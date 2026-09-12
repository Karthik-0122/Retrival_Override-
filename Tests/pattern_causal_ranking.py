"""
PATTERN-BASED causal head ranking, via TransformerLens's hook_pattern.

This is DELIBERATELY DIFFERENT from the existing DLA ranking (which used
hook_z -- a head's OUTPUT VALUE contribution to the logits). This instead
tests: if a head's ATTENTION PATTERN put more weight on the retrieved
passage specifically, does the correct answer become more likely? That is
the direct causal question Phase 3's original ROI metric was built
around (attention divergence over context) but was never actually tested
with an intervention until now -- everything tried so far (ablation,
DLA, hook_z steering) worked on what a head WRITES, not WHERE it LOOKS.

METHOD: for each candidate head, at the last token position (the
position generating the next token), take its real attention distribution
over the prompt, multiply the weight on PASSAGE-TOKEN positions by a
boost factor, renormalize the whole row back to sum to 1 (a valid
attention distribution must sum to 1 -- this is a proportional
reallocation, not an additive corruption), and measure the resulting
shift in (gold - wrong) logit difference versus the unmodified baseline.
Average across many queries; rank heads by this causal shift.

IMPORTANT APPROXIMATION, stated plainly: passage-token positions are
identified by tokenizing the passage text ALONE and taking that many
tokens from the start of the prompt (since build_prompt puts the passage
first). This can be off by a token or two at the boundary due to how
BPE tokenization merges across a text join point -- a well-known,
accepted approximation for this kind of span identification, not an
exact ground truth. Good enough to identify "roughly the passage
region," not intended as token-perfect.

Run from repo root:
    python Tests/pattern_causal_ranking.py

Requires:
    data/final/Phase_02/analysis_dataset.jsonl
    data/final/Phase_02/retrieval_results.jsonl
    data/final/Phase_04/ablation_test_sample.json
    data/final/Phase_03/phase3_length_controlled.json

Writes:
    data/final/Phase_04/pattern_causal_ranking_results.json
"""

import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gemma-2-9b"
LOCAL_MODEL_PATH = "google/gemma-2-9b"  # Hub repo ID, not a local path -- this
                             # box has no pre-downloaded model files, so this
                             # downloads Gemma's real bf16 checkpoint (~18GB)
                             # directly, correctly sized (not the ~37GB fp32
                             # download that happens if TransformerLens tries
                             # to fetch its own copy via a bare model name
                             # instead of an explicit hf_model handoff)
DATASET_PATH = "data/final/Phase_02/analysis_dataset.jsonl"
RETRIEVAL_PATH = "data/final/Phase_02/retrieval_results.jsonl"
TEST_SAMPLE_PATH = "data/final/Phase_04/ablation_test_sample.json"
LENGTH_CONTROLLED_PATH = "data/final/Phase_03/phase3_length_controlled.json"
OUT_PATH = "data/final/Phase_04/pattern_causal_ranking_results.json"

N_QUERIES = 15  # kept deliberately small: this is a diagnostic RANKING pass,
# not a statistical confirmation. Cost note: this script runs one separate
# forward pass per (layer, head, query) combination -- at 16 layers x 16
# heads x N_QUERIES, that's 256*N_QUERIES total forward passes. At N=15
# that's ~3,840 passes, roughly 60-80 minutes at the per-pass timing seen
# elsewhere tonight. Raising N_QUERIES scales runtime linearly -- know that
# before increasing it.
BOOST_FACTOR = 3.0


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def get_validated_layers(model_key="gemma"):
    data = json.load(open(LENGTH_CONTROLLED_PATH))
    return sorted(r["layer"] for r in data[model_key] if r.get("fdr_significant"))


def build_prompt_and_passage_token_count(model, passages, question):
    passage_text = "\n\n".join(passages)
    prompt = f"{passage_text}\n\nQuestion: {question}\nAnswer:"
    passage_tokens = model.to_tokens(passage_text, prepend_bos=True)
    n_passage_tokens = passage_tokens.shape[1]
    return prompt, n_passage_tokens


def get_gold_and_wrong_token_ids(model, gold_answer_text, baseline_logits):
    if not gold_answer_text:
        return None, None
    gold_ids = model.to_tokens(" " + gold_answer_text, prepend_bos=False)[0]
    if len(gold_ids) == 0:
        return None, None
    gold_id = gold_ids[0].item()
    top_wrong_id = baseline_logits.argmax().item()
    if top_wrong_id == gold_id:
        top_wrong_id = baseline_logits.topk(2).indices[1].item()
    return gold_id, top_wrong_id


def make_pattern_boost_hook(head_idx, n_passage_tokens, boost_factor, state):
    def hook_fn(pattern, hook):
        if not state["armed"]:
            return pattern
        pattern = pattern.clone()
        row = pattern[0, head_idx, -1, :]
        n_pass = min(n_passage_tokens, row.shape[0])
        row[:n_pass] = row[:n_pass] * boost_factor
        row = row / row.sum()
        pattern[0, head_idx, -1, :] = row
        return pattern
    return hook_fn


def load_model(device):
    print(f"Loading HF model from local path {LOCAL_MODEL_PATH} (bf16, on CPU first)...")
    # device_map="cpu", NOT device -- loading straight to GPU here would mean
    # TWO full ~19GB copies exist in VRAM at once during the handoff below
    # (this HF copy, plus the internal copy HookedTransformer builds when
    # wrapping it) -- confirmed as the actual OOM cause on a 22GB GPU.
    # Loading to CPU RAM first means only ONE full copy ever touches the
    # GPU, which HookedTransformer creates itself via the device= argument
    # below.
    hf_model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH, dtype=torch.bfloat16, device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)

    print("Wrapping with HookedTransformer on CPU first (moving to GPU after)...")
    # CORRECTION: from_pretrained_no_processing was assumed to skip the
    # float32-upcasting weight-processing step -- confirmed WRONG by a
    # real second OOM with an identical traceback (it still internally
    # calls from_pretrained -> load_and_process_state_dict ->
    # ProcessWeights.process_weights -> v.float() on every tensor,
    # regardless of the "no_processing" name, in this installed version).
    # More robust fix, that doesn't depend on trusting that internal
    # behavior: do ALL processing with device="cpu" first, where there's
    # much more headroom than a 22GB GPU, then move the FINISHED model to
    # GPU as an explicit, separate final step. This works no matter what
    # from_pretrained_no_processing actually does internally, since
    # nothing touches CUDA memory until processing is already complete.
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME, hf_model=hf_model, tokenizer=tokenizer,
        device="cpu", dtype=torch.bfloat16,
    )
    del hf_model
    print(f"Moving fully-processed model to {device}...")
    model = model.to(device)
    torch.cuda.empty_cache()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)

    layers = get_validated_layers()
    n_heads = model.cfg.n_heads
    print(f"Checking {len(layers)} validated layers x {n_heads} heads = {len(layers) * n_heads} candidates")

    dataset_records = load_jsonl(DATASET_PATH)
    retrieval_records = load_jsonl(RETRIEVAL_PATH)
    dataset_by_id = {r["query_id"]: r for r in dataset_records}
    retrieval_by_id = {r["query_id"]: r for r in retrieval_records}
    test_sample = json.load(open(TEST_SAMPLE_PATH))
    queries = test_sample["gemma"]["override_cases"][:N_QUERIES]

    all_shifts = {}

    for qi, q in enumerate(queries):
        qid = q["query_id"]
        ret = retrieval_by_id.get(qid)
        row = dataset_by_id.get(qid)
        if not ret or not row or not ret.get("retrieved_passages"):
            continue

        prompt, n_passage_tokens = build_prompt_and_passage_token_count(
            model, ret["retrieved_passages"], q["question"]
        )
        tokens = model.to_tokens(prompt)

        with torch.no_grad():
            baseline_logits = model(tokens)[0, -1, :]
        gold_id, wrong_id = get_gold_and_wrong_token_ids(model, row.get("gold_answer_text"), baseline_logits)
        if gold_id is None:
            continue

        baseline_logprobs = torch.log_softmax(baseline_logits, dim=-1)
        baseline_diff = (baseline_logprobs[gold_id] - baseline_logprobs[wrong_id]).item()

        for layer in layers:
            hook_name = f"blocks.{layer}.attn.hook_pattern"
            for head in range(n_heads):
                state = {"armed": True}
                hook_fn = make_pattern_boost_hook(head, n_passage_tokens, BOOST_FACTOR, state)
                with torch.no_grad():
                    boosted_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])[0, -1, :]
                boosted_logprobs = torch.log_softmax(boosted_logits, dim=-1)
                boosted_diff = (boosted_logprobs[gold_id] - boosted_logprobs[wrong_id]).item()

                shift = boosted_diff - baseline_diff
                all_shifts.setdefault((layer, head), []).append(shift)

        if (qi + 1) % 5 == 0:
            print(f"  Processed {qi + 1}/{len(queries)} queries...")

    mean_shifts = {f"{l}_{h}": sum(v) / len(v) for (l, h), v in all_shifts.items()}
    ranked = sorted(mean_shifts.items(), key=lambda x: -x[1])

    print(f"\n{'='*70}")
    print("TOP 15 HEADS: boosting attention to the passage helps MOST")
    print(f"{'='*70}")
    for key, val in ranked[:15]:
        layer, head = key.split("_")
        print(f"  layer {layer:>2s} head {head:>2s}: mean (gold-wrong) logit-diff shift = {val:+.4f}")

    print(f"\n{'='*70}")
    print("BOTTOM 10 HEADS: boosting attention to the passage HURTS (for contrast)")
    print(f"{'='*70}")
    for key, val in ranked[-10:]:
        layer, head = key.split("_")
        print(f"  layer {layer:>2s} head {head:>2s}: mean (gold-wrong) logit-diff shift = {val:+.4f}")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"boost_factor": BOOST_FACTOR, "mean_shifts": mean_shifts}, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    print("\nTop-ranked heads here are the candidates for a PATTERN-based steering")
    print("test -- a genuinely different follow-up from the hook_z steering already")
    print("validated, testing whether increasing real attention to the passage (not")
    print("adding a direction to output value) can reach the cases hook_z couldn't.")


if __name__ == "__main__":
    main()
