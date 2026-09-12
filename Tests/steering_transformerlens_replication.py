"""
TRANSFORMERLENS REPLICATION of the validated steering result.

PURPOSE: this is a REPLICATION, not a new experiment. It re-tests the
EXACT SAME 8 heads, EXACT SAME magnitude (10.0), EXACT SAME 100
override-case test set, and EXACT SAME depth-matched random control
already validated in Tests/steering_confirmation.py (p=0.031, real
6/100 vs control 0/100) -- but implemented through TransformerLens
instead of raw PyTorch forward hooks on HuggingFace. If this reproduces
a comparable result, that's strong evidence the finding is a real
property of the model, not an artifact of the custom hook
implementation.

===========================================================================
KEY DESIGN DECISION: folded vs. raw numerics -- UPDATED after a real OOM
===========================================================================
Originally this script planned to use TransformerLens's LayerNorm-folded
("compatibility mode") numerics, reasoning that folding makes per-head
decomposition cleaner. That turned out to be the wrong call for THIS
script specifically, discovered via a real CUDA OOM: the folding step
needs a float32 intermediate regardless of requested dtype, which doesn't
fit in 22GB on top of an already-loaded 18.9GB bf16 model.

On reflection, folding was never actually necessary here in the first
place -- it only helps LINEAR decomposition (like DLA, which separates a
head's output-contribution from a later LayerNorm's scaling). This script
does a direct intervention (measure a direction, add it to hook_z, read
off the real result) -- not a linear decomposition -- so skipping folding
costs nothing. This script now uses from_pretrained_no_processing
instead, which also has a side benefit: it makes the comparison against
the original HuggingFace result MORE direct, not less, since both now
operate in the same unfolded numerical space -- isolating the test to
"does the hook MECHANISM change the result," without folding as an extra
variable.

===========================================================================
KEY DESIGN DECISION: precision (bf16, not 4-bit)
===========================================================================
Mirrors the bf16 DLA check already done tonight, which confirmed the
4-bit-selected heads were NOT distorted by quantization (8/10 top heads
identical) but were somewhat inflated in magnitude. Using bf16 here avoids
re-introducing a quantization variable into a brand-new framework at the
same time -- if something doesn't match the HF result, this way it can
only be the FRAMEWORK, not framework+quantization confounded together.
Needs ~18GB VRAM, same as the earlier bf16 DLA run -- same OOM risk noted
there applies here.

===========================================================================
BUILT-IN SELF-CHECK
===========================================================================
Before running the real experiment, this script runs a handful of
override-case prompts through the model with NO steering and prints the
predicted answer. COMPARE THESE BY EYE against what you already know
those queries' baseline behavior to be (they're override cases, so the
model should predict something OTHER than the gold answer). This is a
cheap, fast sanity check that the model is loaded and behaving as
expected before trusting anything downstream -- same discipline as every
other script tonight.

Run from repo root:
    python Tests/steering_transformerlens_replication.py

Requires:
    data/final/Phase_02/analysis_dataset.jsonl
    data/final/Phase_02/retrieval_results.jsonl
    data/final/Phase_04/ablation_test_sample.json

Writes:
    data/final/Phase_04/steering_transformerlens_replication_results.json
"""

import json
import random
from pathlib import Path

import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gemma-2-9b"   # TransformerLens's registry name -- used only to
                             # tell it WHICH architecture config to use, NOT
                             # to trigger a download (see load_model() below)
LOCAL_MODEL_PATH = "google/gemma-2-9b"  # same local path every other
                             # script tonight already uses -- avoids
                             # TransformerLens re-downloading a fresh
                             # (fp32, ~37GB) copy from HuggingFace's hub
N_LAYERS = 42
N_HEADS = 16

DATASET_PATH = "data/final/Phase_02/analysis_dataset.jsonl"
RETRIEVAL_PATH = "data/final/Phase_02/retrieval_results.jsonl"
TEST_SAMPLE_PATH = "data/final/Phase_04/ablation_test_sample.json"
OUT_PATH = "data/final/Phase_04/steering_transformerlens_replication_results.json"

# EXACT SAME 8 heads as the validated HF result (from dla_results_bf16.json,
# hardcoded here rather than re-derived, since this is a REPLICATION of an
# already-identified finding, not a fresh search)
REAL_TARGET_HEADS = [
    (34, 14), (37, 14), (40, 14), (36, 0), (37, 12), (38, 14), (34, 15), (30, 6),
]
CONFIRMED_MAGNITUDE = 10.0
N_QUERIES = 100
N_DIRECTION_QUERIES = 40
DEPTH_WINDOW = 5
SEED = 2024


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def build_prompt(passages, question):
    passage_text = "\n\n".join(passages)
    return f"{passage_text}\n\nQuestion: {question}\nAnswer:"


def get_gold_first_token_id(model, gold_answer_text):
    if not gold_answer_text:
        return None
    ids = model.to_tokens(" " + gold_answer_text, prepend_bos=False)[0]
    return ids[0].item() if len(ids) > 0 else None


def pick_random_heads(n, exclude_heads):
    """Same depth-matching logic validated earlier tonight -- random
    controls only drawn from within DEPTH_WINDOW layers of a real target,
    not the whole network, to avoid the depth confound found and fixed in
    joint_intervention.py."""
    rng = random.Random(SEED)
    real_layers = sorted({l for l, _ in exclude_heads})
    exclude_set = set(exclude_heads)
    candidates = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)
                  if (l, h) not in exclude_set
                  and any(abs(l - rl) <= DEPTH_WINDOW for rl in real_layers)]
    rng.shuffle(candidates)
    return candidates[:n]


def compute_direction_means(model, target_heads, dataset_records, retrieval_by_id, exclude_ids, device):
    """Computes mean(faithful activation) - mean(override activation) at
    each target head, reading from TransformerLens's hook_z -- which is
    ALREADY split by head (shape [batch, seq, n_heads, d_head]), unlike the
    raw HF version, which required manually slicing a concatenated tensor
    by head_dim offsets. This removes an entire category of "did I slice
    the right offset" risk that existed in the HF implementation."""

    def pool_for(label):
        pool = [r for r in dataset_records if r.get("gemma_label") == label
                and r.get("source_category") != "confiqa"
                and r["query_id"] not in exclude_ids][:N_DIRECTION_QUERIES]
        for r in pool:
            ret = retrieval_by_id.get(r["query_id"])
            r["retrieved_passages"] = ret["retrieved_passages"] if ret else []
        return pool

    faithful_pool = pool_for("faithful")
    override_pool = pool_for("override")

    layers_needed = sorted({l for l, _ in target_heads})
    hook_names = [f"blocks.{l}.attn.hook_z" for l in layers_needed]

    def get_mean_activations(pool, label):
        sums = {k: torch.zeros(model.cfg.d_head, device=device) for k in target_heads}
        counts = {k: 0 for k in target_heads}
        for r in pool:
            if not r.get("retrieved_passages"):
                continue
            prompt = build_prompt(r["retrieved_passages"], r["question"])
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in hook_names)
            for (l, h) in target_heads:
                z = cache[f"blocks.{l}.attn.hook_z"][0, -1, h, :]  # [d_head] at last position
                sums[(l, h)] += z
                counts[(l, h)] += 1
        return {k: sums[k] / max(counts[k], 1) for k in target_heads}

    print(f"  Computing faithful means ({len(faithful_pool)} queries)...")
    faithful_means = get_mean_activations(faithful_pool, "faithful")
    print(f"  Computing override means ({len(override_pool)} queries)...")
    override_means = get_mean_activations(override_pool, "override")

    return {k: faithful_means[k] - override_means[k] for k in target_heads}


def make_steering_hook(head_idx, direction, magnitude):
    """TransformerLens hook function: adds magnitude*direction to hook_z
    at one specific head, last token position only. armed/magnitude is a
    mutable dict so the SAME hook function object can be reused across
    baseline/steered calls just by changing state.magnitude, mirroring the
    SteeringHook.arm()/disarm() pattern from the HF version."""
    state = {"magnitude": magnitude}

    def hook_fn(z, hook):
        if state["magnitude"] == 0.0:
            return z
        z = z.clone()
        z[0, -1, head_idx, :] += state["magnitude"] * direction
        return z

    return hook_fn, state


def run_condition(model, hook_specs, queries, dataset_by_id, retrieval_by_id):
    """hook_specs: list of (hook_name, hook_fn, state) tuples. Runs each
    query with hooks active, records baseline (hooks off) vs steered
    (hooks on) top-token prediction."""
    results = []
    for q in queries:
        qid = q["query_id"]
        ret = retrieval_by_id.get(qid)
        row = dataset_by_id.get(qid)
        if not ret or not row or not ret.get("retrieved_passages"):
            continue
        prompt = build_prompt(ret["retrieved_passages"], q["question"])
        gold_id = get_gold_first_token_id(model, row.get("gold_answer_text"))
        if gold_id is None:
            continue

        tokens = model.to_tokens(prompt)

        for _, _, state in hook_specs:
            state["magnitude"] = 0.0
        with torch.no_grad():
            baseline_logits = model(tokens)[0, -1, :]
        baseline_correct = baseline_logits.argmax().item() == gold_id

        for _, _, state in hook_specs:
            state["magnitude"] = CONFIRMED_MAGNITUDE
        fwd_hooks = [(name, fn) for name, fn, _ in hook_specs]
        with torch.no_grad():
            steered_logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)[0, -1, :]
        for _, _, state in hook_specs:
            state["magnitude"] = 0.0
        steered_correct = steered_logits.argmax().item() == gold_id

        results.append({
            "query_id": qid,
            "baseline_correct": baseline_correct,
            "steered_correct": steered_correct,
            "flipped_to_correct": (not baseline_correct) and steered_correct,
        })
    return results


def run_mcnemar(real_results, control_results):
    real_by_id = {r["query_id"]: r for r in real_results}
    control_by_id = {r["query_id"]: r for r in control_results}
    common = set(real_by_id) & set(control_by_id)
    both = real_only = control_only = neither = 0
    for qid in common:
        rf, cf = real_by_id[qid]["flipped_to_correct"], control_by_id[qid]["flipped_to_correct"]
        if rf and cf: both += 1
        elif rf: real_only += 1
        elif cf: control_only += 1
        else: neither += 1
    print(f"  n={len(common)} paired queries")
    print(f"  both: {both}  real-only: {real_only}  control-only: {control_only}  neither: {neither}")
    if real_only + control_only == 0:
        print("  No discordant pairs -- p=1.0")
        return 1.0
    table = [[both, real_only], [control_only, neither]]
    result = mcnemar(table, exact=True)
    print(f"  McNemar's exact test: p = {result.pvalue:.6f}")
    return result.pvalue


def load_model(device):
    """Loads from the LOCAL model path already used by every other script
    tonight, in bf16, then hands the already-loaded HF model to
    HookedTransformer -- avoiding a fresh ~37GB fp32 download from
    HuggingFace's hub, which is what happens if you pass a bare model
    name string instead. This exact failure occurred once already
    tonight (disk filled to 100% mid-download)."""
    print(f"Loading HF model from local path {LOCAL_MODEL_PATH} (bf16, on CPU first)...")
    # device_map="cpu", NOT device -- loading straight to GPU would mean
    # TWO full ~19GB copies exist in VRAM during the handoff below (this
    # HF copy, plus HookedTransformer's own internal copy) -- confirmed
    # as a real OOM cause on a 22GB GPU. CPU-first means only one full
    # copy ever touches the GPU, built by HookedTransformer itself.
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

    assert model.cfg.n_layers == N_LAYERS, f"Expected {N_LAYERS} layers, got {model.cfg.n_layers}"
    assert model.cfg.n_heads == N_HEADS, f"Expected {N_HEADS} heads, got {model.cfg.n_heads}"

    dataset_records = load_jsonl(DATASET_PATH)
    retrieval_records = load_jsonl(RETRIEVAL_PATH)
    dataset_by_id = {r["query_id"]: r for r in dataset_records}
    retrieval_by_id = {r["query_id"]: r for r in retrieval_records}
    test_sample = json.load(open(TEST_SAMPLE_PATH))
    override_queries = test_sample["gemma"]["override_cases"][:N_QUERIES]

    print("\n" + "=" * 70)
    print("SELF-CHECK: baseline predictions on 3 override-case prompts (no steering)")
    print("These are OVERRIDE cases -- the model SHOULD predict something OTHER")
    print("than the gold answer here. If it predicts the gold answer instead,")
    print("something is wrong with prompt construction or model loading.")
    print("=" * 70)
    for q in override_queries[:3]:
        ret = retrieval_by_id.get(q["query_id"])
        row = dataset_by_id.get(q["query_id"])
        if not ret or not row:
            continue
        prompt = build_prompt(ret["retrieved_passages"], q["question"])
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model(tokens)[0, -1, :]
        top_token = model.to_string(logits.argmax().unsqueeze(0))
        print(f"  Q: {q['question'][:70]}")
        print(f"    gold: {row.get('gold_answer_text')}  |  model top prediction: {top_token!r}")
    print("=" * 70)
    input("\nPress Enter if the self-check looks correct, Ctrl+C to abort...")

    exclude_ids = {c["query_id"] for c in test_sample["gemma"]["faithful_control_cases"]}
    exclude_ids |= {c["query_id"] for c in test_sample["gemma"]["override_cases"]}

    random_target_heads = pick_random_heads(len(REAL_TARGET_HEADS), REAL_TARGET_HEADS)
    print(f"\nReal heads:    {REAL_TARGET_HEADS}")
    print(f"Control heads: {random_target_heads}")

    print("\nComputing steering directions...")
    all_heads = REAL_TARGET_HEADS + random_target_heads
    directions = compute_direction_means(model, all_heads, dataset_records, retrieval_by_id, exclude_ids, device)

    real_hook_specs = []
    for (l, h) in REAL_TARGET_HEADS:
        fn, state = make_steering_hook(h, directions[(l, h)], 0.0)
        real_hook_specs.append((f"blocks.{l}.attn.hook_z", fn, state))

    control_hook_specs = []
    for (l, h) in random_target_heads:
        fn, state = make_steering_hook(h, directions[(l, h)], 0.0)
        control_hook_specs.append((f"blocks.{l}.attn.hook_z", fn, state))

    print(f"\nRunning REAL condition (magnitude={CONFIRMED_MAGNITUDE})...")
    real_results = run_condition(model, real_hook_specs, override_queries, dataset_by_id, retrieval_by_id)
    n_flip = sum(1 for r in real_results if r["flipped_to_correct"])
    print(f"  REAL: {n_flip}/{len(real_results)} flipped to correct")

    print(f"\nRunning CONTROL condition (magnitude={CONFIRMED_MAGNITUDE})...")
    control_results = run_condition(model, control_hook_specs, override_queries, dataset_by_id, retrieval_by_id)
    n_flip_c = sum(1 for r in control_results if r["flipped_to_correct"])
    print(f"  CONTROL: {n_flip_c}/{len(control_results)} flipped to correct")

    print("\n" + "=" * 70)
    print("STATISTICAL TEST")
    print("=" * 70)
    pvalue = run_mcnemar(real_results, control_results)

    print("\n" + "=" * 70)
    print("COMPARISON TO THE ORIGINAL HUGGINGFACE RESULT")
    print("=" * 70)
    print(f"  HuggingFace (original):  real=6/100, control=0/100, p=0.031")
    print(f"  TransformerLens (this):  real={n_flip}/{len(real_results)}, control={n_flip_c}/{len(control_results)}, p={pvalue:.4f}")
    if pvalue < 0.05 and n_flip > n_flip_c:
        print("  -> REPLICATED. Independent evidence the effect is real, not a")
        print("     framework-specific artifact of the custom HF hook implementation.")
    else:
        print("  -> DID NOT CLEANLY REPLICATE. Worth investigating why before")
        print("     treating either result as more trustworthy than the other --")
        print("     see the module docstring's notes on folded numerics as one")
        print("     place a real difference could legitimately arise.")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "real_heads": REAL_TARGET_HEADS, "control_heads": random_target_heads,
            "magnitude": CONFIRMED_MAGNITUDE,
            "real_results": real_results, "control_results": control_results,
            "pvalue": pvalue,
        }, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
