"""
Phase 5 (scoped down, not full circuit tracing): Direct Logit Attribution.

Motivated by McDougall, Conmy, Rushing, McGrath, Nanda (BlackboxNLP 2024)
"Copy Suppression" -- a specific, well-precedented phenomenon where ONE
attention head (not a block of 96) suppresses a token that earlier
layers were leaning toward predicting. Their paper explicitly notes
ablation-based methods can MISS negative heads because backup/redundant
components compensate -- exactly the pattern found in tonight's block-
level ablation tests (real ~= random control, once properly depth-matched).

DLA is a DIFFERENT diagnostic than ablation: no intervention, just a
linear decomposition of the final logit into each component's direct
contribution, via a single forward pass. This gives HEAD-LEVEL
resolution that block ablation structurally could not provide, and
doesn't depend on ablation having worked -- it's a complementary
method, not a downstream step gated on Phase 4's outcome.

WHAT TO LOOK FOR: heads with a NEGATIVE contribution to
(gold_logit - top_wrong_logit) specifically within the validated block
-- a head that's actively pushing the model AWAY from the correct
retrieved answer, even while the block's average behavior (tonight's
finding) is that removing the WHOLE block hurts overall. A single
negative head can exist inside a block that's net-positive on average.

SELF-CHECK BUILT IN: for each layer, verifies that summing all heads'
individually-computed contributions reconstructs the ACTUAL residual-
stream delta from that layer almost exactly. If this check fails, the
weight-slicing math has an orientation bug -- do not trust the DLA
results until it passes.

Run from repo root:
    python Tests/direct_logit_attribution.py

Requires:
    data/final/Phase_03/phase3_length_controlled.json
    data/final/Phase_04/ablation_test_sample.json
    data/final/Phase_02/analysis_dataset.jsonl
    data/final/Phase_02/retrieval_results.jsonl

Writes:
    data/final/Phase_04/dla_results.json
"""

import json
import sys
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

MODEL_CONFIGS = {
    "gemma": "/home/models/gemma-2-9b",
    "llama": "/home/models/Llama-3.1-8B",
}
HEAD_DIMS = {"gemma": 256, "llama": 128}
NUM_HEADS = {"gemma": 16, "llama": 32}

LENGTH_CONTROLLED_PATH = "data/final/Phase_03/phase3_length_controlled.json"
TEST_SAMPLE_PATH = "data/final/Phase_04/ablation_test_sample.json"
DATASET_PATH = "data/final/Phase_02/analysis_dataset.jsonl"
RETRIEVAL_PATH = "data/final/Phase_02/retrieval_results.jsonl"
OUT_PATH = "data/final/Phase_04/dla_results.json"

N_QUERIES = 40  # override cases only, this is diagnostic not statistical -- doesn't need n=100

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def get_validated_layers(model_key):
    data = json.load(open(LENGTH_CONTROLLED_PATH))
    return sorted(r["layer"] for r in data[model_key] if r.get("fdr_significant"))


def build_prompt(passages, question):
    passage_text = "\n\n".join(passages)
    return f"{passage_text}\n\nQuestion: {question}\nAnswer:"


def get_gold_first_token_id(tokenizer, gold_answer_text):
    if not gold_answer_text:
        return None
    ids = tokenizer(" " + gold_answer_text, add_special_tokens=False).input_ids
    return ids[0] if ids else None


def compute_dla_for_query(model, tokenizer, prompt, gold_token_id, layers, num_heads, head_dim, device):
    """
    Returns: dict {(layer, head): contribution_to_logit_diff}
    Also returns the self-check discrepancy per layer (should be ~0).
    """
    captured = {}

    def make_hook(layer_idx):
        def hook(module, args):
            captured[layer_idx] = args[0][0, -1, :].detach()  # pre-o_proj, last position
            return None
        return hook

    handles = [model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(make_hook(l))
               for l in layers]

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(device)
    with torch.no_grad():
        output = model(input_ids, output_hidden_states=True)

    for h in handles:
        h.remove()

    logits = output.logits[0, -1, :]
    top_wrong_id = logits.argmax().item()
    if top_wrong_id == gold_token_id:
        # model already predicts correctly -- find the 2nd-highest as "the alternative" instead
        top_wrong_id = logits.topk(2).indices[1].item()

    final_hidden = output.hidden_states[-1][0, -1, :]  # residual stream just before final norm
    rms = final_hidden.float().pow(2).mean().sqrt()  # RMSNorm denominator at this exact operating point

    final_norm_weight = model.model.norm.weight.float()
    unembed = model.lm_head.weight.float()  # (vocab, hidden)

    direction = unembed[gold_token_id, :] - unembed[top_wrong_id, :]  # (hidden,) -- logit-diff direction

    contributions = {}
    self_check = {}

    for layer_idx in layers:
        pre_o_proj = captured[layer_idx].float()  # (num_heads*head_dim,)
        o_proj_module = model.model.layers[layer_idx].self_attn.o_proj

        # NOTE: model is 4-bit quantized -- o_proj.weight is PACKED storage,
        # not a usable float matrix (confirmed via a real shape-mismatch
        # crash on first run: manually slicing raw .weight does not work
        # with bitsandbytes 4-bit layers). Fix: use the module's own
        # forward() on masked input instead of touching weights directly --
        # this handles dequantization internally and sidesteps the problem
        # entirely. Since o_proj is linear, masking the input to just one
        # head's slice and running it through forward() gives exactly that
        # head's contribution (module(masked) - module(zeros) removes any
        # bias term so contributions are purely linear and summable).
        zeros_input = torch.zeros_like(pre_o_proj).unsqueeze(0).to(pre_o_proj.dtype)
        with torch.no_grad():
            baseline_out = o_proj_module(zeros_input).squeeze(0).float()  # captures bias, if any
            full_layer_output = o_proj_module(pre_o_proj.unsqueeze(0)).squeeze(0).float()

        per_head_sum = baseline_out.clone()  # one copy of bias, added back once at the end

        for head_idx in range(num_heads):
            h_start, h_end = head_idx * head_dim, (head_idx + 1) * head_dim
            masked = torch.zeros_like(pre_o_proj)
            masked[h_start:h_end] = pre_o_proj[h_start:h_end]
            with torch.no_grad():
                head_out_with_bias = o_proj_module(masked.unsqueeze(0).to(pre_o_proj.dtype)).squeeze(0).float()
            head_contrib_to_residual = head_out_with_bias - baseline_out  # pure linear part, bias removed
            per_head_sum += head_contrib_to_residual

            normed = (head_contrib_to_residual / (rms + 1e-6)) * final_norm_weight
            contrib_to_logit_diff = (normed @ direction).item()
            contributions[(layer_idx, head_idx)] = contrib_to_logit_diff

        # self-check: baseline + sum of all per-head linear contributions should equal the real full output
        discrepancy = (per_head_sum - full_layer_output).abs().max().item()
        self_check[layer_idx] = discrepancy

    return contributions, self_check


def run_model(model_key, model_path, device):
    print(f"\n{'='*70}\nLoading {model_key} ({model_path})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=QUANT_CONFIG, device_map=device, attn_implementation="eager",
    )
    model.eval()

    layers = get_validated_layers(model_key)
    num_heads = NUM_HEADS[model_key]
    head_dim = HEAD_DIMS[model_key]
    print(f"  Checking {len(layers)} validated layers x {num_heads} heads = {len(layers)*num_heads} candidates")

    dataset_records = load_jsonl(DATASET_PATH)
    retrieval_records = load_jsonl(RETRIEVAL_PATH)
    dataset_by_id = {r["query_id"]: r for r in dataset_records}
    retrieval_by_id = {r["query_id"]: r for r in retrieval_records}

    test_sample = json.load(open(TEST_SAMPLE_PATH))
    queries = test_sample[model_key]["override_cases"][:N_QUERIES]

    all_contributions = {}  # (layer, head) -> list of per-query contributions
    max_discrepancy_seen = 0.0

    for q in tqdm(queries, desc=f"DLA ({model_key})"):
        qid = q["query_id"]
        ret = retrieval_by_id.get(qid)
        row = dataset_by_id.get(qid)
        if not ret or not row or not ret.get("retrieved_passages"):
            continue
        prompt = build_prompt(ret["retrieved_passages"], q["question"])
        gold_token_id = get_gold_first_token_id(tokenizer, row.get("gold_answer_text"))
        if gold_token_id is None:
            continue

        contribs, self_check = compute_dla_for_query(
            model, tokenizer, prompt, gold_token_id, layers, num_heads, head_dim, device
        )
        max_discrepancy_seen = max(max_discrepancy_seen, max(self_check.values()))

        for key, val in contribs.items():
            all_contributions.setdefault(key, []).append(val)

    print(f"\n  SELF-CHECK: max reconstruction discrepancy across all layers/queries: {max_discrepancy_seen:.6f}")
    if max_discrepancy_seen > 0.5:
        print(f"  *** WARNING: discrepancy is large -- weight-slicing math likely has a bug. ***")
        print(f"  *** Do not trust the DLA results below until this is fixed. ***")
    else:
        print(f"  Self-check passed (small discrepancy expected from fp precision + norm-linearization approximation).")

    mean_contribs = {f"{k[0]}_{k[1]}": sum(v) / len(v) for k, v in all_contributions.items()}
    ranked = sorted(mean_contribs.items(), key=lambda x: x[1])  # most negative first

    print(f"\n  Top 10 MOST NEGATIVE heads (candidates for 'suppresses correct answer'):")
    for key, val in ranked[:10]:
        layer, head = key.split("_")
        print(f"    layer {layer:>2s} head {head:>2s}: mean contribution to (gold-wrong) logit diff = {val:+.4f}")

    print(f"\n  Top 10 MOST POSITIVE heads (for contrast -- these push TOWARD correct answer):")
    for key, val in ranked[-10:][::-1]:
        layer, head = key.split("_")
        print(f"    layer {layer:>2s} head {head:>2s}: mean contribution to (gold-wrong) logit diff = {val:+.4f}")

    del model
    torch.cuda.empty_cache()
    return mean_contribs, max_discrepancy_seen


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = {}
    for model_key, model_path in MODEL_CONFIGS.items():
        mean_contribs, discrepancy = run_model(model_key, model_path, device)
        output[model_key] = {"contributions": mean_contribs, "self_check_max_discrepancy": discrepancy}

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
