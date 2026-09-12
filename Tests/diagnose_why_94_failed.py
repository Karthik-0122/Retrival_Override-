"""
Diagnostic: WHY did steering fail on 94/100 override cases?

The confirmed result only recorded binary flip/no-flip. This measures
the actual continuous shift in the gold answer's probability for the
94 non-flipped cases, real steering vs control steering, at the
already-confirmed setup (top-8 heads, magnitude=10).

Two possible outcomes, with different implications:
  A) Real steering shows a small but CONSISTENT positive shift on these
     94 cases (even though not enough to flip) -- the mechanism IS
     reaching these cases, just not with enough force. Supports trying
     more heads / higher magnitude (steering_hard_cases.py).
  B) Real steering shows ~NO shift on these 94 cases, indistinguishable
     from control -- the mechanism genuinely does not reach these cases
     at all. Would mean something else, elsewhere in the network, is
     responsible for locking in high-confidence wrong answers -- a
     different, informative finding, not a "push harder" problem.

Run from repo root:
    python Tests/diagnose_why_94_failed.py

Requires:
    data/final/Phase_04/steering_confirmation_results.json
    data/final/Phase_04/dla_results_bf16.json
    data/final/Phase_02/analysis_dataset.jsonl
    data/final/Phase_02/retrieval_results.jsonl
    data/final/Phase_04/ablation_test_sample.json

Writes:
    data/final/Phase_04/why_94_failed_results.json
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from scipy import stats

sys.path.insert(0, "Tests")
from steering_screen import (
    MODEL_PATH, HEAD_DIM, DATASET_PATH, RETRIEVAL_PATH, TEST_SAMPLE_PATH,
    load_jsonl, build_prompt, get_gold_first_token_id,
    SteeringHook, compute_direction_means, QUANT_CONFIG,
    load_top_negative_heads, pick_random_heads,
)

CONFIRMATION_RESULTS_PATH = "data/final/Phase_04/steering_confirmation_results.json"
OUT_PATH = "data/final/Phase_04/why_94_failed_results.json"
CONFIRMED_MAGNITUDE = 10.0


def get_hard_case_ids():
    data = json.load(open(CONFIRMATION_RESULTS_PATH))
    return [r["query_id"] for r in data["real_results"] if not r["flipped_to_correct"]]


def get_gold_logprob(logits, gold_id):
    logprobs = torch.log_softmax(logits, dim=-1)
    return logprobs[gold_id].item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=QUANT_CONFIG, device_map=device, attn_implementation="eager",
    )
    model.eval()

    real_target_heads = load_top_negative_heads()
    random_target_heads = pick_random_heads(len(real_target_heads), real_target_heads)
    print(f"Real heads: {real_target_heads}")
    print(f"Control heads: {random_target_heads}")

    all_heads = real_target_heads + random_target_heads
    directions = compute_direction_means(model, tokenizer, all_heads, device)

    real_hooks, real_handles = [], []
    for (l, h) in real_target_heads:
        hook = SteeringHook(h, HEAD_DIM, directions[(l, h)], CONFIRMED_MAGNITUDE)
        real_handles.append(hook.register(model.model.layers[l]))
        real_hooks.append(hook)

    control_hooks, control_handles = [], []
    for (l, h) in random_target_heads:
        hook = SteeringHook(h, HEAD_DIM, directions[(l, h)], CONFIRMED_MAGNITUDE)
        control_handles.append(hook.register(model.model.layers[l]))
        control_hooks.append(hook)

    hard_ids = set(get_hard_case_ids())
    print(f"\nDiagnosing {len(hard_ids)} previously-unflipped cases")

    dataset_records = load_jsonl(DATASET_PATH)
    retrieval_records = load_jsonl(RETRIEVAL_PATH)
    dataset_by_id = {r["query_id"]: r for r in dataset_records}
    retrieval_by_id = {r["query_id"]: r for r in retrieval_records}
    test_sample = json.load(open(TEST_SAMPLE_PATH))
    all_override_queries = test_sample["gemma"]["override_cases"]
    hard_queries = [q for q in all_override_queries if q["query_id"] in hard_ids]

    results = []
    for q in tqdm(hard_queries, desc="Measuring probability shifts"):
        qid = q["query_id"]
        ret = retrieval_by_id.get(qid)
        row = dataset_by_id.get(qid)
        if not ret or not row or not ret.get("retrieved_passages"):
            continue
        prompt = build_prompt(ret["retrieved_passages"], q["question"])
        gold_id = get_gold_first_token_id(tokenizer, row.get("gold_answer_text"))
        if gold_id is None:
            continue

        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(device)

        for h in real_hooks + control_hooks:
            h.disarm()
        with torch.no_grad():
            baseline_logits = model(input_ids).logits[0, -1, :]
        baseline_logprob = get_gold_logprob(baseline_logits, gold_id)

        for h in real_hooks:
            h.arm()
        with torch.no_grad():
            real_logits = model(input_ids).logits[0, -1, :]
        for h in real_hooks:
            h.disarm()
        real_logprob = get_gold_logprob(real_logits, gold_id)

        for h in control_hooks:
            h.arm()
        with torch.no_grad():
            control_logits = model(input_ids).logits[0, -1, :]
        for h in control_hooks:
            h.disarm()
        control_logprob = get_gold_logprob(control_logits, gold_id)

        results.append({
            "query_id": qid,
            "pcs_gemma": row.get("pcs_gemma"),
            "baseline_logprob": baseline_logprob,
            "real_logprob": real_logprob,
            "control_logprob": control_logprob,
            "real_shift": real_logprob - baseline_logprob,
            "control_shift": control_logprob - baseline_logprob,
        })

    for h in real_handles + control_handles:
        h.remove()

    real_shifts = np.array([r["real_shift"] for r in results])
    control_shifts = np.array([r["control_shift"] for r in results])

    print("\n" + "=" * 70)
    print("RESULT: probability shift on the 94 unflipped cases")
    print("=" * 70)
    print(f"n = {len(results)}")
    print(f"Real steering:    mean shift = {real_shifts.mean():+.4f}  (positive = toward correct)")
    print(f"Control steering: mean shift = {control_shifts.mean():+.4f}")
    n_positive_real = int((real_shifts > 0).sum())
    n_positive_control = int((control_shifts > 0).sum())
    print(f"Real shifts positive:    {n_positive_real}/{len(results)}")
    print(f"Control shifts positive: {n_positive_control}/{len(results)}")

    t, p = stats.ttest_rel(real_shifts, control_shifts)
    print(f"\nPaired t-test (real vs control shift): t={t:.3f}, p={p:.4f}")

    print("\n" + "=" * 70)
    if p < 0.05 and real_shifts.mean() > control_shifts.mean():
        print("INTERPRETATION: Real steering shows a significantly larger positive")
        print("shift than control on these 'failed' cases -- the mechanism IS")
        print("reaching them, just not with enough force to flip the top answer.")
        print("-> Supports testing more heads / higher magnitude (steering_hard_cases.py).")
    else:
        print("INTERPRETATION: No significant difference from control on these cases --")
        print("the mechanism genuinely does not move these cases at all, not even")
        print("partially. This suggests the decision for high-confidence override")
        print("cases is being made elsewhere -- possibly earlier in the network")
        print("(consistent with the earlier finding that fact storage overlaps this")
        print("same layer block) -- not a 'needs more force' problem.")
    print("=" * 70)

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "n": len(results),
            "real_mean_shift": float(real_shifts.mean()),
            "control_mean_shift": float(control_shifts.mean()),
            "pvalue": float(p),
            "per_query": results,
        }, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()

