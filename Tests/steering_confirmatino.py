"""
Steering test CONFIRMATION run -- the screen (n=20) found a real signal:
magnitude 10.0, REAL flips (10/20) clearly beating CONTROL flips (2/20).
This scales that up to n=100 with a proper paired statistical test
(McNemar's exact test on per-query flip outcomes), the same rigor used
for every other result in this project.

Run from repo root:
    python Tests/steering_confirmation.py

Requires the same files as steering_screen.py.

Writes:
    data/final/Phase_04/steering_confirmation_results.json
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, "Tests")
from steering_screen import (
    MODEL_PATH, HEAD_DIM, DATASET_PATH, RETRIEVAL_PATH, TEST_SAMPLE_PATH,
    load_jsonl, build_prompt, get_gold_first_token_id, pick_random_heads,
    SteeringHook, compute_direction_means, QUANT_CONFIG,
    load_top_negative_heads,
)

WINNING_MAGNITUDE = 10.0
N_CONFIRM_QUERIES = 100
OUT_PATH = "data/final/Phase_04/steering_confirmation_results.json"


def run_full(model, tokenizer, hooks, queries, dataset_by_id, retrieval_by_id, device, magnitude):
    for h in hooks:
        h.magnitude = magnitude

    results = []
    for q in tqdm(queries, desc=f"Confirmation run (magnitude={magnitude})"):
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

        for h in hooks:
            h.disarm()
        with torch.no_grad():
            baseline_logits = model(input_ids).logits[0, -1, :]
        baseline_correct = baseline_logits.argmax().item() == gold_id

        for h in hooks:
            h.arm()
        with torch.no_grad():
            steered_logits = model(input_ids).logits[0, -1, :]
        for h in hooks:
            h.disarm()
        steered_correct = steered_logits.argmax().item() == gold_id

        results.append({
            "query_id": qid,
            "baseline_correct": baseline_correct,
            "steered_correct": steered_correct,
            "flipped_to_correct": (not baseline_correct) and steered_correct,
            "flipped_to_incorrect": baseline_correct and (not steered_correct),
        })

    return results


def run_mcnemar(real_results, control_results):
    real_by_id = {r["query_id"]: r for r in real_results}
    control_by_id = {r["query_id"]: r for r in control_results}
    common = set(real_by_id) & set(control_by_id)

    both = real_only = control_only = neither = 0
    for qid in common:
        rf = real_by_id[qid]["flipped_to_correct"]
        cf = control_by_id[qid]["flipped_to_correct"]
        if rf and cf:
            both += 1
        elif rf:
            real_only += 1
        elif cf:
            control_only += 1
        else:
            neither += 1

    print(f"\n  n={len(common)} paired queries")
    print(f"  both flipped: {both}  real-only: {real_only}  control-only: {control_only}  neither: {neither}")

    if real_only + control_only == 0:
        print("  No discordant pairs -- p=1.0")
        return 1.0

    table = [[both, real_only], [control_only, neither]]
    result = mcnemar(table, exact=True)
    print(f"  McNemar's exact test: p = {result.pvalue:.6f}")
    if result.pvalue < 0.05:
        direction = "REAL > CONTROL" if real_only > control_only else "CONTROL > REAL"
        print(f"  -> SIGNIFICANT at p<0.05 ({direction})")
        if real_only > control_only:
            print("  -> This is a validated causal effect: steering these specific heads")
            print("     flips override cases to correct significantly more than a random")
            print("     comparable intervention does.")
    else:
        print("  -> NOT significant at p<0.05")
    return result.pvalue


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=QUANT_CONFIG, device_map=device, attn_implementation="eager",
    )
    model.eval()

    real_target_heads = load_top_negative_heads()
    random_target_heads = pick_random_heads(len(real_target_heads), real_target_heads)
    print(f"Real target heads: {real_target_heads}")
    print(f"Random control heads: {random_target_heads}")

    all_target_heads = real_target_heads + random_target_heads
    directions = compute_direction_means(model, tokenizer, all_target_heads, device)

    real_hooks, real_handles = [], []
    for (l, h) in real_target_heads:
        hook = SteeringHook(h, HEAD_DIM, directions[(l, h)], WINNING_MAGNITUDE)
        real_handles.append(hook.register(model.model.layers[l]))
        real_hooks.append(hook)

    control_hooks, control_handles = [], []
    for (l, h) in random_target_heads:
        hook = SteeringHook(h, HEAD_DIM, directions[(l, h)], WINNING_MAGNITUDE)
        control_handles.append(hook.register(model.model.layers[l]))
        control_hooks.append(hook)

    dataset_records = load_jsonl(DATASET_PATH)
    retrieval_records = load_jsonl(RETRIEVAL_PATH)
    dataset_by_id = {r["query_id"]: r for r in dataset_records}
    retrieval_by_id = {r["query_id"]: r for r in retrieval_records}
    test_sample = json.load(open(TEST_SAMPLE_PATH))
    queries = test_sample["gemma"]["override_cases"][:N_CONFIRM_QUERIES]

    print(f"\nRunning REAL (magnitude={WINNING_MAGNITUDE})...")
    real_results = run_full(model, tokenizer, real_hooks, queries, dataset_by_id, retrieval_by_id, device, WINNING_MAGNITUDE)
    n_flip = sum(1 for r in real_results if r["flipped_to_correct"])
    print(f"  REAL: {n_flip}/{len(real_results)} flipped to correct")

    print(f"\nRunning CONTROL (magnitude={WINNING_MAGNITUDE})...")
    control_results = run_full(model, tokenizer, control_hooks, queries, dataset_by_id, retrieval_by_id, device, WINNING_MAGNITUDE)
    n_flip_c = sum(1 for r in control_results if r["flipped_to_correct"])
    print(f"  CONTROL: {n_flip_c}/{len(control_results)} flipped to correct")

    for h in real_handles + control_handles:
        h.remove()

    print("\n" + "=" * 70)
    print("STATISTICAL TEST")
    print("=" * 70)
    pvalue = run_mcnemar(real_results, control_results)

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "magnitude": WINNING_MAGNITUDE,
            "real_target_heads": real_target_heads,
            "control_target_heads": random_target_heads,
            "real_results": real_results,
            "control_results": control_results,
            "pvalue": pvalue,
        }, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
