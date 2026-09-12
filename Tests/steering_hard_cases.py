"""
Targeted follow-up to the validated steering result (p=0.031, 6/100 flipped
at magnitude=10, top-8 DLA heads). Tests whether the 94 UNFLIPPED cases
(high parametric-confidence cases, per the case analysis) can be reached
by either MORE heads or HIGHER magnitude -- or whether there's a real
ceiling, consistent with yesterday's finding that this layer block
overlaps fact STORAGE, not just the override decision.

Efficient design: only re-tests the 94 cases that DIDN'T flip already --
no need to re-spend compute on the 6 that already work.

Run from repo root:
    python Tests/steering_hard_cases.py

Requires:
    data/final/Phase_04/steering_confirmation_results.json  (to know which cases are "hard")
    data/final/Phase_04/dla_results_bf16.json
    data/final/Phase_02/analysis_dataset.jsonl
    data/final/Phase_02/retrieval_results.jsonl

Writes:
    data/final/Phase_04/steering_hard_cases_results.json
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.path.insert(0, "Tests")
from steering_screen import (
    MODEL_PATH, HEAD_DIM, DATASET_PATH, RETRIEVAL_PATH, TEST_SAMPLE_PATH,
    load_jsonl, build_prompt, get_gold_first_token_id,
    SteeringHook, compute_direction_means, QUANT_CONFIG,
)

CONFIRMATION_RESULTS_PATH = "data/final/Phase_04/steering_confirmation_results.json"
DLA_PATH = "data/final/Phase_04/dla_results_bf16.json"
OUT_PATH = "data/final/Phase_04/steering_hard_cases_results.json"

HEAD_COUNTS_TO_TEST = [8, 16]  # top-8 (already confirmed) vs top-16
MAGNITUDES_TO_TEST = [10.0, 20.0, 30.0]


def load_top_negative_heads(n, model_key="gemma"):
    data = json.load(open(DLA_PATH))
    contribs = data[model_key]["contributions"]
    ranked = sorted(contribs.items(), key=lambda x: x[1])[:n]
    return [tuple(map(int, k.split("_"))) for k, _ in ranked]


def get_hard_case_ids():
    """The cases that did NOT flip in the confirmed run -- these are the
    ones worth re-testing, not the ones that already work."""
    data = json.load(open(CONFIRMATION_RESULTS_PATH))
    return [r["query_id"] for r in data["real_results"] if not r["flipped_to_correct"]]


def run_condition(model, tokenizer, hooks, magnitude, query_objs, dataset_by_id, retrieval_by_id, device):
    for h in hooks:
        h.magnitude = magnitude

    n_flip = n_total = 0
    for q in tqdm(query_objs, desc=f"heads={len(hooks)} mag={magnitude}"):
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

        n_total += 1
        if (not baseline_correct) and steered_correct:
            n_flip += 1

    return n_flip, n_total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=QUANT_CONFIG, device_map=device, attn_implementation="eager",
    )
    model.eval()

    hard_ids = set(get_hard_case_ids())
    print(f"Testing {len(hard_ids)} hard (previously unflipped) cases")

    test_sample = json.load(open(TEST_SAMPLE_PATH))
    all_override_queries = test_sample["gemma"]["override_cases"]
    hard_query_objs = [q for q in all_override_queries if q["query_id"] in hard_ids]
    print(f"Matched {len(hard_query_objs)} query objects from the test sample")

    dataset_records = load_jsonl(DATASET_PATH)
    retrieval_records = load_jsonl(RETRIEVAL_PATH)
    dataset_by_id = {r["query_id"]: r for r in dataset_records}
    retrieval_by_id = {r["query_id"]: r for r in retrieval_records}

    max_heads_needed = max(HEAD_COUNTS_TO_TEST)
    all_heads = load_top_negative_heads(max_heads_needed)
    directions = compute_direction_means(model, tokenizer, all_heads, device)

    results = []
    print(f"\n{'Heads':>6} {'Magnitude':>10} {'Flips':>12}")
    for n_heads in HEAD_COUNTS_TO_TEST:
        target_heads = all_heads[:n_heads]
        hooks, handles = [], []
        for (l, h) in target_heads:
            hook = SteeringHook(h, HEAD_DIM, directions[(l, h)], 0.0)
            handles.append(hook.register(model.model.layers[l]))
            hooks.append(hook)

        for mag in MAGNITUDES_TO_TEST:
            n_flip, n_total = run_condition(model, tokenizer, hooks, mag, hard_query_objs,
                                              dataset_by_id, retrieval_by_id, device)
            print(f"{n_heads:>6} {mag:>10.1f} {f'{n_flip}/{n_total}':>12}")
            results.append({"n_heads": n_heads, "magnitude": mag, "flips": n_flip, "total": n_total})

        for h in handles:
            h.remove()

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"target_heads_by_count": {n: all_heads[:n] for n in HEAD_COUNTS_TO_TEST}, "results": results}, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    print("\nRead: if ALL rows stay near 0, that's a real ceiling at this location --")
    print("consistent with the decision being made earlier in the network than")
    print("where these heads sit. If any row shows real flips, that lever")
    print("(more heads, or higher magnitude) is worth scaling up properly.")


if __name__ == "__main__":
    main()
