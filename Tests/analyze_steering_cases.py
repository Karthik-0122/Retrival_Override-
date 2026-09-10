"""
Analyzes the steering confirmation results: what distinguishes the
cases that flipped to correct from the ones that didn't? Uses data
already collected (PCS gap, answer length, source category) -- no GPU
needed.

Run from repo root:
    python Tests/analyze_steering_cases.py

Requires:
    data/final/Phase_04/steering_confirmation_results.json
    data/final/Phase_02/analysis_dataset.jsonl
"""

import json
from collections import Counter

import numpy as np
from scipy import stats

RESULTS_PATH = "data/final/Phase_04/steering_confirmation_results.json"
DATASET_PATH = "data/final/Phase_02/analysis_dataset.jsonl"


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    results = json.load(open(RESULTS_PATH))
    dataset = {r["query_id"]: r for r in load_jsonl(DATASET_PATH)}

    real_results = results["real_results"]
    flipped_ids = {r["query_id"] for r in real_results if r["flipped_to_correct"]}
    non_flipped_ids = {r["query_id"] for r in real_results if not r["flipped_to_correct"]}

    print(f"Flipped: {len(flipped_ids)}  Non-flipped: {len(non_flipped_ids)}\n")

    print("=" * 70)
    print("FLIPPED CASES -- full detail")
    print("=" * 70)
    for qid in flipped_ids:
        row = dataset.get(qid)
        if not row:
            continue
        print(f"\n  {qid}")
        print(f"    question: {row.get('question')}")
        print(f"    source: {row.get('source_category')}")
        print(f"    gold answer: {row.get('gold_answer_text')}")
        print(f"    gemma_answer (original, with context): {row.get('gemma_answer')}")
        print(f"    pcs_gemma: {row.get('pcs_gemma')}")
        print(f"    popularity_tier: {row.get('popularity_tier')}")

    def get_stats(ids):
        pcs_vals, len_vals = [], []
        for qid in ids:
            row = dataset.get(qid)
            if not row:
                continue
            pcs = row.get("pcs_gemma")
            if pcs is not None:
                pcs_vals.append(abs(pcs))
            ans = row.get("gemma_answer") or ""
            len_vals.append(len(ans.split()))
        return pcs_vals, len_vals

    flip_pcs, flip_len = get_stats(flipped_ids)
    non_flip_pcs, non_flip_len = get_stats(non_flipped_ids)

    print("\n" + "=" * 70)
    print("NUMERIC COMPARISON")
    print("=" * 70)
    print(f"\n|PCS| (parametric confidence gap):")
    print(f"  flipped:     mean={np.mean(flip_pcs):.3f}  (n={len(flip_pcs)})")
    print(f"  non-flipped: mean={np.mean(non_flip_pcs):.3f}  (n={len(non_flip_pcs)})")
    if len(flip_pcs) >= 2 and len(non_flip_pcs) >= 2:
        t, p = stats.ttest_ind(flip_pcs, non_flip_pcs)
        print(f"  t-test p={p:.4f}")

    print(f"\nOriginal answer length (words):")
    print(f"  flipped:     mean={np.mean(flip_len):.2f}  (n={len(flip_len)})")
    print(f"  non-flipped: mean={np.mean(non_flip_len):.2f}  (n={len(non_flip_len)})")
    if len(flip_len) >= 2 and len(non_flip_len) >= 2:
        t, p = stats.ttest_ind(flip_len, non_flip_len)
        print(f"  t-test p={p:.4f}")

    print(f"\nSource category breakdown:")
    flip_cats = Counter(dataset[qid].get("source_category") for qid in flipped_ids if qid in dataset)
    non_flip_cats = Counter(dataset[qid].get("source_category") for qid in non_flipped_ids if qid in dataset)
    all_cats = set(flip_cats) | set(non_flip_cats)
    for cat in sorted(all_cats):
        f, nf = flip_cats.get(cat, 0), non_flip_cats.get(cat, 0)
        total = f + nf
        rate = f / total * 100 if total else 0
        print(f"  {cat:20s}: {f}/{total} flipped ({rate:.1f}%)")

    print("\nRead: if flipped cases cluster on low PCS gap, short answers, or one")
    print("specific source category, that tells you WHICH kind of override case")
    print("this mechanism handles -- a more precise, more citable finding than a")
    print("flat percentage.")


if __name__ == "__main__":
    main()
