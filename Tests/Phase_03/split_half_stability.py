"""
Split-half stability check for Test 1 (faithful vs override, per layer).
Randomly splits the query pool in half, reruns the per-layer t-test on
each half independently, and checks whether the significant-layer set
is stable across the split — or whether it's sensitive to which
queries happen to land in which half.

Run from repo root:
    python tests/split_half_stability.py
"""

import json
import random
import numpy as np
from scipy import stats

random.seed(42)
np.random.seed(42)

RESULTS_PATH = "data/final/phase3_roi_results.jsonl"
LABELS_PATH = "data/final/analysis_dataset.jsonl"


def load_results():
    rows = []
    with open(RESULTS_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_labels():
    # maps query_id -> {'gemma': 'faithful'/'override', 'llama': 'faithful'/'override'}
    label_map = {}
    with open(LABELS_PATH) as f:
        for line in f:
            r = json.loads(line)
            label_map[r["query_id"]] = {
                "gemma": r.get("gemma_label"),
                "llama": r.get("llama_label"),
            }
    return label_map


def layer_mean(row, layer_idx):
    # mean divergence across heads for a given layer, for this query
    return float(np.mean(row["divergence_by_layer"][layer_idx]))


def run_test1(rows, label_map, model):
    n_layers = len(rows[0]["divergence_by_layer"])
    sig_layers = []
    for layer_idx in range(n_layers):
        faithful_vals = [
            layer_mean(r, layer_idx) for r in rows
            if label_map[r["query_id"]][model] == "faithful"
        ]
        override_vals = [
            layer_mean(r, layer_idx) for r in rows
            if label_map[r["query_id"]][model] == "override"
        ]
        if len(faithful_vals) < 2 or len(override_vals) < 2:
            continue
        _, p = stats.ttest_ind(faithful_vals, override_vals)
        if p < 0.05:
            sig_layers.append(layer_idx)
    return set(sig_layers)


def main():
    rows = load_results()
    label_map = load_labels()

    for model in ["gemma", "llama"]:
        model_rows = [r for r in rows if r["model"] == model and r["query_id"] in label_map]
        query_ids = list({r["query_id"] for r in model_rows})
        random.shuffle(query_ids)
        half = len(query_ids) // 2
        half_a_ids, half_b_ids = set(query_ids[:half]), set(query_ids[half:])

        rows_a = [r for r in model_rows if r["query_id"] in half_a_ids]
        rows_b = [r for r in model_rows if r["query_id"] in half_b_ids]

        sig_a = run_test1(rows_a, label_map, model)
        sig_b = run_test1(rows_b, label_map, model)
        full_sig = run_test1(model_rows, label_map, model)

        overlap = sig_a & sig_b
        union = sig_a | sig_b
        overlap_pct = len(overlap) / len(union) * 100 if union else 0

        print(f"\n{model.upper()} (full N={len(model_rows)}, half A N={len(rows_a)}, half B N={len(rows_b)})")
        print(f"  Full-pool significant layers: {sorted(full_sig)}")
        print(f"  Half A significant layers:    {sorted(sig_a)}")
        print(f"  Half B significant layers:    {sorted(sig_b)}")
        print(f"  A∩B overlap: {sorted(overlap)}  ({overlap_pct:.0f}% of union)")
        print(f"  A∩full: {len(sig_a & full_sig)}/{len(sig_a)}   B∩full: {len(sig_b & full_sig)}/{len(sig_b)}")


if __name__ == "__main__":
    main()