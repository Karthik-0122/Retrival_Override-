"""
Per-head extension of Test 1 (faithful vs override divergence).
Instead of collapsing each layer to a head-mean before testing, this
runs one t-test per (layer, head) pair, then applies FDR correction
across the full set of tests for that model (since we're now running
hundreds of tests per model, not dozens).

Run from repo root:
    python tests/per_head_analysis.py

Writes: data/final/phase3_per_head_analysis.json
"""

import json
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

RESULTS_PATH = "data/final/phase3_roi_results.jsonl"
LABELS_PATH = "data/final/analysis_dataset.jsonl"
OUT_PATH = "data/final/phase3_per_head_analysis.json"


def load_results():
    rows = []
    with open(RESULTS_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_labels():
    label_map = {}
    with open(LABELS_PATH) as f:
        for line in f:
            r = json.loads(line)
            label_map[r["query_id"]] = {
                "gemma": r.get("gemma_label"),
                "llama": r.get("llama_label"),
            }
    return label_map


def run_per_head(rows, label_map, model):
    n_layers = len(rows[0]["divergence_by_layer"])
    n_heads = len(rows[0]["divergence_by_layer"][0])

    faithful_rows = [r for r in rows if label_map[r["query_id"]][model] == "faithful"]
    override_rows = [r for r in rows if label_map[r["query_id"]][model] == "override"]

    results = []  # flat list, one entry per (layer, head)
    pvals = []

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            faithful_vals = [r["divergence_by_layer"][layer_idx][head_idx] for r in faithful_rows]
            override_vals = [r["divergence_by_layer"][layer_idx][head_idx] for r in override_rows]

            mean_f = float(np.mean(faithful_vals))
            mean_o = float(np.mean(override_vals))
            _, p = stats.ttest_ind(faithful_vals, override_vals)

            results.append({
                "layer": layer_idx,
                "head": head_idx,
                "mean_faithful": mean_f,
                "mean_override": mean_o,
                "effect": mean_o - mean_f,
                "p_value": float(p),
            })
            pvals.append(float(p))

    # FDR correction across ALL layer x head tests for this model
    rejected, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    for r, rej, p in zip(results, rejected, p_adj):
        r["fdr_significant"] = bool(rej)
        r["fdr_p"] = float(p)

    n_sig_raw = sum(1 for r in results if r["p_value"] < 0.05)
    n_sig_fdr = sum(rejected)
    print(f"{model.upper()}: {n_layers} layers x {n_heads} heads = {len(results)} tests")
    print(f"  raw p<0.05: {n_sig_raw}/{len(results)}")
    print(f"  FDR significant: {n_sig_fdr}/{len(results)}")

    # summarize which heads survive FDR, grouped by layer, sorted by |effect|
    surviving = [r for r in results if r["fdr_significant"]]
    surviving.sort(key=lambda r: -abs(r["effect"]))
    print(f"  Top 10 surviving (layer, head) by |effect|:")
    for r in surviving[:10]:
        print(f"    layer {r['layer']:2d} head {r['head']:2d}: effect={r['effect']:+.5f} fdr_p={r['fdr_p']:.2e}")

    return results


def main():
    rows = load_results()
    label_map = load_labels()

    output = {}
    for model in ["gemma", "llama"]:
        model_rows = [r for r in rows if r["model"] == model and r["query_id"] in label_map]
        print(f"\n{'='*70}")
        results = run_per_head(model_rows, label_map, model)
        output[model] = results

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Wrote per-head results to {OUT_PATH}")


if __name__ == "__main__":
    main()