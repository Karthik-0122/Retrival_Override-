"""
FDR correction on the length-controlled per-layer p-values.
Run from repo root: python tests/length_controlled_fdr.py
"""
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

RESULTS_PATH = "data/final/phase3_roi_results.jsonl"
LABELS_PATH = "data/final/analysis_dataset.jsonl"
OUT_PATH = "data/final/phase3_length_controlled.json"

def load_results():
    return [json.loads(l) for l in open(RESULTS_PATH)]

def load_meta():
    meta = {}
    for line in open(LABELS_PATH):
        r = json.loads(line)
        meta[r["query_id"]] = {
            "gemma_label": r.get("gemma_label"), "llama_label": r.get("llama_label"),
            "gemma_len": len((r.get("gemma_answer") or "").split()),
            "llama_len": len((r.get("llama_answer") or "").split()),
        }
    return meta

def layer_mean(row, layer_idx):
    return float(np.mean(row["divergence_by_layer"][layer_idx]))

def run_model(rows, meta, model):
    label_key, len_key = f"{model}_label", f"{model}_len"
    model_rows = [r for r in rows if r["model"] == model and r["query_id"] in meta]
    n_layers = len(model_rows[0]["divergence_by_layer"])

    layer_results = []
    for layer_idx in range(n_layers):
        df = pd.DataFrame({
            "divergence": [layer_mean(r, layer_idx) for r in model_rows],
            "label": [meta[r["query_id"]][label_key] for r in model_rows],
            "length": [meta[r["query_id"]][len_key] for r in model_rows],
        }).dropna()
        fit = smf.ols("divergence ~ C(label) + length", data=df).fit()
        cand = [k for k in fit.pvalues.index if "label" in k]
        p = fit.pvalues[cand[0]] if cand else float("nan")
        layer_results.append({"layer": layer_idx, "p_value": float(p)})

    pvals = [r["p_value"] for r in layer_results]
    rejected, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    for r, rej, padj in zip(layer_results, rejected, p_adj):
        r["fdr_significant"] = bool(rej)
        r["fdr_p"] = float(padj)

    n_raw = sum(1 for r in layer_results if r["p_value"] < 0.05)
    n_fdr = sum(rejected)
    print(f"{model.upper()}: {n_raw}/{n_layers} raw -> {n_fdr}/{n_layers} survive FDR")
    survivors = [r["layer"] for r in layer_results if r["fdr_significant"]]
    print(f"  Surviving layers: {survivors}")
    return layer_results

def main():
    rows, meta = load_results(), load_meta()
    output = {}
    for model in ["gemma", "llama"]:
        output[model] = run_model(rows, meta, model)
    json.dump(output, open(OUT_PATH, "w"), indent=2)
    print(f"\nWrote {OUT_PATH}")

if __name__ == "__main__":
    main()