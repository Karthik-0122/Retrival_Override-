"""
Length-controlled re-test of Test 1 (faithful vs override, per layer).

The plain t-test can't tell 'override mechanism' apart from 'whatever
correlates with answer length' since override answers are ~3-4x longer
than faithful ones (confirmed: p<1e-49 both models). This script instead
fits, per layer, per model:

    layer_mean_divergence ~ label + answer_length

and reports the p-value on the label coefficient -- i.e. does faithful
vs override still predict divergence AFTER answer length is partialed
out. This is a much stronger test than a plain group comparison.

Run from repo root:
    python tests/length_controlled_test.py

Requires: pip install statsmodels (already installed for FDR step)
"""

import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

RESULTS_PATH = "data/final/phase3_roi_results.jsonl"
LABELS_PATH = "data/final/analysis_dataset.jsonl"

# Layers already flagged as the robust/interesting core from earlier steps --
# split-half stability + effect-size triage. Edit if your own lists differ.
FLAGGED_CORE = {
    "gemma": [7, 13, 18, 19, 22, 30, 31, 33, 34, 35, 36, 38, 40],
    "llama": [2, 3, 4, 7, 9, 27, 28],
}


def load_results():
    rows = []
    with open(RESULTS_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_meta():
    """query_id -> {gemma_label, llama_label, gemma_len, llama_len}"""
    meta = {}
    with open(LABELS_PATH) as f:
        for line in f:
            r = json.loads(line)
            meta[r["query_id"]] = {
                "gemma_label": r.get("gemma_label"),
                "llama_label": r.get("llama_label"),
                "gemma_len": len((r.get("gemma_answer") or "").split()),
                "llama_len": len((r.get("llama_answer") or "").split()),
            }
    return meta


def layer_mean(row, layer_idx):
    return float(np.mean(row["divergence_by_layer"][layer_idx]))


def run_model(rows, meta, model):
    label_key = f"{model}_label"
    len_key = f"{model}_len"

    model_rows = [r for r in rows if r["model"] == model and r["query_id"] in meta]
    n_layers = len(model_rows[0]["divergence_by_layer"])

    print(f"\n{'='*70}\n{model.upper()} (N={len(model_rows)})\n{'='*70}")

    survived = []
    for layer_idx in range(n_layers):
        df = pd.DataFrame({
            "divergence": [layer_mean(r, layer_idx) for r in model_rows],
            "label": [meta[r["query_id"]][label_key] for r in model_rows],
            "length": [meta[r["query_id"]][len_key] for r in model_rows],
        })
        df = df.dropna()
        if df["label"].nunique() < 2:
            continue

        try:
            model_fit = smf.ols("divergence ~ C(label) + length", data=df).fit()
            p_label = model_fit.pvalues.get("C(label)[T.override]")
            if p_label is None:
                # category ordering might differ
                cand = [k for k in model_fit.pvalues.index if "label" in k]
                p_label = model_fit.pvalues[cand[0]] if cand else float("nan")
        except Exception as e:
            print(f"  layer {layer_idx}: regression failed ({e})")
            continue

        raw_flag = layer_idx in FLAGGED_CORE.get(model, [])
        sig = p_label < 0.05
        marker = "  <-- flagged core layer" if raw_flag else ""
        status = "SURVIVES" if sig else "drops out"
        print(f"  layer {layer_idx:2d}: length-controlled p={p_label:.4f}  [{status}]{marker}")

        if raw_flag:
            survived.append((layer_idx, sig))

    n_core = len(survived)
    n_surv = sum(1 for _, s in survived if s)
    print(f"\n  Of {n_core} previously-flagged core layers: {n_surv}/{n_core} survive length control")


def main():
    rows = load_results()
    meta = load_meta()
    for model in ["gemma", "llama"]:
        run_model(rows, meta, model)


if __name__ == "__main__":
    main()