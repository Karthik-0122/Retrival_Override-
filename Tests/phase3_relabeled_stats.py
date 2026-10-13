"""
Phase 3 re-analysis with corrected labels. No GPU.

ROI was measured before the model generated anything, so the ROI values
themselves are unaffected by the labeling problem -- only which group each
query falls into (override vs faithful) changes. This reruns the Phase 3
statistics twice on the SAME queries: once with the original strict labels,
once with the corrected full-answer labels, side by side.

Tests (per model):
  1. Per-layer difference in ROI between override and faithful
     (Welch t-test on each layer's head-averaged ROI), FDR-corrected.
  2. The same, controlling for answer length (regression: ROI ~ override +
     answer length), FDR-corrected. This is the test that defined your
     validated layers.
  3. Per-head differences, FDR-corrected across all heads.
  4. AUROC: how well head-averaged ROI over the ORIGINAL validated layers
     separates override from faithful. Optimistic (same data), but comparable
     across the two labelings.

Both columns use the same queries and exclude ConFiQA, so the only thing that
differs between them is the labels. Your originally reported Phase 3 counts
may differ slightly if they included ConFiQA or other queries.

Run from repo root (after relabel_full_answer.py):
    python Tests/phase3_relabeled_stats.py

Writes:
    data/final/Phase_03/phase3_relabeled_analysis.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
import statsmodels.api as sm

ROI_CANDIDATES = ["data/final/Phase_03/phase3_roi_results.jsonl",
                  "data/final/phase3_roi_results.jsonl"]
ORIGINAL_LAYERS = {"gemma": [5, 7, 12] + list(range(27, 39)) + [40], "llama": [2, 3, 4]}
ALPHA = 0.05


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def bh_fdr(pvals, alpha=ALPHA):
    """Benjamini-Hochberg. Returns (q-values, significant flags)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [1.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvals[i] * m / rank)
        q[i] = running
    return q, [qi <= alpha for qi in q]


def model_of(row):
    name = str(row.get("model", "")).lower()
    if "gemma" in name:
        return "gemma"
    if "llama" in name:
        return "llama"
    return None


def find_roi_key(row):
    if "divergence_by_layer" in row:
        return "divergence_by_layer"
    for k, v in row.items():
        if (isinstance(v, list) and v and isinstance(v[0], list) and v[0]
                and isinstance(v[0][0], (int, float))):
            return k
    return None


def layer_ttests(L, y):
    p = []
    for l in range(L.shape[1]):
        a, b = L[y, l], L[~y, l]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        p.append(float(stats.ttest_ind(a, b, equal_var=False).pvalue)
                 if len(a) > 1 and len(b) > 1 else 1.0)
    return p


def layer_length_controlled(L, y, length):
    p = []
    for l in range(L.shape[1]):
        yv = L[:, l]
        ok = ~np.isnan(yv)
        design = sm.add_constant(np.column_stack([y[ok].astype(float), length[ok]]))
        p.append(float(sm.OLS(yv[ok], design).fit().pvalues[1]))
    return p


def head_ttests(X, y):
    p = []
    for l in range(X.shape[1]):
        for h in range(X.shape[2]):
            a, b = X[y, l, h], X[~y, l, h]
            a, b = a[~np.isnan(a)], b[~np.isnan(b)]
            p.append(float(stats.ttest_ind(a, b, equal_var=False).pvalue)
                     if len(a) > 1 and len(b) > 1 else 1.0)
    return p


def auroc(score, y):
    ok = ~np.isnan(score)
    pos, neg = score[ok & y], score[ok & ~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def analyze(X, L, y, length, model):
    p_raw = layer_ttests(L, y)
    _, sig_fdr = bh_fdr(p_raw)
    p_len = layer_length_controlled(L, y, length)
    _, sig_len = bh_fdr(p_len)
    p_head = head_ttests(X, y)
    _, sig_head = bh_fdr(p_head)
    layers = [l for l in ORIGINAL_LAYERS[model] if l < L.shape[1]]
    block = np.nanmean(L[:, layers], axis=1)
    return {
        "n_override": int(y.sum()), "n_faithful": int((~y).sum()),
        "layers_sig_raw": int(sum(p < ALPHA for p in p_raw)),
        "layers_sig_fdr": int(sum(sig_fdr)),
        "layers_length_controlled": [l for l, s in enumerate(sig_len) if s],
        "heads_sig_fdr": int(sum(sig_head)),
        "auroc_original_layers": auroc(block, y),
        "p_layer_raw": p_raw, "p_layer_length_controlled": p_len,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", default=None)
    ap.add_argument("--dataset", default="data/final/Phase_02/analysis_dataset_relabeled.jsonl")
    ap.add_argument("--out", default="data/final/Phase_03/phase3_relabeled_analysis.json")
    args = ap.parse_args()

    roi_path = args.roi or next((c for c in ROI_CANDIDATES if Path(c).exists()), None)
    if roi_path is None:
        sys.exit(f"ROI results not found. Tried: {ROI_CANDIDATES}. Pass --roi <path>.")
    rows = load_jsonl(roi_path)
    key = find_roi_key(rows[0])
    if key is None or model_of(rows[0]) is None:
        sys.exit(f"Couldn't read the ROI file format. First row keys: {list(rows[0].keys())} "
                 f"-- paste this line so the script can be adjusted.")
    print(f"Using ROI field '{key}' from {roi_path}")

    dataset = {r["query_id"]: r for r in load_jsonl(args.dataset)}
    results = {}

    for model in ("gemma", "llama"):
        X, lengths, old, new = [], [], [], []
        for r in rows:
            if model_of(r) != model:
                continue
            d = dataset.get(r["query_id"])
            if not d or d.get("source_category") == "confiqa":
                continue
            if d.get(f"{model}_relabel_status") != "ok":
                continue
            o, n = d.get(f"{model}_label"), d.get(f"{model}_label_full")
            if o not in ("override", "faithful") or n not in ("override", "faithful"):
                continue
            X.append(np.array(r[key], dtype=float))
            lengths.append(len((d.get(f"{model}_answer") or "").split()))
            old.append(o == "override")
            new.append(n == "override")
        if not X:
            print(f"\n{model.upper()}: no usable rows")
            continue

        X = np.stack(X)                       # (queries, layers, heads)
        L = np.nanmean(X, axis=2)             # head-averaged ROI per layer
        length = np.array(lengths, dtype=float)
        old, new = np.array(old), np.array(new)
        n_layers, n_heads = X.shape[1], X.shape[2]

        a = analyze(X, L, old, length, model)
        b = analyze(X, L, new, length, model)
        changed = int((old & ~new).sum())
        results[model] = {"n_queries": len(old), "labels_changed": changed,
                          "original_labels": a, "corrected_labels": b}

        print(f"\n{'=' * 72}\n{model.upper()}: {len(old)} queries (same set in both columns), "
              f"{changed} relabeled override -> faithful\n{'=' * 72}")
        print(f"{'':<40}{'original':>15}{'corrected':>15}")
        print(f"{'override / faithful':<40}{a['n_override']:>7} / {a['n_faithful']:<5}"
              f"{b['n_override']:>7} / {b['n_faithful']:<5}")
        print(f"{'layers significant (raw)':<40}{a['layers_sig_raw']:>12}/{n_layers}{b['layers_sig_raw']:>12}/{n_layers}")
        print(f"{'layers significant (FDR)':<40}{a['layers_sig_fdr']:>12}/{n_layers}{b['layers_sig_fdr']:>12}/{n_layers}")
        print(f"{'layers surviving length control + FDR':<40}{len(a['layers_length_controlled']):>12}/{n_layers}"
              f"{len(b['layers_length_controlled']):>12}/{n_layers}")
        print(f"{'heads significant (FDR)':<40}{a['heads_sig_fdr']:>10}/{n_layers * n_heads}"
              f"{b['heads_sig_fdr']:>10}/{n_layers * n_heads}")
        print(f"{'AUROC, original validated layers':<40}{a['auroc_original_layers']:>15.3f}{b['auroc_original_layers']:>15.3f}")

        la, lb = set(a["layers_length_controlled"]), set(b["layers_length_controlled"])
        print(f"\nLayers surviving length control + FDR:")
        print(f"  original labels:   {sorted(la)}")
        print(f"  corrected labels:  {sorted(lb)}")
        print(f"  in both:           {sorted(la & lb)}")
        print(f"  your reported set: {ORIGINAL_LAYERS[model]}")

    print("\nHow to read this:")
    print("  - Corrected layers keep the core of your reported set -> the localization holds.")
    print("  - Corrected labels give MORE significant layers or a higher AUROC -> label noise was")
    print("    hiding the signal (check Llama especially).")
    print("  - Most layers vanish with corrected labels -> the original signal partly tracked the")
    print("    mislabeled, wordier answers, and Phase 3 needs to be restated.")
    print("  - AUROC below 0.5 means override has LOWER ROI; distance from 0.5 is the strength.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()