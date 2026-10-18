"""
Experiment 3 -- Failure-type-specific ROI re-analysis. No GPU.

ROI (attention shift on the question tokens when the passage is added) was
measured before generation for every question, so nothing is recomputed:
this regroups the existing per-head ROI by the failure types from
Experiments 1-2 and reruns the Phase 3 statistics.

Comparisons (guide's spec):
  primary     F1 genuine override vs faithful
  secondary   F1 vs F2 (wrong extraction), F1 vs F3 (evidence absent),
              F1 vs F4 (partial)
  plotted     every type vs faithful (layer-by-failure-type profile)

For each comparison with at least MIN_N cases per group:
  per layer   Welch t, Hedges' g with bootstrap 95% CI, Benjamini-Hochberg FDR
              (same correction as the original study)
  length      OLS  ROI ~ group + answer length  per layer, FDR (original approach)
  per head    Welch t per (layer, head), FDR over all heads, Hedges' g map
  stability   split-half: per-layer t in two random halves, Spearman r across
              layers and overlap of the top-5 layers (100 splits)
  prediction  AUROC of mean ROI over the ORIGINAL validated layers (fixed in
              advance) and a split-half AUROC (layers chosen on one half,
              scored on the other) -- does ROI predict individual cases?

Label sources, side by side:
  hybrid      human label where validated, automatic label elsewhere (PRIMARY)
  human       human-validated cases only
  auto        Experiment 1 automatic labels (unverified, larger n)
  original    the old strict override/faithful labels -- for the
              original-vs-corrected localization comparison

'Faithful' = correct under both the strict and the full-answer scoring
(so F5 'actually correct' cases are not mixed into the control group).
ConFiQA is excluded (ROI was not measured for it).

Groups smaller than MIN_N are reported descriptively only -- no tests.
ROI is correlational evidence; nothing here is a causal claim.

Run from repo root:
    python Tests/experiment3_roi_by_failure_type.py

Writes to data/final/Phase_05/exp3/:
    exp3_results.json                        everything
    exp3_layers_<model>_<source>.csv         per-layer table, every comparison
    exp3_candidate_heads_<model>.csv         top heads, F1 vs faithful (primary)
    exp3_layer_profile_<model>.png           g by layer, each type vs faithful
    exp3_head_heatmap_<model>.png            g per head, F1 vs faithful
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
import statsmodels.api as sm

MODELS = ("gemma", "llama")
TYPES = ["F1", "F2", "F3", "F4", "F5", "F6"]
MIN_N = 5
N_BOOT = 1000
N_SPLITS = 100
TOP_K = 5
ALPHA = 0.05
SEED = 2026
ORIGINAL_LAYERS = {"gemma": [5, 7, 12] + list(range(27, 39)) + [40], "llama": [2, 3, 4]}
COMPARISONS = [("F1", "faithful"), ("F1", "F2"), ("F1", "F3"), ("F1", "F4")]
PROFILE_TYPES = ["F1", "F2", "F3", "F4"]
# Validated categorical slots 1-4 (light surface); diverging blue / gray / red.
SERIES = {"F1": "#2a78d6", "F2": "#eb6834", "F3": "#1baf7a", "F4": "#eda100"}
MARKERS = {"F1": "o", "F2": "s", "F3": "^", "F4": "D"}
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]
INK, MUTED, GRID, SURFACE = "#1f1f1d", "#6b6b66", "#e4e3df", "#fcfcfb"
TYPE_NAMES = {"F1": "F1 genuine override", "F2": "F2 wrong extraction",
              "F3": "F3 evidence absent", "F4": "F4 partial"}


# ------------------------------------------------------------------ io
def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def model_of(name):
    n = str(name).lower()
    return "gemma" if "gemma" in n else ("llama" if "llama" in n else None)


def write_csv(p, rows):
    if not rows:
        return
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ stats
def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p[i] * m / rank)
        q[i] = running
    return q


def hedges_g(a, b, axis=0):
    """a: (na, ...), b: (nb, ...). Positive g = group a has higher ROI."""
    na, nb = a.shape[axis], b.shape[axis]
    va, vb = np.nanvar(a, axis=axis, ddof=1), np.nanvar(b, axis=axis, ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    j = 1 - 3 / (4 * (na + nb) - 9)
    with np.errstate(divide="ignore", invalid="ignore"):
        return j * (np.nanmean(a, axis=axis) - np.nanmean(b, axis=axis)) / sp


def welch_p(a, b, axis=0):
    return stats.ttest_ind(a, b, axis=axis, equal_var=False, nan_policy="omit").pvalue


def welch_t(a, b, axis=0):
    return stats.ttest_ind(a, b, axis=axis, equal_var=False, nan_policy="omit").statistic


def boot_g_ci(a, b, rng):
    """Bootstrap 95% CI of Hedges' g for every column at once."""
    gs = np.empty((N_BOOT,) + a.shape[1:])
    for i in range(N_BOOT):
        gs[i] = hedges_g(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))])
    return np.nanpercentile(gs, 2.5, axis=0), np.nanpercentile(gs, 97.5, axis=0)


def length_controlled_p(La, Lb, len_a, len_b):
    y_grp = np.r_[np.ones(len(La)), np.zeros(len(Lb))]
    length = np.r_[len_a, len_b]
    L = np.vstack([La, Lb])
    out = []
    for l in range(L.shape[1]):
        yv = L[:, l]
        ok = ~np.isnan(yv)
        X = sm.add_constant(np.column_stack([y_grp[ok], length[ok]]))
        out.append(float(sm.OLS(yv[ok], X).fit().pvalues[1]))
    return np.array(out)


def auroc(score_a, score_b):
    """P(random a-case has higher score than random b-case)."""
    a, b = score_a[~np.isnan(score_a)], score_b[~np.isnan(score_b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(u / (len(a) * len(b)))


def split_half(La, Lb, rng):
    """Per-layer t in two random halves: Spearman r, top-K overlap, and an
    out-of-half AUROC (top-K layers picked on half 1, scored on half 2)."""
    rs, jac, aucs = [], [], []
    for _ in range(N_SPLITS):
        ia, ib = rng.permutation(len(La)), rng.permutation(len(Lb))
        ha, hb = len(ia) // 2, len(ib) // 2
        a1, a2, b1, b2 = La[ia[:ha]], La[ia[ha:]], Lb[ib[:hb]], Lb[ib[hb:]]
        t1, t2 = welch_t(a1, b1), welch_t(a2, b2)
        ok = ~(np.isnan(t1) | np.isnan(t2))
        if ok.sum() > 2:
            rs.append(stats.spearmanr(t1[ok], t2[ok]).correlation)
        top1 = set(np.argsort(-np.abs(np.nan_to_num(t1)))[:TOP_K])
        top2 = set(np.argsort(-np.abs(np.nan_to_num(t2)))[:TOP_K])
        jac.append(len(top1 & top2) / len(top1 | top2))
        layers = sorted(top1)
        sign = np.sign(np.nanmean(t1[layers])) or 1.0
        aucs.append(auroc(sign * np.nanmean(a2[:, layers], 1), sign * np.nanmean(b2[:, layers], 1)))
    pct = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if v else [None, None]
    return {"spearman_r_mean": float(np.nanmean(rs)) if rs else None, "spearman_r_95": pct(rs),
            "top5_overlap_jaccard_mean": float(np.mean(jac)),
            "out_of_half_auroc_mean": float(np.nanmean(aucs)), "out_of_half_auroc_95": pct(aucs)}


def compare(X, lengths, groups, ga, gb, model, rng):
    ia, ib = [i for i, g in enumerate(groups) if g == ga], [i for i, g in enumerate(groups) if g == gb]
    res = {"a": ga, "b": gb, "n_a": len(ia), "n_b": len(ib)}
    if len(ia) == 0 or len(ib) == 0:
        res["status"] = "empty group"
        return res
    Xa, Xb = X[ia], X[ib]
    La, Lb = np.nanmean(Xa, 2), np.nanmean(Xb, 2)
    res["mean_roi_by_layer_a"] = np.nanmean(La, 0).tolist()
    res["mean_roi_by_layer_b"] = np.nanmean(Lb, 0).tolist()
    if len(ia) < MIN_N or len(ib) < MIN_N:
        res["status"] = f"descriptive only (need >= {MIN_N} per group)"
        res["g_by_layer"] = hedges_g(La, Lb).tolist() if min(len(ia), len(ib)) > 1 else None
        return res
    res["status"] = "tested"
    g = hedges_g(La, Lb)
    lo, hi = boot_g_ci(La, Lb, rng)
    p = welch_p(La, Lb)
    q = bh_fdr(p)
    p_len = length_controlled_p(La, Lb, lengths[ia], lengths[ib])
    q_len = bh_fdr(p_len)
    gh = hedges_g(Xa, Xb)
    ph = welch_p(Xa, Xb)
    qh = bh_fdr(np.nan_to_num(ph.ravel(), nan=1.0)).reshape(ph.shape)
    layers0 = [l for l in ORIGINAL_LAYERS[model] if l < La.shape[1]]
    res.update({
        "g_by_layer": g.tolist(), "g_ci_low": lo.tolist(), "g_ci_high": hi.tolist(),
        "p_layer": p.tolist(), "q_layer": q.tolist(),
        "p_layer_length_controlled": p_len.tolist(), "q_layer_length_controlled": q_len.tolist(),
        "layers_p05": [int(l) for l in np.where(p < ALPHA)[0]],
        "layers_fdr": [int(l) for l in np.where(q < ALPHA)[0]],
        "layers_fdr_length_controlled": [int(l) for l in np.where(q_len < ALPHA)[0]],
        "g_by_head": gh.tolist(), "p_by_head": ph.tolist(), "q_by_head": qh.tolist(),
        "heads_fdr": [[int(l), int(h)] for l, h in zip(*np.where(qh < ALPHA))],
        "auroc_original_layers": auroc(np.nanmean(La[:, layers0], 1), np.nanmean(Lb[:, layers0], 1)),
        "split_half": split_half(La, Lb, rng),
    })
    return res


# ------------------------------------------------------------------ labels
def build_labels(model, dataset, exp1, human):
    """Returns {source: {qid: group}} with groups F1..F6, 'faithful', 'override'."""
    auto = {e["question_id"]: e["failure_type"] for e in exp1 if e["model"] == model}
    hum = {h["question_id"]: h["human_type"] for h in human if h["model"] == model}
    faithful = {qid for qid, r in dataset.items()
                if r.get("source_category") != "confiqa" and r.get(f"{model}_relabel_status") == "ok"
                and r.get(f"{model}_label") == "faithful" and r.get(f"{model}_label_full") == "faithful"}
    out = {"auto": {**{q: "faithful" for q in faithful}, **auto},
           "hybrid": {**{q: "faithful" for q in faithful}, **auto, **hum},
           "human": {**{q: "faithful" for q in faithful}, **hum},
           "original": {}}
    for qid, r in dataset.items():
        if r.get("source_category") == "confiqa":
            continue
        if r.get(f"{model}_label") in ("override", "faithful"):
            out["original"][qid] = r[f"{model}_label"]
    return out


# ------------------------------------------------------------------ figures
def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_profile(model, comps, path, n_layers, source):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(n_layers)
    ax.axhline(0, color=MUTED, linewidth=1)
    ends = []
    for t in PROFILE_TYPES:
        c = comps.get(t)
        if not c or c.get("g_by_layer") is None:
            continue
        g = np.array(c["g_by_layer"], dtype=float)
        tested = c.get("status") == "tested"
        ax.plot(x, g, color=SERIES[t], linewidth=2, linestyle="-" if tested else ":",
                marker=MARKERS[t], markersize=4, markevery=3,
                label=f"{TYPE_NAMES[t]} (n={c['n_a']}{'' if tested else ', too small to test'})")
        if tested and t == "F1":
            sig = c["layers_fdr_length_controlled"]
            ax.scatter(sig, g[sig], s=70, facecolor="none", edgecolor=INK, linewidth=1.5, zorder=5)
        ends.append((g[~np.isnan(g)][-1] if np.any(~np.isnan(g)) else 0, t))
    lo, hi = ax.get_ylim()
    gap = 0.07 * (hi - lo)
    ends.sort()
    placed = []
    for y, t in ends:                                   # short direct labels, spaced apart
        y = max(y, placed[-1] + gap) if placed else y
        placed.append(y)
        ax.annotate(t, (n_layers - 1, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_ylim(lo, max(hi, placed[-1] + gap) if placed else hi)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5,
                    labelcolor=INK, handlelength=2.5)
    ax.set_xlim(-0.5, n_layers - 1 + 0.5)
    ax.set_xlabel("Layer", color=MUTED, fontsize=9.5)
    ax.set_ylabel("Hedges' g vs faithful  (+ = higher ROI)", color=MUTED, fontsize=9.5)
    src_note = ("human-validated labels where available" if source == "hybrid"
                else "AUTOMATIC labels (unverified) -- too few human-confirmed F1 to test")
    ax.set_title(f"{model.capitalize()}: ROI difference from faithful answers, by failure type\n{src_note}",
                 color=INK, fontsize=11, loc="left", pad=46)
    ax.text(0, -0.2, "Dotted = group too small to test. Circles = F1 layers significant after "
            "length control + FDR.", transform=ax.transAxes, fontsize=8, color=MUTED)
    fig.subplots_adjust(right=0.94, bottom=0.2, top=0.8)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_heatmap(model, comp, path, source):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    g = np.array(comp["g_by_head"], dtype=float)
    lim = float(np.nanmax(np.abs(g))) or 1.0
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    fig, ax = plt.subplots(figsize=(7, 8), facecolor=SURFACE)
    im = ax.imshow(g, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
    for l, h in comp.get("heads_fdr", []):
        ax.add_patch(plt.Rectangle((h - 0.5, l - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.5))
    ax.set_xlabel("Head", color=MUTED, fontsize=9.5)
    ax.set_ylabel("Layer", color=MUTED, fontsize=9.5)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Hedges' g, F1 vs faithful (+ = higher ROI in F1)", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    ax.set_title(f"{model.capitalize()}: per-head ROI effect, F1 (n={comp['n_a']}) vs faithful "
                 f"(n={comp['n_b']})\n{'AUTOMATIC labels (unverified). ' if source == 'auto' else ''}Outlined = significant after FDR", color=INK, fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", default="data/final/Phase_03/phase3_roi_results.jsonl")
    ap.add_argument("--dataset", default="data/final/Phase_02/analysis_dataset_relabeled.jsonl")
    ap.add_argument("--exp1", default="data/final/Phase_05/exp1_failure_types.jsonl")
    ap.add_argument("--human", default="data/final/Phase_05/exp2/validated_labels.jsonl")
    ap.add_argument("--outdir", default="data/final/Phase_05/exp3")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    roi = load_jsonl(args.roi)
    dataset = {r["query_id"]: r for r in load_jsonl(args.dataset)}
    exp1 = load_jsonl(args.exp1)
    human = load_jsonl(args.human) if Path(args.human).exists() else []
    if not human:
        print(f"!! {args.human} not found -- run experiment2_annotation.py validate first.")
        print("!! Continuing with automatic labels only ('human' and 'hybrid' = auto).\n")

    results = {"config": {"min_n": MIN_N, "n_boot": N_BOOT, "n_splits": N_SPLITS, "seed": SEED,
                          "original_layers": ORIGINAL_LAYERS, "human_labels_found": bool(human)}}
    for model in MODELS:
        rows = [r for r in roi if model_of(r.get("model")) == model and r["query_id"] in dataset]
        if not rows:
            continue
        X = np.array([np.array(r["divergence_by_layer"], dtype=float) for r in rows])   # (q, L, H)
        qids = [r["query_id"] for r in rows]
        lengths = np.array([len((dataset[q].get(f"{model}_answer") or "").split()) for q in qids], float)
        n_layers, n_heads = X.shape[1], X.shape[2]
        labels = build_labels(model, dataset, exp1, human)
        mres = {"n_layers": n_layers, "n_heads": n_heads, "sources": {}}
        print(f"\n{'=' * 78}\n{model.upper()}  ({len(rows)} questions with ROI, {n_layers} layers x {n_heads} heads)\n{'=' * 78}")

        for source in ("hybrid", "human", "auto", "original"):
            groups = [labels[source].get(q) for q in qids]
            counts = Counter(g for g in groups if g)
            sres = {"group_counts": dict(counts), "comparisons": {}, "profile": {}}
            comp_list = [("override", "faithful")] if source == "original" else COMPARISONS
            for ga, gb in comp_list:
                sres["comparisons"][f"{ga}_vs_{gb}"] = compare(X, lengths, groups, ga, gb, model, rng)
            if source != "original":
                for t in PROFILE_TYPES:
                    key = f"{t}_vs_faithful"
                    sres["profile"][t] = (sres["comparisons"][key] if key in sres["comparisons"]
                                          else compare(X, lengths, groups, t, "faithful", model, rng))
            mres["sources"][source] = sres

            # ---- printout ----
            print(f"\n[{source}] groups: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
            for key, c in sres["comparisons"].items():
                if c.get("status") != "tested":
                    print(f"  {key:<22} n={c['n_a']}/{c['n_b']}  -> {c.get('status')}")
                    continue
                sh = c["split_half"]
                print(f"  {key:<22} n={c['n_a']}/{c['n_b']}  layers p<.05: {len(c['layers_p05'])}  "
                      f"FDR: {c['layers_fdr']}  FDR+length: {c['layers_fdr_length_controlled']}  "
                      f"heads FDR: {len(c['heads_fdr'])}")
                g = np.array(c["g_by_layer"])
                top = np.argsort(-np.abs(np.nan_to_num(g)))[:3]
                print(f"  {'':<22} strongest layers (g [95% CI]): " + "; ".join(
                    f"L{l} {g[l]:+.2f} [{c['g_ci_low'][l]:+.2f},{c['g_ci_high'][l]:+.2f}]" for l in top))
                print(f"  {'':<22} AUROC original layers {c['auroc_original_layers']:.3f} | split-half: "
                      f"r={sh['spearman_r_mean']:.2f}, top-5 overlap {sh['top5_overlap_jaccard_mean']:.2f}, "
                      f"out-of-half AUROC {sh['out_of_half_auroc_mean']:.3f}")

            # ---- per-layer CSV ----
            table = []
            for l in range(n_layers):
                row = {"layer": l, "normalized_depth": round(l / (n_layers - 1), 3)}
                for key, c in sres["comparisons"].items():
                    if c.get("status") == "tested":
                        row.update({f"{key}_g": round(c["g_by_layer"][l], 4),
                                    f"{key}_ci_low": round(c["g_ci_low"][l], 4),
                                    f"{key}_ci_high": round(c["g_ci_high"][l], 4),
                                    f"{key}_p": c["p_layer"][l], f"{key}_q": c["q_layer"][l],
                                    f"{key}_q_length": c["q_layer_length_controlled"][l]})
                table.append(row)
            write_csv(out / f"exp3_layers_{model}_{source}.csv", table)

        # ---- original vs corrected localization ----
        prim = mres["sources"]["hybrid"]["comparisons"]["F1_vs_faithful"]
        orig = mres["sources"]["original"]["comparisons"]["override_vs_faithful"]
        if prim.get("status") == "tested" and orig.get("status") == "tested":
            a, b = set(orig["layers_fdr_length_controlled"]), set(prim["layers_fdr_length_controlled"])
            mres["localization_change"] = {"original": sorted(a), "corrected_F1": sorted(b),
                                           "both": sorted(a & b), "only_original": sorted(a - b),
                                           "only_corrected": sorted(b - a)}
            print(f"\nLocalization (length-controlled + FDR):\n  original labels:  {sorted(a)}"
                  f"\n  corrected F1:     {sorted(b)}\n  in both:          {sorted(a & b)}")
        else:
            print(f"\nLocalization comparison skipped: primary F1 vs faithful is "
                  f"'{prim.get('status')}' (n_F1={prim.get('n_a')}).")

        # ---- candidate heads (primary comparison) ----
        src_for_heads = "hybrid" if prim.get("status") == "tested" else "auto"
        hc = mres["sources"][src_for_heads]["comparisons"]["F1_vs_faithful"]
        if hc.get("status") == "tested":
            g = np.array(hc["g_by_head"]); p = np.array(hc["p_by_head"]); q = np.array(hc["q_by_head"])
            flat = np.argsort(-np.abs(np.nan_to_num(g)).ravel())[:10]
            cand = [{"rank": i + 1, "layer": int(f // n_heads), "head": int(f % n_heads),
                     "normalized_depth": round((f // n_heads) / (n_layers - 1), 3),
                     "g": round(float(g.ravel()[f]), 4), "p": float(p.ravel()[f]), "q": float(q.ravel()[f]),
                     "fdr_significant": bool(q.ravel()[f] < ALPHA), "label_source": src_for_heads}
                    for i, f in enumerate(flat)]
            write_csv(out / f"exp3_candidate_heads_{model}.csv", cand)
            mres["candidate_heads"] = cand
            print(f"\nCandidate heads, F1 vs faithful [{src_for_heads} labels"
                  f"{', UNVERIFIED' if src_for_heads == 'auto' else ''}]:")
            for c in cand[:5]:
                print(f"  L{c['layer']}H{c['head']}  g={c['g']:+.2f}  q={c['q']:.3f}"
                      f"{'  (FDR sig)' if c['fdr_significant'] else ''}")
            plot_heatmap(model, hc, out / f"exp3_head_heatmap_{model}.png", src_for_heads)

        # ---- figure: layer profile ----
        prof_src = "hybrid" if prim.get("status") == "tested" else "auto"
        plot_profile(model, mres["sources"][prof_src]["profile"], out / f"exp3_layer_profile_{model}.png",
                     n_layers, prof_src)
        mres["figure_label_source"] = prof_src
        results[model] = mres

    with open(out / "exp3_results.json", "w") as f:
        json.dump(results, f)
    print(f"\nWrote {out}/exp3_results.json, per-layer CSVs, candidate heads, and figures.")
    print("Reminder: ROI is correlational. Groups under the minimum size are descriptive only.")


if __name__ == "__main__":
    main()