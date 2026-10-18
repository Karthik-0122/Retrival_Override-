"""
Revision analyses requested in the guide's feedback (15 Oct 2026) on the
Experiments 1-3 report. No GPU -- everything uses data already on disk.

  correction 1   split-half stability (100 splits) for every human-label
                 comparison, with Gemma F3 vs faithful printed first
  correction 2   human-label figures: layer profiles (F2, F3, F4 vs faithful)
                 for both models and the Gemma F3 vs faithful head heatmap;
                 the automatic-label figures are copied to an appendix name
  correction 5   reconciles the no-passage counts (247 / 322) with the
                 Experiment 1 counts (246 / 320)
  analysis 1     the human-validated F1 cases against their no-passage
                 answers: same wrong answer without the passage, or not?
                 (measured on the answers' own words -- words echoed from
                 the question are excluded, unlike the Experiment 1 measure)
                 Also records the no-passage match for all 200 validated
                 items (labels are NOT changed)
  analysis 2     passage-relevance covariate: does the F3 ROI effect
                 survive when question-passage relevance is controlled?
                 (token overlap always; embedding similarity if
                 sentence-transformers is installed)
  analysis 4     the automatic-label F1 result for Llama, as text
  correction 3   exports the 7 F1 cases + the F1-vs-F2 disputes for the
                 guide's third pass (no labels shown, random order) plus a
                 private key file

Run from repo root, after experiment3_roi_by_failure_type.py:
    python Tests/revision_analyses.py
    python Tests/revision_analyses.py --no-embed     # skip embedding similarity

Writes to data/final/Phase_05/revision/
"""

import argparse
import csv
import json
import random
import re
import shutil
import string
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
import statsmodels.api as sm

MODELS = ("gemma", "llama")
PROFILE_TYPES = ["F2", "F3", "F4"]
TYPE_NAMES = {"F1": "F1 genuine override", "F2": "F2 wrong extraction",
              "F3": "F3 evidence absent", "F4": "F4 partial"}
SERIES = {"F1": "#2a78d6", "F2": "#eb6834", "F3": "#1baf7a", "F4": "#eda100"}   # same colour per type as before
MARKERS = {"F1": "o", "F2": "s", "F3": "^", "F4": "D"}
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]
INK, MUTED, GRID, SURFACE = "#1f1f1d", "#6b6b66", "#e4e3df", "#fcfcfb"
ALPHA = 0.05
SEED = 2026
STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is", "was", "are",
             "were", "be", "by", "with", "as", "from", "that", "this", "it", "its", "what", "who",
             "when", "where", "which", "how", "why", "did", "does", "do", "whose", "whom"}
_PUNCT = set(string.punctuation)


# ------------------------------------------------------------------ io / text
def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def read_csv(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(p, rows):
    if not rows:
        return
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def normalize(t):
    t = "".join(ch for ch in (t or "").lower() if ch not in _PUNCT)
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    return " ".join(t.split())


def content_tokens(t):
    return [w for w in normalize(t).split() if w not in STOPWORDS]


def parse_gold(text):
    if not text:
        return []
    s = text.strip()
    if s.startswith("["):
        try:
            v = json.loads(s)
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return [x for x in v if normalize(x)]
        except (json.JSONDecodeError, TypeError):
            pass
    return [text] if normalize(text) else []


def contains(hay, needle):
    n = normalize(needle)
    return bool(n) and f" {n} " in f" {normalize(hay)} "


def containment(noctx, answer):
    """The Experiment 1 measure: share of the no-passage answer's content words
    that appear in the with-passage answer. Kept only for comparison -- it
    over-counts matches when both answers restate the question."""
    nt = content_tokens(noctx) or normalize(noctx).split()
    at = set(content_tokens(answer)) | set(normalize(answer).split())
    return sum(t in at for t in nt) / len(nt) if nt else 0.0


def _stem(t):
    return t[:-1] if len(t) > 3 and t.endswith("s") else t


def answer_words(text, question):
    """Content words of an answer that are NOT already in the question."""
    q = {_stem(t) for t in content_tokens(question)}
    return [_stem(t) for t in content_tokens(text) if _stem(t) not in q]


def containment_q(noctx, answer, question):
    """Corrected measure: share of the no-passage answer's own content words
    (question words removed) that appear in the with-passage answer.
    'The SA node begins ... at rest' vs '... stimulated by the autonomic
    nervous system' -> 0.0 (the old measure gave 0.56 from the echoed question)."""
    nt = answer_words(noctx, question)
    at = set(answer_words(answer, question))
    return sum(t in at for t in nt) / len(nt) if nt else 0.0


def model_of(name):
    n = str(name).lower()
    return "gemma" if "gemma" in n else ("llama" if "llama" in n else None)


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


def ranges(layers):
    """[21,22,23,25] -> '21-23, 25'"""
    layers = sorted(layers)
    out, start = [], None
    for i, l in enumerate(layers):
        if start is None:
            start = l
        if i == len(layers) - 1 or layers[i + 1] != l + 1:
            out.append(f"{start}" if start == l else f"{start}-{l}")
            start = None
    return ", ".join(out) if out else "none"


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# ------------------------------------------------------------------ correction 1
def _f(v, fmt):
    return "n/a" if v is None or v == "" else format(v, fmt)


def correction1_split_half(exp3, out):
    banner("CORRECTION 1 -- split-half stability, human labels (100 random splits)")
    rows = []
    for m in MODELS:
        prof = exp3.get(m, {}).get("sources", {}).get("human", {}).get("profile", {})
        for t in ["F1"] + PROFILE_TYPES:
            c = prof.get(t)
            if not c:
                continue
            if c.get("status") != "tested":
                rows.append({"model": m, "comparison": f"{t} vs faithful", "n": f"{c.get('n_a')}/{c.get('n_b')}",
                             "status": c.get("status"), "split_half_r": "", "r_95_low": "", "r_95_high": "",
                             "top5_overlap": "", "out_of_half_auroc": ""})
                continue
            sh = c["split_half"]
            rnd = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 3)
            rows.append({"model": m, "comparison": f"{t} vs faithful", "n": f"{c['n_a']}/{c['n_b']}",
                         "status": "tested", "split_half_r": rnd(sh["spearman_r_mean"]),
                         "r_95_low": rnd(sh["spearman_r_95"][0]), "r_95_high": rnd(sh["spearman_r_95"][1]),
                         "top5_overlap": rnd(sh["top5_overlap_jaccard_mean"]),
                         "out_of_half_auroc": rnd(sh["out_of_half_auroc_mean"])})
    write_csv(out / "c1_split_half_human.csv", rows)
    key = [r for r in rows if r["model"] == "gemma" and r["comparison"] == "F3 vs faithful"]
    if key and key[0]["status"] == "tested":
        r = key[0]
        print(f">> FOR TABLE 11 (Gemma, human F3 vs faithful, n={r['n']}): split-half r = {_f(r['split_half_r'], '.2f')} "
              f"(95% range {_f(r['r_95_low'], '.2f')} to {_f(r['r_95_high'], '.2f')}); top-5 layer overlap "
              f"(Jaccard) = {_f(r['top5_overlap'], '.2f')}; out-of-half AUROC = {_f(r['out_of_half_auroc'], '.2f')}")
    for r in rows:
        if r["status"] == "tested":
            print(f"  {r['model']:<6} {r['comparison']:<16} n={r['n']:<8} r={_f(r['split_half_r'], '+.2f')} "
                  f"[{_f(r['r_95_low'], '+.2f')},{_f(r['r_95_high'], '+.2f')}]  top-5 overlap "
                  f"{_f(r['top5_overlap'], '.2f')}  out-of-half AUROC {_f(r['out_of_half_auroc'], '.2f')}")
        else:
            print(f"  {r['model']:<6} {r['comparison']:<16} n={r['n']:<8} {r['status']}")
    print("  (r = rank correlation of the per-layer effect between the two halves; top-5 overlap = share of\n"
          "   the five strongest layers found in both halves; AUROC > 0.5 = layers picked in one half\n"
          "   separate the groups in the other half)")
    return rows


# ------------------------------------------------------------------ correction 2
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_profile_human(model, prof, n_layers, path, n_f1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    _style(ax)
    x = np.arange(n_layers)
    ax.axhline(0, color=MUTED, linewidth=1)
    ends = []
    for t in PROFILE_TYPES:
        c = prof.get(t)
        if not c or c.get("g_by_layer") is None:
            continue
        g = np.array(c["g_by_layer"], dtype=float)
        tested = c.get("status") == "tested"
        ax.plot(x, g, color=SERIES[t], linewidth=2, linestyle="-" if tested else ":", marker=MARKERS[t],
                markersize=4, markevery=3,
                label=f"{TYPE_NAMES[t]} (n={c['n_a']}{'' if tested else ', too small to test'})")
        if tested and c.get("layers_fdr_length_controlled"):
            sig = c["layers_fdr_length_controlled"]
            ax.scatter(sig, g[sig], s=46, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=5)
        if np.any(~np.isnan(g)):
            ends.append((g[~np.isnan(g)][-1], t))
    lo, hi = ax.get_ylim()
    gap = 0.07 * (hi - lo)
    placed = []
    for y, t in sorted(ends):
        y = max(y, placed[-1] + gap) if placed else y
        placed.append(y)
        ax.annotate(t, (n_layers - 1, y), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK, fontweight="bold")
    if placed:
        ax.set_ylim(lo, max(hi, placed[-1] + gap))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5,
              labelcolor=INK, handlelength=2.5)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_xlabel("Layer", color=MUTED, fontsize=9.5)
    ax.set_ylabel("Hedges' g vs faithful  (+ = higher ROI)", color=MUTED, fontsize=9.5)
    ax.set_title(f"{model.capitalize()}: ROI difference from faithful answers, by failure type\n"
                 f"Human-validated labels only", color=INK, fontsize=11, loc="left", pad=46)
    ax.text(0, -0.2, f"Circles = layers significant after length control + FDR. F1 not shown "
            f"(n = {n_f1}, too few to test).", transform=ax.transAxes, fontsize=8, color=MUTED)
    fig.subplots_adjust(right=0.94, bottom=0.2, top=0.8)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_heatmap(model, c, path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    g = np.array(c["g_by_head"], dtype=float)
    lim = float(np.nanmax(np.abs(g))) or 1.0
    fig, ax = plt.subplots(figsize=(7, 8), facecolor=SURFACE)
    im = ax.imshow(g, cmap=LinearSegmentedColormap.from_list("div", DIVERGING), vmin=-lim, vmax=lim,
                   aspect="auto", interpolation="nearest")
    for l, h in c.get("heads_fdr", []):
        ax.add_patch(plt.Rectangle((h - 0.5, l - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.3))
    ax.set_xlabel("Head", color=MUTED, fontsize=9.5)
    ax.set_ylabel("Layer", color=MUTED, fontsize=9.5)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(f"Hedges' g, {label} vs faithful (+ = higher ROI)", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    ax.set_title(f"{model.capitalize()}: per-head ROI effect, {label} (n={c['n_a']}) vs faithful "
                 f"(n={c['n_b']})\nHuman-validated labels. Outlined = significant after FDR "
                 f"({len(c.get('heads_fdr', []))} heads)", color=INK, fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def correction2_figures(exp3, exp3_dir, out):
    banner("CORRECTION 2 -- figures from human-validated labels")
    made = []
    for i, m in enumerate(MODELS, 1):
        mres = exp3.get(m)
        if not mres:
            continue
        prof = mres["sources"]["human"]["profile"]
        n_f1 = mres["sources"]["human"]["group_counts"].get("F1", 0)
        p = out / f"figure{i}_layer_profile_human_{m}.png"
        plot_profile_human(m, prof, mres["n_layers"], p, n_f1)
        made.append(p)
    f3 = exp3.get("gemma", {}).get("sources", {}).get("human", {}).get("profile", {}).get("F3")
    if f3 and f3.get("status") == "tested":
        p = out / "figure3_head_heatmap_human_gemma_F3.png"
        plot_heatmap("gemma", f3, p, "F3 evidence absent")
        made.append(p)
    else:
        print("  Gemma human F3 vs faithful not tested -- heatmap skipped")
    for j, m in enumerate(MODELS, 1):                     # automatic-label figures -> appendix
        src = exp3_dir / f"exp3_layer_profile_{m}.png"
        if src.exists():
            dst = out / f"appendixA{j}_layer_profile_AUTOMATIC_{m}.png"
            shutil.copy(src, dst)
            made.append(dst)
    for p in made:
        print(f"  wrote {p}")


# ------------------------------------------------------------------ correction 5
def correction5_reconcile(noctx_rows, exp1_rows, exp1_summary):
    banner("CORRECTION 5 -- no-passage counts vs Experiment 1 counts")
    why = {"no_valid_gold": "without a usable gold answer", "no_answer": "with an empty model answer",
           "not_labeled": "without an override/faithful label", "no_passage_stored": "without a stored passage"}
    parts = []
    for m in MODELS:
        n_noctx = sum(1 for r in noctx_rows if r["model"] == m)
        n_exp1 = sum(1 for r in exp1_rows if r["model"] == m)
        exc = {k: v for k, v in exp1_summary.get(m, {}).get("excluded", {}).items() if k != "confiqa"}
        print(f"  {m}: no-passage answers {n_noctx}, Experiment 1 cases {n_exp1}, difference {n_noctx - n_exp1}; "
              f"exclusions (non-ConFiQA): {exc}")
        detail = ", ".join(f"{v} {why.get(k, k)}" for k, v in exc.items()) or "none"
        parts.append((m, n_noctx, n_exp1, sum(exc.values()), detail))
    print("\n  Sentence for the report:")
    s = "No-passage answers were generated for every apparent failure before any filtering ("
    s += " and ".join(f"{n0} {m.capitalize()}" for m, n0, _, _, _ in parts) + "). Experiment 1 then excluded "
    s += "; ".join(f"{k} {m.capitalize()} case{'s' if k != 1 else ''} ({d})" for m, _, _, k, d in parts)
    s += ", giving " + " and ".join(f"{n1}" for _, _, n1, _, _ in parts) + " apparent failures."
    print("  \"" + s + "\"")
    return s


# ------------------------------------------------------------------ analysis 1
def analysis1_noctx(validated, dataset, noctx, out):
    banner("ANALYSIS 1 -- human-validated F1 cases vs their no-passage answers")
    rows_all, f1_rows = [], []
    for v in validated:
        qid, m = v["question_id"], v["model"]
        d = dataset.get(qid, {})
        ans = d.get(f"{m}_answer") or ""
        nc = noctx.get((qid, m))
        gold = parse_gold(d.get("gold_answer_text"))
        if nc is None:
            cont, old, match, nc_correct = None, None, None, None
        else:
            cont = containment_q(nc, ans, d.get("question", ""))
            old = containment(nc, ans)
            nc_correct = any(contains(nc, g) for g in gold)
            match = cont >= 0.5
        row = {"question_id": qid, "model": m, "human_type": v["human_type"], "question": d.get("question", ""),
               "gold": d.get("gold_answer_text", ""), "with_passage_answer": ans,
               "no_passage_answer": "" if nc is None else nc,
               "containment": "" if cont is None else round(cont, 2),
               "containment_exp1_measure": "" if old is None else round(old, 2),
               "same_answer_without_passage": "" if match is None else ("yes" if match else "no"),
               "no_passage_answer_correct": "" if nc_correct is None else ("yes" if nc_correct else "no")}
        rows_all.append(row)
        if v["human_type"] == "F1":
            f1_rows.append(row)
    write_csv(out / "a1_f1_noctx_crosscheck.csv", f1_rows)
    write_csv(out / "a1_validated_labels_with_noctx.csv", rows_all)   # labels unchanged; adds the record only
    same = sum(r["same_answer_without_passage"] == "yes" for r in f1_rows)
    old_same = sum(isinstance(r["containment_exp1_measure"], float) and r["containment_exp1_measure"] >= 0.5
                   for r in f1_rows)
    knew = sum(r["no_passage_answer_correct"] == "yes" for r in f1_rows)
    for r in f1_rows:
        print(f"\n  [{r['model']}] {r['question_id']}  Q: {r['question'][:90]}")
        print(f"     gold: {r['gold'][:80]}")
        print(f"     with passage:    {r['with_passage_answer'][:110]}")
        print(f"     without passage: {r['no_passage_answer'][:110]}")
        print(f"     same answer without passage: {r['same_answer_without_passage']} "
              f"(containment {r['containment']}, question words excluded; Exp 1 measure "
              f"{r['containment_exp1_measure']}); no-passage answer correct: {r['no_passage_answer_correct']}")
    print(f"\n  SUMMARY: {same}/{len(f1_rows)} F1 cases give the same wrong answer without the passage "
          f"(direct evidence of parametric origin).")
    print(f"  (The Experiment 1 measure, which does not remove words echoed from the question, would say "
          f"{old_same}/{len(f1_rows)}.)")
    if knew:
        print(f"  NOTE: {knew} F1 case(s) answer CORRECTLY without the passage -- the passage made them worse;")
        print("  that is not parametric override and should be raised with the guide (labels not changed).")
    return f1_rows


# ------------------------------------------------------------------ analysis 2
def chunks(text, size=150, step=100, cap=30):
    w = text.split()
    if len(w) <= size:
        return [" ".join(w)] if w else [""]
    return [" ".join(w[i:i + size]) for i in range(0, max(1, len(w) - size + step), step)][:cap]


def relevance_scores(qids, dataset, passages, use_embed):
    overlap = {}
    for q in qids:
        qt = content_tokens(dataset[q].get("question", ""))
        pt = set(normalize(" ".join(passages.get(q, []))).split())
        overlap[q] = (sum(t in pt for t in qt) / len(qt)) if qt else np.nan
    scores = {"token_overlap": overlap}
    if use_embed:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("  sentence-transformers not installed -> embedding similarity skipped "
                  "(pip install sentence-transformers to add it)")
            return scores
        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        qtext = [dataset[q].get("question", "") for q in qids]
        qemb = enc.encode(qtext, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        ch, owner = [], []
        for i, q in enumerate(qids):
            for c in chunks(" ".join(passages.get(q, []))):
                ch.append(c)
                owner.append(i)
        cemb = enc.encode(ch, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        best = np.full(len(qids), -1.0)
        for e, i in zip(cemb, owner):
            best[i] = max(best[i], float(np.dot(qemb[i], e)))
        scores["embedding_max_cosine"] = {q: best[i] for i, q in enumerate(qids)}
    return scores


RANK_PROBLEMS = Counter()


def ols_group_effect(y, grp, covs, tag=""):
    """Coefficient and p-value of the group term in OLS y ~ group + covariates.
    Returns (nan, 1.0) and records the problem if the design is rank-deficient
    (e.g. a covariate that is perfectly collinear with the group)."""
    ok = ~np.isnan(y)
    for c in covs:
        ok &= ~np.isnan(c)
    X = sm.add_constant(np.column_stack([grp[ok]] + [c[ok] for c in covs]), has_constant="add")
    if np.linalg.matrix_rank(X) < X.shape[1]:
        RANK_PROBLEMS[tag] += 1
        return float("nan"), 1.0
    fit = sm.OLS(y[ok], X).fit()
    return float(fit.params[1]), float(fit.pvalues[1])


def analysis2_relevance(roi, dataset, passages, human, out, use_embed):
    banner("ANALYSIS 2 -- does the failure-type ROI effect survive a passage-relevance covariate?")
    hum = {(h["question_id"], h["model"]): h["human_type"] for h in human}
    all_q = sorted({r["query_id"] for r in roi if r["query_id"] in dataset})
    rel = relevance_scores(all_q, dataset, passages, use_embed)
    write_csv(out / "a2_relevance_scores.csv",
              [{"query_id": q, **{k: (None if np.isnan(v[q]) else round(float(v[q]), 4)) for k, v in rel.items()}}
               for q in all_q])
    summary, layer_rows = {}, []
    for m in MODELS:
        rows = [r for r in roi if model_of(r.get("model")) == m and r["query_id"] in dataset]
        if not rows:
            continue
        faithful = {q for q, d in dataset.items() if d.get("source_category") != "confiqa"
                    and d.get(f"{m}_relabel_status") == "ok" and d.get(f"{m}_label") == "faithful"
                    and d.get(f"{m}_label_full") == "faithful"}
        for t in PROFILE_TYPES:
            sel = [r for r in rows if hum.get((r["query_id"], m)) == t or r["query_id"] in faithful]
            grp = np.array([1.0 if hum.get((r["query_id"], m)) == t else 0.0 for r in sel])
            if grp.sum() < 5:
                continue
            L = np.array([np.nanmean(np.array(r["divergence_by_layer"], dtype=float), axis=1) for r in sel])
            length = np.array([len((dataset[r["query_id"]].get(f"{m}_answer") or "").split()) for r in sel], float)
            res = {"n_type": int(grp.sum()), "n_faithful": int((grp == 0).sum())}
            for name, sc in rel.items():
                rv = np.array([sc[r["query_id"]] for r in sel], dtype=float)
                a, b = rv[grp == 1], rv[grp == 0]
                a, b = a[~np.isnan(a)], b[~np.isnan(b)]
                res[f"{name}_mean_type"] = float(np.mean(a))
                res[f"{name}_mean_faithful"] = float(np.mean(b))
                res[f"{name}_mwu_p"] = float(stats.mannwhitneyu(a, b).pvalue)
                au = float(stats.mannwhitneyu(b, a).statistic / (len(a) * len(b)))
                res[f"{name}_auroc_alone"] = max(au, 1 - au)
            base = [ols_group_effect(L[:, l], grp, [length], f"{m} {t} length") for l in range(L.shape[1])]
            q_base = bh_fdr([p for _, p in base])
            sig_base = [l for l in range(L.shape[1]) if q_base[l] < ALPHA]
            res["layers_base"] = sig_base
            for name, sc in rel.items():
                rv = np.array([sc[r["query_id"]] for r in sel], dtype=float)
                adj = [ols_group_effect(L[:, l], grp, [length, rv], f"{m} {t} length+{name}") for l in range(L.shape[1])]
                q_adj = bh_fdr([p for _, p in adj])
                sig_adj = [l for l in range(L.shape[1]) if q_adj[l] < ALPHA]
                band = sig_base or list(range(L.shape[1]))
                cb = np.nanmean([abs(base[l][0]) for l in band])
                ca = np.nanmean([abs(adj[l][0]) for l in band])
                res[f"layers_with_{name}"] = sig_adj
                res[f"attenuation_{name}"] = float(1 - ca / cb) if (cb > 0 and not np.isnan(ca)) else None
                for l in range(L.shape[1]):
                    layer_rows.append({"model": m, "type": t, "covariate": name, "layer": l,
                                       "coef_base": base[l][0], "q_base": q_base[l],
                                       "coef_adjusted": adj[l][0], "q_adjusted": q_adj[l]})
            summary[f"{m}_{t}"] = res
            print(f"\n  {m} {t} vs faithful (n={res['n_type']}/{res['n_faithful']})")
            print(f"    length-controlled (as reported):        layers {ranges(sig_base)}  [{len(sig_base)}]")
            for name in rel:
                sa = res[f"layers_with_{name}"]
                att = res[f"attenuation_{name}"]
                print(f"    + {name:<22}           layers {ranges(sa)}  [{len(sa)}]   "
                      f"effect size change over the band: {'n/a' if att is None else f'{-100 * att:+.0f}%'}")
                print(f"      {name}: {t} mean {res[f'{name}_mean_type']:.3f} vs faithful "
                      f"{res[f'{name}_mean_faithful']:.3f} (Mann-Whitney p={res[f'{name}_mwu_p']:.3g}; "
                      f"relevance alone separates them with AUROC {res[f'{name}_auroc_alone']:.2f})")
    if RANK_PROBLEMS:
        print("\n  !! Regression could not be fitted (collinear covariates) for:", dict(RANK_PROBLEMS))
        print("  !! Those layers are reported as not significant; check the covariates before using them.")
    write_csv(out / "a2_relevance_covariate_by_layer.csv", layer_rows)
    json.dump(summary, open(out / "a2_relevance_covariate_summary.json", "w"), indent=2)
    g = summary.get("gemma_F3")
    if g:
        print("\n  READING THE GEMMA F3 RESULT:")
        for name in rel:
            att, n0, n1 = g[f"attenuation_{name}"], len(g["layers_base"]), len(g[f"layers_with_{name}"])
            if n0 == 0:
                verdict = "no length-controlled effect to begin with"
            elif n1 == 0 or (att is not None and att >= 0.5):
                verdict = "LARGELY EXPLAINED by passage relevance -> treat as a property of the retrieval"
            elif n1 >= 0.5 * n0 and (att is None or att < 0.25):
                verdict = "SURVIVES the relevance control -> not explained by topical relevance"
            else:
                verdict = "PARTLY explained -> report both numbers; still exploratory"
            print(f"    {name}: {n0} -> {n1} significant layers, effect "
                  f"{'n/a' if att is None else f'{-100 * att:+.0f}%'}  =>  {verdict}")
        print("    Caveat: F3 is defined by the passage lacking the answer, so even a surviving effect is a\n"
              "    response to the input; it shows ROI tracks more than topical relevance, not that it is a\n"
              "    failure mechanism.")
    return summary


# ------------------------------------------------------------------ analysis 4
def analysis4_llama_auto(exp3):
    banner("ANALYSIS 4 -- automatic-label F1 result for Llama (text for the report)")
    c = exp3.get("llama", {}).get("sources", {}).get("auto", {}).get("comparisons", {}).get("F1_vs_faithful")
    if not c or c.get("status") != "tested":
        print("  not available")
        return
    g = np.array(c["g_by_layer"], dtype=float)
    top = np.argsort(-np.abs(np.nan_to_num(g)))[:3]
    print(f"  \"With automatic labels, Llama F1 (n={c['n_a']}) vs faithful (n={c['n_b']}) shows "
          f"{len(c['layers_p05'])} layers at uncorrected p < .05, {len(c['layers_fdr'])} after FDR and "
          f"{len(c['layers_fdr_length_controlled'])} after length control and FDR "
          f"(strongest: {', '.join(f'L{l} g = {g[l]:+.2f}' for l in top)}; AUROC on the pre-specified "
          f"layers {max(c['auroc_original_layers'], 1 - c['auroc_original_layers']):.2f}). "
          f"{len(c['heads_fdr'])} heads pass FDR. As for Gemma, these automatic F1 cases are mostly not "
          f"genuine override, so the result is not interpreted.\"")


# ------------------------------------------------------------------ correction 3
def export_f1_packet(adj_rows, key_rows, passages, out):
    banner("CORRECTION 3 -- F1 cases and F1-vs-F2 disputes for the guide's third pass")
    key = {k["item_id"]: k for k in key_rows}
    picked = []
    for r in adj_rows:
        a, b = r["annot_A"].strip().upper(), r["annot_B"].strip().upper()
        final = r["final_type"].strip().upper()
        dispute = {a, b} == {"F1", "F2"}
        if final == "F1" or dispute:
            picked.append((r, final, dispute))
    random.Random(SEED).shuffle(picked)
    packet, private = [], []
    for i, (r, final, dispute) in enumerate(picked, 1):
        k = key.get(r["item_id"], {})
        full = "\n\n---\n\n".join(passages.get(k.get("question_id"), [])) or r.get("retrieved_passages", "")
        pid = f"R{i:02d}"
        packet.append({"case": pid, "question": r["question"], "gold_answers": r["gold_answers"],
                       "model_answer": r["model_answer"], "retrieved_passages": full,
                       "rationale_annotator_A": r["rationale_A"], "rationale_annotator_B": r["rationale_B"],
                       "guide_label (F1-F6)": "", "guide_notes": ""})
        private.append({"case": pid, "item_id": r["item_id"], "question_id": k.get("question_id"),
                        "model": k.get("model"), "annot_A": r["annot_A"], "annot_B": r["annot_B"],
                        "final_type": final, "in_set_because": ("final F1" if final == "F1" else "")
                        + (" + " if final == "F1" and dispute else "") + ("F1-vs-F2 dispute" if dispute else "")})
    write_csv(out / "c3_F1_review_packet_FOR_GUIDE.csv", packet)
    write_csv(out / "c3_F1_review_packet_PRIVATE_KEY.csv", private)
    nf1 = sum(p["final_type"] == "F1" for p in private)
    nd = sum("dispute" in p["in_set_because"] for p in private)
    print(f"  {len(packet)} unique cases: {nf1} final F1, {nd} F1-vs-F2 disputes (overlap counted once).")
    print("  Send ONLY c3_F1_review_packet_FOR_GUIDE.csv (labels hidden, random order). Keep the PRIVATE_KEY file.")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", default="data/final/Phase_03/phase3_roi_results.jsonl")
    ap.add_argument("--dataset", default="data/final/Phase_02/analysis_dataset_relabeled.jsonl")
    ap.add_argument("--retrieval", default="data/final/Phase_02/retrieval_results.jsonl")
    ap.add_argument("--exp1", default="data/final/Phase_05/exp1_failure_types.jsonl")
    ap.add_argument("--exp1-summary", default="data/final/Phase_05/exp1_summary.json")
    ap.add_argument("--noctx", default="data/final/Phase_05/no_passage_answers.jsonl")
    ap.add_argument("--exp2dir", default="data/final/Phase_05/exp2")
    ap.add_argument("--exp3dir", default="data/final/Phase_05/exp3")
    ap.add_argument("--outdir", default="data/final/Phase_05/revision")
    ap.add_argument("--no-embed", action="store_true", help="skip embedding similarity")
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    exp2, exp3_dir = Path(a.exp2dir), Path(a.exp3dir)
    dataset = {r["query_id"]: r for r in load_jsonl(a.dataset)}
    passages = {r["query_id"]: r.get("retrieved_passages") or [] for r in load_jsonl(a.retrieval)}
    exp3 = json.load(open(exp3_dir / "exp3_results.json"))
    validated = load_jsonl(exp2 / "validated_labels.jsonl")
    noctx_rows = load_jsonl(a.noctx)
    noctx = {(r["query_id"], r["model"]): r["noctx_answer"] for r in noctx_rows}

    correction1_split_half(exp3, out)
    correction2_figures(exp3, exp3_dir, out)
    correction5_reconcile(noctx_rows, load_jsonl(a.exp1), json.load(open(a.exp1_summary)))
    analysis1_noctx(validated, dataset, noctx, out)
    analysis2_relevance(load_jsonl(a.roi), dataset, passages, validated, out, use_embed=not a.no_embed)
    analysis4_llama_auto(exp3)
    export_f1_packet(read_csv(exp2 / "adjudication.csv"), read_csv(exp2 / "HIDDEN_key.csv"), passages, out)
    print(f"\nAll outputs in {out}/")


if __name__ == "__main__":
    main()