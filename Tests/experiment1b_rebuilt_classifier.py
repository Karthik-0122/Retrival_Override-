"""
Rebuilt failure-type classifier (analysis 3) and pilot yield. No GPU --
reads the judge outputs written by llm_judge.py.

The rules are the written adjudication rules, applied in the same order, with
the LLM judge answering each question instead of string matching:

  1. F5  answer correct                         (judge: correct = yes)
  2. F4  right but less specific / incomplete   (judge: correct = partial)
  3. F3  passages do not state the gold         (judge: contains = no)
  4. F6  no real answer given                   (judge: answers = no)
  5. F2  wrong answer is written in the passage (judge: supported = yes)
  6. F1  HARD REQUIREMENT: the no-passage answer is the same answer
         (judge: same = yes) and is itself not correct
  7. F6  otherwise

Nothing is tuned on the human labels, so every human-labelled case is out of
sample. As the guide asked, a held-out fold is still fixed in advance:
the 200 cases are split 50/50 within each (model, human type) cell with seed
2026, and the split is written to disk. Precision and recall for F1 are
reported on the held-out fold (primary) and on all 200, with 95% Wilson
intervals -- with 7 human F1 cases these intervals are wide by necessity.

Tasks:
  --task existing   analysis 3 + predictions for all 566 apparent failures
  --task pilot      enrichment-pilot yield tables + F1 candidate list

Run from repo root:
    python Tests/experiment1b_rebuilt_classifier.py --task existing
    python Tests/experiment1b_rebuilt_classifier.py --task pilot
"""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from gpu_utils import containment_q, load_jsonl

TYPES = ["F1", "F2", "F3", "F4", "F5", "F6"]
SEED = 2026


def classify(J):
    c = J.get("correct")
    if c is None:
        return None
    if c == "yes":
        return "F5"
    if c == "partial":
        return "F4"
    if J.get("contains") == "no":
        return "F3"
    if J.get("answers") == "no":
        return "F6"
    if J.get("supported") == "yes":
        return "F2"
    if J.get("same") == "yes" and J.get("noctx_correct") != "yes":
        return "F1"
    return "F6"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def metrics(pairs):
    """pairs: list of (human, predicted)."""
    out = {"n": len(pairs), "accuracy": sum(h == p for h, p in pairs) / len(pairs) if pairs else float("nan")}
    for t in TYPES:
        tp = sum(h == t and p == t for h, p in pairs)
        fp = sum(h != t and p == t for h, p in pairs)
        fn = sum(h == t and p != t for h, p in pairs)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        out[t] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
                  "precision_ci": wilson(tp, tp + fp), "recall_ci": wilson(tp, tp + fn),
                  "n_human": tp + fn, "n_pred": tp + fp}
    return out


def fmt(v):
    return "  n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}"


def print_metrics(name, m, old=None):
    print(f"\n  {name}: n = {m['n']}, accuracy {fmt(m['accuracy'])}"
          + (f"   (old classifier on the same cases: {fmt(old['accuracy'])})" if old else ""))
    print(f"    {'type':<5}{'prec':>6}{'95% CI':>15}{'recall':>8}{'95% CI':>15}{'n_human':>9}{'n_pred':>8}"
          + ("   old prec / recall" if old else ""))
    for t in TYPES:
        r = m[t]
        line = (f"    {t:<5}{fmt(r['precision']):>6}  [{fmt(r['precision_ci'][0])},{fmt(r['precision_ci'][1])}]"
                f"{fmt(r['recall']):>8}  [{fmt(r['recall_ci'][0])},{fmt(r['recall_ci'][1])}]{r['n_human']:>9}{r['n_pred']:>8}")
        if old:
            line += f"   {fmt(old[t]['precision'])} / {fmt(old[t]['recall'])}"
        print(line)


def confusion(pairs):
    m = {h: Counter() for h in TYPES}
    for h, p in pairs:
        if h in m:
            m[h][p] += 1
    print(f"    {'human/pred':<11}" + "".join(f"{t:>5}" for t in TYPES))
    for h in TYPES:
        print(f"    {h:<11}" + "".join(f"{m[h][t]:>5}" for t in TYPES))


def kappa(x, y):
    n = len(x)
    if n == 0:
        return float("nan")
    labs = sorted(set(x) | set(y))
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[l] * cy[l] for l in labs) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def read_csv(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(p, rows):
    if rows:
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


# ------------------------------------------------------------------ analysis 3
def task_existing(a):
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    jl = {tuple(r["key"]): r["label"] for r in load_jsonl(Path(a.judge_dir) / "judge_existing.jsonl")}
    if not jl:
        raise SystemExit("no judgments found -- run python Tests/llm_judge.py --tasks existing first")
    exp1 = load_jsonl(a.exp1)
    preds, missing = {}, 0
    rows = []
    for r in exp1:
        q, m = r["question_id"], r["model"]
        J = {"correct": jl.get((q, m, "correct")), "noctx_correct": jl.get((q, m, "noctx_correct")),
             "contains": jl.get((q, "-", "contains")), "answers": jl.get((q, m, "answers")),
             "supported": jl.get((q, m, "supported")), "same": jl.get((q, m, "same"))}
        p = classify(J)
        if p is None:
            missing += 1
            continue
        preds[(q, m)] = p
        rows.append({"question_id": q, "model": m, "dataset": r.get("dataset"), "rebuilt_type": p,
                     "old_auto_type": r.get("failure_type"),
                     "same_lexical": round(containment_q(r.get("noctx_answer") or "", r.get("model_answer") or "",
                                                         r.get("question") or ""), 2), **{f"judge_{k}": v for k, v in J.items()}})
    write_csv(out / "rebuilt_predictions_all.csv", rows)
    if missing:
        print(f"!! {missing} cases have no judgments yet (judge run incomplete?)")

    print("=" * 78 + "\nANALYSIS 3 -- rebuilt classifier vs the 200 human labels\n" + "=" * 78)
    for m in ("gemma", "llama"):
        c = Counter(p for (q, mm), p in preds.items() if mm == m)
        n = sum(c.values())
        print(f"  all {n} {m} apparent failures, rebuilt types: " + ", ".join(f"{t}={c[t]}" for t in TYPES))

    val = load_jsonl(Path(a.exp2dir) / "validated_labels.jsonl")
    # pre-registered split, written to disk
    cells = defaultdict(list)
    for v in sorted(val, key=lambda v: (v["model"], v["human_type"], v["question_id"])):
        cells[(v["model"], v["human_type"])].append(v)
    rng = random.Random(SEED)
    fold = {}
    for k in sorted(cells):
        items = cells[k]
        rng.shuffle(items)
        for i, v in enumerate(items):
            fold[(v["question_id"], v["model"])] = "test" if i % 2 == 0 else "dev"
    write_csv(out / "rebuilt_split.csv", [{"question_id": q, "model": m, "fold": f} for (q, m), f in sorted(fold.items())])

    pairs = {"test": [], "dev": [], "all": []}
    old = {"test": [], "dev": [], "all": []}
    for v in val:
        key = (v["question_id"], v["model"])
        if key not in preds:
            continue
        h, p, o = v["human_type"], preds[key], v.get("auto_type")
        for f in (fold[key], "all"):
            pairs[f].append((h, p))
            old[f].append((h, o))
    res = {f: metrics(pairs[f]) for f in pairs}
    res_old = {f: metrics(old[f]) for f in old}
    print_metrics("HELD-OUT FOLD (primary)", res["test"], res_old["test"])
    print_metrics("dev fold", res["dev"], res_old["dev"])
    print_metrics("all 200", res["all"], res_old["all"])
    print("\n  Confusion, all 200 (rows human, columns rebuilt):")
    confusion(pairs["all"])
    t = res["test"]["F1"]
    print(f"\n  >> F1 on the held-out fold: precision {fmt(t['precision'])} "
          f"[{fmt(t['precision_ci'][0])},{fmt(t['precision_ci'][1])}], recall {fmt(t['recall'])} "
          f"[{fmt(t['recall_ci'][0])},{fmt(t['recall_ci'][1])}] (human F1 cases in fold: {t['n_human']}); "
          f"old classifier: precision {fmt(res_old['test']['F1']['precision'])}, recall {fmt(res_old['test']['F1']['recall'])}")

    # judge vs annotators, question by question
    key = {r["item_id"]: (r["question_id"], r["model"]) for r in read_csv(Path(a.exp2dir) / "HIDDEN_key.csv")}
    sheets = {}
    for name in ("A", "B"):
        p = Path(a.exp2dir) / f"annotation_sheet_{name}.csv"
        if p.exists():
            sheets[name] = {key[r["item_id"]]: r for r in read_csv(p) if r["item_id"] in key}

    def hum(v, col):
        v = (v or "").strip().lower()
        v = {"partialy": "partially"}.get(v, v)
        if col == "passage_contains_answer" and v == "partially":
            v = "unclear"
        return v

    judge_map = {"passage_contains_answer": ("contains", lambda x: x),
                 "answer_correct": ("correct", lambda x: {"partial": "partially"}.get(x, x)),
                 "answer_supported_by_passage": ("supported", lambda x: x)}
    allowed = {"passage_contains_answer": {"yes", "no"}, "answer_correct": {"yes", "no", "partially"},
               "answer_supported_by_passage": {"yes", "no"}}
    print("\n  Judge vs annotators, question by question (items where all three gave an allowed value):")
    agree_rows = []
    for col, (kind, conv) in judge_map.items():
        trip = []
        for (q, m) in preds:
            if "A" not in sheets or "B" not in sheets or (q, m) not in sheets["A"] or (q, m) not in sheets["B"]:
                continue
            ja = jl.get((q, "-", "contains") if kind == "contains" else (q, m, kind))
            if ja is None:
                continue
            x, y, z = hum(sheets["A"][(q, m)].get(col), col), hum(sheets["B"][(q, m)].get(col), col), conv(ja)
            if x in allowed[col] and y in allowed[col] and z in allowed[col]:
                trip.append((x, y, z))
        if not trip:
            continue
        A_, B_, J_ = zip(*trip)
        row = {"question": col, "n": len(trip),
               "A_vs_B_agree": round(sum(a1 == b1 for a1, b1 in zip(A_, B_)) / len(trip), 3), "A_vs_B_kappa": round(kappa(A_, B_), 3),
               "judge_vs_A_agree": round(sum(a1 == b1 for a1, b1 in zip(J_, A_)) / len(trip), 3), "judge_vs_A_kappa": round(kappa(J_, A_), 3),
               "judge_vs_B_agree": round(sum(a1 == b1 for a1, b1 in zip(J_, B_)) / len(trip), 3), "judge_vs_B_kappa": round(kappa(J_, B_), 3)}
        agree_rows.append(row)
        print(f"    {col:<30} n={row['n']:<4} A-B {row['A_vs_B_agree']:.2f} (k {row['A_vs_B_kappa']:+.2f})   "
              f"judge-A {row['judge_vs_A_agree']:.2f} (k {row['judge_vs_A_kappa']:+.2f})   "
              f"judge-B {row['judge_vs_B_agree']:.2f} (k {row['judge_vs_B_kappa']:+.2f})")
    write_csv(out / "rebuilt_judge_vs_annotators.csv", agree_rows)

    # same-answer check: judge vs the corrected lexical measure
    sj = [(r["judge_same"], r["same_lexical"] >= 0.5) for r in rows if r["judge_same"] in ("yes", "no")]
    if sj:
        agree = sum((j == "yes") == lex for j, lex in sj) / len(sj)
        print(f"\n  Same-answer check, judge vs corrected word measure: agree on {agree:.0%} of {len(sj)} cases")

    json.dump({"metrics_rebuilt": res, "metrics_old": res_old, "judge_vs_annotators": agree_rows},
              open(out / "rebuilt_classifier_report.json", "w"), indent=2, default=str)
    unl = [r for r in rows if (r["question_id"], r["model"]) not in fold and r["rebuilt_type"] == "F3"]
    c = Counter(r["model"] for r in unl)
    print(f"\n  Unlabelled cases predicted F3 (candidates for the extra F3 labelling): {dict(c)}")
    print(f"\nWrote {out}/rebuilt_predictions_all.csv, rebuilt_split.csv, rebuilt_judge_vs_annotators.csv, "
          f"rebuilt_classifier_report.json")


# ------------------------------------------------------------------ pilot
def task_pilot(a):
    d = Path(a.pilot_dir)
    pool = {r["pid"]: r for r in load_jsonl(d / "pilot_pool.jsonl")}
    jl = {tuple(r["key"]): r["label"] for r in load_jsonl(d / "judge_pilot.jsonl")}
    prec = None
    rep = Path(a.outdir) / "rebuilt_classifier_report.json"
    if rep.exists():
        p = json.load(open(rep))["metrics_rebuilt"]["all"]["F1"]["precision"]
        prec = None if p in (None, "nan") or (isinstance(p, float) and math.isnan(p)) else float(p)
    print("=" * 78 + "\nENRICHMENT PILOT -- yield of genuine-override (F1) candidates\n" + "=" * 78)
    summary, cands = {}, []
    for m in ("gemma", "llama"):
        cb = load_jsonl(d / f"pilot_closedbook_{m}.jsonl")
        wp = {(r["pid"], r["cond"]): r for r in load_jsonl(d / f"pilot_withpassage_{m}.jsonl")}
        if not cb:
            continue
        n_pool = len(cb)
        wrong = [r for r in cb if jl.get((r["pid"], m, "-", "cb_correct")) == "no" and not r["cb_string_correct"]]
        judged = sum((r["pid"], m, "-", "cb_correct") in jl for r in cb)
        print(f"\n{m.upper()}: {n_pool} questions, closed-book judged {judged}, wrong {len(wrong)} "
              f"({100 * len(wrong) / max(n_pool, 1):.0f}%)")
        confs = sorted(r["cb_mean_logprob"] for r in wrong)
        cuts = {"all wrong": -1e9}
        for q, lab in ((0.25, "top 75% confidence"), (0.5, "top 50%"), (0.75, "top 25%")):
            if confs:
                cuts[lab] = confs[int(q * (len(confs) - 1))]
        msum = {"n_pool": n_pool, "n_wrong": len(wrong), "conditions": {}}
        for cond in ("gold", "mixed"):
            labels = {}
            for r in wrong:
                w = wp.get((r["pid"], cond))
                if not w:
                    continue
                c = "yes" if w["wp_string_correct"] else jl.get((r["pid"], m, cond, "wp_correct"))
                J = {"correct": c, "noctx_correct": "no",
                     "contains": jl.get((r["pid"], "-", cond, "contains")),
                     "answers": jl.get((r["pid"], m, cond, "answers")),
                     "supported": jl.get((r["pid"], m, cond, "supported")),
                     "same": jl.get((r["pid"], m, cond, "same"))}
                t = classify(J)
                if t:
                    labels[r["pid"]] = {"F5": "fixed_by_passage", "F4": "partial"}.get(t, t)
            dist = Counter(labels.values())
            print(f"  [{cond}] with-passage outcome for the {len(labels)} wrong-closed-book questions: "
                  + ", ".join(f"{k}={v}" for k, v in sorted(dist.items())))
            rows = []
            for lab, cut in cuts.items():
                sel = [r for r in wrong if r["cb_mean_logprob"] >= cut and r["pid"] in labels]
                f1 = [r for r in sel if labels[r["pid"]] == "F1"]
                rate_q = len(f1) / n_pool
                rows.append({"selection": lab, "n_selected": len(sel), "n_F1_candidates": len(f1),
                             "F1_share_of_selected": round(len(f1) / len(sel), 3) if sel else None,
                             "F1_candidates_per_1000_questions": round(1000 * rate_q, 1),
                             "questions_for_40_candidates": round(40 / rate_q) if rate_q else None,
                             "questions_for_40_validated_if_precision_holds":
                                 round(40 / (rate_q * prec)) if (rate_q and prec) else None})
                print(f"    {lab:<20} selected {len(sel):>4}  F1 candidates {len(f1):>3}  "
                      f"({rows[-1]['F1_share_of_selected']})  per 1,000 questions {rows[-1]['F1_candidates_per_1000_questions']:>5}  "
                      f"questions for 40 candidates: {rows[-1]['questions_for_40_candidates']}")
            by_src = Counter(pool[pid]["source"] for pid, t in labels.items() if t == "F1")
            print(f"    F1 candidates by source: {dict(by_src)}")
            msum["conditions"][cond] = {"distribution": dict(dist), "yield": rows, "by_source": dict(by_src)}
            for r in wrong:
                if labels.get(r["pid"]) == "F1":
                    p, w = pool[r["pid"]], wp[(r["pid"], cond)]
                    cands.append({"pid": r["pid"], "model": m, "cond": cond, "source": p["source"],
                                  "question": p["question"], "answers": " | ".join(p["answers"]),
                                  "closed_book_answer": r["cb_answer"], "cb_mean_logprob": r["cb_mean_logprob"],
                                  "with_passage_answer": w["wp_answer"]})
        summary[m] = msum
    if prec is not None:
        print(f"\n  ('validated' projections assume the F1 precision measured on the 200 human labels, {prec:.2f};"
              f" the real figure comes from human-checking these candidates)")
    write_csv(d / "pilot_F1_candidates.csv", cands)
    json.dump(summary, open(d / "pilot_yield_summary.json", "w"), indent=2)
    print(f"\nWrote {d}/pilot_yield_summary.json and pilot_F1_candidates.csv ({len(cands)} candidate rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["existing", "pilot"], required=True)
    ap.add_argument("--exp1", default="data/final/Phase_05/exp1_failure_types.jsonl")
    ap.add_argument("--exp2dir", default="data/final/Phase_05/exp2")
    ap.add_argument("--judge-dir", default="data/final/Phase_05/judge")
    ap.add_argument("--outdir", default="data/final/Phase_05/judge")
    ap.add_argument("--pilot-dir", default="data/final/Phase_05/pilot")
    a = ap.parse_args()
    {"existing": task_existing, "pilot": task_pilot}[a.task](a)


if __name__ == "__main__":
    main()
