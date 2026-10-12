"""
Experiment 2 -- Human validation of ~200 cases. No GPU. Three steps:

  1) sample     Stratified sample (model x predicted type, spread over datasets).
                Writes two BLIND annotation sheets (no automatic label, no
                signals, different row order) + a hidden key + guidelines.

                    python Tests/experiment2_annotation.py sample

  2) agreement  After both annotators fill their sheet independently:
                Cohen's kappa (all types, and F1-vs-rest) with bootstrap CI,
                percent agreement, disagreement matrix. Writes an
                adjudication sheet listing only the disagreements.

                    python Tests/experiment2_annotation.py agreement

  3) validate   After the adjudication sheet's final_type column is filled:
                ground-truth labels, automatic-vs-human confusion matrix,
                accuracy, per-type precision/recall/F1 (F1 genuine override
                reported separately), macro F1. Writes the validated dataset
                used by Experiments 3-7.

                    python Tests/experiment2_annotation.py validate

With only one annotator: `agreement --single` skips kappa and treats sheet A
as final (report this as a limitation).

All files in data/final/Phase_05/exp2/
"""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

TYPES = ["F1", "F2", "F3", "F4", "F5", "F6"]
TARGET_N = 200
SEED = 2026
MAX_PASSAGE_CHARS = 30000           # Excel cell limit is 32,767
ANNOT_COLS = ["failure_type", "passage_contains_answer", "answer_correct",
              "answer_supported_by_passage", "answer_conflicts_with_passage",
              "confidence", "rationale"]
ALLOWED = {
    "failure_type": set(TYPES),
    "passage_contains_answer": {"yes", "no", "unclear"},
    "answer_correct": {"yes", "no", "partially", "unclear"},
    "answer_supported_by_passage": {"yes", "no", "partially", "unclear"},
    "answer_conflicts_with_passage": {"yes", "no", "unclear"},
    "confidence": {"high", "medium", "low"},
}

GUIDELINES = """ANNOTATION GUIDELINES -- Retrieval failure types (read fully before starting)

You see: question, gold answer(s), retrieved passage(s), the model's answer.
Work ALONE. Do not discuss cases with the other annotator until both sheets are done.
Fill every annotation column. Allowed values are listed below; use them exactly.

Answer these in order, then pick ONE failure_type:
  passage_contains_answer       yes / no / unclear
      Does the passage explicitly state the gold answer (any valid form)?
  answer_correct                yes / no / partially / unclear
      Is the model's answer factually correct for the question?
  answer_supported_by_passage   yes / no / partially / unclear
      Is what the model said stated in the passage (even if it's the wrong fact)?
  answer_conflicts_with_passage yes / no / unclear
      Does the model's answer contradict what the passage says about the question?

failure_type (exactly one):
  F1  Genuine parametric override -- passage contains the correct answer, the model's
      answer conflicts with it, and the answer is NOT taken from the passage, i.e. it is
      better explained by the model's own prior knowledge.
  F2  Wrong extraction / wrong interpretation -- the model uses the passage but takes the
      wrong fact, answers a different aspect of the question, or misreads the evidence.
  F3  Evidence absent -- the passage does not actually support the gold answer.
  F4  Partially correct / incomplete -- part of the correct answer is present but
      something required is missing or altered.
  F5  Actually correct -- the answer is right; the automatic scoring was too strict.
  F6  Ambiguous -- you cannot reliably decide among the above.

CRITICAL: a wrong answer is NOT automatically F1. Choose F1 only when the passage has the
answer AND the model's answer is not drawn from the passage. If the model's wrong answer
appears in the passage, that is F2. If you can't tell, use F6.

confidence: high / medium / low.   rationale: one short sentence.
"""


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def read_csv(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(p, rows, cols):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ 1) sample
def allocate(cells, target):
    """Equal share per cell, capped by availability; leftovers redistributed."""
    quota = {k: 0 for k in cells}
    remaining = target
    open_cells = [k for k in cells if cells[k] > 0]
    while remaining > 0 and open_cells:
        share = max(1, remaining // len(open_cells))
        for k in list(open_cells):
            give = min(share, cells[k] - quota[k], remaining)
            quota[k] += give
            remaining -= give
            if quota[k] >= cells[k]:
                open_cells.remove(k)
            if remaining == 0:
                break
    return quota


def cmd_sample(a):
    rows = load_jsonl(a.exp1)
    passages = {r["query_id"]: r.get("retrieved_passages") or [] for r in load_jsonl(a.retrieval)}
    rng = random.Random(SEED)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["failure_type"])].append(r)
    quota = allocate({k: len(v) for k, v in cells.items()}, a.n)

    picked = []
    for k, n in sorted(quota.items()):
        by_ds = defaultdict(list)                 # round-robin over datasets inside each cell
        for r in cells[k]:
            by_ds[r["dataset"]].append(r)
        for v in by_ds.values():
            rng.shuffle(v)
        order = sorted(by_ds)
        i = 0
        while sum(1 for p in picked if (p["model"], p["failure_type"]) == k) < n:
            ds = order[i % len(order)]
            if by_ds[ds]:
                picked.append(by_ds[ds].pop())
            i += 1
    rng.shuffle(picked)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    items, key = [], []
    for i, r in enumerate(picked, 1):
        iid = f"item_{i:03d}"
        ptxt = "\n\n---\n\n".join(passages.get(r["question_id"], []))
        if len(ptxt) > MAX_PASSAGE_CHARS:
            ptxt = ptxt[:MAX_PASSAGE_CHARS] + " [...TRUNCATED -- see full passage via question_id]"
        items.append({"item_id": iid, "question": r["question"], "gold_answers": r["gold"],
                      "model_answer": r["model_answer"], "retrieved_passages": ptxt,
                      **{c: "" for c in ANNOT_COLS}})
        key.append({"item_id": iid, "question_id": r["question_id"], "model": r["model"],
                    "dataset": r["dataset"], "auto_type": r["failure_type"],
                    "auto_confidence": r["confidence"], "original_label": r["original_label"]})
    cols = ["item_id", "question", "gold_answers", "model_answer", "retrieved_passages"] + ANNOT_COLS
    for name, s in (("A", SEED + 1), ("B", SEED + 2)):
        sheet = items[:]
        random.Random(s).shuffle(sheet)
        write_csv(outdir / f"annotation_sheet_{name}.csv", sheet, cols)
    write_csv(outdir / "HIDDEN_key.csv", key, list(key[0].keys()))
    (outdir / "annotation_guidelines.txt").write_text(GUIDELINES)

    print(f"Sampled {len(picked)} cases. Per (model, predicted type):")
    for k in sorted(quota):
        print(f"  {k[0]:<6} {k[1]}  {quota[k]:>4} of {len(cells[k])}")
    print("Per dataset:", dict(Counter(r["dataset"] for r in picked)))
    print(f"\nGive annotation_sheet_A.csv to annotator A and _B.csv to annotator B, with")
    print("annotation_guidelines.txt. Do NOT share HIDDEN_key.csv until both are finished.")


# ------------------------------------------------------------------ stats
def kappa(x, y, labels):
    n = len(x)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[l] * cy[l] for l in labels) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def boot_kappa(x, y, labels, reps=2000):
    rng = random.Random(0)
    n, vals = len(x), []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        k = kappa([x[i] for i in idx], [y[i] for i in idx], labels)
        if not math.isnan(k):
            vals.append(k)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def matrix(x, y, labels):
    m = {a: Counter() for a in labels}
    for a, b in zip(x, y):
        m[a][b] += 1
    return m


def print_matrix(m, labels, rlab, clab):
    corner = f"{rlab} / {clab}"
    print(f"{corner:<14}" + "".join(f"{l:>6}" for l in labels))
    for a in labels:
        print(f"{a:<14}" + "".join(f"{m[a][b]:>6}" for b in labels))


def validate_sheet(rows, name):
    bad = []
    for r in rows:
        for c, ok in ALLOWED.items():
            v = (r.get(c) or "").strip()
            v = v.upper() if c == "failure_type" else v.lower()
            r[c] = v
            if v not in ok:
                bad.append((r["item_id"], c, v))
    if bad:
        print(f"Sheet {name}: {len(bad)} missing/invalid entries, e.g. {bad[:5]}")
        raise SystemExit("Fix these before computing agreement.")


# ------------------------------------------------------------------ 2) agreement
def cmd_agreement(a):
    d = Path(a.outdir)
    A = {r["item_id"]: r for r in read_csv(d / "annotation_sheet_A.csv")}
    validate_sheet(list(A.values()), "A")
    if a.single:
        final = [{"item_id": i, "annot_A": A[i]["failure_type"], "annot_B": "",
                  "final_type": A[i]["failure_type"], "how": "single_annotator"} for i in sorted(A)]
        write_csv(d / "adjudication.csv", final, list(final[0].keys()))
        print("Single annotator: sheet A taken as final. Report this as a limitation (no agreement statistic).")
        return
    B = {r["item_id"]: r for r in read_csv(d / "annotation_sheet_B.csv")}
    validate_sheet(list(B.values()), "B")
    ids = sorted(set(A) & set(B))
    x, y = [A[i]["failure_type"] for i in ids], [B[i]["failure_type"] for i in ids]

    k = kappa(x, y, TYPES); lo, hi = boot_kappa(x, y, TYPES)
    xb = ["F1" if v == "F1" else "other" for v in x]
    yb = ["F1" if v == "F1" else "other" for v in y]
    kb = kappa(xb, yb, ["F1", "other"]); lob, hib = boot_kappa(xb, yb, ["F1", "other"])
    agree = sum(p == q for p, q in zip(x, y))
    rest = [(p, q) for p, q in zip(x, y) if p != "F1" and q != "F1"]
    k_rest = kappa([p for p, _ in rest], [q for _, q in rest], TYPES[1:]) if rest else float("nan")

    print(f"Items annotated by both: {len(ids)}")
    print(f"Percent agreement (6 types):     {100 * agree / len(ids):.1f}%")
    print(f"Cohen's kappa (6 types):         {k:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"Cohen's kappa (F1 vs rest):      {kb:.3f}  95% CI [{lob:.3f}, {hib:.3f}]")
    print(f"Cohen's kappa (F2-F6, both non-F1): {k_rest:.3f}  (n={len(rest)})")
    print("  (Landis & Koch: 0.61-0.80 substantial, >0.80 almost perfect)\n")
    print_matrix(matrix(x, y, TYPES), TYPES, "A", "B")

    adj = []
    for i in ids:
        same = A[i]["failure_type"] == B[i]["failure_type"]
        adj.append({"item_id": i, "annot_A": A[i]["failure_type"], "annot_B": B[i]["failure_type"],
                    "rationale_A": A[i]["rationale"], "rationale_B": B[i]["rationale"],
                    "final_type": A[i]["failure_type"] if same else "",
                    "how": "agreed" if same else "NEEDS_ADJUDICATION"})
    write_csv(d / "adjudication.csv", adj, list(adj[0].keys()))
    json.dump({"n": len(ids), "percent_agreement": agree / len(ids), "kappa": k, "kappa_ci": [lo, hi],
               "kappa_F1_vs_rest": kb, "kappa_F1_vs_rest_ci": [lob, hib], "kappa_non_F1": k_rest,
               "disagreement_matrix": {r: dict(c) for r, c in matrix(x, y, TYPES).items()}},
              open(d / "agreement.json", "w"), indent=2)
    n_dis = sum(r["how"] == "NEEDS_ADJUDICATION" for r in adj)
    print(f"\n{n_dis} disagreements -> adjudication.csv. Procedure (fixed in advance):")
    print("  both annotators discuss each case with the passage open; if still split, the")
    print("  guide decides. Write the result in final_type and 'adjudicated' in how.")


# ------------------------------------------------------------------ 3) validate
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def cmd_validate(a):
    d = Path(a.outdir)
    adj = {r["item_id"]: r for r in read_csv(d / "adjudication.csv")}
    missing = [i for i, r in adj.items() if r["final_type"].strip().upper() not in TYPES]
    if missing:
        raise SystemExit(f"{len(missing)} items still lack a valid final_type, e.g. {missing[:5]}")
    key = {r["item_id"]: r for r in read_csv(d / "HIDDEN_key.csv")}
    ids = sorted(adj)
    human = [adj[i]["final_type"].strip().upper() for i in ids]
    auto = [key[i]["auto_type"] for i in ids]

    acc = sum(h == p for h, p in zip(human, auto)) / len(ids)
    print(f"Ground-truth items: {len(ids)}   automatic accuracy: {100 * acc:.1f}%\n")
    print_matrix(matrix(human, auto, TYPES), TYPES, "human", "auto")
    print(f"\n{'type':<5}{'prec':>7}{'recall':>8}{'F1':>7}{'n_human':>9}")
    per, f1s = {}, []
    for t in TYPES:
        tp = sum(h == t and p == t for h, p in zip(human, auto))
        fp = sum(h != t and p == t for h, p in zip(human, auto))
        fn = sum(h == t and p != t for h, p in zip(human, auto))
        p, r, f = prf(tp, fp, fn)
        per[t] = {"precision": p, "recall": r, "f1": f, "n_human": human.count(t), "n_auto": auto.count(t)}
        if human.count(t) or auto.count(t):
            f1s.append(f)
        print(f"{t:<5}{p:>7.2f}{r:>8.2f}{f:>7.2f}{human.count(t):>9}")
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    print(f"Macro F1: {macro:.3f}")
    g = per["F1"]
    print(f"\nGENUINE OVERRIDE (F1): precision {g['precision']:.2f}, recall {g['recall']:.2f}, "
          f"F1 {g['f1']:.2f}; human-confirmed F1 cases: {g['n_human']}")
    if g["n_human"] < 40:
        print("!! Fewer than 40 confirmed F1 cases in the sample. Exp 4 needs held-out F1 cases --")
        print("!! plan to label additional predicted-F1 cases beyond this sample.")

    with open(d / "validated_labels.jsonl", "w") as f:
        for i, h in zip(ids, human):
            k = key[i]
            f.write(json.dumps({"question_id": k["question_id"], "model": k["model"], "dataset": k["dataset"],
                                "human_type": h, "auto_type": k["auto_type"],
                                "original_label": k["original_label"], "how": adj[i]["how"]}) + "\n")
    json.dump({"n": len(ids), "accuracy": acc, "macro_f1": macro, "per_type": per,
               "confusion_human_rows_auto_cols": {r: dict(c) for r, c in matrix(human, auto, TYPES).items()}},
              open(d / "validation_report.json", "w"), indent=2)
    print(f"\nWrote {d / 'validated_labels.jsonl'} (ground truth for Exp 3-7)")
    print(f"Wrote {d / 'validation_report.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sample", "agreement", "validate"])
    ap.add_argument("--exp1", default="data/final/Phase_05/exp1_failure_types.jsonl")
    ap.add_argument("--retrieval", default="data/final/Phase_02/retrieval_results.jsonl")
    ap.add_argument("--outdir", default="data/final/Phase_05/exp2")
    ap.add_argument("--n", type=int, default=TARGET_N)
    ap.add_argument("--single", action="store_true")
    a = ap.parse_args()
    {"sample": cmd_sample, "agreement": cmd_agreement, "validate": cmd_validate}[a.step](a)


if __name__ == "__main__":
    main()