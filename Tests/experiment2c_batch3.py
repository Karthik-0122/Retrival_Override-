"""
Human batch 3 (Experiment 4, Option B). No GPU. Same annotation sheet as batches 1 and 2, with the
guide's clarifications of 25 October written in (they apply from batch 3 on, never retroactively):

  C1  'Appears in the passage' means present in ANY passage (distractors included, A4) in ANY role,
      including a negation or a related role. A case that passes every other F1 test but whose wrong
      answer is mentioned in a passage is F1-M ('F1-m', mentioned): reported separately, not in the
      primary set.
  C2  F2 also covers an answer to a different reading of the question. If the gold itself does not
      answer the question as asked, the case is F6.
  C3  F1 requires the same answer without the passage (judge same-answer prompt). Every candidate
      passes this by construction (the classifier's hard requirement); it is recorded per case.
  A5  (i) for every disagreement the adjudicator writes first_decision BEFORE seeing any AI-drafted
      suggestion; final_type may differ afterwards and the change rate is reported.
      (iii) before batch 3, both annotators label 20 calibration items together (from batch 2, not
      part of batch 3) -- `calibration` step.

What goes in (shuffled together; nobody can tell which group an item is from), fixed by the plan
written on the GPU box (data/final/Phase_06/plan/feature_cases.jsonl, seed 2026):
  candidate   every F1 candidate of the main run in blocks 1-3 (block 4 = reserve)
  probe       5 per model x condition: other wrong answers (missed overrides)
  followed    10 per model x condition: faithful controls (closed-book wrong, with-passage correct)

Steps (run from the repo root):
  python Tests/experiment2c_batch3.py calibration      -> 20-item calibration sheet (do this first)
  python Tests/experiment2c_batch3.py sample           -> sheets for the two annotators
  python Tests/experiment2c_batch3.py agreement        -> after both sheets are back
  python Tests/experiment2c_batch3.py report           -> after adjudication.xlsx is filled
  python Tests/experiment2c_batch3.py sample --reserve -> only if the report says Gemma is short

Files: data/final/Phase_06/exp2c/  (HIDDEN_key.csv and the plan reveal the candidates -- do not push
data/final/Phase_06 to the public repo until both sheets are back)
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import experiment2b_batch2 as b2
from experiment2b_batch2 import (ANNOT_COLS, GUIDELINES, boot_kappa, found_in, kappa, load_jsonl, matrix,
                                 print_matrix, ptext, read_csv, read_sheet, read_table, validate_sheet, wilson,
                                 write_csv, write_sheet)

SEED = 2026
MODELS = ("gemma", "llama")
CONDS = ("gold", "mixed")
GROUPS = ("candidate", "probe", "followed")
TYPES = ["F1", "F1-M", "F2", "F3", "F4", "F5", "F6"]
FLOOR = {"gemma": 40, "llama": 30}          # guide: no steering on Gemma below 40; Llama floor 30

# batch 3 uses the 7 labels everywhere the batch-2 helpers check or offer failure types
b2.ALLOWED["failure_type"] = set(TYPES)

NOTE = """BATCH 3 NOTE -- the batch-1/2 rules below still apply, with these additions (guide, 25 Oct 2026):

  * NEW LABEL F1-M ("mentioned"). Use it when everything points to F1 (the passage states the correct
    answer, the model gives a different, wrong answer that is not taken from the passage as its
    answer) BUT the wrong answer, or the fact it states, is MENTIONED somewhere in ANY passage --
    in any role, including a negation ("not the Florida Department of Transportation") or a related
    role. Check all passages, including the extra ones separated by ---.
  * F2 needs the model's wrong answer, or the fact it states, to be TAKEN FROM a passage (the model
    gives what a passage says, even from the wrong passage or for the wrong role). Before choosing F2,
    find that answer in the passage. If the passage states the correct answer and the model's wrong
    answer is not in any passage, it is not F2. (More than half of the batch-2 disagreements were here.)
  * F2 also covers answering a DIFFERENT READING of the question (e.g. "domain" read as a biological
    domain when the question is about internet domains).
  * If the gold answer itself does not answer the question as asked (malformed question, gold answers
    something else), choose F6.
  * If the model's answer is correct (any wording), choose F5. Some items are correct on purpose.
  * The same question can appear more than once (different model or different passages). Judge each
    item on its own. The batch is long: take breaks and apply the same rules to the last item.

"""

RULES = """ADJUDICATION RULES (batch 3) -- fixed BEFORE any disagreement is resolved.
Adjudicator: a third person (not annotator A or B) with all passages open.
Apply the checks in this order; the first that fits decides the final type.

  1. F5   The answer is correct for the question (alias, other wording, or the gold is demonstrably wrong).
  2. F4   Right but less specific than required (year only, broader category, part of a multi-part answer).
  3. F3   No passage states the gold answer (or any valid answer).
  4. F6   The gold itself does not answer the question as asked (C2), or the model gives no answer.
  5. F2   The wrong answer, or the fact it states, is taken from a passage (any of the passages, A4),
          or the answer addresses a different reading of the question (C2).
  6. F1-M The passage states the gold, the wrong answer is not taken from a passage, but it (or the
          fact it states) is MENTIONED in some passage in any role, incl. a negation or related role (C1).
  7. F1   The passage states the gold and the wrong answer does not appear in any passage.
  8. F6   Still undecidable.
C3 (same answer without the passage) is checked from the data, not by the adjudicator: every candidate
passes it by construction. If the adjudicator cannot decide with these rules, the guide decides.

HOW TO USE THE ADJUDICATION SHEET (A5)
  1. For every yellow row write first_decision BEFORE looking at any AI-drafted suggestion.
  2. Only then, if suggestions are provided, you may read them and write final_type. Leave final_type
     empty to keep first_decision. The report counts how often final_type differs from first_decision.
  3. Do not change agreed rows. Do not open HIDDEN_key.csv before adjudication is finished.
"""

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")     # characters Excel cannot store


def clean(t):
    return _CTRL.sub("", t if isinstance(t, str) else str(t))


def guide_text():
    return NOTE + GUIDELINES + "\nF1-M  see the batch 3 note at the top.\n"


# ------------------------------------------------------------------ 0) calibration (A5 iii)
def cmd_calibration(a):
    """20 batch-2 items with known adjudicated labels, labelled by both annotators TOGETHER, with
    discussion, before batch 3. Not part of batch 3; the session is reported."""
    out = Path(a.outdir) / "calibration"
    if any(out.glob("calibration_sheet*")):
        raise SystemExit(f"{out} already has a calibration sheet -- refusing to overwrite it")
    d2 = Path(a.batch2_dir)
    adj = read_table(d2 / "adjudication.xlsx")
    key = {r["item_id"]: r for r in read_csv(d2 / "HIDDEN_key.csv")}
    pool = [r for r in adj if key[r["item_id"]]["group"] in ("pilot", "probe")]
    want = {"F1": 4, "F2": 5, "F5": 4, "F3": 3, "F4": 2, "F6": 2}
    rng = random.Random(SEED + 7)
    items = []
    for t, n in want.items():
        cand = sorted([r for r in pool if r["final_type"] == t], key=lambda r: r["item_id"])
        rng.shuffle(cand)
        cand.sort(key=lambda r: r["annot_A"] == r["annot_B"])      # disagreements first: they teach most
        items += cand[:n]
    rng.shuffle(items)
    rows = []
    for i, r in enumerate(items, 1):
        rows.append({"item_id": f"cal_{i:02d}", "question": r["question"], "gold_answers": r["gold_answers"],
                     "model_answer": r["model_answer"], "retrieved_passages": r["retrieved_passages"],
                     **{c: "" for c in ANNOT_COLS}, "_b2": r})
    out.mkdir(parents=True, exist_ok=True)
    path = write_sheet(str(out / "calibration_sheet"), rows, guide_text())
    from openpyxl import load_workbook
    wb = load_workbook(path)
    k = wb.create_sheet("after_discussion")
    k.append(["item_id", "batch-2 final label", "batch-2 annotator A", "batch-2 annotator B", "note"])
    for r in rows:
        x = r["_b2"]
        note = ("annotators disagreed in batch 2" if x["annot_A"] != x["annot_B"] else "")
        k.append([r["item_id"], x["final_type"], x["annot_A"], x["annot_B"], note])
    for col, w in zip("ABCDE", (10, 20, 20, 20, 35)):
        k.column_dimensions[col].width = w
    wb.save(path)
    write_csv(out / "calibration_items.csv", [{"item_id": r["item_id"], "batch2_item": r["_b2"]["item_id"],
                                               "batch2_final": r["_b2"]["final_type"]} for r in rows],
              ["item_id", "batch2_item", "batch2_final"])
    print(f"Calibration: {len(rows)} batch-2 items -> {path}")
    print("  Both annotators label them TOGETHER, discuss every item, and only then open the "
          "'after_discussion' tab.\n  Keep the filled sheet: the session is reported (guide's A5 iii).")


# ------------------------------------------------------------------ 1) sample
def cmd_sample(a):
    out = Path(a.outdir if not a.reserve else a.reserve_outdir)
    if any(out.glob("annotation_sheet_*")):
        raise SystemExit(f"{out} already has annotation sheets -- refusing to overwrite them. "
                         "Delete the folder only if the annotators have not started.")
    plan = load_jsonl(Path(a.plan_dir) / "feature_cases.jsonl")
    if not plan:
        raise SystemExit(f"no plan in {a.plan_dir} -- copy the GPU results (data/final/Phase_06) into the repo first")
    want = ("candidate_reserve",) if a.reserve else GROUPS
    items = []
    for r in plan:
        if r.get("batch3_group") not in want:
            continue
        items.append({"group": "candidate" if r["batch3_group"] == "candidate_reserve" else r["batch3_group"],
                      "case_id": r["case_id"], "pid": r["pid"], "model": r["model"], "cond": r["cond"],
                      "block": r["block"], "auto_type": r["outcome"], "sampling_weight": r.get("sampling_weight") or 1.0,
                      "closed_book_answer": clean(r["closed_book_answer"]), "cb_mean_logprob": r["cb_mean_logprob"],
                      "question": clean(r["question"]), "gold_answers": clean(" | ".join(r["answers"])),
                      "model_answer": clean(r["model_answer"]), "retrieved_passages": clean(ptext(r["passages"]))})
    if not items:
        raise SystemExit("nothing to label in the plan for these groups")
    rng = random.Random(SEED + (100 if a.reserve else 0))
    items.sort(key=lambda it: (it["group"], it["model"], it["case_id"]))
    rng.shuffle(items)
    tag = "b3r" if a.reserve else "b3"
    for i, it in enumerate(items, 1):
        it["item_id"] = f"{tag}_{i:03d}"
        for c in ANNOT_COLS:
            it[c] = ""
    out.mkdir(parents=True, exist_ok=True)
    guide = guide_text()
    files = []
    for name, s in (("A", SEED + 21), ("B", SEED + 22)):
        sheet = items[:]
        random.Random(s).shuffle(sheet)
        files.append(write_sheet(str(out / f"annotation_sheet_{name}"), sheet, guide))
    key_cols = ["item_id", "group", "case_id", "pid", "model", "cond", "block", "auto_type", "sampling_weight",
                "closed_book_answer", "cb_mean_logprob"]
    write_csv(out / "HIDDEN_key.csv", [{c: it.get(c, "") for c in key_cols} for it in items], key_cols)
    (out / "annotation_guidelines.txt").write_text(guide, encoding="utf-8")
    (out / "adjudication_rules.txt").write_text(RULES, encoding="utf-8")
    (out / ".gitignore").write_text("HIDDEN_key.csv\n", encoding="utf-8")
    json.dump({"seed": SEED, "reserve": a.reserve, "n_items": len(items)}, open(out / "sample_info.json", "w"), indent=2)
    c = Counter((it["group"], it["model"]) for it in items)
    print(f"Batch 3{' (reserve, block 4)' if a.reserve else ''}: {len(items)} items")
    for g in GROUPS:
        if c[(g, "gemma")] or c[(g, "llama")]:
            print(f"  {g:<10} gemma {c[(g, 'gemma')]:>3}   llama {c[(g, 'llama')]:>3}")
    print(f"\nSend {Path(files[0]).name} to annotator A and {Path(files[1]).name} to annotator B, with "
          "annotation_guidelines.txt (also in the sheets' guidelines tab). Run the calibration session first.")
    print("Do NOT share or push HIDDEN_key.csv or data/final/Phase_06 until both sheets are back (the repo is public).")


# ------------------------------------------------------------------ 2) agreement
def write_adjudication(path, rows):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    cols = ["item_id", "how", "annot_A", "annot_B", "first_decision", "final_type", "adjudication_note",
            "model_answer_found_in_passage", "a_gold_answer_found_in_passage", "A_contains", "B_contains",
            "A_correct", "B_correct", "A_supported", "B_supported", "rationale_A", "rationale_B", "question",
            "gold_answers", "model_answer", "retrieved_passages"]
    order = sorted(rows, key=lambda r: (r["how"] == "agreed", r["annot_A"], r["annot_B"], r["item_id"]))
    wb = Workbook()
    ws = wb.active
    ws.title = "adjudicate"
    ws.append(cols)
    for r in order:
        ws.append([r.get(c, "") for c in cols])
    widths = {"item_id": 9, "how": 22, "first_decision": 13, "final_type": 11, "adjudication_note": 40,
              "rationale_A": 45, "rationale_B": 45, "question": 40, "gold_answers": 25, "model_answer": 30,
              "retrieved_passages": 90}
    todo = PatternFill("solid", fgColor="FFF2CC")
    for i, c in enumerate(cols, 1):
        col = ws.cell(row=1, column=i).column_letter
        ws.column_dimensions[col].width = widths.get(c, 12)
        ws.cell(row=1, column=i).font = Font(bold=True)
    for name in ("first_decision", "final_type"):
        dv = DataValidation(type="list", formula1='"' + ",".join(TYPES) + '"', allow_blank=True, showErrorMessage=True)
        ws.add_data_validation(dv)
        col = ws.cell(row=1, column=cols.index(name) + 1).column_letter
        dv.add(f"{col}2:{col}{len(order) + 1}")
    wrap = Alignment(wrap_text=True, vertical="top")
    for k, row in enumerate(ws.iter_rows(min_row=2, max_row=len(order) + 1)):
        for cell in row:
            cell.alignment = wrap
        if order[k]["how"] != "agreed":
            row[cols.index("first_decision")].fill = todo
    ws.freeze_panes = "F2"
    g = wb.create_sheet("rules")
    g.column_dimensions["A"].width = 110
    for line in RULES.splitlines():
        g.append([line])
    wb.save(path)


def cmd_agreement(a):
    d = Path(a.outdir if not a.reserve else a.reserve_outdir)
    A_rows, pa = read_sheet(d, "A")
    A = {r["item_id"]: r for r in A_rows}
    validate_sheet(list(A.values()), "A")
    B_rows, pb = read_sheet(d, "B")
    B = {r["item_id"]: r for r in B_rows}
    validate_sheet(list(B.values()), "B")
    print(f"read {pa.name} and {pb.name}")
    common = sorted(set(A) & set(B))
    ids = [i for i in common if A[i]["failure_type"] and B[i]["failure_type"]]
    partial = [i for i in common if i not in ids]
    if not ids:
        raise SystemExit("no item was annotated by both annotators")
    x, y = [A[i]["failure_type"] for i in ids], [B[i]["failure_type"] for i in ids]
    k = kappa(x, y, TYPES)
    lo, hi = boot_kappa(x, y, TYPES)
    xb, yb = ["F1" if v == "F1" else "other" for v in x], ["F1" if v == "F1" else "other" for v in y]
    kb = kappa(xb, yb, ["F1", "other"])
    lob, hib = boot_kappa(xb, yb, ["F1", "other"])
    agree = sum(p == q for p, q in zip(x, y))
    print(f"Items annotated by both: {len(ids)}")
    print(f"Percent agreement (7 labels): {100 * agree / len(ids):.1f}%")
    print(f"Cohen's kappa (7 labels):     {k:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"Cohen's kappa (F1 vs rest):   {kb:.3f}  95% CI [{lob:.3f}, {hib:.3f}]\n")
    print_matrix(matrix(x, y, TYPES), TYPES, "A", "B")
    adj = []
    for i in common:
        a_t, b_t = A[i]["failure_type"], B[i]["failure_type"]
        how = ("ONE_ANNOTATOR_NEEDS_DECISION" if (a_t or b_t) else "UNANNOTATED_NEEDS_DECISION") if i in partial \
            else ("agreed" if a_t == b_t else "NEEDS_ADJUDICATION")
        src = B[i] if B[i].get("question") else A[i]
        golds = [g for g in (src.get("gold_answers") or "").split(" | ") if g.strip()]
        adj.append({"item_id": i, "annot_A": a_t, "annot_B": b_t, "how": how,
                    "first_decision": a_t if how == "agreed" else "", "final_type": a_t if how == "agreed" else "",
                    "rationale_A": A[i].get("rationale", ""), "rationale_B": B[i].get("rationale", ""),
                    "question": src.get("question", ""), "gold_answers": src.get("gold_answers", ""),
                    "model_answer": src.get("model_answer", ""), "retrieved_passages": src.get("retrieved_passages", ""),
                    "adjudication_note": "",
                    "model_answer_found_in_passage": found_in(src.get("model_answer", ""), src.get("retrieved_passages", "")),
                    "a_gold_answer_found_in_passage": "yes" if any(found_in(g, src.get("retrieved_passages", "")) == "yes"
                                                                   for g in golds) else "no",
                    **{f"{who}_{short}": S[i].get(col, "") for who, S in (("A", A), ("B", B))
                       for short, col in (("contains", "passage_contains_answer"), ("correct", "answer_correct"),
                                          ("supported", "answer_supported_by_passage"))}})
    write_csv(d / "adjudication.csv", adj, ["item_id", "annot_A", "annot_B", "first_decision", "final_type", "how",
                                           "rationale_A", "rationale_B", "question", "gold_answers", "model_answer"])
    (d / "adjudication_rules.txt").write_text(RULES, encoding="utf-8")
    write_adjudication(d / "adjudication.xlsx", adj)
    json.dump({"n": len(ids), "single_or_unannotated_items": partial, "percent_agreement": agree / len(ids),
               "kappa": k, "kappa_ci": [lo, hi], "kappa_F1_vs_rest": kb, "kappa_F1_vs_rest_ci": [lob, hib],
               "disagreement_matrix": {r: dict(c) for r, c in matrix(x, y, TYPES).items()}},
              open(d / "agreement.json", "w"), indent=2)
    n_dis = sum(r["how"] != "agreed" for r in adj)
    print(f"\n{n_dis} items to decide -> adjudication.xlsx: write first_decision (yellow) for each, BEFORE any "
          "AI suggestion.\nThen run: python Tests/experiment2c_batch3.py report")


# ------------------------------------------------------------------ 3) report
def load_final(d):
    srcs = [p for p in (d / "adjudication.xlsx", d / "adjudication.csv") if p.exists()]
    if not srcs:
        raise SystemExit(f"no adjudication file in {d} -- run the agreement step first")
    src = max(srcs, key=lambda q: q.stat().st_mtime)
    adj = {r["item_id"]: r for r in read_table(src) if r.get("item_id")}
    bad = []
    for i, r in adj.items():
        first = (r.get("first_decision") or "").strip().upper()
        final = (r.get("final_type") or "").strip().upper() or first
        if r.get("how") != "agreed" and first not in TYPES:
            bad.append(i)
        r["first_decision"], r["final_type"] = first, final
        if final not in TYPES:
            bad.append(i)
        if r.get("how") != "agreed":
            r["how"] = "adjudicated"
    if bad:
        raise SystemExit(f"{src.name}: {len(set(bad))} rows lack a valid first_decision / final_type, e.g. {sorted(set(bad))[:5]}")
    key = {r["item_id"]: r for r in read_csv(d / "HIDDEN_key.csv")}
    return adj, key, src


def cmd_report(a):
    batches = [Path(a.outdir)] + ([Path(a.reserve_outdir)] if (Path(a.reserve_outdir) / "HIDDEN_key.csv").exists() else [])
    adj, key = {}, {}
    for d in batches:
        ad, k, src = load_final(d)
        print(f"final labels from {d.name}/{src.name}")
        adj.update(ad)
        key.update(k)
    H = {i: adj[i]["final_type"] for i in adj}
    rep = {}
    for d in batches:
        if (d / "agreement.json").exists():
            ag = json.load(open(d / "agreement.json"))
            rep[f"kappa_{d.name}"] = {"kappa": ag["kappa"], "kappa_ci": ag["kappa_ci"],
                                      "kappa_F1_vs_rest": ag["kappa_F1_vs_rest"], "ci": ag["kappa_F1_vs_rest_ci"]}
            print(f"{d.name}: kappa {ag['kappa']:.3f} {ag['kappa_ci']}, F1 vs rest {ag['kappa_F1_vs_rest']:.3f}")
    adjd = [i for i in adj if adj[i]["how"] == "adjudicated"]
    changed = sum(adj[i]["first_decision"] != H[i] for i in adjd)
    print(f"Adjudicated items: {len(adjd)}; final differs from first_decision (before any AI draft) on {changed}")
    rep["adjudication"] = {"n": len(adjd), "changed_after_suggestion": changed}

    def dist(ids):
        c = Counter(H[i] for i in ids)
        return ", ".join(f"{t}={c[t]}" for t in TYPES if c[t])

    def ci(k, n):
        lo, hi = wilson(k, n)
        return f"{k}/{n} = {k / n:.2f} [{lo:.2f},{hi:.2f}]" if n else "n/a"

    print("=" * 78 + "\nBATCH 3 RESULTS (human final labels)\n" + "=" * 78)
    pilot = [r for r in load_jsonl(a.batch2) if r.get("group") == "pilot" and r.get("human_type") == "F1"]
    excluded = {x.strip() for x in a.pilot_exclude.split(",") if x.strip()}
    f1_rows = [{"case_id": f"{r['question_id']}|{r['cond']}", "model": r["model"], "cond": r["cond"],
                "source_run": "pilot", "item_id": r["item_id"], "both_agreed": "batch2"}
               for r in pilot if r["item_id"] not in excluded]
    print("\nF1 candidates: human labels (precision of the rebuilt classifier):")
    for m in MODELS:
        for cond in CONDS:
            ids = [i for i in adj if key[i]["group"] == "candidate" and key[i]["model"] == m and key[i]["cond"] == cond]
            ok = [i for i in ids if H[i] == "F1"]
            print(f"  {m:<6} [{cond}] F1 {ci(len(ok), len(ids))}; F1-M {sum(H[i] == 'F1-M' for i in ids)}; all: {dist(ids)}")
            rep[f"candidates_{m}_{cond}"] = {"n": len(ids), "F1": len(ok), "types": dict(Counter(H[i] for i in ids))}
            f1_rows += [{"case_id": key[i]["case_id"], "model": m, "cond": cond, "source_run": "main", "item_id": i,
                         "both_agreed": "yes" if adj[i]["how"] == "agreed" else "no"} for i in ok]
    print("\nValidated F1 (primary set) for Experiment 4: pilot (after the C1/C2 re-screen) + main run")
    short = []
    for m in MODELS:
        rows = [r for r in f1_rows if r["model"] == m]
        qs = {r["case_id"].split("|")[0] for r in rows}
        agreed = sum(r["both_agreed"] == "yes" for r in rows)
        by_cond = Counter(r["cond"] for r in rows)
        flag = "OK" if len(qs) >= FLOOR[m] else f"SHORT by {FLOOR[m] - len(qs)}"
        print(f"  {m:<6} cases {len(rows)} (gold {by_cond['gold']}, mixed {by_cond['mixed']}); questions {len(qs)}; "
              f"main-run cases both annotators labelled F1: {agreed}   (floor {FLOOR[m]} questions: {flag})")
        rep[f"validated_F1_{m}"] = {"cases": len(rows), "questions": len(qs), "by_condition": dict(by_cond),
                                   "main_both_agreed": agreed, "floor": FLOOR[m]}
        if len(qs) < FLOOR[m]:
            short.append(m)
    print("\nProbe (other wrong answers -- missed overrides):")
    for m in MODELS:
        for cond in CONDS:
            ids = [i for i in adj if key[i]["group"] == "probe" and key[i]["model"] == m and key[i]["cond"] == cond]
            f1 = [i for i in ids if H[i] == "F1"]
            est = sum(float(key[i]["sampling_weight"] or 1) for i in f1)
            print(f"  {m:<6} [{cond}] F1 {len(f1)} of {len(ids)} -> about {est:.1f} missed in blocks 1-3; {dist(ids)}")
            rep[f"probe_{m}_{cond}"] = {"n": len(ids), "F1": len(f1), "estimated_missed": est}
    print("\nFaithful controls (should be F5):")
    for m in MODELS:
        ids = [i for i in adj if key[i]["group"] == "followed" and key[i]["model"] == m]
        ok = sum(H[i] == "F5" for i in ids)
        print(f"  {m:<6} {ci(ok, len(ids))}; {dist(ids)}")
        rep[f"followed_{m}"] = {"n": len(ids), "confirmed": ok, "types": dict(Counter(H[i] for i in ids))}
    out = Path(a.outdir)
    with open(out / "validated_labels_batch3.jsonl", "w", encoding="utf-8") as f:
        for i in sorted(adj):
            k = key[i]
            f.write(json.dumps({"item_id": i, "group": k["group"], "case_id": k["case_id"], "pid": k["pid"],
                                "model": k["model"], "cond": k["cond"], "block": k["block"], "human_type": H[i],
                                "first_decision": adj[i]["first_decision"], "annot_A": adj[i]["annot_A"],
                                "annot_B": adj[i]["annot_B"], "auto_type": k["auto_type"], "how": adj[i]["how"],
                                "sampling_weight": k["sampling_weight"]}) + "\n")
    write_csv(out / "exp4_validated_F1.csv", f1_rows or [{"case_id": ""}], list((f1_rows or [{"case_id": ""}])[0].keys()))
    json.dump(rep, open(out / "batch3_report.json", "w"), indent=2)
    print(f"\nWrote {out}/validated_labels_batch3.jsonl, exp4_validated_F1.csv, batch3_report.json")
    if short:
        print(f"\n{', '.join(short)} below the floor: for Gemma label the block-4 reserve "
              "(sample --reserve); Llama shortfalls are reported, not filled.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["calibration", "sample", "agreement", "report"])
    ap.add_argument("--reserve", action="store_true", help="block-4 reserve candidates (separate folder)")
    ap.add_argument("--plan-dir", default="data/final/Phase_06/plan")
    ap.add_argument("--batch2", default="data/final/Phase_05/exp2b/validated_labels_batch2.jsonl")
    ap.add_argument("--batch2-dir", default="data/final/Phase_05/exp2b")
    ap.add_argument("--pilot-exclude", default="b2_025", help="pilot F1 items removed by the C1 re-screen")
    ap.add_argument("--outdir", default="data/final/Phase_06/exp2c")
    ap.add_argument("--reserve-outdir", default="data/final/Phase_06/exp2c_reserve")
    a = ap.parse_args()
    {"calibration": cmd_calibration, "sample": cmd_sample, "agreement": cmd_agreement, "report": cmd_report}[a.step](a)


if __name__ == "__main__":
    main()
