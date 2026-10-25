"""
Human batch 3 (Experiment 4, Option B). No GPU. Same annotation sheet, guidelines and
adjudication rules as batches 1 and 2; two annotators label every item, blind.

What goes in (shuffled together; nobody can tell which group an item is from), all fixed
by the plan written on the GPU box (data/final/Phase_06/plan/feature_cases.jsonl, seed 2026):
  candidate   every F1 candidate of the main enrichment run in blocks 1-3 (both passage
              conditions). Block 4 is a reserve, labelled only if a model is short of 40.
  probe       5 per model x condition: other wrong answers (partial / F2 / F3 / F6), to
              estimate how many genuine overrides the classifier missed
  followed    10 per model x condition: closed-book wrong, with-passage correct -- the
              faithful controls of Experiment 4 (correction-6 style error rate)

Steps (run from the repo root):
  python Tests/experiment2c_batch3.py sample                -> sheets for the two annotators
  python Tests/experiment2c_batch3.py agreement             -> after both sheets are back
  python Tests/experiment2c_batch3.py report                -> after adjudication is filled
  python Tests/experiment2c_batch3.py sample --reserve      -> only if the report says a model is short:
                                                               block-4 candidates, folder exp2c_reserve

Files: data/final/Phase_06/exp2c/  (HIDDEN_key.csv must not reach the annotators -- the repo is
public, so do not push it until both sheets are back)
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from experiment2b_batch2 import (ADJUDICATION_RULES, ANNOT_COLS, GUIDELINES, TYPES, boot_kappa, found_in, kappa,
                                 load_jsonl, matrix, print_matrix, ptext, read_csv, read_sheet, read_table,
                                 validate_sheet, wilson, write_adjudication_xlsx, write_csv, write_sheet)

SEED = 2026
MODELS = ("gemma", "llama")
CONDS = ("gold", "mixed")
GROUPS = ("candidate", "probe", "followed")
TARGET = 40
NOTE = """BATCH 3 NOTE -- the rules below are exactly the batch-1 and batch-2 rules. In addition:
  * This batch mixes different kinds of cases, and some model answers are correct.
    If the model's answer is right, choose F5.
  * Some items show one passage, others several separated by ---. Judge against all of them.
  * The same question can appear more than once (different model or different passages).
    Judge every item on its own.
  * The batch is long: take breaks, and keep applying the same rules from the first item to the last.
  * Reminder of the F2 rule (more than half of the batch-2 disagreements involved F2): F2 needs the model's wrong
    answer, or the fact it states, to be WRITTEN IN THE PASSAGE. Before choosing F2, find that
    answer in the passage. If the passage states the correct answer and the model's wrong answer
    is not in the passage, the item is not F2.

"""
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")     # characters Excel cannot store


def clean(t):
    return _CTRL.sub("", t if isinstance(t, str) else str(t))


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
    guide = NOTE + GUIDELINES
    files = []
    for name, s in (("A", SEED + 21), ("B", SEED + 22)):
        sheet = items[:]
        random.Random(s).shuffle(sheet)
        files.append(write_sheet(str(out / f"annotation_sheet_{name}"), sheet, guide))
    key_cols = ["item_id", "group", "case_id", "pid", "model", "cond", "block", "auto_type", "sampling_weight",
                "closed_book_answer", "cb_mean_logprob"]
    write_csv(out / "HIDDEN_key.csv", [{c: it.get(c, "") for c in key_cols} for it in items], key_cols)
    (out / "annotation_guidelines.txt").write_text(guide, encoding="utf-8")
    (out / ".gitignore").write_text("HIDDEN_key.csv\n", encoding="utf-8")
    json.dump({"seed": SEED, "reserve": a.reserve, "n_items": len(items)}, open(out / "sample_info.json", "w"), indent=2)
    c = Counter((it["group"], it["model"]) for it in items)
    print(f"Batch 3{' (reserve, block 4)' if a.reserve else ''}: {len(items)} items")
    for g in GROUPS:
        if c[(g, "gemma")] or c[(g, "llama")]:
            print(f"  {g:<10} gemma {c[(g, 'gemma')]:>3}   llama {c[(g, 'llama')]:>3}")
    tab = " (the sheets also have a guidelines tab)" if files[0].endswith(".xlsx") else ""
    print(f"\nSend {Path(files[0]).name} to annotator A and {Path(files[1]).name} to annotator B, with "
          f"annotation_guidelines.txt{tab}.")
    print("Do NOT share or push HIDDEN_key.csv until both sheets are back (the repo is public).")


# ------------------------------------------------------------------ 2) agreement (same procedure as batch 2)
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
    edited = [i for i in common if any((A[i].get(c) or "").strip() != (B[i].get(c) or "").strip()
                                       for c in ("question", "gold_answers", "model_answer"))]
    if edited:
        print(f"note: {len(edited)} items where the two sheets' question/gold/answer text differs: {', '.join(edited)}"
              " -- the adjudication sheet shows sheet B's text")
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
    print(f"Percent agreement (6 types):  {100 * agree / len(ids):.1f}%")
    print(f"Cohen's kappa (6 types):      {k:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"Cohen's kappa (F1 vs rest):   {kb:.3f}  95% CI [{lob:.3f}, {hib:.3f}]\n")
    print_matrix(matrix(x, y, TYPES), TYPES, "A", "B")
    adj = []
    for i in common:
        a_t, b_t = A[i]["failure_type"], B[i]["failure_type"]
        if i in partial:
            how = "ONE_ANNOTATOR_NEEDS_DECISION" if (a_t or b_t) else "UNANNOTATED_NEEDS_DECISION"
        else:
            how = "agreed" if a_t == b_t else "NEEDS_ADJUDICATION"
        src = B[i] if B[i].get("question") else A[i]
        golds = [g for g in (src.get("gold_answers") or "").split(" | ") if g.strip()]
        adj.append({"item_id": i, "annot_A": a_t, "annot_B": b_t, "final_type": a_t if how == "agreed" else "",
                    "how": how, "rationale_A": A[i].get("rationale", ""), "rationale_B": B[i].get("rationale", ""),
                    "question": src.get("question", ""), "gold_answers": src.get("gold_answers", ""),
                    "model_answer": src.get("model_answer", ""), "retrieved_passages": src.get("retrieved_passages", ""),
                    "adjudication_note": "",
                    "model_answer_found_in_passage": found_in(src.get("model_answer", ""), src.get("retrieved_passages", "")),
                    "a_gold_answer_found_in_passage": "yes" if any(found_in(g, src.get("retrieved_passages", "")) == "yes"
                                                                   for g in golds) else "no",
                    **{f"{who}_{short}": S[i].get(col, "") for who, S in (("A", A), ("B", B))
                       for short, col in (("contains", "passage_contains_answer"), ("correct", "answer_correct"),
                                          ("supported", "answer_supported_by_passage"),
                                          ("conflicts", "answer_conflicts_with_passage"))}})
    cols = ["item_id", "annot_A", "annot_B", "final_type", "how", "rationale_A", "rationale_B", "question",
            "gold_answers", "model_answer", "retrieved_passages"]
    write_csv(d / "adjudication.csv", adj, cols)
    (d / "adjudication_rules.txt").write_text(ADJUDICATION_RULES, encoding="utf-8")
    write_adjudication_xlsx(d / "adjudication.xlsx", adj)
    json.dump({"n": len(ids), "single_or_unannotated_items": partial, "percent_agreement": agree / len(ids),
               "kappa": k, "kappa_ci": [lo, hi], "kappa_F1_vs_rest": kb, "kappa_F1_vs_rest_ci": [lob, hib],
               "disagreement_matrix": {r: dict(c) for r, c in matrix(x, y, TYPES).items()}},
              open(d / "agreement.json", "w"), indent=2)
    n_dis = sum(r["how"] == "NEEDS_ADJUDICATION" for r in adj)
    print(f"\n{n_dis} disagreements + {len(partial)} single-annotator items to decide -> adjudication.xlsx (yellow cells)")
    print("A third person applies the fixed rules (rules tab) to every row that is not 'agreed', without opening\n"
          f"HIDDEN_key.csv. Then run: python Tests/experiment2c_batch3.py report{' --reserve' if a.reserve else ''}")


# ------------------------------------------------------------------ 3) report
def load_final(d):
    srcs = [p for p in (d / "adjudication.xlsx", d / "adjudication.csv") if p.exists()]
    if not srcs:
        raise SystemExit(f"no adjudication.xlsx / adjudication.csv in {d} -- run the agreement step first")
    src = max(srcs, key=lambda q: q.stat().st_mtime)
    adj = {r["item_id"]: r for r in read_table(src) if r.get("item_id")}
    missing = [i for i, r in adj.items() if (r.get("final_type") or "").strip().upper() not in TYPES]
    if missing:
        raise SystemExit(f"{src.name}: {len(missing)} items still lack a valid final_type, e.g. {missing[:5]}")
    for r in adj.values():
        if r.get("how") != "agreed":
            r["how"] = "adjudicated" if r.get("how") in ("NEEDS_ADJUDICATION", "", None) else r["how"]
    key = {r["item_id"]: r for r in read_csv(d / "HIDDEN_key.csv")}
    return adj, key, src


def cmd_report(a):
    batches = [Path(a.outdir)] + ([Path(a.reserve_outdir)] if (Path(a.reserve_outdir) / "HIDDEN_key.csv").exists() else [])
    adj, key, H = {}, {}, {}
    for d in batches:
        ad, k, src = load_final(d)
        print(f"final labels from {d.name}/{src.name}")
        adj.update(ad)
        key.update(k)
    H = {i: adj[i]["final_type"].strip().upper() for i in adj}
    rep = {}

    print("\nAnnotator agreement by group (before adjudication):")
    for g in GROUPS:
        ids = [i for i in adj if key[i]["group"] == g]
        ag = sum(adj[i]["annot_A"] == adj[i]["annot_B"] for i in ids)
        fa = sum(H[i] == adj[i]["annot_A"] for i in ids)
        fb = sum(H[i] == adj[i]["annot_B"] for i in ids)
        print(f"  {g:<10} n={len(ids):<4} A=B {ag:>3} ({ag / max(len(ids), 1):.0%})   final=A {fa:>3}   final=B {fb:>3}")
        rep[f"agreement_{g}"] = {"n": len(ids), "A_equals_B": ag, "final_equals_A": fa, "final_equals_B": fb}
    sug = [i for i in adj if adj[i]["how"] == "adjudicated" and (adj[i].get("claude_suggestion") or "").strip()]
    if sug:
        same = sum(H[i] == adj[i]["claude_suggestion"].strip().upper() for i in sug)
        print(f"Adjudicated items with an AI-drafted suggestion: {len(sug)}; final = suggestion on {same}, "
              f"changed on {len(sug) - same} (report this in the methods)")
        rep["adjudication_suggestions"] = {"n": len(sug), "accepted": same, "changed": len(sug) - same}
    print("Rule check (F2 needs the wrong answer to be written in the passage), short answers only:")
    for who in ("A", "B"):
        f2 = [i for i in adj if adj[i][f"annot_{who}"] == "F2" and len(adj[i]["model_answer"].split()) <= 5]
        off = sum(found_in(adj[i]["model_answer"], adj[i]["retrieved_passages"]) == "no" for i in f2)
        print(f"  annotator {who}: {off} of {len(f2)} F2 labels are on answers not found in the passage")
        rep[f"rule_check_F2_{who}"] = {"short_answer_F2": len(f2), "answer_not_in_passage": off}

    def dist(ids):
        c = Counter(H[i] for i in ids)
        return ", ".join(f"{t}={c[t]}" for t in TYPES if c[t])

    def ci(k, n):
        lo, hi = wilson(k, n)
        return f"{k}/{n} = {k / n:.2f} [{lo:.2f},{hi:.2f}]" if n else "n/a"

    print("=" * 78 + "\nBATCH 3 RESULTS (human final labels)\n" + "=" * 78)
    pilot = [r for r in load_jsonl(a.batch2) if r.get("group") == "pilot" and r.get("human_type") == "F1"]
    f1_rows = []
    for r in pilot:
        f1_rows.append({"case_id": f"{r['question_id']}|{r['cond']}", "model": r["model"], "source_run": "pilot",
                        "item_id": r["item_id"]})
    print("\nF1 candidates confirmed by humans (precision of the rebuilt classifier):")
    for m in MODELS:
        for cond in CONDS:
            ids = [i for i in adj if key[i]["group"] == "candidate" and key[i]["model"] == m and key[i]["cond"] == cond]
            ok = [i for i in ids if H[i] == "F1"]
            print(f"  {m:<6} [{cond}] {ci(len(ok), len(ids))}   human types of all candidates: {dist(ids)}")
            rep[f"candidates_{m}_{cond}"] = {"n": len(ids), "F1": len(ok), "types": dict(Counter(H[i] for i in ids))}
            f1_rows += [{"case_id": key[i]["case_id"], "model": m, "source_run": "main", "item_id": i} for i in ok]
    print("\nValidated genuine-override cases available for Experiment 4 (pilot + main run):")
    short = []
    for m in MODELS:
        n_p = sum(r["model"] == m and r["source_run"] == "pilot" for r in f1_rows)
        n_m = sum(r["model"] == m and r["source_run"] == "main" for r in f1_rows)
        tot = n_p + n_m
        flag = "OK" if tot >= TARGET else f"SHORT by {TARGET - tot}"
        print(f"  {m:<6} pilot {n_p} + main {n_m} = {tot}   (target {TARGET}: {flag})")
        rep[f"validated_F1_{m}"] = {"pilot": n_p, "main": n_m, "total": tot, "target": TARGET}
        if tot < TARGET:
            short.append(m)
    print("\nProbe -- other wrong answers (where the classifier could have missed an override):")
    for m in MODELS:
        for cond in CONDS:
            ids = [i for i in adj if key[i]["group"] == "probe" and key[i]["model"] == m and key[i]["cond"] == cond]
            f1 = [i for i in ids if H[i] == "F1"]
            est = sum(float(key[i]["sampling_weight"] or 1) for i in f1)
            print(f"  {m:<6} [{cond}] human F1 {len(f1)} of {len(ids)} -> about {est:.1f} missed in blocks 1-3; types: {dist(ids)}")
            rep[f"probe_{m}_{cond}"] = {"n": len(ids), "F1": len(f1), "estimated_missed": est}
    print("\nFaithful controls (closed-book wrong, with-passage correct; should be F5):")
    for m in MODELS:
        ids = [i for i in adj if key[i]["group"] == "followed" and key[i]["model"] == m]
        ok = sum(H[i] == "F5" for i in ids)
        print(f"  {m:<6} confirmed correct {ci(ok, len(ids))}   human types: {dist(ids)}")
        rep[f"followed_{m}"] = {"n": len(ids), "confirmed": ok, "types": dict(Counter(H[i] for i in ids))}

    out = Path(a.outdir)
    with open(out / "validated_labels_batch3.jsonl", "w", encoding="utf-8") as f:
        for i in sorted(adj):
            k = key[i]
            f.write(json.dumps({"item_id": i, "group": k["group"], "case_id": k["case_id"], "pid": k["pid"],
                                "model": k["model"], "cond": k["cond"], "block": k["block"], "human_type": H[i],
                                "auto_type": k["auto_type"], "how": adj[i]["how"],
                                "ai_suggestion": (adj[i].get("claude_suggestion") or "").strip(),
                                "sampling_weight": k["sampling_weight"]}) + "\n")
    write_csv(out / "exp4_validated_F1.csv", f1_rows or [{"case_id": ""}], list((f1_rows or [{"case_id": ""}])[0].keys()))
    json.dump(rep, open(out / "batch3_report.json", "w"), indent=2)
    print(f"\nWrote {out / 'validated_labels_batch3.jsonl'}, {out / 'exp4_validated_F1.csv'}, {out / 'batch3_report.json'}")
    if short:
        print(f"\n{', '.join(short)} short of {TARGET}: label the block-4 reserve with "
              "python Tests/experiment2c_batch3.py sample --reserve (then agreement --reserve, report).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sample", "agreement", "report"])
    ap.add_argument("--reserve", action="store_true", help="block-4 reserve candidates (separate folder)")
    ap.add_argument("--plan-dir", default="data/final/Phase_06/plan")
    ap.add_argument("--batch2", default="data/final/Phase_05/exp2b/validated_labels_batch2.jsonl")
    ap.add_argument("--outdir", default="data/final/Phase_06/exp2c")
    ap.add_argument("--reserve-outdir", default="data/final/Phase_06/exp2c_reserve")
    a = ap.parse_args()
    {"sample": cmd_sample, "agreement": cmd_agreement, "report": cmd_report}[a.step](a)


if __name__ == "__main__":
    main()
