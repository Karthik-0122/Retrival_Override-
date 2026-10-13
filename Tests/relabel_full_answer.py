"""
Relabel the dataset with full-answer scoring. No GPU -- pure Python on the
answers the models already generated in Phase 2.

Phase 2 labeled an answer 'override' when it didn't match the gold answer
closely enough. That marks correct answers written as full sentences as
failures, e.g. "The torch is held in the right hand" for gold "Right".
This script re-checks every answer the same way steering_improved.py does:
an answer is correct if any gold alias appears in it (case, punctuation and
articles ignored; PopQA's JSON-list aliases parsed).

It reports, per model and per source:
  * how many 'override' labels were actually correct answers,
  * what that means for the 100-query override test set and the six flips,
  * examples to eyeball, and short gold answers that need a manual look
    (a short gold like "one" can appear inside a wrong answer by accident).

ConFiQA is left untouched: its 'override' means the opposite (resisting a
false passage), so full-answer rescoring against its gold doesn't apply.

It does NOT overwrite anything. It writes a new copy with both labels.

Run from repo root:
    python Tests/relabel_full_answer.py

Writes:
    data/final/Phase_02/analysis_dataset_relabeled.jsonl
    data/final/Phase_02/relabel_summary.json
"""

import argparse
import json
import random
import re
import string
from collections import defaultdict
from pathlib import Path

MODELS = ("gemma", "llama")
SIX_FLIPS = ["triviaqa_17766", "nq_00276", "nq_01389", "nq_02413", "triviaqa_09779", "nq_02986"]
SHORT_GOLD_CHARS = 3
N_EXAMPLES = 10


# ----------------------------------------------------------------- helpers
def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def parse_gold_answers(gold_answer_text):
    """Plain string -> [string]. JSON-list string (the PopQA format) -> list of aliases."""
    if not gold_answer_text:
        return []
    text = gold_answer_text.strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return [gold_answer_text]


_PUNCT = set(string.punctuation)


def normalize(text):
    text = "".join(ch for ch in text.lower() if ch not in _PUNCT)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def contains_answer(generated, aliases):
    gen = " " + normalize(generated) + " "
    return any(" " + normalize(a) + " " in gen for a in aliases if normalize(a))


def relabel(record, model):
    """Returns (new_label, status)."""
    old = record.get(f"{model}_label")
    if old not in ("override", "faithful"):
        return old, "not_labeled"
    if record.get("source_category") == "confiqa":
        return old, "confiqa_skipped"
    aliases = [a for a in parse_gold_answers(record.get("gold_answer_text")) if normalize(a)]
    if not aliases:
        return old, "no_valid_gold"
    answer = record.get(f"{model}_answer")
    if not answer:
        return old, "no_answer"
    return ("faithful" if contains_answer(answer, aliases) else "override"), "ok"


def short_gold(record):
    aliases = [a for a in parse_gold_answers(record.get("gold_answer_text")) if normalize(a)]
    return aliases and min(len(normalize(a)) for a in aliases) <= SHORT_GOLD_CHARS


def words(text):
    return len((text or "").split())


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/Phase_02/analysis_dataset.jsonl")
    ap.add_argument("--test-sample", default="data/final/Phase_04/ablation_test_sample.json")
    ap.add_argument("--out", default="data/final/Phase_02/analysis_dataset_relabeled.jsonl")
    ap.add_argument("--summary", default="data/final/Phase_02/relabel_summary.json")
    args = ap.parse_args()

    records = load_jsonl(args.dataset)
    by_id = {r["query_id"]: r for r in records}
    summary = {}

    for model in MODELS:
        counts = defaultdict(lambda: defaultdict(int))
        mislabeled, flagged = [], []
        len_mislabeled, len_true = [], []

        for r in records:
            new, status = relabel(r, model)
            r[f"{model}_label_full"] = new
            r[f"{model}_relabel_status"] = status
            src = r.get("source_category", "unknown")
            counts[src][status] += 1
            old = r.get(f"{model}_label")
            if status != "ok":
                continue
            counts[src][f"old_{old}"] += 1
            if old == "override" and new == "faithful":
                counts[src]["override_to_faithful"] += 1
                mislabeled.append(r)
                len_mislabeled.append(words(r.get(f"{model}_answer")))
                if short_gold(r):
                    flagged.append(r)
            elif old == "override":
                len_true.append(words(r.get(f"{model}_answer")))
            elif old == "faithful" and new == "override":
                counts[src]["faithful_to_override"] += 1

        print(f"\n{'=' * 78}\n{model.upper()}: override labels re-checked with full-answer scoring\n{'=' * 78}")
        print(f"{'source':<22}{'old override':>14}{'really correct':>16}{'still override':>16}{'% wrong':>10}")
        tot_old = tot_bad = 0
        model_summary = {}
        for src in sorted(counts):
            c = counts[src]
            old_ov, bad = c.get("old_override", 0), c.get("override_to_faithful", 0)
            if old_ov == 0:
                continue
            tot_old += old_ov
            tot_bad += bad
            print(f"{src:<22}{old_ov:>14}{bad:>16}{old_ov - bad:>16}{100 * bad / old_ov:>9.1f}%")
            model_summary[src] = {"old_override": old_ov, "really_correct": bad,
                                  "faithful_to_override": c.get("faithful_to_override", 0)}
        if tot_old:
            print(f"{'ALL (excl. ConFiQA)':<22}{tot_old:>14}{tot_bad:>16}{tot_old - tot_bad:>16}{100 * tot_bad / tot_old:>9.1f}%")
        skipped = sum(counts[s].get("no_valid_gold", 0) + counts[s].get("no_answer", 0) for s in counts)
        print(f"(Skipped {skipped} records with no usable gold answer or no model answer.)")

        if len_mislabeled and len_true:
            print(f"\nAverage answer length: mislabeled 'override' = {sum(len_mislabeled) / len(len_mislabeled):.1f} words, "
                  f"true override = {sum(len_true) / len(len_true):.1f} words")

        print(f"\nExamples of 'override' labels that were actually correct (eyeball these):")
        random.Random(0).shuffle(mislabeled)
        for r in mislabeled[:N_EXAMPLES]:
            print(f"  {r['query_id']}: gold={r.get('gold_answer_text')!r}")
            print(f"      answer: {(r.get(f'{model}_answer') or '')[:140]!r}")

        if flagged:
            print(f"\n{len(flagged)} relabels have very short gold answers -- check by hand, "
                  f"they can match a wrong answer by accident:")
            for r in flagged[:N_EXAMPLES]:
                print(f"  {r['query_id']}: gold={r.get('gold_answer_text')!r} | "
                      f"answer: {(r.get(f'{model}_answer') or '')[:100]!r}")

        model_summary["_total"] = {"old_override": tot_old, "really_correct": tot_bad,
                                   "short_gold_flagged": len(flagged)}
        summary[model] = model_summary

    # ---- impact on the Phase 4 test set and the six flips (Gemma) ----
    print(f"\n{'=' * 78}\nIMPACT ON PHASE 4 (Gemma)\n{'=' * 78}")
    if Path(args.test_sample).exists():
        test = json.load(open(args.test_sample))
        ids = [c["query_id"] for c in test["gemma"]["override_cases"]]
        still = [q for q in ids if q in by_id and by_id[q].get("gemma_label_full") == "override"]
        print(f"Override test set: {len(still)} of {len(ids)} are still override under full-answer scoring.")
        summary["test_set"] = {"total": len(ids), "still_override": len(still)}
    else:
        print("(ablation_test_sample.json not found -- skipping test-set impact)")

    print("\nThe six 'flipped' cases:")
    survivors = 0
    for q in SIX_FLIPS:
        r = by_id.get(q)
        if not r:
            print(f"  {q}: not in dataset")
            continue
        new = r.get("gemma_label_full")
        survivors += new == "override"
        verdict = "real override" if new == "override" else "ALREADY CORRECT -- not a real fix"
        print(f"  {q}: gold={r.get('gold_answer_text')!r} -> {verdict}")
        print(f"      original answer: {(r.get('gemma_answer') or '')[:120]!r}")
    print(f"\n{survivors} of 6 flips were real override cases. Whether the steered answers were")
    print("actually correct still needs steering_improved.py (it saves the full steered text).")
    summary["six_flips_real_override"] = survivors

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}\nWrote {args.summary}")


if __name__ == "__main__":
    main()