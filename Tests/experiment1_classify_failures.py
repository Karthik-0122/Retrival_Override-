"""
Experiment 1 -- Automatic failure-type classification (guide's taxonomy).
No GPU.

Classifies EVERY apparent retrieval failure (original strict label 'override'
OR full-answer label 'override'; ConFiQA excluded because its 'override'
means resisting a false passage) into exactly one primary type:

  F1  Genuine parametric override   passage has the answer; the model's answer
                                    is not drawn from the passage AND matches
                                    what the model says with NO passage
  F2  Wrong extraction              answer drawn from the passage, wrong fact
  F3  Evidence absent               passage does not contain the gold answer
  F4  Partial / incomplete          answer contains part of the gold answer
  F5  Actually correct              gold answer is in the model's answer;
                                    strict matching marked it wrong
  F6  Ambiguous / needs review      signals can't separate the types

Critical rule (from the guide): F1 is NEVER assigned just because the answer
is wrong. Without a no-passage answer that matches, the case goes to F6.
So run generate_no_passage_answers.py first; if it is missing, the script
still runs but F1 will be empty and says so loudly.

Decision order (first match wins):
  empty answer                         -> F6
  original=override, full=faithful     -> F5
  gold not in passage                  -> F3 (F6 if the passage holds most of
                                          the gold's words: may be paraphrased)
  answer has >=50% of gold's words     -> F4
  >=50% of answer's 4-grams in passage -> F2
  no-passage answer available:
      matches with-passage answer (word overlap >=0.5) and is itself wrong -> F1
      otherwise                                                            -> F6
  no no-passage answer                 -> F6

The original data is not modified. The automatic labels are NOT ground
truth; Experiment 2 validates them by hand.

Run from repo root (after relabel_full_answer.py and, ideally,
generate_no_passage_answers.py):
    python Tests/experiment1_classify_failures.py

Writes:
    data/final/Phase_05/exp1_failure_types.csv      guide's columns, one row per case
    data/final/Phase_05/exp1_failure_types.jsonl    same + all raw signals
    data/final/Phase_05/exp1_summary.json           counts/percentages, examples
"""

import argparse
import csv
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

MODELS = ("gemma", "llama")
TYPES = ["F1", "F2", "F3", "F4", "F5", "F6"]
TYPE_NAMES = {
    "F1": "Genuine parametric override", "F2": "Wrong extraction / interpretation",
    "F3": "Evidence absent", "F4": "Partially correct / incomplete",
    "F5": "Actually correct / scoring error", "F6": "Ambiguous / needs human review",
}
COPY_T, PARTIAL_T, NOCTX_T, PARAPHRASE_T = 0.5, 0.5, 0.5, 0.5
SHORT_GOLD_CHARS = 3
STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is", "was",
             "are", "were", "be", "by", "with", "as", "from", "that", "this", "it", "its"}
_PUNCT = set(string.punctuation)


# ------------------------------------------------------------------ text utils
def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def parse_gold_answers(text):
    if not text:
        return []
    t = text.strip()
    if t.startswith("["):
        try:
            p = json.loads(t)
            if isinstance(p, list) and all(isinstance(x, str) for x in p):
                return p
        except (json.JSONDecodeError, TypeError):
            pass
    return [text]


def normalize(text):
    text = "".join(ch for ch in (text or "").lower() if ch not in _PUNCT)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def content_tokens(text):
    return [t for t in normalize(text).split() if t not in STOPWORDS]


def contains(hay_norm, needle):
    n = normalize(needle)
    return bool(n) and (" " + n + " ") in (" " + hay_norm + " ")


def gold_parts(alias):
    parts = [p for p in re.split(r"\s*(?:,|;|&|\band\b)\s*", alias, flags=re.I) if normalize(p)]
    return parts or [alias]


def gold_in_passage(aliases, passage_norm):
    """Whole alias present, or every part of a multi-part alias present."""
    return any(contains(passage_norm, a) or all(contains(passage_norm, p) for p in gold_parts(a))
               for a in aliases)


def gold_word_share_in_passage(aliases, passage_norm):
    ptoks = set(passage_norm.split())
    best = 0.0
    for a in aliases:
        g = content_tokens(a)
        if g:
            best = max(best, sum(t in ptoks for t in g) / len(g))
    return best


def gold_recall(answer, aliases):
    ans = set(content_tokens(answer))
    best = 0.0
    for a in aliases:
        g = content_tokens(a) or normalize(a).split()
        if g:
            best = max(best, sum(t in ans for t in g) / len(g))
    return best


def copy_score(answer_norm, passage_norm):
    toks = answer_norm.split()
    if not toks:
        return 0.0
    n = min(4, len(toks))
    pt = passage_norm.split()
    pgrams = {tuple(pt[i:i + n]) for i in range(len(pt) - n + 1)}
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return sum(g in pgrams for g in grams) / len(grams)


def jaccard(a, b):
    sa, sb = set(content_tokens(a)), set(content_tokens(b))
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def passage_answer_snippet(raw, aliases, width=120):
    low = raw.lower()
    for a in aliases:
        for target in [a] + gold_parts(a):
            t = target.lower().strip()
            i = low.find(t) if t else -1
            if i >= 0:
                return raw[max(0, i - width): i + len(t) + width].replace("\n", " ").strip()
    return ""


# ------------------------------------------------------------------ classifier
def classify(original, full, answer, aliases, raw_passage, noctx):
    """Returns dict with failure_type, confidence, evidence fields, rationale."""
    pn, an = normalize(raw_passage), normalize(answer)
    sig = {
        "gold_in_passage": gold_in_passage(aliases, pn),
        "gold_word_share_in_passage": round(gold_word_share_in_passage(aliases, pn), 3),
        "gold_recall": round(gold_recall(answer, aliases), 3),
        "copy_score": round(copy_score(an, pn), 3),
        "noctx_similarity": round(jaccard(answer, noctx), 3) if noctx is not None else None,
        "noctx_correct": (any(contains(normalize(noctx), a) for a in aliases) if noctx is not None else None),
        "short_gold": bool(aliases) and min(len(normalize(a)) for a in aliases) <= SHORT_GOLD_CHARS,
    }
    evidence = "yes" if sig["gold_in_passage"] else (
        "unclear" if sig["gold_word_share_in_passage"] >= PARAPHRASE_T else "no")
    correct = "yes" if full == "faithful" else ("partial" if sig["gold_recall"] >= PARTIAL_T else "no")
    supported = "yes" if sig["copy_score"] >= COPY_T else ("partial" if sig["copy_score"] > 0 else "no")

    def out(t, conf, why):
        return {"failure_type": t, "confidence": round(conf, 2), "evidence_support": evidence,
                "answer_correct": correct, "answer_supported_by_passage": supported,
                "rationale": why, "signals": sig}

    if not an:
        return out("F6", 0.3, "Empty model answer.")
    if original == "override" and full == "faithful":
        conf = 0.6 if sig["short_gold"] else 0.9
        why = "A gold alias appears in the answer; strict scoring marked it wrong."
        if sig["short_gold"]:
            why += " Gold is very short -- the match may be accidental; check by hand."
        return out("F5", conf, why)
    if not sig["gold_in_passage"]:
        if evidence == "unclear":
            return out("F6", 0.4, f"Exact gold not in passage, but {sig['gold_word_share_in_passage']:.0%} "
                                  "of its words are -- the passage may state it in another form.")
        return out("F3", 0.85, "No gold alias (or all parts of one) appears in the retrieved passage.")
    if sig["gold_recall"] >= PARTIAL_T:
        return out("F4", 0.7, f"Answer contains {sig['gold_recall']:.0%} of the gold answer's words, not all.")
    if sig["copy_score"] >= COPY_T:
        return out("F2", 0.55 + 0.35 * sig["copy_score"],
                   f"{sig['copy_score']:.0%} of the answer's word sequences are copied from the passage, "
                   "but it is not the gold fact.")
    if noctx is None:
        return out("F6", 0.3, "Passage has the answer and the model's answer is not copied from it, "
                              "but there is no no-passage answer to show it came from memory.")
    if sig["noctx_similarity"] >= NOCTX_T and not sig["noctx_correct"]:
        return out("F1", 0.55 + 0.4 * sig["noctx_similarity"],
                   f"Passage has the answer; the model's answer is not from the passage and matches its "
                   f"no-passage answer (overlap {sig['noctx_similarity']:.2f}).")
    if sig["noctx_correct"]:
        return out("F6", 0.35, "Model answers correctly WITHOUT the passage but wrongly WITH it -- "
                               "the passage made it worse; not parametric override.")
    return out("F6", 0.35, f"Answer neither copied from the passage nor matching the no-passage answer "
                           f"(overlap {sig['noctx_similarity']:.2f}).")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/final/Phase_02/analysis_dataset_relabeled.jsonl")
    ap.add_argument("--retrieval", default="data/final/Phase_02/retrieval_results.jsonl")
    ap.add_argument("--noctx", default="data/final/Phase_05/no_passage_answers.jsonl")
    ap.add_argument("--outdir", default="data/final/Phase_05")
    args = ap.parse_args()

    records = load_jsonl(args.dataset)
    passages = {r["query_id"]: r.get("retrieved_passages") or [] for r in load_jsonl(args.retrieval)}
    noctx = {}
    if Path(args.noctx).exists():
        for x in load_jsonl(args.noctx):
            noctx[(x["query_id"], x["model"])] = x["noctx_answer"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows, summary = [], {"type_names": TYPE_NAMES}
    for model in MODELS:
        have_noctx = any(k[1] == model for k in noctx)
        print(f"\n{'=' * 78}\n{model.upper()} -- apparent failures by type\n{'=' * 78}")
        if not have_noctx:
            print("!! No no-passage answers for this model. F1 cannot be assigned (critical rule);")
            print("!! run generate_no_passage_answers.py --model", model, "and re-run this script.")
        excluded = Counter()
        mrows = []
        for r in records:
            orig, full = r.get(f"{model}_label"), r.get(f"{model}_label_full")
            if orig != "override" and full != "override":
                continue
            if r.get("source_category") == "confiqa":
                excluded["confiqa"] += 1
                continue
            if r.get(f"{model}_relabel_status") != "ok":
                excluded[r.get(f"{model}_relabel_status") or "unknown"] += 1
                continue
            raw = "\n\n".join(passages.get(r["query_id"], []))
            if not raw.strip():
                excluded["no_passage_stored"] += 1
                continue
            aliases = [a for a in parse_gold_answers(r.get("gold_answer_text")) if normalize(a)]
            answer = r.get(f"{model}_answer") or ""
            nc = noctx.get((r["query_id"], model))
            c = classify(orig, full, answer, aliases, raw, nc)
            mrows.append({
                "question_id": r["query_id"], "model": model, "dataset": r.get("source_category"),
                "original_label": orig, "full_answer_label": full, **{k: v for k, v in c.items() if k != "signals"},
                "passage_answer": passage_answer_snippet(raw, aliases), "model_answer": answer,
                "noctx_answer": nc, "question": r.get("question"), "gold": r.get("gold_answer_text"),
                "pcs": r.get(f"pcs_{model}"), "signals": c["signals"],
            })
        rows.extend(mrows)

        # ---- counts by dataset ----
        by_ds = defaultdict(Counter)
        for x in mrows:
            by_ds[x["dataset"]][x["failure_type"]] += 1
            by_ds["ALL"][x["failure_type"]] += 1
        print(f"{'dataset':<16}" + "".join(f"{t:>8}" for t in TYPES) + f"{'total':>8}")
        for ds in sorted(by_ds, key=lambda s: (s == "ALL", s)):
            row = by_ds[ds]
            n = sum(row.values())
            print(f"{ds:<16}" + "".join(f"{row[t]:>8}" for t in TYPES) + f"{n:>8}")
        n_all = sum(by_ds["ALL"].values()) or 1
        print(f"{'% of ALL':<16}" + "".join(f"{100 * by_ds['ALL'][t] / n_all:>7.1f}%" for t in TYPES))
        if excluded:
            print(f"Excluded: {dict(excluded)}")

        # ---- original 'override' set: what is it made of? ----
        orig_ov = [x for x in mrows if x["original_label"] == "override"]
        comp = Counter(x["failure_type"] for x in orig_ov)
        print(f"\nOriginal 'override' label ({len(orig_ov)} cases) decomposes into: "
              + ", ".join(f"{t}={comp[t]}" for t in TYPES))
        print(f"Genuine override (F1) share of original label: "
              f"{100 * comp['F1'] / max(len(orig_ov), 1):.1f}%   <- headline number")

        # ---- PCS by type ----
        print("Mean PCS by type (near 0 = confident without passage):")
        for t in TYPES:
            v = [x["pcs"] for x in mrows if x["failure_type"] == t and isinstance(x["pcs"], (int, float))]
            if v:
                print(f"  {t} {TYPE_NAMES[t]:<36} {sum(v) / len(v):7.2f}  (n={len(v)})")

        # ---- examples ----
        examples = {}
        for t in TYPES:
            ex = sorted([x for x in mrows if x["failure_type"] == t], key=lambda x: -x["confidence"])[:3]
            examples[t] = [{k: x[k] for k in ("question_id", "question", "gold", "model_answer",
                                               "noctx_answer", "passage_answer", "rationale")} for x in ex]
        summary[model] = {
            "n_apparent_failures": len(mrows), "noctx_available": have_noctx,
            "counts": dict(by_ds["ALL"]),
            "percent": {t: round(100 * by_ds["ALL"][t] / n_all, 1) for t in TYPES},
            "by_dataset": {k: dict(v) for k, v in by_ds.items() if k != "ALL"},
            "original_override_composition": dict(comp), "excluded": dict(excluded),
            "n_ambiguous": by_ds["ALL"]["F6"], "examples": examples,
        }

    cols = ["question_id", "model", "dataset", "original_label", "full_answer_label", "failure_type",
            "confidence", "evidence_support", "answer_correct", "answer_supported_by_passage",
            "passage_answer", "model_answer", "noctx_answer", "rationale"]
    with open(outdir / "exp1_failure_types.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(outdir / "exp1_failure_types.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    with open(outdir / "exp1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {outdir / 'exp1_failure_types.csv'} ({len(rows)} rows)")
    print(f"Wrote {outdir / 'exp1_failure_types.jsonl'}\nWrote {outdir / 'exp1_summary.json'}")
    print("Next: python Tests/experiment2_annotation.py sample")


if __name__ == "__main__":
    main()