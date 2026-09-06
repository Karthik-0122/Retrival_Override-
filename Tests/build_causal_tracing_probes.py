"""
Builds the probe set for causal tracing (fact-storage localization).
Filters existing queries down to short, single-entity-answer facts —
the format causal tracing needs — and finds the subject span in each
prompt using the same char_to_token approach as the ROI extraction.

Run from repo root:
    python tests/build_causal_tracing_probes.py

Output: data/final/causal_tracing_probes.jsonl
"""

import json
import re
from transformers import AutoTokenizer

DATASET_PATH = "data/final/dataset_1500_with_titles_1.jsonl"
SOURCE_FIELD = "source"
SOURCE_VALUE = "popqa"
OUT_PATH = "data/final/causal_tracing_probes.jsonl"
N_PROBES = 100

# Use one tokenizer just to find spans consistently; re-tokenized per-model at extraction time
TOKENIZER_FOR_SPAN_CHECK = "/root/models/gemma-2-9b"  # adjust if needed


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def looks_like_single_entity_answer(answer):
    """Heuristic: short answer, no sentence punctuation, likely a name/place/date."""
    if not answer:
        return False
    words = answer.strip().split()
    if len(words) > 4:
        return False
    if any(c in answer for c in ".!?;"):
        return False
    return True


def find_subject_span(question, subject, tokenizer):
    """Find subject's char span in the question, then convert to token span."""
    idx = question.find(subject)
    if idx == -1:
        return None
    char_start, char_end = idx, idx + len(subject)
    encoding = tokenizer(question, return_tensors="pt", return_offsets_mapping=True)
    token_start = encoding.char_to_token(char_start)
    token_end_char = encoding.char_to_token(char_end - 1)
    token_end = (token_end_char + 1) if token_end_char is not None else None
    if token_start is None or token_end is None:
        return None
    return token_start, token_end


def parse_answer(gold):
    """gold_answers can be a dict, a real list, or (seen in this dataset for
    popqa) a JSON-encoded string that LOOKS like a list, e.g. '["Granada"]'
    -- must be json.loads'd, not used as-is, or you get the literal
    bracket-and-quote text as your 'answer'."""
    if isinstance(gold, dict):
        return gold.get("value") or (gold.get("aliases") or [""])[0]
    if isinstance(gold, list):
        return gold[0] if gold else ""
    if isinstance(gold, str):
        s = gold.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                return parsed[0] if parsed else ""
            except (json.JSONDecodeError, IndexError):
                return gold
        return gold
    return ""


# PopQA questions are templated -- parsing the template directly is far more
# reliable than generic capitalized-word matching, which breaks on
# sentence-initial question words ("Who...") and on titles containing
# lowercase connector words ("Crossing the Bridge").
SUBJECT_TEMPLATES = [
    r"^(?:Who|What) (?:is|was) the (?:author|director|screenwriter|capital) (?:of|for) (.+?)\??$",
    r"^In what city was (.+?) born\??$",
    r"^What sport does (.+?) play\??$",
    r"^What is (.+?)(?:'s| is)? .+\??$",
]


def guess_subject(question):
    """
    Try known PopQA question templates first (reliable). Falls back to the
    old capitalized-span heuristic only if no template matches -- fallback
    rows are flagged in the output so you know to check them by hand.
    """
    q = question.strip()
    for pattern in SUBJECT_TEMPLATES:
        m = re.match(pattern, q, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().rstrip("?")
            if candidate:
                return candidate, "template"

    # fallback: capitalized-span heuristic, excluding common sentence-initial
    # question words so they don't get mistaken for the subject
    QUESTION_WORDS = {"who", "what", "when", "where", "which", "how", "why", "in", "is", "was", "did", "does"}
    candidates = re.findall(r"([A-Z][a-zA-Z0-9&'.\-]*(?:\s+[A-Z][a-zA-Z0-9&'.\-]*)*)", q)
    candidates = [c for c in candidates if len(c.split()) <= 5 and c.lower() not in QUESTION_WORDS]
    if not candidates:
        return None, "fallback_failed"
    return max(candidates, key=len), "fallback"


def main():
    records = load_jsonl(DATASET_PATH)
    popqa_records = [r for r in records if r.get(SOURCE_FIELD) == SOURCE_VALUE]
    print(f"Loaded {len(popqa_records)} popqa records to filter from")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_FOR_SPAN_CHECK)

    probes = []
    for r in popqa_records:
        question = r["question"]
        gold = r.get("gold_answers")
        answer = parse_answer(gold)
        if not answer or not looks_like_single_entity_answer(answer):
            continue

        subject, method = guess_subject(question)
        if subject is None:
            continue

        span = find_subject_span(question, subject, tokenizer)
        if span is None:
            continue
        token_start, token_end = span

        probes.append({
            "query_id": r["query_id"],
            "question": question,
            "subject": subject,
            "subject_extraction_method": method,
            "subject_token_start": token_start,
            "subject_token_end": token_end,
            "answer_text": answer,
        })

        if len(probes) >= N_PROBES:
            break

    with open(OUT_PATH, "w") as f:
        for p in probes:
            f.write(json.dumps(p) + "\n")

    n_template = sum(1 for p in probes if p["subject_extraction_method"] == "template")
    n_fallback = sum(1 for p in probes if p["subject_extraction_method"] == "fallback")
    print(f"Wrote {len(probes)} probes to {OUT_PATH}")
    print(f"  {n_template} via reliable template match, {n_fallback} via fallback heuristic")
    print(f"  -> inspect the {n_fallback} fallback rows especially closely")
    print("\nIMPORTANT: inspect this file by hand before running extraction.")
    print("The subject-guessing heuristic is rough (capitalized-span matching,")
    print("not real NER) and will misfire on some questions -- e.g. questions")
    print("starting with a capitalized question word, or multi-entity questions.")
    print("Manually delete or fix any obviously wrong rows before Step 3.")


if __name__ == "__main__":
    main()
