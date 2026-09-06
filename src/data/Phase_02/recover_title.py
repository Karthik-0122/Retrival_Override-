"""
Recovers gold Wikipedia titles for NQ/TriviaQA/PopQA rows in the merged
dataset. ConFiQA is intentionally skipped — it already has its own
counterfactual_context passage per row and doesn't need a Wikipedia lookup.

Matching notes per source:
  - nq_open has NO title/document field at all (it's the simplified
    open-domain version). Titles are recovered by matching question text
    against the full `natural_questions` validation split instead.
  - trivia_qa was loaded with config "rc.nocontext", which strips the
    entity_pages field. Titles are recovered by matching question text
    against the "rc" config instead, which retains entity_pages.
  - PopQA (akariasai/PopQA) already has s_wiki_title + s_pop directly —
    no cross-dataset matching needed, and this also fills popularity_tier.

pip install datasets
"""

import json
from pathlib import Path

from datasets import load_dataset

MERGED_FILE = "data/final/merged_circuit_analysis_data.jsonl"
OUTPUT_FILE = "data/final/dataset_1500_with_titles_1.jsonl"
NQ_LOOKUP_CACHE = Path("data/artifacts/nq_title_lookup.json")


def _normalize_question(question: str) -> str:
    return question.strip().lower()


def _extract_nq_question_and_title(example):
    question = example["question"]
    if isinstance(question, dict):
        question = question.get("text", "")
    title = example.get("document", {}).get("title")
    return _normalize_question(question), title

def build_nq_lookup(needed_questions):
    needed = {_normalize_question(q) for q in needed_questions}
    lookup = {}

    if NQ_LOOKUP_CACHE.exists():
        with open(NQ_LOOKUP_CACHE, "r", encoding="utf-8") as f:
            cached = json.load(f)

        for q_key, title in cached.items():
            if q_key in needed and title:
                lookup[q_key] = title

        needed -= set(lookup.keys())

        if not needed:
            print(f"Loaded {len(lookup)} NQ titles from cache.")
            return lookup

        print(
            f"Loaded {len(lookup)} titles from cache. "
            f"{len(needed)} still need to be recovered."
        )

    print(
        f"\nLoading Natural Questions validation split "
        f"(streaming, {len(needed)} questions)..."
    )

    ds = load_dataset(
        "google-research-datasets/natural_questions",
        split="validation",
        streaming=True,
    )

    print("Dataset loaded successfully.")
    print("Beginning scan...\n")

    scanned = 0
    total_needed = len(needed)

    for example in ds:
        scanned += 1

        if scanned % 500 == 0:
            print(
                f"[Progress] "
                f"Scanned={scanned} | "
                f"Matched={len(lookup)}/{total_needed} | "
                f"Remaining={len(needed)}"
            )

        q_key, title = _extract_nq_question_and_title(example)

        if q_key in needed:
            lookup[q_key] = title
            needed.remove(q_key)

            print(
                f"Found match {len(lookup)}/{total_needed}: "
                f"{title}"
            )

            # Removed the early 'break' here so it continues scanning 
            # until the entire split is checked or all items are found.
            if not needed:
                print("\nRecovered every required title.")
                break

    print(f"\nFinished scanning {scanned} examples.")

    if needed:
        print(f"WARNING: {len(needed)} questions could not be matched.")

    NQ_LOOKUP_CACHE.parent.mkdir(parents=True, exist_ok=True)

    merged_cache = {}
    if NQ_LOOKUP_CACHE.exists():
        with open(NQ_LOOKUP_CACHE, "r", encoding="utf-8") as f:
            merged_cache = json.load(f)

    merged_cache.update(lookup)

    with open(NQ_LOOKUP_CACHE, "w", encoding="utf-8") as f:
        json.dump(merged_cache, f, ensure_ascii=False, indent=2)

    print(f"Recovered {len(lookup)} titles.")

    return lookup

def build_triviaqa_lookup():
    print("Loading trivia_qa (rc config, for title recovery)...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation")
    lookup = {}
    for ex in ds:
        q = _normalize_question(ex["question"])
        titles = ex["entity_pages"]["title"]
        lookup[q] = titles[0] if titles else None
    print(f"  {len(lookup)} TriviaQA questions indexed")
    return lookup


def build_popqa_lookup():
    print("Loading PopQA (title + popularity)...")
    ds = load_dataset("akariasai/PopQA", split="test")
    lookup = {}
    for ex in ds:
        q = _normalize_question(ex["question"])
        lookup[q] = {"title": ex.get("s_wiki_title"), "popularity": ex.get("s_pop")}
    print(f"  {len(lookup)} PopQA questions indexed")
    return lookup


def assign_popularity_tier(popularity, thresholds=(1000, 100000)):
    """Placeholder thresholds — check PopQA's actual s_pop percentile
    distribution before trusting these cutoffs."""
    if popularity is None:
        return None
    if popularity < thresholds[0]:
        return "low"
    elif popularity < thresholds[1]:
        return "medium"
    return "high"


def main():
    print(f"Loading {MERGED_FILE}...")
    records = []
    with open(MERGED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    nq_questions = [r["question"] for r in records if r["source"] == "nq"]
    nq_lookup = build_nq_lookup(nq_questions)
    triviaqa_lookup = build_triviaqa_lookup()
    popqa_lookup = build_popqa_lookup()

    n_matched, n_unmatched, n_skipped_confiqa = 0, 0, 0

    for r in records:
        source = r["source"]
        q_key = _normalize_question(r["question"])

        if source == "confiqa":
            n_skipped_confiqa += 1
            r["gold_title"] = None  # not applicable — uses counterfactual_context instead
            continue

        if source == "nq":
            title = nq_lookup.get(q_key)
        elif source == "triviaqa":
            title = triviaqa_lookup.get(q_key)
        elif source == "popqa":
            hit = popqa_lookup.get(q_key)
            title = hit["title"] if hit else None
            r["popularity_tier"] = assign_popularity_tier(hit["popularity"]) if hit else None
        else:
            print(f"  [warn] unrecognized source '{source}' for query_id={r['query_id']}")
            title = None

        r["gold_title"] = title
        if title:
            n_matched += 1
        else:
            n_unmatched += 1

    print(f"\nMatched: {n_matched}")
    print(f"Unmatched (question text didn't match raw source): {n_unmatched}")
    print(f"Skipped (ConFiQA, not applicable): {n_skipped_confiqa}")
    if n_unmatched:
        print("  Unmatched rows are usually phrasing/whitespace mismatches vs the raw "
              "dataset — spot check a handful before proceeding to corpus build.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
