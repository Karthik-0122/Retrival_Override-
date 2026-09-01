import json
from collections import Counter

MERGED_FILE = "data/final/merged_circuit_analysis_data.jsonl"


def main():
    records = []
    with open(MERGED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Total records: {len(records)}")

    source_counts = Counter(r["source"] for r in records)
    print("\nPer-source counts (expect nq=400, triviaqa=300, popqa=400, confiqa=400):")
    for src, count in source_counts.items():
        print(f"  {src}: {count}")

    confiqa_rows = [r for r in records if r["source"] == "confiqa"]
    has_context = sum(1 for r in confiqa_rows if r.get("counterfactual_context"))
    has_orig_answer = sum(1 for r in confiqa_rows if r.get("answer_original"))
    print(f"\nConFiQA rows with counterfactual_context populated: {has_context}/{len(confiqa_rows)}")
    print(f"ConFiQA rows with answer_original populated: {has_orig_answer}/{len(confiqa_rows)}")
    if has_context < len(confiqa_rows):
        print("  WARNING: some/all ConFiQA rows missing counterfactual_context — "
              "check build_datasets.py's extra_column_candidates matched the real column name.")

    non_confiqa = [r for r in records if r["source"] != "confiqa"]
    leaked = sum(1 for r in non_confiqa if r.get("counterfactual_context") or r.get("answer_original"))
    print(f"\nNon-ConFiQA rows with unexpected counterfactual fields set: {leaked} (should be 0)")

    empty_q = sum(1 for r in records if not r.get("question"))
    empty_a = sum(1 for r in records if not r.get("gold_answers"))
    print(f"\nRows with empty question: {empty_q}")
    print(f"Rows with empty gold_answers: {empty_a}")

    ids = [r["query_id"] for r in records]
    dupes = len(ids) - len(set(ids))
    print(f"\nDuplicate query_ids: {dupes}")


if __name__ == "__main__":
    main()