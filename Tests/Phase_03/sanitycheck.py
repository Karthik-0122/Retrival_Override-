"""
Sanity check for T5's generation_results.jsonl before moving to T6.

Checks:
  1. How many rows have None for answer / entropy (the hook-not-firing
     failure mode flagged before running T5)
  2. Entropy value range/distribution -- should be positive, non-trivial
     numbers, not all identical (which would suggest the hook fired but
     wasn't actually varying, e.g. always hitting the same fallback path)
  3. A handful of example rows: question, gold answer, both models'
     generated answers, both entropy values -- eyeball for sanity
"""

import json

RECORDS_FILE = "data/final/dataset_1500_with_titles_1.jsonl"
GENERATION_FILE = "data/final/generation_results.jsonl"


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def get_gold_answer_text(record):
    ga = record.get("gold_answers")
    if isinstance(ga, dict):
        return ga.get("value", "")
    if isinstance(ga, list):
        return ga[0] if ga else ""
    return str(ga) if ga else ""


def main():
    records = {r["query_id"]: r for r in load_jsonl(RECORDS_FILE)}
    generations = load_jsonl(GENERATION_FILE)
    print(f"Loaded {len(generations)} generation results\n")

    for model in ["gemma", "llama"]:
        answer_col = f"{model}_answer"
        entropy_col = f"{model}_attn_entropy_on_passage"

        n_none_answer = sum(1 for g in generations if g.get(answer_col) is None)
        n_empty_answer = sum(1 for g in generations if g.get(answer_col) == "")
        n_none_entropy = sum(1 for g in generations if g.get(entropy_col) is None)

        entropies = [g[entropy_col] for g in generations if g.get(entropy_col) is not None]

        print(f"=== {model.upper()} ===")
        print(f"  None answers: {n_none_answer}/{len(generations)}")
        print(f"  Empty-string answers: {n_empty_answer}/{len(generations)}")
        print(f"  None entropy (hook may not have fired): {n_none_entropy}/{len(generations)}")

        if entropies:
            entropies_sorted = sorted(entropies)
            n = len(entropies_sorted)
            print(f"  Entropy range: min={entropies_sorted[0]:.4f}  "
                  f"median={entropies_sorted[n//2]:.4f}  max={entropies_sorted[-1]:.4f}")
            unique_vals = len(set(round(e, 4) for e in entropies))
            print(f"  Unique entropy values (rounded to 4dp): {unique_vals} "
                  f"(low uniqueness relative to n would suggest the hook isn't "
                  f"actually varying per query)")
        else:
            print("  WARNING: no entropy values at all -- hook likely never fired.")
        print()

    print("--- 5 example rows ---")
    for g in generations[:5]:
        r = records.get(g["query_id"], {})
        print(f"\nQ: {r.get('question')}")
        print(f"Gold: {get_gold_answer_text(r)}")
        print(f"Gemma answer: {g.get('gemma_answer')!r}")
        print(f"Gemma entropy: {g.get('gemma_attn_entropy_on_passage')}")
        print(f"Llama answer: {g.get('llama_answer')!r}")
        print(f"Llama entropy: {g.get('llama_attn_entropy_on_passage')}")


if __name__ == "__main__":
    main()