"""
Sanity check on PCS (Parametric Confidence Score) from Phase 2.

Pulls a spread of real examples -- some with very high |PCS|, some with
very low |PCS| -- so they can be read by eye and checked against
intuition: does a well-known, confidently-known fact actually show high
|PCS|, and an obscure/uncertain one show low |PCS|? This is a cheap,
direct way to catch a sign error or labeling mixup, as opposed to
trusting an unusual result (Phase 2's "backwards from hypothesis"
finding) without ever having looked at real examples.

No GPU needed -- pure data inspection.

Run from repo root:
    python Tests/pcs_sanity_check.py

Requires:
    data/final/Phase_02/analysis_dataset.jsonl
"""

import json

DATASET_PATH = "data/final/Phase_02/analysis_dataset.jsonl"
N_EXAMPLES_PER_BUCKET = 3


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def print_example(r, model_key):
    pcs_key = f"pcs_{model_key}"
    label_key = f"{model_key}_label"
    answer_key = f"{model_key}_answer"
    print(f"\n  {r['query_id']}  [{r.get('source_category')}]")
    print(f"    question: {r.get('question')}")
    print(f"    gold answer: {r.get('gold_answer_text')}")
    print(f"    model's with-context answer: {r.get(answer_key)}")
    print(f"    label: {r.get(label_key)}")
    print(f"    PCS ({model_key}): {r.get(pcs_key)}")


def main():
    records = load_jsonl(DATASET_PATH)
    records = [r for r in records if r.get("source_category") != "confiqa"]

    for model_key in ["gemma", "llama"]:
        pcs_key = f"pcs_{model_key}"
        scored = [(r.get(pcs_key), r) for r in records if r.get(pcs_key) is not None]

        print("\n" + "=" * 70)
        print(f"{model_key.upper()} -- PCS SANITY CHECK")
        print("=" * 70)

        scored.sort(key=lambda x: -abs(x[0]))
        print(f"\n--- {N_EXAMPLES_PER_BUCKET} HIGHEST |PCS| (should be facts the model")
        print("    seems VERY confident/certain about from memory alone) ---")
        for pcs, r in scored[:N_EXAMPLES_PER_BUCKET]:
            print_example(r, model_key)

        scored.sort(key=lambda x: abs(x[0]))
        print(f"\n--- {N_EXAMPLES_PER_BUCKET} LOWEST |PCS| (should be facts the model")
        print("    seems UNCERTAIN/not confident about from memory alone) ---")
        for pcs, r in scored[:N_EXAMPLES_PER_BUCKET]:
            print_example(r, model_key)

    print("\n" + "=" * 70)
    print("WHAT TO CHECK BY EYE")
    print("=" * 70)
    print("For the HIGH |PCS| examples: does the question look like something")
    print("the model plausibly has confidently memorized (well-known facts,")
    print("famous entities)? If instead these look obscure/random, PCS may be")
    print("inverted or miscomputed.")
    print()
    print("For the LOW |PCS| examples: does the question look genuinely obscure")
    print("or uncertain? If these look like well-known facts instead, that")
    print("would also suggest something is backwards.")


if __name__ == "__main__":
    main()