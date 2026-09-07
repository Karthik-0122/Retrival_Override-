"""
Phase 4, T20: Select the test sample for ablation.

Pulls, per model:
  - OVERRIDE cases: the actual causal question is "does ablating the
    target heads flip override -> faithful". These are the test set.
  - FAITHFUL cases: a control group. If ablation flips these to
    override/incorrect too, the ablation is just breaking generation
    indiscriminately, not doing anything specific.

Run from repo root:
    python Tests/Phase_04/select_test_sample.py

Requires:
    data/final/analysis_dataset.jsonl

Writes:
    data/final/Phase_04/ablation_test_sample.json
"""

import json
import random
from pathlib import Path

DATASET_PATH = "data/final/Phase_02/analysis_dataset.jsonl"
OUT_PATH = "data/final/Phase_04/ablation_test_sample.json"

N_PER_GROUP = 40  # override cases and faithful-control cases, per model
SEED = 42


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def select_for_model(records, model):
    label_key = f"{model}_label"
    # Exclude ConFiQA: its "override" label means the model correctly
    # RESISTED a false/counterfactual passage (the good outcome there),
    # opposite of what "override" means for NQ/TriviaQA/PopQA (wrongly
    # ignoring a CORRECT passage). Scoring correctness the same way for
    # both would be backwards for ConFiQA rows. Kept separate throughout
    # this project for exactly this reason -- same here.
    non_confiqa = [r for r in records if r.get("source_category") != "confiqa"]
    override = [r for r in non_confiqa if r.get(label_key) == "override"]
    faithful = [r for r in non_confiqa if r.get(label_key) == "faithful"]

    rng = random.Random(SEED)
    rng.shuffle(override)
    rng.shuffle(faithful)

    n_override = min(N_PER_GROUP, len(override))
    n_faithful = min(N_PER_GROUP, len(faithful))

    if n_override < N_PER_GROUP or n_faithful < N_PER_GROUP:
        print(f"  WARNING ({model}): requested {N_PER_GROUP} per group, "
              f"only got {n_override} override / {n_faithful} faithful available")

    return {
        "override_cases": [
            {"query_id": r["query_id"], "question": r["question"]}
            for r in override[:n_override]
        ],
        "faithful_control_cases": [
            {"query_id": r["query_id"], "question": r["question"]}
            for r in faithful[:n_faithful]
        ],
    }


def main():
    records = load_jsonl(DATASET_PATH)
    print(f"Loaded {len(records)} total records")

    output = {}
    for model in ["gemma", "llama"]:
        sample = select_for_model(records, model)
        output[model] = sample
        print(f"{model}: {len(sample['override_cases'])} override cases, "
              f"{len(sample['faithful_control_cases'])} faithful control cases")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
