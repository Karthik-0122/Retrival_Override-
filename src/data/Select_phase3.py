"""
Stratified sample for Phase 3's FIRST ROI extraction run.

Rather than extracting ROI for all 1500 queries blind, pull a deliberate
sample covering the specific comparison Phase 3 exists to test: does the
attention mechanism differ between PopQA (where override correlates with
HIGHER PCS) and natural_rag_other (where override correlates with LOWER
PCS)? A stratified sample lets you answer this with a fraction of the
GPU time a full run would cost, before deciding whether to scale up.

Output: data/final/phase3_stratified_sample.jsonl
"""

import json
import random

ANALYSIS_FILE = "data/final/analysis_dataset.jsonl"
OUTPUT_FILE = "data/final/phase3_stratified_sample.jsonl"

# Per-cell sample size. 30 x 4 cells x 2 models-worth-of-labels = manageable
# on rented GPU for a first pass; scale up later if the signal looks real.
# Set very high so min(PER_CELL_N, available) always takes the FULL pool
# per stratum -- the 120-query test showed a strong effect (28/42 and
# 10/32 layers significant, far beyond chance), so this scales to the
# largest available confirmation set rather than another fixed subsample.
# Expected pool sizes (from analysis_dataset.jsonl, gemma_label):
#   natural_rag_other/faithful: 298   natural_rag_other/override: 141
#   popqa/faithful: 66                popqa/override: 42
# Total ~547 queries -- still cheap: 120 queries took ~80s/model total,
# so ~547 should land well under 10 minutes total across both models.
PER_CELL_N = 100000
RANDOM_SEED = 42

# The four cells that directly test the reversal question. Confiqa
# excluded from this first stratified pass -- it's a different
# phenomenon (counterfactual resistance, not natural override) and
# would dilute a focused first test; add it back for a broader run later.
STRATA = [
    ("natural_rag_other", "faithful"),
    ("natural_rag_other", "override"),
    ("popqa", "faithful"),
    ("popqa", "override"),
]


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    random.seed(RANDOM_SEED)
    records = load_jsonl(ANALYSIS_FILE)
    print(f"Loaded {len(records)} rows from {ANALYSIS_FILE}")

    # Use gemma_label as the primary stratification label (arbitrary choice --
    # could also require agreement between gemma_label and llama_label for a
    # cleaner sample; here we just use gemma's label, both models still get
    # ROI extracted for whichever queries are selected).
    sample = []
    for cat, label in STRATA:
        candidates = [r for r in records
                      if r.get("source_category") == cat and r.get("gemma_label") == label]
        chosen = random.sample(candidates, min(PER_CELL_N, len(candidates)))
        for r in chosen:
            r["_phase3_stratum"] = f"{cat}/{label}"
        sample.extend(chosen)
        print(f"  {cat}/{label}: {len(chosen)} selected (full available pool)")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {OUTPUT_FILE} ({len(sample)} total queries)")
    print("Run Phase 3's ROI extraction against THIS file first, not the full 1500 --")
    print("confirms the extraction works and shows an early signal (or absence of one)")
    print("before spending GPU time on the full set.")


if __name__ == "__main__":
    main()