"""
Phase 4, T25: Statistical validation of the ablation effect.

Joins the REAL ablation results (ablation_results_escalated.jsonl) with
the RANDOM CONTROL results (ablation_results_control.jsonl) on
query_id -- both tested the SAME queries, just with different heads
ablated. This makes it a PAIRED comparison: for each query, did it flip
under the real heads, the random heads, both, or neither. McNemar's
test is the correct test for exactly this paired-binary-outcome setup
(not an independent-samples test, since it's the same queries twice).

Run from repo root:
    python Tests/statistical_validation.py

Requires:
    data/final/Phase_04/ablation_results_escalated.jsonl
    data/final/Phase_04/ablation_results_control.jsonl

Requires: pip install statsmodels (already installed earlier this project)
"""

import json
from statsmodels.stats.contingency_tables import mcnemar

REAL_PATH = "data/final/Phase_04/ablation_results_escalated.jsonl"
CONTROL_PATH = "data/final/Phase_04/ablation_results_control.jsonl"


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def run_mcnemar_for_group(real_rows, control_rows, model, group, outcome_field):
    """outcome_field: 'flipped_to_correct' (for override_cases) or
    'flipped_to_incorrect' (for faithful_control_cases)."""
    real_by_id = {r["query_id"]: r for r in real_rows if r["model"] == model and r["group"] == group}
    control_by_id = {r["query_id"]: r for r in control_rows if r["model"] == model and r["group"] == group}

    common_ids = set(real_by_id) & set(control_by_id)
    if not common_ids:
        print(f"  {model} {group}: no overlapping query_ids between real and control -- can't run test")
        return None

    # 2x2 contingency: rows = real outcome, cols = control outcome
    both = neither = real_only = control_only = 0
    for qid in common_ids:
        real_flip = real_by_id[qid][outcome_field]
        control_flip = control_by_id[qid][outcome_field]
        if real_flip and control_flip:
            both += 1
        elif real_flip and not control_flip:
            real_only += 1
        elif not real_flip and control_flip:
            control_only += 1
        else:
            neither += 1

    table = [[both, real_only], [control_only, neither]]
    n = len(common_ids)

    print(f"\n  {model} {group} (n={n} paired queries, outcome={outcome_field}):")
    print(f"    both flipped: {both}   real-only: {real_only}   control-only: {control_only}   neither: {neither}")

    if real_only + control_only == 0:
        print(f"    No discordant pairs at all -- real and control are identical on this outcome. p=1.0 (no difference).")
        return 1.0

    result = mcnemar(table, exact=True)  # exact binomial test, appropriate for small discordant counts
    print(f"    McNemar's exact test: p = {result.pvalue:.4f}")
    if result.pvalue < 0.05:
        direction = "REAL > CONTROL" if real_only > control_only else "CONTROL > REAL"
        print(f"    -> SIGNIFICANT at p<0.05 ({direction})")
    else:
        print(f"    -> NOT significant at p<0.05 -- real and control are statistically indistinguishable on this outcome")
    return result.pvalue


def main():
    real_rows = load_jsonl(REAL_PATH)
    control_rows = load_jsonl(CONTROL_PATH)

    print("=" * 70)
    print("McNemar's test: REAL (Phase-3-identified heads) vs RANDOM CONTROL")
    print("=" * 70)

    for model in ["gemma", "llama"]:
        print(f"\n{'='*70}\n{model.upper()}\n{'='*70}")
        run_mcnemar_for_group(real_rows, control_rows, model, "override_cases", "flipped_to_correct")
        run_mcnemar_for_group(real_rows, control_rows, model, "faithful_control_cases", "flipped_to_incorrect")


if __name__ == "__main__":
    main()
    