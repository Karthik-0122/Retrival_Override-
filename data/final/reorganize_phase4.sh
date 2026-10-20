#!/usr/bin/env bash
# Reorganize Phase 4 files in Retrieval_Override repo
# Run from the repo root: ~/PROJECTS/Retrival_Override/
#
# DRY RUN by default — prints what it would do without moving anything.
# Review the "NEEDS YOUR CALL" section at the bottom before running for real.
#
# Usage:
#   ./reorganize_phase4.sh          # dry run (safe, just prints)
#   ./reorganize_phase4.sh --apply  # actually moves files with git mv

set -euo pipefail

APPLY=false
if [[ "${1:-}" == "--apply" ]]; then
    APPLY=true
fi

move() {
    local src="$1"
    local dst="$2"
    if [[ ! -e "$src" ]]; then
        echo "  SKIP (not found): $src"
        return
    fi
    if $APPLY; then
        mkdir -p "$(dirname "$dst")"
        git mv "$src" "$dst"
        echo "  MOVED: $src -> $dst"
    else
        echo "  WOULD MOVE: $src -> $dst"
    fi
}

echo "=== Creating Phase_04 directories ==="
if $APPLY; then
    mkdir -p src/data/Phase_04/utils
    mkdir -p Tests/Phase_04
fi

echo ""
echo "=== Moving Phase 4 pipeline scripts -> src/data/Phase_04/ ==="
move "src/data/extract_causal_tracing.py"       "src/data/Phase_04/extract_causal_tracing.py"
move "Tests/build_causal_tracing_probes.py"     "src/data/Phase_04/build_causal_tracing_probes.py"
move "Tests/compare_storage_vs_override.py"     "src/data/Phase_04/compare_storage_vs_override.py"
move "Tests/compute_faithful_means.py"          "src/data/Phase_04/compute_faithful_means.py"
move "Tests/direct_logit_attribution.py"        "src/data/Phase_04/direct_logit_attribution.py"
move "Tests/find_early_layer_candidates.py"     "src/data/Phase_04/find_early_layer_candidates.py"
move "Tests/joint_intervention.py"              "src/data/Phase_04/joint_intervention.py"
move "Tests/mlp_ablation.py"                    "src/data/Phase_04/mlp_ablation.py"
move "Tests/mlp_stage1_screen.py"               "src/data/Phase_04/mlp_stage1_screen.py"
move "Tests/run_ablation.py"                    "src/data/Phase_04/run_ablation.py"
move "Tests/run_ablation_control.py"            "src/data/Phase_04/run_ablation_control.py"
move "Tests/select_ablation_targets.py"         "src/data/Phase_04/select_ablation_targets.py"
move "Tests/phase4_final_summary.py"            "src/data/Phase_04/phase4_final_summary.py"

echo ""
echo "=== Moving Phase 4 utilities -> src/data/Phase_04/utils/ ==="
move "Tests/ablation_utils.py"                  "src/data/Phase_04/utils/ablation_utils.py"
move "Tests/mlp_ablation_utils.py"              "src/data/Phase_04/utils/mlp_ablation_utils.py"

echo ""
echo "=== Moving Phase 4 test/validation scripts -> Tests/Phase_04/ ==="
move "Tests/pcs_sanity_check.py"                "Tests/Phase_04/pcs_sanity_check.py"
move "Tests/statistical_validation.py"          "Tests/Phase_04/statistical_validation.py"
move "Tests/smoke_test_batching.py"             "Tests/Phase_04/smoke_test_batching.py"
move "Tests/select_test_sample.py"              "Tests/Phase_04/select_test_sample.py"
move "Tests/cross_ref.py"                       "Tests/Phase_04/cross_ref.py"
move "Tests/diagonize_ablation.py"              "Tests/Phase_04/diagonize_ablation.py"

echo ""
echo "=== NOT MOVED — needs your decision, edit this script once you've decided ==="
echo "  Tests/ablation_util           (no .py — dupe of ablation_utils.py?)"
echo "  Tests/Mlp_ablation_utils.py   (case-dupe of mlp_ablation_utils.py?)"
echo "  Tests/computer_faithful_means.py (typo-dupe of compute_faithful_means.py?)"
echo "  Tests/practice_ablation_gpt2.py  (scratch file — keep, discard, or move to a scratch/ dir?)"
echo "  =0.46.1                       (looks like stray output of a botched pip install — delete?)"
echo "  apply_fdr.py                  (repo root — which phase?)"
echo "  inspect2.py                   (repo root — which phase?)"

echo ""
if $APPLY; then
    echo "Done. Run 'git status' to review, then commit."
else
    echo "Dry run complete. Re-run with --apply to actually move files."
fi