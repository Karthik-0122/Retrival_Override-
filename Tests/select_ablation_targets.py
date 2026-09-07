"""
Phase 4, T17: Select ablation targets.

Pulls the strongest (layer, head) candidates from the per-head analysis,
restricted to layers that survived the length-controlled + FDR test
(the final validated layer set from Phase 3) -- not just raw per-head
FDR significance, which includes layers that later dropped out (e.g.
Gemma's mid-network 13-22 block).

Run from repo root:
    python Tests/select_ablation_targets.py

Requires:
    data/final/Phase_03/phase3_per_head_analysis.json
    data/final/Phase_03/phase3_length_controlled.json

Writes:
    data/final/Phase_04/ablation_targets.json
"""

import json
from pathlib import Path

PER_HEAD_PATH = "data/final/Phase_03/phase3_per_head_analysis.json"
LENGTH_CONTROLLED_PATH = "data/final/Phase_03/phase3_length_controlled.json"
OUT_PATH = "data/final/Phase_04/ablation_targets_escalated.json"

N_TARGETS_PER_MODEL = None  # None = ablate ALL FDR-significant heads within
# validated layers, not just a top-N subset. Escalated from the original
# top-8 test, which showed 0/40 flips for both models -- likely because
# 8 heads is a small fraction of the ~96 (gemma) / ~68 (llama) candidate
# heads found significant within the validated block. This tests the
# redundancy hypothesis directly: does removing MOST of the block's
# significant heads (not just the individually-strongest few) produce a
# real causal effect.


def load():
    per_head = json.load(open(PER_HEAD_PATH))
    length_ctrl = json.load(open(LENGTH_CONTROLLED_PATH))
    return per_head, length_ctrl


def select_targets(model, per_head_results, length_ctrl_results):
    validated_layers = {r["layer"] for r in length_ctrl_results if r.get("fdr_significant")}

    candidates = [
        r for r in per_head_results
        if r["layer"] in validated_layers and r.get("fdr_significant")
    ]
    candidates.sort(key=lambda r: -abs(r["effect"]))

    top = candidates if N_TARGETS_PER_MODEL is None else candidates[:N_TARGETS_PER_MODEL]

    print(f"\n{model.upper()}: {len(validated_layers)} validated layers, "
          f"{len(candidates)} candidate heads within them")
    print(f"  Selecting {len(top)} targets ({'ALL candidates' if N_TARGETS_PER_MODEL is None else f'top {N_TARGETS_PER_MODEL}'}):")
    for r in top[:10]:
        direction = "override > faithful" if r["effect"] > 0 else "override < faithful"
        print(f"    layer {r['layer']:2d} head {r['head']:2d}: effect={r['effect']:+.5f} ({direction})")
    if len(top) > 10:
        print(f"    ... and {len(top) - 10} more")

    return top


def main():
    per_head, length_ctrl = load()
    output = {}
    for model in ["gemma", "llama"]:
        output[model] = select_targets(model, per_head[model], length_ctrl.get(model, []))

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
