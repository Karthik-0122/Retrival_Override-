"""
Cross-reference check: do DLA's top 'negative heads' (candidates for
actively suppressing the correct answer) also show up as top hits in
Phase 3's original per-head attention-divergence analysis? If the same
specific heads are flagged by two completely independent methods
(divergence-based correlation vs. logit-attribution-based causation-
adjacent measurement), that's much stronger evidence than either alone.

No GPU needed -- pure comparison of two JSON files you already have.

Run from repo root:
    python Tests/crossref_dla_phase3.py

Requires:
    data/final/Phase_04/dla_results.json
    data/final/Phase_03/phase3_per_head_analysis.json
"""

import json

DLA_PATH = "data/final/Phase_04/dla_results.json"
PER_HEAD_PATH = "data/final/Phase_03/phase3_per_head_analysis.json"

N_TOP_DLA = 15  # how many top-negative DLA heads to check


def main():
    dla = json.load(open(DLA_PATH))
    per_head = json.load(open(PER_HEAD_PATH))

    for model in ["gemma", "llama"]:
        print(f"\n{'='*70}\n{model.upper()}\n{'='*70}")

        if model not in dla or model not in per_head:
            print(f"  Missing data for {model} in one of the input files -- skipping.")
            continue

        contribs = dla[model]["contributions"]
        ranked = sorted(contribs.items(), key=lambda x: x[1])[:N_TOP_DLA]

        # build a lookup: (layer, head) -> phase3 per-head record
        phase3_lookup = {(r["layer"], r["head"]): r for r in per_head[model]}

        print(f"{'Layer':>6} {'Head':>5} {'DLA contrib':>12} {'Phase3 FDR sig?':>16} {'Phase3 effect':>14} {'Phase3 |rank|':>14}")

        # rank all phase3 heads by |effect| to report where each DLA head falls
        phase3_ranked = sorted(per_head[model], key=lambda r: -abs(r["effect"]))
        phase3_rank_lookup = {(r["layer"], r["head"]): i + 1 for i, r in enumerate(phase3_ranked)}
        total_heads = len(phase3_ranked)

        n_overlap_sig = 0
        n_overlap_top10pct = 0
        for key, dla_val in ranked:
            layer, head = map(int, key.split("_"))
            p3 = phase3_lookup.get((layer, head))
            if p3 is None:
                print(f"{layer:>6} {head:>5} {dla_val:>12.4f} {'not found':>16}")
                continue
            sig = p3.get("fdr_significant", False)
            effect = p3.get("effect", None)
            rank = phase3_rank_lookup.get((layer, head), None)
            rank_pct = f"{rank}/{total_heads}" if rank else "?"
            if sig:
                n_overlap_sig += 1
            if rank and rank <= total_heads * 0.1:
                n_overlap_top10pct += 1
            print(f"{layer:>6} {head:>5} {dla_val:>12.4f} {str(sig):>16} {effect:>14.4f} {rank_pct:>14}")

        print(f"\n  {n_overlap_sig}/{N_TOP_DLA} of DLA's top negative heads were ALSO FDR-significant in Phase 3")
        print(f"  {n_overlap_top10pct}/{N_TOP_DLA} of DLA's top negative heads were in Phase 3's top 10% by effect size")
        print(f"  (chance baseline for top-10% overlap, if unrelated: ~{N_TOP_DLA * 0.1:.1f}/{N_TOP_DLA})")


if __name__ == "__main__":
    main()