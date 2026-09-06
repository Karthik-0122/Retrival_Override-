"""
Applies Benjamini-Hochberg FDR correction to phase3_roi_analysis.json,
in place, per model per test. Adds 'fdr_p' and 'fdr_significant' to
every layer entry.
"""

import json
from statsmodels.stats.multitest import multipletests

PATH = "data/final/phase3_roi_analysis.json"
TESTS = ["per_layer_faithful_vs_override", "per_layer_category_comparison"]
MODELS = ["gemma", "llama"]

def main():
    with open(PATH) as f:
        data = json.load(f)

    print("=" * 70)
    for test in TESTS:
        print(f"\n{test}")
        for model in MODELS:
            layers = data[test][model]
            pvals = [l["p_value"] for l in layers]

            raw_sig = sum(l["significant"] for l in layers)

            rejected, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

            for l, r, p in zip(layers, rejected, p_adj):
                l["fdr_p"] = float(p)
                l["fdr_significant"] = bool(r)

            fdr_sig = sum(rejected)
            print(f"  {model:6s}: {raw_sig:2d}/{len(layers)} raw significant  ->  {fdr_sig:2d}/{len(layers)} survive FDR")

    with open(PATH, "w") as f:
        json.dump(data, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Wrote fdr_p / fdr_significant fields into {PATH}")

if __name__ == "__main__":
    main()
