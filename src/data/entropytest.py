"""
Entropy significance test: mean_attn_entropy_on_passage(override) vs
mean_attn_entropy_on_passage(faithful).

The boxplot in T7 showed override entropy visually lower than faithful
for both models, but that was never actually tested statistically --
this closes that gap, using the same t-test approach as T6's PCS
hypothesis check, applied to entropy instead.

Hypothesis: LOWER entropy in override cases (model concentrating less
on the passage when it ends up ignoring/overriding it).

Reads data/final/analysis_dataset.jsonl directly (T6's output) -- no
need to re-merge T3/T4/T5.
"""

import json

import pandas as pd
from scipy import stats

ANALYSIS_FILE = "data/final/analysis_dataset.jsonl"
MODELS = ["gemma", "llama"]
MIN_N_FOR_RELIABLE = 30
SOURCE_CATEGORIES = ["natural_rag_other", "popqa", "confiqa"]


def load_analysis_df() -> pd.DataFrame:
    records = []
    with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    if "source_category" not in df.columns:
        df["source_category"] = df["source"].apply(
            lambda s: "confiqa" if s == "confiqa" else ("popqa" if s == "popqa" else "natural_rag_other")
        )
    return df


def _entropy_test(faithful_vals, override_vals, label=""):
    if len(faithful_vals) < 2 or len(override_vals) < 2:
        print(f"    {label}: not enough samples to test "
              f"(faithful n={len(faithful_vals)}, override n={len(override_vals)})")
        return None

    mean_faithful = faithful_vals.mean()
    mean_override = override_vals.mean()
    t_stat, p_value = stats.ttest_ind(override_vals, faithful_vals, equal_var=False)

    direction = ("override < faithful (CONFIRMS hypothesis: lower entropy in override)"
                 if mean_override < mean_faithful
                 else "override > faithful (CONTRADICTS hypothesis)")

    n_min = min(len(faithful_vals), len(override_vals))
    n_flag = f"  [SMALL N -- unreliable, n={n_min} < {MIN_N_FOR_RELIABLE}]" if n_min < MIN_N_FOR_RELIABLE else ""

    print(f"    {label}: faithful n={len(faithful_vals)} mean={mean_faithful:.4f}  |  "
          f"override n={len(override_vals)} mean={mean_override:.4f}{n_flag}")
    print(f"      t={t_stat:.3f}, p={p_value:.4f} -> {direction} "
          f"{'(p < 0.05, significant)' if p_value < 0.05 else '(NOT significant at this N)'}")

    return {
        "mean_faithful": mean_faithful, "mean_override": mean_override,
        "t_stat": t_stat, "p_value": p_value,
        "n_faithful": len(faithful_vals), "n_override": len(override_vals),
        "significant": p_value < 0.05,
        "confirms_hypothesis": mean_override < mean_faithful,
    }


def main():
    print(f"Loading {ANALYSIS_FILE}...")
    df = load_analysis_df()
    print(f"Loaded {len(df)} rows\n")

    print("=== Attention Entropy Hypothesis Check: entropy(override) < entropy(faithful)? ===")

    results = {}
    for model in MODELS:
        label_col = f"{model}_label"
        entropy_col = f"{model}_attn_entropy_on_passage"

        if entropy_col not in df.columns:
            print(f"\n  {model}: no {entropy_col} column found, skipping")
            continue

        results[model] = {}
        print(f"\n  {model.upper()}:")

        print(f"  -- Pooled (all sources) --")
        pooled = _entropy_test(
            df.loc[df[label_col] == "faithful", entropy_col].dropna(),
            df.loc[df[label_col] == "override", entropy_col].dropna(),
            label="pooled",
        )
        results[model]["pooled"] = pooled

        print(f"  -- Per source category --")
        for cat in SOURCE_CATEGORIES:
            group = df[df["source_category"] == cat]
            cat_result = _entropy_test(
                group.loc[group[label_col] == "faithful", entropy_col].dropna(),
                group.loc[group[label_col] == "override", entropy_col].dropna(),
                label=cat,
            )
            results[model][cat] = cat_result

    out_path = "data/final/entropy_significance_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()