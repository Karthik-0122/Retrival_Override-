"""
T7: Analysis -- Experiment 1 Results (revised).

Reads data/final/analysis_dataset.jsonl (T6 output).

Revision from the original version: the PCS-vs-override effect reverses
in PopQA vs the other sources, and the popularity-tier split within
PopQA didn't reach significance in most cells (small N). So:
  - The PCS distribution plot is now split by source_category
    (natural_rag_other = NQ+TriviaQA, popqa, confiqa) instead of pooling
    everything into one histogram, since pooling would visually wash out
    a real, well-powered reversal.
  - The popularity-tier plot is KEPT but explicitly labeled exploratory /
    underpowered in its title -- don't present it as a clean finding.

Output:
  data/final/primary_results_table.csv
  data/final/figures/pcs_distribution_by_category.png
  data/final/figures/override_rate_vs_popularity_EXPLORATORY.png
  data/final/figures/attention_entropy_comparison.png
  data/final/spearman_results.json

Requirements:
  pip install pandas matplotlib seaborn scipy
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ANALYSIS_FILE = "data/final/analysis_dataset.jsonl"
OUTPUT_DIR = Path("data/final")
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS = ["gemma", "llama"]
SOURCE_CATEGORIES = ["natural_rag_other", "popqa", "confiqa"]
MIN_N_FOR_RELIABLE = 30

sns.set_theme(style="whitegrid")


def load_analysis_df() -> pd.DataFrame:
    records = []
    with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    if "source_category" not in df.columns:
        # fallback if running against an older analysis_dataset.jsonl
        df["source_category"] = df["source"].apply(
            lambda s: "confiqa" if s == "confiqa" else ("popqa" if s == "popqa" else "natural_rag_other")
        )
    return df


def build_primary_results_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        label_col = f"{model}_label"
        em_col = f"{model}_em"
        pcs_col = f"pcs_{model}"

        for cat, group in df.groupby("source_category"):
            n_faithful = (group[label_col] == "faithful").sum()
            n_override = (group[label_col] == "override").sum()
            total = n_faithful + n_override
            override_rate = n_override / total if total > 0 else float("nan")

            mean_em = group[em_col].mean() if em_col in group.columns else float("nan")
            mean_pcs_override = group.loc[group[label_col] == "override", pcs_col].mean()
            mean_pcs_faithful = group.loc[group[label_col] == "faithful", pcs_col].mean()

            rows.append({
                "model": model,
                "source_category": cat,
                "n": total,
                "reliable_n": total >= MIN_N_FOR_RELIABLE,
                "em": mean_em,
                "override_rate": override_rate,
                "mean_pcs_override": mean_pcs_override,
                "mean_pcs_faithful": mean_pcs_faithful,
                "pcs_direction": ("override<faithful" if mean_pcs_override < mean_pcs_faithful
                                   else "override>faithful"),
            })

    return pd.DataFrame(rows)


def plot_pcs_distribution_by_category(df: pd.DataFrame):
    """3 categories x 2 models = 6 panels. This is the corrected version
    of the original single pooled histogram -- pooling hid the fact that
    PopQA reverses the direction seen everywhere else."""
    fig, axes = plt.subplots(len(MODELS), len(SOURCE_CATEGORIES),
                              figsize=(15, 8), sharey=True)

    for row_idx, model in enumerate(MODELS):
        label_col = f"{model}_label"
        pcs_col = f"pcs_{model}"

        for col_idx, cat in enumerate(SOURCE_CATEGORIES):
            ax = axes[row_idx, col_idx]
            subset = df[(df["source_category"] == cat) & df[label_col].isin(["faithful", "override"])]

            for label, color in [("faithful", "#2ca02c"), ("override", "#d62728")]:
                values = subset.loc[subset[label_col] == label, pcs_col].dropna()
                ax.hist(values, bins=20, alpha=0.6, label=label, color=color)

            n_total = len(subset)
            reliability = "" if n_total >= MIN_N_FOR_RELIABLE else " (small N)"
            ax.set_title(f"{model.upper()} / {cat}\nn={n_total}{reliability}", fontsize=10)
            if row_idx == len(MODELS) - 1:
                ax.set_xlabel("PCS")
            if col_idx == 0:
                ax.set_ylabel("Count")
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8)

    fig.suptitle("PCS distribution by source category and label\n"
                  "(natural_rag_other + confiqa: override<faithful is well-powered; "
                  "popqa: reversed, but tier-level driver is unresolved)", fontsize=11)
    fig.tight_layout()
    out_path = FIGURES_DIR / "pcs_distribution_by_category.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_override_rate_vs_popularity_exploratory(df: pd.DataFrame):
    """EXPLORATORY -- most tier cells in the underlying data did not
    reach statistical significance for the PCS hypothesis. This plot is
    kept for completeness but should be captioned as exploratory /
    underpowered in the paper, not presented as a confirmed trend."""
    if "popularity_tier" not in df.columns or not df["popularity_tier"].notna().any():
        print("No popularity_tier data available -- skipping this plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    tier_order = ["low", "medium", "high"]

    for model in MODELS:
        label_col = f"{model}_label"
        rates = []
        tiers_present = []
        for tier in tier_order:
            group = df[(df["source"] == "popqa") & (df["popularity_tier"] == tier)]
            if len(group) == 0:
                continue
            n_f = (group[label_col] == "faithful").sum()
            n_o = (group[label_col] == "override").sum()
            total = n_f + n_o
            if total == 0:
                continue
            rates.append(n_o / total)
            tiers_present.append(f"{tier}\n(n={total})")
        ax.plot(tiers_present, rates, marker="o", label=model)

    ax.set_xlabel("Popularity tier (PopQA only)")
    ax.set_ylabel("Override rate")
    ax.set_title("EXPLORATORY -- override rate vs popularity tier\n"
                  "(most cells below n=30; NOT a confirmed monotonic trend)",
                  color="darkred")
    ax.legend()
    fig.tight_layout()
    out_path = FIGURES_DIR / "override_rate_vs_popularity_EXPLORATORY.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_attention_entropy_comparison(df: pd.DataFrame):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(12, 5), sharey=True)
    if len(MODELS) == 1:
        axes = [axes]

    for ax, model in zip(axes, MODELS):
        label_col = f"{model}_label"
        entropy_col = f"{model}_attn_entropy_on_passage"
        if entropy_col not in df.columns:
            ax.set_title(f"{model.upper()}: no entropy data")
            continue

        subset = df[df[label_col].isin(["faithful", "override"])].copy()
        subset = subset.dropna(subset=[entropy_col])

        sns.boxplot(data=subset, x=label_col, y=entropy_col, ax=ax,
                    order=["faithful", "override"], palette=["#2ca02c", "#d62728"])
        ax.set_title(f"{model.upper()}: attention entropy over passage")
        ax.set_xlabel("")
        ax.set_ylabel("Mean attention entropy" if model == MODELS[0] else "")

    fig.tight_layout()
    out_path = FIGURES_DIR / "attention_entropy_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def compute_spearman_correlation(df: pd.DataFrame):
    results = {}
    for model in MODELS:
        label_col = f"{model}_label"
        pcs_col = f"pcs_{model}"

        results[model] = {}
        for cat in SOURCE_CATEGORIES + ["all"]:
            subset = df if cat == "all" else df[df["source_category"] == cat]
            subset = subset[subset[label_col].isin(["faithful", "override"])].dropna(subset=[pcs_col])

            if len(subset) < 3:
                print(f"{model}/{cat}: not enough data for correlation")
                continue

            override_binary = (subset[label_col] == "override").astype(int)
            corr, p_value = stats.spearmanr(subset[pcs_col], override_binary)

            results[model][cat] = {"spearman_r": corr, "p_value": p_value, "n": len(subset)}
            print(f"\n{model.upper()} / {cat} (n={len(subset)}):")
            print(f"  r = {corr:.4f}, p = {p_value:.4f} "
                  f"{'(significant)' if p_value < 0.05 else '(NOT significant)'}")

    return results


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {ANALYSIS_FILE}...")
    df = load_analysis_df()
    print(f"Loaded {len(df)} rows (faithful+override subset)")

    print("\nBuilding primary results table...")
    table = build_primary_results_table(df)
    table_path = OUTPUT_DIR / "primary_results_table.csv"
    table.to_csv(table_path, index=False)
    print(f"Saved {table_path}")
    print(table.to_string(index=False))

    print("\nGenerating figures...")
    plot_pcs_distribution_by_category(df)
    plot_override_rate_vs_popularity_exploratory(df)
    plot_attention_entropy_comparison(df)

    print("\nComputing Spearman correlations (pooled + per source_category)...")
    spearman_results = compute_spearman_correlation(df)
    spearman_path = OUTPUT_DIR / "spearman_results.json"
    with open(spearman_path, "w") as f:
        json.dump(spearman_results, f, indent=2)
    print(f"Saved {spearman_path}")

    print("\nT7 analysis complete.")


if __name__ == "__main__":
    main()