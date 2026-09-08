"""
Phase 4, T26: Visualize joint intervention results.

Plots the distribution of per-query effects (real ablation vs random
control) for each model, as a paired comparison -- this is the figure
for the paper regardless of whether the result comes back significant
or not; either way it's the honest picture of what happened.

Run from repo root, AFTER joint_intervention.py has produced results:
    python Tests/visualize_joint_intervention.py

Requires:
    data/final/Phase_04/joint_intervention_results.jsonl

Writes:
    figures/joint_intervention_gemma.png
    figures/joint_intervention_llama.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_PATH = "data/final/Phase_04/joint_intervention_results.jsonl"
OUT_DIR = Path("figures")


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def plot_model(rows, model_key):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, group, title in zip(
        axes,
        ["override_cases", "faithful_control_cases"],
        ["Override cases\n(want: real effect > control effect)",
         "Faithful control cases\n(want: both near zero -- no side effects)"],
    ):
        group_rows = [r for r in rows if r["group"] == group]
        if not group_rows:
            ax.set_title(f"{title}\n(no data)")
            continue

        real_effects = [r["real_effect"] for r in group_rows]
        control_effects = [r["control_effect"] for r in group_rows]

        positions = [1, 2]
        bp = ax.boxplot([real_effects, control_effects], positions=positions, widths=0.5,
                         patch_artist=True, showmeans=True)
        bp["boxes"][0].set_facecolor("#2563eb55")
        bp["boxes"][1].set_facecolor("#88888855")

        # jittered scatter of individual points on top
        rng = np.random.default_rng(0)
        ax.scatter(rng.normal(1, 0.04, len(real_effects)), real_effects, alpha=0.4, s=12, color="#2563eb")
        ax.scatter(rng.normal(2, 0.04, len(control_effects)), control_effects, alpha=0.4, s=12, color="#666666")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(["Real\n(identified components)", "Control\n(random components)"])
        ax.set_ylabel("Effect on gold-answer log-probability")
        ax.set_title(title, fontsize=10)

    fig.suptitle(f"{model_key.upper()}: joint attention+MLP intervention effect on gold-answer probability",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"joint_intervention_{model_key}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    rows = load_jsonl(RESULTS_PATH)
    for model_key in ["gemma", "llama"]:
        model_rows = [r for r in rows if r["model"] == model_key]
        if not model_rows:
            print(f"No rows for {model_key}, skipping")
            continue
        plot_model(model_rows, model_key)


if __name__ == "__main__":
    main()
