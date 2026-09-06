"""
Phase 3, item 6 (T15): layer x head heatmap of the override-vs-faithful
divergence signal, for both models, with significance overlays.

Produces one PNG per model:
    figures/phase3_heatmap_gemma.png
    figures/phase3_heatmap_llama.png

Each cell = mean divergence (override - faithful) for that (layer, head).
Cells outlined in black = FDR-significant at the raw per-head test.
Column (layer) header marked with * if that layer survived BOTH FDR
and length-control (your final validated core layers).

Run from repo root:
    python tests/make_heatmap.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PER_HEAD_PATH = "data/final/phase3_per_head_analysis.json"
LENGTH_CONTROLLED_PATH = "data/final/phase3_length_controlled.json"
OUT_DIR = Path("figures")

def load():
    per_head = json.load(open(PER_HEAD_PATH))
    length_ctrl = json.load(open(LENGTH_CONTROLLED_PATH))
    return per_head, length_ctrl

def make_heatmap(model, per_head_results, length_ctrl_results):
    layers = sorted(set(r["layer"] for r in per_head_results))
    heads = sorted(set(r["head"] for r in per_head_results))
    n_layers, n_heads = len(layers), len(heads)

    effect_grid = np.zeros((n_heads, n_layers))
    sig_grid = np.zeros((n_heads, n_layers), dtype=bool)

    layer_idx_map = {l: i for i, l in enumerate(layers)}
    head_idx_map = {h: i for i, h in enumerate(heads)}

    for r in per_head_results:
        li, hi = layer_idx_map[r["layer"]], head_idx_map[r["head"]]
        effect_grid[hi, li] = r["effect"]
        sig_grid[hi, li] = r["fdr_significant"]

    core_layers = {r["layer"] for r in length_ctrl_results if r["fdr_significant"]}

    vmax = np.abs(effect_grid).max()
    fig, ax = plt.subplots(figsize=(max(10, n_layers * 0.35), max(4, n_heads * 0.3)))
    im = ax.imshow(effect_grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for hi in range(n_heads):
        for li in range(n_layers):
            if sig_grid[hi, li]:
                ax.add_patch(plt.Rectangle((li - 0.5, hi - 0.5), 1, 1,
                                            fill=False, edgecolor="black", linewidth=0.6))

    xtick_labels = [f"{l}*" if l in core_layers else str(l) for l in layers]
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(xtick_labels, fontsize=7, rotation=90)
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels(heads, fontsize=7)
    ax.set_xlabel("Layer  (* = survives FDR + length control)")
    ax.set_ylabel("Head")
    ax.set_title(f"{model.upper()}: override - faithful divergence, per layer x head\n"
                 f"(black outline = FDR-significant per-head)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("effect (override - faithful)")

    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"phase3_heatmap_{model}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")

def main():
    per_head, length_ctrl = load()
    for model in ["gemma", "llama"]:
        make_heatmap(model, per_head[model], length_ctrl[model])

if __name__ == "__main__":
    main()