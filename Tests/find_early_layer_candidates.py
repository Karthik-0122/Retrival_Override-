"""
Finds early-layer (before the validated 27-38 block) candidates for a
new steering attempt, using data ALREADY COLLECTED from yesterday's
causal-tracing work -- no new extraction needed.

causal_tracing_results.jsonl recorded recovery_by_layer for EVERY layer
(not just Phase 3's validated block) when patching clean MLP activations
into a corrupted run. High recovery at an early layer means that layer
is causally important for retrieving/producing the correct fact --
independent of Phase 3's attention-divergence metric, which only looked
at the validated block and might be blind to an earlier decision point.

Run from repo root:
    python Tests/find_early_layer_candidates.py

Requires:
    data/final/causal_tracing_results.jsonl

Writes:
    data/final/Phase_04/early_layer_candidates.json
"""

import json
from pathlib import Path
import numpy as np

RESULTS_PATH = "data/final/causal_tracing_results.jsonl"
OUT_PATH = "data/final/Phase_04/early_layer_candidates.json"

VALIDATED_BLOCK_START = 27
N_TOP_CANDIDATES = 8


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    rows = load_jsonl(RESULTS_PATH)
    gemma_rows = [r for r in rows if r["model"] == "gemma"]
    print(f"Loaded {len(gemma_rows)} gemma causal-tracing rows")

    n_layers = len(gemma_rows[0]["recovery_by_layer"])
    print(f"{n_layers} layers recorded per row")

    recovery_matrix = np.array([r["recovery_by_layer"] for r in gemma_rows])
    clean = np.array([r["clean_prob"] for r in gemma_rows])
    corrupted = np.array([r["corrupted_prob"] for r in gemma_rows])
    denom = clean - corrupted
    denom[denom < 1e-6] = 1e-6
    recovery_frac = (recovery_matrix - corrupted[:, None]) / denom[:, None]
    mean_recovery = recovery_frac.mean(axis=0)

    print("\n" + "=" * 70)
    print(f"MEAN RECOVERY IMPORTANCE, ALL {n_layers} LAYERS")
    print("=" * 70)
    for layer_idx in range(n_layers):
        marker = " <-- EARLY (candidate)" if layer_idx < VALIDATED_BLOCK_START else " (inside validated block)"
        print(f"  layer {layer_idx:2d}: {mean_recovery[layer_idx]:+.4f}{marker}")

    early_layers = [(i, mean_recovery[i]) for i in range(min(VALIDATED_BLOCK_START, n_layers))]
    early_layers.sort(key=lambda x: -x[1])
    top_early = early_layers[:N_TOP_CANDIDATES]

    print("\n" + "=" * 70)
    print(f"TOP {N_TOP_CANDIDATES} EARLY-LAYER CANDIDATES (before layer {VALIDATED_BLOCK_START})")
    print("=" * 70)
    for layer_idx, score in top_early:
        print(f"  layer {layer_idx:2d}: recovery={score:+.4f}")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "all_layer_recovery": {i: float(mean_recovery[i]) for i in range(n_layers)},
            "top_early_candidates": [{"layer": l, "recovery": float(s)} for l, s in top_early],
        }, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    print("\nThese early layers can now be targeted with a new MLP-level steering")
    print("test, the same design as the attention-head steering that worked, just")
    print("applied to a different, earlier location in the network.")


if __name__ == "__main__":
    main()