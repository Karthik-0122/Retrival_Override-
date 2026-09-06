"""
Step 4 + 5: aggregate causal-tracing recovery curves per model, and
overlay against the existing override signature (length-controlled
FDR results) to check whether fact-storage and override-decision
layers are actually separated.

Run from repo root:
    python tests/compare_storage_vs_override.py

Requires:
    data/final/causal_tracing_results.jsonl   (from extract_causal_tracing.py)
    data/final/Phase_03/phase3_length_controlled.json  (already exists)

Writes:
    figures/storage_vs_override_gemma.png
    figures/storage_vs_override_llama.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt

TRACING_PATH = "data/final/causal_tracing_results.jsonl"
OVERRIDE_PATH = "data/final/Phase_03/phase3_length_controlled.json"


def load_tracing():
    rows = [json.loads(l) for l in open(TRACING_PATH)]
    return rows


def load_override():
    return json.load(open(OVERRIDE_PATH))


def normalize(arr):
    arr = np.array(arr)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return arr
    return (arr - lo) / (hi - lo)


def plot_model(model, tracing_rows, override_layers):
    model_rows = [r for r in tracing_rows if r["model"] == model]
    if not model_rows:
        print(f"No causal-tracing rows for {model}, skipping.")
        return

    n_layers = len(model_rows[0]["recovery_by_layer"])
    recovery_matrix = np.array([r["recovery_by_layer"] for r in model_rows])
    mean_recovery = recovery_matrix.mean(axis=0)

    # storage importance = how much this layer's patch recovers relative to
    # clean/corrupted baseline -- normalize per-probe using clean/corrupted probs
    clean = np.array([r["clean_prob"] for r in model_rows])
    corrupted = np.array([r["corrupted_prob"] for r in model_rows])
    denom = (clean - corrupted)
    denom[denom < 1e-6] = 1e-6
    recovery_frac = (recovery_matrix - corrupted[:, None]) / denom[:, None]
    mean_recovery_frac = recovery_frac.mean(axis=0)

    storage_peak_layer = int(np.argmax(mean_recovery_frac))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(range(n_layers), normalize(mean_recovery_frac), label="Fact-storage recovery (causal tracing)", color="#2563eb", linewidth=2)

    override_sig_layers = set(l["layer"] for l in override_layers if l.get("fdr_significant"))
    override_curve = np.array([1.0 if l in override_sig_layers else 0.0 for l in range(n_layers)])
    ax.fill_between(range(n_layers), 0, override_curve, alpha=0.25, color="#d64545", label="Override-significant layer (length-controlled + FDR)")

    ax.axvline(storage_peak_layer, color="#2563eb", linestyle="--", alpha=0.6)
    ax.text(storage_peak_layer, 1.02, f"storage peak: L{storage_peak_layer}", color="#2563eb", fontsize=8, ha="center")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized recovery / significance")
    ax.set_title(f"{model.upper()}: fact-storage recovery vs. override-significant layers")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    out_path = f"figures/storage_vs_override_{model}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")

    gap_layers = [l for l in range(n_layers) if l not in override_sig_layers]
    print(f"  {model}: storage peak at layer {storage_peak_layer}; "
          f"override-significant layers = {sorted(override_sig_layers)}")
    if storage_peak_layer not in override_sig_layers:
        print(f"  -> Storage peak layer {storage_peak_layer} is OUTSIDE the override-significant set. "
              f"Supports a storage/decision separation.")
    else:
        print(f"  -> Storage peak layer {storage_peak_layer} FALLS INSIDE the override-significant set. "
              f"Does not cleanly support separation -- treat the 'commitment circuit' framing cautiously.")


def main():
    tracing_rows = load_tracing()
    override_data = load_override()
    for model in ["gemma", "llama"]:
        plot_model(model, tracing_rows, override_data.get(model, []))


if __name__ == "__main__":
    main()
