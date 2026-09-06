"""
T13+T14: Merge Phase 3 ROI results with T6 labels, then test whether
divergence differs meaningfully between faithful/override cases, and
between natural_rag_other/popqa strata (the PopQA reversal question
this whole phase exists to investigate).

Per-layer testing (NOT one pooled average) -- same lesson as T5's
entropy failure and T6/T7's ConFiQA/PopQA conflation: a real effect can
hide inside an average, or a fake one can appear from pooling unrelated
groups. Test at the finest grain the data supports (per layer here;
per-head would need more N than 120 queries currently provides).

Output: data/final/phase3_roi_analysis.json + printed summary
"""

import json
from collections import defaultdict

import numpy as np
from scipy import stats

ROI_FILE = "data/final/phase3_roi_results.jsonl"
ANALYSIS_FILE = "data/final/analysis_dataset.jsonl"  # for labels
MIN_N_FOR_RELIABLE = 15  # lower bar than T6/T7's 30, since this is a
                          # deliberately small first-pass sample (120
                          # total, ~30 per stratum before splitting by model)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def get_layer_means(roi_row):
    """Collapse each layer's per-head list to a mean, for the per-layer
    test. (Per-head would be the finer-grained follow-up once a layer
    shows a signal worth chasing down to head level.)"""
    return [np.mean(layer) if layer else None for layer in roi_row["divergence_by_layer"]]


def main():
    roi_records = load_jsonl(ROI_FILE)
    analysis_records = {r["query_id"]: r for r in load_jsonl(ANALYSIS_FILE)}

    print(f"Loaded {len(roi_records)} ROI extraction rows")

    # Attach labels + source_category to each ROI row
    merged = []
    unmatched = 0
    for roi in roi_records:
        qid = roi["query_id"]
        analysis = analysis_records.get(qid)
        if analysis is None:
            unmatched += 1
            continue
        model = roi["model"]
        label = analysis.get(f"{model}_label")
        if label not in ("faithful", "override"):
            continue  # skip retrieval_failed or missing labels
        merged.append({
            "query_id": qid,
            "model": model,
            "label": label,
            "source_category": analysis.get("source_category"),
            "layer_means": get_layer_means(roi),
        })

    print(f"Merged {len(merged)} rows with valid labels ({unmatched} unmatched query_ids)")

    num_layers_by_model = {}
    for model in sorted(set(r["model"] for r in merged)):
        model_rows = [r for r in merged if r["model"] == model]
        num_layers_by_model[model] = len(model_rows[0]["layer_means"]) if model_rows else 0
    models = sorted(num_layers_by_model.keys())
    categories = sorted(set(r["source_category"] for r in merged if r["source_category"]))
    print(f"Layer counts by model: {num_layers_by_model}  (Gemma 2=42, Llama 3.1=32 -- different, don't share a loop bound)")

    results = {"per_layer_faithful_vs_override": {}, "per_layer_category_comparison": {}}

    # --- Test 1: does divergence differ between faithful and override, per layer, per model ---
    print("\n=== Test 1: Divergence, faithful vs override, per layer ===")
    for model in models:
        print(f"\n  {model.upper()}:")
        results["per_layer_faithful_vs_override"][model] = []
        num_layers = num_layers_by_model[model]
        for layer_idx in range(num_layers):
            faithful_vals = [r["layer_means"][layer_idx] for r in merged
                              if r["model"] == model and r["label"] == "faithful"
                              and r["layer_means"][layer_idx] is not None]
            override_vals = [r["layer_means"][layer_idx] for r in merged
                              if r["model"] == model and r["label"] == "override"
                              and r["layer_means"][layer_idx] is not None]

            if len(faithful_vals) < 2 or len(override_vals) < 2:
                continue

            t_stat, p_value = stats.ttest_ind(override_vals, faithful_vals, equal_var=False)
            n_min = min(len(faithful_vals), len(override_vals))
            flag = " [SMALL N]" if n_min < MIN_N_FOR_RELIABLE else ""

            entry = {
                "layer": layer_idx, "n_faithful": len(faithful_vals), "n_override": len(override_vals),
                "mean_faithful": float(np.mean(faithful_vals)), "mean_override": float(np.mean(override_vals)),
                "t_stat": float(t_stat), "p_value": float(p_value), "significant": bool(p_value < 0.05),
            }
            results["per_layer_faithful_vs_override"][model].append(entry)

            if p_value < 0.05:
                direction = "override > faithful" if entry["mean_override"] > entry["mean_faithful"] else "override < faithful"
                print(f"    layer {layer_idx}: p={p_value:.4f} SIGNIFICANT ({direction}){flag}")

        sig_count = sum(1 for e in results["per_layer_faithful_vs_override"][model] if e["significant"])
        print(f"    -> {sig_count}/{num_layers} layers significant at p<0.05")

    # --- Test 2: does divergence differ between natural_rag_other and popqa, per layer, per model ---
    print("\n=== Test 2: Divergence, natural_rag_other vs popqa, per layer (the reversal question) ===")
    for model in models:
        print(f"\n  {model.upper()}:")
        results["per_layer_category_comparison"][model] = []
        num_layers = num_layers_by_model[model]
        for layer_idx in range(num_layers):
            natural_vals = [r["layer_means"][layer_idx] for r in merged
                             if r["model"] == model and r["source_category"] == "natural_rag_other"
                             and r["layer_means"][layer_idx] is not None]
            popqa_vals = [r["layer_means"][layer_idx] for r in merged
                          if r["model"] == model and r["source_category"] == "popqa"
                          and r["layer_means"][layer_idx] is not None]

            if len(natural_vals) < 2 or len(popqa_vals) < 2:
                continue

            t_stat, p_value = stats.ttest_ind(popqa_vals, natural_vals, equal_var=False)
            n_min = min(len(natural_vals), len(popqa_vals))
            flag = " [SMALL N]" if n_min < MIN_N_FOR_RELIABLE else ""

            entry = {
                "layer": layer_idx, "n_natural": len(natural_vals), "n_popqa": len(popqa_vals),
                "mean_natural": float(np.mean(natural_vals)), "mean_popqa": float(np.mean(popqa_vals)),
                "t_stat": float(t_stat), "p_value": float(p_value), "significant": bool(p_value < 0.05),
            }
            results["per_layer_category_comparison"][model].append(entry)

            if p_value < 0.05:
                print(f"    layer {layer_idx}: p={p_value:.4f} SIGNIFICANT{flag}")

        sig_count = sum(1 for e in results["per_layer_category_comparison"][model] if e["significant"])
        print(f"    -> {sig_count}/{num_layers} layers significant at p<0.05")

    with open("data/final/phase3_roi_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote data/final/phase3_roi_analysis.json")

    print("\n=== Honest read of what this means ===")
    print("This is a SMALL first-pass sample (120 queries, ~15-30 per group after")
    print("splitting by model/label/category). Treat any significant layer here as a")
    print("candidate worth confirming on more data, not a finalized finding -- same")
    print("caution as T7's small-N warnings, just at an earlier stage of this phase.")


if __name__ == "__main__":
    main()