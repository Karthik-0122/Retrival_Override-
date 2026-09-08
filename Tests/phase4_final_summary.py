"""
Phase 4, T27: Final checkpoint and consolidated summary.

Reads every ablation result produced across Phase 4 and prints one
consolidated table plus an honest, auto-generated written summary --
regardless of whether the joint intervention came back significant or
not. This is the artifact-consistency check plus the "is this clean
enough to write up" decision point, done at the same time.

Run from repo root, AFTER joint_intervention.py has completed:
    python Tests/phase4_final_summary.py

Reads (whichever of these exist -- missing files are noted, not fatal):
    data/final/Phase_04/ablation_results_escalated.jsonl        (attention-only, real)
    data/final/Phase_04/ablation_results_control.jsonl          (attention-only, control)
    data/final/Phase_04/mlp_ablation_results.jsonl               (MLP wholesale, real+control)
    data/final/Phase_04/mlp_stage1_screen.jsonl                  (MLP per-layer screen)
    data/final/Phase_04/joint_intervention_results.jsonl         (joint attn+MLP, sensitive metric)

Writes:
    data/final/Phase_04/phase4_final_summary.txt
"""

import json
from pathlib import Path
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

FILES = {
    "attn_real": "data/final/Phase_04/ablation_results_escalated.jsonl",
    "attn_control": "data/final/Phase_04/ablation_results_control.jsonl",
    "mlp_wholesale": "data/final/Phase_04/mlp_ablation_results.jsonl",
    "mlp_screen": "data/final/Phase_04/mlp_stage1_screen.jsonl",
    "joint": "data/final/Phase_04/joint_intervention_results.jsonl",
}
OUT_PATH = "data/final/Phase_04/phase4_final_summary.txt"


def load_jsonl_if_exists(path):
    p = Path(path)
    if not p.exists():
        return None
    return [json.loads(l) for l in open(path) if l.strip()]


def section(title):
    return f"\n{'='*70}\n{title}\n{'='*70}\n"


def summarize_attention(data_real, data_control, out):
    out.append(section("1. ATTENTION-HEAD ABLATION (block-level, all validated heads)"))
    if data_real is None:
        out.append("  No data found -- skipped.\n")
        return
    for model in ["gemma", "llama"]:
        real_by_id = {r["query_id"]: r for r in data_real if r["model"] == model and r["group"] == "override_cases"}
        n_flip = sum(1 for r in real_by_id.values() if r["flipped_to_correct"])
        out.append(f"  {model}: {n_flip}/{len(real_by_id)} override cases flipped to correct (real targets)")
        if data_control is not None:
            control_by_id = {r["query_id"]: r for r in data_control if r["model"] == model and r["group"] == "override_cases"}
            common = set(real_by_id) & set(control_by_id)
            if common:
                both = real_only = control_only = neither = 0
                for qid in common:
                    rf, cf = real_by_id[qid]["flipped_to_correct"], control_by_id[qid]["flipped_to_correct"]
                    if rf and cf:
                        both += 1
                    elif rf:
                        real_only += 1
                    elif cf:
                        control_only += 1
                    else:
                        neither += 1
                table = [[both, real_only], [control_only, neither]]
                if real_only + control_only > 0:
                    p = mcnemar(table, exact=True).pvalue
                    out.append(f"    vs random control (n={len(common)} paired): McNemar p={p:.4f} "
                               f"{'SIGNIFICANT' if p < 0.05 else 'not significant'}")
                else:
                    out.append(f"    vs random control (n={len(common)} paired): no discordant pairs, p=1.0")
    out.append("")


def summarize_mlp(data_wholesale, data_screen, out):
    out.append(section("2. MLP ABLATION"))
    if data_wholesale is not None:
        for model in ["gemma", "llama"]:
            for condition in ["REAL", "CONTROL"]:
                rows = [r for r in data_wholesale if r["model"] == model and r["condition"] == condition
                        and r["group"] == "override_cases"]
                if rows:
                    n_flip = sum(1 for r in rows if r["flipped_to_correct"])
                    out.append(f"  {model} wholesale [{condition}]: {n_flip}/{len(rows)} flipped to correct")
    else:
        out.append("  No wholesale MLP data found -- skipped.")

    if data_screen is not None:
        out.append("\n  Per-layer screen (Gemma, n=10/layer -- underpowered, see notes):")
        for r in data_screen:
            out.append(f"    layer {r['layer']:2d} [{r['condition']}]: "
                       f"flip_correct={r['flip_correct']}/{r['n_override']}")
    else:
        out.append("\n  No per-layer screen data found -- skipped.")
    out.append("")


def summarize_joint(data, out):
    out.append(section("3. JOINT ATTENTION + MLP INTERVENTION (sensitive logprob metric)"))
    if data is None:
        out.append("  No data found -- this is the most important result and it's missing. Run Tests/joint_intervention.py.\n")
        return
    for model in ["gemma", "llama"]:
        for group in ["override_cases", "faithful_control_cases"]:
            rows = [r for r in data if r["model"] == model and r["group"] == group]
            if len(rows) < 2:
                continue
            real_e = [r["real_effect"] for r in rows]
            control_e = [r["control_effect"] for r in rows]
            t_stat, p = stats.ttest_rel(real_e, control_e)
            mean_real, mean_control = sum(real_e) / len(real_e), sum(control_e) / len(control_e)
            sig = "SIGNIFICANT" if p < 0.05 else "not significant"
            out.append(f"  {model} {group} (n={len(rows)}): real={mean_real:+.4f}, "
                       f"control={mean_control:+.4f}, p={p:.4f} [{sig}]")
    out.append("")


def write_conclusion(out):
    out.append(section("OVERALL PHASE 4 CONCLUSION"))
    out.append(
        "Phase 3 established a robust, multiply-validated correlational localization\n"
        "of the retrieval-override signature (Gemma: layers 27-38; Llama: layers 2-4).\n"
        "Phase 4 tested whether this localization is causally load-bearing.\n\n"
        "Three approaches were tested, in order of increasing methodological rigor:\n"
        "  1. Attention-head ablation alone (block-level) -- see Section 1 above.\n"
        "  2. MLP ablation alone (wholesale, then per-layer screen) -- see Section 2.\n"
        "  3. Joint attention+MLP ablation with a sensitive continuous outcome metric,\n"
        "     modeled on the approach used successfully in ReDeEP (Sun et al., ICLR 2025)\n"
        "     -- see Section 3. This is the definitive result; read Section 3's p-values\n"
        "     to determine which conclusion applies:\n\n"
        "  IF SIGNIFICANT (p<0.05, real > control): the identified layers are causally\n"
        "  validated as driving override behavior, confirming Phase 3's localization is\n"
        "  not merely correlational.\n\n"
        "  IF NOT SIGNIFICANT: across all three tested approaches, the correlational\n"
        "  signature could not be causally validated via targeted ablation. This is\n"
        "  consistent with a distributed, redundant implementation of the mechanism\n"
        "  (supported by Phase 3's own finding that 40-60% of heads in the block carry\n"
        "  signal) rather than a small set of causally necessary components. This is a\n"
        "  legitimate, informative negative result, not a failure of the investigation.\n"
    )


def main():
    data = {k: load_jsonl_if_exists(v) for k, v in FILES.items()}

    out = []
    out.append("PHASE 4 FINAL CONSOLIDATED SUMMARY")
    summarize_attention(data["attn_real"], data["attn_control"], out)
    summarize_mlp(data["mlp_wholesale"], data["mlp_screen"], out)
    summarize_joint(data["joint"], out)
    write_conclusion(out)

    full_text = "\n".join(out)
    print(full_text)

    with open(OUT_PATH, "w") as f:
        f.write(full_text)
    print(f"\n\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
