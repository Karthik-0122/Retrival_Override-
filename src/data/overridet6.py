"""
T6: Override / Faithful Labeling.

Merges the outputs of T3 (PCS), T4 (retrieval), and T5 (generation) by
query_id, then:
  1. Normalize extracted answer + gold answer (lowercase, strip articles,
     strip punctuation)
  2. Compute EM and token F1 per query per model
  3. Apply label logic per model:
       faithful        = retrieval_succeeded AND model_correct
       override        = retrieval_succeeded AND NOT model_correct
       retrieval_failed = NOT retrieval_succeeded -> excluded from analysis
  4. Compute override rate = |override| / (|faithful| + |override|)
     per model, per dataset (source), per popularity tier
  5. Verify key hypothesis: mean PCS in override cases > mean PCS in
     faithful cases (t-test + simple mean comparison)

retrieval_succeeded here means retrieval_success != "none" (i.e. verbatim
or soft counts as retrieval succeeding) -- adjust if your definition of
"succeeded" should be stricter (verbatim only).

Output: data/final/analysis_dataset.jsonl (faithful + override subset only,
        per your T8 artifact spec) plus a printed summary table.

Requirements:
  pip install pandas scipy
"""

import json
import re
import string
from pathlib import Path

import pandas as pd
from scipy import stats

RECORDS_FILE = "data/final/dataset_1500_with_titles_1.jsonl"
PCS_FILE = "data/final/pcs_scores.jsonl"
RETRIEVAL_FILE = "data/final/retrieval_results.jsonl"
GENERATION_FILE = "data/final/generation_results.jsonl"
OUTPUT_FILE = "data/final/analysis_dataset.jsonl"
EVAL_RESULTS_FILE = "data/final/eval_results.jsonl"  # T8 artifact: full unfiltered record

MODELS = ["gemma", "llama"]
EM_MATCH_THRESHOLD = 1.0  # exact match after normalization
CORRECT_F1_THRESHOLD = 0.5  # F1 above this also counts as "correct" if EM fails
MIN_N_FOR_RELIABLE_RATE = 30  # cells below this get an explicit small-N warning

# ConFiQA's gold answer is the counterfactual-consistent answer, so its
# "override" means the model resisted an injected counterfactual and
# answered from real-world knowledge -- the OPPOSITE of what "override"
# means for NQ/TriviaQA/PopQA (ignoring a correctly-retrieved passage and
# getting it wrong). Pooling these under one "override rate" conflates two
# structurally different behaviors. Report them separately.
CONFLICT_INJECTED_SOURCES = {"confiqa"}
NATURAL_RAG_SOURCES = {"nq", "triviaqa", "popqa"}


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    num_common = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def get_gold_answer_aliases(record) -> list:
    """Returns ALL valid answer strings (primary value + aliases) --
    matches the same fix applied to retrieve_t4.py, so correctness
    checking is consistent with retrieval matching."""
    ga = record.get("gold_answers")
    if isinstance(ga, dict):
        aliases = ga.get("aliases") or []
        value = ga.get("value")
        all_answers = ([value] if value else []) + list(aliases)
        return [a for a in all_answers if a]
    if isinstance(ga, list):
        return [str(a) for a in ga if a]
    return [str(ga)] if ga else []


def get_gold_answer_text(record) -> str:
    """Primary answer only -- kept for display/output. Use
    get_gold_answer_aliases() for correctness checking."""
    aliases = get_gold_answer_aliases(record)
    return aliases[0] if aliases else ""


def best_match_em_f1(pred: str, aliases: list):
    """Checks pred against EVERY alias, returns the best (max) EM/F1
    across all of them -- a model answering any valid alias should count
    as correct, not just the one alias that happened to be listed first."""
    if not aliases:
        return False, 0.0
    best_em = False
    best_f1 = 0.0
    for alias in aliases:
        em = exact_match(pred, alias)
        f1 = token_f1(pred, alias)
        best_em = best_em or em
        best_f1 = max(best_f1, f1)
    return best_em, best_f1


def build_merged_dataframe():
    records = load_jsonl(RECORDS_FILE)
    pcs_records = load_jsonl(PCS_FILE)
    retrieval_records = load_jsonl(RETRIEVAL_FILE)
    generation_records = load_jsonl(GENERATION_FILE)

    records_df = pd.DataFrame(records)
    records_df["gold_answer_text"] = records_df.apply(get_gold_answer_text, axis=1)
    records_df["gold_answer_aliases"] = records_df.apply(get_gold_answer_aliases, axis=1)

    pcs_df = pd.DataFrame(pcs_records)
    retrieval_df = pd.DataFrame(retrieval_records)[["query_id", "retrieval_success"]]
    generation_df = pd.DataFrame(generation_records)

    merged = records_df.merge(pcs_df, on="query_id", how="left")
    merged = merged.merge(retrieval_df, on="query_id", how="left")
    merged = merged.merge(generation_df, on="query_id", how="left")
    return merged


def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["retrieval_succeeded"] = df["retrieval_success"] != "none"

    for model in MODELS:
        answer_col = f"{model}_answer"
        em_col = f"{model}_em"
        f1_col = f"{model}_f1"
        correct_col = f"{model}_correct"
        label_col = f"{model}_label"

        df[em_col] = df.apply(
            lambda r: best_match_em_f1(r.get(answer_col, ""), r["gold_answer_aliases"])[0], axis=1
        )
        df[f1_col] = df.apply(
            lambda r: best_match_em_f1(r.get(answer_col, ""), r["gold_answer_aliases"])[1], axis=1
        )
        df[correct_col] = df[em_col] | (df[f1_col] >= CORRECT_F1_THRESHOLD)

        def label_row(r):
            if not r["retrieval_succeeded"]:
                return "retrieval_failed"
            return "faithful" if r[correct_col] else "override"

        df[label_col] = df.apply(label_row, axis=1)

    return df


def _n_warning(n: int) -> str:
    return f"  [SMALL N -- unreliable, n={n} < {MIN_N_FOR_RELIABLE_RATE}]" if n < MIN_N_FOR_RELIABLE_RATE else ""


def compute_override_rates(df: pd.DataFrame):
    print("\n=== Override Rates ===")
    for model in MODELS:
        label_col = f"{model}_label"
        analyzable = df[df[label_col] != "retrieval_failed"]

        # --- Pooled rate, split by source category (NOT mixed together) ---
        natural = analyzable[analyzable["source"].isin(NATURAL_RAG_SOURCES)]
        n_f = (natural[label_col] == "faithful").sum()
        n_o = (natural[label_col] == "override").sum()
        t = n_f + n_o
        rate = n_o / t if t > 0 else float("nan")
        print(f"\n{model.upper()} -- Natural RAG datasets only (NQ/TriviaQA/PopQA), "
              f"override = ignored correctly-retrieved evidence:")
        print(f"  pooled: {rate:.3f} ({n_o}/{t}){_n_warning(t)}")

        confiqa = analyzable[analyzable["source"].isin(CONFLICT_INJECTED_SOURCES)]
        n_f_c = (confiqa[label_col] == "faithful").sum()
        n_o_c = (confiqa[label_col] == "override").sum()
        t_c = n_f_c + n_o_c
        rate_c = n_o_c / t_c if t_c > 0 else float("nan")
        print(f"\n{model.upper()} -- ConFiQA only, 'override' here = model RESISTED an "
              f"injected counterfactual (structurally different from above, do not pool):")
        print(f"  pooled: {rate_c:.3f} ({n_o_c}/{t_c}){_n_warning(t_c)}")

        print(f"\n  By source dataset (for reference):")
        for source, group in analyzable.groupby("source"):
            n_f = (group[label_col] == "faithful").sum()
            n_o = (group[label_col] == "override").sum()
            t = n_f + n_o
            r = n_o / t if t > 0 else float("nan")
            print(f"    {source}: {r:.3f} ({n_o}/{t}){_n_warning(t)}")

        if "popularity_tier" in analyzable.columns and analyzable["popularity_tier"].notna().any():
            print(f"  By popularity tier (natural RAG sources only, since only PopQA has this field):")
            pop_natural = natural
            for tier, group in pop_natural.groupby("popularity_tier"):
                n_f = (group[label_col] == "faithful").sum()
                n_o = (group[label_col] == "override").sum()
                t = n_f + n_o
                r = n_o / t if t > 0 else float("nan")
                print(f"    {tier}: {r:.3f} ({n_o}/{t}){_n_warning(t)}")


def _pcs_test(faithful_pcs, override_pcs, label=""):
    if len(faithful_pcs) < 2 or len(override_pcs) < 2:
        print(f"    {label}: not enough samples to test "
              f"(faithful n={len(faithful_pcs)}, override n={len(override_pcs)})")
        return
    mean_faithful = faithful_pcs.mean()
    mean_override = override_pcs.mean()
    t_stat, p_value = stats.ttest_ind(override_pcs, faithful_pcs, equal_var=False)
    direction = "override > faithful (confirms original hypothesis)" if mean_override > mean_faithful \
        else "override < faithful (CONTRADICTS original hypothesis)"
    n_flag = _n_warning(min(len(faithful_pcs), len(override_pcs)))
    print(f"    {label}: faithful n={len(faithful_pcs)} mean={mean_faithful:.4f}  |  "
          f"override n={len(override_pcs)} mean={mean_override:.4f}{n_flag}")
    print(f"      t={t_stat:.3f}, p={p_value:.4f} -> {direction} "
          f"{'(p < 0.05)' if p_value < 0.05 else '(NOT significant at this N)'}")


def verify_pcs_hypothesis(df: pd.DataFrame):
    print("\n=== PCS Hypothesis Check: mean PCS(override) vs mean PCS(faithful) ===")
    print("(checked POOLED, then per-source -- pooled alone can hide a pattern that's")
    print(" only true for some sources, or driven entirely by one source)")

    for model in MODELS:
        label_col = f"{model}_label"
        pcs_col = f"pcs_{model}"
        if pcs_col not in df.columns:
            print(f"\n  {model}: no {pcs_col} column found, skipping")
            continue

        print(f"\n  {model.upper()}:")

        print(f"  -- Pooled, natural RAG sources only (NQ/TriviaQA/PopQA) --")
        natural = df[df["source"].isin(NATURAL_RAG_SOURCES)]
        _pcs_test(
            natural.loc[natural[label_col] == "faithful", pcs_col].dropna(),
            natural.loc[natural[label_col] == "override", pcs_col].dropna(),
            label="pooled (natural)",
        )

        print(f"  -- ConFiQA only (structurally different override meaning) --")
        confiqa = df[df["source"].isin(CONFLICT_INJECTED_SOURCES)]
        _pcs_test(
            confiqa.loc[confiqa[label_col] == "faithful", pcs_col].dropna(),
            confiqa.loc[confiqa[label_col] == "override", pcs_col].dropna(),
            label="pooled (confiqa)",
        )

        print(f"  -- Per source dataset (does the effect hold everywhere, or is it")
        print(f"     concentrated in one source?) --")
        for source, group in df.groupby("source"):
            _pcs_test(
                group.loc[group[label_col] == "faithful", pcs_col].dropna(),
                group.loc[group[label_col] == "override", pcs_col].dropna(),
                label=source,
            )

        if "popularity_tier" in df.columns and df["popularity_tier"].notna().any():
            print(f"\n  -- PopQA only, split by popularity tier (is the PopQA reversal")
            print(f"     really about popularity, or just 'being PopQA'?) --")
            popqa = df[(df["source"] == "popqa") & df["popularity_tier"].notna()]
            for tier, group in popqa.groupby("popularity_tier"):
                _pcs_test(
                    group.loc[group[label_col] == "faithful", pcs_col].dropna(),
                    group.loc[group[label_col] == "override", pcs_col].dropna(),
                    label=f"popqa/{tier}",
                )


def main():
    print("Merging T3/T4/T5 outputs...")
    df = build_merged_dataframe()
    print(f"Merged dataframe: {len(df)} rows")

    df = apply_labels(df)

    compute_override_rates(df)
    verify_pcs_hypothesis(df)

    def categorize(s):
        if s in CONFLICT_INJECTED_SOURCES:
            return "confiqa"
        if s == "popqa":
            return "popqa"  # kept separate -- this is the source showing the reversal
        return "natural_rag_other"  # nq, triviaqa

    df["source_category"] = df["source"].apply(categorize)

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    # --- T8 artifact: full unfiltered eval_results.jsonl (ALL rows, incl. retrieval_failed) ---
    eval_cols = ["query_id", "source", "source_category", "question", "gold_answer_text",
                 "popularity_tier", "pcs_gemma", "pcs_llama", "retrieval_success",
                 "retrieved_passages" if "retrieved_passages" in df.columns else None,
                 "gemma_answer", "gemma_em", "gemma_f1", "gemma_label",
                 "gemma_attn_entropy_on_passage",
                 "llama_answer", "llama_em", "llama_f1", "llama_label",
                 "llama_attn_entropy_on_passage"]
    eval_cols = [c for c in eval_cols if c and c in df.columns]
    df[eval_cols].to_json(EVAL_RESULTS_FILE, orient="records", lines=True, force_ascii=False)
    print(f"\nWrote {EVAL_RESULTS_FILE} ({len(df)} rows -- full unfiltered record, T8 artifact)")

    # --- analysis_dataset.jsonl: faithful+override subset only, as before ---
    keep_cols = ["query_id", "source", "source_category", "question", "gold_answer_text",
                 "popularity_tier", "pcs_gemma", "pcs_llama", "retrieval_success",
                 "gemma_answer", "gemma_em", "gemma_f1", "gemma_label",
                 "gemma_attn_entropy_on_passage",
                 "llama_answer", "llama_em", "llama_f1", "llama_label",
                 "llama_attn_entropy_on_passage"]
    keep_cols = [c for c in keep_cols if c in df.columns]

    analysis_subset = df[
        (df["gemma_label"] != "retrieval_failed") | (df["llama_label"] != "retrieval_failed")
    ][keep_cols]

    analysis_subset.to_json(OUTPUT_FILE, orient="records", lines=True, force_ascii=False)
    print(f"Wrote {OUTPUT_FILE} ({len(analysis_subset)} rows -- faithful+override subset)")


if __name__ == "__main__":
    main()