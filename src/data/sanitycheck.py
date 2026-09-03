"""
Diagnostic for T4's soft-tier threshold calibration.

Re-computes F1 and cosine similarity (best passage-alias pair per query)
for every query currently labeled "none", and prints the distribution --
so you can pick SOFT_F1_THRESHOLD / SOFT_COSINE_THRESHOLD based on where
the real scores actually cluster, instead of guessing twice.

Does NOT re-run full retrieval -- reuses retrieval_results.jsonl's
already-retrieved passages, just recomputes the match scores.
"""

import json
from sentence_transformers import SentenceTransformer, util
import re
import string

RECORDS_FILE = "data/final/dataset_1500_with_titles_1.jsonl"
RETRIEVAL_FILE = "data/final/retrieval_results.jsonl"


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


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
    ga = record.get("gold_answers")
    if isinstance(ga, dict):
        aliases = ga.get("aliases") or []
        value = ga.get("value")
        all_answers = ([value] if value else []) + list(aliases)
        return [a for a in all_answers if a]
    if isinstance(ga, list):
        return [str(a) for a in ga if a]
    return [str(ga)] if ga else []


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    records = {r["query_id"]: r for r in load_jsonl(RECORDS_FILE)}
    retrieval = load_jsonl(RETRIEVAL_FILE)

    print("Loading embedder...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    none_tier = [r for r in retrieval if r["retrieval_success"] == "none"]
    print(f"Analyzing {len(none_tier)} 'none'-tier queries...")

    best_f1s = []
    best_cosines = []

    for r in none_tier[:300]:  # sample for speed -- full set if you want exact
        record = records.get(r["query_id"])
        if not record:
            continue
        aliases = get_gold_answer_aliases(record)
        if not aliases:
            continue

        alias_embs = [embedder.encode(a, convert_to_tensor=True) for a in aliases]

        best_f1 = 0.0
        best_cos = 0.0
        for p in r["retrieved_passages"]:
            if not p:
                continue
            p_emb = embedder.encode(p[:500], convert_to_tensor=True)
            for alias, alias_emb in zip(aliases, alias_embs):
                f1 = token_f1(p, alias)
                best_f1 = max(best_f1, f1)
                cos = util.cos_sim(alias_emb, p_emb).item()
                best_cos = max(best_cos, cos)

        best_f1s.append(best_f1)
        best_cosines.append(best_cos)

    best_f1s.sort()
    best_cosines.sort()
    n = len(best_f1s)

    def percentile(lst, p):
        idx = int(len(lst) * p)
        return lst[min(idx, len(lst) - 1)]

    print(f"\n--- Best F1 per query (n={n}) ---")
    print(f"  min={best_f1s[0]:.3f}  p25={percentile(best_f1s,0.25):.3f}  "
          f"median={percentile(best_f1s,0.5):.3f}  p75={percentile(best_f1s,0.75):.3f}  "
          f"p90={percentile(best_f1s,0.9):.3f}  max={best_f1s[-1]:.3f}")

    print(f"\n--- Best cosine similarity per query (n={n}) ---")
    print(f"  min={best_cosines[0]:.3f}  p25={percentile(best_cosines,0.25):.3f}  "
          f"median={percentile(best_cosines,0.5):.3f}  p75={percentile(best_cosines,0.75):.3f}  "
          f"p90={percentile(best_cosines,0.9):.3f}  max={best_cosines[-1]:.3f}")

    print(f"\nHow many would flip to 'soft' at various thresholds:")
    for f1_t in [0.2, 0.3, 0.4, 0.5]:
        count = sum(1 for f1 in best_f1s if f1 > f1_t)
        print(f"  F1 > {f1_t}: {count}/{n} ({100*count/n:.1f}%)")
    for cos_t in [0.5, 0.6, 0.7, 0.8, 0.85]:
        count = sum(1 for c in best_cosines if c > cos_t)
        print(f"  cosine > {cos_t}: {count}/{n} ({100*count/n:.1f}%)")


if __name__ == "__main__":
    main()