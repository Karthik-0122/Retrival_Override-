"""
T4: BM25 Retrieval.

For each query: search the Lucene index built in T2, keep top-5 passages
+ BM25 scores, and compute GRADED retrieval success (three tiers, not
binary):
  - Verbatim: gold answer string appears in any retrieved passage
  - Soft:     token F1 > 0.5 OR cosine similarity > 0.85 between gold
              answer and passage (via all-MiniLM-L6-v2)
  - None:     neither -- retrieval failed for this query

Output: data/final/retrieval_results.jsonl
  {"query_id": ..., "retrieved_passages": [...], "bm25_scores": [...],
   "retrieval_success": "verbatim" | "soft" | "none"}

Requirements:
  pip install pyserini sentence-transformers
"""

import json
import re
import string
from pathlib import Path

from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

RECORDS_FILE = "data/final/dataset_1500_with_titles_1.jsonl"
INDEX_DIR = "data/artifacts/lucene_index"
METADATA_FILE = "data/artifacts/passage_metadata.json"
OUTPUT_FILE = "data/final/retrieval_results.jsonl"
TOP_K = 5
SOFT_F1_THRESHOLD = 0.5
SOFT_COSINE_THRESHOLD = 0.85


def normalize_text(s: str) -> str:
    """Standard EM/F1 normalization: lowercase, strip articles/punctuation."""
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


def get_gold_answer_text(record) -> str:
    ga = record.get("gold_answers")
    if isinstance(ga, list):
        return str(ga[0]) if ga else ""
    if isinstance(ga, dict):
        v = ga.get("value") or ga.get("aliases")
        if isinstance(v, list):
            return str(v[0]) if v else ""
        return str(v) if v else ""
    return str(ga) if ga else ""


def load_records():
    records = []
    with open(RECORDS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def compute_retrieval_success(gold_answer: str, passages: list, embedder) -> str:
    gold_norm = normalize_text(gold_answer)
    if not gold_norm:
        return "none"

    # Tier 1: Verbatim -- exact substring match, normalized
    for p in passages:
        if gold_norm in normalize_text(p):
            return "verbatim"

    # Tier 2: Soft -- token F1 or embedding cosine similarity
    gold_emb = embedder.encode(gold_answer, convert_to_tensor=True)
    for p in passages:
        f1 = token_f1(p, gold_answer)
        if f1 > SOFT_F1_THRESHOLD:
            return "soft"
        p_emb = embedder.encode(p[:500], convert_to_tensor=True)  # truncate for speed
        cosine = util.cos_sim(gold_emb, p_emb).item()
        if cosine > SOFT_COSINE_THRESHOLD:
            return "soft"

    return "none"


def main():
    print(f"Loading Lucene index from {INDEX_DIR}...")
    searcher = LuceneSearcher(INDEX_DIR)

    print(f"Loading passage metadata from {METADATA_FILE}...")
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        passage_metadata = json.load(f)

    print("Loading embedder for soft-match tier (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device = 'CUDA')
    print(f"Embedder device: {embedder.device}")

    records = load_records()
    print(f"Loaded {len(records)} records")

    results = []
    tier_counts = {"verbatim": 0, "soft": 0, "none": 0}

    for r in tqdm(records, desc="Retrieving"):
        hits = searcher.search(r["question"], k=TOP_K)

        passage_texts = []
        bm25_scores = []
        for hit in hits:
            meta = passage_metadata.get(hit.docid)
            text = meta["text"] if meta else ""
            passage_texts.append(text)
            bm25_scores.append(hit.score)

        gold_answer = get_gold_answer_text(r)
        success = compute_retrieval_success(gold_answer, passage_texts, embedder)
        tier_counts[success] += 1

        results.append({
            "query_id": r["query_id"],
            "retrieved_passages": passage_texts,
            "retrieved_passage_ids": [hit.docid for hit in hits],
            "bm25_scores": bm25_scores,
            "retrieval_success": success,
        })

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {OUTPUT_FILE}")
    print(f"Retrieval success breakdown:")
    for tier, count in tier_counts.items():
        pct = 100 * count / len(records)
        print(f"  {tier}: {count}/{len(records)} ({pct:.1f}%)")


if __name__ == "__main__":
    main()