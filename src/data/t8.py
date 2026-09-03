"""
T8: Artifacts and Checkpoints.

Verifies every artifact from the original plan exists, reports size/row
counts, and writes a manifest for the paper's reproducibility section.

Per your original T8 spec:
  01. bm25_index.pkl + passage_map.json  -- serialized index, reused in Phase 3
      (actual: lucene_index/ + passage_metadata.json, since we switched
      from rank_bm25 to Pyserini/Lucene for scale -- see T2 discussion)
  02. eval_results.jsonl -- full per-query record (all fields including
      PCS, label, passages, entropy)
  03. analysis_dataset.jsonl -- faithful + override subset only, used
      for Phase 3 circuit analysis
"""

import json
from pathlib import Path

ARTIFACTS = [
    {
        "name": "Lucene index (bm25_index.pkl equivalent)",
        "path": "data/artifacts/lucene_index",
        "type": "directory",
        "note": "Pyserini/Lucene index -- replaces rank_bm25's pickle for "
                "scale (see T2). Reload with: "
                "LuceneSearcher('data/artifacts/lucene_index')",
    },
    {
        "name": "Passage metadata (passage_map.json equivalent)",
        "path": "data/artifacts/passage_metadata.json",
        "type": "json",
        "note": "docid -> {text, title, source, type[, query_id]}",
    },
    {
        "name": "Wikipedia collection (raw passages, Pyserini input format)",
        "path": "data/artifacts/collection/passages.jsonl",
        "type": "jsonl",
        "note": "id + contents per line -- input to the Lucene index build",
    },
    {
        "name": "PCS scores (T3)",
        "path": "data/final/pcs_scores.jsonl",
        "type": "jsonl",
        "note": "query_id, pcs_gemma, pcs_llama",
    },
    {
        "name": "Retrieval results (T4)",
        "path": "data/final/retrieval_results.jsonl",
        "type": "jsonl",
        "note": "query_id, retrieved_passages, bm25_scores, retrieval_success",
    },
    {
        "name": "Generation results (T5)",
        "path": "data/final/generation_results.jsonl",
        "type": "jsonl",
        "note": "query_id, {model}_answer, {model}_attn_entropy_on_passage",
    },
    {
        "name": "Full eval results (T6, unfiltered)",
        "path": "data/final/eval_results.jsonl",
        "type": "jsonl",
        "note": "ALL rows including retrieval_failed -- per your T8 spec 02",
    },
    {
        "name": "Analysis dataset (T6, faithful+override subset)",
        "path": "data/final/analysis_dataset.jsonl",
        "type": "jsonl",
        "note": "Used for Phase 3 circuit analysis, per your T8 spec 03",
    },
    {
        "name": "Primary results table (T7)",
        "path": "data/final/primary_results_table.csv",
        "type": "csv",
        "note": "EM, override rate, mean PCS -- per model x source_category",
    },
    {
        "name": "Spearman correlation results (T7)",
        "path": "data/final/spearman_results.json",
        "type": "json",
        "note": "PCS vs override probability, pooled + per source_category",
    },
    {
        "name": "PCS distribution figure (T7)",
        "path": "data/final/figures/pcs_distribution_by_category.png",
        "type": "png",
        "note": "Paper figure",
    },
    {
        "name": "Override rate vs popularity figure (T7, EXPLORATORY)",
        "path": "data/final/figures/override_rate_vs_popularity_EXPLORATORY.png",
        "type": "png",
        "note": "Underpowered -- caption as exploratory in the paper, not a confirmed trend",
    },
    {
        "name": "Attention entropy comparison figure (T7)",
        "path": "data/final/figures/attention_entropy_comparison.png",
        "type": "png",
        "note": "Paper figure",
    },
]


def human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"


def count_jsonl_rows(path: Path) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return -1


def main():
    manifest = []
    print("=" * 70)
    print("T8: Artifact Manifest")
    print("=" * 70)

    all_present = True

    for artifact in ARTIFACTS:
        path = Path(artifact["path"])
        entry = {**artifact, "exists": path.exists()}

        if path.exists():
            if artifact["type"] == "directory":
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                entry["size"] = human_size(size)
                entry["file_count"] = sum(1 for _ in path.rglob("*") if _.is_file())
                status = f"OK  ({entry['size']}, {entry['file_count']} files)"
            elif artifact["type"] == "jsonl":
                size = path.stat().st_size
                rows = count_jsonl_rows(path)
                entry["size"] = human_size(size)
                entry["row_count"] = rows
                status = f"OK  ({entry['size']}, {rows} rows)"
            else:
                size = path.stat().st_size
                entry["size"] = human_size(size)
                status = f"OK  ({entry['size']})"
        else:
            all_present = False
            status = "MISSING"

        print(f"\n[{status}] {artifact['name']}")
        print(f"  path: {artifact['path']}")
        print(f"  note: {artifact['note']}")

        manifest.append(entry)

    manifest_path = Path("data/final/t8_artifact_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print("\n" + "=" * 70)
    if all_present:
        print("All T8 artifacts present. Wrote manifest to", manifest_path)
        print("T1-T8 pipeline complete.")
    else:
        print("Some artifacts are MISSING -- check the list above before "
              "considering T1-T8 complete.")
        print("Wrote partial manifest to", manifest_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
    