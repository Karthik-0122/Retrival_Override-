"""
T2: Builds the controlled Wikipedia corpus and indexes it with Lucene
(via Pyserini) instead of rank_bm25 — needed once the corpus goes past a
few tens of thousands of passages, since BM25Okapi is pure Python and
doesn't scale to this size or query volume.

Passage pool = three sources:
  1. Gold Wikipedia pages for NQ/TriviaQA/PopQA rows (via gold_title from
     recover_titles.py), pulled from psgs_w100.
  2. ConFiQA's own counterfactual_context passages (from context_modified) —
     not from Wikipedia, added directly, keyed by query_id.
  3. A stratified random sample from psgs_w100 as the distractor pool.

Two-stage process (Pyserini requires this):
  Stage A: write all passages to a JSONL "collection" file in Pyserini's
           required format: {"id": ..., "contents": ...}
  Stage B: shell out to `python -m pyserini.index.lucene` to build the
           actual Lucene index from that collection.

Output:
  data/artifacts/collection/passages.jsonl   # Pyserini input format
  data/artifacts/lucene_index/               # built Lucene index dir
  data/artifacts/passage_metadata.json       # id -> {title, source, type, query_id}

Requirements:
  pip install pyserini datasets tqdm
  Java 21 (JDK) installed and on PATH.
"""

import gzip
import json
import os
import random
import subprocess
import sys
import unicodedata
import urllib.request
from pathlib import Path

from tqdm import tqdm

TITLES_FILE = "data/final/dataset_1500_with_titles_1.jsonl"
OUTPUT_DIR = Path("data/artifacts")
COLLECTION_DIR = OUTPUT_DIR / "collection"
INDEX_DIR = OUTPUT_DIR / "lucene_index"
PASSAGES_URL = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
PASSAGES_LOCAL_GZ = Path("data/raw/psgs_w100.tsv.gz")  # ~13GB download, cached after first run
DISTRACTOR_SAMPLE_SIZE = 300_000  # tune to your disk/time budget; test small first
MAX_SCAN = None  # None = full scan (~21M lines). Set to an int (e.g. 200_000) for a
                  # quick dry run only — with a cap set, DISTRACTOR_SAMPLE_SIZE should
                  # be well under the cap or the reservoir just takes everything scanned.
RANDOM_SEED = 42
INDEX_THREADS = 4  # adjust to your CPU core count

random.seed(RANDOM_SEED)


def normalize_title(t: str) -> str:
    """Case/whitespace/unicode/punctuation-insensitive title key. Catches
    formatting drift (casing, a missing '!', underscores vs spaces, accented
    chars) but NOT genuine Wikipedia renames (e.g. 'United Kingdom general
    election, 2010' -> '2010 United Kingdom general election') — those are
    real retitles between snapshots, not formatting differences, and need
    redirect resolution rather than string normalization."""
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("_", " ").strip().lower()
    t = "".join(ch for ch in t if ch.isalnum() or ch.isspace())
    return " ".join(t.split())


def download_passages_file(url: str, local_path: Path):
    if local_path.exists():
        print(f"Using cached download at {local_path} "
              f"({local_path.stat().st_size / 1e9:.1f} GB)")
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {local_path} (this is ~13GB compressed, "
          f"one-time — cached for future runs)...")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(local_path, "wb") as out_f:
        total = int(resp.headers.get("Content-Length", 0))
        chunk_size = 1024 * 1024
        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                pbar.update(len(chunk))
    print("Download complete.")


def load_records():
    records = []
    print(f"DEBUG: Opening titles file at path -> {TITLES_FILE}")
    with open(TITLES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"DEBUG: Total rows loaded from file: {len(records)}")
    return records


def collect_gold_titles(records):
    titles = set()
    for r in records:
        # falls back to 'title' if 'gold_title' is empty/missing
        t = r.get("gold_title") or r.get("title")
        if t:
            titles.add(t)
    return titles


def build_gold_and_distractor_passages(gold_titles, sample_size, max_scan=None):
    """Single streaming pass over the official DPR psgs_w100.tsv.gz file
    (columns: id, text, title — tab-separated, header row first): keeps
    gold-title matches (via normalized comparison), reservoir-samples the
    rest as distractors.
    """
    download_passages_file(PASSAGES_URL, PASSAGES_LOCAL_GZ)

    print(f"Reading psgs_w100.tsv.gz (target: {len(gold_titles)} gold titles "
          f"+ {sample_size} distractors"
          f"{f', capped at {max_scan} lines' if max_scan else ', full scan'})...")

    # normalized lookup: normalized_title -> original gold_title
    norm_to_gold = {normalize_title(t): t for t in gold_titles}

    gold_passages = []
    reservoir = []
    seen_titles_hit = set()
    n_seen = 0

    with gzip.open(PASSAGES_LOCAL_GZ, "rt", encoding="utf-8") as f:
        header = f.readline()  # "id\ttext\ttitle"
        for line in tqdm(f, desc="Reading passages", total=max_scan):
            if max_scan and n_seen >= max_scan:
                print(f"\nStopped at max_scan={max_scan} (not a full pass — "
                      f"gold title coverage and distractor sample may be incomplete).")
                break
            n_seen += 1

            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue  # malformed line, skip
            pid, text, title = parts
            text = text.strip('"')  # DPR tsv wraps text in quotes

            matched_gold = norm_to_gold.get(normalize_title(title))

            if matched_gold is not None:
                gold_passages.append({
                    "passage_id": f"gold_{pid}",
                    "title": title,
                    "matched_gold_title": matched_gold,
                    "text": text,
                    "source": "wikipedia",
                    "type": "gold",
                })
                seen_titles_hit.add(matched_gold)
            else:
                if len(reservoir) < sample_size:
                    reservoir.append({
                        "passage_id": f"distractor_{pid}",
                        "title": title,
                        "text": text,
                        "source": "wikipedia",
                        "type": "distractor",
                    })
                else:
                    j = random.randint(0, n_seen - 1)
                    if j < sample_size:
                        reservoir[j] = {
                            "passage_id": f"distractor_{pid}",
                            "title": title,
                            "text": text,
                            "source": "wikipedia",
                            "type": "distractor",
                        }

    missed = gold_titles - seen_titles_hit
    print(f"\nTotal lines scanned: {n_seen}")
    print(f"Gold titles found: {len(seen_titles_hit)}/{len(gold_titles)}")
    if missed:
        print(f"  WARNING: {len(missed)} gold titles not found even after "
              f"normalization (likely genuine Wikipedia renames/redirects "
              f"between snapshots) — examples:")
        for t in list(missed)[:5]:
            print(f"    - {t!r}")

    print(f"Distractors sampled: {len(reservoir)}")
    return gold_passages, reservoir


def build_confiqa_passages(records):
    passages = []
    for r in records:
        if r["source"] == "confiqa" and r.get("counterfactual_context"):
            passages.append({
                "passage_id": f"confiqa_{r['query_id']}",
                "title": None,
                "text": r["counterfactual_context"],
                "source": "confiqa",
                "type": "counterfactual",
                "query_id": r["query_id"],
            })
    print(f"ConFiQA counterfactual passages added: {len(passages)}")
    return passages


def write_pyserini_collection(all_passages, collection_dir: Path):
    """Pyserini's JSONL format requires exactly {"id": ..., "contents": ...}
    per line. Extra metadata is written to a separate sidecar file."""
    collection_dir.mkdir(parents=True, exist_ok=True)
    collection_file = collection_dir / "passages.jsonl"

    print(f"Writing Pyserini collection to {collection_file}...")
    with open(collection_file, "w", encoding="utf-8") as f:
        for p in tqdm(all_passages, desc="Writing collection"):
            f.write(json.dumps({"id": p["passage_id"], "contents": p["text"]}) + "\n")

    return collection_file


def run_pyserini_indexing(collection_dir: Path, index_dir: Path, threads: int):
    index_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # Java 21+ memory-segment MMapDirectory support triggers a LinkageError
    # with the Lucene version Pyserini bundles -- disable it.
    env["JAVA_TOOL_OPTIONS"] = "-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false"

    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(collection_dir),
        "--index", str(index_dir),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
        "--storePositions",
    ]
    print(f"\nRunning Pyserini indexing:\n  {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    print(result.stdout[-3000:])
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise RuntimeError(
            "Pyserini indexing failed — check Java is installed (`java -version`) "
            "and `pip install pyserini` succeeded."
        )
    print(f"Lucene index built at {index_dir}")


def main():
    records = load_records()
    gold_titles = collect_gold_titles(records)
    print(f"Unique gold titles to pull: {len(gold_titles)}")

    gold_passages, distractor_passages = build_gold_and_distractor_passages(
        gold_titles, DISTRACTOR_SAMPLE_SIZE, max_scan=MAX_SCAN
    )
    confiqa_passages = build_confiqa_passages(records)

    all_passages = gold_passages + distractor_passages + confiqa_passages
    print(f"\nTotal passages: {len(all_passages)} "
          f"(gold={len(gold_passages)}, distractor={len(distractor_passages)}, "
          f"confiqa={len(confiqa_passages)})")

    write_pyserini_collection(all_passages, COLLECTION_DIR)

    metadata_path = OUTPUT_DIR / "passage_metadata.json"
    metadata = {p["passage_id"]: p for p in all_passages}
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    print(f"Wrote passage metadata sidecar to {metadata_path}")

    run_pyserini_indexing(COLLECTION_DIR, INDEX_DIR, INDEX_THREADS)

    print("\nT2 corpus + Lucene index build complete.")
    print(f"  Retrieve later with: from pyserini.search.lucene import LuceneSearcher; "
          f"searcher = LuceneSearcher('{INDEX_DIR}')")


if __name__ == "__main__":
    main()