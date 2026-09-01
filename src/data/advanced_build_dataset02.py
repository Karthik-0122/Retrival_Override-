from datasets import load_dataset
import random
import json
from pathlib import Path


def load_data(dataset_path, split="validation", config_name=None):
    print("=" * 60)
    print(f"Loading Dataset: {dataset_path} [{split}]")
    print("=" * 60)

    dataset = load_dataset(dataset_path, name=config_name, split=split)

    print(f"Dataset loaded successfully!")
    print(f"Total samples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    print("\nOriginal sample:\n")
    print(dataset[0])

    return dataset


def find_column(dataset, candidates, required=False, label=""):
    """Returns the first candidate name that's actually a column in this
    dataset, so we don't hardcode a guess. Prints what it found (or didn't)
    so you can verify against the printed column list from load_data."""
    for c in candidates:
        if c in dataset.column_names:
            print(f"  [{label}] matched column: '{c}'")
            return c
    msg = f"  [{label}] WARNING: none of {candidates} found in {dataset.column_names}"
    if required:
        raise ValueError(msg)
    print(msg)
    return None


def standardize_data(dataset, source_name, column_mapping, extra_column_candidates=None):
    """
    Standardizes the dataset format.

    :param column_mapping: dict mapping standard keys to actual column names
        for question/answer (required, as before).
    :param extra_column_candidates: optional dict of
        {standard_key: [candidate_col_name_1, candidate_col_name_2, ...]}
        for fields that vary by dataset (e.g. ConFiQA's counterfactual
        context / original answer). Only added to the record if a match
        is found — missing fields are set to None, not dropped silently.
    """
    records = []

    q_col = column_mapping.get("question", "question")
    a_col = column_mapping.get("answer", "answer")

    resolved_extra_cols = {}
    if extra_column_candidates:
        for std_key, candidates in extra_column_candidates.items():
            resolved_extra_cols[std_key] = find_column(dataset, candidates, label=std_key)

    for idx, sample in enumerate(dataset):
        record = {
            "query_id": f"{source_name}_{idx:05d}",
            "source": source_name,
            "question": sample[q_col],
            "gold_answers": sample[a_col],
            "popularity_tier": None,
        }
        for std_key, actual_col in resolved_extra_cols.items():
            record[std_key] = sample[actual_col] if actual_col else None
        records.append(record)

    print(f"Successfully standardized {len(records)} records from {source_name}.")
    return records


def sample_records(records, sample_size=400, seed=42):
    random.seed(seed)
    actual_sample_size = min(sample_size, len(records))
    sampled_records = random.sample(records, actual_sample_size)

    print("\n" + "=" * 60)
    print("Sampling Records")
    print("=" * 60)
    print(f"Original Records : {len(records)}")
    print(f"Sample Size      : {actual_sample_size}")
    print(f"Random Seed      : {seed}")

    return sampled_records


def save_jsonl(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("Dataset Saved")
    print("=" * 60)
    print(f"Output File   : {output_path}")
    print(f"Total Records : {len(records)}")


def process_dataset(config):
    dataset = load_data(
        dataset_path=config["dataset_path"],
        split=config["split"],
        config_name=config.get("config_name"),
    )

    records = standardize_data(
        dataset=dataset,
        source_name=config["source_name"],
        column_mapping=config["column_mapping"],
        extra_column_candidates=config.get("extra_column_candidates"),
    )

    sampled_records = sample_records(records=records, sample_size=config["sample_size"])
    save_jsonl(records=sampled_records, output_path=config["output_path"])


def main():
    nq_config = {
        "dataset_path": "google-research-datasets/nq_open",
        "split": "validation",
        "source_name": "nq",
        "column_mapping": {"question": "question", "answer": "answer"},
        "sample_size": 400,
        "output_path": "data/processed/nq_sample.jsonl",
    }

    triviaqa_config = {
        "dataset_path": "mandarjoshi/trivia_qa",
        "config_name": "rc.nocontext",
        "split": "validation",
        "source_name": "triviaqa",
        "column_mapping": {"question": "question", "answer": "answer"},
        "sample_size": 300,
        "output_path": "data/processed/triviaqa_sample.jsonl",
    }

    popqa_config = {
        "dataset_path": "akariasai/PopQA",
        "split": "test",
        "source_name": "popqa",
        "column_mapping": {"question": "question", "answer": "possible_answers"},
        "sample_size": 400,
        "output_path": "data/processed/popqa_sample.jsonl",
    }

    # ConFiQA — the previous version silently dropped the counterfactual
    # context and the original (parametric) answer. Both are captured now.
    # Candidate column names are guesses based on the ConFiQA/Context-DPO
    # paper's terminology (Bi et al. 2024) — CHECK the printed "Columns:"
    # and "Original sample:" output against these before trusting the run.
    confiqa_config = {
        "dataset_path": "RajMaheshwari/ConFiQA",
        "config_name": "QA",
        "split": "test",
        "source_name": "confiqa",
        "column_mapping": {"question": "question", "answer": "answer_modified"},
        "extra_column_candidates": {
            # Confirmed schema (RajMaheshwari/ConFiQA, QA config) as of your
            # inspect_confiqa_schema.py output:
            #   question, context_original, context_modified, answer_original,
            #   answer_modified, truth, modified_aliases, context_piece_original,
            #   context_piece_modified, path_original, path_modified,
            #   path_labeled_original, path_labeled_modified, triple_original,
            #   triple_modified
            "counterfactual_context": ["context_modified"],
            "answer_original": ["answer_original"],
            "original_context": ["context_original"],
            "counterfactual_context_piece": ["context_piece_modified"],
            "original_context_piece": ["context_piece_original"],
            "answer_aliases": ["modified_aliases"],
        },
        "sample_size": 400,
        "output_path": "data/processed/confiqa_sample.jsonl",
    }

    print("Running NQ Pipeline...")
    process_dataset(nq_config)

    print("Running TriviaQA Pipeline...")
    process_dataset(triviaqa_config)

    print("Running PopQA Pipeline...")
    process_dataset(popqa_config)

    print("Running ConFiQA Pipeline...")
    process_dataset(confiqa_config)


if __name__ == "__main__":
    main()