from datasets import load_dataset
import random
import json
from pathlib import Path

def load_data(dataset_path, split="validation", config_name=None):
    print("=" * 60)
    print(f"Loading Dataset: {dataset_path} [{split}]")
    print("=" * 60)

    # config_name is useful for datasets with multiple subsets (e.g., "en" for wikipedia)
    dataset = load_dataset(
        dataset_path,
        name=config_name,
        split=split
    )

    print(f"Dataset loaded successfully!")
    print(f"Total samples: {len(dataset)}")
    print("\nOriginal sample:\n")
    print(dataset[0])

    return dataset

def standardize_data(dataset, source_name, column_mapping):
    """
    Standardizes the dataset format.
    
    :param dataset: The HuggingFace dataset object
    :param source_name: A short prefix for the dataset (e.g., 'nq', 'triviaqa')
    :param column_mapping: Dict mapping standard keys to the dataset's actual column names.
                           Example: {"question": "question_text", "answer": "answers"}
    """
    records = []
    
 
    q_col = column_mapping.get("question", "question")
    a_col = column_mapping.get("answer", "answer")

    for idx, sample in enumerate(dataset):
        record = {
            "query_id": f"{source_name}_{idx:05d}",
            "source": source_name,
            "question": sample[q_col],
            "gold_answers": sample[a_col],
            "popularity_tier": None,
        }
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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    print("\n" + "=" * 60)
    print("Dataset Saved")
    print("=" * 60)
    print(f"Output File   : {output_path}")
    print(f"Total Records : {len(records)}")


def process_dataset(config):
    """Orchestrator function that takes a configuration dictionary."""
    dataset = load_data(
        dataset_path=config["dataset_path"], 
        split=config["split"],
        config_name=config.get("config_name")
    )
    
    records = standardize_data(
        dataset=dataset, 
        source_name=config["source_name"],
        column_mapping=config["column_mapping"]
    )
    
    sampled_records = sample_records(
        records=records, 
        sample_size=config["sample_size"]
    )
    
    save_jsonl(
        records=sampled_records, 
        output_path=config["output_path"]
    )


def main():
    
    # nq_config = {
    #     "dataset_path": "google-research-datasets/nq_open",
    #     "split": "validation",
    #     "source_name": "nq",
    #     "column_mapping": {
    #         "question": "question", 
    #         "answer": "answer"
    #     },
    #     "sample_size": 400,
    #     "output_path": "data/processed/nq_sample.jsonl"
    # }


    # triviaqa_config = {
    #     "dataset_path": "mandarjoshi/trivia_qa",
    #     "config_name": "rc.nocontext",
    #     "split": "validation",
    #     "source_name": "triviaqa",
    #     "column_mapping": {
    #         "question": "question", 
    #         "answer": "answer" 
    #     },
    #     "sample_size": 300,
    #     "output_path": "data/processed/triviaqa_sample.jsonl"
    # }

    # print("Running trivia question...")
    # process_dataset(triviaqa_config)
    
    # # PopQA Configuration
    # popqa_config = {
    #     "dataset_path": "akariasai/PopQA",
    #     "split": "test", 
    #     "source_name": "popqa",
    #     "column_mapping": {
    #         "question": "question", 
    #         "answer": "possible_answers" 
    #     },
    #     "sample_size": 400,
    #     "output_path": "data/processed/popqa_sample.jsonl"
    # }
    
    # print("Running PopQA Pipeline...")
    # process_dataset(popqa_config)
    
    
    # ConFiQA Configuration
    confiqa_config = {
        "dataset_path": "RajMaheshwari/ConFiQA",
        "config_name": "QA",         # Subsets available: 'QA' (single-hop), 'MR' (multi-hop), or 'MC' (multi-conflict)
        "split": "test",             # ConFiQA provides 'train' and 'test' splits
        "source_name": "confiqa",
        "column_mapping": {
            "question": "question", 
            "answer": "answer_modified" # Use 'answer_modified' for the context-faithful answer, or 'answer_original' for the parametric answer
        },
        "sample_size": 400,
        "output_path": "data/processed/confiqa_sample.jsonl"
    }
    
    print("Running ConFiQA Pipeline...")
    process_dataset(confiqa_config)

if __name__ == "__main__":
    main()