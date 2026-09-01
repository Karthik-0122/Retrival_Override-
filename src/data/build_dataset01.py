from datasets import load_dataset
import random
import json
from pathlib import Path

def load_nq():

    print("=" * 60)
    print("Loading Natural Questions...")
    print("=" * 60)

    dataset = load_dataset(
        "google-research-datasets/nq_open",
        split="validation"
    )

    print(f"Dataset loaded successfully!")
    print(f"Total samples: {len(dataset)}")

    print("\nOriginal sample:\n")
    print(dataset[0])

    return dataset

def standardize_nq(dataset):

    record=[]

    for idx, sample in enumerate(dataset):

        records={
            "query_id": f"nq_{idx:05d}",
            "source": "nq",
            "question": sample["question"],
            "gold_answers": sample["answer"],
            "popularity_tier": None,
        }

        record.append(records)

        print(f"\nConverted {len(record)} record.")

    return record


def sample_records(record, sample_size= 400, seed=42):

    random.seed(seed)

    sampled_records= random.sample(record, sample_size)

    print("\n" + "=" * 60)
    print("Sampling Records")
    print("=" * 60)
    print(f"Original Records : {len(record)}")
    print(f"Sample Size      : {sample_size}")
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
    print(f"Output File : {output_path}")
    print(f"Total Records : {len(records)}")

def main():

    dataset = load_nq()
    records = standardize_nq(dataset)
    sampled_records = sample_records(records)
    save_jsonl(sampled_records, "data/processed/nq.jsonl1")

if __name__ == "__main__":
    main()
  
