import json
import random
from pathlib import Path

def merge_datasets(input_dir, output_file, shuffle=True, seed=42):
    """
    Combines all .jsonl files in input_dir into a single JSONL file.
    """
    input_path = Path(input_dir)
    merged_records = []
    
    # Read all processed datasets
    for file_path in input_path.glob("*.jsonl"):
        print(f"Reading: {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    merged_records.append(json.loads(line))
                    
    print(f"\nTotal combined records: {len(merged_records)}")
    
    # Shuffle combined dataset
    if shuffle:
        random.seed(seed)
        random.shuffle(merged_records)
        print(f"Dataset shuffled (Seed: {seed})")
        
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save final file
    with open(output_path, "w", encoding="utf-8") as f:
        for record in merged_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"Merged dataset saved to: {output_path}")

if __name__ == "__main__":
    merge_datasets(
        input_dir="data/processed/", 
        output_file="data/final/merged_circuit_analysis_data.jsonl",
        shuffle=True
    )