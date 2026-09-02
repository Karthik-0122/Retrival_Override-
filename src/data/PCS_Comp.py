"""
T3: Parametric Confidence Score (PCS) computation.

For each query, run each model with NO retrieved context — prompt:
  "Question: {q}\nAnswer:"
PCS = mean log-probability of the gold answer tokens (teacher-forced),
normalized by answer length. This must run BEFORE retrieval (T4) so PCS
is uncontaminated by passage context — it's your uncontaminated parametric
memory signal.

Design for a single rented GPU (RunPod/Vast.ai, 24GB class card):
  - 4-bit quantized (bitsandbytes) — both models fit comfortably at 4-bit,
    but are run SEQUENTIALLY (load, process all 1500 queries, unload,
    load the next) rather than concurrently, to keep VRAM headroom safe
    regardless of card size.
  - Batches by padding to the same prompt length isn't done here (answer
    lengths vary token-by-token log-prob extraction is per-example) —
    this is correctness-first, not throughput-optimized. If it's too
    slow, batching is the first thing to add.

Output: data/final/pcs_scores.jsonl
  {"query_id": ..., "pcs_gemma": float, "pcs_llama": float}
  Merge this back into your main record file by query_id before T4/T6.

Requirements:
  pip install transformers accelerate bitsandbytes torch tqdm
  HF_TOKEN env var set if these are gated model repos on your account.
"""

import json
import gc
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

RECORDS_FILE = "data/final/dataset_1500_with_titles_1.jsonl"
OUTPUT_FILE = "data/final/pcs_scores.jsonl"

MODEL_CONFIGS = {
    "gemma": "google/gemma-2-9b",
    "llama": "meta-llama/Llama-3.1-8B",
}

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def load_records():
    records = []
    with open(RECORDS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def get_gold_answer_text(record) -> str:
    """Same answer field used throughout the pipeline (T6 grading target),
    so PCS measures confidence on the exact string you'll later check
    correctness against."""
    ga = record.get("gold_answers")
    if isinstance(ga, list):
        return str(ga[0]) if ga else ""
    if isinstance(ga, dict):
        v = ga.get("value") or ga.get("aliases")
        if isinstance(v, list):
            return str(v[0]) if v else ""
        return str(v) if v else ""
    return str(ga) if ga else ""


@torch.no_grad()
def compute_pcs(model, tokenizer, question: str, answer: str, device: str) -> float:
    """Mean log-probability of answer tokens given the no-context prompt,
    via teacher forcing: feed prompt+answer, read off the model's log-probs
    at the positions predicting each answer token."""
    if not answer.strip():
        return None

    prompt = f"Question: {question}\nAnswer:"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    # leading space before the answer, matching how it'd naturally follow "Answer:"
    answer_ids = tokenizer(" " + answer, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    if answer_ids.shape[1] == 0:
        return None

    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    outputs = model(full_ids)
    logits = outputs.logits  # [1, seq_len, vocab]

    # logits at position i predict token i+1 -- we want the logits that
    # predict each answer token, i.e. positions (prompt_len-1) .. (full_len-2)
    prompt_len = prompt_ids.shape[1]
    answer_len = answer_ids.shape[1]

    relevant_logits = logits[0, prompt_len - 1: prompt_len - 1 + answer_len, :]
    log_probs = torch.log_softmax(relevant_logits, dim=-1)

    target_tokens = answer_ids[0]
    token_log_probs = log_probs[torch.arange(answer_len), target_tokens]

    return token_log_probs.mean().item()


def run_model_pass(model_key: str, model_path: str, records: list, device: str) -> dict:
    print(f"\nLoading {model_key} ({model_path})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=QUANT_CONFIG, device_map=device
    )
    model.eval()

    results = {}
    for r in tqdm(records, desc=f"PCS ({model_key})"):
        answer = get_gold_answer_text(r)
        pcs = compute_pcs(model, tokenizer, r["question"], answer, device)
        results[r["query_id"]] = pcs

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Done with {model_key}, VRAM freed.")
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected — this will be extremely slow for 9B/8B models.")

    records = load_records()
    print(f"Loaded {len(records)} records")

    pcs_gemma = run_model_pass("gemma", MODEL_CONFIGS["gemma"], records, device)
    pcs_llama = run_model_pass("llama", MODEL_CONFIGS["llama"], records, device)

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            qid = r["query_id"]
            f.write(json.dumps({
                "query_id": qid,
                "pcs_gemma": pcs_gemma.get(qid),
                "pcs_llama": pcs_llama.get(qid),
            }) + "\n")

    n_none_gemma = sum(1 for v in pcs_gemma.values() if v is None)
    n_none_llama = sum(1 for v in pcs_llama.values() if v is None)
    print(f"\nWrote {OUTPUT_FILE}")
    print(f"  Gemma: {len(pcs_gemma) - n_none_gemma}/{len(records)} scored ({n_none_gemma} skipped — empty answer)")
    print(f"  Llama: {len(pcs_llama) - n_none_llama}/{len(records)} scored ({n_none_llama} skipped — empty answer)")


if __name__ == "__main__":
    main()