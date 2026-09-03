"""
T5: RAG Generation (Both Models).

For each query:
  1. Prompt = top-5 retrieved passages concatenated + "Question: {q}\nAnswer:"
  2. Greedy decode, max_new_tokens=50
  3. Extract answer span: first sentence / text before first newline
  4. During generation, a forward hook logs MEAN ATTENTION ENTROPY
     computed only over PASSAGE token positions (not the whole prompt) --
     this requires knowing which token indices in the prompt are passage
     vs. question, tracked via prompt construction below.

Run sequentially per model (same pattern as T3's compute_pcs.py) to keep
VRAM usage safe on a single rented GPU.

Output: data/final/generation_results.jsonl
  {"query_id": ..., "gemma_answer": ..., "gemma_attn_entropy_on_passage": ...,
   "llama_answer": ..., "llama_attn_entropy_on_passage": ...}

Requirements:
  pip install transformers accelerate bitsandbytes torch tqdm
"""

import json
import gc
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

RECORDS_FILE = "data/final/dataset_1500_with_titles_1.jsonl"
RETRIEVAL_FILE = "data/final/retrieval_results.jsonl"
OUTPUT_FILE = "data/final/generation_results.jsonl"

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

MAX_NEW_TOKENS = 50


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_prompt_with_span(tokenizer, passages: list, question: str):
    """
    Builds the prompt and returns the token index range covering the
    passage text specifically, so the entropy hook can restrict its
    computation to those positions. This tokenizes the passage segment
    separately first to get its length -- a bit redundant but the
    simplest way to get an accurate boundary without guessing.
    """
    passage_text = "\n\n".join(passages)
    full_prompt = f"{passage_text}\n\nQuestion: {question}\nAnswer:"

    passage_ids = tokenizer(passage_text, add_special_tokens=False).input_ids
    passage_token_len = len(passage_ids)

    # Account for the tokenizer's leading special token(s) on the full prompt
    full_ids_check = tokenizer(full_prompt, return_tensors="pt").input_ids
    leading_offset = full_ids_check.shape[1] - len(
        tokenizer(full_prompt, add_special_tokens=False).input_ids
    )

    passage_start = leading_offset
    passage_end = leading_offset + passage_token_len  # exclusive

    return full_prompt, passage_start, passage_end


def extract_answer_span(generated_text: str) -> str:
    """First sentence / text before first newline, per spec."""
    text = generated_text.strip()
    newline_idx = text.find("\n")
    if newline_idx != -1:
        text = text[:newline_idx]
    # also cut at first sentence-ending period if it comes before any newline
    period_idx = text.find(". ")
    if period_idx != -1:
        text = text[: period_idx + 1]
    return text.strip()


class EntropyHookState:
    """Holds the passage span + accumulates entropy across generation
    steps. A class instead of bare globals so multiple queries don't
    leak state into each other if you forget to reset -- reset() is
    called explicitly per query below regardless, but this makes an
    accidental miss visible instead of silently reusing stale values."""

    def __init__(self):
        self.passage_start = None
        self.passage_end = None
        self.entropies = []

    def reset(self, passage_start, passage_end):
        self.passage_start = passage_start
        self.passage_end = passage_end
        self.entropies = []

    def hook(self, module, input, output):
        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            return
        attn_weights = output[1]  # (batch, heads, query_len, key_len)

        # Only the last query position -- the token currently being generated.
        last_attn = attn_weights[:, :, -1, :]  # (batch, heads, key_len)

        key_len = last_attn.shape[-1]
        p_start = min(self.passage_start, key_len)
        p_end = min(self.passage_end, key_len)
        if p_end <= p_start:
            return  # passage span not in this key range yet/at all, skip

        passage_attn = last_attn[:, :, p_start:p_end]  # (batch, heads, passage_len)
        # Renormalize to a valid distribution over JUST the passage positions
        passage_attn_sum = passage_attn.sum(dim=-1, keepdim=True)
        passage_attn_sum = torch.clamp(passage_attn_sum, min=1e-12)
        renormalized = passage_attn / passage_attn_sum

        eps = 1e-12
        entropy = -torch.sum(renormalized * torch.log(renormalized + eps), dim=-1)
        self.entropies.append(entropy.mean().item())


def run_model_pass(model_key, model_path, records, retrieval_by_id, device):
    print(f"\nLoading {model_key} ({model_path})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=QUANT_CONFIG, device_map=device,
        attn_implementation="eager",  # needed to get attention weights in the hook
    )
    model.eval()

    hook_state = EntropyHookState()
    handles = []
    for layer in model.model.layers:  # adjust attribute path if architecture differs
        h = layer.self_attn.register_forward_hook(hook_state.hook)
        handles.append(h)

    results = {}
    for r in tqdm(records, desc=f"Generation ({model_key})"):
        qid = r["query_id"]
        retrieval = retrieval_by_id.get(qid)
        if retrieval is None:
            results[qid] = {"answer": None, "attn_entropy_on_passage": None}
            continue

        passages = retrieval["retrieved_passages"]
        prompt, p_start, p_end = build_prompt_with_span(tokenizer, passages, r["question"])
        hook_state.reset(p_start, p_end)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        answer = extract_answer_span(generated_text)

        mean_entropy = (
            sum(hook_state.entropies) / len(hook_state.entropies)
            if hook_state.entropies else None
        )

        results[qid] = {"answer": answer, "attn_entropy_on_passage": mean_entropy}

    for h in handles:
        h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Done with {model_key}, VRAM freed.")
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected -- this will be extremely slow.")

    records = load_jsonl(RECORDS_FILE)
    retrieval_records = load_jsonl(RETRIEVAL_FILE)
    retrieval_by_id = {r["query_id"]: r for r in retrieval_records}
    print(f"Loaded {len(records)} records, {len(retrieval_records)} retrieval results")

    gemma_results = run_model_pass("gemma", MODEL_CONFIGS["gemma"], records, retrieval_by_id, device)
    llama_results = run_model_pass("llama", MODEL_CONFIGS["llama"], records, retrieval_by_id, device)

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            qid = r["query_id"]
            g = gemma_results.get(qid, {})
            l = llama_results.get(qid, {})
            f.write(json.dumps({
                "query_id": qid,
                "gemma_answer": g.get("answer"),
                "gemma_attn_entropy_on_passage": g.get("attn_entropy_on_passage"),
                "llama_answer": l.get("answer"),
                "llama_attn_entropy_on_passage": l.get("attn_entropy_on_passage"),
            }) + "\n")

    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()