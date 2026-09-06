"""
Phase 3: ROI Extraction (Gemma 2 9B / Llama 3.1 8B).

Computes divergence between attention over the QUESTION tokens:
  (a) with retrieved passages present (with-context)
  (b) with no context at all (no-context baseline)
per head, per layer -- NOT averaged across layers/heads (lesson from
T5's entropy failure).

This is the real-model version of 05_roi_divergence_practice.py, which
validated: the offset-mapping span-finding approach (fixes a real BPE
boundary bug), that divergence varies meaningfully across layers/heads,
and that it responds to different questions.

Runs on the ~120-query STRATIFIED SAMPLE first (select_phase3_sample.py's
output), not the full 1500 -- confirm this shows something before
scaling up.

Sequential model loading (same pattern as T3/T5) to keep VRAM safe.

Output: data/final/phase3_roi_results.jsonl
  {"query_id": ..., "model": ..., "stratum": ...,
   "divergence_by_layer": [[head0, head1, ...], ...],  # one list per layer
   "overall_mean_divergence": float}

Requirements:
  pip install transformers accelerate bitsandbytes torch tqdm
"""

import json
import gc
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

SAMPLE_FILE = "data/final/phase3_stratified_sample.jsonl"
RETRIEVAL_FILE = "data/final/retrieval_results.jsonl"
OUTPUT_FILE = "data/final/phase3_roi_results.jsonl"

MODEL_CONFIGS = {
    "gemma": "/root/models/gemma-2-9b",
    "llama": "/root/models/Llama-3.1-8B",
}

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def find_question_span(full_prompt, question_text, tokenizer):
    """Offset-mapping based span finder -- validated in the GPT-2 practice
    script. Fixes a real bug where tokenizing the prefix separately and
    adding lengths silently dropped the first word of the question due
    to BPE/SentencePiece boundary re-tokenization."""
    char_start = full_prompt.index(question_text)
    char_end = char_start + len(question_text)

    encoding = tokenizer(full_prompt, return_tensors="pt", return_offsets_mapping=True,
                          truncation=True, max_length=4096)
    offsets = encoding["offset_mapping"][0].tolist()

    token_start = None
    token_end = None
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if token_start is None and e > char_start:
            token_start = i
        if s < char_end:
            token_end = i + 1

    return token_start, token_end, encoding["input_ids"]


class AttentionCapture:
    def __init__(self, num_layers, span_start, span_end):
        self.num_layers = num_layers
        self.span_start = span_start
        self.span_end = span_end
        self.per_layer = [None] * num_layers
        self._current_layer = 0

    def reset(self, span_start, span_end):
        self.span_start = span_start
        self.span_end = span_end
        self.per_layer = [None] * self.num_layers
        self._current_layer = 0

    def hook(self, module, input, output):
        layer_idx = self._current_layer
        self._current_layer = (self._current_layer + 1) % self.num_layers

        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            return
        attn_weights = output[1]  # (batch, heads, query_len, key_len)

        last_attn = attn_weights[0, :, -1, :]  # (heads, key_len)
        key_len = last_attn.shape[-1]
        s = min(self.span_start, key_len)
        e = min(self.span_end, key_len)
        if e <= s:
            return
        span_attn = last_attn[:, s:e]
        span_attn = span_attn / (span_attn.sum(dim=-1, keepdim=True) + 1e-12)
        self.per_layer[layer_idx] = span_attn.detach().cpu()


def cosine_divergence(a, b):
    if a is None or b is None:
        return None
    min_heads = min(a.shape[0], b.shape[0])
    a, b = a[:min_heads], b[:min_heads]
    sims = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return (1 - sims).tolist()


def merge_passages(sample_records, retrieval_records):
    retrieval_by_id = {r["query_id"]: r for r in retrieval_records}
    for r in sample_records:
        retrieval = retrieval_by_id.get(r["query_id"])
        r["retrieved_passages"] = retrieval["retrieved_passages"] if retrieval else []
    return sample_records


def run_model_pass(model_key, model_path, records, device):
    print(f"\nLoading {model_key} ({model_path})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=QUANT_CONFIG, device_map=device,
        attn_implementation="eager",
    )
    model.eval()

    num_layers = len(model.model.layers)
    capture = AttentionCapture(num_layers, 0, 0)
    handles = [layer.self_attn.register_forward_hook(capture.hook) for layer in model.model.layers]

    results = []
    skipped = 0

    for r in tqdm(records, desc=f"ROI extraction ({model_key})"):
        question = r["question"]
        passages = r.get("retrieved_passages", [])
        if not passages:
            skipped += 1
            continue

        no_context_prompt = f"Question: {question}\nAnswer:"
        passage_text = "\n\n".join(passages)
        with_context_prompt = f"{passage_text}\n\nQuestion: {question}\nAnswer:"

        try:
            no_ctx_start, no_ctx_end, no_ctx_ids = find_question_span(no_context_prompt, question, tokenizer)
            with_ctx_start, with_ctx_end, with_ctx_ids = find_question_span(with_context_prompt, question, tokenizer)
        except ValueError:
            skipped += 1
            continue

        if no_ctx_start is None or with_ctx_start is None:
            skipped += 1
            continue

        with torch.no_grad():
            capture.reset(no_ctx_start, no_ctx_end)
            model(no_ctx_ids.to(device))
            baseline_attn = list(capture.per_layer)

            capture.reset(with_ctx_start, with_ctx_end)
            model(with_ctx_ids.to(device))
            context_attn = list(capture.per_layer)

        divergence_by_layer = []
        for layer_idx in range(num_layers):
            div = cosine_divergence(context_attn[layer_idx], baseline_attn[layer_idx])
            divergence_by_layer.append(div if div is not None else [])

        flat_valid = [v for layer in divergence_by_layer for v in layer]
        overall_mean = sum(flat_valid) / len(flat_valid) if flat_valid else None

        results.append({
            "query_id": r["query_id"],
            "model": model_key,
            "stratum": r.get("_phase3_stratum"),
            "divergence_by_layer": divergence_by_layer,
            "overall_mean_divergence": overall_mean,
        })

    for h in handles:
        h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Done with {model_key}. Processed {len(results)}, skipped {skipped} "
          f"(no passages or span-finding failure).")
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected -- this will be extremely slow.")

    sample_records = load_jsonl(SAMPLE_FILE)
    retrieval_records = load_jsonl(RETRIEVAL_FILE)
    sample_records = merge_passages(sample_records, retrieval_records)
    print(f"Loaded {len(sample_records)} stratified sample queries "
          f"(with passages merged from {RETRIEVAL_FILE})")

    all_results = []
    for model_key, model_path in MODEL_CONFIGS.items():
        all_results.extend(run_model_pass(model_key, model_path, sample_records, device))

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {OUTPUT_FILE} ({len(all_results)} rows)")
    print("Next: check whether overall_mean_divergence (or specific layers) differ")
    print("between 'faithful' and 'override' strata, and between 'natural_rag_other'")
    print("and 'popqa' strata -- this is the actual test of the Phase 3 hypothesis.")


if __name__ == "__main__":
    main()