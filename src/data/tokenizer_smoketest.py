"""
Smoke test: does find_question_span work correctly on the REAL tokenizers
(Gemma's BPE-family tokenizer, Llama's SentencePiece-based one), not just
GPT-2? Tests 3 real queries from the stratified sample, both models.

Run this BEFORE the full extract_roi_phase3.py -- if either tokenizer's
spans come out wrong, fix it here first rather than discovering it after
running all 120 x 2 queries on paid GPU time.
"""

import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

SAMPLE_FILE = "data/final/phase3_stratified_sample.jsonl"
RETRIEVAL_FILE = "data/final/retrieval_results.jsonl"
N_TEST_QUERIES = 3

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


def find_question_span(full_prompt, question_text, tokenizer):
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


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    sample = load_jsonl(SAMPLE_FILE)[:N_TEST_QUERIES]
    retrieval = {r["query_id"]: r for r in load_jsonl(RETRIEVAL_FILE)}

    for model_key, model_path in MODEL_CONFIGS.items():
        print(f"\n{'=' * 70}")
        print(f"TOKENIZER: {model_key} ({model_path})")
        print(f"{'=' * 70}")

        print("Loading tokenizer only (no model weights needed for this test)...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Tokenizer class: {type(tokenizer).__name__}")

        for r in sample:
            question = r["question"]
            ret = retrieval.get(r["query_id"])
            passages = ret["retrieved_passages"] if ret else []
            passage_text = "\n\n".join(passages) if passages else "[no passage found]"

            no_context_prompt = f"Question: {question}\nAnswer:"
            with_context_prompt = f"{passage_text}\n\nQuestion: {question}\nAnswer:"

            print(f"\n--- Query: {r['query_id']} ---")
            print(f"Question: {question!r}")

            try:
                no_s, no_e, no_ids = find_question_span(no_context_prompt, question, tokenizer)
                with_s, with_e, with_ids = find_question_span(with_context_prompt, question, tokenizer)

                no_span_text = tokenizer.decode(no_ids[0][no_s:no_e])
                with_span_text = tokenizer.decode(with_ids[0][with_s:with_e])

                print(f"  No-context span:   {no_span_text!r}")
                print(f"  With-context span: {with_span_text!r}")

                no_match = no_span_text.strip() == question.strip()
                with_match = with_span_text.strip() == question.strip()
                print(f"  Exact match? no-context={no_match}  with-context={with_match}")
                if not (no_match and with_match):
                    print("  *** MISMATCH -- span finder needs fixing for this tokenizer ***")

            except Exception as e:
                print(f"  ERROR: {e}")

    print(f"\n{'=' * 70}")
    print("If any span above did NOT exactly match the question text, stop and")
    print("fix find_question_span for that tokenizer before running the full")
    print("extract_roi_phase3.py -- do not proceed on a failing tokenizer.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()