"""
Smoke test for the batched version of run_ablation.py.

CRITICAL CHECK: runs the same few queries twice -- once each alone
(batch_size=1) and once together in a real batch -- and confirms the
outputs are IDENTICAL. If left-padding and the attention mask are
implemented correctly, padding tokens should have zero influence on the
real tokens' computation, so batch-of-1 and batch-of-N results for the
same query must match exactly. If they don't match, something about the
padding/masking/hook-indexing is silently wrong and the full batched run
should NOT be trusted until this is fixed.

Run from repo root, BEFORE trusting a full batched run_ablation.py run:
    python Tests/smoke_test_batching.py
"""

import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, "Tests")
from ablation_utils import AblationHook

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
)
MODEL_PATH = "/home/models/gemma-2-9b"
HEAD_DIM = 256
MAX_NEW_TOKENS = 24
N_TEST_QUERIES = 4


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def generate_batch(model, tokenizer, prompts, device):
    inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=4096, padding=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, attention_mask=attention_mask, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    input_len = input_ids.shape[1]
    return [tokenizer.decode(row, skip_special_tokens=True) for row in output_ids[:, input_len:]]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=QUANT_CONFIG, device_map="cuda", attn_implementation="eager"
    )
    model.eval()

    targets = json.load(open("data/final/Phase_04/ablation_targets_escalated.json"))["gemma"]
    means = torch.load("data/final/Phase_04/faithful_means_escalated.pt")

    hooks, handles = [], []
    for t in targets:
        key = f"gemma_{t['layer']}_{t['head']}"
        if key not in means:
            continue
        h = AblationHook(t["head"], HEAD_DIM, means[key])
        handles.append(h.register(model.model.layers[t["layer"]]))
        hooks.append(h)
    print(f"{len(hooks)} hooks registered\n")

    test_sample = json.load(open("data/final/Phase_04/ablation_test_sample.json"))
    retrieval = {r["query_id"]: r for r in load_jsonl("data/final/Phase_02/retrieval_results.jsonl")}

    queries = test_sample["gemma"]["override_cases"][:N_TEST_QUERIES]
    prompts = []
    for q in queries:
        ret = retrieval.get(q["query_id"])
        passage_text = "\n\n".join(ret["retrieved_passages"]) if ret else ""
        prompts.append(f"{passage_text}\n\nQuestion: {q['question']}\nAnswer:")

    print("=" * 70)
    print("PASS 1: each query run ALONE (batch_size=1)")
    print("=" * 70)
    alone_results_baseline = []
    alone_results_ablated = []
    for p in prompts:
        for h in hooks:
            h.disarm()
        alone_results_baseline.append(generate_batch(model, tokenizer, [p], "cuda")[0])
        for h in hooks:
            h.arm()
        alone_results_ablated.append(generate_batch(model, tokenizer, [p], "cuda")[0])
        for h in hooks:
            h.disarm()

    print("\n" + "=" * 70)
    print(f"PASS 2: all {len(prompts)} queries run TOGETHER as one batch")
    print("=" * 70)
    for h in hooks:
        h.disarm()
    batch_results_baseline = generate_batch(model, tokenizer, prompts, "cuda")
    for h in hooks:
        h.arm()
    batch_results_ablated = generate_batch(model, tokenizer, prompts, "cuda")
    for h in hooks:
        h.disarm()

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    all_match = True
    for i, q in enumerate(queries):
        baseline_match = alone_results_baseline[i] == batch_results_baseline[i]
        ablated_match = alone_results_ablated[i] == batch_results_ablated[i]
        status = "MATCH" if (baseline_match and ablated_match) else "MISMATCH"
        if not (baseline_match and ablated_match):
            all_match = False
        print(f"\nQuery {i} ({q['query_id']}): {status}")
        print(f"  alone  baseline: {alone_results_baseline[i]!r}")
        print(f"  batch  baseline: {batch_results_baseline[i]!r}")
        print(f"  alone  ablated:  {alone_results_ablated[i]!r}")
        print(f"  batch  ablated:  {batch_results_ablated[i]!r}")

    print("\n" + "=" * 70)
    if all_match:
        print("ALL MATCH -- batching is safe to trust. Proceed to the full run_ablation.py.")
    else:
        print("MISMATCH DETECTED -- DO NOT trust a full batched run yet.")
        print("Likely causes: padding_side not set before tokenization, attention_mask")
        print("not passed correctly, or the hook applying to the wrong position for")
        print("some rows in the batch.")

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
