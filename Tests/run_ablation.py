"""
Phase 4, T22: Real ablation run.

SCOPE DECISION: ablates all target heads for a model TOGETHER (block-level),
not one at a time. This answers the primary causal question -- "does
ablating the flagged block flip override cases toward correct answers,
without breaking faithful cases" -- at a fraction of the compute cost of
testing every head individually. Per-head attribution (which single head
matters most) is a natural follow-up ONCE the block-level effect is
confirmed to exist -- no point paying for that granularity first.

For each query (override + faithful-control), per model:
  1. Generate normally (no ablation) -- this is the baseline.
  2. Generate with ALL target heads ablated (mean-patched to their
     faithful-case values, computed in T22a) -- this is the ablated run.
  3. Score correctness of both generations against gold answers.

Run from repo root:
    python Tests/run_ablation.py

Requires:
    data/final/Phase_04/ablation_targets.json
    data/final/Phase_04/ablation_test_sample.json
    data/final/Phase_04/faithful_means.pt
    data/final/Phase_02/analysis_dataset.jsonl
    data/final/Phase_02/retrieval_results.jsonl

Writes:
    data/final/Phase_04/ablation_results.jsonl
"""

import json
import re
import sys
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation_utils import AblationHook

TARGETS_PATH = "data/final/Phase_04/ablation_targets_escalated.json"
TEST_SAMPLE_PATH = "data/final/Phase_04/ablation_test_sample.json"
MEANS_PATH = "data/final/Phase_04/faithful_means_escalated.pt"
DATASET_PATH = "data/final/Phase_02/analysis_dataset.jsonl"
RETRIEVAL_PATH = "data/final/Phase_02/retrieval_results.jsonl"
OUT_PATH = "data/final/Phase_04/ablation_results_escalated.jsonl"

MODEL_CONFIGS = {
    "gemma": "/home/models/gemma-2-9b",
    "llama": "/home/models/Llama-3.1-8B",
}
HEAD_DIMS = {"gemma": 256, "llama": 128}
MAX_NEW_TOKENS = 24  # generous for short factual answers, matches typical answer length seen in this dataset

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def normalize_text(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def get_gold_candidates(gold_answer_text):
    """This dataset version stores the gold answer as a plain string in
    'gold_answer_text', not the more complex dict/list structure seen in
    an earlier dataset version. Simple wrapper for a single candidate."""
    if not gold_answer_text:
        return []
    return [gold_answer_text]


def is_correct(generated_text, gold_answer_text):
    candidates = get_gold_candidates(gold_answer_text)
    norm_gen = normalize_text(generated_text)
    for c in candidates:
        if normalize_text(c) in norm_gen:
            return True
    return False


def merge_passages_and_gold(records, retrieval_records, dataset_records):
    retrieval_by_id = {r["query_id"]: r for r in retrieval_records}
    dataset_by_id = {r["query_id"]: r for r in dataset_records}
    merged = []
    for r in records:
        retrieval = retrieval_by_id.get(r["query_id"])
        dataset_row = dataset_by_id.get(r["query_id"])
        if retrieval is None or dataset_row is None:
            continue
        merged.append({
            "query_id": r["query_id"],
            "question": r["question"],
            "retrieved_passages": retrieval.get("retrieved_passages", []),
            "gold_answer_text": dataset_row.get("gold_answer_text"),
        })
    return merged


def generate_text(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs.input_ids.to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy -- deterministic, reproducible
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    # Generation frequently ran past the actual answer into a hallucinated
    # continuation of a fake follow-up "Question:...Answer:..." turn,
    # confirmed by inspecting raw output on this dataset. Truncate at the
    # first sign of that runaway continuation so correctness scoring sees
    # a clean answer, not answer-plus-fabricated-next-turn.
    for stop_str in ["\nQuestion:", "\nquestion:", "\n\n"]:
        idx = text.find(stop_str)
        if idx != -1:
            text = text[:idx]
    text = text.split("\n")[0].strip()  # also cut at first plain newline as a final guard

    return text


def run_model(model_key, model_path, targets, test_sample, means, device):
    print(f"\nLoading {model_key} ({model_path})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=QUANT_CONFIG, device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    head_dim = HEAD_DIMS[model_key]

    # set up one AblationHook per target, all initially disarmed
    hooks = []
    handles = []
    for t in targets:
        layer_idx, head_idx = t["layer"], t["head"]
        mean_key = f"{model_key}_{layer_idx}_{head_idx}"
        if mean_key not in means:
            print(f"  WARNING: no faithful mean found for layer {layer_idx} head {head_idx}, skipping this target")
            continue
        hook = AblationHook(head_idx, head_dim, means[mean_key])
        layer = model.model.layers[layer_idx]
        handle = hook.register(layer)
        hooks.append(hook)
        handles.append(handle)

    print(f"  {len(hooks)} ablation hooks registered (block-level, all armed/disarmed together)")

    dataset_records = load_jsonl(DATASET_PATH)
    retrieval_records = load_jsonl(RETRIEVAL_PATH)

    results = []
    for group_name in ["override_cases", "faithful_control_cases"]:
        queries = test_sample[model_key][group_name]
        merged = merge_passages_and_gold(queries, retrieval_records, dataset_records)

        for r in tqdm(merged, desc=f"Ablation ({model_key}, {group_name})"):
            if not r["retrieved_passages"]:
                continue
            passage_text = "\n\n".join(r["retrieved_passages"])
            prompt = f"{passage_text}\n\nQuestion: {r['question']}\nAnswer:"

            for h in hooks:
                h.disarm()
            baseline_text = generate_text(model, tokenizer, prompt, device)
            baseline_correct = is_correct(baseline_text, r["gold_answer_text"])

            for h in hooks:
                h.arm()
            ablated_text = generate_text(model, tokenizer, prompt, device)
            ablated_correct = is_correct(ablated_text, r["gold_answer_text"])
            for h in hooks:
                h.disarm()

            results.append({
                "query_id": r["query_id"],
                "model": model_key,
                "group": group_name,
                "baseline_text": baseline_text,
                "baseline_correct": baseline_correct,
                "ablated_text": ablated_text,
                "ablated_correct": ablated_correct,
                "flipped_to_correct": (not baseline_correct) and ablated_correct,
                "flipped_to_incorrect": baseline_correct and (not ablated_correct),
            })

    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    targets_by_model = json.load(open(TARGETS_PATH))
    test_sample = json.load(open(TEST_SAMPLE_PATH))
    means = torch.load(MEANS_PATH)

    all_results = []
    for model_key, model_path in MODEL_CONFIGS.items():
        results = run_model(model_key, model_path, targets_by_model[model_key], test_sample, means, device)
        all_results.extend(results)

        n_override = sum(1 for r in results if r["group"] == "override_cases")
        n_flipped_correct = sum(1 for r in results if r["group"] == "override_cases" and r["flipped_to_correct"])
        n_faithful = sum(1 for r in results if r["group"] == "faithful_control_cases")
        n_flipped_incorrect = sum(1 for r in results if r["group"] == "faithful_control_cases" and r["flipped_to_incorrect"])

        print(f"\n{model_key.upper()} SUMMARY:")
        print(f"  Override cases: {n_flipped_correct}/{n_override} flipped to CORRECT when ablated")
        print(f"  Faithful cases: {n_flipped_incorrect}/{n_faithful} flipped to INCORRECT when ablated (side-effect check)")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {OUT_PATH} ({len(all_results)} rows)")


if __name__ == "__main__":
    main()
