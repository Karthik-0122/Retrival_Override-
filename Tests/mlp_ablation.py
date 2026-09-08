"""
Phase 4, MLP ablation (follow-up after attention-head ablation showed no
advantage over random control).

Reads Phase 3's length-controlled + FDR-validated significant layers
directly from data/final/Phase_03/phase3_length_controlled.json (not
hardcoded, to avoid a transcription mismatch). Ablates the MLP output at
ALL validated layers together (block-level), for both REAL validated
layers and a RANDOM set of layers (same count, excluding the real ones)
as a control -- run back to back in the same execution, so the control
can't be accidentally skipped this time.

Run from repo root:
    python Tests/run_mlp_ablation.py

Requires:
    data/final/Phase_03/phase3_length_controlled.json
    data/final/Phase_04/ablation_test_sample.json
    data/final/Phase_02/analysis_dataset.jsonl
    data/final/Phase_02/retrieval_results.jsonl

Writes:
    data/final/Phase_04/mlp_ablation_results.jsonl  (both real and control rows, tagged)
"""

import json
import random
import re
import sys
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp_ablation_utils import MLPAblationHook

LENGTH_CONTROLLED_PATH = "data/final/Phase_03/phase3_length_controlled.json"
TEST_SAMPLE_PATH = "data/final/Phase_04/ablation_test_sample.json"
DATASET_PATH = "data/final/Phase_02/analysis_dataset.jsonl"
RETRIEVAL_PATH = "data/final/Phase_02/retrieval_results.jsonl"
OUT_PATH = "data/final/Phase_04/mlp_ablation_results.jsonl"

MODEL_CONFIGS = {
    "gemma": "/home/models/gemma-2-9b",
    "llama": "/home/models/Llama-3.1-8B",
}
NUM_LAYERS = {"gemma": 42, "llama": 32}
MAX_NEW_TOKENS = 24
N_MEANS_QUERIES = 60
SEED = 456  # yet another independent draw, distinct from other seeds used this project

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


def is_correct(generated_text, gold_answer_text):
    if not gold_answer_text:
        return False
    return normalize_text(gold_answer_text) in normalize_text(generated_text)


def get_validated_layers(model_key):
    data = json.load(open(LENGTH_CONTROLLED_PATH))
    return sorted(r["layer"] for r in data[model_key] if r.get("fdr_significant"))


def pick_random_layers(model_key, n, exclude):
    rng = random.Random(SEED)
    pool = [l for l in range(NUM_LAYERS[model_key]) if l not in exclude]
    rng.shuffle(pool)
    return sorted(pool[:n])


def build_prompt(passages, question):
    passage_text = "\n\n".join(passages)
    return f"{passage_text}\n\nQuestion: {question}\nAnswer:"


def generate_text(model, tokenizer, prompt, device):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
    for stop_str in ["\nQuestion:", "\nquestion:", "\n\n"]:
        idx = text.find(stop_str)
        if idx != -1:
            text = text[:idx]
    return text.split("\n")[0].strip()


def compute_mlp_means(model, tokenizer, layers, faithful_pool, device):
    sums = {l: None for l in layers}
    counts = {l: 0 for l in layers}
    captured = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            captured[layer_idx] = output[0, -1, :].detach().float().cpu()
            return output
        return hook

    handles = [model.model.layers[l].mlp.register_forward_hook(make_hook(l)) for l in layers]

    for r in tqdm(faithful_pool, desc="Computing MLP faithful means"):
        if not r.get("retrieved_passages"):
            continue
        prompt = build_prompt(r["retrieved_passages"], r["question"])
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(device)
        captured.clear()
        with torch.no_grad():
            model(input_ids)
        for l in layers:
            if l not in captured:
                continue
            vec = captured[l]
            if sums[l] is None:
                sums[l] = torch.zeros_like(vec)
            sums[l] += vec
            counts[l] += 1

    for h in handles:
        h.remove()

    return {l: sums[l] / counts[l] for l in layers if counts[l] > 0}


def run_condition(model, tokenizer, layers, means, test_sample, model_key, device, condition_label,
                   dataset_records, retrieval_records):
    hooks, handles = [], []
    for l in layers:
        if l not in means:
            continue
        h = MLPAblationHook(means[l])
        handles.append(h.register(model.model.layers[l]))
        hooks.append(h)

    dataset_by_id = {r["query_id"]: r for r in dataset_records}
    retrieval_by_id = {r["query_id"]: r for r in retrieval_records}

    results = []
    for group_name in ["override_cases", "faithful_control_cases"]:
        queries = test_sample[model_key][group_name]
        for q in tqdm(queries, desc=f"MLP {condition_label} ({model_key}, {group_name})"):
            qid = q["query_id"]
            ret = retrieval_by_id.get(qid)
            row = dataset_by_id.get(qid)
            if not ret or not row or not ret.get("retrieved_passages"):
                continue
            prompt = build_prompt(ret["retrieved_passages"], q["question"])
            gold = row.get("gold_answer_text")

            for h in hooks:
                h.disarm()
            baseline_text = generate_text(model, tokenizer, prompt, device)
            baseline_correct = is_correct(baseline_text, gold)

            for h in hooks:
                h.arm()
            ablated_text = generate_text(model, tokenizer, prompt, device)
            ablated_correct = is_correct(ablated_text, gold)
            for h in hooks:
                h.disarm()

            results.append({
                "query_id": qid, "model": model_key, "group": group_name, "condition": condition_label,
                "baseline_correct": baseline_correct, "ablated_correct": ablated_correct,
                "flipped_to_correct": (not baseline_correct) and ablated_correct,
                "flipped_to_incorrect": baseline_correct and (not ablated_correct),
            })

    for h in handles:
        h.remove()
    return results


def print_summary(results, model_key, condition_label):
    n_override = sum(1 for r in results if r["group"] == "override_cases")
    n_flip_correct = sum(1 for r in results if r["group"] == "override_cases" and r["flipped_to_correct"])
    n_faithful = sum(1 for r in results if r["group"] == "faithful_control_cases")
    n_flip_incorrect = sum(1 for r in results if r["group"] == "faithful_control_cases" and r["flipped_to_incorrect"])
    print(f"\n{model_key.upper()} -- {condition_label}:")
    print(f"  Override cases: {n_flip_correct}/{n_override} flipped to CORRECT")
    print(f"  Faithful cases: {n_flip_incorrect}/{n_faithful} flipped to INCORRECT (side effect)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_sample = json.load(open(TEST_SAMPLE_PATH))

    all_results = []
    for model_key, model_path in MODEL_CONFIGS.items():
        print(f"\n{'='*70}\nLoading {model_key} ({model_path})...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=QUANT_CONFIG, device_map=device, attn_implementation="eager",
        )
        model.eval()

        real_layers = get_validated_layers(model_key)
        random_layers = pick_random_layers(model_key, len(real_layers), set(real_layers))
        print(f"  Real validated layers ({len(real_layers)}): {real_layers}")
        print(f"  Random control layers ({len(random_layers)}): {random_layers}")

        dataset_records = load_jsonl(DATASET_PATH)
        retrieval_records = load_jsonl(RETRIEVAL_PATH)
        retrieval_by_id = {r["query_id"]: r for r in retrieval_records}

        exclude_ids = {c["query_id"] for c in test_sample[model_key]["faithful_control_cases"]}
        exclude_ids |= {c["query_id"] for c in test_sample[model_key]["override_cases"]}
        label_key = f"{model_key}_label"
        faithful_pool = [
            r for r in dataset_records
            if r.get(label_key) == "faithful"
            and r.get("source_category") != "confiqa"
            and r["query_id"] not in exclude_ids
        ][:N_MEANS_QUERIES]
        for r in faithful_pool:
            ret = retrieval_by_id.get(r["query_id"])
            r["retrieved_passages"] = ret["retrieved_passages"] if ret else []

        all_layers_needed = sorted(set(real_layers) | set(random_layers))
        means = compute_mlp_means(model, tokenizer, all_layers_needed, faithful_pool, device)

        real_results = run_condition(model, tokenizer, real_layers, means, test_sample, model_key,
                                      device, "REAL", dataset_records, retrieval_records)
        print_summary(real_results, model_key, "REAL (validated layers)")
        all_results.extend(real_results)

        control_results = run_condition(model, tokenizer, random_layers, means, test_sample, model_key,
                                         device, "CONTROL", dataset_records, retrieval_records)
        print_summary(control_results, model_key, "CONTROL (random layers)")
        all_results.extend(control_results)

        del model
        torch.cuda.empty_cache()

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
