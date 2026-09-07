"""
Phase 4, MLP Stage 1: fast per-layer screening (Gemma only).

Fixes the previous MLP attempt's core flaw: ablating all 16 validated
layers' MLP output SIMULTANEOUSLY made 53% faithful-case damage
uninterpretable -- couldn't tell "this layer matters" from "we broke
general computation." This tests each validated layer INDIVIDUALLY,
against 4 randomly chosen non-validated layers (also tested
individually) as a control baseline.

Small sample (n=10/group) by design -- this is a SCREEN, not a final
result. Purpose: find which single layer(s), if any, show real
separation from the random-layer baseline. Whatever looks promising
gets a full n=100 confirmation run separately, later.

Run from repo root:
    python Tests/mlp_stage1_screen.py

Requires:
    data/final/Phase_03/phase3_length_controlled.json
    data/final/Phase_04/ablation_test_sample.json
    data/final/Phase_02/analysis_dataset.jsonl
    data/final/Phase_02/retrieval_results.jsonl

Writes:
    data/final/Phase_04/mlp_stage1_screen.jsonl
"""

import json
import random
import re
import sys
import os
import time
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
OUT_PATH = "data/final/Phase_04/mlp_stage1_screen.jsonl"

MODEL_PATH = "/home/models/gemma-2-9b"
NUM_LAYERS = 42
MAX_NEW_TOKENS = 24
N_MEANS_QUERIES = 40
N_SCREEN_PER_GROUP = 10  # SMALL on purpose -- this is a screen, not a final result
N_RANDOM_CONTROL_LAYERS = 4
SEED = 789

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


def get_validated_layers():
    data = json.load(open(LENGTH_CONTROLLED_PATH))
    return sorted(r["layer"] for r in data["gemma"] if r.get("fdr_significant"))


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


def compute_means_for_layers(model, tokenizer, layers, faithful_pool, device):
    sums = {l: None for l in layers}
    counts = {l: 0 for l in layers}
    captured = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            captured[layer_idx] = output[0, -1, :].detach().float().cpu()
            return output
        return hook

    handles = [model.model.layers[l].mlp.register_forward_hook(make_hook(l)) for l in layers]

    for r in tqdm(faithful_pool, desc="Computing per-layer MLP means"):
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


def test_single_layer(model, tokenizer, layer_idx, mean_vec, screen_queries, dataset_by_id, retrieval_by_id, device):
    hook = MLPAblationHook(mean_vec)
    handle = hook.register(model.model.layers[layer_idx])

    n_override = n_flip_correct = n_faithful = n_flip_incorrect = 0
    for q, group in screen_queries:
        qid = q["query_id"]
        ret = retrieval_by_id.get(qid)
        row = dataset_by_id.get(qid)
        if not ret or not row or not ret.get("retrieved_passages"):
            continue
        prompt = build_prompt(ret["retrieved_passages"], q["question"])
        gold = row.get("gold_answer_text")

        hook.disarm()
        baseline_text = generate_text(model, tokenizer, prompt, device)
        baseline_correct = is_correct(baseline_text, gold)

        hook.arm()
        ablated_text = generate_text(model, tokenizer, prompt, device)
        ablated_correct = is_correct(ablated_text, gold)
        hook.disarm()

        if group == "override_cases":
            n_override += 1
            if (not baseline_correct) and ablated_correct:
                n_flip_correct += 1
        else:
            n_faithful += 1
            if baseline_correct and (not ablated_correct):
                n_flip_incorrect += 1

    handle.remove()
    return n_flip_correct, n_override, n_flip_incorrect, n_faithful


def main():
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading gemma ({MODEL_PATH})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=QUANT_CONFIG, device_map=device, attn_implementation="eager",
    )
    model.eval()

    real_layers = get_validated_layers()
    rng = random.Random(SEED)
    non_validated = [l for l in range(NUM_LAYERS) if l not in real_layers]
    rng.shuffle(non_validated)
    random_layers = sorted(non_validated[:N_RANDOM_CONTROL_LAYERS])
    print(f"Real validated layers ({len(real_layers)}): {real_layers}")
    print(f"Random control layers ({len(random_layers)}): {random_layers}")

    test_sample = json.load(open(TEST_SAMPLE_PATH))
    dataset_records = load_jsonl(DATASET_PATH)
    retrieval_records = load_jsonl(RETRIEVAL_PATH)
    dataset_by_id = {r["query_id"]: r for r in dataset_records}
    retrieval_by_id = {r["query_id"]: r for r in retrieval_records}

    rng2 = random.Random(SEED)
    override_pool = list(test_sample["gemma"]["override_cases"])
    faithful_pool_q = list(test_sample["gemma"]["faithful_control_cases"])
    rng2.shuffle(override_pool)
    rng2.shuffle(faithful_pool_q)
    screen_queries = (
        [(q, "override_cases") for q in override_pool[:N_SCREEN_PER_GROUP]] +
        [(q, "faithful_control_cases") for q in faithful_pool_q[:N_SCREEN_PER_GROUP]]
    )
    print(f"Screening with {N_SCREEN_PER_GROUP} override + {N_SCREEN_PER_GROUP} faithful queries per layer")

    exclude_ids = {c["query_id"] for c in test_sample["gemma"]["faithful_control_cases"]}
    exclude_ids |= {c["query_id"] for c in test_sample["gemma"]["override_cases"]}
    means_faithful_pool = [
        r for r in dataset_records
        if r.get("gemma_label") == "faithful"
        and r.get("source_category") != "confiqa"
        and r["query_id"] not in exclude_ids
    ][:N_MEANS_QUERIES]
    for r in means_faithful_pool:
        ret = retrieval_by_id.get(r["query_id"])
        r["retrieved_passages"] = ret["retrieved_passages"] if ret else []

    all_layers_needed = sorted(set(real_layers) | set(random_layers))
    means = compute_means_for_layers(model, tokenizer, all_layers_needed, means_faithful_pool, device)

    results = []
    print("\n" + "=" * 70)
    print("SCREENING REAL VALIDATED LAYERS")
    print("=" * 70)
    for layer_idx in real_layers:
        if layer_idx not in means:
            continue
        flip_correct, n_override, flip_incorrect, n_faithful = test_single_layer(
            model, tokenizer, layer_idx, means[layer_idx], screen_queries, dataset_by_id, retrieval_by_id, device
        )
        net = flip_correct - flip_incorrect
        print(f"  Layer {layer_idx:2d} [REAL]:    flip_correct={flip_correct}/{n_override}  "
              f"flip_incorrect={flip_incorrect}/{n_faithful}  net={net:+d}")
        results.append({"layer": layer_idx, "condition": "real", "flip_correct": flip_correct,
                         "n_override": n_override, "flip_incorrect": flip_incorrect, "n_faithful": n_faithful})

    print("\n" + "=" * 70)
    print("SCREENING RANDOM CONTROL LAYERS")
    print("=" * 70)
    for layer_idx in random_layers:
        if layer_idx not in means:
            continue
        flip_correct, n_override, flip_incorrect, n_faithful = test_single_layer(
            model, tokenizer, layer_idx, means[layer_idx], screen_queries, dataset_by_id, retrieval_by_id, device
        )
        net = flip_correct - flip_incorrect
        print(f"  Layer {layer_idx:2d} [CONTROL]: flip_correct={flip_correct}/{n_override}  "
              f"flip_incorrect={flip_incorrect}/{n_faithful}  net={net:+d}")
        results.append({"layer": layer_idx, "condition": "control", "flip_correct": flip_correct,
                         "n_override": n_override, "flip_incorrect": flip_incorrect, "n_faithful": n_faithful})

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    elapsed_min = (time.time() - start_time) / 60
    print(f"\n{'='*70}")
    print(f"Wrote {OUT_PATH}")
    print(f"Total time: {elapsed_min:.1f} minutes")
    print("\nLook for REAL layers with net scores clearly higher than the")
    print("CONTROL layers' net scores -- those are candidates for a full")
    print("n=100 confirmation run. This screen is small-sample and noisy")
    print("by design; treat any single layer's number as a lead, not proof.")


if __name__ == "__main__":
    main()
