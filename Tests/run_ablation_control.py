"""
Phase 4, control experiment: random-head ablation.

CRITICAL MISSING PIECE identified after seeing real (non-zero, but small)
flip rates from ablating the Phase-3-identified heads: since generation
is greedy/deterministic, baseline is always fixed, so ANY ablation has
some nonzero chance of nudging a marginal case toward a different
answer -- just by perturbing computation at all, regardless of whether
the ablated heads are scientifically meaningful.

This runs the IDENTICAL pipeline (same queries, same mean-ablation
method, same number of heads ablated) but on a RANDOMLY SELECTED set of
heads instead of the Phase-3-identified ones. If the real targets show a
meaningfully HIGHER flip-to-correct rate than this random control, that
supports the localization finding causally. If the random control shows
a SIMILAR flip rate, it means disrupting almost any large head set does
about the same thing, and the Phase 3 localization hasn't been causally
validated by this experiment.

Run from repo root, AFTER run_ablation.py has produced real results:
    python Tests/run_ablation_control.py

Requires the same files as run_ablation.py.

Writes:
    data/final/Phase_04/ablation_results_control.jsonl
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
from ablation_utils import AblationHook

TARGETS_PATH = "data/final/Phase_04/ablation_targets_escalated.json"
TEST_SAMPLE_PATH = "data/final/Phase_04/ablation_test_sample.json"
DATASET_PATH = "data/final/Phase_02/analysis_dataset.jsonl"
RETRIEVAL_PATH = "data/final/Phase_02/retrieval_results.jsonl"
OUT_PATH = "data/final/Phase_04/ablation_results_control.jsonl"

MODEL_CONFIGS = {
    "gemma": "/home/models/gemma-2-9b",
    "llama": "/home/models/Llama-3.1-8B",
}
HEAD_DIMS = {"gemma": 256, "llama": 128}
NUM_LAYERS = {"gemma": 42, "llama": 32}
NUM_HEADS = {"gemma": 16, "llama": 32}
MAX_NEW_TOKENS = 24
SEED = 123  # deliberately different from other seeds used this project, for an independent random draw

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


def pick_random_targets(model_key, n_targets, exclude_targets):
    rng = random.Random(SEED)
    exclude_set = {(t["layer"], t["head"]) for t in exclude_targets}
    all_possible = [
        (l, h) for l in range(NUM_LAYERS[model_key]) for h in range(NUM_HEADS[model_key])
        if (l, h) not in exclude_set
    ]
    rng.shuffle(all_possible)
    chosen = all_possible[:n_targets]
    return [{"layer": l, "head": h} for l, h in chosen]


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
            input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    for stop_str in ["\nQuestion:", "\nquestion:", "\n\n"]:
        idx = text.find(stop_str)
        if idx != -1:
            text = text[:idx]
    return text.split("\n")[0].strip()


def build_prompt(passages, question):
    passage_text = "\n\n".join(passages)
    return f"{passage_text}\n\nQuestion: {question}\nAnswer:"


def compute_random_means(model, tokenizer, random_targets, model_key, faithful_pool, device):
    head_dim = HEAD_DIMS[model_key]
    sums = {(t["layer"], t["head"]): torch.zeros(head_dim, dtype=torch.float32) for t in random_targets}
    counts = {(t["layer"], t["head"]): 0 for t in random_targets}

    layer_indices = sorted({t["layer"] for t in random_targets})
    captured = {}

    def make_hook(layer_idx):
        def hook(module, args):
            captured[layer_idx] = args[0][0, -1, :].detach().float().cpu()
            return None
        return hook

    handles = [model.model.layers[li].self_attn.o_proj.register_forward_pre_hook(make_hook(li))
               for li in layer_indices]

    for r in tqdm(faithful_pool, desc=f"Computing control means ({model_key})"):
        passages = r.get("retrieved_passages", [])
        if not passages:
            continue
        prompt = build_prompt(passages, r["question"])
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(device)
        captured.clear()
        with torch.no_grad():
            model(input_ids)
        for t in random_targets:
            li, hi = t["layer"], t["head"]
            if li not in captured:
                continue
            vec = captured[li]
            h_start, h_end = hi * head_dim, (hi + 1) * head_dim
            sums[(li, hi)] += vec[h_start:h_end]
            counts[(li, hi)] += 1

    for h in handles:
        h.remove()

    return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}


def run_model_control(model_key, model_path, real_targets, test_sample, device):
    print(f"\nLoading {model_key} ({model_path})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=QUANT_CONFIG, device_map=device, attn_implementation="eager",
    )
    model.eval()

    random_targets = pick_random_targets(model_key, len(real_targets), real_targets)
    print(f"  Selected {len(random_targets)} RANDOM control heads (excluding real targets)")

    dataset_records = load_jsonl(DATASET_PATH)
    retrieval_records = load_jsonl(RETRIEVAL_PATH)

    exclude_ids = {c["query_id"] for c in test_sample[model_key]["faithful_control_cases"]}
    exclude_ids |= {c["query_id"] for c in test_sample[model_key]["override_cases"]}
    label_key = f"{model_key}_label"
    faithful_pool = [
        r for r in dataset_records
        if r.get(label_key) == "faithful"
        and r.get("source_category") != "confiqa"
        and r["query_id"] not in exclude_ids
    ][:60]
    retrieval_by_id = {r["query_id"]: r for r in retrieval_records}
    for r in faithful_pool:
        ret = retrieval_by_id.get(r["query_id"])
        r["retrieved_passages"] = ret["retrieved_passages"] if ret else []

    means = compute_random_means(model, tokenizer, random_targets, model_key, faithful_pool, device)

    head_dim = HEAD_DIMS[model_key]
    hooks, handles = [], []
    for t in random_targets:
        key = (t["layer"], t["head"])
        if key not in means:
            continue
        h = AblationHook(t["head"], head_dim, means[key])
        handles.append(h.register(model.model.layers[t["layer"]]))
        hooks.append(h)
    print(f"  {len(hooks)} random control hooks registered")

    results = []
    for group_name in ["override_cases", "faithful_control_cases"]:
        queries = test_sample[model_key][group_name]
        merged = merge_passages_and_gold(queries, retrieval_records, dataset_records)

        for r in tqdm(merged, desc=f"Control ablation ({model_key}, {group_name})"):
            if not r["retrieved_passages"]:
                continue
            prompt = build_prompt(r["retrieved_passages"], r["question"])

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
                "query_id": r["query_id"], "model": model_key, "group": group_name,
                "baseline_correct": baseline_correct, "ablated_correct": ablated_correct,
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

    all_results = []
    for model_key, model_path in MODEL_CONFIGS.items():
        results = run_model_control(model_key, model_path, targets_by_model[model_key], test_sample, device)
        all_results.extend(results)

        n_override = sum(1 for r in results if r["group"] == "override_cases")
        n_flip_correct = sum(1 for r in results if r["group"] == "override_cases" and r["flipped_to_correct"])
        n_faithful = sum(1 for r in results if r["group"] == "faithful_control_cases")
        n_flip_incorrect = sum(1 for r in results if r["group"] == "faithful_control_cases" and r["flipped_to_incorrect"])

        print(f"\n{model_key.upper()} RANDOM CONTROL SUMMARY:")
        print(f"  Override cases: {n_flip_correct}/{n_override} flipped to CORRECT (random heads)")
        print(f"  Faithful cases: {n_flip_incorrect}/{n_faithful} flipped to INCORRECT (random heads)")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
