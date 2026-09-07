"""
Phase 4, T22a: Compute faithful-case mean activations.

For each ablation target (layer, head) selected in T17, computes the
MEAN o_proj-input activation for that head, averaged across a batch of
real FAITHFUL-case queries (run WITH retrieved context -- same prompt
format as the original ROI extraction, since that's the condition
override/faithful behavior was actually measured in).

Uses a SEPARATE pool of faithful queries from the ones in T20's
faithful_control_cases, to avoid computing the replacement value from
the exact same queries used later to test whether ablation breaks
faithful behavior (that would be circular).

Run from repo root:
    python Tests/Phase_04/compute_faithful_means.py

Requires:
    data/final/Phase_04/ablation_targets.json  (T17)
    data/final/Phase_04/ablation_test_sample.json  (T20, to exclude those query_ids)
    data/final/analysis_dataset.jsonl
    data/final/retrieval_results.jsonl

Writes:
    data/final/Phase_04/faithful_means.pt   (torch tensors, keyed by "model_layer_head")
"""

import json
import sys
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TARGETS_PATH = "data/final/Phase_04/ablation_targets_escalated.json"
TEST_SAMPLE_PATH = "data/final/Phase_04/ablation_test_sample.json"
DATASET_PATH = "data/final/Phase_02/analysis_dataset.jsonl"
RETRIEVAL_PATH = "data/final/Phase_02/retrieval_results.jsonl"
OUT_PATH = "data/final/Phase_04/faithful_means_escalated.pt"

MODEL_CONFIGS = {
    "gemma": "/home/models/gemma-2-9b",
    "llama": "/home/models/Llama-3.1-8B",
}
HEAD_DIMS = {"gemma": 256, "llama": 128}

N_QUERIES_FOR_MEAN = 60  # how many faithful queries to average over, per model
QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def select_faithful_pool(model, exclude_ids):
    records = load_jsonl(DATASET_PATH)
    label_key = f"{model}_label"
    pool = [r for r in records if r.get(label_key) == "faithful" and r["query_id"] not in exclude_ids]
    return pool[:N_QUERIES_FOR_MEAN]


def merge_passages(records, retrieval_records):
    retrieval_by_id = {r["query_id"]: r for r in retrieval_records}
    for r in records:
        retrieval = retrieval_by_id.get(r["query_id"])
        r["retrieved_passages"] = retrieval["retrieved_passages"] if retrieval else []
    return records


def compute_means_for_model(model_key, model_path, targets, faithful_pool, device):
    print(f"\nLoading {model_key} ({model_path})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=QUANT_CONFIG, device_map=device,
        attn_implementation="eager",
    )
    model.eval()

    head_dim = HEAD_DIMS[model_key]

    # accumulate sums per (layer, head), then divide by count at the end
    sums = {}
    counts = {}
    for t in targets:
        key = (t["layer"], t["head"])
        sums[key] = torch.zeros(head_dim, dtype=torch.float32)
        counts[key] = 0

    # one capture hook per unique target layer, reused across queries
    layer_indices = sorted({t["layer"] for t in targets})
    captured_this_pass = {}

    def make_hook(layer_idx):
        def hook(module, args):
            captured_this_pass[layer_idx] = args[0][0, -1, :].detach().float().cpu()
            return None
        return hook

    handles = []
    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]
        h = layer.self_attn.o_proj.register_forward_pre_hook(make_hook(layer_idx))
        handles.append(h)

    for r in tqdm(faithful_pool, desc=f"Computing means ({model_key})"):
        passages = r.get("retrieved_passages", [])
        if not passages:
            continue
        passage_text = "\n\n".join(passages)
        prompt = f"{passage_text}\n\nQuestion: {r['question']}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = inputs.input_ids.to(device)

        captured_this_pass.clear()
        with torch.no_grad():
            model(input_ids)

        for t in targets:
            layer_idx, head_idx = t["layer"], t["head"]
            if layer_idx not in captured_this_pass:
                continue
            full_vec = captured_this_pass[layer_idx]  # shape (num_heads * head_dim,)
            h_start = head_idx * head_dim
            h_end = h_start + head_dim
            sums[(layer_idx, head_idx)] += full_vec[h_start:h_end]
            counts[(layer_idx, head_idx)] += 1

    for h in handles:
        h.remove()

    means = {}
    for key, total in sums.items():
        n = counts[key]
        if n == 0:
            print(f"  WARNING: no successful captures for layer {key[0]} head {key[1]}, skipping")
            continue
        means[key] = total / n

    print(f"Done with {model_key}. Computed means for {len(means)}/{len(targets)} targets "
          f"(averaged over up to {N_QUERIES_FOR_MEAN} queries).")

    del model
    torch.cuda.empty_cache()
    return means


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    targets_by_model = json.load(open(TARGETS_PATH))
    test_sample = json.load(open(TEST_SAMPLE_PATH))

    all_means = {}
    for model_key, model_path in MODEL_CONFIGS.items():
        targets = targets_by_model[model_key]
        exclude_ids = {c["query_id"] for c in test_sample[model_key]["faithful_control_cases"]}
        exclude_ids |= {c["query_id"] for c in test_sample[model_key]["override_cases"]}

        pool = select_faithful_pool(model_key, exclude_ids)
        retrieval_records = load_jsonl(RETRIEVAL_PATH)
        pool = merge_passages(pool, retrieval_records)
        print(f"{model_key}: {len(pool)} faithful queries available for computing means "
              f"(excluding test-sample queries)")

        means = compute_means_for_model(model_key, model_path, targets, pool, device)
        for (layer, head), vec in means.items():
            all_means[f"{model_key}_{layer}_{head}"] = vec

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_means, OUT_PATH)
    print(f"\nWrote {OUT_PATH} ({len(all_means)} target means)")


if __name__ == "__main__":
    main()
