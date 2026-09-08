"""
Phase 4, joint intervention (the properly-scoped follow-up after tonight's
attention-only and MLP-only tests both came back not-significant against
a random control).

TWO FIXES vs tonight's attempts, based on how ReDeEP (ICLR 2025) achieved
a validated result on a similar question:

  1. JOINT intervention: ablates the validated attention heads AND the
     validated MLP layers TOGETHER, in the same run -- not as two
     separate tests. If the mechanism involves both working together,
     testing either alone (what happened tonight) can miss the effect
     entirely.

  2. SENSITIVE metric: instead of binary "did the exact generated text
     flip from wrong to right" (which tonight's own sanity check showed
     can miss a 100,000-point logit shift if it doesn't cross the #1
     token boundary), this tracks the actual shift in P(gold answer)
     via a single forward pass -- a continuous number that captures
     partial, real movement even when it doesn't flip the top prediction.
     This is much closer to ReDeEP's NLL-difference approach than
     tonight's flip-counting was.

     Bonus: single forward passes are far cheaper than full .generate()
     calls, so this runs faster despite testing more per query.

Run from repo root:
    python Tests/joint_intervention.py

Requires:
    data/final/Phase_04/ablation_targets_escalated.json
    data/final/Phase_04/faithful_means_escalated.pt
    data/final/Phase_03/phase3_length_controlled.json
    data/final/Phase_04/ablation_test_sample.json
    data/final/Phase_02/analysis_dataset.jsonl
    data/final/Phase_02/retrieval_results.jsonl

Writes:
    data/final/Phase_04/joint_intervention_results.jsonl
"""

import json
import random
import sys
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation_utils import AblationHook
from mlp_ablation_utils import MLPAblationHook

ATTN_TARGETS_PATH = "data/final/Phase_04/ablation_targets_escalated.json"
ATTN_MEANS_PATH = "data/final/Phase_04/faithful_means_escalated.pt"
MLP_LAYERS_PATH = "data/final/Phase_03/phase3_length_controlled.json"
TEST_SAMPLE_PATH = "data/final/Phase_04/ablation_test_sample.json"
DATASET_PATH = "data/final/Phase_02/analysis_dataset.jsonl"
RETRIEVAL_PATH = "data/final/Phase_02/retrieval_results.jsonl"
OUT_PATH = "data/final/Phase_04/joint_intervention_results.jsonl"

MODEL_CONFIGS = {
    "gemma": "/home/models/gemma-2-9b",
    "llama": "/home/models/Llama-3.1-8B",
}
HEAD_DIMS = {"gemma": 256, "llama": 128}
NUM_LAYERS = {"gemma": 42, "llama": 32}
NUM_HEADS = {"gemma": 16, "llama": 32}
N_MEANS_QUERIES = 60
SEED = 2024  # independent draw, distinct from every other seed used tonight

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def build_prompt(passages, question):
    passage_text = "\n\n".join(passages)
    return f"{passage_text}\n\nQuestion: {question}\nAnswer:"


def get_gold_token_ids(tokenizer, gold_answer_text):
    """Full multi-token gold answer, not just the first token -- a richer,
    less noisy signal than a single-token check, closer to how ReDeEP
    judges whole-response truthfulness rather than one token."""
    if not gold_answer_text:
        return None
    ids = tokenizer(" " + gold_answer_text, add_special_tokens=False).input_ids
    return ids if ids else None


def get_validated_attn_targets(model_key):
    return json.load(open(ATTN_TARGETS_PATH))[model_key]


def get_validated_mlp_layers(model_key):
    data = json.load(open(MLP_LAYERS_PATH))
    return sorted(r["layer"] for r in data[model_key] if r.get("fdr_significant"))


DEPTH_WINDOW = 5  # random controls must be sampled within this many layers of
# a real target, not uniformly across the whole network -- otherwise "random"
# can land in a totally different depth region (e.g. Llama's real targets at
# layers 2-4 vs a control drawn from layers 12/24/31), which confounds
# "is this specific region special" with "is this depth range generically
# more load-bearing." Found via inspecting the first joint_intervention.py run.


def _depth_matched_pool(candidates, real_positions, window):
    real_set = set(real_positions)
    return [c for c in candidates
            if c not in real_set and any(abs(c - r) <= window for r in real_set)]


def pick_random_attn_targets(model_key, n, exclude_targets):
    rng = random.Random(SEED)
    real_layers = sorted({t["layer"] for t in exclude_targets})
    exclude_set = {(t["layer"], t["head"]) for t in exclude_targets}
    all_layers = list(range(NUM_LAYERS[model_key]))
    depth_matched_layers = _depth_matched_pool(all_layers, real_layers, DEPTH_WINDOW)
    if not depth_matched_layers:
        depth_matched_layers = all_layers  # fallback, shouldn't happen with window=5

    pool = [(l, h) for l in depth_matched_layers for h in range(NUM_HEADS[model_key])
            if (l, h) not in exclude_set]
    rng.shuffle(pool)
    return [{"layer": l, "head": h} for l, h in pool[:n]]


def pick_random_mlp_layers(model_key, n, exclude_layers):
    rng = random.Random(SEED + 1)
    all_layers = list(range(NUM_LAYERS[model_key]))
    depth_matched = _depth_matched_pool(all_layers, sorted(exclude_layers), DEPTH_WINDOW)
    if not depth_matched:
        depth_matched = [l for l in all_layers if l not in exclude_layers]
    rng.shuffle(depth_matched)
    return sorted(depth_matched[:n])


def compute_attn_means(model, tokenizer, targets, head_dim, faithful_pool, device):
    sums, counts = {}, {}
    for t in targets:
        key = (t["layer"], t["head"])
        sums[key] = torch.zeros(head_dim, dtype=torch.float32)
        counts[key] = 0

    layer_indices = sorted({t["layer"] for t in targets})
    captured = {}

    def make_hook(li):
        def hook(module, args):
            captured[li] = args[0][0, -1, :].detach().float().cpu()
            return None
        return hook

    handles = [model.model.layers[li].self_attn.o_proj.register_forward_pre_hook(make_hook(li))
               for li in layer_indices]

    for r in tqdm(faithful_pool, desc="Computing attn means"):
        if not r.get("retrieved_passages"):
            continue
        prompt = build_prompt(r["retrieved_passages"], r["question"])
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(device)
        captured.clear()
        with torch.no_grad():
            model(input_ids)
        for t in targets:
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


def compute_mlp_means(model, tokenizer, layers, faithful_pool, device):
    sums = {l: None for l in layers}
    counts = {l: 0 for l in layers}
    captured = {}

    def make_hook(li):
        def hook(module, input, output):
            captured[li] = output[0, -1, :].detach().float().cpu()
            return output
        return hook

    handles = [model.model.layers[l].mlp.register_forward_hook(make_hook(l)) for l in layers]

    for r in tqdm(faithful_pool, desc="Computing MLP means"):
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


def get_gold_logprob(model, tokenizer, prompt, gold_token_ids, device):
    """Total log-probability of the FULL gold answer (all its tokens).

    IMPORTANT: does this via one forward pass PER gold token, each time
    feeding (prompt + gold tokens seen so far) and reading the prediction
    for the next gold token at position -1. This is deliberately NOT a
    single all-at-once forward pass over (prompt + full gold answer) --
    the ablation hooks (AblationHook, MLPAblationHook) both patch
    position -1 only, which is correct when -1 is "the position about to
    predict the next token" (true at every step here) but would be WRONG
    in an all-at-once pass, where -1 would be the LAST gold token,
    leaving the prompt and earlier gold positions completely unablated.
    Gold answers here are short (typically 1-4 tokens), so this costs at
    most a few extra forward passes per query, not a real slowdown."""
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=4096 - len(gold_token_ids)).input_ids.to(device)

    total_logprob = 0.0
    current_ids = prompt_ids
    for tok_id in gold_token_ids:
        with torch.no_grad():
            logits = model(current_ids).logits[0, -1, :]
        logprobs = torch.log_softmax(logits, dim=-1)
        total_logprob += logprobs[tok_id].item()
        current_ids = torch.cat([current_ids, torch.tensor([[tok_id]], device=device)], dim=1)

    return total_logprob


def run_model(model_key, model_path, device):
    print(f"\n{'='*70}\nLoading {model_key} ({model_path})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=QUANT_CONFIG, device_map=device, attn_implementation="eager",
    )
    model.eval()
    head_dim = HEAD_DIMS[model_key]

    real_attn_targets = get_validated_attn_targets(model_key)
    real_mlp_layers = get_validated_mlp_layers(model_key)
    random_attn_targets = pick_random_attn_targets(model_key, len(real_attn_targets), real_attn_targets)
    random_mlp_layers = pick_random_mlp_layers(model_key, len(real_mlp_layers), set(real_mlp_layers))

    print(f"  Real: {len(real_attn_targets)} attn heads, {len(real_mlp_layers)} MLP layers -- {real_mlp_layers}")
    print(f"  Control: {len(random_attn_targets)} random attn heads, {len(random_mlp_layers)} random MLP layers -- {random_mlp_layers}")

    dataset_records = load_jsonl(DATASET_PATH)
    retrieval_records = load_jsonl(RETRIEVAL_PATH)
    dataset_by_id = {r["query_id"]: r for r in dataset_records}
    retrieval_by_id = {r["query_id"]: r for r in retrieval_records}

    test_sample = json.load(open(TEST_SAMPLE_PATH))
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

    all_attn_targets_needed = real_attn_targets + random_attn_targets
    all_mlp_layers_needed = sorted(set(real_mlp_layers) | set(random_mlp_layers))

    attn_means = compute_attn_means(model, tokenizer, all_attn_targets_needed, head_dim, faithful_pool, device)
    mlp_means = compute_mlp_means(model, tokenizer, all_mlp_layers_needed, faithful_pool, device)

    def make_hooks(attn_targets, mlp_layers):
        hooks, handles = [], []
        for t in attn_targets:
            key = (t["layer"], t["head"])
            if key not in attn_means:
                continue
            h = AblationHook(t["head"], head_dim, attn_means[key])
            handles.append(h.register(model.model.layers[t["layer"]]))
            hooks.append(h)
        for l in mlp_layers:
            if l not in mlp_means:
                continue
            h = MLPAblationHook(mlp_means[l])
            handles.append(h.register(model.model.layers[l]))
            hooks.append(h)
        return hooks, handles

    real_hooks, real_handles = make_hooks(real_attn_targets, real_mlp_layers)
    control_hooks, control_handles = make_hooks(random_attn_targets, random_mlp_layers)
    print(f"  {len(real_hooks)} real hooks, {len(control_hooks)} control hooks registered (both sets present; armed independently)")

    results = []
    for group_name in ["override_cases", "faithful_control_cases"]:
        queries = test_sample[model_key][group_name]
        if group_name == "override_cases":
            # Filter to the CLEAREST override cases using PCS gap already
            # computed in Phase 2 -- the closest legitimate substitute for
            # human-confirmed hallucination spans (which we don't have)
            # without new annotation work. A bigger gap between parametric
            # confidence and correct-context confidence means the model
            # more unambiguously chose memory over evidence -- the
            # cleanest test cases for a causal effect, if one exists.
            pcs_key = f"pcs_{model_key}"
            scored = []
            for q in queries:
                row = dataset_by_id.get(q["query_id"])
                pcs = row.get(pcs_key) if row else None
                scored.append((abs(pcs) if pcs is not None else 0.0, q))
            scored.sort(key=lambda x: -x[0])
            queries = [q for _, q in scored[:max(1, int(len(queries) * 0.7))]]
            print(f"  Filtered to {len(queries)}/{len(test_sample[model_key][group_name])} "
                  f"clearest override cases by |PCS| gap")
        for q in tqdm(queries, desc=f"Joint intervention ({model_key}, {group_name})"):
            qid = q["query_id"]
            ret = retrieval_by_id.get(qid)
            row = dataset_by_id.get(qid)
            if not ret or not row or not ret.get("retrieved_passages"):
                continue
            prompt = build_prompt(ret["retrieved_passages"], q["question"])
            gold_token_ids = get_gold_token_ids(tokenizer, row.get("gold_answer_text"))
            if gold_token_ids is None:
                continue

            for h in real_hooks + control_hooks:
                h.disarm()
            baseline_logprob = get_gold_logprob(model, tokenizer, prompt, gold_token_ids, device)

            for h in real_hooks:
                h.arm()
            real_logprob = get_gold_logprob(model, tokenizer, prompt, gold_token_ids, device)
            for h in real_hooks:
                h.disarm()

            for h in control_hooks:
                h.arm()
            control_logprob = get_gold_logprob(model, tokenizer, prompt, gold_token_ids, device)
            for h in control_hooks:
                h.disarm()

            results.append({
                "query_id": qid, "model": model_key, "group": group_name,
                "baseline_logprob": baseline_logprob,
                "real_logprob": real_logprob,
                "control_logprob": control_logprob,
                "real_effect": real_logprob - baseline_logprob,
                "control_effect": control_logprob - baseline_logprob,
            })

    for h in real_handles + control_handles:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return results


def analyze(results, model_key, group_name):
    rows = [r for r in results if r["model"] == model_key and r["group"] == group_name]
    if len(rows) < 2:
        print(f"  {model_key} {group_name}: not enough rows to test")
        return
    real_effects = [r["real_effect"] for r in rows]
    control_effects = [r["control_effect"] for r in rows]

    mean_real = sum(real_effects) / len(real_effects)
    mean_control = sum(control_effects) / len(control_effects)

    t_stat, p_value = stats.ttest_rel(real_effects, control_effects)

    print(f"\n  {model_key} {group_name} (n={len(rows)}):")
    print(f"    mean real effect on gold logprob:    {mean_real:+.4f}")
    print(f"    mean control effect on gold logprob: {mean_control:+.4f}")
    print(f"    paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
    if p_value < 0.05:
        direction = "REAL > CONTROL" if mean_real > mean_control else "CONTROL > REAL"
        print(f"    -> SIGNIFICANT at p<0.05 ({direction})")
    else:
        print(f"    -> NOT significant at p<0.05")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = []
    for model_key, model_path in MODEL_CONFIGS.items():
        results = run_model(model_key, model_path, device)
        all_results.extend(results)

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {OUT_PATH}")

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS (paired t-test: real effect vs control effect)")
    print("=" * 70)
    for model_key in MODEL_CONFIGS:
        print(f"\n{model_key.upper()}")
        analyze(all_results, model_key, "override_cases")
        analyze(all_results, model_key, "faithful_control_cases")


if __name__ == "__main__":
    main()
