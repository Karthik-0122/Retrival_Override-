"""
Steering-vector screen (fast, small-sample) -- targets Gemma's top DLA-
flagged heads specifically (37/12, 34/14, 37/14, 36/0, 38/14, 29/14,
40/14, 33/7), not the whole 96-head block. This sits deliberately
between "one head" (may be too narrow) and "96 heads" (already shown
to fail -- too diluted by redundant components).

DIFFERENT INTERVENTION TYPE than everything tried earlier tonight:
instead of DELETING a component's contribution (ablation/mean-
replacement, which destroys whatever else it does too), this ADDS a
direction vector -- computed as (mean faithful-case activation) minus
(mean override-case activation) -- at a chosen magnitude. This nudges
the representation toward "looks more like a faithful case" without
deleting the component's other functions. Tests several magnitudes,
since steering strength matters (too weak = no effect, too strong =
out-of-distribution breakdown -- same lesson from yesterday's causal-
tracing noise-scale tuning).

Includes a depth-matched random-head control at the SAME magnitudes,
built in from the start.

Uses the single-token gold logprob metric (cheap, one forward pass) for
this screen -- full-answer scoring is for a confirmation run later, if
this screen shows anything worth confirming.

Run from repo root:
    python Tests/steering_screen.py

Requires:
    data/final/Phase_04/ablation_test_sample.json
    data/final/Phase_02/analysis_dataset.jsonl
    data/final/Phase_02/retrieval_results.jsonl

Writes:
    data/final/Phase_04/steering_screen_results.json
"""

import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

MODEL_PATH = "/home/models/gemma-2-9b"
HEAD_DIM = 256
NUM_LAYERS = 42
NUM_HEADS = 16

DATASET_PATH = "data/final/Phase_02/analysis_dataset.jsonl"
RETRIEVAL_PATH = "data/final/Phase_02/retrieval_results.jsonl"
TEST_SAMPLE_PATH = "data/final/Phase_04/ablation_test_sample.json"
OUT_PATH = "data/final/Phase_04/steering_screen_results.json"

# Head selection is now DYNAMIC, not hardcoded -- reads from whichever DLA
# result is most trustworthy, so this never silently runs on a stale or
# unconfirmed head list again. Prefers the bf16 (full-precision) DLA run
# if it exists, since that's the one checking whether 4-bit quantization
# was distorting which heads looked important; falls back to the 4-bit
# run only if bf16 hasn't been run yet.
DLA_BF16_PATH = "data/final/Phase_04/dla_results_bf16.json"
DLA_4BIT_PATH = "data/final/Phase_04/dla_results.json"
N_STEERING_TARGETS = 8


def load_top_negative_heads(model_key="gemma", n=N_STEERING_TARGETS):
    source_path = None
    if Path(DLA_BF16_PATH).exists():
        source_path = DLA_BF16_PATH
        print(f"Using BF16 (full-precision) DLA results: {source_path}")
    elif Path(DLA_4BIT_PATH).exists():
        source_path = DLA_4BIT_PATH
        print(f"WARNING: no bf16 DLA results found -- falling back to 4-bit DLA results.")
        print(f"  These may be distorted by quantization noise (see the self-check discrepancy")
        print(f"  in that run). Run Tests/direct_logit_attribution_bf16.py first if possible.")
    else:
        raise FileNotFoundError(
            "No DLA results found at either path. Run Tests/direct_logit_attribution.py "
            "(or the bf16 version) before running the steering screen -- it needs a real "
            "head list to target, not a guess."
        )

    data = json.load(open(source_path))
    contribs = data[model_key]["contributions"]
    ranked = sorted(contribs.items(), key=lambda x: x[1])[:n]
    heads = [tuple(map(int, k.split("_"))) for k, _ in ranked]
    print(f"Top {n} negative heads from {source_path}:")
    for (l, h), (_, v) in zip(heads, ranked):
        print(f"  layer {l:2d} head {h:2d}: {v:+.4f}")
    return heads


REAL_TARGET_HEADS = load_top_negative_heads()

N_DIRECTION_QUERIES = 40  # queries used to compute the faithful-minus-override direction, per class
N_SCREEN_QUERIES = 20     # small sample for the screen itself
MAGNITUDES = [0.0, 2.0, 5.0, 10.0, 20.0]  # scalar multiples of the raw mean-difference vector
DEPTH_WINDOW = 5
SEED = 3141

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


def get_gold_first_token_id(tokenizer, gold_answer_text):
    if not gold_answer_text:
        return None
    ids = tokenizer(" " + gold_answer_text, add_special_tokens=False).input_ids
    return ids[0] if ids else None


def pick_random_heads(n, exclude_heads):
    rng = random.Random(SEED)
    real_layers = sorted({l for l, _ in exclude_heads})
    exclude_set = set(exclude_heads)
    all_candidates = [(l, h) for l in range(NUM_LAYERS) for h in range(NUM_HEADS)
                       if (l, h) not in exclude_set
                       and any(abs(l - rl) <= DEPTH_WINDOW for rl in real_layers)]
    rng.shuffle(all_candidates)
    return all_candidates[:n]


class SteeringHook:
    """Adds magnitude * direction to a specific head's pre-o_proj slice,
    instead of replacing it (the key difference from AblationHook used
    everywhere else tonight)."""
    def __init__(self, head_idx, head_dim, direction, magnitude):
        self.head_idx = head_idx
        self.head_dim = head_dim
        self.direction = direction
        self.magnitude = magnitude
        self.armed = False

    def pre_hook(self, module, args):
        if not self.armed or self.magnitude == 0.0:
            return None
        hidden_states = args[0]
        h_start, h_end = self.head_idx * self.head_dim, (self.head_idx + 1) * self.head_dim
        patched = hidden_states.clone()
        patched[:, -1, h_start:h_end] += (self.magnitude * self.direction).to(patched.device, patched.dtype)
        return (patched,) + args[1:]

    def arm(self):
        self.armed = True

    def disarm(self):
        self.armed = False

    def register(self, layer):
        return layer.self_attn.o_proj.register_forward_pre_hook(self.pre_hook)


def compute_direction_means(model, tokenizer, target_heads, device):
    """Returns dict {(layer, head): direction_vector} = mean(faithful) - mean(override),
    computed from queries NOT in the test sample."""
    records = load_jsonl(DATASET_PATH)
    retrieval = {r["query_id"]: r for r in load_jsonl(RETRIEVAL_PATH)}
    test_sample = json.load(open(TEST_SAMPLE_PATH))
    exclude_ids = {c["query_id"] for c in test_sample["gemma"]["faithful_control_cases"]}
    exclude_ids |= {c["query_id"] for c in test_sample["gemma"]["override_cases"]}

    def pool_for(label):
        p = [r for r in records if r.get("gemma_label") == label
             and r.get("source_category") != "confiqa"
             and r["query_id"] not in exclude_ids][:N_DIRECTION_QUERIES]
        for r in p:
            ret = retrieval.get(r["query_id"])
            r["retrieved_passages"] = ret["retrieved_passages"] if ret else []
        return p

    faithful_pool = pool_for("faithful")
    override_pool = pool_for("override")

    layer_indices = sorted({l for l, _ in target_heads})
    captured = {}

    def make_hook(li):
        def hook(module, args):
            captured[li] = args[0][0, -1, :].detach().float().cpu()
            return None
        return hook

    handles = [model.model.layers[li].self_attn.o_proj.register_forward_pre_hook(make_hook(li))
               for li in layer_indices]

    def get_mean_activations(pool, desc):
        sums = {k: torch.zeros(HEAD_DIM) for k in target_heads}
        counts = {k: 0 for k in target_heads}
        for r in tqdm(pool, desc=desc):
            if not r.get("retrieved_passages"):
                continue
            prompt = build_prompt(r["retrieved_passages"], r["question"])
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(device)
            captured.clear()
            with torch.no_grad():
                model(input_ids)
            for (li, hi) in target_heads:
                if li not in captured:
                    continue
                h_start, h_end = hi * HEAD_DIM, (hi + 1) * HEAD_DIM
                sums[(li, hi)] += captured[li][h_start:h_end]
                counts[(li, hi)] += 1
        return {k: sums[k] / max(counts[k], 1) for k in target_heads}

    faithful_means = get_mean_activations(faithful_pool, "Computing faithful means")
    override_means = get_mean_activations(override_pool, "Computing override means")

    for h in handles:
        h.remove()

    return {k: faithful_means[k] - override_means[k] for k in target_heads}


def run_screen(model, tokenizer, hooks, magnitude, queries, dataset_by_id, retrieval_by_id, device):
    for h in hooks:
        h.magnitude = magnitude

    n_flip = n_total = 0
    for q in queries:
        qid = q["query_id"]
        ret = retrieval_by_id.get(qid)
        row = dataset_by_id.get(qid)
        if not ret or not row or not ret.get("retrieved_passages"):
            continue
        prompt = build_prompt(ret["retrieved_passages"], q["question"])
        gold_id = get_gold_first_token_id(tokenizer, row.get("gold_answer_text"))
        if gold_id is None:
            continue

        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(device)

        for h in hooks:
            h.disarm()
        with torch.no_grad():
            baseline_logits = model(input_ids).logits[0, -1, :]
        baseline_logprob = torch.log_softmax(baseline_logits, dim=-1)[gold_id].item()

        for h in hooks:
            h.arm()
        with torch.no_grad():
            steered_logits = model(input_ids).logits[0, -1, :]
        for h in hooks:
            h.disarm()
        steered_logprob = torch.log_softmax(steered_logits, dim=-1)[gold_id].item()

        n_total += 1
        if steered_logprob > baseline_logprob + 0.5:  # meaningful improvement, not just fp noise
            n_flip += 1

    return n_flip, n_total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=QUANT_CONFIG, device_map=device, attn_implementation="eager",
    )
    model.eval()

    random_target_heads = pick_random_heads(len(REAL_TARGET_HEADS), REAL_TARGET_HEADS)
    print(f"Real target heads: {REAL_TARGET_HEADS}")
    print(f"Random control heads (depth-matched): {random_target_heads}")

    all_target_heads = REAL_TARGET_HEADS + random_target_heads
    directions = compute_direction_means(model, tokenizer, all_target_heads, device)

    real_hooks, real_handles = [], []
    for (l, h) in REAL_TARGET_HEADS:
        hook = SteeringHook(h, HEAD_DIM, directions[(l, h)], 0.0)
        real_handles.append(hook.register(model.model.layers[l]))
        real_hooks.append(hook)

    control_hooks, control_handles = [], []
    for (l, h) in random_target_heads:
        hook = SteeringHook(h, HEAD_DIM, directions[(l, h)], 0.0)
        control_handles.append(hook.register(model.model.layers[l]))
        control_hooks.append(hook)

    dataset_records = load_jsonl(DATASET_PATH)
    retrieval_records = load_jsonl(RETRIEVAL_PATH)
    dataset_by_id = {r["query_id"]: r for r in dataset_records}
    retrieval_by_id = {r["query_id"]: r for r in retrieval_records}
    test_sample = json.load(open(TEST_SAMPLE_PATH))
    screen_queries = test_sample["gemma"]["override_cases"][:N_SCREEN_QUERIES]

    print(f"\n{'Magnitude':>10} {'REAL flips':>12} {'CONTROL flips':>14}")
    results = []
    for mag in MAGNITUDES:
        real_flip, real_total = run_screen(model, tokenizer, real_hooks, mag, screen_queries,
                                             dataset_by_id, retrieval_by_id, device)
        control_flip, control_total = run_screen(model, tokenizer, control_hooks, mag, screen_queries,
                                                    dataset_by_id, retrieval_by_id, device)
        print(f"{mag:>10.1f} {f'{real_flip}/{real_total}':>12} {f'{control_flip}/{control_total}':>14}")
        results.append({"magnitude": mag, "real_flip": real_flip, "real_total": real_total,
                         "control_flip": control_flip, "control_total": control_total})

    for h in real_handles + control_handles:
        h.remove()

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"real_heads": REAL_TARGET_HEADS, "control_heads": random_target_heads, "results": results}, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    print("\nLook for a magnitude where REAL flips clearly exceed CONTROL flips.")
    print("If none exists, or REAL and CONTROL track each other, this doesn't")
    print("support a specific steering effect at this head set.")


if __name__ == "__main__":
    main()
