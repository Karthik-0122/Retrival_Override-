"""
Diagnostic follow-up to run_ablation.py's null result.

The sanity check on a toy prompt showed massive logit change from
ablation (diff=100k+) but no argmax flip -- greedy decoding only cares
about rank-1, so a large but "diffuse" perturbation can leave the #1
token unchanged even while meaningfully reshaping the whole distribution.

This checks REAL override-case queries (with retrieved context, matching
the actual test setup) and looks at whether the GOLD answer's rank/logit
improves under ablation, even if it doesn't reach rank 1. This tells us
whether ablation is nudging in the right direction (just not far enough)
or not really engaging with the override-relevant computation at all.

Run from repo root:
    python Tests/diagnose_ablation_direction.py
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
N_QUERIES_TO_CHECK = 5


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def get_gold_first_token_ids(tokenizer, gold):
    """Get token id(s) for plausible gold answer strings, tried with a
    leading space (how they'd appear after 'Answer:')."""
    candidates = []
    if isinstance(gold, dict):
        if gold.get("value"):
            candidates.append(gold["value"])
        candidates.extend(gold.get("aliases") or [])
    elif isinstance(gold, str):
        s = gold.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                candidates.extend(json.loads(s))
            except json.JSONDecodeError:
                candidates.append(gold)
        else:
            candidates.append(gold)
    elif isinstance(gold, list):
        candidates.extend(gold)

    ids = []
    for c in candidates[:3]:  # just check first few candidates
        tok_ids = tokenizer(" " + c, add_special_tokens=False).input_ids
        if tok_ids:
            ids.append((c, tok_ids[0]))
    return ids


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
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
    dataset = {r["query_id"]: r for r in load_jsonl("data/final/Phase_02/analysis_dataset.jsonl")}
    retrieval = {r["query_id"]: r for r in load_jsonl("data/final/Phase_02/retrieval_results.jsonl")}

    override_queries = test_sample["gemma"]["override_cases"][:N_QUERIES_TO_CHECK]

    for q in override_queries:
        qid = q["query_id"]
        ret = retrieval.get(qid)
        row = dataset.get(qid)
        if not ret or not row or not ret.get("retrieved_passages"):
            continue

        passage_text = "\n\n".join(ret["retrieved_passages"])
        prompt = f"{passage_text}\n\nQuestion: {q['question']}\nAnswer:"
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).input_ids.to("cuda")

        gold_ids = get_gold_first_token_ids(tokenizer, row.get("gold_answers"))
        if not gold_ids:
            continue

        with torch.no_grad():
            baseline_logits = model(input_ids).logits[0, -1, :]
        for h in hooks:
            h.arm()
        with torch.no_grad():
            ablated_logits = model(input_ids).logits[0, -1, :]
        for h in hooks:
            h.disarm()

        baseline_probs = torch.softmax(baseline_logits, dim=-1)
        ablated_probs = torch.softmax(ablated_logits, dim=-1)
        baseline_rank_all = baseline_logits.argsort(descending=True)
        ablated_rank_all = ablated_logits.argsort(descending=True)

        print(f"Query {qid}: {q['question']!r}")
        print(f"  Baseline top token: {tokenizer.decode([baseline_logits.argmax()])!r}  "
              f"(p={baseline_probs.max().item():.4f})")
        print(f"  Ablated  top token: {tokenizer.decode([ablated_logits.argmax()])!r}  "
              f"(p={ablated_probs.max().item():.4f})")

        for gold_text, gold_id in gold_ids:
            b_rank = (baseline_rank_all == gold_id).nonzero().item()
            a_rank = (ablated_rank_all == gold_id).nonzero().item()
            b_p = baseline_probs[gold_id].item()
            a_p = ablated_probs[gold_id].item()
            direction = "IMPROVED" if a_rank < b_rank else ("WORSE" if a_rank > b_rank else "unchanged")
            print(f"  Gold candidate {gold_text!r}: rank {b_rank} -> {a_rank} "
                  f"(prob {b_p:.5f} -> {a_p:.5f})  [{direction}]")
        print()

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
