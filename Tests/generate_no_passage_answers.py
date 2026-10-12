"""
Experiment 1 prerequisite: what does each model answer WITHOUT the passage?

The guide's critical rule: F1 (genuine parametric override) may only be
assigned when there is evidence the answer came from the model's own memory,
not merely because it is wrong. The cleanest evidence is the model's
closed-book answer: if the with-passage answer matches what the model says
with no passage at all, the passage did not change its mind.

Runs only on apparent failures (original or full-answer label 'override',
non-ConFiQA) for one model at a time. Greedy decoding, same "Question/Answer"
format as Phase 2 minus the passages. ~1-2k short generations per model:
roughly 10-20 minutes on an A5000.

Run from repo root, once per model:
    python Tests/generate_no_passage_answers.py --model gemma
    python Tests/generate_no_passage_answers.py --model llama

Resumable: already-written query_ids are skipped.

Writes (appends):
    data/final/Phase_05/no_passage_answers.jsonl
      {"query_id", "model", "noctx_answer"}
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATHS = {"gemma": "google/gemma-2-9b", "llama": "meta-llama/Llama-3.1-8B"}
MAX_NEW_TOKENS = 32


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def build_prompt_no_passage(question):
    # Phase 2 prompt is "<passages>\n\nQuestion: {q}\nAnswer:" -- same, minus passages.
    return f"Question: {question}\nAnswer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PATHS), required=True)
    ap.add_argument("--model-path", default=None, help="override HF id / local path")
    ap.add_argument("--dataset", default="data/final/Phase_02/analysis_dataset_relabeled.jsonl")
    ap.add_argument("--out", default="data/final/Phase_05/no_passage_answers.jsonl")
    ap.add_argument("--use-4bit", action="store_true", help="use on GPUs under ~24GB")
    ap.add_argument("--include-faithful", action="store_true",
                    help="also run faithful cases (useful later for Exp 3 controls)")
    args = ap.parse_args()
    m = args.model

    records = load_jsonl(args.dataset)
    todo = []
    for r in records:
        if r.get("source_category") == "confiqa":
            continue
        apparent_fail = r.get(f"{m}_label") == "override" or r.get(f"{m}_label_full") == "override"
        if apparent_fail or (args.include_faithful and r.get(f"{m}_label") == "faithful"):
            todo.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(x["query_id"], x["model"]) for x in load_jsonl(out)}
    todo = [r for r in todo if (r["query_id"], m) not in done]
    print(f"{m}: {len(todo)} questions to answer without the passage ({len(done)} already done)")
    if not todo:
        return

    path = args.model_path or MODEL_PATHS[m]
    tok = AutoTokenizer.from_pretrained(path)
    kw = {"device_map": "cuda"}
    if args.use_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(path, **kw).eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    with open(out, "a") as f:
        for r in tqdm(todo):
            ids = tok(build_prompt_no_passage(r["question"]), return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=pad_id)
            text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
            text = text.split("\n")[0].strip()          # first line only, like a short answer
            f.write(json.dumps({"query_id": r["query_id"], "model": m, "noctx_answer": text}) + "\n")
            f.flush()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()