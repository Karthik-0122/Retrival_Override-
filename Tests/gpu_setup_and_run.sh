cat > /tmp/ro_fix.sh <<'RO_FIX'
#!/usr/bin/env bash
# Fix for: "Disabling PyTorch because PyTorch >= 2.6 is required but found 2.2.1"
#   1/4 updated scripts   2/4 transformers 4.x (works with this PyTorch)
#   3/4 GPU self-test     4/4 restart the run (smoke test, then the full run by itself)
set -euo pipefail
say() { echo; echo "========== $* =========="; }
die() { echo; echo "STOP: $*"; exit 1; }
pipi() { python -m pip install -q "$@" 2>/dev/null || python -m pip install -q --break-system-packages "$@"; }

P=""
for d in /workspace /home "$HOME" /root; do
  if [ -f "$d/Retrival_Override-/logs/run_session.sh" ]; then P="$d/Retrival_Override-"; break; fi
done
[ -n "$P" ] || die "project folder not found (looked for Retrival_Override-/logs/run_session.sh)"
cd "$P"
eval "$(grep '^export HF_HOME=' logs/run_session.sh || true)"     # same model cache as the run
echo "project: $P | model cache: ${HF_HOME:-default}"

say "1/4 updated scripts"
mkdir -p logs/backup && cp Tests/gpu_utils.py Tests/enrichment_pilot.py logs/backup/ 2>/dev/null || true
cat > Tests/gpu_utils.py <<'RO_END_OF_FILE'
"""
Shared utilities for the GPU scripts (llm_judge.py, enrichment_pilot.py).

  * model loading (bf16, or 4-bit with --use-4bit), left padding for batches
  * token-budget batching: prompts sorted by length and packed so that
    (longest prompt + new tokens) x batch size stays under a budget
  * out-of-memory safety: a batch that runs out of GPU memory is split in
    half and retried, so one long prompt cannot crash a multi-hour run
  * greedy generation that also returns per-token log-probabilities
    (first-token and mean log-prob = the model's confidence)
  * last-token label scoring for the judge (one forward pass, no sampling)
  * resumable JSONL output: every stage skips ids already written
"""

import gzip
import json
import math
import re
import string
import time
from pathlib import Path

import torch

MODEL_PATHS = {"gemma": "google/gemma-2-9b", "llama": "meta-llama/Llama-3.1-8B",
               "judge": "Qwen/Qwen2.5-7B-Instruct"}
_PUNCT = set(string.punctuation)
STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is", "was", "are",
             "were", "be", "by", "with", "as", "from", "that", "this", "it", "its", "what", "who",
             "when", "where", "which", "how", "why", "did", "does", "do", "whose", "whom"}


# ------------------------------------------------------------------ text
def normalize(t):
    t = "".join(ch for ch in (t or "").lower() if ch not in _PUNCT)
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    return " ".join(t.split())


def content_tokens(t):
    return [w for w in normalize(t).split() if w not in STOPWORDS]


def parse_gold(text_or_list):
    if isinstance(text_or_list, list):
        return [x for x in text_or_list if isinstance(x, str) and normalize(x)]
    s = (text_or_list or "").strip()
    if s.startswith("["):
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, str) and normalize(x)]
        except (json.JSONDecodeError, TypeError):
            pass
    return [s] if normalize(s) else []


def contains(hay, needle):
    n = normalize(needle)
    return bool(n) and f" {n} " in f" {normalize(hay)} "


def contains_any(hay, aliases):
    return any(contains(hay, a) for a in aliases)


def _stem(t):
    return t[:-1] if len(t) > 3 and t.endswith("s") else t


def answer_words(text, question):
    q = {_stem(t) for t in content_tokens(question)}
    return [_stem(t) for t in content_tokens(text) if _stem(t) not in q]


def containment_q(noctx, answer, question):
    """Share of the no-passage answer's own words (question words removed)
    that appear in the with-passage answer (the corrected measure)."""
    nt = answer_words(noctx, question)
    at = set(answer_words(answer, question))
    return sum(t in at for t in nt) / len(nt) if nt else 0.0


def first_line(text):
    return (text or "").split("\n")[0].strip()


# ------------------------------------------------------------------ jsonl io
def load_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def done_keys(path, key_fn):
    return {key_fn(r) for r in load_jsonl(path)}


# ------------------------------------------------------------------ models
def pick_dtype(path, use_4bit):
    """bf16 on Ampere or newer (as in every earlier phase). Older GPUs (Turing,
    Volta: T4, V100, RTX 20xx, Quadro RTX) have no bf16. Llama and Qwen run fine
    in fp16 there, but Gemma-2 can overflow in fp16 (NaN / garbage output), so
    Gemma falls back to fp32 when the GPU has room, and stops otherwise."""
    if not torch.cuda.is_available():
        return torch.float32
    major, minor = torch.cuda.get_device_capability()
    if major >= 8:
        return torch.bfloat16
    name = torch.cuda.get_device_name()
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"!! {name} (compute {major}.{minor}) has no bfloat16 support", flush=True)
    if "gemma" not in str(path).lower():
        print("!! using float16 for this model", flush=True)
        return torch.float16
    if use_4bit:
        raise SystemExit("Gemma-2 in 4-bit needs bf16 compute, which this GPU lacks -- use an Ampere or newer GPU.")
    if vram >= 44:
        print(f"!! Gemma-2 is unstable in float16 -> using float32 ({vram:.0f} GB VRAM; slower, results comparable)",
              flush=True)
        return torch.float32
    raise SystemExit(f"Gemma-2 needs bf16 (or ~40 GB for fp32); this GPU has neither ({vram:.0f} GB). "
                     "Use an Ampere or newer GPU (A5000, A6000, A40, L40S, A100, RTX 3090/4090).")


def load_model(path, use_4bit=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    on_gpu = torch.cuda.is_available()
    dtype = pick_dtype(path, use_4bit)
    kw = {"device_map": "cuda" if on_gpu else "cpu"}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
    else:
        import transformers
        from packaging.version import Version
        kw["dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"] = dtype
    if "gemma" in str(path).lower():
        kw["attn_implementation"] = "eager"        # Gemma-2 attention soft-capping; same as earlier phases
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(path, **kw).eval()
    except ValueError as e:                        # no `accelerate` installed: load, then move
        if "accelerate" not in str(e) or use_4bit:
            raise
        kw.pop("device_map")
        model = AutoModelForCausalLM.from_pretrained(path, **kw).eval().to("cuda" if on_gpu else "cpu")
    print(f"loaded {path} in {time.time() - t0:.0f}s ({'4-bit' if use_4bit else model.dtype}, "
          f"{'GPU' if on_gpu else 'CPU'})", flush=True)
    return model, tok


def budget_batches(lengths, budget, max_bs):
    """Index batches, longest first, with max_len * batch_size <= budget."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batch, cur = [], 0
    for i in order:
        new = max(cur, lengths[i])
        if batch and (new * (len(batch) + 1) > budget or len(batch) >= max_bs):
            yield batch
            batch, new = [], lengths[i]
        batch.append(i)
        cur = new
    if batch:
        yield batch


def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def run_safely(fn, items):
    """fn(list) -> list of results. On OOM, split the batch and retry."""
    try:
        return fn(items)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if not _is_oom(e) or len(items) == 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mid = len(items) // 2
        print(f"  [OOM] splitting a batch of {len(items)}", flush=True)
        return run_safely(fn, items[:mid]) + run_safely(fn, items[mid:])


# ------------------------------------------------------------------ generation
@torch.no_grad()
def _generate_batch(model, tok, prompts, max_new_tokens):
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    enc.pop("token_type_ids", None)               # some tokenizers add these; causal LMs reject them
    kw = dict(max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id,
              return_dict_in_generate=True, output_logits=True, temperature=None, top_p=None, top_k=None)
    try:
        out = model.generate(**enc, stop_strings=["\n"], tokenizer=tok, **kw)
    except (TypeError, ValueError):
        out = model.generate(**enc, **kw)
    new = out.sequences[:, enc["input_ids"].shape[1]:]
    T = len(out.logits)
    lps = torch.zeros(new.shape[0], T)
    for t in range(T):
        lp = torch.log_softmax(out.logits[t].float(), dim=-1)
        lps[:, t] = lp.gather(1, new[:, t:t + 1].to(lp.device)).squeeze(1).cpu()
    stop_ids = {tok.eos_token_id, tok.pad_token_id}
    res = []
    for b in range(new.shape[0]):
        ids = new[b].tolist()
        n = 0                                     # tokens that belong to the first line
        for t in range(len(ids)):
            if ids[t] in stop_ids:
                break
            if "\n" in tok.decode(ids[:t + 1], skip_special_tokens=True):
                break
            n = t + 1
        text = first_line(tok.decode(ids[:max(n, 1)], skip_special_tokens=True)) if n else ""
        ans_lp = lps[b, :n].tolist() if n else []
        res.append({"text": text, "n_tokens": n,
                    "first_logprob": float(lps[b, 0]),
                    "mean_logprob": float(sum(ans_lp) / len(ans_lp)) if ans_lp else float(lps[b, 0])})
    return res


def generate(model, tok, prompts, max_new_tokens=32, token_budget=8000, max_bs=32, desc="generate"):
    """Greedy first-line answers for all prompts, in input order."""
    lengths = [len(tok(p)["input_ids"]) + max_new_tokens for p in prompts]
    out = [None] * len(prompts)
    batches = list(budget_batches(lengths, token_budget, max_bs))
    t0 = time.time()
    for k, b in enumerate(batches, 1):
        res = run_safely(lambda idx: _generate_batch(model, tok, [prompts[i] for i in idx], max_new_tokens), b)
        for i, r in zip(b, res):
            out[i] = r
        if k % 10 == 0 or k == len(batches):
            done = sum(len(x) for x in batches[:k])
            print(f"  {desc}: {done}/{len(prompts)} ({time.time() - t0:.0f}s)", flush=True)
    return out


# ------------------------------------------------------------------ label scoring (judge)
def label_token_ids(tok, label):
    ids = set()
    for v in (label, label.capitalize(), label.upper(), " " + label, " " + label.capitalize()):
        enc = tok(v, add_special_tokens=False)["input_ids"]
        if enc:
            ids.add(enc[0])
    return sorted(ids)


@torch.no_grad()
def _last_logprobs(model, tok, prompts):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    enc.pop("token_type_ids", None)
    try:
        out = model(**enc, logits_to_keep=1)
    except TypeError:
        out = model(**enc)
    return torch.log_softmax(out.logits[:, -1, :].float(), dim=-1).cpu()


def score_labels(model, tok, prompts, labels, token_budget=12000, max_bs=32, desc="judge"):
    """For each prompt, the probability of each label as the next token
    (case variants pooled), renormalised over the allowed labels."""
    ids = {l: label_token_ids(tok, l) for l in labels}
    lengths = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    out = [None] * len(prompts)
    batches = list(budget_batches(lengths, token_budget, max_bs))
    t0 = time.time()

    def run(idx):
        lp = _last_logprobs(model, tok, [prompts[i] for i in idx])
        rows = []
        for r in range(lp.shape[0]):
            s = {l: float(torch.logsumexp(lp[r, ids[l]], 0)) for l in labels}
            m = max(s.values())
            z = sum(math.exp(v - m) for v in s.values())
            rows.append({l: math.exp(s[l] - m) / z for l in labels})
        return rows

    for k, b in enumerate(batches, 1):
        for i, r in zip(b, run_safely(run, b)):
            out[i] = r
        if k % 20 == 0 or k == len(batches):
            done = sum(len(x) for x in batches[:k])
            print(f"  {desc}: {done}/{len(prompts)} ({time.time() - t0:.0f}s)", flush=True)
    return out
RO_END_OF_FILE
cat > Tests/enrichment_pilot.py <<'RO_END_OF_FILE'
"""
Enrichment pilot (guide's Experiment 4 decision, Option B). GPU except `pool`.

Idea: genuine override should be far more common when the model holds a
confident WRONG belief. So: ask closed-book first, keep the wrong answers,
then give the model the passage that contains the right answer.

Stages (run in this order; each one resumes where it stopped):

  pool         Download Tevatron NQ and TriviaQA dev files (answer-bearing
               Wikipedia passages from the DPR data), keep questions that have
               a passage containing a gold alias and 4 hard-negative passages
               without any alias, drop questions already in our dataset,
               sample 1,000 per source (seed 2026). CPU.
  closedbook   --model gemma|llama: "Question: ...\\nAnswer:" with no passage,
               greedy, first line kept, plus confidence (first-token and mean
               token log-probability of the answer).
  [judge]      python Tests/llm_judge.py --tasks pilot-closedbook
  withpassage  --model gemma|llama: for every closed-book answer the judge
               calls wrong (and that does not contain a gold alias), generate
               WITH passages in the Phase 2 prompt format, two ways:
                   gold   the gold passage alone
                   mixed  the gold passage + 4 hard negatives, gold position
                          fixed per question (same for both models)
  [judge]      python Tests/llm_judge.py --tasks pilot-withpassage
  [classify]   python Tests/experiment1b_rebuilt_classifier.py --task pilot

Smoke test first (about 2 minutes once models are downloaded):
    python Tests/enrichment_pilot.py pool --n-per-source 10 --pilot-dir data/final/Phase_05/pilot_smoke
    python Tests/enrichment_pilot.py closedbook --model gemma --pilot-dir data/final/Phase_05/pilot_smoke
"""

import argparse
import random
from pathlib import Path

from gpu_utils import (MODEL_PATHS, append_jsonl, contains_any, done_keys, generate, load_jsonl, load_model,
                       normalize, parse_gold)

SOURCES = {"nq": ("Tevatron/wikipedia-nq", "nq-dev.jsonl.gz"),
           "trivia": ("Tevatron/wikipedia-trivia", "trivia-dev.jsonl.gz")}
SEED = 2026
N_DISTRACTORS = 4
CHUNK = 256


def ptext(p):
    return f"{p.get('title') or ''} {p.get('text') or ''}".strip()


def stage_pool(a):
    out = Path(a.pilot_dir) / "pilot_pool.jsonl"
    if out.exists():
        print(f"{out} exists -- delete it to rebuild the pool")
        return
    existing = {normalize(r.get("question")) for r in load_jsonl(a.dataset)}
    print(f"excluding {len(existing)} questions already in {a.dataset}", flush=True)
    rng = random.Random(SEED)
    pool = []
    for src, (repo, fname) in SOURCES.items():
        if a.local_dir:
            path = Path(a.local_dir) / fname
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset")
        print(f"{src}: reading {path}", flush=True)
        rows = load_jsonl(path)
        order = list(range(len(rows)))
        random.Random(f"{SEED}-{src}").shuffle(order)       # shuffle first, then take the first eligible ones
        picked, seen, checked = [], set(), 0
        why = {"no_answer": 0, "no_gold_passage": 0, "few_distractors": 0, "already_used": 0, "duplicate": 0}
        for i in order:
            if len(picked) >= a.n_per_source:
                break
            checked += 1
            r = rows[i]
            q = (r.get("query") or "").strip()
            nq = normalize(q)
            aliases = parse_gold(r.get("answers") or [])
            keys = [f" {normalize(x)} " for x in aliases if normalize(x)]
            if not keys:
                why["no_answer"] += 1
                continue
            if nq in existing:
                why["already_used"] += 1
                continue
            if nq in seen:
                why["duplicate"] += 1
                continue

            def has_alias(text):                              # each passage normalised once
                t = f" {normalize(text)} "
                return any(k in t for k in keys)

            gold = next((ptext(p) for p in (r.get("positive_passages") or []) if has_alias(ptext(p))), None)
            if not gold:
                why["no_gold_passage"] += 1
                continue
            negs = []
            for p in (r.get("negative_passages") or []):
                t = ptext(p)
                if not has_alias(t):
                    negs.append(t)
                    if len(negs) == N_DISTRACTORS:
                        break
            if len(negs) < N_DISTRACTORS:
                why["few_distractors"] += 1
                continue
            seen.add(nq)
            picked.append({"pid": f"pilot_{src}_{r.get('query_id')}", "source": src, "question": q,
                           "answers": aliases, "gold_passage": gold, "distractors": negs,
                           "gold_position": rng.randrange(N_DISTRACTORS + 1)})
        pool += picked
        print(f"{src}: {len(rows)} questions, checked {checked}, sampled {len(picked)}; excluded {why}", flush=True)
        del rows
    append_jsonl(out, pool)
    print(f"wrote {out} ({len(pool)} questions)")


def stage_closedbook(a):
    d = Path(a.pilot_dir)
    pool = load_jsonl(d / "pilot_pool.jsonl")
    if a.limit:
        pool = pool[:a.limit]
    out = d / f"pilot_closedbook_{a.model}.jsonl"
    done = done_keys(out, lambda r: r["pid"])
    todo = [p for p in pool if p["pid"] not in done]
    print(f"closedbook {a.model}: {len(todo)} to do ({len(done)} done)")
    if not todo:
        return
    model, tok = load_model(a.model_path or MODEL_PATHS[a.model], a.use_4bit)
    for s in range(0, len(todo), CHUNK):
        chunk = todo[s:s + CHUNK]
        res = generate(model, tok, [f"Question: {p['question']}\nAnswer:" for p in chunk],
                       max_new_tokens=32, token_budget=a.token_budget, max_bs=a.max_bs,
                       desc=f"closedbook {a.model} {s + len(chunk)}/{len(todo)}")
        append_jsonl(out, [{"pid": p["pid"], "model": a.model, "cb_answer": r["text"],
                            "cb_first_logprob": round(r["first_logprob"], 4),
                            "cb_mean_logprob": round(r["mean_logprob"], 4), "cb_n_tokens": r["n_tokens"],
                            "cb_string_correct": contains_any(r["text"], p["answers"])}
                           for p, r in zip(chunk, res)])
    print(f"wrote {out}")


def passages_for(p, cond):
    if cond == "gold":
        return [p["gold_passage"]]
    ps = list(p["distractors"])
    ps.insert(p["gold_position"], p["gold_passage"])
    return ps


def stage_withpassage(a):
    d = Path(a.pilot_dir)
    pool = {r["pid"]: r for r in load_jsonl(d / "pilot_pool.jsonl")}
    cb = load_jsonl(d / f"pilot_closedbook_{a.model}.jsonl")
    judged = {tuple(r["key"]): r["label"] for r in load_jsonl(d / "judge_pilot.jsonl")}
    missing = [r["pid"] for r in cb if (r["pid"], a.model, "-", "cb_correct") not in judged]
    if missing:
        raise SystemExit(f"{len(missing)} closed-book answers are not judged yet -- run "
                         f"python Tests/llm_judge.py --tasks pilot-closedbook first")
    wrong = [r for r in cb if judged[(r["pid"], a.model, "-", "cb_correct")] == "no" and not r["cb_string_correct"]]
    print(f"withpassage {a.model}: {len(wrong)} of {len(cb)} closed-book answers judged wrong")
    if a.limit:
        wrong = wrong[:a.limit]
    out = d / f"pilot_withpassage_{a.model}.jsonl"
    done = done_keys(out, lambda r: (r["pid"], r["cond"]))
    jobs = [(r["pid"], c) for r in wrong for c in ("gold", "mixed") if (r["pid"], c) not in done]
    print(f"  {len(jobs)} generations to do ({len(done)} done)")
    if not jobs:
        return
    model, tok = load_model(a.model_path or MODEL_PATHS[a.model], a.use_4bit)
    for s in range(0, len(jobs), CHUNK):
        chunk = jobs[s:s + CHUNK]
        prompts = ["\n\n".join(passages_for(pool[pid], c)) + f"\n\nQuestion: {pool[pid]['question']}\nAnswer:"
                   for pid, c in chunk]
        res = generate(model, tok, prompts, max_new_tokens=32, token_budget=a.token_budget, max_bs=a.max_bs,
                       desc=f"withpassage {a.model} {s + len(chunk)}/{len(jobs)}")
        append_jsonl(out, [{"pid": pid, "model": a.model, "cond": c, "wp_answer": r["text"],
                            "wp_first_logprob": round(r["first_logprob"], 4),
                            "wp_mean_logprob": round(r["mean_logprob"], 4),
                            "wp_string_correct": contains_any(r["text"], pool[pid]["answers"])}
                           for (pid, c), r in zip(chunk, res)])
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pool", "closedbook", "withpassage"])
    ap.add_argument("--model", choices=["gemma", "llama"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--use-4bit", action="store_true")
    ap.add_argument("--token-budget", type=int, default=8000)
    ap.add_argument("--max-bs", type=int, default=32)
    ap.add_argument("--n-per-source", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke test)")
    ap.add_argument("--local-dir", default=None, help="folder with nq-dev.jsonl.gz / trivia-dev.jsonl.gz")
    ap.add_argument("--dataset", default="data/final/Phase_02/analysis_dataset_relabeled.jsonl")
    ap.add_argument("--pilot-dir", default="data/final/Phase_05/pilot")
    a = ap.parse_args()
    if a.stage != "pool" and not a.model:
        raise SystemExit("--model gemma|llama is required for this stage")
    Path(a.pilot_dir).mkdir(parents=True, exist_ok=True)
    {"pool": stage_pool, "closedbook": stage_closedbook, "withpassage": stage_withpassage}[a.stage](a)


if __name__ == "__main__":
    main()
RO_END_OF_FILE
cat > Tests/gpu_selftest.py <<'RO_END_OF_FILE'
"""
GPU self-test: can this PyTorch + transformers pair run our exact code?

Builds tiny random-weight copies (2 layers) of Gemma-2, Llama-3.1 and the Qwen
judge from their real configs, with their real tokenizers (small files that
are already in the model cache), puts them on the GPU in the precision the real
run uses, and calls the same gpu_utils functions as the real run:
generate (batched, left-padded, stops at a newline, per-token log-probs) and
score_labels (the judge's yes/no/partial probabilities). About 30 seconds.

    python Tests/gpu_selftest.py                  # on the GPU box; exit code 0 = OK
    python Tests/gpu_selftest.py G L J --cpu-ok   # local test with three model folders
"""

import math
import sys
import time
import traceback
from pathlib import Path

t0 = time.time()
import torch
import transformers
from transformers.utils import is_torch_available

print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) | transformers {transformers.__version__}", flush=True)
if not is_torch_available():
    sys.exit("FAIL: this transformers version refuses this PyTorch version")
if not torch.cuda.is_available() and "--cpu-ok" not in sys.argv:
    sys.exit("FAIL: PyTorch cannot see the GPU")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gpu_utils import MODEL_PATHS, generate, pick_dtype, score_labels  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
PATHS = dict(zip(["gemma", "llama", "judge"], args)) if len(args) == 3 else MODEL_PATHS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS = ["Question: who wrote the play hamlet?\nAnswer:",
           "Paris is the capital and most populous city of France.\n\nQuestion: what is the capital of france?\nAnswer:"]


def tokenizer(path):                                   # same settings as gpu_utils.load_model
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def tiny_model(path):
    cfg = AutoConfig.from_pretrained(path)
    d = cfg.to_dict()
    d.update(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=4,
             num_key_value_heads=2, head_dim=16)
    d.pop("layer_types", None)                         # rebuilt for 2 layers
    cfg = type(cfg).from_dict(d)
    kw = {"attn_implementation": "eager"} if "gemma" in str(path).lower() else {}
    model = AutoModelForCausalLM.from_config(cfg, **kw)
    return model.to(device=DEVICE, dtype=pick_dtype(path, False)).eval()


def finite(*xs):
    return all(isinstance(x, float) and math.isfinite(x) for x in xs)


failed = []
for name in ("gemma", "llama", "judge"):
    path = PATHS[name]
    try:
        tok, model = tokenizer(path), tiny_model(path)
        if name == "judge":
            msgs = [{"role": "user", "content": "Is 'Paris' a correct answer to: what is the capital of France? "
                                                "Answer yes, no or partial."}]
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            rows = score_labels(model, tok, [p, "Context: none.\n" + p], ["yes", "no", "partial"],
                                token_budget=4000, max_bs=8, desc="  judge scoring")
            ok = len(rows) == 2 and all(finite(*r.values()) and abs(sum(r.values()) - 1) < 1e-3 for r in rows)
            what = "label probabilities"
        else:
            res = generate(model, tok, PROMPTS, max_new_tokens=8, token_budget=4000, max_bs=8,
                           desc=f"  {name} generation")
            ok = len(res) == 2 and all(isinstance(r["text"], str) and finite(r["first_logprob"], r["mean_logprob"])
                                       for r in res)
            what = "batched generation + log-probs"
        print(f"  {'OK  ' if ok else 'FAIL'} {name:5s} {what} ({model.dtype}, {DEVICE})", flush=True)
        if not ok:
            failed.append(name)
        del model
    except Exception:
        traceback.print_exc()
        print(f"  FAIL {name}", flush=True)
        failed.append(name)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if failed:
    sys.exit(f"SELF-TEST FAILED for: {', '.join(failed)}")
print(f"SELF-TEST PASSED ({time.time() - t0:.0f}s)")
RO_END_OF_FILE
md5sum -c --quiet <<'SUMS' || die "a file got garbled while pasting -- paste the whole block again"
e41ba555731cf5738f4b599724abed39  Tests/gpu_utils.py
a2aa9f60f41019f1bf4e2c30c99a1ec2  Tests/enrichment_pilot.py
fb9dd6ed26ef67e02f5797352f57701e  Tests/gpu_selftest.py
SUMS
echo "OK: gpu_utils.py (works with transformers 4.x and 5.x), enrichment_pilot.py (fast question pool), gpu_selftest.py"

say "2/4 transformers 4.x (the newest 5.x refuses PyTorch below 2.6)"
pipi "transformers>=4.56,<5"

say "3/4 GPU self-test (tiny copies of the 3 models, ~30 s)"
if ! python Tests/gpu_selftest.py; then
  echo; echo "transformers 4.x did not pass -- upgrading PyTorch to 2.6 instead (~3 GB download, 3-5 min)"
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  IDX=cu124; [ "${DRV:-0}" -lt 525 ] && IDX=cu118
  pipi "torch==2.6.0" --index-url "https://download.pytorch.org/whl/$IDX" || die "PyTorch upgrade failed -- send Claude the lines above"
  pipi -U "transformers>=4.56" huggingface_hub
  python Tests/gpu_selftest.py || die "self-test still fails -- send Claude the lines above"
fi

say "4/4 restarting the run: fresh smoke test (~10 min), then the full run starts by itself"
rm -rf data/final/Phase_05/pilot_smoke      # rebuilt with the fast pool, so the smoke test checks it too
if [ -n "${TMUX:-}" ]; then
  echo "Running here inside tmux. Detach: Ctrl+B then D. Reattach later: tmux attach"
  bash logs/run_session.sh 2>&1 | tee -a logs/session.log
else
  tmux kill-session -t ro_run 2>/dev/null || true
  tmux new-session -d -s ro_run "bash '$P/logs/run_session.sh' 2>&1 | tee -a '$P/logs/session.log'; exec bash"
  echo "Running in tmux session ro_run. If SSH drops, reconnect and type:  tmux attach -t ro_run"
  sleep 2; tmux attach -t ro_run
fi
RO_FIX
bash /tmp/ro_fix.sh