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
        kw["dtype"] = dtype
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
    print(f"loaded {path} in {time.time() - t0:.0f}s ({'4-bit' if use_4bit else kw.get('dtype')}, "
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