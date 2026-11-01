"""
Experiment 4, step 1: collect the cases. NO STEERING happens here -- the guide
approves the frozen protocol before any steering run.

Option B (primary, guide's decision of 15 Oct): the enrichment pipeline of the
pilot, unchanged, on 10,000 new Natural Questions (NQ-train questions never
used before; every human-confirmed override in the pilot came from NQ, none
from TriviaQA). Same 10,000 questions for both models. The pool is cut into 4
blocks of 2,500 in a fixed random order: batch 3 labels every F1 candidate of
blocks 1-3, and block 4 only if a model is still short of 40 validated cases.

Option A (in parallel): the 328 ConFiQA questions (the passage states a
counterfactual answer), re-run in the same precision as Option B, closed-book
and with the same 5 passages as Phase 2, then judged: the answer follows the
passage / keeps the model's own closed-book answer (parametric dominance) /
other.

Features for choosing candidate heads later (computed for every planned case;
which cases are genuine override is only known after batch 3):
  ROI  per head: 1 - cosine(last-token attention over the question tokens with
       the passages, the same without passages) -- the Phase 3 definition
       (Extract_roi.py), here in bf16 like the generation that defines the cases
  DLA  per head: direct contribution to logit(token of the answer the passage
       supports) - logit(the model's own top token), as in Phase 4, with two
       corrections: (1) read where the two answers part, not blindly at the
       first token -- a date or number starts with a bare space or a shared
       digit in both models' tokenizers, so the first token decides nothing;
       (2) Gemma-2's attention output passes through a post-attention RMSNorm
       and its RMSNorms scale by (1 + weight); the Phase 4 DLA ignored both.
       All layers and heads.

Stages (each one resumes where it stopped; run from the repo root):
  pool           CPU  10,000 NQ-train questions -> data/final/Phase_06/main/pilot_pool.jsonl
  [the pilot stages, unchanged: enrichment_pilot.py closedbook/withpassage, llm_judge.py
   pilot-closedbook/pilot-withpassage, experiment1b_rebuilt_classifier.py --task pilot,
   all with --pilot-dir data/final/Phase_06/main]
  confiqa-gen    GPU  --model gemma|llama
  confiqa-judge  GPU  the judge model
  plan           CPU  the cases that get features + the batch-3 samples (fixed seed)
  features       GPU  --model gemma|llama
  summary        CPU  counts for the log
The order and the exact commands are in Tests/run_exp4_collect.sh.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from gpu_utils import (MODEL_PATHS, append_jsonl, contains_any, done_keys, generate, load_jsonl, load_model,
                       normalize, parse_gold)

SEED = 2026
N_DISTRACTORS = 4
CHUNK = 256
TYPES_WRONG = ("F2", "F3", "F6", "partial")          # non-candidate wrong outcomes (probe strata)


def hkey(*parts):
    """Deterministic pseudo-random key, independent of file order."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def ptext(p):
    return f"{p.get('title') or ''} {p.get('text') or ''}".strip()


def passages_for(p, cond):                            # identical to enrichment_pilot.passages_for
    if cond == "gold":
        return [p["gold_passage"]]
    ps = list(p["distractors"])
    ps.insert(p["gold_position"], p["gold_passage"])
    return ps


def wp_prompt(passages, question):                    # Phase 2 / pilot prompt format
    return "\n\n".join(passages) + f"\n\nQuestion: {question}\nAnswer:"


def cb_prompt(question):
    return f"Question: {question}\nAnswer:"


def write_csv(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ================================================================== pool (CPU)
def stage_pool(a):
    d = Path(a.main_dir)
    out = d / "pilot_pool.jsonl"
    if out.exists():
        print(f"{out} exists -- delete it to rebuild the pool")
        return
    used = {normalize(r.get("question")) for r in load_jsonl(a.dataset)}
    n_ds = len(used)
    pilot_q = {normalize(r.get("question")) for r in load_jsonl(Path(a.pilot_dir) / "pilot_pool.jsonl")}
    used |= pilot_q
    print(f"excluding {n_ds} Phase 2 questions and {len(pilot_q)} pilot questions", flush=True)
    if a.local_dir:
        path = Path(a.local_dir) / a.train_file
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="Tevatron/wikipedia-nq", filename=a.train_file, repo_type="dataset")
    print(f"reading {path} (streamed)", flush=True)
    t0 = time.time()
    why, eligible, n_rows = Counter(), {}, 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_rows += 1
            r = json.loads(line)
            qid = str(r.get("query_id"))
            q = (r.get("query") or "").strip()
            nq = normalize(q)
            aliases = parse_gold(r.get("answers") or [])
            keys = [f" {normalize(x)} " for x in aliases if normalize(x)]
            if not keys:
                why["no_answer"] += 1
                continue
            if nq in used:
                why["already_used"] += 1
                continue

            def has_alias(text):
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
            k = hkey(SEED, "nq-train", qid)
            if nq in eligible and eligible[nq]["_k"] <= k:      # duplicate question: keep the smaller key
                why["duplicate"] += 1
                continue
            if nq in eligible:
                why["duplicate"] += 1
            eligible[nq] = {"_k": k, "qid": qid, "question": q, "answers": aliases, "gold_passage": gold,
                            "distractors": negs}
            if n_rows % 10000 == 0:
                print(f"  {n_rows} rows read, {len(eligible)} eligible ({time.time() - t0:.0f}s)", flush=True)
    rows = sorted(eligible.values(), key=lambda r: r["_k"])
    if len(rows) < a.n:
        print(f"!! only {len(rows)} eligible questions, fewer than the {a.n} asked for")
    pool = []
    for i, r in enumerate(rows[:a.n]):
        pool.append({"pid": f"main_nq_{r['qid']}", "source": "nq", "split": "nq-train", "question": r["question"],
                     "answers": r["answers"], "gold_passage": r["gold_passage"], "distractors": r["distractors"],
                     "gold_position": int(hkey(SEED, "pos", r["qid"])[:8], 16) % (N_DISTRACTORS + 1),
                     "block": 1 + i // a.block_size, "order": i})
    append_jsonl(out, pool)
    info = {"seed": SEED, "file": a.train_file, "rows_read": n_rows, "eligible": len(rows), "selected": len(pool),
            "block_size": a.block_size, "blocks": dict(Counter(p["block"] for p in pool)),
            "excluded": dict(why), "excluded_sources": {"phase2_questions": n_ds, "pilot_questions": len(pilot_q)}}
    json.dump(info, open(d / "pool_info.json", "w"), indent=2)
    print(f"{n_rows} rows read, {len(rows)} eligible, {len(pool)} selected in {time.time() - t0:.0f}s; "
          f"excluded {dict(why)}")
    print(f"wrote {out}")


# ================================================================== ConFiQA (Option A)
def confiqa_cases(a):
    ret = {r["query_id"]: r for r in load_jsonl(a.retrieval)}
    out = []
    for r in load_jsonl(a.dataset):
        if r.get("source_category") != "confiqa":
            continue
        ps = (ret.get(r["query_id"]) or {}).get("retrieved_passages") or []
        if not ps or not normalize(r.get("gold_answer_text")):
            continue
        out.append({"qid": r["query_id"], "question": r["question"], "cf_answer": r["gold_answer_text"],
                    "passages": ps, "phase2_label": {m: r.get(f"{m}_label") for m in ("gemma", "llama")}})
    return out


def stage_confiqa_gen(a):
    d = Path(a.confiqa_dir)
    cases = confiqa_cases(a)
    if a.limit:
        cases = cases[:a.limit]
    out = d / f"confiqa_gen_{a.model}.jsonl"
    done = done_keys(out, lambda r: r["qid"])
    todo = [c for c in cases if c["qid"] not in done]
    print(f"confiqa-gen {a.model}: {len(cases)} ConFiQA questions, {len(todo)} to do")
    if not todo:
        return
    model, tok = load_model(a.model_path or MODEL_PATHS[a.model], a.use_4bit)
    for s in range(0, len(todo), CHUNK):
        chunk = todo[s:s + CHUNK]
        prompts = [cb_prompt(c["question"]) for c in chunk] + [wp_prompt(c["passages"], c["question"]) for c in chunk]
        res = generate(model, tok, prompts, max_new_tokens=32, token_budget=a.token_budget, max_bs=a.max_bs,
                       desc=f"confiqa {a.model} {s + len(chunk)}/{len(todo)}")
        cb, wp = res[:len(chunk)], res[len(chunk):]
        append_jsonl(out, [{"qid": c["qid"], "model": a.model, "cb_answer": x["text"],
                            "cb_first_logprob": round(x["first_logprob"], 4), "cb_mean_logprob": round(x["mean_logprob"], 4),
                            "wp_answer": y["text"], "wp_first_logprob": round(y["first_logprob"], 4),
                            "wp_mean_logprob": round(y["mean_logprob"], 4),
                            "cb_has_cf": contains_any(x["text"], [c["cf_answer"]]),
                            "wp_has_cf": contains_any(y["text"], [c["cf_answer"]]),
                            "phase2_label": c["phase2_label"][a.model]}
                           for c, x, y in zip(chunk, cb, wp)])
    print(f"wrote {out}")


def stage_confiqa_judge(a):
    from llm_judge import YN, q_answers, q_same, run_jobs
    d = Path(a.confiqa_dir)
    cases = {c["qid"]: c for c in confiqa_cases(a)}
    jobs = []
    for m in ("gemma", "llama"):
        for g in load_jsonl(d / f"confiqa_gen_{m}.jsonl"):
            c = cases.get(g["qid"])
            if not c:
                continue
            q, cf = c["question"], c["cf_answer"]
            # "same answer" prompts only: the correctness prompt would let the judge prefer the
            # real-world answer over the counterfactual reference, which is the opposite of what we need
            jobs += [{"key": [g["qid"], m, "wp_is_cf"], "kind": "same", "user": q_same(q, cf, g["wp_answer"]), "labels": YN},
                     {"key": [g["qid"], m, "cb_is_cf"], "kind": "same", "user": q_same(q, cf, g["cb_answer"]), "labels": YN},
                     {"key": [g["qid"], m, "same"], "kind": "same", "user": q_same(q, g["cb_answer"], g["wp_answer"]), "labels": YN},
                     {"key": [g["qid"], m, "answers"], "kind": "answers", "user": q_answers(q, g["wp_answer"]), "labels": YN}]
    print(f"confiqa-judge: {len(jobs)} prompts")
    if not jobs:
        return
    model, tok = load_model(a.model_path or MODEL_PATHS["judge"], a.use_4bit)
    run_jobs(model, tok, jobs, d / "judge_confiqa.jsonl", a, "confiqa")


def confiqa_class(g, J):
    if g["cb_has_cf"] or J.get("cb_is_cf") == "yes":
        return "no_conflict"                 # the model already gives the passage's answer without it
    if g["wp_has_cf"] or J.get("wp_is_cf") == "yes":
        return "follows_passage"
    if J.get("answers") == "no":
        return "no_answer"
    if J.get("same") == "yes":
        return "parametric_dominance"        # keeps its own closed-book answer against the passage
    return "other"


# ================================================================== Option B outcomes (shared with the classifier)
def b_outcomes(d, m):
    """Every with-passage outcome for the closed-book-wrong questions in folder d, classified exactly as
    experiment1b_rebuilt_classifier.py --task pilot does (F5 -> followed, F4 -> partial)."""
    from experiment1b_rebuilt_classifier import classify
    d = Path(d)
    jl = {tuple(r["key"]): r["label"] for r in load_jsonl(d / "judge_pilot.jsonl")}
    cb = load_jsonl(d / f"pilot_closedbook_{m}.jsonl")
    wp = {(r["pid"], r["cond"]): r for r in load_jsonl(d / f"pilot_withpassage_{m}.jsonl")}
    res = []
    for r in cb:
        if jl.get((r["pid"], m, "-", "cb_correct")) != "no" or r["cb_string_correct"]:
            continue
        for cond in ("gold", "mixed"):
            w = wp.get((r["pid"], cond))
            if not w:
                continue
            c = "yes" if w["wp_string_correct"] else jl.get((r["pid"], m, cond, "wp_correct"))
            J = {"correct": c, "noctx_correct": "no", "contains": jl.get((r["pid"], "-", cond, "contains")),
                 "answers": jl.get((r["pid"], m, cond, "answers")), "supported": jl.get((r["pid"], m, cond, "supported")),
                 "same": jl.get((r["pid"], m, cond, "same"))}
            t = classify(J)
            if t is None:
                continue
            res.append({"pid": r["pid"], "model": m, "cond": cond, "type": {"F5": "followed", "F4": "partial"}.get(t, t),
                        "cb_answer": r["cb_answer"], "cb_mean_logprob": r["cb_mean_logprob"], "wp_answer": w["wp_answer"]})
    return res


def target_alias(p):
    """The gold alias the gold passage actually states (DLA target); else the first alias."""
    for x in p["answers"]:
        if contains_any(p["gold_passage"], [x]):
            return x
    return p["answers"][0]


def allocate(sizes, total):
    """Largest-remainder allocation of `total` over strata, proportional to size."""
    n = sum(sizes.values())
    if n == 0 or total == 0:
        return {k: 0 for k in sizes}
    total = min(total, n)
    raw = {k: total * v / n for k, v in sizes.items()}
    q = {k: min(sizes[k], int(math.floor(x))) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: -(raw[k] - math.floor(raw[k]))):
        if sum(q.values()) >= total:
            break
        if q[k] < sizes[k]:
            q[k] += 1
    return q


# ================================================================== plan (CPU)
def stage_plan(a):
    out_dir = Path(a.plan_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summ = [], {"seed": SEED}
    for m in ("gemma", "llama"):
        for src, d in (("main", a.main_dir), ("pilot", a.pilot_dir)):
            pool = {r["pid"]: r for r in load_jsonl(Path(d) / "pilot_pool.jsonl")}
            if not pool:
                continue
            outs = b_outcomes(d, m)
            summ[f"{src}_{m}_outcomes"] = dict(Counter(o["type"] for o in outs))

            def case(o, kind, group=None, weight=None):
                p = pool[o["pid"]]
                return {"case_id": f"{o['pid']}|{o['cond']}", "model": m, "kind": kind, "source_run": src,
                        "pid": o["pid"], "cond": o["cond"], "block": p.get("block"), "batch3_group": group,
                        "sampling_weight": weight, "outcome": o["type"], "question": p["question"],
                        "answers": p["answers"], "target": target_alias(p), "passages": passages_for(p, o["cond"]),
                        "closed_book_answer": o["cb_answer"], "cb_mean_logprob": o["cb_mean_logprob"],
                        "model_answer": o["wp_answer"]}

            # every F1 candidate (main: all blocks; pilot: all) -- batch 3 labels main blocks 1-3
            for o in outs:
                if o["type"] == "F1":
                    blk = pool[o["pid"]].get("block")
                    grp = None if src == "pilot" else ("candidate" if blk <= a.label_blocks else "candidate_reserve")
                    rows.append(case(o, "B_candidate", grp, 1.0))
            if src != "main":
                continue
            labelled = [o for o in outs if pool[o["pid"]]["block"] <= a.label_blocks]
            for cond in ("gold", "mixed"):
                # faithful controls: closed-book wrong, with-passage correct ("followed the passage")
                fol = sorted([o for o in labelled if o["cond"] == cond and o["type"] == "followed"],
                             key=lambda o: hkey(SEED, "followed", m, cond, o["pid"]))
                for i, o in enumerate(fol[:a.n_followed]):
                    rows.append(case(o, "B_followed", "followed" if i < a.n_followed_check else None,
                                     round(len(fol) / max(1, min(len(fol), a.n_followed_check)), 4)
                                     if i < a.n_followed_check else None))
                # probe: other wrong answers, where missed overrides would hide (proportional to stratum size)
                strata = {t: sorted([o for o in labelled if o["cond"] == cond and o["type"] == t],
                                    key=lambda o: hkey(SEED, "probe", m, cond, o["pid"])) for t in TYPES_WRONG}
                quota = allocate({t: len(v) for t, v in strata.items()}, a.n_probe)
                for t, v in strata.items():
                    for o in v[:quota[t]]:
                        rows.append(case(o, "B_probe", "probe", round(len(v) / quota[t], 4)))
        # Option A
        cq = Path(a.confiqa_dir)
        gens = load_jsonl(cq / f"confiqa_gen_{m}.jsonl")
        if gens:
            J = defaultdict(dict)
            for r in load_jsonl(cq / "judge_confiqa.jsonl"):
                q, mm, kind = r["key"]
                J[(q, mm)][kind] = r["label"]
            cases = {c["qid"]: c for c in confiqa_cases(a)}
            cls = Counter()
            for g in gens:
                c = cases.get(g["qid"])
                if not c:
                    continue
                k = confiqa_class(g, J[(g["qid"], m)])
                cls[k] += 1
                rows.append({"case_id": g["qid"], "model": m, "kind": "A_confiqa", "source_run": "confiqa",
                             "pid": g["qid"], "cond": "confiqa", "block": None, "batch3_group": None,
                             "sampling_weight": None, "outcome": k, "question": c["question"],
                             "answers": [c["cf_answer"]], "target": c["cf_answer"], "passages": c["passages"],
                             "closed_book_answer": g["cb_answer"], "cb_mean_logprob": g["cb_mean_logprob"],
                             "model_answer": g["wp_answer"]})
            summ[f"confiqa_{m}_classes"] = dict(cls)
    out = out_dir / "feature_cases.jsonl"
    if out.exists():
        out.unlink()
    append_jsonl(out, rows)
    summ["cases"] = {f"{m}:{k}": n for (m, k), n in Counter((r["model"], r["kind"]) for r in rows).items()}
    summ["batch3"] = {f"{m}:{g}": n for (m, g), n in Counter((r["model"], r["batch3_group"]) for r in rows
                                                              if r["batch3_group"]).items()}
    json.dump(summ, open(out_dir / "plan_summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=2))
    print(f"wrote {out} ({len(rows)} rows)")


# ================================================================== features (GPU)
def norm_eps(mod):
    for name in ("eps", "variance_epsilon"):
        if hasattr(mod, name):
            return float(getattr(mod, name))
    return 1e-6


def is_gemma(model):
    return str(getattr(model.config, "model_type", "")).startswith("gemma")


def head_dim(model):
    c = model.config
    return getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads


class Probe:
    """Forward hooks that keep, for the LAST position of a batch-size-1 forward pass: each layer's
    attention row over all keys (eager attention), the input and output of o_proj, the input of the
    final norm and the lm_head output (logits before Gemma's soft-capping)."""

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.reset()
        for i, layer in enumerate(model.model.layers):
            self.handles += [layer.self_attn.register_forward_hook(self._attn(i)),
                             layer.self_attn.o_proj.register_forward_pre_hook(self._z(i)),
                             layer.self_attn.o_proj.register_forward_hook(self._o(i))]
        self.handles += [model.model.norm.register_forward_pre_hook(self._final),
                         model.lm_head.register_forward_hook(self._logits)]

    def reset(self):
        self.attn, self.z, self.o, self.final, self.logits = {}, {}, {}, None, None

    def _attn(self, i):
        def f(mod, args, out):
            w = out[1] if isinstance(out, tuple) and len(out) > 1 else None
            if w is not None:
                self.attn[i] = w[0, :, -1, :].float()
        return f

    def _z(self, i):
        def f(mod, args):
            self.z[i] = args[0][0, -1, :].float()
        return f

    def _o(self, i):
        def f(mod, args, out):
            self.o[i] = out[0, -1, :].float()
        return f

    def _final(self, mod, args):
        self.final = args[0][0, -1, :].float()

    def _logits(self, mod, args, out):
        self.logits = out[0, -1, :].float()

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def forward(model, probe, input_ids):
    probe.reset()
    try:
        model(input_ids=input_ids, logits_to_keep=1)
    except TypeError:
        model(input_ids=input_ids)


def question_span(tok, prompt, question):
    """Token span of the question inside the prompt: the occurrence after 'Question: ' (Extract_roi.py
    used the first occurrence, which is the same unless a passage repeats the question verbatim)."""
    s = prompt.rindex("Question: " + question) + len("Question: ")
    e = s + len(question)
    enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True)
    t0 = enc.char_to_token(s)
    t1 = enc.char_to_token(e - 1)
    if t0 is None or t1 is None:
        return None, None, enc["input_ids"]
    return t0, t1 + 1, enc["input_ids"]


def span_rows(attn, s, e):
    """Per layer: (heads, span) attention of the last token over the question, renormalised (Phase 3)."""
    out = {}
    for layer, w in attn.items():
        x = w[:, s:e]
        out[layer] = x / (x.sum(-1, keepdim=True) + 1e-12)
    return out


def snapshot(probe):
    """The captures of the last forward pass (each forward creates new tensors, so references suffice)."""
    return {"z": dict(probe.z), "o": dict(probe.o), "final": probe.final, "logits": probe.logits}


@torch.no_grad()
def dla_matrix(model, st, tgt, alt, check=False):
    """[layers, heads] direct contribution of each head to logit[tgt] - logit[alt] (before soft-capping)
    at the position captured in st, with every RMSNorm linearised at its actual scale there."""
    gem = is_gemma(model)
    W_U = model.lm_head.weight
    du = W_U[tgt].float() - W_U[alt].float()
    nF = model.model.norm
    gF = nF.weight.float() + (1.0 if gem else 0.0)            # Gemma RMSNorm scales by (1 + weight)
    rF = st["final"]
    rmsF = torch.sqrt(rF.pow(2).mean() + norm_eps(nF))
    v = du * gF / rmsF
    H, D = model.config.num_attention_heads, head_dim(model)
    rows, chk = [], {}
    for li, layer in enumerate(model.model.layers):
        WO = layer.self_attn.o_proj.weight.float()                 # [hidden, H*D]
        z = st["z"][li]
        if gem:                                                    # Gemma-2: post-attention RMSNorm
            pn = layer.post_attention_layernorm
            scale = (pn.weight.float() + 1.0) / torch.sqrt(st["o"][li].pow(2).mean() + norm_eps(pn))
        else:
            scale = torch.ones_like(v)
        rows.append(((WO.t() @ (v * scale)) * z).view(H, D).sum(-1))
        if check and li in (0, len(model.model.layers) - 1):
            per_head = (WO.view(WO.shape[0], H, D) * z.view(1, H, D)).sum(-1).t()   # [H, hidden]
            o_true = st["o"][li]
            chk[f"L{li}_heads_sum_vs_o_proj"] = float((per_head.sum(0) - o_true).norm() / (o_true.norm() + 1e-9))
            if gem:
                pn = layer.post_attention_layernorm
                normed_true = pn(o_true.to(pn.weight.dtype).view(1, -1)).float().view(-1)
                normed_lin = (per_head * scale).sum(0)
                chk[f"L{li}_post_norm_linear_vs_module"] = float((normed_lin - normed_true).norm() / (normed_true.norm() + 1e-9))
            direct = ((per_head * scale) @ (du * gF / rmsF))           # per-head direct effect, explicit route
            chk[f"L{li}_fast_vs_explicit"] = float((direct - rows[-1]).abs().max() / (direct.abs().max() + 1e-9))
    final_lin = float((rF * gF / rmsF) @ du)
    final_true = float(st["logits"][tgt] - st["logits"][alt])
    chk["final_norm_identity_abs_err"] = abs(final_lin - final_true)
    chk["logit_diff"] = final_true
    chk["logit_scale"] = max(1.0, abs(float(st["logits"][tgt])), abs(float(st["logits"][alt])))
    return torch.stack(rows), chk


MAX_SHARED = 8


@torch.no_grad()
def dla_position(model, probe, tok, ids, first_state, target):
    """Where to read the DLA. Phase 4 used the first token at the end of the prompt, but there a
    date or number starts with a bare space or a shared digit for both answers (Gemma splits digits,
    Llama-3 splits off the space), so the first token says nothing about which answer wins.
    Here: follow the target answer token by token while the model's top prediction agrees with it;
      diverge  at the first position where the top prediction differs: target token vs that prediction
      agree    the model's greedy path reproduces the target: first non-space target token vs the
               runner-up there
    Returns (state, tgt, alt, n_target_tokens_fed, mode) or None."""
    gold = tok(" " + target, add_special_tokens=False)["input_ids"]
    if not gold:
        return None
    states, cur = [first_state], ids
    for k in range(min(len(gold), MAX_SHARED + 1)):
        top = int(torch.argmax(states[k]["logits"]))
        if top != gold[k]:
            return states[k], gold[k], top, k, "diverge"
        if k + 1 >= min(len(gold), MAX_SHARED + 1):
            break
        cur = torch.cat([cur, torch.tensor([[gold[k]]], device=cur.device)], dim=1)
        forward(model, probe, cur)
        states.append(snapshot(probe))
    j = next((i for i, t in enumerate(gold[:len(states)]) if tok.decode([t]).strip()), 0)
    st = states[j]
    top2 = torch.topk(st["logits"], 2).indices.tolist()
    return st, gold[j], (top2[1] if top2[0] == gold[j] else top2[0]), j, "agree"


def features_loop(model, tok, todo, out, model_key):
    """ROI + corrected DLA for each case in todo (plan-format rows), appended to out. The model must use
    eager attention. Shared by the collection run and the steering session (original faithful set)."""
    dev = model.device
    probe = Probe(model)
    L, H = len(model.model.layers), model.config.num_attention_heads
    soft = getattr(model.config, "final_logit_softcapping", None)
    noctx_cache, buf, t0, n_checked = {}, [], time.time(), 0
    notes = Counter()
    for k, r in enumerate(todo, 1):
        q = r["question"]
        rec = {"case_id": r["case_id"], "model": model_key, "kind": r["kind"]}
        # no-passage pass (shared by the gold and mixed versions of a question)
        if r["pid"] not in noctx_cache:
            p0 = cb_prompt(q)
            s0, e0, ids0 = question_span(tok, p0, q)
            if s0 is None:
                noctx_cache[r["pid"]] = None
            else:
                forward(model, probe, ids0.to(dev))
                noctx_cache[r["pid"]] = span_rows(probe.attn, s0, e0) if len(probe.attn) == L else None
            if len(noctx_cache) > 64:
                noctx_cache.pop(next(iter(noctx_cache)))
        base = noctx_cache.get(r["pid"])
        p1 = wp_prompt(r["passages"], q)
        s1, e1, ids1 = question_span(tok, p1, q)
        ids1 = ids1.to(dev)
        forward(model, probe, ids1)
        rec["n_tokens"] = int(ids1.shape[1])
        first = snapshot(probe)
        # ROI (end of the prompt, as in Phase 3)
        roi = None
        if base is None or s1 is None:
            notes["roi_no_span"] += 1
        elif len(probe.attn) != L:
            notes["roi_no_attention_weights"] += 1
        else:
            ctx = span_rows(probe.attn, s1, e1)
            if any(ctx[l].shape != base[l].shape for l in range(L)):
                notes["roi_span_length_differs"] += 1
            else:
                roi = [[round(float(x), 5) for x in (1 - torch.nn.functional.cosine_similarity(ctx[l], base[l], dim=-1)).tolist()]
                       for l in range(L)]
        rec["roi"] = roi
        top_first = int(torch.argmax(first["logits"]))
        rec["top_token_prompt_end"] = tok.decode([top_first])
        # DLA at the position where the passage's answer and the model's own answer part
        pos = dla_position(model, probe, tok, ids1, first, r["target"])
        if pos is None:
            rec["dla"] = None
            notes["dla_no_target_token"] += 1
        else:
            st, tgt, alt, k_fed, mode = pos
            do_check = n_checked < 3
            M, chk = dla_matrix(model, st, tgt, alt, check=do_check)
            n_checked += do_check
            notes[f"dla_{mode}"] += 1
            rec["dla"] = [[round(float(x), 4) for x in row] for row in M.tolist()]
            rec["dla_mode"], rec["dla_tokens_fed"] = mode, k_fed
            rec["logit_diff"] = round(chk["logit_diff"], 4)
            rec["dla_final_identity_err"] = round(chk["final_norm_identity_abs_err"], 5)
            rec["logit_scale"] = round(chk["logit_scale"], 3)
            if do_check:
                rec["dla_checks"] = {k2: round(v2, 6) for k2, v2 in chk.items()}
            lg = st["logits"] if not soft else torch.tanh(st["logits"] / soft) * soft
            lp = torch.log_softmax(lg, -1)
            rec.update({"tgt_token": tok.decode([tgt]), "alt_token": tok.decode([alt]),
                        "logprob_tgt": round(float(lp[tgt]), 4), "logprob_alt": round(float(lp[alt]), 4)})
        buf.append(rec)
        if len(buf) >= 50 or k == len(todo):
            append_jsonl(out, buf)
            buf = []
            print(f"  features {model_key}: {k}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    probe.remove()
    if notes:
        print(f"  notes: {dict(notes)}")
    print(f"wrote {out}  (layers {L}, heads {H})")


def stage_features(a):
    plan = [r for r in load_jsonl(Path(a.plan_dir) / "feature_cases.jsonl") if r["model"] == a.model]
    plan.sort(key=lambda r: (r["pid"], r["cond"]))                # gold and mixed of a question together
    if a.limit:
        plan = plan[:a.limit]
    out = Path(a.feat_dir) / f"features_{a.model}.jsonl"
    done = done_keys(out, lambda r: r["case_id"])
    todo = [r for r in plan if r["case_id"] not in done]
    print(f"features {a.model}: {len(plan)} planned cases, {len(todo)} to do")
    if not todo:
        return
    model, tok = load_model(a.model_path or MODEL_PATHS[a.model], a.use_4bit, attn_impl="eager")
    features_loop(model, tok, todo, out, a.model)




# ================================================================== summary (CPU)
def stage_summary(a):
    S = {}
    pool = load_jsonl(Path(a.main_dir) / "pilot_pool.jsonl")
    blocks = {p["pid"]: p["block"] for p in pool}
    print("=" * 78 + "\nEXPERIMENT 4, STEP 1 -- what was collected\n" + "=" * 78)
    print(f"Option B pool: {len(pool)} NQ-train questions, blocks {dict(Counter(blocks.values()))}")
    for m in ("gemma", "llama"):
        outs = b_outcomes(a.main_dir, m)
        cb = load_jsonl(Path(a.main_dir) / f"pilot_closedbook_{m}.jsonl")
        wrong = {o["pid"] for o in outs}
        S[m] = {"closed_book": len(cb), "closed_book_wrong": len(wrong)}
        print(f"\n{m.upper()}: {len(cb)} closed-book answers, {len(wrong)} wrong ({100 * len(wrong) / max(1, len(cb)):.0f}%)")
        for cond in ("gold", "mixed"):
            c = Counter(o["type"] for o in outs if o["cond"] == cond)
            f1 = Counter(blocks[o["pid"]] for o in outs if o["cond"] == cond and o["type"] == "F1")
            S[m][cond] = {"outcomes": dict(c), "F1_candidates_by_block": dict(sorted(f1.items()))}
            print(f"  [{cond}] {dict(sorted(c.items()))}")
            print(f"         F1 candidates by block: {dict(sorted(f1.items()))}")
    for m in ("gemma", "llama"):
        g = load_jsonl(Path(a.confiqa_dir) / f"confiqa_gen_{m}.jsonl")
        if g:
            print(f"\nOption A ConFiQA {m}: {len(g)} questions re-run")
    ps = Path(a.plan_dir) / "plan_summary.json"
    if ps.exists():
        P = json.load(open(ps))
        S["plan"] = P
        print("\nPlan:", json.dumps({k: v for k, v in P.items() if k in ("cases", "batch3")}, indent=1))
        for m in ("gemma", "llama"):
            if f"confiqa_{m}_classes" in P:
                print(f"  ConFiQA {m}: {P[f'confiqa_{m}_classes']}")
    for m in ("gemma", "llama"):
        F = load_jsonl(Path(a.feat_dir) / f"features_{m}.jsonl")
        if F:
            ok_roi = sum(r.get("roi") is not None for r in F)
            ok_dla = sum(r.get("dla") is not None for r in F)
            err = max((r.get("dla_final_identity_err") or 0) for r in F)
            checks = [r["dla_checks"] for r in F if r.get("dla_checks")]
            S[f"features_{m}"] = {"n": len(F), "roi": ok_roi, "dla": ok_dla, "max_final_identity_err": err,
                                  "checks": checks[:3]}
            print(f"\nfeatures {m}: {len(F)} cases, ROI {ok_roi}, DLA {ok_dla}, max final-norm identity error {err}")
            for c in checks[:3]:
                print(f"  DLA self-check: {c}")
    json.dump(S, open(Path(a.plan_dir) / "collect_summary.json", "w"), indent=2)
    print(f"\nwrote {Path(a.plan_dir) / 'collect_summary.json'}")


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pool", "confiqa-gen", "confiqa-judge", "plan", "features", "summary"])
    ap.add_argument("--model", choices=["gemma", "llama"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--use-4bit", action="store_true")
    ap.add_argument("--token-budget", type=int, default=8000)
    ap.add_argument("--max-bs", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=512, help="judge prompts per saved chunk")
    ap.add_argument("--limit", type=int, default=0, help="only the first N cases (smoke test)")
    ap.add_argument("--n", type=int, default=10000, help="pool size")
    ap.add_argument("--block-size", type=int, default=2500)
    ap.add_argument("--label-blocks", type=int, default=3, help="batch 3 labels candidates of blocks 1..N")
    ap.add_argument("--n-followed", type=int, default=150, help="faithful controls with features, per model x condition")
    ap.add_argument("--n-followed-check", type=int, default=10, help="of those, human-checked in batch 3")
    ap.add_argument("--n-probe", type=int, default=5, help="per model x condition")
    ap.add_argument("--train-file", default="nq-train.jsonl.gz")
    ap.add_argument("--local-dir", default=None, help="folder that holds the Tevatron file (offline)")
    ap.add_argument("--dataset", default="data/final/Phase_02/analysis_dataset_relabeled.jsonl")
    ap.add_argument("--retrieval", default="data/final/Phase_02/retrieval_results.jsonl")
    ap.add_argument("--pilot-dir", default="data/final/Phase_05/pilot")
    ap.add_argument("--main-dir", default="data/final/Phase_06/main")
    ap.add_argument("--confiqa-dir", default="data/final/Phase_06/confiqa")
    ap.add_argument("--plan-dir", default="data/final/Phase_06/plan")
    ap.add_argument("--feat-dir", default="data/final/Phase_06/features")
    a = ap.parse_args()
    if a.stage in ("confiqa-gen", "features") and not a.model:
        raise SystemExit("--model gemma|llama is required for this stage")
    for d in (a.main_dir, a.confiqa_dir, a.plan_dir, a.feat_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    {"pool": stage_pool, "confiqa-gen": stage_confiqa_gen, "confiqa-judge": stage_confiqa_judge,
     "plan": stage_plan, "features": stage_features, "summary": stage_summary}[a.stage](a)


if __name__ == "__main__":
    main()
