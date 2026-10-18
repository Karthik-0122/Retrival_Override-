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
    print(f"excluding {len(existing)} questions already in {a.dataset}")
    rng = random.Random(SEED)
    pool = []
    for src, (repo, fname) in SOURCES.items():
        if a.local_dir:
            path = Path(a.local_dir) / fname
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset")
        rows = load_jsonl(path)
        eligible, seen, why = [], set(), {"no_answer": 0, "no_gold_passage": 0, "few_distractors": 0,
                                          "already_used": 0, "duplicate": 0}
        for r in rows:
            q = (r.get("query") or "").strip()
            nq = normalize(q)
            aliases = parse_gold(r.get("answers") or [])
            if not aliases:
                why["no_answer"] += 1
                continue
            if nq in existing:
                why["already_used"] += 1
                continue
            if nq in seen:
                why["duplicate"] += 1
                continue
            gold = next((ptext(p) for p in (r.get("positive_passages") or []) if contains_any(ptext(p), aliases)), None)
            if not gold:
                why["no_gold_passage"] += 1
                continue
            negs = [ptext(p) for p in (r.get("negative_passages") or []) if not contains_any(ptext(p), aliases)]
            if len(negs) < N_DISTRACTORS:
                why["few_distractors"] += 1
                continue
            seen.add(nq)
            eligible.append({"pid": f"pilot_{src}_{r.get('query_id')}", "source": src, "question": q,
                             "answers": aliases, "gold_passage": gold, "distractors": negs[:N_DISTRACTORS]})
        k = min(a.n_per_source, len(eligible))
        picked = rng.sample(eligible, k)
        for p in picked:
            p["gold_position"] = rng.randrange(N_DISTRACTORS + 1)
        pool += picked
        print(f"{src}: {len(rows)} questions, {len(eligible)} eligible, sampled {k}; excluded {why}")
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