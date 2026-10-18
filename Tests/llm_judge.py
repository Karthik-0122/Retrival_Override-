"""
LLM judge (analysis 3 and the enrichment pilot). GPU.

An instruction-tuned model from a different family (default
Qwen/Qwen2.5-7B-Instruct) answers five fixed questions about each case. Each
mirrors a step of the written adjudication rules:

  correct    Is the answer correct for the question?        yes / partial / no
  contains   Do the passages state the gold answer?        yes / no
  answers    Does the model actually give an answer?       yes / no
  supported  Is the model's answer written in the passages? yes / no
  same       Are the with-passage and no-passage answers
             the same answer?                              yes / no

The judge never generates free text: for each prompt we read the probability
of each allowed label as the next token (one forward pass, deterministic).

Tasks (several can run in one model load, comma-separated):
  existing           the 566 apparent failures of Experiment 1 (all five
                     questions, every case -- also lets us compare the judge
                     with the human annotators question by question)
  pilot-closedbook   is each pilot closed-book answer correct?
  pilot-withpassage  the five questions for every pilot with-passage answer
                     (later questions only for answers judged wrong)

Run from repo root:
    python Tests/llm_judge.py --tasks existing,pilot-closedbook
    python Tests/llm_judge.py --tasks pilot-withpassage
Resumable: judgments already written are skipped.
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

from gpu_utils import (MODEL_PATHS, append_jsonl, done_keys, load_jsonl, load_model, parse_gold,
                       score_labels)

PROMPT_VERSION = "v1-2026-10-18"
SYSTEM = ("You are a careful annotator for a question-answering research study. "
          "Follow the instruction exactly and reply with a single word.")
YN, YPN = ["yes", "no"], ["yes", "partial", "no"]
MAX_PASSAGE_CHARS = 10000


def fmt_passages(passages):
    text, out = 0, []
    for i, p in enumerate(passages, 1):
        p = " ".join(p.split())
        if text + len(p) > MAX_PASSAGE_CHARS:
            p = p[:max(0, MAX_PASSAGE_CHARS - text)] + " [...]"
        out.append(f"[{i}] {p}")
        text += len(p)
        if text >= MAX_PASSAGE_CHARS:
            break
    return "\n\n".join(out)


def golds_str(golds):
    return " | ".join(golds[:12]) if golds else "(none)"


def q_correct(q, golds, ans):
    return (f"Question: {q}\nReference answer(s): {golds_str(golds)}\nCandidate answer: {ans or '(empty)'}\n\n"
            "Is the candidate answer correct for this question? Judge the meaning, not the wording: an alias, "
            "a longer sentence or an equivalent form of a reference answer counts as correct. If the candidate is "
            "right but clearly less specific or incomplete (for example only the year of a full date, or only "
            "part of a multi-part answer), reply partial. If the reference answer is clearly wrong for the "
            "question and the candidate is right, reply yes.\nReply with one word: yes, partial, or no.")


def q_contains(q, golds, passages):
    return (f"Passages:\n{fmt_passages(passages)}\n\nQuestion: {q}\nReference answer(s): {golds_str(golds)}\n\n"
            "Do the passages explicitly state a reference answer, or an equivalent form of it, as the answer "
            "to this question?\nReply with one word: yes or no.")


def q_answers(q, ans):
    return (f"Question: {q}\nCandidate answer: {ans or '(empty)'}\n\n"
            "Does the candidate actually give an answer to the question? Reply no if it only repeats the "
            "question, is placeholder text, is empty, or stops before giving any answer.\n"
            "Reply with one word: yes or no.")


def q_supported(q, passages, ans):
    return (f"Passages:\n{fmt_passages(passages)}\n\nQuestion: {q}\nCandidate answer: {ans or '(empty)'}\n\n"
            "Is the candidate answer, or the specific fact it states, written in the passages? It does not "
            "matter whether the candidate answers the question correctly.\nReply with one word: yes or no.")


def q_same(q, a, b):
    return (f"Question: {q}\nAnswer A: {a or '(empty)'}\nAnswer B: {b or '(empty)'}\n\n"
            "Do answers A and B give the same answer to the question (the same entity, value or claim), "
            "regardless of wording or extra detail?\nReply with one word: yes or no.")


def chat(tok, user):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM}\n\n{user}\n\nAnswer:"


def run_jobs(model, tok, jobs, out_path, args, desc):
    """jobs: list of dicts with key (list), kind, user, labels. Skips keys already written."""
    done = done_keys(out_path, lambda r: tuple(r["key"]))
    n_all = len(jobs)
    jobs = [j for j in jobs if tuple(j["key"]) not in done]
    if not jobs:
        print(f"  {desc}: nothing to do (all {n_all} already judged)")
        return
    if n_all > len(jobs):
        print(f"  {desc}: resuming, {n_all - len(jobs)} of {n_all} already judged")
    by_labels = defaultdict(list)
    for j in jobs:
        by_labels[tuple(j["labels"])].append(j)
    for labels, js in by_labels.items():
        for start in range(0, len(js), args.chunk):
            chunk = js[start:start + args.chunk]
            probs = score_labels(model, tok, [chat(tok, j["user"]) for j in chunk], list(labels),
                                 token_budget=args.token_budget, max_bs=args.max_bs,
                                 desc=f"{desc} [{'/'.join(labels)}] {start + len(chunk)}/{len(js)}")
            append_jsonl(out_path, [{"key": j["key"], "kind": j["kind"], "label": max(p, key=p.get),
                                     "probs": {k: round(v, 4) for k, v in p.items()},
                                     "judge": args.model_path, "prompt_version": PROMPT_VERSION}
                                    for j, p in zip(chunk, probs)])


def labels_from(path):
    return {tuple(r["key"]): r["label"] for r in load_jsonl(path)}


# ------------------------------------------------------------------ tasks
def task_existing(model, tok, args):
    passages = {r["query_id"]: r.get("retrieved_passages") or [] for r in load_jsonl(args.retrieval)}
    rows = load_jsonl(args.exp1)
    jobs, seen_q = [], set()
    for r in rows:
        q, m = r["question_id"], r["model"]
        golds, ques, ans, nc = parse_gold(r.get("gold")), r.get("question") or "", r.get("model_answer") or "", r.get("noctx_answer") or ""
        P = passages.get(q, [])
        if q not in seen_q:
            seen_q.add(q)
            jobs.append({"key": [q, "-", "contains"], "kind": "contains", "user": q_contains(ques, golds, P), "labels": YN})
        jobs += [{"key": [q, m, "correct"], "kind": "correct", "user": q_correct(ques, golds, ans), "labels": YPN},
                 {"key": [q, m, "noctx_correct"], "kind": "correct", "user": q_correct(ques, golds, nc), "labels": YPN},
                 {"key": [q, m, "answers"], "kind": "answers", "user": q_answers(ques, ans), "labels": YN},
                 {"key": [q, m, "supported"], "kind": "supported", "user": q_supported(ques, P, ans), "labels": YN},
                 {"key": [q, m, "same"], "kind": "same", "user": q_same(ques, nc, ans), "labels": YN}]
    print(f"existing: {len(rows)} apparent failures -> {len(jobs)} judge prompts")
    run_jobs(model, tok, jobs, Path(args.outdir_existing) / "judge_existing.jsonl", args, "existing")


def task_pilot_closedbook(model, tok, args):
    pool = {r["pid"]: r for r in load_jsonl(Path(args.pilot_dir) / "pilot_pool.jsonl")}
    jobs = []
    for m in ("gemma", "llama"):
        for r in load_jsonl(Path(args.pilot_dir) / f"pilot_closedbook_{m}.jsonl"):
            p = pool[r["pid"]]
            jobs.append({"key": [r["pid"], m, "-", "cb_correct"], "kind": "correct",
                         "user": q_correct(p["question"], p["answers"], r["cb_answer"]), "labels": YPN})
    print(f"pilot-closedbook: {len(jobs)} closed-book answers")
    run_jobs(model, tok, jobs, Path(args.pilot_dir) / "judge_pilot.jsonl", args, "pilot-closedbook")


def pilot_passages(p, cond):
    if cond == "gold":
        return [p["gold_passage"]]
    ps = list(p["distractors"])
    ps.insert(p["gold_position"], p["gold_passage"])
    return ps


def task_pilot_withpassage(model, tok, args):
    d = Path(args.pilot_dir)
    out = d / "judge_pilot.jsonl"
    pool = {r["pid"]: r for r in load_jsonl(d / "pilot_pool.jsonl")}
    cb = {(r["pid"], m): r for m in ("gemma", "llama") for r in load_jsonl(d / f"pilot_closedbook_{m}.jsonl")}
    wp = [r for m in ("gemma", "llama") for r in load_jsonl(d / f"pilot_withpassage_{m}.jsonl")]
    # pass 1: is the with-passage answer correct?
    jobs = [{"key": [r["pid"], r["model"], r["cond"], "wp_correct"], "kind": "correct",
             "user": q_correct(pool[r["pid"]]["question"], pool[r["pid"]]["answers"], r["wp_answer"]), "labels": YPN}
            for r in wp]
    print(f"pilot-withpassage: {len(wp)} with-passage answers")
    run_jobs(model, tok, jobs, out, args, "pilot-withpassage pass 1")
    # pass 2: the remaining questions, only for answers judged wrong
    lab = labels_from(out)
    jobs, seen = [], set()
    for r in wp:
        if lab.get((r["pid"], r["model"], r["cond"], "wp_correct")) != "no":
            continue
        p, ans = pool[r["pid"]], r["wp_answer"]
        P = pilot_passages(p, r["cond"])
        if (r["pid"], r["cond"]) not in seen:
            seen.add((r["pid"], r["cond"]))
            jobs.append({"key": [r["pid"], "-", r["cond"], "contains"], "kind": "contains",
                         "user": q_contains(p["question"], p["answers"], P), "labels": YN})
        cba = cb[(r["pid"], r["model"])]["cb_answer"]
        jobs += [{"key": [r["pid"], r["model"], r["cond"], "answers"], "kind": "answers", "user": q_answers(p["question"], ans), "labels": YN},
                 {"key": [r["pid"], r["model"], r["cond"], "supported"], "kind": "supported", "user": q_supported(p["question"], P, ans), "labels": YN},
                 {"key": [r["pid"], r["model"], r["cond"], "same"], "kind": "same", "user": q_same(p["question"], cba, ans), "labels": YN}]
    print(f"pilot-withpassage pass 2: {len(jobs)} prompts for answers judged wrong")
    run_jobs(model, tok, jobs, out, args, "pilot-withpassage pass 2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="comma-separated: existing, pilot-closedbook, pilot-withpassage")
    ap.add_argument("--model-path", default=MODEL_PATHS["judge"])
    ap.add_argument("--use-4bit", action="store_true")
    ap.add_argument("--token-budget", type=int, default=12000)
    ap.add_argument("--max-bs", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=512, help="prompts per saved chunk (resume granularity)")
    ap.add_argument("--exp1", default="data/final/Phase_05/exp1_failure_types.jsonl")
    ap.add_argument("--retrieval", default="data/final/Phase_02/retrieval_results.jsonl")
    ap.add_argument("--outdir-existing", default="data/final/Phase_05/judge")
    ap.add_argument("--pilot-dir", default="data/final/Phase_05/pilot")
    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    bad = [t for t in tasks if t not in ("existing", "pilot-closedbook", "pilot-withpassage")]
    if bad:
        raise SystemExit(f"unknown task(s): {bad}")
    model, tok = load_model(args.model_path, args.use_4bit)
    t0 = time.time()
    for t in tasks:
        {"existing": task_existing, "pilot-closedbook": task_pilot_closedbook,
         "pilot-withpassage": task_pilot_withpassage}[t](model, tok, args)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()