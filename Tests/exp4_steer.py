"""
Experiment 4, steering (frozen protocol v2, confirmed by the guide on 1 Nov 2026). Nothing here may be
changed after outcomes are seen; every rule below is the protocol's, written as code.

Stages (run from the repo root; each resumes where it stopped):
  split           CPU  final labels (+ label_fixes.csv) -> selection / test halves by question hash
  faithful2-gen   GPU  original faithful set (Q2): closed-book-correct questions, answered with passages
  [judge]              exp4_steer.py judge  -- is each answer correct?
  faithful2-feat  GPU  ROI + corrected DLA for 150 judged-correct cases per condition
  select          CPU  selection set only: ROI layers (length control + FDR), candidate heads (primary, A2
                       secondary), random depth-matched controls (1 primary + 20 robustness), balance check
  steer           GPU  directions from the selection set; generation at strengths 0, 2, 5, 10, 20 for every
                       planned (case, head set); teacher-forced log-probabilities for the test override cases
  judge           GPU  Qwen judge on every steered output (+ the faithful2 answers)
  check-sheet     CPU  A1: every test output the judge marks corrected + an equal random number not
                       corrected, for one annotator blind to heads and strength
  analyze         CPU  everything in protocol sections 7-8; --final once the A1 sheet is back

Option B = natural override (F1 vs faithful controls). Option A = ConFiQA (parametric dominance vs follows
the passage), reported as dominance over conflicting evidence, never as override of correct evidence.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

SEED = 2026
STRENGTHS = [0.0, 2.0, 5.0, 10.0, 20.0]
N_HEADS = 8
DEPTH_WINDOW = 5
RAND_SEED_PRIMARY = 3141
RAND_SEEDS_ROBUST = list(range(3142, 3162))          # 20 sets
RAND_SEED_SECONDARY = 4141
MIN_GROUP = 5                                        # A8
SMD_LIMIT = 0.25                                     # Q2 balance rule: |standardised mean difference| above this
FDR_Q = 0.05
PRIMARY_MODEL = "gemma"                              # Q4: Experiments 4-6 on Gemma; Experiment 7 on Llama
TRANSFER_EXP2 = {"gemma": ["triviaqa_02254", "triviaqa_03486", "nq_00676", "nq_03608"],
                 "llama": ["nq_03565", "popqa_07744", "triviaqa_01543"]}
CONFIQA_FOLLOW, CONFIQA_DOM = "follows_passage", "parametric_dominance"


def hkey(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def load_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def append_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_csv(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(p, rows, cols=None):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    cols = cols or (list(rows[0].keys()) if rows else ["empty"])
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def norm(t):
    t = "".join(ch for ch in (t or "").lower() if ch.isalnum() or ch.isspace())
    return " ".join(w for w in t.split() if w not in ("a", "an", "the"))


def contains_any(text, aliases):
    t = f" {norm(text)} "
    return any(norm(a) and f" {norm(a)} " in t for a in aliases)


# ================================================================== labels + split
def final_labels(a, model):
    """case_id -> {label, both_agreed, source, item_id} for every human-labelled case of this model."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from experiment2c_batch3 import load_final
    lab = {}
    # batch 2 pilot candidates (pilot re-screen under C1: Saban -> F1-M)
    adj2 = {r["item_id"]: r for r in read_csv(Path(a.batch2_dir) / "adjudication.csv")}
    for r in load_jsonl(Path(a.batch2_dir) / "validated_labels_batch2.jsonl"):
        if r["group"] != "pilot" or r["model"] != model:
            continue
        t = "F1-M" if r["item_id"] in a.pilot_f1m.split(",") else r["human_type"]
        x = adj2.get(r["item_id"], {})
        lab[f"{r['question_id']}|{r['cond']}"] = {"label": t, "item_id": r["item_id"], "source": "pilot",
                                                  "both_agreed": x.get("annot_A") == x.get("annot_B") == t}
    fixes = {r["item_id"]: r for r in read_csv(a.fixes)} if Path(a.fixes).exists() else {}
    pending = [i for i, r in fixes.items() if (r.get("new_label") or "").strip().upper() in ("", "PENDING")]
    if pending:
        raise SystemExit(f"label_fixes.csv: no decision yet for {pending} -- write new_label and reason first")
    for d in [Path(a.batch3_dir)] + ([Path(a.reserve_dir)] if (Path(a.reserve_dir) / "HIDDEN_key.csv").exists() else []):
        try:
            adj, key, _ = load_final(d)
        except SystemExit as e:
            if d == Path(a.reserve_dir):
                print(f"  ({d.name}: not adjudicated yet -- {e}; skipped)")
                continue
            raise
        for i, r in adj.items():
            k = key[i]
            if k["model"] != model:
                continue
            t = r["final_type"]
            fx = fixes.get(i)
            if fx:
                if fx["old_label"].strip().upper() != t.upper():
                    raise SystemExit(f"label_fixes.csv: {i} is {t} in the adjudication file, not {fx['old_label']}")
                t = fx["new_label"].strip().upper()
            lab[k["case_id"]] = {"label": t, "item_id": i, "source": d.name, "group": k["group"],
                                 "both_agreed": r["how"] == "agreed" and t == r["annot_A"]}
    return lab


def stage_split(a):
    out = Path(a.steer_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = {r["case_id"]: r for r in load_jsonl(Path(a.plan_dir) / "feature_cases.jsonl") if r["model"] == a.model}
    lab = final_labels(a, a.model)
    rows = []
    for cid, r in plan.items():
        L = lab.get(cid)
        if r["kind"] == "B_candidate" and L and L["label"] == "F1":
            g = "F1"
        elif r["kind"] == "B_followed":
            if L and L["label"] != "F5":
                continue                                    # hand-checked control that is not correct
            g = "faithful"
        elif r["kind"] in ("B_candidate", "B_probe") and L and L["label"] in ("F2", "F3", "F4"):
            g = L["label"]
        elif r["kind"] == "A_confiqa" and r["outcome"] in (CONFIQA_DOM, CONFIQA_FOLLOW):
            g = "dominance" if r["outcome"] == CONFIQA_DOM else "follows"
        else:
            continue
        rows.append({"case_id": cid, "pid": r["pid"], "cond": r["cond"], "model": a.model,
                     "option": "A" if r["kind"] == "A_confiqa" else "B", "group": g,
                     "both_agreed": bool(L and L.get("both_agreed")), "item_id": (L or {}).get("item_id", "")})
    # halves: questions ordered by hash, assigned alternately -- override (resp. dominance) questions first,
    # then each other group's unassigned questions; every case of a question stays in its question's half
    half = {}
    for opt, order in (("B", ["F1", "faithful", "F2", "F3", "F4"]), ("A", ["dominance", "follows"])):
        for g in order:
            qs = sorted({r["pid"] for r in rows if r["option"] == opt and r["group"] == g and r["pid"] not in half},
                        key=lambda q: hkey(SEED, a.model, opt, q))
            for i, q in enumerate(qs):
                half[q] = "selection" if i % 2 == 0 else "test"
    for r in rows:
        r["half"] = half[r["pid"]]
    p = out / f"split_{a.model}.jsonl"
    if p.exists():
        old = {r["case_id"]: r["half"] for r in load_jsonl(p)}
        changed = [c for c in old if c in {r["case_id"] for r in rows} and old[c] != half[plan[c]["pid"]]]
        if changed:
            raise SystemExit(f"{p} exists and {len(changed)} cases would change half -- the split is frozen")
        p.unlink()
    append_jsonl(p, rows)
    # report (the corrected counts for the guide)
    print("=" * 78 + f"\nSPLIT {a.model.upper()} (seed {SEED})\n" + "=" * 78)
    f1 = [r for r in rows if r["group"] == "F1"]
    q_all = {r["pid"] for r in f1}
    print(f"Validated F1: {len(f1)} cases, {len(q_all)} questions; by condition {dict(Counter(r['cond'] for r in f1))}")
    agreed = [r for r in f1 if r["both_agreed"]]
    print(f"  both annotators labelled F1 independently: {len(agreed)} cases, {len({r['pid'] for r in agreed})} questions")
    for h in ("selection", "test"):
        x = [r for r in f1 if r["half"] == h]
        prim = primary_cases([r for r in x])
        print(f"  {h:<9} F1 questions {len({r['pid'] for r in x})} (cases {len(x)}; primary unit = {len(prim)})")
    for g in ("faithful", "F2", "F3", "F4", "dominance", "follows"):
        c = Counter(r["half"] for r in rows if r["group"] == g)
        if c:
            print(f"  {g:<10} selection {c['selection']:>4}   test {c['test']:>4}")
    lab_counts = Counter(v["label"] for v in lab.values())
    print(f"All human labels for {a.model} after fixes: {dict(sorted(lab_counts.items()))}")
    json.dump({"model": a.model, "F1_cases": len(f1), "F1_questions": len(q_all),
               "F1_by_condition": dict(Counter(r["cond"] for r in f1)), "F1_both_agreed_cases": len(agreed),
               "F1_both_agreed_questions": len({r["pid"] for r in agreed}),
               "halves": {g: dict(Counter(r["half"] for r in rows if r["group"] == g))
                          for g in sorted({r["group"] for r in rows})}, "labels": dict(lab_counts)},
              open(out / f"split_summary_{a.model}.json", "w"), indent=2)
    print(f"wrote {p}")


def primary_cases(rows):
    """A3: one case per question -- the gold-condition case where validated, otherwise the mixed one."""
    by = defaultdict(dict)
    for r in rows:
        by[r["pid"]][r["cond"]] = r
    return [v.get("gold") or v.get("mixed") or next(iter(v.values())) for v in by.values()]


# ================================================================== statistics helpers
def bh(p):
    import numpy as np
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    from scipy.stats import binom
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * min(binom.cdf(min(b, c), n, 0.5), binom.sf(max(b, c) - 1, n, 0.5))))


def paired_diff_ci(x, y, z=1.96):
    """Newcombe-style Wald CI for a paired difference of proportions (x, y: 0/1 lists)."""
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"))
    d = [a - b for a, b in zip(x, y)]
    m = sum(d) / n
    var = sum((v - m) ** 2 for v in d) / max(1, n - 1) / n
    return (m - z * math.sqrt(var), m + z * math.sqrt(var))


def cohens_h(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


# ================================================================== covariates
DATE_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b|\b(january|february|march|april|may|june|july|august|september|"
                     r"october|november|december)\b", re.I)
NUM_RE = re.compile(r"^[\d.,\s%$]+$|\b\d+(\.\d+)?\b")
STOP = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is", "was", "are", "were", "be", "by",
        "with", "as", "from", "that", "this", "it", "its", "what", "who", "when", "where", "which", "how", "why",
        "did", "does", "do", "whose", "whom"}


def answer_type(t):
    t = (t or "").strip()
    if DATE_RE.search(t):
        return "date"
    if NUM_RE.search(t):
        return "number"
    if t and t[0].isupper():
        return "entity"
    return "other"


def relevance(question, passages):
    q = [w for w in norm(question).split() if w not in STOP]
    p = set(norm(" ".join(passages)).split())
    return sum(w in p for w in q) / len(q) if q else 0.0


def covariates(case):
    return {"length": len((case.get("model_answer") or "").split()),
            "relevance": relevance(case["question"], case["passages"]),
            "confidence": float(case.get("cb_mean_logprob") or 0.0),
            "type": answer_type(case.get("target") or (case.get("answers") or [""])[0])}


def smd(x, y):
    import numpy as np
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2) if len(x) > 1 and len(y) > 1 else 0.0
    return float((x.mean() - y.mean()) / s) if s > 0 else 0.0


# ================================================================== select (CPU, selection set only)
def roi_layers(Xa, Xb, cov_a, cov_b, extra):
    """Per layer: OLS ROI ~ group + length (+ extra covariates), BH-FDR across layers."""
    import numpy as np
    import statsmodels.api as sm
    La, Lb = np.nanmean(Xa, 2), np.nanmean(Xb, 2)
    y_grp = np.r_[np.ones(len(La)), np.zeros(len(Lb))]
    cols = [y_grp, np.r_[[c["length"] for c in cov_a], [c["length"] for c in cov_b]]]
    for e in extra:
        if e.startswith("type_"):
            t = e[5:]
            cols.append(np.r_[[c["type"] == t for c in cov_a], [c["type"] == t for c in cov_b]].astype(float))
        else:
            cols.append(np.r_[[c[e] for c in cov_a], [c[e] for c in cov_b]])
    X = sm.add_constant(np.column_stack(cols))
    p, coef = [], []
    for li in range(La.shape[1]):
        y = np.r_[La[:, li], Lb[:, li]]
        ok = np.isfinite(y)
        fit = sm.OLS(y[ok], X[ok]).fit()
        p.append(float(fit.pvalues[1]))
        coef.append(float(fit.params[1]))
    q = bh(p)
    return [i for i in range(len(p)) if q[i] < FDR_Q], p, list(map(float, q)), coef


def random_heads(cands, n_layers, n_heads, seed):
    """Phase 4 rule: depth-matched (within +-5 layers of a candidate's layer), candidates excluded."""
    rng = random.Random(seed)
    layers = sorted({l for l, _ in cands})
    pool = [(l, h) for l in range(n_layers) for h in range(n_heads)
            if (l, h) not in set(cands) and any(abs(l - x) <= DEPTH_WINDOW for x in layers)]
    rng.shuffle(pool)
    return sorted(pool[:len(cands)])


def stage_select(a):
    import numpy as np
    out = Path(a.steer_dir)
    split = load_jsonl(out / f"split_{a.model}.jsonl")
    plan = {r["case_id"]: r for r in load_jsonl(Path(a.plan_dir) / "feature_cases.jsonl") if r["model"] == a.model}
    feats = {r["case_id"]: r for r in load_jsonl(Path(a.feat_dir) / f"features_{a.model}.jsonl")}
    f2 = load_jsonl(out / f"faithful2_cases_{a.model}.jsonl")
    f2feat = {r["case_id"]: r for r in load_jsonl(out / f"features_faithful2_{a.model}.jsonl")}
    res = {}
    for opt, (ga, gb) in (("B", ("F1", "faithful")), ("A", ("dominance", "follows"))):
        A_ = [r for r in split if r["option"] == opt and r["group"] == ga and r["half"] == "selection"]
        B_ = [r for r in split if r["option"] == opt and r["group"] == gb and r["half"] == "selection"]
        A_ = [r for r in A_ if feats.get(r["case_id"], {}).get("roi") and feats[r["case_id"]].get("dla")]
        B_ = [r for r in B_ if feats.get(r["case_id"], {}).get("roi") and feats[r["case_id"]].get("dla")]
        if len(A_) < MIN_GROUP or len(B_) < MIN_GROUP:
            print(f"{opt}: too few selection cases ({len(A_)} / {len(B_)}) -- skipped")
            continue
        Xa = np.array([feats[r["case_id"]]["roi"] for r in A_], float)
        Xb = np.array([feats[r["case_id"]]["roi"] for r in B_], float)
        Da = np.array([feats[r["case_id"]]["dla"] for r in A_], float)
        Db = np.array([feats[r["case_id"]]["dla"] for r in B_], float)
        n_layers, n_heads = Xa.shape[1], Xa.shape[2]
        ca = [covariates(plan[r["case_id"]]) for r in A_]
        cb = [covariates(plan[r["case_id"]]) for r in B_]
        # Q2 balance check, fixed rule: any covariate with |SMD| > 0.25 joins the regression
        bal, extra = {}, []
        for c in ("relevance", "confidence"):
            s = smd([x[c] for x in ca], [x[c] for x in cb])
            bal[c] = round(s, 3)
            if abs(s) > SMD_LIMIT:
                extra.append(c)
        for t in ("date", "number", "entity", "other"):
            s = smd([x["type"] == t for x in ca], [x["type"] == t for x in cb])
            bal[f"type_{t}"] = round(s, 3)
            if abs(s) > SMD_LIMIT and t != "other":             # 'other' is the reference category
                extra.append(f"type_{t}")
        bal["length"] = round(smd([x["length"] for x in ca], [x["length"] for x in cb]), 3)
        sig, p, q, coef = roi_layers(Xa, Xb, ca, cb, extra)
        restrict = sig if sig else list(range(n_layers))
        mA, mB = Da.mean(0), Db.mean(0)

        def top(score):
            cand = [(float(score[l, h]), l, h) for l in restrict for h in range(n_heads)]
            cand.sort()
            return sorted((l, h) for _, l, h in cand[:N_HEADS])
        prim, sec = top(mA), top(mA - mB)
        rnd = {"rand_primary": random_heads(prim, n_layers, n_heads, RAND_SEED_PRIMARY),
               "rand_secondary": random_heads(sec, n_layers, n_heads, RAND_SEED_SECONDARY)}
        for i, s in enumerate(RAND_SEEDS_ROBUST, 1):
            rnd[f"rand_{i:02d}"] = random_heads(prim, n_layers, n_heads, s)
        r = {"n_selection": [len(A_), len(B_)], "balance_smd": bal, "covariates_added": extra,
             "roi_layers_fdr": sig, "roi_p": p, "roi_q": q, "roi_coef": coef, "layers_restricted_to": restrict,
             "heads": {"cand_primary": prim, "cand_secondary": sec, **rnd},
             "overlap_primary_secondary": sorted(set(prim) & set(sec))}
        # Q2: localisation against the original faithful set (secondary control, reported alongside)
        if opt == "B" and f2:
            S2 = [c for c in f2 if c.get("half") == "selection" and f2feat.get(c["case_id"], {}).get("roi")]
            if len(S2) >= MIN_GROUP:
                X2 = np.array([f2feat[c["case_id"]]["roi"] for c in S2], float)
                c2 = [covariates(c) for c in S2]
                sig2, _, _, _ = roi_layers(Xa, X2, ca, c2, [e for e in extra])
                r["roi_layers_fdr_vs_original_faithful"] = sig2
                r["n_original_faithful_selection"] = len(S2)
        res[opt] = r
        print(f"{a.model} option {opt}: selection {len(A_)} {ga} vs {len(B_)} {gb}; balance {bal}; added {extra}")
        print(f"  ROI layers (length control + FDR): {sig or 'none -> all layers'}")
        if "roi_layers_fdr_vs_original_faithful" in r:
            print(f"  vs original faithful set: {r['roi_layers_fdr_vs_original_faithful'] or 'none'}")
        print(f"  candidate heads (primary): {prim}\n  candidate heads (A2 secondary): {sec}; overlap {r['overlap_primary_secondary']}")
    p = out / f"heads_{a.model}.json"
    if p.exists():
        old = json.load(open(p))
        if any(old.get(o, {}).get("heads") != res.get(o, {}).get("heads") for o in res):
            raise SystemExit(f"{p} exists with different heads -- selection is frozen once written")
    json.dump(res, open(p, "w"), indent=2)
    print(f"wrote {p}")


# ================================================================== original faithful set (Q2)
def stage_faithful2_gen(a):
    from gpu_utils import MODEL_PATHS, generate, load_model
    out = Path(a.steer_dir) / f"faithful2_gen_{a.model}.jsonl"
    if out.exists():
        print(f"{out} exists -- skipped")
        return
    pool = {r["pid"]: r for r in load_jsonl(Path(a.main_dir) / "pilot_pool.jsonl")}
    jl = {tuple(r["key"]): r["label"] for r in load_jsonl(Path(a.main_dir) / "judge_pilot.jsonl")}
    cb = load_jsonl(Path(a.main_dir) / f"pilot_closedbook_{a.model}.jsonl")
    ok = [r for r in cb if pool[r["pid"]]["block"] <= 3 and
          (r["cb_string_correct"] or jl.get((r["pid"], a.model, "-", "cb_correct")) == "yes")]
    ok.sort(key=lambda r: hkey(SEED, "faithful2", a.model, r["pid"]))
    ok = ok[:a.n_faithful2_draw]
    model, tok = load_model(a.model_path or MODEL_PATHS[a.model], a.use_4bit)
    jobs = [(r, c) for r in ok for c in ("gold", "mixed")]
    prompts = [build_prompt(passages_for(pool[r["pid"]], c), pool[r["pid"]]["question"]) for r, c in jobs]
    res = generate(model, tok, prompts, max_new_tokens=32, token_budget=a.token_budget, max_bs=a.max_bs,
                   desc=f"faithful2 {a.model}")
    rows = []
    for (r, c), g in zip(jobs, res):
        p = pool[r["pid"]]
        rows.append({"case_id": f"{r['pid']}|{c}|f2", "pid": r["pid"], "cond": c, "model": a.model,
                     "kind": "B_faithful2", "question": p["question"], "answers": p["answers"],
                     "target": next((x for x in p["answers"] if contains_any(p["gold_passage"], [x])), p["answers"][0]),
                     "passages": passages_for(p, c), "closed_book_answer": r["cb_answer"],
                     "cb_mean_logprob": r["cb_mean_logprob"], "model_answer": g["text"],
                     "string_correct": contains_any(g["text"], p["answers"])})
    append_jsonl(out, rows)
    print(f"wrote {out} ({len(rows)} answers)")


def passages_for(p, cond):
    if cond == "gold":
        return [p["gold_passage"]]
    ps = list(p["distractors"])
    ps.insert(p["gold_position"], p["gold_passage"])
    return ps


def build_prompt(passages, question):
    return "\n\n".join(passages) + f"\n\nQuestion: {question}\nAnswer:"


def stage_faithful2_feat(a):
    from exp4_collect import features_loop
    from gpu_utils import MODEL_PATHS, load_model
    out_dir = Path(a.steer_dir)
    gen = load_jsonl(out_dir / f"faithful2_gen_{a.model}.jsonl")
    jl = judged(out_dir)
    keep, per = [], Counter()
    for r in gen:
        v = "yes" if r["string_correct"] else jl.get(jkey("correct", r["question"], r["answers"], r["model_answer"]))
        if v == "yes" and per[r["cond"]] < a.n_faithful2:
            keep.append(r)
            per[r["cond"]] += 1
    if any(v is None for v in [jl.get(jkey("correct", r["question"], r["answers"], r["model_answer"]))
                               for r in gen if not r["string_correct"]][:50]):
        raise SystemExit("faithful2 answers are not judged yet -- run the judge stage first")
    # halves by the same rule (questions ordered by hash, alternating)
    qs = sorted({r["pid"] for r in keep}, key=lambda q: hkey(SEED, a.model, "B2", q))
    half = {q: ("selection" if i % 2 == 0 else "test") for i, q in enumerate(qs)}
    for r in keep:
        r["half"] = half[r["pid"]]
    cp = out_dir / f"faithful2_cases_{a.model}.jsonl"
    if cp.exists():
        cp.unlink()
    append_jsonl(cp, keep)
    print(f"original faithful set {a.model}: {dict(per)} kept of {len(gen)} answers")
    fout = out_dir / f"features_faithful2_{a.model}.jsonl"
    done = {r["case_id"] for r in load_jsonl(fout)}
    todo = [r for r in keep if r["case_id"] not in done]
    if not todo:
        return
    model, tok = load_model(a.model_path or MODEL_PATHS[a.model], a.use_4bit, attn_impl="eager")
    features_loop(model, tok, todo, fout, a.model)


# ================================================================== steer (GPU)
class Steer:
    """Adds strength * direction to each listed head's slice of the o_proj input. mode 'last': last position of
    every forward pass (generation: end of the prompt, then each new token). mode 'from': positions >= start
    (teacher forcing, so the same positions are steered as during generation)."""

    def __init__(self, model):
        self.model, self.hd = model, None
        c = model.config
        self.hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        self.handles, self.vecs, self.strength, self.mode, self.start = [], {}, 0.0, "last", 0

    def set(self, heads, dirs, strength, mode="last", start=0):
        self.clear()
        self.strength, self.mode, self.start = strength, mode, start
        if strength == 0 or not heads:
            return
        by_layer = defaultdict(list)
        for (l, h) in heads:
            by_layer[l].append((h, dirs[(l, h)]))
        for l, lst in by_layer.items():
            self.handles.append(self.model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(self._hook(lst)))

    def _hook(self, lst):
        def f(mod, args):
            x = args[0].clone()
            for h, v in lst:
                v = (self.strength * v).to(x.device, x.dtype)
                if self.mode == "last":
                    x[:, -1, h * self.hd:(h + 1) * self.hd] += v
                else:
                    x[:, self.start:, h * self.hd:(h + 1) * self.hd] += v
            return (x,) + tuple(args[1:])
        return f

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def head_inputs(model, tok, prompts, layers, budget, max_bs):
    """Mean o_proj input (all heads) at the last prompt token, per layer, over the prompts."""
    import torch
    from gpu_utils import budget_batches
    cap, sums, n = {}, {l: None for l in layers}, 0
    hs = [model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
        (lambda li: (lambda m, a_: cap.__setitem__(li, a_[0][:, -1, :].float().sum(0).cpu())))(l)) for l in layers]
    lengths = [len(tok(p)["input_ids"]) for p in prompts]
    with torch.no_grad():
        for b in budget_batches(lengths, budget, max_bs):
            enc = tok([prompts[i] for i in b], return_tensors="pt", padding=True).to(model.device)
            enc.pop("token_type_ids", None)
            cap.clear()
            try:
                model(**enc, logits_to_keep=1)
            except TypeError:
                model(**enc)
            for l in layers:
                sums[l] = cap[l] if sums[l] is None else sums[l] + cap[l]
            n += len(b)
    for h in hs:
        h.remove()
    return {l: sums[l] / max(n, 1) for l in layers}


def teacher_logprob(model, tok, prompt, answer, steer, heads, dirs, strength):
    """Sum of log-probabilities of ' ' + answer after the prompt, steered at the answer positions. (The model's own
    forward already applies Gemma-2's final logit soft-capping, so the logits are used as returned.)"""
    import torch
    p_ids = tok(prompt, return_tensors="pt")["input_ids"]
    a_ids = tok(" " + answer.strip(), add_special_tokens=False, return_tensors="pt")["input_ids"]
    if a_ids.shape[1] == 0:
        return None
    ids = torch.cat([p_ids, a_ids], 1).to(model.device)
    start = p_ids.shape[1] - 1
    steer.set(heads, dirs, strength, mode="from", start=start)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0].float()
    steer.clear()
    lp = torch.log_softmax(logits[start:-1], -1)
    return float(lp.gather(1, ids[0, start + 1:].unsqueeze(1)).sum())


def stage_steer(a):
    import torch
    from gpu_utils import MODEL_PATHS, generate, load_model
    out_dir = Path(a.steer_dir)
    heads = json.load(open(out_dir / f"heads_{a.model}.json"))
    split = load_jsonl(out_dir / f"split_{a.model}.jsonl")
    plan = {r["case_id"]: r for r in load_jsonl(Path(a.plan_dir) / "feature_cases.jsonl") if r["model"] == a.model}
    f2 = load_jsonl(out_dir / f"faithful2_cases_{a.model}.jsonl")
    out = out_dir / f"steer_{a.model}.jsonl"
    lp_out = out_dir / f"steer_logprob_{a.model}.jsonl"
    done = {(r["case_id"], r["set"], r["strength"]) for r in load_jsonl(out)}
    lp_done = {(r["case_id"], r["set"], r["strength"]) for r in load_jsonl(lp_out)}
    case = lambda cid: plan.get(cid) or next(c for c in f2 if c["case_id"] == cid)

    def cases(opt, groups, h):
        return [r for r in split if r["option"] == opt and r["group"] in groups and r["half"] == h]
    jobs = []                     # (option, head-set option, set name, case rows)
    robust = [f"rand_{i:02d}" for i in range(1, len(RAND_SEEDS_ROBUST) + 1)]
    main4 = ["cand_primary", "rand_primary", "cand_secondary", "rand_secondary"]
    for opt, ga, gb, others in (("B", "F1", "faithful", ["F2", "F3", "F4"]), ("A", "dominance", "follows", [])):
        if opt not in heads:
            continue
        jobs.append((opt, opt, ["cand_primary", "rand_primary"], cases(opt, [ga], "selection")))
        jobs.append((opt, opt, main4 + robust, cases(opt, [ga], "test")))
        jobs.append((opt, opt, main4, cases(opt, [gb], "test")))
        if others:
            jobs.append((opt, opt, main4, cases(opt, others, "test")))
    if "B" in heads and f2:
        jobs.append(("B", "B", ["cand_primary", "rand_primary"],
                     [{"case_id": c["case_id"], "pid": c["pid"], "cond": c["cond"], "option": "B",
                       "group": "faithful2", "half": "test"} for c in f2 if c["half"] == "test"]))
    # transfer checks (descriptive): B heads on A test and the Experiment 2 cases; A heads on B test
    if "A" in heads and "B" in heads:
        jobs.append(("A", "B", ["cand_primary"], cases("A", ["dominance"], "test")))
        jobs.append(("B", "A", ["cand_primary"], cases("B", ["F1"], "test")))
    exp2 = exp2_cases(a) if "B" in heads else []
    if exp2:
        jobs.append(("B", "B", ["cand_primary"], exp2))
        for c in exp2:
            plan[c["case_id"]] = c
    model, tok = load_model(a.model_path or MODEL_PATHS[a.model], a.use_4bit)
    steer = Steer(model)
    # directions from the selection set: mean(faithful-like) - mean(override-like), per head
    dirs = {}
    for opt, ga, gb in (("B", "F1", "faithful"), ("A", "dominance", "follows")):
        if opt not in heads:
            continue
        hs = sorted({tuple(x) for v in heads[opt]["heads"].values() for x in v})
        layers = sorted({l for l, _ in hs})
        pa = [build_prompt(case(r["case_id"])["passages"], case(r["case_id"])["question"]) for r in cases(opt, [ga], "selection")]
        pb = [build_prompt(case(r["case_id"])["passages"], case(r["case_id"])["question"]) for r in cases(opt, [gb], "selection")]
        ma = head_inputs(model, tok, pa, layers, a.token_budget, a.max_bs)
        mb = head_inputs(model, tok, pb, layers, a.token_budget, a.max_bs)
        d = steer.hd
        dirs[opt] = {(l, h): (mb[l][h * d:(h + 1) * d] - ma[l][h * d:(h + 1) * d]) for l, h in hs}
        print(f"directions {opt}: {len(hs)} heads from {len(pa)} {ga} and {len(pb)} {gb} selection cases", flush=True)
    t0 = time.time()
    for opt, hopt, sets, rows in jobs:
        if not rows:
            continue
        prompts = [build_prompt(case(r["case_id"])["passages"], case(r["case_id"])["question"]) for r in rows]
        for sname in sets:
            hl = [tuple(x) for x in heads[hopt]["heads"][sname]]
            tag = sname if hopt == opt else f"{hopt}:{sname}"
            for s in STRENGTHS:
                stag = "baseline" if s == 0 else tag
                todo = [i for i, r in enumerate(rows) if (r["case_id"], stag, s) not in done]
                if not todo:
                    continue
                steer.set(hl, dirs[hopt], s)
                res = generate(model, tok, [prompts[i] for i in todo], max_new_tokens=32, token_budget=a.token_budget,
                               max_bs=a.max_bs, desc=f"{a.model} {opt} {rows[0]['group']} {tag} s={s}")
                steer.clear()
                append_jsonl(out, [{"case_id": rows[i]["case_id"], "pid": rows[i]["pid"], "cond": rows[i]["cond"],
                                    "option": opt, "group": rows[i]["group"], "half": rows[i]["half"], "set": stag,
                                    "heads_from": hopt, "strength": s, "text": g["text"], "n_tokens": g["n_tokens"]}
                                   for i, g in zip(todo, res)])
                for i in todo:
                    done.add((rows[i]["case_id"], stag, s))
        print(f"  {a.model} {opt} {rows[0]['group']} {rows[0]['half']}: done ({time.time() - t0:.0f}s)", flush=True)
    # teacher-forced log-probabilities: test override-like cases, main head sets, all strengths
    lp_rows = []
    for opt, ga in (("B", "F1"), ("A", "dominance")):
        if opt not in heads:
            continue
        for r in cases(opt, [ga], "test"):
            c = case(r["case_id"])
            pr = build_prompt(c["passages"], c["question"])
            for sname in ["cand_primary", "rand_primary", "cand_secondary", "rand_secondary"]:
                hl = [tuple(x) for x in heads[opt]["heads"][sname]]
                for s in STRENGTHS:
                    stag = "baseline" if s == 0 else sname
                    if (r["case_id"], stag, s) in lp_done:
                        continue
                    lp_done.add((r["case_id"], stag, s))
                    lp_rows.append({"case_id": r["case_id"], "option": opt, "set": stag, "strength": s,
                                    "lp_gold": teacher_logprob(model, tok, pr, c["target"], steer, hl, dirs[opt], s),
                                    "lp_orig": teacher_logprob(model, tok, pr, c["model_answer"], steer, hl, dirs[opt], s)})
            if len(lp_rows) >= 50:
                append_jsonl(lp_out, lp_rows)
                lp_rows = []
    append_jsonl(lp_out, lp_rows)
    del torch
    print(f"wrote {out} and {lp_out} ({time.time() - t0:.0f}s)")


def exp2_cases(a):
    """The 7 Experiment 2 F1 cases (external transfer check, descriptive), in the Phase 2 prompt format."""
    ids = TRANSFER_EXP2.get(a.model, [])
    if not ids:
        return []
    ds = {r["query_id"]: r for r in load_jsonl(a.dataset)}
    ret = {r["query_id"]: r for r in load_jsonl(a.retrieval)}
    out = []
    for q in ids:
        if q not in ds or q not in ret:
            continue
        r = ds[q]
        golds = parse_gold(r.get("gold_answer_text"))
        out.append({"case_id": f"exp2_{q}", "pid": q, "cond": "phase2", "option": "B", "group": "exp2_F1",
                    "half": "transfer", "question": r["question"], "answers": golds, "target": golds[0] if golds else "",
                    "passages": ret[q].get("retrieved_passages") or [], "model_answer": r.get(f"{a.model}_answer") or "",
                    "closed_book_answer": "", "cb_mean_logprob": 0.0})
    return out


def parse_gold(t):
    if isinstance(t, list):
        return t
    s = (t or "").strip()
    if s.startswith("["):
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, str)]
        except json.JSONDecodeError:
            pass
    return [s] if s else []


# ================================================================== judge (GPU)
def jkey(kind, question, golds, text):
    return hkey(kind, question, json.dumps(golds, ensure_ascii=False), text)


def judged(out_dir):
    return {r["key"]: r["label"] for r in load_jsonl(Path(out_dir) / "judge_steer.jsonl")}


def stage_judge(a):
    from gpu_utils import MODEL_PATHS, load_model
    from llm_judge import YN, YPN, q_answers, q_correct, q_same, run_jobs
    out_dir = Path(a.steer_dir)
    jobs, seen = [], set(judged(out_dir))
    for m in ("gemma", "llama"):
        plan = {r["case_id"]: r for r in load_jsonl(Path(a.plan_dir) / "feature_cases.jsonl") if r["model"] == m}
        for c in exp2_cases(argparse.Namespace(**{**vars(a), "model": m})):
            plan[c["case_id"]] = c
        for r in load_jsonl(out_dir / f"faithful2_gen_{m}.jsonl"):
            if not r["string_correct"]:
                k = jkey("correct", r["question"], r["answers"], r["model_answer"])
                if k not in seen:
                    seen.add(k)
                    jobs.append({"key": k, "kind": "correct", "user": q_correct(r["question"], r["answers"], r["model_answer"]), "labels": YPN})
        f2 = {c["case_id"]: c for c in load_jsonl(out_dir / f"faithful2_cases_{m}.jsonl")}
        for r in load_jsonl(out_dir / f"steer_{m}.jsonl"):
            c = plan.get(r["case_id"]) or f2.get(r["case_id"])
            if c is None:
                continue
            q, t = c["question"], r["text"]
            if r["option"] == "B":
                k = jkey("correct", q, c["answers"], t)
                if k not in seen:
                    seen.add(k)
                    jobs.append({"key": k, "kind": "correct", "user": q_correct(q, c["answers"], t), "labels": YPN})
            else:
                k = jkey("same_cf", q, [c["target"]], t)
                if k not in seen:
                    seen.add(k)
                    jobs.append({"key": k, "kind": "same", "user": q_same(q, c["target"], t), "labels": YN})
            k = jkey("answers", q, [], t)
            if k not in seen:
                seen.add(k)
                jobs.append({"key": k, "kind": "answers", "user": q_answers(q, t), "labels": YN})
    print(f"judge: {len(jobs)} new prompts")
    if not jobs:
        return
    model, tok = load_model(a.judge_path or MODEL_PATHS["judge"], a.use_4bit)
    run_jobs(model, tok, jobs, out_dir / "judge_steer.jsonl", argparse.Namespace(
        chunk=512, token_budget=a.judge_budget, max_bs=a.max_bs, model_path=a.judge_path or MODEL_PATHS["judge"]), "steer")


# ================================================================== outcomes
def degenerate(q, t, answers_label):
    """A7, fixed definition: empty; judge says no answer; mostly the question repeated; mostly non-word
    characters; or one token repeated four or more times in a row."""
    t = (t or "").strip()
    if not t or answers_label == "no":
        return True
    w = norm(t).split()
    qw = set(norm(q).split())
    if w and sum(x in qw for x in w) / len(w) >= 0.8:
        return True
    if sum(ch.isalnum() or ch.isspace() for ch in t) / len(t) < 0.5:
        return True
    return any(len(set(w[i:i + 4])) == 1 for i in range(max(0, len(w) - 3)))


def outcomes(a, model, human=None):
    out_dir = Path(a.steer_dir)
    plan = {r["case_id"]: r for r in load_jsonl(Path(a.plan_dir) / "feature_cases.jsonl") if r["model"] == model}
    for c in exp2_cases(argparse.Namespace(**{**vars(a), "model": model})):
        plan[c["case_id"]] = c
    for c in load_jsonl(out_dir / f"faithful2_cases_{model}.jsonl"):
        plan[c["case_id"]] = c
    jl = judged(out_dir)
    rows = []
    for r in load_jsonl(out_dir / f"steer_{model}.jsonl"):
        c = plan[r["case_id"]]
        q, t = c["question"], r["text"]
        if r["option"] == "B":
            lab = jl.get(jkey("correct", q, c["answers"], t))
            ok = contains_any(t, c["answers"]) or lab == "yes"
        else:
            lab = jl.get(jkey("same_cf", q, [c["target"]], t))
            ok = contains_any(t, [c["target"]]) or lab == "yes"
        hk = hkey(q, t)
        if human and hk in human:
            ok = human[hk]
        rows.append({**r, "correct": bool(ok), "judged": lab is not None,
                     "degenerate": degenerate(q, t, jl.get(jkey("answers", q, [], t)))})
    return rows, plan


# ================================================================== check sheet (A1)
def stage_check_sheet(a):
    rows, plan = outcomes(a, a.model)
    test = [r for r in rows if r["option"] == "B" and r["group"] == "F1" and r["half"] == "test"]
    uniq = {}
    for r in test:
        c = plan[r["case_id"]]
        uniq.setdefault(hkey(c["question"], r["text"]), (r, c))
    corrected = [k for k, (r, c) in uniq.items() if r["correct"] and r["strength"] > 0]
    rest = sorted([k for k, (r, c) in uniq.items() if not r["correct"] and r["strength"] > 0],
                  key=lambda k: hkey(SEED, "a1", k))[:len(corrected)]
    items = sorted(corrected + rest, key=lambda k: hkey(SEED, "a1-order", k))
    out = Path(a.steer_dir) / f"A1_check_{a.model}"
    out.mkdir(parents=True, exist_ok=True)
    sheet = []
    for i, k in enumerate(items, 1):
        r, c = uniq[k]
        sheet.append({"item_id": f"a1_{i:03d}", "question": c["question"], "gold_answers": " | ".join(c["answers"]),
                      "model_answer": r["text"], "correct": "", "note": ""})
    key = [{"item_id": s["item_id"], "hash": k, "judge_corrected": uniq[k][0]["correct"]} for s, k in zip(sheet, items)]
    try:
        from openpyxl import Workbook
        from openpyxl.worksheet.datavalidation import DataValidation
        wb = Workbook()
        ws = wb.active
        ws.title = "check"
        cols = list(sheet[0].keys()) if sheet else ["item_id"]
        ws.append(cols)
        for s in sheet:
            ws.append([s[c] for c in cols])
        dv = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"E2:E{len(sheet) + 1}")
        for col, w in zip("ABCDEF", (9, 50, 30, 45, 9, 30)):
            ws.column_dimensions[col].width = w
        g = wb.create_sheet("instructions")
        for line in ["For each row: is the model's answer a correct answer to the question? (yes / no)",
                     "Judge the meaning, not the wording; an alias or a longer sentence with the right answer is yes.",
                     "A less specific or incomplete answer (year only, part of a list) is no.",
                     "You are not told how the answer was produced. Do not try to find out."]:
            g.append([line])
        wb.save(out / "A1_check_sheet.xlsx")
    except ImportError:
        write_csv(out / "A1_check_sheet.csv", sheet)
    write_csv(out / "HIDDEN_A1_key.csv", key)
    print(f"A1 check {a.model}: {len(corrected)} judge-corrected + {len(rest)} not corrected = {len(sheet)} items -> {out}")


def human_overrides(a, model):
    d = Path(a.steer_dir) / f"A1_check_{model}"
    keyp = d / "HIDDEN_A1_key.csv"
    if not keyp.exists():
        return None, None
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from experiment2b_batch2 import read_table
    src = [p for p in (d / "A1_check_sheet.xlsx", d / "A1_check_sheet.csv") if p.exists()]
    if not src:
        return None, None
    filled = {r["item_id"]: (r.get("correct") or "").strip().lower() for r in read_table(src[0])}
    if not all(v in ("yes", "no") for v in filled.values()):
        return None, None
    key = read_csv(keyp)
    over = {k["hash"]: filled[k["item_id"]] == "yes" for k in key}
    dis = sum((filled[k["item_id"]] == "yes") != (k["judge_corrected"] == "True") for k in key)
    return over, {"items": len(key), "judge_human_disagree": dis}


# ================================================================== analyze
def stage_analyze(a):
    import numpy as np
    human, hstat = human_overrides(a, a.model) if a.final else (None, None)
    if a.final and human is None:
        raise SystemExit("--final needs the filled A1 check sheet (every row yes/no)")
    rows, plan = outcomes(a, a.model, human)
    if not rows:
        raise SystemExit("no steering outputs")
    unjudged = sum(not r["judged"] for r in rows)
    split = {r["case_id"]: r for r in load_jsonl(Path(a.steer_dir) / f"split_{a.model}.jsonl")}
    lps = load_jsonl(Path(a.steer_dir) / f"steer_logprob_{a.model}.jsonl")
    heads = json.load(open(Path(a.steer_dir) / f"heads_{a.model}.json"))
    R = {"model": a.model, "final": a.final, "unjudged_outputs": unjudged, "A1": hstat}
    idx = {(r["case_id"], r["set"], r["strength"]): r for r in rows}
    print("=" * 78 + f"\nEXPERIMENT 4 STEERING -- {a.model.upper()} ({'FINAL, human-checked' if a.final else 'PROVISIONAL, judge only'})\n" + "=" * 78)
    if unjudged:
        print(f"!! {unjudged} outputs have no judgment yet")
    for opt, ga, gb in (("B", "F1", "faithful"), ("A", "dominance", "follows")):
        if opt not in heads:
            continue
        o = {}
        sel = [c for c, s in split.items() if s["option"] == opt and s["group"] == ga and s["half"] == "selection"]
        test = [c for c, s in split.items() if s["option"] == opt and s["group"] == ga and s["half"] == "test"]
        prim = [r["case_id"] for r in primary_cases([split[c] for c in test])]
        prim_sel = [r["case_id"] for r in primary_cases([split[c] for c in sel])]
        rate = lambda ids, st, s: [idx[(c, st if s else "baseline", s)]["correct"] for c in ids if (c, st if s else "baseline", s) in idx]
        # baseline check
        base_ok = [c for c in test if (c, "baseline", 0.0) in idx and not idx[(c, "baseline", 0.0)]["correct"]]
        base_changed = [c for c in test if (c, "baseline", 0.0) in idx and norm(idx[(c, "baseline", 0.0)]["text"]) != norm(plan[c]["model_answer"])]
        o["baseline"] = {"test_cases": len(test), "baseline_wrong": len(base_ok), "baseline_text_changed": base_changed}
        # primary strength on the selection set
        diffs = {}
        for s in STRENGTHS[1:]:
            ca, ra = rate(prim_sel, "cand_primary", s), rate(prim_sel, "rand_primary", s)
            diffs[s] = (sum(ca) - sum(ra)) / max(1, len(ca))
        s_star = min(diffs, key=lambda s: (-diffs[s], s))
        o["strength_choice"] = {"selection_diff_by_strength": diffs, "primary_strength": s_star}
        # primary test (A3 unit)

        def paired(ids, s1, s2, s):
            pr = [(idx[(c, s1, s)]["correct"], idx[(c, s2, s)]["correct"]) for c in ids if (c, s1, s) in idx and (c, s2, s) in idx]
            return [int(x) for x, _ in pr], [int(y) for _, y in pr]

        def test_block(ids, cand, rnd, s):
            x, y = paired(ids, cand, rnd, s)
            n = len(x)
            b = sum(1 for i, j in zip(x, y) if i and not j)
            c_ = sum(1 for i, j in zip(x, y) if j and not i)
            p1, p2 = (sum(x) / n if n else float("nan")), (sum(y) / n if n else float("nan"))
            return {"n": n, "cand_corrected": sum(x), "rand_corrected": sum(y), "cand_rate": p1, "rand_rate": p2,
                    "cand_ci": wilson(sum(x), n), "rand_ci": wilson(sum(y), n), "diff": p1 - p2 if n else float("nan"),
                    "diff_ci": paired_diff_ci(x, y), "cohens_h": cohens_h(p1, p2) if n else float("nan"),
                    "discordant": [b, c_], "mcnemar_p": mcnemar_exact(b, c_)}
        o["primary"] = test_block(prim, "cand_primary", "rand_primary", s_star)
        o["primary_A6_both_agreed"] = test_block([c for c in prim if split[c]["both_agreed"]], "cand_primary", "rand_primary", s_star) if opt == "B" else None
        o["secondary_set_A2"] = test_block(prim, "cand_secondary", "rand_secondary", s_star)
        o["overlap_primary_secondary"] = heads[opt].get("overlap_primary_secondary")
        # all cases, clustered by question (A3 secondary analysis): cluster bootstrap of the difference
        x, y = paired(test, "cand_primary", "rand_primary", s_star)
        ids_ = [c for c in test if (c, "cand_primary", s_star) in idx and (c, "rand_primary", s_star) in idx]
        q_of = [split[c]["pid"] for c in ids_]
        rng = np.random.default_rng(SEED)
        qs = sorted(set(q_of))
        boots = []
        for _ in range(2000):
            pick = rng.choice(len(qs), len(qs))
            sel_ = [i for k in pick for i, q in enumerate(q_of) if q == qs[k]]
            if sel_:
                boots.append(np.mean([x[i] - y[i] for i in sel_]))
        o["all_cases_clustered"] = {"n_cases": len(x), "n_questions": len(qs), "diff": float(np.mean([i - j for i, j in zip(x, y)])) if x else float("nan"),
                                    "diff_ci_cluster_bootstrap": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None}
        # robustness: rank among the 20 random sets
        cand_rate = o["primary"]["cand_rate"]
        rr = []
        for i in range(1, len(RAND_SEEDS_ROBUST) + 1):
            v = rate(prim, f"rand_{i:02d}", s_star)
            rr.append(sum(v) / len(v) if v else float("nan"))
        o["robustness"] = {"random_set_rates": rr, "p": (1 + sum(r_ >= cand_rate for r_ in rr)) / (1 + len(rr))}
        # dose-response + degenerate outputs (A7) + log-probabilities
        dose = []
        lpi = {(r["case_id"], r["set"], r["strength"]): r for r in lps}
        for s in STRENGTHS:
            row = {"strength": s}
            for st in ("cand_primary", "rand_primary", "cand_secondary", "rand_secondary"):
                key_ = st if s else "baseline"
                v = rate(prim, st, s)
                dg = [idx[(c, key_, s)]["degenerate"] for c in prim if (c, key_, s) in idx]
                dl = [(lpi[(c, key_, s)]["lp_gold"] - lpi[(c, "baseline", 0.0)]["lp_gold"],
                       lpi[(c, key_, s)]["lp_orig"] - lpi[(c, "baseline", 0.0)]["lp_orig"])
                      for c in prim if (c, key_, s) in lpi and (c, "baseline", 0.0) in lpi
                      and lpi[(c, key_, s)]["lp_gold"] is not None and lpi[(c, "baseline", 0.0)]["lp_gold"] is not None]
                row[st] = {"rate": sum(v) / len(v) if v else float("nan"), "degenerate": sum(dg) / len(dg) if dg else float("nan"),
                           "d_lp_gold": float(np.mean([d[0] for d in dl])) if dl else None,
                           "d_lp_orig": float(np.mean([d[1] for d in dl if d[1] is not None])) if dl else None}
            dose.append(row)
        o["dose_response"] = dose
        # Wilcoxon on log-probability changes, candidate vs random, at the primary strength
        try:
            from scipy.stats import wilcoxon
            pr = [(lpi[(c, "cand_primary", s_star)]["lp_gold"] - lpi[(c, "rand_primary", s_star)]["lp_gold"])
                  for c in prim if (c, "cand_primary", s_star) in lpi and (c, "rand_primary", s_star) in lpi
                  and lpi[(c, "cand_primary", s_star)]["lp_gold"] is not None]
            o["wilcoxon_lp_gold"] = {"n": len(pr), "p": float(wilcoxon(pr).pvalue) if len(pr) >= 5 and any(pr) else None}
        except Exception as e:                                                  # noqa: BLE001
            o["wilcoxon_lp_gold"] = {"error": str(e)}
        # specificity: faithful damage, other failure types (A8), original faithful set
        fa = [c for c, s in split.items() if s["option"] == opt and s["group"] == gb and s["half"] == "test"]

        def damage(ids, st):
            base = [c for c in ids if (c, "baseline", 0.0) in idx and idx[(c, "baseline", 0.0)]["correct"]]
            hurt = [c for c in base if (c, st, s_star) in idx and not idx[(c, st, s_star)]["correct"]]
            return base, hurt
        bc, hc = damage(fa, "cand_primary")
        br, hr = damage(fa, "rand_primary")
        both = [c for c in bc if c in br]
        b_ = sum(1 for c in both if c in hc and c not in hr)
        c_ = sum(1 for c in both if c in hr and c not in hc)
        dmg_c, dmg_r = len(hc) / max(1, len(bc)), len(hr) / max(1, len(br))
        p_dmg = mcnemar_exact(b_, c_)
        o["faithful_damage"] = {"n_correct_at_baseline": len(bc), "cand_damage": dmg_c, "rand_damage": dmg_r,
                                "mcnemar_p": p_dmg, "substantial": (dmg_c - dmg_r > 0.10) or (p_dmg < 0.05 and dmg_c > dmg_r)}
        if opt == "B":
            f2 = [c["case_id"] for c in load_jsonl(Path(a.steer_dir) / f"faithful2_cases_{a.model}.jsonl") if c["half"] == "test"]
            if f2:
                bc2, hc2 = damage(f2, "cand_primary")
                br2, hr2 = damage(f2, "rand_primary")
                o["faithful2_damage"] = {"n": len(bc2), "cand_damage": len(hc2) / max(1, len(bc2)), "rand_damage": len(hr2) / max(1, len(br2))}
            oth = {}
            for g in ("F2", "F3", "F4"):
                ids = [c for c, s in split.items() if s["group"] == g and s["half"] == "test"]
                t = test_block(ids, "cand_primary", "rand_primary", s_star)
                t["descriptive_only"] = len(ids) < MIN_GROUP
                oth[g] = t
            o["other_failure_types"] = oth
            # Experiment 5 interaction: corrected ~ heads * (F1 vs other wrong types), SE clustered by case
            o["interaction"] = interaction(idx, [c for c in test] + [c for g in ("F2", "F3", "F4") for c, s in split.items() if s["group"] == g and s["half"] == "test"],
                                           split, s_star)
            tr = [r for r in rows if r["group"] == "exp2_F1"]
            o["transfer_exp2"] = {f"{r['case_id']}@{r['strength']}": r["correct"] for r in tr}
        o["transfer_other_option"] = {
            s: (lambda v: sum(v) / len(v) if v else None)([r["correct"] for r in rows if r["option"] == opt and r["heads_from"] != opt and r["strength"] == s])
            for s in STRENGTHS[1:]}
        # conclusion class (protocol Table 3; A uses the primary set)
        loc = bool(heads[opt].get("roi_layers_fdr"))
        prim_sig = o["primary"]["mcnemar_p"] < 0.05 and o["primary"]["diff"] > 0
        rob = o["robustness"]["p"] < 0.05
        spec_ok = (not o["faithful_damage"]["substantial"]) and (opt != "B" or (o["interaction"].get("p") is not None and o["interaction"]["p"] < 0.05 and o["interaction"]["coef"] > 0))
        if prim_sig and rob and spec_ok:
            cls = "A causal and specific"
        elif prim_sig and rob:
            cls = "B causal but not specific"
        elif not prim_sig and loc:
            cls = "C correlational / localisation only"
        else:
            cls = "D no reproducible effect"
        o["conclusion"] = {"class": cls, "primary_significant": prim_sig, "robustness": rob, "specific": spec_ok,
                           "localisation_present": loc}
        R[opt] = o
        pv = o["primary"]
        print(f"\nOPTION {opt} ({'natural override' if opt == 'B' else 'ConFiQA: dominance over conflicting evidence'})")
        print(f"  primary strength (selection set): {s_star}   selection diffs {diffs}")
        print(f"  primary test, {pv['n']} questions: candidate {pv['cand_corrected']} ({pv['cand_rate']:.2f}) vs random {pv['rand_corrected']} "
              f"({pv['rand_rate']:.2f}); discordant {pv['discordant']}; McNemar p = {pv['mcnemar_p']:.4f}; h = {pv['cohens_h']:.2f}")
        print(f"  robustness p = {o['robustness']['p']:.3f}; faithful damage cand {dmg_c:.2f} vs rand {dmg_r:.2f} (p {p_dmg:.3f})")
        if opt == "B":
            print(f"  interaction (Experiment 5): {o['interaction']}")
            print(f"  A6 both-agreed subset: {o['primary_A6_both_agreed']}")
        print(f"  A2 secondary set: cand {o['secondary_set_A2']['cand_corrected']} vs rand {o['secondary_set_A2']['rand_corrected']} "
              f"(p {o['secondary_set_A2']['mcnemar_p']:.3f}); overlap {o['overlap_primary_secondary']}")
        print(f"  baseline: {len(base_ok)}/{len(test)} wrong at strength 0; text changed for {len(base_changed)}")
        print(f"  CONCLUSION: {cls}")
    out = Path(a.steer_dir) / f"analysis_{a.model}{'_final' if a.final else '_provisional'}.json"
    json.dump(R, open(out, "w"), indent=2, default=lambda v: float(v) if hasattr(v, "__float__") else str(v))
    print(f"\nwrote {out}")


def interaction(idx, ids, split, s):
    """Logistic regression of corrected on heads (candidate=1 / random=0), F1 (1) vs other wrong types (0),
    and their interaction; standard errors clustered by case."""
    import numpy as np
    import statsmodels.api as sm
    y, X, g = [], [], []
    for k, c in enumerate(ids):
        f1 = 1.0 if split[c]["group"] in ("F1", "dominance") else 0.0
        for heads_, st in ((1.0, "cand_primary"), (0.0, "rand_primary")):
            if (c, st, s) in idx:
                y.append(float(idx[(c, st, s)]["correct"]))
                X.append([heads_, f1, heads_ * f1])
                g.append(k)
    if len(set(y)) < 2 or len({row[1] for row in X}) < 2:
        return {"p": None, "coef": None, "note": "not estimable (no variation)"}
    try:
        fit = sm.GLM(np.array(y), sm.add_constant(np.array(X)), family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": np.array(g)})
        return {"coef": float(fit.params[3]), "p": float(fit.pvalues[3]), "n_obs": len(y)}
    except Exception as e:                                                      # noqa: BLE001
        return {"p": None, "coef": None, "note": f"not estimable: {e}"}


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["split", "faithful2-gen", "faithful2-feat", "select", "steer", "judge",
                                      "check-sheet", "analyze"])
    ap.add_argument("--model", choices=["gemma", "llama"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--judge-path", default=None)
    ap.add_argument("--use-4bit", action="store_true")
    ap.add_argument("--token-budget", type=int, default=8000)
    ap.add_argument("--judge-budget", type=int, default=12000)
    ap.add_argument("--max-bs", type=int, default=32)
    ap.add_argument("--final", action="store_true", help="analyze with the human A1 decisions")
    ap.add_argument("--n-faithful2", type=int, default=150, help="original faithful cases per condition")
    ap.add_argument("--n-faithful2-draw", type=int, default=250, help="closed-book-correct questions answered")
    ap.add_argument("--pilot-f1m", default="b2_025", help="pilot items moved to F1-M by the C1 re-screen")
    ap.add_argument("--fixes", default="data/final/Phase_06/exp2c/label_fixes.csv")
    ap.add_argument("--batch2-dir", default="data/final/Phase_05/exp2b")
    ap.add_argument("--batch3-dir", default="data/final/Phase_06/exp2c")
    ap.add_argument("--reserve-dir", default="data/final/Phase_06/exp2c_reserve")
    ap.add_argument("--plan-dir", default="data/final/Phase_06/plan")
    ap.add_argument("--feat-dir", default="data/final/Phase_06/features")
    ap.add_argument("--main-dir", default="data/final/Phase_06/main")
    ap.add_argument("--steer-dir", default="data/final/Phase_06/steer")
    ap.add_argument("--dataset", default="data/final/Phase_02/analysis_dataset_relabeled.jsonl")
    ap.add_argument("--retrieval", default="data/final/Phase_02/retrieval_results.jsonl")
    a = ap.parse_args()
    if a.stage != "judge" and not a.model:
        raise SystemExit("--model gemma|llama is required")
    Path(a.steer_dir).mkdir(parents=True, exist_ok=True)
    {"split": stage_split, "faithful2-gen": stage_faithful2_gen, "faithful2-feat": stage_faithful2_feat,
     "select": stage_select, "steer": stage_steer, "judge": stage_judge, "check-sheet": stage_check_sheet,
     "analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    main()
