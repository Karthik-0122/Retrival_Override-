#!/usr/bin/env bash
# Retrieval Override -- Experiment 4, step 1 (collect the cases; NO steering).
# On a fresh GPU box, paste ONE line:
#   curl -fsSL https://raw.githubusercontent.com/Karthik-0122/Retrival_Override-/main/Tests/gpu_setup_exp4.sh -o /tmp/ro_exp4.sh && bash /tmp/ro_exp4.sh
# Part 1 (here, ~5 min, needs you once): libraries, Hugging Face login, repo, access checks.
# Part 2 (inside tmux, unattended, ~3.5 h): downloads, self-tests, smoke test + automatic check,
#        full run, then everything packed into ONE file to copy to your laptop.
set -euo pipefail
REPO_URL="https://github.com/Karthik-0122/Retrival_Override-.git"
AUTO_FULL="${AUTO_FULL:-1}"          # 1 = start the full run automatically if the smoke test passes
say() { echo; echo "========== $* =========="; }
die() { echo; echo "STOP: $*"; exit 1; }

say "1/7 GPU and disk"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found -- this image has no GPU driver."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$VRAM" -ge 22000 ] || die "this GPU has ${VRAM} MiB; Gemma-2 9B in bf16 needs a 24 GB card or larger."
# put everything on the filesystem with the most free space
BASE=""; BEST=0
for d in /workspace /home "$HOME" /root /; do
  [ -d "$d" ] && [ -w "$d" ] || continue
  free=$(df -Pk "$d" 2>/dev/null | awk 'NR==2{print $4}' || true); free=${free:-0}
  if [ "$free" -gt "$BEST" ]; then BEST=$free; BASE=$d; fi
done
[ "$BASE" = "/" ] && BASE=/root
PROJECT="$BASE/Retrival_Override-"
# reuse a Hugging Face cache that already holds the models
if [ -z "${HF_HOME:-}" ]; then
  for c in "$HOME/.cache/huggingface" /root/.cache/huggingface "$BASE/hf_cache" /workspace/hf_cache; do
    if [ -d "$c/hub" ] && ls "$c/hub" 2>/dev/null | grep -q "models--"; then HF_HOME="$c"; break; fi
  done
fi
export HF_HOME="${HF_HOME:-$BASE/hf_cache}"
mkdir -p "$HF_HOME/hub"            # must exist: on a fresh box `du` below would fail and stop the script
HAVE_GB=$(du -sk "$HF_HOME/hub" 2>/dev/null | awk '{print int($1/1024/1024)}'); HAVE_GB=${HAVE_GB:-0}
FREE_GB=$(df -Pk "$HF_HOME" | awk 'NR==2{print int($4/1024/1024)}')
NEED_GB=$(( 76 - HAVE_GB )); [ "$NEED_GB" -lt 15 ] && NEED_GB=15
echo "Project in $BASE | model cache $HF_HOME (already ${HAVE_GB} GB) | ${FREE_GB} GB free, need ~${NEED_GB} GB"
[ "$FREE_GB" -ge "$NEED_GB" ] || die "only $FREE_GB GB free but ~$NEED_GB GB more is needed (models ~68 GB + NQ file). Rent 100 GB of disk."
sed -i '/^export HF_HOME=/d' ~/.bashrc 2>/dev/null || true
echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

say "2/7 system tools"
SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO=sudo
need=""; for t in git tmux; do command -v $t >/dev/null || need="$need $t"; done
if [ -n "$need" ]; then export DEBIAN_FRONTEND=noninteractive; $SUDO apt-get update -qq && $SUDO apt-get install -y -qq $need; fi
PY=$(command -v python || command -v python3) || die "no python found"
if ! command -v python >/dev/null; then $SUDO ln -sf "$PY" /usr/local/bin/python; fi   # scripts call `python`
echo "python: $PY"

say "3/7 Python libraries (PyTorch is replaced only if it cannot run on this GPU)"
python - <<'PY' || exit 1
import torch, sys
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("STOP: PyTorch cannot see the GPU")
print("GPU:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
PY
# A GPU newer than the image's PyTorch build (Blackwell: RTX PRO 4000/6000, RTX 50xx) fails with
# "no kernel image is available". Test with a real GPU operation; if it fails, install
# PyTorch 2.8 built for CUDA 12.8, which supports these cards.
GPUTEST='import torch; x = torch.randn(256, 256, device="cuda"); assert torch.isfinite(x @ x).all(); print("GPU kernels OK: torch", torch.__version__, "on", torch.cuda.get_device_name(0))'
if ! python -c "$GPUTEST" 2>/dev/null; then
  echo "PyTorch $(python -c 'import torch; print(torch.__version__)' 2>/dev/null) has no kernels for this GPU -- installing PyTorch 2.8 for CUDA 12.8 (~3.5 GB, 3-5 min)"
  TIDX=https://download.pytorch.org/whl/cu128
  BSP=""; python -c 'import os, sys, sysconfig; sys.exit(not os.path.exists(os.path.join(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED")))' && BSP="--break-system-packages"
  INST=$(python -c 'import importlib.metadata as m; print((m.distribution("torch").read_text("INSTALLER") or "unknown").strip())' 2>/dev/null || echo none)
  if [ "$INST" = "pip" ]; then
    python -m pip uninstall -y -q torchvision torchaudio 2>/dev/null || true
    python -m pip install $BSP --no-cache-dir "torch==2.8.0" --index-url $TIDX
  else
    # installed by conda, which pip cannot uninstall: remove the old copy by hand, then install fresh
    SP=$(python -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("torch").origin)))')
    [ -d "$SP/torch" ] || die "cannot find the installed PyTorch"
    echo "removing the image's PyTorch (installed by $INST) from $SP"
    (cd "$SP" && rm -rf torch torchgen functorch torch-[0-9]* torchvision* torchaudio* triton triton-[0-9]* torchtriton*)
    python -m pip install $BSP --no-cache-dir --ignore-installed "torch==2.8.0" --index-url $TIDX
  fi
  python -c "$GPUTEST" || die "PyTorch still cannot use this GPU -- paste the error above"
fi
PIPI="python -m pip install -q -U"
$PIPI "transformers>=4.56" accelerate huggingface_hub safetensors 2>/dev/null || \
  $PIPI --break-system-packages "transformers>=4.56" accelerate huggingface_hub safetensors
python -m pip uninstall -y -q torchvision torchaudio 2>/dev/null || true   # avoids an import crash seen before
# old brotli in some images breaks Hugging Face downloads ("process() takes no keyword arguments")
if python -c "import brotli" 2>/dev/null && ! python -c "import brotli; brotli.Decompressor().process(b'', output_buffer_limit=1)" 2>/dev/null; then
  echo "upgrading brotli (the image's version breaks Hugging Face downloads)"
  $PIPI "brotli>=1.2.0" 2>/dev/null || $PIPI --ignore-installed "brotli>=1.2.0" 2>/dev/null || python -m pip uninstall -y -q brotli
fi
if python -m pip show brotlicffi >/dev/null 2>&1; then $PIPI "brotlicffi>=1.2.0" 2>/dev/null || python -m pip uninstall -y -q brotlicffi; fi
# the newest transformers needs PyTorch >= 2.6; on an older image it silently disables PyTorch,
# so check what transformers itself reports, and fall back to the last 4.x release if needed
TEST='import sys, transformers; from transformers.utils import is_torch_available as ok; print("transformers", transformers.__version__, "| works with this PyTorch:", ok()); sys.exit(0 if ok() else 1)'
if ! python -c "$TEST"; then
  echo "the newest transformers needs a newer PyTorch than this image has -- installing the last 4.x release instead"
  $PIPI "transformers>=4.56,<5" 2>/dev/null || $PIPI --break-system-packages "transformers>=4.56,<5"
  python -c "$TEST" || die "transformers cannot use this PyTorch -- paste the error above"
fi

say "4/7 Hugging Face login"
if python -c "from huggingface_hub import whoami; print('logged in as', whoami()['name'])" 2>/dev/null; then
  :
elif [ -n "${HF_TOKEN:-}" ]; then
  python -c "import os; from huggingface_hub import login; login(token=os.environ['HF_TOKEN'])"
else
  echo "Paste your Hugging Face token (hf_...). It is not shown while you type."
  python -c "from huggingface_hub import login; login()"
fi
python -c "from huggingface_hub import whoami; print('logged in as', whoami()['name'])" || die "login failed"

say "5/7 model access check (small files only)"
python - <<'PY' || exit 1
import sys
from huggingface_hub import hf_hub_download
bad = []
for repo in ["google/gemma-2-9b", "meta-llama/Llama-3.1-8B", "Qwen/Qwen2.5-7B-Instruct"]:
    try:
        hf_hub_download(repo, "config.json")
        print(f"  OK   {repo}")
    except Exception as e:
        print(f"  FAIL {repo}: {type(e).__name__}: {str(e)[:160]}")
        bad.append(repo)
if bad:
    sys.exit("STOP: no access to " + ", ".join(bad) + " -- open the model page on huggingface.co while logged in "
             "and accept the licence, then rerun this setup.")
PY

say "6/7 project repository"
if [ -d "$PROJECT/.git" ]; then
  git -C "$PROJECT" pull --ff-only || echo "WARNING: git pull failed -- using the files already on this box"
else
  URL="$REPO_URL"
  [ -n "${GITHUB_TOKEN:-}" ] && URL="https://x-access-token:${GITHUB_TOKEN}@github.com/Karthik-0122/Retrival_Override-.git"
  GIT_TERMINAL_PROMPT=0 git clone -q "$URL" "$PROJECT" || die "clone failed. If the repo is private: export GITHUB_TOKEN=<token> and rerun."
fi
cd "$PROJECT"
git log -1 --format="latest commit: %h %s (%cr)"
missing=0
for f in Tests/exp4_collect.py Tests/run_exp4_collect.sh Tests/exp4_selftest.py Tests/gpu_utils.py Tests/llm_judge.py \
         Tests/enrichment_pilot.py Tests/experiment1b_rebuilt_classifier.py Tests/gpu_selftest.py \
         data/final/Phase_02/analysis_dataset_relabeled.jsonl data/final/Phase_02/retrieval_results.jsonl \
         data/final/Phase_05/pilot/pilot_pool.jsonl data/final/Phase_05/pilot/judge_pilot.jsonl \
         data/final/Phase_05/pilot/pilot_closedbook_gemma.jsonl data/final/Phase_05/pilot/pilot_closedbook_llama.jsonl \
         data/final/Phase_05/pilot/pilot_withpassage_gemma.jsonl data/final/Phase_05/pilot/pilot_withpassage_llama.jsonl \
         data/final/Phase_05/judge/rebuilt_classifier_report.json; do
  [ -s "$f" ] || { echo "  MISSING: $f"; missing=1; }
done
grep -q "attn_impl" Tests/gpu_utils.py || { echo "  OLD VERSION: Tests/gpu_utils.py"; missing=1; }
[ "$missing" -eq 0 ] || die "files above are missing on GitHub -- push them from your laptop first, then rerun."
echo "all scripts and data present"

say "7/7 starting the unattended run in tmux"
if   [ "$VRAM" -ge 40000 ]; then EXTRA="--token-budget 16000"; JEXTRA="--token-budget 24000"
elif [ "$VRAM" -ge 26000 ]; then EXTRA="--token-budget 8000";  JEXTRA="--token-budget 12000"
else                             EXTRA="--token-budget 4000";  JEXTRA="--token-budget 8000"; fi
echo "VRAM ${VRAM} MiB -> model batches: $EXTRA | judge batches: $JEXTRA"
mkdir -p logs
cat > logs/run_exp4.sh <<RUN
#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="$HF_HOME"
cd "$PROJECT"
stamp() { echo; echo "[\$(date '+%H:%M:%S')] ===== \$* ====="; }
stamp "downloading models and the NQ-train file (~70 GB: Gemma-2 is stored in 32-bit, ~37 GB)"
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
for repo in ["google/gemma-2-9b", "meta-llama/Llama-3.1-8B", "Qwen/Qwen2.5-7B-Instruct"]:
    snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"])
    print("downloaded", repo, flush=True)
hf_hub_download("Tevatron/wikipedia-nq", "nq-train.jsonl.gz", repo_type="dataset")
print("downloaded Tevatron/wikipedia-nq nq-train.jsonl.gz", flush=True)
PY
df -h "$BASE" | tail -1
stamp "GPU self-tests (tiny copies of the models, ~1 min)"
python Tests/gpu_selftest.py
python Tests/exp4_selftest.py
stamp "SMOKE TEST (20 questions + 10 ConFiQA)"
EXTRA="$EXTRA" JEXTRA="$JEXTRA" bash Tests/run_exp4_collect.sh smoke 2>&1 | tee logs/exp4_smoke.log
stamp "automatic check of the smoke test"
python - <<'PY'
import json, math, sys
from pathlib import Path
R = Path("data/final/Phase_06/smoke")
def rows(p): return [json.loads(l) for l in open(p)] if p.exists() else []
problems = []
pool = {r["pid"]: r for r in rows(R / "main/pilot_pool.jsonl")}
judge = {tuple(r["key"]): r["label"] for r in rows(R / "main/judge_pilot.jsonl")}
for m in ("gemma", "llama"):
    cb, wp = rows(R / f"main/pilot_closedbook_{m}.jsonl"), rows(R / f"main/pilot_withpassage_{m}.jsonl")
    if not cb:
        problems.append(f"{m}: no closed-book answers"); continue
    empty = sum(not r["cb_answer"].strip() for r in cb) / len(cb)
    nan = sum(not math.isfinite(r["cb_mean_logprob"]) for r in cb)
    labs = [judge.get((r["pid"], m, "-", "cb_correct")) for r in cb]
    print(f"\n{m}: {len(cb)} closed-book answers, {empty:.0%} empty, {nan} NaN, judge {dict((l, labs.count(l)) for l in set(labs))}, "
          f"{len(wp)} with-passage answers")
    for r in cb[:4]:
        print(f"   Q: {pool[r['pid']]['question'][:70]}\n      gold: {pool[r['pid']]['answers'][0]} | answer: {r['cb_answer'][:60]!r} "
              f"| judge: {judge.get((r['pid'], m, '-', 'cb_correct'))}")
    if empty > 0.5: problems.append(f"{m}: more than half the closed-book answers are empty")
    if nan: problems.append(f"{m}: NaN log-probabilities")
    if None in labs: problems.append(f"{m}: some closed-book answers were not judged")
    cq = rows(R / f"confiqa/confiqa_gen_{m}.jsonl")
    if len(cq) < 10: problems.append(f"{m}: only {len(cq)} ConFiQA answers")
    elif sum(not r["wp_answer"].strip() for r in cq) > 5: problems.append(f"{m}: ConFiQA answers mostly empty")
    else:
        print(f"   ConFiQA: {cq[0]['qid']} closed-book {cq[0]['cb_answer'][:40]!r} | with passage {cq[0]['wp_answer'][:40]!r}")
    plan = [r for r in rows(R / "plan/feature_cases.jsonl") if r["model"] == m]
    F = rows(R / f"features/features_{m}.jsonl")
    if len(F) != len(plan) or not F:
        problems.append(f"{m}: features for {len(F)} of {len(plan)} planned cases"); continue
    roi = sum(r["roi"] is not None for r in F) / len(F)
    dla = sum(r["dla"] is not None for r in F) / len(F)
    bad = [r for r in F if r.get("dla") is not None and r["dla_final_identity_err"] > 0.03 * r["logit_scale"] + 0.05]
    print(f"   features: {len(F)} cases, ROI for {roi:.0%}, DLA for {dla:.0%}, read modes "
          f"{dict((k, sum(r.get('dla_mode') == k for r in F)) for k in ('diverge', 'agree'))}")
    if roi < 0.9: problems.append(f"{m}: ROI missing for {1 - roi:.0%} of cases")
    if dla < 0.9: problems.append(f"{m}: DLA missing for {1 - dla:.0%} of cases")
    if bad: problems.append(f"{m}: final-norm identity off for {len(bad)} cases")
    for r in F:
        for k, v in (r.get("dla_checks") or {}).items():
            if (k.endswith("heads_sum_vs_o_proj") or k.endswith("post_norm_linear_vs_module")) and v > 0.05:
                problems.append(f"{m}: DLA check {k} = {v}")
            if k.endswith("fast_vs_explicit") and v > 1e-3:
                problems.append(f"{m}: DLA check {k} = {v}")
if not rows(R / "confiqa/judge_confiqa.jsonl"): problems.append("ConFiQA answers were not judged")
if problems:
    print("\nSMOKE CHECK FAILED:\n  " + "\n  ".join(problems)); sys.exit(1)
print("\nSMOKE CHECK PASSED")
PY
if [ "$AUTO_FULL" = "1" ]; then
  stamp "FULL RUN (10,000 NQ questions per model + ConFiQA + features, ~3 h)"
  EXTRA="$EXTRA" JEXTRA="$JEXTRA" bash Tests/run_exp4_collect.sh full 2>&1 | tee logs/exp4_full.log
  tar -czf exp4_results.tar.gz --exclude=data/final/Phase_06/smoke data/final/Phase_06 logs/exp4_full.log logs/exp4_smoke.log
  stamp "ALL DONE -- results packed in $PROJECT/exp4_results.tar.gz (\$(du -h exp4_results.tar.gz | cut -f1))"
  echo "Copy that ONE file to your laptop (command from Claude), then you can stop the GPU."
else
  stamp "smoke test passed -- start the full run with: bash Tests/run_exp4_collect.sh full"
fi
RUN
chmod +x logs/run_exp4.sh
echo "Full log: $PROJECT/logs/exp4_session.log"
if [ -n "${TMUX:-}" ]; then
  echo "Already inside tmux -- running here. Detach with Ctrl+B then D; reattach later with: tmux attach"
  bash "$PROJECT/logs/run_exp4.sh" 2>&1 | tee -a "$PROJECT/logs/exp4_session.log"
  exit 0
fi
S=exp4
if pgrep -f "logs/run_exp4.sh" >/dev/null 2>&1 && tmux has-session -t $S 2>/dev/null; then
  echo "The run is already going in tmux session '$S' -- attaching to it"
else
  tmux kill-session -t $S 2>/dev/null || true      # an old, finished session; every step resumes safely
  tmux new-session -d -s $S "bash '$PROJECT/logs/run_exp4.sh' 2>&1 | tee -a '$PROJECT/logs/exp4_session.log'; exec bash"
fi
echo
echo "Running in tmux session '$S'. If SSH drops, reconnect and type:  tmux attach -t $S"
sleep 2
tmux attach -t $S
