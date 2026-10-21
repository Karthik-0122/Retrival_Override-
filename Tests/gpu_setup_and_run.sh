cat > /tmp/ro_setup.sh <<'SETUP'
#!/usr/bin/env bash
# Retrieval Override -- one-paste GPU setup + run.
# Part 1 (here, ~5 min, needs you): libraries, Hugging Face login, repo, access checks.
# Part 2 (inside tmux, unattended): downloads, smoke test + automatic check, full run.
set -euo pipefail
REPO_URL="https://github.com/Karthik-0122/Retrival_Override-.git"
AUTO_FULL="${AUTO_FULL:-1}"          # 1 = start the full run automatically if the smoke test passes
say() { echo; echo "========== $* =========="; }
die() { echo; echo "STOP: $*"; exit 1; }

say "1/7 GPU and disk"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found -- this image has no GPU driver."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# put everything on the filesystem with the most free space
BASE=""; BEST=0
for d in /workspace /home "$HOME" /root /; do
  [ -d "$d" ] && [ -w "$d" ] || continue
  free=$(df -Pk "$d" 2>/dev/null | awk 'NR==2{print $4}' || true); free=${free:-0}
  if [ "$free" -gt "$BEST" ]; then BEST=$free; BASE=$d; fi
done
[ "$BASE" = "/" ] && BASE=/root
PROJECT="$BASE/Retrival_Override-"
# reuse a Hugging Face cache that already holds the models (e.g. from an earlier `hf download`)
if [ -z "${HF_HOME:-}" ]; then
  for c in "$HOME/.cache/huggingface" /root/.cache/huggingface "$BASE/hf_cache" /workspace/hf_cache; do
    if [ -d "$c/hub" ] && ls "$c/hub" 2>/dev/null | grep -q "models--"; then HF_HOME="$c"; break; fi
  done
fi
export HF_HOME="${HF_HOME:-$BASE/hf_cache}"
mkdir -p "$HF_HOME/hub"            # must exist: on a fresh box `du` below would fail and stop the script
HAVE_GB=$(du -sk "$HF_HOME/hub" 2>/dev/null | awk '{print int($1/1024/1024)}'); HAVE_GB=${HAVE_GB:-0}
FREE_GB=$(df -Pk "$HF_HOME" | awk 'NR==2{print int($4/1024/1024)}')
NEED_GB=$(( 72 - HAVE_GB )); [ "$NEED_GB" -lt 15 ] && NEED_GB=15
echo "Project in $BASE | model cache $HF_HOME (already ${HAVE_GB} GB) | ${FREE_GB} GB free, need ~${NEED_GB} GB"
[ "$FREE_GB" -ge "$NEED_GB" ] || die "only $FREE_GB GB free but ~$NEED_GB GB more is needed for the models. Rent 100 GB of disk."
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
python -m pip uninstall -y -q torchvision torchaudio 2>/dev/null || true   # avoids the import crash seen last time
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
for f in Tests/run_gpu_session.sh Tests/gpu_utils.py Tests/llm_judge.py Tests/enrichment_pilot.py \
         Tests/experiment1b_rebuilt_classifier.py Tests/gpu_selftest.py \
         data/final/Phase_02/analysis_dataset_relabeled.jsonl data/final/Phase_02/retrieval_results.jsonl \
         data/final/Phase_05/exp1_failure_types.jsonl data/final/Phase_05/exp2/validated_labels.jsonl \
         data/final/Phase_05/exp2/HIDDEN_key.csv data/final/Phase_05/exp2/annotation_sheet_A.csv \
         data/final/Phase_05/exp2/annotation_sheet_B.csv; do
  [ -s "$f" ] || { echo "  MISSING: $f"; missing=1; }
done
grep -q "token_type_ids" Tests/gpu_utils.py || { echo "  OLD VERSION: Tests/gpu_utils.py"; missing=1; }
grep -q "shuffle(order)" Tests/enrichment_pilot.py || { echo "  OLD VERSION: Tests/enrichment_pilot.py (slow question pool)"; missing=1; }
[ "$missing" -eq 0 ] || die "files above are missing on GitHub -- push them from your laptop (Step 3a), then rerun."
echo "all scripts and data present"

say "7/7 starting the unattended run in tmux"
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if   [ "$VRAM" -ge 40000 ]; then EXTRA="--token-budget 16000"; JEXTRA="--token-budget 24000"
elif [ "$VRAM" -ge 26000 ]; then EXTRA="--token-budget 8000";  JEXTRA="--token-budget 12000"
else                             EXTRA="--token-budget 4000";  JEXTRA="--token-budget 8000"; fi
echo "VRAM ${VRAM} MiB -> model batches: $EXTRA | judge batches: $JEXTRA"
mkdir -p logs
cat > logs/run_session.sh <<RUN
#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="$HF_HOME"
cd "$PROJECT"
stamp() { echo; echo "[\$(date '+%H:%M:%S')] ===== \$* ====="; }
stamp "downloading models and datasets (~68 GB: Gemma-2 is stored in 32-bit, ~37 GB)"
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
for repo in ["google/gemma-2-9b", "meta-llama/Llama-3.1-8B", "Qwen/Qwen2.5-7B-Instruct"]:
    snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"])
    print("downloaded", repo, flush=True)
for repo, f in [("Tevatron/wikipedia-nq", "nq-dev.jsonl.gz"), ("Tevatron/wikipedia-trivia", "trivia-dev.jsonl.gz")]:
    hf_hub_download(repo, f, repo_type="dataset")
    print("downloaded", repo, f, flush=True)
PY
df -h "$BASE" | tail -1
stamp "GPU self-test (tiny copies of the 3 models, ~30 s)"
python Tests/gpu_selftest.py
stamp "SMOKE TEST (20 questions)"
EXTRA="$EXTRA" JEXTRA="$JEXTRA" bash Tests/run_gpu_session.sh smoke 2>&1 | tee logs/smoke.log
stamp "automatic check of the smoke test"
python - <<'PY'
import json, math, sys
from pathlib import Path
d = Path("data/final/Phase_05/pilot_smoke")
def rows(p): return [json.loads(l) for l in open(p)] if p.exists() else []
pool = {r["pid"]: r for r in rows(d / "pilot_pool.jsonl")}
judge = {tuple(r["key"]): r["label"] for r in rows(d / "judge_pilot.jsonl")}
problems = []
for m in ("gemma", "llama"):
    cb, wp = rows(d / f"pilot_closedbook_{m}.jsonl"), rows(d / f"pilot_withpassage_{m}.jsonl")
    if not cb:
        problems.append(f"{m}: no closed-book answers"); continue
    empty = sum(not r["cb_answer"].strip() for r in cb) / len(cb)
    nan = sum(not math.isfinite(r["cb_mean_logprob"]) for r in cb)
    labs = [judge.get((r["pid"], m, "-", "cb_correct")) for r in cb]
    print(f"\n{m}: {len(cb)} closed-book answers, {empty:.0%} empty, {nan} NaN log-probs, judge labels "
          f"{ {l: labs.count(l) for l in set(labs)} }, {len(wp)} with-passage answers")
    for r in cb[:5]:
        print(f"   Q: {pool[r['pid']]['question'][:70]}\n      gold: {pool[r['pid']]['answers'][0]} | "
              f"answer: {r['cb_answer'][:70]!r} | judge: {judge.get((r['pid'], m, '-', 'cb_correct'))}")
    if empty > 0.5: problems.append(f"{m}: more than half the answers are empty")
    if nan: problems.append(f"{m}: NaN log-probabilities (precision problem)")
    if None in labs: problems.append(f"{m}: some answers were not judged")
if problems:
    print("\nSMOKE CHECK FAILED:\n  " + "\n  ".join(problems)); sys.exit(1)
print("\nSMOKE CHECK PASSED")
PY
if [ "$AUTO_FULL" = "1" ]; then
  stamp "FULL RUN (2,000 questions per model + the 566 existing cases)"
  EXTRA="$EXTRA" JEXTRA="$JEXTRA" bash Tests/run_gpu_session.sh full 2>&1 | tee logs/full.log
  stamp "ALL DONE"
  echo "Copy the results to your laptop (run this ON YOUR LAPTOP, in ~/PROJECTS/Retrival_Override):"
  echo "  scp -P <port> -r root@<host>:$PROJECT/data/final/Phase_05/pilot data/final/Phase_05/"
  echo "  scp -P <port> -r root@<host>:$PROJECT/data/final/Phase_05/judge data/final/Phase_05/"
  echo "  scp -P <port> root@<host>:$PROJECT/logs/full.log ."
else
  stamp "smoke test passed -- start the full run with: bash Tests/run_gpu_session.sh full"
fi
RUN
chmod +x logs/run_session.sh
echo "Full log: $PROJECT/logs/session.log"
if [ -n "${TMUX:-}" ]; then
  # already inside tmux (e.g. you opened `tmux new -s gpu` first): run right here, no nesting
  echo "Already inside tmux -- running here. Detach with Ctrl+B then D; reattach later with: tmux attach"
  bash "$PROJECT/logs/run_session.sh" 2>&1 | tee -a "$PROJECT/logs/session.log"
  exit 0
fi
S=ro_run
if pgrep -f "logs/run_session.sh" >/dev/null 2>&1 && tmux has-session -t $S 2>/dev/null; then
  echo "The run is already going in tmux session '$S' -- attaching to it"
else
  tmux kill-session -t $S 2>/dev/null || true      # an old, finished session; every step resumes safely
  tmux new-session -d -s $S "bash '$PROJECT/logs/run_session.sh' 2>&1 | tee -a '$PROJECT/logs/session.log'; exec bash"
fi
echo
echo "Running in tmux session '$S'. If SSH drops, reconnect and type:  tmux attach -t $S"
sleep 2
tmux attach -t $S
SETUP
bash /tmp/ro_setup.sh
