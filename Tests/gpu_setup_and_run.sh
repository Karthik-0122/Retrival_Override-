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
  free=$(df -Pk "$d" | awk 'NR==2{print $4}')
  if [ "$free" -gt "$BEST" ]; then BEST=$free; BASE=$d; fi
done
[ "$BASE" = "/" ] && BASE=/root
FREE_GB=$((BEST / 1024 / 1024))
echo "Using $BASE ($FREE_GB GB free)"
[ "$FREE_GB" -ge 70 ] || die "only $FREE_GB GB free; the three models need ~50 GB plus room. Rent 100 GB of disk."
PROJECT="$BASE/Retrival_Override-"
export HF_HOME="$BASE/hf_cache"
grep -q "HF_HOME=$HF_HOME" ~/.bashrc 2>/dev/null || echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

say "2/7 system tools"
SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO=sudo
need=""; for t in git tmux; do command -v $t >/dev/null || need="$need $t"; done
if [ -n "$need" ]; then export DEBIAN_FRONTEND=noninteractive; $SUDO apt-get update -qq && $SUDO apt-get install -y -qq $need; fi
PY=$(command -v python || command -v python3) || die "no python found"
if ! command -v python >/dev/null; then $SUDO ln -sf "$PY" /usr/local/bin/python; fi   # scripts call `python`
echo "python: $PY"

say "3/7 Python libraries (PyTorch is left exactly as the image ships it)"
python - <<'PY' || exit 1
import torch, sys
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("STOP: PyTorch cannot see the GPU")
print("GPU:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
PY
PIPI="python -m pip install -q -U"
$PIPI "transformers>=4.56" accelerate huggingface_hub safetensors 2>/dev/null || \
  $PIPI --break-system-packages "transformers>=4.56" accelerate huggingface_hub safetensors
python -m pip uninstall -y -q torchvision torchaudio 2>/dev/null || true   # avoids the import crash seen last time
TEST='import torch, transformers; from transformers import AutoModelForCausalLM; print("transformers", transformers.__version__, "OK")'
if ! python -c "$TEST"; then
  echo "newest transformers does not work with this PyTorch -- installing the last 4.x release instead"
  $PIPI "transformers>=4.56,<5" 2>/dev/null || $PIPI --break-system-packages "transformers>=4.56,<5"
  python -c "$TEST" || die "transformers does not import -- paste the error above"
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
         Tests/experiment1b_rebuilt_classifier.py \
         data/final/Phase_02/analysis_dataset_relabeled.jsonl data/final/Phase_02/retrieval_results.jsonl \
         data/final/Phase_05/exp1_failure_types.jsonl data/final/Phase_05/exp2/validated_labels.jsonl \
         data/final/Phase_05/exp2/HIDDEN_key.csv data/final/Phase_05/exp2/annotation_sheet_A.csv \
         data/final/Phase_05/exp2/annotation_sheet_B.csv; do
  [ -s "$f" ] || { echo "  MISSING: $f"; missing=1; }
done
grep -q "def pick_dtype" Tests/gpu_utils.py || { echo "  OLD VERSION: Tests/gpu_utils.py (no precision check)"; missing=1; }
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
stamp "downloading models and datasets (~50 GB)"
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
if tmux has-session -t gpu 2>/dev/null; then
  echo "tmux session 'gpu' already exists -- attaching to it"
else
  tmux new-session -d -s gpu "bash '$PROJECT/logs/run_session.sh' 2>&1 | tee -a '$PROJECT/logs/session.log'; exec bash"
fi
echo
echo "Running in tmux session 'gpu'. If SSH drops, reconnect and type:  tmux attach -t gpu"
echo "Full log: $PROJECT/logs/session.log"
sleep 2
tmux attach -t gpu
SETUP
bash /tmp/ro_setup.sh