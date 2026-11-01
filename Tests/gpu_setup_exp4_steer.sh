#!/usr/bin/env bash
# Retrieval Override -- Experiment 4, step 2: STEERING (frozen protocol v2). One model per run.
# 1. On your laptop: make steer_inputs.tar.gz (command from Claude) and copy it to the GPU box (scp).
# 2. On the GPU box, paste ONE line:
#   curl -fsSL https://raw.githubusercontent.com/Karthik-0122/Retrival_Override-/main/Tests/gpu_setup_exp4_steer.sh -o /tmp/ro_steer.sh && STEER_MODEL=gemma bash /tmp/ro_steer.sh
# Part 1 (here, ~5 min, needs you once): libraries, Hugging Face login, repo, your data file, checks.
# Part 2 (inside tmux, unattended, ~3 h): downloads, self-tests, steering, judge, A1 sheet, provisional
#        analysis, then everything packed into ONE file to copy to your laptop.
set -euo pipefail
REPO_URL="https://github.com/Karthik-0122/Retrival_Override-.git"
M="${STEER_MODEL:-gemma}"
[ "$M" = "gemma" ] || [ "$M" = "llama" ] || { echo "STEER_MODEL must be gemma or llama"; exit 1; }
if [ "$M" = "gemma" ]; then MREPO=google/gemma-2-9b; else MREPO=meta-llama/Llama-3.1-8B; fi
say() { echo; echo "========== $* =========="; }
die() { echo; echo "STOP: $*"; exit 1; }

say "1/7 GPU and disk (steering: $M)"
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
NEED_GB=$(( 60 - HAVE_GB )); [ "$NEED_GB" -lt 15 ] && NEED_GB=15
echo "Project in $BASE | model cache $HF_HOME (already ${HAVE_GB} GB) | ${FREE_GB} GB free, need ~${NEED_GB} GB"
[ "$FREE_GB" -ge "$NEED_GB" ] || die "only $FREE_GB GB free but ~$NEED_GB GB more is needed (model + judge ~55 GB). Rent 80 GB of disk."
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
$PIPI "transformers>=4.56" accelerate huggingface_hub safetensors scipy statsmodels openpyxl 2>/dev/null || \
  $PIPI --break-system-packages "transformers>=4.56" accelerate huggingface_hub safetensors scipy statsmodels openpyxl
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
for repo in ["google/gemma-2-9b", "meta-llama/Llama-3.1-8B", "Qwen/Qwen2.5-7B-Instruct"]:   # tokenizers of both are used by the self-test
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
for f in Tests/exp4_steer.py Tests/run_exp4_steer.sh Tests/exp4_steer_selftest.py Tests/exp4_collect.py Tests/gpu_utils.py \
         Tests/llm_judge.py Tests/experiment2c_batch3.py Tests/experiment2b_batch2.py \
         data/final/Phase_02/analysis_dataset_relabeled.jsonl data/final/Phase_02/retrieval_results.jsonl; do
  [ -s "$f" ] || { echo "  MISSING: $f"; missing=1; }
done
grep -q "def features_loop" Tests/exp4_collect.py || { echo "  OLD VERSION: Tests/exp4_collect.py"; missing=1; }
grep -q "a.model and" Tests/experiment2c_batch3.py || { echo "  OLD VERSION: Tests/experiment2c_batch3.py"; missing=1; }
[ "$missing" -eq 0 ] || die "files above are missing on GitHub -- push them from your laptop first, then rerun."
# the labelled data is not on GitHub: it comes in steer_inputs.tar.gz, copied from the laptop
if [ ! -s data/final/Phase_06/steer/split_$M.jsonl ]; then
  TB=""
  for t in "$PROJECT/steer_inputs.tar.gz" "$HOME/steer_inputs.tar.gz" /root/steer_inputs.tar.gz /workspace/steer_inputs.tar.gz; do
    [ -s "$t" ] && { TB="$t"; break; }
  done
  [ -n "$TB" ] || die "steer_inputs.tar.gz not found. Copy it from your laptop to $HOME on this box (scp command from Claude), then rerun the same line."
  tar -xzf "$TB" -C "$PROJECT"
  echo "unpacked $TB"
fi
for f in data/final/Phase_06/plan/feature_cases.jsonl data/final/Phase_06/features/features_$M.jsonl \
         data/final/Phase_06/exp2c/adjudication.xlsx data/final/Phase_06/exp2c/HIDDEN_key.csv \
         data/final/Phase_06/exp2c/label_fixes.csv data/final/Phase_06/main/pilot_pool.jsonl \
         data/final/Phase_06/main/judge_pilot.jsonl data/final/Phase_06/main/pilot_closedbook_$M.jsonl \
         data/final/Phase_05/exp2b/validated_labels_batch2.jsonl data/final/Phase_05/exp2b/adjudication.csv \
         data/final/Phase_06/steer/split_$M.jsonl; do
  [ -s "$f" ] || { echo "  MISSING: $f"; missing=1; }
done
[ "$missing" -eq 0 ] || die "the data file is incomplete -- make steer_inputs.tar.gz again on the laptop (command from Claude)."
cp data/final/Phase_06/steer/split_$M.jsonl /tmp/split_laptop_$M.jsonl
python Tests/exp4_steer.py split --model $M > /tmp/split_check.txt 2>&1 || { cat /tmp/split_check.txt; die "the split could not be rebuilt here"; }
cmp -s <(sort /tmp/split_laptop_$M.jsonl) <(sort data/final/Phase_06/steer/split_$M.jsonl) || die "the split rebuilt here differs from the laptop's -- stop and tell Claude"
grep -E "Validated F1|both annotators|selection F1|test +F1" /tmp/split_check.txt
echo "all scripts and data present"

say "7/7 starting the unattended run in tmux"
if   [ "$VRAM" -ge 40000 ]; then EXTRA="--token-budget 16000"; JEXTRA="--token-budget 24000"
elif [ "$VRAM" -ge 26000 ]; then EXTRA="--token-budget 8000";  JEXTRA="--token-budget 12000"
else                             EXTRA="--token-budget 4000";  JEXTRA="--token-budget 8000"; fi
echo "VRAM ${VRAM} MiB -> model batches: $EXTRA | judge batches: $JEXTRA"
mkdir -p logs
cat > logs/run_exp4_steer.sh <<RUN
#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="$HF_HOME"
cd "$PROJECT"
stamp() { echo; echo "[\$(date '+%H:%M:%S')] ===== \$* ====="; }
stamp "downloading $MREPO and the judge (~55 GB)"
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ["$MREPO", "Qwen/Qwen2.5-7B-Instruct"]:
    snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"])
    print("downloaded", repo, flush=True)
PY
df -h "$BASE" | tail -1
stamp "GPU self-tests (tiny copies of the models, ~1 min)"
python Tests/exp4_selftest.py
python Tests/exp4_steer_selftest.py
stamp "STEERING RUN: $M"
EXTRA="$EXTRA" JEXTRA="$JEXTRA" bash Tests/run_exp4_steer.sh $M 2>&1 | tee logs/exp4_steer_$M.log
tar -czf exp4_steer_${M}_results.tar.gz data/final/Phase_06/steer logs/exp4_steer_$M.log
stamp "ALL DONE -- results packed in $PROJECT/exp4_steer_${M}_results.tar.gz (\$(du -h exp4_steer_${M}_results.tar.gz | cut -f1))"
echo "Copy that ONE file to your laptop (command from Claude), then you can stop the GPU."
RUN
chmod +x logs/run_exp4_steer.sh
echo "Full log: $PROJECT/logs/exp4_steer_session.log"
if [ -n "${TMUX:-}" ]; then
  echo "Already inside tmux -- running here. Detach with Ctrl+B then D; reattach later with: tmux attach"
  bash "$PROJECT/logs/run_exp4_steer.sh" 2>&1 | tee -a "$PROJECT/logs/exp4_steer_session.log"
  exit 0
fi
S=steer
if pgrep -f "logs/run_exp4_steer.sh" >/dev/null 2>&1 && tmux has-session -t $S 2>/dev/null; then
  echo "The run is already going in tmux session '$S' -- attaching to it"
else
  tmux kill-session -t $S 2>/dev/null || true      # an old, finished session; every step resumes safely
  tmux new-session -d -s $S "bash '$PROJECT/logs/run_exp4_steer.sh' 2>&1 | tee -a '$PROJECT/logs/exp4_steer_session.log'; exec bash"
fi
echo
echo "Running in tmux session '$S'. If SSH drops, reconnect and type:  tmux attach -t $S"
sleep 2
tmux attach -t $S
