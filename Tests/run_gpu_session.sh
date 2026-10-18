#!/usr/bin/env bash
# One GPU session: judge + rebuilt classifier (analysis 3) + enrichment pilot.
#
#   bash Tests/run_gpu_session.sh smoke     # ~10 min incl. downloads: 20 questions, checks everything runs
#   bash Tests/run_gpu_session.sh full      # the real run (2,000 questions per model + the 566 existing cases)
#
# Every step resumes where it stopped, so after a crash or disconnect just run
# the same command again. Run inside tmux so a dropped SSH does not kill it.
#
# Optional environment variables:
#   GEMMA, LLAMA, JUDGE   model ids or local folders (defaults: HF ids below)
#   EXTRA                 extra flags for Gemma/Llama, e.g. "--use-4bit" or "--token-budget 4000"
#   JEXTRA                extra flags for the judge, e.g. "--token-budget 6000"
#   POOL_EXTRA            extra flags for the pool step, e.g. "--local-dir /path/to/tevatron"
set -euo pipefail
MODE=${1:-}
if [ "$MODE" != "smoke" ] && [ "$MODE" != "full" ]; then
  echo "usage: bash Tests/run_gpu_session.sh smoke|full"; exit 1
fi
GEMMA=${GEMMA:-google/gemma-2-9b}
LLAMA=${LLAMA:-meta-llama/Llama-3.1-8B}
JUDGE=${JUDGE:-Qwen/Qwen2.5-7B-Instruct}
EXTRA=${EXTRA:-}
JEXTRA=${JEXTRA:-}
POOL_EXTRA=${POOL_EXTRA:-}
if [ "$MODE" = "smoke" ]; then
  DIR=data/final/Phase_05/pilot_smoke; N=10; LIM="--limit 10"; JTASKS="pilot-closedbook"
else
  DIR=data/final/Phase_05/pilot; N=1000; LIM=""; JTASKS="existing,pilot-closedbook"
fi
log() { echo; echo "[$(date '+%H:%M:%S')] ===== $* ====="; }
START=$(date +%s)

log "1/8 question pool ($N per source)"
python Tests/enrichment_pilot.py pool --n-per-source $N --pilot-dir $DIR $POOL_EXTRA

log "2/8 closed-book answers: Gemma"
python Tests/enrichment_pilot.py closedbook --model gemma --model-path "$GEMMA" --pilot-dir $DIR $EXTRA $LIM

log "3/8 closed-book answers: Llama"
python Tests/enrichment_pilot.py closedbook --model llama --model-path "$LLAMA" --pilot-dir $DIR $EXTRA $LIM

log "4/8 judge: $JTASKS"
python Tests/llm_judge.py --tasks $JTASKS --model-path "$JUDGE" --pilot-dir $DIR $JEXTRA

log "5/8 with-passage answers: Gemma"
python Tests/enrichment_pilot.py withpassage --model gemma --model-path "$GEMMA" --pilot-dir $DIR $EXTRA $LIM

log "6/8 with-passage answers: Llama"
python Tests/enrichment_pilot.py withpassage --model llama --model-path "$LLAMA" --pilot-dir $DIR $EXTRA $LIM

log "7/8 judge: pilot-withpassage"
python Tests/llm_judge.py --tasks pilot-withpassage --model-path "$JUDGE" --pilot-dir $DIR $JEXTRA

log "8/8 classifier"
if [ "$MODE" = "full" ]; then
  python Tests/experiment1b_rebuilt_classifier.py --task existing
fi
python Tests/experiment1b_rebuilt_classifier.py --task pilot --pilot-dir $DIR

log "finished in $(( ($(date +%s) - START) / 60 )) minutes"