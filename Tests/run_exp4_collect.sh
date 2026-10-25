#!/usr/bin/env bash
# Experiment 4, step 1 on one GPU: collect the cases (NO steering).
#
#   bash Tests/run_exp4_collect.sh smoke     # ~10 min: 20 questions + 10 ConFiQA, checks everything runs
#   bash Tests/run_exp4_collect.sh full      # the real run: 10,000 NQ questions per model + ConFiQA (~3 h, 24 GB GPU)
#
# Every step resumes where it stopped, so after a crash or disconnect run the
# same command again. Run inside tmux so a dropped SSH does not kill it.
#
# Optional environment variables:
#   GEMMA, LLAMA, JUDGE   model ids or local folders (defaults: HF ids below)
#   EXTRA                 extra flags for Gemma/Llama, e.g. "--token-budget 4000"
#   JEXTRA                extra flags for the judge, e.g. "--token-budget 8000"
#   POOL_EXTRA            extra flags for the pool step, e.g. "--local-dir /path/to/tevatron"
set -euo pipefail
MODE=${1:-}
if [ "$MODE" != "smoke" ] && [ "$MODE" != "full" ]; then
  echo "usage: bash Tests/run_exp4_collect.sh smoke|full"; exit 1
fi
GEMMA=${GEMMA:-google/gemma-2-9b}
LLAMA=${LLAMA:-meta-llama/Llama-3.1-8B}
JUDGE=${JUDGE:-Qwen/Qwen2.5-7B-Instruct}
EXTRA=${EXTRA:-}
JEXTRA=${JEXTRA:-}
POOL_EXTRA=${POOL_EXTRA:-}
if [ "$MODE" = "smoke" ]; then
  R=data/final/Phase_06/smoke; N=20; BLOCK=5; LIM="--limit 10"
else
  R=data/final/Phase_06; N=10000; BLOCK=2500; LIM=""
fi
DIRS="--main-dir $R/main --confiqa-dir $R/confiqa --plan-dir $R/plan --feat-dir $R/features"
log() { echo; echo "[$(date '+%H:%M:%S')] ===== $* ====="; }
START=$(date +%s)

log "1/13 Option B question pool ($N NQ-train questions, blocks of $BLOCK)"
python Tests/exp4_collect.py pool --n $N --block-size $BLOCK $DIRS $POOL_EXTRA

log "2/13 closed-book answers: Gemma"
python Tests/enrichment_pilot.py closedbook --model gemma --model-path "$GEMMA" --pilot-dir $R/main $EXTRA

log "3/13 closed-book answers: Llama"
python Tests/enrichment_pilot.py closedbook --model llama --model-path "$LLAMA" --pilot-dir $R/main $EXTRA

log "4/13 judge: closed-book answers"
python Tests/llm_judge.py --tasks pilot-closedbook --model-path "$JUDGE" --pilot-dir $R/main $JEXTRA

log "5/13 with-passage answers: Gemma (gold passage alone, and with 4 distractors)"
python Tests/enrichment_pilot.py withpassage --model gemma --model-path "$GEMMA" --pilot-dir $R/main $EXTRA

log "6/13 with-passage answers: Llama"
python Tests/enrichment_pilot.py withpassage --model llama --model-path "$LLAMA" --pilot-dir $R/main $EXTRA

log "7/13 Option A: ConFiQA answers, Gemma"
python Tests/exp4_collect.py confiqa-gen --model gemma --model-path "$GEMMA" $DIRS $EXTRA $LIM

log "8/13 Option A: ConFiQA answers, Llama"
python Tests/exp4_collect.py confiqa-gen --model llama --model-path "$LLAMA" $DIRS $EXTRA $LIM

log "9/13 judge: with-passage answers and ConFiQA"
python Tests/llm_judge.py --tasks pilot-withpassage --model-path "$JUDGE" --pilot-dir $R/main $JEXTRA
python Tests/exp4_collect.py confiqa-judge --model-path "$JUDGE" $DIRS $JEXTRA

log "10/13 rebuilt classifier -> F1 candidates"
python Tests/experiment1b_rebuilt_classifier.py --task pilot --pilot-dir $R/main

log "11/13 plan: cases that get features + batch 3 samples"
python Tests/exp4_collect.py plan $DIRS

log "12/13 features (ROI + DLA): Gemma, then Llama"
python Tests/exp4_collect.py features --model gemma --model-path "$GEMMA" $DIRS
python Tests/exp4_collect.py features --model llama --model-path "$LLAMA" $DIRS

log "13/13 summary"
python Tests/exp4_collect.py summary $DIRS

log "finished in $(( ($(date +%s) - START) / 60 )) minutes"
