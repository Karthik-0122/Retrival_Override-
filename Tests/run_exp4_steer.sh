#!/usr/bin/env bash
# Experiment 4, step 2 on one GPU: steering (frozen protocol v2).
#
#   bash Tests/run_exp4_steer.sh gemma      # ~2-3 h on a 24 GB card
#   bash Tests/run_exp4_steer.sh llama      # only after Llama block 4 is labelled and adjudicated
#
# Every step resumes where it stopped: after a crash or disconnect run the same command again.
# Optional environment variables: MODEL_PATH, JUDGE, EXTRA (e.g. "--token-budget 4000"), JEXTRA.
set -euo pipefail
M=${1:-}
if [ "$M" != "gemma" ] && [ "$M" != "llama" ]; then echo "usage: bash Tests/run_exp4_steer.sh gemma|llama"; exit 1; fi
if [ "$M" = "gemma" ]; then DEF=google/gemma-2-9b; else DEF=meta-llama/Llama-3.1-8B; fi
MP=${MODEL_PATH:-$DEF}
JUDGE=${JUDGE:-Qwen/Qwen2.5-7B-Instruct}
EXTRA=${EXTRA:-}
JEXTRA=${JEXTRA:-}
log() { echo; echo "[$(date '+%H:%M:%S')] ===== $* ====="; }
START=$(date +%s)

log "1/9 split (must match the one made on the laptop)"
python Tests/exp4_steer.py split --model $M

log "2/9 original faithful set: answers with passages ($M)"
python Tests/exp4_steer.py faithful2-gen --model $M --model-path "$MP" $EXTRA

log "3/9 judge: original faithful set"
python Tests/exp4_steer.py judge --judge-path "$JUDGE" $JEXTRA

log "4/9 original faithful set: features"
python Tests/exp4_steer.py faithful2-feat --model $M --model-path "$MP"

log "5/9 head selection (selection set only)"
python Tests/exp4_steer.py select --model $M

log "6/9 steering: all head sets x strengths 0, 2, 5, 10, 20"
python Tests/exp4_steer.py steer --model $M --model-path "$MP" $EXTRA

log "7/9 judge: steered outputs"
python Tests/exp4_steer.py judge --judge-path "$JUDGE" $JEXTRA

log "8/9 A1 human check sheet"
python Tests/exp4_steer.py check-sheet --model $M

log "9/9 provisional analysis (judge only; the final one waits for the A1 sheet)"
python Tests/exp4_steer.py analyze --model $M | tee data/final/Phase_06/steer/analysis_${M}_provisional.txt

log "finished in $(( ($(date +%s) - START) / 60 )) minutes"
