#!/usr/bin/env bash
# UDA-Bench Player: extract every attribute the queries use, five repeats per
# model, then run and score the queries. Completed tables are skipped.
set -u
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY=SemBench/.venvs/lotus/bin/python
LOGS=experiments/logs/uda-repeats
mkdir -p "$LOGS"

run() {
  local name=$1; shift
  "$PY" scripts/uda_repeats.py "$@" >> "$LOGS/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) $name exit $?" >> "$LOGS/matrix.log"
}

run player-gptoss --model openai/managed-gpt-oss-120b --repeats 5 &
run player-llama-managed --model openai/managed-llama-3-3-70b-instruct --repeats 5 &
run player-granite --model openai/vllm-granite-3-3-8b-instruct --repeats 5 &
wait
"${UDA_PY:-${UDABENCH:-$HOME/code/UDA-Bench}/.venv/bin/python}" scripts/score_uda_repeats.py >> "$LOGS/score.log" 2>&1
