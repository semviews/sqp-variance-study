#!/usr/bin/env bash
# More executions on the cheap Movie queries: gpt-oss-120B and
# Granite-8B to 40 executions, Llama-70B (W) to 20. The driver skips finished
# cells, so this extends the main matrix in place. Pass deployment names to run
# a subset: gptoss, granite, llama-w.
set -u
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY=SemBench/.venvs/lotus/bin/python
LOGS=experiments/logs/sembench-repeats
mkdir -p "$LOGS"
QUERIES="2 3 4 8 9"
WHICH=${*:-gptoss granite llama-w}

run() {
  local name=$1; shift
  "$PY" scripts/sembench_repeats.py "$@" >> "$LOGS/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) $name exit $?" >> "$LOGS/matrix.log"
}

for which in $WHICH; do
  case $which in
    gptoss) run c5-gptoss --queries $QUERIES --repeats 40 & ;;
    granite) run c5-granite --model openai/vllm-granite-3-3-8b-instruct --logprobs --queries $QUERIES --repeats 40 & ;;
    llama-w) run c5-llama-w --model openai/managed-llama-3-3-70b-instruct --queries $QUERIES --repeats 20 & ;;
  esac
done
wait
