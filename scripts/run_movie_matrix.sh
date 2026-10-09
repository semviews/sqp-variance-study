#!/usr/bin/env bash
# Movie repeats through SemBench's LOTUS runner, one job per model deployment.
# Resumable: finished cells are skipped. Logs in experiments/logs/sembench-repeats/.
set -u
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY=SemBench/.venvs/lotus/bin/python
LOGS=experiments/logs/sembench-repeats
mkdir -p "$LOGS"
QUERIES="2 3 4 8 9 1 10"

run() {
  local name=$1; shift
  "$PY" scripts/sembench_repeats.py "$@" >> "$LOGS/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) $name exit $?" >> "$LOGS/matrix.log"
}

run movie-gptoss --queries $QUERIES --repeats 10 &
run movie-llama-vllm --model openai/vllm-llama-3-3-70b-instruct --logprobs --queries $QUERIES --repeats 5 &
run movie-qwen --model openai/vllm-qwen2-5-72b-instruct --logprobs --queries $QUERIES --repeats 5 &
run movie-llama-managed --model openai/managed-llama-3-3-70b-instruct --queries $QUERIES --repeats 5 &
run movie-mistral --model openai/managed-mistral-small-3-1-24b-2503 --queries $QUERIES --repeats 5 &
run movie-granite --model openai/vllm-granite-3-3-8b-instruct --logprobs --queries $QUERIES --repeats 5 &
wait
