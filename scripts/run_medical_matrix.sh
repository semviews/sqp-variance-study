#!/usr/bin/env bash
# Medical text queries through SemBench's LOTUS runner, one job per model deployment.
set -u
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY=SemBench/.venvs/lotus/bin/python
LOGS=experiments/logs/sembench-repeats
mkdir -p "$LOGS"
QUERIES="1 4 10"

run() {
  local name=$1; shift
  "$PY" scripts/sembench_repeats.py --scenario medical "$@" >> "$LOGS/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) $name exit $?" >> "$LOGS/matrix.log"
}

run medical-gptoss --queries $QUERIES --repeats 10 &
run medical-llama-vllm --model openai/vllm-llama-3-3-70b-instruct --logprobs --queries $QUERIES --repeats 5 &
run medical-qwen --model openai/vllm-qwen2-5-72b-instruct --logprobs --queries $QUERIES --repeats 5 &
run medical-llama-managed --model openai/managed-llama-3-3-70b-instruct --queries $QUERIES --repeats 5 &
run medical-mistral --model openai/managed-mistral-small-3-1-24b-2503 --queries $QUERIES --repeats 5 &
run medical-granite --model openai/vllm-granite-3-3-8b-instruct --logprobs --queries $QUERIES --repeats 5 &
wait
