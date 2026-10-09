#!/usr/bin/env bash
# Movie Q5-Q7 (exact semantic self-joins) on 40 distinct reviews of the joined
# film; build the inputs with scripts/make_movie_join_subset.py. Completed
# cells are skipped, so the script can be re-run to fill gaps.
set -u
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY=SemBench/.venvs/lotus/bin/python
LOGS=experiments/logs/sembench-repeats
JOINS="7 5 6"
COMMON=(--policy exact --data-dir experiments/data/movie-join40 --tag join40 --queries $JOINS)
mkdir -p "$LOGS"

run() {
  local name=$1; shift
  "$PY" scripts/sembench_repeats.py "$@" >> "$LOGS/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) $name exit $?" >> "$LOGS/matrix.log"
}

run joins-gptoss "${COMMON[@]}" --repeats 10 &
run joins-llama-managed --model openai/managed-llama-3-3-70b-instruct "${COMMON[@]}" --repeats 5 &
run joins-mistral --model openai/managed-mistral-small-3-1-24b-2503 "${COMMON[@]}" --repeats 5 &
run joins-llama-vllm --model openai/vllm-llama-3-3-70b-instruct --logprobs "${COMMON[@]}" --repeats 5 &
run joins-qwen --model openai/vllm-qwen2-5-72b-instruct --logprobs "${COMMON[@]}" --repeats 5 &
run joins-granite --model openai/vllm-granite-3-3-8b-instruct --logprobs "${COMMON[@]}" --repeats 5 &
wait
