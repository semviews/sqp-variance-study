#!/usr/bin/env bash
# All scored LRO-Bench queries through LRO-Bench's own pipelines, five repeats
# per implementation. Completed repeats are skipped; pass --retry-failed to
# re-run repeats that failed (e.g. after a proxy outage). Llama 3.3 70B runs on
# both stacks: the managed deployment often answers in a tool-call envelope that
# LRO-Bench cannot parse, the vllm deployment of the same weights does not.
set -u
cd "$(dirname "$0")/.."
PY=${LROBENCH:-$HOME/code/LROBench}/.venv/bin/python
LOGS=experiments/logs/lrobench
mkdir -p "$LOGS"

run() {
  local model=$1 operator=$2; shift 2
  "$PY" scripts/lrobench_repeats.py --model "$model" --operator "$operator" --repeats 5 "$@" \
    >> "$LOGS/$model-$operator.log" 2>&1
  echo "$(date -u +%FT%TZ) $model $operator exit $?" >> "$LOGS/matrix.log"
}

MODELS=${MODELS:-managed-gpt-oss-120b managed-llama-3-3-70b-instruct vllm-llama-3-3-70b-instruct vllm-granite-3-3-8b-instruct}
for model in $MODELS; do
  for operator in select match impute cluster order multi; do
    run "$model" "$operator" "$@" &
  done
done
wait
