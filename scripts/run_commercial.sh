#!/usr/bin/env bash
# Commercial models on the SemBench text queries, five executions,
# through the enterprise endpoint. Each driver stops before a cell once the logged
# spend on the endpoint reaches CAP (USD). Resumable.
# LANES selects models (default "gpt4o sonnet5"). A lane holds a lock directory
# while it runs, so a second launch skips a lane that is already running; after a
# hard kill, remove experiments/logs/sembench-repeats/lane-<name>.lock by hand.
set -u
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY=SemBench/.venvs/lotus/bin/python
LOGS=experiments/logs/sembench-repeats
CAP=${CAP:-50}
LANES=${LANES:-gpt4o sonnet5}
mkdir -p "$LOGS"

run() {
  local name=$1; shift
  "$PY" scripts/sembench_repeats.py --endpoint enterprise --max-cost "$CAP" "$@" >> "$LOGS/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) $name exit $?" >> "$LOGS/matrix.log"
}

lane() {
  local name=$1 model=$2 extra=$3 topup=${4:-}
  local lock="$LOGS/lane-$name.lock"
  if ! mkdir "$lock" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) lane $name already running, skipped" >> "$LOGS/matrix.log"
    return 0
  fi
  trap 'rmdir "$lock"' EXIT
  run "movie-$name" --model "$model" $extra --queries 2 3 4 8 9 1 10 --repeats 5
  run "medical-$name" --scenario medical --model "$model" $extra --queries 1 4 10 --repeats 5
  # Movie queries whose executions failed all three attempts get new executions
  # (repeats 6 and 7); the analysis keeps the first five clean ones.
  if [ -n "$topup" ]; then
    run "movie-$name" --model "$model" $extra --queries $topup --first-repeat 6 --repeats 2
  fi
  rmdir "$lock" 2>/dev/null
}

# One driver per model at a time: four concurrent drivers (80 requests in flight)
# were throttled by the endpoint.
for name in $LANES; do
  case $name in
    gpt4o) (lane gpt4o openai/Azure/gpt-4o --logprobs 10) & ;;
    sonnet5) (lane sonnet5 openai/aws/claude-sonnet-5 "") & ;;
  esac
done
wait
