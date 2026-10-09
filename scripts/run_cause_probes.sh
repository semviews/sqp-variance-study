#!/usr/bin/env bash
# Cause probes on Movie Q3 (filter), Q8 (label map), Q9 (score map).
# One factor varies per probe. Settings are interleaved within each repeat
# so that shared-provider load over time is balanced across settings.
# Granite-8B is the reference: it flips often enough to show an effect and
# returns log-probabilities. Probe settings carry the "probe" tag so they are
# never pooled with the main matrix.
set -u
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY=SemBench/.venvs/lotus/bin/python
LOGS=experiments/logs/sembench-repeats
mkdir -p "$LOGS"
QUERIES="3 8 9"
REF=openai/vllm-granite-3-3-8b-instruct
REPEATS=${REPEATS:-10}

for repeat in $(seq 1 "$REPEATS"); do
  for workers in 1 8 20 64; do
    "$PY" scripts/sembench_repeats.py --model $REF --logprobs --workers $workers --tag probe \
      --queries $QUERIES --first-repeat "$repeat" --repeats 1 >> "$LOGS/probe-concurrency.log" 2>&1
  done
  "$PY" scripts/sembench_repeats.py --model $REF --logprobs --shuffle --tag probe \
    --queries $QUERIES --first-repeat "$repeat" --repeats 1 >> "$LOGS/probe-shuffle.log" 2>&1
  for effort in low medium high; do
    "$PY" scripts/sembench_repeats.py --model openai/managed-gpt-oss-120b --reasoning $effort --tag probe \
      --queries $QUERIES --first-repeat "$repeat" --repeats 1 >> "$LOGS/probe-reasoning.log" 2>&1
  done
  echo "$(date -u +%FT%TZ) probes repeat $repeat done" >> "$LOGS/matrix.log"
done
