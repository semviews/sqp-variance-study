#!/usr/bin/env bash
# The later model-call campaigns, in one resumable job: the temperature
# control, the commercial models, and more executions on the cheap Movie
# queries. gpt-oss-120B's extra
# executions start only after the temperature control has finished, so
# the two never share the deployment. Rerun to resume.
set -u
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY=SemBench/.venvs/lotus/bin/python
LOGS=experiments/logs
mkdir -p "$LOGS/controls" "$LOGS/sembench-repeats"

control() {
  for query in 3 1; do
    for setting in t0 t0-seed t1; do
      "$PY" scripts/temperature_control.py --query $query --setting $setting >> "$LOGS/controls/temperature-Q$query-$setting.log" 2>&1 &
    done
  done
  wait
  echo "$(date -u +%FT%TZ) temperature control done" >> "$LOGS/sembench-repeats/matrix.log"
}

(control; scripts/run_more_executions.sh gptoss) &
scripts/run_commercial.sh &
scripts/run_more_executions.sh granite llama-w &
wait
echo "$(date -u +%FT%TZ) boost calls done" >> "$LOGS/sembench-repeats/matrix.log"
