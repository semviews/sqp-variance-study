#!/usr/bin/env bash
# Regenerate every processed file, and every table, figure, and macro under outputs/. No model is called.
set -eu
cd "$(dirname "$0")/.."
mkdir -p outputs/tables outputs/figures
PY=${PY:-SemBench/.venvs/sembench/bin/python}
UDA_PY=${UDA_PY:-${UDABENCH:-$HOME/code/UDA-Bench}/.venv/bin/python}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl}
"$PY" scripts/analyze_published_repeats.py > /dev/null
"$PY" scripts/report_published_repeats.py > /dev/null
"$PY" scripts/published_engines.py > /dev/null
"$PY" scripts/score_sembench_repeats.py 2> /dev/null | tail -1
"$PY" scripts/analyze_sembench_repeats.py > /dev/null
"$PY" scripts/margin_noise.py > /dev/null
"$PY" scripts/report_sembench_repeats.py > /dev/null
"$PY" scripts/boost_analyses.py > /dev/null
"$PY" scripts/more_executions.py > /dev/null
"$PY" scripts/commercial_report.py > /dev/null
"$PY" scripts/campaigns.py > /dev/null
"$PY" scripts/remedies_sembench_repeats.py > /dev/null
"$PY" scripts/selective_vote.py > /dev/null
"$PY" scripts/answer_bounds.py > /dev/null
"$PY" scripts/report_probes.py > /dev/null
"$PY" scripts/temperature_analysis.py > /dev/null
"$PY" scripts/reasoning_divergence.py > /dev/null
"$PY" scripts/analyze_lrobench_repeats.py > /dev/null
"$PY" scripts/lrobench_failures.py > /dev/null
"$PY" scripts/claim_audit.py > /dev/null
"$UDA_PY" scripts/score_uda_repeats.py > /dev/null
"$PY" scripts/analyze_uda_repeats.py > /dev/null
"$PY" scripts/uda_lineage.py > /dev/null
"$PY" scripts/exposure_rules.py > /dev/null
"$PY" scripts/report_text_suites.py > /dev/null
"$PY" scripts/report_item_examples.py > /dev/null
"$PY" scripts/report_ladder.py > /dev/null
"$PY" scripts/cascade_replay.py > /dev/null
"$PY" scripts/runs_needed.py > /dev/null
"$PY" scripts/plot_published.py
"$PY" scripts/plot_teaser.py
"$PY" scripts/plot_amplification.py > /dev/null
"$PY" scripts/paper_macros.py
