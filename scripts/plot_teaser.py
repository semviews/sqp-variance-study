#!/usr/bin/env python3
"""Figure 1: the count SemBench Movie Q3 returns in every execution, per deployment, against the gold count.

Reads the answers SemBench's runner wrote under experiments/raw/sembench-repeats/movie/
and the gold count from SemBench's Reviews.csv. Writes outputs/figures/teaser_q3.pdf.
No model is called.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_style

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "experiments" / "raw" / "sembench-repeats" / "movie"
REVIEWS = ROOT / "SemBench" / "files" / "movie" / "data" / "sf_2000" / "Reviews.csv"
OUT = ROOT / "outputs" / "figures" / "teaser_q3.pdf"
MAIN_EXECUTIONS = {"managed-gpt-oss-120b": 10}
MAIN_DEFAULT = 5
DEPLOYMENTS = [
    ("Azure-gpt-4o", "w20_lp", "GPT-4o"),
    ("aws-claude-sonnet-5", "w20", "Claude Sonnet 5"),
    ("managed-gpt-oss-120b", "w20", "gpt-oss-120B"),
    ("managed-llama-3-3-70b-instruct", "w20", "Llama-70B (M)"),
    ("vllm-llama-3-3-70b-instruct", "w20_lp", "Llama-70B (V)"),
    ("vllm-qwen2-5-72b-instruct", "w20_lp", "Qwen-72B"),
    ("managed-mistral-small-3-1-24b-2503", "w20", "Mistral-24B"),
    ("vllm-granite-3-3-8b-instruct", "w20_lp", "Granite-8B"),
]


def gold() -> int:
    """SemBench's gold SQL: COUNT(*) FROM Reviews WHERE id = 'taken_3' AND scoreSentiment = 'POSITIVE'."""
    with REVIEWS.open(newline="", encoding="utf-8") as handle:
        return sum(row["id"] == "taken_3" and row["scoreSentiment"] == "POSITIVE" for row in csv.DictReader(handle))


def counts(model: str, setting: str) -> list[int]:
    """Counts of the main executions: the first clean ones, as in analyze_sembench_repeats.py."""
    values = []
    repeats = sorted((RAW / model / setting).glob("repeat-*"), key=lambda path: int(path.name.split("-")[1]))
    for path in repeats:
        metrics_path = path / "Q3.metrics.json"
        if not metrics_path.exists():
            continue
        if json.loads(metrics_path.read_text()).get("call_errors"):
            continue
        rows = list(csv.reader((path / "Q3.csv").open(newline="", encoding="utf-8")))
        values.append(int(float(rows[1][0])))
    return values[: MAIN_EXECUTIONS.get(model, MAIN_DEFAULT)]


def main() -> None:
    plot_style.apply()
    figure, axis = plt.subplots(figsize=(plot_style.COLUMN, 1.7))
    for index, (model, setting, label) in enumerate(DEPLOYMENTS):
        tally = Counter(counts(model, setting))
        for value, number in tally.items():
            axis.scatter([value], [index], s=12 + 9 * number, color="black" if len(tally) > 1 else "grey",
                         zorder=3, linewidths=0)
            if number > 1:
                axis.annotate(f"{number}", (value, index), textcoords="offset points", xytext=(0, 5.5),
                              ha="center", fontsize=plot_style.SMALL - 0.5)
    truth = gold()
    axis.axvline(truth, color="tab:red", linestyle="--", linewidth=0.8)
    axis.annotate("gold", (truth, len(DEPLOYMENTS) - 3), xytext=(-3, 0), textcoords="offset points",
                  ha="right", va="center", color="tab:red", fontsize=plot_style.SMALL)
    axis.set_yticks(range(len(DEPLOYMENTS)))
    axis.set_yticklabels([label for _m, _s, label in DEPLOYMENTS], fontsize=plot_style.FONT - 0.5)
    axis.set_ylim(len(DEPLOYMENTS) - 0.3, -0.6)
    axis.set_xlabel("Count of positive reviews returned by Movie Q3")
    axis.grid(axis="x", linewidth=0.3, alpha=0.5)
    figure.tight_layout(pad=0.2)
    figure.savefig(OUT)


if __name__ == "__main__":
    main()
