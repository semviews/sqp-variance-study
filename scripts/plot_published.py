#!/usr/bin/env python3
"""Figure for RQ1: published quality ranges and executions needed per comparison.

Left: share of SemBench's published headline text cells whose quality range
across five executions exceeds x, per system. Right: share of system pairs
on a query whose comparison needs more than n executions per system, for
SemBench's published systems and for our deployments. Reads
experiments/processed/published_repeats/cells.csv and
experiments/processed/runs_needed.json; writes outputs/figures/published.pdf.
No model is called.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "experiments" / "processed" / "published_repeats" / "cells.csv"
NEEDED = ROOT / "experiments" / "processed" / "runs_needed.json"
OUT = ROOT / "outputs" / "figures" / "published.pdf"
HEADLINE = {"movie": "sf2000", "cars": "sf19672", "ecomm": "sf500", "mmqa": "sf200", "medical": "default"}
NAMES = {"lotus": "LOTUS", "palimpzest": "Palimpzest", "thalamusdb": "ThalamusDB", "bigquery": "BigQuery"}


def exceed(values: list[float], xs: list[float]) -> list[float]:
    return [sum(value > x for value in values) / len(values) for x in xs]


def main() -> None:
    ranges = defaultdict(list)
    with CELLS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["text_only"] == "1" and HEADLINE.get(row["scenario"]) == row["config"]:
                ranges[row["system"]].append(float(row["range"]))
    needed = json.loads(NEEDED.read_text())

    figure, (left, right) = plt.subplots(1, 2, figsize=(3.4, 1.75))
    xs = [index / 400 for index in range(201)]
    for system, name in NAMES.items():
        if ranges.get(system):
            left.step(xs, exceed(ranges[system], xs), where="post", linewidth=1.0, label=name)
    left.set_xlim(0, 0.5)
    left.set_ylim(0, None)
    left.set_xlabel("Quality range $x$", fontsize=7)
    left.set_ylabel("Share of cells, range $> x$", fontsize=7)
    left.legend(fontsize=5.5, frameon=False, loc="upper right")

    ns = list(range(1, 101))
    for key, name, style in (("published_text", "SemBench systems", "-"), ("deployments", "our deployments", "--")):
        counts = needed[key]["needed"]
        right.step(ns, exceed(counts, ns), where="post", linestyle=style, color="black", linewidth=1.0,
                   label=name)
    right.axvline(5, color="0.6", linewidth=0.6, linestyle=":")
    right.set_xscale("log")
    right.set_xlim(1, 100)
    right.set_ylim(0, 0.3)
    right.set_xlabel("Executions per system $n$", fontsize=7)
    right.set_ylabel("Share of pairs, need $> n$", fontsize=7)
    right.legend(fontsize=5.5, frameon=False, loc="upper right")
    for axis in (left, right):
        axis.tick_params(labelsize=6)
    figure.tight_layout(pad=0.3, w_pad=0.8)
    figure.savefig(OUT)


if __name__ == "__main__":
    main()
