#!/usr/bin/env python3
"""How many executions a per-query comparison between two systems needs.

For each pair of systems whose mean quality on a query differs, the number
of executions per system after which a two-sided 95% interval of the
difference of means excludes zero, n* = ceil(1.96^2 (s_a^2 + s_b^2) / d^2),
with s the run-to-run standard deviation of each system and d the
difference of their means. Pairs in which neither system moves need one
execution. Two populations: SemBench's published headline text cells
(systems on Gemini 2.5 Flash) and our SemBench executions (deployments as
systems). Writes experiments/processed/runs_needed.json. No model is called.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from report_sembench_repeats import MODELS

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "experiments" / "processed" / "published_repeats" / "cells.csv"
OURS = ROOT / "experiments" / "processed" / "sembench_repeats" / "queries.csv"
OUT = ROOT / "experiments" / "processed" / "runs_needed.json"
HEADLINE = {"movie": "sf2000", "cars": "sf19672", "ecomm": "sf500", "mmqa": "sf200", "medical": "default"}
TIE = 1e-9
Z = 1.96


def needed(a: list[float], b: list[float]) -> float:
    difference = statistics.fmean(a) - statistics.fmean(b)
    variance = statistics.variance(a) + statistics.variance(b)
    if variance <= TIE:
        return 1
    return max(1, math.ceil(Z * Z * variance / (difference * difference)))


def summarise(groups: dict) -> dict:
    counts = []
    for systems in groups.values():
        for left, right in combinations(sorted(systems), 2):
            a, b = systems[left], systems[right]
            if abs(statistics.fmean(a) - statistics.fmean(b)) <= TIE:
                continue
            counts.append(needed(a, b))
    moving = [n for n in counts if n > 1]
    return {
        "pairs": len(counts),
        "pairs_needing_more_than_one": len(moving),
        "pairs_needing_more_than_five": sum(n > 5 for n in counts),
        "pairs_needing_more_than_ten": sum(n > 10 for n in counts),
        "median_when_more_than_one": statistics.median(moving) if moving else None,
        "needed": sorted(counts),
    }


def main() -> None:
    published = defaultdict(dict)
    with PUBLISHED.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["text_only"] == "1" and HEADLINE.get(row["scenario"]) == row["config"]:
                published[(row["scenario"], row["query"])][row["system"]] = json.loads(row["values"])
    ours = defaultdict(dict)
    with OURS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["setting"] in ("w20", "w20_lp") and row["model"] in MODELS and row["quality_values"]:
                ours[(row["scenario"], row["query_id"])][row["model"]] = json.loads(row["quality_values"])
    facts = {"published_text": summarise(published), "deployments": summarise(ours)}
    OUT.write_text(json.dumps(facts, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
