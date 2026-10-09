#!/usr/bin/env python3
"""Answers with uncertainty, fitted on five executions of gpt-oss-120B and tested on the other five.

COUNT (Movie Q3): the Poisson-binomial distribution of the count, with each
item's probability of qualifying estimated from the first five executions;
we report its central 95% interval, the counts of the later five, and the gold
count. Sets (Medical Q1, all qualifying patients; the Movie Q7 join, all
matching pairs): the certain answer (rows returned by all of the first five)
and the possible answer (rows returned by any), and how often a later
execution falls between them. Writes
experiments/processed/sembench_repeats/answer_bounds.json. No model is called.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_sembench_repeats as base  # noqa: E402
from plot_teaser import gold  # noqa: E402

MODEL = "managed-gpt-oss-120b"
FIT = 5
SETS = [("medical", 1, "w20"), ("movie", 7, "w20_exact_join40")]
OUT = base.PROCESSED / "answer_bounds.json"


def items(scenario: str, query: int, setting: str) -> list[list]:
    with (base.PROCESSED / "items.csv").open(newline="", encoding="utf-8") as handle:
        return [json.loads(row["values"]) for row in csv.DictReader(handle)
                if row["model"] == MODEL and row["scenario"] == scenario
                and int(row["query_id"]) == query and row["setting"] == setting]


def poisson_binomial(probabilities: list[float]) -> list[float]:
    distribution = [1.0]
    for p in probabilities:
        step = [0.0] * (len(distribution) + 1)
        for count, mass in enumerate(distribution):
            step[count] += mass * (1 - p)
            step[count + 1] += mass * p
        distribution = step
    return distribution


def central(distribution: list[float], level: float = 0.95) -> tuple[int, int]:
    tail = (1 - level) / 2
    total, low, high = 0.0, None, None
    for count, mass in enumerate(distribution):
        total += mass
        if low is None and total >= tail:
            low = count
        if high is None and total >= 1 - tail:
            high = count
    return low, high


def main() -> None:
    values = items("movie", 3, "w20")
    probabilities = [sum(value is True for value in runs[:FIT]) / FIT for runs in values]
    low, high = central(poisson_binomial(probabilities))
    later = [sum(runs[r] is True for runs in values) for r in range(FIT, len(values[0]))]
    facts = {"count": {"low": low, "high": high, "later": later,
                       "later_inside": sum(low <= count <= high for count in later), "gold": gold()}}

    facts["sets"] = []
    for scenario, query, setting in SETS:
        values = items(scenario, query, setting)
        certain = {i for i, runs in enumerate(values) if all(value is True for value in runs[:FIT])}
        possible = {i for i, runs in enumerate(values) if any(value is True for value in runs[:FIT])}
        answers = [{i for i, runs in enumerate(values) if runs[r] is True} for r in range(FIT, len(values[0]))]
        facts["sets"].append({
            "scenario": scenario, "query": query, "certain": len(certain), "possible": len(possible),
            "mean_answer": statistics.fmean(len(answer) for answer in answers),
            "bracketed": sum(certain <= answer <= possible for answer in answers), "later": len(answers),
            "mean_certain_missing": statistics.fmean(len(certain - answer) for answer in answers),
            "mean_outside_possible": statistics.fmean(len(answer - possible) for answer in answers),
        })
    OUT.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
