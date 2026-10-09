"""Split SemBench's published quality variance by whether the engine's plan is randomized.

Reads experiments/processed/published_repeats/cells.csv and classifies each
system and query by the randomness of its plan, from SemBench's runners:

- LOTUS runs with policy "approximate" (generic_lotus_runner.py). Queries
  whose implementation passes cascade arguments (Movie Q5-Q7, MMQA Q2a, Q2b,
  Q7, E-commerce Q7, Q9) use a sampled join cascade; every other LOTUS query
  issues one model call per row or pair.
- ThalamusDB processes queries approximately under constraints and samples
  the rows it evaluates.
- Palimpzest's optimizer chooses physical operators per query.
- BigQuery calls the model inside the service with a fixed SQL plan.

The deterministic-plan subset is LOTUS without a cascade, plus BigQuery.
Writes experiments/processed/published_engines.json with a "macros" map.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from analyze_sembench_repeats import wilson
from report_published_repeats import headline

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "experiments" / "processed" / "published_repeats" / "cells.csv"
OUT = ROOT / "experiments" / "processed" / "published_engines.json"

CASCADE = {("movie", "Q5"), ("movie", "Q6"), ("movie", "Q7"), ("mmqa", "Q2a"), ("mmqa", "Q2b"),
           ("mmqa", "Q7"), ("ecomm", "Q7"), ("ecomm", "Q9")}
PLAN = {
    "lotus": "per-row calls, or a sampled cascade on joins",
    "bigquery": "fixed SQL plan, model called inside the service",
    "palimpzest": "optimizer chooses physical operators",
    "thalamusdb": "approximate processing over sampled rows",
}


def plan_class(row: dict) -> str:
    if row["system"] == "lotus":
        return "randomized" if (row["scenario"], row["query"]) in CASCADE else "deterministic"
    return "deterministic" if row["system"] == "bigquery" else "randomized"


def pct(value: float) -> str:
    return f"{100 * value:.0f}\\%"


def main() -> None:
    rows = [row for row in csv.DictReader(CELLS.open()) if headline(row) and row["text_only"] == "1"]
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(plan_class(row), []).append(row)
        groups.setdefault(row["system"], []).append(row)
    summary = {}
    macros = {}
    for name, members in sorted(groups.items()):
        moving = sum(float(row["range"]) > 1e-9 for row in members)
        low, high = wilson(moving, len(members))
        summary[name] = {"cells": len(members), "moving": moving, "share": moving / len(members), "ci": [low, high]}
        key = name.capitalize().replace("db", "Db")
        macros[f"pubPlan{key}Cells"] = len(members)
        macros[f"pubPlan{key}Moving"] = moving
        macros[f"pubPlan{key}MovingPct"] = pct(moving / len(members))
        macros[f"pubPlan{key}MovingCi"] = f"[{100 * low:.0f}, {100 * high:.0f}]\\%"
    OUT.write_text(json.dumps({"plans": PLAN, "cascade_queries": sorted(map(list, CASCADE)),
                               "summary": summary, "macros": macros}, indent=1) + "\n")
    for name, values in summary.items():
        print(f"{name:14s} {values['moving']:3d}/{values['cells']:3d} {values['share']:.1%}")


if __name__ == "__main__":
    main()
