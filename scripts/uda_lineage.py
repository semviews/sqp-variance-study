"""Predict UDA-Bench answer reproduction from the extracted values each query reads.

Each query's sql.json names the tables and attributes its SQL reads. The
query's exposure x_q is the sum of the pairwise disagreement delta of every
extracted (document, attribute) value of those attributes, over all
documents of the table: the expected number of read values on which two
executions differ. A filter reads its attribute on every document, so this
is exact for filters and an upper bound for projected attributes and
aggregates. The possible-worlds model predicts reproduction e^{-x_q} for
answers that expose every value read.

Writes experiments/processed/uda_lineage.json with a "macros" map.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "experiments" / "processed"
QUERIES = ROOT / "experiments" / "data" / "uda" / "queries"
OUT = PROCESSED / "uda_lineage.json"
MACRO = {"managed-gpt-oss-120b": "Gptoss", "managed-llama-3-3-70b-instruct": "LlamaW",
         "vllm-granite-3-3-8b-instruct": "Granite"}
TOLERANCE = 0.15


def disagreement(values: list[str]) -> float:
    pairs = list(combinations(values, 2))
    return sum(a != b for a, b in pairs) / len(pairs)


def reads(dataset: str, query: str) -> list[tuple[str, str]]:
    spec = json.loads((QUERIES / dataset / query / "sql.json").read_text())
    return [(table, attribute) for table, attributes in spec.items() if table != "sql"
            for attribute in attributes]


def main() -> None:
    delta: dict[tuple, dict[tuple, float]] = defaultdict(dict)
    for row in csv.DictReader((PROCESSED / "uda_repeats" / "items.csv").open()):
        key = (row["dataset"], row["model"], row["setting"])
        delta[key][(row["table"], row["attribute"], row["doc"])] = disagreement(json.loads(row["values"]))
    by_attribute: dict[tuple, dict[tuple, float]] = defaultdict(lambda: defaultdict(float))
    for key, values in delta.items():
        for (table, attribute, _), value in values.items():
            by_attribute[key][(table, attribute)] += value

    points, macros, summary = [], {}, {}
    for row in csv.DictReader((PROCESSED / "uda_repeats" / "queries.csv").open()):
        key = (row["dataset"], row["model"], row["setting"])
        exposure = sum(by_attribute[key].get(item, 0.0) for item in reads(row["dataset"], row["query"]))
        measured = float(row["answer_reproduction"])
        points.append({"model": row["model"], "query": row["query"], "task": row["task"],
                       "exposure": exposure, "predicted": math.exp(-exposure), "measured": measured})

    for model, suffix in MACRO.items():
        rows = [point for point in points if point["model"] == model]
        if not rows:
            continue
        within = sum(abs(point["predicted"] - point["measured"]) <= TOLERANCE for point in rows)
        above = sum(point["measured"] > point["predicted"] + TOLERANCE for point in rows)
        tasks = {}
        for task in sorted({point["task"] for point in rows}):
            members = [point for point in rows if point["task"] == task]
            tasks[task] = {
                "queries": len(members),
                "within": sum(abs(point["predicted"] - point["measured"]) <= TOLERANCE for point in members),
                "mean_error": sum(abs(point["predicted"] - point["measured"]) for point in members) / len(members),
            }
        summary[model] = {"queries": len(rows), "within": within, "above": above,
                          "median_exposure": sorted(point["exposure"] for point in rows)[len(rows) // 2],
                          "tasks": tasks}
        macros[f"udaLinQueries{suffix}"] = len(rows)
        macros[f"udaLinWithin{suffix}"] = within
        macros[f"udaLinWithinPct{suffix}"] = f"{100 * within / len(rows):.0f}\\%"
        macros[f"udaLinAbove{suffix}"] = above
        macros[f"udaLinBelow{suffix}"] = len(rows) - within - above
        macros[f"udaLinMedExposure{suffix}"] = f"{summary[model]['median_exposure']:.1f}"
        for task, values in tasks.items():
            macros[f"udaLinWithin{task}{suffix}"] = f"{values['within']} of {values['queries']}"
    OUT.write_text(json.dumps({"points": points, "summary": summary, "macros": macros}, indent=1) + "\n")
    for model, values in summary.items():
        print(model, {k: v for k, v in values.items() if k != "tasks"})
        for task, stats in values["tasks"].items():
            print("   ", task, stats)


if __name__ == "__main__":
    main()
