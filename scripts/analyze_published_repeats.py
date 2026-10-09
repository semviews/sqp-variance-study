#!/usr/bin/env python3
"""Analyze the repeated runs that SemBench publishes in its metrics folders.

SemBench executes each (scenario, scale factor, system, query) five times
with Gemini 2.5 Flash and reports the mean. This script asks what that
mean hides: the spread of the quality metric, the gap between one run,
the mean, and the best run, and how often the order of systems changes
from one run to another.

The quality metric follows SemBench's own precedence: f1_score, then
accuracy, then relative_error, then spearman_correlation. Relative error
is reported as 1 - min(1, error) so that higher is better throughout.
No model is called.
"""

from __future__ import annotations

import csv
import json
import random
import re
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ROOT / "SemBench" / "files"
OUT = ROOT / "experiments" / "processed" / "published_repeats"
SCENARIOS = ("movie", "animals", "cars", "ecomm", "mmqa", "medical")
COVERAGE = ROOT / "SemBench" / "docs" / "static" / "data" / "paper-2025-10-01"
TEXTUAL = {"text", "table"}
SEED = 11
PRECEDENCE = ("f1_score", "accuracy", "relative_error", "spearman_correlation")
TIE = 1e-9


def quality(metrics: dict) -> tuple[str, float] | None:
    if metrics.get("status") != "success":
        return None
    for name in PRECEDENCE:
        value = metrics.get(name)
        if value is not None:
            value = float(value)
            if name == "relative_error":
                value = 1.0 - min(1.0, value)
            return name, value
    return None


def configurations(scenario: str) -> dict[str, list[Path]]:
    """Group repeat folders. A configuration is one model and scale factor."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for folder in sorted((FILES / scenario / "metrics").glob("across_system_*")):
        name = folder.name
        match = re.fullmatch(r"across_system_(2\.5flash|gemini-2\.5-flash)_(sf\d+)_repeat(\d)", name)
        if match:
            groups[f"{match.group(2)}"].append(folder)
            continue
        match = re.fullmatch(r"across_system_(2\.5flash|gemini-2\.5-flash)_(\d)", name)
        if match:
            groups["default"].append(folder)
    return {key: value for key, value in groups.items() if len(value) >= 3}


def load(folders: list[Path]) -> dict[str, dict[str, list[float | None]]]:
    """system -> query -> list of quality per repeat (None when failed)."""
    systems = sorted({path.stem for folder in folders for path in folder.glob("*.json")})
    table: dict[str, dict[str, list]] = {}
    metric_names: dict[str, str] = {}
    for system in systems:
        per_query: dict[str, list] = defaultdict(lambda: [None] * len(folders))
        rows_per_query: dict[str, list] = defaultdict(lambda: [None] * len(folders))
        for index, folder in enumerate(folders):
            path = folder / f"{system}.json"
            if not path.exists():
                continue
            for query, metrics in json.loads(path.read_text()).items():
                scored = quality(metrics)
                if scored is not None:
                    metric_names[query] = scored[0]
                    per_query[query][index] = scored[1]
                    rows_per_query[query][index] = metrics.get("row_count")
        table[system] = dict(per_query)
        ROW_COUNTS[system] = dict(rows_per_query)
    return table, metric_names


ROW_COUNTS: dict[str, dict[str, list]] = {}


def modalities(scenario: str) -> dict[str, list[str]]:
    """Query id -> the modalities SemBench lists for it in its coverage file."""
    coverage = json.loads((COVERAGE / scenario / "query" / "coverage.json").read_text())
    return {entry["id"]: sorted(entry["modalities"]) for entry in coverage["queries"]}


def empty_amid_nonempty(system: str, query: str) -> int:
    """Runs that returned no rows while another run of the same cell returned rows."""
    counts = [count for count in ROW_COUNTS.get(system, {}).get(query, []) if count is not None]
    if not counts or max(counts) == 0:
        return 0
    return sum(count == 0 for count in counts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    cell_rows, pair_rows, scenario_rows = [], [], []
    for scenario in SCENARIOS:
        kinds = modalities(scenario)
        for config, folders in configurations(scenario).items():
            table, metric_names = load(folders)
            repeats = len(folders)
            for system, queries in table.items():
                for query, values in queries.items():
                    finite = [value for value in values if value is not None]
                    if len(finite) < 3:
                        continue
                    draw = rng.choice(finite)
                    mean = statistics.fmean(finite)
                    cell_rows.append(
                        {
                            "scenario": scenario,
                            "config": config,
                            "system": system,
                            "query": query,
                            "modalities": "+".join(kinds[query]),
                            "text_only": int(set(kinds[query]) <= TEXTUAL),
                            "metric": metric_names.get(query, ""),
                            "repeats": len(finite),
                            "values": json.dumps([round(value, 4) for value in finite]),
                            "mean": mean,
                            "sd": statistics.stdev(finite),
                            "min": min(finite),
                            "max": max(finite),
                            "range": max(finite) - min(finite),
                            "random": draw,
                            "best_minus_mean": max(finite) - mean,
                            "random_minus_mean": draw - mean,
                            "empty_runs": empty_amid_nonempty(system, query),
                        }
                    )

            systems = sorted(table)
            queries = sorted({query for system in systems for query in table[system]})
            for query in queries:
                for left, right in combinations(systems, 2):
                    a = table[left].get(query)
                    b = table[right].get(query)
                    if not a or not b:
                        continue
                    both = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
                    if len(both) < 3:
                        continue
                    mean_diff = statistics.fmean(x for x, _ in both) - statistics.fmean(y for _, y in both)
                    signs = [x - y for x, y in both]
                    reversals = sum(1 for s in signs if abs(mean_diff) > TIE and s * mean_diff < -TIE)
                    ties = sum(1 for s in signs if abs(s) <= TIE)
                    pair_rows.append(
                        {
                            "scenario": scenario,
                            "config": config,
                            "query": query,
                            "text_only": int(set(kinds[query]) <= TEXTUAL),
                            "left": left,
                            "right": right,
                            "repeats": len(both),
                            "mean_diff": mean_diff,
                            "reversals": reversals,
                            "single_run_ties": ties,
                            "mean_tied": abs(mean_diff) <= TIE,
                            "any_reversal": reversals > 0,
                        }
                    )

            common = [
                query
                for query in queries
                if all(
                    table[system].get(query) and all(value is not None for value in table[system][query])
                    for system in systems
                )
            ]
            if len(systems) >= 2 and common:
                averages = {
                    system: [statistics.fmean(table[system][query][index] for query in common) for index in range(repeats)]
                    for system in systems
                }
                mean_order = sorted(systems, key=lambda system: -statistics.fmean(averages[system]))
                orders = [sorted(systems, key=lambda system, i=index: -averages[system][i]) for index in range(repeats)]
                scenario_rows.append(
                    {
                        "scenario": scenario,
                        "config": config,
                        "systems": "|".join(systems),
                        "common_queries": len(common),
                        "mean_order": "|".join(mean_order),
                        "orders": json.dumps(["|".join(order) for order in orders]),
                        "repeats_matching_mean_order": sum(order == mean_order for order in orders),
                        "repeats_matching_mean_winner": sum(order[0] == mean_order[0] for order in orders),
                        "repeats": repeats,
                    }
                )

    for name, rows in (("cells", cell_rows), ("pairs", pair_rows), ("scenario_rankings", scenario_rows)):
        with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    moving = [row for row in cell_rows if row["range"] > 1e-9]
    print(f"cells {len(cell_rows)}; metric moves in {len(moving)} ({len(moving) / len(cell_rows):.1%})")
    for threshold in (0.05, 0.1, 0.25):
        count = sum(row["range"] >= threshold for row in cell_rows)
        print(f"  range >= {threshold}: {count} ({count / len(cell_rows):.1%})")
    untied = [row for row in pair_rows if not row["mean_tied"]]
    reversed_pairs = [row for row in untied if row["any_reversal"]]
    print(
        f"system pairs on a query (mean not tied): {len(untied)}; "
        f"at least one single run reverses the mean order: {len(reversed_pairs)} ({len(reversed_pairs) / len(untied):.1%})"
    )
    total_runs = sum(row["repeats"] for row in untied)
    print(f"  single runs that reverse: {sum(row['reversals'] for row in untied)} of {total_runs}")
    for row in scenario_rows:
        print(
            f"{row['scenario']:8s} {row['config']:8s} q={row['common_queries']:2d} "
            f"order={row['mean_order']} same order {row['repeats_matching_mean_order']}/{row['repeats']} "
            f"same winner {row['repeats_matching_mean_winner']}/{row['repeats']}"
        )


if __name__ == "__main__":
    main()
