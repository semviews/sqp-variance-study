#!/usr/bin/env python3
"""Run-to-run variance of LRO-Bench queries run by scripts/lrobench_repeats.py.

A run's answer is the set of predictions its metrics received, in the form
each metric compares (see lrobench_repeats.CAPTURE); for multi-LRO queries it
is the prediction LRO-Bench scores by exact table match. A failed run (the
pipeline could not parse a reply) counts as its own outcome.

Writes experiments/processed/lrobench_repeats/{queries.csv,items.csv,facts.json}.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "experiments" / "raw" / "lrobench"
OUT = ROOT / "experiments" / "processed" / "lrobench_repeats"
FAILED = "FAILED"
JUDGE_ONLY = {"list_agent_acc", "df_agent_acc"}


def answer_of(record: dict) -> str:
    if record["operator"] == "multi":
        return json.dumps(record.get("prediction"), sort_keys=True, default=str)
    seen = []
    for metric in record.get("metrics", []):
        if metric["metric"] in JUDGE_ONLY:
            continue
        if metric["prediction"] not in seen:
            seen.append(metric["prediction"])
    return json.dumps(seen, sort_keys=True, default=str)


def quality_of(record: dict) -> float | None:
    """The headline metric LRO-Bench reports for the operator."""
    returned = record.get("returned")
    operator = record["operator"]
    try:
        if operator in ("select", "match"):
            return float(returned[2])
        if operator == "order":
            return float(returned[1])
        if operator == "cluster":
            return float(returned[0])
        if operator == "impute":
            return float(returned[0] if isinstance(returned, list) else returned)
        if operator == "multi":
            return float(record["score"]["exact_table"])
    except (TypeError, IndexError, KeyError, ValueError):
        return None
    return None


def pair_agreement(values: list) -> float:
    pairs = list(combinations(values, 2))
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else math.nan


def normalise(output: str | None) -> str | None:
    return None if output is None else " ".join(output.split()).casefold()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    query_rows, item_rows = [], []
    cells = defaultdict(dict)
    for path in sorted(RAW.glob("*/*/repeat-*/*.json")):
        repeat = int(path.parent.name.split("-")[1])
        model, setting = path.parent.parent.parent.name, path.parent.parent.name
        query = path.name.split(".")[0]
        cells[(model, setting, query)][repeat] = path

    for (model, setting, query), paths in sorted(cells.items()):
        records = {repeat: json.loads(path.read_text()) for repeat, path in sorted(paths.items())}
        outcomes, answers, qualities, calls_by_run = [], [], [], []
        for repeat, record in records.items():
            if record.get("error"):
                outcomes.append(FAILED)
                continue
            answer = answer_of(record)
            outcomes.append(answer)
            answers.append(answer)
            quality = quality_of(record)
            if quality is not None:
                qualities.append(quality)
            calls_path = paths[repeat].with_name(f"{query}.calls.jsonl")
            outputs = defaultdict(list)
            if calls_path.exists():
                for line in calls_path.read_text().splitlines():
                    call = json.loads(line)
                    if call["role"] == "operator":
                        outputs[call["key"]].append(normalise(call.get("output")))
            calls_by_run.append({f"{key}#{index}": value for key, values in outputs.items()
                                 for index, value in enumerate(values)})
        first = next(iter(records.values()))
        operator, impl = first["operator"], first["impl"]

        flips = items = 0
        if len(calls_by_run) >= 2:
            shared = set.intersection(*(set(run) for run in calls_by_run))
            for key in sorted(shared):
                values = [run[key] for run in calls_by_run]
                flipped = len(set(values)) > 1
                items += 1
                flips += flipped
                if flipped:
                    item_rows.append({"model": model, "setting": setting, "query": query, "item": key,
                                      "distinct": len(set(values))})
        query_rows.append({
            "model": model,
            "setting": setting,
            "operator": operator,
            "impl": impl,
            "query": query,
            "repeats": len(outcomes),
            "failed": outcomes.count(FAILED),
            "ok": len(answers),
            "distinct_answers": len(set(answers)),
            "answer_reproduction": pair_agreement(answers),
            "outcome_reproduction": pair_agreement(outcomes),
            "quality_min": min(qualities) if qualities else "",
            "quality_max": max(qualities) if qualities else "",
            "quality_mean": statistics.fmean(qualities) if qualities else "",
            "calls_compared": items,
            "calls_text_differs": flips,
        })

    with (OUT / "queries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(query_rows[0]))
        writer.writeheader()
        writer.writerows(query_rows)
    with (OUT / "items.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "setting", "query", "item", "distinct"])
        writer.writeheader()
        writer.writerows(item_rows)

    facts = {"by_model": {}, "by_setting": {}}
    complete = [row for row in query_rows if row["repeats"] >= 5]
    groups = defaultdict(list)
    for row in complete:
        groups[("by_model", row["model"])].append(row)
        groups[("by_setting", f"{row['model']}|{row['setting']}")].append(row)
    for (kind, key), rows in sorted(groups.items()):
        answered = [row for row in rows if row["ok"] >= 2]
        spread = [row["quality_max"] - row["quality_min"] for row in rows if row["quality_min"] != ""]
        facts[kind][key] = {
            "queries": len(rows),
            "runs": sum(row["repeats"] for row in rows),
            "failed_runs": sum(row["failed"] for row in rows),
            "queries_any_failure": sum(row["failed"] > 0 for row in rows),
            "queries_mixed_failure": sum(0 < row["failed"] < row["repeats"] for row in rows),
            "answered_queries": len(answered),
            "answers_differ": sum(row["distinct_answers"] > 1 for row in answered),
            "mean_answer_reproduction": statistics.fmean(row["answer_reproduction"] for row in answered)
            if answered else None,
            "outcomes_differ": sum(row["outcome_reproduction"] < 1 for row in rows),
            "quality_moves": sum(value > 0 for value in spread),
            "mean_quality_range": statistics.fmean(spread) if spread else None,
            "max_quality_range": max(spread) if spread else None,
            "calls_compared": sum(row["calls_compared"] for row in rows),
            "calls_text_differs": sum(row["calls_text_differs"] for row in rows),
        }
    (OUT / "facts.json").write_text(json.dumps(facts, indent=1))
    print(f"{len(query_rows)} cells ({len(complete)} with 5 repeats) -> {OUT.relative_to(ROOT)}")
    for model, values in facts["by_model"].items():
        print(f"  {model}: {values['answers_differ']}/{values['answered_queries']} answers differ, "
              f"{values['queries_any_failure']}/{values['queries']} with a failed run, "
              f"quality moves on {values['quality_moves']}")


if __name__ == "__main__":
    main()
