#!/usr/bin/env python3
"""Run-to-run variance of UDA-Bench queries run by scripts/uda_repeats.py.

Items are extracted values, one per (table, document, attribute), after the
same coercion the SQL sees. A query's answer is its result table (rows as a
multiset) from scripts/score_uda_repeats.py. Quality is UDA-Bench's macro F1,
summarised only over queries whose ground-truth answer is non-empty.

Writes experiments/processed/uda_repeats/{items.csv,queries.csv,facts.json}.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "experiments" / "raw" / "uda-repeats"
OUT = ROOT / "experiments" / "processed" / "uda_repeats"
QUERIES = ROOT / "experiments" / "data" / "uda" / "queries"


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def canonical(path: Path) -> str:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return "[]"

    def norm(cell: str) -> str:
        """Integers stay exact, since they include IDs; other numbers absorb float rounding."""
        try:
            number = float(cell)
        except ValueError:
            return cell.strip()
        return str(int(number)) if number.is_integer() else f"{number:.6g}"

    return json.dumps([rows[0], sorted([norm(cell) for cell in row] for row in rows[1:])])


def agreement(values: list) -> float:
    pairs = list(combinations(values, 2))
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else math.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scores = defaultdict(dict)
    score_path = OUT / "scores.csv"
    if score_path.exists():
        for row in read_rows(score_path):
            scores[(row["dataset"], row["model"], row["setting"], row["query"])][int(row["repeat"])] = row

    item_rows, query_rows = [], []
    facts = {"models": {}}
    for setting_dir in sorted(RAW.glob("*/*/*")):
        dataset, model, setting = setting_dir.parent.parent.name, setting_dir.parent.name, setting_dir.name
        repeats = sorted(setting_dir.glob("repeat-*"), key=lambda path: int(path.name.split("-")[1]))
        tables = sorted(path.name.split(".")[0] for path in (repeats[0] / "extract").glob("*.metrics.json")) if repeats else []
        complete = [path for path in repeats if all((path / "extract" / f"{t}.csv").exists() for t in tables)]
        if len(complete) < 2:
            continue

        flips = total = 0
        per_attribute = defaultdict(lambda: [0, 0])
        for table in tables:
            runs = [{row["ID"]: row for row in read_rows(path / "extract" / f"{table}.csv")} for path in complete]
            for doc in sorted(runs[0], key=lambda value: int(value) if value.isdigit() else value):
                for attribute in [column for column in runs[0][doc] if column != "ID"]:
                    values = [run.get(doc, {}).get(attribute, "") for run in runs]
                    flipped = len(set(values)) > 1
                    total += 1
                    flips += flipped
                    per_attribute[f"{table}.{attribute}"][0] += flipped
                    per_attribute[f"{table}.{attribute}"][1] += 1
                    item_rows.append({"dataset": dataset, "model": model, "setting": setting, "table": table,
                                      "doc": doc, "attribute": attribute, "flipped": int(flipped),
                                      "distinct": len(set(values)), "values": json.dumps(values)})

        changed = answered = 0
        quality_moves = 0
        spreads = []
        for manifest in sorted((QUERIES / dataset).glob("*/*/*/sql.json")):
            name = str(manifest.parent.relative_to(QUERIES / dataset))
            results = [path / "queries" / name / "result.csv" for path in complete]
            if not all(path.exists() for path in results):
                continue
            answers = [canonical(path) for path in results]
            cell_scores = scores.get((dataset, model, setting, name), {})
            f1 = [float(row["macro_f1"]) for row in cell_scores.values() if row.get("macro_f1") not in ("", None)]
            gold_empty = any(row.get("gold_empty") == "1" for row in cell_scores.values())
            answered += 1
            changed += len(set(answers)) > 1
            if f1 and not gold_empty:
                spreads.append(max(f1) - min(f1))
                quality_moves += max(f1) > min(f1)
            query_rows.append({
                "dataset": dataset, "model": model, "setting": setting, "query": name,
                "task": name.split("/")[0], "repeats": len(answers), "distinct_answers": len(set(answers)),
                "answer_reproduction": agreement(answers), "gold_empty": int(gold_empty),
                "f1_min": min(f1) if f1 else "", "f1_max": max(f1) if f1 else "",
                "f1_mean": statistics.fmean(f1) if f1 else "",
            })
        facts["models"][model] = {
            "dataset": dataset,
            "repeats": len(complete),
            "items": total,
            "item_flips": flips,
            "item_flip_rate": flips / total if total else None,
            "queries": answered,
            "answers_differ": changed,
            "quality_moves": quality_moves,
            "mean_f1_range": statistics.fmean(spreads) if spreads else None,
            "max_f1_range": max(spreads) if spreads else None,
            "attributes": {key: {"flips": value[0], "items": value[1]} for key, value in sorted(per_attribute.items())},
        }

    for name, rows in (("items.csv", item_rows), ("queries.csv", query_rows)):
        if rows:
            with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (OUT / "facts.json").write_text(json.dumps(facts, indent=1))
    for model, values in facts["models"].items():
        print(f"{model}: {values['repeats']} repeats, item flips {values['item_flips']}/{values['items']}, "
              f"answers differ on {values['answers_differ']}/{values['queries']}, F1 moves on {values['quality_moves']}")


if __name__ == "__main__":
    main()
