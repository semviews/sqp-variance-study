#!/usr/bin/env python3
"""Five temperature-0 repeats of product-review text operators.

One hundred synthetic reviews. Filters and maps only, each a single LOTUS
sem_filter or sem_map call with LOTUS's cache disabled. Without --execute
this prints the plan.

Run with SemBench/.venvs/lotus/bin/python.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from sembench_repeats import _env

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "data" / "product-reviews" / "product_reviews.csv"
OUT = ROOT / "experiments" / "raw" / "product-reviews" / "sf10"
MODEL = "openai/managed-gpt-oss-120b"

TASKS = (
    {
        "task_id": "positive",
        "operator": "filter",
        "field": "description",
        "instruction": (
            "Determine if the following product review is positive. "
            'Review: "{description}".'
        ),
    },
    {
        "task_id": "return",
        "operator": "filter",
        "field": "description",
        "instruction": (
            "Determine if the review mentions returning or refunding the product. "
            'Review: "{description}".'
        ),
    },
    {
        "task_id": "defect",
        "operator": "filter",
        "field": "description",
        "instruction": (
            "Determine if the review says the product is defective, broken, or failed. "
            'Review: "{description}".'
        ),
    },
    {
        "task_id": "sentiment",
        "operator": "map",
        "field": "description",
        "instruction": (
            "Classify the sentiment of this product review as Positive, Negative, or Neutral. "
            "Only output one of those three words. Review: {description}"
        ),
    },
    {
        "task_id": "review_type",
        "operator": "map",
        "field": "description",
        "instruction": (
            "Classify this product review as exactly one of: praise, complaint, "
            "feature-detail, general, question. Output only that label. Review: {description}"
        ),
    },
    {
        "task_id": "brand",
        "operator": "map",
        "field": "description",
        "instruction": (
            "Name the brand this product review is about. Output only the brand name. "
            "Review: {description}"
        ),
    },
    {
        "task_id": "city",
        "operator": "map",
        "field": "customer_location",
        "instruction": "Extract the city from this customer location. Output only the city name. Location: {customer_location}",
    },
    {
        "task_id": "state",
        "operator": "map",
        "field": "customer_location",
        "instruction": "Extract the U.S. state name from this customer location. Output only the state name. Location: {customer_location}",
    },
    {
        "task_id": "product_type",
        "operator": "map",
        "field": "description",
        "instruction": "Name the type of product in this review, such as microwave or perfume. Output only the product type. Review: {description}",
    },
    {
        "task_id": "sentiment_score",
        "operator": "map",
        "field": "description",
        "instruction": "Score the sentiment of this review from 1 to 10. Use 1-3 for negative, 4-7 for neutral, and 8-10 for positive. Output only the integer. Review: {description}",
    },
    {
        "task_id": "quality_score",
        "operator": "map",
        "field": "description",
        "instruction": "Score the product quality in this review from 1 to 5. Output only the integer. Review: {description}",
    },
    {
        "task_id": "summary",
        "operator": "map",
        "field": "description",
        "instruction": "Summarize this review in one short sentence that names the sentiment, the brand, and the product type. Review: {description}",
    },
    {
        "task_id": "department",
        "operator": "map",
        "field": "description",
        "instruction": (
            "Classify the department of this product review as exactly one of: "
            "Kitchen, Electronics, Beauty, Sports, Home, Books, Clothing, Toys. "
            "Output only that word. Review: {description}"
        ),
    },
)


def _records(field: str) -> list[dict[str, str]]:
    with DATA.open(newline="", encoding="utf-8") as handle:
        return [{"id": row["id"], field: row[field]} for row in csv.DictReader(handle)]


def _execute(operator: str, records: list[dict[str, str]], instruction: str) -> tuple[dict, dict]:
    """One LOTUS filter or map over all records at temperature 0: labels by id, and token usage."""
    import lotus
    import pandas as pd
    from lotus.models import LM

    language_model = LM(MODEL, temperature=0.0, max_tokens=8192, max_batch_size=8)
    lotus.settings.configure(lm=language_model)
    lotus.settings.enable_cache = False
    frame = pd.DataFrame(records)
    if operator == "filter":
        kept = set(frame.sem_filter(instruction)["id"].astype(str))
        labels = {str(row["id"]): str(row["id"]) in kept for row in records}
    else:
        mapped = frame.sem_map(instruction)
        labels = {str(row_id): value for row_id, value in zip(frame["id"].astype(str), mapped["_map"], strict=True)}
    usage = language_model.stats.physical_usage
    return labels, {"total_tokens": usage.total_tokens, "model_calls": len(records)}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repeat product-review text operators")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    print(f"reviews: {DATA}")
    print(f"tasks: {', '.join(task['task_id'] for task in TASKS)}")
    print(f"repeats: {args.repeats} temperature 0")
    if not args.execute:
        return 0
    _env("local")
    for task in TASKS:
        records = _records(task["field"])
        for repeat in range(1, args.repeats + 1):
            destination = OUT / task["task_id"] / f"repeat-{repeat}" / "labels.json"
            if destination.exists():
                print(f"skip {task['task_id']} repeat-{repeat}", flush=True)
                continue
            print(f"start {task['task_id']} repeat-{repeat}", flush=True)
            started = time.perf_counter()
            labels, usage = None, None
            last_error = "unknown"
            for attempt in range(1, 4):
                try:
                    labels, usage = _execute(task["operator"], records, task["instruction"])
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    print(f"retry {task['task_id']} repeat-{repeat} attempt {attempt}: {type(exc).__name__}", flush=True)
                    time.sleep(5)
            if labels is None or usage is None:
                print(f"fail {task['task_id']} repeat-{repeat}: {last_error[:200]}", flush=True)
                return 1
            elapsed = time.perf_counter() - started
            _write(
                destination,
                {
                    "task_id": task["task_id"],
                    "operator": task["operator"],
                    "repeat": repeat,
                    "temperature": 0.0,
                    "reviews": len(records),
                    "model": MODEL,
                    "labels": {str(key): value for key, value in labels.items()},
                    "model_calls": usage.get("model_calls", len(records)),
                    "total_tokens": usage.get("total_tokens", 0),
                    "execution_time": elapsed,
                    "status": "success",
                },
            )
            print(f"finish {task['task_id']} repeat-{repeat} seconds={elapsed:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
