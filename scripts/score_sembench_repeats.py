#!/usr/bin/env python3
"""Score every repeated SemBench result with SemBench's own evaluator.

Reads experiments/raw/sembench-repeats/<scenario>/<model>/<setting>/repeat-<r>/Q<q>.csv
and writes one row per cell. Ground truth is regenerated in memory from
the gold SQL; nothing is written into the SemBench tree.

Run with SemBench/.venvs/sembench/bin/python.
"""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "SemBench" / "src"
RAW = ROOT / "experiments" / "raw" / "sembench-repeats"
OUT = ROOT / "experiments" / "processed" / "sembench_repeats" / "scores.csv"
sys.path.insert(0, str(SRC))

PRECEDENCE = ("f1_score", "accuracy", "relative_error", "spearman_correlation")


def _evaluator(scenario: str, scale: int | None, data_dir: str | None = None):
    path = SRC / "scenario" / scenario / "evaluation" / "evaluate.py"
    spec = importlib.util.spec_from_file_location(f"sembench_{scenario}_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = next(
        value
        for name, value in vars(module).items()
        if isinstance(value, type) and name.endswith("Evaluator") and name != "GenericEvaluator"
    )

    class Evaluator(base):
        def _get_ground_truth(self, query_id):
            import duckdb

            if scenario != "movie":
                return super()._get_ground_truth(query_id)
            sql = (self._root / "query" / "gold_sql" / f"Q{query_id}.sql").read_text().strip()
            conn = duckdb.connect()
            conn.register("Movies", self.movies_df)
            conn.register("Reviews", self.reviews_df)
            try:
                return conn.execute(sql).fetchdf()
            finally:
                conn.close()

    evaluator = Evaluator(scenario, scale)
    evaluator._load_domain_data()
    if data_dir:
        if scenario != "movie":
            raise ValueError(f"replacement inputs are only supported for movie, not {scenario}")
        evaluator.movies_df = pd.read_csv(ROOT / data_dir / "Movies.csv")
        evaluator.reviews_df = pd.read_csv(ROOT / data_dir / "Reviews.csv")
    return evaluator


def quality(scored: dict) -> tuple[str, float | None]:
    for name in PRECEDENCE:
        value = scored.get(name)
        if value is not None:
            value = float(value)
            return name, (1.0 - min(1.0, value)) if name == "relative_error" else value
    return "", None


def main() -> None:
    rows = []
    evaluators = {}
    truths = {}
    for result in sorted(RAW.glob("*/*/*/repeat-*/Q*.csv")):
        repeat_dir = result.parent
        setting_dir = repeat_dir.parent
        model_dir = setting_dir.parent
        scenario = model_dir.parent.name
        query = int(re.fullmatch(r"Q(\d+)", result.stem).group(1))
        meta_path = repeat_dir / f"Q{query}.metrics.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        source = (scenario, meta.get("data_dir"))
        if source not in evaluators:
            evaluators[source] = _evaluator(scenario, meta.get("scale_factor"), meta.get("data_dir"))
        evaluator = evaluators[source]
        if (source, query) not in truths:
            truths[(source, query)] = evaluator._get_ground_truth(query)
        answer = pd.read_csv(result)
        try:
            scored = dataclasses.asdict(evaluator._evaluate_single_query(query, answer, truths[(source, query)]))
            error = ""
        except Exception as exc:
            scored, error = {}, f"{type(exc).__name__}: {exc}"
        metric, value = quality(scored)
        rows.append(
            {
                "scenario": scenario,
                "model": model_dir.name,
                "setting": setting_dir.name,
                "repeat": int(repeat_dir.name.split("-")[1]),
                "query_id": query,
                "metric": metric,
                "quality": value,
                "model_calls": meta.get("model_calls"),
                "token_usage": meta.get("token_usage"),
                "execution_time": meta.get("execution_time"),
                "detail": json.dumps({k: v for k, v in scored.items() if v is not None}, default=float),
                "error": error,
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"scored {len(rows)} cells -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
