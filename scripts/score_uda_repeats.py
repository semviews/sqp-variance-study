#!/usr/bin/env python3
"""Run and score UDA-Bench queries over the tables each repeat extracted.

For every query, UDA-Bench's own GtRunner executes the query's SQL (with the
id columns its evaluator adds) over the extracted tables instead of the
ground truth; that is the system's result. UDA-Bench's run_eval then scores
the result against the ground truth with lexical comparison
(--llm-provider none), so the score of a result never depends on a model.

gt_check.json records each query's result when the ground truth itself is
passed in as the extraction. Queries that fail there are excluded; queries
whose ground-truth answer is empty, or that score below 1 on the ground truth
(lexical comparison of numbers and names), are kept for answer comparison and
flagged so quality summaries can leave them out.

Run with UDA-Bench's environment, e.g. ~/code/UDA-Bench/.venv/bin/python.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UDA = Path(os.environ.get("UDABENCH", Path.home() / "code" / "UDA-Bench"))
QUERIES = ROOT / "experiments" / "data" / "uda" / "queries"
RAW = ROOT / "experiments" / "raw" / "uda-repeats"
PROCESSED = ROOT / "experiments" / "processed" / "uda_repeats"

os.chdir(UDA)
sys.path.insert(0, str(UDA))
from evaluation import run_eval  # noqa: E402
from evaluation.tools.gt_runner import GtRunner  # noqa: E402
from evaluation.tools.query_manifest import QueryManifest  # noqa: E402
from evaluation.tools.sql_parser import SqlParser  # noqa: E402

logging.disable(logging.WARNING)


def queries(dataset: str) -> list[tuple[str, Path]]:
    base = QUERIES / dataset
    paths = sorted(base.glob("*/*/*/sql.json"), key=lambda p: (str(p.parent.parent), int(p.parent.name)))
    return [(str(path.parent.relative_to(base)), path) for path in paths]


def execute(dataset: str, sql_file: Path, tables: Path, out: Path) -> None:
    attributes = UDA / "Query" / dataset / f"{dataset}_attributes.json"
    manifest = QueryManifest.from_files(sql_file=sql_file, attributes_file=attributes, parser=SqlParser())
    runner = GtRunner(gt_dir=tables, attributes=manifest.attributes, logger_name="uda_result")
    result = runner.run(run_eval._inject_id_columns(manifest, logging.getLogger("uda_result")))
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "result.csv", index=False)


def evaluate(dataset: str, sql_file: Path, out: Path) -> dict:
    task = sql_file.parent.parent.parent.name
    argv = [
        "run_eval", "--dataset", dataset, "--task", task, "--sql-file", str(sql_file),
        "--result-csv", str(out / "result.csv"), "--output-dir", str(out / "acc_result"),
        "--llm-provider", "none", "--log-level", "ERROR",
    ]
    saved = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run_eval.main()
    finally:
        sys.argv = saved
    return json.loads((out / "acc_result" / "acc.json").read_text())


def run_one(dataset: str, sql_file: Path, tables: Path, out: Path) -> dict:
    try:
        execute(dataset, sql_file, tables, out)
        acc = evaluate(dataset, sql_file, out)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    rows = acc.get("rows", {})
    return {
        "macro_f1": acc.get("macro_f1"),
        "len_gold": rows.get("len_gold"),
        "len_pred": rows.get("len_pred"),
        "matched_rows": rows.get("matched_rows"),
        "error": "",
    }


def gt_check(dataset: str) -> dict[str, dict]:
    path = QUERIES / dataset / "gt_check.json"
    if path.exists():
        return json.loads(path.read_text())
    import tempfile

    checked = {}
    with tempfile.TemporaryDirectory() as scratch:
        for name, sql_file in queries(dataset):
            checked[name] = run_one(dataset, sql_file, UDA / "Query" / dataset, Path(scratch) / name)
    path.write_text(json.dumps(checked, indent=1))
    return checked


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_dir in sorted(RAW.iterdir()):
        dataset = dataset_dir.name
        checked = gt_check(dataset)
        valid = {name for name, result in checked.items() if not result["error"]}
        print(f"{dataset}: {len(valid)} of {len(checked)} queries run on the ground truth", flush=True)
        tables = json.loads((UDA / "Query" / dataset / f"{dataset}_attributes.json").read_text())
        for repeat_dir in sorted(dataset_dir.glob("*/*/repeat-*")):
            extract = repeat_dir / "extract"
            if any(not (extract / f"{table}.csv").exists() for table in tables):
                continue
            model, setting = repeat_dir.parent.parent.name, repeat_dir.parent.name
            extracted_at = max((extract / f"{table}.csv").stat().st_mtime for table in tables)
            for name, sql_file in queries(dataset):
                if name not in valid:
                    continue
                out = repeat_dir / "queries" / name
                cached = out / "score.json"
                if cached.exists() and cached.stat().st_mtime > extracted_at:
                    result = json.loads(cached.read_text())
                else:
                    result = run_one(dataset, sql_file, extract, out)
                    out.mkdir(parents=True, exist_ok=True)
                    cached.write_text(json.dumps(result))
                rows.append({"dataset": dataset, "model": model, "setting": setting,
                             "repeat": int(repeat_dir.name.split("-")[1]), "query": name, **result,
                             "gold_empty": int(checked[name]["len_gold"] == 0),
                             "gt_self_f1": checked[name]["macro_f1"]})
    with (PROCESSED / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "model", "setting", "repeat", "query", "macro_f1",
                                                    "len_gold", "len_pred", "matched_rows", "error",
                                                    "gold_empty", "gt_self_f1"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"scored {len(rows)} query runs -> {(PROCESSED / 'scores.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
