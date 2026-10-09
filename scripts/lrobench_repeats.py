#!/usr/bin/env python3
"""Repeat LRO-Bench queries through LRO-Bench's own pipelines and operators.

Each pipeline in LRO-Bench's eval/*.py builds its operator, calls the model
through src/utils/LLMCaller.py and passes its prediction to a metric. This
driver runs those pipelines unchanged, once per repeat, and records

  * every model call (prompt hash, output, and whether the operator or the
    LLM judge of an impute metric made it), and
  * the prediction each metric received, in the form the metric compares,
    so two runs can be checked for the same answer.

LRO-Bench's main.py retries a failing query and averages up to three valid
runs; here a repeat is one run, and a failed run is retried up to three times.
The settings follow LRO-Bench's scripts/*_eval.sh without --thinking and with
the first listed example count; impute 400-404 are unscored and omitted.
LRO-Bench's impute judge uses the model under test; here it is one fixed
model for every run, so a judge failure or drift cannot change the score of
an unchanged prediction.

Run with LRO-Bench's own environment, e.g. ~/code/LROBench/.venv/bin/python.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LRO = Path(os.environ.get("LROBENCH", Path.home() / "code" / "LROBench")).resolve()
OUT = ROOT / "experiments" / "raw" / "lrobench"
ATTEMPTS = 3

# operator -> [(implementation, sort algorithm, [(first id, last id, example count)])]
SETTINGS = {
    "select": [
        (impl, None, [(100, 129, None), (200, 214, 0), (300, 314, 0)]) for impl in ("llm_all", "llm_one")
    ],
    "match": [
        (impl, None, [(100, 129, None), (200, 214, None), (300, 314, 0)])
        for impl in ("llm_all", "llm_one", "llm_semi")
    ],
    "impute": [
        ("llm_all", None, [(100, 129, None), (200, 229, None), (300, 314, 0)]),
        ("llm_one", None, [(100, 129, 0), (200, 229, None)]),
    ],
    "cluster": [
        (impl, None, [(100, 129, None), (200, 214, 0), (300, 314, 0)]) for impl in ("llm_all", "llm_one")
    ],
    "order": [
        ("llm_all", None, [(0, 29, None)]),
        ("llm_one", "simple", [(0, 29, None)]),
        ("llm_one", "heap", [(0, 29, None)]),
        ("llm_semi", None, [(0, 29, None)]),
    ],
    "multi": [("llm_all", None, [(1, 60, None)])],
}

CALLS: list[dict] | None = None
ROLE = "operator"
JUDGE_MODEL = "managed-gpt-oss-120b"


def load_env() -> None:
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    os.environ["BASE_URL"] = os.environ["LOCAL_LITELLM_ENDPOINT"]
    os.environ["API_KEY"] = os.environ["LOCAL_LITELLM_API_KEY"]


def message_key(query) -> str:
    return hashlib.sha256(json.dumps(query, sort_keys=True, default=str).encode()).hexdigest()[:16]


def record_call(query, output, started: float) -> None:
    if CALLS is not None:
        CALLS.append(
            {
                "key": message_key(query),
                "role": ROLE,
                "t": round(started, 3),
                "seconds": round(time.time() - started, 3),
                "output": output,
            }
        )


def patch_caller() -> None:
    from src.utils.LLMCaller import LLMCaller

    original_call, original_async = LLMCaller.call, LLMCaller.async_call

    def call(self, query):
        started = time.time()
        output = original_call(self, query)
        record_call(query, output, started)
        return output

    async def async_call(self, query):
        started = time.time()
        output = await original_async(self, query)
        record_call(query, output, started)
        return output

    LLMCaller.call, LLMCaller.async_call = call, async_call


def rows_as_strings(frame) -> list[str]:
    import pandas as pd

    if isinstance(frame, pd.DataFrame):
        return [str(row) for row in frame.astype(str).values]
    return [str(value) for value in frame]


def partition(labels) -> list[int]:
    """Cluster labels renamed by first appearance, so equal partitions compare equal."""
    names: dict = {}
    return [names.setdefault(str(label), len(names)) for label in labels]


def imputed_cells(frame, log) -> list[str]:
    return [str(frame.at[row, column]) for row, column, _truth in log]


# metric -> (form of the prediction the metric compares, form of the truth)
CAPTURE = {
    "f1_score": lambda truth, pred: (sorted(set(rows_as_strings(pred))), sorted(set(rows_as_strings(truth)))),
    "kendall_tau_at_k": lambda truth, pred: ([str(value) for value in pred], [str(value) for value in truth]),
    "ari": lambda truth, pred: (partition(pred), partition(truth)),
    "nmi": lambda truth, pred: (partition(pred), partition(truth)),
    "list_em_acc": lambda truth, pred: (rows_as_strings(pred), rows_as_strings(truth)),
    "list_agent_acc": lambda truth, pred: (rows_as_strings(pred), rows_as_strings(truth)),
    "df_em_acc": lambda frame, log: (imputed_cells(frame, log), [str(value) for _row, _column, value in log]),
    "df_agent_acc": lambda frame, log: (imputed_cells(frame, log), [str(value) for _row, _column, value in log]),
}
JUDGED = {"list_agent_acc", "df_agent_acc"}


def patch_metrics(module, captured: list) -> None:
    for name, capture in CAPTURE.items():
        original = getattr(module, name, None)
        if original is None:
            continue

        def wrapped(first, second, _name=name, _original=original, _capture=capture):
            global ROLE
            prediction, truth = _capture(first, second)
            model = os.environ["MODEL"]
            if _name in JUDGED:
                # LLMCaller reads MODEL when the judge constructs it.
                ROLE, os.environ["MODEL"] = "judge", JUDGE_MODEL
            try:
                value = _original(first, second)
            finally:
                ROLE, os.environ["MODEL"] = "operator", model
            captured.append({"metric": _name, "value": value, "prediction": prediction, "truth": truth})
            return value

        setattr(module, name, wrapped)


def to_jsonable(value):
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        return value
    if isinstance(value, pd.DataFrame):
        return {"columns": [str(c) for c in value.columns], "rows": value.astype(str).values.tolist()}
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def setting_name(impl: str, sort_algo: str | None) -> str:
    return impl + (f"-{sort_algo}" if sort_algo else "")


def main(argv: list[str] | None = None) -> None:
    global CALLS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model name on the LiteLLM proxy")
    parser.add_argument("--operator", required=True, choices=sorted(SETTINGS))
    parser.add_argument("--impl", default=None, help="one implementation; default all for the operator")
    parser.add_argument("--sort-algo", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--first-repeat", type=int, default=1)
    parser.add_argument("--ids", default=None, help="comma-separated query ids; default every scored id")
    parser.add_argument("--retry-failed", action="store_true", help="re-run queries whose repeat failed")
    args = parser.parse_args(argv)

    load_env()
    os.environ["MODEL"] = args.model
    os.chdir(LRO)
    sys.path.insert(0, str(LRO))
    patch_caller()
    import importlib

    module = importlib.import_module(f"eval.{args.operator}_eval")
    os.chdir(LRO)
    captured: list = []
    patch_metrics(module, captured)
    from src.core.enums import ImplType

    truths = {}
    if args.operator == "multi":
        from forward_eval_interface import build_query, score

        truths = {str(query.query_id): query for query in build_query("multi_lro")}

    wanted = {int(value) for value in args.ids.split(",")} if args.ids else None
    plans = [
        (impl, sort_algo, ranges)
        for impl, sort_algo, ranges in SETTINGS[args.operator]
        if (args.impl is None or impl == args.impl) and (args.sort_algo is None or sort_algo == args.sort_algo)
    ]
    for repeat in range(args.first_repeat, args.first_repeat + args.repeats):
        for impl, sort_algo, ranges in plans:
            folder = OUT / args.model / f"{args.operator}_{setting_name(impl, sort_algo)}" / f"repeat-{repeat}"
            folder.mkdir(parents=True, exist_ok=True)
            for first, last, example_num in ranges:
                for query_id in range(first, last + 1):
                    if wanted and query_id not in wanted:
                        continue
                    target = folder / f"{query_id}.json"
                    failed = folder / f"{query_id}.failed.json"
                    if target.exists() or (failed.exists() and not args.retry_failed):
                        continue
                    name = f"pipeline_{query_id}" if args.operator == "multi" else f"pipeline_{args.operator}{query_id}"
                    pipeline = getattr(module, name)
                    namespace = argparse.Namespace(
                        operator=args.operator,
                        impl=ImplType(impl),
                        thinking=False,
                        example_num=example_num,
                        sort_algo=sort_algo,
                    )
                    failures = []
                    for attempt in range(1, ATTEMPTS + 1):
                        CALLS, error, returned = [], None, None
                        captured.clear()
                        started = time.time()
                        try:
                            with contextlib.redirect_stdout(io.StringIO()):
                                returned = pipeline(namespace)
                        except Exception:
                            error = traceback.format_exc(limit=3)
                        seconds = time.time() - started
                        if error is None and any(call["output"] is None for call in CALLS):
                            error = "a model call failed after LLMCaller's retries"
                        if error is None:
                            break
                        failures.append(
                            {"error": error.strip().splitlines()[-1][:500],
                             "operator_calls": sum(call["role"] == "operator" for call in CALLS)}
                        )
                        print(f"{args.model} {args.operator} {impl} Q{query_id} repeat-{repeat} attempt {attempt}: "
                              f"{error.strip().splitlines()[-1]}", flush=True)
                    calls = CALLS
                    CALLS = None
                    record = {
                        "benchmark": "lrobench",
                        "operator": args.operator,
                        "impl": impl,
                        "sort_algo": sort_algo,
                        "example_num": example_num,
                        "thinking": False,
                        "query_id": query_id,
                        "model": args.model,
                        "repeat": repeat,
                        "attempt": attempt,
                        "seconds": round(seconds, 3),
                        "error": error,
                        "failed_attempts": failures if error is None else failures[:-1],
                        "judge_model": JUDGE_MODEL if args.operator == "impute" else None,
                        "operator_calls": sum(call["role"] == "operator" for call in calls),
                        "judge_calls": sum(call["role"] == "judge" for call in calls),
                    }
                    if error is None:
                        result, tokens = returned
                        record["returned"] = to_jsonable(result)
                        record["tokens"] = to_jsonable(tokens)
                        record["metrics"] = to_jsonable(list(captured))
                        if args.operator == "multi":
                            query = truths[str(query_id)]
                            record["prediction"] = to_jsonable(result)
                            record["score"] = to_jsonable(score("multi_lro", query.ground_truth, result, query.attributes))
                    with (folder / f"{query_id}.calls.jsonl").open("w", encoding="utf-8") as handle:
                        for call in calls:
                            handle.write(json.dumps(call, default=str) + "\n")
                    if error is None:
                        target.write_text(json.dumps(record, indent=1, default=str))
                        failed.unlink(missing_ok=True)
                    else:
                        failed.write_text(json.dumps(record, indent=1, default=str))
                    print(f"{args.model} {args.operator} {setting_name(impl, sort_algo)} Q{query_id} repeat-{repeat} "
                          f"ok={error is None} calls={record['operator_calls']}+{record['judge_calls']} "
                          f"seconds={seconds:.1f}", flush=True)


if __name__ == "__main__":
    main()
