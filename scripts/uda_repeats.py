#!/usr/bin/env python3
"""Repeat UDA-Bench queries with the extract-then-query plan of its LOTUS adapter.

UDA-Bench's systems/Lotus/extract.py extracts each attribute a query needs
from every document with one sem_map call per (document, attribute), using
the instruction, descriptions, and examples in that folder; the query is then
plain SQL over the extracted tables (scripts/score_uda_repeats.py runs and
scores it). The instruction carries the closing
sentence that the adapter's filter.py adds ("Please keep each extracted value
concise ..."); without it, tables that have no examples get prose answers. This driver does the same for every
attribute the dataset's queries use, once per repeat, then runs each query's
SQL with duckdb. The adapter only ships descriptions and examples for the
player table; the other tables use the descriptions in the dataset's
*_attributes.json and no examples.

Before the SQL runs, each extracted value is cut to its first non-empty line
(models that explain themselves put the value first), coerced to the type of
the matching ground-truth column (a number is read from the text when that
column is numeric), and the adapter's "empty" marker becomes NULL. The raw
outputs are kept next to the coerced tables.

Run with SemBench/.venvs/lotus/bin/python.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sembench_repeats as common  # noqa: E402

ROOT = common.ROOT
UDA = Path(os.environ.get("UDABENCH", Path.home() / "code" / "UDA-Bench"))
QUERIES = ROOT / "experiments" / "data" / "uda" / "queries"
OUT = ROOT / "experiments" / "raw" / "uda-repeats"
FOLDERS = {"Player": {"player": "player", "team": "team", "city": "city", "manager": "owner"}}
ADAPTER_KEYS = {"Player": {"player": "player"}}


def manifests(dataset: str) -> list[tuple[str, dict]]:
    """(query path relative to the dataset, sql.json) for every prepared query."""
    base = QUERIES / dataset
    return [
        (str(path.parent.relative_to(base)), json.loads(path.read_text()))
        for path in sorted(base.glob("*/*/*/sql.json"), key=lambda p: (p.parent.parent, int(p.parent.name)))
    ]


def attributes_used(dataset: str) -> dict[str, list[str]]:
    used: dict[str, set] = {}
    for _name, manifest in manifests(dataset):
        for table, columns in manifest.items():
            if table != "sql":
                used.setdefault(table, set()).update(columns)
    return {table: sorted(columns) for table, columns in sorted(used.items())}


def instructions(dataset: str, table: str, attribute: str):
    """The adapter's instruction and examples for one attribute."""
    import pandas as pd

    adapter = UDA / "systems" / "Lotus"
    key = ADAPTER_KEYS.get(dataset, {}).get(table)
    examples = None
    if key:
        names = json.loads((adapter / "extractions.json").read_text())[key]
        if attribute in names:
            index = names.index(attribute)
            description = json.loads((adapter / "descriptions.json").read_text())[key][index]
            examples = pd.DataFrame(json.loads((adapter / "examples.json").read_text())[key][index])
        else:
            key = None
    if not key:
        schema = json.loads((UDA / "Query" / dataset / f"{dataset}_attributes.json").read_text())
        description = schema[table][attribute]["description"]
    instruction = (
        "What" + attribute + "in {context}?" + description
        + "If there are multiple values, separate them with '||' and leave empty if not applicable."
        + " Please keep each extracted value concise and avoid lengthy content."
    )
    return instruction, examples


def coerce(value, numeric: bool):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = next((line.strip() for line in str(value).splitlines() if line.strip()), "")
    if not text or "empty" in text.lower():
        return None
    if not numeric:
        return text
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not match:
        return None
    number = float(match.group(0).replace(",", ""))
    return int(number) if number.is_integer() else number


RETRY = "\x00retry"
ATTEMPTS = 5


def documents(dataset: str, table: str):
    import pandas as pd

    truth = pd.read_csv(UDA / "Query" / dataset / f"{table}.csv")
    docs = UDA / "datasets" / dataset / FOLDERS[dataset][table]
    ids = [str(int(value)) if float(value).is_integer() else str(value) for value in truth["ID"].dropna()]
    texts = [(docs / f"{doc}.txt").read_text(encoding="utf-8").strip() for doc in ids]
    return truth, pd.DataFrame({"context": texts}, index=ids)


def extract_attribute(calls: list, dataset: str, table: str, attribute: str, frame, path: Path) -> bool:
    """One value per document; a document is asked again only after a transient failure."""
    instruction, examples = instructions(dataset, table, attribute)
    outputs: dict[str, str] = {}
    log: list[dict] = []
    for attempt in range(1, ATTEMPTS + 1):
        todo = [doc for doc in frame.index if doc not in outputs]
        if not todo:
            break
        start = len(calls)
        try:
            subset = frame.loc[todo]
            mapped = subset.sem_map(instruction, examples=examples) if examples is not None else subset.sem_map(instruction)
        except Exception:
            print(f"{table}.{attribute} attempt {attempt}: {traceback.format_exc(limit=2).strip().splitlines()[-1][:200]}",
                  flush=True)
            time.sleep(30 * attempt)
            continue
        for doc, call, value in zip(todo, calls[start:], mapped["_map"]):
            call.update({"table": table, "attribute": attribute, "doc": doc, "attempt": attempt})
            if value != RETRY:
                outputs[doc] = value
        log.extend(calls[start:])
        if len(outputs) < len(frame.index):
            time.sleep(30 * attempt)
    if len(outputs) < len(frame.index):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"outputs": [outputs[doc] for doc in frame.index], "calls": log}, default=str))
    return True


def assemble(dataset: str, table: str, columns: list[str], truth, frame, folder: Path) -> list[dict]:
    import pandas as pd

    ids = list(frame.index)
    raw, extracted, calls = {"ID": ids}, {"ID": ids}, []
    for column in columns:
        part = json.loads((folder / "parts" / table / f"{column}.json").read_text())
        numeric = column in truth and pd.api.types.is_numeric_dtype(truth[column])
        raw[column] = part["outputs"]
        extracted[column] = [coerce(value, numeric) for value in part["outputs"]]
        calls.extend(part["calls"])
    (folder / "raw").mkdir(exist_ok=True)
    pd.DataFrame(raw).to_csv(folder / "raw" / f"{table}.csv", index=False)
    pd.DataFrame(extracted, dtype=object).to_csv(folder / f"{table}.csv", index=False)
    return calls


def drop_float_suffix(path: Path, truth) -> None:
    """Undo pandas' float upcast of integer columns that hold a NULL; coerce() emits integers as int."""
    import pandas as pd

    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in table.columns:
        if column in truth and pd.api.types.is_numeric_dtype(truth[column]):
            table[column] = table[column].str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)
    table.to_csv(path, index=False)


def answered(response) -> bool:
    try:
        return response.choices[0].message.content is not None
    except (AttributeError, IndexError):
        return False


TRANSPORT = "Invalid HTTP request received"


def rejected(error: Exception) -> bool:
    """The endpoint refused this request; the gateway's malformed-request error is transport, not refusal."""
    return type(error).__name__ in {"BadRequestError", "ContextWindowExceededError"} and TRANSPORT not in str(error)


def repair_attribute(calls: list, dataset: str, table: str, attribute: str, frame, path: Path) -> int:
    """Ask again the documents whose call failed on transport before TRANSPORT counted as transient."""
    part = json.loads(path.read_text())
    redo = sorted({call["doc"] for call in part["calls"] if TRANSPORT in str(call.get("failed_call", ""))})
    if not redo:
        return 0
    fresh = path.with_suffix(".repair.json")
    if not extract_attribute(calls, dataset, table, attribute, frame.loc[redo], fresh):
        return -1
    redone = json.loads(fresh.read_text())
    index = list(frame.index)
    for doc, value in zip(redo, redone["outputs"]):
        part["outputs"][index.index(doc)] = value
    for call in part["calls"]:
        if call.get("doc") in redo and TRANSPORT in str(call.get("failed_call", "")):
            call["transient"] = call.pop("failed_call")
    for call in redone["calls"]:
        call["repair"] = True
    part["calls"].extend(redone["calls"])
    path.write_text(json.dumps(part, default=str))
    fresh.unlink()
    return len(redo)


def empty_response(content: str = ""):
    from litellm import ModelResponse

    return ModelResponse(
        choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="Player", choices=sorted(FOLDERS))
    parser.add_argument("--model", default="openai/managed-gpt-oss-120b")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--first-repeat", type=int, default=1)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--tables", default=None, help="comma-separated tables to extract; default all")
    parser.add_argument("--reassemble", action="store_true",
                        help="rebuild extracted tables of finished repeats from their saved parts")
    parser.add_argument("--repair", action="store_true",
                        help="ask again documents of finished repeats whose call failed on transport")
    args = parser.parse_args(argv)

    common._env()
    import lotus
    from lotus.models import LM

    calls: list[dict] = []
    lm = LM(args.model, temperature=0.0, max_batch_size=args.workers, max_tokens=args.max_tokens)
    original = lm._process_uncached_messages

    def logged(uncached_data, all_kwargs, show_progress_bar, progress_bar_desc):
        started = time.time()
        responses = original(uncached_data, all_kwargs, show_progress_bar, progress_bar_desc)
        kept = []
        for (messages, _hash), response in zip(uncached_data, responses):
            record = common._record(messages, response, started)
            if not answered(response):
                if isinstance(response, Exception) and not rejected(response):
                    # Timeouts, rate limits, server errors: the document is asked again.
                    record["transient"] = record.pop("error", type(response).__name__)
                    response = empty_response(RETRY)
                else:
                    # The endpoint refused the request or the model returned no content: the
                    # system yields no value for this item, as it would in production.
                    record["failed_call"] = record.pop("error", "no content")
                    response = empty_response()
            calls.append(record)
            kept.append(response)
        return kept

    lm._process_uncached_messages = logged
    lotus.settings.configure(lm=lm)
    lotus.settings.enable_cache = False

    used = attributes_used(args.dataset)
    setting = f"w{args.workers}"
    for repeat in range(args.first_repeat, args.first_repeat + args.repeats):
        folder = OUT / args.dataset / common.model_tag(args.model) / setting / f"repeat-{repeat}"
        extract_dir = folder / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        for table, columns in used.items():
            if args.tables and table not in args.tables.split(","):
                continue
            started = time.time()
            elapsed = 0.0
            truth, frame = documents(args.dataset, table)
            parts = [extract_dir / "parts" / table / f"{column}.json" for column in columns]
            if (extract_dir / f"{table}.csv").exists():
                if not (args.reassemble or args.repair):
                    continue
                if not all(part.exists() for part in parts):
                    drop_float_suffix(extract_dir / f"{table}.csv", truth)
                    continue
                if args.repair:
                    calls.clear()
                    redone = [repair_attribute(calls, args.dataset, table, column, frame, part)
                              for column, part in zip(columns, parts)]
                    print(f"repair {table} repeat-{repeat} documents={sum(max(n, 0) for n in redone)} "
                          f"ok={-1 not in redone}", flush=True)
                    if -1 in redone or not any(redone):
                        continue
                    elapsed = json.loads((extract_dir / f"{table}.metrics.json").read_text()).get("seconds", 0.0)
                else:
                    assemble(args.dataset, table, columns, truth, frame, extract_dir)
                    continue
            complete = True
            for column, part in zip(columns, parts):
                if part.exists():
                    continue
                calls.clear()
                done = extract_attribute(calls, args.dataset, table, column, frame, part)
                complete &= done
                print(f"{table}.{column} repeat-{repeat} ok={done} calls={len(calls)}", flush=True)
            if not complete:
                continue
            log = assemble(args.dataset, table, columns, truth, frame, extract_dir)
            with (extract_dir / f"{table}.calls.jsonl").open("w", encoding="utf-8") as handle:
                for call in log:
                    handle.write(json.dumps(call, default=str) + "\n")
            (extract_dir / f"{table}.metrics.json").write_text(json.dumps({
                "benchmark": "uda-bench", "dataset": args.dataset, "table": table, "attributes": columns,
                "model": args.model, "workers": args.workers, "repeat": repeat,
                "calls": len(log), "documents": len(frame.index),
                "transient_retries": sum("transient" in call for call in log),
                "failed_calls": sum("failed_call" in call for call in log),
                "seconds": round(elapsed + time.time() - started, 3),
                "prompt_tokens": sum(call.get("prompt_tokens") or 0 for call in log),
            }, indent=1))
            print(f"finish {args.dataset} {table} repeat-{repeat} calls={len(log)} "
                  f"seconds={time.time() - started:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
