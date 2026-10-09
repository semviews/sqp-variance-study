#!/usr/bin/env python3
"""Temperature control: replay exact LOTUS requests one at a time.

The requests are rebuilt with LOTUS's own sem_filter formatter on SemBench's
Movie data, and kept only if their hash equals a key recorded in the main
campaign's call log, so they are the requests the benchmark sent. Items are
fixed in a manifest (half that flipped in the main campaign, half that did
not; seed 11). Each item is then sent `--repeats` times at concurrency 1 in
one of three settings:

- `t0`: temperature 0, the benchmark's setting;
- `t0-seed`: temperature 0 with `seed` = 11;
- `t1`: temperature 1, a positive control for sampling.

Every response is logged with its full reasoning text, response id,
fingerprint, and provider headers. Resumable. Run with
SemBench/.venvs/lotus/bin/python.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sembench_repeats import OUT as SEMBENCH_OUT, SEMBENCH, _env, _message_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "experiments" / "raw" / "controls" / "temperature"
INSTRUCTIONS = {
    1: ('Determine if the following movie review is clearly positive. Review: "{reviewText}".', None),
    3: ("Determine if the following review is clearly positive. Review: {reviewText}", "taken_3"),
}
SETTINGS = {
    "t0": {"temperature": 0.0},
    "t0-seed": {"temperature": 0.0, "seed": 11},
    "t1": {"temperature": 1.0},
}


class _Captured(Exception):
    pass


def capture_requests(model: str, query: int) -> tuple[list[list[dict]], dict]:
    """The messages and keyword arguments LOTUS sends for one Movie filter query."""
    import pandas as pd
    import lotus
    from lotus.models import LM

    captured: dict = {}
    lm = LM(model, temperature=0.0, max_batch_size=20, max_tokens=8192)

    def capture(uncached_data, all_kwargs, show_progress_bar, progress_bar_desc):
        captured["messages"] = [messages for messages, _hash in uncached_data]
        captured["kwargs"] = dict(all_kwargs)
        raise _Captured

    lm._process_uncached_messages = capture
    lotus.settings.configure(lm=lm, enable_cache=False)
    reviews = pd.read_csv(SEMBENCH / "files" / "movie" / "data" / "sf_2000" / "Reviews.csv")
    instruction, movie = INSTRUCTIONS[query]
    if movie:
        reviews = reviews[reviews["id"] == movie]
    try:
        reviews.sem_filter(instruction)
    except _Captured:
        pass
    return captured["messages"], captured["kwargs"]


def recorded_outputs(model: str, query: int) -> dict[str, list[str]]:
    base = SEMBENCH_OUT / "movie" / model.split("/", 1)[-1] / "w20"
    outputs: dict[str, list[str]] = defaultdict(list)
    for calls in sorted(base.glob(f"repeat-*/Q{query}.calls.jsonl")):
        for line in calls.read_text().splitlines():
            call = json.loads(line)
            outputs[call["key"]].append(_label(call.get("output")))
    return outputs


def _label(output: str | None) -> str:
    text = (output or "").split("Answer:")[-1].strip().lower()
    return "true" if text.startswith("true") else "false" if text.startswith("false") else "other"


def manifest(model: str, query: int, items: int) -> dict:
    path = DEST / f"manifest-Q{query}.json"
    if path.exists():
        return json.loads(path.read_text())
    messages, kwargs = capture_requests(model, query)
    by_key = {_message_key(m): m for m in messages}
    outputs = recorded_outputs(model, query)
    matched = sorted(set(by_key) & set(outputs))
    flipped = [k for k in matched if len(set(outputs[k])) > 1]
    stable = [k for k in matched if len(set(outputs[k])) == 1]
    rng = random.Random(11)
    take_flipped = sorted(rng.sample(flipped, min(len(flipped), items // 2)))
    take_stable = sorted(rng.sample(stable, items - len(take_flipped)))
    record = {
        "model": model,
        "query": query,
        "seed": 11,
        "requests_built": len(messages),
        "requests_matching_recorded_keys": len(matched),
        "recorded_executions": max(len(v) for v in outputs.values()),
        "flipped_available": len(flipped),
        "kwargs": {k: v for k, v in kwargs.items()},
        "items": [
            {"key": k, "flipped_in_campaign": k in take_flipped, "campaign_outputs": outputs[k], "messages": by_key[k]}
            for k in take_flipped + take_stable
        ],
    }
    DEST.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")
    return record


def call(litellm, model: str, messages: list[dict], kwargs: dict, setting: str) -> dict:
    started = time.time()
    request = {k: v for k, v in kwargs.items() if k not in ("temperature", "seed")}
    request.update(SETTINGS[setting])
    try:
        response = litellm.completion(model=model, messages=messages, drop_params=True, timeout=120, num_retries=0, **request)
    except Exception as exc:
        return {"t": round(started, 3), "error": f"{type(exc).__name__}: {exc}"[:500]}
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    hidden = getattr(response, "_hidden_params", None) or {}
    headers = hidden.get("additional_headers") or {}
    reasoning = getattr(choice.message, "reasoning_content", None)
    return {
        "t": round(started, 3),
        "seconds": round(time.time() - started, 2),
        "id": getattr(response, "id", None),
        "model_returned": getattr(response, "model", None),
        "output": choice.message.content,
        "reasoning_text": reasoning,
        "finish_reason": choice.finish_reason,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "fingerprint": getattr(response, "system_fingerprint", None),
        "provider_headers": {k: v for k, v in headers.items() if k.startswith("llm_provider-")},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/managed-gpt-oss-120b")
    parser.add_argument("--query", type=int, default=3)
    parser.add_argument("--items", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--setting", choices=sorted(SETTINGS), required=True)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args(argv)
    _env("local")
    record = manifest(args.model, args.query, args.items)
    print(
        f"manifest: {len(record['items'])} items, {record['requests_matching_recorded_keys']} of "
        f"{record['requests_built']} rebuilt requests match recorded keys",
        flush=True,
    )
    if args.manifest_only:
        return 0
    import litellm

    out = DEST / f"Q{args.query}-{args.setting}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            row = json.loads(line)
            if "error" not in row:
                done.add((row["repeat"], row["key"]))
    with out.open("a", encoding="utf-8") as handle:
        for repeat in range(1, args.repeats + 1):
            for item in record["items"]:
                if (repeat, item["key"]) in done:
                    continue
                for attempt in range(1, 4):
                    row = {"setting": args.setting, "repeat": repeat, "key": item["key"], "attempt": attempt}
                    row.update(call(litellm, args.model, item["messages"], record["kwargs"], args.setting))
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    if "error" not in row:
                        break
                    time.sleep(10 * attempt)
            print(f"{args.setting} repeat {repeat} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
