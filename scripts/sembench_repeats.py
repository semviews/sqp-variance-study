#!/usr/bin/env python3
"""Repeat SemBench queries through SemBench's own LOTUS runner.

The runner class, prompts, and query code are SemBench's. This script
subclasses the runner to point LOTUS at the local LiteLLM proxy, to write
results outside the SemBench tree, and to log every model call. Each
(model, setting, repeat, query) cell is written once and skipped on rerun.

Run with SemBench/.venvs/lotus/bin/python.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEMBENCH = ROOT / "SemBench"
SRC = SEMBENCH / "src"
OUT = ROOT / "experiments" / "raw" / "sembench-repeats"
SCALE = {"movie": 2000, "animals": 200, "medical": 11112, "ecomm": 500, "mmqa": 200, "cars": 9836}


ENDPOINTS = {"local": "LOCAL_LITELLM", "enterprise": "ENTERPRISE_LITELLM"}


def _env(endpoint: str = "local") -> None:
    values = {}
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    prefix = ENDPOINTS[endpoint]
    os.environ["OPENAI_API_BASE"] = values[f"{prefix}_ENDPOINT"]
    os.environ["OPENAI_API_KEY"] = values[f"{prefix}_API_KEY"]


def _runner_class(scenario: str):
    sys.path.insert(0, str(SRC))
    path = SRC / "scenario" / scenario / "runner" / "lotus_runner" / "lotus_runner.py"
    spec = importlib.util.spec_from_file_location(f"sembench_{scenario}_lotus_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LotusRunner


def _message_key(messages) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _text_of(messages) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content[:300]
            return json.dumps(content, default=str)[:300]
    return ""


def data_fingerprint(scenario: str, data_dir: Path | None = None) -> dict[str, str]:
    """SHA-256 prefix of every input CSV, so a cell can prove which inputs it read."""
    folder = data_dir or SEMBENCH / "files" / scenario / "data"
    return {
        str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        for path in sorted(folder.rglob("*.csv"))
    }


def _disable_scenario_setup() -> None:
    """Scenario setup regenerates input CSVs; concurrent runs then race on them."""
    from runner import generic_runner

    original = generic_runner.GenericRunner.get_scenario_handler

    def no_setup(use_case, scale_factor):
        handler = original(use_case, scale_factor)
        if handler is not None:
            handler.setup_scenario = lambda *args, **kwargs: None
        return handler

    generic_runner.GenericRunner.get_scenario_handler = staticmethod(no_setup)


def build(scenario: str, model: str, workers: int, reasoning: str | None, shuffle_seed: int | None, logprobs: bool):
    import lotus
    from lotus.models import LM

    base = _runner_class(scenario)
    _disable_scenario_setup()

    class Runner(base):
        calls: list[dict] = []

        def _configure_lm(self):
            extra = {}
            if reasoning:
                extra["reasoning_effort"] = reasoning
            lm = LM(model, temperature=0.0, max_batch_size=workers, max_tokens=self.max_tokens, **extra)
            original = lm._process_uncached_messages
            runner = self

            def logged(uncached_data, all_kwargs, show_progress_bar, progress_bar_desc):
                started = time.time()
                if logprobs and not all_kwargs.get("logprobs"):
                    all_kwargs = {**all_kwargs, "logprobs": True, "top_logprobs": 5}
                responses = list(original(uncached_data, all_kwargs, show_progress_bar, progress_bar_desc))
                retries = [0] * len(responses)
                for round_ in range(RATE_LIMIT_ROUNDS):
                    pending = [i for i, response in enumerate(responses) if _rate_limited(response)]
                    if not pending:
                        break
                    time.sleep(RATE_LIMIT_WAIT * (round_ + 1))
                    again = original([uncached_data[i] for i in pending], all_kwargs, False, progress_bar_desc)
                    for i, response in zip(pending, again):
                        responses[i] = response
                        retries[i] += 1
                for (messages, _hash), response, retried in zip(uncached_data, responses, retries):
                    entry = _record(messages, response, started)
                    if retried:
                        entry["rate_limit_retries"] = retried
                    runner.calls.append(entry)
                return responses

            lm._process_uncached_messages = logged
            return lm

        def load_data(self, *args, **kwargs):
            frame = super().load_data(*args, **kwargs)
            if shuffle_seed is not None and hasattr(frame, "sample"):
                frame = frame.sample(frac=1.0, random_state=shuffle_seed)
            return frame

    lotus.settings.enable_cache = False
    return Runner


# A request the gateway refused for rate limiting got no model output, so sending
# it again is not a second sample of the model.
RATE_LIMIT_ROUNDS, RATE_LIMIT_WAIT = 5, 30


def _rate_limited(response) -> bool:
    return isinstance(response, Exception) and (
        "RateLimit" in type(response).__name__ or "Rate limit exceeded" in str(response))


def _record(messages, response, started: float) -> dict:
    entry = {"key": _message_key(messages), "text": _text_of(messages), "t": round(started, 3)}
    if isinstance(response, Exception):
        entry["error"] = f"{type(response).__name__}: {response}"[:500]
        return entry
    try:
        choice = response.choices[0]
        message = choice.message
        entry["output"] = message.content
        reasoning = getattr(message, "reasoning_content", None)
        entry["reasoning_chars"] = len(reasoning) if reasoning else 0
        if reasoning:
            entry["reasoning_text"] = reasoning
        entry["finish_reason"] = choice.finish_reason
        usage = getattr(response, "usage", None)
        if usage is not None:
            entry["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
            entry["completion_tokens"] = getattr(usage, "completion_tokens", None)
        entry["fingerprint"] = getattr(response, "system_fingerprint", None)
        cost = (getattr(response, "_hidden_params", None) or {}).get("response_cost")
        if cost is not None:
            entry["cost"] = cost
        logprobs = getattr(choice, "logprobs", None)
        content = getattr(logprobs, "content", None) if logprobs is not None else None
        if content:
            entry["logprobs"] = [
                {
                    "token": item.token,
                    "logprob": item.logprob,
                    "top": [[alt.token, alt.logprob] for alt in (item.top_logprobs or [])],
                }
                for item in content[:4]
            ]
    except Exception as exc:
        entry["error"] = f"parse {type(exc).__name__}: {exc}"[:500]
    return entry


def model_tag(model: str) -> str:
    return model.split("/", 1)[-1].replace("/", "-")


def setting_tag(workers: int, reasoning: str | None, shuffle: bool, logprobs: bool) -> str:
    parts = [f"w{workers}"]
    if reasoning:
        parts.append(f"r-{reasoning}")
    if shuffle:
        parts.append("shuffle")
    if logprobs:
        parts.append("lp")
    return "_".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="movie")
    parser.add_argument("--queries", type=int, nargs="+", required=True)
    parser.add_argument("--model", default="openai/managed-gpt-oss-120b")
    parser.add_argument("--endpoint", default="local", choices=sorted(ENDPOINTS), help="LiteLLM proxy whose .env variables to use")
    parser.add_argument("--max-cost", type=float, default=None, help="stop before the next cell once logged spend (USD) on this endpoint reaches this")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--first-repeat", type=int, default=1)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--reasoning", default=None)
    parser.add_argument("--shuffle", action="store_true", help="shuffle input rows, seed = repeat")
    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--policy", default="approximate", choices=("approximate", "exact"))
    parser.add_argument("--scale-factor", type=int, default=None)
    parser.add_argument("--tag", default=None, help="suffix that keeps a probe apart from the main matrix")
    parser.add_argument("--data-dir", default=None, help="input folder replacing SemBench's, relative to the repository")
    args = parser.parse_args(argv)
    args.data_dir = (ROOT / args.data_dir).resolve() if args.data_dir else None

    _env(args.endpoint)
    scale = args.scale_factor or SCALE[args.scenario]
    setting = setting_tag(args.workers, args.reasoning, args.shuffle, args.logprobs)
    if args.policy != "approximate":
        setting += f"_{args.policy}"
    if args.tag:
        setting += f"_{args.tag}"
    base = OUT / args.scenario / model_tag(args.model) / setting
    rejected = {query for query in args.queries if policy_rejected(base, query)}
    todo = [
        (repeat, query)
        for repeat in range(args.first_repeat, args.first_repeat + args.repeats)
        for query in args.queries
        if not (base / f"repeat-{repeat}" / f"Q{query}.metrics.json").exists()
        and not exhausted(base / f"repeat-{repeat}" / f"Q{query}.failures.jsonl")
        and query not in rejected
    ]
    if rejected:
        print(f"skip Q{sorted(rejected)}: a content policy rejected every attempt of one execution", flush=True)
    print(f"{args.scenario} {args.model} {setting}: {len(todo)} cells to run", flush=True)
    if not todo:
        return 0

    runner = None
    current_shuffle = object()
    for repeat, query in todo:
        if args.max_cost is not None:
            spent = logged_spend(args.endpoint)
            if spent >= args.max_cost:
                print(f"stop: logged spend ${spent:.2f} on {args.endpoint} reached --max-cost ${args.max_cost:.2f}", flush=True)
                return 2
        shuffle_seed = repeat if args.shuffle else None
        if runner is None or shuffle_seed != current_shuffle:
            Runner = build(args.scenario, args.model, args.workers, args.reasoning, shuffle_seed, args.logprobs)
            runner = Runner(args.scenario, scale, model_name=args.model, concurrent_llm_worker=args.workers, skip_setup=True)
            runner.policy = args.policy
            if args.data_dir:
                runner.data_path = args.data_dir
            current_shuffle = shuffle_seed
        destination = base / f"repeat-{repeat}"
        destination.mkdir(parents=True, exist_ok=True)
        if query in rejected:
            continue
        failures = destination / f"Q{query}.failures.jsonl"
        earlier = len(failures.read_text().splitlines()) if failures.exists() else 0
        for attempt in range(earlier + 1, MAX_ATTEMPTS + 1):
            ok = run_cell(runner, args, setting, scale, shuffle_seed, repeat, query, destination, attempt)
            if ok:
                break
            time.sleep(10 * attempt)
        if policy_rejected(base, query):
            rejected.add(query)
            print(f"skip Q{query} from now on: a content policy rejected every attempt", flush=True)
    return 0


MAX_ATTEMPTS = 3
POLICY_REJECTION = "ContentPolicyViolation"


def exhausted(failures: Path) -> bool:
    """A cell that failed MAX_ATTEMPTS times stays a recorded failure."""
    return failures.exists() and len(failures.read_text().splitlines()) >= MAX_ATTEMPTS


def policy_rejected(base: Path, query: int) -> bool:
    """Every attempt of some execution of the query was rejected by the provider's content policy."""
    for failures in base.glob(f"repeat-*/Q{query}.failures.jsonl"):
        records = [json.loads(line) for line in failures.read_text().splitlines()]
        if len(records) >= MAX_ATTEMPTS and all(POLICY_REJECTION in json.dumps(r) for r in records):
            return True
    return False


def logged_spend(endpoint: str, root: Path = OUT.parent) -> float:
    """USD logged by every call in cells (kept or failed) that ran on this endpoint."""
    total = 0.0
    for metrics in root.rglob("Q*.metrics.json"):
        record = json.loads(metrics.read_text())
        if record.get("endpoint") != endpoint:
            continue
        calls = metrics.with_name(metrics.name.replace(".metrics.json", ".calls.jsonl"))
        if calls.exists():
            for line in calls.read_text().splitlines():
                total += json.loads(line).get("cost") or 0.0
    for failures in root.rglob("Q*.failures.jsonl"):
        for line in failures.read_text().splitlines():
            record = json.loads(line)
            if record.get("endpoint") == endpoint:
                total += record.get("spend") or 0.0
    return total


def run_cell(runner, args, setting, scale, shuffle_seed, repeat, query, destination, attempt) -> bool:
    """One execution. A cell with any failed model call is a failure, not an answer."""
    runner.calls.clear()
    print(f"start Q{query} repeat-{repeat} attempt {attempt}", flush=True)
    try:
        metric = runner.execute_query(query)
    except Exception as exc:
        metric = type("Failed", (), {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                     "execution_time": None, "results": None})()
    record = {
        "scenario": args.scenario,
        "model": args.model,
        "setting": setting,
        "policy": args.policy,
        "repeat": repeat,
        "query_id": query,
        "status": metric.status,
        "error": getattr(metric, "error", None),
        "execution_time": metric.execution_time,
        "token_usage": getattr(metric, "token_usage", None),
        "model_calls": len(runner.calls),
        "call_errors": sum("error" in call for call in runner.calls),
        "scale_factor": scale,
        "temperature": 0.0,
        "endpoint": args.endpoint,
        "workers": args.workers,
        "reasoning": args.reasoning,
        "shuffle_seed": shuffle_seed,
        "attempt": attempt,
        "data_dir": str(args.data_dir.relative_to(ROOT)) if args.data_dir else None,
        "data": data_fingerprint(args.scenario, args.data_dir),
    }
    ok = metric.status == "success" and metric.results is not None and record["call_errors"] == 0
    if ok:
        with (destination / f"Q{query}.calls.jsonl").open("w", encoding="utf-8") as handle:
            for call in runner.calls:
                handle.write(json.dumps(call, default=str) + "\n")
        metric.results.to_csv(destination / f"Q{query}.csv", index=False)
        (destination / f"Q{query}.metrics.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    else:
        record["call_error_samples"] = sorted({call["error"] for call in runner.calls if "error" in call})[:3]
        record["spend"] = sum(call.get("cost") or 0.0 for call in runner.calls)
        with (destination / f"Q{query}.failures.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    print(
        f"finish Q{query} repeat-{repeat} status={metric.status} ok={ok} calls={len(runner.calls)} "
        f"errors={record['call_errors']} seconds={metric.execution_time}",
        flush=True,
    )
    return ok


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
