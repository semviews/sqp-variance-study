"""Classify why LRO-Bench executions fail, from the recorded replies.

For every failed execution, each operator reply is put into one class:

- envelope: JSON wrapped in a tool call ({"tool": ..., "args": ...});
- leading text: not JSON as returned, but a JSON value follows some text;
- valid JSON: parses as returned, so the failure is in its content or schema;
- no JSON: no JSON value anywhere in the reply.

An execution takes the first class in that order that any of its replies
has. A lenient parser that strips envelopes and leading text could rescue
at most the executions in the first two classes.

It also checks the impute judge: pairs of executions of a
query that return identical answers should receive identical scores. No
model is called.

Writes experiments/processed/lrobench_failures.json with a "macros" map.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

from analyze_lrobench_repeats import answer_of, quality_of

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "experiments" / "raw" / "lrobench"
OUT = ROOT / "experiments" / "processed" / "lrobench_failures.json"
ORDER = ["envelope", "leading text", "valid JSON", "no JSON"]
MACRO = {"managed-gpt-oss-120b": "Gptoss", "managed-llama-3-3-70b-instruct": "LlamaW",
         "vllm-llama-3-3-70b-instruct": "LlamaR", "vllm-granite-3-3-8b-instruct": "Granite"}
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parses(text: str) -> object | None:
    try:
        return json.loads(FENCE.sub("", text.strip()))
    except (json.JSONDecodeError, ValueError):
        return None


def embedded(text: str) -> bool:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            decoder.raw_decode(text[match.start():])
            return True
        except json.JSONDecodeError:
            continue
    return False


def classify(reply: str | None) -> str:
    if not reply:
        return "no JSON"
    value = parses(reply)
    if isinstance(value, dict) and "tool" in value and ("args" in value or "arguments" in value):
        return "envelope"
    if value is not None:
        return "valid JSON"
    if '"tool"' in reply and embedded(reply):
        return "envelope"
    return "leading text" if embedded(reply) else "no JSON"


def judge_check() -> tuple[int, int]:
    groups: dict[tuple, list] = {}
    for path in RAW.glob("*/impute_*/repeat-*/*.json"):
        if path.name.endswith(".failed.json"):
            continue
        record = json.loads(path.read_text())
        if record.get("error") or quality_of(record) is None:
            continue
        groups.setdefault((path.parts[-4], path.parts[-3], path.name), []).append((answer_of(record), quality_of(record)))
    same = differ = 0
    for runs in groups.values():
        for (answer_a, score_a), (answer_b, score_b) in combinations(runs, 2):
            if answer_a == answer_b:
                same += 1
                differ += abs(score_a - score_b) > 1e-9
    return same, differ


def main() -> None:
    result, macros = {}, {}
    for model_dir in sorted(RAW.iterdir()):
        counts = Counter()
        for failed in model_dir.glob("*/repeat-*/*.failed.json"):
            calls = failed.with_name(failed.name.replace(".failed.json", ".calls.jsonl"))
            replies = [json.loads(line).get("output") for line in calls.read_text().splitlines()] if calls.exists() else []
            classes = {classify(reply) for reply in replies} or {"no JSON"}
            counts[next(name for name in ORDER if name in classes)] += 1
        total = sum(counts.values())
        result[model_dir.name] = {"failed_executions": total, **{name: counts[name] for name in ORDER}}
        if model_dir.name in MACRO and total:
            suffix = MACRO[model_dir.name]
            rescuable = counts["envelope"] + counts["leading text"]
            macros[f"lroFailedExec{suffix}"] = total
            macros[f"lroFailEnvelopePct{suffix}"] = f"{100 * counts['envelope'] / total:.0f}\\%"
            macros[f"lroFailTextPct{suffix}"] = f"{100 * counts['leading text'] / total:.0f}\\%"
            macros[f"lroFailJsonPct{suffix}"] = f"{100 * counts['valid JSON'] / total:.0f}\\%"
            macros[f"lroFailNoJsonPct{suffix}"] = f"{100 * counts['no JSON'] / total:.0f}\\%"
            macros[f"lroFailRescuablePct{suffix}"] = f"{100 * rescuable / total:.0f}\\%"
    same, differ = judge_check()
    macros["lroJudgeSamePairs"] = f"{same:,}".replace(",", "{,}")
    macros["lroJudgeDifferPairs"] = differ
    OUT.write_text(json.dumps({"failures": result, "judge": {"identical_answer_pairs": same, "different_scores": differ},
                               "macros": macros}, indent=1) + "\n")
    for model, values in result.items():
        print(model, values)


if __name__ == "__main__":
    main()
