#!/usr/bin/env python3
"""Selective voting replayed on the ten recorded SemBench executions of gpt-oss-120B.

The executions are split into three disjoint groups of three. Within a group,
every item takes the output of the group's first execution, except the share q
of items whose first call produced the longest reasoning trace, which take the
majority (median for scores) of all three. The cost is 1 + 2q calls per item.
A random selection of the same share is the baseline. Reports the pairwise
item disagreement between the combined executions, relative to one execution
and to a vote of three on every item. Writes
experiments/processed/sembench_repeats/selective_vote.json. No model is called.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_sembench_repeats as base  # noqa: E402
from remedies_sembench_repeats import combine  # noqa: E402

MODEL = "managed-gpt-oss-120b"
SETTING = "w20"
SHARES = (0.1, 0.2, 0.3)
GROUP = 3
RUNS = 9
OUT = base.PROCESSED / "selective_vote.json"


def disagreement(combined: list[dict], keys: list[str]) -> float:
    return statistics.fmean(base.pair_disagreement([run[key] for run in combined]) for key in keys)


def main() -> None:
    rng = random.Random(base.SEED)
    cells = []
    for scenario, queries in base.KINDS.items():
        for query, (kind, plan) in queries.items():
            if plan.startswith("pairs"):
                continue
            folder = base.RAW / scenario / MODEL / SETTING
            dirs = sorted(folder.glob("repeat-*"), key=lambda path: int(path.name.split("-")[1]))
            dirs = [path for path in dirs if (path / f"Q{query}.calls.jsonl").exists()
                    and not json.loads((path / f"Q{query}.metrics.json").read_text()).get("call_errors")][:RUNS]
            if len(dirs) < RUNS:
                continue
            runs = [base.read_calls(path / f"Q{query}.calls.jsonl", kind) for path in dirs]
            values = [{key: value for key, value, _record in calls} for calls in runs]
            length = [{key: record.get("reasoning_chars") or 0 for key, _value, record in calls} for calls in runs]
            keys = [key for key, _value, _record in runs[0] if all(key in run for run in values)]
            groups = [range(start, start + GROUP) for start in range(0, RUNS, GROUP)]

            def combined_with(selected_per_group: list[set]) -> list[dict]:
                result = []
                for members, selected in zip(groups, selected_per_group):
                    first = members[0]
                    result.append({key: combine(kind, [values[r][key] for r in members]) if key in selected
                                   else values[first][key] for key in keys})
                return result

            single = disagreement(combined_with([set()] * len(groups)), keys)
            full = disagreement(combined_with([set(keys)] * len(groups)), keys)
            entry = {"scenario": scenario, "query": query, "items": len(keys), "single": single, "vote3": full}
            for share in SHARES:
                count = round(share * len(keys))
                longest = [set(sorted(keys, key=lambda key: -length[members[0]][key])[:count]) for members in groups]
                randomly = [set(rng.sample(keys, count)) for _members in groups]
                entry[f"longest_{share}"] = disagreement(combined_with(longest), keys)
                entry[f"random_{share}"] = disagreement(combined_with(randomly), keys)
            cells.append(entry)

    names = [name for name in cells[0] if name not in ("scenario", "query", "items")]
    total = {name: sum(cell[name] * cell["items"] for cell in cells) / sum(cell["items"] for cell in cells)
             for name in names}
    removed = {name: 1 - value / total["single"] for name, value in total.items()}
    facts = {"cells": cells, "pooled_disagreement": total, "share_removed": removed,
             "share_of_full_vote": {name: value / removed["vote3"] for name, value in removed.items()}}
    OUT.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    for name in total:
        print(f"{name:12s} disagreement={100 * total[name]:.2f}% removed={100 * removed[name]:.0f}% "
              f"of-full={100 * facts['share_of_full_vote'][name]:.0f}%")


if __name__ == "__main__":
    main()
