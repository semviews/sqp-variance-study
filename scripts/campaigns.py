#!/usr/bin/env python3
"""Campaign windows, call totals, and cost.

Reads every per-call log of the three suites and the controls and writes
experiments/processed/campaigns.json:

- per suite, deployment, and campaign: first and last call (UTC date),
  executions, calls, prompt and completion tokens, and logged cost;
- study totals (calls, tokens, enterprise spend);
- for every query that the main matrix and a later campaign both ran on
  the same deployment and setting family, the pairwise disagreement in each
  and an item-bootstrap interval of the difference (10,000 resamples,
  seed 11).

Campaigns: SemBench `main` (w20 or w20_lp, the first executions),
`extension` (later executions of the same cells), `joins`, `probes`,
`commercial`; LRO-Bench and UDA-Bench `main`; `controls` for the
temperature control. No model is called.
"""

from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_sembench_repeats import KINDS, parse, seed_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "experiments" / "raw"
OUT = ROOT / "experiments" / "processed" / "campaigns.json"
SEED = 11
RESAMPLES = 10_000
COMMERCIAL = ("Azure-", "aws-")
# Executions in the main matrix before later campaigns extended it.
MAIN_EXECUTIONS = {"managed-gpt-oss-120b": 10}
MAIN_DEFAULT = 5


def day(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")


def sembench_campaign(model: str, setting: str, repeat: int) -> str:
    if model.startswith(COMMERCIAL):
        return "commercial"
    if setting.endswith("_probe"):
        return "probes"
    if "_exact_join" in setting:
        return "joins"
    if setting in ("w20", "w20_lp"):
        return "main" if repeat <= MAIN_EXECUTIONS.get(model, MAIN_DEFAULT) else "extension"
    return "other"


def calls_of(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def kept(path: Path) -> bool:
    """A SemBench cell counts only if its metrics file exists (failed cells keep no calls file)."""
    return path.with_name(path.name.replace(".calls.jsonl", ".metrics.json")).exists()


def main() -> None:
    groups: dict[tuple, dict] = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                                       "cost": 0.0, "t_min": None, "t_max": None,
                                                       "executions": set()})

    def add(key: tuple, execution: tuple, rows: list[dict]) -> None:
        group = groups[key]
        group["executions"].add(execution)
        for row in rows:
            group["calls"] += 1
            group["prompt_tokens"] += row.get("prompt_tokens") or 0
            group["completion_tokens"] += row.get("completion_tokens") or 0
            group["cost"] += row.get("cost") or 0.0
            t = row.get("t")
            if t:
                group["t_min"] = t if group["t_min"] is None else min(group["t_min"], t)
                group["t_max"] = t if group["t_max"] is None else max(group["t_max"], t)

    per_query_items: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for path in sorted((RAW / "sembench-repeats").glob("*/*/*/repeat-*/Q*.calls.jsonl")):
        if not kept(path):
            continue
        repeat = int(path.parent.name.split("-")[1])
        setting, model, scenario = path.parent.parent.name, path.parent.parent.parent.name, path.parents[3].name
        campaign = sembench_campaign(model, setting, repeat)
        rows = calls_of(path)
        add(("SemBench", model, campaign), (scenario, setting, repeat), rows)
        query = int(re.match(r"Q(\d+)", path.name).group(1))
        if campaign in ("main", "extension", "probes") and query in KINDS.get(scenario, {}):
            kind = KINDS[scenario][query][0]
            seen: Counter = Counter()
            outputs = []
            for row in rows:
                seen[row["key"]] += 1
                outputs.append((f"{row['key']}#{seen[row['key']]}", str(parse(kind, row.get("output")))))
            per_query_items[(scenario, model, setting, query, campaign)][repeat] = outputs

    for path in sorted((RAW / "lrobench").glob("*/*/repeat-*/*.calls.jsonl")):
        model = path.parents[2].name
        add(("LRO-Bench", model, "main"), (path.parents[1].name, path.parent.name), calls_of(path))
    for path in sorted((RAW / "uda-repeats").glob("*/*/*/repeat-*/extract/*.calls.jsonl")):
        model = path.parents[3].name
        add(("UDA-Bench", model, "main"), (path.parents[4].name, path.parents[1].name), calls_of(path))
    for path in sorted((RAW / "controls" / "temperature").glob("Q*-*.jsonl")):
        rows = [row for row in calls_of(path) if "error" not in row]
        add(("Controls", "managed-gpt-oss-120b", "temperature"), (path.stem,), rows)

    campaigns = []
    for (suite, model, campaign), group in sorted(groups.items()):
        campaigns.append({
            "suite": suite, "model": model, "campaign": campaign,
            "first": day(group["t_min"]) if group["t_min"] else None,
            "last": day(group["t_max"]) if group["t_max"] else None,
            "executions": len(group["executions"]),
            "calls": group["calls"],
            "prompt_tokens": group["prompt_tokens"],
            "completion_tokens": group["completion_tokens"],
            "cost": round(group["cost"], 4),
        })

    totals = {
        "calls": sum(row["calls"] for row in campaigns),
        "prompt_tokens": sum(row["prompt_tokens"] for row in campaigns),
        "completion_tokens": sum(row["completion_tokens"] for row in campaigns),
        "enterprise_cost": round(sum(row["cost"] for row in campaigns if row["campaign"] == "commercial"), 2),
        "first": min(row["first"] for row in campaigns if row["first"]),
        "last": max(row["last"] for row in campaigns if row["last"]),
        "by_suite": {suite: sum(row["calls"] for row in campaigns if row["suite"] == suite)
                     for suite in sorted({row["suite"] for row in campaigns})},
    }

    comparisons = compare_campaigns(per_query_items)
    OUT.write_text(json.dumps({"campaigns": campaigns, "totals": totals, "comparisons": comparisons}, indent=1) + "\n")
    OUT.with_name("campaigns_macros.json").write_text(json.dumps({"macros": macros(campaigns, totals, comparisons)}, indent=1) + "\n")
    for row in campaigns:
        print(f"{row['suite']:9s} {row['model'][:34]:34s} {row['campaign']:10s} {row['first']}..{row['last']} "
              f"exec={row['executions']:4d} calls={row['calls']:7d} cost={row['cost']}")
    print(json.dumps(totals))
    for row in comparisons:
        print(row)


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def macros(campaigns: list[dict], totals: dict, comparisons: list[dict]) -> dict:
    result = {
        "studyCalls": thousands(totals["calls"]),
        "studyCallsMillions": f"{totals['calls'] / 1e6:.1f}",
        "studyTokensMillions": f"{(totals['prompt_tokens'] + totals['completion_tokens']) / 1e6:.0f}",
        "studyCost": f"{totals['enterprise_cost']:.2f}",
        "studyFirst": totals["first"],
        "studyLast": totals["last"],
    }
    artifact = ROOT / "versions"
    for name, file in (("versionSemBench", "sembench_commit.txt"), ("versionLroBench", "lrobench_commit.txt"),
                       ("versionUdaBench", "udabench_commit.txt")):
        if (artifact / file).exists():
            result[name] = (artifact / file).read_text().strip()[:7]
    result["versionLotus"] = "1.1.3"
    for suite, calls in totals["by_suite"].items():
        result["calls" + suite.replace("-", "").replace(" ", "")] = thousands(calls)
    windows = {}
    for row in campaigns:
        windows.setdefault((row["suite"], row["campaign"]), []).append(row)
    names = {("SemBench", "main"): "SemMain", ("SemBench", "joins"): "SemJoins", ("SemBench", "probes"): "SemProbes",
             ("SemBench", "extension"): "SemExtension", ("SemBench", "commercial"): "SemCommercial",
             ("LRO-Bench", "main"): "Lro", ("UDA-Bench", "main"): "Uda", ("Controls", "temperature"): "Control"}
    for key, rows in windows.items():
        if key in names:
            first = min(row["first"] for row in rows if row["first"])
            last = max(row["last"] for row in rows if row["last"])
            result[f"window{names[key]}"] = first if first == last else f"{first} to {last}"
    probe = [row for row in comparisons if row["model"] == "managed-gpt-oss-120b" and row["query"] == 3
             and row["setting"] == "w20_r-medium_probe"]
    if probe:
        row = probe[0]
        result["probeRefMainDisagree"] = f"{100 * row['disagreement_main']:.2f}\\%"
        result["probeRefProbeDisagree"] = f"{100 * row['disagreement_other']:.2f}\\%"
        result["probeRefDiffCi"] = f"[{100 * row['diff_low']:.2f}, {100 * row['diff_high']:.2f}]"
    return result


def disagreement(values: list[str]) -> float:
    k = len(values)
    counts = Counter(values)
    return 1.0 - sum(n * (n - 1) for n in counts.values()) / (k * (k - 1))


def item_disagreements(repeats: dict[int, list]) -> dict[str, float]:
    items: dict[str, list] = defaultdict(list)
    for _repeat, outputs in sorted(repeats.items()):
        for key, value in outputs:
            items[key].append(value)
    k = len(repeats)
    return {key: disagreement(values) for key, values in items.items() if len(values) == k and k >= 2}


def compare_campaigns(per_query_items: dict) -> list[dict]:
    """Main campaign vs each later campaign on the same deployment and query."""
    rows = []
    index = defaultdict(dict)
    for (scenario, model, setting, query, campaign), repeats in per_query_items.items():
        index[(scenario, model, query)][(setting, campaign)] = repeats
    for (scenario, model, query), variants in sorted(index.items()):
        main = [(s, c) for s, c in variants if c == "main"]
        if not main:
            continue
        reference = item_disagreements(variants[main[0]])
        for (setting, campaign), repeats in sorted(variants.items()):
            if campaign == "main" or len(repeats) < 2:
                continue
            other = item_disagreements(repeats)
            keys = sorted(set(reference) & set(other))
            if len(keys) < 20:
                continue
            rng = random.Random(f"{SEED}:{scenario}:{seed_name(model)}:{query}:{setting}:{campaign}")
            diffs = []
            for _ in range(RESAMPLES):
                sample = [keys[rng.randrange(len(keys))] for _ in keys]
                diffs.append(statistics.fmean(other[k] for k in sample) - statistics.fmean(reference[k] for k in sample))
            diffs.sort()
            rows.append({
                "scenario": scenario, "model": model, "query": query, "campaign": campaign, "setting": setting,
                "executions_main": len(variants[main[0]]), "executions_other": len(repeats), "items": len(keys),
                "disagreement_main": statistics.fmean(reference[k] for k in keys),
                "disagreement_other": statistics.fmean(other[k] for k in keys),
                "diff_low": diffs[int(0.025 * RESAMPLES)], "diff_high": diffs[int(0.975 * RESAMPLES) - 1],
            })
    return rows


if __name__ == "__main__":
    main()
