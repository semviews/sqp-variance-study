"""Analyse the temperature control of gpt-oss-120B.

scripts/temperature_control.py replays the recorded LOTUS requests of Movie
Q1 and Q3 for a manifest of items (half that changed in the campaign, half
that did not), 20 times under each setting: temperature 0 (t0), temperature
0 with a fixed seed (t0-seed), and temperature 1 (t1). Outputs are parsed by
LOTUS's filter rule.

Per setting: mean pairwise disagreement over items (separately for items
that changed in the campaign and items that did not), the expected flip rate
over five executions, and bootstrap intervals over items for the differences
t0-seed - t0 and t1 - t0 (10,000 resamples, seed 11). The campaign
disagreement of the same items comes from the manifest.

Writes experiments/processed/controls/temperature.json with a "macros" map.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from analyze_sembench_repeats import pair_disagreement, parse, subset_flip

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "experiments" / "raw" / "controls" / "temperature"
OUT = ROOT / "experiments" / "processed" / "controls" / "temperature.json"
SETTINGS = ("t0", "t0-seed", "t1")
NAMES = {"t0": "Zero", "t0-seed": "Seed", "t1": "One"}
RESAMPLES, SEED, MIN_REPEATS = 10_000, 11, 5


def campaign_value(text: str) -> bool:
    return str(text).strip().lower() == "true"


def load(query: int, setting: str) -> dict[str, dict[int, bool]]:
    path = RAW / f"Q{query}-{setting}.jsonl"
    outputs: dict[str, dict[int, bool]] = defaultdict(dict)
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("output") is not None:
                outputs[row["key"]][int(row["repeat"])] = parse("bool", row["output"])
    return outputs


def bootstrap_difference(a: list[float], b: list[float]) -> tuple[float, float]:
    rng = random.Random(SEED)
    pairs = list(zip(a, b))
    values = sorted(sum(y - x for x, y in rng.choices(pairs, k=len(pairs))) / len(pairs) for _ in range(RESAMPLES))
    return values[int(0.025 * RESAMPLES)], values[int(0.975 * RESAMPLES) - 1]


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}\\%"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result, macros = {}, {}
    for query in (3, 1):
        manifest_path = RAW / f"manifest-Q{query}.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        items = {item["key"]: item for item in manifest["items"]}
        per_setting = {setting: load(query, setting) for setting in SETTINGS}
        complete = [key for key in items
                    if all(len(per_setting[s].get(key, {})) >= MIN_REPEATS for s in SETTINGS)]
        repeats = {s: min((len(per_setting[s][k]) for k in complete), default=0) for s in SETTINGS}
        entry = {"items": len(complete), "repeats_min": repeats, "settings": {}}
        disagreement = {}
        for setting in SETTINGS:
            values = {key: [per_setting[setting][key][r] for r in sorted(per_setting[setting][key])] for key in complete}
            disagreement[setting] = [pair_disagreement(values[key]) for key in complete]
            groups = {}
            for label, flag in (("changed", True), ("stable", False)):
                keys = [key for key in complete if items[key]["flipped_in_campaign"] == flag]
                groups[label] = {
                    "items": len(keys),
                    "disagreement": sum(pair_disagreement(values[k]) for k in keys) / len(keys) if keys else None,
                    "items_varying": sum(len(set(values[k])) > 1 for k in keys),
                }
            entry["settings"][setting] = {
                "disagreement": sum(disagreement[setting]) / len(complete) if complete else None,
                "flip_k5": sum(subset_flip(values[k], 5) for k in complete) / len(complete) if complete else None,
                "groups": groups,
                "fingerprints": sorted({str(json.loads(line).get("fingerprint"))
                                        for line in (RAW / f"Q{query}-{setting}.jsonl").read_text().splitlines()}),
            }
        campaign = [pair_disagreement([campaign_value(v) for v in items[key]["campaign_outputs"]]) for key in complete]
        entry["campaign_disagreement"] = sum(campaign) / len(campaign) if campaign else None
        for setting in ("t0-seed", "t1"):
            if complete:
                low, high = bootstrap_difference(disagreement["t0"], disagreement[setting])
                entry[f"diff_{setting}"] = {"mean": entry["settings"][setting]["disagreement"] - entry["settings"]["t0"]["disagreement"],
                                            "ci": [low, high]}
        if complete:
            low, high = bootstrap_difference(campaign, disagreement["t0"])
            entry["diff_t0_vs_campaign"] = {"mean": entry["settings"]["t0"]["disagreement"] - entry["campaign_disagreement"],
                                            "ci": [low, high]}
        result[f"Q{query}"] = entry

        word = {3: "Three", 1: "One"}[query]
        macros[f"tempItems{word}"] = len(complete)
        macros[f"tempRepeats{word}"] = min(repeats.values())
        if complete:
            macros[f"tempCampaign{word}"] = pct(entry["campaign_disagreement"])
            for setting, name in NAMES.items():
                values = entry["settings"][setting]
                macros[f"temp{name}{word}"] = pct(values["disagreement"])
                macros[f"temp{name}Changed{word}"] = pct(values["groups"]["changed"]["disagreement"])
                macros[f"temp{name}Stable{word}"] = pct(values["groups"]["stable"]["disagreement"])
                macros[f"temp{name}StableVarying{word}"] = values["groups"]["stable"]["items_varying"]
            for setting, name in (("t0-seed", "Seed"), ("t1", "One")):
                low, high = entry[f"diff_{setting}"]["ci"]
                macros[f"tempDiff{name}Ci{word}"] = f"[{100 * low:+.1f}, {100 * high:+.1f}]"
            low, high = entry["diff_t0_vs_campaign"]["ci"]
            macros[f"tempDiffCampaignCi{word}"] = f"[{100 * low:+.1f}, {100 * high:+.1f}]"
    OUT.write_text(json.dumps({"queries": result, "macros": macros}, indent=1) + "\n")
    print(json.dumps({q: {k: v for k, v in e.items() if k != "settings"} | {
        s: {k: v for k, v in e["settings"][s].items() if k != "fingerprints"} for s in SETTINGS}
        for q, e in result.items()}, indent=1))


if __name__ == "__main__":
    main()
