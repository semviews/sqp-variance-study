"""Where two reasoning traces of the same item diverge.

Reads the temperature-0 executions of the gpt-oss-120B control
(experiments/raw/controls/temperature/Q{1,3}-t0.jsonl), which log the full
reasoning text. For every item and every pair of executions, the traces are
split into words and compared: identical, or the first word at which they
differ, as a fraction of the shorter trace. Pairs are split by whether the
final outputs (LOTUS's filter rule) differ.

Reported per query: the share of pairs with identical traces, the number of
identical-trace pairs whose outputs differ, the mean relative divergence
position of output-changing and output-preserving pairs, a bootstrap
interval over items for their difference (10,000 resamples, seed 11), the
share of pairs that diverge in the first tenth of the trace, and the rank
correlation across items between median trace length and the share of
diverging pairs.

Writes experiments/processed/controls/divergence.json with a "macros" map.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median

from analyze_sembench_repeats import parse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "experiments" / "raw" / "controls" / "temperature"
OUT = ROOT / "experiments" / "processed" / "controls" / "divergence.json"
QUERIES = {3: "Three", 1: "One"}
RESAMPLES, SEED, EARLY = 10_000, 11, 0.1


def load(query: int) -> dict[str, dict[int, tuple[bool, list[str]]]]:
    items: dict[str, dict[int, tuple[bool, list[str]]]] = defaultdict(dict)
    path = RAW / f"Q{query}-t0.jsonl"
    if not path.exists():
        return items
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("output") is None or row.get("reasoning_text") is None:
            continue
        items[row["key"]][int(row["repeat"])] = (parse("bool", row["output"]), row["reasoning_text"].split())
    return items


def first_divergence(a: list[str], b: list[str]) -> float | None:
    """Relative position of the first differing word, or None if identical."""
    if a == b:
        return None
    shorter = min(len(a), len(b))
    index = next((i for i in range(shorter) if a[i] != b[i]), shorter)
    return index / max(shorter, 1)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            result[order[k]] = (i + j) / 2
        i = j + 1
    return result


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return cov / var if var else 0.0


def pct(value: float) -> str:
    return f"{100 * value:.1f}\\%"


def analyse(query: int) -> dict:
    items = load(query)
    pairs = identical = identical_flip = 0
    flip_rel: list[float] = []
    same_rel: list[float] = []
    per_item: list[tuple[float, float]] = []
    lengths: list[float] = []
    diverging: list[float] = []
    for runs in items.values():
        if len(runs) < 2:
            continue
        item_flip, item_same, item_pairs, item_div = [], [], 0, 0
        for (_, (out_a, text_a)), (_, (out_b, text_b)) in combinations(sorted(runs.items()), 2):
            pairs += 1
            item_pairs += 1
            position = first_divergence(text_a, text_b)
            if position is None:
                identical += 1
                identical_flip += out_a != out_b
                continue
            item_div += 1
            (item_flip if out_a != out_b else item_same).append(position)
        flip_rel += item_flip
        same_rel += item_same
        if item_flip and item_same:
            per_item.append((mean(item_flip), mean(item_same)))
        lengths.append(median(len(text) for _, text in runs.values()))
        diverging.append(item_div / item_pairs)
    rng = random.Random(SEED)
    diffs = sorted(
        mean(f - s for f, s in rng.choices(per_item, k=len(per_item))) for _ in range(RESAMPLES)
    ) if per_item else []
    return {
        "items": len(lengths),
        "pairs": pairs,
        "identical": identical,
        "identical_flip": identical_flip,
        "flip_pairs": len(flip_rel),
        "same_pairs": len(same_rel),
        "flip_rel": mean(flip_rel) if flip_rel else None,
        "same_rel": mean(same_rel) if same_rel else None,
        "flip_early": sum(p < EARLY for p in flip_rel) / len(flip_rel) if flip_rel else None,
        "same_early": sum(p < EARLY for p in same_rel) / len(same_rel) if same_rel else None,
        "items_both": len(per_item),
        "diff_ci": [diffs[int(0.025 * RESAMPLES)], diffs[int(0.975 * RESAMPLES) - 1]] if diffs else None,
        "length_rho": spearman(lengths, diverging) if len(lengths) > 2 else None,
    }


def main() -> None:
    results = {query: analyse(query) for query in QUERIES}
    macros: dict[str, str] = {}
    for query, name in QUERIES.items():
        r = results[query]
        if not r["pairs"]:
            continue
        macros[f"divPairs{name}"] = f"{r['pairs']:,}".replace(",", "{,}")
        macros[f"divIdenticalPct{name}"] = pct(r["identical"] / r["pairs"])
        macros[f"divIdenticalFlip{name}"] = str(r["identical_flip"])
        macros[f"divFlipPairs{name}"] = f"{r['flip_pairs']:,}".replace(",", "{,}")
        if r["flip_rel"] is not None:
            macros[f"divFlipRel{name}"] = f"{r['flip_rel']:.2f}"
            macros[f"divFlipEarly{name}"] = pct(r["flip_early"])
        if r["same_rel"] is not None:
            macros[f"divSameRel{name}"] = f"{r['same_rel']:.2f}"
            macros[f"divSameEarly{name}"] = pct(r["same_early"])
        if r["diff_ci"]:
            macros[f"divRelDiffCi{name}"] = f"[{r['diff_ci'][0]:+.2f}, {r['diff_ci'][1]:+.2f}]"
        if r["length_rho"] is not None:
            macros[f"divLengthRho{name}"] = f"{r['length_rho']:.2f}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"results": results, "macros": macros}, indent=1) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
