#!/usr/bin/env python3
"""Figure: how the flip rate grows with executions, and how plans amplify item disagreement.

Left: the expected item flip rate of gpt-oss-120B over m of its ten SemBench
executions, per output type. Right: measured answer reproduction of every
SemBench cell, including the joins, against the expected number of items on
which two executions disagree, sum_i delta_i, with the possible-worlds
prediction exp(-x) for answers that expose every item.
Reads experiments/processed/sembench_repeats/{items,queries}.csv. Writes
outputs/figures/amplification.pdf and experiments/processed/sembench_repeats/amplification.json.
No model is called.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from report_sembench_repeats import MODELS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "processed" / "sembench_repeats"
OUT = ROOT / "outputs" / "figures" / "amplification.pdf"
FACTS = DATA / "amplification.json"
SETTINGS = {"w20", "w20_lp", "w20_exact_join40", "w20_lp_exact_join40"}
TEN = "managed-gpt-oss-120b"
EXPOSES_EVERY_ITEM = {"vector", "set", "pairs"}
KIND_LABELS = {"bool": "Boolean", "label": "2-way label", "score": "1–5 score", "extract": "24-way extract"}


def subset_flip(values: list, m: int) -> float:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return 1.0 - sum(math.comb(n, m) for n in counts.values()) / math.comb(len(values), m)


def read(name: str) -> list[dict]:
    with (DATA / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    queries = {(row["scenario"], row["model"], row["setting"], row["query_id"]): row
               for row in read("queries") if row["setting"] in SETTINGS and row["model"] in MODELS}
    expected_disagreements = defaultdict(float)
    by_kind = defaultdict(list)
    for row in read("items"):
        key = (row["scenario"], row["model"], row["setting"], row["query_id"])
        if key not in queries:
            continue
        expected_disagreements[key] += float(row["pair_disagreement"])
        if row["model"] == TEN and not row["setting"].endswith("join40"):
            by_kind[queries[key]["kind"]].append(json.loads(row["values"]))

    figure, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    curves = {}
    for kind in ("bool", "label", "score", "extract"):
        items = by_kind[kind]
        repeats = min(len(values) for values in items)
        ms = list(range(2, repeats + 1))
        rates = [100 * statistics.fmean(subset_flip(values, m) for values in items) for m in ms]
        curves[kind] = dict(zip(ms, rates))
        left.plot(ms, rates, marker="o", markersize=2.5, label=KIND_LABELS[kind])
    left.set_xlabel("Executions of gpt-oss-120B observed", fontsize=7)
    left.set_ylabel("Item flip rate (%)", fontsize=7)
    left.tick_params(labelsize=7)
    left.legend(fontsize=6, frameon=False)

    points = []
    for key, row in queries.items():
        x = expected_disagreements[key]
        if x <= 0:
            continue
        points.append({"plan": row["plan"], "x": x, "measured": float(row["answer_reproduction"]),
                       "every_item": row["plan"] in EXPOSES_EVERY_ITEM, "query": f"{key[0]} Q{key[3]}",
                       "model": key[1], "join": key[2].endswith("join40")})
    for every_item, style in ((True, dict(marker="o", color="black")), (False, dict(marker="o", facecolors="none", edgecolors="grey"))):
        chosen = [point for point in points if point["every_item"] == every_item]
        right.scatter([point["x"] for point in chosen], [point["measured"] for point in chosen], s=12,
                      label="answer exposes every item" if every_item else "LIMIT, aggregate, or ranking", **style)
    xs = [10 ** (exponent / 20) for exponent in range(-55, 61)]
    right.plot(xs, [math.exp(-x) for x in xs], color="tab:red", linewidth=0.9, label="$e^{-x}$")
    right.set_xscale("log")
    right.set_xlim(0.003, 1000)
    right.set_xlabel("Expected items on which two executions disagree, $x=\\sum_i\\delta_i$", fontsize=7)
    right.set_ylabel("Answer reproduction", fontsize=7)
    right.tick_params(labelsize=7)
    right.legend(fontsize=6, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.94))
    figure.tight_layout()
    figure.savefig(OUT)

    every = [point for point in points if point["every_item"]]
    other = [point for point in points if not point["every_item"]]
    facts = {
        "flip_by_k_by_kind": curves,
        "every_item_cells": len(every),
        "every_item_mean_abs_error": statistics.fmean(abs(point["measured"] - math.exp(-point["x"])) for point in every),
        "every_item_within_015": sum(abs(point["measured"] - math.exp(-point["x"])) <= 0.15 for point in every),
        "other_cells": len(other),
        "other_above_curve": sum(point["measured"] >= math.exp(-point["x"]) - 0.05 for point in other),
        "points": points,
    }
    FACTS.write_text(json.dumps(facts, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in facts.items() if key != "points"}, indent=1))


if __name__ == "__main__":
    main()
