#!/usr/bin/env python3
"""Replay proxy cascades with statistical guarantees against an oracle that varies across executions.

A cascade in the style of SUPG returns the rows whose proxy score clears a
threshold. The threshold is calibrated on a uniform sample labelled by the
oracle, so that precision (or recall) with respect to the oracle is at least
a target with high probability. The guarantee assumes one oracle labelling.
Here the oracle is a model with several recorded executions, so we calibrate
on one execution and measure the achieved precision and recall against the
same execution and against each other one.

Proxy: Granite-8B, whose first-execution decision margin, signed by its
output, is the score. Oracles: the other deployments' recorded executions.
Reads experiments/processed/sembench_repeats/items.csv. Writes
experiments/processed/sembench_repeats/cascade.json and outputs/tables/cascade.tex.
No model is called.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "processed" / "sembench_repeats"
TABLES = ROOT / "outputs" / "tables"
SETTINGS = {"w20", "w20_lp"}
PROXY = "vllm-granite-3-3-8b-instruct"
ORACLES = {"managed-gpt-oss-120b": "gpt-oss-120B", "managed-llama-3-3-70b-instruct": "Llama-70B (M)"}
QUERIES = {("medical", "1"): "Medical Q1", ("medical", "4"): "Medical Q4"}
TARGET = 0.9
Z = 1.645  # one-sided 95%
SAMPLE = 400
DRAWS = 300
SEED = 11


def lower_bound(successes: int, total: int) -> float:
    """One-sided Wilson lower bound."""
    if total == 0:
        return 0.0
    p = successes / total
    denominator = 1 + Z * Z / total
    centre = p + Z * Z / (2 * total)
    half = Z * math.sqrt(p * (1 - p) / total + Z * Z / (4 * total * total))
    return (centre - half) / denominator


def precision_threshold(sample: list[tuple[float, bool]]) -> float:
    """Smallest threshold whose calibrated precision clears the target with 95% confidence."""
    ordered = sorted(sample, key=lambda pair: -pair[0])
    best, positives = math.inf, 0
    for count, (score, label) in enumerate(ordered, start=1):
        positives += label
        if count >= 10 and lower_bound(positives, count) >= TARGET:
            best = score
    return best


def recall_threshold(sample: list[tuple[float, bool]]) -> float:
    """Largest threshold whose calibrated recall clears the target with 95% confidence."""
    positive_scores = sorted((score for score, label in sample if label), reverse=True)
    total = len(positive_scores)
    for count, score in enumerate(positive_scores, start=1):
        if lower_bound(count, total) >= TARGET:
            return score
    return -math.inf


def achieved(selected: set[str], truth: set[str]) -> tuple[float, float]:
    precision = len(selected & truth) / len(selected) if selected else 1.0
    recall = len(selected & truth) / len(truth) if truth else 1.0
    return precision, recall


def main() -> None:
    values = defaultdict(dict)
    margins = {}
    with (DATA / "items.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario"], row["query_id"])
            if row["setting"] not in SETTINGS or key not in QUERIES:
                continue
            values[key, row["model"]][row["item"]] = json.loads(row["values"])
            if row["model"] == PROXY:
                margin = row["first_margin"]
                margins[key, row["item"]] = 0.0 if margin in ("", "inf") else float(margin)

    rng = random.Random(SEED)
    results, lines = [], []
    for query, query_name in QUERIES.items():
        proxy = values[query, PROXY]
        score = {item: (1 if runs[0] is True else -1) * margins[query, item] for item, runs in proxy.items()}
        for oracle, oracle_name in ORACLES.items():
            runs = values[query, oracle]
            items = sorted(set(runs) & set(score))
            k = min(len(runs[item]) for item in items)
            truth = [{item for item in items if runs[item][r] is True} for r in range(k)]
            counts = {"precision": defaultdict(lambda: [0, 0]), "recall": defaultdict(lambda: [0, 0])}
            thresholds = {"precision": [], "recall": []}
            shortfall = {"precision": [], "recall": []}
            reproduced = {"precision": [], "recall": []}
            sizes = {"precision": [], "recall": []}
            for _ in range(DRAWS):
                sample = rng.sample(items, SAMPLE)
                per_run = {"precision": [], "recall": []}
                for r in range(k):
                    labelled = [(score[item], item in truth[r]) for item in sample]
                    for target, choose in (("precision", precision_threshold), ("recall", recall_threshold)):
                        tau = choose(labelled)
                        per_run[target].append(tau)
                        selected = {item for item in items if score[item] >= tau}
                        for s in range(k):
                            value = achieved(selected, truth[s])[0 if target == "precision" else 1]
                            bucket = "same" if s == r else "other"
                            counts[target][bucket][0] += value < TARGET
                            counts[target][bucket][1] += 1
                            if s != r:
                                shortfall[target].append(max(0.0, TARGET - value))
                for target in per_run:
                    finite = [tau for tau in per_run[target] if math.isfinite(tau)]
                    thresholds[target].append(len(set(per_run[target])) > 1 if finite else False)
                    returned = [frozenset(item for item in items if score[item] >= tau) for tau in per_run[target]]
                    pairs = [(a, b) for i, a in enumerate(returned) for b in returned[i + 1:]]
                    reproduced[target].append(sum(a == b for a, b in pairs) / len(pairs))
                    sizes[target].append(statistics.fmean(len(a) for a in returned))
            oracle_agreement = statistics.fmean(
                len(truth[r] ^ truth[s]) / len(items) for r in range(k) for s in range(k) if r < s
            )
            entry = {
                "query": query_name, "oracle": oracle_name, "items": len(items), "oracle_runs": k,
                "oracle_positive_rate": statistics.fmean(len(t) for t in truth) / len(items),
                "oracle_disagreement": oracle_agreement,
            }
            for target in ("precision", "recall"):
                same, other = counts[target]["same"], counts[target]["other"]
                entry[f"{target}_violation_same"] = same[0] / same[1]
                entry[f"{target}_violation_other"] = other[0] / other[1]
                entry[f"{target}_threshold_moves"] = statistics.fmean(thresholds[target])
                entry[f"{target}_max_shortfall"] = max(shortfall[target])
                entry[f"{target}_reproduction"] = statistics.fmean(reproduced[target])
                entry[f"{target}_returned"] = statistics.fmean(sizes[target])
            results.append(entry)
            print(json.dumps({key: round(value, 4) if isinstance(value, float) else value for key, value in entry.items()}))

    (DATA / "cascade.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    lines = [
        "% Generated by scripts/cascade_replay.py. Do not edit by hand.",
        "\\begin{table}[t]",
        "\\caption{Proxy cascades calibrated for 90\\% recall with 95\\% confidence on "
        f"{SAMPLE} oracle-labelled rows ({DRAWS} samples), with Granite-8B as proxy. Dis.: pairwise disagreement of "
        "the oracle's executions. Rows: mean rows returned. Violation: share of calibrations whose recall falls "
        "below 90\\% against the oracle execution that labelled the sample (same) or another one (other). Repro: "
        "for a fixed sample, the share of pairs of oracle executions whose calibrations return the same rows. "
        "All rates in \\%.}",
        "\\label{tab:cascade}",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}llrrrrr@{}}",
        "\\toprule",
        "& & & & \\multicolumn{2}{c}{Violation} & \\\\",
        "\\cmidrule(lr){5-6}",
        "Query & Oracle & Dis. & Rows & Same & Other & Repro \\\\",
        "\\midrule",
    ]
    for entry in results:
        cells = [f"{100 * entry['oracle_disagreement']:.2f}", f"{entry['recall_returned']:.0f}",
                 f"{100 * entry['recall_violation_same']:.1f}", f"{100 * entry['recall_violation_other']:.1f}",
                 f"{100 * entry['recall_reproduction']:.0f}"]
        lines.append(f"{entry['query']} & {entry['oracle']} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "cascade.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
