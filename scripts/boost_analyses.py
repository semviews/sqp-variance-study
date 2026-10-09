#!/usr/bin/env python3
"""Supplementary analyses over recorded SemBench executions.

- permutation test of whether item changes co-occur in executions.
- answer distances between pairs of executions, per answer type.
- run-to-run spread of cost (completion tokens, reasoning) and latency.
- prompt tokens of identical requests on the two Llama-3.3-70B stacks.
- AUC of first-execution reasoning length within strata of contestedness.
- bias and variance of error against SemBench's labels and gold count.
- storage per recorded output and calls saved by within-execution dedup.

Reads experiments/processed/sembench_repeats/{items,queries}.csv and the raw
cells; writes experiments/processed/sembench_repeats/boost.json (with a
"macros" map that scripts/paper_macros.py turns into LaTeX macros) and
outputs/tables/distances.tex. No model is called.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_sembench_repeats as base  # noqa: E402
from report_sembench_repeats import auc, most_common  # noqa: E402

ROOT = base.ROOT
PROCESSED = base.PROCESSED
RAW = base.RAW
OUT = PROCESSED / "boost.json"
TABLE = ROOT / "outputs" / "tables" / "distances.tex"
SEED = 11
PERMUTATIONS = 2000
OPEN = {
    "managed-gpt-oss-120b": ("gpt-oss-120B", "Gptoss"),
    "managed-llama-3-3-70b-instruct": ("Llama-70B (M)", "LlamaW"),
    "vllm-llama-3-3-70b-instruct": ("Llama-70B (V)", "LlamaR"),
    "vllm-qwen2-5-72b-instruct": ("Qwen-72B", "Qwen"),
    "managed-mistral-small-3-1-24b-2503": ("Mistral-24B", "Mistral"),
    "vllm-granite-3-3-8b-instruct": ("Granite-8B", "Granite"),
}
COMMERCIAL = {
    "Azure-gpt-4o": ("GPT-4o", "GptFour"),
    "aws-claude-sonnet-5": ("Claude Sonnet 5", "Sonnet"),
}
DEPLOYMENTS = {**OPEN, **COMMERCIAL}
MAIN = {"w20", "w20_lp"}
JOINS = {"w20_exact_join40", "w20_lp_exact_join40"}


def pct(value: float, digits: int = 1) -> str:
    return "--" if value != value else f"{100 * value:.{digits}f}\\%"


def read(name: str) -> list[dict]:
    with (PROCESSED / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cells(settings: set[str]):
    """(scenario, model, setting, query) -> sorted repeat directories with a kept answer."""
    found = defaultdict(list)
    for metrics in RAW.glob("*/*/*/repeat-*/Q*.metrics.json"):
        setting = metrics.parent.parent.name
        model = metrics.parents[2].name
        if setting not in settings or model not in DEPLOYMENTS:
            continue
        repeat = int(metrics.parent.name.split("-")[1])
        if setting in MAIN and repeat > base.MAIN_EXECUTIONS.get(model, base.MAIN_DEFAULT):
            continue
        query = int(metrics.name[1:].split(".")[0])
        found[(metrics.parents[3].name, model, setting, query)].append(metrics.parent)
    return {key: sorted(dirs, key=lambda path: int(path.name.split("-")[1])) for key, dirs in found.items()}


# ---------------------------------------------------------------- co-occurrence of item changes
def permutation_test(items: list[dict]) -> dict:
    """Do deviations from each item's majority pile up in some executions more than chance allows?

    Statistic: variance over executions of the number of deviating items.
    Null: each item's outputs are exchangeable across executions, so its
    values are permuted independently of the other items.
    """
    by_cell = defaultdict(list)
    for row in items:
        by_cell[(row["scenario"], row["query_id"], row["model"])].append([str(v) for v in json.loads(row["values"])])
    results = []
    for (scenario, query, model), rows in sorted(by_cell.items()):
        deviating = []
        for values in rows:
            majority = most_common(values)
            flags = [value != majority for value in values]
            if any(flags):
                deviating.append(flags)
        if sum(sum(flags) for flags in deviating) < 10:
            continue
        k = len(deviating[0])

        def statistic(matrix: list[list[bool]]) -> float:
            return statistics.pvariance([sum(row[r] for row in matrix) for r in range(k)])

        # Seeded per cell so results do not depend on which other cells exist or how deployments are named.
        rng = random.Random(f"{SEED}:{scenario}:{query}:{len(rows)}")
        observed = statistic(deviating)
        at_least = 0
        for _ in range(PERMUTATIONS):
            shuffled = []
            for flags in deviating:
                copy = flags[:]
                rng.shuffle(copy)
                shuffled.append(copy)
            at_least += statistic(shuffled) >= observed - 1e-12
        results.append({"scenario": scenario, "query": int(query), "model": model, "executions": k,
                        "deviations": sum(sum(flags) for flags in deviating),
                        "p": (1 + at_least) / (1 + PERMUTATIONS)})
    return {"cells": results, "tested": len(results), "below_005": sum(r["p"] < 0.05 for r in results),
            "below_001": sum(r["p"] < 0.01 for r in results),
            "median_p": statistics.median(r["p"] for r in results) if results else math.nan}


# ---------------------------------------------------------------- answer distances
def read_answer(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return rows[1:]


def kendall_tau_b(x: list[float], y: list[float]) -> float:
    concordant = discordant = ties_x = ties_y = 0
    for i, j in combinations(range(len(x)), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            ties_x += 1
        elif dy == 0:
            ties_y += 1
        elif dx * dy > 0:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return (concordant - discordant) / denominator if denominator else 1.0


DISTANCE = {
    ("movie", 1): ("set", "Jaccard distance"),
    ("movie", 2): ("set", "Jaccard distance"),
    ("movie", 3): ("number", "$|\\Delta|$ count"),
    ("movie", 4): ("number_pp", "$|\\Delta|$ ratio (points)"),
    ("movie", 8): ("labelcount", "$|\\Delta|$ positive count"),
    ("movie", 9): ("rows", "rows that differ (\\%)"),
    ("movie", 10): ("rank", "$1-\\tau_b$ of the ranking"),
    ("medical", 1): ("set", "Jaccard distance"),
    ("medical", 4): ("relative", "relative $|\\Delta|$ (\\%)"),
    ("medical", 10): ("rows", "rows that differ (\\%)"),
    ("movie", 5): ("pairs", "Jaccard distance"),
    ("movie", 6): ("pairs", "Jaccard distance"),
    ("movie", 7): ("pairs", "Jaccard distance"),
}


def distance(kind: str, left: list[list[str]], right: list[list[str]]) -> float:
    if kind in ("set", "pairs"):
        a = {tuple(row) for row in left}
        b = {tuple(row) for row in right}
        union = a | b
        return 1 - len(a & b) / len(union) if union else 0.0
    if kind in ("number", "number_pp", "relative"):
        x, y = float(left[0][0]), float(right[0][0])
        if kind == "relative":
            return abs(x - y) / abs(x) if x else math.nan
        return abs(x - y) * (100 if kind == "number_pp" else 1)
    if kind == "labelcount":
        def positive(rows):
            return sum(int(count) for label, count in rows if label.strip().upper().startswith("POS"))
        return abs(positive(left) - positive(right))
    if kind == "rows":
        a = {row[0]: tuple(row[1:]) for row in left}
        b = {row[0]: tuple(row[1:]) for row in right}
        keys = set(a) | set(b)
        return sum(a.get(key) != b.get(key) for key in keys) / len(keys) if keys else 0.0
    if kind == "rank":
        a = {row[0]: float(row[1]) for row in left}
        b = {row[0]: float(row[1]) for row in right}
        keys = sorted(set(a) & set(b))
        return 1 - kendall_tau_b([a[key] for key in keys], [b[key] for key in keys])
    raise ValueError(kind)


def answer_distances() -> list[dict]:
    rows = []
    for (scenario, model, setting, query), dirs in sorted(cells(MAIN | JOINS).items()):
        key = (scenario, query)
        if key not in DISTANCE or (query in (5, 6, 7)) != (setting in JOINS):
            continue
        answers = [read_answer(path / f"Q{query}.csv") for path in dirs if (path / f"Q{query}.csv").exists()]
        if len(answers) < 2:
            continue
        kind = DISTANCE[key][0]
        values = [distance(kind, a, b) for a, b in combinations(answers, 2)]
        values = [value for value in values if value == value]
        if kind in ("rows", "relative"):
            values = [100 * value for value in values]
        rows.append({"scenario": scenario, "query": query, "model": model, "kind": kind,
                     "pairs": len(values), "mean": statistics.fmean(values), "max": max(values)})
    return rows


def distance_table(rows: list[dict]) -> None:
    present = [model for model in DEPLOYMENTS if any(row["model"] == model for row in rows)]
    by = {(row["scenario"], row["query"], row["model"]): row for row in rows}
    lines = [
        "% Generated by scripts/boost_analyses.py. Do not edit by hand.",
        "\\begin{table*}[t]",
        "\\caption{How far apart two executions' answers are: mean over pairs of executions of a distance suited to the "
        "answer type (0 means identical). Set answers use the Jaccard distance; per-row answers the share of rows that "
        "differ; rankings $1-\\tau_b$ (Kendall); aggregates the absolute difference.}",
        "\\label{tab:distances}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}ll" + "r" * len(present) + "@{}}",
        "\\toprule",
        "Query & Distance & " + " & ".join(DEPLOYMENTS[model][0] for model in present) + " \\\\",
        "\\midrule",
    ]
    order = [(s, q) for (s, q) in DISTANCE if any((s, q, m) in by for m in present)]
    for scenario, query in order:
        cells_text = []
        for model in present:
            row = by.get((scenario, query, model))
            if not row:
                cells_text.append("--")
                continue
            value = row["mean"]
            digits = 1 if row["kind"] in ("number", "labelcount", "rows", "relative", "number_pp") else 2
            text = f"{value:.{digits}f}"
            cells_text.append("0" if float(text) == 0 else text)
        lines.append(f"{scenario.capitalize()} Q{query} & {DISTANCE[(scenario, query)][1]} & " + " & ".join(cells_text) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    TABLE.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- cost and latency spread
def cost_spread() -> list[dict]:
    rows = []
    for (scenario, model, setting, query), dirs in sorted(cells(MAIN).items()):
        if query not in base.KINDS.get(scenario, {}):
            continue
        tokens, reasoning, seconds = [], [], []
        for path in dirs:
            meta = json.loads((path / f"Q{query}.metrics.json").read_text())
            calls = [json.loads(line) for line in (path / f"Q{query}.calls.jsonl").read_text().splitlines()]
            tokens.append(sum(call.get("completion_tokens") or 0 for call in calls))
            reasoning.append(sum(call.get("reasoning_chars") or 0 for call in calls))
            if meta.get("execution_time"):
                seconds.append(float(meta["execution_time"]))
        if len(tokens) < 2:
            continue

        def cv(values: list[float]) -> float:
            mean = statistics.fmean(values)
            return statistics.stdev(values) / mean if mean else math.nan

        rows.append({"scenario": scenario, "query": query, "model": model, "executions": len(tokens),
                     "tokens_cv": cv(tokens), "tokens_min": min(tokens), "tokens_max": max(tokens),
                     "reasoning_cv": cv(reasoning) if any(reasoning) else None,
                     "seconds_cv": cv(seconds) if len(seconds) >= 2 else None,
                     "seconds_min": min(seconds) if seconds else None, "seconds_max": max(seconds) if seconds else None})
    return rows


# ---------------------------------------------------------------- prompt tokens on the two Llama stacks
def prompt_tokens_by_stack() -> dict:
    """Same request hash on W and R: does the reported prompt length differ?"""
    stacks = {"managed-llama-3-3-70b-instruct": "w20", "vllm-llama-3-3-70b-instruct": "w20_lp"}
    tokens = {}
    for model, setting in stacks.items():
        seen = {}
        for path in RAW.glob(f"*/{model}/{setting}/repeat-1/Q*.calls.jsonl"):
            for line in path.read_text().splitlines():
                call = json.loads(line)
                if call.get("prompt_tokens") is not None:
                    seen.setdefault(call["key"], set()).add(call["prompt_tokens"])
        tokens[model] = seen
    w, r = (tokens[model] for model in stacks)
    shared = sorted(set(w) & set(r))
    differences = [min(w[key]) - min(r[key]) for key in shared]
    within_w = sum(len(w[key]) > 1 for key in w)
    within_r = sum(len(r[key]) > 1 for key in r)
    return {
        "shared_requests": len(shared),
        "differ": sum(d != 0 for d in differences),
        "mean_difference": statistics.fmean(differences) if differences else math.nan,
        "difference_values": sorted(Counter(differences).items())[:10],
        "w_requests_with_two_lengths": within_w,
        "r_requests_with_two_lengths": within_r,
    }


# ---------------------------------------------------------------- reasoning length within strata
def reasoning_by_contestedness(items: list[dict]) -> dict:
    by_item = defaultdict(dict)
    for row in items:
        if row["model"] not in OPEN:
            continue
        values = [str(v) for v in json.loads(row["values"])]
        by_item[(row["scenario"], row["query_id"], row["item"])][row["model"]] = (row, most_common(values))
    model = "managed-gpt-oss-120b"
    strata = defaultdict(lambda: ([], []))
    for per_model in by_item.values():
        if len(per_model) < len(OPEN) or model not in per_model:
            continue
        row, _majority = per_model[model]
        others = [m for name, (_r, m) in per_model.items() if name != model]
        plurality = most_common(others)
        dissent = sum(m != plurality for m in others)
        values = [str(v) for v in json.loads(row["values"])]
        later = len(set(values[1:])) > 1
        score = -float(row["first_reasoning_chars"])
        strata[dissent][0 if later else 1].append(score)
    result = {}
    for dissent, (positive, negative) in sorted(strata.items()):
        result[dissent] = {"changing": len(positive), "stable": len(negative),
                           "auc": auc(positive, negative) if len(positive) >= 10 and len(negative) >= 10 else None}
    usable = [(v["auc"], v["changing"]) for v in result.values() if v["auc"] is not None]
    weighted = sum(a * n for a, n in usable) / sum(n for _a, n in usable) if usable else math.nan
    return {"strata": result, "weighted_auc": weighted}


# ---------------------------------------------------------------- bias and variance
def bias_variance(items: list[dict]) -> dict:
    labels = None
    item_level = defaultdict(lambda: {"items": 0, "majority_wrong": 0.0, "flip_k5": 0.0})
    from remedies_sembench_repeats import movie_labels, truth  # noqa: E402

    labels = movie_labels()
    texts = {}
    for path in RAW.glob("movie/managed-gpt-oss-120b/w20/repeat-1/Q*.calls.jsonl"):
        query = int(path.name[1:].split(".")[0])
        seen: Counter = Counter()
        for line in path.read_text().splitlines():
            call = json.loads(line)
            seen[call["key"]] += 1
            texts[(query, f"{call['key']}#{seen[call['key']]}")] = call
    for row in items:
        if row["scenario"] != "movie" or int(row["query_id"]) not in (3, 4, 8, 9) or row["model"] not in DEPLOYMENTS:
            continue
        query = int(row["query_id"])
        kind = base.MOVIE[query][0]
        record = texts.get((query, row["item"]))
        if record is None:
            continue
        gold = truth(kind, query, record, labels)
        if gold is None:
            continue
        values = json.loads(row["values"])
        majority = Counter(str(v) for v in values).most_common(1)[0][0]
        if kind == "score":
            wrong = abs(float(majority) - float(gold)) > 0.5
        else:
            wrong = majority != str(gold)
        cell = item_level[(row["model"], query)]
        cell["items"] += 1
        cell["majority_wrong"] += wrong
        cell["flip_k5"] += float(row["flip_k5"])
    from plot_teaser import gold as gold_count  # noqa: E402

    truth_count = gold_count()
    count_level = {}
    for (scenario, model, setting, query), dirs in cells(MAIN).items():
        if (scenario, query) != ("movie", 3):
            continue
        counts = [int(float(read_answer(path / "Q3.csv")[0][0])) for path in dirs]
        mean = statistics.fmean(counts)
        mse = statistics.fmean((c - truth_count) ** 2 for c in counts)
        variance = statistics.pvariance(counts)
        count_level[model] = {"gold": truth_count, "mean": mean, "mse": mse, "bias_sq": (mean - truth_count) ** 2,
                              "variance": variance, "variance_share": variance / mse if mse else 0.0}
    return {"items": {f"{m}|{q}": v for (m, q), v in item_level.items()}, "count": count_level}


# ---------------------------------------------------------------- storage and deduplication
def storage_and_dedup() -> dict:
    model, setting = "managed-gpt-oss-120b", "w20"
    outputs = keys = calls = 0
    output_bytes = 0
    for path in sorted(RAW.glob(f"*/{model}/{setting}/repeat-1/Q*.calls.jsonl")):
        query = int(path.name[1:].split(".")[0])
        if query in (5, 6, 7):
            continue
        seen = set()
        for line in path.read_text().splitlines():
            call = json.loads(line)
            calls += 1
            seen.add(call["key"])
            output_bytes += len((call.get("output") or "").encode())
        keys += len(seen)
    outputs = calls
    key_bytes = 8
    return {
        "calls_per_execution": calls,
        "distinct_requests": keys,
        "dedup_saved_share": (calls - keys) / calls,
        "bytes_per_output": output_bytes / outputs + key_bytes,
        "bytes_per_output_with_signal": output_bytes / outputs + key_bytes + 4,
        "kilobytes_per_execution": (output_bytes + calls * (key_bytes + 4)) / 1024,
    }


def main() -> None:
    items = [row for row in read("items") if row["setting"] in MAIN and row["model"] in DEPLOYMENTS]
    open_items = [row for row in items if row["model"] in OPEN]
    facts = {
        "permutation": permutation_test(open_items),
        "distances": answer_distances(),
        "cost_spread": cost_spread(),
        "prompt_tokens": prompt_tokens_by_stack(),
        "reasoning_strata": reasoning_by_contestedness(open_items),
        "bias_variance": bias_variance(open_items),
        "storage": storage_and_dedup(),
    }
    distance_table(facts["distances"])
    macros = {}
    perm = facts["permutation"]
    macros["permCells"] = perm["tested"]
    macros["permBelowFive"] = perm["below_005"]
    macros["permBelowOne"] = perm["below_001"]
    macros["permMedianP"] = f"{perm['median_p']:.2f}"
    dist = {(row["scenario"], row["query"], row["model"]): row for row in facts["distances"]}
    for (scenario, query, model), row in dist.items():
        suffix = DEPLOYMENTS[model][1]
        name = f"dist{scenario.capitalize()}{base_word(query)}{suffix}"
        digits = 2 if row["kind"] in ("set", "pairs", "rank") else 1
        macros[name] = f"{row['mean']:.{digits}f}"
        macros[name + "Max"] = f"{row['max']:.{digits}f}"
    spread = facts["cost_spread"]
    for model, (_label, suffix) in DEPLOYMENTS.items():
        rows = [row for row in spread if row["model"] == model]
        if not rows:
            continue
        macros[f"tokCvMed{suffix}"] = pct(statistics.median(row["tokens_cv"] for row in rows), 1)
        macros[f"tokCvMax{suffix}"] = pct(max(row["tokens_cv"] for row in rows), 1)
        timed = [row["seconds_cv"] for row in rows if row["seconds_cv"] is not None]
        if timed:
            macros[f"secCvMed{suffix}"] = pct(statistics.median(timed), 0)
            macros[f"secCvMax{suffix}"] = pct(max(timed), 0)
        timed_rows = [row for row in rows if row["seconds_min"]]
        if timed_rows:
            ratio = max(row["seconds_max"] / row["seconds_min"] for row in timed_rows)
            macros[f"secRatioMax{suffix}"] = f"{ratio:.1f}"
    gpt_rows = [row for row in spread if row["model"] == "managed-gpt-oss-120b" and row["reasoning_cv"]]
    if gpt_rows:
        macros["reasonCvMedGptoss"] = pct(statistics.median(row["reasoning_cv"] for row in gpt_rows), 1)
    tokens = facts["prompt_tokens"]
    macros["stackSharedRequests"] = f"{tokens['shared_requests']:,}".replace(",", "{,}")
    macros["stackPromptDiffer"] = f"{tokens['differ']:,}".replace(",", "{,}")
    macros["stackPromptDifferPct"] = pct(tokens["differ"] / tokens["shared_requests"], 0) if tokens["shared_requests"] else "--"
    macros["stackPromptMeanDiff"] = f"{tokens['mean_difference']:.1f}"
    strata = facts["reasoning_strata"]
    words = ["Zero", "One", "Two", "Three", "Four", "Five"]
    for dissent, values in strata["strata"].items():
        if values["auc"] is not None:
            macros[f"reasonAucDissent{words[int(dissent)]}"] = f"{values['auc']:.2f}"
            macros[f"reasonAucDissent{words[int(dissent)]}N"] = values["changing"]
    macros["reasonAucStratified"] = f"{strata['weighted_auc']:.2f}"
    bv = facts["bias_variance"]
    for model, values in bv["count"].items():
        suffix = DEPLOYMENTS[model][1]
        macros[f"countVarShare{suffix}"] = pct(values["variance_share"], 0)
        macros[f"countMean{suffix}"] = f"{values['mean']:.1f}"
    pooled = defaultdict(lambda: [0, 0.0, 0.0])
    for key, values in bv["items"].items():
        model = key.split("|")[0]
        pooled[model][0] += values["items"]
        pooled[model][1] += values["majority_wrong"]
        pooled[model][2] += values["flip_k5"]
    for model, (count, wrong, flip) in pooled.items():
        suffix = DEPLOYMENTS[model][1]
        macros[f"majWrong{suffix}"] = pct(wrong / count, 1)
        macros[f"labFlip{suffix}"] = pct(flip / count, 1)
    store = facts["storage"]
    macros["storeBytes"] = f"{store['bytes_per_output']:.0f}"
    macros["storeKb"] = f"{store['kilobytes_per_execution']:.0f}"
    macros["dedupSaved"] = pct(store["dedup_saved_share"], 1)
    macros["dedupCalls"] = f"{store['calls_per_execution']:,}".replace(",", "{,}")
    facts["macros"] = macros
    OUT.write_text(json.dumps(facts, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in facts.items() if k not in ("distances", "cost_spread")}, indent=1, default=str)[:6000])


def base_word(query: int) -> str:
    return ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"][query]


if __name__ == "__main__":
    main()
