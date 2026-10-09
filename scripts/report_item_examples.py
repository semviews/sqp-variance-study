#!/usr/bin/env python3
"""Identical prompts within one execution, and example items that change.

SemBench's inputs contain duplicate rows, which LOTUS sends as separate,
identical requests (its cache is disabled). Comparing the two copies within
one execution against one copy across executions tells whether variance
belongs to the call or to the execution. Writes
experiments/processed/sembench_repeats/duplicates.json and
outputs/tables/examples.tex. No model is called.
"""

from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "experiments" / "processed" / "sembench_repeats"
RAW = ROOT / "experiments" / "raw" / "sembench-repeats"
TABLES = ROOT / "outputs" / "tables"
MAIN_SETTINGS = {"w20", "w20_lp"}
REFERENCE = "managed-gpt-oss-120b"
OTHERS = ["vllm-granite-3-3-8b-instruct", "vllm-llama-3-3-70b-instruct", "vllm-qwen2-5-72b-instruct",
          "managed-llama-3-3-70b-instruct", "managed-mistral-small-3-1-24b-2503"]

# Hand-picked items whose gpt-oss-120B output changes; chosen to cover a
# Boolean filter, a score with a duplicate input, and an extraction.
EXAMPLES = [
    ("movie", 3, "058bc5511280cd03", "Q3, clearly positive?"),
    ("movie", 3, "8552f1906afa7e9a", "Q3, clearly positive?"),
    ("movie", 9, "87bf1881f8285c0a", "Q9, score 1--5:"),
    ("medical", 10, "c48b82d3d730271b", "Q10, disease:"),
]


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / (1 + z * z / total)
    return (max(0.0, centre - half), min(1.0, centre + half))


def label(value) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, str):
        return value.lower().replace("common cold", "cold")
    return str(value)


def counts(values: list) -> str:
    return ", ".join(f"{label(value)} ({count})" for value, count in Counter(values).most_common())


def excerpt(text: str, words: int = 22) -> str:
    match = re.search("\u00ab(.*?)\u00bb", text, re.S)
    body = html.unescape(match.group(1) if match else text).strip()
    tokens = body.split()
    short = " ".join(tokens[:words]) + (" \\ldots" if len(tokens) > words else "")
    return short.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#").replace("\u2019", "'")


def main() -> None:
    values = defaultdict(dict)
    for row in csv.DictReader((PROCESSED / "items.csv").open(newline="", encoding="utf-8")):
        if row["setting"] not in MAIN_SETTINGS:
            continue
        key, _, copy = row["item"].rpartition("#")
        values[(row["model"], row["scenario"], int(row["query_id"]), key)][copy] = json.loads(row["values"])

    tally = defaultdict(Counter)
    for (model, _scenario, _query, _key), copies in values.items():
        if len(copies) < 2:
            continue
        first, second = (copies[name] for name in sorted(copies)[:2])
        runs = min(len(first), len(second))
        tally[model]["inputs"] += 1
        tally[model]["within"] += sum(first[r] != second[r] for r in range(runs))
        tally[model]["within_pairs"] += runs
        for copy in (first[:runs], second[:runs]):
            pairs = list(combinations(range(runs), 2))
            tally[model]["across"] += sum(copy[a] != copy[b] for a, b in pairs)
            tally[model]["across_pairs"] += len(pairs)
    facts = {}
    for model, entry in tally.items():
        within = entry["within"] / entry["within_pairs"]
        across = entry["across"] / entry["across_pairs"]
        facts[model] = {"duplicated_inputs": entry["inputs"], "within": within, "across": across,
                        "within_ci": wilson(entry["within"], entry["within_pairs"]),
                        "within_differ": entry["within"], "within_pairs": entry["within_pairs"]}
    duplicated = [copies for (model, *_rest), copies in values.items() if model == REFERENCE and len(copies) > 1]
    example = next(values[(REFERENCE, scenario, query, key)] for scenario, query, key, _title in EXAMPLES
                   if len(values[(REFERENCE, scenario, query, key)]) > 1)
    first, second = (example[name] for name in sorted(example)[:2])
    facts["example"] = {"executions": len(first),
                        "copies_differ": sum(a != b for a, b in zip(first, second)),
                        "reference_inputs_any_within": sum(
                            any(a != b for a, b in zip(*(copies[name] for name in sorted(copies)[:2])))
                            for copies in duplicated)}
    (PROCESSED / "duplicates.json").write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")

    lines = ["% Generated by scripts/report_item_examples.py. Do not edit by hand.", "\\begin{table}[tb]",
             "\\caption{Items whose output changes on gpt-oss-120B: its outputs over ten executions with their counts, "
             "and the majority output of each other deployment (Granite-8B, Llama-70B (V), Qwen-72B, "
             "Llama-70B (M), Mistral-24B). The Q9 review occurs twice in the input; each copy is a separate, "
             "identical request.}",
             "\\label{tab:examples}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{@{}p{0.47\\columnwidth}p{0.5\\columnwidth}@{}}", "\\toprule",
             "gpt-oss-120B & Other deployments \\\\", "\\midrule"]
    for scenario, query, key, title in EXAMPLES:
        copies = values[(REFERENCE, scenario, query, key)]
        calls = RAW / scenario / REFERENCE / "w20" / "repeat-1" / f"Q{query}.calls.jsonl"
        text = next(json.loads(line)["text"] for line in calls.read_text().splitlines()
                    if json.loads(line)["key"] == key)
        mine = " / ".join(counts(copies[name]) for name in sorted(copies))
        others = ", ".join(label(Counter(values[(model, scenario, query, key)][sorted(copies)[0]]).most_common(1)[0][0])
                           for model in OTHERS)
        lines.append(f"\\multicolumn{{2}}{{@{{}}p{{\\columnwidth}}@{{}}}}{{\\textit{{{scenario.capitalize()} "
                     f"{title}}} ``{excerpt(text)}''}} \\\\")
        lines.append(f"{mine} & {others} \\\\")
        lines.append("\\addlinespace[3pt]")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLES / "examples.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(facts["example"])
    for model, entry in facts.items():
        if model == "example":
            continue
        print(f"{model:36s} inputs={entry['duplicated_inputs']} within={entry['within']:.4f} "
              f"across={entry['across']:.4f}")


if __name__ == "__main__":
    main()
