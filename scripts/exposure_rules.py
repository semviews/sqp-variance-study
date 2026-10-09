"""Exposure rule per plan operator, each validated on the cells that use it.

For every plan class of the SemBench queries, the table states which items
decide the answer and compares the possible-worlds prediction with measured
answer reproduction over the main open-weight cells. For answers that expose
every item, the prediction is the closed form e^{-x} with x the sum of item
disagreements; for the other plans, the independence simulation through the
plan. The UDA-Bench row uses the lineage exposure of scripts/uda_lineage.py.

Writes experiments/processed/exposure.json and outputs/tables/exposure.tex.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "experiments" / "processed"
OUT = PROCESSED / "exposure.json"
TABLE = ROOT / "outputs" / "tables" / "exposure.tex"
OPEN = {"managed-gpt-oss-120b", "managed-llama-3-3-70b-instruct", "vllm-llama-3-3-70b-instruct",
        "vllm-qwen2-5-72b-instruct", "managed-mistral-small-3-1-24b-2503", "vllm-granite-3-3-8b-instruct"}
MAIN = {"w20", "w20_lp", "w20_exact_join40", "w20_lp_exact_join40"}
TOLERANCE = 0.15
# (label, plan classes, which items decide the answer, prediction used)
RULES = [
    ("Full output (map, set, join)", {"vector", "set", "pairs"}, "all $n$ items", "closed"),
    ("\\texttt{LIMIT} $k$", {"limit5", "pairs_limit10"}, "prefix up to the $k$-th match", "simulated"),
    ("\\texttt{COUNT}, ratio, counts", {"count", "ratio", "label_counts"}, "spread of the count", "simulated"),
    ("\\texttt{AVG}", {"mean_of_attribute"}, "spread of the mean", "simulated"),
    ("Rank by group mean", {"group_mean_rank"}, "groups near a tie", "simulated"),
]


def main() -> None:
    cells = [row for row in csv.DictReader((PROCESSED / "sembench_repeats" / "queries.csv").open())
             if row["model"] in OPEN and row["setting"] in MAIN and row["answer_reproduction"]]
    rows, macros = [], {}
    for label, plans, decides, kind in RULES:
        members = [row for row in cells if row["plan"] in plans]
        errors = []
        for row in members:
            measured = float(row["answer_reproduction"])
            if kind == "closed":
                x = float(row["item_pair_disagreement"]) * int(row["items"])
                predicted = math.exp(-x)
            else:
                predicted = float(row["answer_reproduction_independent"])
            errors.append(abs(predicted - measured))
        within = sum(error <= TOLERANCE for error in errors)
        rows.append({"rule": label, "decides": decides, "prediction": kind, "cells": len(members),
                     "within": within, "mean_error": sum(errors) / len(errors) if errors else None})
    uda = json.loads((PROCESSED / "uda_lineage.json").read_text())["summary"]
    within = sum(values["within"] for values in uda.values())
    total = sum(values["queries"] for values in uda.values())
    rows.append({"rule": "SQL over extractions", "decides": "values the SQL reads",
                 "prediction": "closed", "cells": total, "within": within, "mean_error": None})
    for index, row in enumerate(rows):
        name = "ABCDEF"[index]
        macros[f"expoCells{name}"] = row["cells"]
        macros[f"expoWithin{name}"] = row["within"]
    OUT.write_text(json.dumps({"rules": rows, "macros": macros}, indent=1) + "\n")

    lines = [
        "\\begin{table}[t]",
        "\\caption{Which items decide the answer under each plan operator, and how often the possible-worlds prediction "
        "is within 0.15 of measured reproduction: the closed form $e^{-x}$ (C) or the simulation through the plan (S). "
        "SemBench rows count query--deployment cells; the last row counts UDA-Bench query--deployment pairs.}",
        "\\label{tab:exposure}",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}llcr@{}}",
        "\\toprule",
        "Plan operator & Items deciding the answer & & Within \\\\",
        "\\midrule",
    ]
    for row in rows:
        mark = "C" if row["prediction"] == "closed" else "S"
        lines.append(f"{row['rule']} & {row['decides']} & {mark} & {row['within']}/{row['cells']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    TABLE.write_text("\n".join(lines))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
