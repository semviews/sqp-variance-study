"""Exposure rule per plan operator, each validated on the cells that use it (work item C3, revision R2).

For every plan class of the SemBench queries, the table states which items
decide the answer and compares the possible-worlds prediction with measured
answer reproduction over the main open-weight cells. For answers that expose
every item, the prediction is the exact product prod_i (1 - delta_i) of the
items' pairwise agreement (about e^{-x}, x = sum_i delta_i); for the other
plans, the independence simulation through the plan. A cell is informative
when its prediction lies strictly between 0.05 and 0.95; elsewhere any
monotone curve fits. With five executions, measured reproduction moves in
steps of 0.1, so the tolerance of 0.15 is one and a half steps; we also
report 0.10. The UDA-Bench row uses the lineage exposure of
scripts/uda_lineage.py, for which the product is a lower bound, and counts
the queries on which the bound holds within the tolerance.

Writes experiments/processed/exposure.json and outputs/tables/exposure.tex.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "experiments" / "processed"
OUT = PROCESSED / "exposure.json"
TABLE = ROOT / "outputs" / "tables" / "exposure.tex"
OPEN = {"managed-gpt-oss-120b", "managed-llama-3-3-70b-instruct", "vllm-llama-3-3-70b-instruct",
        "vllm-qwen2-5-72b-instruct", "managed-mistral-small-3-1-24b-2503", "vllm-granite-3-3-8b-instruct"}
MAIN = {"w20", "w20_lp", "w20_exact_join40", "w20_lp_exact_join40"}
TOLERANCE = 0.15
STRICT = 0.10
INFORMATIVE = (0.05, 0.95)
# (label, plan classes, which items decide the answer, prediction used)
RULES = [
    ("Full output (map, set, join)", {"vector", "set", "pairs"}, "all $n$ items", "closed"),
    ("\\texttt{LIMIT} $k$", {"limit5", "pairs_limit10"}, "prefix to the $k$-th match", "simulated"),
    ("\\texttt{COUNT}, ratio, counts", {"count", "ratio", "label_counts"}, "spread of the count", "simulated"),
    ("\\texttt{AVG}", {"mean_of_attribute"}, "spread of the mean", "simulated"),
    ("Rank by group mean", {"group_mean_rank"}, "groups near a tie", "simulated"),
]


def informative(predicted: float) -> bool:
    return INFORMATIVE[0] < predicted < INFORMATIVE[1]


def main() -> None:
    cells = [row for row in csv.DictReader((PROCESSED / "sembench_repeats" / "queries.csv").open())
             if row["model"] in OPEN and row["setting"] in MAIN and row["answer_reproduction"]]
    keys = {(row["scenario"], row["model"], row["setting"], row["query_id"]) for row in cells}
    product: dict[tuple, float] = defaultdict(lambda: 1.0)
    for row in csv.DictReader((PROCESSED / "sembench_repeats" / "items.csv").open()):
        key = (row["scenario"], row["model"], row["setting"], row["query_id"])
        if key in keys:
            product[key] *= 1 - float(row["pair_disagreement"])
    rows, macros, points = [], {}, []
    for label, plans, decides, kind in RULES:
        members = [row for row in cells if row["plan"] in plans]
        pairs = []
        for row in members:
            key = (row["scenario"], row["model"], row["setting"], row["query_id"])
            measured = float(row["answer_reproduction"])
            predicted = product[key] if kind == "closed" else float(row["answer_reproduction_independent"])
            pairs.append((predicted, measured))
            points.append({"cell": "/".join(key), "plan": row["plan"], "predicted": predicted, "measured": measured})
        chosen = [pair for pair in pairs if informative(pair[0])]
        rows.append({
            "rule": label, "decides": decides, "prediction": kind, "cells": len(pairs),
            "within": sum(abs(p - m) <= TOLERANCE for p, m in pairs),
            "within_strict": sum(abs(p - m) <= STRICT for p, m in pairs),
            "informative": len(chosen),
            "informative_within": sum(abs(p - m) <= TOLERANCE for p, m in chosen),
            "informative_within_strict": sum(abs(p - m) <= STRICT for p, m in chosen),
            "mean_error": sum(abs(p - m) for p, m in pairs) / len(pairs) if pairs else None,
            "informative_mean_error": sum(abs(p - m) for p, m in chosen) / len(chosen) if chosen else None,
        })
    uda = json.loads((PROCESSED / "uda_lineage.json").read_text())["summary"]
    rows.append({"rule": "SQL over extractions", "decides": "values read", "prediction": "bound",
                 "cells": sum(values["queries"] for values in uda.values()),
                 "within": sum(values["bound_holds"] for values in uda.values()),
                 "informative": sum(values["informative"] for values in uda.values()),
                 "informative_within": sum(values["informative_bound_holds"] for values in uda.values()),
                 "mean_error": None, "informative_mean_error": None})
    for index, row in enumerate(rows):
        name = "ABCDEF"[index]
        macros[f"expoCells{name}"] = row["cells"]
        macros[f"expoWithin{name}"] = row["within"]
    sembench = rows[:-1]
    chosen = [point for point in points if informative(point["predicted"])]
    macros["expoSemCells"] = sum(row["cells"] for row in sembench)
    macros["expoSemWithin"] = sum(row["within"] for row in sembench)
    macros["expoSemInformative"] = len(chosen)
    macros["expoSemInformativeWithin"] = sum(row["informative_within"] for row in sembench)
    macros["expoSemInformativeStrict"] = sum(row["informative_within_strict"] for row in sembench)
    macros["expoSemInformativeMae"] = f"{sum(abs(p['predicted'] - p['measured']) for p in chosen) / len(chosen):.2f}"
    macros["expoSemOverPredicted"] = sum(p["predicted"] > p["measured"] for p in chosen)
    macros["expoUdaHolds"] = rows[-1]["within"]
    macros["expoUdaCells"] = rows[-1]["cells"]
    macros["expoUdaInformative"] = rows[-1]["informative"]
    macros["expoUdaInformativeHolds"] = rows[-1]["informative_within"]
    OUT.write_text(json.dumps({"rules": rows, "points": points, "macros": macros}, indent=1) + "\n")

    def mae(value):
        return "--" if value is None else f"{value:.2f}"

    lines = [
        "\\begin{table}[t]",
        "\\caption{Which items decide the answer under each plan operator, and how often the possible-worlds "
        "prediction is within 0.15 of measured answer reproduction, over all cells and over the informative ones "
        "(prediction between 0.05 and 0.95). P: the exact product $\\prod_i(1-\\delta_i)$, about $e^{-x}$; "
        "S: simulation through the plan; B: the product as a lower bound, counting where the bound holds. "
        "SemBench rows count query--deployment cells of the open-weight deployments; the last row counts "
        "UDA-Bench query--deployment pairs. MAE: mean absolute error over informative cells.}",
        "\\label{tab:exposure}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}llcrrr@{}}",
        "\\toprule",
        "Plan operator & Deciding items & & All & Inform. & MAE \\\\",
        "\\midrule",
    ]
    for row in rows:
        mark = {"closed": "P", "simulated": "S", "bound": "B"}[row["prediction"]]
        if row["prediction"] == "bound":
            lines.append("\\midrule")
        lines.append(f"{row['rule']} & {row['decides']} & {mark} & {row['within']}/{row['cells']} & "
                     f"{row['informative_within']}/{row['informative']} & {mae(row['informative_mean_error'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    TABLE.write_text("\n".join(lines))
    for row in rows:
        print(row)
    print(macros)


if __name__ == "__main__":
    main()
