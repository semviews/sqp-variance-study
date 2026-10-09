"""Test LRO-Bench's published implementation recommendations on repeated executions.

Claims come from the key take-aways of the LRO-Bench paper (Section 5, one
per operator and operand type). A claim "A beats B" is tested per
deployment on the queries of that operand type:

- mean: per query, the mean quality of the successful executions of each
  implementation; the difference A - B averaged over queries, with a
  percentile bootstrap over queries (10,000 resamples, seed 11);
- per execution: the same difference using only execution r of each
  implementation, for every r both have.

Verdicts: "every execution" (interval above 0 and every execution agrees),
"mean only" (interval above 0, some execution disagrees), "unstable"
(interval contains 0), "reversed" (interval below 0).

Writes experiments/processed/claims.json and outputs/tables/claims.tex.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path

from analyze_lrobench_repeats import quality_of

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "experiments" / "raw" / "lrobench"
LRO = Path(os.environ.get("LROBENCH", Path.home() / "code" / "LROBench"))
OUT = ROOT / "experiments" / "processed" / "claims.json"
TABLE = ROOT / "outputs" / "tables" / "claims.tex"
RESAMPLES, SEED = 10_000, 11
DEPLOYMENTS = [("managed-gpt-oss-120b", "gpt-oss"), ("managed-llama-3-3-70b-instruct", "Llama (M)"),
               ("vllm-llama-3-3-70b-instruct", "Llama (V)"), ("vllm-granite-3-3-8b-instruct", "Granite")]
# (operator, operand types, better, worse, claim as worded in the table)
CLAIMS = [
    ("select", {"row"}, "llm_one", "llm_all", "Filter, rows: ONE $>$ ALL"),
    ("select", {"column", "table"}, "llm_all", "llm_one", "Filter, columns, tables: ALL $>$ ONE"),
    ("match", {"row"}, "llm_semi", "llm_all", "Match, rows: SEMI $\\geq$ ALL"),
    ("match", {"cell", "column"}, "llm_all", "llm_semi", "Match, cells, columns: ALL $>$ SEMI"),
    ("impute", {"column"}, "llm_one", "llm_all", "Impute, columns: ONE $>$ ALL"),
    ("impute", {"cell", "row"}, "llm_all", "llm_one", "Impute, cells, rows: ALL $>$ ONE"),
    ("cluster", None, "llm_all", "llm_one", "Cluster: ALL $>$ ONE"),
    ("order", None, "llm_all", "llm_semi", "Order: ALL $>$ SEMI"),
    ("order", None, "llm_all", "llm_one-heap", "Order: ALL $>$ ONE (heap)"),
]
SYMBOL = {"every execution": "$\\checkmark$", "mean only": "mean", "unstable": "$\\sim$",
          "reversed": "$\\times$", "too few": "--"}


def operand_types() -> dict[tuple[str, str], str]:
    types = {}
    for operator in ("select", "match", "impute", "cluster", "order"):
        for query in json.loads((LRO / f"{operator}_metadata.json").read_text())["queries"]:
            key = str(query["query_id"]).removeprefix(operator)
            types[(operator, key)] = query["attributes"].get("operand_type", "")
    return types


def qualities(model: str, setting: str) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = defaultdict(dict)
    for path in (RAW / model / setting).glob("repeat-*/*.json"):
        if path.name.endswith(".failed.json"):
            continue
        record = json.loads(path.read_text())
        quality = None if record.get("error") else quality_of(record)
        if quality is not None:
            result[path.name.split(".")[0]][int(path.parent.name.split("-")[1])] = quality
    return result


def bootstrap(differences: list[float]) -> tuple[float, float]:
    rng = random.Random(SEED)
    means = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(RESAMPLES))
    return means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]


def test(better: dict, worse: dict, queries: set[str]) -> dict:
    shared = sorted(query for query in queries if better.get(query) and worse.get(query))
    if len(shared) < 3:
        return {"queries": len(shared), "verdict": "too few"}
    differences = [sum(better[q].values()) / len(better[q]) - sum(worse[q].values()) / len(worse[q]) for q in shared]
    low, high = bootstrap(differences)
    per_execution = []
    for repeat in sorted({r for q in shared for r in better[q]} & {r for q in shared for r in worse[q]}):
        pairs = [better[q][repeat] - worse[q][repeat] for q in shared if repeat in better[q] and repeat in worse[q]]
        if pairs:
            per_execution.append(sum(pairs) / len(pairs))
    agree = sum(value > 0 for value in per_execution)
    if low > 0:
        verdict = "every execution" if agree == len(per_execution) else "mean only"
    else:
        verdict = "reversed" if high < 0 else "unstable"
    return {"queries": len(shared), "mean": sum(differences) / len(differences), "ci": [low, high],
            "executions": len(per_execution), "agree": agree, "verdict": verdict}


def main() -> None:
    types = operand_types()
    rows = []
    for operator, operands, better, worse, text in CLAIMS:
        row = {"claim": text, "operator": operator, "better": better, "worse": worse, "results": {}}
        for model, _ in DEPLOYMENTS:
            good, bad = qualities(model, f"{operator}_{better}"), qualities(model, f"{operator}_{worse}")
            queries = {q for q in set(good) | set(bad) if operands is None or types.get((operator, q)) in operands}
            row["results"][model] = test(good, bad, queries)
        rows.append(row)

    verdicts = [result["verdict"] for row in rows for result in row["results"].values() if result["verdict"] != "too few"]
    macros = {
        "claimCount": len(rows),
        "claimTests": len(verdicts),
        "claimEvery": verdicts.count("every execution"),
        "claimMeanOnly": verdicts.count("mean only"),
        "claimUnstable": verdicts.count("unstable"),
        "claimReversed": verdicts.count("reversed"),
        "claimSplit": sum(0 < r["agree"] < r["executions"] for row in rows for r in row["results"].values()
                          if r["verdict"] != "too few"),
        "claimHoldAll": sum(all(r["verdict"] == "every execution" for r in row["results"].values()
                                if r["verdict"] != "too few") for row in rows),
    }
    OUT.write_text(json.dumps({"claims": rows, "macros": macros}, indent=1) + "\n")

    lines = [
        "\\begin{table}[t]",
        "\\caption{LRO-Bench's implementation recommendations on our executions. $\\checkmark$: the 95\\% interval of the "
        "mean quality difference over queries is above zero and every execution agrees; $\\sim$: the interval contains zero; "
        "--: fewer than three queries.}",
        "\\label{tab:claims}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{@{}l" + "c" * len(DEPLOYMENTS) + "@{}}",
        "\\toprule",
        "Recommendation & " + " & ".join(label for _, label in DEPLOYMENTS) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [SYMBOL[row["results"][model]["verdict"]] for model, _ in DEPLOYMENTS]
        lines.append(f"{row['claim']} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    TABLE.write_text("\n".join(lines))
    for row in rows:
        print(row["claim"])
        for model, result in row["results"].items():
            print(f"   {model:34s} {result}")
    print(macros)


if __name__ == "__main__":
    main()
