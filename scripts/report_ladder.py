#!/usr/bin/env python3
"""Reproducibility at three levels across SemBench, LRO-Bench, and UDA-Bench.

For each deployment and benchmark, the share of scored queries whose
operator outputs, final answer, and benchmark quality are identical over five
executions (for ten SemBench executions, the exact expectation over subsets of
five). Writes outputs/tables/ladder.tex and experiments/processed/ladder.json.
No model is called.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "experiments" / "processed"
TABLES = ROOT / "outputs" / "tables"
NAMES = {
    "managed-gpt-oss-120b": "gpt-oss-120B (M)",
    "managed-llama-3-3-70b-instruct": "Llama-70B (M)",
    "vllm-llama-3-3-70b-instruct": "Llama-70B (V)",
    "vllm-qwen2-5-72b-instruct": "Qwen-72B (V)",
    "managed-mistral-small-3-1-24b-2503": "Mistral-24B (M)",
    "vllm-granite-3-3-8b-instruct": "Granite-8B (V)",
}
SEMBENCH_SETTINGS = {"w20", "w20_lp", "w20_exact_join40", "w20_lp_exact_join40"}
LEVELS = ("outputs", "answer", "quality")


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def share(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def sembench() -> dict:
    rows = [row for row in read(PROCESSED / "sembench_repeats" / "queries.csv")
            if row["setting"] in SEMBENCH_SETTINGS]
    result = {}
    for model in NAMES:
        mine = [row for row in rows if row["model"] == model and row["quality_identical_k5"] != ""]
        result[model] = {
            "queries": len(mine),
            "outputs": share([float(row["outputs_identical_k5"]) for row in mine]),
            "answer": share([float(row["answer_identical_k5"]) for row in mine]),
            "quality": share([float(row["quality_identical_k5"]) for row in mine]),
        }
    return result


def lrobench() -> dict:
    rows = [row for row in read(PROCESSED / "lrobench_repeats" / "queries.csv") if int(row["repeats"]) >= 5]
    result = {}
    for model in NAMES:
        scored = [row for row in rows if row["model"] == model and int(row["ok"]) >= 2 and row["quality_min"] != ""]
        if not scored:
            continue
        compared = [row for row in scored if int(row["calls_compared"]) > 0]
        result[model] = {
            "queries": len(scored),
            "outputs": share([float(row["calls_text_differs"] == "0") for row in compared]),
            "answer": share([float(row["distinct_answers"] == "1") for row in scored]),
            "quality": share([float(abs(float(row["quality_max"]) - float(row["quality_min"])) < 1e-9)
                              for row in scored]),
        }
    return result


def udabench() -> dict:
    rows = read(PROCESSED / "uda_repeats" / "queries.csv")
    lineage = json.loads((PROCESSED / "uda_lineage.json").read_text())["points"]
    exposure = {(point["model"], point["query"]): point["exposure"] for point in lineage}
    result = {}
    for model in NAMES:
        mine = [row for row in rows if row["model"] == model]
        if not mine:
            continue
        scored = [row for row in mine if row["gold_empty"] == "0" and row["f1_min"] != ""]
        result[model] = {
            "queries": len(scored),
            "outputs": share([float(exposure[(model, row["query"])] == 0) for row in scored]),
            "answer": share([float(row["distinct_answers"] == "1") for row in scored]),
            "quality": share([float(abs(float(row["f1_max"]) - float(row["f1_min"])) < 1e-9) for row in scored]),
        }
    return result


def plot(suites: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {"sembench": "SemBench", "lrobench": "LRO-Bench", "udabench": "UDA-Bench"}
    markers = dict(zip(NAMES, "osD^vP"))
    fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.9), sharey=True)
    for axis, (suite, title) in zip(axes, titles.items()):
        for model, name in NAMES.items():
            entry = suites[suite].get(model)
            if not entry:
                continue
            values = [100 * entry[level] for level in LEVELS]
            axis.plot(range(3), values, marker=markers[model], color="0.25", linewidth=0.9, markersize=4,
                      label=name)
        axis.set_xticks(range(3), ["Out.", "Ans.", "Qual."])
        axis.set_xlim(-0.3, 2.3)
        axis.set_title(title, fontsize=8)
        axis.set_ylim(-3, 103)
        axis.grid(axis="y", linewidth=0.3)
        axis.tick_params(labelsize=7)
    axes[0].set_ylabel("Identical (% of queries)", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.17))
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "figures" / "ladder.pdf", bbox_inches="tight")


def cell(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.0f}"


def main() -> None:
    suites = {"sembench": sembench(), "lrobench": lrobench(), "udabench": udabench()}
    columns = [("sembench", LEVELS), ("lrobench", LEVELS), ("udabench", LEVELS)]
    lines = ["% Generated by scripts/report_ladder.py. Do not edit by hand.", "\\begin{table}[tb]",
             "\\caption{Share of queries (\\%) whose operator outputs (Out.), final answer (Ans.), and benchmark "
             "quality (Qual.) are identical over five executions; for gpt-oss-120B on SemBench, the expectation "
             "over five of its ten. All levels count the same queries: those with a quality score (on LRO-Bench, "
             "answered in at least two executions). LRO-Bench compares raw replies; on UDA-Bench, the outputs "
             "are the extracted values of the attributes the query's SQL reads.}",
             "\\label{tab:ladder}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.6pt}",
             "\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule",
             "& \\multicolumn{3}{c}{SemBench} & \\multicolumn{3}{c}{LRO-Bench} & \\multicolumn{3}{c}{UDA-Bench} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-10}",
             "Deployment & Out. & Ans. & Qual. & Out. & Ans. & Qual. & Out. & Ans. & Qual. \\\\", "\\midrule"]
    for model, name in NAMES.items():
        cells = []
        for suite, levels in columns:
            entry = suites[suite].get(model)
            cells += [cell(entry[level] if entry else None) for level in levels]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    pooled = {}
    cells = []
    for suite, levels in columns:
        pooled[suite] = {}
        for level in levels:
            values = [entry[level] for entry in suites[suite].values() if entry[level] is not None]
            pooled[suite][level] = statistics.fmean(values) if values else None
            cells.append(cell(pooled[suite][level]))
    lines += ["Mean & " + " & ".join(cells) + " \\\\", "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLES / "ladder.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot(suites)
    (PROCESSED / "ladder.json").write_text(json.dumps({"suites": suites, "pooled": pooled}, indent=2) + "\n")
    print(json.dumps(pooled, indent=2))


if __name__ == "__main__":
    main()
