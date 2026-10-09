"""Commercial models on the SemBench filters and maps.

Applies the main matrix's definitions (expected item flip rate over five
executions, pairwise disagreement, answer reproduction) to GPT-4o and
Claude Sonnet 5, run through the enterprise endpoint on the same ten queries,
and lists the open-weight range next to them. Cost comes from the call logs.

Writes experiments/processed/commercial.json and outputs/tables/commercial.tex.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from analyze_sembench_repeats import wilson
from report_sembench_repeats import MODELS, QUERY_NAMES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "processed" / "sembench_repeats"
RAW = ROOT / "experiments" / "raw" / "sembench-repeats"
OUT = ROOT / "experiments" / "processed" / "commercial.json"
TABLE = ROOT / "outputs" / "tables" / "commercial.tex"
COMMERCIAL = {"Azure-gpt-4o": ("GPT-4o", "GptFour"), "aws-claude-sonnet-5": ("Claude Sonnet 5", "Sonnet")}
SETTINGS = {"w20", "w20_lp"}
MIN_REPEATS = 5
MAX_ATTEMPTS = 3


def read(name: str) -> list[dict]:
    with (DATA / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "\\pending{}"
    return f"{100 * value:.{2 if 0 < value < 0.01 else digits}f}\\%"


def cost(model: str) -> float:
    total = 0.0
    for path in RAW.glob(f"*/{model}/*/repeat-*/*.calls.jsonl"):
        for line in path.read_text().splitlines():
            total += float(json.loads(line).get("cost") or 0.0)
    for path in RAW.glob(f"*/{model}/*/repeat-*/*.failures.jsonl"):
        for line in path.read_text().splitlines():
            total += float(json.loads(line).get("spend") or 0.0)
    return total


def policy_rejected(model: str, key: tuple) -> bool:
    """Every attempt of some execution was rejected by the provider's content policy (see sembench_repeats.py)."""
    for path in RAW.glob(f"{key[0]}/{model}/*/repeat-*/Q{key[1]}.failures.jsonl"):
        lines = path.read_text().splitlines()
        if len(lines) >= MAX_ATTEMPTS and all("ContentPolicyViolation" in line for line in lines):
            return True
    return False


def series(names: list[str]) -> str:
    """Movie Q1, Medical Q1, Q4, and Q10."""
    parts, last = [], None
    for name in names:
        scenario, query = name.split(" ")
        parts.append(query if scenario == last else name)
        last = scenario
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + (", and " if len(parts) > 2 else " and ") + parts[-1]


def main() -> None:
    queries = [row for row in read("queries") if row["setting"] in SETTINGS]
    all_keys = [key for key in QUERY_NAMES if not (key[0] == "movie" and key[1] in (5, 6, 7))]
    rejected = {key: [COMMERCIAL[m][0] for m in COMMERCIAL if policy_rejected(m, key)] for key in all_keys}
    rejected = {key: who for key, who in rejected.items() if who}
    keys = [key for key in all_keys if key not in rejected]
    items = [row for row in read("items") if row["setting"] in SETTINGS
             and (row["scenario"], int(row["query_id"])) in set(keys)]
    result, macros = {"models": {}, "open_range": {}, "rejected": {f"{k[0].capitalize()} Q{k[1]}": v for k, v in rejected.items()}}, {}

    def cell(model: str, key: tuple) -> dict | None:
        rows = [row for row in queries if row["model"] == model and (row["scenario"], int(row["query_id"])) == key]
        return rows[0] if rows and int(rows[0]["repeats"]) >= MIN_REPEATS else None

    for model, (label, suffix) in COMMERCIAL.items():
        complete = [key for key in keys if cell(model, key)]
        mine = [row for row in items if row["model"] == model
                and (row["scenario"], int(row["query_id"])) in set(complete)]
        entry = {"label": label, "queries": len(complete), "cost": cost(model), "per_query": {}}
        for key in complete:
            row = cell(model, key)
            entry["per_query"][QUERY_NAMES[key]] = {"flip_k5": float(row["item_flip_rate_k5"]),
                                                   "reproduction": float(row["answer_reproduction"])}
        if mine and len(complete) == len(keys):
            expected = sum(float(row["flip_k5"]) for row in mine)
            low, high = wilson(round(expected), len(mine))
            entry.update(flip_rate=expected / len(mine), flip_ci=[low, high],
                         disagreement=statistics.fmean(float(row["pair_disagreement"]) for row in mine),
                         not_reproduced=sum(entry["per_query"][QUERY_NAMES[k]]["reproduction"] < 1 for k in complete))
        result["models"][model] = entry
        macros[f"flip{suffix}"] = pct(entry.get("flip_rate"))
        macros[f"flipCi{suffix}"] = (f"[{pct(entry['flip_ci'][0], 2)}, {pct(entry['flip_ci'][1], 2)}]"
                                     if "flip_ci" in entry else "\\pending{}")
        macros[f"disagree{suffix}"] = pct(entry.get("disagreement"), 2)
        macros[f"notReproduced{suffix}"] = entry.get("not_reproduced", "\\pending{}")
        macros[f"queries{suffix}"] = len(keys)
        macros[f"cost{suffix}"] = f"{entry['cost']:.2f}"

    open_rates = {}
    for model in MODELS:
        mine = [row for row in items if row["model"] == model]
        if mine:
            open_rates[model] = sum(float(row["flip_k5"]) for row in mine) / len(mine)
    result["open_range"] = open_rates
    macros["commercialCost"] = f"{sum(e['cost'] for e in result['models'].values()):.2f}"
    macros["commercialExcluded"] = series(list(result["rejected"])) if result["rejected"] else "none"
    OUT.write_text(json.dumps({**result, "macros": macros}, indent=1) + "\n")

    labels = [COMMERCIAL[m][0] for m in COMMERCIAL]
    kind_of = {(row["scenario"], row["query_id"]): row["kind"] for row in queries}
    kinds = [("bool", "Boolean filters"), ("label", "Two-way label"), ("score", "1--5 scores"), ("extract", "24-way extraction")]

    def rate(model: str, kind: str | None) -> float | None:
        rows = [row for row in items if row["model"] == model
                and (kind is None or kind_of.get((row["scenario"], row["query_id"])) == kind)]
        if model in COMMERCIAL and result["models"][model]["queries"] < len(keys):
            return None
        return sum(float(row["flip_k5"]) for row in rows) / len(rows) if rows else None

    by_filter = {}
    for name, who in result["rejected"].items():
        by_filter.setdefault(" and ".join(who), []).append(name)
    excluded = "".join(
        f" {series(names)} {'is' if len(names) == 1 else 'are'} excluded for all deployments:"
        f" the content filter of {who} rejected every attempt."
        for who, names in by_filter.items())
    lines = [
        "% Generated by scripts/commercial_report.py. Do not edit by hand.",
        "\\begin{table}[t]",
        "\\caption{Item flip rate over five executions, in percent, of the commercial models on the SemBench filters and "
        "maps, by output type, next to the lowest and highest rate among the six open-weight deployments." + excluded + "}",
        "\\label{tab:commercial}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}l" + "r" * len(labels) + "r@{}}",
        "\\toprule",
        "Output & " + " & ".join(labels) + " & Open-weight \\\\",
        "\\midrule",
    ]
    for kind, label in kinds + [(None, "Pooled")]:
        if kind is None:
            lines.append("\\midrule")
        opens = [r for r in (rate(model, kind) for model in MODELS) if r is not None]
        if not opens:
            continue
        values = [pct(rate(model, kind), 2) for model in COMMERCIAL]
        span = f"{100 * min(opens):.2f}--{100 * max(opens):.1f}" if opens else "--"
        lines.append(f"{label} & " + " & ".join(values) + f" & {span} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    TABLE.write_text("\n".join(lines).replace("\\%", ""))
    print(json.dumps({m: {k: v for k, v in e.items() if k != "per_query"} for m, e in result["models"].items()}, indent=1))


if __name__ == "__main__":
    main()
