#!/usr/bin/env python3
"""Tables, figures, and quoted facts from experiments/processed/sembench_repeats/.

Reads queries.csv and items.csv written by analyze_sembench_repeats.py.
Writes outputs/tables/repeats_*.tex, outputs/figures/repeats_*.pdf, and
experiments/processed/sembench_repeats/facts.json. No model is called.
"""

from __future__ import annotations

import bisect
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
FIGURES = ROOT / "outputs" / "figures"

MODELS = {
    "managed-gpt-oss-120b": "gpt-oss-120B",
    "managed-llama-3-3-70b-instruct": "Llama-70B (M)",
    "vllm-llama-3-3-70b-instruct": "Llama-70B (V)",
    "vllm-qwen2-5-72b-instruct": "Qwen-72B",
    "managed-mistral-small-3-1-24b-2503": "Mistral-24B",
    "vllm-granite-3-3-8b-instruct": "Granite-8B",
}
MAIN_SETTING = {"w20", "w20_lp"}
COMMON_K = 5
BOOTSTRAP_SEED = 11
QUERY_NAMES = {
    ("movie", 1): "Movie Q1 filter, LIMIT 5",
    ("movie", 2): "Movie Q2 filter, LIMIT 5",
    ("movie", 3): "Movie Q3 filter, COUNT",
    ("movie", 4): "Movie Q4 filter, ratio",
    ("movie", 8): "Movie Q8 label, GROUP BY",
    ("movie", 9): "Movie Q9 score per review",
    ("movie", 10): "Movie Q10 score, AVG, rank",
    ("medical", 1): "Medical Q1 filter, set",
    ("medical", 4): "Medical Q4 filter, AVG(age)",
    ("medical", 10): "Medical Q10 24-way extract",
}


# Shorter labels for the wide repeats table; the operator of each query is in the queries table.
TABLE_NAMES = {
    ("movie", 1): "Movie Q1 (LIMIT)",
    ("movie", 2): "Movie Q2 (LIMIT)",
    ("movie", 3): "Movie Q3 (COUNT)",
    ("movie", 4): "Movie Q4 (ratio)",
    ("movie", 8): "Movie Q8 (counts)",
    ("movie", 9): "Movie Q9 (scores)",
    ("movie", 10): "Movie Q10 (rank)",
    ("medical", 1): "Medical Q1 (set)",
    ("medical", 4): "Medical Q4 (AVG)",
    ("medical", 10): "Medical Q10 (labels)",
}

ANSWERS = {
    ("movie", 1): "first 5 positive reviews",
    ("movie", 2): "first 5 qualifying reviews",
    ("movie", 3): "count of positive reviews",
    ("movie", 4): "share of positive reviews",
    ("movie", 8): "reviews per sentiment",
    ("movie", 9): "score per review",
    ("movie", 10): "movies ranked by mean score",
    ("medical", 1): "patients with allergy symptoms",
    ("medical", 4): "mean age of patients with acne",
    ("medical", 10): "disease per symptom text",
}
JOIN_ANSWERS = {
    5: "first 10 same-sentiment pairs",
    6: "first 10 opposite-sentiment pairs",
    7: "all opposite-sentiment pairs",
}


def most_common(values: list[str]) -> str:
    """Most frequent value; ties go to the smallest value, so the result does not depend on run order."""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return min(counts, key=lambda value: (-counts[value], value))


def subset_flip(values: list, m: int) -> float:
    """Probability that m of the recorded runs, drawn without replacement, are not unanimous."""
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return 1.0 - sum(math.comb(n, m) for n in counts.values()) / math.comb(len(values), m)


def combinations_of(values: list) -> list[tuple]:
    return [(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values))]


def read(name: str) -> list[dict]:
    path = DATA / f"{name}.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def auc(scores_positive: list[float], scores_negative: list[float]) -> float:
    """Probability that a random flipped item has a lower margin than a random stable one."""
    if not scores_positive or not scores_negative:
        return math.nan
    negative = sorted(scores_negative)
    wins = 0.0
    for p in scores_positive:
        above = len(negative) - bisect.bisect_right(negative, p)
        ties = bisect.bisect_right(negative, p) - bisect.bisect_left(negative, p)
        wins += above + 0.5 * ties
    return wins / (len(scores_positive) * len(negative))


def auc_interval(positive: list[float], negative: list[float], resamples: int = 2000) -> list[float]:
    """Percentile bootstrap over items, resampling flipped and stable items separately."""
    if len(positive) < 2 or len(negative) < 2:
        return [math.nan, math.nan]
    rng = random.Random(BOOTSTRAP_SEED)
    values = sorted(
        auc([rng.choice(positive) for _ in positive], [rng.choice(negative) for _ in negative])
        for _ in range(resamples)
    )
    return [values[int(0.025 * resamples)], values[int(0.975 * resamples) - 1]]


def pct(value: float) -> str:
    return f"{100 * value:.1f}" if value == value else "--"


def main() -> None:
    queries = [row for row in read("queries") if row["setting"] in MAIN_SETTING and row["model"] in MODELS]
    items = [row for row in read("items") if row["setting"] in MAIN_SETTING and row["model"] in MODELS]
    if not queries:
        print("no data")
        return
    cell = {(row["scenario"], int(row["query_id"]), row["model"]): row for row in queries}
    keys = [key for key in QUERY_NAMES if any((key[0], key[1], model) in cell for model in MODELS)]
    models = [model for model in MODELS if any((key[0], key[1], model) in cell for key in keys)]

    def short(value: float) -> str:
        """Two decimals with the leading zero, the format of every fraction in the paper."""
        return f"{value:.2f}"

    def flip_text(value: float) -> str:
        """Flip rate in percent with one decimal; an exact zero prints as 0, so it is not mistaken for a rounded rate."""
        return "0" if value == 0 else pct(value)

    # Spare page width goes between deployments, so that each one's three columns stay together.
    group = "@{\\extracolsep{\\fill}}r@{\\extracolsep{0pt}\\hspace{6pt}}r@{\\hspace{6pt}}r"
    lines = [
        "% Generated by scripts/report_sembench_repeats.py. Do not edit by hand.",
        "\\begin{table*}[t]",
        "\\caption{Repeated SemBench executions at temperature 0. F: item flip rate (\\%), the share of items whose "
        "output is not identical in five executions (for gpt-oss-120B, the expected rate over five of its ten; "
        "0 is an exact zero). R: answer reproduction, the share of execution pairs that return an identical answer "
        "(ten pairs, so steps of 0.1; 45 pairs for gpt-oss-120B). "
        "$\\Delta$Q: range of SemBench's quality metric across executions. Shaded: the answer changes but the "
        "quality does not.}",
        "\\label{tab:flip}",
        "\\label{tab:repro}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{1.9pt}",
        "\\begin{tabular*}{\\linewidth}{@{}l" + group * len(models) + "@{}}",
        "\\toprule",
        "& " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{MODELS[model]}}}" for model in models) + " \\\\",
        "".join(f"\\cmidrule(lr){{{2 + 3 * index}-{4 + 3 * index}}}" for index in range(len(models))),
        "Query & " + " & ".join(["F", "R", "$\\Delta$Q"] * len(models)) + " \\\\",
        "\\midrule",
    ]
    for key in keys:
        values = []
        for model in models:
            row = cell.get((key[0], key[1], model))
            if not row:
                values += ["--"] * 3
                continue
            quality = json.loads(row["quality_values"]) if row["quality_values"] else []
            spread = max(quality) - min(quality) if quality else math.nan
            reproduction = float(row["answer_reproduction"])
            hidden = reproduction < 1 and spread == spread and spread < 0.005
            r_text = short(reproduction)
            values += [
                flip_text(float(row["item_flip_rate_k5"])),
                f"\\cellcolor{{black!12}}{r_text}" if hidden else r_text,
                "--" if spread != spread else short(spread),
            ]
        lines.append(f"{TABLE_NAMES[key]} & " + " & ".join(values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table*}", ""]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "repeats.tex").write_text("\n".join(lines), encoding="utf-8")

    facts: dict = {"models": {}, "queries": len(keys)}
    for model in models:
        model_items = [row for row in items if row["model"] == model]
        expected_flips = sum(float(row["flip_k5"]) for row in model_items)
        low, high = wilson(round(expected_flips), len(model_items))
        # The margin of the first execution predicts whether the item changes in the later executions,
        # so the outcome does not include the execution the margin came from.
        later_flip = {id(row): len({str(v) for v in json.loads(row["values"])[1:]}) > 1 for row in model_items}
        with_margin = [row for row in model_items if row["first_margin"] not in ("", "inf")]
        flipped_margin = [float(row["first_margin"]) for row in with_margin if later_flip[id(row)]]
        stable_margin = [float(row["first_margin"]) for row in with_margin if not later_flip[id(row)]]
        model_queries = [row for row in queries if row["model"] == model]
        facts["models"][model] = {
            "label": MODELS[model],
            "items": len(model_items),
            "repeats": max(int(row["repeats"]) for row in model_queries),
            "flipped": round(expected_flips),
            "flip_rate": expected_flips / len(model_items) if model_items else math.nan,
            "flip_ci": [low, high],
            "flip_rate_all_runs": sum(int(row["flipped"]) for row in model_items) / len(model_items),
            "flip_rate_macro": statistics.fmean(float(row["item_flip_rate_k5"]) for row in model_queries),
            "pair_disagreement": statistics.fmean(float(row["pair_disagreement"]) for row in model_items),
            "queries_with_any_flip": sum(float(row["item_flip_rate"]) > 0 for row in model_queries),
            "queries": len(model_queries),
            "mean_answer_reproduction": statistics.fmean(float(row["answer_reproduction"]) for row in model_queries),
            "queries_not_reproduced": sum(float(row["answer_reproduction"]) < 1 for row in model_queries),
            "margin_auc": auc(flipped_margin, stable_margin),
            "margin_auc_ci": auc_interval(flipped_margin, stable_margin),
            "margin_flipped": len(flipped_margin),
            "margin_items": len(flipped_margin) + len(stable_margin),
        }
        reasoning_flipped = [float(row["mean_reasoning_chars"]) for row in model_items if row["flipped"] == "1"]
        reasoning_stable = [float(row["mean_reasoning_chars"]) for row in model_items if row["flipped"] == "0"]
        if reasoning_flipped and any(reasoning_stable):
            facts["models"][model]["reasoning_chars_flipped"] = statistics.fmean(reasoning_flipped)
            facts["models"][model]["reasoning_chars_stable"] = statistics.fmean(reasoning_stable)
            # Longer first-execution reasoning as a predictor of later changes, pooled and
            # within each query, since reasoning length differs between operators.
            first = {id(row): -float(row["first_reasoning_chars"]) for row in model_items}
            flipped_first = [first[id(row)] for row in model_items if later_flip[id(row)]]
            stable_first = [first[id(row)] for row in model_items if not later_flip[id(row)]]
            within, weights = [], []
            for query_key in {(row["scenario"], row["query_id"]) for row in model_items}:
                rows = [row for row in model_items if (row["scenario"], row["query_id"]) == query_key]
                positive = [first[id(row)] for row in rows if later_flip[id(row)]]
                negative = [first[id(row)] for row in rows if not later_flip[id(row)]]
                if len(positive) >= 10 and len(negative) >= 10:
                    within.append(auc(positive, negative))
                    weights.append(len(positive))
            facts["models"][model].update({
                "reasoning_auc": auc(flipped_first, stable_first),
                "reasoning_auc_ci": auc_interval(flipped_first, stable_first),
                "reasoning_auc_within": sum(a * w for a, w in zip(within, weights)) / sum(weights)
                if weights else math.nan,
                "reasoning_auc_within_min": min(within) if within else math.nan,
                "reasoning_auc_within_max": max(within) if within else math.nan,
                "reasoning_auc_queries": len(within),
                "reasoning_flipped": len(flipped_first),
            })

    # Items that flip in one model and the agreement of the other models on them.
    # Each item enters as flipped with weight flip_k5 and as stable with weight
    # 1 - flip_k5, so models with ten executions count like models with five.
    by_item = defaultdict(dict)
    for row in items:
        values = json.loads(row["values"])
        majority = most_common([str(value) for value in values])
        by_item[(row["scenario"], row["query_id"], row["item"])][row["model"]] = (float(row["flip_k5"]), majority)
    contested = defaultdict(float)
    for per_model in by_item.values():
        if len(per_model) < 3:
            continue
        for model, (flip_weight, _majority) in per_model.items():
            # Contested among the other deployments only, so that an item's own
            # unstable majority cannot make it look contested.
            others = {majority for name, (_f, majority) in per_model.items() if name != model}
            disagree = len(others) > 1
            contested["flipped_items"] += flip_weight
            contested["stable_items"] += 1 - flip_weight
            contested["flipped_items_cross_model_disagree"] += flip_weight * disagree
            contested["stable_items_cross_model_disagree"] += (1 - flip_weight) * disagree
    facts["cross_model"] = dict(contested)

    independence = [
        {
            "scenario": row["scenario"],
            "query": int(row["query_id"]),
            "model": row["model"],
            "measured": float(row["answer_reproduction"]),
            "independent": float(row["answer_reproduction_independent"]),
            "flip": float(row["item_flip_rate_k5"]),
        }
        for row in queries
    ]
    facts["independence"] = independence
    heldout = [
        {"query": QUERY_NAMES[(row["scenario"], int(row["query_id"]))], "model": row["model"],
         "predicted": float(row["heldout_predicted"]), "measured": float(row["heldout_measured"])}
        for row in queries if row.get("heldout_predicted") not in (None, "")
    ]
    facts["heldout"] = heldout
    if heldout:
        facts["heldout_mean_abs_error"] = statistics.fmean(abs(row["predicted"] - row["measured"]) for row in heldout)
        facts["heldout_max_abs_error"] = max(abs(row["predicted"] - row["measured"]) for row in heldout)

    # How the flip rate of the ten-execution model grows with the number of executions counted.
    ten = [json.loads(row["values"]) for row in items if row["model"] == models[0]]
    repeats = min(len(values) for values in ten)
    facts["flip_by_k"] = {
        "model": models[0],
        "rates": {m: statistics.fmean(subset_flip(values, m) for values in ten) for m in range(2, repeats + 1)},
    }
    facts["correlated_cells"] = [
        row for row in independence if row["independent"] - row["measured"] >= 0.2
    ]

    # Same weights, two serving stacks: how often do their majority outputs differ,
    # compared with how often either stack flips against itself?
    stacks = ("managed-llama-3-3-70b-instruct", "vllm-llama-3-3-70b-instruct")
    def stack_counts(selected: list[dict]) -> dict:
        return {
            "items": len(selected),
            "majority_differs": sum(per_model[stacks[0]][1] != per_model[stacks[1]][1] for per_model in selected),
            "flipped_either": sum(bool(per_model[stacks[0]][0] or per_model[stacks[1]][0]) for per_model in selected),
            "flipped_w": sum(bool(per_model[stacks[0]][0]) for per_model in selected),
            "flipped_r": sum(bool(per_model[stacks[1]][0]) for per_model in selected),
        }

    shared = {key: per_model for key, per_model in by_item.items() if all(model in per_model for model in stacks)}
    facts["stacks"] = stack_counts(list(shared.values()))
    facts["stacks"]["by_scenario"] = {
        scenario: stack_counts([per_model for key, per_model in shared.items() if key[0] == scenario])
        for scenario in sorted({key[0] for key in shared})
    }
    facts["stacks"]["queries"] = []
    for key in keys:
        left, right = cell.get((key[0], key[1], stacks[0])), cell.get((key[0], key[1], stacks[1]))
        if left and right and left["quality_values"] and right["quality_values"]:
            facts["stacks"]["queries"].append(
                {
                    "query": QUERY_NAMES[key],
                    "quality_w": json.loads(left["quality_values"]),
                    "quality_r": json.loads(right["quality_values"]),
                }
            )

    # Do flips concentrate in one execution? Share of all deviations from the
    # per-item majority that the most deviant execution accounts for.
    deviations = defaultdict(lambda: defaultdict(int))
    for row in items:
        values = [str(value) for value in json.loads(row["values"])]
        majority = most_common(values)
        for index, value in enumerate(values):
            if value != majority:
                deviations[(row["scenario"], row["query_id"], row["model"])][index] += 1
    concentration = []
    for (scenario, query, model), per_run in deviations.items():
        total = sum(per_run.values())
        row = cell.get((scenario, int(query), model))
        if total >= 10 and row:
            concentration.append(
                {
                    "scenario": scenario,
                    "query": int(query),
                    "model": model,
                    "repeats": int(row["repeats"]),
                    "deviations": total,
                    "top_run_share": max(per_run.values()) / total,
                }
            )
    facts["concentration"] = sorted(concentration, key=lambda row: -row["top_run_share"])

    # Would a benchmark that runs each deployment once order them the same way?
    reversals = {"pairs": 0, "pairs_reversed": 0, "draws": 0, "draws_reversed": 0, "per_query": []}
    for key in keys:
        runs = {}
        for model in models:
            row = cell.get((key[0], key[1], model))
            if row and row["quality_values"]:
                runs[model] = json.loads(row["quality_values"])
        query_pairs = query_reversed = 0
        for first, second in combinations_of(list(runs)):
            mean_gap = statistics.fmean(runs[first]) - statistics.fmean(runs[second])
            if abs(mean_gap) < 1e-9:
                continue
            draws = [(a, b) for a in runs[first] for b in runs[second]]
            wrong = sum((a - b) * mean_gap < 0 for a, b in draws)
            reversals["pairs"] += 1
            reversals["draws"] += len(draws)
            reversals["draws_reversed"] += wrong
            reversals["pairs_reversed"] += wrong > 0
            query_pairs += 1
            query_reversed += wrong > 0
        reversals["per_query"].append({"query": QUERY_NAMES[key], "pairs": query_pairs, "reversed": query_reversed})
    facts["model_reversals"] = reversals

    # Flip rate by scenario and by operator output kind.
    kind_of = {(row["scenario"], row["query_id"], row["model"]): row["kind"] for row in queries}
    grouped = defaultdict(lambda: [0.0, 0])
    for row in items:
        for group in (("scenario", row["scenario"], row["model"]), ("kind", kind_of[(row["scenario"], row["query_id"], row["model"])], row["model"])):
            grouped[group][0] += float(row["flip_k5"])
            grouped[group][1] += 1
    facts["groups"] = [
        {"by": by, "group": group, "model": model, "flipped": flipped, "items": total, "flip_rate": flipped / total}
        for (by, group, model), (flipped, total) in sorted(grouped.items())
    ]

    query_lines = [
        "% Generated by scripts/report_sembench_repeats.py. Do not edit by hand.",
        "\\begin{table}[t]",
        "\\caption{SemBench queries executed repeatedly. Items is the number of semantic-operator outputs analysed per execution; for the joins, on a 40-review sample, it excludes each review paired with itself.}",
        "\\label{tab:queries}",
        "\\footnotesize",
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lllr@{}}",
        "\\toprule",
        "Query & Output & Answer & Items \\\\",
        "\\midrule",
    ]
    for key in keys:
        rows = [cell[(key[0], key[1], model)] for model in models if (key[0], key[1], model) in cell]
        items_per_run = f"{max(int(row['items']) for row in rows):,}".replace(",", "{,}")
        query_lines.append(f"{key[0].capitalize()} Q{key[1]} & {rows[0]['kind']} & {ANSWERS[key]} & {items_per_run} \\\\")
    join_rows = [row for row in read("queries") if row["setting"].endswith("_exact_join40")]
    if join_rows:
        query_lines.append("\\midrule")
        calls = max(int(row["calls"]) // int(row["repeats"]) for row in join_rows)
        for query, answer in JOIN_ANSWERS.items():
            query_lines.append(f"Movie Q{query} & pair & {answer} & {calls:,} \\\\".replace(",", "{,}"))
    query_lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""]
    (TABLES / "queries.tex").write_text("\n".join(query_lines), encoding="utf-8")

    (DATA / "facts.json").write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in facts.items() if key != "independence"}, indent=1))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    markers = dict(zip(models, "osD^vP"))
    for model in models:
        rows = [row for row in independence if row["model"] == model]
        axes[0].scatter(
            [max(row["flip"], 1e-4) for row in rows],
            [row["measured"] for row in rows],
            marker=markers[model],
            s=18,
            label=MODELS[model],
            alpha=0.8,
        )
        axes[1].scatter([row["independent"] for row in rows], [row["measured"] for row in rows], marker=markers[model], s=18, alpha=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Item flip rate (log; 0 plotted at $10^{-4}$)")
    axes[0].set_ylabel("Answer reproduction")
    axes[0].legend(fontsize=6, frameon=False, loc="lower left")
    axes[1].plot([0, 1], [0, 1], color="grey", linewidth=0.8)
    axes[1].set_xlabel("Predicted if items flip independently")
    axes[1].set_ylabel("Measured reproduction")
    figure.tight_layout()
    figure.savefig(FIGURES / "repeats_reproduction.pdf")

    import plot_style

    plot_style.apply()
    kinds = list(plot_style.KINDS)
    rates = {(row["group"], row["model"]): row for row in facts["groups"] if row["by"] == "kind"}
    figure, axis = plt.subplots(figsize=(plot_style.COLUMN, 1.55))
    width = 0.8 / len(models)
    floor = 1e-2
    for index, model in enumerate(models):
        xs, ys, zeros = [], [], []
        for position, kind in enumerate(kinds):
            row = rates.get((kind, model))
            if not row:
                continue
            x = position + (index - (len(models) - 1) / 2) * width
            if row["flip_rate"] > 0:
                xs.append(x)
                ys.append(100 * row["flip_rate"])
            else:
                zeros.append(x)
        axis.bar(xs, ys, width=width, color=plot_style.color(model), label=plot_style.label(model))
        # A rate of zero has no bar on a log axis; mark it on the floor instead.
        axis.scatter(zeros, [floor * 1.25] * len(zeros), marker="o", s=7, facecolors="none",
                     edgecolors=plot_style.color(model), linewidths=0.7, clip_on=False, zorder=3)
    axis.set_yscale("log")
    axis.set_ylim(floor, 100)
    axis.set_xticks(range(len(kinds)))
    axis.set_xticklabels([plot_style.KINDS[kind][0] for kind in kinds])
    axis.set_ylabel("Item flip rate (%, log)")
    axis.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.0)
    figure.savefig(FIGURES / "repeats_by_kind.pdf")

    # Sources: margin distributions, and flip rate against cross-model dissent.
    figure, axes = plt.subplots(1, 2, figsize=(plot_style.DOUBLE, 1.8))
    noise_path = DATA / "margin_noise.json"
    noise = json.loads(noise_path.read_text()) if noise_path.exists() else {}
    bin_labels = []
    for model, fit in noise.items():
        bins = fit["bins"]
        if fit["items_changing"] < 10:
            continue
        positions = list(range(len(bins)))
        bin_labels = bin_labels if len(bin_labels) >= len(bins) else [
            f"{row['low']:g}–{row['high']:g}" if row["high"] is not None else f">{row['low']:g}" for row in bins]
        colour = plot_style.color(model)
        axes[0].plot(positions, [max(100 * row["observed"], 1e-2) for row in bins], marker=plot_style.marker(model),
                     color=colour, label=f"{fit['label']} ($\\sigma$={fit['sigma']:.2f})")
        axes[0].plot(positions, [max(100 * row["fitted"], 1e-2) for row in bins], ":", color=colour, linewidth=0.9)
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(bin_labels)))
    axes[0].set_xticklabels(bin_labels)
    axes[0].set_xlabel("First execution's decision margin (nats)")
    axes[0].set_ylabel("Later outputs that differ (%)")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.56, 1.0), ncol=3, handlelength=1.2, columnspacing=0.6, handletextpad=0.4)

    dissent_bins = defaultdict(lambda: [0.0, 0])
    for per_model in by_item.values():
        if len(per_model) < len(models):
            continue
        for model, (flip_weight, _majority) in per_model.items():
            others = [m for name, (_f, m) in per_model.items() if name != model]
            plurality = most_common(others)
            dissent = sum(m != plurality for m in others)
            dissent_bins[(model, dissent)][0] += flip_weight
            dissent_bins[(model, dissent)][1] += 1
    facts["dissent"] = [
        {"model": model, "dissent": dissent, "flipped": flipped, "items": total}
        for (model, dissent), (flipped, total) in sorted(dissent_bins.items())
    ]
    for model in models:
        points = sorted((dissent, flipped / total) for (name, dissent), (flipped, total) in dissent_bins.items() if name == model and total >= 20)
        if points:
            axes[1].plot(
                [p[0] for p in points], [max(100 * p[1], 1e-2) for p in points],
                marker=plot_style.marker(model), color=plot_style.color(model), label=plot_style.label(model),
            )
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(4))
    axes[1].set_xlabel("Other deployments that dissent")
    axes[1].set_ylabel("Item flip rate (%)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    figure.tight_layout(w_pad=1.5)
    figure.savefig(FIGURES / "repeats_sources.pdf")
    (DATA / "facts.json").write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
