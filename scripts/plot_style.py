"""Shared style for the paper's figures: one color and marker per deployment, one font size.

Colors are the Okabe-Ito palette, which stays distinguishable for common
color-vision deficiencies. Import `apply()` before creating a figure and use
`DEPLOYMENTS` for every per-deployment series, so a deployment looks the
same in every figure.
"""

from __future__ import annotations

COLUMN = 3.5  # inches, one column of the IEEE conference template
DOUBLE = 7.0  # inches, both columns
FONT = 7
SMALL = 6

# model id -> (label, color, marker)
DEPLOYMENTS = {
    "managed-gpt-oss-120b": ("gpt-oss-120B", "#0072B2", "o"),
    "managed-llama-3-3-70b-instruct": ("Llama-70B (M)", "#E69F00", "s"),
    "vllm-llama-3-3-70b-instruct": ("Llama-70B (V)", "#009E73", "D"),
    "vllm-qwen2-5-72b-instruct": ("Qwen-72B", "#D55E00", "^"),
    "managed-mistral-small-3-1-24b-2503": ("Mistral-24B", "#CC79A7", "v"),
    "vllm-granite-3-3-8b-instruct": ("Granite-8B", "#56B4E9", "P"),
    "Azure-gpt-4o": ("GPT-4o", "#000000", "X"),
    "aws-claude-sonnet-5": ("Claude Sonnet 5", "#7F7F7F", "*"),
}
# Output types of SemBench operators, in increasing order of their flip rate.
KINDS = {
    "bool": ("Boolean", "#0072B2"),
    "label": ("2-way label", "#E69F00"),
    "score": ("1–5 score", "#009E73"),
    "extract": ("24-way extract", "#D55E00"),
}


def label(model: str) -> str:
    return DEPLOYMENTS[model][0]


def color(model: str) -> str:
    return DEPLOYMENTS[model][1]


def marker(model: str) -> str:
    return DEPLOYMENTS[model][2]


def apply() -> None:
    import matplotlib

    matplotlib.rcParams.update({
        "font.size": FONT,
        "axes.labelsize": FONT,
        "axes.titlesize": FONT,
        "xtick.labelsize": SMALL,
        "ytick.labelsize": SMALL,
        "legend.fontsize": SMALL,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
