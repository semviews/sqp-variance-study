# Reproducing the results

The tables, number macros, and figures of the submitted paper are in
`reference/`. There are two levels of reproduction. Each goes further back
toward the model calls.

| Level | What you do | Needs | Time |
|---|---|---|---|
| **A** | Regenerate every processed file, table, figure, and number from the shipped raw logs, and compare with `reference/` | the three benchmark checkouts at pinned commits and their Python environments; no model access | about 4 minutes |
| **B** | Re-execute the benchmarks against live models | level A, plus an OpenAI-compatible gateway that serves the deployments, and an API budget | several days of gateway time (the study's campaigns ran over eight days) |

Level A calls no model. Level B cannot return the same outputs as ours:
run-to-run variance is what the paper measures, and the deployments behind
the gateway change over time. Level B reproduces the procedure, so its
results should be compared with ours as distributions, not number by
number.

Related documents: [ARCHITECTURE.md](ARCHITECTURE.md) explains the code and
[DATA.md](DATA.md) the files.

## 1. Requirements

**Platform.** The study ran on macOS on Apple silicon (arm64) with Python
3.12 for SemBench and LOTUS, 3.10 for LRO-Bench, and 3.11 for UDA-Bench.
Other platforms are untested. No GPU is needed: every model is reached
through a gateway.

**Software.**

- Git, and [uv](https://docs.astral.sh/uv/) or `python -m venv` with pip.
- LaTeX is not needed. The scripts write tables and macros as LaTeX source
  fragments, and figures as PDF files through matplotlib.

**Pinned versions** (`versions/`):

| Component | Commit or version | File |
|---|---|---|
| [SemBench](https://github.com/SemBench/SemBench) | `c814e38` | `versions/sembench_commit.txt` |
| [LOTUS](https://github.com/lotus-data/lotus) | `lotus-ai==1.1.3` from PyPI (source commit `136ae4f`) | `versions/lotus_commit.txt`, `versions/env-lotus.txt` |
| [LRO-Bench](https://github.com/LROBench/LROBench) | `6eb2930` | `versions/lrobench_commit.txt` |
| [UDA-Bench](https://github.com/BIT-DataLab/UDA-Bench) | `f2da981` | `versions/udabench_commit.txt` |
| Python packages of the four environments | `uv pip freeze` output | `versions/env-{sembench,lotus,lrobench,udabench}.txt` |

## 2. What the release contains

| Path | Contents |
|---|---|
| `experiments/raw/sembench-repeats/` | Every SemBench execution: answers, per-call logs, metadata, failure records |
| `experiments/raw/lrobench/` | Every LRO-Bench execution: per-query results and per-call logs |
| `experiments/raw/uda-repeats/` | Every UDA-Bench execution: extracted tables, per-call logs, query results and scores |
| `experiments/raw/controls/temperature/` | The temperature control: item manifests and every response with its reasoning text |
| `experiments/raw/product-reviews/` | The product-review executions |
| `experiments/processed/` | Outputs of the analysis scripts, as used for the submitted paper |
| `experiments/data/` | The 40-review join input, the UDA-Bench query manifests, and the synthetic product reviews |
| `reference/tables/` | The generated tables as submitted, and `macros.tex`, which defines every number quoted in the paper's prose |
| `reference/figures/` | The generated figures as submitted |

The benchmarks' own data are not redistributed. Section 3 fetches them.

## 3. Setup

All commands run from the repository root. The external checkouts can live
anywhere; the defaults below match the scripts.

**SemBench and LOTUS.** SemBench is expected at `./SemBench`. Its setup
script creates two environments: `.venvs/sembench` runs SemBench's
evaluator and all our analysis scripts, and `.venvs/lotus` runs LOTUS for
the drivers.

```bash
git clone https://github.com/SemBench/SemBench.git SemBench
git -C SemBench checkout c814e3807e72d4cf876b852b17e77f3cc94575c2
(cd SemBench && bash scripts/setup_envs.sh lotus)
```

The Movie and Medical inputs and SemBench's published per-run metrics are
part of the SemBench repository. To match our package versions exactly,
install `versions/env-sembench.txt` and `versions/env-lotus.txt` into the two
environments (`uv pip install --python SemBench/.venvs/<name>/bin/python -r
versions/env-<name>.txt`).

**LRO-Bench.** Level A reads only LRO-Bench's query metadata. Level B also
needs its environment and its databases.

```bash
git clone https://github.com/LROBench/LROBench.git ~/code/LROBench
git -C ~/code/LROBench checkout 6eb2930468ef892b03421a4d7f1232de1739d6fb
# level B only:
(cd ~/code/LROBench && unzip -q databases.zip && uv venv --python 3.10 .venv \
  && uv pip install --python .venv/bin/python -r requirements.txt)
```

`versions/env-lrobench.txt` lists the exact packages we used, if
LRO-Bench's own requirements resolve differently.

**UDA-Bench.** Scoring runs in UDA-Bench's own environment, which needs the
Player dataset and its ground truth. Download both from the links in
UDA-Bench's README (Google Drive): the documents go to
`datasets/Player/`, and the ground-truth tables go to `Query/Player/` beside
the query files.

```bash
ARTIFACT=$PWD
git clone https://github.com/BIT-DataLab/UDA-Bench.git ~/code/UDA-Bench
git -C ~/code/UDA-Bench checkout f2da981b5d37c058345f27f23c7979eb44697f0c
(cd ~/code/UDA-Bench && uv venv --python 3.11 .venv \
  && uv pip install --python .venv/bin/python -r "$ARTIFACT/versions/env-udabench.txt")
```

If the checkouts are elsewhere, set `LROBENCH` and `UDABENCH` to their
paths. `PY` overrides the analysis interpreter and `UDA_PY` the UDA-Bench
interpreter.

**Tests.** The unit tests need only pytest:

```bash
make test        # $(PY) -m pytest -q tests
```

## 4. Level A: regenerate everything from the raw logs

```bash
make rebuild     # scripts/rebuild.sh
make check       # scripts/compare_reference.py
```

`scripts/rebuild.sh` runs each analysis step in the order listed in
[ARCHITECTURE.md](ARCHITECTURE.md#3-analysis-pipeline), then `paper_macros.py`.
It overwrites `experiments/processed/` and writes `outputs/tables/` and
`outputs/figures/`.

`make check` compares every regenerated table in `outputs/tables/` with
`reference/tables/`. For `macros.tex` it compares each macro and prints the
ones whose value differs. All bootstraps and simulations are seeded, so the
expected result is no difference. On the machine that produced the paper, a
rebuild inside this release takes about four minutes and reproduces every
table and every macro exactly; `make check` then reports `N of N identical`
for both. Figures are regenerated too but not compared, because the PDF
files embed a creation time; compare `outputs/figures/` with
`reference/figures/` by eye.

On macOS, numpy in the SemBench environments may crash inside restricted
sandboxes; run the rebuild in a normal shell.

## 5. Level B: re-execute the benchmarks

### Gateway

The drivers call models through a [LiteLLM](https://docs.litellm.ai)-style
OpenAI-compatible gateway. Create a file `.env` in the repository root:

```bash
LOCAL_LITELLM_ENDPOINT=https://<your gateway>/v1
LOCAL_LITELLM_API_KEY=<key>
# only for the commercial models (--endpoint enterprise):
ENTERPRISE_LITELLM_ENDPOINT=https://<second gateway>/v1
ENTERPRISE_LITELLM_API_KEY=<key>
```

The gateway must serve these model names. The study used two hosted
services that we do not operate; any deployment of the same weights can
stand in, but it is a different deployment.

| Model name the drivers request | Weights | Service |
|---|---|---|
| `managed-gpt-oss-120b` | gpt-oss-120b | managed inference service (M) |
| `managed-llama-3-3-70b-instruct` | Llama-3.3-70B-Instruct | M |
| `managed-mistral-small-3-1-24b-2503` | Mistral-Small-3.1-24B-Instruct-2503 | M |
| `vllm-llama-3-3-70b-instruct` | Llama-3.3-70B-Instruct | shared vLLM service (V) |
| `vllm-qwen2-5-72b-instruct` | Qwen2.5-72B-Instruct | V |
| `vllm-granite-3-3-8b-instruct` | Granite-3.3-8B-Instruct | V |
| `Azure/gpt-4o` | GPT-4o | commercial, second gateway |
| `aws/claude-sonnet-5` | Claude Sonnet 5 | commercial, second gateway |

`scripts/list_models.py` prints what a gateway serves, and
`scripts/probe_models.py` checks whether a model accepts temperature 0 and
returns log-probabilities.

### Campaigns

Drivers run with the LOTUS environment
(`SemBench/.venvs/lotus/bin/python`), except LRO-Bench, which uses its own.
Every launcher is resumable: finished executions are skipped, and a launcher
can be re-run after an interruption. Logs go to `experiments/logs/`. To
start from nothing, move `experiments/raw/` aside first; otherwise every
cell is already finished and nothing runs.

| Order | Command | What it runs | Model calls in the study |
|---|---|---|---|
| 1 | `scripts/run_movie_matrix.sh` | SemBench Movie Q1–Q4 and Q8–Q10 on the six open-weight deployments (10 executions of gpt-oss-120B, 5 of the others) | part of 291,760 (main matrix) |
| 2 | `scripts/run_medical_matrix.sh` | SemBench Medical Q1, Q4, Q10, same deployments | part of the main matrix |
| 3 | `SemBench/.venvs/sembench/bin/python scripts/make_movie_join_subset.py`, then `scripts/run_movie_joins.sh` | Movie Q5–Q7 exact self-joins on 40 reviews | 168,000 |
| 4 | `scripts/run_cause_probes.sh` | One factor varied at a time on Movie Q3, Q8, Q9: concurrency, input order, reasoning effort | 39,680 |
| 5 | `scripts/run_lrobench.sh` | Every scored LRO-Bench query, every implementation, five executions | 567,591 |
| 6 | `scripts/run_uda.sh` | UDA-Bench Player extraction, five executions, then scoring | 33,267 |
| 7 | `scripts/run_boost_calls.sh` | The temperature control (4,800 calls); more executions on Movie Q2–Q4, Q8, Q9 (58,880); and the commercial models (`run_commercial.sh`) | 63,680 plus the commercial calls |
| 8 | `SemBench/.venvs/lotus/bin/python scripts/run_product_reviews.py --execute` | 13 product-review tasks, five executions | 6,500 |

The commercial lane (`scripts/run_commercial.sh`) made 55,360 kept calls.
Every enterprise run passes `--max-cost`, and each driver stops before a
cell once the spend logged on that gateway reaches the cap (`CAP`, default
USD 50). The commercial models cost about USD 0.04 per execution of Movie
Q3 (120 calls).

Practical notes from the study:

- Run at most one driver per deployment. The launchers already do; several
  20-worker drivers on one gateway were rate-limited.
- The drivers resend calls that the gateway refused for rate limiting, and
  record any other failed call as a failed execution. A cell is retried at
  most three times in total.
- GPT-4o's content filter rejected some SemBench prompts. A query whose
  every attempt of one execution was rejected is skipped for that
  deployment and excluded from the commercial comparison for all
  deployments.
- After a campaign, run level A to regenerate the tables, figures, and
  macros from the new logs.

## 6. Where each display comes from

Each output file has a counterpart with the same name under `reference/`.

| Display in the paper | Script | Output |
| --- | --- | --- |
| Figure 1 (Movie Q3 counts) | `plot_teaser.py` | `outputs/figures/teaser_q3.pdf` |
| Exposure rules per plan operator | `exposure_rules.py` | `outputs/tables/exposure.tex` |
| Queries | `report_sembench_repeats.py` | `outputs/tables/queries.tex` |
| Model deployments | written in the paper; its counts are macros from `paper_macros.py` | — |
| Published results | `report_published_repeats.py` | `outputs/tables/published.tex` |
| Published ranges and executions needed | `plot_published.py` (with `runs_needed.py`) | `outputs/figures/published.pdf` |
| Item flip rates and answer reproduction | `report_sembench_repeats.py` | `outputs/tables/repeats.tex` |
| Commercial models | `commercial_report.py` | `outputs/tables/commercial.tex` |
| Flip rate by output type | `report_sembench_repeats.py` | `outputs/figures/repeats_by_kind.pdf` |
| Flip growth and amplification | `plot_amplification.py` | `outputs/figures/amplification.pdf` |
| LRO-Bench | `report_text_suites.py` | `outputs/tables/lro_repeats.tex` |
| Joins | `report_text_suites.py` | `outputs/tables/joins.tex` |
| UDA-Bench | `report_text_suites.py` | `outputs/tables/uda.tex` |
| LRO-Bench recommendations under repetition | `claim_audit.py` | `outputs/tables/claims.tex` |
| Three levels of reproducibility | `report_ladder.py` | `outputs/figures/ladder.pdf` |
| Sources of variance (margins, reasoning) | `report_sembench_repeats.py` (with `margin_noise.py`) | `outputs/figures/repeats_sources.pdf` |
| Cause probes | `report_probes.py` | `outputs/tables/probes.tex` |
| Remedies | `remedies_sembench_repeats.py` | `outputs/tables/remedies.tex` |
| Cascades | `cascade_replay.py` | `outputs/tables/cascade.tex` |
| Every number in the prose | `paper_macros.py` | `outputs/tables/macros.tex` |

To find where a number in the paper's prose comes from, search
`reference/tables/macros.tex` for the value to find its macro name, then
search for the name (or its prefix) in `scripts/paper_macros.py` and the
processed JSON files.
