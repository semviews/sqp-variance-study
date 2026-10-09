# Same Query, Different Answer: artifact

This repository accompanies the paper *Same Query, Different Answer:
Run-to-Run Variance in Semantic Query Processing [Experiments & Analysis]*.
It contains the drivers that executed three benchmarks (SemBench, LRO-Bench,
and UDA-Bench) repeatedly at temperature 0, the per-call logs of those
executions, and the analysis that turns the logs into every number, table,
and figure in the paper.

The study re-executes each benchmark through its own harness. Prompts, plans,
output parsers, and evaluators are the benchmarks' own. Our code wraps them to
log every model call and to record each execution separately. The paper lists
every deviation from the benchmarks' defaults.

## Quick start

The tables, figures, and number macros of the submitted paper are in
`reference/`. To regenerate all of them from the raw call logs and compare
with that copy:

```bash
make rebuild          # scripts/rebuild.sh; writes outputs/; no model is called
make check            # compares outputs/tables/ with reference/tables/
```

`make rebuild` needs checkouts of the three benchmarks at pinned commits and
their Python environments; [docs/REPRODUCE.md](docs/REPRODUCE.md) gives the
setup. Re-executing the benchmarks against live models is also described
there. It needs an OpenAI-compatible endpoint and costs money.

Run the unit tests (the variance measures, the output parsers, and the
driver's bookkeeping):

```bash
make test
```

## Documentation

| Document | Contents |
|---|---|
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | Setup, the two levels of reproduction, the command for every campaign, and a map from each display in the paper to the script that makes it |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the code works: the drivers and their logging, the analysis pipeline, the measures, and the statistics |
| [docs/DATA.md](docs/DATA.md) | Benchmarks, deployments, the raw directory layout, the schema of every log and processed file |
| [docs/AI_USE.md](docs/AI_USE.md) | Use of AI tools in this work (ACM policy disclosure) |

## Repository layout

```
scripts/                 drivers (model calls), analysis, report, and plot scripts
  rebuild.sh             runs every analysis step in order
  run_*.sh               launchers for the model-call campaigns
experiments/
  raw/                   per-execution outputs and per-call logs (about 890 MB)
  processed/             analysis outputs read by the report scripts
  data/                  inputs we derived or created (join subset, UDA-Bench queries, product reviews)
reference/               the paper's tables, number macros, and figures as submitted
outputs/                 created by `make rebuild`: the regenerated tables, macros, and figures
versions/                benchmark commits and frozen package lists of the four environments
tests/                   unit tests that need no model and no benchmark checkout
```

## Naming

Deployments are named `<service>-<model>`. `managed-*` is the managed
inference service, written (M) in the paper. `vllm-*` is the shared
vLLM-based service, written (V). `Azure-gpt-4o` and `aws-claude-sonnet-5` are
the two commercial models, reached through a second gateway.

## Licence

The code is released under the MIT licence (`LICENSE`). The benchmarks, their
data, and their evaluators are not redistributed here. They remain under
their own licences and are fetched from their public repositories at the
commits in `versions/`. The raw logs contain model outputs and prompt
excerpts from those benchmarks' inputs.

## Citation

If you use this code or these logs, please cite the paper (also in
`CITATION.bib` and `CITATION.cff`):

```bibtex
@unpublished{hassanzadeh2026samequery,
  title  = {Same Query, Different Answer: Run-to-Run Variance in Semantic Query Processing},
  author = {Hassanzadeh, Oktie and Subramanian, Dharmashankar},
  note   = {Under review},
  year   = {2026},
  url    = {https://github.com/semviews/sqp-variance-study}
}
```
