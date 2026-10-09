# Use of AI tools

The SIGMOD 2027 call for papers requires compliance with the ACM Policy on
Authorship, including its section on the use of artificial intelligence.
That policy asks authors to describe in detail any use of AI tools in
conducting the research: designing experiments, collecting data, coding,
analysing data, validating results, and archiving code and data. AI used
only to assist with writing need not be disclosed. We describe both.

## Tools

We used an AI coding agent in the Cursor editor, backed by Anthropic Claude
models (including Claude Opus 5.5), under the authors' direction. The agent
worked in this repository: it read and wrote files, ran commands, and
launched and monitored experiment scripts.

The language models that the experiments execute (gpt-oss-120B,
Llama-3.3-70B, Mistral-Small-3.1-24B, Qwen2.5-72B, Granite-3.3-8B, GPT-4o,
and Claude Sonnet 5) are the subjects of the study, not authoring tools.
[DATA.md](DATA.md) and the paper describe them.

## Use in conducting the research

- **Code.** The agent wrote most of the code in this repository from the
  authors' specifications: the drivers that wrap each benchmark's harness and
  log every model call, the analysis and statistics scripts, the scripts
  that generate tables, figures, and number macros, the unit tests, and the
  scripts that build this release.
- **Running experiments.** The agent started the model-call campaigns with
  the launchers in `scripts/`, monitored them, and relaunched interrupted
  drivers. Retry rules, attempt limits, and spending caps are fixed in the
  driver code, not left to the agent's judgement during a run.
- **Analysis.** The agent ran the analysis scripts and summarised their
  outputs for the authors. It proposed analyses and controls (for example
  the temperature control, the cause probes, and the replay of remedies on
  recorded executions). The authors decided which to run and how to report
  them.
- **Data.** The 100 synthetic product reviews used in one control
  (`experiments/data/product-reviews/`) were produced by a data generation
  script that draws on a knowledge graph (Wikidata) and uses LLMs to write
  the reviews. All other inputs are the benchmarks' own data, or subsets of
  them selected by seeded scripts.
- **Design and planning.** The agent helped draft the experiment plan,
  the item manifests, and the lists of deviations from the benchmarks'
  defaults. The research questions, the choice of benchmarks and models,
  and the decisions about method were the authors'.
- **Literature.** The agent searched for related work and checked
  bibliographic details against the sources.
- **Artifact.** The agent prepared this release: the anonymisation of the
  logs, the documentation, and the comparison with the submitted tables.

## Use in writing

The agent drafted and revised parts of the paper's text and gave feedback
on drafts. The authors revised and approved all text.

## Safeguards

- **No number typed by hand.** Every number in the paper is a macro written
  by `scripts/paper_macros.py` from the processed files, and every table is
  generated. `make rebuild` regenerates all of them from the raw logs without
  calling a model, and `make check` compares them with the submitted copy
  in `reference/`.
- **Benchmark fidelity.** Prompts, plans, output parsers, and evaluators are
  the benchmarks' own, run from their public repositories at pinned commits.
  Our code only wraps them. The paper lists every deviation.
- **Fixed procedures.** Item subsets and seeds were fixed before the runs
  that use them. Failed executions are recorded and retried at most three
  times, never imputed. Intervals use seeded bootstraps (10,000 resamples,
  seed 11).
- **Raw evidence.** Every model call is logged, and the logs are released,
  so any reported value can be traced to the calls behind it.
- **Review.** The authors reviewed the code, the analyses, and the text,
  and are responsible for all content of the paper and this repository.
