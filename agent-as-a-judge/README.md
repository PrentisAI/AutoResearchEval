# autoresearcheval

[![PyPI](https://img.shields.io/pypi/v/autoresearcheval)](https://pypi.org/project/autoresearcheval/)

A two-stage pipeline for turning raw AI-agent research trajectories into a
structured, evidence-grounded failure-taxonomy classification.

```
raw trajectory log  --[Stage 1]-->  analysis.md  --[Stage 2]-->  ARFT labels
```

```python
from autoresearcheval import generate_analysis, label_arft, pattern_info

analysis = generate_analysis("path/to/trajectory_dir")        # Stage 1
result   = label_arft(analysis["analysis"], api_key="sk-...")  # Stage 2

print(result["summary"], result["total_failures"])
for code in result["failure_modes"]:
    print(code, pattern_info(code)["name"])
```

A trajectory directory (or a single trajectory JSON, or a dict) and an API key are all
that is required.

**Stage 1** spawns one fresh Claude Code session per trajectory to write a deep,
ONBOARDING-conformant `analysis.md` — a structured, six-stage critique (ideation,
retrieval & synthesis, execution, analysis, writing, self-review) with a claim-by-claim
verdict table and independent numerical sanity checks, not a summary.

**Stage 2** classifies each `analysis.md` against **ARFT** (the AutoResearch Failure
Taxonomy: `A.1`–`X.8`, 45 patterns spanning six lifecycle stages plus a cross-cutting
layer, rolling up to four root-cause pillars — see [`ARFT.md`](ARFT.md)) and rolls the
results into a pattern × model matrix, a root-cause breakdown, co-occurrence stats, and
cross-model agreement.

## Why the judge is artifact-aware

Many failures leave no trace in the report — a result the code never produced, a method
the logs never ran — so catching them means checking the manuscript against the
artifacts. Stage 1 therefore runs a fresh, zero-history session per trajectory, with
shell access and no network, handed the full evidence package (task statement, execution
log, delivered filesystem, the scorer's own source, every scoring call, read-only gold)
and required to anchor every finding to a line, file, or value.

Against three-expert annotation on 50 stratified trajectories this reaches **κ = 0.75**
(pattern) and **0.83** (root cause), versus 0.53 / 0.62 for a single-call LLM-as-a-judge
on the transcript alone. Almost all of the gain is recall — which is the point: artifact
access is what makes transcript-invisible failures detectable.

## Install

```bash
pip install autoresearcheval              # the library and the CLIs
pip install 'autoresearcheval[stats]'     # adds pandas, needed for the corpus rollups
```

- Python 3.10+. The only hard dependency is `httpx`.
- **Stage 2** needs an API key for any OpenAI-compatible chat-completions endpoint. Key
  resolution: the `api_key` argument, then `ARFT_OPENROUTER_KEY`, then
  `~/.openrouter_key`, then `OPENROUTER_API_KEY`. Point it elsewhere with `base_url=` or
  the `AAJ_ENDPOINT` env var.
- **Stage 1**, and Stage 2's alternate executor, spawn headless
  [Claude Code](https://claude.com/product/claude-code) sessions and need the `claude`
  CLI installed and authenticated.

## The two calls

| | What it does | Cost | Needs |
|---|---|---|---|
| `generate_analysis(trajectory, ...)` | reads the trajectory's artifacts and writes a six-stage critique with a claim-by-claim verdict table | minutes | the `claude` CLI |
| `label_arft(analysis, ...)` | maps that critique onto the 45 ARFT codes | one completion | an API key |

They are separate on purpose. Stage 1 is the expensive half, and it produces the
artifact that makes the labels auditable; folding the two together would hide both.

`generate_analysis` returns `{"task_id", "analysis", "qa", "path", "workspace",
"duration_s", "returncode"}`. `qa["ok"]` is the depth gate described below — False means
the analysis came back thinner than the framework's bar, not that the call failed.

`label_arft` returns the classification plus `failure_modes` (every established code),
`total_failures`, and `qa` (the schema-and-polarity gate). Each hit carries its `code`,
`name`, `stage`, `pillar`, `root_cause`, `confidence`, `evidence` and `why`.

### Telling the analyst about your harness

Two facts change what counts as a finding: whether the harness's retrieval tools did
real network I/O or were mocked, and whether gold values are reachable locally. By
default the analyst is told to **work both out from the trajectory** and report what it
concluded — so the defaults are correct for any harness and nothing needs editing.

Override them with `retrieval_note=` / `gold_note=` (or the `RETRIEVAL_NOTE` /
`GOLD_NOTE` constants for the batch CLI) **only if you can state the truth**. A declared
fact beats an inferred one, but a wrong one is worse than neither: an analyst told that
a real search tool is mocked will report fabricated retrieval that never happened, and
one told that a shim is real will credit calibration against literature that was never
fetched.

## Batch CLIs

The commands that produced the paper's corpus install alongside the library:
`aaj-generate`, `aaj-classify`, `aaj-classify-cc`, `aaj-aggregate`, `aaj-status`,
`aaj-verify`, `aaj-analysis-qa`, `aaj-label-qa`.

## Quickstart — Stage 1: trajectory → analysis.md

```bash
aaj-generate --run-dir /path/to/your_model__your_suite \
    --concurrency 4 --resume --model claude-opus-4-8
```

Expects `<run-dir>/traj/*.json`, one JSON object per trajectory with at least a
`task_id` field and a log field `traj_tools.py` can recognize. `traj_tools.py` normalizes
three log formats out of the box — Claude Code stream-JSON, Gemini CLI NDJSON, Codex CLI
JSONL — so multi-megabyte logs need no truncation; see `traj_tools.detect_format`.
Writes `<model>/<task_id>/analysis.md` under `./corpus` by default (override with the
`AAJ_CORPUS_DIR` env var).

### The depth exemplar

Each session is handed two references: [`ONBOARDING.md`](src/autoresearcheval/data/ONBOARDING.md) (the framework — workflow,
iron rules, required skeleton) and [`analysis_long.md`](src/autoresearcheval/data/analysis_long.md) (a worked example
of the bar being met). The exemplar is a real analysis of a real trajectory, not a
template: a microkinetics rollout whose headline finding is refuted by a sweep table the
agent itself printed. It is what "every issue is a paragraph with a mechanism, a
fair-credit reading and a numeric anchor, plus a `[stage | root cause]` trailer" looks
like in practice, and it clears every gate of the quality checker:

```bash
python -c "from autoresearcheval import config; print(config.exemplar())"   # where it lives
aaj-analysis-qa "$(python -c 'from autoresearcheval import config; print(config.exemplar())')" \
    --reason "soft[current_density]"
```

Point `AAJ_EXEMPLAR` at a different file to calibrate against your own corpus instead.
If neither exists, the prompt drops the exemplar line and falls back to ONBOARDING §3;
depth then rests on the QA gate alone, so writing one reference analysis by hand for
your own domain is worth the effort.

Note what the exemplar depends on: several of its sharpest findings turn on knowing that
this run's `WebSearch` was a shim while `WebFetch` was real. That is exactly the fact
`RETRIEVAL_NOTE` carries into your own runs — get it wrong and the analyst will confidently
make the opposite mistake.

## Quickstart — Stage 2: analysis.md → ARFT classification

```bash
export ARFT_OPENROUTER_KEY=...        # or drop a key in ~/.openrouter_key
scripts/run_all_arft_api.sh           # self-healing: resumes, retries QA failures
```

Reads `./corpus/<model>/<task>/analysis.md` (`$AAJ_CORPUS_DIR` — the same default
Stage 1 writes to, so the two stages compose with no extra flags), writes
per-analysis `<model>/<task_id>.json` plus the rolled-up stats to `./results`
(`$AAJ_OUT_DIR`):

| Output | What |
|---|---|
| `<model>/<task>.json` | Per-analysis classification, evidence-backed |
| `agg.json` | Dense `[model, task, {code: score}]` grid |
| `SUMMARY.md` | Pattern × model HIT/PARTIAL matrix, ranked |
| `root_cause_stats.md` | Lifecycle stage × root-cause pillar breakdown |
| `matrix_long.csv` | Tidy long-format table everything else derives from |
| `cooccurrence.csv` / `agreement.csv` | Pattern co-occurrence; cross-model agreement |
| `tables.tex` | Paper-ready LaTeX |
| `UNCOVERED.md` | Findings that fit no existing pattern — taxonomy-gap review |

Then check the result is trustworthy before you rely on it:

```bash
aaj-verify   # polarity regression + (optionally) a prior-run comparison
```

`aaj-verify` checks that the classifier isn't mistaking exculpatory language for a
finding (a real failure mode — some diagnostic vocabulary shows up almost entirely in
*clearing* statements in this kind of writeup) and, if you pass `--pass2` against a
second independent run, reports per-pattern Cohen's κ so you know which codes are
reliably distinguishable and which need their guide entry sharpened.

## Taxonomy

The 45-pattern label space, its four root-cause pillars, and what the 800-trajectory
audit found are documented in [`ARFT.md`](ARFT.md). The code list lives in
`autoresearcheval/patterns.py` and the classifier's operational guide ships as package
data (`config.arft_guide()`).

## Reasoning budget

The reasoning-token budget is the main quality lever on Stage 2 — don't turn it down.
Disabling reasoning entirely measured **38% recall** against a hand-verified reference
labelling, missing several genuinely-present patterns; `3000` reasoning tokens
(the `reasoning_tokens` default) measured **81% recall**.

Stage 1 is the heavier stage per item, being open-ended authoring rather than
extraction. Use `--dry-run` / `--n` to size a pilot before committing to a full run.

## License

MIT — see `LICENSE`.
