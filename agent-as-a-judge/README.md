# agent-as-a-judge

A two-stage pipeline for turning raw AI-agent research trajectories into a
structured, evidence-grounded failure-taxonomy classification.

```
raw trajectory log  --[Stage 1: generate/]-->  analysis.md  --[Stage 2: classify/]-->  taxonomy stats
```

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
pip install -r requirements.txt          # httpx, pandas
```

- Python 3.10+.
- **Stage 2's default executor** (`arft_classify_api.py`) needs only an OpenRouter API
  key (or any OpenAI-compatible endpoint — see `arft_classify_api.py`'s
  `ENDPOINT`/`load_key`).
- **Stage 1**, and Stage 2's alternate executor (`arft_classify_cc.py`), spawn headless
  [Claude Code](https://claude.com/product/claude-code) sessions and need the `claude`
  CLI installed and authenticated.

## Quickstart — Stage 1: trajectory → analysis.md

```bash
cd generate/
# Edit RETRIEVAL_NOTE and GOLD_NOTE at the top of generate_analysis_cc.py first —
# they describe facts specific to YOUR harness (is WebSearch real or mocked? are
# gold values available locally?) and ship as TODO placeholders.

python3 generate_analysis_cc.py --run-dir /path/to/your_model__your_suite \
    --concurrency 4 --resume --model claude-opus-4-8
```

Expects `<run-dir>/traj/*.json`, one JSON object per trajectory with at least a
`task_id` field and a log field `traj_tools.py` can recognize. `traj_tools.py` normalizes
three log formats out of the box — Claude Code stream-JSON, Gemini CLI NDJSON, Codex CLI
JSONL — so multi-megabyte logs need no truncation; see `traj_tools.detect_format`.
Writes `<model>/<task_id>/analysis.md` under `./corpus` by default (override with the
`AAJ_CORPUS_DIR` env var).

### The depth exemplar

Each session is handed two references: `ONBOARDING.md` (the framework — workflow, iron
rules, required skeleton) and [`analysis_long.md`](analysis_long.md) (a worked example
of the bar being met). The exemplar is a real analysis of a real trajectory, not a
template: a microkinetics rollout whose headline finding is refuted by a sweep table the
agent itself printed. It is what "every issue is a paragraph with a mechanism, a
fair-credit reading and a numeric anchor, plus a `[stage | root cause]` trailer" looks
like in practice, and it clears `qa_check_analysis.py` on every gate:

```bash
# from agent-as-a-judge/
python3 generate/qa_check_analysis.py analysis_long.md --reason "soft[current_density]"
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
cd classify/
export ARFT_OPENROUTER_KEY=...        # or drop a key in ~/.openrouter_key
./run_all_arft_api.sh                 # self-healing: resumes, retries QA failures
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
python3 arft_verify.py   # polarity regression + (optionally) a prior-run comparison
```

`arft_verify.py` checks that the classifier isn't mistaking exculpatory language for a
finding (a real failure mode — some diagnostic vocabulary shows up almost entirely in
*clearing* statements in this kind of writeup) and, if you pass `--pass2` against a
second independent run, reports per-pattern Cohen's κ so you know which codes are
reliably distinguishable and which need their guide entry sharpened.

## Taxonomy

The 45-pattern label space, its four root-cause pillars, and what the 800-trajectory
audit found are documented in [`ARFT.md`](ARFT.md). The code list lives in
`classify/arft_patterns.py` and the classifier's operational guide in
`classify/arft_guide.md`.

## Reasoning budget

The reasoning-token budget is the main quality lever on Stage 2 — don't turn it down.
Disabling reasoning entirely measured **38% recall** against a hand-verified reference
labelling, missing several genuinely-present patterns; `3000` reasoning tokens
(`arft_classify_api.py`'s default) measured **81% recall**.

Stage 1 is the heavier stage per item, being open-ended authoring rather than
extraction. Use `--dry-run` / `--n` to size a pilot before committing to a full run.

## License

MIT — see `LICENSE`.
