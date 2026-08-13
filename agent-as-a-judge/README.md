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
layer, rolling up to four root-cause pillars) and rolls the results into a pattern ×
model matrix, a root-cause breakdown, co-occurrence stats, and cross-model agreement.

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
`task_id` field and a log field `traj_tools.py` can recognize (Claude Code, Gemini CLI,
and Codex CLI log shapes are supported out of the box — see `traj_tools.detect_format`).
Writes `<model>/<task_id>/analysis.md` under `./corpus` by default (override with the
`AAJ_CORPUS_DIR` env var).

## Quickstart — Stage 2: analysis.md → ARFT classification

```bash
cd classify/
export ARFT_OPENROUTER_KEY=...        # or drop a key in ~/.openrouter_key
./run_all_arft_api.sh                 # self-healing: resumes, retries QA failures
```

Reads `./corpus/<model>/<task>/analysis.md` (`$AAJ_CORPUS_DIR` — the same default
Stage 1 writes to, so the two stages compose with no extra flags), writes
per-analysis `classification.json` plus the rolled-up stats to `./results`
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

**ARFT** — the AutoResearch Failure Taxonomy — is defined in `classify/arft_patterns.py`
(source of truth for the code list) and `classify/arft_guide.md` (the operational guide
handed to the classifier — scoring rubric, discrimination rules for easily confused
patterns, and a Do-NOT-label list worth rereading if you retarget this at a different
kind of trajectory).

An earlier, incompatible 29-leaf taxonomy is kept for reference in `legacy/` — see
`legacy/README.md` for why it was superseded and why no runnable classifier ships for
it. **Never mix codes between the two systems**; they collide in meaning (`C1` and
`C.1` mean different things).

## Cost

Classifying 800 analyses (8 models × 100 tasks) through the default OpenRouter
executor at `anthropic/claude-sonnet-5` cost **~$283 total** (~$0.35/analysis
including QA retries). The reasoning-token budget is the main quality lever, not a
cost knob to shave: disabling reasoning entirely measured **38% recall** against a
hand-verified reference labelling (missing several genuinely-present patterns);
`3000` reasoning tokens (`arft_classify_api.py`'s default) measured **81% recall** at
roughly double the disabled-reasoning cost. Don't reduce it to save money.

Stage 1 (trajectory → analysis.md) is the more expensive stage per-item since it's an
open-ended authoring task rather than extraction — budget accordingly and use
`--dry-run` / `--n` to size a pilot before committing to a full run.

## License

MIT — see `LICENSE`.
