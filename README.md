<h1 align="center">AutoResearchEval</h1>

<p align="center">
  <b>How do Agents Fail on AutoResearch?</b><br/>
  End-to-end diagnostic evaluation on 100 real-world frontier research tasks.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/tasks-100-blue" alt="100 tasks"/>
  <img src="https://img.shields.io/badge/domains-7-blue" alt="7 domains"/>
  <img src="https://img.shields.io/badge/trajectories-800-orange" alt="800 trajectories"/>
  <img src="https://img.shields.io/badge/ARFT-45%20failure%20patterns-red" alt="45 ARFT patterns"/>
  <img src="https://img.shields.io/badge/python-3.10%2B-green" alt="Python 3.10+"/>
</p>

This repository contains the code behind **AutoResearchEval** — a benchmark and diagnostic study
of *AutoResearch*: LLM agents that carry a study end to end, from an initial hypothesis through
literature, experiments, and analysis to a written report.

Existing evaluations score the endpoint, which says whether a run matched a reference but not how
the agent worked or where it broke. AutoResearchEval instead builds **100 tasks** from published
frontier science across **seven domains** and the full research lifecycle, runs them as
autonomous six-stage rollouts under **eight harness–model combinations** (**800 trajectories**),
and annotates every trajectory at the process level against **ARFT**, the AutoResearch Failure
Taxonomy — **45 empirically-grounded failure patterns**. The headline finding is that failures
span every stage but converge on one limitation: agents lack a **metacognitive loop**.

## Overview

![Construction, rollout, and evaluation of AutoResearchEval](assets/overview.jpg)

**a — Task construction.** 5,878 candidate papers are parsed into seven fields and filtered to
100 tasks across seven domains. The agent sees only the query (Premise, Tension); the target
(KeyClaims, Conclusion) is withheld — so many paths through a task are admissible and none of
them is the reference path.

**b — Rollout.** Each task runs once per harness–model pair as a six-stage episode with revision,
yielding 800 trajectories with all artifacts retained.

**c — Diagnosis.** ARFT is induced bottom-up: experts annotate failures in full trajectories,
group them into patterns, and refine until all agree — 45 patterns under 4 root-cause pillars. A
judge agent then reviews the full artifact set under a per-stage rubric, anchoring every issue to
concrete evidence, with a quality checker regenerating weak analyses in a self-healing loop.

## What is in this repository

| Paper component | Code |
|---|---|
| Task construction — mine venues, parse the seven fields, filter, author tasks | `adapters/`, `reconstruct/`, `examples/` |
| Six-stage rollout task packages (`instruction.md`, verifier, rubric) | `examples/materialize_discovery_tasks.py` |
| Reward harness for target-anchored optimization tasks | `harness/`, `verify/` |
| Trajectory intermediate representation + SFT export | `ir/`, `export/` |
| Agent-as-a-judge → `analysis.md` → ARFT classification | [`agent-as-a-judge/`](agent-as-a-judge/README.md) |

| Module | Purpose |
|---|---|
| `adapters/` | source ingestion — OpenAlex metadata + bronze/silver/golden tiering, MinerU-parsed paper corpus |
| `reconstruct/` | seven-field extraction, novelty-move classification, OpenRouter teacher client, held-out ground truth |
| `harness/` | domain-agnostic rollout environment, recompute oracle/tools, three-dimensional verifier, catalysis-QE plugin |
| `ir/` | unified trajectory IR + typed action registries (`ir/actions/`) |
| `export/` | trajectory → SFT/ReAct message export with observation loss masking |
| `verify/` | MLIP pre-filter physical checks |
| `examples/` | pipeline entry points — crawl → mine → export → materialize → roll out |
| `agent-as-a-judge/` | the two-stage annotator: trajectory → `analysis.md` → ARFT |

The **task suite and the 800-trajectory annotated corpus are released separately**; this
repository is the pipeline that produces them. No API keys, data, or model weights are included.

## Setup

```bash
pip install -r requirements.lock          # pinned, verified-importable versions
export OPENROUTER_API_KEY=...             # teacher LLM; never commit keys
```

Core runtime deps are deliberately tiny (`pydantic` only) so the IR / export / verify core imports
without a scientific-computing stack. Heavy, source-specific deps are optional extras:

```bash
pip install -e ".[aiida]"        # AiiDA provenance adapter
pip install -e ".[mlip]"         # MACE / CHGNet / M3GNet prefilter
pip install -e ".[atomate2,mp]"  # atomate2 TaskDocs, Materials Project
```

`requirements.freeze.txt` is the full transitive freeze (226 packages) of the environment these
were verified in. `agent-as-a-judge/` installs separately (`httpx`, `pandas`) and needs the
`claude` CLI authenticated for Stage 1.

## Building tasks from papers

Crawl a corpus and grade it. Tiering is automatic, from OpenAlex signals only — no hand-curated
journal whitelist: **bronze** (breadth), **silver** (the workhorse for extraction), **golden**
(elite venue *and* field-leading impact, weighted up downstream). Young papers are graded on
venue and institution signals; citations only ever promote, never punish recency.

```bash
python examples/crawl_topic_set.py --per-topic 60 --set-name diverse_v1
python examples/fetch_corpus_pdfs.py --set diverse_v1 --unpaywall --s2 --core
python examples/discovery_pattern_mine.py --zip corpus.zip     # -> seven fields per paper
```

Extraction emits the seven fields per paper — **Premise**, **Tension**, **Motivation**,
**Method**, **Experiment**, **KeyClaims**, **Conclusion** — plus the novelty move. Two fields
become the agent's query; **KeyClaims** and **Conclusion** are withheld as ground truth; the
remaining three drive construction, gating whether a paper can become a runnable task and fixing
whether it becomes an *open-ended discovery* or a *target-anchored optimization* task.

Then turn extracted patterns into runnable task packages:

```bash
python examples/export_discovery_tasks.py          # -> discovery_tasks.jsonl
python examples/materialize_discovery_tasks.py \
    --jsonl examples/output/discovery_tasks.jsonl \
    --out examples/output/discovery_tasks_v2 --prompt-version v2_af
```

Each task directory holds `instruction.md` (premise + tension, open decision schema, **no gold
leaked**), a `task.toml`, and the per-paper rubric the verifier scores against. The `v2_af` prompt
version is the domain-general one carrying the explicit six-stage process — **A** Ideation →
**B** Retrieval & Synthesis → **C** Execution → **D** Analysis → **E** Writing → **F** Review —
that the agent must work through; `v1_domain_specific` is kept side by side so instruction
wordings can be A/B'd on the identical task set.

## Rewards for target-anchored optimization tasks

Open-ended discovery tasks expose no objective, so their rollouts are judged on process. For
target-anchored optimization tasks, where an explicit metric exists, the repo ships a live-oracle
rollout whose terminal reward is a real calculation rather than a judgment:

```bash
python examples/discovery_rollout.py --calc emt              # validate the loop, instant
QE_NP=32 QE_NPOOL=4 python examples/discovery_rollout.py --calc qe   # real DFT, admissible
```

| Tier | What it is | Admissible? |
|---|---|---|
| `emt` | instant EMT — plumbing / CI only | never (physically meaningless) |
| `mlip` | CHGNet universal MLIP — cheap prefilter | no (not the honest reward) |
| `qe` | real Quantum ESPRESSO 7.5 PBE-PAW | **yes** — minutes per calc, MPI |

The reward is a gate, not a weighted sum:

```
reward = correctness × (0.5 + 0.25·significance + 0.25·novelty)
```

`correctness ∈ {0,1}` is hard — a re-executed, sane number matching the held-out gold within
tolerance. A beautiful-but-wrong or un-run rollout scores **0**, so the soft dimensions can only
*rank* rollouts that already passed; soft-dim hacking cannot mint reward from nothing. The episode
gate has the same shape: `sane ∧ decisive ∧ valid`, not "produced a number".

Two further invariants are encoded as data rather than prose. Nothing is verified until an
execution check says so: `Trajectory.is_admissible()` returns `verification.passed`, and export
refuses anything else. And failure branches are first-class — a step with a non-zero exit or a
correction is flagged `is_failure_branch=True` and *retained*, becoming error→recovery supervision
in the SFT export rather than being dropped.

`harness/discovery_env.py` imports nothing domain-specific: no ASE, no QE, no chemistry. That
import-poverty is the structural check that the skeleton (frame a tension → design an experiment →
read the result → decide if it is resolved) is domain-agnostic; computational catalysis is the
first plugged-in domain (`harness/domains/catalysis_qe.py`), and a second domain means a new
oracle, not a new driver.

## Agent-as-a-judge

Many failures leave no trace in the report: the agent claims a result its own code does not
produce, or describes a method its logs show it never ran. Detecting these requires comparing the
manuscript against the artifacts. The judge is therefore **artifact-aware** — a fresh,
zero-history Claude Code session per trajectory with shell access and no network tools, given the
rollout's complete evidence package (task statement, full execution log, the delivered container
filesystem, the scorer's real source code, every call the rollout made to the scoring service, and
read-only ground truth) and required to anchor every finding to a line number, file, or exact
value. Against three-expert human annotation it reaches **κ = 0.75** at the pattern level and
**0.83** at the root-cause level, versus 0.53 / 0.62 for a single-call LLM-as-a-judge on the
transcript alone — the gain is almost entirely recall.

```bash
# Stage 1 — trajectory -> analysis.md (six-stage critique, claim-by-claim verdicts)
cd agent-as-a-judge/generate/
python3 generate_analysis_cc.py --run-dir /path/to/your_model__your_suite \
    --concurrency 4 --resume --model claude-opus-4-8

# Stage 2 — analysis.md -> ARFT labels, matrices, root-cause rollups
cd ../classify/
export ARFT_OPENROUTER_KEY=...
./run_all_arft_api.sh        # self-healing: resumes, retries QA failures
python3 arft_verify.py       # polarity regression + cross-run Cohen's κ
```

A `traj_tools.py` extraction CLI normalizes three rollout-log formats (Claude Code stream-JSON,
Gemini CLI NDJSON, Codex CLI JSONL) so multi-megabyte logs stay tractable without truncation. An
automated checker enforces coverage, depth, and anchor density before an analysis is accepted;
documents that fail are regenerated with gate-specific feedback. Full details — rubric, iron
rules, cost, outputs — in [`agent-as-a-judge/README.md`](agent-as-a-judge/README.md).

### ARFT, the label space

The taxonomy Stage 2 classifies against lives in `agent-as-a-judge/classify/arft_patterns.py`
(source of truth for the code list) and `arft_guide.md` (the operational guide handed to the
classifier). Its 45 patterns sit on two orthogonal axes — the **stage** where a failure surfaces
and the **root cause** of why it happens:

| Root-cause pillar | Core failure focus | Patterns |
|---|---|---|
| **R1 · Grounding & Faithfulness** | Claims disconnect from the code, data, logs, or literature that should license them | A.6, B.1, B.2, B.5, C.3, D.1, D.4, D.6, E.1, E.4, F.6, X.6 |
| **R2 · Cognitive Depth & Adaptability** | Shallow reasoning and search, passive self-critique, inability to re-plan | A.1, A.3, B.4, B.6, C.6, C.7, D.5, F.1–F.4, X.3, X.7 |
| **R3 · Scientific Integrity & Alignment** | Metric hacking, shortcut reliance, concealed failure, conclusions fixed in advance | A.2, A.5, C.1, C.2, D.2, D.3, D.7, E.2, E.3, F.5, X.2, X.4, X.5 |
| **R4 · Engineering Robustness** | Numerical faults, unhandled runtime errors, broken CLI/OS interaction | A.4, B.3, C.4, C.5, C.8, X.1, X.8 |

Stages: **A** Ideation (6 patterns) · **B** Retrieval (6) · **C** Execution (8) · **D** Analysis
(7) · **E** Writing (4) · **F** Review (6) · **X** Cross-stage (8).

## Action vocabulary

Two registries under `ir/actions/`, both induced from data (143 GitHub science agents + real
provenance diffs) rather than designed top-down:

- **`registry.json`** — 26 verifier-bound *execution* actions across 12 categories
  (`build_structure`, `run_dft`, `check_convergence`, `triage_failure`, `train_mlip`, …). Every
  entry pins a verifier this repo owns: external tools supply the action space, verification is ours.
- **`discovery_registry.json`** — 10 atomic *reasoning* moves in three phases (FRAME → PROBE →
  RESOLVE): `survey_consensus`, `identify_tension`, `formulate_question`, `propose_hypothesis`,
  `select_system`, `choose_method`, `run_calculation`, `compare_reference`, `interpret_result`,
  `draw_conclusion`.

## Environment variables

| Variable | Used by |
|---|---|
| `OPENROUTER_API_KEY` | teacher LLM for field extraction and move generation |
| `ARFT_OPENROUTER_KEY` | agent-as-a-judge Stage 2 classifier |
| `OPENALEX_API_KEY`, `S2_API_KEY`, `CORE_API_KEY` | corpus crawl and PDF fallback chain (all optional) |
| `QE_PW`, `QE_MPIRUN`, `QE_PSEUDO_DIR`, `QE_NP`, `QE_NPOOL` | Quantum ESPRESSO recompute oracle |
| `MLIP_DEVICE`, `RECOMPUTE_WORKERS` | MLIP prefilter / recompute parallelism |
| `AAJ_CORPUS_DIR`, `AAJ_OUT_DIR` | agent-as-a-judge corpus and results roots |

Keys resolve as: explicit argument → environment → a gitignored `.env` in the repo root.

## Caveats

- Several `examples/` scripts still carry absolute paths from the machine they were developed on
  (`/data/xmyu/scicoder`, conda interpreter paths). Point them at your checkout before running;
  the library modules under `ir/`, `reconstruct/`, `harness/`, and `export/` are path-clean.
- The provenance-reconstruction path does not re-run Quantum ESPRESSO on archived runs — there,
  `reexecuted` means the archive nodes reload and the trajectory faithfully reproduces the
  recorded execution, with an MLIP check as the independent physics signal. The `--calc qe`
  rollout path *does* run real DFT.
- Extraction output is soft by construction: mined patterns are stored as `pending-soft-verify`
  and never enter the admissible pool on their own.

## Citation

The overview figure in `assets/` is the paper's own.

```bibtex
@article{autoresearcheval2026,
  title  = {How do Agents Fail on AutoResearch: End-to-end Diagnostic Evaluation
            on 100 Real-world Frontier Research Tasks},
  year   = {2026},
  note   = {Preprint}
}
```

## License

`agent-as-a-judge/` is MIT — see [`agent-as-a-judge/LICENSE`](agent-as-a-judge/LICENSE).
