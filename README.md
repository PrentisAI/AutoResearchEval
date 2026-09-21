<h1 align="center">AutoResearchEval</h1>

<p align="center">
  <b>How Do Agents Fail on AutoResearch?</b><br/>
  End-to-End Diagnostic Evaluation on 100 Real-World Frontier Research Tasks
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.14905"><b>Paper</b></a> ·
  <a href="https://titanresearchlabs.github.io/AutoResearchEval-site/"><b>Website</b></a> ·
  <a href="#citation"><b>Citation</b></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/tasks-100-blue" alt="100 tasks"/>
  <img src="https://img.shields.io/badge/domains-7-blue" alt="7 domains"/>
  <img src="https://img.shields.io/badge/trajectories-800-orange" alt="800 trajectories"/>
  <img src="https://img.shields.io/badge/ARFT-45%20failure%20patterns-red" alt="45 ARFT patterns"/>
  <img src="https://img.shields.io/badge/python-3.10%2B-green" alt="Python 3.10+"/>
</p>

*AutoResearch* is the paradigm where one LLM agent carries a study end to end — hypothesis,
literature, experiment, analysis, write-up. Endpoint scoring tells you whether the final answer
matched a reference; it does not tell you how the agent worked or where it broke.

**AutoResearchEval** is a diagnostic study of that gap. It builds **100 tasks** from published
frontier science across **seven domains**, runs each one as an autonomous six-stage rollout under
**eight harness–model combinations** (**800 trajectories**), and annotates every trajectory at the
*process* level — against the artifacts it produced, not just its report. Failures appear at every
stage but converge on one limitation: agents lack a **metacognitive loop** — the ability to check
what they produced against what they found, revise when it does not hold up, and question whether
the path they took was sound.

## The study in one figure

![Construction, rollout, and evaluation of AutoResearchEval](assets/overview.png)

| | Stage | What happens |
|---|---|---|
| **a** | **Task construction** | 5,878 candidate papers across nine fields → 100 tasks in seven domains. Each paper is parsed into seven fields; the agent is shown only the query (`Premise`, `Tension`), while the target (`KeyClaims`, `Conclusion`) is withheld. No method is given, so many paths are admissible and none is *the* reference path. |
| **b** | **Rollout** | Each task runs once per harness–model pair as a six-stage episode with revision: 800 trajectories, 73k tool calls, 92.3 steps per episode on average, all artifacts retained. |
| **c** | **Diagnosis** | ARFT is induced bottom-up — three experts annotate full trajectories, group them, and refine to consensus (κ = 0.85 after five rounds) → 45 patterns under 4 root-cause pillars. A judge agent then reviews the whole artifact set under a per-stage rubric, anchoring each issue to concrete evidence, with a quality checker regenerating weak analyses. |

Two task types are reported separately and never compared: **open-ended discovery** (*n* = 70; no
metric exists, so the process is what is judged) and **target-anchored optimization** (*n* = 30;
an explicit human SOTA or computable metric exists). The eight agents — one harness supplies the
tool loop, file system, and code execution; the backbone model drives it:

| Harness | Backbone models |
|---|---|
| Claude Code | opus-4.8, claude-sonnet-5, qwen3.7-max, glm-5.2, minimax-m3, deepseek-v4-pro |
| Codex | gpt-5-mini |
| Gemini CLI | gemini-3.5-flash |

## What is in this repository

The study has three moving parts — build the tasks, run the agents, diagnose the trajectories.
This repository is the code for the first and the third.

```
papers ──▶ [ this repo: pipelines/ ] ──▶ tasks ──▶ [ container rollouts: NOT here ]
                                                        │
                                      trajectories ◀────┘
                                            │
                                            ▼
                               [ this repo: agent-as-a-judge/ ] ──▶ analysis.md ──▶ ARFT labels
```

| Area | What it is | Docs |
|---|---|---|
| [`pipelines/`](pipelines/) | **papers → tasks** (panel **a**): crawl and tier a corpus, extract the seven fields, materialize runnable task directories | [`pipelines/README.md`](pipelines/README.md) |
| [`agent-as-a-judge/`](agent-as-a-judge/) | **trajectories → ARFT labels** (panel **c**): the two-stage artifact-aware judge and the 45-pattern taxonomy, published as [`autoresearcheval`](https://pypi.org/project/autoresearcheval/) on PyPI | [README](agent-as-a-judge/README.md) · [ARFT](agent-as-a-judge/ARFT.md) |
| [`harness/`](harness/), [`verify/`](verify/) | the rubric scorer, the domain-agnostic episode skeleton, and computational catalysis as the one worked recompute oracle | [`harness/README.md`](harness/README.md) |
| [`ir/`](ir/), [`export/`](export/) | trajectory IR, the two induced action registries, SFT/ReAct export with loss masking | [`ir/README.md`](ir/README.md) |
| [`adapters/`](adapters/), [`reconstruct/`](reconstruct/) | library code behind `pipelines/`: OpenAlex + PDF-corpus adapters, the extraction prompts, the OpenRouter teacher client | — |

Groups 3 and 4 are the machinery the task-construction line grew out of. They matter if you want
to extend the suite with a new scored domain, and are unnecessary if you only want to reproduce
the diagnosis.

### Not in this repository

- **The rollout infrastructure** (panel **b**). Rollouts ran as one Docker container per
  model–task on a SLURM node (8×B300, 32 cores and 150 GB per container, 4 h wall clock), driving
  Claude Code / Codex / Gemini CLI against models served through OpenRouter. That orchestration is
  not part of this release.
- **The task suite and the annotated corpus.** The 100 tasks and the 800 trajectories with their
  process-level annotations are released as data, separately from this code.
- **Keys, credentials, model weights.** None are included; see [Environment variables](#environment-variables).

## Setup

```bash
pip install -r requirements.lock          # pinned, verified-importable versions
export OPENROUTER_API_KEY=...             # teacher LLM for extraction; never commit keys
```

Core deps are `pydantic` only, so the IR / export / verify core imports without a
scientific-computing stack. Heavy per-source deps are extras:

```bash
pip install -e ".[aiida]"        # AiiDA provenance adapter
pip install -e ".[mlip]"         # MACE / CHGNet / M3GNet prefilter
pip install -e ".[atomate2,mp]"  # atomate2 TaskDocs, Materials Project
```

`requirements.freeze.txt` is the full 226-package freeze of the verified environment.
The judge is a separate, published package — `pip install autoresearcheval` — and needs an
authenticated `claude` CLI for Stage 1.

## Quickstart

**Papers → tasks** — full walkthrough in [`pipelines/README.md`](pipelines/README.md):

```bash
python pipelines/corpus/crawl_topic_set.py --per-topic 60 --set-name diverse_v1
python pipelines/corpus/fetch_corpus_pdfs.py --set diverse_v1 --unpaywall --s2 --core
python pipelines/tasks/discovery_pattern_mine.py --zip corpus.zip
python pipelines/tasks/export_discovery_tasks.py
python pipelines/tasks/materialize_discovery_tasks.py \
    --jsonl pipelines/output/discovery_tasks.jsonl \
    --template path/to/env_template \
    --out pipelines/output/discovery_tasks_v6 --prompt-version v6_report_review
```

**Trajectories → ARFT labels** — published as the `autoresearcheval` package; full
walkthrough in [`agent-as-a-judge/README.md`](agent-as-a-judge/README.md):

```bash
pip install autoresearcheval
```

```python
from autoresearcheval import generate_analysis, label_arft, pattern_info

analysis = generate_analysis(trajectory, retrieval_note=..., gold_note=...)
result   = label_arft(analysis["analysis"], api_key="sk-...")

for code in result["failure_modes"]:
    print(code, pattern_info(code)["name"])
```

Or over a whole corpus, with the batch CLIs the package installs:

```bash
aaj-generate --run-dir /path/to/your_model__your_suite --concurrency 4 --resume
export ARFT_OPENROUTER_KEY=...
agent-as-a-judge/scripts/run_all_arft_api.sh   # self-healing: resumes, retries QA failures
aaj-verify                                     # polarity regression + cross-run Cohen's kappa
```

## ARFT in one paragraph

45 empirically-grounded failure patterns on two axes — the lifecycle **stage** where a failure
surfaces (A–F plus a cross-cutting X), and its **root cause** (grounding, depth, integrity,
robustness). Auditing all 800 trajectories yields 12,712 hits; the three cognitive pillars account
for 92.1% of them and engineering robustness for 7.9%. The single most frequent pattern is
**F.4 · Uncorrected Self-Awareness** — the agent names a severe flaw in its own review and ships
anyway — in 82.5% of analyses. Full label space and per-pillar breakdown:
[`agent-as-a-judge/ARFT.md`](agent-as-a-judge/ARFT.md).

## Environment variables

| Variable | Used by |
|---|---|
| `OPENROUTER_API_KEY` | teacher LLM for field extraction and move generation |
| `ARFT_OPENROUTER_KEY` | agent-as-a-judge Stage 2 classifier |
| `AAJ_CORPUS_DIR`, `AAJ_OUT_DIR` | agent-as-a-judge corpus and results roots |
| `AAJ_EXEMPLAR`, `AAJ_ONBOARDING`, `AAJ_GUIDE` | override the packaged Stage 1 exemplar, framework, or Stage 2 guide |
| `AAJ_ENDPOINT` | OpenAI-compatible chat-completions URL for Stage 2 (default OpenRouter) |
| `OPENALEX_API_KEY`, `S2_API_KEY`, `CORE_API_KEY` | corpus crawl and PDF fallback chain (optional) |
| `QE_PW`, `QE_MPIRUN`, `QE_PSEUDO_DIR`, `QE_NP`, `QE_NPOOL` | Quantum ESPRESSO recompute oracle |
| `MLIP_DEVICE`, `RECOMPUTE_WORKERS` | MLIP prefilter / recompute parallelism |

Keys resolve as: explicit argument → environment → a gitignored `.env` in the repo root.

## Citation

The overview figure in `assets/` is the paper's own.

```bibtex
@article{fei2026autoresearcheval,
  title   = {How Do Agents Fail on AutoResearch: End-to-End Diagnostic Evaluation
             on 100 Real-World Frontier Research Tasks},
  author  = {Fei, Yanlin and Liu, Nazhou and Yu, Xinmiao and Chen, Shaolong and
             Li, Lei and Thapa, Rahul and Ciobanu, Madalina and Mao, Qingqing and
             Das, Ritankar},
  journal = {arXiv preprint arXiv:2608.14905},
  year    = {2026}
}
```

## License

`agent-as-a-judge/` is MIT — see [`agent-as-a-judge/LICENSE`](agent-as-a-judge/LICENSE).
