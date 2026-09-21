# Pipeline 1 — papers to tasks

Published papers in, runnable task directories out. This is panel **a** of the study
figure: 5,878 candidate papers across nine fields reduced to 100 tasks in seven domains.

```
crawl ──▶ fetch ──▶ mine ──▶ export ──▶ materialize
corpus/    corpus/   tasks/   tasks/     tasks/
```

| Directory | Step |
|---|---|
| [`corpus/`](corpus/) | find and grade candidate papers, then pull their PDFs |
| [`tasks/`](tasks/) | extract the seven fields, select tasks, write task directories |
| [`rollout/`](rollout/) | the local discovery-episode driver (see [`../harness/README.md`](../harness/README.md)) |

Every script resolves the repository root itself, so run them from anywhere. Generated
data goes to `pipelines/output/` (gitignored).

## Crawl and grade a corpus

Tiering is automatic from OpenAlex signals — no hand-curated whitelist: **bronze**
(breadth), **silver** (the workhorse for extraction), **golden** (elite venue *and*
field-leading impact, weighted up downstream). Recent papers are graded on venue and
institution; citations only ever promote, so a 2026 paper is never punished for having
no citations yet.

```bash
python pipelines/corpus/crawl_topic_set.py --per-topic 60 --set-name diverse_v1
python pipelines/corpus/fetch_corpus_pdfs.py --set diverse_v1 --unpaywall --s2 --core
```

`crawl_tiered_corpus.py` deepens a single topic instead of spreading over a set;
`backfill_corpus_tiers.py` retrofits tiers onto a corpus crawled before tiering existed.

## Extract the seven fields

One LLM pass per paper returns the seven fields, the key claims, and one novelty-move
label (`consensus-overturn`, `method-correction`, `new-regime`, `mechanism`,
`scaling-relation`, `reconciliation`, `incremental`). It is a reconstruction, not an
evaluation: the model reports what the paper states and never invents numbers.

```bash
python pipelines/tasks/discovery_pattern_mine.py --zip corpus.zip   # -> one record per paper
```

Each field has a job. **Premise** and **Tension** become the agent's query. **KeyClaims**
and **Conclusion** are withheld as gold. **Experiment** and **Method** gate whether a
paper can become a runnable task at all — a purely wet-lab experiment or a single-step
lookup is dropped. **Motivation** and **Conclusion** decide whether the task is
*open-ended discovery* or *target-anchored optimization*.

The extraction prompt itself lives in
[`reconstruct/discovery_pattern.py`](../reconstruct/discovery_pattern.py) and is the one
reproduced in the paper's appendix.

## Materialize task directories

```bash
python pipelines/tasks/export_discovery_tasks.py        # -> pipelines/output/discovery_tasks.jsonl
python pipelines/tasks/materialize_discovery_tasks.py \
    --jsonl pipelines/output/discovery_tasks.jsonl \
    --template path/to/env_template \
    --out pipelines/output/discovery_tasks_v6 --prompt-version v6_report_review
```

`--template` points at a directory holding `environment/Dockerfile`, copied into every
task. Each task directory holds `instruction.md` (premise + tension, an open decision
schema, **no gold leaked**), a `task.toml`, and the per-paper rubric the verifier scores
against.

> The rubric evaluator and the QE relax driver are copied into each task's `tests/`
> directory when present, but they ship with the task-data release rather than with this
> repository, so directories materialized from this repo alone have no `tests/evaluate.py`.

### Prompt versions

`instruction.md` carries the six-stage process the agent is asked to work through:
**A** Ideation & Planning → **B** Retrieval & Synthesis → **C** Execution &
Implementation → **D** Analysis & Interpretation → **E** Writing & Documentation →
**F** Self-Verification & Review. Six wordings live side by side in `PROMPT_VERSIONS` so
they can be A/B'd on an identical task set; each one's rationale is recorded in the
comment above it.

| `--prompt-version` | |
|---|---|
| `v6_report_review` | **the prompt used for the paper's 800 rollouts**, reproduced in the appendix: `decision.json` first, then a narrative `report.md` with a `## Peer Review` section |
| `v5_report_review` | same ordering fix, no peer-review section |
| `v4_report_review`, `v3_failure_taxonomy` | intermediate: added `process_log`, then `report.md` |
| `v2_af` | domain-general six-stage prompt, no `process_log` (the script's default) |
| `v1_domain_specific` | original chemistry/materials wording, no explicit process structure |
