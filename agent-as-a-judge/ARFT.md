# ARFT — the AutoResearch Failure Taxonomy

45 patterns on two axes: the **stage** where a failure surfaces, and the **root cause** of
why it happens. Every pattern maps to exactly one pillar.

The source of truth for the code list is
[`patterns.py`](src/autoresearcheval/patterns.py); the operational guide handed to
the classifier — scoring rubric, discrimination rules for easily confused patterns, and a
Do-NOT-label list — is [`arft_guide.md`](src/autoresearcheval/data/arft_guide.md). Reread the
latter if you retarget this at a different kind of trajectory.

## Root-cause pillars

| Pillar | Core failure focus | Patterns | Share of hits |
|---|---|---|---|
| **P1 · Grounding & Faithfulness** | Claims disconnect from the code, data, logs, or literature that should license them | A.6, B.1, B.2, B.5, C.3, D.1, D.4, D.6, E.1, E.4, F.6, X.6 | 31.0% |
| **P2 · Cognitive Depth & Adaptability** | Shallow reasoning and search, passive self-critique, inability to re-plan | A.1, A.3, B.4, B.6, C.6, C.7, D.5, F.1–F.4, X.3, X.7 | 27.6% |
| **P3 · Integrity & Alignment** | Metric hacking, shortcut reliance, concealed failure, conclusions fixed in advance | A.2, A.5, C.1, C.2, D.2, D.3, D.7, E.2, E.3, F.5, X.2, X.4, X.5 | 33.5% |
| **P4 · Engineering Robustness** | Numerical faults, unhandled runtime errors, broken CLI/OS interaction | A.4, B.3, C.4, C.5, C.8, X.1, X.8 | 7.9% |

The four pillars roll up to a single systemic root cause: **metacognitive deficit**.

In the per-issue `[stage: <A-F,X> | root cause: <word>]` trailer that every analysis
carries, the pillars appear as the lowercase words `grounding`, `depth`, `integrity` and
`robustness` — the mapping is `ROOT_CAUSE_WORD` in `patterns.py`.

## Lifecycle stages

| Stage | | Patterns |
|---|---|---|
| **A** | Ideation & Planning | 6 |
| **B** | Retrieval & Synthesis | 6 |
| **C** | Execution & Implementation | 8 |
| **D** | Analysis & Interpretation | 7 |
| **E** | Writing & Documentation | 4 |
| **F** | Self-Verification & Review | 6 |
| **X** | Cross-cutting meta-failures | 8 |

## What the audit found

Auditing all 800 trajectories yields 12,712 hits. The three cognitive pillars account for
92.1% of them; engineering robustness for 7.9%. The single most frequent pattern is
**F.4 · Uncorrected Self-Awareness** — the agent identifies a severe flaw during its own
review and ships anyway — present in 82.5% of analyses.

The evidence that would refute most failures is already sitting in the agent's own run
directory; the comparison is simply never performed.

> **Note on numbering.** This taxonomy is the *dotted* one (`A.1` … `X.8`). An older
> undotted taxonomy (`A1` … `X6`) exists and collides in meaning — there `C1` might mean
> "impl bugs", here `C.1` is "Circular Validation & Shortcut Reliance". Never mix codes
> from the two systems in one table.
