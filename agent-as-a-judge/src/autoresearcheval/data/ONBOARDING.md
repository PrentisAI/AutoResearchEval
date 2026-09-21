# Agentic-Research Trajectory Deep-Dive Framework (ONBOARDING)

> Purpose: for every trajectory in an agentic-research benchmark run, produce a
> code-level, reproducible, fair-credit `analysis.md` that identifies the failure
> patterns actually present in it.

---

## 0. The one-line goal

Not a restatement of what the agent wrote, but an excavation of its execution log,
a line-by-line audit of the computation/simulation/retrieval code it actually ran, and
a judgment of whether its conclusion actually holds — while giving full, honest credit
for what it got right. Verification leans on "read the code + numerical sanity checks";
full re-execution is selective (see §2).

---

## 1. Data sources and directory conventions

One run typically has three kinds of artifacts (paths are illustrative; only the paths
change between runs):

| Kind | Path (example) | Contents |
|---|---|---|
| Execution log | `<run>/traj/<task_id>.json` → the harness log field inside it | The agent's full reasoning + every shell/tool call + output. **The main excavation site.** |
| Deliverables | `problem_readme.md`, `data_description.md`, `result.json`, `submissions.jsonl`, `evaluator/`, `verification.md`, `agent_code/` | What the agent's rollout actually delivered, plus the scoring service's own source and the harness's real execution status — see the Evidence Package table below. |
| Your output | `<corpus>/<model>/<task_id>/analysis.md` | One deep-dive per task. |

**Whether a task is already analyzed**: check whether its output directory exists and
spot-check whether `analysis.md` clears the bar (§3).

### Evidence package provided to the judge

| Artifact | Role |
|---|---|
| `problem_readme.md`, `data_description.md` | The task statement and data schema the rollout itself was given. |
| the rollout's execution log | The rollout's complete execution log (the primary evidence base). |
| `result.json` | Container-level execution status (status/duration/returncode) only — **explicitly not a quality signal**. |
| `submissions.jsonl` | Every call the rollout made to the benchmark's own scoring service, with the score returned each time (when present). |
| `evaluator/` | The scoring service's real source code, copied verbatim from the benchmark definition — lets you check the rollout's method against the exact metric being computed, not just its stated name. |
| `verification.md` | Human-written notes (paper provenance, held-out set construction, oracle score ceiling) where available. Read this first — it often names the single most common systematic error on the task. |
| `agent_code/` | The rollout's actual final container filesystem (its real `workspace/`), capped at 60 files / 1 MB per file / 20 MB total, source-suffix filtered. |
| Sealed ground truth & raw task data | Referenced by real host path (mounted read-only) rather than copied — these can reach tens of GB per task — query on demand, never read in full. |

This combination — execution log, delivered filesystem, and sealed ground truth — is
what makes grounding failures (e.g. circular validation on a synthetic substrate)
detectable at all; none of them leaves a signature in the final report alone.

---

## 2. Workflow (four fixed steps per task)

```
① grep the code  →  ② read the key spans  →  ③ sanity-check (every task) + selective re-run  →  ④ write analysis.md
```

1. **Grep to locate the real computation**: in the execution log, grep for
   `def |import|<model keywords>|numbers|formulas`, filter out reasoning noise
   (`grep -ivE "I need|I'll|maybe|let me|..."`), and find the DGP / solver / retrieval
   calls the agent actually ran.
2. **Read the key spans in full**: `sed -n` out the DGP, scoring logic, and result
   numbers in their entirety — a few dozen lines, not a summary of them.
3. **Sanity-check every task (cheap) + selective re-run (layered, never exhaustive)**:
   - **Every task, mandatory, seconds, mental arithmetic**: a numerical sanity check —
     is the headline number the right **order of magnitude / unit / within a few×** of
     what's expected? Examples that should raise an eyebrow on sight: a protein–DNA
     interface burial reported as 47 Å² where >1000 is expected (almost certainly an
     nm²/Å² unit error); a water-dimer interaction energy of −0.16 where ∼−5 is
     expected; a nitrobenzene excitation energy of 1.38 eV where ∼2.6 is expected.
     **Most magnitude/unit bugs are caught right here, no re-run needed.**
   - **Also check "too good / extreme" results, not only "wrong" ones**: a result that
     looks nearly perfect (metric ≈100%, error ≈0) is itself a warning sign — decompose
     it and check whether it's propped up by a single convenient assumption or a
     credit/cancellation artifact.
   - **Read the code + trace every number to its source (mandatory)**: circular DGPs,
     observable mismatch, tautologies, strawman comparisons — these are found entirely
     by reading plus the sanity check above. This is the bulk of the value.
   - **Selective re-run (minutes, only when triggered)**: (a) a number is suspicious and
     can't be resolved by reading alone (e.g. "does feedback actually have a causal
     effect?" → run fb=0 vs 0.02); (b) the conclusion depends on exact reproduction;
     (c) the dependency is **light** (numpy/pandas/sklearn — you do not need to fight
     PySCF/scanpy/torch environments the way the agent did). When you do re-run,
     **rebuild only the core few lines** (e.g. a timing formula is ~6 lines), never the
     whole pipeline.
   - **Skip the re-run**: dependency is heavy / the conclusion is already clear from
     reading / `reason=judge_unavailable`·`no_decision` (an infrastructure problem — a
     re-run wouldn't change the verdict).
   - Note: many of the sharpest findings are **pure analytical derivations** — worked
     out on paper, zero re-execution. Insight is not synonymous with re-running.
4. **Write `analysis.md`** per the standard in §3 and the structure in §5.

> **Key point**: most trajectories contain **multiple code versions** (false starts,
> abandoned fallbacks). You must **follow the evolution to completion and pin down
> which version the number in the final delivered `decision.json`/report actually came
> from** before judging the conclusion (see the Iron Rules in §6).

---

## 3. The depth standard (three dimensions, all required)

1. **Depth**: get into the code the agent actually ran — do not restate `decision.json`.
   Dig out circularity, bugs, observable mismatch, strawman comparisons.
2. **Every issue is a paragraph-length mechanism analysis**: each issue =
   **mechanism (what it concretely did) + why it's harmful + the honest/charitable
   reading (fair credit) + evidence (a line number or a number)**. Not a one-line bullet.
3. **Breadth**: computational/buggy trajectories should cover **~25–40 issues**,
   spanning all six lifecycle stages plus the cross-stage layer (§5); add a
   **Sentence-by-Sentence Checklist** (every key claim in `report.md`/`review.md`,
   marked ✅/⚠️/❌).

**Calibration hard floor**: `wc -m analysis.md` character count ÷ issue count **≥ 280**
(target ≥ 290). Below that, each issue is too thin.

**Do not pad to hit a count**: genuinely limited trajectories (a single catastrophic
bug / `no_decision` / `judge_unavailable` / a pure tautology) naturally have fewer
issues — 16–20 is fine **as long as each one is deep** (≥350 characters/issue). Depth
is prioritized over hitting a raw count.

**Per-issue writing standard, beyond the quota**:
- At least one verifiable anchor per issue — a concrete number, a log line index, a
  file name, or a code identifier.
- Every issue also carries a one-line trailer `[stage: <A–F,X> | root cause: <grounding
  | depth | integrity | robustness>]`, giving the issue's two coordinates on the
  lifecycle-stage and root-cause axes. Downstream aggregation depends on this trailer
  being present and well-formed on every issue.
- A minimum length (≥200 characters); one-sentence bullets get merged into a fuller
  issue, not counted separately.
- No large verbatim pasting from `problem_readme.md`/`result.json` — pasting is not
  analysis and is penalized as padding.
- No templated/recycled phrasing across issues — every issue must state a fact unique
  to that specific rollout.

---

## 4. Disambiguating the scoring `reason` (the easiest trap to fall into)

`reward=0` does **not** necessarily mean poor quality. Look at `reason`:

| `reason` | Meaning | Quality signal? |
|---|---|---|
| `judge_unavailable` | Scoring infrastructure was down | ❌ **No** — the rollout may be excellent |
| `no_decision` | Malformed/incomplete decision artifact (missing fields, `result` is null, JSON syntax error) | ❌ **No** — a delivery failure, not a science failure |
| `soft[<observable>]` | Genuine score against the gold observable | ✅ **Yes** — a real quality signal |

This taxonomy is pool-specific — an open-ended pool with no reward/reason field has no
such disambiguation to make; quality there is established purely from internal
evidence (code, log, numerical self-consistency, `evaluator/` source logic).

When writing the analysis, always make this distinction explicit: is `reward=0`
really "infrastructure/delivery," or is it "science quality"?

A related, high-frequency failure to check explicitly is **observable mismatch**: the
rollout computes a plausible, correctly executed proxy quantity that simply isn't the
one the benchmark scores against (e.g. computing group delay when the gold metric is
efficiency; computing contact probability when the gold metric is chain length),
which yields zero credit regardless of code quality.

---

## 5. The `analysis.md` skeleton (stages A–F plus the cross-stage layer X)

Fixed skeleton (headings reproduced verbatim):

```
# <title: task + one-line verdict>
> Core Verdict (2–4 sentences, the 2–3 sharpest findings)
## Metadata (task/model/harness, execution status, gold vs. agent observable)
## Trajectory Arc (ideation → retrieval → execution → result → conclusion, one narrative paragraph)
## Credit Due (genuine strengths — required, not optional)
## A. Ideation & Planning
## B. Retrieval & Synthesis
## C. Execution & Implementation (heaviest stage)
## D. Analysis & Interpretation
## E. Writing & Documentation
## F. Self-Verification & Review (guard against being fooled by an agent's own confident self-diagnosis)
## X. Cross-Stage Dynamics (error propagation, goal drift, right-for-the-wrong-reason outcomes)
## Sentence-by-Sentence Checklist (every key claim in the rollout's final report, marked pass/partial/fail)
## Numerical Grounding Notes (what was independently re-derived, and the result)
## Retraction / Correction Log (honest record of any self-corrected misjudgment)
## One-Line Verdict
```

Every stage section should both point out problems and mark where credit is due. `X`
is for dynamics that don't belong to a single stage — cascading error propagation
across stages, goal drift over the course of the trajectory, or a target metric hit
through a hidden bug/leak/luck rather than the claimed mechanism.

---

## 6. Iron rules (hard-won lessons — non-negotiable)

1. **Follow code evolution to the final delivered artifact.** Rollouts frequently
   contain multiple superseded versions (false starts, abandoned fallbacks). Never
   indict the final conclusion using a version the rollout itself discarded; when the
   final report/review explicitly states what it did, prefer that over a misleading
   earlier draft.
2. **Sanity-check every task; re-run selectively, never exhaustively.** A near-zero-cost
   order-of-magnitude/units check catches most bugs (see the worked examples in §2). Full
   re-execution is reserved for cases where a number is both suspicious and unresolvable
   by inspection, and cheap to reproduce. Insight is not synonymous with re-running —
   many of the sharpest findings are pure analytical derivations with zero re-execution.
3. **Give credit where due.** Honestly reported null results, genuine mechanistic
   modeling, and self-caught bugs are real strengths that must be written up with the
   same rigor as failures.
4. **Retract and log honestly.** If you (the judge) misjudged something upon further
   reading, record the retraction and the lesson explicitly rather than silently editing
   it away. This project has gone through multiple such corrections itself — always
   caught by following the code and reading the final artifact, never by assumption.
5. **Judge each trajectory on its own terms.** No cross-trajectory template conclusions.
6. **Check for answer contamination.** If a rollout fetches the source paper's full text
   before modeling, its apparent "independent replication" of the paper's conclusion may
   simply be reading the answer first; any gold-adjacent fact that appears in text the
   rollout is shown to have already read cannot be credited as an independent finding
   (divergence from the paper, not mere disclosure, is what counts as evidence of
   independence).
7. **Judge retrieval sufficiency, not only retrieval honesty.** A separate failure mode
   from fabrication/contamination is simply not searching for information the gold
   observable required (e.g. skipping a dataset search and using "no data" as an excuse
   to fall back to a synthetic model). This failure often disguises itself as a
   downstream ideation defect.
8. **Verify every citation before crediting "honest, no fabrication."** Grep each cited
   DOI/arXiv ID/title against the actual retrieval log; an ID that never appears in a
   real search result but shows up in the final report is a hallucinated citation, not a
   formatting artifact. Separately, distinguish "read the full text" from "saw a
   title+snippet only" — the latter has grounded existence but is not full verification.
9. **Don't be disarmed by a rollout's own eloquent self-diagnosis.** Rollouts frequently
   name their own core defect in a review paragraph and then ship the finding unchanged.
   "Identified the mechanism" ≠ "addressed it." Credit for a review stage is recorded
   only against what the rollout independently found *and then acted on* — naming a flaw
   without correcting it is a problem to flag, not a strength to credit.

---

## 7. Ready to paste into an agent session

> I'm doing a deep trajectory audit of one agentic-scientific-discovery rollout. The
> data is at `<your run path>`: the execution log, the deliverables
> (`decision.json`/`report.md`/`review.md` or the evidence package in §1), and the
> scoring summary (reward + reason), if this pool has one.
> Please follow this framework's rules to write an `analysis.md` for `<task_id>`: get
> into the log and find what it actually ran; do a numerical sanity check on every task
> (order of magnitude/units/off-by-how-much); only re-run a core few lines in a scratch
> directory when a number is suspicious, unreadable from the code, and the dependency is
> light; write it in the six-stage-plus-X structure; write every issue as a full
> paragraph (mechanism + why it's harmful + fair credit + line-number evidence) with a
> `[stage | root cause]` trailer; add a Sentence-by-Sentence Checklist; disambiguate
> whether the reward/reason reflects infrastructure/delivery or science quality; give
> full credit where it's due; keep characters/issues ≥ 280. Follow multi-version code to
> the final delivered artifact.
