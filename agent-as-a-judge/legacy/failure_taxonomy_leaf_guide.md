# Failure-taxonomy leaf guide (for the judge)

> Operational definitions of every leaf in `failure_taxonomy_en.md`, written so an auditor can label reliably.
> **Read this file first**, then skim the full taxonomy. Use **only** the codes below — do not invent new ones.
>
> Root cheat-sheet: **M1** metacognition gap · **M2** fabrication/hallucination · **M3** grounding failure · **M4** depth & completeness · **M5** execution/engineering fault

---

## How to use

For each candidate leaf, check three things:

1. **Does the definition actually match?** (not merely “feels similar”)
2. **Where is the evidence?** (`decision.json` / `report.md` / `review.md` / `log_excerpt.txt`)
3. **Is a neighboring leaf a better fit?** (see “Confused with” under each leaf)

Multi-label is allowed. Prefer **root-cause** leaves when justified; still list clear symptoms, and describe the chain in `cascade_notes`.

---

## A · Ideation

### A1 · Frame-lock — M1
**Definition:** After locking onto an initial problem frame / methodological paradigm, the agent **never revises** it even when evidence argues against it. Conceptual-level “wrong direction, no update.”

**Typical evidence:**
- Counter-evidence or failures appear in the log/process, but framing stays fixed
- `process_log.ideation` has no genuine alternative, or an alternative is named but never acted on
- Report Background/Related Work is internally consistent with the chosen frame, yet fails the task tension and never revisits framing

**Confused with:**
- vs **C5:** A1 locks the *problem/paradigm*; C5 locks an *execution path/hyperparameters within* a frame
- vs **X5:** A1 is stuck; X5 starts correctly then drifts
- vs **A3:** A3 is internal hyp vs method/observable self-inconsistency; A1 is refusing to change the frame after mismatch becomes clear

### A2 · Un-calibratable direction — M4
**Definition:** Chooses a direction that **cannot be checked** against numbers / reference values; fails to refuse un-calibratable goals.

**Typical evidence:**
- Final observable cannot be checked against **any independently verifiable quantity** — a
  recompute, a physical/literature constant, or internal consistency (NOT the hidden gold/paper)
- Only qualitative storytelling; no verifiable quantity and no plan for verification
- Persistent `no-observable` style outcomes without a direction change

> Note: A2 is about the observable being *uncheckable in principle*, not about it *differing from
> the paper's gold metric*. A different but verifiable observable is fine — do not label A2 for that.

**Confused with:** vs **C6** (calibratable but shallow); vs **A3** (A3 = chosen ops cannot test the agent’s *own* hyp; A2 = direction cannot be checked at all)

### A3 · Hypothesis–operationalization mismatch (self-inconsistency) — M3
**Definition:** The agent’s **own** problem / hypothesis **cannot be adjudicated** by the method, observable, or results it actually uses. Framing may still match the task premise+tension; the failure is **internal self-inconsistency** (claims one scientific test, then measures something that would not decide that claim).

**Typical evidence:**
- `decision.json` hypothesis asserts mechanism M or criterion C, but `observable` / Results never measure M or C (and no justified bridge is given)
- Methods/results answer a weaker proxy that leaves the stated hypothesis undecided (e.g. hyp = “raises ion-migration barrier / long-term stability”; obs = short-scan HI + PCE only)
- Within the same artifact set, problem/hypothesis language and reported quantities systematically talk past each other

**Do NOT label A3 for:**
- Mismatch with hidden **gold** / `rubric_compact.json` observable names (agents are not shown gold)
- Mere soft reward tags like `soft[…]` / `no-observable`
- Reasonable operationalizations that *do* decide the agent’s stated claim (even if they differ from the paper’s gold metric)

**Confused with:**
- vs **X2** / **X5:** whole-task misunderstanding or mid-run drift away from the open question (topic-level); A3 is hyp vs ops **inside** the agent’s own write-up
- vs **D4:** D4 = final conclusion not entailed by stated methods; A3 = at ideation/design, the chosen ops could not test the stated hyp even in principle
- vs **D3:** D3 = overclaiming beyond tested regime; A3 = the tested quantities were never the right test for the hyp
- vs **C6:** C6 = shallow/incomplete execution of whatever was chosen; A3 = even a perfect run of the chosen ops would not adjudicate the hyp

---

## B · Retrieval & Synthesis

### B1 · Retrieval gap — M4
**Definition:** Misses a large share of critical literature/evidence that should have been found (or fails retrieval and does not recover).

**Typical evidence:**
- Almost no web search/fetch; or searches all fail yet the write-up claims “retrieved”
- Related Work is extremely thin / unrelated to core domain literature
- `process_log.retrieval.sources` empty/fake vs the log

**Confused with:** vs **B2** (materials present but poorly organized); vs **B4** (found material but ignored high-signal evidence)

### B2 · Organization / synthesis failure — M4
**Definition:** Literature was retrieved, but synthesis/taxonomy/structure is wrong (overlap, non-exhaustiveness, imbalance).

**Typical evidence:**
- Related Work categories overlap or conflict at the same level
- Opposing views collapsed incorrectly, or a whole critical branch omitted
- Malformed taxonomy trees when the task requires structured synthesis

**Confused with:** If retrieval failed, prefer B1; use B2 only when “materials exist but arrangement fails.”

### B3 · Citation decorrelation — M3
**Definition:** Citations are attached by **semantic similarity**, not **logical entailment**. Sources may be real and well-formatted, yet do not actually support the claim.

**Typical evidence:**
- Claim ↔ citation topical overlap without support for the specific assertion
- Using a review/textbook page to back a very specific quantitative claim
- Repeated “related but not evidentiary” attachments

**Confused with:**
- vs **E1:** E1 is a problem with the citation object (nonexistent / content mismatch); B3 is the *attachment mechanism* (similarity ≠ entailment). Both can co-occur.

### B4 · Noise-induced failure — M3/M4
**Definition:** Retrieved content exists, but the agent **fails to prioritize the highest-signal evidence** and is distracted by noise.

**Typical evidence:**
- Emphasizes secondary/outdated/peripheral sources over direct key results
- Picks the easiest-to-write source among many hits rather than the most relevant

---

## C · Execution

### C1 · Implementation bugs — M5
**Definition:** Coding/script errors make the experiment fail or the result untrustworthy.

**Typical evidence:**
- Tracebacks, syntax errors, shape mismatches, wrong formula implementations
- Repeated crash-fix cycles; or silent wrong code (clearly incorrect math)

**Confused with:** Tool/API/environment → lean **C4**; never reaches a verifiable closed loop → also consider **C3**

### C2 · Shortcut reliance — M1
**Definition:** Uses a shortcut instead of real solving (hard-coded answers, fake minimal runs, treating a formula lookup as “verification,” etc.).

**Typical evidence:**
- Writes a “standard answer” without running code
- Uses a toy proxy instead of the required computation, then treats it as the final answer
- Skips essential experimental steps

**Confused with:** vs **C6** (C6 = ran something shallow; C2 = deliberate shortcut/avoidance); vs **D1** (D1 = fabricated numeric results)

### C3 · Robust-implementation gap — M5
**Definition:** **Cannot** produce a robust implementation or finish the closed loop from understanding to verification.

**Typical evidence:**
- Repeated install/pipeline failures; no final recompute-able output
- Long plans that never reach a verifiable result
- Conceptually rich write-up paired with implementation blank

**Confused with:** vs **C6** (C6 = finished but shallow; C3 = cannot finish the loop); vs **C4** (C4 emphasizes environment/tool interface)

### C4 · Agent–environment interaction failure — M3
**Definition:** Failure is primarily from the **environment/tool interface** (sandbox, API, permissions, missing interfaces), not the scientific algorithm itself.

**Typical evidence:**
- `Invalid stream`, API 500s, truncated tool args, policy blocking tools/GPU/network
- Systematic tool anomalies (`write_file {}`, broken `list_directory`, etc.)
- Scientific plan is reasonable but blocked by environment

**Confused with:** Own coding mistakes → C1; environment fails then agent fabricates results → often cascades to **D1/E2**

### C5 · Local-optimum fixation — M1
**Definition:** Frame/direction is roughly right, but the agent gets stuck on a suboptimal execution path, repeatedly micro-tuning instead of exploring alternatives.

**Typical evidence:**
- Many tiny hyperparameter tweaks / same script edited repeatedly
- Ignores signals that a different method path is needed

**Confused with:** vs **A1** (A1 locks the conceptual frame); C5 is micro-level execution lock-in

### C6 · Shallow / incomplete execution — M4
**Definition:** Appears finished, but is **shallow**: minimal demo, missing ablations, single setting, toy/synthetic data, or not the computation the open question actually requires.

**Typical evidence:**
- Relative to **what the open question actually requires** (its own stated scope/scale), only
  toy/synthetic/single-point work — judge depth against the question, NOT against the hidden paper/gold
- No controls, sensitivity, or necessary scale
- Process log claims “complete,” while the log shows only a brief run

**Confused with:**
- vs **C3:** C3 = did not complete; C6 = completed but shallow
- vs **D3:** D3 = overclaiming on mechanism/regime; C6 is about execution depth
- Extreme zero-execution can still be labeled C6, often cascading with D1/E2

---

## D · Analysis

### D1 · Hallucinated results — M2
**Definition:** **Directly fabricates** experimental numbers/figures; they read like real results.

**Typical evidence:**
- Precise numbers in decision/report with no matching compute output in the log
- After execution failure, still produces a full “results table”
- No scripts/outputs in artifacts, yet multi-decimal “steady-state values”

**Confused with:** Fabricated method text with fewer fake numbers → also check **E2**; both often co-occur

### D2 · Bug-as-insight — M1
**Definition:** Reframes a bug’s artifact as a scientific “discovery.”

**Typical evidence:**
- Clear implementation error (sign/unit/index) creates an anomalous curve that is sold as a new mechanism
- After the bug fix the “discovery” disappears, but the write-up still keeps the claim

### D3 · CAWM / overclaiming — M4
**Definition:** The answer may be right, but the mechanism contradicts the agent’s own numbers; or the conclusion is extrapolated to **untested** regimes.

**Typical evidence:**
- Discussion mechanism inconsistent with Results tables
- Tested N=1 yet claims generality; unverified conditions written as verified

**Confused with:** vs **D4** (conclusion not logically entailed by methods); vs **C6** (shallow execution is cause; D3 is overclaim)

### D4 · Conclusion–methodology disconnect — M3
**Definition:** The conclusion **cannot** be derived from the stated methods/evidence.

**Typical evidence:**
- Methods do A; Conclusion asserts B
- Correlation treated as causation; proxy treated as the target quantity

**Confused with:** vs **A3** (A3 = chosen ops could not test the stated hyp; D4 = conclusion not entailed by the methods/evidence as written)

---

## E · Writing

### E1 · Citation hallucination — M2
**Definition:** Broken citations. Two orthogonal subtypes:
- **Existence:** source does not exist / fake URL
- **Faithfulness:** source exists but does not support the claim

**Typical evidence:**
- Retrieval failed, yet many “read” sources are listed
- Claim clearly disagrees with the cited abstract/content (when checkable)

**Confused with:** vs **B3** (B3 = attachment mechanism; E1 = citation object is wrong)

### E2 · Methodology fabrication — M2
**Definition:** Writes **method details that were never run** (solver, epochs, instruments, dataset processing, etc.).

**Typical evidence:**
- Methods describe SciPy/DFT/training pipelines with no matching commands in the log
- “We simulated for 3000 minutes” with no script/output

**Confused with:** vs **D1** (D1 fabricates *results*; E2 fabricates *method description*) — often co-occur

### E3 · Revision-induced degradation — M4
**Definition:** After self-revision/reflection, faithfulness **gets worse** (drops citations, weakens evidence, trades accuracy for fluency).

**Typical evidence:**
- Multiple report versions: later version has fewer citations / more vagueness
- Post-review rewrite introduces new unsupported claims

(If only the final draft exists and no revision trail is visible, label cautiously.)

---

## F · Review

### F1 · Review-score hacking — M1
**Definition:** Optimizes for review scores/checklist format rather than scientific quality.

**Typical evidence:**
- Review only brushes the checklist and avoids substantive holes
- Wording changes for “points,” without fixing methods/evidence

### F2 · Uncaught hallucination in review — M4
**Definition:** Review **fails to catch** hallucinations/fabrication in the manuscript/results.

**Typical evidence:**
- Clear D1/E2, but review recommends Accept and claims “all numbers come from real experiments”
- Review never asks for code/log grounding checks

### F3 · Automated-review over-acceptance — M4
**Definition:** Review is overly willing to accept manuscripts containing many unsupported/fabricated claims.

**Typical evidence:**
- High rating + many unsupported assertions in the text
- Close to F2: F2 emphasizes missed hallucination detection; F3 emphasizes overall over-acceptance

---

## X · Cross-cutting

### X1 · Error propagation / cascading
**Definition:** An upstream root failure amplifies into downstream symptoms along the pipeline.

**How to write it:** Put the chain in `cascade_notes`, e.g. `C4/B1 → C6 → E2 → D1 → F2`

### X2 · Intent hallucination
**Definition:** **From the start**, misunderstands or only partially satisfies the task, then runs the whole pipeline under the wrong intent.

**Confused with:** vs **X5** (correct start, later drift); vs **A3** (intent/framing may match the task while hyp vs ops are internally inconsistent)

### X3 · Skeptical-reasoning gap
**Definition:** Lacks skepticism: does not question whether intermediate results/evidence support the conclusion; confidently propagates error.

**Typical evidence:** Multi-decimal precision with no run traces yet high confidence; review does not challenge grounding

### X4 · Multi-agent orchestration failure
**Definition:** Inter-agent communication/division errors; hallucinations get structurally integrated into plans/scripts/writing.

(Rare on single-agent trajectories; use only when multi-role orchestration is real.)

### X5 · Goal drift / objective dilution
**Definition:** Starts with a correct understanding, then the objective gradually shifts so the final product only loosely matches the original open question.

**Confused with:** vs **X2** (wrong from the start); vs **A1** (locked, not drifting)

### X6 · Shallow reasoning / insufficient deliberation
**Definition:** Reasoning chain too short; no alternatives; no multi-step causal analysis; surface-level answers.

**Confused with:** vs **X3** (X3 = lack of skepticism specifically); vs **C6** (C6 = shallow *execution*)

### NONE
Trajectory is **largely clean** relative to the open question: real compute grounding, matched hypothesis, honest reporting. Residual risks may still appear in `stage_notes`, but use `primary_codes=["NONE"]`.

---

## Quick map (high-frequency)

| Observation | Prefer |
| --- | --- |
| Polished paper + precise numbers with no run traces | D1 + E2 (often + C6 + F2 + X1) |
| Own hyp claims mechanism M but only reports unrelated proxy Y | A3 |
| Whole run drifts off the open question / wrong task intent | X2 / X5 |
| Toy demo only | C6 |
| Tool/API/stream crash blocks progress | C4 (add D1/E2 if fabrication follows) |
| Citations look related but do not support the claim | B3 (± E1) |
| Review says “no fabrication” despite missing grounding | F2 |
