# Autoresearch Agent Failure Taxonomy and Root-Cause Analysis

> A structured framework for failure modes of end-to-end autoresearch / AI-scientist agents.
> Organization: a 2D grid of **pipeline stages (A–F) × root mechanisms (M1–M5)**, plus a cross-cutting layer (X) for multi-stage dynamics.

---

## 1. Framework overview

### 1.1 Two axes

**Stage axis (rows)** — six stages of the research pipeline:

| Stage | Name | Meaning |
| --- | --- | --- |
| A | Ideation | Problem framing, direction choice, hypothesis generation |
| B | Retrieval & Synthesis | Literature retrieval, organization, citation |
| C | Execution | Experiment design, coding, running |
| D | Analysis | Interpreting results, inferring mechanisms |
| E | Writing | Manuscript writing, citations, method description |
| F | Review | Self-evaluation and peer-review loops |

**Root-cause axis (columns)** — five underlying mechanisms:

| Root | Name | Meaning |
| --- | --- | --- |
| M1 | Metacognition gap | Does not question its own frame or direction; lacks step-back |
| M2 | Fabrication / hallucination | Invents nonexistent results, methods, or citations |
| M3 | Grounding failure | Output decoupled from evidence / intent |
| M4 | Depth & completeness | Did something, but shallow or incomplete |
| M5 | Execution / engineering fault | Hard failures at the implementation layer |

**Cross-cutting layer (X)** — failures that do not sit in a single cell; they are cross-cell dynamics (see Section 4).

### 1.2 Stage × root matrix

Each cell lists failure-mode codes for that stage–root pair. Empty cells mean no failure mode identified yet (emptiness itself can be informative).

| Stage ↓ / Root → | M1 metacognition | M2 fabrication | M3 grounding | M4 depth | M5 execution |
| --- | --- | --- | --- | --- | --- |
| **A · Ideation** | A1 frame-lock | — | A3 hyp–ops mismatch | A2 un-calibratable direction | — |
| **B · Retrieval** | — | — | B3 citation decorrelation | B1 retrieval gap · B2 organization failure | — |
| **C · Execution** | C2 shortcut · C5 local-optimum | — | C4 agent–env interaction | C6 shallow execution | C1 impl. bugs · C3 robust-impl gap |
| **D · Analysis** | D2 bug-as-insight | D1 hallucinated results | D4 concl.–method disconnect | D3 CAWM / overclaim | — |
| **E · Writing** | — | E1 citation hallucination · E2 methodology fabrication | — | E3 revision degradation | — |
| **F · Review** | F1 review-score hacking | — | — | F2 uncaught halluc. · F3 over-acceptance | — |

### 1.3 Structural observations from the matrix

1. **M1 (metacognition gap) spans the widest vertically** — it appears from ideation through review (A1 / C2·C5 / D2 / F1). Lack of skeptical reasoning / step-back is likely a systemic root, not a local bug.
2. **M2 (fabrication) concentrates in D and E** — only when results and text must be produced is there something to invent. That distribution itself is informative.
3. **Empty cells flag gaps** — e.g. B×M1 is empty (is there “retrieval metacognition lock-in,” such as repeatedly searching similar queries without changing strategy?). Empty M5 cells for A/E/F are expected (those stages do not run code).

---

## 2. Stage-by-stage failure modes (A–F)

### A · Ideation

**A1 · Frame-lock** — root M1  
Locks onto an initial problem frame / methodological paradigm and cannot escape it, even when evidence argues otherwise. Conceptual-level failure: wrong direction, no revision.

- Related to C5 (local-optimum fixation): same root M1, different lock target — A1 locks the conceptual frame; C5 locks the execution path. They can occur independently.

**A2 · Un-calibratable direction selection** — root M4  
Chooses a research direction that cannot be checked against numbers / reference values; lacks grounding that would refuse un-calibratable goals.

**A3 · Hypothesis–operationalization mismatch (self-inconsistency)** — root M3  
The agent’s **own** stated problem / hypothesis cannot be adjudicated by the method, observable, or results it actually uses. The write-up may still sit inside the task’s premise+tension; the failure is internal: claims one scientific test, then measures something that would not decide that claim (e.g. hypothesizes an ion-migration barrier / long-term stability mechanism but only reports a short-scan hysteresis index and PCE, without measuring the claimed barrier or a stability quantity).

Do **not** treat mismatch with a hidden gold observable / paper metric name as A3 — agents are not given gold. True topic drift relative to the open question belongs under **X2** / **X5** when applicable.

### B · Retrieval & Synthesis

**B1 · Retrieval gap** — root M4  
Misses a large share of critical literature that should have been found.

**B2 · Organization / synthesis failure** — root M4  
Retrieves the right materials, but fails when organizing them into a structure (survey, taxonomy, classification). Independent of B1: B1 is “materials incomplete”; B2 is “materials present but poorly arranged.” Typical forms:

- **Sibling overlap**: same-level categories overlap with unclear boundaries.
- **MECE violation**: not mutually exclusive and/or not collectively exhaustive — omission or overlap.
- **Structural imbalance**: malformed trees; branches uneven in depth / density.

**B3 · Citation decorrelation** — root M3  
Architectural failure: when attaching citations to claims, the agent optimizes for semantic similarity in embedding space (e.g. cosine similarity) rather than logical evidential support (entailment). Citations become plausibility signals (“looks related”) rather than evidential links — format may be correct and sources real, yet they may not support the attached assertion.

- Distinct from E1: E1 is a problem with the citation object (nonexistent source / content mismatch); B3 is deeper — the **citation-generation mechanism** optimizes similarity rather than entailment. B3 is often a root; E1 is often an observable symptom.

**B4 · Noise-induced failure** — root M3 (or M4)  
Retrieved content exists, but the agent fails to prioritize the highest-signal evidence.

### C · Execution

**C1 · Implementation bugs** — root M5  
Coding errors cause experiments to fail or become untrustworthy.

**C2 · Shortcut reliance** — root M1  
Uses shortcuts instead of real solving.

**C3 · Robust-implementation gap** — root M5  
Cannot produce a robust implementation or finish a full closed loop from understanding to verified results. Agents are often stronger at planning / summarization than at robust implementation.

**C4 · Agent–environment interaction failure** — root M3  
Failures caused by the environment / tool interface, distinct from the agent’s own scientific mistakes.

**C5 · Local-optimum fixation / over-tuning** — root M1 — *provisional*  
The frame may be correct and the direction reasonable, but the agent gets stuck on a suboptimal point within that direction, repeatedly micro-tuning instead of exploring other execution paths. A micro-level execution counterpart of A1 frame-lock.

**C6 · Shallow / incomplete execution** — root M4 — *provisional*  
Appears finished, but is shallow: minimal demo treated as verification, missing ablations, single setting only. Sits between C3 (never finished) and D3 (finished but overclaims).

### D · Analysis

**D1 · Hallucinated results** — root M2  
Directly fabricates experimental numbers. Especially dangerous because forged results can read identically to real ones.

**D2 · Bug-as-insight reframing** — root M1  
Reframes a bug’s artifact as a scientific “discovery.”

**D3 · Correct Answer, Wrong Mechanism (CAWM) / overclaiming** — root M4  
The answer may be right, but the mechanism contradicts the agent’s own numbers; and/or overclaiming — extrapolating conclusions to untested regimes. “Did the agent overclaim?” maps here.

**D4 · Conclusion–methodology disconnect** — root M3  
The conclusion cannot be derived from the stated methodology. (Borderline with M1 if framed as failing to question one’s own reasoning chain.)

### E · Writing

**E1 · Citation hallucination** — root M2  
Two orthogonal subtypes:

- **Existence failure**: the cited source does not exist.
- **Faithfulness failure**: the source exists but does not support the claim.

**E2 · Methodology fabrication** — root M2  
Writes method details that were never actually performed.

**E3 · Revision-induced degradation** — root M4  
Counterintuitive but important: self-reflection / revision can *reduce* faithfulness (e.g. later drafts drop citations or weaken grounding).

### F · Review

**F1 · Review-score hacking** — root M1  
Optimizes for review scores rather than scientific quality.

**F2 · Uncaught hallucination in review** — root M4  
Review fails to catch hallucinations in the manuscript / results.

**F3 · Automated-review over-acceptance** — root M4  
Automated review accepts papers that contain many fabricated / unsupported claims that careful human review would catch.

---

## 3. Cross-cutting layer (X): multi-stage dynamics

These failures do not belong to a single stage × root cell; they are cross-cell dynamics:

**X1 · Error propagation / cascading**  
A root-cause failure amplifies into downstream symptoms along the pipeline. Often useful to model as a DAG (claims/actions as nodes; causal dependencies as edges).

**X2 · Intent hallucination**  
At the front of planning, misunderstands or only partially satisfies the user query / task, then runs the whole pipeline under the wrong understanding. The hallucinated object is the **task itself**, not a single fact.

**X3 · Skeptical-reasoning gap**  
Lacks skeptical reasoning: easily misled, confidently propagates error, does not stop to ask whether intermediate results or evidence support the conclusion. Cross-cutting expression of M1 — frame-lock, bug-as-insight, and accepting numbers that contradict common sense often share this root.

**X4 · Multi-agent orchestration failure**  
Inter-agent communication / division errors; hallucinations can be structurally integrated into plans, scripts, and writing, making real computation hard to distinguish from plausible fabrication.

**X5 · Goal drift / objective dilution** — *provisional*  
Starts with a correct understanding, then the objective gradually shifts / dilutes over a long chain so the final product only loosely matches the original goal. Distinct from X2 (wrong from the start) and A1 (locked, not drifting).

**X6 · Shallow reasoning / insufficient deliberation** — *provisional*  
Reasoning chain too short; no multi-step causal analysis; no alternative hypotheses; surface-level answers to complex questions. Broader than X3 (X3 = lack of skepticism specifically; X6 = insufficient depth in general).
