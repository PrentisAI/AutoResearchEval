"""Materialize discovery_tasks.jsonl -> one rLLM task dir per paper (GENERALIZED, 2026-06-30).

REWRITE: the old version baked a CO/metal-slab decision schema + an adsorption-only reward into
every task. This version is general — each task carries the PAPER's own rubric (premise, tension,
conclusion, observable, gold) and an OPEN decision schema; the agent models whatever system the
paper is actually about. Reward = examples/discovery_rubric_evaluate.py (observable-routed
correctness + the 4-dim rubric). recompute is one backend, not the whole reward.

Per task dir <task_id>/:
  task.toml            - [metadata] + [verifier] module="tests.evaluate" + [agent]
  instruction.md       - generalized goal (premise+tension, open decision.json, NO gold leaked)
  tests/task_meta.json - the per-paper RUBRIC the verifier scores against
Shared at dataset root tests/: evaluate.py (rubric reward) + recompute_tools.py + qe_relax.py.

Run:
  python examples/materialize_discovery_tasks.py \
     --jsonl examples/output/discovery_tasks.jsonl \
     --template <dir with environment/Dockerfile> \
     --out examples/output/discovery_tasks_v2 \
     [--prompt-version v2_af]

Prompt versions (PROMPT_VERSIONS below) are kept side by side, never overwritten in place, so
different instruction.md wordings can be A/B'd against each other on the identical task set:
  v1_domain_specific - original: chemistry/materials-flavored language (DFT/adsorption_energy/...),
                        no explicit process structure. Superseded 2026-07-15, kept for comparison.
  v2_af              - domain-general language (works across all 9 corpus domains) + explicit
                        six-stage process (Ideation/Retrieval&Synthesis/Execution/Analysis/Writing/
                        Review) the agent must actually work through, not just "compute a number".
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path("/data/from_139/xmyu/scicoder")
_REWARD_SRC = [  # copied to dataset-root tests/ (shared, self-contained)
    ("examples/discovery_rubric_evaluate.py", "evaluate.py"),
    ("harness/recompute_tools.py", "recompute_tools.py"),
    ("harness/qe_relax.py", "qe_relax.py"),
]


def slug(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in (s or "task"))[:120]


def gold_block(row: dict) -> dict:
    """Normalize the paper's gold for this observable: {value, unit, claim, site_preference}."""
    g = row.get("gold") or {}
    out = {"claim": g.get("claim", "")}
    v = g.get("energy_eV")
    if v is None:
        v = g.get("numeric")
    if isinstance(v, (int, float)) and g.get("unit"):
        out["value"] = float(v); out["unit"] = g.get("unit")
    vr = (g.get("value_raw") or g.get("claim") or "").lower()
    for s in ("atop", "ontop", "fcc", "hcp", "bridge", "hollow", "top"):
        if (row.get("handle") == "site_preference") and s in vr:
            out["site_preference"] = "ontop" if s in ("atop", "ontop", "top") else ("fcc" if s == "hollow" else s)
            break
    return out


INSTRUCTION_V1_DOMAIN_SPECIFIC = """# {title}

## Background (established consensus)
{premise}

## The open question (tension)
{tension}

**Your task:** investigate THIS question computationally, using whatever method actually fits the
quantity the paper concerns — first-principles DFT, molecular dynamics, (micro)kinetic / rate
modeling, thermodynamic analysis, statistical / machine-learning modeling, etc. CHOOSE THE METHOD
THAT ANSWERS THE PAPER'S QUESTION; do NOT reduce a kinetics / activity / data-driven / spectroscopy
problem to a generic energy calculation just because one tool is handy. Model the *actual* system
the paper concerns (molecule, surface, crystal, defect, reaction, dataset, ...).

The sandbox has Python with network access — `pip install` whatever your chosen method needs (e.g.
pyscf / psi4 for quantum chemistry, an MD engine, scikit-learn for ML, pymatgen, scipy for rate /
kinetics modeling, a DFT code, etc.). Pick the package that computes the paper's OWN observable; do
NOT fall back to a generic energy / interatomic-potential calculation just because one is convenient
when the paper's quantity is a rate, activity, transport, spectroscopic, or data-driven property.
Install only what the task needs; don't go on open-ended web searches. Build/compute, inspect each
run, then report.

**A GPU is attached to this sandbox** (check with `nvidia-smi` or `torch.cuda.is_available()` after
`pip install torch`). If your method benefits from it — training/inference for an ML surrogate,
GPU-accelerated MLIP relaxation (e.g. CHGNet/MACE), large batched simulations — install a
CUDA-enabled build and use it; CPU-only is completely fine if the method doesn't need one. Don't
force GPU use where it doesn't fit the task.

## Required output — write `/workspace/decision.json` BEFORE you stop (it is the only graded artifact):
```json
{{
  "problem": "the specific question you investigated (1 sentence)",
  "system": {{"description": "what you modeled",
             "spec": {{"formula": "...", "structure_type": "...", "...": "machine-readable build hints"}}}},
  "method": {{"approach": "the computational method you actually used (dft / md / microkinetic / ML / ...)",
             "tools": "packages/engines used", "key_params": {{}}}},
  "observable": "the quantity you computed — use the paper's OWN quantity (adsorption_energy / activation_energy / reaction_rate / band_gap / faradaic_efficiency / catalytic_activity / formation_energy / ... whatever fits)",
  "result": {{"value": 0.0, "unit": "...", "details": {{}}}},
  "conclusion": "one-sentence finding that resolves the open question, grounded in YOUR computed result"
}}
```
**Write `/workspace/decision.json` EARLY** — as soon as you have a first computed estimate, write a
preliminary version, then keep OVERWRITING it with refined values as you compute more. Long
first-principles / MD runs may not finish within limits; a preliminary grounded decision is far
better than none, so NEVER leave decision.json unwritten while a computation is still running.
Only re-executed, grounded results count — base your conclusion on what you actually computed,
not on a remembered literature value.
"""


INSTRUCTION_V2_AF = """# {title}

## Background (established consensus)
{premise}

## The open question (tension)
{tension}

**Your task:** investigate this open question yourself, end-to-end, as a real research process —
not just "compute one number and stop." This corpus spans many fields (physics, chemistry, biology,
materials, medicine, geophysics, energy, industrial systems, scientific computing, ...), so there is
no default method — work through the six stages below for real, in order; your final decision.json
should read as the natural output of having gone through all of them, not a number found first and
rationalized backward.

**A · Ideation** — Restate the tension in your own words before touching any tool, and commit to a
specific, falsifiable hypothesis: what claim will your work support or refute? Decide what system,
dataset, or scale you will actually study, and why it's tractable in this sandbox.

**B · Retrieval & Synthesis** — Ground the hypothesis: use `WebSearch` / `WebFetch` (or your own
domain knowledge where lookups aren't fruitful) to check what's already established about this
specific question, and update your plan if it changes anything. Don't skip this to save time — an
ungrounded hypothesis is the most common way this kind of task goes wrong.

**C · Execution** — Choose whatever method actually answers THIS paper's own question — that could
be a first-principles/quantum calculation, a molecular or agent-based simulation, a (micro)kinetic
or rate model, a statistical or machine-learning model, a numerical/PDE solver, a phylogenetic or
sequence analysis, a signal-processing pipeline, an econometric or optimization model, or something
else entirely. Pick whatever the paper's own observable actually calls for, not whatever's most
familiar to you. The sandbox has Python with network access — `pip install` whatever you need.
Build the system/dataset, run it for real, and inspect actual output at each step; never fabricate
a plausible-looking number.

**D · Analysis** — Interpret what you computed: does it support or refute the Stage-A hypothesis?
Compare against the premise/tension framing above. Note failure modes, sensitivity, or alternative
explanations you considered — not just the headline number.

**E · Writing** — Write `/workspace/decision.json` (schema below) as the report of the whole
process above: a reader should be able to reconstruct your hypothesis, method, and reasoning from
it, not just read off a number.

**F · Review** — Before you stop, re-read your own decision.json as a skeptical reviewer would: is
the conclusion actually supported by the result, given the method's real limitations? A well-hedged,
honest conclusion scores better than a confident but unsupported one — say so if the evidence is
weak, rather than overclaiming.

**A GPU is attached to this sandbox** (check with `nvidia-smi` or `torch.cuda.is_available()` after
`pip install torch`). If your method benefits from it — training/inference for an ML model,
GPU-accelerated simulation, large batched computation — install a CUDA-enabled build and use it;
CPU-only is completely fine if the method doesn't need one. Don't force GPU use where it doesn't fit.

## Required output — write `/workspace/decision.json` BEFORE you stop (it is the only graded artifact):
```json
{{
  "problem": "the specific question you investigated (1 sentence)",
  "hypothesis": "the falsifiable claim from Stage A, stated before you had results",
  "system": {{"description": "what you modeled or studied",
             "spec": {{"...": "machine-readable build hints — whatever fields make sense for your system"}}}},
  "method": {{"approach": "the method you actually used, in your own words (not a category label)",
             "tools": "packages/engines/datasets used", "key_params": {{}}}},
  "observable": "the quantity you computed — name it however it's actually named in this field, using the paper's OWN quantity, not a generic placeholder",
  "result": {{"value": 0.0, "unit": "...", "details": {{}}}},
  "conclusion": "one-sentence finding that resolves the open question, grounded in YOUR computed result — hedge honestly if the evidence is weak (Stage F)"
}}
```
**Write `/workspace/decision.json` EARLY** — as soon as you have a first computed estimate, write a
preliminary version, then keep OVERWRITING it with refined values as you work through Stages D-F.
Long computations may not finish within limits; a preliminary grounded decision is far better than
none, so NEVER leave decision.json unwritten while a computation is still running. Only re-executed,
grounded results count — base your conclusion on what you actually computed, not on a remembered
literature value.
"""

INSTRUCTION_V3_FAILURE_TAXONOMY = """# {title}

## Background (established consensus)
{premise}

## The open question (tension)
{tension}

**Your task:** investigate this open question yourself, end-to-end, as a real research process —
not just "compute one number and stop." This corpus spans many fields (physics, chemistry, biology,
materials, medicine, geophysics, energy, industrial systems, scientific computing, ...), so there is
no default method — work through the six stages below for real, in order. At each stage, record the
specific artifact requested for `process_log` (schema below) as you go, not reconstructed
afterward — it should reflect what you actually did and considered, including anything that didn't
work or that you decided against.

**A · Ideation** — Restate the tension in your own words, then commit to a specific, falsifiable
hypothesis. Before moving on, name at least one other way you could have framed this question or
approached it, and say why you picked this one instead. Also say concretely: what result, number,
or comparison would tell you your hypothesis is WRONG — if nothing could tell you that, the
hypothesis isn't ready yet.

**B · Retrieval & Synthesis** — Use `WebSearch` / `WebFetch` (or your own domain knowledge where
lookups aren't fruitful) to check what's already established about this specific question. For
each source you actually use later (in Stage E), be able to say what it establishes and which
specific claim of yours it supports — not just that it seemed topically related. Organize what you
find before moving on; don't carry it forward as an unsorted pile.

**C · Execution** — Choose whatever method actually answers THIS paper's own question — that could
be a first-principles/quantum calculation, a molecular or agent-based simulation, a (micro)kinetic
or rate model, a statistical or machine-learning model, a numerical/PDE solver, a phylogenetic or
sequence analysis, a signal-processing pipeline, an econometric or optimization model, or something
else entirely. Pick whatever the paper's own observable actually calls for, not whatever's most
familiar to you. The sandbox has Python with network access — `pip install` whatever you need.
Build the system/dataset, run it for real, and inspect actual output at each step; never fabricate
a plausible-looking number. Say explicitly what you tested (which settings, scales, or conditions)
and what you deliberately left untested, and note anything that broke, stalled, or needed a
workaround along the way.

**D · Analysis** — Interpret what you computed: does it support or refute the Stage-A hypothesis?
Name at least one alternative explanation for your result other than the one you're going with, and
say concretely why you ruled it out (or didn't). State how confident you are and what specifically
that confidence is based on.

**E · Writing** — Write `/workspace/decision.json` (schema below) as the report of the whole
process above: a reader should be able to reconstruct your hypothesis, method, and reasoning from
it, not just read off a number. Every citation you list must be one you actually retrieved and read
in Stage B, tied to the specific claim it supports.

**F · Review** — Before you stop, re-read your own decision.json as a skeptical reviewer would: is
the conclusion actually supported by the result, given the method's real limitations? Name the
single weakest point in your own argument, and say what evidence — if it existed — would change
your mind. A well-hedged, honest conclusion scores better than a confident but unsupported one.

**A GPU is attached to this sandbox** (check with `nvidia-smi` or `torch.cuda.is_available()` after
`pip install torch`). If your method benefits from it — training/inference for an ML model,
GPU-accelerated simulation, large batched computation — install a CUDA-enabled build and use it;
CPU-only is completely fine if the method doesn't need one. Don't force GPU use where it doesn't fit.

## Required output — write `/workspace/decision.json` BEFORE you stop (it is the only graded artifact):
```json
{{
  "problem": "the specific question you investigated (1 sentence)",
  "hypothesis": "the falsifiable claim from Stage A, stated before you had results",
  "system": {{"description": "what you modeled or studied",
             "spec": {{"...": "machine-readable build hints — whatever fields make sense for your system"}}}},
  "method": {{"approach": "the method you actually used, in your own words (not a category label)",
             "tools": "packages/engines/datasets used", "key_params": {{}}}},
  "observable": "the quantity you computed — name it however it's actually named in this field, using the paper's OWN quantity, not a generic placeholder",
  "result": {{"value": 0.0, "unit": "...", "details": {{}}}},
  "conclusion": "one-sentence finding that resolves the open question, grounded in YOUR computed result — hedge honestly if the evidence is weak (Stage F)",
  "process_log": {{
    "ideation": {{"alternative_framing_considered": "the other way you could have approached this",
                 "falsification_check": "what result would have told you the hypothesis was wrong"}},
    "retrieval": {{"sources": [{{"source": "...", "establishes": "...", "supports_claim": "..."}}],
                  "synthesis_note": "how you organized what you found before moving on"}},
    "execution": {{"settings_tested": ["..."], "deliberately_not_tested": ["..."],
                  "issues_encountered": ["anything that broke, stalled, or needed a workaround"]}},
    "analysis": {{"alternative_explanation_considered": "...", "why_ruled_out_or_not": "...",
                 "confidence": "...", "confidence_basis": "what specifically that confidence rests on"}},
    "review": {{"weakest_point": "...", "what_would_change_your_mind": "..."}}
  }}
}}
```
**Write `/workspace/decision.json` EARLY** — as soon as you have a first computed estimate, write a
preliminary version, then keep OVERWRITING it with refined values as you work through Stages D-F.
Long computations may not finish within limits; a preliminary grounded decision is far better than
none, so NEVER leave decision.json unwritten while a computation is still running. Only re-executed,
grounded results count — base your conclusion on what you actually computed, not on a remembered
literature value. Fill in `process_log` honestly — an incomplete or hedged entry is far more useful
than a reconstructed one that just makes the process look clean.
"""


# v4: audited v3 trajectories (2026-07-15) showed process_log.review/analysis WERE being filled in
# (48/48 and 46/46 non-empty across two 50-task batches), but ZERO trajectories out of 100 ever
# wrote a separate report file — "Stage E · Writing" collapsed into filling a few short JSON string
# fields in one shot, and "Stage F · Review" was re-reading those same few lines, not a real
# critical re-read of a written document. v3's own wording caused this: it told the agent
# decision.json itself WAS "the report", so there was never a narrative document to review. v4
# splits the deliverable in two: a real prose report.md (Stage E) that a reader could follow with
# no other context, reviewed by literally re-opening and re-reading it (Stage F) and appending a
# reviewer-voice section — THEN distilling the unchanged decision.json schema from report.md, so
# every existing verifier/regime-eval/rescore script still parses it exactly as before.
INSTRUCTION_V4_REPORT_REVIEW = INSTRUCTION_V3_FAILURE_TAXONOMY.replace(
    '''**E · Writing** — Write `/workspace/decision.json` (schema below) as the report of the whole
process above: a reader should be able to reconstruct your hypothesis, method, and reasoning from
it, not just read off a number. Every citation you list must be one you actually retrieved and read
in Stage B, tied to the specific claim it supports.

**F · Review** — Before you stop, re-read your own decision.json as a skeptical reviewer would: is
the conclusion actually supported by the result, given the method's real limitations? Name the
single weakest point in your own argument, and say what evidence — if it existed — would change
your mind. A well-hedged, honest conclusion scores better than a confident but unsupported one.''',
    '''**E · Writing** — Write `/workspace/report.md` FIRST: a real written report of the whole
investigation, in your own prose (not bullet fragments), with these sections in order:
  - `## Summary` — 2-4 sentences: what you investigated, the method you used, and the key result —
    a reader should get the whole story from this paragraph alone before reading further.
  - `## Introduction & Hypothesis` — the tension, your hypothesis, and the falsification check (Stage A).
  - `## Method` — what you built/ran and why it answers this question (Stage C), in your own words.
  - `## Results` — what you actually computed, with the real numbers (Stage C output).
  - `## Discussion` — interpretation, the alternative explanation you weighed, your confidence (Stage D).
  - `## Limitations` — what you didn't test and why, anything that broke or needed a workaround.
  - `## References` — every source you actually retrieved and read in Stage B, one per line, each
    with what it establishes and which specific claim above it supports (title/URL + 1 sentence) —
    do not list a source here you did not actually open and read; do not cite from memory.
A reader with none of your context should be able to reconstruct your hypothesis, method, and
reasoning end-to-end from this document alone. Only after report.md exists, distill
`/workspace/decision.json` (schema below) FROM it — decision.json is the compact, machine-graded
summary of report.md, not a replacement for it; every field in it should be traceable to a passage
in report.md, and `process_log.retrieval.sources` must match `## References` one-to-one.

**F · Review** — Close report.md, then actually re-open and re-read it — not from memory — as a
skeptical external reviewer would: is the conclusion actually supported by the result, given the
method's real limitations? Append a `## Review` section to report.md written in the reviewer's
voice (third person, e.g. "The authors claim X, but Y is not ruled out because..."): name the
single weakest point in the argument, and say what evidence — if it existed — would change the
verdict. A well-hedged, honest review scores better than a confident but unsupported one. Mirror
the same two points into decision.json's `process_log.review` for automated scoring, but the full
reasoning belongs in report.md.''',
)
INSTRUCTION_V4_REPORT_REVIEW = INSTRUCTION_V4_REPORT_REVIEW.replace(
    '''## Required output — write `/workspace/decision.json` BEFORE you stop (it is the only graded artifact):''',
    '''## Required output — two artifacts, written in this order, BEFORE you stop:
1. `/workspace/report.md` — the full written report + its appended `## Review` section (Stage E + F).
   This is the primary human-readable artifact; write it in real prose.
2. `/workspace/decision.json` — the compact, machine-graded summary distilled FROM report.md (schema
   below). This is the only artifact automated scoring reads, but every field must be traceable to
   a passage in report.md — do not let the two diverge.''',
)
INSTRUCTION_V4_REPORT_REVIEW = INSTRUCTION_V4_REPORT_REVIEW.replace(
    '''**Write `/workspace/decision.json` EARLY** — as soon as you have a first computed estimate, write a
preliminary version, then keep OVERWRITING it with refined values as you work through Stages D-F.
Long computations may not finish within limits; a preliminary grounded decision is far better than
none, so NEVER leave decision.json unwritten while a computation is still running. Only re-executed,
grounded results count — base your conclusion on what you actually computed, not on a remembered
literature value. Fill in `process_log` honestly — an incomplete or hedged entry is far more useful
than a reconstructed one that just makes the process look clean.''',
    '''**Write both files EARLY** — as soon as you have a first computed estimate, write a preliminary
report.md and decision.json, then keep OVERWRITING both with refined content as you work through
Stages D-F. Long computations may not finish within limits; a preliminary grounded pair of files is
far better than none, so NEVER leave either file unwritten while a computation is still running.
Only re-executed, grounded results count — base your conclusion on what you actually computed, not
on a remembered literature value. Fill in `process_log` and report.md's `## Review` section
honestly — an incomplete or hedged entry is far more useful than a reconstructed one that just
makes the process look clean.''',
)


# v5: v4's 50-task glm-5.2 run (2026-07-15/16, subset50) showed wrote_decision drop to 32/50 (64%)
# vs v3's 48/50 (96%) on the IDENTICAL task set + model. Inspecting the 18 no_decision cases: all
# ended with the CLI's own "result" event reporting subtype=success/is_error=false/stop_reason=
# tool_use at only 30-80 of the 200 allowed turns — i.e. the session ended itself mid-task, before
# ever reaching decision.json, NOT from hitting max-turns/timeout/API errors (gateway logs show
# ~29k requests, almost all 200 OK). Root cause: v4's Stage E made decision.json (the ONLY graded
# artifact) strictly DEPENDENT on report.md being fully written first ("Only after report.md
# exists, distill decision.json FROM it") — if the session ends for any reason while still drafting
# the multi-section report.md, decision.json never gets written at all, and the task scores zero
# even if the agent had a perfectly good grounded result in hand. v5 restores v3's proven-safe
# ordering — write a preliminary decision.json THE MOMENT a grounded result exists, before any
# long-form writing — and makes report.md an EXPANSION of that decision, not a gate in front of it.
INSTRUCTION_V5_REPORT_REVIEW = INSTRUCTION_V4_REPORT_REVIEW.replace(
    '''**E · Writing** — Write `/workspace/report.md` FIRST: a real written report of the whole
investigation, in your own prose (not bullet fragments), with these sections in order:
  - `## Summary` — 2-4 sentences: what you investigated, the method you used, and the key result —
    a reader should get the whole story from this paragraph alone before reading further.
  - `## Introduction & Hypothesis` — the tension, your hypothesis, and the falsification check (Stage A).
  - `## Method` — what you built/ran and why it answers this question (Stage C), in your own words.
  - `## Results` — what you actually computed, with the real numbers (Stage C output).
  - `## Discussion` — interpretation, the alternative explanation you weighed, your confidence (Stage D).
  - `## Limitations` — what you didn't test and why, anything that broke or needed a workaround.
  - `## References` — every source you actually retrieved and read in Stage B, one per line, each
    with what it establishes and which specific claim above it supports (title/URL + 1 sentence) —
    do not list a source here you did not actually open and read; do not cite from memory.
A reader with none of your context should be able to reconstruct your hypothesis, method, and
reasoning end-to-end from this document alone. Only after report.md exists, distill
`/workspace/decision.json` (schema below) FROM it — decision.json is the compact, machine-graded
summary of report.md, not a replacement for it; every field in it should be traceable to a passage
in report.md, and `process_log.retrieval.sources` must match `## References` one-to-one.''',
    '''**E · Writing** — The MOMENT you have a first grounded result, write a preliminary
`/workspace/decision.json` (schema below) — this is the only artifact automated scoring reads, so it
must exist as early as possible and never be blocked on anything else. Only once that exists, EXPAND
it into `/workspace/report.md`: a real written report of the whole investigation, in your own prose
(not bullet fragments), with these sections in order:
  - `## Summary` — 2-4 sentences: what you investigated, the method you used, and the key result —
    a reader should get the whole story from this paragraph alone before reading further.
  - `## Introduction & Hypothesis` — the tension, your hypothesis, and the falsification check (Stage A).
  - `## Method` — what you built/ran and why it answers this question (Stage C), in your own words.
  - `## Results` — what you actually computed, with the real numbers (Stage C output).
  - `## Discussion` — interpretation, the alternative explanation you weighed, your confidence (Stage D).
  - `## Limitations` — what you didn't test and why, anything that broke or needed a workaround.
  - `## References` — every source you actually retrieved and read in Stage B, one per line, each
    with what it establishes and which specific claim above it supports (title/URL + 1 sentence) —
    do not list a source here you did not actually open and read; do not cite from memory.
A reader with none of your context should be able to reconstruct your hypothesis, method, and
reasoning end-to-end from report.md alone, and every field in decision.json should be traceable to
a passage in it (`process_log.retrieval.sources` should match `## References` one-to-one) — but if
you run low on turns or time, an unfinished report.md is fine, an unfinished decision.json is not.''',
)
INSTRUCTION_V5_REPORT_REVIEW = INSTRUCTION_V5_REPORT_REVIEW.replace(
    '''## Required output — two artifacts, written in this order, BEFORE you stop:
1. `/workspace/report.md` — the full written report + its appended `## Review` section (Stage E + F).
   This is the primary human-readable artifact; write it in real prose.
2. `/workspace/decision.json` — the compact, machine-graded summary distilled FROM report.md (schema
   below). This is the only artifact automated scoring reads, but every field must be traceable to
   a passage in report.md — do not let the two diverge.''',
    '''## Required output — two artifacts, written in this order, BEFORE you stop:
1. `/workspace/decision.json` (schema below) — write a preliminary version the moment you have a
   first grounded result; this is the ONLY artifact automated scoring reads, so it must exist even
   if nothing else does.
2. `/workspace/report.md` — the full written report + its appended `## Review` section (Stage E + F),
   expanding on decision.json in real prose. Every field in decision.json must be traceable to a
   passage here — do not let the two diverge — but decision.json existing does not depend on this
   file being finished.''',
)
INSTRUCTION_V5_REPORT_REVIEW = INSTRUCTION_V5_REPORT_REVIEW.replace(
    '''**Write both files EARLY** — as soon as you have a first computed estimate, write a preliminary
report.md and decision.json, then keep OVERWRITING both with refined content as you work through
Stages D-F. Long computations may not finish within limits; a preliminary grounded pair of files is
far better than none, so NEVER leave either file unwritten while a computation is still running.
Only re-executed, grounded results count — base your conclusion on what you actually computed, not
on a remembered literature value. Fill in `process_log` and report.md's `## Review` section
honestly — an incomplete or hedged entry is far more useful than a reconstructed one that just
makes the process look clean.''',
    '''**Write decision.json EARLY, report.md second** — as soon as you have a first computed estimate,
write a preliminary decision.json, then keep OVERWRITING it with refined values as you work through
Stages D-F; only start report.md once decision.json exists. Long computations may not finish within
limits; a preliminary grounded decision is far better than none, so NEVER leave decision.json
unwritten while a computation is still running, and never let writing report.md delay decision.json.
Only re-executed, grounded results count — base your conclusion on what you actually computed, not
on a remembered literature value. Fill in `process_log` and report.md's `## Review` section
honestly — an incomplete or hedged entry is far more useful than a reconstructed one that just
makes the process look clean.''',
)


# v6: v5 fixed the ordering bug (decision.json first) but kept v4's wording, which framed report.md
# as something "distilled/expanded FROM decision.json" with bookkeeping requirements (traceable to
# a passage, one-to-one match with process_log.retrieval.sources) — verbose, and conceptually
# backwards. User feedback (2026-07-16, two rounds): (1) report.md isn't a derivative of the compact
# JSON, drop the bookkeeping language; (2) it should read like a paper a human would actually write
# after finishing a full research project, not a terse internal summary — so use standard paper
# sections (Abstract/Introduction/Methods/Results/Discussion/Limitations/Conclusion/References) and
# name the appended critique "Peer Review" so the whole artifact reads as "submitted paper + the
# reviewer report on it". Same decision-json-first safety ordering as v5 — unchanged.
INSTRUCTION_V6_REPORT_REVIEW = INSTRUCTION_V5_REPORT_REVIEW.replace(
    '''**E · Writing** — The MOMENT you have a first grounded result, write a preliminary
`/workspace/decision.json` (schema below) — this is the only artifact automated scoring reads, so it
must exist as early as possible and never be blocked on anything else. Only once that exists, EXPAND
it into `/workspace/report.md`: a real written report of the whole investigation, in your own prose
(not bullet fragments), with these sections in order:
  - `## Summary` — 2-4 sentences: what you investigated, the method you used, and the key result —
    a reader should get the whole story from this paragraph alone before reading further.
  - `## Introduction & Hypothesis` — the tension, your hypothesis, and the falsification check (Stage A).
  - `## Method` — what you built/ran and why it answers this question (Stage C), in your own words.
  - `## Results` — what you actually computed, with the real numbers (Stage C output).
  - `## Discussion` — interpretation, the alternative explanation you weighed, your confidence (Stage D).
  - `## Limitations` — what you didn't test and why, anything that broke or needed a workaround.
  - `## References` — every source you actually retrieved and read in Stage B, one per line, each
    with what it establishes and which specific claim above it supports (title/URL + 1 sentence) —
    do not list a source here you did not actually open and read; do not cite from memory.
A reader with none of your context should be able to reconstruct your hypothesis, method, and
reasoning end-to-end from report.md alone, and every field in decision.json should be traceable to
a passage in it (`process_log.retrieval.sources` should match `## References` one-to-one) — but if
you run low on turns or time, an unfinished report.md is fine, an unfinished decision.json is not.

**F · Review** — Close report.md, then actually re-open and re-read it — not from memory — as a
skeptical external reviewer would: is the conclusion actually supported by the result, given the
method's real limitations? Append a `## Review` section to report.md written in the reviewer's
voice (third person, e.g. "The authors claim X, but Y is not ruled out because..."): name the
single weakest point in the argument, and say what evidence — if it existed — would change the
verdict. A well-hedged, honest review scores better than a confident but unsupported one. Mirror
the same two points into decision.json's `process_log.review` for automated scoring, but the full
reasoning belongs in report.md.''',
    '''**E · Writing** — The MOMENT you have a first grounded result, write a preliminary
`/workspace/decision.json` (schema below) — it's the only artifact automated scoring reads, so get
it down early and never block it on anything else. Then write `/workspace/report.md`: a real paper,
the kind a scientist writes after actually finishing this investigation, with these sections:
  - `## Abstract` — 2-3 sentences: what you investigated, your method, and the key result.
  - `## Introduction` — the tension/open question and your hypothesis (Stage A).
  - `## Methods` — what you built or ran, and why it answers this question (Stage C).
  - `## Results` — what you actually computed, with the real numbers.
  - `## Discussion` — your interpretation, the alternative explanation you weighed, and your
    confidence (Stage D).
  - `## Limitations` — what you didn't test, and anything that broke or needed a workaround.
  - `## Conclusion` — the one-paragraph takeaway.
  - `## References` — each source you actually retrieved and read in Stage B, and what it
    establishes. Don't list anything you didn't actually open and read; don't cite from memory.
If you run low on turns or time, an unfinished report.md is fine, an unfinished decision.json is not.

**F · Review** — Re-open and re-read report.md — not from memory — as a skeptical peer reviewer
would. Append a `## Peer Review` section in the reviewer's voice (third person, e.g. "The authors
claim X, but Y is not ruled out because..."): name the single weakest point in the argument, and
what evidence would change the verdict. Mirror it into decision.json's `process_log.review`.''',
)
INSTRUCTION_V6_REPORT_REVIEW = INSTRUCTION_V6_REPORT_REVIEW.replace(
    '''## Required output — two artifacts, written in this order, BEFORE you stop:
1. `/workspace/decision.json` (schema below) — write a preliminary version the moment you have a
   first grounded result; this is the ONLY artifact automated scoring reads, so it must exist even
   if nothing else does.
2. `/workspace/report.md` — the full written report + its appended `## Review` section (Stage E + F),
   expanding on decision.json in real prose. Every field in decision.json must be traceable to a
   passage here — do not let the two diverge — but decision.json existing does not depend on this
   file being finished.''',
    '''## Required output — two artifacts, written in this order, BEFORE you stop:
1. `/workspace/decision.json` (schema below) — write a preliminary version the moment you have a
   first grounded result. This is the ONLY artifact automated scoring reads.
2. `/workspace/report.md` — a paper on the whole investigation (Stage E), plus its appended
   `## Peer Review` section (Stage F).''',
)
INSTRUCTION_V6_REPORT_REVIEW = INSTRUCTION_V6_REPORT_REVIEW.replace(
    '''**Write decision.json EARLY, report.md second** — as soon as you have a first computed estimate,
write a preliminary decision.json, then keep OVERWRITING it with refined values as you work through
Stages D-F; only start report.md once decision.json exists. Long computations may not finish within
limits; a preliminary grounded decision is far better than none, so NEVER leave decision.json
unwritten while a computation is still running, and never let writing report.md delay decision.json.
Only re-executed, grounded results count — base your conclusion on what you actually computed, not
on a remembered literature value. Fill in `process_log` and report.md's `## Review` section
honestly — an incomplete or hedged entry is far more useful than a reconstructed one that just
makes the process look clean.''',
    '''**Write decision.json EARLY, report.md second** — write a preliminary decision.json as soon as
you have a first computed estimate, keep refining it through Stages D-F, and only start report.md
once decision.json exists — never let writing report.md delay it. Only re-executed, grounded
results count — base your conclusion on what you actually computed, not a remembered literature
value. Fill in `process_log` and report.md's `## Peer Review` section honestly — an incomplete or
hedged entry is far more useful than a reconstructed one that just makes the process look clean.''',
)


PROMPT_VERSIONS = {
    "v1_domain_specific": INSTRUCTION_V1_DOMAIN_SPECIFIC,
    "v2_af": INSTRUCTION_V2_AF,
    "v3_failure_taxonomy": INSTRUCTION_V3_FAILURE_TAXONOMY,
    "v4_report_review": INSTRUCTION_V4_REPORT_REVIEW,
    "v5_report_review": INSTRUCTION_V5_REPORT_REVIEW,
    "v6_report_review": INSTRUCTION_V6_REPORT_REVIEW,
}


def task_toml(row: dict) -> str:
    return f"""version = "1.0"

[metadata]
benchmark = "SciDiscovery"
instance_id = "{slug(row.get('task_id',''))}"
category = "discovery"
observable = "{row.get('handle') or ''}"
novelty_move = "{row.get('novelty_move') or ''}"
tier = "{row.get('tier') or ''}"
tier_weight = {row.get('tier_weight') or 1.0}
qe_ready = {str(bool(row.get('qe_ready'))).lower()}
source = "scidata-engine/discovery"

[verifier]
module = "tests.evaluate"
timeout_sec = 1800.0

[agent]
timeout_sec = 10800.0
"""


def rubric(row: dict) -> dict:
    return {
        "task_id": row.get("task_id"),
        "premise": row.get("premise") or "",
        "tension": row.get("tension") or "",
        "paper_conclusion": row.get("conclusion") or "",
        "novelty_move": row.get("novelty_move") or "",
        "observable": row.get("handle") or "",
        "all_observables": row.get("all_handles") or [],
        "gold": gold_block(row),
        "paper_gt": row.get("paper_gt") or [],
        "tier": row.get("tier") or "",
        "tier_weight": row.get("tier_weight") or 1.0,
        "qe_ready": bool(row.get("qe_ready")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(REPO / "examples/output/discovery_tasks.jsonl"))
    ap.add_argument("--template", required=True, help="dir with environment/Dockerfile")
    ap.add_argument("--out", default=str(REPO / "examples/output/discovery_tasks_v2"))
    ap.add_argument("--prompt-version", default="v2_af", choices=sorted(PROMPT_VERSIONS),
                    help="which instruction.md wording to materialize (see module docstring)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    template = Path(args.template)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # shared reward source at dataset-root tests/
    (outdir / "tests").mkdir(parents=True, exist_ok=True)
    for src, dst in _REWARD_SRC:
        if (REPO / src).exists():
            shutil.copy(REPO / src, outdir / "tests" / dst)

    prompt_tpl = PROMPT_VERSIONS[args.prompt_version]

    emitted = 0
    import collections
    obs = collections.Counter()
    for row in rows:
        tid = slug(row.get("task_id", "task"))
        d = outdir / tid
        (d / "tests").mkdir(parents=True, exist_ok=True)
        (d / "environment").mkdir(parents=True, exist_ok=True)
        (d / "task.toml").write_text(task_toml(row))
        (d / "instruction.md").write_text(prompt_tpl.format(
            title=row.get("title") or row.get("task_id"),
            premise=row.get("premise") or "(not stated)",
            tension=row.get("tension") or "(not stated)"))
        (d / "tests" / "task_meta.json").write_text(json.dumps({"rubric": rubric(row)}, ensure_ascii=False, indent=2))
        td = template / "environment" / "Dockerfile"
        if td.exists():
            shutil.copy(td, d / "environment" / "Dockerfile")
        obs[row.get("handle") or "(none)"] += 1
        emitted += 1

    (outdir / "dataset.toml").write_text(
        'version = "1.0"\n[metadata]\nname = "scidiscovery"\nbenchmark = "SciDiscovery"\n')
    print(f"emitted {emitted} task dirs -> {outdir}")
    print(f"observable distribution: {dict(obs.most_common())}")


if __name__ == "__main__":
    main()
