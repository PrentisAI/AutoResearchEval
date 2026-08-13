"""Model a paper's discovery arc as an agent-learnable (s,a,o,r) trajectory.

This is the bridge the user asked for: §18's ``DiscoveryPattern`` is a *flat* record
(premise → tension → motivation → method → experiment → conclusion). A real paper,
though, is an *incremental* sequence of small reasoning moves — "the field believed
X, but Y was untested, so I asked Q, picked system S, computed it, compared to the
reference, and concluded Z". To make discovery learnable we lay that arc onto an
**atomic discovery action space** and emit one IR ``Trajectory`` (CLAUDE.md §0.9
discovery→RL, §18.3).

Two-level action structure (deliberate):

  * **Discovery moves** (this module) — the *reasoning* layer: survey_consensus,
    identify_tension, formulate_question, … draw_conclusion. Mostly SOFT (the
    reasoning skeleton is the discovery supervision).
  * **Execution tools** (``ir/actions`` — the 26 verifier-bound actions) — the
    *doing* layer. The ``run_calculation`` discovery move GROUNDS into one of those
    (run_dft/run_md/…), and the trajectory's terminal reward grounds into the
    claim's ``recompute_handle`` (the rigor gate, §0.9 "hinge": RL reward = the
    deterministic gate applied to the model's own move).

Honesty (§1.1, §1.2, §14): this is RECONSTRUCTION. The moves/observations restate
what the paper reports — we never fabricate numbers. The arc is **not** admissible
via the hard physics gate (no QE-in-loop yet) — it stays ``pending-soft-verify``
and carries the recompute anchor so the future oracle can score it. We do not
overclaim admissibility.

The action vocabulary is SEEDED here but INDUCED-validated: the decomposition lets
the teacher emit ``other:<label>`` for anything that doesn't fit, and the driver
reports escape-hatch frequency (coverage). Per §15.3/§16.1: vocab from data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from ir import (
    Action,
    Difficulty,
    Observation,
    ProcessState,
    Provenance,
    ReconstructMethod,
    Reward,
    RewardStyle,
    SourceType,
    Step,
    Trajectory,
    Verification,
)
from reconstruct.discovery_pattern import DiscoveryPattern

# --------------------------------------------------------------------------- #
# The atomic discovery action space (seed vocabulary, 3 phases × moves).
# Grounded in the user's stated arc (background → motivation → question →
# design → analyze conclusion) + the §18 fields + the scientific method. The
# decomposition pass validates coverage (escape-hatch frequency).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DiscoveryMove:
    id: str
    phase: str          # FRAME | PROBE | RESOLVE
    description: str
    expects: tuple      # arg names the move carries
    produces: str       # what the observation is
    verifiable: str     # soft | grounds-exec | recompute-anchor


DISCOVERY_MOVES: list[DiscoveryMove] = [
    # ---- FRAME: from background to a sharp, answerable question ----
    DiscoveryMove("survey_consensus", "FRAME",
                  "State the established prior understanding the work takes as given.",
                  ("consensus",), "the field's current belief", "soft"),
    DiscoveryMove("identify_tension", "FRAME",
                  "Locate the gap/contradiction/anomaly in that consensus — the discovery seed.",
                  ("gap",), "the unsatisfied/untested point", "soft"),
    DiscoveryMove("formulate_question", "FRAME",
                  "Sharpen the tension into one concrete, answerable question.",
                  ("question",), "the question to resolve", "soft"),
    DiscoveryMove("propose_hypothesis", "FRAME",
                  "State a candidate answer / mechanism to test (optional).",
                  ("hypothesis",), "the testable hypothesis", "soft"),
    # ---- PROBE: design and execute the test ----
    DiscoveryMove("select_system", "PROBE",
                  "Choose the concrete system/surface/sites/conditions that can answer the question.",
                  ("system", "conditions"), "the chosen experimental setup", "soft"),
    DiscoveryMove("choose_method", "PROBE",
                  "Choose method + level of theory (functional/code/technique) and justify it.",
                  ("method", "level"), "the method to run", "soft"),
    DiscoveryMove("run_calculation", "PROBE",
                  "Execute the probe. GROUNDS into an ir/actions execution tool; observation is the computed/measured result.",
                  ("tool", "computes", "target"), "the result the paper reports", "grounds-exec"),
    DiscoveryMove("compare_reference", "PROBE",
                  "Compare the result against experiment / a prior value / a competing model.",
                  ("reference", "delta"), "agreement or discrepancy vs reference", "soft"),
    # ---- RESOLVE: interpret and conclude ----
    DiscoveryMove("interpret_result", "RESOLVE",
                  "Analyze what the observation means for the question (no new claim of victory yet).",
                  ("reading",), "the interpretation", "soft"),
    DiscoveryMove("draw_conclusion", "RESOLVE",
                  "State the terminal claim that resolves the tension / updates the belief.",
                  ("claim", "recompute_handle"), "the discovery", "recompute-anchor"),
]

MOVE_IDS = {m.id for m in DISCOVERY_MOVES}
MOVE_BY_ID = {m.id: m for m in DISCOVERY_MOVES}
PHASE_ORDER = {"FRAME": 0, "PROBE": 1, "RESOLVE": 2}

# Which execution action (ir/actions) a recompute_handle is naturally produced by.
# This is the bridge from the discovery layer to the 26-action verifier-bound layer.
HANDLE_TO_EXEC: dict[str, str] = {
    "co_adsorption_energy": "run_dft",
    "reaction_barrier": "run_dft",
    "site_preference": "run_dft",
    "vibrational_frequency": "run_dft",
    "coverage_shift": "run_dft",
    "work_function": "run_dft",
    "scaling_relation": "compute_descriptor",
}

# --------------------------------------------------------------------------- #
# LLM decomposition: arc → ordered atomic moves (the trajectory material).
# --------------------------------------------------------------------------- #
DECOMPOSE_SYSTEM = (
    "You are a computational-catalysis researcher replaying how a paper's discovery "
    "actually unfolded, step by step, as a sequence of atomic reasoning MOVES. You are "
    "reconstructing the researcher's first-person reasoning — not summarising the paper. "
    "Output strict JSON only. Be faithful: never invent numbers; reuse the figures the "
    "record already states. CRITICAL ANTI-HINDSIGHT RULE: a move's `thought` and "
    "`observation` may use ONLY what is known up to that point. Do NOT state the final "
    "conclusion, or claim the result confirms/refutes anything, before the run_calculation "
    "that produces it and the final draw_conclusion. The opening moves must read like open "
    "questions, not foregone answers."
)

_DECOMPOSE_PROMPT = """Replay this paper's discovery as an ordered list of atomic MOVES.

Use moves from this vocabulary (id : when to use):
{vocab}

Sequence rules:
- Follow FRAME → PROBE → RESOLVE order. Moves may repeat (e.g. several run_calculation)
  or be skipped if the paper genuinely lacks them. propose_hypothesis is optional.
- The FIRST move is survey_consensus (establish background). The LAST move is draw_conclusion.
- Emit a run_calculation for each distinct quantity the paper computes/measures; set
  `args.computes` to the recompute_handle it produces (or "none") and `args.tool` to the
  execution tool used (e.g. run_dft, run_md, compute_descriptor, analyze_dft_output).
- If a step truly fits none of the vocabulary, use "other:<short_label>" — but prefer a
  listed move.

For each move output: {{"move": "<id or other:label>", "thought": "<first-person reasoning
for making THIS move, grounded in the record, obeying the anti-hindsight rule>", "args":
{{...}}, "observation": "<what the agent now knows after the move — a real value/finding for
run_calculation, the stated question/setup/interpretation otherwise>"}}

Output JSON: {{"steps": [ ... ]}}. Output ONLY the JSON object.

DISCOVERY RECORD (paper {pid}):
- premise/consensus: {premise}
- tension: {tension}
- motivation: {motivation}
- method: {method}
- experiment: {experiment}
- conclusion: {conclusion}
- key_claims: {claims}
- novelty_move: {move}
"""


def _vocab_block() -> str:
    return "\n".join(f"  {m.id} ({m.phase}) : {m.description}" for m in DISCOVERY_MOVES)


def _extract_json(text: str) -> Optional[dict]:
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:
        s, e = text.find("{"), text.rfind("}")
        blob = text[s:e + 1] if s != -1 and e > s else None
    if not blob:
        return None
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


def decompose_arc(llm, pattern: DiscoveryPattern) -> Optional[list[dict]]:
    """Ask the teacher to replay the arc as ordered atomic moves. Returns the raw
    step dicts (move/thought/args/observation), or None if unparseable."""
    claims = json.dumps([
        {"claim": c.get("claim", ""), "value": c.get("value", ""),
         "recompute_handle": c.get("recompute_handle", "none")}
        for c in (pattern.key_claims or [])
    ], ensure_ascii=False)
    prompt = _DECOMPOSE_PROMPT.format(
        vocab=_vocab_block(), pid=pattern.paper_id,
        premise=pattern.premise_consensus, tension=pattern.tension,
        motivation=pattern.motivation, method=pattern.method,
        experiment=pattern.experiment, conclusion=pattern.conclusion,
        claims=claims, move=pattern.novelty_move)
    reply = llm.complete(prompt, system=DECOMPOSE_SYSTEM)
    d = _extract_json(reply)
    if not d or not isinstance(d.get("steps"), list) or not d["steps"]:
        return None
    return d["steps"]


# --------------------------------------------------------------------------- #
# Honesty lint (§1.2, §17.2 negation-robust): pre-conclusion moves must not leak
# the verdict. Returns a list of violation strings (empty == clean).
# --------------------------------------------------------------------------- #
_VERDICT_WORDS = re.compile(
    r"\b(we (conclude|demonstrate|show that)|in conclusion|this (confirms|proves|establishes))\b", re.I)


def lint_steps(steps: list[dict]) -> list[str]:
    """Honesty + shape gate (§1.2, §17.2). The invariants that actually matter:

      * the arc opens on background (survey_consensus) and closes on a claim
        (draw_conclusion);
      * once experimentation begins you do not return to FRAMING — you can't
        re-survey the consensus mid-experiment. (PROBE and RESOLVE interleave
        freely: real discovery loops compute → interpret → compute again.)
      * anti-hindsight: the answer must not be asserted BEFORE any computation
        exists. After the first run_calculation, interpreting toward the
        conclusion is legitimate; before it, a verdict word is leakage.
    """
    bad: list[str] = []
    moves = [str(s.get("move", "")) for s in steps]
    if not moves:
        return ["empty step list"]
    if not moves[0].startswith("survey_consensus"):
        bad.append(f"first move is {moves[0]!r}, expected survey_consensus")
    if not moves[-1].startswith("draw_conclusion"):
        bad.append(f"last move is {moves[-1]!r}, expected draw_conclusion")

    # no return to FRAME once we've left it for PROBE/RESOLVE
    left_frame = False
    for mv in moves:
        base = mv.split(":")[0]
        phase = MOVE_BY_ID[base].phase if base in MOVE_BY_ID else None
        if phase in ("PROBE", "RESOLVE"):
            left_frame = True
        elif phase == "FRAME" and left_frame:
            bad.append(f"framing move {mv!r} after experimentation started")

    # verdict leakage only counts BEFORE the first run_calculation (no evidence yet)
    first_calc = next((i for i, mv in enumerate(moves) if mv.split(":")[0] == "run_calculation"), len(moves))
    for i in range(min(first_calc, len(steps) - 1)):
        s = steps[i]
        txt = f"{s.get('thought','')} {s.get('observation','')}"
        if _VERDICT_WORDS.search(txt):
            bad.append(f"verdict leakage before any computation, step {i} ({moves[i]!r})")
    return bad


# --------------------------------------------------------------------------- #
# Deterministic converter: pattern + decomposition → IR Trajectory.
# --------------------------------------------------------------------------- #
DISCOVERY_SYSTEM_PROMPT = (
    "You are a computational-catalysis discovery agent. Resolve an open question by an "
    "incremental sequence of atomic moves: survey_consensus, identify_tension, "
    "formulate_question, propose_hypothesis, select_system, choose_method, run_calculation, "
    "compare_reference, interpret_result, draw_conclusion. run_calculation grounds into a "
    "real execution tool (e.g. run_dft) and your final claim must be backed by a recomputable "
    "quantity."
)


def _goal(pattern: DiscoveryPattern) -> str:
    """Back-translate the discovery task an agent faces — premise + open question, WITHOUT
    the answer (the answer is what RL/the agent must produce)."""
    q = pattern.tension.strip() or pattern.motivation.strip()
    return (f"Given the prior understanding that {pattern.premise_consensus.strip()} "
            f"resolve the following open question: {q}").strip()


def build_trajectory(pattern: DiscoveryPattern, steps: list[dict]) -> Trajectory:
    """Lay the decomposed move sequence onto the IR. Soft (pending-soft-verify); the
    terminal reward carries the recompute anchor for the future rigor gate."""
    handles = pattern.recompute_handles()
    ir_steps: list[Step] = []
    for i, s in enumerate(steps):
        move = str(s.get("move", "") or "")
        base = move.split(":")[0]
        is_exec = base == "run_calculation"
        is_terminal = (i == len(steps) - 1)

        obs_content: dict = {}
        if is_exec:
            obs_content = {
                "computes": (s.get("args") or {}).get("computes", "none"),
                "exec_tool": (s.get("args") or {}).get("tool", ""),
            }
        if is_terminal and base == "draw_conclusion":
            # surface the discovery as the trajectory's "result" so the exported
            # Final Answer is the actual claim, not a generic "completed".
            obs_content = {"res": pattern.conclusion or str(s.get("observation", ""))}
        observation = Observation(
            content=obs_content,
            text=str(s.get("observation", "") or ""),
            # reasoning moves "succeed" trivially; execution moves are reported-true
            process_state=ProcessState.FINISHED,
            exit_status=0,
            raw_ref=pattern.paper_id,
        )

        reward = None
        if is_terminal:
            if handles:
                reward = Reward(
                    style=RewardStyle.RULE, terminal=True,
                    ground_truth={"recompute_handles": handles,
                                  "claims": [c for c in pattern.key_claims
                                             if c.get("recompute_handle", "none") not in ("none", "")]},
                    # verifiable stays EMPTY until a QE/MLIP rerun fills it (§18.3). Honest.
                    verifiable={},
                )
            else:
                # no recomputable anchor → soft reward, needs reward-model / AI-Verifier
                reward = Reward(style=RewardStyle.MODEL, terminal=True,
                                ground_truth={"qualitative_claim": pattern.conclusion})

        ir_steps.append(Step(
            index=i,
            thought=str(s.get("thought", "") or "") or None,
            action=Action(name=base, params=(s.get("args") or {}), raw_ref=pattern.paper_id),
            observation=observation,
            reward=reward,
        ))

    prov = Provenance(
        source_type=SourceType.PAPER_REPO,
        source_id=pattern.paper_id,
        artifact_uri=pattern.artifact_uri or None,
        reconstruct_method=ReconstructMethod.THOUGHT_COMPLETION,
        teacher_model=pattern.model_id or None,
        reconstruct_prompt_ref="discovery_trajectory/decompose/v1",
    )
    move_seq = [str(s.get("move", "")) for s in steps]
    escape = [m for m in move_seq if m.split(":")[0] not in MOVE_IDS]
    return Trajectory(
        id=f"discovery::{pattern.paper_id}",
        goal=_goal(pattern),
        task_spec=("recompute anchors: " + ", ".join(handles)) if handles else None,
        system_prompt=DISCOVERY_SYSTEM_PROMPT,
        steps=ir_steps,
        provenance=prov,
        verification=Verification(notes="pending-soft-verify: discovery reasoning is soft; "
                                  "recompute anchor attached for future rigor gate (§18.3)"),
        difficulty=Difficulty.L3,
        metadata={
            "data_class": "discovery",          # §0.9 two-classes: discovery→RL
            "train_use": "RL",
            "status": "pending-soft-verify",
            "novelty_move": pattern.novelty_move,
            "recompute_handles": handles,
            "move_sequence": move_seq,
            "escape_hatch_moves": escape,
            "title": pattern.title,
        },
    )
