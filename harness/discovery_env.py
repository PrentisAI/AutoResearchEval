"""Domain-agnostic scientific-discovery environment (CLAUDE.md §0.9 hinge, §18, RL track).

THE SEAM. This module is the *skeleton* of a discovery episode and deliberately imports
NOTHING domain-specific (no ASE, no QE, no `recompute_tools`). That import-poverty is the
structural proof that the agent ability it trains — frame a tension → design an experiment
to adjudicate it → read the result → decide if the tension is resolved — is domain-agnostic.
Computational catalysis is merely our first plugged-in domain (`harness/domains/catalysis_qe.py`).

What lives HERE (domain-invariant):
  * the discovery MOVE sequence + state machine (FRAME → DESIGN → EXECUTE → RESOLVE);
  * the REWARD LOGIC — gate on ``sane ∧ decisive ∧ valid`` (defined below), NOT on
    "produced a number". This is the anti-reward-hacking rule from the design discussion:
    the episode only scores if the oracle says the experiment *decisively adjudicated the
    tension* and the tier is *admissible* (a real oracle, not a cheap proxy);
  * termination + IR/Reward/Verification wiring.

What a DOMAIN plugin supplies (via the ``DomainOracle`` protocol below):
  * a persona priming the reasoning moves;
  * ``design`` — turn the framed question into a typed, runnable experiment (the domain's
    select-system / choose-method moves);
  * ``execute`` — run the live oracle and return the raw result;
  * ``assess`` — map that raw result onto the three domain-invariant verdict flags.

The skeleton never sees a slab, a cutoff, or a Hubbard-U. The compile-time test for "is
this line domain-agnostic?" is simply: *would it work unchanged for a second domain (an
ML fail-to-pass oracle, a wet-lab database)?* If not, it belongs in the plugin, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

# ir is the engine's generic trajectory IR — NOT domain-specific.
from ir import Reward, RewardStyle, Verification


# --------------------------------------------------------------------------- #
# Domain-invariant value objects exchanged across the seam.
# --------------------------------------------------------------------------- #
@dataclass
class DiscoveryTask:
    """A framed discovery target — the domain-agnostic view of a paper pattern.

    The skeleton only ever reads ``premise`` + ``tension`` (the seed) to frame the episode
    and ``handle`` to tell the oracle which quantity adjudicates the tension. Everything
    else is provenance the IR carries through."""
    task_id: str
    title: str
    premise: str               # what the field already believes
    tension: str               # the crack in the consensus — the discovery seed
    handle: str                # the quantity whose recompute adjudicates the tension
    novelty_move: str = ""     # incremental | new-regime | … (curriculum label)
    provenance: dict = field(default_factory=dict)

    def goal(self) -> str:
        return f"Given that {self.premise.strip()} resolve: {self.tension.strip()}"


@dataclass
class Verdict:
    """The oracle's domain-agnostic judgement of one experiment. The skeleton's reward is
    a pure function of the three flags — the domain decides HOW to set them, the skeleton
    decides what they MEAN for the reward.

      * ``sane``     — is the result physically/logically admissible (not garbage)?
      * ``decisive`` — does it actually adjudicate the tension (e.g. compared both arms),
                       rather than merely emit one number? (the anti-hack flag)
      * ``valid``    — is this an *admissible-tier* oracle (a real measurement, not a cheap
                       proxy like EMT/MLIP)? Only a valid tier can mint admissible data.

    ``summary`` is the observation text; the remaining dicts are generic IR payloads the
    domain fills and the skeleton plumbs verbatim into Reward/Verification."""
    sane: bool
    decisive: bool
    valid: bool
    summary: str
    ground_truth: dict = field(default_factory=dict)
    verifiable: dict = field(default_factory=dict)
    physics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    notes_valid: str = ""
    notes_invalid: str = ""

    @property
    def rewarded(self) -> bool:
        """The single anti-hack gate: a real oracle that sanely AND decisively settled it."""
        return bool(self.sane and self.decisive and self.valid)


@dataclass
class Design:
    """Output of a domain ``design`` step: the move records to lay onto the IR, plus the
    opaque ``experiment`` handle the skeleton hands straight back to ``execute``."""
    records: list[dict]
    experiment: Any            # opaque to the skeleton — only the oracle interprets it
    metadata: dict = field(default_factory=dict)


@dataclass
class Execution:
    """Output of a domain ``execute`` step."""
    result: Any                # opaque to the skeleton — passed to assess / reference_note
    record: dict               # the run_* move record (thought/args/observation)
    seconds: float = 0.0


# --------------------------------------------------------------------------- #
# The plugin contract.
# --------------------------------------------------------------------------- #
@runtime_checkable
class DomainOracle(Protocol):
    """What every domain must implement to plug into the discovery skeleton.

    A conforming oracle is constructed already knowing its TIER (real vs proxy) and any
    compute budget/overrides; the skeleton treats it as a black box that designs, runs, and
    judges experiments. The three verdict flags are the ONLY domain knowledge that crosses
    back into the skeleton's reward."""

    name: str
    persona: str               # primes the reasoning moves for this domain
    engine_name: str           # human label of the oracle engine (for IR)
    admissible: bool           # is this tier real enough to mint admissible data?

    def design(self, llm, ctx: dict) -> Design:
        """Frame → a typed, runnable experiment (the domain's select/method moves)."""
        ...

    def execute(self, design: Design, **run_kw) -> Execution:
        """Run the live oracle; the observation IS the real result."""
        ...

    def assess(self, execution: Execution) -> Verdict:
        """Map the raw result onto the domain-invariant verdict flags + IR payloads."""
        ...

    def reference_note(self, execution: Execution) -> str:
        """The reference (experiment / prior value) the RESOLVE moves compare against."""
        ...


# --------------------------------------------------------------------------- #
# Domain-agnostic reasoning moves. Goals are generic discovery goals; the persona
# (which primes them for a domain) is injected by the oracle. select/method/execute
# are NOT here — they are the domain's job.
# --------------------------------------------------------------------------- #
FRAME_MOVES = ("survey_consensus", "identify_tension", "formulate_question")
RESOLVE_MOVES = ("compare_reference", "interpret_result", "draw_conclusion")

# discovery-move id -> the context key its output feeds (for the prompt view).
_CTX_KEY = {
    "survey_consensus": "consensus", "identify_tension": "tension",
    "formulate_question": "question", "propose_hypothesis": "hypothesis",
    "compare_reference": "comparison", "interpret_result": "interpretation",
}


def _reasoning_move(llm, move_id: str, ctx: dict, persona: str) -> dict:
    """Run one domain-agnostic reasoning move via the shared move machinery, primed with
    the domain persona. Imported lazily so the skeleton stays import-light."""
    from reconstruct.discovery_moves import run_move
    r = run_move(llm, move_id, ctx, persona=persona)
    return {"move": move_id, "thought": r.thought, "args": {}, "observation": r.thought}


# --------------------------------------------------------------------------- #
# The environment.
# --------------------------------------------------------------------------- #
class DiscoveryEnv:
    """Drives a discovery episode over a pluggable ``DomainOracle``.

    Two entry points sharing one state machine:
      * ``rollout(llm)``  — run the whole episode in one call (data-construction / eval);
      * ``reset()`` + ``step(action)`` — gym-like interface for an external RL policy
        (rLLM/verl agent loop), where the policy emits the moves and the env enforces the
        sequence, fulfils EXECUTE deterministically, and scores only at the terminal move.
    """

    def __init__(self, task: DiscoveryTask, oracle: DomainOracle):
        self.task = task
        self.oracle = oracle
        # episode state (used by reset/step)
        self._ctx: dict = {}
        self._records: list[dict] = []
        self._phase: int = 0
        self._design: Optional[Design] = None
        self._execution: Optional[Execution] = None
        self._verdict: Optional[Verdict] = None

    # ---- whole-episode driver (parity with the old run_rollout) ------------- #
    def rollout(self, llm, **run_kw) -> dict:
        """Run FRAME → DESIGN → EXECUTE → RESOLVE end to end. ``run_kw`` is forwarded to
        the oracle's ``execute`` (e.g. relax/fmax/steps). Returns a dict bundle the caller
        lays onto the IR via :meth:`to_trajectory`."""
        ctx = {"goal": self.task.goal()}
        records: list[dict] = []

        for mid in FRAME_MOVES:
            rec = _reasoning_move(llm, mid, ctx, self.oracle.persona)
            ctx[_CTX_KEY[mid]] = rec["thought"]
            records.append(rec)

        design = self.oracle.design(llm, ctx)
        records.extend(design.records)

        execution = self.oracle.execute(design, **run_kw)
        records.append(execution.record)
        ctx["observations"] = (execution.record["observation"] + " | Reference: "
                               + self.oracle.reference_note(execution))

        for mid in RESOLVE_MOVES:
            rec = _reasoning_move(llm, mid, ctx, self.oracle.persona)
            ctx[_CTX_KEY.get(mid, mid)] = rec["thought"]
            records.append(rec)

        verdict = self.oracle.assess(execution)
        self._ctx, self._records = ctx, records
        self._design, self._execution, self._verdict = design, execution, verdict
        return {"records": records, "design": design, "execution": execution, "verdict": verdict}

    # ---- gym-like interface for an external RL policy ----------------------- #
    # Move plan the policy walks: reasoning moves it authors, the design moves the oracle
    # parses from the policy's spec, and a single env-fulfilled EXECUTE.
    _PLAN = FRAME_MOVES + ("design", "run_experiment") + RESOLVE_MOVES

    def reset(self) -> dict:
        """Begin an episode. Returns the initial observation: the framed tension plus the
        ordered move plan the policy must follow (the state machine = §14 scaffolding)."""
        self._ctx = {"goal": self.task.goal()}
        self._records, self._phase = [], 0
        self._design = self._execution = self._verdict = None
        return {"goal": self.task.goal(), "tension": self.task.tension,
                "handle": self.task.handle, "next_move": self._PLAN[0],
                "move_plan": list(self._PLAN), "persona": self.oracle.persona}

    @property
    def expected_move(self) -> Optional[str]:
        return self._PLAN[self._phase] if self._phase < len(self._PLAN) else None

    def step(self, action: dict) -> dict:
        """Advance one move. ``action`` = ``{"move": <id>, "thought": str, "spec": {...}?}``.

        Reward is 0 for every non-terminal move; only ``draw_conclusion`` (the terminal)
        carries the oracle verdict's reward. ``run_experiment`` is env-fulfilled — the
        policy emits it, the env runs the oracle. Returns
        ``{observation, reward, done, info}``."""
        expected = self.expected_move
        if expected is None:
            raise RuntimeError("episode already terminated; call reset()")
        move = action.get("move", expected)
        if move != expected:
            # state-machine guard (scaffolding): wrong move → no progress, no reward.
            return {"observation": f"expected move {expected!r}, got {move!r}",
                    "reward": 0.0, "done": False, "info": {"rejected": True,
                    "next_move": expected}}

        obs, reward, done, info = "", 0.0, False, {}

        if move in FRAME_MOVES:
            thought = action.get("thought", "")
            self._ctx[_CTX_KEY[move]] = thought
            self._records.append({"move": move, "thought": thought, "args": {}, "observation": thought})
            obs = thought
        elif move == "design":
            # the policy's spec is materialised by the oracle (parsing/clamp = domain).
            self._design = self.oracle.design_from_action(action, self._ctx)
            self._records.extend(self._design.records)
            obs = "; ".join(r["observation"] for r in self._design.records)
        elif move == "run_experiment":
            if self._design is None:
                return {"observation": "no design to run", "reward": 0.0, "done": False,
                        "info": {"rejected": True, "next_move": "design"}}
            self._execution = self.oracle.execute(self._design)
            self._records.append(self._execution.record)
            self._ctx["observations"] = (self._execution.record["observation"] + " | Reference: "
                                         + self.oracle.reference_note(self._execution))
            obs = self._execution.record["observation"]
        elif move in RESOLVE_MOVES:
            thought = action.get("thought", "")
            self._ctx[_CTX_KEY.get(move, move)] = thought
            self._records.append({"move": move, "thought": thought, "args": {}, "observation": thought})
            obs = thought
            if move == "draw_conclusion":                 # terminal → score
                self._verdict = self.oracle.assess(self._execution)
                reward, done = (1.0 if self._verdict.rewarded else 0.0), True
                info = {"verdict": self._verdict, "admissible": self._verdict.rewarded}

        self._phase += 1
        info.setdefault("next_move", self.expected_move)
        return {"observation": obs, "reward": reward, "done": done, "info": info}

    # ---- IR wiring (domain-agnostic: only generic flags + oracle payloads) -- #
    def to_trajectory(self, pattern):
        """Lay the recorded moves onto the engine IR and wire the verdict as the reward.
        ``pattern`` is the DiscoveryPattern the records were built for (kept so the existing
        ``reconstruct.discovery_trajectory.build_trajectory`` formatting is reused)."""
        from reconstruct.discovery_trajectory import build_trajectory
        if self._verdict is None:
            raise RuntimeError("no verdict yet — run rollout() or step() to a conclusion")
        v = self._verdict
        traj = build_trajectory(pattern, self._records)
        traj.terminal_step.reward = Reward(
            value=1.0 if v.rewarded else 0.0, terminal=True, style=RewardStyle.RULE,
            ground_truth=v.ground_truth, verifiable=v.verifiable,
        )
        if v.valid:
            traj.verification = Verification(
                reexecuted=True, reexecute_reproduced=True, physics=v.physics,
                judge_votes=[v.sane, v.decisive], min_judges=2, notes=v.notes_valid,
            )
        else:
            traj.verification = Verification(
                reexecuted=True, reexecute_reproduced=False, physics=v.physics,
                mlip_prefiltered=v.metadata.get("mlip_prefiltered", False),
                judge_votes=[], min_judges=2, notes=v.notes_invalid,
            )
        traj.metadata.update({
            "data_class": "discovery", "train_use": "RL",
            "status": ("reward-verified (terminal recompute)" if v.valid
                       else f"loop-validated ({self.oracle.name})"),
            "reward_source": f"{v.metadata.get('engine', self.oracle.engine_name)} live recompute",
            "rollout": True,
        })
        traj.metadata.update(v.metadata.get("trajectory_metadata", {}))
        return traj
