"""Unified intermediate representation (IR) for SciCoder trajectories.

This module is the **convergence point** of the whole engine (CLAUDE.md §8):

    source ──adapter──▶ IR ──reconstruct──▶ verify ──filter──▶ export

Every source (AiiDA provenance, atomate2 TaskDocs, Custodian logs, MP tasks,
mlflow/wandb runs, papers+repos, commits, notebooks) is first lowered into the
types here. Reconstruction (§5), verification (§6), filtering, and export (§7)
all operate on the IR and nothing else. Adding a new source therefore means
writing exactly one adapter that emits `Trajectory` objects.

Design follows the §4 mapping rules and the §7 SFT/RLVR output contract:

    episode/subgoal-tree ← top-level WorkChainNode (recursive via .called)
    action               ← each CalcJobNode
    action_params        ← input Dict node (INCAR/KPOINTS/...)
    observation          ← output Data node + parsed node.res, plus status fields
    reward/terminal      ← exit_status==0 AND physics validator passed
    thought              ← reconstructed (Thought Completion / STaR), may be None

Two hard invariants from §1 are encoded as data, not prose:

  * Nothing is `verified` until an execution check sets it (§1.1). The IR carries
    a `Verification` record; `Trajectory.is_admissible()` refuses un-verified or
    un-gated trajectories.
  * Failure branches are first-class, never dropped (§1.3, §10). A `Step` with a
    non-zero exit / excepted state / Custodian correction is flagged
    `is_failure_branch=True` and retained.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class SourceType(str, Enum):
    """Where a trajectory was reconstructed from (CLAUDE.md §2 three layers)."""

    # layer 3 — provenance / tracking (⭐ the novel, independent space)
    AIIDA = "aiida"
    ATOMATE2_TASKDOC = "atomate2_taskdoc"
    MP_TASK = "mp_task"
    CUSTODIAN = "custodian"
    MLFLOW = "mlflow"
    WANDB = "wandb"
    # layer 2 — version control / notebooks
    COMMIT = "commit"
    NOTEBOOK = "notebook"
    # layer 1 — papers + repos
    PAPER_REPO = "paper_repo"
    # synthetic / rollout
    ROLLOUT = "rollout"


class ProcessState(str, Enum):
    """AiiDA-style process state (CLAUDE.md §4). Terminal = last three."""

    CREATED = "created"
    RUNNING = "running"
    WAITING = "waiting"
    FINISHED = "finished"   # terminal — check exit_status for ok/fail
    EXCEPTED = "excepted"   # terminal — failure branch, KEEP (§1.3)
    KILLED = "killed"       # terminal — failure branch, KEEP (§1.3)

    @property
    def is_terminal(self) -> bool:
        return self in (ProcessState.FINISHED, ProcessState.EXCEPTED, ProcessState.KILLED)


class ReconstructMethod(str, Enum):
    """How thought/goal was recovered (CLAUDE.md §5). NONE = literal from source."""

    NONE = "none"
    STAR = "star"                          # STaR rationalization (keep-if-correct)
    HUMPBACK = "humpback"                  # back-translation of task/goal
    THOUGHT_COMPLETION = "thought_completion"
    HINDSIGHT_RELABEL = "hindsight_relabel"  # AgentHER


class RewardStyle(str, Enum):
    """verl-style reward_model.style (CLAUDE.md §7)."""

    RULE = "rule"      # deterministic physics / fail-to-pass check
    MODEL = "model"    # judge model
    HYBRID = "hybrid"


class Difficulty(str, Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


# --------------------------------------------------------------------------- #
# Provenance — every trajectory is source-traceable (§1.4, §11)
# --------------------------------------------------------------------------- #
class Provenance(BaseModel):
    """`{source_type, source_id, artifact_uri, reconstruct_method, verified_by}`.

    Carried on every admissible trajectory for audit + de-contamination (§11).
    """

    source_type: SourceType
    source_id: str = Field(..., description="stable id within the source, e.g. AiiDA pk/uuid, MP task_id, commit sha")
    artifact_uri: Optional[str] = Field(None, description="pointer to the raw artifact (path/URL/uri)")
    reconstruct_method: ReconstructMethod = ReconstructMethod.NONE
    verified_by: list[str] = Field(default_factory=list, description="names of verifiers/judges that signed off")

    # For every LLM reconstruction step we log prompt + teacher version + gate
    # outcome (§11). Kept here so provenance travels with the data.
    teacher_model: Optional[str] = None
    reconstruct_prompt_ref: Optional[str] = Field(None, description="ref/hash of the reconstruction prompt")


# --------------------------------------------------------------------------- #
# Action / Observation / Reward
# --------------------------------------------------------------------------- #
class Action(BaseModel):
    """A single agent action — e.g. one CalcJobNode (§4), one tool call, one cell."""

    name: str = Field(..., description="action / tool / task_type, e.g. 'relax', 'static', 'bandstructure'")
    params: dict[str, Any] = Field(default_factory=dict, description="action_params, e.g. INCAR/KPOINTS dict")
    raw_ref: Optional[str] = Field(None, description="pointer back to the source node/cell/diff")


class Observation(BaseModel):
    """Environment feedback after an action (§4).

    Holds both the parsed result and the raw status fields that drive reward.
    """

    content: dict[str, Any] = Field(default_factory=dict, description="parsed result, e.g. node.res / TaskDoc output")
    text: Optional[str] = Field(None, description="human/agent-facing rendering, e.g. 'Observation: converged=True ...'")
    raw_ref: Optional[str] = None

    # status fields (§4) — drive reward & failure-branch detection
    process_state: Optional[ProcessState] = None
    exit_status: Optional[int] = Field(None, description="0 = success; non-zero = failure (KEEP, §1.3)")
    exit_message: Optional[str] = Field(None, description="failure-branch diagnostic")
    exception: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        """Process-level success: finished-ok with zero exit status.

        NOTE: this is *necessary, not sufficient*. Dataset admission additionally
        requires a passed physics / fail-to-pass verifier (§1.1, §6). Never let a
        model grade itself where a deterministic check exists.
        """
        if self.exit_status is not None and self.exit_status != 0:
            return False
        if self.process_state is not None and self.process_state != ProcessState.FINISHED:
            return False
        return True


class Reward(BaseModel):
    """Verifiable reward (§7). `value` is meaningful only once `verified`."""

    value: float = 0.0
    terminal: bool = False
    style: RewardStyle = RewardStyle.RULE
    ground_truth: Optional[Any] = Field(None, description="reward_model.ground_truth (verl style)")
    # physics / fail-to-pass check results carried as verifiable reward (§7).
    verifiable: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Step
# --------------------------------------------------------------------------- #
class Step(BaseModel):
    """One (thought?, action, observation, reward?) tuple in a trajectory.

    `thought` is optional and, when present, is a *reconstructed* artifact whose
    provenance method is recorded on the trajectory; it must only survive if the
    re-execution reproduced the product (STaR keep-if-correct, §1.2).
    """

    index: int = Field(..., ge=0)
    thought: Optional[str] = None
    action: Action
    observation: Observation
    reward: Optional[Reward] = None

    is_failure_branch: bool = Field(
        False,
        description="non-zero exit / excepted / killed / Custodian-corrected step. KEEP — only source of error→recovery supervision (§1.3, §10).",
    )
    # Custodian correction that recovered from this step's error, if any (§4).
    # Shape e.g. {"error": "brmix", "actions": [{"dict": "INCAR", "action": {"_set": {"IMIX": 1}}}]}
    correction: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def _flag_failure_branch(self) -> "Step":
        """Auto-flag failure branches so adapters can't silently drop them."""
        obs = self.observation
        is_fail = (
            (obs.exit_status is not None and obs.exit_status != 0)
            or (obs.process_state in (ProcessState.EXCEPTED, ProcessState.KILLED))
            or (obs.exception is not None)
            or (self.correction is not None)
        )
        if is_fail:
            self.is_failure_branch = True
        return self


# --------------------------------------------------------------------------- #
# Verification record — the hard gate (§1.1, §1.5, §6)
# --------------------------------------------------------------------------- #
class Verification(BaseModel):
    """Outcome of re-execution + multi-judge gating for a whole trajectory.

    A trajectory is admissible to the dataset ONLY if `reexecuted` is True and at
    least `min_judges` judges agree (default 2, per §1.5 / AgentHER).
    """

    reexecuted: bool = Field(False, description="re-ran end-to-end (fail-to-pass / physics) — §1.1")
    reexecute_reproduced: bool = Field(False, description="re-execution reproduced the original product")
    physics: dict[str, Any] = Field(default_factory=dict, description="§6 check results (scf_converged, e_above_hull, energy_drift, ...)")
    judge_votes: list[bool] = Field(default_factory=list, description="independent judge accept/reject votes (§1.5)")
    min_judges: int = 2
    mlip_prefiltered: bool = Field(False, description="passed cheap MLIP prefilter before DFT (§6)")
    notes: Optional[str] = None

    @property
    def judges_agree(self) -> bool:
        accepts = sum(1 for v in self.judge_votes if v)
        return accepts >= self.min_judges

    @property
    def passed(self) -> bool:
        """Hard gate: re-executed AND reproduced AND ≥min_judges agree."""
        return self.reexecuted and self.reexecute_reproduced and self.judges_agree


# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #
class Trajectory(BaseModel):
    """A reconstructed, (to-be-)verified agent trajectory — the IR unit.

    Subgoal structure is represented by `parent_id` + ordering across a set of
    trajectories (AiiDA WorkChain `.called` recursion → child episodes; `.caller`
    → parent context, §4). A single trajectory holds one ordered list of steps.
    """

    id: str
    goal: str = Field(..., description="natural-language task goal (Humpback back-translated, §5)")
    task_spec: Optional[str] = Field(None, description="formal task spec / reference solution pointer (layer-1 §2)")
    system_prompt: Optional[str] = Field(None, description="tools/instructions for the system message (§7)")

    steps: list[Step] = Field(default_factory=list)
    provenance: Provenance
    verification: Verification = Field(default_factory=Verification)

    # subgoal tree (§4): top-level WorkChain expanded via .called
    parent_id: Optional[str] = None
    child_ids: list[str] = Field(default_factory=list)

    difficulty: Optional[Difficulty] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ----- derived views -----
    @property
    def has_failure_branch(self) -> bool:
        return any(s.is_failure_branch for s in self.steps)

    @property
    def terminal_step(self) -> Optional[Step]:
        return self.steps[-1] if self.steps else None

    @property
    def succeeded(self) -> bool:
        """Process-level success of the final step (necessary, not sufficient)."""
        t = self.terminal_step
        return bool(t and t.observation.succeeded)

    def is_admissible(self) -> bool:
        """May this trajectory enter the SFT/RLVR dataset?

        Encodes §1.1 (execution-verify hard gate) + §1.5 (multi-judge). Export
        refuses anything for which this is False. Note: a *failed* trajectory can
        still be admissible — its verification just asserts the failure (and any
        recovery) is faithfully reproduced; failure branches are wanted (§1.3).
        """
        return self.verification.passed

    def reindex(self) -> "Trajectory":
        """Renumber steps 0..n-1 (e.g. after step-level filtering)."""
        for i, s in enumerate(self.steps):
            s.index = i
        return self
