"""Shared plumbing for reconstruction techniques (CLAUDE.md §5).

Reconstruction narrates a *result* back into a *process* (thought/goal/relabel).
Every technique here takes a pluggable :class:`LLMClient` so the pipeline wires
up without binding to any vendor, and tests inject a deterministic fake.

Two cross-cutting rules from §1 / §11 are enforced as code, not convention:

  * **Keep-if-correct (§1.2):** an LLM-reconstructed artifact may only survive if
    a *verifier* (typically ``verify.reexecute``) confirms the product still
    reproduces. ``gated_reconstruct`` wraps any technique in this gate.
  * **Provenance (§11):** every LLM step logs prompt ref + teacher model +
    whether it passed the gate, onto ``Trajectory.provenance``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from ir import ReconstructMethod, Trajectory


class LLMClient(Protocol):
    """Minimal teacher-LLM surface used by all reconstruction techniques."""

    def complete(self, prompt: str, *, system: Optional[str] = None) -> str: ...

    @property
    def model_id(self) -> str: ...


@dataclass
class ReconstructLog:
    method: ReconstructMethod
    teacher_model: str
    prompt_ref: str
    kept: bool
    detail: dict


def stamp_provenance(traj: Trajectory, log: ReconstructLog) -> None:
    """Record an LLM reconstruction step on the trajectory's provenance (§11)."""
    traj.provenance.reconstruct_method = log.method
    traj.provenance.teacher_model = log.teacher_model
    traj.provenance.reconstruct_prompt_ref = log.prompt_ref


# A verifier returns True iff the reconstructed product still reproduces (§1.2).
Verifier = Callable[[Trajectory], bool]


def gated_reconstruct(
    traj: Trajectory,
    apply: Callable[[Trajectory], ReconstructLog],
    verifier: Optional[Verifier],
) -> tuple[Trajectory, ReconstructLog]:
    """Apply a reconstruction, then KEEP only if the verifier passes (§1.2).

    `apply(traj)` mutates the trajectory in place (adds thoughts / goal / relabel)
    and returns a :class:`ReconstructLog`. If `verifier` is provided and returns
    False, the reconstruction is rolled back conceptually (caller should discard)
    by setting ``log.kept=False`` and leaving the trajectory un-stamped. With no
    verifier, the artifact is provisional and ``kept`` reflects only that it ran
    — never export such a trajectory without a downstream ``verify`` pass.
    """
    log = apply(traj)
    if verifier is not None:
        log.kept = bool(verifier(traj))
    if log.kept:
        stamp_provenance(traj, log)
    return traj, log
