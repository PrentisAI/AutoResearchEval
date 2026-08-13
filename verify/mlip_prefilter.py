"""MLIP prefilter (§6) — cheap screen before expensive DFT re-execution.

CLAUDE.md §6: "先 MLIP 预过滤再上 DFT" — use a universal MLIP (MACE-MP-0 /
CHGNet / M3GNet) to quickly reject obviously-bad structures/relaxations before
spending DFT, then re-run survivors at DFT and apply ``verify.physics_checks``.

The heavy ML packages are optional extras (``pip install -e ".[mlip]"``) and
pinned in requirements.lock; they are imported lazily behind ``_load_calculator``.
The *interface* — :func:`prefilter` and the :class:`Calculator` protocol — is
defined here so the pipeline wires up without the models installed; calling it
without the extra raises a clear, actionable error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol


class Calculator(Protocol):
    """Minimal ASE-style calculator surface the prefilter needs."""

    def get_potential_energy(self) -> float: ...
    def get_forces(self) -> Any: ...


@dataclass
class PrefilterResult:
    passed: bool
    detail: dict


_MODEL_ANCHORS = {
    "mace": "mace-torch==0.3.16",   # MACE-MP-0
    "chgnet": "chgnet==0.4.2",
    "m3gnet": "matgl==4.0.1",       # torch-geometric backend (no DGL)
}


def _load_calculator(model: str):
    """Lazily construct an MLIP ASE calculator; actionable error if extra absent."""
    model = model.lower()
    try:
        if model == "mace":
            from mace.calculators import mace_mp  # type: ignore
            return mace_mp(model="medium", default_dtype="float64")
        if model == "chgnet":
            from chgnet.model.dynamics import CHGNetCalculator  # type: ignore
            return CHGNetCalculator()
        if model == "m3gnet":
            import matgl  # type: ignore
            from matgl.ext.ase import PESCalculator  # type: ignore
            pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
            return PESCalculator(pot)
    except ImportError as exc:  # pragma: no cover - only without the extra
        anchor = _MODEL_ANCHORS.get(model, "the matching package")
        raise ImportError(
            f"MLIP model '{model}' needs its package. Install the extra and pin it:\n"
            f'    pip install -e ".[mlip]"   # then set {anchor} in requirements.lock'
        ) from exc
    raise ValueError(f"unknown MLIP model '{model}'; choose from {sorted(_MODEL_ANCHORS)}")


def prefilter(
    structure: Any,
    *,
    model: str = "mace",
    max_force_tol: float = 0.5,
    reference_energy_per_atom: Optional[float] = None,
    energy_tol_per_atom: float = 0.5,
    _calculator: Optional[Calculator] = None,
) -> PrefilterResult:
    """Cheap MLIP screen of a structure before DFT.

    Rejects if max force exceeds ``max_force_tol`` (eV/Å) or, when a reference
    energy/atom is given, if the MLIP energy/atom deviates beyond
    ``energy_tol_per_atom``. Pass ``_calculator`` to inject a (real or fake)
    calculator and bypass model loading — used by tests and for custom MLIPs.

    Note: a prefilter is a *screen*, not a verifier. Passing here only earns a
    structure a DFT re-run; admission still requires DFT + physics_checks +
    multi-judge (§1, §6).
    """
    calc = _calculator if _calculator is not None else _load_calculator(model)

    # Attach the calculator to the structure when possible. Modern ASE Atoms
    # expose a ``.calc`` property; older code used ``set_calculator``.
    try:
        structure.calc = calc
    except (AttributeError, TypeError):
        setter = getattr(structure, "set_calculator", None)
        if callable(setter):
            setter(calc)

    # Energy/forces: prefer the ASE *atoms* API (``atoms.get_potential_energy``),
    # which passes the atoms through to the calculator — calling the calculator
    # directly with no atoms attached raises. Fall back to the calculator's own
    # methods for injected / non-ASE calculators that already hold their state.
    if hasattr(structure, "get_potential_energy"):
        energy = structure.get_potential_energy()
        forces = structure.get_forces()
    else:
        energy = calc.get_potential_energy()
        forces = calc.get_forces()
    n_atoms = _n_atoms(structure) or 1

    max_force = _max_norm(forces)
    detail: dict = {"model": model, "energy": energy, "max_force": max_force, "n_atoms": n_atoms}

    passed = max_force is not None and max_force <= max_force_tol
    if reference_energy_per_atom is not None:
        e_per_atom = energy / n_atoms
        dev = abs(e_per_atom - reference_energy_per_atom)
        detail.update(e_per_atom=e_per_atom, ref_e_per_atom=reference_energy_per_atom, energy_dev=dev)
        passed = passed and dev <= energy_tol_per_atom

    # Coerce out of numpy bool/float so downstream JSON + identity checks behave.
    return PrefilterResult(passed=bool(passed), detail=detail)


def _n_atoms(structure: Any) -> Optional[int]:
    try:
        return len(structure)
    except TypeError:
        return getattr(structure, "num_atoms", None)


def _max_norm(forces: Any) -> Optional[float]:
    if forces is None:
        return None
    try:
        return max((sum(c * c for c in row)) ** 0.5 for row in forces)
    except TypeError:
        return None
