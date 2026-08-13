"""Semantic action lifting: raw provenance Steps → clean agent tool calls (§4→§7).

A raw AiiDA CalcJob Step has ``action.name="Legacy JobCalculation"`` and
``params={the entire QE input dict}`` — faithful, but NOT an agent action. An
agent acts in a *defined tool space*; the meaningful move in a restart chain is
"resubmit with mixing_beta reduced", which is buried in the param diff.

This module defines a small SCF-convergence tool space and lifts a restart-chain
trajectory onto it:

  * step 0           → ``run_scf(...)``      (the initial submission, key inputs)
  * each restart i>0 → ``restart_scf(...)``  (ONLY the changed parameters)

It also enriches observation text into grounded natural language (convergence /
energy), and sets the trajectory ``system_prompt`` to the tool definitions (§7).
Thoughts are left for ``reconstruct.thought_completion`` to fill afterwards.

This is a domain design choice (the action vocabulary), kept separate from the
adapter so the raw IR stays faithful and the tool space can evolve independently.
"""

from __future__ import annotations

from typing import Any, Optional

from ir import Action, Trajectory

# --------------------------------------------------------------------------- #
# Tool space (goes into the SFT system message, §7).
# --------------------------------------------------------------------------- #
SCF_TOOLS = [
    {
        "name": "run_scf",
        "description": "Submit an initial plane-wave DFT SCF calculation for a structure.",
        "parameters": {
            "formula": "chemical formula of the structure",
            "functional": "XC/vdW functional, e.g. rvv10",
            "ecutwfc": "wavefunction cutoff (Ry)",
            "ecutrho": "charge-density cutoff (Ry)",
            "conv_thr": "SCF convergence threshold",
            "smearing": "smearing type",
            "degauss": "smearing width (Ry)",
        },
    },
    {
        "name": "restart_scf",
        "description": ("Restart the SCF from the previous run's charge density, changing only the "
                        "given parameters. Use to recover from non-convergence (e.g. lower "
                        "mixing_beta to damp charge-density oscillations)."),
        "parameters": {
            "mixing_beta": "linear mixing factor (lower = more damping)",
            "restart_mode": "'restart' to continue from previous density",
            "...": "any other QE parameter to override",
        },
    },
]

SYSTEM_PROMPT = (
    "You are a DFT computational-chemistry agent driving plane-wave SCF calculations to "
    "convergence. You have these tools:\n"
    "- run_scf(formula, functional, ecutwfc, ecutrho, conv_thr, smearing, degauss): start an SCF.\n"
    "- restart_scf(**changes): restart from the previous charge density, overriding only the given "
    "parameters; use it to recover from non-convergence.\n"
    "Think step by step (Thought), call exactly one tool per step (Action), and after convergence "
    "give a Final Answer with the total energy."
)

# parameters worth surfacing on the initial run_scf action (others stay implicit)
_INITIAL_KEYS = {
    "SYSTEM": ["input_dft", "ecutwfc", "ecutrho", "occupations", "smearing", "degauss",
              "nbnd", "nspin", "starting_magnetization"],
    "ELECTRONS": ["conv_thr", "mixing_beta", "electron_maxstep"],
    "CONTROL": ["calculation"],
}
_RENAME = {"input_dft": "functional", "calculation": "calculation"}


def _flatten(qe_params: dict[str, Any]) -> dict[str, Any]:
    """{'SYSTEM':{'ecutwfc':40},...} → {'SYSTEM.ecutwfc':40, ...} (+ top-level scalars)."""
    flat: dict[str, Any] = {}
    for sec, d in qe_params.items():
        if isinstance(d, dict):
            for k, v in d.items():
                flat[f"{sec}.{k}"] = v
        else:
            flat[sec] = d
    return flat


def _qe_params(step) -> dict[str, Any]:
    return (step.action.params or {}).get("parameters") or {}


def _diff(prev: dict[str, Any], cur: dict[str, Any]) -> dict[str, Any]:
    """Changed/added flat params cur vs prev, with section prefixes stripped to leaf names."""
    out: dict[str, Any] = {}
    for k, v in cur.items():
        if k not in prev or prev[k] != v:
            leaf = k.split(".")[-1]
            out[leaf] = v
    return out


def lift_restart_chain(traj: Trajectory, *, formula: Optional[str] = None) -> Trajectory:
    """Rewrite a raw restart-chain trajectory's actions into the SCF tool space.

    Mutates and returns ``traj``: sets ``system_prompt``, rewrites each step's
    ``action`` to ``run_scf`` / ``restart_scf`` with minimal params, and rewrites
    observation text into grounded natural language. Raw params are preserved in
    ``action.params['_raw_changes']`` only as needed for audit-free minimalism we
    drop them; the original IR remains available upstream if needed.
    """
    traj.system_prompt = SYSTEM_PROMPT
    prev_flat: dict[str, Any] = {}

    for i, step in enumerate(traj.steps):
        flat = _flatten(_qe_params(step))
        if i == 0:
            params: dict[str, Any] = {}
            if formula:
                params["formula"] = formula
            for sec, keys in _INITIAL_KEYS.items():
                for k in keys:
                    fk = f"{sec}.{k}"
                    if fk in flat:
                        params[_RENAME.get(k, k)] = flat[fk]
            step.action = Action(name="run_scf", params=params, raw_ref=step.action.raw_ref)
        else:
            changes = _diff(prev_flat, flat)
            # drop noise that isn't a real recovery knob
            changes.pop("max_seconds", None)
            step.action = Action(name="restart_scf", params=changes, raw_ref=step.action.raw_ref)
        prev_flat = flat

        # grounded observation text
        _rewrite_observation(step)

    # episode-level goal hint (will be replaced by Humpback back-translation)
    if formula and (not traj.goal or "pk:" in traj.goal):
        traj.goal = f"Compute the converged DFT SCF ground-state energy of {formula}."
    return traj


def _rewrite_observation(step) -> None:
    o = step.observation
    res = (o.content or {}).get("res") or {}
    energy = res.get("energy")
    if o.exit_status == 0:
        e = f"; total energy = {energy:.4f} eV" if isinstance(energy, (int, float)) else ""
        o.text = f"Observation: SCF converged{e}."
    else:
        o.text = (f"Observation: SCF did NOT converge (exit_status={o.exit_status}); "
                  "charge density failed to settle.")


# =========================================================================== #
# Named SCF recovery tool space (data-derived, see examples/analyze_recovery_vocab.py).
#
# Replaces the catch-all restart_scf with a vocabulary of NAMED recovery moves,
# each carrying intent. The move set + frequencies were mined from real mc2d
# (364 chains) + ACWF (cross-source) restart chains: reducing mixing_beta (the
# dominant move, geometric ×0.8), adding empty bands, restarting from the saved
# charge density, reseeding magnetization, and full resubmission. Classification
# is DETERMINISTIC from the input diff (no LLM) — the agent learns "diagnose →
# named recovery", not "change some param".
# =========================================================================== #
SCF_NAMED_TOOLS = [
    {"name": "run_scf",
     "description": "Submit an initial plane-wave DFT SCF calculation for a structure.",
     "parameters": {"formula": "chemical formula", "functional": "XC/vdW functional",
                    "ecutwfc": "wavefunction cutoff (Ry)", "ecutrho": "charge-density cutoff (Ry)",
                    "conv_thr": "SCF convergence threshold", "mixing_beta": "linear charge mixing factor",
                    "smearing": "smearing type", "degauss": "smearing width (Ry)",
                    "nbnd": "number of bands", "nspin": "1=non-spin, 2=spin-polarized",
                    "starting_magnetization": "per-species initial magnetization"}},
    {"name": "damp_charge_mixing",
     "description": ("Restart from the saved charge density with a SMALLER linear mixing factor to "
                     "suppress charge-density oscillations that prevented SCF convergence."),
     "parameters": {"mixing_beta": "new (smaller) mixing factor, e.g. previous × 0.8"}},
    {"name": "add_empty_bands",
     "description": ("Restart with MORE bands (and reuse the saved charge density). Use when a metal / "
                     "f-electron / smeared system lacked enough empty states to converge."),
     "parameters": {"nbnd": "new (larger) number of bands"}},
    {"name": "restart_from_charge_density",
     "description": "Restart continuing from the previous run's saved charge density, no parameter change.",
     "parameters": {}},
    {"name": "reseed_magnetization",
     "description": "Restart a spin-polarized run with a revised initial magnetization to escape a bad magnetic state.",
     "parameters": {"starting_magnetization": "revised per-species initial magnetization"}},
    {"name": "resubmit_from_scratch",
     "description": "Resubmit the calculation from scratch with a revised input set (a full reset).",
     "parameters": {"changes": "the revised key inputs"}},
    {"name": "adjust_scf_parameters",
     "description": "Restart adjusting other SCF knob(s) not covered by the named moves above (long tail).",
     "parameters": {"changes": "the changed parameter(s)"}},
]

SCF_NAMED_SYSTEM_PROMPT = (
    "You are a DFT computational-chemistry agent driving plane-wave SCF calculations to convergence. "
    "You start a calculation with run_scf, then if it fails to converge you recover with ONE named move:\n"
    "- damp_charge_mixing(mixing_beta): smaller mixing to damp charge-density oscillations.\n"
    "- add_empty_bands(nbnd): more bands for metals / f-electrons / smeared systems.\n"
    "- restart_from_charge_density(): continue from the saved density, no change.\n"
    "- reseed_magnetization(starting_magnetization): revise magnetic seeding (spin-polarized).\n"
    "- resubmit_from_scratch(changes) / adjust_scf_parameters(changes): full reset / other knob.\n"
    "Diagnose the observation, then call exactly one tool per step (Thought then Action). After "
    "convergence give a Final Answer with the total energy."
)

# DFPT / phonon (ph.x) restart vocabulary — a DISTINCT sub-process; SCF lifting skips these.
_PHONON_KEYS = {"alpha_mix(1)", "fildrho", "fildvscf", "epsil", "tr2_ph", "niter_ph", "recover", "fildyn"}
_DROP_KEYS = {"max_seconds"}


def is_phonon_chain(traj: Trajectory) -> bool:
    """True if any step's QE params touch DFPT/phonon-only keys (ph.x, not pw.x SCF)."""
    for s in traj.steps:
        if _PHONON_KEYS & set(_flatten(_qe_params(s))):
            return True
    return False


def _leaf_flatten(qe_params: dict[str, Any]) -> dict[str, Any]:
    """{'ELECTRONS':{'mixing_beta':0.4},...} → {'mixing_beta':0.4, ...} (leaf names only)."""
    out: dict[str, Any] = {}
    for sec, d in (qe_params or {}).items():
        if isinstance(d, dict):
            out.update(d)
        else:
            out[sec] = d
    return out


def _num(x):
    return x if isinstance(x, (int, float)) else None


def classify_restart(prev_flat: dict[str, Any], cur_flat: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Map a restart's param diff onto ONE named SCF recovery tool (deterministic)."""
    changed = {k: cur_flat.get(k) for k in set(prev_flat) | set(cur_flat)
               if prev_flat.get(k) != cur_flat.get(k) and k not in _DROP_KEYS}
    keys = set(changed)
    if not keys:
        return "restart_from_charge_density", {}
    mb_old, mb_new = _num(prev_flat.get("mixing_beta")), _num(cur_flat.get("mixing_beta"))
    if "mixing_beta" in keys and mb_old is not None and mb_new is not None and mb_new < mb_old:
        return "damp_charge_mixing", {"mixing_beta": mb_new}
    nb_old, nb_new = _num(prev_flat.get("nbnd")), _num(cur_flat.get("nbnd"))
    if "nbnd" in keys and nb_new is not None and nb_new > (nb_old or 0):
        return "add_empty_bands", {"nbnd": nb_new}
    if "starting_magnetization" in keys:
        return "reseed_magnetization", {"starting_magnetization": cur_flat.get("starting_magnetization")}
    if keys <= {"restart_mode", "startingpot"}:
        return "restart_from_charge_density", {}
    if len(keys) >= 5:
        return "resubmit_from_scratch", {k: changed[k] for k in sorted(keys)[:6]}
    return "adjust_scf_parameters", changed


def _diagnose_observation(step) -> None:
    """Enrich the observation into a diagnosed signal (problem only — never the fix, §19)."""
    o = step.observation
    res = (o.content or {}).get("res") or {}
    it = res.get("scf_iterations") or res.get("total_number_of_scf_iterations")
    if o.exit_status == 0:
        e = res.get("energy")
        parts = ["Observation: SCF converged"]
        if it:
            parts.append(f" in {it} iterations")
        if isinstance(e, (int, float)):
            parts.append(f"; total energy = {e:.4f} eV")
        o.text = "".join(parts) + "."
    else:
        # Report ONLY hard facts the archive actually records: exit_status + how many
        # SCF iterations ran + that no converged energy was produced. Do NOT name a
        # failure *mechanism* ("charge density failed to settle" / "oscillations"):
        # exit_status here is a generic non-convergence code, identical across causes,
        # so any mechanism claim would be an unsupported diagnosis that also leaks the
        # fix direction (§19, §20.3). The causal lint (recovery_and_thought_ok) keeps
        # reconstructed thoughts within these facts.
        after = f" after {it} iterations" if it else ""
        o.text = (f"Observation: SCF did NOT converge (exit_status={o.exit_status}); "
                  f"the run terminated{after} without producing a converged ground-state energy.")


def lift_restart_chain_named(traj: Trajectory, *, formula: Optional[str] = None) -> Trajectory:
    """Lift a raw SCF restart chain onto the NAMED recovery tool space + diagnosed observations.

    step 0 → run_scf(...); each restart → the classified named move. Mutates and returns traj.
    Caller should skip phonon chains via ``is_phonon_chain`` first.
    """
    traj.system_prompt = SCF_NAMED_SYSTEM_PROMPT
    prev_leaf: dict[str, Any] = {}
    for i, step in enumerate(traj.steps):
        qp = _qe_params(step)
        flat = _flatten(qp)          # section.key — for run_scf initial-key lookup
        leaf = _leaf_flatten(qp)     # leaf names — for restart classification
        if i == 0:
            params: dict[str, Any] = {}
            if formula:
                params["formula"] = formula
            for sec, keys in _INITIAL_KEYS.items():
                for k in keys:
                    fk = f"{sec}.{k}"
                    if fk in flat:
                        params[_RENAME.get(k, k)] = flat[fk]
            step.action = Action(name="run_scf", params=params, raw_ref=step.action.raw_ref)
        else:
            name, p = classify_restart(prev_leaf, leaf)
            step.action = Action(name=name, params=p, raw_ref=step.action.raw_ref)
        prev_leaf = leaf
        _diagnose_observation(step)
    if formula and (not traj.goal or "pk:" in traj.goal):
        traj.goal = f"Compute the converged DFT SCF ground-state energy of {formula}."
    return traj
