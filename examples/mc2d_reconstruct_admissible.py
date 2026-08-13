"""Reconstruct ADMISSIBLE, ground-truth trajectories from real mc2d provenance
(CLAUDE.md §2-layer3, §13.1 reconstruction-mode, §19 "only make correct trajectories").

This is the EASY / most-ground-truth tier of the difficulty ramp: the actions and
observations are REAL recorded QE execution from the mc2d archive (11,123 CalcJob
nodes) — nothing is generated from scratch, so correctness is guaranteed by the
record, not by an LLM guessing (contrast: SAB rollout mode, §13.2). We only
(a) semantically lift the raw `Legacy JobCalculation` steps into a clean SCF tool
space, (b) reconstruct the implicit Thought before each action, and (c) admit the
trajectory through THREE deterministic checks.

Honesty boundary (§1.1): we do NOT re-run Quantum ESPRESSO on this box. So
`reexecuted` here means "reloaded the archive nodes and confirmed the trajectory
faithfully reproduces the real recorded execution (exit_status chain + final
energy/structure)" — a real, runnable check, NOT a fake QE re-run. The independent
physics signal is a CHGNet MLIP check on the final structure.

Admission gate (encodes ir.Verification.passed = reexecuted ∧ reproduced ∧ ≥2 judges):
  reexecuted/reproduced  ← FAITHFULNESS: every step's node reloads and its live
                           exit_status matches the trajectory (deterministic).
  judge 1 (physics)      ← CHGNet MLIP on the recovered final structure passes.
  judge 2 (recovery)     ← the chain is a real error→recovery (nonzero…→0 exit)
                           AND reconstructed Thoughts don't leak future outcomes
                           (STaR-style keep-if-correct on the reconstructed reasoning).

Run: python examples/mc2d_reconstruct_admissible.py
"""


from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "examples" / "output"
OUT.mkdir(parents=True, exist_ok=True)

from aiida import load_profile, orm  # noqa: E402

load_profile("mc2d")

from adapters import aiida_walker as W  # noqa: E402
from export import to_sft_react  # noqa: E402
from ir import Reward, RewardStyle, Verification  # noqa: E402
from reconstruct import thought_completion  # noqa: E402
from reconstruct.base import gated_reconstruct  # noqa: E402
from reconstruct.llm_openrouter import OpenRouterClient  # noqa: E402
from reconstruct.tool_lift import is_phonon_chain, lift_restart_chain_named  # noqa: E402
from verify.mlip_prefilter import prefilter  # noqa: E402

TARGET_ADMITTED = 8        # size of the clean batch we want
SCAN_LIMIT = 600           # episodes to scan (error→recovery chains are rare, ~1/25)
MLIP_FORCE_TOL = 2.0       # eV/Å — relaxed structure should have small residual forces


def _native(x):
    """Coerce numpy scalars/containers to JSON-serializable Python types."""
    import numpy as np
    if isinstance(x, dict):
        return {k: _native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_native(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def is_error_recovery(traj) -> bool:
    """Cheap exit-chain pre-filter: a real failure→…→success chain (nonzero…→0).
    Run BEFORE any LLM/MLIP work so the (majority) clean [0→0] chains cost nothing."""
    exits = [s.observation.exit_status for s in traj.steps]
    return exits[-1] == 0 and any(e not in (0, None) for e in exits[:-1])


# --------------------------------------------------------------------------- #
# Provenance helpers (reload the REAL nodes; never trust the transcript alone).
# --------------------------------------------------------------------------- #
def _pk(ref: str | None) -> int | None:
    if ref and ref.startswith("pk:"):
        return int(ref.split(":")[1])
    return None


def root_formula(traj) -> str | None:
    """Reduced chemical formula of the structure the first calc was run on."""
    pk = _pk(traj.steps[0].action.raw_ref)
    if pk is None:
        return None
    node = orm.load_node(pk)
    for t in node.base.links.get_incoming().all():
        if isinstance(t.node, orm.StructureData):
            try:
                return t.node.get_pymatgen().composition.reduced_formula
            except Exception:  # noqa: BLE001
                return t.node.get_formula()
    return None


def final_structure(traj):
    pk = _pk(traj.terminal_step.action.raw_ref)
    if pk is None:
        return None
    node = orm.load_node(pk)
    for t in node.base.links.get_outgoing().all():
        if t.link_label == "output_structure":
            return t.node
    return None


# --------------------------------------------------------------------------- #
# The three deterministic admission checks.
# --------------------------------------------------------------------------- #
def faithfulness_ok(traj) -> bool:
    """Reload every step's node; its live exit_status must match the trajectory.

    This is what makes `reexecuted/reexecute_reproduced` honest WITHOUT a QE re-run:
    we prove the trajectory is a faithful transcription of the real archived run.
    """
    for s in traj.steps:
        pk = _pk(s.action.raw_ref)
        if pk is None:
            return False
        try:
            live = getattr(orm.load_node(pk), "exit_status", "MISSING")
        except Exception:  # noqa: BLE001
            return False
        if live != s.observation.exit_status:
            return False
    return True


def recovery_and_thought_ok(traj) -> bool:
    """Judge 2: a real error→recovery chain whose reconstructed Thoughts don't
    leak future outcomes (STaR-style keep-if-correct on the reasoning)."""
    exits = [s.observation.exit_status for s in traj.steps]
    real_recovery = exits[-1] == 0 and any(e not in (0, None) for e in exits[:-1])
    if not real_recovery:
        return False
    # future-leak lint on reconstructed thoughts
    SUCCESS_WORDS = ("converged", "success", "finished successfully", "settled")
    # CAUSAL lint: the archive only records a generic non-convergence exit_status — it
    # does NOT record *why* it failed (no per-iteration SCF accuracy / oscillation trace
    # is parsed into the Dict). So a reconstructed thought may not assert a specific
    # physical failure MECHANISM the observation can't support — that is hindsight
    # rationalization of what was, in mc2d, a fixed ×0.8 retry policy (§1.2, §20.3).
    MECHANISM_CLAIMS = (
        "oscillat", "charge density", "charge-density", "charge sloshing", "sloshing",
        "failed to settle", "did not settle", "instabilit", "diverg", "too large",
    )
    supported = " ".join((s.observation.text or "") for s in traj.steps).lower()
    for i, s in enumerate(traj.steps):
        th = (s.thought or "").lower()
        if not th:
            return False
        # a step that did NOT converge must not claim convergence/success in its Thought
        if s.observation.exit_status not in (0, None) and any(w in th for w in SUCCESS_WORDS):
            return False
        # the first action cannot have been a restart/recovery (nothing ran before it)
        if i == 0 and ("restart" in th or "recover" in th or "previous run" in th):
            return False
        # a mechanism claim is only allowed if some observation actually states it
        for claim in MECHANISM_CLAIMS:
            if claim in th and claim not in supported:
                return False
    return True


def mlip_physics_ok(traj) -> tuple[bool, dict]:
    """Judge 1: CHGNet MLIP on the recovered final structure — real, runnable."""
    struct = final_structure(traj)
    if struct is None:
        return False, {"error": "no output_structure on terminal node"}
    from pymatgen.io.ase import AseAtomsAdaptor
    atoms = AseAtomsAdaptor.get_atoms(struct.get_pymatgen())
    res = prefilter(atoms, model="chgnet", max_force_tol=MLIP_FORCE_TOL)
    return bool(res.passed), res.detail


# --------------------------------------------------------------------------- #
# Pipeline.
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 78)
    print("mc2d → admissible reconstructed trajectories (reconstruction mode, §13.1)")
    print("=" * 78)
    print(f"scanning up to {SCAN_LIMIT} recovered restart episodes; want {TARGET_ADMITTED} admitted\n")

    llm = OpenRouterClient(max_tokens=400)
    print(f"teacher model (thought completion only): {llm.model_id}\n")

    admitted, scanned, skipped_clean = [], 0, 0
    for traj in W.iter_restart_episodes(min_steps=2, limit=SCAN_LIMIT):
        if not traj.metadata.get("recovered"):
            continue
        # pre-filter to genuine error→recovery BEFORE spending any LLM/MLIP budget
        if not is_error_recovery(traj):
            skipped_clean += 1
            continue
        # DFPT/phonon (ph.x) restarts are a distinct sub-process — out of the SCF tool space
        if is_phonon_chain(traj):
            continue
        scanned += 1
        tag = traj.id
        exits = "→".join(str(s.observation.exit_status) for s in traj.steps)

        # 1) semantic lifting onto the NAMED SCF recovery tool space (data-derived)
        formula = root_formula(traj)
        lift_restart_chain_named(traj, formula=formula)

        # 2) thought completion, gated by the recovery + future-leak verifier (STaR §1.2)
        _, log = gated_reconstruct(
            traj,
            lambda t: thought_completion.complete_thoughts(t, llm),
            verifier=recovery_and_thought_ok,
        )
        if not log.kept:
            print(f"  [{tag}] {formula or '?':14s} exits=[{exits}]  -> DROP (thought gate)")
            continue

        # 3) physics + faithfulness
        phys_ok, phys = mlip_physics_ok(traj)
        phys = _native(phys)
        faith = faithfulness_ok(traj)

        traj.verification = Verification(
            reexecuted=faith,
            reexecute_reproduced=faith,
            judge_votes=[phys_ok, log.kept],   # judge1 physics, judge2 recovery+thoughts
            min_judges=2,
        )
        traj.terminal_step.reward = Reward(
            value=1.0, terminal=True, style=RewardStyle.RULE,
            ground_truth=json.dumps({"recovered": True, "mlip_max_force_tol": MLIP_FORCE_TOL}),
            verifiable={"faithful_to_archive": faith, "mlip": phys},
        )

        ok = traj.is_admissible()
        mf = phys.get("max_force")
        mf_s = f"{mf:.3f}" if isinstance(mf, (int, float)) else "n/a"
        print(f"  [{tag}] {formula or '?':14s} exits=[{exits}]  "
              f"faithful={faith} physics={phys_ok}(maxF={mf_s}) -> "
              f"{'ADMIT' if ok else 'DROP'}")
        if ok:
            traj.metadata["reconstruction"] = {
                "semantic_lift": "scf_tool_space", "thought_completion": llm.model_id,
                "admission": "faithfulness(archive) + chgnet_physics + recovery/thought_lint",
                "qe_reexecuted": False, "note": "actions/observations are REAL archived QE; not re-run here",
            }
            admitted.append(traj)
        if len(admitted) >= TARGET_ADMITTED:
            break

    # ----------------------------------------------------------------------- #
    print("\n" + "-" * 78)
    print(f"skipped {skipped_clean} clean (no-error) chains; scanned "
          f"{scanned} error→recovery episodes -> {len(admitted)} ADMITTED")
    if not admitted:
        print("no admissible trajectory produced.")
        return

    ir_path = OUT / "mc2d_admissible_ir.jsonl"
    with open(ir_path, "w") as f:
        for t in admitted:
            f.write(t.model_dump_json() + "\n")
    n_sft = to_sft_react.export_jsonl(admitted, str(OUT / "mc2d_admissible_sft.jsonl"),
                                      require_admissible=True)   # hard gate: refuses un-verified
    print(f"wrote {len(admitted)} IR -> {ir_path.name}")
    print(f"wrote {n_sft} admissible SFT ReAct records -> mc2d_admissible_sft.jsonl")
    print(f"LLM calls (thought completion): {llm.n_calls}")

    # show one admitted trajectory end-to-end
    t = admitted[0]
    print("\n" + "=" * 78)
    print(f"SAMPLE ADMITTED TRAJECTORY: {t.id}")
    print(f"goal: {t.goal}")
    for s in t.steps:
        flag = "  [FAILURE BRANCH]" if s.is_failure_branch else ""
        print(f"\n  step[{s.index}] {s.action.name}({json.dumps(s.action.params)})")
        print(f"    Thought: {s.thought}")
        print(f"    {s.observation.text}{flag}")


if __name__ == "__main__":
    main()
