"""Move-driven, live-oracle discovery rollout (CLAUDE.md §0.9 hinge, §18.3).

THIN DRIVER over the refactored architecture (2026-06-22):
  * ``harness/discovery_env.py``       — the domain-AGNOSTIC discovery skeleton (move
    sequence, anti-hack reward gate, IR wiring; imports no chemistry);
  * ``harness/domains/catalysis_qe.py`` — the computational-catalysis PLUGIN (builds
    systems, chooses/clamps QE params, runs the live recompute, maps the result onto the
    three domain-invariant verdict flags).

This file just (1) frames the task as a DiscoveryPattern, (2) constructs the catalysis
oracle at the chosen tier/budget, (3) runs the episode through ``DiscoveryEnv``, and
(4) writes the IR + SFT products. The chemistry that used to live here is now in the
plugin; to add a second domain you write a new oracle, not a new driver.

Unlike the 92 corpus trajectories (``examples/discovery_trajectory_build.py``) whose
``run_calculation`` observations are *narrated from the paper* (pending-soft-verify),
here ``run_calculation`` EXECUTES the recompute and the terminal reward IS that
deterministic recompute (the "live recompute oracle in loop" the discovery→RL data needs).

Calc tiers (swap the oracle tier without touching the skeleton):
  --calc emt    instant EMT — PLUMBING/CI only (physically meaningless for CO); never admissible
  --calc mlip   CHGNet universal MLIP — cheap prefilter (§6); not the honest reward → not admissible
  --calc qe     real QE 7.5 PBE-PAW — the honest, admissible reward (minutes/calc, MPI)

Run (scicoder env for emt/mlip plumbing; qe env reaches pw.x for the real reward):
  PY=/home/ubuntu/miniconda3/envs/scicoder/bin/python
  $PY examples/discovery_rollout.py --calc emt        # validate the loop (instant)
  QE_NP=32 QE_NPOOL=4 /home/ubuntu/miniconda3/envs/qe/bin/python \
      examples/discovery_rollout.py --calc qe          # real, admissible reward
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/data/xmyu/scicoder")

from export.to_sft_react import trajectory_to_messages
from harness.discovery_env import DiscoveryEnv
from harness.domains.catalysis_qe import CatalysisQEOracle, pick_handle as _pick_handle, task_from_pattern
from reconstruct.discovery_moves import sanitize_method_spec  # re-exported for callers/tests
from reconstruct.discovery_pattern import DiscoveryPattern
from reconstruct.llm_openrouter import OpenRouterClient

REPO = Path("/data/xmyu/scicoder")
PATTERNS_DIR = REPO / "examples/output/discovery_patterns/patterns"
OUTDIR = REPO / "examples/output/discovery_rollouts"


def canonical_co_pt() -> DiscoveryPattern:
    """Default self-contained task: CO site preference on Pt(111) — the premise the whole
    CO/Pt corpus stands on, posed as an open question (no answer leaked into the goal)."""
    return DiscoveryPattern(
        paper_id="co_pt111_rollout",
        title="CO site preference on Pt(111): atop vs fcc-hollow (move-driven live recompute)",
        premise_consensus=(
            "CO chemisorbs on transition-metal surfaces via the Blyholder donor–acceptor model; "
            "on Pt(111) at low coverage experiment (IR/LEED) finds CO bound terminally ATOP a single "
            "Pt atom."),
        tension=(
            "Semilocal DFT-GGA (PBE) is notorious for the 'CO/Pt(111) puzzle' — it tends to "
            "over-stabilise the high-coordination fcc-hollow site against the experimental atop "
            "preference. For a given PBE-PAW setup, which site is favoured and by how much?"),
        motivation="Establish a recomputable CO/Pt(111) site-preference anchor as a discovery reward.",
        method="Plane-wave DFT, PBE; Pt(111) p(2×2) slab; E_ads at atop and fcc-hollow.",
        experiment="Relax clean slab, gas CO, and CO at each site; E_ads = E(slab+CO) − E(slab) − E(CO).",
        conclusion="",  # the agent must produce it; not leaked
        key_claims=[
            {"claim": "PBE site preference for CO on Pt(111).", "kind": "qualitative",
             "value": "", "recompute_handle": "site_preference"},
            {"claim": "CO adsorption energy on Pt(111).", "kind": "quantitative",
             "value": "", "recompute_handle": "co_adsorption_energy"},
        ],
        novelty_move="method-correction",
        artifact_uri="harness/recompute_tools.py (live recompute)",
    )


def canonical_sac() -> DiscoveryPattern:
    """Single-atom-catalysis task: CO binding at a metal atom anchored on a graphene
    vacancy — the cleanest QE-recomputable supported anchor (Pt/C/O, no Hubbard-U),
    representing the corpus-dominant supported regime (vs the clean-metal CO/Pt case)."""
    return DiscoveryPattern(
        paper_id="sac_metal_graphene_rollout",
        title="CO adsorption at a single metal atom on defective graphene (move-driven live recompute)",
        premise_consensus=(
            "Single-atom catalysts maximise precious-metal efficiency; a metal atom trapped at a "
            "graphene vacancy is a canonical anchored active site, and CO is the standard probe of "
            "its electronic state via its adsorption energy and stretch frequency."),
        tension=(
            "The CO binding strength at such an anchored single atom depends sensitively on the "
            "metal and the anchoring defect, and is not obvious a priori — what adsorption energy "
            "does a given metal-on-defective-graphene site give?"),
        motivation="Establish a recomputable CO/single-atom-on-support adsorption-energy anchor.",
        method="Plane-wave DFT, PBE; metal atom at a graphene vacancy; CO on the metal site.",
        experiment="Relax metal@vacancy-graphene, gas CO, and CO on the metal; E_ads = E(sub+CO) − E(sub) − E(CO).",
        conclusion="",  # agent must produce it
        key_claims=[{"claim": "CO adsorption energy at the single-atom site.", "kind": "quantitative",
                     "value": "", "recompute_handle": "co_adsorption_energy"}],
        novelty_move="new-regime",
        artifact_uri="harness/recompute_tools.py (supported live recompute)",
    )


def canonical_oxide_sac() -> DiscoveryPattern:
    """Oxide-supported single-atom catalysis: CO binding at a metal atom on a reducible
    oxide surface (e.g. Pt/CeO₂) — the corpus-dominant supported regime. Recompute is
    CHGNet-tier (no Ti/Ce UPF here); the Hubbard-U is carried for the future QE path."""
    return DiscoveryPattern(
        paper_id="oxide_sac_rollout",
        title="CO adsorption at a single metal atom on a reducible oxide (move-driven recompute)",
        premise_consensus=(
            "Metal single atoms on reducible oxides (CeO₂, TiO₂) are a leading single-atom-catalyst "
            "platform; the oxide's redox activity and the metal–support charge transfer set the CO "
            "binding strength, which standard DFT struggles with (needs Hubbard-U)."),
        tension=(
            "What CO adsorption energy does a metal atom on a given oxide facet give, and is the "
            "metal–support interaction strong enough to anchor it — this is sensitive to the metal, "
            "facet, and U, and not obvious a priori?"),
        motivation="Establish a recomputable CO/metal-on-oxide adsorption-energy anchor.",
        method="Plane-wave DFT+U; metal atom on an oxide surface; CO on the metal site.",
        experiment="Relax metal@oxide, gas CO, and CO on the metal; E_ads = E(sub+CO) − E(sub) − E(CO).",
        conclusion="",
        key_claims=[{"claim": "CO adsorption energy at the oxide-supported single-atom site.",
                     "kind": "quantitative", "value": "", "recompute_handle": "co_adsorption_energy"}],
        novelty_move="new-regime",
        artifact_uri="harness/recompute_tools.py (oxide-supported live recompute)",
    )


_TASKS = {"co_pt": canonical_co_pt, "sac": canonical_sac, "oxide_sac": canonical_oxide_sac}


def load_pattern(paper_id: str | None, task: str = "co_pt",
                 patterns_dir: Path | None = None) -> DiscoveryPattern:
    if paper_id:
        f = (patterns_dir or PATTERNS_DIR) / f"{paper_id}.json"
        if not f.exists():
            sys.exit(f"pattern not found: {f}")
        return DiscoveryPattern.from_dict(json.loads(f.read_text()))
    return _TASKS[task]()


def calc_factory_for(tier: str, method_spec):
    """Return (calc_factory, engine_name, admissible_tier). Thin shim over the catalysis
    plugin's tier machinery (kept for callers/tests that probe tiers directly)."""
    from harness import recompute_tools as RT
    from harness.domains.catalysis_qe import _ENGINE
    if tier == "emt":
        return RT.emt_factory(), _ENGINE["emt"], False
    if tier == "mlip":
        return RT.mlip_factory("chgnet"), _ENGINE["mlip"], False
    if tier == "qe":
        return RT.qe_factory(method_spec), _ENGINE["qe"], True
    sys.exit(f"unknown --calc {tier!r} (use emt|mlip|qe)")


def pick_handle(pattern: DiscoveryPattern) -> str:
    """The clean-metal adsorption anchor to recompute (the recipe's scope)."""
    try:
        return _pick_handle(pattern.recompute_handles())
    except ValueError as e:
        sys.exit(f"{e} (see harness/recompute_tools.recompute_for_handle)")


# build_trajectory is re-exported so the integration test can lay records onto the IR.
from reconstruct.discovery_trajectory import build_trajectory  # noqa: E402


def _fmt_result(handle: str, res: dict) -> str:
    if res.get("handle") == "defect_formation_energy":
        return (f"Recompute ({handle}): E_form({res.get('formula')} "
                f"{res.get('defect',{}).get('kind')})={res['E_form_eV']} eV; "
                f"sane={res['defect_sane']}, decisive={res['decisive']}.")
    eads = ", ".join(f"E_ads({k})={v} eV" for k, v in res["E_ads_eV"].items())
    return (f"Recompute ({handle}): {eads}; site_preference={res['site_preference']} "
            f"(Δ={res['delta_eV']} eV); chemisorption_sane={res['chemisorption_sane']}, "
            f"decisive={res['decisive']}.")


def run_rollout(llm, pattern: DiscoveryPattern, tier: str, *, relax: bool, fmax: float,
                steps_cap: int, compute_budget: dict | None = None,
                overrides: dict | None = None) -> tuple[list[dict], dict, dict]:
    """Drive the discovery episode and execute run_calculation live, via the refactored
    DiscoveryEnv + catalysis plugin. Returns (move_records, recompute_result, used) — the
    same tuple shape the integration test consumes — by constructing the catalysis oracle
    at `tier` (with the real-DFT budget/overrides) and running one episode through the
    domain-agnostic skeleton."""
    handle = pick_handle(pattern)
    task = task_from_pattern(pattern)
    oracle = CatalysisQEOracle(tier=tier, handle=handle, compute_budget=compute_budget,
                               overrides=overrides)
    env = DiscoveryEnv(task, oracle)
    bundle = env.rollout(llm, relax=relax, fmax=fmax, steps=steps_cap)
    res = bundle["execution"].result
    used = {"engine": oracle.engine_name, "handle": handle,
            "system": oracle._struct_spec.__dict__ if oracle._struct_spec else {},
            "method": oracle._method_spec.__dict__ if oracle._method_spec else {},
            "_env": env}   # stash the env so main() can reuse to_trajectory()
    if res.get("handle") == "defect_formation_energy":
        print(f"  [run_calculation/{tier}] {res.get('formula')} "
              f"{res.get('defect',{}).get('kind')}  E_form={res['E_form_eV']} eV  "
              f"({bundle['execution'].seconds:.0f}s)", flush=True)
    else:
        print(f"  [run_calculation/{tier}] {res['site_preference']} preferred  "
              f"E_ads={res['E_ads_eV']}  ({bundle['execution'].seconds:.0f}s)", flush=True)
    return bundle["records"], res, used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calc", choices=["emt", "mlip", "qe"], default="emt")
    ap.add_argument("--task", choices=["co_pt", "sac", "oxide_sac"], default="co_pt",
                    help="built-in task: co_pt (clean metal) | sac (carbon SAC) | oxide_sac (oxide SAC)")
    ap.add_argument("--paper-id", default=None, help="corpus pattern id (overrides --task)")
    ap.add_argument("--patterns-dir", default=None,
                    help="dir holding <paper-id>.json (default: examples/output/discovery_patterns/patterns)")
    ap.add_argument("--metal", default=None, help="pin the active/slab metal (QE-ready combo)")
    ap.add_argument("--support", default=None, help="pin the support (graphene|ceo2|tio2|tio2_anatase)")
    ap.add_argument("--respect-system", action="store_true",
                    help="honor the LLM's chosen system kind (oxide/SAC/clean) instead of forcing "
                         "clean-metal when --metal is set; clamps repair it to a buildable system "
                         "of that kind (recipe_coverage_expansion.md Stage 1)")
    ap.add_argument("--no-relax", action="store_true", help="single-point (skip BFGS) — faster plumbing")
    ap.add_argument("--relax-mlip", choices=["chgnet", "mace"], default=None,
                    help="route B: relax geometry with this MLIP (GPU), score final single-point "
                         "with --calc engine. Turns 4 QE relaxations into 4 QE single-points.")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=80, help="max BFGS steps per relaxation")
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    pdir = Path(args.patterns_dir) if args.patterns_dir else None
    pattern = load_pattern(args.paper_id, args.task, patterns_dir=pdir)
    llm = OpenRouterClient(max_tokens=1200, temperature=args.temperature)
    # Per-run QE scratch (unless the caller already pinned one) so multiple rollouts can
    # run CONCURRENTLY without clobbering each other's espresso.pwi/.pwo + QE wavefunctions.
    if args.calc == "qe" and "QE_SCRATCH" not in os.environ:
        combo = "_".join(filter(None, (pattern.paper_id, args.metal, args.support)))
        os.environ["QE_SCRATCH"] = str(REPO / "research/agent_survey/qe_demo" / f"scratch_{combo}")
    # route B: install an MLIP relax surrogate (GPU) — geometry from MLIP, energy from --calc.
    if args.relax_mlip:
        from harness.recompute_tools import mlip_factory, set_relax_surrogate
        set_relax_surrogate(mlip_factory(args.relax_mlip))
        print(f"[relax-surrogate] geometry via {args.relax_mlip} (MLIP), energy via {args.calc}")
    print(f"[rollout] paper={pattern.paper_id}  calc={args.calc}  teacher={llm._model}")

    # real-DFT is expensive → cap the model to the smallest meaningful size
    budget = None
    if args.calc == "qe":
        budget = {"hint": ("Compute budget: this will be evaluated with REAL plane-wave DFT, so "
                           "choose the SMALLEST physically meaningful model — at most a 2x2 "
                           "supercell and 3 layers (clean slab) or a 3x3 cell (supported) — and "
                           "MODEST cutoffs (ecutwfc ~40-45 Ry, enough for PAW)."),
                  "max_supercell": 2, "max_layers": 3, "max_ecutwfc": 45.0}
    overrides = {k: v for k, v in (("metal", args.metal), ("support", args.support)) if v}
    if args.respect_system:
        overrides["respect_system"] = True
    rec, res, used = run_rollout(llm, pattern, args.calc, relax=not args.no_relax, fmax=args.fmax,
                                 steps_cap=args.steps, compute_budget=budget, overrides=overrides or None)

    # ---- lay the moves onto the IR + wire the verdict as reward (skeleton owns this) ----
    traj = used["_env"].to_trajectory(pattern)
    reproduces_exp = traj.metadata.get("reproduces_experiment")

    # encode the metal/support combo so multiple QE-ready runs don't clobber each other
    combo = "_".join(filter(None, (args.metal, args.support)))
    tag = f"_{combo}" if combo else ""
    out_ir = OUTDIR / f"{pattern.paper_id}{tag}_{args.calc}_ir.json"
    out_sft = OUTDIR / f"{pattern.paper_id}{tag}_{args.calc}_sft.jsonl"
    out_ir.write_text(traj.model_dump_json(indent=2))
    out_sft.write_text(json.dumps(trajectory_to_messages(traj, require_admissible=False),
                                  ensure_ascii=False))

    _obs = (f"E_form={res['E_form_eV']} eV" if res.get("handle") == "defect_formation_energy"
            else f"site={res.get('site_preference')}  reproduces_experiment={reproduces_exp}")
    print(f"[reward] value={traj.terminal_step.reward.value}  admissible={traj.is_admissible()}  {_obs}")
    print(f"[moves ] {' → '.join(s['move'] for s in rec)}")
    print(f"[llm   ] {llm.n_calls} calls\n[out] {out_ir}\n[out] {out_sft}")


if __name__ == "__main__":
    main()
