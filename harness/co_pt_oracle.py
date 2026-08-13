"""Real QE 7.5 recompute oracle for CO adsorption on Pt(111) — the deterministic
reward behind a discovery trajectory (CLAUDE.md §18.3, §0.9 hinge).

The CO/Pt(111) site-preference puzzle is the cleanest hard anchor in the CO/Pt
corpus: a clean metal surface (no Hubbard-U), the textbook fact that CO binds
atop at low coverage, and a known DFT-GGA failure mode (PBE over-stabilises the
fcc-hollow site). So `site_preference` + `co_adsorption_energy` are recomputable
with our QE, and the recomputed numbers become a real, deterministic reward.

This oracle:
  1. relaxes a clean Pt(111) slab            → E_slab
  2. relaxes a gas-phase CO molecule         → E_CO
  3. relaxes CO on the slab at ATOP and FCC  → E_site
  4. E_ads(site) = E_site − E_slab − E_CO ;  site_preference = argmin E_ads

PSLibrary 1.0.0 PBE-PAW pseudos (Pt/C/O), QE 7.5 via ASE Espresso + MPI k-point
pools. First-pass settings are modest (documented) — a proof that the recompute
runs and yields a physically sane E_ads with the right sign, not a converged
publication number. Convergence is a knob (--ecutwfc/--kpts/--layers).

ENV: run with the `qe` conda env python (has ASE + reaches pw.x). pseudos in
research/agent_survey/qe_demo/pseudo/.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# MPI run: 1 OMP thread per rank (k-point pools give the parallelism).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

_REPO = Path(__file__).resolve().parents[1]
# pw.x / mpirun are env-overridable so a large-cell (TMD/oxide) rollout can point at the
# GPU build (QE_PW=/home/ubuntu/qe_gpu_build/.../build-gpu/bin/pw.x + the NVHPC mpirun +
# its LD_LIBRARY_PATH) without code changes (recipe_coverage_expansion Stage 2b). Default
# = the conda CPU pw.x (the honest, validated path for small cells).
QE_PW = Path(os.environ.get("QE_PW", "/home/ubuntu/miniconda3/envs/qe/bin/pw.x"))
QE_MPIRUN = Path(os.environ.get("QE_MPIRUN", "/home/ubuntu/miniconda3/envs/qe/bin/mpirun"))
PSEUDO_DIR = _REPO / "research/agent_survey/qe_demo/pseudo"
SCRATCH = _REPO / "research/agent_survey/qe_demo/scratch_co_pt"
PSEUDOPOTENTIALS = {
    "Pt": "Pt.pbe-n-kjpaw_psl.1.0.0.UPF",
    "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
}
A_PT_PBE = 3.97  # PBE fcc Pt lattice constant (Å); exp 3.92

# CO–Pt anchoring heights (Å, C above the site) — sensible starting geometries.
SITE_HEIGHT = {"ontop": 1.85, "fcc": 1.35}


def _espresso(directory, kpts, ecutwfc, ecutrho, np_ranks, npool, calc="relax"):
    from ase.calculators.espresso import Espresso, EspressoProfile
    if not QE_PW.exists():
        raise FileNotFoundError(f"pw.x not found at {QE_PW}")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    Path(directory).mkdir(parents=True, exist_ok=True)
    cmd = f"{QE_MPIRUN} -np {np_ranks} {QE_PW} -npool {npool}"
    profile = EspressoProfile(command=cmd, pseudo_dir=str(PSEUDO_DIR))
    input_data = {
        "control": {"calculation": calc, "tprnfor": True, "forc_conv_thr": 1e-3,
                    "nstep": 80, "disk_io": "none", "verbosity": "low"},
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho,
                   "occupations": "smearing", "smearing": "mv", "degauss": 0.02},
        "electrons": {"conv_thr": 1e-6, "mixing_beta": 0.3, "electron_maxstep": 200},
        "ions": {"ion_dynamics": "bfgs"},
    }
    return Espresso(profile=profile, directory=str(directory),
                    pseudopotentials=PSEUDOPOTENTIALS, input_data=input_data, kpts=kpts)


def _build_slab(layers, vacuum):
    from ase.build import fcc111
    from ase.constraints import FixAtoms
    slab = fcc111("Pt", size=(2, 2, layers), a=A_PT_PBE, vacuum=vacuum)
    # fix the bottom layer (z below the midpoint of the lowest two layers)
    zs = sorted(set(round(a.z, 3) for a in slab))
    fix_below = (zs[0] + zs[1]) / 2 if len(zs) > 1 else zs[0] + 0.1
    slab.set_constraint(FixAtoms(mask=[a.z < fix_below for a in slab]))
    return slab


def _slab_with_co(layers, vacuum, site):
    from ase import Atoms
    from ase.build import add_adsorbate
    slab = _build_slab(layers, vacuum)
    co = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.14]])  # C at anchor, O above
    add_adsorbate(slab, co, height=SITE_HEIGHT[site], position=site, mol_index=0)
    return slab


def _co_molecule(box):
    from ase import Atoms
    co = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.14]], cell=[box, box, box], pbc=True)
    co.center()
    return co


def _relax_energy(atoms, label, *, kpts, ecutwfc, ecutrho, np_ranks, npool):
    t0 = time.time()
    atoms.calc = _espresso(SCRATCH / label, kpts, ecutwfc, ecutrho, np_ranks, npool)
    e = atoms.get_potential_energy()          # eV; final (relaxed) energy
    print(f"  [{label}] E = {e:.4f} eV  ({time.time()-t0:.0f}s)", flush=True)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--vacuum", type=float, default=7.0)
    ap.add_argument("--ecutwfc", type=float, default=40.0)
    ap.add_argument("--ecutrho", type=float, default=320.0)
    ap.add_argument("--kx", type=int, default=4)
    ap.add_argument("--np", dest="np_ranks", type=int, default=32)
    ap.add_argument("--npool", type=int, default=4)
    ap.add_argument("--box", type=float, default=12.0, help="CO molecule box (Å)")
    ap.add_argument("--out", default=str(_REPO / "examples/output/co_pt111_recompute.json"))
    args = ap.parse_args()

    slab_kpts = (args.kx, args.kx, 1)
    common = dict(ecutwfc=args.ecutwfc, ecutrho=args.ecutrho, np_ranks=args.np_ranks)
    print(f"[co_pt_oracle] Pt(111) p(2x2) {args.layers}L  k={slab_kpts}  "
          f"ecut={args.ecutwfc}/{args.ecutrho} Ry  np={args.np_ranks} npool={args.npool}", flush=True)

    e_slab = _relax_energy(_build_slab(args.layers, args.vacuum), "slab",
                           kpts=slab_kpts, npool=args.npool, **common)
    # gas-phase CO: gamma only → npool 1; cap ranks (2 atoms can't feed 32 ranks)
    e_co = _relax_energy(_co_molecule(args.box), "co_gas", kpts=(1, 1, 1), npool=1,
                         ecutwfc=args.ecutwfc, ecutrho=args.ecutrho,
                         np_ranks=min(8, args.np_ranks))

    sites = {"ontop": "atop", "fcc": "fcc_hollow"}
    eads = {}
    energies = {"E_slab": e_slab, "E_CO": e_co}
    for site, name in sites.items():
        e_site = _relax_energy(_slab_with_co(args.layers, args.vacuum, site), f"co_{site}",
                               kpts=slab_kpts, npool=args.npool, **common)
        energies[f"E_slab_CO_{name}"] = e_site
        eads[name] = e_site - e_slab - e_co
        print(f"  => E_ads({name}) = {eads[name]:.3f} eV", flush=True)

    pref = min(eads, key=eads.get)            # most negative = strongest binding
    result = {
        "system": "CO on Pt(111) p(2x2), 0.25 ML",
        "method": f"QE 7.5 PBE-PAW (PSL 1.0.0), {args.layers}L slab, k={slab_kpts}, "
                  f"ecutwfc={args.ecutwfc} ecutrho={args.ecutrho} Ry, MV smearing 0.02 Ry",
        "E_ads_eV": {k: round(v, 4) for k, v in eads.items()},
        "site_preference": pref,
        "delta_atop_minus_hollow_eV": round(eads["atop"] - eads["fcc_hollow"], 4),
        "energies_eV": {k: round(v, 4) for k, v in energies.items()},
        "note": ("Proof-of-concept recompute: modest first-pass settings (convergence is a knob). "
                 "PBE is known to over-stabilise the fcc-hollow site (the 'CO/Pt(111) puzzle'); "
                 "experiment binds atop at low coverage. We report the recomputed numbers honestly."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\n[done] site_preference={pref}  E_ads={result['E_ads_eV']}\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
