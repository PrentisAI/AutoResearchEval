"""Generic, calculator-agnostic recompute tools — the executor recipes behind the
discovery recompute_handles (CLAUDE.md §0.9 hinge, §6 verification, §18.3).

These are NOT new physics engines; each is a thin COMPOSITION over a single generic
calculator (`calc_factory(atoms) -> ASE calculator`):

  adsorption_energy      -> co_adsorption_energy / site_preference / coverage_shift
  reaction_barrier       -> reaction_barrier        (ASE NEB / climbing image)
  vibrational_frequencies-> vibrational_frequency   (ASE finite-difference Hessian)

Calculator-agnostic on purpose: swap `emt_factory` (instant, plumbing/tests),
`mlip_factory` (cheap prefilter, §6), or `qe_factory` (real DFT, the reward). This
is the generalised, decomposed form of harness/co_pt_oracle.py — no per-system oracle.

The deterministic VERIFIER is the recipe itself: a converged, finite, decisive number
(e.g. E_ads sign/magnitude, a positive barrier, real frequencies). Method-matching
(§16.4 lesson: absolute energies aren't comparable across functionals) lives in the
MethodSpec the caller hands to qe_factory; prefer relative/comparative quantities
(site A vs B, ΔE) when the target value came from a different method.
"""
from __future__ import annotations

import itertools
import os
import tempfile
from pathlib import Path
from typing import Callable, Optional

# ASE site-name synonyms (LLMs say "atop"; ASE wants "ontop")
_SITE_ALIASES = {"atop": "ontop", "on-top": "ontop", "top": "ontop", "on top": "ontop",
                 "hollow": "fcc", "fcc-hollow": "fcc", "fcc hollow": "fcc",
                 "hcp-hollow": "hcp", "hcp hollow": "hcp", "3-fold": "fcc"}


def _norm_site(site):
    return _SITE_ALIASES.get((site or "ontop").strip().lower(), (site or "ontop").strip().lower())


# --------------------------------------------------------------------------- #
# Calculator factories — calc_factory(atoms) -> a fresh ASE calculator.
# --------------------------------------------------------------------------- #
def emt_factory(method_spec=None) -> Callable:
    """Instant EMT potential — for plumbing/tests only (physically meaningless for CO)."""
    from ase.calculators.emt import EMT
    return lambda atoms: EMT()


def mlip_factory(model: str = "chgnet") -> Callable:
    """Cheap universal-MLIP tier (§6 prefilter): CHGNet or MACE-MP. Loads the heavy model
    weights ONCE, but returns a FRESH calculator per call — ASE calculators cache results
    on the instance, so a shared calc is NOT thread-safe; recipe-level parallelism
    (RECOMPUTE_WORKERS>1) runs several relaxations concurrently, which would race a shared
    calc ("forces not present"). A fresh lightweight wrapper around the shared weights is
    both cheap and safe. Use for relaxation/screening before the QE reward."""
    # Device: honour MLIP_DEVICE, else auto-pick CUDA when available (route B runs the MLIP
    # relax surrogate on the free GPUs; CUDA_VISIBLE_DEVICES pins which cards).
    device = os.environ.get("MLIP_DEVICE")
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"
    if model == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator
        from chgnet.model.model import CHGNet
        weights = CHGNet.load()                              # heavy: load once
        return lambda atoms: CHGNetCalculator(model=weights, use_device=device)  # cheap per call
    elif model in ("mace", "mace_mp"):
        from mace.calculators import mace_mp
        # mace_mp builds a calculator; rebuild per call (model download is cached on disk).
        return lambda atoms: mace_mp(model="small", default_dtype="float64", device=device)
    else:
        raise ValueError(f"unknown MLIP {model!r} (use 'chgnet' or 'mace')")


def qe_factory(method_spec) -> Callable:
    """Real QE 7.5 (PBE-PAW) via ASE Espresso + MPI, auto-resolving pseudos per element.
    `method_spec` is a reconstruct.discovery_moves.MethodSpec (or compatible dict)."""
    from ase.calculators.espresso import Espresso, EspressoProfile

    from harness.co_pt_oracle import QE_MPIRUN, QE_PW
    from harness.co_pt_oracle import PSEUDO_DIR as _CURATED_PSEUDO
    from reconstruct.discovery_moves import hubbard_cards, to_qe_input_data

    ms = method_spec
    input_data = to_qe_input_data(ms)
    kpts = tuple(getattr(ms, "kpts", (4, 4, 1)))

    # Pseudo dir: QE_PSEUDO_DIR env > the merged SSSP+curated dir (102 elements, curated
    # qe_demo UPFs win conflicts so the +U-tested Ti/Ce stay) > the curated 8-element dir.
    # This widens QE-admissibility (Stage 2): Mo/S/W/Se/Fe/Co/... now resolve.
    _merged = Path(_CURATED_PSEUDO).parent / "pseudo_merged"
    PSEUDO_DIR = Path(os.environ.get("QE_PSEUDO_DIR")
                      or (_merged if _merged.is_dir() else _CURATED_PSEUDO))

    def _resolve(symbols):
        pseudos = {}
        for el in sorted(set(symbols)):
            # case-insensitive (SSSP files are e.g. 'pt_pbe_v1', 's_pbe_v1'); merged dir
            # exposes canonical '<El>.UPF' symlinks so a direct hit is preferred.
            hits = sorted(Path(PSEUDO_DIR).glob(f"{el}.*UPF")) or \
                   sorted(Path(PSEUDO_DIR).glob(f"{el}.*upf")) or \
                   [p for p in Path(PSEUDO_DIR).glob("*")
                    if p.name.lower().split(".")[0].split("_")[0] == el.lower()]
            if not hits:
                raise FileNotFoundError(f"no UPF for {el} in {PSEUDO_DIR}; download it first")
            pseudos[el] = hits[0].name
        return pseudos

    np_ranks = int(os.environ.get("QE_NP", "16"))
    npool = int(os.environ.get("QE_NPOOL", "4"))
    # Per-run scratch so concurrent rollouts don't clobber each other's espresso.pwi/.pwo
    # AND QE wavefunctions (ASE runs pw.x with cwd=directory, so QE's outdir lands inside
    # it too). Override with QE_SCRATCH to parallelise; default is the shared dir.
    scratch = Path(os.environ.get("QE_SCRATCH", str(Path(PSEUDO_DIR).parent / "scratch_recompute")))
    _call = itertools.count()                     # unique sub-scratch per make() call

    def make(atoms):
        # Each calc gets its OWN sub-scratch: the recipe's 4 QE calls (slab/gas/atop/fcc)
        # can run CONCURRENTLY (recipe-level parallelism) without clobbering each other's
        # espresso.pwi/.pwo + QE wavefunctions. Thread-safe via an atomic counter.
        cdir = scratch / f"c{next(_call)}"
        cdir.mkdir(parents=True, exist_ok=True)
        # QE 7.x HUBBARD card: only emit U for elements ACTUALLY in this structure —
        # a stray element (e.g. an LLM-proposed Ce on a TiO2 slab) makes pw.x fail with
        # "Hubbard atom does not match any type in ATOMIC_SPECIES".
        present = set(atoms.get_chemical_symbols())
        cards = hubbard_cards({el: u for el, u in (getattr(ms, "hubbard_u", {}) or {}).items()
                               if el in present})
        profile = EspressoProfile(command=f"{QE_MPIRUN} -np {np_ranks} {QE_PW} -npool {npool}",
                                  pseudo_dir=str(PSEUDO_DIR))
        return Espresso(profile=profile, directory=str(cdir),   # keep pwi/pwo out of cwd
                        pseudopotentials=_resolve(atoms.get_chemical_symbols()),
                        input_data=input_data, kpts=kpts,
                        additional_cards=cards or None)
    return make


# --------------------------------------------------------------------------- #
# Structure helpers
# --------------------------------------------------------------------------- #
# crystal structure of common catalytic metals (extend as needed)
_CRYSTAL = {
    "fcc": set("Pt Pd Ni Cu Au Ag Rh Ir Pb Al Ca Sr".split()),
    "bcc": set("Fe W Mo Cr V Nb Ta Ba".split()),
    "hcp": set("Co Ru Re Ti Zn Mg Zr Os Cd Y Sc".split()),
}
# (crystal, facet) -> ASE named builder (these set adsorbate_info sites)
_FACET_BUILDERS = {
    ("fcc", "111"): "fcc111", ("fcc", "100"): "fcc100", ("fcc", "110"): "fcc110",
    ("bcc", "110"): "bcc110", ("bcc", "100"): "bcc100", ("bcc", "111"): "bcc111",
    ("hcp", "0001"): "hcp0001",
    # note: stepped facets (fcc211 etc.) need special supercell sizing — not wired yet.
}


def _crystal_of(metal):
    for k, v in _CRYSTAL.items():
        if metal in v:
            return k
    return None


def metal_slab(metal="Pt", facet="111", supercell=(2, 2), layers=3, vacuum=7.0, a=None,
               fix_bottom=True, crystal=None):
    """Clean metal slab for any wired (crystal, facet). fcc 111/100/110/211, bcc
    110/100/111, hcp 0001. Unwired combos raise NotImplementedError (honest)."""
    import ase.build as B
    from ase.constraints import FixAtoms
    cr = crystal or _crystal_of(metal)
    if cr is None:
        raise NotImplementedError(f"unknown crystal structure for {metal!r}; pass crystal=fcc|bcc|hcp")
    key = (cr, str(facet))
    if key not in _FACET_BUILDERS:
        raise NotImplementedError(f"facet {facet!r} for {cr} not wired (have {sorted(f for c,f in _FACET_BUILDERS if c==cr)})")
    builder = getattr(B, _FACET_BUILDERS[key])
    kw = dict(size=(*supercell, layers), vacuum=vacuum)
    if a:
        kw["a"] = a
    slab = builder(metal, **kw)
    if fix_bottom:
        zs = sorted(set(round(at.z, 3) for at in slab))
        cut = (zs[0] + zs[1]) / 2 if len(zs) > 1 else zs[0] + 0.1
        slab.set_constraint(FixAtoms(mask=[at.z < cut for at in slab]))
    return slab


def with_adsorbate(slab, adsorbate="CO", site="ontop", height=1.8):
    """Place a molecular adsorbate. Site names are facet-dependent, so we map synonyms
    and fall back to an available site (never crash on an unknown site name)."""
    from ase import Atoms
    from ase.build import add_adsorbate
    from ase.symbols import string2symbols
    s = slab.copy()
    syms = string2symbols(adsorbate)            # atom count from parsed formula, not str len
    ads = Atoms(syms, positions=[[0, 0, i * 1.14] for i in range(len(syms))])
    want = _norm_site(site)
    avail = (s.info.get("adsorbate_info", {}) or {}).get("sites", {})
    if avail and want not in avail:
        for alt in (want, "ontop", "hollow", "fcc", "bridge", "hcp"):
            if alt in avail:
                want = alt
                break
        else:
            want = next(iter(avail))
    add_adsorbate(s, ads, height=height, position=want, mol_index=0)
    return s


def gas_molecule(formula="CO", box=12.0):
    from ase import Atoms
    from ase.symbols import string2symbols
    # Atom count from the PARSED formula, not the string length: a 2-char element like "Cl"
    # is one atom (len("Cl")==2 would wrongly request 2 positions). Linear chain along z.
    syms = string2symbols(formula)
    m = Atoms(syms, positions=[[0, 0, i * 1.14] for i in range(len(syms))],
              cell=[box] * 3, pbc=True)
    m.center()
    return m


# --------------------------------------------------------------------------- #
# 2D / TMD monolayers + point defects (the TMD corpus regime, §recipe_coverage_expansion
# Stage 3). These papers' real observable is a DEFECT FORMATION ENERGY (e.g. a sulfur
# vacancy or carbon substitution in MoS2), not a CO adsorption energy — a single-structure
# energy difference, deterministically buildable, large supercell (QE-GPU sweet spot).
# --------------------------------------------------------------------------- #
# TMD lattice constants (Å) for the 2H monolayer (a, layer thickness).
_TMD_PRESETS = {
    "MoS2": (3.16, 3.17), "WS2": (3.15, 3.14), "MoSe2": (3.29, 3.34),
    "WSe2": (3.28, 3.36), "MoTe2": (3.52, 3.60),
}


def tmd_monolayer(formula="MoS2", supercell=(4, 4), vacuum=8.0):
    """2H TMD monolayer (MX2) supercell. Returns an ase.Atoms with vacuum along z."""
    from ase.build import mx2
    a, thick = _TMD_PRESETS.get(formula, (3.16, 3.17))
    return mx2(formula=formula, kind="2H", a=a, thickness=thick,
               size=(*supercell, 1), vacuum=vacuum)


def make_point_defect(monolayer, kind="vacancy", element=None, dopant=None):
    """Introduce ONE point defect into a TMD monolayer:
       - vacancy: remove one atom of `element` (default the chalcogen, e.g. S in MoS2)
       - substitution: replace one `element` atom by `dopant` (e.g. C at an S site)
    Returns (defect_atoms, removed_symbol or (host,dopant)). Deterministic: picks the first
    matching site (periodic supercell → all equivalent)."""
    s = monolayer.copy()
    syms = s.get_chemical_symbols()
    # default target = the chalcogen (the more numerous species in MX2)
    if element is None:
        from collections import Counter
        element = Counter(syms).most_common(1)[0][0]
    idx = [i for i, e in enumerate(syms) if e == element]
    if not idx:
        raise ValueError(f"no {element} atom in {s.get_chemical_formula()} to defect")
    site = idx[0]
    if kind == "vacancy":
        host = syms[site]
        del s[site]
        return s, {"kind": "vacancy", "removed": host}
    elif kind == "substitution":
        if not dopant:
            raise ValueError("substitution needs dopant=")
        s[site].symbol = dopant
        return s, {"kind": "substitution", "host": syms[site], "dopant": dopant}
    raise ValueError(f"unknown defect kind {kind!r} (use vacancy|substitution)")


# Elemental/molecular chemical-potential references for defect formation energy (eV/atom).
# The reference reservoir an atom is removed-to / added-from: gas-phase dimer for O/N/H,
# a small molecular/elemental cluster otherwise. Honest scope: μ is reservoir-dependent;
# we use a fixed, documented gas/elemental reference so the number is reproducible.
def _chempot(element, calc_factory):
    """μ per atom from a simple reference (½ X2 for O/N/H/S as S2; bulk-ish dimer else)."""
    from ase import Atoms
    dimer = {"O": ("O2", 1.21), "N": ("N2", 1.10), "H": ("H2", 0.74),
             "S": ("S2", 1.89), "Se": ("Se2", 2.17), "C": ("C2", 1.31)}
    if element in dimer:
        f, d = dimer[element]
        m = Atoms(f, positions=[[0, 0, 0], [0, 0, d]], cell=[14] * 3, pbc=True); m.center()
        return _energy(m, calc_factory, relax=True, fmax=0.05, steps=40) / 2.0
    a = Atoms(element, positions=[[0, 0, 0]], cell=[14] * 3, pbc=True); a.center()
    return _energy(a, calc_factory, relax=False)


def defect_formation_energy(*, formula="MoS2", defect="vacancy", element=None, dopant=None,
                            supercell=(4, 4), vacuum=8.0, calc_factory,
                            relax=True, fmax=0.05, steps=80) -> dict:
    """E_form(defect) = E(defect) − E(pristine) + Σ μ(removed) − Σ μ(added).

    For a vacancy: +μ(removed). For a substitution host→dopant: +μ(host) − μ(dopant).
    Generic over calc_factory; QE-admissible when all elements have a UPF (Mo/S/W/Se now do,
    Stage 2a). Large supercell → route the QE tier to the GPU pw.x for speed."""
    pristine = tmd_monolayer(formula, supercell, vacuum)
    defected, info = make_point_defect(pristine, defect, element, dopant)
    e_pristine = _energy(pristine, calc_factory, relax=relax, fmax=fmax, steps=steps)
    e_defect = _energy(defected, calc_factory, relax=relax, fmax=fmax, steps=steps)
    if info["kind"] == "vacancy":
        mu = _chempot(info["removed"], calc_factory)
        e_form = e_defect - e_pristine + mu               # atom returned to reservoir
        chem = {"removed": info["removed"], "mu_removed": round(mu, 4)}
    else:  # substitution host→dopant
        mu_h = _chempot(info["host"], calc_factory)
        mu_d = _chempot(info["dopant"], calc_factory)
        e_form = e_defect - e_pristine + mu_h - mu_d
        chem = {"host": info["host"], "dopant": info["dopant"],
                "mu_host": round(mu_h, 4), "mu_dopant": round(mu_d, 4)}
    return {
        "handle": "defect_formation_energy",
        "formula": formula, "defect": info, "supercell": list(supercell),
        "E_form_eV": round(e_form, 4),
        "energies_eV": {"E_pristine": round(e_pristine, 4), "E_defect": round(e_defect, 4), **chem},
        # sane window: defect formation energies are typically 0–10 eV (negative = exothermic
        # incorporation, also physical); decisive = a finite, non-degenerate number.
        "defect_sane": -6.0 < e_form < 15.0,
        "decisive": True,
    }


# --------------------------------------------------------------------------- #
# Supported systems (single-atom catalysis) — the dominant corpus regime: a metal
# active site anchored on a support, not a clean metal slab. Same adsorption recipe
# (E_ads = E(sub+CO) − E(sub) − E(CO)); only the SUBSTRATE builder differs.
#
# QE-recomputable subset (the honest, admissible reward): the support + metal +
# adsorbate elements must all have a UPF and not need Hubbard-U. Pt/C/O do (clean
# carbon support, no +U) → metal single atom on (defective) graphene is the cleanest
# admissible SAC anchor. Oxide / N-doped / +U supports are wired to raise honestly
# (need their UPF / a +U MethodSpec) so we never silently fake them.
# --------------------------------------------------------------------------- #
_GRAPHENE_A = 2.46    # graphene lattice constant (Å)
# supports whose elements have a PSL UPF and need no Hubbard-U → qe-admissible
_QE_READY_SUPPORTS = {"graphene"}


def graphene_support(supercell=(3, 3), vacuum=7.5, a=_GRAPHENE_A):
    """Pristine graphene sheet, `supercell` repetition, vacuum along z."""
    from ase.build import graphene
    return graphene(formula="C2", a=a, size=(*supercell, 1), vacuum=vacuum)


def make_vacancy(slab, n=1):
    """Remove the n atoms nearest the in-plane centre → mono/di-vacancy. Returns
    (slab_with_vacancy, (site_xy, z_plane)) so the defect site can host a metal atom."""
    import numpy as np
    s = slab.copy()
    cen = s.cell.diagonal()[:2] / 2.0
    pos = s.get_positions()
    order = np.argsort(np.linalg.norm(pos[:, :2] - cen, axis=1))
    idx = sorted(order[:n].tolist())
    site_xy = pos[idx[0], :2].copy()
    z_plane = float(pos[:, 2].mean())
    del s[idx]
    return s, (site_xy, z_plane)


def single_atom_on_support(support="graphene", metal="Pt", defect="vacancy",
                           supercell=(3, 3), vacuum=7.5, height=1.2):
    """Build a single metal atom anchored on a (defective) support. Returns
    (atoms, metal_index). defect: 'vacancy'|'monovacancy' (mono), 'divacancy', or
    None/'pristine' (atop the sheet). Only carbon supports are QE-ready (see
    _QE_READY_SUPPORTS); others must supply their own UPF/+U."""
    import numpy as np
    from ase import Atom
    if support != "graphene":
        raise NotImplementedError(
            f"support {support!r} not wired (have: graphene). Oxide/N-doped/perovskite "
            "supports need their UPF (and often Hubbard-U) — out of the QE-ready set "
            f"{sorted(_QE_READY_SUPPORTS)}.")
    sub = graphene_support(supercell, vacuum)
    if defect in ("vacancy", "monovacancy"):
        sub, (xy, z0) = make_vacancy(sub, n=1)
    elif defect == "divacancy":
        sub, (xy, z0) = make_vacancy(sub, n=2)
    elif defect in (None, "none", "pristine"):
        xy = sub.cell.diagonal()[:2] / 2.0
        z0 = float(sub.get_positions()[:, 2].mean())
    else:
        raise NotImplementedError(f"defect {defect!r} (use vacancy|divacancy|pristine)")
    sub.append(Atom(metal, position=(float(xy[0]), float(xy[1]), z0 + height)))
    return sub, len(sub) - 1


def adsorbate_on_atom(substrate, anchor_index, adsorbate="CO", height=1.8):
    """Place a molecular adsorbate vertically above a specific substrate atom
    (first atom of `adsorbate` down, e.g. CO binds C-end down on the metal)."""
    from ase import Atoms
    from ase.symbols import string2symbols
    s = substrate.copy()
    p = s.get_positions()[anchor_index]
    syms = string2symbols(adsorbate)            # atom count from parsed formula, not str len
    ads = Atoms(syms, positions=[[p[0], p[1], p[2] + height + i * 1.14]
                                 for i in range(len(syms))])
    s += ads
    return s


# --------------------------------------------------------------------------- #
# Oxide supports (the corpus-dominant supported regime beyond carbon): a metal
# single atom on a reducible/irreducible oxide surface. Built deterministically
# from the bulk spacegroup via pymatgen SlabGenerator → a non-polar, (where
# possible) symmetric termination. These need Hubbard-U (reducible oxides) and
# their own UPF, so the QE tier is NOT admissible yet (no Ti/Ce UPF on this box);
# CHGNet (universal MLIP) is the plumbing/prefilter tier — EMT can't do Ti/Ce/Zn.
# Termination / slab thickness / U value are documented knobs, like cutoffs.
# --------------------------------------------------------------------------- #
_OXIDE_PRESETS = {
    # name: bulk spacegroup + lattice + Wyckoff seeds, default Miller face, Hubbard-U
    "ceo2": dict(spacegroup="Fm-3m", lattice=("cubic", (5.41,)), species=["Ce", "O"],
                 coords=[[0, 0, 0], [0.25, 0.25, 0.25]], miller=(1, 1, 1), hubbard_u={"Ce": 4.5}),
    "tio2": dict(spacegroup="P4_2/mnm", lattice=("tetragonal", (4.59, 2.96)), species=["Ti", "O"],
                 coords=[[0, 0, 0], [0.305, 0.305, 0]], miller=(1, 1, 0), hubbard_u={"Ti": 3.0}),
    "tio2_anatase": dict(spacegroup="I41/amd", lattice=("tetragonal", (3.78, 9.51)),
                         species=["Ti", "O"], coords=[[0, 0, 0], [0, 0, 0.208]],
                         miller=(1, 0, 1), hubbard_u={"Ti": 3.0}),
}


def oxide_support(oxide="ceo2", miller=None, min_slab=6.0, min_vacuum=12.0, supercell=(1, 1)):
    """Build an oxide surface slab from its bulk spacegroup (pymatgen). Returns
    (ase_atoms, hubbard_u). Deterministically picks a non-polar (and, where available,
    symmetric/stoichiometric) termination. Bottom half is fixed for relaxation."""
    from ase.constraints import FixAtoms
    from pymatgen.core import Lattice, Structure
    from pymatgen.core.surface import SlabGenerator
    from pymatgen.io.ase import AseAtomsAdaptor
    if oxide not in _OXIDE_PRESETS:
        raise NotImplementedError(f"oxide {oxide!r} not wired (have {sorted(_OXIDE_PRESETS)}); "
                                  "add its spacegroup/lattice/U to _OXIDE_PRESETS.")
    p = _OXIDE_PRESETS[oxide]
    lat = getattr(Lattice, p["lattice"][0])(*p["lattice"][1])
    bulk = Structure.from_spacegroup(p["spacegroup"], lat, p["species"], p["coords"])
    # accept only a 3-index Miller; an LLM 4-index hex/Miller-Bravais (e.g. [1,1,-2,0]) or
    # any malformed value falls back to the preset's default face (else pymatgen SlabGenerator
    # crashes with a shape mismatch).
    hkl = tuple(int(x) for x in miller) if (miller and len(miller) == 3) else p["miller"]
    slabs = SlabGenerator(bulk, hkl, min_slab, min_vacuum, center_slab=True,
                          lll_reduce=True).get_slabs()
    if not slabs:
        raise RuntimeError(f"no slab generated for {oxide} {hkl}")
    # deterministic termination: non-polar first, then symmetric/stoichiometric
    slab = sorted(slabs, key=lambda s: (not s.is_polar(), s.is_symmetric()), reverse=True)[0]
    atoms = AseAtomsAdaptor.get_atoms(slab)
    if tuple(supercell) != (1, 1):
        atoms = atoms.repeat((*supercell, 1))
    zs = atoms.get_positions()[:, 2]
    mid = (zs.min() + zs.max()) / 2.0
    atoms.set_constraint(FixAtoms(mask=[z < mid for z in zs]))
    return atoms, dict(p["hubbard_u"])


def metal_on_oxide(oxide_slab, metal="Pt", height=2.0):
    """Anchor a single metal atom above the top surface of an oxide slab (over the
    highest surface O, near the cell centre). Returns (atoms, metal_index)."""
    import numpy as np
    from ase import Atom
    s = oxide_slab.copy()
    pos = s.get_positions()
    ztop = pos[:, 2].max()
    near_top = [i for i, z in enumerate(pos[:, 2]) if ztop - z < 1.2]   # top-layer atoms
    cen = s.cell.diagonal()[:2] / 2.0
    anchor = min(near_top, key=lambda i: np.linalg.norm(pos[i, :2] - cen))
    xy = pos[anchor, :2]
    s.append(Atom(metal, position=(float(xy[0]), float(xy[1]), ztop + height)))
    return s, len(s) - 1


def oxide_supported_adsorption_energy(*, oxide="ceo2", metal="Pt", miller=None,
                                      supercell=(1, 1), min_slab=6.0, adsorbate="CO",
                                      height=1.9, box=12.0, calc_factory, relax=True,
                                      fmax=0.05, steps=80) -> dict:
    """E_ads of a molecule on a single metal atom on an oxide surface (oxide-supported
    SAC). Generic over calc_factory; QE is NOT admissible (no Ti/Ce UPF + needs +U) so
    use the CHGNet tier as a physical prefilter. Hubbard-U is returned for the QE path."""
    sub0, u = oxide_support(oxide, miller, min_slab, supercell=supercell)
    e_sub = _energy(metal_on_oxide(sub0, metal, height=2.0)[0],
                    calc_factory, relax=relax, fmax=fmax, steps=steps)
    e_gas = _energy(gas_molecule(adsorbate, box), calc_factory, relax=relax, fmax=fmax, steps=steps)
    sub1, m_idx = metal_on_oxide(oxide_support(oxide, miller, min_slab, supercell=supercell)[0],
                                 metal, height=2.0)
    e_sub_ads = _energy(adsorbate_on_atom(sub1, m_idx, adsorbate, height),
                        calc_factory, relax=relax, fmax=fmax, steps=steps)
    eads = e_sub_ads - e_sub - e_gas
    sane = bool(-5.0 < eads < 0.5)
    site = f"{metal}@{oxide}"
    return {
        "handle": "co_adsorption_energy",
        "system": f"{metal}@{oxide}{tuple(miller) if miller else ''} p{tuple(supercell)}",
        "E_ads_eV": {site: round(eads, 4)},
        "site_preference": site,
        "delta_eV": 0.0,
        "energies_eV": {"E_sub": round(e_sub, 4), "E_gas": round(e_gas, 4),
                        "E_sub_ads": round(e_sub_ads, 4)},
        "chemisorption_sane": sane,
        "decisive": sane,
        "anchor": f"{metal} single atom on {oxide} surface (atom index {m_idx})",
        "hubbard_u": u,
        "qe_admissible_support": False,   # needs Ti/Ce UPF + Hubbard-U; CHGNet-tier only for now
    }


# Optional MLIP relax-surrogate: when set, BFGS geometry optimisation is driven by THIS
# (cheap) calculator (e.g. CHGNet/MACE on GPU), and only the FINAL single-point energy is
# taken with the expensive `calc_factory` (QE). This is the "relax with MLIP, score with
# QE" split (route B): turns each recompute from 4 full QE relaxations into 4 QE
# single-points, while the geometry still comes from a real (MLIP) optimisation.
# Set via set_relax_surrogate(); None = legacy behaviour (relax with calc_factory itself).
_RELAX_SURROGATE = None


def set_relax_surrogate(calc_factory) -> None:
    """Install a cheap relax calculator factory (or None to clear). When set, _energy
    relaxes geometry with it and scores the final single-point with the main calc_factory."""
    global _RELAX_SURROGATE
    _RELAX_SURROGATE = calc_factory


def _energy(atoms, calc_factory, *, relax=True, fmax=0.05, steps=80):
    """Energy (eV). If ``relax``: optimise geometry, then return the final energy.

    With a relax surrogate installed (route B), BFGS runs on the cheap surrogate
    (MLIP/GPU) and the final energy is a single-point with ``calc_factory`` (QE) on the
    relaxed geometry. Without one, relax+score both use ``calc_factory`` (legacy)."""
    from ase.optimize import BFGS
    atoms = atoms.copy()
    if relax and _RELAX_SURROGATE is not None:
        atoms.calc = _RELAX_SURROGATE(atoms)        # cheap geometry (MLIP, e.g. on GPU)
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
        atoms.calc = calc_factory(atoms)            # expensive single-point (QE) on relaxed geom
        return float(atoms.get_potential_energy())
    atoms.calc = calc_factory(atoms)
    if relax:
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    return float(atoms.get_potential_energy())


# --------------------------------------------------------------------------- #
# Tool 1 — adsorption_energy  (co_adsorption_energy / site_preference / coverage_shift)
# --------------------------------------------------------------------------- #
def adsorption_energy(*, metal="Pt", facet="111", supercell=(2, 2), layers=3, vacuum=7.0, a=None,
                      adsorbate="CO", sites=("ontop", "fcc"), heights=None, box=12.0,
                      calc_factory, relax=True, fmax=0.05, steps=80) -> dict:
    """E_ads(site) = E(slab+ads) − E(slab) − E(ads_gas), for each site. Generic over
    calc_factory. Returns the per-site E_ads, the site preference, and the raw energies."""
    heights = heights or {}
    base = metal_slab(metal, facet, supercell, layers, vacuum, a)
    # The slab / gas / per-site energies are INDEPENDENT (E_ads subtracts them only after
    # all finish), so run them CONCURRENTLY (recipe-level parallelism). Each _energy spends
    # its time blocked on the pw.x subprocess (GIL released), so threads parallelise well;
    # qe_factory hands each call its own sub-scratch so they don't collide. Cap concurrency
    # via RECOMPUTE_WORKERS so (workers × QE_NP) stays within the physical-core budget.
    jobs = {"E_slab": metal_slab(metal, facet, supercell, layers, vacuum, a),
            "E_gas": gas_molecule(adsorbate, box)}
    for site in sites:
        jobs[f"E_slab_ads_{_norm_site(site)}"] = with_adsorbate(
            base, adsorbate, site, heights.get(site, 1.8))
    n_workers = max(1, int(os.environ.get("RECOMPUTE_WORKERS", "1")))
    if n_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(n_workers, len(jobs))) as ex:
            futs = {k: ex.submit(_energy, at, calc_factory, relax=relax, fmax=fmax, steps=steps)
                    for k, at in jobs.items()}
            energies = {k: f.result() for k, f in futs.items()}
    else:
        energies = {k: _energy(at, calc_factory, relax=relax, fmax=fmax, steps=steps)
                    for k, at in jobs.items()}
    e_slab, e_gas = energies["E_slab"], energies["E_gas"]
    eads = {}
    for site in sites:
        ns = _norm_site(site)
        eads[ns] = energies[f"E_slab_ads_{ns}"] - e_slab - e_gas
    pref = min(eads, key=eads.get)
    return {
        "handle": "co_adsorption_energy",
        "E_ads_eV": {k: round(v, 4) for k, v in eads.items()},
        "site_preference": pref,
        "delta_eV": round(max(eads.values()) - min(eads.values()), 4) if len(eads) > 1 else 0.0,
        "energies_eV": {k: round(v, 4) for k, v in energies.items()},
        "chemisorption_sane": all(-4.0 < v < 0.5 for v in eads.values()),
        "decisive": (len(eads) > 1 and abs(list(eads.values())[0] - list(eads.values())[1]) > 0.01),
    }


def supported_adsorption_energy(*, support="graphene", metal="Pt", defect="vacancy",
                                supercell=(3, 3), vacuum=7.5, adsorbate="CO", height=1.8,
                                box=12.0, calc_factory, relax=True, fmax=0.05, steps=80) -> dict:
    """E_ads of a molecule on a single-metal-atom active site anchored on a support
    (single-atom catalysis). E_ads = E(metal@support + ads) − E(metal@support) − E(ads_gas).
    Generic over calc_factory; the QE tier is admissible only for _QE_READY_SUPPORTS."""
    e_sub = _energy(single_atom_on_support(support, metal, defect, supercell, vacuum)[0],
                    calc_factory, relax=relax, fmax=fmax, steps=steps)
    e_gas = _energy(gas_molecule(adsorbate, box), calc_factory, relax=relax, fmax=fmax, steps=steps)
    sub, m_idx = single_atom_on_support(support, metal, defect, supercell, vacuum)  # fresh, unrelaxed
    e_sub_ads = _energy(adsorbate_on_atom(sub, m_idx, adsorbate, height),
                        calc_factory, relax=relax, fmax=fmax, steps=steps)
    eads = e_sub_ads - e_sub - e_gas
    sane = bool(-4.0 < eads < 0.5)
    site = f"{metal}@{defect or 'pristine'}"
    return {
        "handle": "co_adsorption_energy",
        "system": f"{metal}@{defect or 'pristine'}-{support} p{tuple(supercell)}",
        "E_ads_eV": {site: round(eads, 4)},
        "site_preference": site,            # SAC: one well-defined active site
        "delta_eV": 0.0,
        "energies_eV": {"E_sub": round(e_sub, 4), "E_gas": round(e_gas, 4),
                        "E_sub_ads": round(e_sub_ads, 4)},
        "chemisorption_sane": sane,
        "decisive": sane,                   # a finite, physical E_ads at the single site
        "anchor": f"{metal} single atom on {defect or 'pristine'} {support} (atom index {m_idx})",
        "qe_admissible_support": support in _QE_READY_SUPPORTS,
    }


# --------------------------------------------------------------------------- #
# Tool 2 — reaction_barrier  (NEB / climbing image)
# --------------------------------------------------------------------------- #
def reaction_barrier(initial, final, *, calc_factory, n_images=5, fmax=0.1, steps=60,
                     climb=True, relax_endpoints=True) -> dict:
    """CI-NEB barrier (eV) between two endpoints. Generic over calc_factory."""
    from ase.mep import NEB
    from ase.optimize import BFGS
    ini, fin = initial.copy(), final.copy()
    if relax_endpoints:
        for a in (ini, fin):
            a.calc = calc_factory(a)
            BFGS(a, logfile=None).run(fmax=fmax, steps=steps)
    images = [ini] + [ini.copy() for _ in range(n_images)] + [fin]
    for im in images[1:-1]:
        im.calc = calc_factory(im)
    neb = NEB(images, climb=climb, method="improvedtangent")
    neb.interpolate()
    BFGS(neb, logfile=None).run(fmax=fmax, steps=steps)
    path = [float(im.get_potential_energy()) for im in images]
    barrier = max(path) - path[0]
    return {
        "handle": "reaction_barrier",
        "barrier_eV": round(barrier, 4),
        "delta_E_eV": round(path[-1] - path[0], 4),
        "energy_path_eV": [round(e, 4) for e in path],
        "positive_barrier": bool(barrier > 0),
    }


# --------------------------------------------------------------------------- #
# Tool 3 — vibrational_frequencies  (finite-difference Hessian)
# --------------------------------------------------------------------------- #
def vibrational_frequencies(atoms, *, indices=None, calc_factory, relax=False,
                            fmax=0.03, steps=100) -> dict:
    """Harmonic frequencies (cm⁻¹) of the selected atoms via ASE finite differences."""
    from ase.optimize import BFGS
    from ase.vibrations import Vibrations
    a = atoms.copy()
    a.calc = calc_factory(a)
    if relax:
        BFGS(a, logfile=None).run(fmax=fmax, steps=steps)
    idx = indices if indices is not None else list(range(len(a)))
    with tempfile.TemporaryDirectory() as tmp:
        vib = Vibrations(a, indices=idx, name=str(Path(tmp) / "vib"))
        vib.run()
        freqs = vib.get_frequencies()             # cm⁻¹ (imaginary → complex)
        vib.clean()
    real = [round(float(f.real), 2) for f in freqs if abs(f.imag) < 1e-6]
    n_imag = sum(1 for f in freqs if abs(f.imag) > 1e-6)
    return {
        "handle": "vibrational_frequency",
        "frequencies_cm1": real,
        "max_cm1": round(max(real), 2) if real else None,
        "n_imaginary": n_imag,
    }


# --------------------------------------------------------------------------- #
# Tool 4 — run_md  (stability / energy drift; generic over calc_factory)
# --------------------------------------------------------------------------- #
def run_md(atoms, *, calc_factory, steps=50, timestep_fs=1.0, temperature_K=300.0,
           friction=0.01) -> dict:
    """Langevin MD; reports energy drift + blow-up flag (the §6 'MD didn't explode' check)."""
    import math

    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    a = atoms.copy()
    a.calc = calc_factory(a)
    MaxwellBoltzmannDistribution(a, temperature_K=temperature_K)
    e0 = float(a.get_potential_energy())
    dyn = Langevin(a, timestep_fs * units.fs, temperature_K=temperature_K, friction=friction)
    dyn.run(steps)
    ef = float(a.get_potential_energy())
    drift = abs(ef - e0)
    exploded = (not math.isfinite(ef)) or drift > 50.0
    return {
        "handle": "md_stability", "n_steps": steps,
        "E_initial_eV": round(e0, 4), "E_final_eV": round(ef, 4),
        "energy_drift_eV": round(drift, 4), "exploded": bool(exploded),
    }


# --------------------------------------------------------------------------- #
# compare_reference — the DETERMINISTIC verdict (turns a recomputed number into
# "matches the claim or not"). Numeric values must be method-matched (§16.4); for
# cross-method claims, compare relative/categorical quantities (site, ordering).
# --------------------------------------------------------------------------- #
def compare_reference(computed, reference, *, kind="energy", tol=0.2) -> dict:
    """kind='energy'|'numeric' → |Δ|≤tol; kind='site'|'categorical' → exact (case-insensitive)."""
    if kind in ("site", "categorical", "ordinal"):
        agrees = str(computed).strip().lower() == str(reference).strip().lower()
        return {"kind": kind, "computed": computed, "reference": reference, "agrees": bool(agrees)}
    delta = float(computed) - float(reference)
    return {"kind": "numeric", "computed": float(computed), "reference": float(reference),
            "delta": round(delta, 4), "abs_delta": round(abs(delta), 4), "tol": tol,
            "agrees": bool(abs(delta) <= tol)}


# --------------------------------------------------------------------------- #
# Orchestrator — draw_conclusion's recompute_handle → the right recipe → reward.
# Closes the loop: given a handle + system/method, run the matching tool. (barrier
# needs explicit endpoints; work_function/scaling_relation are honestly not wired.)
# --------------------------------------------------------------------------- #
def recompute_for_handle(handle: str, *, calc_factory, **kw) -> dict:
    if handle in ("co_adsorption_energy", "site_preference", "coverage_shift"):
        # routing by substrate kwarg: `oxide` → oxide-supported SAC; `support` →
        # carbon/graphene SAC; neither → clean-metal slab. (incompatible kwargs dropped.)
        if kw.get("oxide"):
            for k in ("support", "defect", "facet", "layers", "a", "sites", "heights", "vacuum"):
                kw.pop(k, None)
            return oxide_supported_adsorption_energy(calc_factory=calc_factory, **kw)
        if kw.get("support"):
            for k in ("oxide", "miller", "min_slab", "facet", "layers", "a", "sites", "heights"):
                kw.pop(k, None)
            return supported_adsorption_energy(calc_factory=calc_factory, **kw)
        for k in ("support", "defect", "oxide", "miller", "min_slab"):
            kw.pop(k, None)
        return adsorption_energy(calc_factory=calc_factory, **kw)
    if handle == "defect_formation_energy":
        # TMD/2D point-defect formation energy (Stage 3). Drop adsorption-only kwargs.
        for k in ("oxide", "miller", "min_slab", "support", "facet", "layers", "a",
                  "sites", "heights", "adsorbate", "site", "height", "box", "metal"):
            kw.pop(k, None)
        return defect_formation_energy(calc_factory=calc_factory, **kw)
    if handle == "vibrational_frequency":
        return vibrational_frequencies(calc_factory=calc_factory, **kw)
    if handle == "reaction_barrier":
        if "initial" not in kw or "final" not in kw:
            raise ValueError("reaction_barrier needs initial= and final= endpoint Atoms")
        return reaction_barrier(kw.pop("initial"), kw.pop("final"), calc_factory=calc_factory, **kw)
    if handle == "work_function":
        raise NotImplementedError("work_function needs QE pp.x electrostatic-potential extraction "
                                  "(vacuum level − E_Fermi) — not wired; QE-only, can't EMT-test")
    if handle == "scaling_relation":
        raise NotImplementedError("scaling_relation needs a descriptor (e.g. d-band centre via "
                                  "projwfc) over a family of systems — not wired (1 corpus paper)")
    raise NotImplementedError(f"no recompute recipe for handle {handle!r}")
