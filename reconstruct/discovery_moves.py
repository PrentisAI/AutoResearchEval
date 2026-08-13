"""Each abstract discovery move as a PROMPTED FUNCTION (CLAUDE.md §18, §14).

The discussion that produced this (2026-06-15): the engine moves (`run_dft`/`run_md`/…)
are deterministic tools; the *abstract* moves (`select_system`, `choose_method`, and the
reasoning moves) are not compute engines — they MAP free-form research intent into a
typed output via an LLM. So we implement each as a function that shares ONE machinery
(context → LLM → parse to typed output) but carries a move-SPECIFIC prompt, because each
move's goal differs.

Two output kinds:
  * engine-feeding moves → a TYPED SPEC that materialises into the real engine's input:
      select_system → StructureSpec → build_structure() → ase.Atoms
      choose_method → MethodSpec    → to_qe_input_data() → run_dft params
  * reasoning moves → text (the agent's reasoning; the Thought channel).

This is the data-CONSTRUCTION scaffold (one focused prompt per move = higher-quality,
more controllable than one monolithic decomposition prompt). The final trained agent
just generates these as thought; here we elicit them as functions to build clean data.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from reconstruct.discovery_trajectory import _extract_json

# --------------------------------------------------------------------------- #
# Typed outputs for the two engine-feeding moves (the only ones that must be
# machine-consumable; everything downstream is deterministic).
# --------------------------------------------------------------------------- #
@dataclass
class StructureSpec:
    kind: str = "slab"                 # slab | molecule | bulk | supported
    element: str = ""                  # slab metal, e.g. "Pt"
    facet: str = "111"                 # slab facet
    supercell: tuple = (2, 2)          # in-plane repetition
    layers: int = 3
    vacuum: float = 7.0
    adsorbate: Optional[str] = None    # e.g. "CO"
    site: Optional[str] = None         # ontop | fcc | hcp | bridge
    height: Optional[float] = None     # Å, C-anchor above the site
    formula: Optional[str] = None      # for molecule/bulk
    box: float = 12.0                  # molecule cell (Å)
    a: Optional[float] = None          # lattice constant (Å); None → ASE default
    # supported / single-atom-catalysis fields (kind="supported")
    support: Optional[str] = None      # graphene (QE-ready) | ceo2 | tio2 | tio2_anatase (need UPF/+U)
    active_metal: Optional[str] = None # the anchored single atom, e.g. "Pt" (falls back to element)
    defect: Optional[str] = None       # graphene anchoring: vacancy | divacancy | pristine
    miller: Optional[tuple] = None     # oxide surface face, e.g. [1,1,1]; None → preset default
    # 2D / TMD monolayer defect fields (kind="monolayer_2d"): formula = "MoS2"/"WS2"/…,
    # defect = vacancy|substitution, element = which species to defect, dopant = substituent.
    dopant: Optional[str] = None       # substitution dopant, e.g. "C" at an S site


@dataclass
class MethodSpec:
    code: str = "qe"
    functional: str = "PBE"
    ecutwfc: float = 40.0
    ecutrho: float = 320.0
    kpts: tuple = (4, 4, 1)
    smearing: str = "mv"
    degauss: float = 0.02
    hubbard_u: dict = field(default_factory=dict)   # {"Ce": 4.5} etc.
    calc: str = "relax"                              # scf | relax


@dataclass
class MoveResult:
    """Uniform return of every move: the reasoning (→ Thought channel) plus, for the
    engine-feeding moves, the typed spec that materialises into the engine's input.
    For reasoning moves `spec` is None and `thought` carries the move's text."""
    move: str
    thought: str
    spec: object = None                              # StructureSpec | MethodSpec | None


# --------------------------------------------------------------------------- #
# Per-move prompt table — the goal differs per move, so the prompt differs.
# out: text | structure_spec | method_spec
# --------------------------------------------------------------------------- #
_PERSONA = ("You are a computational-catalysis researcher working a discovery step by step. "
            "Be faithful and concrete; never invent numbers you were not given. ")

MOVE_SPECS: dict[str, dict] = {
    "survey_consensus": dict(out="text", system=_PERSONA,
        goal="State the established prior understanding this question builds on — what does the "
             "field already believe? 1–2 sentences. Do NOT state a conclusion."),
    "identify_tension": dict(out="text", system=_PERSONA,
        goal="Name the gap / contradiction / anomaly in that consensus — the discovery seed. "
             "One sentence, phrased as an open问题, NOT a foregone answer."),
    "formulate_question": dict(out="text", system=_PERSONA,
        goal="Turn the tension into ONE concrete, computationally answerable question."),
    "propose_hypothesis": dict(out="text", system=_PERSONA,
        goal="State one testable candidate answer or mechanism to probe (or 'none' if the work is "
             "purely exploratory)."),
    "select_system": dict(out="structure_spec", system=_PERSONA + "Output strict JSON only.",
        goal="Choose ONE concrete, buildable atomistic system that can answer the question. "
             "Output JSON {\"thought\": \"<1–2 sentences: why this system/size/site lets you answer "
             "the question>\", \"spec\": {kind(slab|molecule|bulk|supported|monolayer_2d), element, "
             "facet, supercell([a,b]), layers, vacuum, adsorbate(or null), site(ontop|fcc|hcp|bridge|null), "
             "height(Å or null), formula(or null), support(graphene|ceo2|tio2|tio2_anatase|null), "
             "active_metal(or null), defect(vacancy|divacancy|pristine|substitution|null), dopant(or null), "
             "miller(oxide face [h,k,l] or null)}}. Use kind='supported' for single-atom catalysis (a "
             "metal atom anchored on a support): set support + active_metal; for a carbon support also set "
             "defect, for an oxide support optionally set miller. Use kind='slab' for a clean "
             "extended metal surface. Use kind='monolayer_2d' for a 2D TMD whose question is a DEFECT "
             "FORMATION ENERGY (sulfur vacancy, dopant substitution in MoS2/WS2/MoSe2/WSe2): set "
             "formula (e.g. 'MoS2'), defect ('vacancy' or 'substitution'), element (the species to "
             "defect, e.g. 'S'), and for substitution also dopant (e.g. 'C'). Pick the smallest model "
             "that is still physically meaningful. "
             "IMPORTANT: set `adsorbate` to the molecule THIS PAPER's reaction actually probes — "
             "O2 (or OOH/OH/O) for ORR/oxygen reduction, H for HER/hydrogen evolution, N2 (or NNH) "
             "for nitrogen reduction, CO2 (or COOH) for CO2 reduction, CO only if the paper is about "
             "CO adsorption/oxidation. Do NOT default to CO when the paper studies a different "
             "adsorbate — the computed adsorption energy must match the paper's tension."),
    "choose_method": dict(out="method_spec", system=_PERSONA + "Output strict JSON only.",
        goal="Choose a plane-wave DFT method + parameters appropriate to the system and question. "
             "Output JSON {\"thought\": \"<1–2 sentences: why this functional/cutoffs/k-points/smearing, "
             "and which literature method you are matching>\", \"spec\": {functional, ecutwfc, ecutrho, "
             "kpts([kx,ky,kz]), smearing(mv|gauss|none), degauss, hubbard_u(dict elem→U, {} if none), "
             "calc(scf|relax)}}. Match the literature method for this system where you can."),
    "compare_reference": dict(out="text", system=_PERSONA,
        goal="Given the computed result and the reference (experiment / prior value / competing model) "
             "in the context, state agreement or discrepancy and the numeric delta."),
    "interpret_result": dict(out="text", system=_PERSONA,
        goal="Say what the observation means for the question — no victory claim, no overreach."),
    "draw_conclusion": dict(out="text", system=_PERSONA,
        goal="State the terminal claim that resolves the tension, grounded in the computed numbers "
             "in the context. Be specific and honest (state the level of theory)."),
}


def _ctx_view(context: dict) -> str:
    """Compact, ordered view of the trajectory-so-far for the prompt."""
    order = ["goal", "consensus", "tension", "question", "hypothesis", "system", "method",
             "observations", "comparison", "interpretation"]
    lines = []
    for k in order:
        v = context.get(k)
        if v:
            lines.append(f"- {k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}")
    return "\n".join(lines) or "(no prior context)"


def _strip_move_label(text: str, move_id: str) -> str:
    """Drop a leading echo of the move id the teacher sometimes prepends, e.g.
    'compare_reference: ...', '**draw_conclusion** ...', 'select_system - ...'."""
    label = move_id.replace("_", r"[_\s]")
    return re.sub(rf"^\s*\**\s*{label}\s*\**\s*[:\-–]?\s*", "", text or "", count=1, flags=re.I).strip()


def run_move(client, move_id: str, context: dict, *, retries: int = 2,
             persona: str | None = None) -> MoveResult:
    """Execute one abstract move via its purpose-built prompt. Always returns a
    MoveResult(thought, spec): reasoning moves carry only `thought`; the engine-feeding
    moves carry `thought` + a typed StructureSpec/MethodSpec.

    `persona` overrides the system-prompt persona prefix (used by harness/discovery_env.py
    to prime the domain-agnostic reasoning moves for a specific domain); when None the
    move's built-in computational-catalysis persona is kept (back-compatible default).

    For spec moves we retry on a failed/empty parse and NEVER silently return all-default
    values — a failure is surfaced as spec=None with a flagged thought (so the caller can
    drop or re-ask instead of running an unintended default calculation)."""
    if move_id not in MOVE_SPECS:
        raise KeyError(f"unknown move {move_id!r}")
    mspec = MOVE_SPECS[move_id]
    system = mspec["system"] if persona is None else mspec["system"].replace(_PERSONA, persona)
    prompt = (f"Discovery step: **{move_id}**.\n{mspec['goal']}\n\n"
              f"Context so far:\n{_ctx_view(context)}\n\n"
              + ("Output ONLY the JSON object." if mspec["out"].endswith("_spec")
                 else "Output ONLY the step text (no preamble)."))

    if mspec["out"] not in ("structure_spec", "method_spec"):
        raw = client.complete(prompt, system=system)
        return MoveResult(move=move_id, thought=_strip_move_label(raw, move_id))

    cls = StructureSpec if mspec["out"] == "structure_spec" else MethodSpec
    for _ in range(retries + 1):
        d = _extract_json(client.complete(prompt, system=system)) or {}
        sd = d.get("spec", d)
        sd = sd if isinstance(sd, dict) else {}
        known = {k: v for k, v in sd.items() if k in cls.__annotations__}
        if not known:                              # parse failed / junk → retry, don't default-silently
            continue
        for tup in ("supercell", "kpts", "miller"):
            if known.get(tup):
                known[tup] = tuple(known[tup])
        return MoveResult(move=move_id, thought=_strip_move_label(str(d.get("thought", "")), move_id),
                          spec=cls(**known))
    return MoveResult(move=move_id, thought="[parse-failed: no valid spec after retries]", spec=None)


# Ergonomic named wrappers — "each move is a function".
def survey_consensus(client, ctx):   return run_move(client, "survey_consensus", ctx)
def identify_tension(client, ctx):   return run_move(client, "identify_tension", ctx)
def formulate_question(client, ctx): return run_move(client, "formulate_question", ctx)
def propose_hypothesis(client, ctx): return run_move(client, "propose_hypothesis", ctx)
def select_system(client, ctx):      return run_move(client, "select_system", ctx)
def choose_method(client, ctx):      return run_move(client, "choose_method", ctx)
def compare_reference(client, ctx):  return run_move(client, "compare_reference", ctx)
def interpret_result(client, ctx):   return run_move(client, "interpret_result", ctx)
def draw_conclusion(client, ctx):    return run_move(client, "draw_conclusion", ctx)


# --------------------------------------------------------------------------- #
# Deterministic materialisation: typed spec → real engine input (the bridge).
# --------------------------------------------------------------------------- #
def build_structure(spec: StructureSpec):
    """StructureSpec → ase.Atoms (no physics — pure construction)."""
    from ase import Atoms
    if spec.kind == "molecule":
        a = Atoms(spec.formula or "CO",
                  positions=[[0, 0, i * 1.14] for i in range(len(spec.formula or "CO"))],
                  cell=[spec.box] * 3, pbc=True)
        a.center()
        return a
    if spec.kind == "bulk":
        from ase.build import bulk
        return bulk(spec.element or spec.formula or "Pt", a=spec.a)
    if spec.kind == "supported":
        # single metal atom on a support — single-atom catalysis (carbon or oxide)
        from harness.recompute_tools import (_OXIDE_PRESETS, adsorbate_on_atom,
                                             metal_on_oxide, oxide_support, single_atom_on_support)
        metal = spec.active_metal or spec.element or "Pt"
        if (spec.support or "") in _OXIDE_PRESETS:
            slab, _u = oxide_support(spec.support, miller=spec.miller,
                                     supercell=tuple(spec.supercell))
            sub, m_idx = metal_on_oxide(slab, metal, height=2.0)
        else:
            sub, m_idx = single_atom_on_support(
                support=spec.support or "graphene", metal=metal,
                defect=spec.defect or "vacancy", supercell=tuple(spec.supercell), vacuum=spec.vacuum)
        if spec.adsorbate:
            sub = adsorbate_on_atom(sub, m_idx, spec.adsorbate, spec.height or 1.8)
        return sub
    # slab (default) — delegate to the canonical multi-facet builder (fcc/bcc/hcp)
    from harness.recompute_tools import metal_slab, with_adsorbate
    slab = metal_slab(spec.element or "Pt", facet=spec.facet, supercell=tuple(spec.supercell),
                      layers=spec.layers, vacuum=spec.vacuum, a=spec.a)
    if spec.adsorbate:
        slab = with_adsorbate(slab, spec.adsorbate, spec.site or "ontop", spec.height or 1.8)
    return slab


# ASE fcc111 site names; map the synonyms an LLM tends to produce.
_SITE_ALIASES = {"atop": "ontop", "on-top": "ontop", "top": "ontop", "on top": "ontop",
                 "hollow": "fcc", "fcc-hollow": "fcc", "fcc hollow": "fcc",
                 "hcp-hollow": "hcp", "hcp hollow": "hcp", "3-fold": "fcc"}


def _norm_site(site: Optional[str]) -> str:
    s = (site or "ontop").strip().lower()
    return _SITE_ALIASES.get(s, s)


def sanitize_method_spec(spec: MethodSpec):
    """Clamp LLM-proposed values to physical/affordable ranges (the deterministic layer
    that catches e.g. ecutwfc=400 Ry). Returns (sanitized_copy, notes). PSL PAW PBE:
    ecutwfc ~40–60 Ry, ecutrho 8–12× ecutwfc, degauss ~0.01–0.02 Ry."""
    import copy
    s = copy.deepcopy(spec)
    notes = []
    if s.ecutwfc < 30:
        notes.append(f"ecutwfc {s.ecutwfc}->30"); s.ecutwfc = 30.0
    elif s.ecutwfc > 90:
        notes.append(f"ecutwfc {s.ecutwfc}->90 (was unphysically high)"); s.ecutwfc = 90.0
    lo, hi = 8 * s.ecutwfc, 12 * s.ecutwfc
    if s.ecutrho < lo:
        notes.append(f"ecutrho {s.ecutrho}->{lo}"); s.ecutrho = lo
    elif s.ecutrho > hi:
        notes.append(f"ecutrho {s.ecutrho}->{hi}"); s.ecutrho = hi
    if s.smearing and s.smearing != "none":
        if s.degauss > 0.05:
            notes.append(f"degauss {s.degauss}->0.02"); s.degauss = 0.02
        elif s.degauss < 0.002:
            notes.append(f"degauss {s.degauss}->0.01"); s.degauss = 0.01
    k = tuple(max(1, min(16, int(x))) for x in (s.kpts or (4, 4, 1)))
    if k != tuple(s.kpts):
        notes.append(f"kpts {tuple(s.kpts)}->{k}"); s.kpts = k
    return s, notes


def to_qe_input_data(spec: MethodSpec) -> dict:
    """MethodSpec → ASE Espresso input_data dict (namelists only). Hubbard-U is NOT a
    namelist in QE 7.x — it goes in a HUBBARD card (see hubbard_cards), passed to ASE as
    additional_cards by qe_factory."""
    system = {"ecutwfc": spec.ecutwfc, "ecutrho": spec.ecutrho}
    if spec.smearing and spec.smearing != "none":
        system.update({"occupations": "smearing", "smearing": spec.smearing, "degauss": spec.degauss})
    # DFT+U prints Hubbard occupation tables that make ASE's parser miscount spin unless the
    # band eigenvalues are also printed → use verbosity='high' when +U is active (else 'low').
    verbosity = "high" if spec.hubbard_u else "low"
    return {
        "control": {"calculation": spec.calc, "tprnfor": True, "disk_io": "none", "verbosity": verbosity},
        "system": system,
        "electrons": {"conv_thr": 1e-6, "mixing_beta": 0.3},
        "ions": {"ion_dynamics": "bfgs"} if spec.calc == "relax" else {},
    }


# Hubbard manifold per element (which shell carries U) — QE 7.x HUBBARD card needs it.
_HUBBARD_MANIFOLD = {
    # 3d transition metals
    "Sc": "3d", "Ti": "3d", "V": "3d", "Cr": "3d", "Mn": "3d", "Fe": "3d", "Co": "3d",
    "Ni": "3d", "Cu": "3d", "Zn": "3d",
    # 4d
    "Nb": "4d", "Mo": "4d", "Ru": "4d", "Rh": "4d", "Pd": "4d",
    # 5d
    "Ta": "5d", "W": "5d",
    # 4f lanthanides
    "La": "4f", "Ce": "4f", "Pr": "4f", "Nd": "4f", "Sm": "4f", "Eu": "4f", "Gd": "4f",
}


def hubbard_cards(hubbard_u: dict, projector: str = "ortho-atomic") -> list[str]:
    """Build the QE 7.x HUBBARD card lines for a {element: U_eV} mapping (the card itself
    activates DFT+U in QE 7.x — no lda_plus_u flag). Species labels must match
    ATOMIC_SPECIES (ASE writes the bare element symbol). Unknown manifolds raise (honest)."""
    if not hubbard_u:
        return []
    lines = [f"HUBBARD ({projector})"]
    for el, u in hubbard_u.items():
        man = _HUBBARD_MANIFOLD.get(el)
        if man is None:
            raise ValueError(f"no Hubbard manifold known for {el!r}; add it to _HUBBARD_MANIFOLD")
        lines.append(f"U {el}-{man} {float(u)}")
    return lines


def spec_to_dict(x) -> dict:
    return asdict(x)
