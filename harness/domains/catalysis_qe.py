"""Computational-catalysis domain plugin for the discovery skeleton (CLAUDE.md §18).

This is the FIRST domain plugged into ``harness/discovery_env.py``. It owns ALL the
chemistry the skeleton must not know: building atomistic systems (slab / carbon-SAC /
oxide-SAC), choosing/clamping QE parameters, running the live recompute oracle, and
mapping the raw E_ads result onto the three domain-invariant verdict flags. Everything
domain-specific that used to be inlined in ``examples/discovery_rollout.py`` lives here.

Tiers (the oracle is constructed knowing which it is):
  emt   — ASE EMT, instant, PLUMBING/CI only (meaningless for CO) → never admissible
  mlip  — CHGNet universal MLIP, cheap prefilter (§6)             → not the honest reward
  qe    — real QE 7.5 PBE-PAW                                      → the admissible reward

The skeleton's anti-hack reward gate (``sane ∧ decisive ∧ valid``) is fed by :meth:`assess`:
  * sane     ← chemisorption energy in a physical window
  * decisive ← the experiment actually adjudicated the tension (compared both sites for a
               clean-metal site-preference task; a finite single-site E_ads for SAC)
  * valid    ← tier == "qe" (only real ab-initio mints admissible data)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from harness.discovery_env import Design, DiscoveryTask, Execution, Verdict
from reconstruct.discovery_moves import run_move, sanitize_method_spec, spec_to_dict

_PERSONA = ("You are a computational-catalysis researcher working a discovery step by step. "
            "Be faithful and concrete; never invent numbers you were not given. ")

# clean-metal adsorption anchors the recompute recipe can serve as a live reward.
_ADSORPTION_HANDLES = ("site_preference", "co_adsorption_energy", "coverage_shift")

_ENGINE = {
    "emt": "ASE EMT (plumbing only — physically meaningless for CO)",
    "mlip": "CHGNet universal MLIP (cheap prefilter, §6)",
    "qe": "Quantum ESPRESSO 7.5 (real ab-initio, PBE-PAW)",
}


@dataclass
class _RunDefaults:
    relax: bool = True
    fmax: float = 0.05
    steps: int = 80


class CatalysisQEOracle:
    """A ``DomainOracle`` for CO-adsorption discovery tasks, backed by ``recompute_tools``.

    Construct with a tier and (optionally) a real-DFT compute budget + metal/support
    overrides (§14 scaffolding for pinning a QE-ready combo on a data-generation run)."""

    name = "catalysis_qe"
    persona = _PERSONA

    def __init__(self, tier: str = "emt", *, handle: str = "site_preference",
                 compute_budget: Optional[dict] = None, overrides: Optional[dict] = None,
                 run_defaults: Optional[_RunDefaults] = None):
        if tier not in _ENGINE:
            raise ValueError(f"unknown tier {tier!r} (use emt|mlip|qe)")
        self.tier = tier
        self.handle = handle
        self.engine_name = _ENGINE[tier]
        self.admissible = (tier == "qe")
        self.compute_budget = compute_budget
        self.overrides = overrides or {}
        self.run_defaults = run_defaults or _RunDefaults()
        self._method_spec = None       # set during design (needed to build the qe factory)
        self._struct_spec = None

    # ---- helpers -------------------------------------------------------------
    def _calc_factory(self):
        from harness import recompute_tools as RT
        if self.tier == "emt":
            return RT.emt_factory()
        if self.tier == "mlip":
            return RT.mlip_factory("chgnet")
        return RT.qe_factory(self._method_spec)

    @staticmethod
    def _is_oxide(support) -> bool:
        from harness.recompute_tools import _OXIDE_PRESETS
        return (support or "") in _OXIDE_PRESETS

    # ---- DESIGN: question → typed, runnable experiment -----------------------
    def design(self, llm, ctx: dict) -> Design:
        """select_system → choose_method, with the domain's budget clamps + overrides.
        Returns the two move records + the opaque experiment (struct/method specs)."""
        # real-DFT budget: front-load a smallest-model prior before system design (§14)
        if self.compute_budget and self.compute_budget.get("hint"):
            ctx["question"] = (ctx.get("question", "") + "\n" + self.compute_budget["hint"]).strip()

        rs = run_move(llm, "select_system", ctx, persona=self.persona)
        if rs.spec is None:
            raise RuntimeError("select_system produced no usable StructureSpec")
        sspec = rs.spec
        cap_note = self._apply_overrides(sspec) + self._clamp_structure(sspec)
        ctx["system"] = spec_to_dict(sspec)
        sys_label, sys_obs = self._describe_system(sspec)
        sel_rec = {"move": "select_system", "thought": rs.thought,
                   "args": {"system": sys_label, "conditions": f"adsorbate={sspec.adsorbate}"},
                   "observation": sys_obs + cap_note}

        rm = run_move(llm, "choose_method", ctx, persona=self.persona)
        if rm.spec is None:
            raise RuntimeError("choose_method produced no usable MethodSpec")
        mspec, notes = self._sanitize_method(rm.spec, sspec)
        ctx["method"] = spec_to_dict(mspec)
        note_s = f" (clamped: {'; '.join(notes)})" if notes else ""
        meth_rec = {"move": "choose_method", "thought": rm.thought,
                    "args": {"method": f"{mspec.functional} ecut={mspec.ecutwfc}/{mspec.ecutrho} Ry",
                             "level": f"k={mspec.kpts}, {mspec.smearing} {mspec.degauss}"},
                    "observation": f"Method: {mspec.functional} PAW, ecutwfc={mspec.ecutwfc} Ry, "
                                   f"k={mspec.kpts}{note_s}."}
        self._struct_spec, self._method_spec = sspec, mspec
        return Design(records=[sel_rec, meth_rec], experiment=(sspec, mspec),
                      metadata={"system": spec_to_dict(sspec), "method": spec_to_dict(mspec)})

    def design_from_action(self, action: dict, ctx: dict) -> Design:
        """RL-policy variant: build the experiment from a policy-emitted spec dict instead
        of a fresh LLM call. Reuses the same clamps so the env's reward is consistent."""
        from reconstruct.discovery_moves import MethodSpec, StructureSpec
        sd = action.get("system", {}) or {}
        md = action.get("method", {}) or {}
        sspec = StructureSpec(**{k: v for k, v in sd.items() if k in StructureSpec.__annotations__})
        mspec_in = MethodSpec(**{k: v for k, v in md.items() if k in MethodSpec.__annotations__})
        cap_note = self._apply_overrides(sspec) + self._clamp_structure(sspec)
        sys_label, sys_obs = self._describe_system(sspec)
        mspec, notes = self._sanitize_method(mspec_in, sspec)
        self._struct_spec, self._method_spec = sspec, mspec
        recs = [
            {"move": "select_system", "thought": action.get("thought", ""),
             "args": {"system": sys_label}, "observation": sys_obs + cap_note},
            {"move": "choose_method", "thought": action.get("method_thought", ""),
             "args": {"method": f"{mspec.functional} ecut={mspec.ecutwfc} Ry"},
             "observation": f"Method: {mspec.functional} PAW, ecutwfc={mspec.ecutwfc} Ry, k={mspec.kpts}."
                            + (f" (clamped: {'; '.join(notes)})" if notes else "")},
        ]
        return Design(records=recs, experiment=(sspec, mspec))

    # ---- EXECUTE: run the live oracle ----------------------------------------
    def execute(self, design: Design, *, relax=None, fmax=None, steps=None) -> Execution:
        from harness.recompute_tools import _OXIDE_PRESETS, recompute_for_handle
        sspec, mspec = design.experiment
        relax = self.run_defaults.relax if relax is None else relax
        fmax = self.run_defaults.fmax if fmax is None else fmax
        steps = self.run_defaults.steps if steps is None else steps
        ads = self._sane_adsorbate(sspec.adsorbate)
        cf = self._calc_factory()
        t0 = time.time()
        if sspec.kind == "monolayer_2d":
            # TMD/2D point-defect formation energy (Stage 3) — the paper's real observable,
            # not a CO adsorption. handle is forced to defect_formation_energy here.
            formula = sspec.formula or "MoS2"
            dkind = (sspec.defect or "vacancy")
            dkind = dkind if dkind in ("vacancy", "substitution") else "vacancy"
            res = recompute_for_handle(
                "defect_formation_energy", calc_factory=cf, formula=formula, defect=dkind,
                element=(sspec.element or None), dopant=sspec.dopant,
                supercell=tuple(sspec.supercell or (4, 4)), vacuum=float(sspec.vacuum or 8.0),
                relax=relax, fmax=fmax, steps=steps)
            d = res.get("defect", {})
            thought = (f"Execute the probe: build the {formula} monolayer and its {dkind} "
                       f"defect, relax both, and form E_form = E_defect − E_pristine + Δμ.")
            target = f"{formula} {dkind}"
            dt = time.time() - t0
            record = {"move": "run_calculation", "thought": thought,
                      "args": {"tool": "run_dft", "computes": "defect_formation_energy",
                               "target": target},
                      "observation": (f"E_form({formula} {dkind}) = {res['E_form_eV']} eV; "
                                      f"sane={res['defect_sane']}.")}
            return Execution(result=res, record=record, seconds=dt)
        if sspec.kind == "supported":
            metal = sspec.active_metal or sspec.element or "Pt"
            support = sspec.support or "graphene"
            if self._is_oxide(support):
                res = recompute_for_handle(
                    self.handle, calc_factory=cf, oxide=support, metal=metal,
                    miller=tuple(sspec.miller) if sspec.miller else None,
                    supercell=tuple(sspec.supercell or (1, 1)), adsorbate=ads,
                    height=float(sspec.height or 1.9), relax=relax, fmax=fmax, steps=steps)
                thought = (f"Execute the probe: relax the {metal} single atom on the {support} "
                           f"surface, gas-phase {ads}, and {ads} on the {metal} site, then form E_ads.")
                target = f"{metal}@{support}"
            else:
                defect = sspec.defect or "vacancy"
                res = recompute_for_handle(
                    self.handle, calc_factory=cf, support=support, metal=metal, defect=defect,
                    supercell=tuple(sspec.supercell or (3, 3)), vacuum=float(sspec.vacuum or 7.5),
                    adsorbate=ads, height=float(sspec.height or 1.8), relax=relax, fmax=fmax, steps=steps)
                thought = (f"Execute the probe: relax the {metal} single atom on {defect} {support}, "
                           f"gas-phase {ads}, and {ads} on the {metal} site, then form E_ads.")
                target = f"{metal}@{defect}-{support}"
        else:
            metal = sspec.element or "Pt"
            res = recompute_for_handle(
                self.handle, calc_factory=cf, metal=metal, facet=str(sspec.facet or "111"),
                supercell=tuple(sspec.supercell or (2, 2)), layers=int(sspec.layers or 3),
                vacuum=float(sspec.vacuum or 7.0), adsorbate=ads,
                sites=("ontop", "fcc"), relax=relax, fmax=fmax, steps=steps)
            thought = (f"Execute the probe on the chosen {metal}({sspec.facet}) slab: relax the clean "
                       f"slab, gas-phase {ads}, and {ads} at atop and fcc-hollow, then form E_ads.")
            target = "atop & fcc-hollow"
        dt = time.time() - t0
        record = {"move": "run_calculation", "thought": thought,
                  "args": {"tool": "run_dft", "computes": self.handle, "target": target},
                  "observation": _fmt_result(self.handle, res)}
        return Execution(result=res, record=record, seconds=dt)

    def reference_note(self, execution: Execution) -> str:
        # The CO/Pt(111) atop experimental fact only applies when CO on clean Pt(111) is
        # what was actually computed; for any other adsorbate/system, give a neutral,
        # non-answer-leaking reference so the RESOLVE moves compare against the right thing.
        s = self._struct_spec
        ads = (getattr(s, "adsorbate", None) or "CO") if s else "CO"
        is_pt111_co = bool(s and s.kind != "supported" and (s.element or "Pt") == "Pt"
                           and str(getattr(s, "facet", "111")) == "111" and ads == "CO")
        if is_pt111_co:
            return "experiment binds CO atop on Pt(111) at low coverage."
        return (f"reference: the {ads} adsorption energy/site on this system, as reported in "
                "the paper (compare magnitude and preferred site).")

    # ---- ASSESS: raw result → domain-invariant verdict -----------------------
    def assess(self, execution: Execution) -> Verdict:
        res = execution.result
        # TMD/2D defect-formation-energy result has a different shape (no E_ads/site) —
        # score it on its own sane/decisive flags (Stage 3).
        if res.get("handle") == "defect_formation_energy":
            sane, decisive, valid = bool(res["defect_sane"]), bool(res["decisive"]), self.admissible
            verifiable = {"engine": self.engine_name, "valid_reward": valid,
                          "E_form_eV": res["E_form_eV"], "defect": res.get("defect"),
                          "formula": res.get("formula"), "energies_eV": res["energies_eV"],
                          "defect_sane": sane, "system_kind": "monolayer_2d"}
            return Verdict(
                sane=sane, decisive=decisive, valid=valid,
                summary=f"E_form({res.get('formula')} {res.get('defect',{}).get('kind')}) = "
                        f"{res['E_form_eV']} eV",
                ground_truth={"recompute_handle": "defect_formation_energy",
                              "E_form_eV": res["E_form_eV"], "defect": res.get("defect")},
                verifiable=verifiable,
                physics={"E_form_eV": res["E_form_eV"], "defect_sane": sane},
                metadata={"engine": self.engine_name, "mlip_prefiltered": self.tier == "mlip",
                          "calc_tier": self.tier, "system_kind": "monolayer_2d",
                          "trajectory_metadata": {}},
                notes_valid="Terminal recompute by real QE 7.5 (PBE-PAW): TMD defect formation energy.",
                notes_invalid=f"Loop validated with {self.engine_name} — not an admissible reward.")
        sane, decisive, valid = bool(res["chemisorption_sane"]), bool(res["decisive"]), self.admissible
        is_clean_metal = self._struct_spec is None or self._struct_spec.kind != "supported"
        # the CO/Pt(111) atop-vs-hollow puzzle framing only applies to CO on clean Pt(111);
        # for any other adsorbate/metal the 'site==ontop ⇒ reproduces experiment' heuristic
        # is meaningless, so leave it None (the soft conclusion_match judge handles those).
        s = self._struct_spec
        is_co_pt111 = bool(is_clean_metal and s and (s.element or "Pt") == "Pt"
                           and str(getattr(s, "facet", "111")) == "111"
                           and (getattr(s, "adsorbate", None) or "CO") == "CO")
        reproduces_exp = (res["site_preference"] == "ontop") if is_co_pt111 else None

        verifiable = {
            "engine": self.engine_name, "valid_reward": valid,
            "E_ads_eV": res["E_ads_eV"], "site_preference": res["site_preference"],
            "delta_eV": res["delta_eV"], "energies_eV": res["energies_eV"],
            "chemisorption_sane": sane, "decisive_site_preference": decisive,
            "system_kind": "slab" if is_clean_metal else "supported",
        }
        traj_meta: dict = {}
        if is_co_pt111:
            verifiable["reproduces_experiment"] = reproduces_exp   # False = the CO/Pt(111) puzzle
            traj_meta["reproduces_experiment"] = reproduces_exp
            traj_meta["co_pt111_puzzle_reproduced"] = (
                self.handle in ("site_preference", "co_adsorption_energy") and not reproduces_exp)

        return Verdict(
            sane=sane, decisive=decisive, valid=valid,
            summary=_fmt_result(self.handle, res),
            ground_truth={"recompute_handle": self.handle, "site_preference": res["site_preference"],
                          "E_ads_eV": res["E_ads_eV"]},
            verifiable=verifiable,
            physics={"E_ads_eV": res["E_ads_eV"], "site_preference": res["site_preference"],
                     "delta_eV": res["delta_eV"], "chemisorption_sane": sane},
            metadata={"engine": self.engine_name, "mlip_prefiltered": self.tier == "mlip",
                      "calc_tier": self.tier, "system_kind": verifiable["system_kind"],
                      "trajectory_metadata": traj_meta},
            notes_valid=("Terminal recompute by real QE 7.5 (PBE-PAW); what is verified is the terminal "
                         "E_ads/site, not each reconstructed reasoning step. PBE may prefer fcc-hollow "
                         "(the CO/Pt(111) puzzle) — then this is also a method-correction signal."),
            notes_invalid=(f"Loop validated with {self.engine_name} — NOT an admissible reward "
                           f"({'cheap MLIP prefilter' if self.tier == 'mlip' else 'EMT plumbing only'}); "
                           f"rerun with tier='qe' for the honest, admissible reward."),
        )

    # ---- domain clamps (the affordability/physics backstops) -----------------
    def _apply_overrides(self, sspec) -> str:
        note = ""
        ovr = self.overrides
        if ovr.get("support"):
            sspec.kind = "supported"; sspec.support = ovr["support"]
            note += f" [support→{ovr['support']}]"
        elif ovr.get("metal") and not ovr.get("respect_system"):
            # pinning a metal without a support forces a CLEAN-METAL slab — the LLM can't
            # divert to an oxide/exotic support (which would need pymatgen / +U and not be
            # QE-admissible here). §14 scaffolding: nail a QE-ready combo by construction.
            # respect_system=True opts OUT: honor the LLM's chosen kind (oxide/SAC/clean),
            # relying on _clamp_supported to repair it to a buildable system of that kind —
            # so papers aren't all collapsed to Pt/CO (recipe_coverage_expansion.md Stage 1).
            if sspec.kind != "slab":
                note += f" [kind {sspec.kind}→slab (clean-metal pin)]"
                sspec.kind = "slab"
        if ovr.get("metal"):
            sspec.active_metal = ovr["metal"]
            if sspec.kind != "supported":
                sspec.element = ovr["metal"]
            note += f" [metal→{ovr['metal']}]"
        return note

    @staticmethod
    def _sane_adsorbate(ads) -> str:
        """The recipe anchors a CO-adsorption energy; the LLM occasionally emits an
        adsorbate string ASE's Formula parser rejects (e.g. a charge/site suffix or a
        non-molecule token), which would crash gas_molecule(). Validate it parses as a
        chemical formula, else fall back to CO (§14: keep the run valid by construction)."""
        ads = (ads or "CO").strip()
        try:
            from ase.symbols import string2symbols
            string2symbols(ads)
        except Exception:  # noqa: BLE001
            return "CO"
        return ads or "CO"

    def _clamp_facet(self, sspec) -> str:
        """Clamp an LLM-chosen clean-metal slab to a recipe-buildable (crystal, facet).
        The recipe wires fcc/bcc {111,100,110} + hcp 0001; an out-of-set facet (e.g.
        'basal', 'edge') or a metal with no wired crystal would crash metal_slab, so we
        snap to the metal's lowest-index facet (or Pt(111) if the metal is unknown).
        Clean-metal slabs only — supported/oxide/2D systems don't use a facet builder."""
        if sspec.kind in ("supported", "monolayer_2d", "molecule", "bulk"):
            return ""
        from harness.recompute_tools import _FACET_BUILDERS, _crystal_of
        metal = sspec.element or "Pt"
        cr = _crystal_of(metal)
        note = ""
        if cr is None:
            note += f" [metal {metal!r} has no wired crystal → Pt(111)]"
            sspec.element, metal, cr = "Pt", "Pt", "fcc"
        if (cr, str(sspec.facet)) not in _FACET_BUILDERS:
            valid = sorted(f for c, f in _FACET_BUILDERS if c == cr)
            default = "111" if "111" in valid else (valid[0] if valid else "111")
            note += f" [facet {sspec.facet!r}→{default} (not wired for {cr})]"
            sspec.facet = default
        return note

    def _clamp_supported(self, sspec) -> str:
        """Clamp an LLM-chosen kind=supported spec to a BUILDABLE support, so the oxide/SAC
        recipes can be honored (not force-collapsed to clean-metal). Carbon SAC → graphene
        (the only wired carbon support) + a valid defect; oxide SAC → nearest wired oxide
        preset (ceo2/tio2/tio2_anatase). Unknown support that isn't an oxide → graphene."""
        if sspec.kind != "supported":
            return ""
        from harness.recompute_tools import _OXIDE_PRESETS
        note = ""
        sup = (sspec.support or "").lower()
        if sup in _OXIDE_PRESETS:
            sspec.support = sup                          # normalize case (recipe keys are lowercase)
            return note                                  # already a buildable oxide
        # map common oxide names/aliases to a wired preset (substring match — the LLM emits
        # free text like "rutile TiO2", "anatase titania"). Order matters: anatase before tio2.
        _OXIDE_ALIAS = [("anatase", "tio2_anatase"), ("rutile", "tio2"), ("tio2", "tio2"),
                        ("titania", "tio2"), ("ceria", "ceo2"), ("ceo2", "ceo2"),
                        ("cerium", "ceo2")]
        for key, preset in _OXIDE_ALIAS:
            if key in sup:
                if sspec.support != preset:
                    note += f" [support {sup!r}→{preset} (wired oxide preset)]"
                sspec.support = preset
                return note
        # any other oxide-ish token (contains 'oxide'/ends in o2/o3) → default ceo2
        if "oxide" in sup or sup.endswith("o2") or sup.endswith("o3"):
            note += f" [oxide support {sup!r}→ceo2 (nearest wired preset)]"
            sspec.support = "ceo2"
            return note
        # otherwise treat as a carbon SAC on graphene (the only wired carbon support)
        if sup not in ("graphene",):
            note += f" [support {sup or 'none'!r}→graphene (only wired carbon support)]"
            sspec.support = "graphene"
        if (sspec.defect or "").lower() not in ("vacancy", "divacancy", "pristine"):
            note += f" [defect {sspec.defect!r}→vacancy]"
            sspec.defect = "vacancy"
        return note

    def _clamp_structure(self, sspec) -> str:
        note = self._clamp_facet(sspec) + self._clamp_supported(sspec)
        if not self.compute_budget:
            return note
        b = self.compute_budget
        msc, mly = b.get("max_supercell"), b.get("max_layers")
        if sspec.kind == "supported":
            if self._is_oxide(sspec.support):
                if tuple(sspec.supercell or (1, 1)) != (1, 1):
                    sspec.supercell = (1, 1)
                    note += " [oxide supercell→(1,1) for real-DFT budget]"
            else:
                if any(x > 3 for x in (sspec.supercell or (3, 3))):
                    sspec.supercell = tuple(min(int(x), 3) for x in sspec.supercell)
                    note += f" [supercell capped to {sspec.supercell}]"
        else:
            if msc and tuple(sspec.supercell or (2, 2)) != tuple(min(x, msc) for x in (sspec.supercell or (2, 2))):
                sspec.supercell = tuple(min(int(x), msc) for x in sspec.supercell)
                note += f" [supercell capped to {sspec.supercell} for real-DFT budget]"
            if mly and (sspec.layers or 3) > mly:
                sspec.layers = mly
                note += f" [layers capped to {mly}]"
            # floor at 2 layers: a 1-layer slab freezes EVERY atom under fix_bottom, so QE
            # (with all if_pos=0) prints no forces and ASE's BFGS get_forces() fails. Two
            # layers leaves the top layer mobile — the minimum for a meaningful relaxation.
            if (sspec.layers or 3) < 2:
                note += f" [layers {sspec.layers}→2 (1-layer slab freezes all atoms, no forces)]"
                sspec.layers = 2
        return note

    def _sanitize_method(self, mspec_in, sspec):
        from harness.recompute_tools import _OXIDE_PRESETS
        mspec, notes = sanitize_method_spec(mspec_in)
        # the recipe drives geometry with ASE BFGS, so QE must do single-point scf+forces
        # (calc="relax" would trigger a nested QE internal relaxation on every ASE step).
        if self.tier == "qe" and mspec.calc != "scf":
            mspec.calc = "scf"
            notes = list(notes) + ["calc relax→scf (ASE BFGS drives geometry)"]
        # real-DFT budget: cap the plane-wave cutoff (cost ∝ ecut^1.5).
        if self.compute_budget and self.compute_budget.get("max_ecutwfc") \
                and mspec.ecutwfc > self.compute_budget["max_ecutwfc"]:
            old = mspec.ecutwfc
            mspec.ecutwfc = float(self.compute_budget["max_ecutwfc"])
            mspec.ecutrho = 8.0 * mspec.ecutwfc
            notes = list(notes) + [f"ecutwfc {old}->{mspec.ecutwfc} (real-DFT budget)"]
        # an oxide support REQUIRES Hubbard-U → ensure it is in the MethodSpec (don't override LLM).
        if sspec.kind == "supported" and self._is_oxide(sspec.support):
            for el, u in _OXIDE_PRESETS[sspec.support]["hubbard_u"].items():
                mspec.hubbard_u.setdefault(el, u)
        return mspec, notes

    def _describe_system(self, sspec):
        from harness.recompute_tools import _OXIDE_PRESETS
        if sspec.kind == "supported":
            m = sspec.active_metal or sspec.element or "Pt"
            sup = sspec.support or "graphene"
            if sup in _OXIDE_PRESETS:
                face = f"{tuple(sspec.miller)}" if sspec.miller else ""
                return (f"{m}@{sup}{face} {sspec.supercell}",
                        f"System: {m} single atom on {sup}{face} surface "
                        f"p{sspec.supercell}, adsorbate {sspec.adsorbate}.")
            return (f"{m}@{sspec.defect or 'vacancy'}-{sup} {sspec.supercell}",
                    f"System: {m} single atom on {sspec.defect or 'vacancy'} {sup} "
                    f"p{sspec.supercell}, adsorbate {sspec.adsorbate}.")
        return (f"{sspec.element}({sspec.facet}) {sspec.supercell} {sspec.layers}L",
                f"System: {sspec.element}({sspec.facet}) p{sspec.supercell} "
                f"{sspec.layers}-layer slab, adsorbate {sspec.adsorbate}.")


def _fmt_result(handle: str, res: dict) -> str:
    if res.get("handle") == "defect_formation_energy":
        return (f"Recompute ({handle}): E_form={res['E_form_eV']} eV; "
                f"sane={res['defect_sane']}, decisive={res['decisive']}.")
    eads = ", ".join(f"E_ads({k})={v} eV" for k, v in res["E_ads_eV"].items())
    return (f"Recompute ({handle}): {eads}; site_preference={res['site_preference']} "
            f"(Δ={res['delta_eV']} eV); chemisorption_sane={res['chemisorption_sane']}, "
            f"decisive={res['decisive']}.")


# --------------------------------------------------------------------------- #
# Task factories (DiscoveryPattern → domain-agnostic DiscoveryTask). These keep the
# catalysis premise/tension wording with the domain; the skeleton only reads the
# generic DiscoveryTask fields.
# --------------------------------------------------------------------------- #
def pick_handle(handles: list[str]) -> str:
    """The clean-metal adsorption anchor to recompute (the recipe's scope)."""
    for h in _ADSORPTION_HANDLES:
        if h in handles:
            return h
    if "site_preference" in handles or not handles:
        return "site_preference"
    raise ValueError(f"no clean-metal adsorption anchor in {handles} — out of recompute-recipe scope")


def task_from_pattern(pattern) -> DiscoveryTask:
    """DiscoveryPattern → DiscoveryTask (handle picked from the pattern's anchors)."""
    handle = pick_handle(pattern.recompute_handles())
    return DiscoveryTask(
        task_id=pattern.paper_id, title=pattern.title,
        premise=pattern.premise_consensus, tension=pattern.tension,
        handle=handle, novelty_move=pattern.novelty_move,
        provenance={"artifact_uri": pattern.artifact_uri},
    )
