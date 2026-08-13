"""Parse the paper's reported numbers (the CALCULATION ground truth) out of a discovery
pattern's ``key_claims`` into structured, comparable gold for reward computation.

The user's goal: a discovery task keeps the paper's own GT — both the calculation GT
(numbers the paper computed) and the conclusion GT — so reward can be scored against the
PAPER, not against the rollout's own recomputed value (self-gold). ``key_claims[i].value``
is free text ("201 kJ/mol (Ni6In2) vs 34 kJ/mol (Ni2In6)", "−4.50 eV", "~0.5 eV"); this
module extracts the leading numeric value + unit and normalises energies to eV so the
verifier's gold can compare magnitudes within tolerance (never across-code absolute
energies — magnitudes/signs/orderings only).

Honest scope: orderings ("UroM5 > UroM6") and dimensionless claims ("S = 1/2") yield no
numeric gold (``ev=None``); the raw text is always kept so nothing is lost.
"""
from __future__ import annotations

import re
from typing import Optional

# energy unit → eV (for magnitude comparison; site/sign matching is unit-free)
_U2EV = {"eV": 1.0, "meV": 1e-3, "kJ/mol": 1.0 / 96.485, "kcal/mol": 1.0 / 23.061,
         "Ha": 27.2114, "Ry": 13.6057}
_ENERGY_HANDLES = {"co_adsorption_energy", "reaction_barrier", "coverage_shift"}
# tolerate an uncertainty notation between number and unit, e.g. "-3.37(1) eV" → -3.37
_UNIT_RE = re.compile(
    r"(-?\d+\.?\d*)\s*(?:\(\d+\))?\s*(eV|meV|kJ\s*/?\s*mol|kcal\s*/?\s*mol|Ha|Ry|Å|cm-?1|K)\b",
    re.I)


def _norm_unit(u: str) -> str:
    u = u.replace(" ", "").lower()
    return {"ev": "eV", "mev": "meV", "kj/mol": "kJ/mol", "kjmol": "kJ/mol",
            "kcal/mol": "kcal/mol", "kcalmol": "kcal/mol", "ha": "Ha", "ry": "Ry",
            "å": "Å", "cm-1": "cm-1", "cm1": "cm-1", "k": "K"}.get(u, u)


def parse_value(value: str, handle: str = "") -> dict:
    """Extract {raw, numeric, unit, eV} from a free-text claim value.

    eV is filled only for energy units (and energy handles); a bare number under an energy
    handle is assumed already in eV. Non-numeric/ordering values → numeric=None, eV=None."""
    raw = value or ""
    v = raw.replace("−", "-").replace("–", "-").replace("⁻¹", "-1").replace("⁻", "-")
    m = _UNIT_RE.search(v)
    if m:
        num = float(m.group(1))
        unit = _norm_unit(m.group(2))
        ev = round(num * _U2EV[unit], 4) if unit in _U2EV else None
        return {"raw": raw, "numeric": num, "unit": unit, "eV": ev}
    # NO explicit energy unit → we do NOT manufacture an eV gold. The old "bare number under
    # an energy handle = eV" heuristic produced garbage anchors: "≥7-fold sites"→7.0 eV,
    # "2–3 orders of magnitude"→2.0 eV, coordination numbers, site/count indices. A real
    # adsorption/barrier energy is essentially always written WITH a unit. So a unit-less
    # value yields numeric (for the record) but eV=None → it cannot be a hard energy anchor;
    # the task falls back to site-preference / soft conclusion_match instead.
    m2 = re.search(r"(-?\d+\.?\d*)", v)
    if m2:
        return {"raw": raw, "numeric": float(m2.group(1)), "unit": None, "eV": None}
    return {"raw": raw, "numeric": None, "unit": None, "eV": None}


def extract_paper_gt(pattern: dict) -> list[dict]:
    """All recomputable claims of a pattern → list of structured GT records:
       {handle, claim, kind, value(raw), numeric, unit, eV}. Empty if none recomputable."""
    out = []
    for c in pattern.get("key_claims", []) or []:
        c = c or {}
        h = (c.get("observable") or c.get("recompute_handle") or "").strip()
        if not h or h.lower() == "none":
            continue
        # new prompt gives an explicit `unit`; fold it into the value string so parse_value's
        # unit regex catches it (legacy claims wrote the unit inline in value).
        val = str(c.get("value", "")); unit = str(c.get("unit", "") or "")
        combined = f"{val} {unit}".strip() if (unit and unit not in val) else val
        parsed = parse_value(combined, h)
        out.append({"handle": h, "claim": c.get("claim", ""), "kind": c.get("kind", ""),
                    "computable": str(c.get("computable", "")).lower().startswith("y"), **parsed})
    return out


# a paper energy obtained by a higher-level method is NOT comparable to our MLIP/PBE-level
# recompute → it must not be used as a hard energy anchor (only as a soft reference).
_CROSS_METHOD_RE = re.compile(
    r"\bDMC\b|\bQMC\b|diffusion monte|\bCCSD|coupled.?cluster|\bGW\b|\bRPA\b|\bMP2\b|"
    r"experiment|measured|\bIR\b spectro|calorimet", re.I)


def _is_cross_method(claim: str, raw: str) -> bool:
    return bool(_CROSS_METHOD_RE.search(f"{claim} {raw}"))


def gold_for_handle(paper_gt: list[dict], handle: str) -> Optional[dict]:
    """Pick the paper-GT record matching a handle (prefer one with a numeric eV value) →
    a verifier ``gold`` dict. Returns None if no recomputable claim for that handle.

    A cross-method energy (DMC/CCSD/experiment/…) is carried as a soft ``reference_eV`` +
    ``cross_method`` flag, NOT as the hard ``energy_eV`` anchor — comparing it against an
    MLIP/PBE recompute would be apples-to-oranges (the user: GT must align with the paper,
    and the paper's DMC number doesn't align with a PBE-level recompute)."""
    cands = [g for g in paper_gt if g["handle"] == handle]
    if not cands:
        return None
    cands.sort(key=lambda g: (g["eV"] is None, g["numeric"] is None))  # numeric first
    g = cands[0]
    gold = {"source": "paper", "claim": g["claim"], "value_raw": g["raw"]}
    if g["eV"] is not None:
        if _is_cross_method(g["claim"], g["raw"]):
            gold["reference_eV"] = g["eV"]     # soft reference only (not a hard gate)
            gold["cross_method"] = True
        else:
            gold["energy_eV"] = g["eV"]        # same-method magnitude anchor (within tol)
    if g["numeric"] is not None:
        gold["numeric"] = g["numeric"]
        gold["unit"] = g["unit"]
    return gold
