"""Mine the *discovery pattern* of a paper: premise/consensus → motivation →
method → experiment → conclusion (CLAUDE.md §2-layer-1, the user's discovery axis).

Two kinds of supervision live in a paper and we keep them separate (v5 §0.9
verification decoupling):

  • 严谨性 (rigor) — terminal numeric/qualitative claims that map to something we
    can deterministically recompute (e.g. CO adsorption energy on Pt(111), atop
    vs fcc-hollow site preference). We do NOT verify here; we tag each claim with a
    ``recompute_handle`` so a later hard gate (QE/MLIP) can pick it up.
  • 发现性 (discovery) — the reasoning skeleton: what consensus the work pushed
    against (the *tension*), why it mattered, how the method resolved it, and what
    *kind* of move it was (consensus-overturn / method-correction / new-regime /
    mechanism / scaling). This is soft and stays ``pending-soft-verify``.

This is RECONSTRUCTION (§5): an LLM reads the cleaned paper sections and emits a
structured record. It does NOT fabricate observations — it summarises what the
paper itself reports, conditioned on the result (EXP-Bench style). Nothing here
is admissible until the external AI Verifier (discovery) or our deterministic
recompute (rigor) signs off.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

from adapters.paper_corpus import Paper

# The 12-way move taxonomy the miner is asked to classify into (kept small + orthogonal).
NOVELTY_MOVES = [
    "consensus-overturn",   # prior consensus was wrong; this work flips it
    "method-correction",    # a known method gives wrong answers; fix the method
    "new-regime",           # extends a finding to a regime nobody measured (coverage/T/p…)
    "mechanism",            # explains *why* an observed effect happens
    "scaling-relation",     # finds a descriptor/relation predicting many systems
    "reconciliation",       # resolves a contradiction between two prior results
    "incremental",          # confirms/refines consensus without breaking it
]

EXTRACT_SYSTEM = (
    "You are a computational-science researcher (any domain: chemistry, physics, materials, "
    "biology, etc.) reading a paper to extract its DISCOVERY PATTERN — not to summarise it. "
    "You output strict JSON only, no prose around it. Be faithful to what the paper actually "
    "claims; never invent numbers. If a field is not stated, use an empty string or empty list. "
    "Quote real numbers WITH UNITS when the paper gives them. Do NOT force the paper into any "
    "particular subfield — describe the system and quantities the paper is actually about."
)

_EXTRACT_PROMPT = """Extract the discovery pattern of this paper as JSON with EXACTLY these keys:

{{
  "premise_consensus": "the established prior understanding the paper builds on / takes as given (from the introduction). What did the field already believe?",
  "tension": "the gap, contradiction, or anomaly in that consensus that motivated this work. This is the discovery seed — what was unsatisfying or unknown? Empty string if the paper is purely confirmatory.",
  "motivation": "why resolving that tension matters (the stated goal).",
  "method": "the approach taken to resolve it (e.g. DFT/MD/MLIP setup, experiment, model, code). One or two sentences.",
  "experiment": "what system was studied and what was actually computed/measured, with key conditions/comparisons — whatever they are for THIS paper (molecule, surface, crystal, defect, interface, device, reaction, ...).",
  "conclusion": "the terminal claim — what the paper concludes. Be specific.",
  "key_claims": [
     {{"claim": "a single concrete claim from the conclusions",
       "kind": "quantitative" or "qualitative",
       "observable": "the NAME of the physical quantity this claim is about, snake_case, OPEN VOCABULARY using the paper's own quantity (e.g. adsorption_energy, formation_energy, band_gap, reaction_barrier, diffusion_coefficient, elastic_modulus, binding_affinity, redox_potential, vibrational_frequency, conductivity, magnetic_moment, lattice_constant, ...). Empty if purely qualitative with no measurable quantity.",
       "value": "the numeric value if quantitative, else a short phrase",
       "unit": "the unit of value (eV, eV/atom, Å, cm^-1, K, GPa, ...), empty if dimensionless/none",
       "computable": "yes if an independent first-principles/atomistic calc (DFT/MD/MLIP/quantum-chem) or code could in principle re-derive this quantity from the described system; else no"}}
  ],
  "novelty_move": "classify the paper's primary move as ONE of: {moves}",
  "novelty_rationale": "one sentence justifying the move label, grounded in the tension+conclusion."
}}

Rules:
- "tension" is the most important field for discovery. A good tension reads like "consensus said X, but Y was unexplained / measured wrong / never tested." Do not restate the conclusion as the tension.
- "observable" is OPEN — name the quantity the paper actually reports; do NOT shoehorn into adsorption/catalysis terms unless the paper is genuinely about that.
- key_claims: 1-4 items. Prefer claims that are quantitative AND computable=yes (the rigor anchors), but keep the paper's real quantities regardless.
- Output ONLY the JSON object.

PAPER (id: {pid}, title: {title}):
{narrative}
"""

AGGREGATE_SYSTEM = (
    "You are a senior reviewer synthesising a focused literature corpus. You compare the "
    "extracted claims of many papers on ONE topic and surface where they AGREE (consensus) "
    "and where they CONTRADICT each other (the live questions). Output strict JSON only."
)

_AGGREGATE_PROMPT = """Below are discovery-pattern records extracted from {n} papers on the same topic.
Synthesise the corpus-level picture as JSON with EXACTLY these keys:

{{
  "topic": "one phrase naming what this corpus is about",
  "consensus_claims": [{{"claim": "...", "supported_by": ["paper_id", ...], "recompute_handle": "..."}}],
  "contradictions": [{{"question": "the open/disputed point", "position_a": "...", "papers_a": ["id"], "position_b": "...", "papers_b": ["id"]}}],
  "recurring_tensions": ["the discovery seeds that show up across multiple papers"],
  "rigor_anchors": ["the specific recomputable quantities (with handles) that a deterministic gate could verify across this corpus"],
  "move_distribution": {{"consensus-overturn": int, "method-correction": int, "...": int}}
}}

Focus the contradictions on disputes a CALCULATION could adjudicate (e.g. site preference, adsorption-energy magnitude, coverage dependence). Output ONLY the JSON object.

RECORDS:
{records}
"""


@dataclass
class DiscoveryPattern:
    paper_id: str
    title: str
    premise_consensus: str = ""
    tension: str = ""
    motivation: str = ""
    method: str = ""
    experiment: str = ""
    conclusion: str = ""
    key_claims: list = field(default_factory=list)
    novelty_move: str = ""
    novelty_rationale: str = ""
    # provenance (§10) — every reconstruction stamps its source + teacher model
    artifact_uri: str = ""
    model_id: str = ""
    reconstruct_method: str = "discovery_pattern.extract"
    status: str = "pending-soft-verify"          # never admissible from extraction alone

    @classmethod
    def from_dict(cls, d: dict) -> "DiscoveryPattern":
        """Build from a stored record, ignoring extra keys (e.g. tier/tier_weight/
        tier_reasons appended by the tier backfill/join) that aren't dataclass fields."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def recompute_handles(self) -> list[str]:
        """The paper's observables (open vocabulary), computable-first. Reads the new
        `observable` field (falls back to legacy `recompute_handle`)."""
        comp, other = [], []
        for c in self.key_claims:
            c = c or {}
            obs = (c.get("observable") or c.get("recompute_handle") or "").strip()
            if not obs or obs.lower() == "none":
                continue
            (comp if str(c.get("computable", "")).lower().startswith("y") else other).append(obs)
        return comp + other


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of an LLM reply (tolerant of code fences / prose)."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:
        s = text.find("{")
        e = text.rfind("}")
        blob = text[s:e + 1] if s != -1 and e > s else None
    if not blob:
        return None
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


def extract_pattern(llm, paper: Paper, *, narrative_cap: int = 14000) -> Optional[DiscoveryPattern]:
    """Reconstruct one paper's discovery pattern. Returns None if the LLM reply isn't parseable."""
    narrative = paper.narrative()[:narrative_cap]
    if len(narrative) < 200:
        return None                                  # too little parsed text to be meaningful
    prompt = _EXTRACT_PROMPT.format(
        pid=paper.paper_id, title=paper.title, narrative=narrative,
        moves=", ".join(NOVELTY_MOVES))
    reply = llm.complete(prompt, system=EXTRACT_SYSTEM)
    d = _extract_json(reply)
    if not d:
        return None
    return DiscoveryPattern(
        paper_id=paper.paper_id, title=paper.title,
        premise_consensus=str(d.get("premise_consensus", "")),
        tension=str(d.get("tension", "")),
        motivation=str(d.get("motivation", "")),
        method=str(d.get("method", "")),
        experiment=str(d.get("experiment", "")),
        conclusion=str(d.get("conclusion", "")),
        key_claims=d.get("key_claims", []) if isinstance(d.get("key_claims"), list) else [],
        novelty_move=str(d.get("novelty_move", "")),
        novelty_rationale=str(d.get("novelty_rationale", "")),
        artifact_uri=paper.artifact_uri,
        model_id=getattr(llm, "_model", "") or "",
    )


def aggregate(llm, patterns: list[DiscoveryPattern]) -> Optional[dict]:
    """Cross-paper synthesis: consensus vs contradiction + rigor anchors over the corpus."""
    compact = [{
        "paper_id": p.paper_id,
        "tension": p.tension,
        "conclusion": p.conclusion,
        "novelty_move": p.novelty_move,
        "key_claims": [{"claim": c.get("claim", ""), "value": c.get("value", ""),
                        "recompute_handle": c.get("recompute_handle", "none")}
                       for c in p.key_claims],
    } for p in patterns]
    prompt = _AGGREGATE_PROMPT.format(n=len(patterns), records=json.dumps(compact, ensure_ascii=False))
    reply = llm.complete(prompt, system=AGGREGATE_SYSTEM)
    return _extract_json(reply)


def to_dict(p: DiscoveryPattern) -> dict:
    return asdict(p)
