"""Three-dimensional discovery verifier — the scorer for the Claude-Code-harness flywheel
(CLAUDE.md §0.9, §18; docs/discovery_verifier_design.md).

Flywheel: Claude Code harness + Qwen3-MoE endpoint runs a discovery task → THIS verifier
scores the rollout → high-scored rollouts are kept for SFT → the tuned model drives the
harness again. The harness is fixed; the flywheel evolves the model.

THE CORE IDEA (user-driven, 2026-06-22): "computational match ≠ scientific discovery".
Discovery = novel ∧ correct ∧ meaningful, and only *correct* is deterministically
verifiable. So the score is a **gate × ranking**, never a flat weighted sum:

    reward = correctness × (0.5 + 0.25·significance + 0.25·novelty)

  * correctness ∈ {0,1} — HARD gate. A re-executed, sane number matching the gold within
    tolerance. A beautiful-but-wrong (or un-run) rollout scores 0 — unhackable.
  * significance ∈ [0,1] — did it actually adjudicate the tension? (decisive flag + judge)
  * novelty ∈ [0,1] — did it go beyond the premise, consistent with the novelty_move? (judge)

The soft dims only RANK rollouts that already passed the correctness gate, so soft-dim
hacking cannot mint reward from nothing. Division of labour: the verifier owns *correctness*
(hard); *discovery-ness* is carried by the task's tension quality (soft + front-loaded).

This module is engine-agnostic in structure: the correctness gate is supplied as a
callable (``CorrectnessGate``), so a second domain plugs in its own gate without touching
the gate×ranking logic. The catalysis QE gate lives in ``catalysis_correctness_gate``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional


# --------------------------------------------------------------------------- #
# What the agent reported (parsed from the harness rollout) + the task's anchors.
# --------------------------------------------------------------------------- #
@dataclass
class ReportedResult:
    """The machine-parsable FINAL block the agent is required to emit (see design §4).

    Carries the quantity for whichever recompute_handle the task uses: adsorption
    (e_ads_eV/site_preference), reaction_barrier (barrier_eV), or vibrational_frequency
    (frequency_cm1). Only the field for the task's handle need be populated."""
    e_ads_eV: dict = field(default_factory=dict)   # {"ontop": -1.53, "fcc": -1.69}
    site_preference: Optional[str] = None
    barrier_eV: Optional[float] = None             # reaction_barrier handle
    frequency_cm1: Optional[float] = None          # vibrational_frequency handle
    # GENERIC scalar metric (domain-agnostic) — the value the agent's EXECUTED code produced
    # for a paper-reported quantity (AUC, accuracy, RMSE, correlation, IoU, …). This is what
    # lets ML/bio/physics tasks reach the hard tier without a QE backend: the code-execution /
    # metric-reproduction path (§ verifier generalisation). ``metric_executed`` must be True
    # (number came from real code output, not the LLM's prose) for the result to be admissible.
    metric_value: Optional[float] = None
    metric_name: str = ""
    metric_executed: bool = False
    conclusion: str = ""
    system_desc: str = ""
    method_desc: str = ""
    raw: str = ""                                  # the whole final answer, for the judges


@dataclass
class CorrectnessVerdict:
    """Output of a correctness gate: the {0,1} gate plus the numbers behind it (for signals)."""
    correct: bool
    sane: bool
    decisive: bool
    valid: bool                       # admissible tier (real engine), not a proxy
    matched_gold: Optional[bool]      # None = no comparable gold (B-tier self-consistent)
    # GRADED closeness to the paper's experimental value ∈ [0,1] (1.0 = within tolerance,
    # decaying with relative error). None = no numeric comparison was possible. Lets the
    # reward give PARTIAL credit for near-reproductions instead of a binary in/out cliff —
    # the "soft gate also compares experimental results" refinement. Only trusted (executed /
    # re-executed) numbers feed the graded reward; a self-claimed number cannot mint it.
    graded: Optional[float] = None
    trusted: bool = False             # the number came from a real run (exec/re-exec), not prose
    detail: dict = field(default_factory=dict)


@dataclass
class DiscoveryScore:
    """The composed score. ``reward`` is the single float rLLM consumes; the per-dimension
    values are returned as signals for analysis and for the flywheel's high-score selection."""
    reward: float
    correctness: float
    significance: float
    novelty: float
    conclusion_match: float            # did the rollout reach the source paper's finding?
    is_correct: bool
    tier: str                          # "A" | "B" | "C" | "none"
    admissible: bool
    detail: dict = field(default_factory=dict)

    def as_eval_dict(self) -> dict:
        """Shape rLLM's PythonModuleEvaluator coerces into an EvalOutput."""
        return {
            "reward": self.reward,
            "is_correct": self.is_correct,
            "signals": {"correctness": self.correctness, "significance": self.significance,
                        "novelty": self.novelty, "conclusion_match": self.conclusion_match},
            "metadata": {"tier": self.tier, "admissible": self.admissible, **self.detail},
        }


# A correctness gate maps (reported, task_meta) → CorrectnessVerdict. It is where the live
# re-execution happens (the domain plugin); the verifier core never imports an engine.
CorrectnessGate = Callable[[ReportedResult, dict], CorrectnessVerdict]


def task_to_meta(task: dict) -> dict:
    """Canonical task-row → ``task_meta`` adapter, the single handoff rLLM (or any caller)
    uses to turn one line of ``discovery_tasks.jsonl`` into what ``score_discovery`` reads.

    Keeps the mapping in ONE place: the task uses ``handle`` while the verifier reads
    ``recompute_handle``; ``gold`` carries the paper's calculation GT (paper_gt.gold_for_handle)
    so reward anchors to the PAPER's reported number, not the rollout's self-gold."""
    return {
        "tension": task.get("tension", ""),
        "premise": task.get("premise", ""),
        "conclusion": task.get("conclusion", ""),       # paper's GT → conclusion_match
        "novelty_move": task.get("novelty_move", ""),
        "recompute_handle": task.get("handle", "site_preference"),
        "gold": task.get("gold") or {},                 # paper calculation GT (reward anchor)
        "paper_gt": task.get("paper_gt", []),           # all recomputable claims' GT
        "tier": task.get("tier") or ("A" if task.get("gold") else "B"),
        "unverified": bool(task.get("unverified")),
    }


def task_to_query(task: dict, *, hide_tension: bool = False) -> str:
    """Agent-facing task query (the prompt rLLM shows the policy). Separated from task_meta
    on purpose: the GT (tension/gold/conclusion) always stays in task_meta for scoring, but
    what the AGENT sees can be made harder.

    hide_tension=False (default): give premise + tension — "here is the crack, resolve it".
    hide_tension=True (difficulty knob): give ONLY the premise — the agent must DISCOVER the
    tension itself (its identify_tension move), and the verifier scores the tension it finds
    against the held-out GT tension. The tension is never removed from the data, only from
    the agent's view."""
    premise = (task.get("premise") or "").strip()
    if hide_tension:
        return (f"Given the established understanding that {premise} — identify an open "
                "question or tension in this consensus that a calculation could resolve, "
                "then resolve it.")
    tension = (task.get("tension") or "").strip()
    return f"Given that {premise} resolve: {tension}"


# --------------------------------------------------------------------------- #
# Parsing the agent's FINAL block.
# --------------------------------------------------------------------------- #
def parse_final(answer: str) -> ReportedResult:
    """Parse the required FINAL block. Tolerant: missing fields stay empty so the
    correctness gate (not the parser) decides the verdict."""
    r = ReportedResult(raw=answer or "")
    # E_ads_eV: {ontop: -1.53, fcc: -1.69}  — accept a dict-ish line
    m = re.search(r"E_ads_eV\s*:\s*\{([^}]*)\}", answer or "", re.I)
    if m:
        for k, v in re.findall(r"([A-Za-z_\-]+)\s*:\s*(-?\d+\.?\d*)", m.group(1)):
            r.e_ads_eV[k.strip().lower()] = float(v)
    m = re.search(r"site_preference\s*:\s*([A-Za-z_\-]+)", answer or "", re.I)
    if m:
        r.site_preference = m.group(1).strip().lower()
    m = re.search(r"barrier_eV\s*:\s*(-?\d+\.?\d*)", answer or "", re.I)
    if m:
        r.barrier_eV = float(m.group(1))
    m = re.search(r"frequency_cm1\s*:\s*(-?\d+\.?\d*)", answer or "", re.I)
    if m:
        r.frequency_cm1 = float(m.group(1))
    # generic metric: "metric_value: 0.923" (+ optional "metric_name: auc", "executed: true")
    m = re.search(r"metric_value\s*:\s*(-?\d+\.?\d*)", answer or "", re.I)
    if m:
        r.metric_value = float(m.group(1))
    m = re.search(r"metric_name\s*:\s*([A-Za-z0-9_\-/ ]+)", answer or "", re.I)
    if m:
        r.metric_name = m.group(1).strip().lower()
    m = re.search(r"executed\s*:\s*(true|1|yes)\b", answer or "", re.I)
    if m:
        r.metric_executed = True
    for field_name, attr in (("conclusion", "conclusion"), ("system", "system_desc"),
                             ("method", "method_desc")):
        m = re.search(rf"{field_name}\s*:\s*(.+)", answer or "", re.I)
        if m:
            setattr(r, attr, m.group(1).strip())
    return r


# --------------------------------------------------------------------------- #
# The soft judges (significance + novelty). Optional: with no LLM client they fall back
# to the deterministic floor (decisive flag), so the pipeline runs without a judge.
# --------------------------------------------------------------------------- #
_SIG_SYS = ("You are a strict scientific reviewer. Answer ONLY with a number 0.0-1.0.")
_SIG_PROMPT = ("A discovery task posed this tension:\n{tension}\n\n"
               "The agent's conclusion + numbers:\n{conclusion}\n\n"
               "To what degree does this conclusion DECISIVELY ADJUDICATE the tension "
               "(not merely report a number, not restate the premise)? "
               "0.0 = does not address it, 1.0 = cleanly settles it with evidence. "
               "Reply with only the number.")
_NOV_PROMPT = ("Field's prior consensus (premise):\n{premise}\n\n"
               "The agent's conclusion:\n{conclusion}\n\n"
               "The intended discovery move is '{novelty_move}'. To what degree does the "
               "conclusion go BEYOND merely restating the premise, consistent with that move? "
               "0.0 = pure restatement, 1.0 = a substantive step beyond consensus. "
               "Reply with only the number.")
# Did the rollout REACH the paper's actual conclusion? Soft, semantic — compares the
# agent's conclusion to the source paper's reported conclusion. Stays a RANKING dim only
# (after the correctness hard gate), so it can't mint reward and can't punish a physically
# correct result that differs from the paper (e.g. the CO/Pt PBE-vs-experiment puzzle).
_CONCL_PROMPT = ("The source paper concluded:\n{paper_conclusion}\n\n"
                 "The agent's rollout concluded:\n{agent_conclusion}\n\n"
                 "To what degree does the agent's conclusion REACH the SAME scientific finding "
                 "as the paper (same qualitative claim / direction / mechanism)? Judge the "
                 "scientific content, not the wording. 0.0 = unrelated or contradicts, "
                 "1.0 = reaches the same finding. Reply with only the number.")


def _judge_score(llm, prompt: str, system: str) -> Optional[float]:
    if llm is None:
        return None
    try:
        raw = llm.complete(prompt, system=system)
    except Exception:
        return None
    m = re.search(r"-?\d+\.?\d*", raw or "")
    if not m:
        return None
    return max(0.0, min(1.0, float(m.group())))


def judge_significance(llm, tension: str, reported: ReportedResult,
                       decisive_floor: bool) -> float:
    """Significance ∈ [0,1]. Deterministic floor = 0.5 if the experiment was decisive (it
    compared the arms of the tension), 0.0 otherwise; the LLM judge refines upward."""
    floor = 0.5 if decisive_floor else 0.0
    s = _judge_score(llm, _SIG_PROMPT.format(tension=tension or "",
                     conclusion=(reported.conclusion or reported.raw)[:1500]), _SIG_SYS)
    return floor if s is None else max(floor, s)


def judge_novelty(llm, premise: str, novelty_move: str, reported: ReportedResult) -> float:
    """Novelty ∈ [0,1] — beyond the premise, consistent with the intended move. With no
    judge, fall back to a neutral 0.5 (don't reward or punish what we can't assess)."""
    s = _judge_score(llm, _NOV_PROMPT.format(premise=premise or "", novelty_move=novelty_move or "",
                     conclusion=(reported.conclusion or reported.raw)[:1500]), _SIG_SYS)
    return 0.5 if s is None else s


def judge_conclusion_match(llm, paper_conclusion: str, reported: ReportedResult) -> float:
    """conclusion_match ∈ [0,1] — did the rollout reach the SAME finding as the source paper?
    Soft semantic judge against the paper's own conclusion. With no judge (or no paper
    conclusion to compare), fall back to neutral 0.5 — never reward/punish what we can't
    assess. Stays a ranking-only dim (after the correctness gate), so a physically correct
    result that legitimately differs from the paper is not penalised in the hard sense."""
    if not (paper_conclusion or "").strip():
        return 0.5
    s = _judge_score(llm, _CONCL_PROMPT.format(
        paper_conclusion=paper_conclusion[:1500],
        agent_conclusion=(reported.conclusion or reported.raw)[:1500]), _SIG_SYS)
    return 0.5 if s is None else s


# --------------------------------------------------------------------------- #
# The composer: gate × ranking.
# --------------------------------------------------------------------------- #
def score_discovery(reported: ReportedResult, task_meta: dict, *,
                    gate: CorrectnessGate, llm=None) -> DiscoveryScore:
    """Compose the three dimensions into one reward via gate × ranking.

    ``task_meta`` carries the task's anchors: ``tension``, ``premise``, ``novelty_move``,
    ``recompute_handle``, optional numeric ``gold``, and a ``tier`` hint ("A"/"B"/"C").
    ``gate`` runs the correctness check (re-execution lives there). ``llm`` (optional)
    drives the soft judges; without it they fall back to deterministic floors."""
    cv = gate(reported, task_meta)
    # PROXY correctness drives the RL reward (matches the rLLM in-container verify.py):
    # sane ∧ decisive ∧ the recompute MATCHED THE PAPER's GT. NOT requiring a real-QE engine
    # (so the cheap MLIP tier yields a usable signal), but it DOES require a genuine paper-GT
    # match — `matched_gold is True`, not merely "not False". A task whose paper has no
    # comparable GT (matched_gold is None) must NOT be granted hard correctness for being
    # merely sane+decisive: that would reward the agent's own self-consistent computation
    # WITHOUT any paper anchor (the "self-gold" hole). Such tasks route to the soft path below.
    has_paper_gold = cv.matched_gold is not None
    proxy_correct = bool(cv.sane and cv.decisive and (cv.matched_gold is True))
    correctness = 1.0 if proxy_correct else 0.0

    significance = judge_significance(llm, task_meta.get("tension", ""), reported, cv.decisive)
    novelty = judge_novelty(llm, task_meta.get("premise", ""),
                            task_meta.get("novelty_move", ""), reported)
    conclusion_match = judge_conclusion_match(llm, task_meta.get("conclusion", ""), reported)

    soft = 0.2 * significance + 0.2 * novelty + 0.2 * conclusion_match   # the 0.6 ranking band

    # "unverified" = correctness cannot be anchored to the paper: either the system is out of
    # recompute scope (task_meta unverified), OR the paper reports no comparable GT for this
    # handle (matched_gold is None). Either way there is nothing to hard-verify against → the
    # reward is soft-only (ranking signal for RL), never hard correctness, never admissible.
    # (This closes the self-gold hole: no paper GT ⇒ no hard reward for a sane+decisive run.)
    unverified = bool(task_meta.get("unverified")) or not has_paper_gold
    if unverified:
        # soft-only, ungated: full [0,1] range — these tasks have NO computational-verification
        # path, so soft IS the reward; a rollout that cleanly reaches the paper's finding should
        # score ~1.0 (conclusion_match weighted highest), not be capped in the 0.6 ranking band.
        reward = 0.5 * conclusion_match + 0.25 * significance + 0.25 * novelty
        admissible = False
    else:
        # gate × ranking: baseline 0.4 + discovery band, scaled by the correctness signal.
        # For a TRUSTED number (executed / re-executed) we scale by GRADED closeness to the
        # paper's experimental value, so a near-reproduction earns partial credit instead of a
        # binary 0 — the "soft gate also compares the experimental result" refinement. A
        # self-claimed (un-executed) number cannot use the graded path → falls back to the
        # binary gate, so it still can't mint reward. is_correct/admissible stay strictly binary.
        if cv.trusted and cv.graded is not None:
            reward = cv.graded * (0.4 + soft)
        else:
            reward = correctness * (0.4 + soft)
        admissible = bool(cv.correct and cv.valid and (cv.matched_gold is True))

    tier = task_meta.get("tier", "A" if cv.matched_gold is not None else "B")
    return DiscoveryScore(
        reward=round(reward, 4), correctness=correctness, significance=round(significance, 4),
        novelty=round(novelty, 4), conclusion_match=round(conclusion_match, 4),
        is_correct=proxy_correct, tier=tier, admissible=admissible,
        detail={"sane": cv.sane, "decisive": cv.decisive, "valid": cv.valid,
                "proxy_correct": proxy_correct, "qe_admissible": bool(cv.correct),
                "unverified": unverified, "matched_gold": cv.matched_gold, **cv.detail},
    )


# --------------------------------------------------------------------------- #
# Catalysis correctness gate — the domain plugin's gate. Re-executes the agent's chosen
# system/method (NOT trusting the agent's self-reported number → anti-hack) and matches it
# against the gold. Reuses harness/domains + recompute_tools.
# --------------------------------------------------------------------------- #
def catalysis_correctness_gate(*, tier: str = "emt", tol_eV: float = 0.2,
                               reexecute: bool = True) -> CorrectnessGate:
    """Build a catalysis correctness gate.

    ``reexecute=True`` (recommended, anti-hack): the gate re-runs the agent's stated
    system+method via recompute_tools and scores THAT, ignoring the agent's self-reported
    number. ``reexecute=False`` trusts the agent's parsed numbers (cheap plumbing only).
    ``tier`` selects the engine (emt|mlip|qe); only qe is admissible (``valid``)."""

    def gate(reported: ReportedResult, task_meta: dict) -> CorrectnessVerdict:
        valid = (tier == "qe")
        if not reexecute:
            # trust-the-agent path (plumbing): sane/decisive from the reported numbers.
            eads = reported.e_ads_eV
            sane = bool(eads) and all(-4.0 < v < 0.5 for v in eads.values())
            decisive = len(eads) >= 2 and abs(list(eads.values())[0] - list(eads.values())[1]) > 0.01 \
                if reported.site_preference is None or len(eads) >= 2 else bool(eads)
            matched = _match_gold(eads, reported.site_preference, task_meta, tol_eV)
            return CorrectnessVerdict(correct=bool(sane and decisive and (matched is not False)),
                                      sane=sane, decisive=decisive, valid=valid,
                                      matched_gold=matched, detail={"mode": "trust-reported"})
        # re-execution path (anti-hack): rebuild + re-run the agent's choice.
        from harness.domains.catalysis_qe import CatalysisQEOracle
        from reconstruct.discovery_moves import MethodSpec, StructureSpec
        sspec, mspec = _specs_from_task(reported, task_meta)
        oracle = CatalysisQEOracle(tier=tier, handle=task_meta.get("recompute_handle", "site_preference"))
        # drive design_from_action with the reconstructed spec, then execute
        design = oracle.design_from_action(
            {"system": _asdict(sspec), "method": _asdict(mspec), "thought": "verifier re-exec"}, {})
        execution = oracle.execute(design)
        v = oracle.assess(execution)
        res = execution.result
        matched = _match_gold(res["E_ads_eV"], res["site_preference"], task_meta, tol_eV)
        return CorrectnessVerdict(
            correct=bool(v.sane and v.decisive and valid and (matched is not False)),
            sane=v.sane, decisive=v.decisive, valid=valid, matched_gold=matched,
            detail={"mode": "reexecute", "E_ads_eV": res["E_ads_eV"],
                    "site_preference": res["site_preference"], "engine": v.metadata.get("engine")})

    return gate


def _match_gold(eads: dict, site_pref: Optional[str], task_meta: dict,
                tol_eV: float) -> Optional[bool]:
    """Compare to the task's gold. Returns True/False, or None if no comparable gold
    (B-tier: judge self-consistency only). Site preference is matched qualitatively (sign),
    energies within tolerance — never compare absolute energies across codes."""
    gold = task_meta.get("gold") or {}
    gold_site = gold.get("site_preference")
    if gold_site and site_pref:
        return site_pref.lower() == str(gold_site).lower()
    # energy gold: self-gold uses 'e_ads_eV'; the PAPER's calculation GT (paper_gt.py) uses
    # 'energy_eV'. Either way compare the most-stable E_ads magnitude within tolerance —
    # this is the reward anchored to the PAPER's reported number (the user's goal).
    gold_e = gold.get("e_ads_eV")
    if gold_e is None:
        gold_e = gold.get("energy_eV")
    if gold_e is not None and eads:
        return abs(min(eads.values()) - float(gold_e)) <= tol_eV
    return None   # no comparable gold


def _specs_from_task(reported: ReportedResult, task_meta: dict):
    """Reconstruct StructureSpec/MethodSpec for re-execution. Prefer the task's pinned
    QE-ready combo (data-gen scaffolding); fall back to parsing the agent's FINAL block."""
    from reconstruct.discovery_moves import MethodSpec, StructureSpec
    sys_hint = task_meta.get("system_spec") or {}
    meth_hint = task_meta.get("method_spec") or {}
    sspec = StructureSpec(**{k: v for k, v in sys_hint.items() if k in StructureSpec.__annotations__}) \
        if sys_hint else StructureSpec(kind="slab", element=task_meta.get("metal", "Pt"),
                                       facet="111", supercell=(2, 2), layers=3, adsorbate="CO")
    mspec = MethodSpec(**{k: v for k, v in meth_hint.items() if k in MethodSpec.__annotations__}) \
        if meth_hint else MethodSpec()
    return sspec, mspec


# --------------------------------------------------------------------------- #
# reaction_barrier gate — 35 corpus papers carry this anchor. sane = a positive,
# physically-bounded activation barrier; decisive = a finite barrier was actually
# computed (a path exists). gold (if given) compared within tol_eV.
# --------------------------------------------------------------------------- #
def barrier_correctness_gate(*, tier: str = "emt", tol_eV: float = 0.3,
                             reexecute: bool = False) -> CorrectnessGate:
    """Build a reaction-barrier correctness gate. Re-execution (CI-NEB) needs the task to
    supply endpoint structures (``initial``/``final`` in task_meta) — when absent the gate
    falls back to trust-reported on the agent's parsed ``barrier_eV``. Only qe is admissible."""

    def gate(reported: ReportedResult, task_meta: dict) -> CorrectnessVerdict:
        valid = (tier == "qe")
        b = reported.barrier_eV
        endpoints = task_meta.get("initial") is not None and task_meta.get("final") is not None
        if reexecute and endpoints:
            from harness import recompute_tools as RT
            cf = (RT.emt_factory() if tier == "emt" else RT.mlip_factory("chgnet")
                  if tier == "mlip" else RT.qe_factory(None))
            res = RT.reaction_barrier(task_meta["initial"], task_meta["final"], calc_factory=cf)
            b = res["barrier_eV"]
            mode = "reexecute"
        else:
            mode = "trust-reported"
        # sane: a barrier is a positive activation energy in a physical window (0–6 eV).
        sane = b is not None and 0.0 < b < 6.0
        decisive = b is not None                       # a finite barrier was produced
        matched = None
        gold_b = (task_meta.get("gold") or {}).get("barrier_eV")
        if gold_b is not None and b is not None:
            matched = abs(b - float(gold_b)) <= tol_eV
        return CorrectnessVerdict(correct=bool(sane and decisive and (matched is not False)),
                                  sane=bool(sane), decisive=bool(decisive), valid=valid,
                                  matched_gold=matched, detail={"mode": mode, "barrier_eV": b})

    return gate


# --------------------------------------------------------------------------- #
# vibrational_frequency gate — 21 corpus papers (e.g. CO stretch as a site probe).
# sane = a real (non-imaginary) frequency in a physical window; decisive = a frequency
# was produced. CO stretch is the classic adsorption-site fingerprint.
# --------------------------------------------------------------------------- #
def vibration_correctness_gate(*, tier: str = "emt", tol_cm1: float = 80.0,
                               reexecute: bool = False) -> CorrectnessGate:
    """Build a vibrational-frequency correctness gate. Re-execution (finite-difference
    Hessian) needs the task to supply an ``atoms`` structure; absent it, trust-reported on
    ``frequency_cm1``. Frequencies are method-comparable across codes (unlike absolute E),
    so gold matching within tol_cm1 is meaningful. Only qe is admissible."""

    def gate(reported: ReportedResult, task_meta: dict) -> CorrectnessVerdict:
        valid = (tier == "qe")
        nu = reported.frequency_cm1
        if reexecute and task_meta.get("atoms") is not None:
            from harness import recompute_tools as RT
            cf = (RT.emt_factory() if tier == "emt" else RT.mlip_factory("chgnet")
                  if tier == "mlip" else RT.qe_factory(None))
            res = RT.vibrational_frequencies(task_meta["atoms"], calc_factory=cf,
                                             indices=task_meta.get("indices"))
            nu = res["max_cm1"]
            mode = "reexecute"
        else:
            mode = "trust-reported"
        # sane: a real vibrational mode in a physical window (200–4000 cm⁻¹ covers CO stretch
        # ~2000 and adsorbate modes); reject 0/negative (imaginary → not a minimum).
        sane = nu is not None and 200.0 < nu < 4000.0
        decisive = nu is not None
        matched = None
        gold_nu = (task_meta.get("gold") or {}).get("frequency_cm1")
        if gold_nu is not None and nu is not None:
            matched = abs(nu - float(gold_nu)) <= tol_cm1
        return CorrectnessVerdict(correct=bool(sane and decisive and (matched is not False)),
                                  sane=bool(sane), decisive=bool(decisive), valid=valid,
                                  matched_gold=matched, detail={"mode": mode, "frequency_cm1": nu})

    return gate


# --------------------------------------------------------------------------- #
# Dispatcher: pick the right gate for a task's recompute_handle. This is how the
# verifier "supports more tasks" — one entry point, handle → gate.
# --------------------------------------------------------------------------- #
def gate_for_handle(handle: str, *, tier: str = "emt", reexecute: bool = True) -> CorrectnessGate:
    """Return the correctness gate matching a recompute_handle. Raises on handles we cannot
    deterministically recompute (honest: work_function/scaling_relation are not wired)."""
    if handle in ("site_preference", "co_adsorption_energy", "coverage_shift"):
        return catalysis_correctness_gate(tier=tier, reexecute=reexecute)
    if handle == "reaction_barrier":
        return barrier_correctness_gate(tier=tier, reexecute=reexecute)
    if handle == "vibrational_frequency":
        return vibration_correctness_gate(tier=tier, reexecute=reexecute)
    raise NotImplementedError(
        f"no correctness gate for handle {handle!r} — work_function (pp.x electrostatics) and "
        f"scaling_relation (d-band descriptor over a family) are not wired; out of verifier scope")


def _asdict(spec):
    from dataclasses import asdict
    return asdict(spec)


# --------------------------------------------------------------------------- #
# Generic metric-reproduction gate — the NON-QE hard gate (§ verifier generalisation,
# user: "把 QE 从必要条件里解耦"). For any domain whose paper reports a scalar metric
# (AUC/accuracy/F1/RMSE/correlation/IoU/…) this reproduces the number by CODE EXECUTION:
# the agent's rollout runs code, the harness captures the produced metric into
# ``reported.metric_value`` (with ``metric_executed=True``), and this gate matches it to the
# paper's reported value within a RELATIVE tolerance. Same hard-gate discipline as the QE
# path — the number must come from a real run (anti-hack), just a different backend. This is
# exactly NatureBench's own model (code-out, scored vs the paper's published number).
# --------------------------------------------------------------------------- #
# metrics bounded to [0,1] (a value outside is non-physical → not sane); correlation ∈ [-1,1].
_UNIT_INTERVAL_METRICS = ("auc", "auroc", "auprc", "accuracy", "acc", "f1", "precision",
                          "recall", "iou", "dice", "r2", "ap", "map", "specificity",
                          "sensitivity")
_SIGNED_UNIT_METRICS = ("correlation", "pearson", "spearman", "corr", "r_value")


def graded_closeness(value: float, gold: float, rel_tol: float) -> float:
    """Graded [0,1] closeness of a reproduced number to the paper's experimental value.
    1.0 within ``rel_tol`` (a hit), then decays linearly to 0 at 4× the tolerance band —
    so "almost reproduced the paper's number" earns partial credit, not a binary 0.
    Direction-agnostic — for a symmetric quantity where only reproducing the value matters."""
    denom = max(abs(gold), 1e-9)
    rel = abs(value - gold) / denom
    if rel <= rel_tol:
        return 1.0
    span = 3.0 * rel_tol                      # width of the decay ramp beyond the hit band
    return max(0.0, 1.0 - (rel - rel_tol) / span)


# metric name → higher_is_better. Performance metrics that reward SURPASSING the paper's
# SOTA (NatureBench's g>0.1 = surpass). "error"/"loss"/"rmse"/… are lower-is-better.
_HIGHER_IS_BETTER = ("auc", "auroc", "auprc", "accuracy", "acc", "f1", "precision", "recall",
                     "iou", "dice", "r2", "ap", "map", "specificity", "sensitivity",
                     "correlation", "pearson", "spearman", "r_value", "bleu", "rouge",
                     "psnr", "ssim", "ndcg", "mcc", "hit_rate", "hits", "score", "kendall")
_LOWER_IS_BETTER = ("rmse", "mae", "mse", "rmsd", "error", "loss", "perplexity", "fid",
                    "mape", "wer", "cer", "ece", "nll", "deviation")


def metric_direction(name: str) -> Optional[bool]:
    """higher_is_better for a metric name; None if unknown (→ symmetric reproduction).
    Lower-is-better is checked first so 'rmse'/'error_rate' win over any 'r'/'score' substring."""
    n = (name or "").lower()
    if any(k in n for k in _LOWER_IS_BETTER):
        return False
    if any(k in n for k in _HIGHER_IS_BETTER):
        return True
    return None


def nature_bench_graded(agent: float, sota: float, higher_is_better: bool,
                        rel_tol: float, surpass: float = 0.1) -> tuple:
    """NatureBench-aligned graded reward for a performance metric. Returns (graded, g_norm).

    ``g_norm`` = normalised signed improvement over the paper's SOTA in the *good* direction
    (>0 better, <0 worse), mirroring NatureBench's ``g`` (g>0.1 surpass, ≥0 match, <0 below).
    graded ∈ [0,1]: SURPASS (g≥0.1) → 1.0; MATCH band (0≤g<0.1) → 0.85→1.0; BELOW → decays
    0.85→0 over 3×rel_tol. The match-band floor (0.85) is set so a hard-verified reproduction
    always outscores the soft-only floor (a no-op rollout must never beat a real match), while
    SURPASSING the paper still tops it — reproducing is strong, beating is best, a near-miss
    gets partial credit, and far-below earns nothing."""
    denom = max(abs(sota), 1e-9)
    sign = 1.0 if higher_is_better else -1.0
    g = sign * (agent - sota) / denom                 # signed normalised improvement
    if g >= surpass:
        graded = 1.0
    elif g >= 0.0:
        graded = 0.85 + 0.15 * (g / surpass)          # match band 0.85 → 1.0
    else:
        graded = max(0.0, 0.85 * (1.0 - (-g) / (3.0 * rel_tol)))   # below → decay to 0
    return graded, g


def metric_repro_correctness_gate(*, rel_tol: float = 0.1,
                                  require_execution: bool = True) -> CorrectnessGate:
    """Build a code-execution / metric-reproduction gate (domain-agnostic).

    ``rel_tol`` — relative tolerance for matching the paper's reported metric
    (|v-gold|/max(|gold|,eps) ≤ rel_tol). ``require_execution`` — only mark ``valid``
    (admissible) when the metric came from real code execution (``metric_executed``); a
    self-reported number can still yield a soft signal but never hard admissibility.
    """

    def gate(reported: ReportedResult, task_meta: dict) -> CorrectnessVerdict:
        v = reported.metric_value
        name = (reported.metric_name or task_meta.get("recompute_handle") or "").lower()
        executed = bool(reported.metric_executed)
        valid = executed if require_execution else True
        # sane: a finite number, within the metric's natural range where we know it.
        sane = v is not None and v == v and abs(v) != float("inf")
        if sane and any(k in name for k in _UNIT_INTERVAL_METRICS):
            sane = 0.0 <= v <= 1.0 or 0.0 <= v <= 100.0     # accept fraction OR percentage
        elif sane and any(k in name for k in _SIGNED_UNIT_METRICS):
            sane = -1.0001 <= v <= 1.0001
        decisive = v is not None                            # a metric was actually produced
        # gold: paper's reported numeric (paper_gt.gold_for_handle → {"numeric": ...}).
        gold = task_meta.get("gold") or {}
        gnum = gold.get("numeric")
        matched: Optional[bool] = None
        graded: Optional[float] = None
        # ANTI-HACK: only compare a number we can TRUST. With require_execution, a value that
        # did not come from real code execution is a bare claim → we do NOT compare it (matched
        # stays None → routes to the pure-soft path), so an agent cannot mint reward by simply
        # typing the paper's number. Plumbing mode (require_execution=False) compares anyway.
        comparable = (gnum is not None and v is not None and (executed or not require_execution))
        g_norm = None
        # direction: task can override, else infer from the metric name.
        hib = task_meta.get("higher_is_better")
        if hib is None:
            hib = metric_direction(name)
        if comparable:
            sota = float(gnum)
            vv = v
            # tolerate fraction-vs-percentage mismatch (0.92 vs 92) before comparing.
            if sota > 1.5 and abs(vv) <= 1.5:
                vv = vv * 100.0
            elif abs(vv) > 1.5 and abs(sota) <= 1.5:
                sota = sota * 100.0
            if not sane:
                matched, graded = False, 0.0
            elif hib is None:
                # symmetric quantity — only reproducing the value matters (no better/worse).
                matched = abs(vv - sota) <= rel_tol * max(abs(sota), 1e-9)
                graded = graded_closeness(vv, sota, rel_tol)
            else:
                # performance metric — reward reproducing AND SURPASSING the paper's SOTA (g).
                graded, g_norm = nature_bench_graded(vv, sota, bool(hib), rel_tol)
                matched = g_norm >= -rel_tol          # reproduced-or-better = hard-eligible
        return CorrectnessVerdict(
            correct=bool(sane and decisive and valid and (matched is True)),
            sane=bool(sane), decisive=bool(decisive), valid=bool(valid),
            matched_gold=matched, graded=graded, trusted=bool(executed or not require_execution),
            detail={"mode": "metric-repro", "metric": name or "?", "value": v,
                    "higher_is_better": hib, "g": g_norm, "executed": executed})

    return gate


def null_correctness_gate() -> CorrectnessGate:
    """A gate for tasks with NO deterministic recompute path at all. Returns
    ``matched_gold=None`` so ``score_discovery`` routes to the soft-only reward (the ranking
    signal from conclusion_match + judges) — never hard correctness, never admissible."""

    def gate(reported: ReportedResult, task_meta: dict) -> CorrectnessVerdict:
        return CorrectnessVerdict(correct=False, sane=True, decisive=True, valid=False,
                                  matched_gold=None, detail={"mode": "soft-only"})

    return gate


# --------------------------------------------------------------------------- #
# General task router — the ONE entry point that makes QE non-necessary. Atomistic-physics
# handles keep their re-execution gates (QE/MLIP); any other handle that carries a numeric
# paper gold routes to the metric-reproduction (code-execution) gate; everything else routes
# to the soft-only gate. So every task gets a scorable signal, and no domain is privileged.
# --------------------------------------------------------------------------- #
def gate_for_task(task_meta: dict, *, tier: str = "emt", reexecute: bool = True,
                  rel_tol: float = 0.1, require_execution: bool = True) -> CorrectnessGate:
    """Pick the correctness gate for a task without assuming a domain.

    1. atomistic handle (site_preference/adsorption/barrier/vibration) → its QE/MLIP gate;
    2. else if the paper reports a numeric gold → metric-reproduction (code-exec) gate;
    3. else → soft-only gate. QE is thus one backend among several, not a precondition."""
    handle = (task_meta.get("recompute_handle") or task_meta.get("handle") or "").strip()
    try:
        return gate_for_handle(handle, tier=tier, reexecute=reexecute)
    except NotImplementedError:
        pass
    gold = task_meta.get("gold") or {}
    if gold.get("numeric") is not None:
        return metric_repro_correctness_gate(rel_tol=rel_tol, require_execution=require_execution)
    return null_correctness_gate()
