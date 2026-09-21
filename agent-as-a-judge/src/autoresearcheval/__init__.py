"""AutoResearchEval — process-level failure diagnosis for autonomous research agents.

Two stages, two calls:

    from autoresearcheval import generate_analysis, label_arft

    analysis = generate_analysis(trajectory, retrieval_note=..., gold_note=...)
    result   = label_arft(analysis["analysis"], api_key="...")

    for code in result["failure_modes"]:
        print(code, pattern_info(code)["name"])

Stage 1 reads a trajectory's artifacts — execution log, delivered files, the scorer's
own source — and writes a structured six-stage critique. Stage 2 maps that critique onto
**ARFT**, the AutoResearch Failure Taxonomy: 45 patterns across six lifecycle stages plus
a cross-cutting layer, rolling up to four root-cause pillars.

Judging against artifacts rather than the transcript alone is the point: on 50 stratified
trajectories against three-expert annotation this reaches kappa 0.75 (pattern) and 0.83
(root cause), versus 0.53 / 0.62 for a single-call judge shown only the transcript.
Almost all of the difference is recall.

The batch CLIs that produced the paper's corpus are installed alongside: ``aaj-generate``
(Stage 1 over a run directory), ``aaj-classify`` (Stage 2 over a corpus), plus
``aaj-aggregate``, ``aaj-status`` and ``aaj-verify``.
"""
from .api import all_patterns, generate_analysis, label_arft, pattern_info

__version__ = "0.1.0"
__all__ = [
    "generate_analysis",
    "label_arft",
    "pattern_info",
    "all_patterns",
    "__version__",
]
