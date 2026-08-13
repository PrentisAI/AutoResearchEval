"""Reconstruction techniques (CLAUDE.md §5) — narrate "result" back to "process".

All take a pluggable ``base.LLMClient`` and respect the keep-if-correct (§1.2)
and provenance (§11) invariants. Pipeline position: IR → **reconstruct** →
verify → filter → export.

  thought_completion  — fill implicit thoughts (PC Agent-E)
  humpback_backtranslate — infer task/goal, self-curate (keep score==5)
  star_rationalize    — hint with answer, KEEP only if re-execution agrees
  hindsight_relabel   — failed-at-A → positive-at-B (AgentHER)
"""

from reconstruct import (
    hindsight_relabel,
    humpback_backtranslate,
    star_rationalize,
    thought_completion,
)
from reconstruct.base import LLMClient, ReconstructLog, gated_reconstruct, stamp_provenance

__all__ = [
    "thought_completion",
    "humpback_backtranslate",
    "star_rationalize",
    "hindsight_relabel",
    "LLMClient",
    "ReconstructLog",
    "gated_reconstruct",
    "stamp_provenance",
]
