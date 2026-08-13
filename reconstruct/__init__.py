"""Reconstruction techniques (CLAUDE.md §5) — narrate "result" back to "process".

All take a pluggable ``base.LLMClient`` and respect the keep-if-correct (§1.2)
and provenance (§11) invariants. Pipeline position: IR → **reconstruct** →
verify → filter → export.

  discovery_pattern    — paper → premise/tension/…/conclusion + key claims
  discovery_moves      — each abstract discovery move as a prompted function
  discovery_trajectory — a mined pattern → an agent trajectory
  paper_gt             — the paper's reported numbers → comparable gold
  tool_lift            — raw provenance steps → clean agent tool calls
  llm_openrouter       — the OpenRouter teacher client

Submodules are imported on demand (``from reconstruct.discovery_pattern import …``)
so the package stays importable without the optional per-source dependencies.
"""

from reconstruct.base import LLMClient, ReconstructLog, gated_reconstruct, stamp_provenance

__all__ = [
    "LLMClient",
    "ReconstructLog",
    "gated_reconstruct",
    "stamp_provenance",
]
