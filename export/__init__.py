"""Export (CLAUDE.md §7) — IR → SFT(ReAct) / RLVR(verl·OpenRLHF·NeMo-RL).

This is the only place the engine emits its product format; training lives
downstream (out of scope). Export refuses non-admissible trajectories by default
(§1.1). Observation/tool tokens are masked out of loss via the shared
message-level mask convention (``loss_mask`` / ``response_mask`` / ``loss_multiplier``).
"""

from export import to_rlvr, to_sft_react

__all__ = ["to_sft_react", "to_rlvr"]
