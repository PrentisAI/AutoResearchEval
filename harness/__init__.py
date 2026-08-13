"""Rollout harness (CLAUDE.md §8, §9 stage-4) — MLGym-style scale-up. Optional/later.

Out of scope for the initial build (§9: quality-first cold start needs only
hundreds–low-thousands of verified trajectories; scale via RL rollout, not static
piling, §1.6). This package will host an MLGym-style harness (§3) to roll out an
agent in the verify-able environments and score the resulting trajectories with
``verify/`` — feeding new IR trajectories back into the same pipeline.

Intentionally left as an interface marker until stages 0–3 clear their Go/No-Go
gates (§9).
"""

__all__: list[str] = []
