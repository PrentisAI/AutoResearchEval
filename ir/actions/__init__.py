"""SciData Engine action space — the 26 normalized, verifier-bound actions.

This is the **foundation** for trajectory generation: `id` here == `Action.name`
in `ir/trajectory.py`. The vocabulary was INDUCED from two corpora (CLAUDE.md
§0.9 / §16.1: vocab from data, not top-down) — 143 GitHub science agents +
real provenance diffs (mc2d/ACWF restart chains) — and every action pins a
SciEngine-OWNED deterministic verifier (external tools give the action space,
verification is always ours).

Source of truth: `registry.json` (machine) + `SCHEMA.md` (human).
Derivation/verification harness lives in `research/agent_survey/`
(`discover/extract/analyze.py`, `verify_actions.py`).

Usage:
    from ir.actions import REGISTRY, ACTIONS, ACTION_IDS, get
    a = get("run_dft")                 # one action dict, or None
    {x["id"] for x in ACTIONS}         # == ACTION_IDS
"""
from __future__ import annotations

import json
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent / "registry.json"

REGISTRY: dict = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
ACTIONS: list[dict] = REGISTRY["actions"]
META: dict = REGISTRY.get("_meta", {})
BY_ID: dict[str, dict] = {a["id"]: a for a in ACTIONS}
ACTION_IDS: set[str] = set(BY_ID)

# ── discovery (reasoning) action space — the layer above the execution actions ──
# Models the incremental scientific-discovery arc (CLAUDE.md §0.9 discovery→RL,
# §18). run_calculation grounds into the execution actions above; the terminal
# reward grounds into the recompute_handle. See discovery_registry.json + the
# loader in reconstruct/discovery_trajectory.py.
DISCOVERY_REGISTRY_PATH = Path(__file__).resolve().parent / "discovery_registry.json"
DISCOVERY_REGISTRY: dict = json.loads(DISCOVERY_REGISTRY_PATH.read_text(encoding="utf-8"))
DISCOVERY_ACTIONS: list[dict] = DISCOVERY_REGISTRY["moves"]
DISCOVERY_BY_ID: dict[str, dict] = {a["id"]: a for a in DISCOVERY_ACTIONS}
DISCOVERY_ACTION_IDS: set[str] = set(DISCOVERY_BY_ID)


def get(action_id: str) -> dict | None:
    """Return the action definition (execution or discovery) for `action_id`, or None."""
    return BY_ID.get(action_id) or DISCOVERY_BY_ID.get(action_id)


__all__ = [
    "REGISTRY", "ACTIONS", "META", "BY_ID", "ACTION_IDS", "REGISTRY_PATH",
    "DISCOVERY_REGISTRY", "DISCOVERY_ACTIONS", "DISCOVERY_BY_ID",
    "DISCOVERY_ACTION_IDS", "DISCOVERY_REGISTRY_PATH", "get",
]
