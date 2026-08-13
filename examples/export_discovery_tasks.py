"""Export all mined discovery patterns into a unified, agent-executable task list.

A ``DiscoveryPattern`` (premise→tension→move + key_claims) is the SOFT artifact; a
``task`` is what an agent actually executes: a framed goal (premise + tension, NO answer
leaked), the recompute_handle that adjudicates it, and — for scoring — the paper's own
conclusion (soft conclusion_match target, §discovery_verifier). This collapses the four
mined corpora into one JSONL the batch rollout + verifier consume.

Each task row:
  task_id, title, premise, tension, conclusion (paper's, soft target), novelty_move,
  handle (recompute anchor or ""), tier (bronze/silver/golden), set, artifact_uri,
  qe_ready (handle ∈ adsorption-recipe scope), unverified (no re-executable correctness).

Two reward dimensions follow from `qe_ready` (§0.9 two-data-classes):
  • qe_ready=True  → hard correctness gate (live QE recompute) × soft discovery dims.
  • qe_ready=False → unverified: soft-only (significance/novelty/conclusion_match),
                     never enters the admissible hard-data pool.

Run:
  python examples/export_discovery_tasks.py [--out examples/output/discovery_tasks.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/data/from_139/xmyu/scicoder")
sys.path.insert(0, str(REPO))

from reconstruct.discovery_pattern import DiscoveryPattern  # noqa: E402
from reconstruct.paper_gt import extract_paper_gt, gold_for_handle  # noqa: E402
from harness.discovery_verifier import metric_direction  # noqa: E402

# The output contract the agent's FINAL block must satisfy so a code-execution verifier can
# score it (mirrors the NatureBench submit-and-score loop). Kept on every task for the RL harness.
_SUBMISSION_CONTRACT = ("Write and RUN code to produce the metric on the provided data, then "
                        "emit a FINAL block with: metric_name: <name>, metric_value: <number>, "
                        "executed: true (only if the number came from real code execution), "
                        "conclusion: <one sentence>.")

# Pattern corpora to fold in (dir stem → set label).
PATTERN_DIRS = {
    "discovery_patterns": "co_pt",
    "discovery_patterns_diverse": "diverse_sampled",
    "discovery_patterns_diverse_1k": "diverse_1k",
    "discovery_patterns_diverse_2k": "diverse_2k",
    "discovery_patterns_naturebench_dist_v1": "naturebench_dist_v1",
    "discovery_patterns_naturebench_ml_v1": "naturebench_ml_v1",
    "discovery_patterns_naturebench_ml_v2": "naturebench_ml_v2",
    "discovery_patterns_subset3": "subset3",
}
OUTROOT = REPO / "examples/output"
_SET_DIR = {  # set label → output dir stem (reverse of PATTERN_DIRS, for patterns_dir field)
    "co_pt": "discovery_patterns", "diverse_sampled": "discovery_patterns_diverse",
    "diverse_1k": "discovery_patterns_diverse_1k", "diverse_2k": "discovery_patterns_diverse_2k",
    "naturebench_dist_v1": "discovery_patterns_naturebench_dist_v1",
    "naturebench_ml_v1": "discovery_patterns_naturebench_ml_v1",
    "naturebench_ml_v2": "discovery_patterns_naturebench_ml_v2",
    "subset3": "discovery_patterns_subset3",
}

# Handles the QE adsorption recipe can serve as a live correctness gate (§18.4 scope).
_ADSORPTION_HANDLES = {"co_adsorption_energy", "site_preference", "coverage_shift"}


def _load(path: Path) -> DiscoveryPattern:
    return DiscoveryPattern.from_dict(json.loads(path.read_text()))


def _task_row(p: DiscoveryPattern, set_label: str, raw: dict) -> dict:
    # GENERAL: pick the paper's primary observable WITHOUT preferring adsorption. Prefer one
    # that actually has a numeric gold (a hard-comparable anchor), else the first observable.
    handles = p.recompute_handles()            # open vocabulary, computable-first
    paper_gt = extract_paper_gt(raw)
    with_gold = [g["handle"] for g in paper_gt if g.get("numeric") is not None]
    handle = (with_gold[0] if with_gold else (handles[0] if handles else ""))
    # qe_ready = this observable is re-executable by an EXISTING recompute backend today
    # (only the adsorption family so far; others are soft until their backend lands).
    qe_ready = bool(handle) and (handle in _ADSORPTION_HANDLES or "adsorption" in handle.lower())
    # the paper's CALCULATION GT (numbers it reported) — reward scores against the PAPER.
    gold = gold_for_handle(paper_gt, handle) if handle else None
    # code-out objective (NatureBench alignment): if the paper reports a numeric SOTA, the task
    # is to REPRODUCE-OR-SURPASS it; direction says which way "better" is. Else reach the finding.
    sota = (gold or {}).get("numeric")
    higher_is_better = metric_direction(handle) if handle else None
    if sota is not None:
        objective = "reproduce_or_surpass" if higher_is_better is not None else "reproduce"
    else:
        objective = "reach_conclusion"
    return {
        "task_id": p.paper_id,
        "set": set_label,
        "patterns_dir": f"examples/output/{_SET_DIR[set_label]}/patterns",  # for unambiguous rollout
        "title": p.title,
        "premise": p.premise_consensus,        # framing only — agent sees this
        "tension": p.tension,                  # the discovery seed (no answer leaked)
        "conclusion": p.conclusion,            # paper's finding → soft conclusion_match target (GT)
        "novelty_move": p.novelty_move,
        "handle": handle,                      # recompute anchor (may be "")
        "all_handles": handles,
        "paper_gt": paper_gt,                  # ALL recomputable claims' numeric GT (calc GT)
        "gold": gold,                          # GT for the picked handle → reward anchor
        # --- code-out objective + submission contract (for the code-execution RL harness) ---
        "objective": objective,                # reproduce_or_surpass | reproduce | reach_conclusion
        "metric": handle,                      # the metric name the agent must produce
        "sota": sota,                          # the paper's reported number = the target to beat
        "higher_is_better": higher_is_better,  # direction of "better" (None = symmetric/none)
        "submission_contract": _SUBMISSION_CONTRACT,
        "qe_ready": qe_ready,                  # hard correctness gate available
        "unverified": not qe_ready,            # soft-only scoring otherwise
        "tier": raw.get("tier", ""),           # bronze/silver/golden (from tier backfill)
        "tier_weight": raw.get("tier_weight"),
        "artifact_uri": p.artifact_uri,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTROOT / "discovery_tasks.jsonl"))
    ap.add_argument("--qe-ready-only", action="store_true",
                    help="emit only tasks with a live correctness gate (the hard-data subset)")
    args = ap.parse_args()

    rows: list[dict] = []
    per_set = Counter()
    for stem, label in PATTERN_DIRS.items():
        pdir = OUTROOT / stem / "patterns"
        if not pdir.is_dir():
            continue
        for f in sorted(pdir.glob("*.json")):
            raw = json.loads(f.read_text())
            row = _task_row(_load(f), label, raw)
            if args.qe_ready_only and not row["qe_ready"]:
                continue
            rows.append(row)
            per_set[label] += 1

    out = Path(args.out)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    qe_ready = sum(1 for r in rows if r["qe_ready"])
    has_handle = sum(1 for r in rows if r["handle"])
    moves = Counter(r["novelty_move"] for r in rows)
    tiers = Counter(r["tier"] or "untiered" for r in rows)
    print(f"[export] {len(rows)} tasks → {out}")
    print(f"  per set: {dict(per_set)}")
    print(f"  qe_ready (hard gate): {qe_ready}  ·  any recompute_handle: {has_handle}  "
          f"·  unverified (soft-only): {len(rows) - qe_ready}")
    print(f"  novelty_move: {dict(moves.most_common())}")
    print(f"  tier: {dict(tiers.most_common())}")


if __name__ == "__main__":
    main()
