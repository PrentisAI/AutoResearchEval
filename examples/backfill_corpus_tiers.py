"""Backfill bronze/silver/golden tiers onto the existing discovery-pattern corpus
(CLAUDE.md §18 + the user's tiering plan).

The 93 CO/Pt patterns in ``examples/output/discovery_patterns/patterns/`` were
mined before tiering existed; their filenames are DOIs (``10.1021_ACSCATAL.6B00476``
→ ``10.1021/ACSCATAL.6B00476``). This script resolves each DOI on OpenAlex, grades
it (``adapters/openalex.score_tier``), and:

  • writes a side-car ``tiers.json`` (doi → {tier, weight, reasons, signals});
  • augments each pattern record in place with ``tier`` / ``tier_weight`` /
    ``tier_reasons`` so the downstream miner + SFT/RL export can weight by taste
    (golden up, bronze for breadth) without re-resolving metadata.

Idempotent + resumable: re-running reuses ``tiers.json`` unless ``--force``.

Run:
  python examples/backfill_corpus_tiers.py [--patterns-dir DIR] [--force] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.openalex import (  # noqa: E402
    OpenAlexClient,
    TierConfig,
    enrich_signals,
    extract_signals,
    score_tier,
)

DEFAULT_DIR = Path(__file__).resolve().parent / "output" / "discovery_patterns" / "patterns"


_WORK_ID_RE = re.compile(r"^W\d+$")


def _filename_to_doi(stem: str) -> str:
    """``10.1021_ACSCATAL.6B00476`` → ``10.1021/ACSCATAL.6B00476`` (first '_' after the
    registrant prefix is the prefix/suffix separator; the rest of the DOI keeps its chars)."""
    # DOIs are 10.<registrant>/<suffix>; the corpus encodes the single '/' as the first '_'.
    return stem.replace("_", "/", 1)


def _resolve_work(client: OpenAlexClient, stem: str):
    """Resolve a pattern filename to its OpenAlex work. Filenames are either an OpenAlex
    work id (``W7143594206``) or a DOI with '/' encoded as the first '_'. Returns
    ``(work_or_None, key)`` where ``key`` is the cache key (work id or DOI)."""
    if _WORK_ID_RE.match(stem):
        return client.work_by_id(stem), stem
    doi = _filename_to_doi(stem)
    return client.work_by_doi(doi), doi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--force", action="store_true", help="re-resolve DOIs already in tiers.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mailto", default=None)
    ap.add_argument("--no-augment", action="store_true",
                    help="write tiers.json only; do not edit the pattern records")
    args = ap.parse_args()

    pdir = Path(args.patterns_dir)
    files = sorted(pdir.glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    tiers_path = pdir.parent / "tiers.json"
    cache: dict = json.loads(tiers_path.read_text()) if tiers_path.exists() else {}

    client = OpenAlexClient(**({"mailto": args.mailto} if args.mailto else {}))
    cfg = TierConfig()
    counts: Counter = Counter()
    n_resolved, n_unresolved = 0, 0
    t0 = time.time()

    for i, f in enumerate(files):
        # peek the cache under either possible key (work id or DOI) before fetching
        cache_key = f.stem if _WORK_ID_RE.match(f.stem) else _filename_to_doi(f.stem)
        if cache_key in cache and not args.force:
            rec = cache[cache_key]
        else:
            work, cache_key = _resolve_work(client, f.stem)
            if work is None:
                n_unresolved += 1
                print(f"  [{i+1}] {cache_key[:48]:48s} UNRESOLVED -> bronze fallback")
                rec = {"tier": "bronze", "weight": cfg.weight("bronze"),
                       "reasons": ["not found on OpenAlex"], "signals": {}, "resolved": False}
            else:
                sig = enrich_signals(client, work, extract_signals(work))
                tier, weight, reasons = score_tier(sig, cfg)
                rec = {"tier": tier, "weight": weight, "reasons": reasons,
                       "signals": asdict(sig), "resolved": True}
                n_resolved += 1
                print(f"  [{i+1}] {cache_key[:48]:48s} {tier:7s} w={weight:g}  "
                      f"({'; '.join(reasons) or 'baseline'})"[:120])
            cache[cache_key] = rec
        counts[rec["tier"]] += 1

        if not args.no_augment:
            pat = json.loads(f.read_text())
            pat["tier"] = rec["tier"]
            pat["tier_weight"] = rec["weight"]
            pat["tier_reasons"] = rec["reasons"]
            f.write_text(json.dumps(pat, ensure_ascii=False, indent=2))

    tiers_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    print(f"\n[done] {len(files)} patterns in {time.time()-t0:.0f}s  "
          f"(resolved {n_resolved}, unresolved {n_unresolved})")
    print(f"[tiers] {dict(counts)}")
    print(f"[out] {tiers_path}" + ("" if args.no_augment else "  + augmented pattern records"))


if __name__ == "__main__":
    main()
