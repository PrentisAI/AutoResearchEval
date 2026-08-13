"""Tiered paper-corpus crawler (CLAUDE.md §18, the user's bronze/silver/golden plan).

Front-end of the discovery line: given a topic query, crawl OpenAlex, grade every
hit **bronze / silver / golden** from automatic metadata signals (no hand-curated
whitelist), and emit a download manifest the existing PDF→MinerU step consumes.

The grade encodes scientific *taste*:
  • bronze — normal paper, kept for breadth of the knowledge surface.
  • silver — high-quality venue/impact; the workhorse for discovery-pattern mining.
  • golden — elite venue AND field-leading impact / prominent group; weighted up
             downstream (sampling into the miner + the SFT/RL high-score pool).

Recency-safe: young papers (null fwci/percentile) are graded on venue/institution
signals; citations only ever promote (§ adapters/openalex.py).

Outputs (examples/output/tiered_corpus/<slug>/):
  manifest.jsonl     one line per work: {work_id, doi, tier, weight, pdf_url, signals…}
  download_list.txt  pdf_url<TAB>work_id  for the OA papers (feeds the PDF fetcher)
  SUMMARY.md         tier counts + the golden picks, for manual review (breadth phase)

Run (uses the scicoder env's plain python; only stdlib networking):
  python examples/crawl_tiered_corpus.py \
      --query "CO oxidation Pt single atom catalyst" --max 150 \
      [--filters "from_publication_date:2018-01-01"] [--slug co_pt]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.openalex import (  # noqa: E402
    OpenAlexClient,
    TierConfig,
    enrich_signals,
    extract_signals,
    score_tier,
)
from dataclasses import asdict  # noqa: E402

OUTROOT = Path(__file__).resolve().parent / "output" / "tiered_corpus"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:48]


def crawl_topic(client: OpenAlexClient, query: str, *, max_works: int = 150,
                filters: Optional[str] = None, sort: str = "cited_by_count:desc",
                cfg: Optional[TierConfig] = None, skip_ids: Optional[set] = None,
                sample: bool = False, seed: int = 13, progress: bool = True) -> list[dict]:
    """Crawl + grade one topic. Returns a list of manifest rows (one per work).

    ``skip_ids`` (cross-topic de-dup): work ids already claimed by an earlier topic
    are skipped so each paper lands in exactly one topic folder. Reusable by both the
    single-topic ``main()`` and the multi-topic driver (``crawl_topic_set.py``).
    ``sample=True`` draws a representative (mixed-tier) slice instead of the high-impact
    head — what topic-set diversity wants (see ``OpenAlexClient.search_works``).
    """
    cfg = cfg or TierConfig()
    skip_ids = skip_ids if skip_ids is not None else set()
    rows: list[dict] = []
    tier_counts: Counter = Counter()
    for i, work in enumerate(client.search_works(
            query, max_works=max_works, filters=filters, sort=sort,
            sample=sample, seed=seed)):
        wid = (work.get("id") or "").rsplit("/", 1)[-1]
        if wid in skip_ids:
            continue
        sig = enrich_signals(client, work, extract_signals(work))
        tier, weight, reasons = score_tier(sig, cfg)
        tier_counts[tier] += 1
        skip_ids.add(wid)
        rows.append({
            "work_id": sig.work_id, "doi": sig.doi, "title": sig.title, "year": sig.year,
            "topic": query, "tier": tier, "weight": weight, "tier_reasons": reasons,
            "pdf_url": sig.pdf_url, "is_oa": sig.is_oa,
            "signals": asdict(sig),
        })
        if progress and (i + 1) % 25 == 0:
            print(f"    ..{i+1} works  {dict(tier_counts)}")
    return rows


def write_topic_outputs(outdir: Path, query: str, sort: str, filters: Optional[str],
                        rows: list[dict], cfg: TierConfig) -> int:
    """Write manifest.jsonl + download_list.txt + SUMMARY.md for one topic folder.
    Returns the number of OA pdf links written."""
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""))
    dl = [f"{r['pdf_url']}\t{r['work_id']}" for r in rows if r["pdf_url"]]
    (outdir / "download_list.txt").write_text("\n".join(dl) + ("\n" if dl else ""))
    _write_summary(outdir, query, sort, filters, rows, cfg)
    return len(dl)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="OpenAlex search query (topic)")
    ap.add_argument("--max", type=int, default=150, help="max works to crawl")
    ap.add_argument("--filters", default=None,
                    help="OpenAlex filter string, e.g. 'from_publication_date:2018-01-01'")
    ap.add_argument("--sort", default="cited_by_count:desc",
                    help="OpenAlex sort (default brings the influential papers first)")
    ap.add_argument("--slug", default=None, help="output subdir name (default: from query)")
    ap.add_argument("--mailto", default=None, help="contact for OpenAlex polite pool")
    args = ap.parse_args()

    slug = args.slug or _slug(args.query)
    outdir = OUTROOT / slug
    client = OpenAlexClient(**({"mailto": args.mailto} if args.mailto else {}))
    cfg = TierConfig()

    print(f"[crawl] query={args.query!r} max={args.max} filters={args.filters}")
    t0 = time.time()
    rows = crawl_topic(client, args.query, max_works=args.max, filters=args.filters,
                       sort=args.sort, cfg=cfg)
    n_pdf = write_topic_outputs(outdir, args.query, args.sort, args.filters, rows, cfg)
    counts = Counter(r["tier"] for r in rows)
    print(f"[done] {len(rows)} works in {time.time()-t0:.0f}s  tiers={dict(counts)}  oa_pdfs={n_pdf}")
    print(f"[out] {outdir}/  (manifest.jsonl, download_list.txt, SUMMARY.md)")


def _write_summary(outdir: Path, query: str, sort: str, filters: Optional[str],
                   rows: list[dict], cfg: TierConfig) -> None:
    n = len(rows) or 1
    counts = Counter(r["tier"] for r in rows)
    L = [f"# Tiered corpus — {query}", "",
         f"- works crawled: **{len(rows)}**",
         f"- filters: `{filters or 'none'}`  · sort: `{sort}`",
         "",
         "## Tier distribution", "",
         "| tier | weight | count | share |",
         "| --- | --- | --- | --- |"]
    for t in ("golden", "silver", "bronze"):
        c = counts.get(t, 0)
        L.append(f"| {t} | {cfg.weight(t):g} | {c} | {100*c/n:.0f}% |")
    L += ["", "## Golden picks (high taste — weighted up downstream)", ""]
    golden = [r for r in rows if r["tier"] == "golden"]
    golden.sort(key=lambda r: (r["signals"].get("fwci") or 0), reverse=True)
    for r in golden[:40]:
        why = "; ".join(r["tier_reasons"])
        L.append(f"- **{(r['title'] or '')[:80]}** ({r['year']}) — {why}")
    if not golden:
        L.append("_(none reached the golden gate for this query)_")
    (outdir / "SUMMARY.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
