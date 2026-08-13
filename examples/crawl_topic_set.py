"""Multi-topic tiered crawler — diversity over a *set* of topics, foldered by topic
(CLAUDE.md §18.9, the user's "提升话题多样性 + 按 topic 分子文件夹").

The single-topic crawler (``crawl_tiered_corpus.py``) deepens ONE query; this driver
widens the corpus across a curated set of distinct computational-catalysis / materials
topics so the discovery miner sees a broad knowledge surface (§18 breadth phase), not
just CO/Pt. Each topic is crawled + bronze/silver/golden graded into its OWN folder.

Layout (examples/output/tiered_corpus/<set_name>/):
  <topic_slug>/manifest.jsonl      one folder per topic (self-contained: manifest +
                  download_list.txt   download list + SUMMARY.md, same shape as the
                  SUMMARY.md          single-topic crawler emits)
  INDEX.md        cross-topic rollup: per-topic tier counts + grand totals
  topics.json     machine-readable: {set_name, topics:[{query, slug, counts, n_pdf}]}

Cross-topic de-dup: a work returned by two topics is kept only under the FIRST topic
that claims it (so the per-topic folders partition the corpus, no double counting).

The default DIVERSE_TOPICS set spans heterogeneous/single-atom/electro/photo catalysis,
MLIPs, electrolytes, MOFs, 2D materials, perovskites, alloys — deliberately orthogonal
so the union maximises topical coverage. Override with --topics or a --topics-file.

Run:
  python examples/crawl_topic_set.py [--per-topic 60] [--set-name diverse_v1] \
      [--sort relevance_score:desc] [--filters from_publication_date:2018-01-01] \
      [--topics-file my_topics.txt]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.openalex import OpenAlexClient, TierConfig  # noqa: E402
from examples.crawl_tiered_corpus import OUTROOT, _slug, crawl_topic, write_topic_outputs  # noqa: E402

# A deliberately orthogonal spread of computational catalysis / materials topics.
# Each is a focused query (so OpenAlex relevance is tight) but the SET is broad.
DIVERSE_TOPICS = [
    "CO oxidation Pt single atom catalyst",
    "oxygen evolution reaction electrocatalyst DFT",
    "nitrogen reduction reaction single atom catalyst",
    "CO2 reduction copper electrocatalysis mechanism",
    "hydrogen evolution reaction transition metal dichalcogenide",
    "machine learning interatomic potential catalysis",
    "zeolite methanol to olefins acid site",
    "metal organic framework gas adsorption DFT",
    "perovskite oxide surface oxygen vacancy catalysis",
    "high entropy alloy catalyst descriptor",
    "photocatalytic water splitting TiO2 band structure",
    "ammonia synthesis catalyst nitrogen activation",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-topic", type=int, default=60, help="max works per topic")
    ap.add_argument("--set-name", default="diverse_v1", help="output set folder name")
    ap.add_argument("--sort", default="relevance_score:desc",
                    help="OpenAlex sort (relevance keeps each topic on-subject; ignored if --sample)")
    ap.add_argument("--sample", action="store_true",
                    help="draw a REPRESENTATIVE (mixed-tier) slice per topic instead of the "
                         "high-impact head — better breadth/diversity for the discovery miner")
    ap.add_argument("--seed", type=int, default=13, help="random seed for --sample (reproducible)")
    ap.add_argument("--filters", default=None, help="OpenAlex filter string applied to every topic")
    ap.add_argument("--topics", nargs="*", default=None, help="override the topic list inline")
    ap.add_argument("--topics-file", default=None, help="file with one topic query per line")
    ap.add_argument("--mailto", default=None)
    ap.add_argument("--sleep", type=float, default=0.12,
                    help="base delay between OpenAlex requests (raise to avoid 429 throttling)")
    ap.add_argument("--max-retries", type=int, default=4, help="OpenAlex retry attempts (exp backoff)")
    ap.add_argument("--resume", action="store_true",
                    help="skip topics whose folder already has a manifest.jsonl (resume after a 429)")
    args = ap.parse_args()

    topics = args.topics
    if args.topics_file:
        topics = [ln.strip() for ln in Path(args.topics_file).read_text().splitlines()
                  if ln.strip() and not ln.startswith("#")]
    topics = topics or DIVERSE_TOPICS

    setdir = OUTROOT / args.set_name
    setdir.mkdir(parents=True, exist_ok=True)
    client_kw = {"sleep": args.sleep, "max_retries": args.max_retries}
    if args.mailto:
        client_kw["mailto"] = args.mailto
    client = OpenAlexClient(**client_kw)
    cfg = TierConfig()

    mode = f"sample(seed={args.seed})" if args.sample else f"sort={args.sort}"
    print(f"[topic-set] {len(topics)} topics · per_topic={args.per_topic} · {mode}")
    t0 = time.time()
    seen: set = set()                       # cross-topic de-dup (first topic wins)
    summary: list[dict] = []
    grand: Counter = Counter()
    # --resume: re-seed `seen` from already-written manifests so cross-topic de-dup and
    # the grand totals survive a restart after a 429, and finished topics are skipped.
    done_slugs: set = set()
    if args.resume:
        for man in setdir.glob("*/manifest.jsonl"):
            # NOT .splitlines(): it also breaks on  /\r/etc that can legitimately
            # appear inside a JSON string value (paper titles/abstracts), which would
            # slice one JSONL record into two invalid fragments. JSONL is \n-delimited.
            rows_prev = [json.loads(ln) for ln in man.read_text().split("\n") if ln.strip()]
            if not rows_prev:
                continue
            slug_prev = man.parent.name
            done_slugs.add(slug_prev)
            for r in rows_prev:
                seen.add(r.get("work_id"))
            counts = Counter(r["tier"] for r in rows_prev)
            grand.update(counts)
            summary.append({"query": rows_prev[0].get("topic", slug_prev), "slug": slug_prev,
                            "n_works": len(rows_prev), "counts": dict(counts),
                            "n_pdf": sum(1 for r in rows_prev if r.get("is_oa"))})
        print(f"[resume] {len(done_slugs)} topics already done, {len(seen)} works seen")
    for ti, query in enumerate(topics):
        slug = _slug(query)
        if slug in done_slugs:
            print(f"[{ti+1}/{len(topics)}] {query!r} -> {slug}/  (resume: skip)")
            continue
        print(f"[{ti+1}/{len(topics)}] {query!r} -> {slug}/")
        rows = crawl_topic(client, query, max_works=args.per_topic, filters=args.filters,
                           sort=args.sort, cfg=cfg, skip_ids=seen,
                           sample=args.sample, seed=args.seed)
        n_pdf = write_topic_outputs(setdir / slug, query, args.sort, args.filters, rows, cfg)
        counts = Counter(r["tier"] for r in rows)
        grand.update(counts)
        summary.append({"query": query, "slug": slug, "n_works": len(rows),
                        "counts": dict(counts), "n_pdf": n_pdf})
        print(f"      {len(rows)} works  tiers={dict(counts)}  oa_pdfs={n_pdf}")

    _write_index(setdir, args, summary, grand, cfg, n_unique=len(seen), mode=mode)
    (setdir / "topics.json").write_text(json.dumps(
        {"set_name": args.set_name, "mode": mode, "sort": args.sort,
         "sample": args.sample, "seed": args.seed, "filters": args.filters,
         "topics": summary}, ensure_ascii=False, indent=2))
    n_total = sum(s["n_works"] for s in summary)
    print(f"\n[done] {n_total} works across {len(topics)} topics in {time.time()-t0:.0f}s "
          f"(unique={len(seen)})  grand tiers={dict(grand)}")
    print(f"[out] {setdir}/  (per-topic folders + INDEX.md + topics.json)")


def _write_index(setdir: Path, args, summary: list[dict], grand: Counter,
                 cfg: TierConfig, n_unique: int, mode: str = "") -> None:
    n_total = sum(s["n_works"] for s in summary) or 1
    L = [f"# Tiered corpus set — {args.set_name}", "",
         f"- topics: **{len(summary)}**  ·  unique works: **{n_unique}**",
         f"- per-topic cap: {args.per_topic}  ·  mode: `{mode}`  ·  filters: `{args.filters or 'none'}`",
         "",
         "## Grand tier distribution", "",
         "| tier | weight | count | share |",
         "| --- | --- | --- | --- |"]
    for t in ("golden", "silver", "bronze"):
        c = grand.get(t, 0)
        L.append(f"| {t} | {cfg.weight(t):g} | {c} | {100*c/n_total:.0f}% |")
    L += ["", "## Per-topic breakdown", "",
          "| topic | folder | works | 🥇 | 🥈 | 🥉 | OA pdf |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for s in summary:
        c = s["counts"]
        L.append(f"| {s['query']} | `{s['slug']}/` | {s['n_works']} | "
                 f"{c.get('golden',0)} | {c.get('silver',0)} | {c.get('bronze',0)} | {s['n_pdf']} |")
    (setdir / "INDEX.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
