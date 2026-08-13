"""Fetch the open-access PDFs for a tiered corpus set, into each topic's folder
(CLAUDE.md §18.9 — follows the metadata crawl; the user asked to actually pull the
原文 PDFs after grading).

Reads each topic's ``manifest.jsonl`` (the authoritative record: work_id + pdf_url +
is_oa + tier) and downloads the OA PDFs to ``<topic_slug>/pdfs/<work_id>.pdf`` — so the
PDFs sit next to the grading, ready for the existing PDF→MinerU parse step.

Honest about what "OA pdf_url" actually is:
  • OpenAlex ``best_oa_location.pdf_url`` sometimes points at a graphical-abstract image
    (.jpg/.png) or a landing page, not the article PDF. We therefore (a) skip obvious
    image extensions up front, and (b) verify the downloaded bytes start with ``%PDF``
    — anything else is discarded and logged, never saved as a .pdf.
  • Many publisher OA links 403/redirect to paywalls. Those are recorded as failures,
    not retried forever. We only attempt works flagged ``is_oa``.

Polite + resumable: a realistic User-Agent, per-request timeout, small inter-request
sleep, bounded retries, and skip-if-already-downloaded. Failures land in
``<topic_slug>/pdfs/_failures.jsonl`` with the reason, so a rerun only retries gaps.

Fallback chain (each stage only adds candidates, never replaces earlier ones — all are
tried in order until one downloads a real ``%PDF``):
  manifest best_oa -> OpenAlex locations (repo hosts first) -> Semantic Scholar
  (openAccessPdf + arXiv id, when OpenAlex missed it) -> CORE.ac.uk (harvests
  repositories worldwide; needs a free key, see --core) -> Unpaywall -> PMC
  (Europe PMC render endpoint, inserted first if any candidate carries a PMCID).
Repository-hosted copies (arXiv/ChemRxiv/HAL/PMC) rarely block bots; publisher
OA links (Wiley/ACS/Springer/Elsevier) are the ones that 403 or serve a landing page.

Run:
  python examples/fetch_corpus_pdfs.py --set diverse_sampled          # all topics
  python examples/fetch_corpus_pdfs.py --set diverse_sampled --topic co_oxidation_pt_single_atom_catalyst
  python examples/fetch_corpus_pdfs.py --set diverse_sampled --tiers silver golden   # subset
  python examples/fetch_corpus_pdfs.py --set diverse_sampled --unpaywall --s2 --core   # full fallback chain
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.openalex import OpenAlexClient  # noqa: E402
from examples.crawl_tiered_corpus import OUTROOT  # noqa: E402

# Obvious non-article assets OpenAlex sometimes returns as "pdf_url".
_SKIP_EXT = (".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".svg", ".bmp")
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/124.0 Safari/537.36 SciDataEngine/1.0 (mailto:scidata-engine@inceptlabs.ai)")
_UNPAYWALL_EMAIL = "scidata-engine@inceptlabs.ai"
_CORE_API_KEY = os.environ.get("CORE_API_KEY")   # free key: core.ac.uk/services/api
_S2_API_KEY = os.environ.get("S2_API_KEY")       # optional: api.semanticscholar.org/api-key
_core_warned = False


def _unpaywall_pdf_urls(doi: str, timeout: float) -> list[str]:
    """Ordered PDF-url candidates from Unpaywall for one DOI, **repository (non-publisher)
    hosts first** — arXiv/ChemRxiv/HAL/PMC rarely block bots, publisher OA usually does.
    Unpaywall resolves OA copies OpenAlex's best_oa misses (the industry-standard OA index).
    Empty list on any error (Unpaywall is an enrichment, never a hard dependency)."""
    if not doi:
        return []
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email={_UNPAYWALL_EMAIL}"
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
    except Exception:                                       # noqa: BLE001
        return []
    locs = d.get("oa_locations") or []
    repo = [l.get("url_for_pdf") for l in locs if l.get("host_type") == "repository"]
    pub = [l.get("url_for_pdf") for l in locs if l.get("host_type") != "repository"]
    return [u for u in (repo + pub) if u]


def _semantic_scholar_pdf_urls(doi: str, timeout: float) -> list[str]:
    """Semantic Scholar's own OA resolution (``openAccessPdf.url``) plus — the real win —
    its ``externalIds.ArXiv`` mapping: when OpenAlex/Unpaywall didn't surface an arXiv
    location (it happens), S2 often still knows the paper is on arXiv, letting us build
    a direct ``arxiv.org/pdf/<id>`` link (arXiv never blocks bots, always a real PDF).
    No API key required for light use; sends one if S2_API_KEY is set. Never hard-fails."""
    if not doi:
        return []
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=openAccessPdf,externalIds"
        headers = {"User-Agent": _UA}
        if _S2_API_KEY:
            headers["x-api-key"] = _S2_API_KEY
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
    except Exception:                                       # noqa: BLE001
        return []
    out = []
    arxiv_id = (d.get("externalIds") or {}).get("ArXiv")
    if arxiv_id:
        out.append(f"https://arxiv.org/pdf/{arxiv_id}.pdf")
    pdf = (d.get("openAccessPdf") or {}).get("url")
    if pdf:
        out.append(pdf)
    return out


def _core_pdf_urls(doi: str, timeout: float) -> list[str]:
    """CORE.ac.uk aggregates full text harvested from repositories worldwide, so it
    sometimes has a copy OpenAlex/Unpaywall/S2 all missed. Requires a free API key
    (core.ac.uk/services/api) via $CORE_API_KEY — silently returns [] without one
    (warns once so it's clear the fallback chain has an unused stage, not a bug)."""
    global _core_warned
    if not doi:
        return []
    if not _CORE_API_KEY:
        if not _core_warned:
            print("  [core] CORE_API_KEY not set — skipping CORE.ac.uk fallback "
                  "(get a free key at core.ac.uk/services/api)")
            _core_warned = True
        return []
    try:
        url = "https://api.core.ac.uk/v3/search/works"
        body = json.dumps({"q": f'doi:"{doi}"', "limit": 3}).encode()
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "User-Agent": _UA, "Content-Type": "application/json",
            "Authorization": f"Bearer {_CORE_API_KEY}",
        })
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
    except Exception:                                       # noqa: BLE001
        return []
    out = []
    for res in (d.get("results") or []):
        for u in (res.get("downloadUrl"), res.get("fullTextIdentifier")):
            if u:
                out.append(u)
    return out


def _looks_like_image(url: str) -> bool:
    return url.lower().split("?")[0].endswith(_SKIP_EXT)


def _candidate_urls(manifest_url: str, work: dict | None, *, unpaywall: bool = False,
                    s2: bool = False, core: bool = False,
                    timeout: float = 30.0) -> list[str]:
    """Ordered, de-duped PDF-url candidates for one work.

    The manifest's ``best_oa_location.pdf_url`` is tried first, but publishers
    (Wiley/ACS/Springer) often 403 or serve an HTML landing page there. So we then
    fall back to ALL of the work's OA locations from OpenAlex, **repository hosts
    first** (arXiv/ChemRxiv/PMC/HAL/institutional repos rarely block bots and almost
    always serve a real PDF), then Semantic Scholar (arXiv-id reconstruction + its own
    OA resolution), then CORE.ac.uk (repository harvester), and finally Unpaywall's OA
    copies. Each stage only ADDS candidates the earlier ones may have missed — this is
    what lifts the OA hit-rate past best_oa-only.
    """
    cands: list[str] = []
    if manifest_url and not _looks_like_image(manifest_url):
        cands.append(manifest_url)
    doi = ""
    if work:
        doi = (work.get("doi") or "").replace("https://doi.org/", "")
        locs = work.get("locations") or []
        repo = [l for l in locs if (l.get("source") or {}).get("type") == "repository"]
        other = [l for l in locs if (l.get("source") or {}).get("type") != "repository"]
        for l in repo + other:
            u = l.get("pdf_url")
            if u and not _looks_like_image(u):
                cands.append(u)
    if s2 and doi:
        cands.extend(u for u in _semantic_scholar_pdf_urls(doi, timeout) if not _looks_like_image(u))
    if core and doi:
        cands.extend(u for u in _core_pdf_urls(doi, timeout) if not _looks_like_image(u))
    if unpaywall and doi:
        cands.extend(u for u in _unpaywall_pdf_urls(doi, timeout) if not _looks_like_image(u))
    # PMC fix: NCBI's own /pmc/articles/PMC…/pdf 403s or serves HTML to urllib. Europe PMC's
    # fullTextPDF endpoint serves the real OA PDF reliably (no browser). If any candidate (or
    # the work's ids) carries a PMCID, try that FIRST — recovers the big PMC failure bucket.
    pmc = None
    ids = (work or {}).get("ids") or {}
    for s in [ids.get("pmcid") or "", manifest_url or ""] + cands:
        m = re.search(r"PMC(\d+)", s or "", re.I)
        if m:
            pmc = m.group(1)
            break
    if pmc:
        cands.insert(0, f"https://europepmc.org/articles/PMC{pmc}?pdf=render")
    # de-dup preserving order
    seen, out = set(), []
    for u in cands:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _download_pdf(url: str, dest: Path, *, timeout: float, max_bytes: int) -> tuple[bool, str]:
    """Fetch ``url`` → ``dest`` iff the body is a real PDF. Returns (ok, reason)."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/pdf,*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ctype = (r.headers.get("Content-Type") or "").lower()
            data = r.read(max_bytes + 1)
    except Exception as e:                                  # noqa: BLE001
        return False, f"fetch-error: {type(e).__name__}: {str(e)[:120]}"
    if len(data) > max_bytes:
        return False, f"too-large (>{max_bytes//1_000_000}MB)"
    if not data.startswith(b"%PDF"):
        # not a PDF (landing page / image / html paywall) — don't save it as one
        hint = "html" if b"<html" in data[:512].lower() else f"ctype={ctype or '?'}"
        return False, f"not-a-pdf ({hint})"
    dest.write_bytes(data)
    return True, f"{len(data)//1024} KB"


def _fetch_topic(topic_dir: Path, client: OpenAlexClient, *, tiers: set, timeout: float,
                 sleep: float, max_bytes: int, max_retries: int, fallback: bool,
                 unpaywall: bool, s2: bool = False, core: bool = False,
                 only_ids: set | None = None) -> dict:
    manifest = topic_dir / "manifest.jsonl"
    if not manifest.exists():
        return {"topic": topic_dir.name, "skipped": "no manifest"}
    rows = []
    # NOT .splitlines(): it also breaks on \r/U+0085/etc that can legitimately appear
    # inside a JSON string value (paper titles/abstracts). JSONL is \n-delimited.
    for ln, l in enumerate(manifest.read_text().split("\n"), 1):
        if not l.strip():
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError as e:  # one corrupt line must not kill the batch
            print(f"  [warn] {topic_dir.name}: skipping malformed manifest line {ln}: {e}")
    pdfs = topic_dir / "pdfs"
    pdfs.mkdir(exist_ok=True)
    fail_log = pdfs / "_failures.jsonl"
    failures: list[dict] = []

    n_ok = n_skip_have = n_skip_nooa = n_fail = 0
    for r in rows:
        wid, url, tier = r.get("work_id"), r.get("pdf_url"), r.get("tier")
        if only_ids is not None and wid not in only_ids:
            continue
        if tier not in tiers:
            continue
        dest = pdfs / f"{wid}.pdf"
        if dest.exists() and dest.stat().st_size > 1000:    # resume: already have it
            n_skip_have += 1
            continue
        if not r.get("is_oa"):
            n_skip_nooa += 1
            continue
        # candidate urls: manifest best_oa first, then all OA locations (repos first),
        # then S2 / CORE / Unpaywall's OA copies. fallback off → manifest url only.
        work = client.work_by_id(wid) if (fallback or unpaywall or s2 or core) else None
        cands = _candidate_urls(url, work, unpaywall=unpaywall, s2=s2, core=core, timeout=timeout)
        if not cands:
            n_skip_nooa += 1
            continue
        ok, reason = False, "no candidates"
        for cand in cands:
            for attempt in range(max_retries):
                ok, reason = _download_pdf(cand, dest, timeout=timeout, max_bytes=max_bytes)
                if ok:
                    break
                time.sleep(sleep * (1 + attempt))
            if ok:
                break
            time.sleep(sleep)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            failures.append({"work_id": wid, "tier": tier, "n_candidates": len(cands),
                             "last_reason": reason, "urls": cands})
        time.sleep(sleep)

    if failures:
        fail_log.write_text("\n".join(json.dumps(f, ensure_ascii=False) for f in failures) + "\n")
    stats = {"topic": topic_dir.name, "downloaded": n_ok, "already_had": n_skip_have,
             "skip_non_oa": n_skip_nooa, "failed": n_fail, "total_rows": len(rows)}
    print(f"  [{topic_dir.name[:46]:46s}] +{n_ok} dl  ={n_skip_have} have  "
          f"{n_skip_nooa} non-oa  {n_fail} fail")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="set folder name under output/tiered_corpus/")
    ap.add_argument("--topic", default=None, help="only this topic slug (default: all topics)")
    ap.add_argument("--tiers", nargs="*", default=["bronze", "silver", "golden"],
                    help="only fetch these tiers (default: all)")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--sleep", type=float, default=0.8, help="polite delay between requests (s)")
    ap.add_argument("--max-mb", type=int, default=80, help="skip PDFs larger than this")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--no-fallback", action="store_true",
                    help="only try the manifest best_oa pdf_url (skip the multi-location fallback)")
    ap.add_argument("--unpaywall", action="store_true",
                    help="also resolve OA copies via Unpaywall (repository-first); lifts hit-rate")
    ap.add_argument("--s2", action="store_true",
                    help="also resolve via Semantic Scholar (arXiv-id reconstruction + its own OA link)")
    ap.add_argument("--core", action="store_true",
                    help="also resolve via CORE.ac.uk (needs $CORE_API_KEY, free at core.ac.uk/services/api)")
    ap.add_argument("--ids-file", default=None,
                    help="restrict to work_ids listed in this file (one per line) — "
                         "used to prioritize a pre-screened candidate subset over the full set")
    args = ap.parse_args()

    setdir = OUTROOT / args.set
    if not setdir.is_dir():
        sys.exit(f"no such set: {setdir}")
    topic_dirs = ([setdir / args.topic] if args.topic
                  else sorted(d for d in setdir.iterdir() if d.is_dir()))
    tiers = set(args.tiers)
    only_ids = None
    if args.ids_file:
        only_ids = {l.strip() for l in Path(args.ids_file).read_text().split("\n") if l.strip()}
    client = OpenAlexClient()
    print(f"[fetch] set={args.set} topics={len(topic_dirs)} tiers={sorted(tiers)} "
          f"fallback={'off' if args.no_fallback else 'on (multi-location)'} "
          f"unpaywall={'on' if args.unpaywall else 'off'} "
          f"s2={'on' if args.s2 else 'off'} core={'on' if args.core else 'off'} "
          f"ids_file={'off' if only_ids is None else f'on ({len(only_ids)} ids)'}")
    t0 = time.time()
    allstats = []
    for td in topic_dirs:
        allstats.append(_fetch_topic(td, client, tiers=tiers, timeout=args.timeout, sleep=args.sleep,
                                     max_bytes=args.max_mb * 1_000_000, max_retries=args.max_retries,
                                     fallback=not args.no_fallback, unpaywall=args.unpaywall,
                                     only_ids=only_ids,
                                     s2=args.s2, core=args.core))
    tot = {k: sum(s.get(k, 0) for s in allstats)
           for k in ("downloaded", "already_had", "skip_non_oa", "failed")}
    (setdir / "fetch_report.json").write_text(json.dumps(
        {"set": args.set, "tiers": sorted(tiers), "per_topic": allstats, "totals": tot},
        ensure_ascii=False, indent=2))
    print(f"\n[done] in {time.time()-t0:.0f}s  totals={tot}")
    print(f"[out] PDFs in {setdir}/<topic>/pdfs/  ·  report: {setdir}/fetch_report.json")


if __name__ == "__main__":
    main()
