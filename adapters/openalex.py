"""OpenAlex metadata adapter + tiered paper scorer (CLAUDE.md §2-layer-1, §18).

The discovery line (§18) mines *patterns* from a paper corpus, but not every
paper deserves equal weight. The user's framing: grade papers **bronze / silver /
golden** so the data engine borrows scientific *taste* —

  • **bronze**  — a normal paper. Kept for *breadth* of knowledge surface; cheap
    coverage of what the field generally does.
  • **silver**  — a high-quality journal/conference paper. The workhorse source
    for extracting scientific-discovery patterns (§18.1 premise→tension→move).
  • **golden**  — a high-quality venue AND highly-cited / from a prominent group.
    These carry extra weight downstream (higher sampling weight into the discovery
    miner + the SFT/RL high-score pool), so the corpus is pulled toward good taste.

Every tiering signal comes from **OpenAlex** (free, no key), so the grade is
reproducible and zero-maintenance — no hand-curated journal whitelist (the user
chose the automatic-signal route). The signals:

  • ``fwci``                          — field-weighted citation impact (age + field
                                        normalised; >1 = above world average).
  • ``citation_normalized_percentile`` — top-1% / top-10% flags within field+year.
  • venue ``2yr_mean_citedness`` + ``h_index`` (per-source ``summary_stats``)
                                        — venue quality, age-independent.
  • lead-institution ``h_index``       — proxy for a prominent group/lab.

Age-awareness (§ honesty): a 2026 paper has ``cited_by_count == 0`` and
``fwci/percentile == None``. We never punish recency — for young papers the
venue + institution signals carry the grade, and citation signals only *promote*.

This adapter is metadata-only: it does not download PDFs. The crawl driver
(``examples/crawl_tiered_corpus.py``) turns a tiered work list into a download
manifest; PDF→MinerU parsing stays the existing external step.

ANCHOR (§1.7, §11):  OpenAlex REST API v1 (api.openalex.org), accessed 2026-06.
                     No SDK dependency — plain ``urllib`` against the public API.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator, Optional

OPENALEX_BASE = "https://api.openalex.org"
# OpenAlex asks for a contact in the "polite pool" for faster, more stable service.
# Overridable via the client; this is only a courtesy header value, not a secret.
DEFAULT_MAILTO = "scidata-engine@inceptlabs.ai"

TIERS = ("bronze", "silver", "golden")


# ---------------------------------------------------------------------------
# Tier policy (pure data — no network). Thresholds are deliberately explicit so
# the grade is auditable and tunable from one place.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TierConfig:
    """Thresholds that map OpenAlex signals → {bronze, silver, golden}.

    Defaults are calibrated against the comp-catalysis corpus (CO/Pt), where a
    strong venue runs ~10+ 2yr-citedness and a genuinely influential paper sits
    in the field's top 10% (silver) or top 1% (golden)."""

    # silver gate — "high-quality venue OR clearly above-average impact"
    silver_venue_citedness: float = 5.0      # venue 2yr_mean_citedness
    silver_venue_h_index: int = 150          # strong, established venue
    silver_fwci: float = 1.5                 # >1 already beats world avg; 1.5 = solid
    silver_top_percent: int = 10             # top-10% in field+year

    # golden gate — "elite venue AND (very high impact OR prominent group)".
    # Venue eliteness uses ONLY 2yr_mean_citedness (volume-independent): h_index is an
    # age/volume artifact — a large workhorse journal (e.g. J. Phys. Chem. C, h≈370 but
    # 2yr-citedness only ~3.4) accumulates a high h-index without being elite, so it must
    # not be a golden-venue signal.
    golden_venue_citedness: float = 9.0      # top journals (JACS≈15, Nature Comms≈16, Nature Catal≈30)
    golden_fwci: float = 4.0                 # field-leading impact
    golden_top_percent: int = 1              # top-1% in field+year
    golden_institution_h_index: int = 400    # lead inst is a major research org

    # downstream sampling weights (used by the discovery miner + SFT/RL pools)
    weights: tuple = (1.0, 2.0, 4.0)         # (bronze, silver, golden)

    def weight(self, tier: str) -> float:
        return dict(zip(TIERS, self.weights)).get(tier, 1.0)


@dataclass
class PaperSignals:
    """Flat, JSON-able view of the tiering signals pulled from one OpenAlex work."""
    work_id: str = ""
    doi: str = ""
    title: str = ""
    year: Optional[int] = None
    cited_by_count: int = 0
    fwci: Optional[float] = None
    top_percentile: Optional[float] = None       # citation_normalized_percentile.value
    is_top_1_percent: bool = False
    is_top_10_percent: bool = False
    venue: str = ""
    venue_type: str = ""                          # journal / conference / repository …
    venue_citedness: Optional[float] = None       # source 2yr_mean_citedness
    venue_h_index: Optional[int] = None
    lead_institution: str = ""
    lead_institution_h_index: Optional[int] = None
    is_oa: bool = False
    pdf_url: str = ""
    source_id: str = ""                           # OpenAlex source id (for metric backfill)


# ---------------------------------------------------------------------------
# Signal extraction + tier scoring — both PURE (unit-tested without network).
# ---------------------------------------------------------------------------
def extract_signals(work: dict) -> PaperSignals:
    """Pull the tiering signals out of a raw OpenAlex /works record. Pure."""
    prim = work.get("primary_location") or {}
    src = prim.get("source") or {}
    best_oa = work.get("best_oa_location") or {}
    pct = work.get("citation_normalized_percentile") or {}
    auths = work.get("authorships") or []
    lead_inst_name, lead_inst_id = "", ""
    for a in auths:                                  # first author with an institution = "lead"
        insts = a.get("institutions") or []
        if insts:
            lead_inst_name = insts[0].get("display_name") or ""
            lead_inst_id = insts[0].get("id") or ""
            break

    return PaperSignals(
        work_id=(work.get("id") or "").rsplit("/", 1)[-1],
        doi=(work.get("doi") or "").replace("https://doi.org/", ""),
        title=work.get("title") or work.get("display_name") or "",
        year=work.get("publication_year"),
        cited_by_count=int(work.get("cited_by_count") or 0),
        fwci=work.get("fwci"),
        top_percentile=pct.get("value"),
        is_top_1_percent=bool(pct.get("is_in_top_1_percent")),
        is_top_10_percent=bool(pct.get("is_in_top_10_percent")),
        venue=src.get("display_name") or "",
        venue_type=src.get("type") or "",
        venue_citedness=None,                        # filled by enrich_venue_metrics (needs /sources)
        venue_h_index=None,
        lead_institution=lead_inst_name,
        lead_institution_h_index=None,               # filled by enrich (needs /institutions)
        is_oa=bool((work.get("open_access") or {}).get("is_oa")),
        pdf_url=best_oa.get("pdf_url") or "",
        source_id=(src.get("id") or "").rsplit("/", 1)[-1],
        # stash ids so the driver can backfill venue/inst metrics without re-fetching
    )


# Venue-quality signals (2yr-citedness, h-index) only count for *peer-reviewed* venues.
# A repository (arXiv, Research Square, institutional repos) is not a quality stamp — arXiv
# carries h_index≈674 purely from volume yet 2yr-citedness ≈0.28. So a preprint must earn
# its grade on its OWN impact (fwci / percentile), never on the host repo's venue metrics.
_PEER_REVIEWED_VENUE_TYPES = {"journal", "conference", "book series"}


def _venue_quality_counts(s: PaperSignals) -> bool:
    """Whether this paper's venue metrics may contribute to its grade (peer-reviewed only)."""
    return s.venue_type in _PEER_REVIEWED_VENUE_TYPES


def _meets_silver(s: PaperSignals, c: TierConfig) -> bool:
    venue_ok = _venue_quality_counts(s) and (
        (s.venue_citedness is not None and s.venue_citedness >= c.silver_venue_citedness) or
        (s.venue_h_index is not None and s.venue_h_index >= c.silver_venue_h_index))
    impact_ok = (s.fwci is not None and s.fwci >= c.silver_fwci) or s.is_top_10_percent
    return venue_ok or impact_ok


def _meets_golden(s: PaperSignals, c: TierConfig) -> bool:
    elite_venue = _venue_quality_counts(s) and \
        s.venue_citedness is not None and s.venue_citedness >= c.golden_venue_citedness
    very_high_impact = (s.fwci is not None and s.fwci >= c.golden_fwci) or s.is_top_1_percent
    prominent_group = (s.lead_institution_h_index is not None and
                       s.lead_institution_h_index >= c.golden_institution_h_index)
    # golden = elite venue AND (field-leading impact OR a prominent group behind it)
    return elite_venue and (very_high_impact or prominent_group)


def score_tier(s: PaperSignals, config: Optional[TierConfig] = None) -> tuple[str, float, list[str]]:
    """Grade one paper. Returns (tier, weight, reasons).

    Monotone by construction: golden ⇒ silver gate also met. Citation signals only
    ever *promote* (recency-safe — a young paper falls back to venue/inst signals).
    """
    c = config or TierConfig()
    reasons: list[str] = []
    tier = "bronze"
    if _meets_silver(s, c):
        tier = "silver"
        if s.is_top_10_percent:
            reasons.append("top-10% in field+year")
        if s.fwci is not None and s.fwci >= c.silver_fwci:
            reasons.append(f"fwci={s.fwci:.1f}")
        if _venue_quality_counts(s):
            if s.venue_citedness is not None and s.venue_citedness >= c.silver_venue_citedness:
                reasons.append(f"venue 2yr-citedness={s.venue_citedness:.1f}")
            elif s.venue_h_index is not None and s.venue_h_index >= c.silver_venue_h_index:
                reasons.append(f"venue h-index={s.venue_h_index}")
    if tier == "silver" and _meets_golden(s, c):
        tier = "golden"
        if s.is_top_1_percent:
            reasons.append("top-1% in field+year")
        if s.fwci is not None and s.fwci >= c.golden_fwci:
            reasons.append(f"fwci={s.fwci:.1f} (field-leading)")
        if s.lead_institution_h_index is not None and \
                s.lead_institution_h_index >= c.golden_institution_h_index:
            reasons.append(f"prominent group ({s.lead_institution}, h={s.lead_institution_h_index})")
    return tier, c.weight(tier), reasons


# ---------------------------------------------------------------------------
# Network client (urllib; no SDK). Lazy + cached so the driver stays cheap.
# ---------------------------------------------------------------------------
class OpenAlexClient:
    """Minimal polite-pool OpenAlex client (works / sources / institutions)."""

    def __init__(self, *, mailto: str = DEFAULT_MAILTO, timeout: float = 20.0,
                 max_retries: int = 4, sleep: float = 0.12,
                 api_key: Optional[str] = None):
        self.mailto = mailto
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep = sleep
        # OpenAlex requires an API key since 2026-02-13; anonymous full-text
        # search is rate-limited (503 "Anonymous search is temporarily
        # rate-limited"). Free key from openalex.org/settings/api, passed as the
        # ``api_key`` query param. Falls back to $OPENALEX_API_KEY.
        self.api_key = api_key or os.environ.get("OPENALEX_API_KEY") or None
        self._source_cache: dict[str, dict] = {}
        self._inst_cache: dict[str, dict] = {}

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        params = dict(params or {})
        params["mailto"] = self.mailto
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{OPENALEX_BASE}/{path.lstrip('/')}?{urllib.parse.urlencode(params)}"
        last = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(url, timeout=self.timeout) as r:
                    data = json.loads(r.read())
                time.sleep(self.sleep)   # pace every call (pagination + entity lookups)
                return data
            except Exception as e:                       # noqa: BLE001
                last = e
                time.sleep(self.sleep * (2 ** attempt))   # exponential backoff
        raise RuntimeError(f"OpenAlex GET failed after {self.max_retries} tries: {url} :: {last!r}")

    def work_by_doi(self, doi: str) -> Optional[dict]:
        """Resolve a single work by DOI (used to backfill the existing corpus)."""
        doi = doi.replace("https://doi.org/", "").strip()
        try:
            return self._get(f"works/doi:{doi}")
        except RuntimeError:
            return None

    def work_by_id(self, work_id: str) -> Optional[dict]:
        """Resolve a single work by its OpenAlex id (e.g. ``W7143594206``)."""
        wid = work_id.rsplit("/", 1)[-1].strip()
        try:
            return self._get(f"works/{wid}")
        except RuntimeError:
            return None

    def search_works(self, query: str, *, per_page: int = 50, max_works: int = 200,
                     filters: Optional[str] = None, sort: Optional[str] = None,
                     sample: bool = False, seed: int = 13) -> Iterator[dict]:
        """Search, yielding raw work records up to ``max_works``.

        Two modes:
          • ranked (default) — cursor-paginated by ``sort`` (e.g. cited_by_count:desc).
            Brings the head of the distribution → great for *depth* on a topic, but a
            biased (high-impact) slice.
          • ``sample=True`` — OpenAlex random ``sample`` (page-based, fixed ``seed`` for
            reproducibility). A *representative* slice of the whole topic (mix of tiers)
            → what topic-set *diversity* wants. Note: ``sample`` is incompatible with
            cursor + sort, so this branch ignores ``sort`` and pages by number.
        """
        # the search term lives in default.search when filtering so it composes with sample
        search_filter = f"default.search:{query}"
        full_filter = f"{search_filter},{filters}" if filters else search_filter
        if sample:
            n, page_no = 0, 1
            while n < max_works:
                size = min(per_page, max_works - n)
                params = {"filter": full_filter, "sample": max_works,
                          "seed": seed, "per-page": size, "page": page_no}
                page = self._get("works", params)
                results = page.get("results") or []
                if not results:
                    break
                for w in results:
                    yield w
                    n += 1
                    if n >= max_works:
                        return
                page_no += 1
            return
        cursor = "*"
        n = 0
        while cursor and n < max_works:
            params = {"search": query, "per-page": min(per_page, max_works - n), "cursor": cursor}
            if filters:
                params["filter"] = filters
            if sort:
                params["sort"] = sort
            page = self._get("works", params)
            results = page.get("results") or []
            if not results:
                break
            for w in results:
                yield w
                n += 1
                if n >= max_works:
                    return
            cursor = (page.get("meta") or {}).get("next_cursor")

    def source_metrics(self, source_id: str) -> dict:
        """Cached per-venue summary_stats (2yr_mean_citedness, h_index)."""
        if not source_id:
            return {}
        if source_id not in self._source_cache:
            try:
                d = self._get(f"sources/{source_id}")
                self._source_cache[source_id] = d.get("summary_stats") or {}
            except RuntimeError:
                self._source_cache[source_id] = {}
        return self._source_cache[source_id]

    def institution_metrics(self, inst_id: str) -> dict:
        """Cached per-institution summary_stats (h_index)."""
        if not inst_id:
            return {}
        iid = inst_id.rsplit("/", 1)[-1]
        if iid not in self._inst_cache:
            try:
                d = self._get(f"institutions/{iid}")
                self._inst_cache[iid] = d.get("summary_stats") or {}
            except RuntimeError:
                self._inst_cache[iid] = {}
        return self._inst_cache[iid]


def enrich_signals(client: OpenAlexClient, work: dict, signals: PaperSignals) -> PaperSignals:
    """Backfill venue + lead-institution metrics that aren't on the /works record.

    These two extra lookups are what separate silver from golden (venue quality +
    group prominence), so the driver calls this before ``score_tier``.
    """
    if signals.source_id:
        st = client.source_metrics(signals.source_id)
        signals.venue_citedness = st.get("2yr_mean_citedness")
        signals.venue_h_index = st.get("h_index")
    # lead institution id was on the work; re-read it (extract_signals kept only the name)
    for a in (work.get("authorships") or []):
        insts = a.get("institutions") or []
        if insts and insts[0].get("id"):
            im = client.institution_metrics(insts[0]["id"])
            signals.lead_institution_h_index = im.get("h_index")
            break
    return signals
