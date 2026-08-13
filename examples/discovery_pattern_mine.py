"""Mine discovery patterns from a focused paper corpus (the user's "从论文里挖掘
发现的范式" probe). Breadth-phase, pure EXTRACTOR: no hard gate — every record is
stored as ``pending-soft-verify`` (v5 §0.9: discovery reasoning is soft, goes to
the external AI Verifier; only the recompute_handle anchors are future rigor gates).

Pipeline (deterministic front-end + LLM reconstruction, §5):
  1. PaperCorpus  : read each MinerU paper from the zip, slice canonical sections.
  2. extract_pattern : LLM → {premise/consensus → tension → motivation → method →
                       experiment → conclusion} + key_claims (with recompute_handle)
                       + novelty_move. Faithful to the paper, result-conditioned.
  3. aggregate    : cross-paper synthesis → consensus vs contradiction + rigor anchors.

Output (examples/output/discovery_patterns/):
  patterns/<paper_id>.json   one record per paper
  corpus_map.json            cross-paper consensus / contradictions / rigor anchors
  SUMMARY.md                 human-readable map for manual review (breadth phase)

Run:
  python examples/discovery_pattern_mine.py \
      [--zip co_pt_corpus.zip] [--limit N] [--no-aggregate]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from adapters.paper_corpus import PaperCorpus
from reconstruct.discovery_pattern import DiscoveryPattern, aggregate, extract_pattern, to_dict
from reconstruct.llm_openrouter import OpenRouterClient

OUTDIR = REPO / "examples/output/discovery_patterns"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(REPO / "co_pt_corpus.zip"))
    ap.add_argument("--limit", type=int, default=None, help="extract only the first N papers (quick test)")
    ap.add_argument("--no-aggregate", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-extract papers whose JSON already exists")
    ap.add_argument("--outdir", default=None, help="write patterns/ + corpus_map.json + SUMMARY.md here "
                    "(default: examples/output/discovery_patterns). Use a separate dir for a new corpus.")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel LLM extraction workers (each paper's call is I/O-bound; "
                         "one OpenRouterClient per worker, resume-cache check stays sequential)")
    args = ap.parse_args()

    global OUTDIR
    if args.outdir:
        OUTDIR = Path(args.outdir)

    corpus = PaperCorpus(args.zip)
    print(f"[corpus] {len(corpus)} papers in {Path(args.zip).name}")
    print(f"[llm] teacher = {OpenRouterClient()._model}  workers={args.workers}")

    (OUTDIR / "patterns").mkdir(parents=True, exist_ok=True)

    patterns = []
    patterns_lock = threading.Lock()
    n_calls = [0]
    t_start = time.time()

    def _process(i, paper):
        t0 = time.time()
        llm = _thread_llm.client
        try:
            pat = extract_pattern(llm, paper)
        except Exception as e:                                  # noqa: BLE001
            print(f"  [{i+1}] {paper.paper_id[:40]:40s} ERROR {e!r}"[:120])
            return
        if pat is None:
            print(f"  [{i+1}] {paper.paper_id[:40]:40s} unparseable/empty -> skip")
            return
        (OUTDIR / "patterns" / f"{paper.paper_id}.json").write_text(
            json.dumps(to_dict(pat), ensure_ascii=False, indent=2))
        anchors = ",".join(pat.recompute_handles()) or "-"
        print(f"  [{i+1}] {paper.paper_id[:32]:32s} move={pat.novelty_move:18s} "
              f"anchors=[{anchors}] ({time.time()-t0:.0f}s)")
        with patterns_lock:
            patterns.append(pat)
            n_calls[0] += llm.n_calls

    class _ThreadLLM(threading.local):
        def __init__(self):
            self.client = OpenRouterClient(max_tokens=2048, temperature=0.2)

    _thread_llm = _ThreadLLM()

    to_submit = []
    for i, paper in enumerate(corpus.papers(limit=args.limit)):
        cached = OUTDIR / "patterns" / f"{paper.paper_id}.json"
        if cached.exists() and not args.force:                  # resume: reuse prior extraction
            patterns.append(DiscoveryPattern(**json.loads(cached.read_text())))
            print(f"  [{i+1}] {paper.paper_id[:32]:32s} cached -> reuse")
            continue
        to_submit.append((i, paper))

    if args.workers <= 1:
        for i, paper in to_submit:
            _process(i, paper)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_process, i, paper) for i, paper in to_submit]
            for _ in as_completed(futs):
                pass

    print(f"[done] {len(patterns)} patterns in {time.time()-t_start:.0f}s, {n_calls[0]} llm calls")
    if not patterns:
        return

    # ---- corpus-level rollups (deterministic) ----
    move_dist = Counter(p.novelty_move for p in patterns)
    anchor_dist = Counter(h for p in patterns for h in p.recompute_handles())
    n_with_tension = sum(1 for p in patterns if len(p.tension.strip()) > 20)
    n_recomputable = sum(1 for p in patterns if p.recompute_handles())

    corpus_map: dict = {
        "n_papers": len(patterns),
        "move_distribution": dict(move_dist.most_common()),
        "recompute_anchor_distribution": dict(anchor_dist.most_common()),
        "n_with_nontrivial_tension": n_with_tension,
        "n_with_recompute_anchor": n_recomputable,
    }

    # ---- LLM cross-paper synthesis (consensus vs contradiction) ----
    if not args.no_aggregate:
        print("[aggregate] synthesising consensus vs contradictions ...")
        agg_llm = OpenRouterClient(max_tokens=3500, temperature=0.2)
        synth = aggregate(agg_llm, patterns)
        if synth:
            corpus_map["synthesis"] = synth
        else:
            print("  [aggregate] synthesis reply unparseable — keeping deterministic rollups only")

    (OUTDIR / "corpus_map.json").write_text(json.dumps(corpus_map, ensure_ascii=False, indent=2))
    _write_summary(corpus_map, patterns)
    print(f"[out] {OUTDIR}/  (patterns/, corpus_map.json, SUMMARY.md)")


def _write_summary(cmap: dict, patterns: list) -> None:
    L = ["# Discovery-pattern map — CO/Pt corpus (pending-soft-verify)",
         "",
         f"- papers extracted: **{cmap['n_papers']}**",
         f"- with non-trivial tension (discovery seed): **{cmap['n_with_nontrivial_tension']}**",
         f"- with a recomputable rigor anchor: **{cmap['n_with_recompute_anchor']}**",
         "",
         "## Novelty-move distribution",
         ""]
    for m, n in cmap["move_distribution"].items():
        L.append(f"- `{m}`: {n}")
    L += ["", "## Recompute anchors (future rigor gates)", ""]
    for h, n in cmap["recompute_anchor_distribution"].items():
        L.append(f"- `{h}`: {n} papers")

    synth = cmap.get("synthesis", {})
    if synth:
        L += ["", "## Cross-paper synthesis", ""]
        if synth.get("topic"):
            L.append(f"**Topic:** {synth['topic']}\n")
        if synth.get("contradictions"):
            L.append("### Contradictions (calculation-adjudicable)\n")
            for c in synth["contradictions"]:
                L.append(f"- **{c.get('question','')}**")
                L.append(f"  - A: {c.get('position_a','')}  ⟵ {', '.join(c.get('papers_a',[]))}")
                L.append(f"  - B: {c.get('position_b','')}  ⟵ {', '.join(c.get('papers_b',[]))}")
            L.append("")
        if synth.get("recurring_tensions"):
            L.append("### Recurring tensions\n")
            for t in synth["recurring_tensions"]:
                L.append(f"- {t}")
            L.append("")
        if synth.get("rigor_anchors"):
            L.append("### Rigor anchors a deterministic gate could verify\n")
            for a in synth["rigor_anchors"]:
                L.append(f"- {a}")
            L.append("")

    L += ["", "## Per-paper discovery arcs", ""]
    for p in patterns:
        L.append(f"### {p.paper_id} — `{p.novelty_move}`")
        L.append(f"- **premise/consensus:** {p.premise_consensus}")
        L.append(f"- **tension:** {p.tension or '_(none — confirmatory)_'}")
        L.append(f"- **method:** {p.method}")
        L.append(f"- **conclusion:** {p.conclusion}")
        if p.recompute_handles():
            L.append(f"- **rigor anchors:** {', '.join(p.recompute_handles())}")
        L.append("")
    (OUTDIR / "SUMMARY.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
