#!/usr/bin/env python3
"""
arft_verify.py — quality checks on the finished ARFT classification.

Up to three independent checks, each answering a different way the labelling could be
wrong:

  polarity   Did the classifier mistake exculpatory language for a finding? Some
             fabrication-vocabulary terms (see FAB_TERMS) are often used to CLEAR an
             agent, not accuse it, in trajectory-analysis writeups. If the
             fabrication-family HIT rate tracks the raw mention rate of those terms,
             the classifier is reading the audit as the finding — a real failure mode,
             not a hypothetical one (this is exactly what motivated this check).
             NOTE: FAB_TERMS below is a reasonable starting list, not empirically
             calibrated against your corpus — if your writeups use different
             vocabulary for "we checked for fabrication and found none," extend it.

  prior      OPTIONAL. If you have an earlier classification run against this same
             taxonomy (e.g. a pilot, or a run over a different corpus) saved in the
             same {"rows": [[model, task, {code: score}]]} shape as arft_aggregate.py's
             agg.json, pass it via --prior-agg to sanity-check that the head of the
             new distribution roughly rhymes with it. Different trajectories, so exact
             agreement isn't expected — but a wildly different head suggests a prompt
             or guide bug, not a real finding. Skipped entirely if --prior-agg isn't given.

  kappa      Blind double-pass agreement. Requires a second pass written to a parallel
             output directory (see --pass2). Reports per-pattern Cohen's kappa; any
             pattern below --kappa-floor needs its guide entry rewritten before the
             numbers are trusted.

Usage:
    python3 arft_verify.py                                    # polarity only
    python3 arft_verify.py --prior-agg /path/to/earlier_agg.json
    python3 arft_verify.py --pass2 ../results_pass2
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arft_classify_cc as c   # noqa: E402
import arft_patterns as P      # noqa: E402

# Fabrication-family codes: the ones most at risk from the exculpatory-language trap.
FABRICATION = ["B.1", "D.6", "E.4"]
FAB_TERMS = ["fabricat", "hallucinat", "no fabrication", "contaminat"]


def load(root):
    recs = {}
    for mk in c.MODELS:
        for t in c.discover(mk):
            j = Path(root) / mk / f"{t['task_id']}.json"
            if not j.exists():
                continue
            try:
                d = json.loads(j.read_text())
            except Exception:
                continue
            hits = {it["code"] for it in (d.get("hits") or [])
                    if it.get("code") in P.VALID_CODES}
            recs[(mk, t["task_id"])] = {"hits": hits, "md": t["analysis_md"]}
    return recs


def check_polarity(recs):
    print("\n=== polarity regression ===")
    if not recs:
        print("  no classifications"); return False
    n = len(recs)
    mention = 0
    for v in recs.values():
        try:
            txt = Path(v["md"]).read_text(errors="ignore").lower()
        except Exception:
            continue
        if any(t in txt for t in FAB_TERMS):
            mention += 1
    m_rate = 100 * mention / n
    print(f"  analyses merely MENTIONING {'/'.join(FAB_TERMS)}: {mention}/{n} = {m_rate:.0f}%")
    ok = True
    for code in FABRICATION:
        h = sum(1 for v in recs.values() if code in v["hits"])
        r = 100 * h / n
        # A fabrication rate anywhere near the mention rate means the classifier is
        # reading the audit as the finding.
        bad = r > 0.6 * m_rate
        ok &= not bad
        print(f"  {code} {P.NAME[code]:38} HIT {h:4}/{n} = {r:5.1f}%"
              f"  {'<-- SUSPICIOUS, tracks the mention rate' if bad else 'ok'}")
    print(f"  verdict: {'PASS' if ok else 'FAIL — revisit the polarity rule in arft_guide.md'}")
    return ok


def check_prior(recs, prior_agg_path):
    print("\n=== head-of-distribution vs a prior run (--prior-agg) ===")
    if not prior_agg_path:
        print("  no --prior-agg given; skipping (this check is optional)"); return True
    prior_agg_path = Path(prior_agg_path)
    if not prior_agg_path.exists():
        print(f"  --prior-agg {prior_agg_path} not found; skipping"); return True
    prior = json.load(open(prior_agg_path))
    prows = prior["rows"]
    pn = len(prows)
    prate = {code: 100 * sum(1 for r in prows if r[2].get(code) == P.SCORE_HIT) / pn
             for code in P.CODES}
    n = len(recs) or 1
    nrate = {code: 100 * sum(1 for v in recs.values() if code in v["hits"]) / n
             for code in P.CODES}
    ptop = sorted(P.CODES, key=lambda x: -prate[x])[:10]
    ntop = sorted(P.CODES, key=lambda x: -nrate[x])[:10]
    print(f"  prior n={pn} (different trajectories)   new n={n}")
    print(f"  {'rank':<5} {'prior':<28} {'new':<28}")
    for i in range(10):
        print(f"  {i+1:<5} {ptop[i]+' '+P.NAME[ptop[i]][:20]:<28} "
              f"{ntop[i]+' '+P.NAME[ntop[i]][:20]:<28}")
    overlap = len(set(ptop) & set(ntop))
    print(f"  top-10 overlap: {overlap}/10  "
          f"{'ok' if overlap >= 4 else '<-- LOW, check the prompt/guide'}")
    return overlap >= 4


def _kappa(a, b):
    """Cohen's kappa for two boolean label vectors."""
    n = len(a)
    if n == 0:
        return None
    both = sum(1 for x, y in zip(a, b) if x and y)
    neither = sum(1 for x, y in zip(a, b) if not x and not y)
    po = (both + neither) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1:
        return 1.0 if po == 1 else 0.0
    return (po - pe) / (1 - pe)


def check_kappa(recs, pass2_root, floor):
    print(f"\n=== blind double-pass agreement (floor={floor}) ===")
    r2 = load(pass2_root)
    keys = sorted(set(recs) & set(r2))
    if not keys:
        print(f"  no overlapping classifications in {pass2_root}; skipping"); return True
    print(f"  overlapping analyses: {len(keys)}")
    low = []
    print(f"  {'code':<6} {'name':<40} {'n1':>4} {'n2':>4} {'kappa':>7}")
    for code in P.CODES:
        a = [code in recs[k]["hits"] for k in keys]
        b = [code in r2[k]["hits"] for k in keys]
        if not any(a) and not any(b):
            continue
        k = _kappa(a, b)
        flag = ""
        if k is not None and k < floor:
            low.append((code, k)); flag = " <-- below floor"
        print(f"  {code:<6} {P.NAME[code][:40]:<40} {sum(a):>4} {sum(b):>4} "
              f"{('n/a' if k is None else f'{k:6.2f}')}{flag}")
    if low:
        print(f"\n  {len(low)} pattern(s) below kappa {floor} — rewrite their guide "
              f"entries before trusting the numbers:")
        for code, k in sorted(low, key=lambda x: x[1]):
            print(f"    {code} {P.NAME[code]}  kappa={k:.2f}")
    else:
        print("  all patterns at or above the floor")
    return not low


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(c.OUT_ROOT))
    ap.add_argument("--prior-agg", default=None,
                    help="optional: path to an earlier agg.json to compare against")
    ap.add_argument("--pass2", default=None)
    ap.add_argument("--kappa-floor", type=float, default=0.4)
    a = ap.parse_args()

    recs = load(a.root)
    print(f"loaded {len(recs)} classifications from {a.root}")
    ok = check_polarity(recs)
    ok &= check_prior(recs, a.prior_agg)
    if a.pass2:
        ok &= check_kappa(recs, a.pass2, a.kappa_floor)
    print(f"\nOVERALL: {'PASS' if ok else 'ATTENTION NEEDED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
