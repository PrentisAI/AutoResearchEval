#!/usr/bin/env python3
"""
arft_qa_check.py — schema + polarity gate for one ARFT classification.json.

Schema-shaped: it cannot tell a well-evidenced label from a plausible-sounding one.
It exists to make the self-healing loop converge on outputs that are *parseable and
locatable*, not to certify correctness. Semantic quality is checked by a hand-check
pilot and a blind double-pass (see arft_verify.py).

One semantic guard is included, because it catches a likely systematic error: a label
whose evidence points only at `## Credit Due`, the fair-credit section. That is a
polarity inversion, not a finding.

Usage:  python3 arft_qa_check.py path/to/classification.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arft_patterns as P  # noqa: E402

# Evidence must be locatable back into the source analysis.md. This corpus is now
# English-only (see ONBOARDING.md's fixed skeleton), so the rules below target the
# citation styles that skeleton and the classifier prompt actually ask for — an
# explicit section+issue reference, a section name, an issue/finding number, a
# section symbol, or a short quoted phrase. There is deliberately NO bare-decimal
# rule: accepting any `\d+\.` would let evidence citing nothing but a number like
# "2.66%" pass by accident.
# Distinctive multi-word headings only — deliberately NOT the bare single words
# ("Analysis", "Execution", "Writing", "Retrieval", "Ideation", "Checklist",
# "Grounding" alone) because those collide with ordinary prose ("the analysis
# shows...", "during execution..."); a bare-word version of this rule matched
# "analysis" inside "the analysis gives credit due here" in testing, which is
# exactly the false-positive this checker exists to avoid.
_SECTION_NAMES = (
    r"Core Verdict|Metadata|Trajectory Arc|Credit Due|"
    r"Ideation\s*&\s*Planning|Retrieval\s*&\s*Synthesis|"
    r"Execution\s*&\s*Implementation|Analysis\s*&\s*Interpretation|"
    r"Writing\s*&\s*Documentation|Self-Verification\s*&\s*Review|"
    r"Cross-Stage\s*Dynamics|Sentence-by-Sentence Checklist|"
    r"Numerical Grounding\s*Notes|Retraction\s*/\s*Correction Log|"
    r"One-Line Verdict"
)
LOCATOR = re.compile(
    # "Section C", "Section C, issue 8", "stage A", "stage X"
    r"\bSection\s+[A-FX]\b"
    r"|\bstage[: ]\s*[A-FX]\b"
    # a named section, e.g. "Trajectory Arc", "One-Line Verdict", "Credit Due"
    r"|\b(?:" + _SECTION_NAMES + r")\b"
    # "issue 12", "Finding 3", "problem 7", with an optional row/# form
    r"|\b(?:[Ii]ssue|[Ff]inding|[Pp]roblem)\s*#?\s*\d+\b"
    r"|\brow\s*\d+\b"
    # explicit section sigils / issue-id shorthands: §C-8, #13, C18, A.2
    r"|§\s*[A-FX]?[-.]?\s*\d*"
    r"|#\d+"
    r"|\b[A-FX]\.\d+\b"
    r"|\b[A-FX]\d{1,3}\b"
    # a real quoted phrase (the evidence is quoting the analysis verbatim)
    r"|\"[^\"\n]{4,}\"|“[^”\n]{4,}”|'[^'\n]{4,}'"
    # general structural invariant: whatever the exact wording, a citation lead is
    # SHORT and is followed by a colon before the substance of the claim — this is
    # what catches phrasing not covered by the explicit rules above.
    r"|^.{0,30}:",
    re.I | re.M)

# The fair-credit section, in its observed heading variants.
CREDIT_MARKERS = ("Credit Due", "credit is due", "give credit", "fair credit")
# Strip the credit reference together with any section sigil that introduces it
# (`## Credit Due`, `§Credit Due`, quoted forms) — otherwise the sigil itself would
# satisfy LOCATOR and mask the polarity error.
CREDIT_REF = re.compile(
    r"[#§\"'“”‘’]{0,3}\s*(?:" + "|".join(re.escape(m) for m in CREDIT_MARKERS) + r")"
    r"(?:\s*\([^)]*\))?\s*[\"'“”‘’]{0,2}", re.I)


def _evidence_ok(ev):
    return bool(ev) and isinstance(ev, str) and len(ev.strip()) >= 10


def _only_credit_sourced(ev):
    """True if the evidence cites the credit section and nothing else.

    `## Credit Due` is the fair-credit section, required in every analysis. A label
    whose evidence points only there is a polarity inversion — the analysis was saying
    the agent did this RIGHT. Evidence that also cites a real finding elsewhere is fine.
    """
    if not ev or not any(m.lower() in ev.lower() for m in CREDIT_MARKERS):
        return False
    return not LOCATOR.search(CREDIT_REF.sub(" ", ev))


def check(path):
    path = Path(path)
    res = {"path": str(path), "ok": False, "problems": [],
           "n_hits": 0, "n_partials": 0, "n_uncovered": 0, "codes": []}
    prob = res["problems"]

    try:
        d = json.loads(path.read_text())
    except Exception as e:
        prob.append(f"invalid JSON: {e}")
        return res
    if not isinstance(d, dict):
        prob.append("top level is not an object")
        return res

    for k in ("task_id", "overall_severity", "summary", "hits", "partials"):
        if k not in d:
            prob.append(f"missing key: {k}")

    sev = d.get("overall_severity")
    if sev not in P.SEVERITIES:
        prob.append(f"overall_severity {sev!r} not in {P.SEVERITIES}")

    if not (d.get("summary") or "").strip():
        prob.append("empty summary")

    seen = {}
    for bucket in ("hits", "partials"):
        items = d.get(bucket)
        if items is None:
            continue
        if not isinstance(items, list):
            prob.append(f"{bucket} is not a list")
            continue
        for i, it in enumerate(items):
            tag = f"{bucket}[{i}]"
            if not isinstance(it, dict):
                prob.append(f"{tag} is not an object")
                continue
            code = it.get("code")
            if code not in P.VALID_CODES:
                prob.append(f"{tag} invalid code {code!r}")
                continue
            if code in seen:
                prob.append(f"{code} appears in both {seen[code]} and {bucket}")
            seen[code] = bucket
            ev = it.get("evidence")
            if not _evidence_ok(ev):
                prob.append(f"{tag} {code}: missing/short evidence")
            elif not LOCATOR.search(ev):
                prob.append(f"{tag} {code}: evidence not locatable (no section/issue ref)")
            elif _only_credit_sourced(ev):
                prob.append(f"{tag} {code}: evidence sourced ONLY from the Credit Due section")
            conf = it.get("confidence")
            if conf is not None and not (isinstance(conf, (int, float)) and 0 <= conf <= 1):
                prob.append(f"{tag} {code}: confidence {conf!r} out of [0,1]")

    res["n_hits"] = len(d.get("hits") or [])
    res["n_partials"] = len(d.get("partials") or [])
    res["codes"] = sorted(seen)

    irc = d.get("iron_rules_cited")
    if irc is not None and not (isinstance(irc, list)
                                and all(isinstance(x, int) for x in irc)):
        prob.append("iron_rules_cited must be a list of ints")

    unc = d.get("uncovered") or []
    res["n_uncovered"] = len(unc)
    if not isinstance(unc, list):
        prob.append("uncovered is not a list")
    else:
        for i, u in enumerate(unc):
            if not isinstance(u, dict):
                prob.append(f"uncovered[{i}] is not an object")
                continue
            for k in ("mechanism", "why_each_fails", "nearest"):
                if not u.get(k):
                    prob.append(f"uncovered[{i}] missing {k}")
            near = u.get("nearest") or []
            bad = [c for c in near if c not in P.VALID_CODES]
            if bad:
                prob.append(f"uncovered[{i}] nearest has invalid codes {bad}")

    # Severity must agree with the labels, but only HITs force a non-none severity:
    # a trajectory carrying only PARTIALs (qualified concerns, nothing established) is
    # legitimately 'none' or 'low', and rejecting that would cause spurious retries.
    n_hits = sum(1 for cd, b in seen.items() if b == "hits")
    if not seen and sev != "none":
        prob.append(f"no hits/partials but overall_severity={sev!r} (expected 'none')")
    if n_hits and sev == "none":
        prob.append(f"{n_hits} HIT codes asserted but overall_severity='none'")

    res["ok"] = not prob
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    a = ap.parse_args()
    r = check(a.path)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
