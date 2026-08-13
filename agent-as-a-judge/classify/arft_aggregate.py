#!/usr/bin/env python3
"""
arft_aggregate.py — roll the per-analysis ARFT classifications up into the stats
deliverables. Pure pandas; no model calls.

Reads   $AAJ_OUT_DIR/<model>/<task>.json      (sparse hits/partials; default ./results)
Writes  $AAJ_OUT_DIR/
          agg.json              rows=[[model, task, {A.1..X.8: 0|1|2}]]
          matrix_long.csv       tidy base table everything else derives from
          SUMMARY.md            pattern x model HIT/PARTIAL matrix + totals
          root_cause_stats.md   stage(A-F,X) x pillar(P1-P4), overall + per model
          by_source_set.md      pattern x source_set (only if your corpus tags one via
                                an optional _MANIFEST.tsv), and W-paper vs other tasks
          cooccurrence.csv      45x45 within-task co-occurrence
          agreement.csv         per-task cross-model agreement (how many models hit
                                each code — most informative on a complete model x
                                task grid, see write_agreement())
          tables.tex            LaTeX for the two headline matrices
          UNCOVERED.md          mechanisms fitting no code, for taxonomy-gap review

Score encoding (do not change): 2 = HIT, 1 = PARTIAL, 0 = miss. Yes, 2 is the stronger
value — see the note in arft_patterns.py. Always reference P.SCORE_* rather than literals.
"""
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arft_classify_cc as c     # noqa: E402
import arft_patterns as P        # noqa: E402

HIT, PART = P.SCORE_HIT, P.SCORE_PARTIAL


# ---------------------------------------------------------------- load

def load_all():
    """-> (records, long_rows, missing). One record per classified (model, task)."""
    recs, long_rows, missing = [], [], []
    for mk in c.MODELS:
        for t in c.discover(mk):
            tid = t["task_id"]
            j = c.OUT_ROOT / mk / f"{tid}.json"
            if not j.exists():
                missing.append((mk, tid))
                continue
            try:
                d = json.loads(j.read_text())
            except Exception:
                missing.append((mk, tid))
                continue
            scores = {code: 0 for code in P.CODES}
            detail = {}
            for bucket, val in (("hits", HIT), ("partials", PART)):
                for it in d.get(bucket) or []:
                    code = it.get("code")
                    if code in P.VALID_CODES:
                        scores[code] = val
                        detail[code] = it
            recs.append({"model": mk, "task": tid, "source_set": t["source_set"],
                         "scores": scores, "raw": d})
            for code in P.CODES:
                if scores[code]:
                    it = detail.get(code, {})
                    long_rows.append({
                        "model": mk, "task": tid, "source_set": t["source_set"],
                        "code": code, "name": P.NAME[code],
                        "stage": P.STAGE_OF[code], "pillar": P.PILLAR_OF[code],
                        "score": scores[code],
                        "verdict": "HIT" if scores[code] == HIT else "PARTIAL",
                        "confidence": it.get("confidence"),
                        "evidence": (it.get("evidence") or "").replace("\n", " "),
                    })
    return recs, long_rows, missing


def task_family(task_id):
    """One example cut: buckets OpenAlex-style paper ids (task_id starting with `W`,
    followed by digits) separately from everything else. This was written for a
    corpus that mixed those with named benchmark tasks (e.g. `echonet_lvef`) — adapt
    or remove this heuristic to match your own task-naming convention."""
    return "W-paper" if task_id.startswith("W") else "named-benchmark"


# ---------------------------------------------------------------- outputs

def write_agg_json(recs, out):
    rows = [[r["model"], r["task"], r["scores"]] for r in recs]
    hit = {code: sum(1 for r in recs if r["scores"][code] == HIT) for code in P.CODES}
    part = {code: sum(1 for r in recs if r["scores"][code] == PART) for code in P.CODES}
    permodel = {}
    for mk in c.MODELS:
        sub = [r for r in recs if r["model"] == mk]
        permodel[mk] = {
            "n": len(sub),
            "hit": {code: sum(1 for r in sub if r["scores"][code] == HIT) for code in P.CODES},
            "part": {code: sum(1 for r in sub if r["scores"][code] == PART) for code in P.CODES},
        }
    (out / "agg.json").write_text(json.dumps(
        {"rows": rows, "hit": hit, "part": part, "permodel": permodel},
        ensure_ascii=False, indent=1))


def _matrix(recs, key, keys, score):
    """code x <key> counts at the given score."""
    m = pd.DataFrame(0, index=P.CODES, columns=keys, dtype=int)
    for r in recs:
        k = r[key]
        if k not in m.columns:
            continue
        for code in P.CODES:
            if r["scores"][code] == score:
                m.at[code, k] += 1
    return m


def write_summary(recs, out):
    hits = _matrix(recs, "model", c.MODELS, HIT)
    parts = _matrix(recs, "model", c.MODELS, PART)
    n_per = {mk: sum(1 for r in recs if r["model"] == mk) for mk in c.MODELS}

    L = [f"# 45-pattern failure matrix — {len(recs)} classified analyses\n",
         f"Models: {', '.join(f'{m}={n_per[m]}' for m in c.MODELS)}\n",
         "Cells are `HIT/PARTIAL` counts. HIT = the analysis presents the failure as "
         "established; PARTIAL = raised but qualified.\n"]

    L.append("\n## Pattern × model\n")
    L.append("| code | name | stage | pillar | " + " | ".join(c.MODELS) + " | ΣHIT | ΣPART | HIT% |")
    L.append("|---|---|---|---|" + "---|" * (len(c.MODELS) + 3))
    tot = len(recs) or 1
    for code in P.CODES:
        cells = " | ".join(f"{hits.at[code, m]}/{parts.at[code, m]}" for m in c.MODELS)
        sh, sp = int(hits.loc[code].sum()), int(parts.loc[code].sum())
        L.append(f"| {code} | {P.NAME[code]} | {P.STAGE_OF[code]} | {P.PILLAR_OF[code]} | "
                 f"{cells} | **{sh}** | {sp} | {100*sh/tot:.1f}% |")

    L.append("\n## Ranked by HIT count\n")
    L.append("| rank | code | name | HIT | PARTIAL | HIT% |")
    L.append("|---|---|---|---|---|---|")
    order = sorted(P.CODES, key=lambda x: (-int(hits.loc[x].sum()), x))
    for i, code in enumerate(order, 1):
        sh, sp = int(hits.loc[code].sum()), int(parts.loc[code].sum())
        L.append(f"| {i} | {code} | {P.NAME[code]} | {sh} | {sp} | {100*sh/tot:.1f}% |")

    L.append("\n## Per-model load\n")
    L.append("| model | n | ΣHIT | ΣPARTIAL | mean HIT/analysis | severity |")
    L.append("|---|---|---|---|---|---|")
    for mk in c.MODELS:
        sub = [r for r in recs if r["model"] == mk]
        sh = int(hits[mk].sum())
        sp = int(parts[mk].sum())
        sev = Counter((r["raw"].get("overall_severity") or "?") for r in sub)
        L.append(f"| {mk} | {len(sub)} | {sh} | {sp} | "
                 f"{sh/max(1,len(sub)):.1f} | {dict(sev)} |")

    (out / "SUMMARY.md").write_text("\n".join(L) + "\n")
    return hits, parts


def write_root_cause(recs, out):
    """stage x pillar HIT tables — overall and per model."""
    def table(sub, title):
        grid = pd.DataFrame(0, index=P.STAGE_ORDER, columns=P.PILLAR_ORDER, dtype=int)
        for r in sub:
            for code in P.CODES:
                if r["scores"][code] == HIT:
                    grid.at[P.STAGE_OF[code], P.PILLAR_OF[code]] += 1
        lines = [f"\n### {title}  (n={len(sub)})\n",
                 "| stage | " + " | ".join(P.PILLAR_ORDER) + " | total |",
                 "|---|" + "---|" * (len(P.PILLAR_ORDER) + 1)]
        for st in P.STAGE_ORDER:
            row = " | ".join(str(grid.at[st, p]) for p in P.PILLAR_ORDER)
            lines.append(f"| {st} {P.STAGES[st]} | {row} | **{int(grid.loc[st].sum())}** |")
        totals = " | ".join(f"**{int(grid[p].sum())}**" for p in P.PILLAR_ORDER)
        lines.append(f"| **total** | {totals} | **{int(grid.values.sum())}** |")
        return lines

    L = ["# Root-cause pillars × lifecycle stage (HIT counts)\n",
         "Pillars roll up to the single systemic root cause, **Metacognitive Deficit**.\n"]
    for p in P.PILLAR_ORDER:
        L.append(f"- **{p} {P.PILLARS[p][0]}** — {P.PILLARS[p][1]}")
    L += table(recs, "Overall")
    for mk in c.MODELS:
        L += table([r for r in recs if r["model"] == mk], mk)
    (out / "root_cause_stats.md").write_text("\n".join(L) + "\n")


def write_by_source_set(recs, out):
    """Pattern prevalence sliced by corpus provenance — separates task-type effects
    from model effects. Two cuts, each skipped if there's nothing to compare:
      - "source_set": whatever distinct values your optional _MANIFEST.tsv assigns.
      - "family": the task_family() heuristic (see its docstring — adapt to your
        own task-naming convention).
    """
    for r in recs:
        r["family"] = task_family(r["task"])

    sets_present = sorted({r["source_set"] for r in recs if r.get("source_set")})
    families_present = sorted({r["family"] for r in recs})

    L = ["# Pattern prevalence by corpus slice\n"]
    any_section = False
    for key, present, title in (("source_set", sets_present, "Source set"),
                                ("family", families_present, "Task family")):
        if len(present) < 2:
            continue
        any_section = True
        hits = _matrix(recs, key, present, HIT)
        n = {k: sum(1 for r in recs if r[key] == k) for k in present}
        L.append(f"\n## {title}\n")
        L.append("| code | name | " + " | ".join(f"{k} (n={n[k]})" for k in present) + " | Δ rate |")
        L.append("|---|---|" + "---|" * (len(present) + 1))
        for code in P.CODES:
            rates = {k: hits.at[code, k] / max(1, n[k]) for k in present}
            cells = " | ".join(f"{hits.at[code,k]} ({100*rates[k]:.0f}%)" for k in present)
            delta = (f"{100*(rates[present[0]]-rates[present[1]]):+.0f} pp"
                     if len(present) == 2 else "-")
            L.append(f"| {code} | {P.NAME[code]} | {cells} | {delta} |")
    if not any_section:
        L.append("\n(no comparison to show — needs either 2+ distinct `source_set` "
                 "values via an optional `_MANIFEST.tsv`, or 2+ task families)\n")
    (out / "by_source_set.md").write_text("\n".join(L) + "\n")


def write_cooccurrence(recs, out):
    co = pd.DataFrame(0, index=P.CODES, columns=P.CODES, dtype=int)
    for r in recs:
        on = [code for code in P.CODES if r["scores"][code] == HIT]
        for code in on:
            co.at[code, code] += 1
        for a, b in combinations(on, 2):
            co.at[a, b] += 1
            co.at[b, a] += 1
    co.to_csv(out / "cooccurrence.csv")

    pairs = sorted(((int(co.at[a, b]), a, b) for a, b in combinations(P.CODES, 2)),
                   reverse=True)[:40]
    L = ["# Top co-occurring pattern pairs (within the same analysis, HIT only)\n",
         "| n | code A | code B | lift |", "|---|---|---|---|"]
    tot = len(recs) or 1
    for n, a, b in pairs:
        pa, pb = co.at[a, a] / tot, co.at[b, b] / tot
        lift = (n / tot) / (pa * pb) if pa and pb else 0
        L.append(f"| {n} | {a} {P.NAME[a]} | {b} {P.NAME[b]} | {lift:.2f} |")
    (out / "cooccurrence_top.md").write_text("\n".join(L) + "\n")


def write_agreement(recs, out):
    """Per task: how many of the models were labelled with each code.

    Separates 'hard task' (most models fail the same way) from 'weak model'
    (one model fails alone). Most informative when your corpus has the same set of
    tasks run through every model (a complete model x task grid) — with a ragged
    corpus, `n_models` per task will vary and the universal/solo counts mean less.
    """
    by_task = defaultdict(list)
    for r in recs:
        by_task[r["task"]].append(r)
    rows = []
    for task, rs in sorted(by_task.items()):
        n = len(rs)
        row = {"task": task, "source_set": rs[0]["source_set"],
               "family": task_family(task), "n_models": n}
        for code in P.CODES:
            row[code] = sum(1 for r in rs if r["scores"][code] == HIT)
        row["universal"] = sum(1 for code in P.CODES if row[code] == n and n > 0)
        row["solo"] = sum(1 for code in P.CODES if row[code] == 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "agreement.csv", index=False)

    L = ["# Cross-model agreement per task\n",
         "`universal` = codes hit by ALL models on this task (task-driven failure). "
         "`solo` = codes hit by exactly one model (model-driven failure).\n",
         "| task | set | n | universal | solo | top shared codes |",
         "|---|---|---|---|---|---|"]
    for _, r in df.sort_values("universal", ascending=False).iterrows():
        top = sorted(((r[code], code) for code in P.CODES), reverse=True)[:4]
        top_s = ", ".join(f"{code}({int(n)})" for n, code in top if n)
        L.append(f"| {r['task']} | {r['source_set']} | {r['n_models']} | "
                 f"{r['universal']} | {r['solo']} | {top_s} |")
    (out / "agreement.md").write_text("\n".join(L) + "\n")


def write_tex(recs, hits, parts, out):
    tot = len(recs) or 1
    esc = lambda s: (s.replace("&", r"\&").replace("_", r"\_")
                      .replace("\"", "").replace("%", r"\%"))
    n_models = len({r["model"] for r in recs})
    L = [r"% Auto-generated by arft_aggregate.py — do not edit by hand.",
         r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{ARFT pattern prevalence across %d agent trajectories "
         r"(%d models). HIT = failure established in the "
         r"trajectory analysis.}" % (len(recs), n_models),
         r"\begin{tabular}{llrrr}", r"\toprule",
         r"Code & Pattern & HIT & PARTIAL & HIT\% \\", r"\midrule"]
    cur = None
    for code in sorted(P.CODES, key=lambda x: (P.STAGE_ORDER.index(P.STAGE_OF[x]), x)):
        st = P.STAGE_OF[code]
        if st != cur:
            cur = st
            L.append(r"\multicolumn{5}{l}{\textit{%s. %s}} \\" % (st, esc(P.STAGES[st])))
        sh, sp = int(hits.loc[code].sum()), int(parts.loc[code].sum())
        L.append(f"{code} & {esc(P.NAME[code])} & {sh} & {sp} & {100*sh/tot:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "", ""]

    L += [r"\begin{table}[t]", r"\centering", r"\small",
          r"\caption{Root-cause pillars by lifecycle stage (HIT counts).}",
          r"\begin{tabular}{l" + "r" * (len(P.PILLAR_ORDER) + 1) + "}", r"\toprule",
          "Stage & " + " & ".join(P.PILLAR_ORDER) + r" & Total \\", r"\midrule"]
    grid = pd.DataFrame(0, index=P.STAGE_ORDER, columns=P.PILLAR_ORDER, dtype=int)
    for r in recs:
        for code in P.CODES:
            if r["scores"][code] == HIT:
                grid.at[P.STAGE_OF[code], P.PILLAR_OF[code]] += 1
    for st in P.STAGE_ORDER:
        cells = " & ".join(str(grid.at[st, p]) for p in P.PILLAR_ORDER)
        L.append(f"{st} {esc(P.STAGES[st])} & {cells} & {int(grid.loc[st].sum())} \\\\")
    L.append(r"\midrule")
    L.append("Total & " + " & ".join(str(int(grid[p].sum())) for p in P.PILLAR_ORDER)
             + f" & {int(grid.values.sum())} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tables.tex").write_text("\n".join(L) + "\n")


def write_uncovered(recs, out):
    L = ["# Mechanisms that fit no existing pattern (UNCOVERED)\n",
         "> Each requires `why_each_fails` refuting the nearest codes. "
         "Use to decide whether the taxonomy needs extending.\n"]
    n = 0
    for r in recs:
        for u in r["raw"].get("uncovered") or []:
            n += 1
            L.append(f"\n### [{r['model']}] {r['task']} — {u.get('mechanism','?')}\n"
                     f"- **description**: {u.get('description','')}\n"
                     f"- **nearest**: {u.get('nearest')}\n"
                     f"- **why each fails**: {u.get('why_each_fails','')}\n"
                     f"- **evidence**: {u.get('evidence','')}\n"
                     f"- **suggested new code**: {u.get('suggested_new_code','')}\n")
    L.insert(2, f"\nTotal uncovered candidates: **{n}**\n")
    (out / "UNCOVERED.md").write_text("\n".join(L) + "\n")
    return n


def write_iron_rules(recs, out):
    """Iron-rule citations — a cross-check on the F/X-family labels."""
    cnt = Counter()
    files = Counter()
    for r in recs:
        rules = set(x for x in (r["raw"].get("iron_rules_cited") or [])
                    if isinstance(x, int))
        for x in rules:
            files[x] += 1
        for x in (r["raw"].get("iron_rules_cited") or []):
            if isinstance(x, int):
                cnt[x] += 1
    L = ["# Iron-rule citations across the corpus\n",
         "Cross-check: Iron Rule 9 (\"identified the mechanism\" != \"addressed it\") "
         "should track F.4 closely.\n",
         "| rule | analyses citing it |", "|---|---|"]
    for rule, nf in sorted(files.items(), key=lambda kv: -kv[1]):
        L.append(f"| Iron Rule {rule} | {nf} |")
    f4 = sum(1 for r in recs if r["scores"]["F.4"] == HIT)
    L.append(f"\nF.4 HIT count: **{f4}** · Iron Rule 9 cited in **{files.get(9,0)}** "
             f"analyses (of {len(recs)}).\n")
    (out / "iron_rules.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- main

def main():
    out = c.OUT_ROOT
    out.mkdir(parents=True, exist_ok=True)
    recs, long_rows, missing = load_all()
    if not recs:
        print("no classifications found under", out)
        return 1

    pd.DataFrame(long_rows).to_csv(out / "matrix_long.csv", index=False)
    write_agg_json(recs, out)
    hits, parts = write_summary(recs, out)
    write_root_cause(recs, out)
    write_by_source_set(recs, out)
    write_cooccurrence(recs, out)
    write_agreement(recs, out)
    write_tex(recs, hits, parts, out)
    n_unc = write_uncovered(recs, out)
    write_iron_rules(recs, out)

    print(f"[agg45] classified={len(recs)}/800  missing={len(missing)}  "
          f"labels={len(long_rows)}  uncovered={n_unc}")
    if missing:
        print(f"[agg45] WARNING: {len(missing)} unclassified, e.g. {missing[:5]}")
        print("[agg45] tables above are over the classified subset only.")
    print(f"[agg45] wrote 12 artifacts to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
