#!/usr/bin/env python3
"""arft_status.py — how many analyses in the corpus have a QA-passing ARFT
classification. Exit 0 iff all do. Loop-termination signal for run_all_arft_*.sh."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arft_classify_cc as c
import arft_qa_check as qa


def main():
    allok = True
    lines = []
    tot_pass = tot_all = 0
    for mk in c.MODELS:
        tasks = c.discover(mk)
        npass, rem = 0, []
        for t in tasks:
            j = c.OUT_ROOT / mk / f"{t['task_id']}.json"
            if j.exists() and qa.check(j)["ok"]:
                npass += 1
            else:
                rem.append(t["task_id"])
        tot_pass += npass
        tot_all += len(tasks)
        if rem:
            allok = False
        shown = ",".join(rem[:12]) + ("…" if len(rem) > 12 else "")
        lines.append(f"{mk:18} {npass:3}/{len(tasks):3} classified; remaining({len(rem)})={shown}")
    lines.append(f"{'TOTAL':18} {tot_pass:3}/{tot_all:3}")
    out = "\n".join(lines)
    print(out)
    c.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (c.OUT_ROOT / "STATUS.txt").write_text(out + "\n")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
