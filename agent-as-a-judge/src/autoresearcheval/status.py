#!/usr/bin/env python3
"""arft_status.py — how many analyses in the corpus have a QA-passing ARFT
classification. Exit 0 iff all do. Loop-termination signal for run_all_arft_*.sh."""
import sys
from pathlib import Path

from . import config
from . import classify_cc as c
from . import label_qa as qa


def main():
    allok = True
    lines = []
    tot_pass = tot_all = 0
    for mk in config.models():
        tasks = c.discover(mk)
        npass, rem = 0, []
        for t in tasks:
            j = config.out_dir() / mk / f"{t['task_id']}.json"
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
    config.out_dir().mkdir(parents=True, exist_ok=True)
    (config.out_dir() / "STATUS.txt").write_text(out + "\n")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
