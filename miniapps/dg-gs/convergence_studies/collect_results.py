#!/usr/bin/env python3
"""Collect convergence-study results into a CSV + summary table.

Parses the "Run summary" block that DG_fixed_boundary_nonlinear_GS prints to each
runs/logs/nonlinear_amg_<config>_L<level>.log (alongside this script), writes
results.csv next to this script, and prints, per config, the L2 error and the
observed order p = log2(e_{L-1}/e_L).

Run after run_convergence_study.sh (works from any CWD):
    python3 collect_results.py
"""

import csv
import glob
import math
import os
import re

# Resolve paths from this script's own location so it works from any CWD.
BASE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE, "runs", "logs")
OUT_CSV = os.path.join(BASE, "results.csv")

LOG_RE = re.compile(r"nonlinear_amg_(?P<config>\w+)_L(?P<level>\d+)\.log$")

FIELDS = [
    "config", "ranks", "level", "dofs",
    "l2_error", "rel_l2_error", "linf_error",
    "newton_iters", "gmres_iters_total", "converged", "solve_time_s",
]


def _grab(pattern, text, cast=str, default=None):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else default


def parse_log(path):
    with open(path) as f:
        text = f.read()

    name = os.path.basename(path)
    m = LOG_RE.search(name)
    if not m:
        return None
    config = m.group("config")
    level = int(m.group("level"))

    # Newton history table: rows of "k  gmres_iters  ||R||  rel||R||" between the
    # header and the following blank line. Sum the GMRES column; count the rows.
    gmres_total, newton_iters = 0, 0
    in_table = False
    for line in text.splitlines():
        if line.startswith("  Iteration"):
            in_table = True
            continue
        if in_table:
            parts = line.split()
            if len(parts) == 4 and parts[0].isdigit():
                gmres_total += int(parts[1])
                newton_iters += 1
            else:
                in_table = False

    return {
        "config": config,
        "ranks": _grab(r"MPI ranks:\s+(\d+)", text, int),
        "level": level,
        "dofs": _grab(r"Global DOFs:\s+(\d+)", text, int),
        "l2_error": _grab(r"L2 error:\s+([-\d.eE+]+)", text, float),
        "rel_l2_error": _grab(r"relative L2 error:\s+([-\d.eE+]+)", text, float),
        "linf_error": _grab(r"Linf error:\s+([-\d.eE+]+)", text, float),
        "newton_iters": newton_iters,
        "gmres_iters_total": gmres_total,
        "converged": _grab(r"Converged:\s+(\w+)", text),
        "solve_time_s": _grab(r"Solve time:\s+([-\d.eE+]+)", text, float),
    }


def main():
    rows = [r for p in sorted(glob.glob(os.path.join(LOG_DIR, "nonlinear_amg_*_L*.log")))
            if (r := parse_log(p))]
    if not rows:
        print(f"No logs found in {LOG_DIR}/ (pattern nonlinear_amg_*_L*.log).")
        return

    rows.sort(key=lambda r: (r["config"], r["level"]))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT_CSV} ({len(rows)} rows)\n")

    # Per-config table with observed order p = log2(e_{L-1}/e_L).
    for config in sorted({r["config"] for r in rows}):
        seq = [r for r in rows if r["config"] == config]
        print(f"[{config}]  ranks={seq[0]['ranks']}")
        print(f"  {'L':>2} {'DOFs':>10} {'L2 error':>12} {'order':>7} "
              f"{'Newton':>7} {'GMRES':>7} {'time[s]':>9} conv")
        prev = None
        for r in seq:
            e = r["l2_error"]
            p = ("%.2f" % (math.log(prev / e, 2))) if (prev and e and e > 0) else "  -  "
            print(f"  {r['level']:>2} {r['dofs']:>10} {e:>12.4e} {p:>7} "
                  f"{r['newton_iters']:>7} {r['gmres_iters_total']:>7} "
                  f"{r['solve_time_s']:>9.3f} {r['converged']:>4}")
            prev = e
        print()


if __name__ == "__main__":
    main()
