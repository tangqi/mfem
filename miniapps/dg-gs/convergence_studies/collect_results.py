#!/usr/bin/env python3
"""Collect convergence-study results into a CSV + summary table.

Parses the "Run summary" block that DG_fixed_boundary_nonlinear_GS prints to each
runs/logs/nonlinear_amg_<config>_L<level>.log (alongside this script), writes
AMG_baseline_results.csv next to this script, and prints, per config, the L2 error
and the observed order p = log2(e_{L-1}/e_L).

Run after run_convergence_study.sh (works from any CWD):
    python3 collect_results.py

With STUDY=linear, instead parses the linear_sipg_cg_<config>_o<order>_L<level>.log
files written by run_linear_convergence_study.sh and only writes the CSV (no tables):
    STUDY=linear python3 collect_results.py
"""

import csv
import glob
import math
import os
import re

# Resolve paths from this script's own location so it works from any CWD.
# RUNS_SUBDIR (which run directory to read) and OUT_NAME (output CSV name) can be
# overridden via env vars to collect a different study without clobbering another
# study's CSV, e.g.:
#   RUNS_SUBDIR=runs_amg_jk OUT_NAME=AMG_Jk_explicit_jacobian_results.csv \
#   python collect_results.py
BASE = os.path.dirname(os.path.abspath(__file__))
# STUDY selects which solver's logs to parse: 'nonlinear' (default) or 'linear'.
STUDY = os.environ.get("STUDY", "nonlinear")
LINEAR = STUDY == "linear"
RUNS_SUBDIR = os.environ.get("RUNS_SUBDIR", "runs_linear_sipg_cg" if LINEAR else "runs_baseline")
LOG_DIR = os.path.join(BASE, RUNS_SUBDIR, "logs")
# The results CSV lives inside the study's own run folder.
OUT_CSV = os.path.join(BASE, RUNS_SUBDIR, os.environ.get(
    "OUT_NAME", "linear_sipg_cg_results.csv" if LINEAR else "AMG_baseline_results.csv"))

# config is lowercase letters (serial/parallel); the "_ks<solver>" and "_o<order>"
# tags are optional, so this matches the h-studies (nonlinear_amg_serial_L0.log), the
# order study (nonlinear_amg_serial_o2_L0.log), and the cg-vs-gmres study
# (nonlinear_amg_parallel_kscg_o2_L0.log). krylov/order come from the log content.
LOG_RE = re.compile(
    r"nonlinear_amg_(?P<config>[a-z]+)(?:_ks(?P<krylov>[a-z]+))?"
    r"(?:_o(?P<order>\d+))?_L(?P<level>\d+)\.log$")

FIELDS = [
    "config", "ranks", "level", "order", "krylov", "dofs",
    "l2_error", "rel_l2_error", "linf_error",
    "newton_iters", "krylov_iters_total", "converged", "solve_time_s",
]

# Linear study logs: linear_sipg_cg_<config>_o<order>_L<level>.log
LINEAR_LOG_RE = re.compile(
    r"linear_sipg_cg_(?P<config>[a-z]+)_o(?P<order>\d+)_L(?P<level>\d+)\.log$")

LINEAR_FIELDS = [
    "config", "ranks", "level", "order", "dofs",
    "l2_error", "rel_l2_error", "linf_error",
    "cg_iters", "converged", "solve_time_s",
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

    # Newton history table: rows of "k  krylov_iters  ||R||  rel||R||" between the
    # header and the following blank line. Sum the Krylov column; count the rows.
    krylov_total, newton_iters = 0, 0
    in_table = False
    for line in text.splitlines():
        if line.startswith("  Iteration"):
            in_table = True
            continue
        if in_table:
            parts = line.split()
            if len(parts) == 4 and parts[0].isdigit():
                krylov_total += int(parts[1])
                newton_iters += 1
            else:
                in_table = False

    return {
        "config": config,
        "ranks": _grab(r"MPI ranks:\s+(\d+)", text, int),
        "level": level,
        "order": _grab(r"Polynomial order:\s+(\d+)", text, int),
        "krylov": _grab(r"Krylov solver:\s+(\w+)", text),
        "dofs": _grab(r"Global DOFs:\s+(\d+)", text, int),
        "l2_error": _grab(r"L2 error:\s+([-\d.eE+]+)", text, float),
        "rel_l2_error": _grab(r"relative L2 error:\s+([-\d.eE+]+)", text, float),
        "linf_error": _grab(r"Linf error:\s+([-\d.eE+]+)", text, float),
        "newton_iters": newton_iters,
        "krylov_iters_total": krylov_total,
        "converged": _grab(r"Converged:\s+(\w+)", text),
        "solve_time_s": _grab(r"Solve time:\s+([-\d.eE+]+)", text, float),
    }


def parse_linear_log(path):
    with open(path) as f:
        text = f.read()

    m = LINEAR_LOG_RE.search(os.path.basename(path))
    if not m:
        return None
    config = m.group("config")

    # The linear solver prints no run summary: CG iterations come from the last
    # "Iteration : N" line of the PCG output, convergence from the absence of
    # "PCG: No convergence!", and ranks from the config.
    iters = re.findall(r"Iteration :\s+(\d+)", text)

    return {
        "config": config,
        "ranks": {"serial": 1, "parallel": 4}.get(config),
        "level": int(m.group("level")),
        "order": int(m.group("order")),
        "dofs": _grab(r"Number of unknowns:\s+(\d+)", text, int),
        "l2_error": _grab(r"\n\s*L2 error:\s+([-\d.eE+]+)", text, float),
        "rel_l2_error": _grab(r"relative L2 error:\s+([-\d.eE+]+)", text, float),
        "linf_error": _grab(r"Linf error:\s+([-\d.eE+]+)", text, float),
        "cg_iters": int(iters[-1]) if iters else None,
        "converged": "PCG: No convergence!" not in text and bool(iters),
        "solve_time_s": _grab(r"Linear solve time:\s+([-\d.eE+]+)", text, float),
    }


def main_linear():
    rows = [r for p in sorted(glob.glob(os.path.join(LOG_DIR, "linear_sipg_cg_*_L*.log")))
            if (r := parse_linear_log(p))]
    if not rows:
        print(f"No logs found in {LOG_DIR}/ (pattern linear_sipg_cg_*_L*.log).")
        return

    rows.sort(key=lambda r: (r["config"], r["order"], r["level"]))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LINEAR_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT_CSV} ({len(rows)} rows)")


def main():
    rows =[r for p in sorted(glob.glob(os.path.join(LOG_DIR, "nonlinear_amg_*_L*.log")))
            if (r := parse_log(p))]
    if not rows:
        print(f"No logs found in {LOG_DIR}/ (pattern nonlinear_amg_*_L*.log).")
        return

    rows.sort(key=lambda r: (r["config"], r["krylov"] or "", r["order"] or 0, r["level"]))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT_CSV} ({len(rows)} rows)\n")

    configs = sorted({r["config"] for r in rows})
    krylovs = sorted({r["krylov"] or "" for r in rows})
    orders = sorted({r["order"] or 0 for r in rows})
    levels = sorted({r["level"] for r in rows})

    def pick(config, krylov, order, level):
        return next((r for r in rows if r["config"] == config
                     and (r["krylov"] or "") == krylov
                     and (r["order"] or 0) == order and r["level"] == level), None)

    # Per (config, krylov, order): h-convergence table, rate p = log2(e_{L-1}/e_L).
    for config in configs:
        for kry in krylovs:
            for order in orders:
                seq = sorted((r for r in rows if r["config"] == config
                              and (r["krylov"] or "") == kry
                              and (r["order"] or 0) == order), key=lambda r: r["level"])
                if not seq:
                    continue
                label = f"[{config}]  ranks={seq[0]['ranks']}  order={order}"
                if kry:
                    label += f"  krylov={kry}"
                print(label)
                print(f"  {'L':>2} {'DOFs':>10} {'L2 error':>12} {'rate':>6} "
                      f"{'Newton':>7} {'Krylov':>7} {'time[s]':>9} conv")
                prev = None
                for r in seq:
                    e = r["l2_error"]
                    rate = ("%.2f" % math.log(prev / e, 2)) if (prev and e and e > 0) else "  -  "
                    print(f"  {r['level']:>2} {r['dofs']:>10} {e:>12.4e} {rate:>6} "
                          f"{r['newton_iters']:>7} {r['krylov_iters_total']:>7} "
                          f"{r['solve_time_s']:>9.3f} {r['converged']:>4}")
                    prev = e
                print()

    # "Relative to order <base>" summary at matched mesh, per config/krylov.
    base = orders[0]
    cfg = "serial" if "serial" in configs else configs[0]
    if len(orders) > 1:
        for kry in krylovs:
            tag = f", krylov={kry}" if kry else ""
            print(f"Relative to order {base} ({cfg}{tag}, same mesh level):")
            print(f"  {'L':>2} {'order':>5} {'DOFs':>10} {'L2 ratio':>10} {'time ratio':>11}")
            for level in levels:
                ref = pick(cfg, kry, base, level)
                if not ref:
                    continue
                for order in orders:
                    r = pick(cfg, kry, order, level)
                    if not r:
                        continue
                    l2r = r["l2_error"] / ref["l2_error"] if ref["l2_error"] else float("nan")
                    tr = r["solve_time_s"] / ref["solve_time_s"] if ref["solve_time_s"] else float("nan")
                    print(f"  {level:>2} {order:>5} {r['dofs']:>10} {l2r:>10.3g} {tr:>11.3g}")
            print()

    # CG vs GMRES comparison (when both solvers are present).
    if "cg" in krylovs and "gmres" in krylovs:
        for config in configs:
            print(f"CG vs GMRES  [{config}]  (Krylov iters total & solve time; ratio = cg/gmres):")
            print(f"  {'ord':>3} {'L':>2} {'DOFs':>10} "
                  f"{'gmres_it':>9} {'cg_it':>6} {'it r':>5} "
                  f"{'gmres_t':>9} {'cg_t':>9} {'t r':>5}  L2")
            for order in orders:
                for level in levels:
                    g = pick(config, "gmres", order, level)
                    c = pick(config, "cg", order, level)
                    if not (g and c):
                        continue
                    gi, ci = g["krylov_iters_total"], c["krylov_iters_total"]
                    gt, ct = g["solve_time_s"], c["solve_time_s"]
                    itr = ci / gi if gi else float("nan")
                    tr = ct / gt if gt else float("nan")
                    l2ok = ("ok" if g["l2_error"] and
                            abs(c["l2_error"] - g["l2_error"]) <= 1e-6 * abs(g["l2_error"])
                            else "DIFF")
                    print(f"  {order:>3} {level:>2} {g['dofs']:>10} "
                          f"{gi:>9} {ci:>6} {itr:>5.2f} "
                          f"{gt:>9.3f} {ct:>9.3f} {tr:>5.2f}  {l2ok}")
            print()


if __name__ == "__main__":
    main_linear() if LINEAR else main()
