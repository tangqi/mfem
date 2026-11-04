#!/usr/bin/env python3
"""2D plots of the numerical solution, exact solution, and pointwise error for
every run of the linear SIPG + CG convergence study.

For each logs/linear_sipg_cg_<config>_o<order>_L<level>.log, samples the saved mesh
and grid function with sample_linear_GS_error (built from the makefile in this
folder) and writes plots/<config>_o<order>_L<level>.png with three panels:
psi_h, psi_exact, and psi_h - psi_exact. The error panel carries a legend with the
solver's global (relative) L2 and Linf errors from linear_sipg_cg_results.csv.

Works from any CWD:
    python3 plot_linear_errors.py
    ONLY=serial_o1 python3 plot_linear_errors.py   # only runs whose name contains this
    KEEP_DATA=1 python3 plot_linear_errors.py      # keep plots/data/*.bin samples
"""

import glob
import os
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# Resolve paths from this script's own location so it works from any CWD.
BASE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE, "logs")
MESH_DIR = os.path.join(BASE, "meshes")
GF_DIR = os.path.join(BASE, "gfs")
PLOT_DIR = os.path.join(BASE, "plots")
DATA_DIR = os.path.join(PLOT_DIR, "data")
CSV = os.path.join(BASE, "linear_sipg_cg_results.csv")
TOOL = os.path.join(BASE, "sample_linear_GS_error")

ONLY = os.environ.get("ONLY", "")
KEEP_DATA = os.environ.get("KEEP_DATA", "0") == "1"

LOG_RE = re.compile(
    r"linear_sipg_cg_(?P<config>[a-z]+)_o(?P<order>\d+)_L(?P<level>\d+)\.log$")
RANKS = {"serial": 1, "parallel": 4}


def _grab(pattern, text, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def run_metrics(results, path, config, order, level):
    """DOFs and global errors: from the results CSV, else parsed from the log."""
    if results is not None:
        row = results[(results["config"] == config) & (results["order"] == order)
                      & (results["level"] == level)]
        if len(row):
            r = row.iloc[0]
            return int(r["dofs"]), r["l2_error"], r["rel_l2_error"], r["linf_error"]
    with open(path) as f:
        text = f.read()
    return (_grab(r"Number of unknowns:\s+(\d+)", text, int),
            _grab(r"\n\s*L2 error:\s+([-\d.eE+]+)", text),
            _grab(r"relative L2 error:\s+([-\d.eE+]+)", text),
            _grab(r"Linf error:\s+([-\d.eE+]+)", text))


def sample(base, config, out):
    """Run the MFEM sampling tool on one saved run and load its binary output."""
    cmd = [TOOL,
           "-m", os.path.join(MESH_DIR, f"{base}_mesh.mesh"),
           "-g", os.path.join(GF_DIR, f"{base}_gf.gf"),
           "-np", str(RANKS[config] if config == "parallel" else 0),
           "-o", out]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    with open(out, "rb") as f:
        npts, ntri, nseg = np.fromfile(f, dtype=np.int64, count=3)
        R, Z, psi_h, psi_ex = np.fromfile(f, dtype=np.float64, count=4*npts).reshape(4, npts)
        tri = np.fromfile(f, dtype=np.int32, count=3*ntri).reshape(ntri, 3)
        rank = np.fromfile(f, dtype=np.int32, count=ntri)
        seg = np.fromfile(f, dtype=np.float64, count=4*nseg).reshape(nseg, 2, 2)
    return R, Z, psi_h, psi_ex, tri, rank, seg


def plot_run(base, config, order, level, metrics, data):
    R, Z, psi_h, psi_ex, tri, rank, seg = data
    dofs, l2, rel_l2, linf = metrics
    err = psi_h - psi_ex
    triang = mtri.Triangulation(R, Z, tri)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), constrained_layout=True)

    # psi_h and psi_exact share one color scale
    vmin = min(psi_h.min(), psi_ex.min())
    vmax = max(psi_h.max(), psi_ex.max())
    for ax, field, title in [(axes[0], psi_h, r"numerical $\psi_h$"),
                             (axes[1], psi_ex, r"exact $\psi$")]:
        tpc = ax.tripcolor(triang, field, shading="gouraud", cmap="viridis",
                           vmin=vmin, vmax=vmax, rasterized=True)
        ax.add_collection(LineCollection(seg, colors="k", linewidths=0.4))
        ax.set_title(title)
    fig.colorbar(tpc, ax=axes[:2], shrink=0.85, label=r"$\psi$")

    # Error, symmetric about 0
    emax = np.abs(err).max()
    tpc = axes[2].tripcolor(triang, err, shading="gouraud", cmap="RdBu_r",
                            vmin=-emax, vmax=emax, rasterized=True)
    axes[2].add_collection(LineCollection(seg, colors="k", linewidths=0.4))
    axes[2].set_title(r"error $\psi_h - \psi$")
    fig.colorbar(tpc, ax=axes[2], shrink=0.85, label=r"$\psi_h - \psi$")

    # Reference box with the solver's global errors
    handles = [Line2D([], [], linestyle="none", label=s) for s in (
        rf"relative $L^2$ error = {rel_l2:.4e}",
        rf"$L^2$ error = {l2:.4e}",
        rf"$L^\infty$ error = {linf:.4e}")]
    axes[2].legend(handles=handles, loc="upper right", handlelength=0,
                   fontsize=9, framealpha=0.9)

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("R")
        ax.set_ylabel("Z")
        ax.autoscale_view()

    ranks = RANKS[config]
    fig.suptitle(f"{config} ({ranks} rank{'s' if ranks > 1 else ''})  ·  P{order}  ·  "
                 f"L{level}  ·  {dofs:,} DOFs" + ("  ·  lines = rank partitions"
                                                 if config == "parallel" else ""))

    out = os.path.join(PLOT_DIR, f"{base}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    subprocess.run(["make", "-s", "-C", BASE, "sample_linear_GS_error"], check=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    results = pd.read_csv(CSV) if os.path.exists(CSV) else None

    logs = sorted(glob.glob(os.path.join(LOG_DIR, "linear_sipg_cg_*_L*.log")))
    for path in logs:
        m = LOG_RE.search(os.path.basename(path))
        if not m:
            continue
        config, order, level = m.group("config"), int(m.group("order")), int(m.group("level"))
        base = f"{config}_o{order}_L{level}"
        if ONLY and ONLY not in base:
            continue

        data_file = os.path.join(DATA_DIR, f"{base}.bin")
        data = sample(base, config, data_file)
        out = plot_run(base, config, order, level,
                       run_metrics(results, path, config, order, level), data)
        if not KEEP_DATA:
            os.remove(data_file)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
