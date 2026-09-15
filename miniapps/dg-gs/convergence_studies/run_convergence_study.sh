#!/usr/bin/env bash
#
# AMG-baseline convergence study for the nonlinear DG Grad-Shafranov solver.
#
# Sweeps uniform mesh refinement L = 0..4 (h halves each level) at fixed order 1
# with the BoomerAMG preconditioner, running each resolution both serial (1 rank)
# and parallel (4 ranks). Per-run artifacts go under convergence_studies/runs/:
# terminal logs in runs/logs/, meshes in runs/meshes/, grid functions in runs/gfs/.
# Parse the logs with collect_results.py afterwards.
#
# Lives in dg-gs/convergence_studies/ and can be launched from anywhere; it cd's
# to the solver directory itself. Must run inside a Slurm allocation with >= 4
# tasks (e.g. `salloc -n 4 ...`), since it calls srun.
#
# Usage:
#   ./run_convergence_study.sh            # full sweep, L = 0 1 2 3 4
#   LEVELS="1" ./run_convergence_study.sh # dry run, just L = 1

set -euo pipefail

# This script lives in dg-gs/convergence_studies/ but drives the binary in dg-gs/.
# Anchor to the script's own location and run from the solver directory so the
# EXE/MESH relative paths below resolve as before.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Executable and fixed problem definition
EXE="./DG_fixed_boundary_nonlinear_GS"
MESH="meshes/ITER.msh"
ORDER=1
PC="amg"

# Baseline solver tolerances (kept fixed = the reference for future preconditioners)
NEWTON_MAX_IT=20
NEWTON_RTOL=1e-8
NEWTON_ATOL=1e-12
GMRES_MAX_IT=500
GMRES_RTOL=1e-6
GMRES_ATOL=0.0
GMRES_KDIM=10

# Refinement levels to sweep (override with the LEVELS env var for a dry run)
LEVELS="${LEVELS:-0 1 2 3 4}"

RUNS_DIR="convergence_studies/runs"
LOGDIR="$RUNS_DIR/logs"
MESHDIR="$RUNS_DIR/meshes"
GFDIR="$RUNS_DIR/gfs"
mkdir -p "$LOGDIR" "$MESHDIR" "$GFDIR"

# run_case <ranks> <tag> <save-flag> <level>
run_case () {
   local ranks="$1" tag="$2" save_flag="$3" L="$4"

   local om="$MESHDIR/${tag}_L${L}_mesh.mesh"
   local og="$GFDIR/${tag}_L${L}_gf.gf"
   local log="$LOGDIR/nonlinear_amg_${tag}_L${L}.log"

   echo ">>> ${tag}  L=${L}  ranks=${ranks}  ->  ${log}"
   srun -n "$ranks" "$EXE" \
      -m "$MESH" \
      -rs 0 \
      -rp "$L" \
      -o "$ORDER" \
      -nmi "$NEWTON_MAX_IT" \
      -nrtol "$NEWTON_RTOL" \
      -natol "$NEWTON_ATOL" \
      -gmi "$GMRES_MAX_IT" \
      -grtol "$GMRES_RTOL" \
      -gatol "$GMRES_ATOL" \
      -gk "$GMRES_KDIM" \
      -pc "$PC" \
      -no-pa \
      -om "$om" \
      -og "$og" \
      "$save_flag" \
      -no-vis \
      > "$log" 2>&1
}

echo "=== Convergence study: order ${ORDER}, pc=${PC}, levels: ${LEVELS} ==="
for L in $LEVELS; do
   run_case 1 serial   -one "$L"
   run_case 4 parallel -sep "$L"
done
echo "=== Done. Parse with: python3 collect_results.py ==="
