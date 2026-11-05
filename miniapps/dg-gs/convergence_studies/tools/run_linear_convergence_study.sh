#!/usr/bin/env bash
#
# SIPG + CG convergence study for the linear DG Grad-Shafranov solver.
#
# Sweeps uniform mesh refinement L = 0..5 (h halves each level) at fixed order 1
# with SIPG (sigma = -1, which selects CG + BoomerAMG in the solver), running each
# resolution both serial (1 rank) and parallel (4 ranks). Per-run artifacts go under
# convergence_studies/runs_linear_sipg_cg/: terminal logs in logs/, meshes in meshes/,
# grid functions in gfs/. Collect the logs into a CSV afterwards with:
#   STUDY=linear python3 convergence_studies/tools/collect_results.py
#
# Lives in dg-gs/convergence_studies/tools/ and can be launched from anywhere; it cd's
# to the solver directory itself. Launches runs with mpirun.
#
# Usage:
#   ./run_linear_convergence_study.sh            # full sweep, L = 0 1 2 3 4 5
#   LEVELS="0 1" ./run_linear_convergence_study.sh # dry run, just L = 0 1

set -euo pipefail

# Anchor to the script's own location and run from the solver directory so the
# EXE/MESH relative paths below resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Executable and fixed problem definition
EXE="${EXE:-./DG_GS_fixed_boundary_linear}"
MESH="meshes/ITER.msh"
ORDER="${ORDER:-1}"

# SIPG (the solver uses CG for sigma = -1)
SIGMA=-1

# Parallel configurations to run: 'serial' (1 rank), 'parallel' (4 ranks), or both.
CONFIGS="${CONFIGS:-serial parallel}"

# Refinement levels to sweep (override with the LEVELS env var for a dry run)
LEVELS="${LEVELS:-0 1 2 3 4 5}"

RUNS_DIR="convergence_studies/${RUNS_SUBDIR:-runs_linear_sipg_cg}"
LOGDIR="$RUNS_DIR/logs"
MESHDIR="$RUNS_DIR/meshes"
GFDIR="$RUNS_DIR/gfs"
mkdir -p "$LOGDIR" "$MESHDIR" "$GFDIR"

# run_case <config:serial|parallel> <level>
run_case () {
   local cfg="$1" L="$2"

   local ranks save_flag
   case "$cfg" in
      serial)   ranks=1; save_flag="-one" ;;
      parallel) ranks=4; save_flag="-sep" ;;
      *) echo "unknown config: $cfg" >&2; return 1 ;;
   esac

   local base="${cfg}_o${ORDER}_L${L}"
   local om="$MESHDIR/${base}_mesh.mesh"
   local og="$GFDIR/${base}_gf.gf"
   local log="$LOGDIR/linear_sipg_cg_${base}.log"

   echo ">>> ${cfg}  order=${ORDER}  L=${L}  ranks=${ranks}  ->  ${log}"
   mpirun -np "$ranks" "$EXE" \
      -m "$MESH" \
      -rs 0 \
      -rp "$L" \
      -o "$ORDER" \
      -s "$SIGMA" \
      -om "$om" \
      -og "$og" \
      "$save_flag" \
      -no-vis \
      > "$log" 2>&1
}

echo "=== Linear convergence study: SIPG + CG, configs: ${CONFIGS}, order: ${ORDER}, levels: ${LEVELS} ==="
for L in $LEVELS; do
   for cfg in $CONFIGS; do
      run_case "$cfg" "$L"
   done
done
echo "=== Done. Collect with: STUDY=linear python3 convergence_studies/tools/collect_results.py ==="
