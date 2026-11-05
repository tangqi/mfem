#!/usr/bin/env bash
#
# AMG-baseline convergence study for the nonlinear DG Grad-Shafranov solver.
#
# Sweeps uniform mesh refinement L = 0..5 (h halves each level) at fixed order 1
# with the BoomerAMG preconditioner, running each resolution both serial (1 rank)
# and parallel (4 ranks). Per-run artifacts go under convergence_studies/runs/:
# terminal logs in runs/logs/, meshes in runs/meshes/, grid functions in runs/gfs/.
# Parse the logs with tools/collect_results.py afterwards.
#
# Lives in dg-gs/convergence_studies/tools/ and can be launched from anywhere; it cd's
# to the solver directory itself. Launches runs with mpirun.
#
# Usage:
#   ./run_convergence_study.sh            # full sweep, L = 0 1 2 3 4
#   LEVELS="1" ./run_convergence_study.sh # dry run, just L = 1

set -euo pipefail

# This script lives in dg-gs/convergence_studies/tools/ but drives the binary in dg-gs/.
# Anchor to the script's own location and run from the solver directory so the
# EXE/MESH relative paths below resolve as before.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Executable and fixed problem definition.
# Defaults to the explicit-Jacobian solver (this driver passes its --krylov-* /
# -kmi/-krtol/-katol flags). EXE and RUNS_SUBDIR can be overridden via env vars to
# benchmark a different solver/preconditioner without clobbering a previous study, e.g.:
#   EXE=./DG_GS_fixed_boundary_nonlinear \
#   RUNS_SUBDIR=runs_amg_jk ./run_convergence_study.sh
# (The JFNK binary uses the older --gmres-* flag names and is not driven by this script.)
EXE="${EXE:-./DG_GS_fixed_boundary_nonlinear}"
MESH="meshes/ITER.msh"
# Polynomial order(s) to sweep (override with the ORDERS env var), e.g.
#   ORDERS="1 2 3" ... ./run_convergence_study.sh
ORDERS="${ORDERS:-1}"
PC="amg"

# Krylov solver(s) passed to the explicit-Jacobian binary via -ks. A space-separated
# list sweeps solvers (e.g. "gmres cg"); empty => the flag is NOT passed, so this stays
# compatible with solvers that lack -ks and with the explicit solver's own default.
KRYLOV="${KRYLOV:-}"

# Parallel configurations to run: 'serial' (1 rank), 'parallel' (4 ranks), or both.
CONFIGS="${CONFIGS:-serial parallel}"

# Solver tolerances (defaults = the baseline reference; override any via env, e.g.
#   NEWTON_RTOL=1e-10 KRYLOV_RTOL=1e-10 ... ./run_convergence_study.sh )
# KRYLOV_* apply to whichever Krylov solver (-ks) is used; GMRES_KDIM is GMRES-only.
NEWTON_MAX_IT="${NEWTON_MAX_IT:-20}"
NEWTON_RTOL="${NEWTON_RTOL:-1e-8}"
NEWTON_ATOL="${NEWTON_ATOL:-1e-12}"
KRYLOV_MAX_IT="${KRYLOV_MAX_IT:-500}"
KRYLOV_RTOL="${KRYLOV_RTOL:-1e-6}"
KRYLOV_ATOL="${KRYLOV_ATOL:-0.0}"
GMRES_KDIM="${GMRES_KDIM:-10}"

# Refinement levels to sweep (override with the LEVELS env var for a dry run)
LEVELS="${LEVELS:-0 1 2 3 4 5}"

RUNS_DIR="convergence_studies/${RUNS_SUBDIR:-runs_baseline}"
LOGDIR="$RUNS_DIR/logs"
MESHDIR="$RUNS_DIR/meshes"
GFDIR="$RUNS_DIR/gfs"
mkdir -p "$LOGDIR" "$MESHDIR" "$GFDIR"

# run_case <config:serial|parallel> <ks> <order> <level>
# ks="" => no -ks flag and no _ks filename tag (backward compatible).
run_case () {
   local cfg="$1" ks="$2" order="$3" L="$4"

   local ranks save_flag
   case "$cfg" in
      serial)   ranks=1; save_flag="-one" ;;
      parallel) ranks=4; save_flag="-sep" ;;
      *) echo "unknown config: $cfg" >&2; return 1 ;;
   esac

   # Encode the Krylov solver in the names so multiple solvers share one folder.
   local ks_tag="" ; local ks_args=()
   [ -n "$ks" ] && { ks_tag="_ks${ks}"; ks_args=(-ks "$ks"); }

   local base="${cfg}${ks_tag}_o${order}_L${L}"
   local om="$MESHDIR/${base}_mesh.mesh"
   local og="$GFDIR/${base}_gf.gf"
   local log="$LOGDIR/nonlinear_amg_${base}.log"

   echo ">>> ${cfg}  ks=${ks:-<default>}  order=${order}  L=${L}  ranks=${ranks}  ->  ${log}"
   mpirun -np "$ranks" "$EXE" \
      -m "$MESH" \
      -rs 0 \
      -rp "$L" \
      -o "$order" \
      -nmi "$NEWTON_MAX_IT" \
      -nrtol "$NEWTON_RTOL" \
      -natol "$NEWTON_ATOL" \
      -kmi "$KRYLOV_MAX_IT" \
      -krtol "$KRYLOV_RTOL" \
      -katol "$KRYLOV_ATOL" \
      -gk "$GMRES_KDIM" \
      -pc "$PC" \
      "${ks_args[@]}" \
      -no-pa \
      -om "$om" \
      -og "$og" \
      "$save_flag" \
      -no-vis \
      > "$log" 2>&1
}

# Krylov slots: empty KRYLOV => one slot with no solver tag/flag (backward compatible).
if [ -z "$KRYLOV" ]; then kry_list=(""); else kry_list=($KRYLOV); fi

echo "=== Convergence study: pc=${PC}, krylov: ${KRYLOV:-<default>}, configs: ${CONFIGS}, orders: ${ORDERS}, levels: ${LEVELS} ==="
for ks in "${kry_list[@]}"; do
   for o in $ORDERS; do
      for L in $LEVELS; do
         for cfg in $CONFIGS; do
            run_case "$cfg" "$ks" "$o" "$L"
         done
      done
   done
done
echo "=== Done. Parse with: python3 convergence_studies/tools/collect_results.py ==="
