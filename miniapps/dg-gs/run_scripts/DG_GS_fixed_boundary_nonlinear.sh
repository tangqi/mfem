#!/usr/bin/env bash

# Runs the fixed-boundary discontinuous Galerkin Grad-Shafranov solver for
# the nonlinear manufactured-solution problem from Section 7.1.2 of the
# DPG Grad-Shafranov paper, using Newton's method with an explicitly
# assembled Jacobian.

set -euo pipefail

# Run from the solver directory (one level up) so the relative paths below resolve
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

# Executable
EXE="./DG_GS_fixed_boundary_nonlinear"

# Input mesh
MESH="meshes/ITER.msh"

# Discretization
SER_REFINEMENT_LEVELS=1
PAR_REFINEMENT_LEVELS=1
ORDER=1

# Newton solver
NEWTON_MAX_IT=20
NEWTON_RTOL=1e-8
NEWTON_ATOL=1e-12

# Krylov solver ('gmres' or 'cg')
KRYLOV_SOLVER="gmres"
KRYLOV_MAX_IT=500
KRYLOV_RTOL=1e-6
KRYLOV_ATOL=0.0
GMRES_KDIM=30

# Preconditioner
PRECONDITIONER="amg"

# Outputs
OUTPUT_MESH="./solutions/solution_mesh.mesh"
OUTPUT_GF="./solutions/solution_gf.gf"

mpirun -np 1 "$EXE" \
    -m "$MESH" \
    -rs "$SER_REFINEMENT_LEVELS" \
    -rp "$PAR_REFINEMENT_LEVELS" \
    -o "$ORDER" \
    -nmi "$NEWTON_MAX_IT" \
    -nrtol "$NEWTON_RTOL" \
    -natol "$NEWTON_ATOL" \
    -ks "$KRYLOV_SOLVER" \
    -kmi "$KRYLOV_MAX_IT" \
    -krtol "$KRYLOV_RTOL" \
    -katol "$KRYLOV_ATOL" \
    -gk "$GMRES_KDIM" \
    -pc "$PRECONDITIONER" \
    -om "$OUTPUT_MESH" \
    -og "$OUTPUT_GF" \
    -one \
    -no-vis
