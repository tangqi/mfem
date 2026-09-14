#!/usr/bin/env bash

# Runs the fixed-boundary discontinuous Galerkin Grad-Shafranov solver for
# the nonlinear manufactured-solution problem from Section 7.1.2 of the
# DPG Grad-Shafranov paper.

set -euo pipefail

# Executable
EXE="./DG_fixed_boundary_nonlinear_GS"

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

# GMRES solver
GMRES_MAX_IT=500
GMRES_RTOL=1e-6
GMRES_ATOL=0.0
GMRES_KDIM=10

# Preconditioner
PRECONDITIONER="amg"

# Outputs
OUTPUT_MESH="./solutions/solution_mesh.mesh"
OUTPUT_GF="./solutions/solution_gf.gf"

srun -n 1 "$EXE" \
    -m "$MESH" \
    -rs "$SER_REFINEMENT_LEVELS" \
    -rp "$PAR_REFINEMENT_LEVELS" \
    -o "$ORDER" \
    -nmi "$NEWTON_MAX_IT" \
    -nrtol "$NEWTON_RTOL" \
    -natol "$NEWTON_ATOL" \
    -gmi "$GMRES_MAX_IT" \
    -grtol "$GMRES_RTOL" \
    -gatol "$GMRES_ATOL" \
    -gk "$GMRES_KDIM" \
    -pc "$PRECONDITIONER" \
    -no-pa \
    -om "$OUTPUT_MESH" \
    -og "$OUTPUT_GF" \
    -one \
    -no-vis
