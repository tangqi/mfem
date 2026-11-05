#!/usr/bin/env bash

# Runs the fixed-boundary discontinuous Galerkin Grad-Shafranov solver for
# the linearized Solov'ev source term.

set -euo pipefail

# Run from the solver directory (one level up) so the relative paths below resolve
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

# Executable
EXE="./DG_GS_fixed_boundary_linear"

# Input mesh
MESH="meshes/ITER.msh"

# Discretization
SER_REFINEMENT_LEVELS=1
PAR_REFINEMENT_LEVELS=2
ORDER=1

# Outputs
OUTPUT_MESH="./solutions/solution_mesh.mesh"
OUTPUT_GF="./solutions/solution_gf.gf"

mpirun -np 1 "$EXE" \
    -m "$MESH" \
    -rs "$SER_REFINEMENT_LEVELS" \
    -rp "$PAR_REFINEMENT_LEVELS" \
    -o "$ORDER" \
    -om "$OUTPUT_MESH" \
    -og "$OUTPUT_GF" \
    -one \
    -no-vis
