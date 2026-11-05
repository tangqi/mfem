# DG Grad-Shafranov solvers

Fixed-boundary discontinuous Galerkin (SIPG/NIPG) solvers for the Grad-Shafranov
equation in the R-Z plane, built on MFEM:

    -div[(1/R) grad(psi)] = -mu_0 J_phi(R, Z, psi)

## Layout

```
dg-gs/
  DG_GS_fixed_boundary_linear.cpp           linear Solov'ev source (DPG paper Sec. 7.1.1)
  DG_GS_fixed_boundary_nonlinear.cpp        nonlinear manufactured solution (Sec. 7.1.2), Newton with an assembled Jacobian
  DG_GS_fixed_boundary_nonlinear_JFNK.cpp   same nonlinear problem, Jacobian-free Newton-Krylov
  makefile
  meshes/ITER.msh                           ITER-like plasma boundary mesh
  run_scripts/                              example runs: <solver>.sh (1 rank), <solver>_parallel.sh (4 ranks)
  convergence_studies/
    tools/                                  study drivers, CSV collector, 2D error plots
    runs_*/                                 results of one study: CSV, logs/, plots/
```

## Build

MFEM was built following the parallel build instructions at https://mfem.org/building/,
with these dependency versions:

| Dependency | Version |
|------------|---------|
| MFEM       | 4.9     |
| hypre      | 2.26.0  |
| METIS      | 5.1.0   |
| Open MPI   | 5.0.11  |
| GLVis      | 4.5     |

Then build the solvers in this directory:

```
make
```

## Run

Each script runs from any directory and writes its mesh and solution to `solutions/`.

```
bash run_scripts/DG_GS_fixed_boundary_linear.sh
bash run_scripts/DG_GS_fixed_boundary_nonlinear_parallel.sh
```
