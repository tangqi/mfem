#ifndef GS_TEST_UTILS_HPP
#define GS_TEST_UTILS_HPP

#include "mfem.hpp"
#include "gs.hpp"

using namespace mfem;


// Prints the entries of a vector to 14 digits of precision.
void Print_(const Vector &y);


// Compare entry-wise differences (Mat) between analytic Jacobian (M1) and
// finite-difference aproximation (M2) matrices.
void CompareSparseMatrices(SparseMatrix *Mat, SparseMatrix *M1, SparseMatrix *M2);


// Tests gradients and Jacobians of SysOperator numerically and
// compares to their finite difference approximations.
void TestGrad(SysOperator *op, GridFunction x, FiniteElementSpace fespace);

#endif
