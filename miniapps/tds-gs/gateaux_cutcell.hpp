#ifndef GATEAUX_CUTCELL_HPP
#define GATEAUX_CUTCELL_HPP

#include "mfem.hpp"
#include "plasma_model.hpp"
#include <set>

using namespace mfem;

// ---------------------------------------------------------------------------
// Gateaux cut-cell diagnostic for the Jacobian (B_y) -- print-only.
//
// Assembles a cut-cell counterpart of the plasma-source contribution to B_y
// using MFEM's moment-fitting cut-volume and cut-surface integration rules,
// and writes Frobenius / max-abs differences against the BFS plasma-source
// pieces built in SysOperator::NonlinearEquationRes (the option-2 mass-like
// SparseMatrix and the option-3 / option-4 column LinearForms).
//
// The reported diffs are independent of `diff_operator` and the Dirichlet
// row-elimination, which are identical in both paths -- only the plasma-
// source pieces contribute to the difference.
//
// Two diffs are returned so the new Eq. 3.10 cut-surface (boundary-motion)
// contribution can be isolated:
//   - `fro_diff_vol_only` : ||X_cut_vol - X_bfs||_F
//   - `fro_diff_full`     : ||X_cut_full - X_bfs||_F
// where X_cut_full = X_cut_vol + (surface bilinear) + (surface ind_x column).
//
// On failure (MFEM without LAPACK, or the limiter contains a 2D element
// that is neither SQUARE nor TRIANGLE) sets ok=false and leaves the four
// output scalars as NAN.
//
//   psi               : poloidal-flux GridFunction
//   psi_x, psi_ma     : separatrix and magnetic-axis flux values
//   ind_ma, ind_x     : the two "special" DOF indices (columns in B_y)
//   model             : plasma source model (drives the integrand)
//   fespace           : H1 space psi lives on (provides mesh and FE)
//   attr_lim          : element attribute of the limiter region
//   plasma_inds       : BFS vertex set (used as the connectivity filter)
//   option2_mat_bfs   : BFS option-2 BilinearForm .SpMat() (mass-like)
//   option3_vec_bfs   : BFS option-3 LinearForm (ind_ma column)
//   option4_vec_bfs   : BFS option-4 LinearForm (ind_x column)
//   fro_diff_vol_only : OUT, ||X_cut_vol - X_bfs||_F
//   fro_diff_full     : OUT, ||X_cut_full - X_bfs||_F
//   maxabs_diff_full  : OUT, max |X_cut_full - X_bfs|
//   fro_plasma_source_bfs : OUT, ||X_bfs||_F (denominator for relative diffs)
//   ok                : OUT, false on skip
//   int_order         : moment-fitting quadrature order (-1 -> default 3)
//   ls_order          : polynomial order for the per-element level-set fit
// ---------------------------------------------------------------------------
void compute_jacobian_diff_cutcell(const GridFunction &psi,
                                   double psi_x, double psi_ma,
                                   int ind_ma, int ind_x,
                                   PlasmaModelBase *model,
                                   FiniteElementSpace *fespace,
                                   int attr_lim,
                                   const std::set<int> &plasma_inds,
                                   const SparseMatrix &option2_mat_bfs,
                                   const Vector &option3_vec_bfs,
                                   const Vector &option4_vec_bfs,
                                   double &fro_diff_vol_only,
                                   double &fro_diff_full,
                                   double &maxabs_diff_full,
                                   double &fro_plasma_source_bfs,
                                   bool &ok,
                                   int int_order = -1,
                                   int ls_order  = 2);

#endif // GATEAUX_CUTCELL_HPP
