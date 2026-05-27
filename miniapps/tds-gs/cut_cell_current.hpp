#ifndef CUT_CELL_CURRENT_HPP
#define CUT_CELL_CURRENT_HPP

#include "mfem.hpp"
#include "plasma_model.hpp"
#include <set>

using namespace mfem;

// ---------------------------------------------------------------------------
// Cut-cell (moment-fitting) total plasma current I_p.
//
// Computes I_p with MFEM's moment-fitting cut-cell integration rules
// (MomentFittingIntRules, cf. examples/ex38.cpp) as a validation alternative
// to the existing BFS whole-element approach in SysOperator::get_plasma_current.
//
//   I_p = - integral over { psi < psi_x } of the option==1 GS plasma-source
//   integrand,
//
// where the sub-region { psi < psi_x } is resolved at sub-element resolution by
// the cut-volume integration rule. The integration is restricted to limiter-
// attribute elements that touch the BFS-connected plasma region (a hybrid
// connectivity filter that excludes disconnected private-flux pockets so the
// cut-cell value is directly comparable to the BFS value).
//
// This is a print-only validation helper: it does not affect the solver state.
//
// Returns the cut-cell I_p (sign matches the existing Plasma_Current). On
// failure -- MFEM built without LAPACK, or the limiter contains 2D elements
// other than SQUARE / TRIANGLE -- sets ok=false and returns NAN.
//
//   psi          : poloidal flux GridFunction
//   psi_x        : flux value at the X-point / separatrix
//   psi_ma       : flux value at the magnetic axis
//   model        : plasma source model
//   fespace      : the H1 space psi lives on (provides the mesh)
//   attr_lim     : element attribute of the limiter region
//   plasma_inds  : BFS vertex set, used as the connectivity filter
//   ok           : output success flag
//   int_order    : moment-fitting quadrature order (-1 -> default of 3)
//   ls_order     : polynomial order for the per-element level-set fit
// ---------------------------------------------------------------------------
double compute_plasma_current_cutcell(const GridFunction &psi,
                                      double psi_x, double psi_ma,
                                      PlasmaModelBase *model,
                                      FiniteElementSpace *fespace,
                                      int attr_lim,
                                      const std::set<int> &plasma_inds,
                                      bool &ok,
                                      int int_order = -1,
                                      int ls_order  = 2);

#endif // CUT_CELL_CURRENT_HPP
