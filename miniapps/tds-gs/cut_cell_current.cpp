#include "cut_cell_current.hpp"
#include <cmath>
#include <iostream>

using namespace std;

namespace
{

// Level-set phi = psi_x - psi.
// phi > 0  <=>  psi < psi_x  <=>  inside the separatrix.
// MomentFittingIntRules builds the cut-volume rule for the sub-region {phi > 0}.
class PsiLevelSetCoefficient : public Coefficient
{
   const GridFunction &psi;
   double psi_x;
public:
   PsiLevelSetCoefficient(const GridFunction &psi_, double psi_x_)
      : psi(psi_), psi_x(psi_x_) { }
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      return psi_x - psi.GetValue(T, ip, 0);  // psi.Getvalue: evaluates psi at integration point ip under element transformation T
   }
};

// Gate-free port of the option==1 plasma-source integrand (GS RHS).
//
// IMPORTANT: this duplicates the physics of NonlinearGridCoefficient::Eval
// option==1 in plasma_model.cpp -- the model_choice switch block and the
// option==1 return expression. The ONLY intentional difference is that the
// attr_lim and plasma_inds geometric gates are removed: here the geometric
// restriction is supplied instead by the cut-volume integration rule and the
// element-loop filters in compute_plasma_current_cutcell().
//
// If the plasma source model in plasma_model.cpp changes, this MUST be updated
// to match. See the cross-reference comment in plasma_model.cpp at the
// option==1 block.
class PlasmaSourceIntegrand : public Coefficient
{
   PlasmaModelBase *model;
   const GridFunction &psi;
   double psi_ma;   // psi at magnetic axis (== psi_max in plasma_model.cpp)
   double psi_x;    // psi at separatrix    (== psi_bdp in plasma_model.cpp)
public:
   PlasmaSourceIntegrand(PlasmaModelBase *model_, const GridFunction &psi_,
                         double psi_ma_, double psi_x_)
      : model(model_), psi(psi_), psi_ma(psi_ma_), psi_x(psi_x_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      const double f_x      = model->get_f_x();
      const double alpha    = model->get_alpha_bar();
      const double beta     = model->get_beta();
      const double gamma    = model->get_gamma();
      const double mu       = model->get_mu();
      const double coeff_u2 = model->get_coeff_u2();
      const double r0       = 6.2;
      const double alpha_0  = 2.0;
      const double beta_0   = 0.5978;
      const double gamma_0  = 1.395;

      // Physical radial coordinate r.
      double x_[3];
      Vector x(x_, 3);
      T.Transform(ip, x);
      const double ri = x(0);

      // psi and normalized psi at the quadrature point.
      double psi_val = psi.GetValue(T, ip, 0);
      double psi_N = (psi_val - psi_ma) / (psi_x - psi_ma);

      // model_choice switches (mirrors plasma_model.cpp).
      int model_choice = model->get_model_choice();
      double switch_beta = 0.0, switch_taylor = 1.0;
      double switch_ff = 0.0, switch_lb = 0.0;
      if (model_choice == 1)
      { switch_beta = 1.0; switch_taylor =  0.0; switch_ff = 0.0; switch_lb = 0.0; }
      else if (model_choice == 2)
      { switch_beta = 0.0; switch_taylor = -1.0; switch_ff = 0.0; switch_lb = 0.0; }
      else if (model_choice == 3)
      { switch_beta = 0.0; switch_taylor =  0.0; switch_ff = 1.0; switch_lb = 0.0; }
      else if (model_choice == 4)
      { switch_beta = 0.0; switch_taylor =  0.0; switch_ff = 0.0; switch_lb = 1.0; }

      const bool in01 = (psi_N > 0.0) && (psi_N < 1.0);

      double S_bar_ffprime =
         switch_beta * alpha * (f_x + alpha * (model->f_bar(psi_N)))
            * (model->f_bar_prime(psi_N)) / (psi_x - psi_ma)
         + switch_taylor * alpha * (-f_x + alpha * (psi_x - psi_val))
         + switch_ff * alpha * (model->S_ff_prime(psi_N))
         + (in01 ? switch_lb * alpha * (1.0 - beta_0) * r0
                   * pow(1.0 - pow(psi_N, alpha_0), gamma_0) : 0.0);

      return
         beta * ri * (model->S_p_prime(psi_N)) * mu
         + gamma * (model->S_ff_prime(psi_N)) / (ri)
         + S_bar_ffprime / (ri)
         + coeff_u2 * pow(psi_val, 2.0)
         + (in01 ? switch_lb * ri * mu * alpha * beta_0 / r0
                   * pow(1.0 - pow(psi_N, alpha_0), gamma_0) : 0.0);
   }
};

} // anonymous namespace


double compute_plasma_current_cutcell(const GridFunction &psi,
                                      double psi_x,
                                      double psi_ma,
                                      PlasmaModelBase *model,
                                      FiniteElementSpace *fespace,
                                      int attr_lim,
                                      const std::set<int> &plasma_inds,
                                      bool &ok,
                                      int int_order,
                                      int ls_order)
{
   ok = false;

#ifndef MFEM_USE_LAPACK
   (void) psi;   (void) psi_x;       (void) psi_ma;   (void) model;
   (void) fespace; (void) attr_lim;  (void) plasma_inds;
   (void) int_order; (void) ls_order;
   cout << "[cut-cell] MFEM built without LAPACK; cut-cell I_p skipped.\n";
   return NAN;
#else
   Mesh *mesh = fespace->GetMesh();

   // Moment-fitting 2D rules support quadrilaterals and triangles (the latter
   // via the local patches to fem/intrules_cut.cpp). Reject anything else
   // cleanly so the rest of the solver isn't blocked.
   for (int e = 0; e < mesh->GetNE(); ++e)
   {
      if (mesh->GetAttribute(e) != attr_lim) { continue; }
      const Geometry::Type g = mesh->GetElementBaseGeometry(e);
      if (g != Geometry::SQUARE && g != Geometry::TRIANGLE)
      {
         cout << "[cut-cell] limiter contains 2D element geometry " << g
              << " which the moment-fitting path does not support; "
                 "cut-cell I_p skipped.\n";
         return NAN;
      }
   }

   if (int_order < 0)
   {
      // Moment-fitting cost scales steeply with this order; ~3 is ample for
      // validating a smooth source integral (ex38 uses 2). Raising it is
      // expensive until the OrthoBasis2D recomputation in intrules_cut.cpp is
      // fixed (see the plan's "Performance, round 3 / phase 2").
      int_order = 3;
   }

   // Level set phi and integrand g
   PsiLevelSetCoefficient phi(psi, psi_x);
   PlasmaSourceIntegrand  g(model, psi, psi_ma, psi_x);

   // LAPACK moment fitting rules
   MomentFittingIntRules mf_ir(int_order, phi, ls_order);

   double Ip = 0.0;
   IntegrationRule vir;  // cut-volume integration rule

   // Loop over elements
   for (int e = 0; e < mesh->GetNE(); ++e)
   {
      // Skip if element is not within limiter region
      if (mesh->GetAttribute(e) != attr_lim) { continue; }

      // Hybrid connectivity filter: keep only limiter elements that touch the
      // BFS-connected plasma region. This excludes disconnected pockets that a
      // pure level set { psi < psi_x } would otherwise pick up, keeping the
      // cut-cell value directly comparable to the BFS value.
      Array<int> verts;
      mesh->GetElementVertices(e, verts);
      bool touches_plasma = false;
      for (int i = 0; i < verts.Size(); ++i)
      {
         if (plasma_inds.count(verts[i])) { touches_plasma = true; break; }
      }
      if (!touches_plasma) { continue; }

      // Own transformation object (matches examples/ex38.cpp usage).
      IsoparametricTransformation T;
      mesh->GetElementTransformation(e, &T);

      // Cut-volume rule for { phi > 0 } on this element.
      mf_ir.GetVolumeIntegrationRule(T, vir);

      for (int q = 0; q < vir.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = vir.IntPoint(q);
         T.SetIntPoint(&ip);

         // Weight convention follows ex38's SubdomainLFIntegrator:
         // reference-space rule weight * mapping Jacobian * integrand.
         Ip += ip.weight * T.Weight() * g.Eval(T, ip);
      }
   }

   ok = true;
   return -Ip;   // sign matches the existing Plasma_Current *= -1.0
#endif
}
