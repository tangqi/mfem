#include "gateaux_cutcell.hpp"
#include <cmath>
#include <iostream>
#include <limits>

using namespace std;

namespace
{

// Level-set phi = psi_x - psi. (Same form as in cut_cell_current.cpp.)
class PsiLevelSetCoefficient : public Coefficient
{
   const GridFunction &psi;
   double psi_x;
public:
   PsiLevelSetCoefficient(const GridFunction &psi_, double psi_x_)
      : psi(psi_), psi_x(psi_x_) { }
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      return psi_x - psi.GetValue(T, ip, 0);
   }
};

// Gate-free port of NonlinearGridCoefficient::Eval for option in {2, 3, 4}.
// See plasma_model.cpp (the "else" branch, ~lines 304-352). Mirrors the same
// model_choice switch block as PlasmaSourceIntegrand in cut_cell_current.cpp.
//
// IMPORTANT: this duplicates physics from plasma_model.cpp. If the Gateaux
// volume integrands there change, mirror the change here. See the cross-
// reference comment in plasma_model.cpp at the option==1/2/3/4 block.
class GateauxCoef : public Coefficient
{
   PlasmaModelBase *model;
   const GridFunction &psi;
   double psi_ma;
   double psi_x;
   int option;  // 2, 3, or 4
public:
   GateauxCoef(PlasmaModelBase *m, const GridFunction &p,
               double ma, double x, int opt)
      : model(m), psi(p), psi_ma(ma), psi_x(x), option(opt) {}

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

      double x_[3];
      Vector xv(x_, 3);
      T.Transform(ip, xv);
      const double ri = xv(0);

      double psi_val = psi.GetValue(T, ip, 0);
      double psi_N = (psi_val - psi_ma) / (psi_x - psi_ma);

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

      double psi_N_multiplier =
         beta * ri * (model->S_prime_p_prime(psi_N)) * mu
         + gamma * (model->S_prime_ff_prime(psi_N)) / (ri)
         + switch_beta * alpha * alpha * pow(model->f_bar_prime(psi_N), 2.0)
            / (psi_x - psi_ma) / (ri)
         + switch_beta * alpha * (f_x + alpha * (model->f_bar(psi_N)))
            * (model->f_bar_double_prime(psi_N)) / (psi_x - psi_ma) / (ri)
         + switch_ff * alpha * (model->S_prime_ff_prime(psi_N)) / (ri)
         - (in01 ? switch_lb * alpha * (1.0 - beta_0) * r0 * gamma_0
                   * pow(1.0 - pow(psi_N, alpha_0), gamma_0 - 1.0) * alpha_0
                   * pow(psi_N, alpha_0 - 1.0) / ri : 0.0)
         - (in01 ? switch_lb * alpha * ri * mu * beta_0 / r0 * gamma_0
                   * pow(1.0 - pow(psi_N, alpha_0), gamma_0 - 1.0) * alpha_0
                   * pow(psi_N, alpha_0 - 1.0) : 0.0);

      double other = - switch_beta * alpha * (f_x + alpha * (model->f_bar(psi_N)))
         * (model->f_bar_prime(psi_N))
         / (psi_x - psi_ma) / (psi_x - psi_ma) / (ri);

      double coeff = 0.0;
      if (option == 2)
      {
         coeff = 1.0 / (psi_x - psi_ma) * psi_N_multiplier
               - switch_taylor * alpha * alpha / (ri);
      }
      else if (option == 3)
      {
         coeff = -(1.0 - psi_N) / (psi_x - psi_ma) * psi_N_multiplier - other;
      }
      else // option == 4
      {
         coeff = -psi_N / (psi_x - psi_ma) * psi_N_multiplier + other
               + switch_taylor * alpha * alpha / (ri);
      }

      return coeff + coeff_u2 * 2.0 * psi_val;
   }
};

// Cut-surface integrand: g(psi_val=psi_x, r) / |grad psi|.
// Evaluates the option=1 plasma-source formula at psi_N=1 (the separatrix),
// divided by |grad psi|. Clamps the denominator from below by `grad_eps` so
// the X-point (where grad psi -> 0) does not produce inf/nan.
//
// IMPORTANT: the on-separatrix value of the source duplicates physics from
// plasma_model.cpp option==1 (evaluated at psi=psi_x). Keep in sync.
class GBoundaryCoef : public Coefficient
{
   PlasmaModelBase *model;
   const GridFunction &psi;
   double psi_ma;
   double psi_x;
   double grad_eps;
public:
   GBoundaryCoef(PlasmaModelBase *m, const GridFunction &p,
                 double ma, double x, double eps = 1e-12)
      : model(m), psi(p), psi_ma(ma), psi_x(x), grad_eps(eps) {}

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
      (void) r0; (void) alpha_0; (void) beta_0; (void) gamma_0; // referenced only via in01-gated terms which vanish at psi_N=1

      double x_[3];
      Vector xv(x_, 3);
      T.Transform(ip, xv);
      const double ri = xv(0);

      // psi_N = 1 on the separatrix; psi_val = psi_x.
      double psi_N = 1.0;

      int model_choice = model->get_model_choice();
      double switch_beta = 0.0, switch_taylor = 1.0;
      double switch_ff = 0.0, switch_lb = 0.0;
      (void) switch_lb;  // in01 is false at psi_N=1 so all switch_lb-gated terms vanish
      if (model_choice == 1)
      { switch_beta = 1.0; switch_taylor =  0.0; switch_ff = 0.0; switch_lb = 0.0; }
      else if (model_choice == 2)
      { switch_beta = 0.0; switch_taylor = -1.0; switch_ff = 0.0; switch_lb = 0.0; }
      else if (model_choice == 3)
      { switch_beta = 0.0; switch_taylor =  0.0; switch_ff = 1.0; switch_lb = 0.0; }
      else if (model_choice == 4)
      { switch_beta = 0.0; switch_taylor =  0.0; switch_ff = 0.0; switch_lb = 1.0; }

      // At psi_N=1: in01 = false, switch_lb-gated terms vanish.
      // switch_taylor term: switch_taylor * alpha * (-f_x + alpha*(psi_x - psi_val))
      //                  =  switch_taylor * alpha * (-f_x + 0) = -switch_taylor * alpha * f_x
      double S_bar_ffprime_bdy =
         switch_beta * alpha * (f_x + alpha * (model->f_bar(psi_N)))
            * (model->f_bar_prime(psi_N)) / (psi_x - psi_ma)
         + switch_taylor * alpha * (-f_x)
         + switch_ff * alpha * (model->S_ff_prime(psi_N));

      double g_bdy =
         beta * ri * (model->S_p_prime(psi_N)) * mu
         + gamma * (model->S_ff_prime(psi_N)) / (ri)
         + S_bar_ffprime_bdy / (ri)
         + coeff_u2 * pow(psi_x, 2.0);

      // |grad psi| at this surface quadrature point.
      T.SetIntPoint(&ip);
      Vector grad;
      const_cast<GridFunction&>(psi).GetGradient(T, grad);
      double grad_norm = grad.Norml2();
      if (grad_norm < grad_eps) { grad_norm = grad_eps; }

      return g_bdy / grad_norm;
   }
};

// Iterate over a finalised SparseMatrix's raw CSR arrays.
inline double frobenius_norm_sq(const SparseMatrix &M)
{
   const int nnz = M.NumNonZeroElems();
   const double *D = M.GetData();
   double s = 0.0;
   for (int k = 0; k < nnz; ++k) { s += D[k] * D[k]; }
   return s;
}
inline double maxabs_entry(const SparseMatrix &M)
{
   const int nnz = M.NumNonZeroElems();
   const double *D = M.GetData();
   double mx = 0.0;
   for (int k = 0; k < nnz; ++k)
   {
      const double a = std::fabs(D[k]);
      if (a > mx) { mx = a; }
   }
   return mx;
}

} // anonymous namespace


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
                                   int int_order,
                                   int ls_order)
{
   ok = false;
   fro_diff_vol_only     = std::numeric_limits<double>::quiet_NaN();
   fro_diff_full         = std::numeric_limits<double>::quiet_NaN();
   maxabs_diff_full      = std::numeric_limits<double>::quiet_NaN();
   fro_plasma_source_bfs = std::numeric_limits<double>::quiet_NaN();

#ifndef MFEM_USE_LAPACK
   (void) psi; (void) psi_x; (void) psi_ma; (void) ind_ma; (void) ind_x;
   (void) model; (void) fespace; (void) attr_lim; (void) plasma_inds;
   (void) option2_mat_bfs; (void) option3_vec_bfs; (void) option4_vec_bfs;
   (void) int_order; (void) ls_order;
   cout << "[B_y comparison] MFEM built without LAPACK; B_y diagnostic skipped.\n";
   return;
#else
   Mesh *mesh = fespace->GetMesh();
   const int m = fespace->GetVSize();

   // Geometry guard: support SQUARE and TRIANGLE in 2D (matches cut-cell I_p).
   for (int e = 0; e < mesh->GetNE(); ++e)
   {
      if (mesh->GetAttribute(e) != attr_lim) { continue; }
      const Geometry::Type g = mesh->GetElementBaseGeometry(e);
      if (g != Geometry::SQUARE && g != Geometry::TRIANGLE)
      {
         cout << "[B_y comparison] limiter contains unsupported geometry "
              << g << "; B_y diagnostic skipped.\n";
         return;
      }
   }

   if (int_order < 0) { int_order = 3; }

   // Coefficients
   PsiLevelSetCoefficient phi(psi, psi_x);
   GateauxCoef   coef2(model, psi, psi_ma, psi_x, 2);
   GateauxCoef   coef3(model, psi, psi_ma, psi_x, 3);
   GateauxCoef   coef4(model, psi, psi_ma, psi_x, 4);
   GBoundaryCoef coef_surf(model, psi, psi_ma, psi_x);

   MomentFittingIntRules mf_ir(int_order, phi, ls_order);

   // Cut-cell assembly targets (global, unfinalised; Add() accumulates).
   SparseMatrix A2_cut(m, m);
   SparseMatrix A_surf_cut(m, m);
   Vector v3_cut(m);     v3_cut     = 0.0;
   Vector v4_cut(m);     v4_cut     = 0.0;
   Vector v_surf_cut(m); v_surf_cut = 0.0;

   IntegrationRule vir, sir;
   Vector surf_w;
   Array<int> verts, dofs;

   for (int e = 0; e < mesh->GetNE(); ++e)
   {
      if (mesh->GetAttribute(e) != attr_lim) { continue; }

      // Hybrid connectivity filter: keep only limiter elements that touch the
      // BFS-connected plasma region.
      mesh->GetElementVertices(e, verts);
      bool touches = false;
      for (int i = 0; i < verts.Size(); ++i)
      {
         if (plasma_inds.count(verts[i])) { touches = true; break; }
      }
      if (!touches) { continue; }

      IsoparametricTransformation T;
      mesh->GetElementTransformation(e, &T);

      const FiniteElement *fe = fespace->GetFE(e);
      const int ndof = fe->GetDof();
      fespace->GetElementDofs(e, dofs);

      Vector shape(ndof);
      DenseMatrix elmat_vol(ndof, ndof);
      DenseMatrix elmat_surf(ndof, ndof);
      Vector elvec3(ndof), elvec4(ndof), elvec_surf(ndof);
      elmat_vol = 0.0;
      elmat_surf = 0.0;
      elvec3 = 0.0;
      elvec4 = 0.0;
      elvec_surf = 0.0;

      // --- Cut-volume contributions (options 2, 3, 4) ---
      mf_ir.GetVolumeIntegrationRule(T, vir);
      for (int q = 0; q < vir.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = vir.IntPoint(q);
         T.SetIntPoint(&ip);
         const double w = ip.weight * T.Weight();
         fe->CalcShape(ip, shape);

         const double c2 = coef2.Eval(T, ip);
         const double c3 = coef3.Eval(T, ip);
         const double c4 = coef4.Eval(T, ip);

         for (int i = 0; i < ndof; ++i)
         {
            const double wsi = w * shape(i);
            for (int j = 0; j < ndof; ++j)
            {
               elmat_vol(i, j) += c2 * wsi * shape(j);
            }
            elvec3(i) += c3 * wsi;
            elvec4(i) += c4 * wsi;
         }
      }

      // --- Cut-surface contributions (new -- the Eq. 3.10 boundary term) ---
      mf_ir.GetSurfaceIntegrationRule(T, sir);
      if (sir.GetNPoints() > 0)
      {
         mf_ir.GetSurfaceWeights(T, sir, surf_w);
         for (int q = 0; q < sir.GetNPoints(); ++q)
         {
            const IntegrationPoint &ip = sir.IntPoint(q);
            T.SetIntPoint(&ip);
            // Surface measure factor follows ex38's SurfaceLFIntegrator
            // convention: ip.weight (from GetSurfaceIntegrationRule)
            //           * surf_w[q] (from GetSurfaceWeights, the Jacobian factor)
            //           * T.Weight() (element transformation weight).
            const double w = ip.weight * surf_w(q) * T.Weight();
            fe->CalcShape(ip, shape);

            const double cs = coef_surf.Eval(T, ip);

            for (int i = 0; i < ndof; ++i)
            {
               const double wsi = w * shape(i);
               for (int j = 0; j < ndof; ++j)
               {
                  elmat_surf(i, j) += cs * wsi * shape(j);
               }
               elvec_surf(i) += cs * wsi;
            }
         }
      }

      // Scatter element pieces into globals.
      A2_cut.AddSubMatrix(dofs, dofs, elmat_vol);
      A_surf_cut.AddSubMatrix(dofs, dofs, elmat_surf);
      for (int i = 0; i < ndof; ++i)
      {
         v3_cut[dofs[i]]     += elvec3(i);
         v4_cut[dofs[i]]     += elvec4(i);
         v_surf_cut[dofs[i]] += elvec_surf(i);
      }
   }
   A2_cut.Finalize();
   A_surf_cut.Finalize();

   // ----- Compose the diff matrices and their norms -----
   //
   // X_bfs(i,j) = -option2_mat_bfs(i,j)
   //            - option3_vec_bfs[i] * delta(j, ind_ma)
   //            - option4_vec_bfs[i] * delta(j, ind_x)
   //
   // X_cut_vol(i,j) = -A2_cut(i,j)
   //                - v3_cut[i] * delta(j, ind_ma)
   //                - v4_cut[i] * delta(j, ind_x)
   //
   // X_cut_full(i,j) = X_cut_vol(i,j)
   //                 + A_surf_cut(i,j)            // surface bilinear, "+"
   //                 - v_surf_cut[i] * delta(j, ind_x)  // surface ind_x col, "-"
   //
   // Build X_bfs, Diff_vol = X_cut_vol - X_bfs, Diff_full = X_cut_full - X_bfs
   // as sparse matrices via Add() (which accumulates in linked-list mode).

   // X_bfs
   SparseMatrix X_bfs(m, m);
   {
      const int *I = option2_mat_bfs.GetI();
      const int *J = option2_mat_bfs.GetJ();
      const double *D = option2_mat_bfs.GetData();
      const int nrows = option2_mat_bfs.Height();
      for (int i = 0; i < nrows; ++i)
      {
         for (int k = I[i]; k < I[i+1]; ++k)
         {
            X_bfs.Add(i, J[k], -D[k]);
         }
      }
   }
   for (int i = 0; i < m; ++i)
   {
      if (option3_vec_bfs(i) != 0.0) { X_bfs.Add(i, ind_ma, -option3_vec_bfs(i)); }
      if (option4_vec_bfs(i) != 0.0) { X_bfs.Add(i, ind_x,  -option4_vec_bfs(i)); }
   }
   X_bfs.Finalize();
   fro_plasma_source_bfs = std::sqrt(frobenius_norm_sq(X_bfs));

   // Diff_vol = -A2_cut + option2_mat_bfs + column corrections
   SparseMatrix Diff_vol(m, m);
   {
      const int *I = option2_mat_bfs.GetI();
      const int *J = option2_mat_bfs.GetJ();
      const double *D = option2_mat_bfs.GetData();
      const int nrows = option2_mat_bfs.Height();
      for (int i = 0; i < nrows; ++i)
      {
         for (int k = I[i]; k < I[i+1]; ++k)
         {
            Diff_vol.Add(i, J[k], +D[k]);
         }
      }
   }
   {
      const int *I = A2_cut.GetI();
      const int *J = A2_cut.GetJ();
      const double *D = A2_cut.GetData();
      const int nrows = A2_cut.Height();
      for (int i = 0; i < nrows; ++i)
      {
         for (int k = I[i]; k < I[i+1]; ++k)
         {
            Diff_vol.Add(i, J[k], -D[k]);
         }
      }
   }
   for (int i = 0; i < m; ++i)
   {
      const double d3 = option3_vec_bfs(i) - v3_cut(i);
      const double d4 = option4_vec_bfs(i) - v4_cut(i);
      if (d3 != 0.0) { Diff_vol.Add(i, ind_ma, d3); }
      if (d4 != 0.0) { Diff_vol.Add(i, ind_x,  d4); }
   }
   Diff_vol.Finalize();
   fro_diff_vol_only = std::sqrt(frobenius_norm_sq(Diff_vol));

   // Diff_full = Diff_vol + A_surf_cut - v_surf_cut col at ind_x
   SparseMatrix Diff_full(m, m);
   {
      const int *I = Diff_vol.GetI();
      const int *J = Diff_vol.GetJ();
      const double *D = Diff_vol.GetData();
      const int nrows = Diff_vol.Height();
      for (int i = 0; i < nrows; ++i)
      {
         for (int k = I[i]; k < I[i+1]; ++k)
         {
            Diff_full.Add(i, J[k], D[k]);
         }
      }
   }
   {
      const int *I = A_surf_cut.GetI();
      const int *J = A_surf_cut.GetJ();
      const double *D = A_surf_cut.GetData();
      const int nrows = A_surf_cut.Height();
      for (int i = 0; i < nrows; ++i)
      {
         for (int k = I[i]; k < I[i+1]; ++k)
         {
            Diff_full.Add(i, J[k], D[k]);
         }
      }
   }
   for (int i = 0; i < m; ++i)
   {
      if (v_surf_cut(i) != 0.0) { Diff_full.Add(i, ind_x, -v_surf_cut(i)); }
   }
   Diff_full.Finalize();
   fro_diff_full    = std::sqrt(frobenius_norm_sq(Diff_full));
   maxabs_diff_full = maxabs_entry(Diff_full);

   ok = true;
#endif
}
