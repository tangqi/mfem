// Projected error of a saved DG Grad-Shafranov solution
//
// Loads one mesh and one DG grid function (saved serially, or in parallel with -one),
// L2-projects a named analytical solution onto the same DG space, and visualizes
// err = psi_h - Pi_h psi_exact in GLVis. Also prints the discrete error
// ||psi_h - Pi_h psi||, the projection error ||Pi_h psi - psi|| and the true error
// ||psi_h - psi||.
//
// Pi_h is the true element-wise L2 projection onto the DG space, which is computed
// by solving the block-diagonal mass matrix system
//
// Usage:
//   DG_GS_projected_error -m mesh.mesh -g psi.gf -p solovev
//   DG_GS_projected_error -m mesh.mesh -g psi.gf -p manufactured -no-vis

#include "mfem.hpp"
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>

using namespace std;
using namespace mfem;

// Solov'ev solution from Section 7.1.1 of the DPG paper; use with linear solver
static real_t psi_solovev(const Vector &x)
{
   const real_t d1 =  0.075385029660066;
   const real_t d2 = -0.206294962187880;
   const real_t d3 = -0.031433707280533;

   const real_t R = x(0);
   const real_t Z = x(1);

   const real_t R2 = R*R;
   const real_t R4 = R2*R2;

   return R4/8.0 + d1 + d2*R2 + d3*(R4 - 4.0*R2*Z*Z);
}

// Manufactured solution from Section 7.1.2 of the DPG paper; use with nonlinear solver
static real_t psi_manufactured(const Vector &x)
{
   const real_t k_R = 1.15 * M_PI;
   const real_t k_Z = 1.15;
   const real_t R_0 = -0.5;

   return sin(k_R * (x(0) + R_0)) * cos(k_Z * x(1));
}

int main(int argc, char *argv[])
{
   const char *mesh_file = "";
   const char *gf_file = "";
   string problem = "solovev";
   bool visualization = true;

   OptionsParser args(argc, argv);

   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file.");

   args.AddOption(&gf_file, "-g", "--gridfunction",
                  "DG grid function file.");

   args.AddOption(&problem, "-p", "--problem",
                  "Exact solution: solovev (linear) or manufactured (nonlinear).");

   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   args.Parse();

   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   // Pick the exact solution
   function<real_t(const Vector &)> exact_fn;
   if (problem == "solovev") { exact_fn = psi_solovev; }
   else if (problem == "manufactured") { exact_fn = psi_manufactured; }
   else
   {
      cerr << "Unknown problem: " << problem << " (use solovev or manufactured)" << endl;
      return 1;
   }
   FunctionCoefficient psi_exact(exact_fn);

   // Load the mesh and grid function
   Mesh mesh(mesh_file, 1, 0, false);
   ifstream gf_ifs(gf_file);
   MFEM_VERIFY(gf_ifs.good(), "Could not open grid function: " << gf_file);
   GridFunction psi_h(&mesh, gf_ifs);

   FiniteElementSpace *fes = psi_h.FESpace();
   MFEM_VERIFY(dynamic_cast<const L2_FECollection *>(fes->FEColl()),
               "The grid function must be in a DG (L2) space.");
   const int order = fes->GetMaxElementOrder();

   // High-order Gaussian quadrature: exact for polynomial solutions up to 
   // degree p+8, which covers the Solov'ev solution exactly and is accurate for
   // the manufactured solution
   const int q_order = 2*order + 8;
   const IntegrationRule *int_rules[Geometry::NumGeom];
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      int_rules[g] = &IntRules.Get(g, q_order);
   }

   /**************************************************************/
   // L2 projection of the exact solution onto the DG space
   /**************************************************************/

   // b_i = (psi_exact, phi_i)
   LinearForm b(fes);
   DomainLFIntegrator *lfi = new DomainLFIntegrator(psi_exact);
   lfi->SetIntRule(int_rules[mesh.GetTypicalElementGeometry()]);
   b.AddDomainIntegrator(lfi);
   b.Assemble();

   // The DG mass matrix is block-diagonal, so assembling the element-wise
   // inverse gives M^{-1} exactly and Pi_h psi = M^{-1} b
   BilinearForm m_inv(fes);
   m_inv.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator));
   m_inv.Assemble();
   m_inv.Finalize();

   GridFunction proj(fes);
   m_inv.Mult(b, proj);

   // err = psi_h - Pi_h psi_exact
   GridFunction err(fes);
   err = psi_h;
   err -= proj;

   /**************************************************************/
   // Error norms
   /**************************************************************/

   ConstantCoefficient zero(0.0);
   const real_t discrete_l2 = err.ComputeL2Error(zero, int_rules);
   const real_t proj_l2 = proj.ComputeL2Error(psi_exact, int_rules);
   const real_t true_l2 = psi_h.ComputeL2Error(psi_exact, int_rules);

   cout << '\n';
   cout << "Elements: " << mesh.GetNE() << "  order: " << order
        << "  DOFs: " << fes->GetVSize() << "  quadrature order: " << q_order << endl;
   cout << "  discrete error   ||psi_h - Pi_h psi||: " << discrete_l2 << endl;
   cout << "  projection error ||Pi_h psi - psi||:   " << proj_l2 << endl;
   cout << "  true error       ||psi_h - psi||:      " << true_l2 << endl;

   /**************************************************************/
   // Visualization
   /**************************************************************/

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << err
               << "window_title 'psi_h - Pi_h psi_exact'" << flush;
   }

   return 0;
}
