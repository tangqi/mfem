//                       MFEM stokes - Serial Version
//
// Compile with: make stokes
//
// Sample runs:  ./stokes -rs 2 -vis
//
// Description:  This example code solves a 2D Stokes problem and compares
//               the solution to a manufactured solution. The system we are
//               solving is
//                                 -\nabla^2 u + \nabla p = f
//                                               div u    = 0
//               with essential boundary conditions.
//               Here, we use a given exact solution (u,p) and compute the
//               corresponding r.h.s. (f,g). The discretization uses the
//               inf-sup stable finite element pair Pn-Pn-1 also known as
//               Taylor-Hood pair.
//
//               This example demonstrates the application of essential boundary
//               conditions to a block system in combination with rectangular blocks.

#include "mfem.hpp"

using namespace std;
using namespace mfem;

void vel_ex(const Vector &x, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   u(0) = -cos(M_PI * xi) * sin(M_PI * yi);
   u(1) = sin(M_PI * xi) * cos(M_PI * yi);
}

double p_ex(const Vector &x)
{
   double xi = x(0);
   double yi = x(1);

   return xi + yi - 1.0;
}

void forcefun(const Vector &x, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   u(0) = 1.0 - 2.0 * M_PI * M_PI * cos(M_PI * xi) * sin(M_PI * yi);
   u(1) = 1.0 + 2.0 * M_PI * M_PI * cos(M_PI * yi) * sin(M_PI * xi);
}

int main(int argc, char *argv[])
{
   int print_level = 2;
   int serial_ref_levels = 0;
   int order = 2;
   bool visualization = 0;
   int basis_type=0;
   double tol = 1e-8;
   const char *mesh_file = "../../data/inline-quad.mesh";

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "");
   args.AddOption(&basis_type, "-basis", "--basis", 
                    "basis type: 0 - H1-H1 (Pn-Pn-1); 1 - H1-L2 (Pn-Pn-1); else - CR-L2");
   args.AddOption(&tol, "-tol", "--tolerance", "Solver relative tolerance");
   args.AddOption(&print_level, "-pl", "--print-level", "Solver print level");
   args.AddOption(&serial_ref_levels,
                  "-rs",
                  "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   int vel_order = order;
   int pres_order = order - 1;

   // Read the (serial) mesh from the given file.
   Mesh *mesh = new Mesh(mesh_file);
   int dim = mesh->Dimension();

   // Refine the serial mesh.
   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // Define a finite element space on the mesh.
   FiniteElementCollection *vel_fec, *pres_fec;
   if (basis_type==0){
      vel_fec = new H1_FECollection(vel_order, dim);
      pres_fec = new H1_FECollection(pres_order, dim);
   }
   else if (basis_type==1){
      vel_fec = new H1_FECollection(vel_order, dim);
      pres_fec = new L2_FECollection(pres_order, dim);
   }
   else{
      vel_fec = new CrouzeixRaviartFECollection();
      pres_fec = new L2_FECollection(0, dim);
   }


   FiniteElementSpace *vel_fes = new FiniteElementSpace(mesh, vel_fec, dim);
   FiniteElementSpace *pres_fes = new FiniteElementSpace(mesh, pres_fec);

   // Create arrays for the essential boundary conditions.
   Array<int> ess_tdof_list, pres_ess_tdof_list;
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   vel_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // Define the block offsets for the Vectors.
   Array<int> block_offsets(3);
   block_offsets[0] = 0;
   block_offsets[1] = vel_fes->GetVSize();
   block_offsets[2] = pres_fes->GetVSize();
   block_offsets.PartialSum();

   {
      cout << "Velocity dofs: " << block_offsets[1]-block_offsets[0] << endl;
      cout << "Pressure dofs: " << block_offsets[2]-block_offsets[1] << endl;
   }

   BlockVector x(block_offsets), rhs(block_offsets);
   GridFunction u_gf(vel_fes), p_gf(pres_fes);

   rhs = 0.0;
   x = 0.0;

   // Define a coefficient for the exact solution for u and p and
   // a coefficient for the rhs.
   VectorFunctionCoefficient uexcoeff(dim, vel_ex);

   VectorFunctionCoefficient fcoeff(dim, forcefun);
   FunctionCoefficient pexcoeff(p_ex);

   // Project the correct boundary condition to the grid function
   // of u. In this example, this also sets the dofs in the velocity
   // offset part of the x vector.
   u_gf.ProjectBdrCoefficient(uexcoeff, ess_bdr);

   LinearForm *fform = new LinearForm(vel_fes);
   fform->AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));
   fform->Assemble();

   BilinearForm *sform = new BilinearForm(vel_fes);
   sform->AddDomainIntegrator(new VectorDiffusionIntegrator);
   sform->Assemble();

   SparseMatrix S;
   Vector X, B;
   // Form the linear system for the first block. This takes the boundary values
   // projected before for the velocity (u_gf) and moves them to B (first component 
   // of rhs).
   //Note it gets confused if I put *.GetBlock(0) in the FormLinearSystem directly
   //(must be a bug in mfem). That works in parallel however.
   sform->FormLinearSystem(ess_tdof_list, u_gf, *fform, S, X, B);
   x.GetBlock(0)=X;
   rhs.GetBlock(0)=B;

   MixedBilinearForm *dform = new MixedBilinearForm(vel_fes, pres_fes);
   dform->AddDomainIntegrator(new VectorDivergenceIntegrator);
   dform->Assemble();

   SparseMatrix D;
   // Form a rectangular matrix of the block system and eliminate the
   // columns (trial dofs). Like in FormLinearSystem, the boundary values
   // are moved from u_gf into the second block of the rhs (B).
   // pres_ess_tdof_list is empty (so we do not eliminate it)
   dform->FormRectangularLinearSystem(ess_tdof_list, pres_ess_tdof_list, u_gf, rhs.GetBlock(1), D, X, B);
   x.GetBlock(0)=X;
   rhs.GetBlock(1)=B;

   SparseMatrix *Dt = Transpose(D);
   // Flip signs of the second block part to make system symmetric.
   rhs.GetBlock(1) *= -1.0;
   
   // The preconditioning technique is to approximate a Schur complement, which
   // is achieved here by forming the pressure mass matrix.
   BilinearForm *mpform = new BilinearForm(pres_fes);
   mpform->AddDomainIntegrator(new MassIntegrator);
   mpform->Assemble();
   mpform->Finalize();
   SparseMatrix &Mp(mpform->SpMat());

   BlockOperator stokesop(block_offsets);
   stokesop.SetBlock(0, 0, &S);
   stokesop.SetBlock(0, 1, Dt, -1.);
   stokesop.SetBlock(1, 0, &D, -1.);

   //For the serial solver, it is better to use suitesparse.
   //The issue is there is no good native smoother for S in the serial mfem
   Solver *invS;
#ifndef MFEM_USE_SUITESPARSE
      invS = new GSSmoother(S);
#else
      invS = new UMFPackSolver(S);
#endif
   Solver *invM = new DSmoother(Mp);
   invM->iterative_mode = false;
   invS->iterative_mode = false;

   BlockDiagonalPreconditioner stokesprec(block_offsets);
   stokesprec.SetDiagonalBlock(0, invS);
   stokesprec.SetDiagonalBlock(1, invM);

   // Since we are solving a symmetric system, a MINRES solver is defined.
   MINRESSolver solver;
   solver.iterative_mode = false;
   solver.SetAbsTol(0.0);
   solver.SetRelTol(tol);
   solver.SetMaxIter(5000);
   solver.SetOperator(stokesop);
   solver.SetPreconditioner(stokesprec);
   solver.SetPrintLevel(print_level);
   solver.Mult(rhs, x);

   // Recover finite element solutions
   u_gf.SetFromTrueDofs(x.GetBlock(0));
   p_gf.SetFromTrueDofs(x.GetBlock(1));

   // Define a quadrature rule used to compare the manufactured solution
   // against the computed solution.
   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double inf=infinity();
   double err_u = u_gf.ComputeL2Error(uexcoeff, irs);
   double errinf_u = u_gf.ComputeLpError(inf,uexcoeff);
   double norm_u = ComputeLpNorm(2, uexcoeff, *mesh, irs);

   double err_p = p_gf.ComputeL2Error(pexcoeff, irs);
   double errinf_p = p_gf.ComputeLpError(inf,pexcoeff);
   double norm_p = ComputeLpNorm(2, pexcoeff, *mesh, irs);

   {
      cout << "|| u_h - u_ex ||_2 = " << err_u << "\n";
      cout << "|| u_h - u_ex ||_2 / || u_ex ||_2 = " << err_u / norm_u << "\n";
      cout << "|| u_h - u_ex ||_i = " << errinf_u << "\n";
      cout << "|| p_h - p_ex ||_2 = " << err_p << "\n";
      cout << "|| p_h - p_ex ||_2 / || p_ex ||_2 = " << err_p / norm_p << "\n";
      cout << "|| p_h - p_ex ||_i = " << errinf_p << "\n";
   }

   if (visualization)
   {
      // Visualize the solution through GLVis.
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream u_sock(vishost, visport);
      u_sock.precision(8);
      u_sock << "solution\n"
             << *mesh << u_gf << "window_title 'velocity'"
             << "keys Rjlc\n"
             << endl;
      u_sock << flush;

      socketstream p_sock(vishost, visport);
      p_sock.precision(8);
      p_sock << "solution\n"
             << *mesh << p_gf << "window_title 'pressure'"
             << "keys Rjlc\n"
             << endl;
      p_sock << flush;
   }

   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("u.gf"), p_ofs("p.gf");
   sol_ofs.precision(8);
   u_gf.Save(sol_ofs);
   p_ofs.precision(8);
   p_gf.Save(p_ofs);

   // Free used memory.
   delete vel_fec;
   delete pres_fec;
   delete vel_fes;
   delete pres_fes;
   delete fform;
   delete sform;
   delete dform;
   delete mpform;
   delete invS;
   delete invM;
   delete Dt;
   delete mesh;

   return 0;
}

