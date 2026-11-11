// Fixed Boundary Grad-Shafranov Solver With Continuous Galerkin
// We are solving: -div[(1/R) grad(psi)] = -R (Solev'ev)
// This is a linearized Grad-Shafranov equation due to the source term having no
// psi dependence
// Coordinate convention:
// x(0) = R
// x(1) = Z
// NOTE: With any chosen mesh, R should be > 0 for all coordinates along the
// mesh. Otherwise, the diffusion term can blow up due to a 1 / R dependency.
//
// This is a CG baseline for comparison with the DG solver.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{

    // Initialize MPI and HYPRE
    Mpi::Init(argc, argv);
    Hypre::Init();

   /**************************************************************/
   // Parse command line options
   /**************************************************************/

   const char *mesh_file = "meshes/ITER.msh";

   string output_mesh = "./solutions/solution_mesh.mesh";
   string output_gf = "./solutions/solution_gf.gf";

   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   int order = 1;

   bool visualization = 1;
   bool save_as_one = false;

   OptionsParser args(argc, argv);

   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");

   args.AddOption(&output_mesh, "-om", "--output-mesh", "Output mesh file path.");

   args.AddOption(&output_gf, "-og", "--output-gf", "Output grid function file path.");

   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");

   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");

   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree.");

   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   args.AddOption(&save_as_one, "-one", "--save-as-one", "-sep",
                  "--save-separately",
                  "Save parallel solution mesh and grid function as one file "
                  "or as separate files for each MPI rank.");
    
   args.Parse();

   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }

   if (Mpi::Root()) { args.PrintOptions(cout); }

   /**************************************************************/
   // Mesh and Finite Element Space
   /**************************************************************/

   // Load serial mesh
   Mesh mesh(mesh_file);
   int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2, "Grad-Shafranov solver requires a 2D R-Z mesh.");

   // Perform uniform mesh refinement on the serial mesh
   for (int l = 0; l < ser_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // Partition the serial mesh into a parallel mesh and delete the serial mesh
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Perform uniform mesh refinement on the parallel mesh
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   // Choose H^1 finite element space
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);
   
   // Get the number of independent DOFs
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   MFEM_VERIFY(size > 0, "Finite element space has no degrees of freedom.");
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << size << endl;
   }

   /**************************************************************/
   // Essential Boundary Conditions
   /**************************************************************/

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;

   MFEM_VERIFY(pmesh.bdr_attributes.Size() > 0, "Mesh has no boundary attributes.");

   // Create an array to mark the essential boundaries, from 1 to the
   // maximum boundary attribute, initialized to zero
   ess_bdr.SetSize(pmesh.bdr_attributes.Max());
   ess_bdr = 0;

   // Mark the external boundaries as essential boundary conditions
   pmesh.MarkExternalBoundaries(ess_bdr);

   // Convert the marked boundaries to essential true DOFs
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   /**************************************************************/
   // Bilinear Form
   /**************************************************************/

   ParBilinearForm a(&fespace);

   // Diffusion operator in GS is like the Poisson diffusion operator, but with a 1/R coefficient
   FunctionCoefficient invR([](const Vector &x)
   {
      const double R = x(0);
      MFEM_VERIFY(R > 0.0, "Grad-Shafranov mesh must lie entirely in R > 0 to prevent blow-up.");
      return 1.0 / R;
   });

   // Add the diffusion term to the bilinear form
   a.AddDomainIntegrator(new DiffusionIntegrator(invR));
   a.Assemble();

   /**************************************************************/
   // Linear Form/GS Source Term
   /**************************************************************/

   // Magnetic permeability
   // const real_t mu0 = 4.0*M_PI*1e-7;  // kept for later use (nonlinear source)

   // Solov'ev equilibrium source--simple linear case
   // Equation 7.1.1 in the DPG paper RHS
   FunctionCoefficient rhs([](const Vector &x) {
      const real_t R = x(0);
      return -R;
   });

   // Assemble linear form
   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(rhs));
   b.Assemble();

   /**************************************************************/
   // Exact Solution
   /**************************************************************/

   // Coefficients from Section 7.1.1 of DPG paper
   const real_t d1 =  0.075385029660066;
   const real_t d2 = -0.206294962187880;
   const real_t d3 = -0.031433707280533;

   // Exact solution
   FunctionCoefficient psi_exact([=](const Vector &x) {
      const real_t R = x(0);
      const real_t Z = x(1);
      
      const real_t R2 = R*R;
      const real_t R4 = R2*R2;

      return R4/8.0 + d1 + d2*R2 + d3*(R4 - 4.0*R2*Z*Z);
   });

   /**************************************************************/
   // Linear System Solve
   /**************************************************************/

   // Define the solution vector psi
   ParGridFunction psi(&fespace);
   psi = 0.0;

   // Apply the exact solution at the Dirichlet boundary
   psi.ProjectBdrCoefficient(psi_exact, ess_bdr);

   // Assemble the linear system in true DOF space
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, psi, b, A, X, B);
   
   // Define BoomerAMG preconditioner
   unique_ptr<HypreBoomerAMG> amg;
   amg.reset(new HypreBoomerAMG(*A.As<HypreParMatrix>()));

   MPI_Barrier(MPI_COMM_WORLD);
   double solve_start = MPI_Wtime();

   // Solve the linear system with PCG
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(500);
   cg.SetPrintLevel(1);
   cg.SetOperator(*A);
   cg.SetPreconditioner(*amg);
   cg.Mult(B, X);

   // Track solve time
   double local_solve_time = MPI_Wtime() - solve_start;
   double solve_time = 0.0;
   MPI_Reduce(&local_solve_time, &solve_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

   // Recover the FEM solution in local DOF space
   a.RecoverFEMSolution(X, b, psi);

   /**************************************************************/
   // Save Output and visualization
   /**************************************************************/
   
   // Save the refined parallel mesh and the solution
   // If input is --save-as-one, then the entire mesh and solution are saved as one file. Else,
   // if input is --save-separately, then separate mesh and solution files are saved for each
   // MPI rank.

   // Create the output directories if they do not already exist
   if (Mpi::Root())
   {
      for (const std::string &path : {output_mesh, output_gf})
      {
         const std::filesystem::path dir = std::filesystem::path(path).parent_path();
         if (!dir.empty())
         {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            MFEM_VERIFY(!ec, "Could not create output directory: " << dir.string()
                        << " (" << ec.message() << ")");
         }
      }
   }
   MPI_Barrier(MPI_COMM_WORLD);

   if (save_as_one)
   {
      pmesh.SaveAsOne(output_mesh);
      psi.SaveAsOne(output_gf.c_str());
   }
   else
   {
      ostringstream mesh_name, gf_name;

      if (Mpi::WorldSize() == 1)
      {
         mesh_name << output_mesh;
         gf_name   << output_gf;
      }
      else
      {
         mesh_name << output_mesh << "." << Mpi::WorldRank();
         gf_name   << output_gf   << "." << Mpi::WorldRank();
      }

      ofstream mesh_ofs(mesh_name.str().c_str());
      MFEM_VERIFY(mesh_ofs.good(), "Could not open mesh output file: " << mesh_name.str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream gf_ofs(gf_name.str());
      MFEM_VERIFY(gf_ofs.good(), "Could not open grid-function output file: " << gf_name.str());
      gf_ofs.precision(8);
      psi.Save(gf_ofs);
   }
   
   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << psi << flush;
   }

   // Print time to solve
   if (Mpi::Root())
   {
      cout << '\n';
      cout << "Linear solve time: " << solve_time << " seconds" << endl;
   }

   /**************************************************************/
   // Compare against analytic solution
   /**************************************************************/

   const real_t psi_l2_error = psi.ComputeL2Error(psi_exact);
   const real_t psi_linf_error = psi.ComputeMaxError(psi_exact);

   // Compute norm of exact solution using zero grid function
   ParGridFunction zero_gf(&fespace);
   zero_gf = 0.0;

   const real_t psi_exact_l2 = zero_gf.ComputeL2Error(psi_exact);
   const real_t psi_relative_l2 = psi_l2_error / psi_exact_l2;

   if (Mpi::Root())
   {
      cout << '\n';
      cout << "Solution error metrics:" << endl;
      cout << "  L2 error:          " << psi_l2_error << endl;
      cout << "  relative L2 error: " << psi_relative_l2 << endl;
      cout << "  Linf error:        " << psi_linf_error << endl;
   }

   return 0;
}
