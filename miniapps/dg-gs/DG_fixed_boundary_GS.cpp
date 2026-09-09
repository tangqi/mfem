// Fixed Boundary Grad-Shafranov Solver With Discontinuous Galerkin
// Coordinate convention:
// x(0) = R
// x(1) = Z
// NOTEL With any chosen mesh, R should be > 0 for all coordinates along the
// mesh. Otherwise, the diffusion term can blow up due to a 1 / R dependency.

// TODO: need analytical solution to compare against

#include "mfem.hpp"
#include <fstream>
#include <iostream>

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

   std::string output_mesh = "./solutions/solution_mesh.mesh";
   std::string output_gf = "./solutions/solution_gf.mesh";

   int ser_ref_levels = 1;
   int par_ref_levels = 2;
   int order = 1;

   real_t sigma = -1.0;
   real_t kappa = -1.0;
   // real_t eta = 0.0;

   bool pa = false;
   bool visualization = 1;
   // const char *device_config = "cpu";
   bool save_as_one = false;

   OptionsParser args(argc, argv);

   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");

   args.AddOption(&output_mesh, "-om", "--output-mesh",
                  "Base path for the output parallel mesh.");

   args.AddOption(&output_gf, "-og", "--output-gridfunction",
                  "Base path for the output solution grid function.");

   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");

   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");

   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");

   args.AddOption(&sigma, "-s", "--sigma",
                  "DG penalty parameter, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");

   args.AddOption(&kappa, "-k", "--kappa",
                  "DG penalty parameterm should be positive."
                  " Negative values are replaced with (order+1)^2.");

   // args.AddOption(&eta, "-e", "--eta", "BR2 penalty parameter.");

   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");

   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   // args.AddOption(&device_config, "-d", "--device",
   //                "Device configuration string, see Device::Configure().");

   args.AddOption(&save_as_one, "-one", "--save-as-one", "-sep",
                  "--save-separately",
                  "Save parallel solution mesh and grid function as one file "
                  "or as separate files for each MPI rank.");

   args.Parse();

   // Print error/help text for invalid command line arguments
   if (!args.Good())
   {
    // This check makes sure that, in a parallel run, only one MPI rank prints to console  
    if (Mpi::Root())
       {
          args.PrintUsage(cout);
       }
       return 1;
   }

   // Kappa should be positive. Otherwise, it is (order + 1)^2
   if (kappa < 0) { kappa = (order+1)*(order+1); }

   // Print command line options
   if (Mpi::Root()) { args.PrintOptions(cout); }

   // // Set up and print device used
   // Device device(device_config);
   // if (Mpi::Root()) { device.Print(); }

   /**************************************************************/
   // Mesh and Finite Element Space
   /**************************************************************/

   // Load serial mesh, either triangular or quadrilateral mesh
   // Note: the entire mesh is loaded for all MPI ranks here.
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

   // If partial assembly, choose the Gauss-Lobatto basis. Otherwise, choose the Gauss-Legendre basis.
   int bt;
   if (pa) { bt = BasisType::GaussLobatto; }
   else { bt = BasisType::GaussLegendre; }

   // Choose discontinuous Galerkin element space with order, dimension, and Gauss Lobatto/Gauss Legendre.
   DG_FECollection fec(order, dim, bt);
   ParFiniteElementSpace fespace(&pmesh, &fec);

   // Get number of independent DOFs
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << size << endl;
   }

   /**************************************************************/
   // Bilinear Form
   /**************************************************************/

   ParBilinearForm a(&fespace);

   // Diffusion operator in GS is like the Poisson diffusion operator, but with a 1/R instead
   FunctionCoefficient invR([](const Vector &x)
      {
         const real_t R = x(0);
         MFEM_VERIFY(R > 0.0, "Grad-Shafranov mesh must lie entirely in R > 0 to prevent blow-up.");
         return 1.0 / R;
      }
   );

   // Build standard bilinear form integral 
   a.AddDomainIntegrator(new DiffusionIntegrator(invR));

   // DG terms on the faces between elements to couple neighboring element solutions
   a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(invR, sigma, kappa));

   // Weakly impose Dirichlet boundary conditions for DG
   a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(invR, sigma, kappa));

   // If partial assembly, don't explicitly form A
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   a.Assemble();
   a.Finalize();  // Compress A into CSR format

   /**************************************************************/
   // Linear Form/GS Source Term
   /**************************************************************/

   // Magnetic permeability
   const real_t mu0 = 4.0*M_PI*1e-7;

   // Solov'ev coefficients
   real_t Cp  = 1.0;
   real_t CFF = 1.0;;

   // Solov'ev equilibrium source--simple linear case
   // P'(psi) = C_P
   // F(psi) F'(psi) = C_F
   FunctionCoefficient rhs([=](const Vector &x) {
      const real_t R = x(0);
      return mu0*R*Cp + CFF/R;
   });
    
   // TODO: implement a more complex, nonlinear case. Luxon and Brown?
   // TODO: add these different cases as command line options

   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(rhs));

   // Handle boundary conditions. In DG, boundary conditions are imposed weakly
   // through boundary integrals rather than by directly fixing boundary DOFs.
   ConstantCoefficient psi_b(0.0);
   b.AddBdrFaceIntegrator(
   new DGDirichletLFIntegrator(psi_b, invR, sigma, kappa));

   b.Assemble();

   /**************************************************************/
   // Linear System Solve
   /**************************************************************/
   
   // Define solution vector x
   ParGridFunction x(&fespace);
   x = 0.0;

   OperatorHandle A;
   std::unique_ptr<HypreBoomerAMG> amg;  // Algebraic Multigrid preconditioner

   // Partial assembly
   if (pa)
   {
      // Set the operator A to point at the bilinear form a, for GMRES or CG later on  
      A.Reset(&a, false);
   }
   else
   {
      // Build HYPRE sparse matrix
      A.SetType(Operator::Hypre_ParCSR);
      a.ParallelAssemble(A);

      // Build Algebraic Multigrid preconditioner from A
      amg.reset(new HypreBoomerAMG(*A.As<HypreParMatrix>()));
   }

   MPI_Barrier(MPI_COMM_WORLD);
   double solve_start = MPI_Wtime();
   
   // Depending on the symmetry of A, define and apply a parallel PCG or
   // GMRES solver for AX=B using the BoomerAMG preconditioner from hypre.
   if (sigma == -1.0)
   {
      // Conjugate gradient: sigma == -1 implies A is SPD
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(500);
      cg.SetPrintLevel(1);
      cg.SetOperator(*A);
      if (amg) { cg.SetPreconditioner(*amg); }  // With partial assembly, there is no AMG preconditioner
      cg.Mult(b, x);
   }
   else
   {
      // GMRES
      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetAbsTol(0.0);
      gmres.SetRelTol(1e-12);
      gmres.SetMaxIter(500);
      gmres.SetKDim(10);
      gmres.SetPrintLevel(1);
      gmres.SetOperator(*A);
      if (amg) { gmres.SetPreconditioner(*amg); }  // With partial assembly, there is no AMG preconditioner
      gmres.Mult(b, x);
   }

   // Track solve time
   double local_solve_time = MPI_Wtime() - solve_start;
   double solve_time = 0.0;
   MPI_Reduce(&local_solve_time, &solve_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

   // TODO: Add nonlinear solve e.g. Newton iteration: residual, delta_x, Newton Loop, JFNK, preconditioner

   /**************************************************************/
   // Output and visualization
   /**************************************************************/
   
   // Save the refined parallel mesh and the solution
   // If input is --save-as-one, then the entire mesh and solution are saved as one file. Else,
   // if input is --save-separately, then separate mesh and solution files are saved for each
   // MPI rank.
   if (save_as_one)
   {
      pmesh.SaveAsOne(output_mesh);
      x.SaveAsOne(output_gf.c_str());
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
      x.Save(gf_ofs);
   }
   
   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }

   // Print time to solve
   if (Mpi::Root()) 
   {
      cout << '\n';
      cout << "Linear solve time: " << solve_time << " seconds" << endl;
   }

   return 0;
}
