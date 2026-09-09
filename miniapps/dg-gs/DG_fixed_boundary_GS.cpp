// Fixed Boundary Grad-Shafranov Solver With Discontinuous Galerkin (SIPG)
// Coordinate convention:
// x(0) = R
// x(1) = Z

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

   // TODO: these are the default ex14p.cpp command line options. Tailor for this problem

   const char *mesh_file = "../data/star.mesh";  // TODO: need different default mesh
   int ser_ref_levels = 1;
   int par_ref_levels = 2;
   int order = 1;
   real_t sigma = -1.0;
   real_t kappa = -1.0;
   real_t eta = 0.0;
   bool pa = false;
   bool visualization = 1;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the three DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the three DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&eta, "-e", "--eta", "BR2 penalty parameter.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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

   // Set up and print device used
   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

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
         return 1.0 / R;  // Potential issue: R = 0
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

   // TODO: Add nonlinear solve e.g. Newton iteration: residual, delta_x, Newton Loop, JFNK, preconditioner

   /**************************************************************/
   // Output and visualization
   /**************************************************************/
   
   // Save the refined mesh and the solution in parallel
   // View in GLVis with: "glvis -np <np> -m mesh -g sol"  // TODO: confirm if this is the case
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << Mpi::WorldRank();
      sol_name << "sol." << setfill('0') << setw(6) << Mpi::WorldRank();

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
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

   return 0;
}

