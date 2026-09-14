// Parallel Fixed Boundary Grad-Shafranov Solver With Discontinuous Galerkin
//
// This implements the nonlinear manufactured-solution test from Section
// 7.1.2 of:
// Z. Peng, Q. Tang, and X.-Z. Tang,
// "An Adaptive Discontinuous Petrov-Galerkin Method for the
// Grad-Shafranov Equation," SIAM J. Sci. Comput. 42(5), 2020.
// DOI: 10.1137/19M1309894
//
// We are solving: -div[(1/R) grad(psi)] = f(R, Z, psi)
//
// Where f(R, Z, psi) = 1 / R F(R, Z, psi), and the source term given by the
// paper is:
//
//   F(R,Z,psi) = (k_R^2 + k_Z^2) psi
//                + (k_R/R) cos(k_R(R + R_0)) cos(k_Z Z)
//                + R [ sin^2(k_R(R + R_0)) cos^2(k_Z Z)
//                      - psi^2
//                      + exp(-sin(k_R(R + R_0)) cos(k_Z Z))
//                      - exp(-psi) ]
//
// with:
//
//   k_R = 1.15*pi,
//   k_Z = 1.15,
//   R_0 = -0.5.
//
// The analytical solution is: psi_exact(R,Z) = sin(k_R(R + R_0)) cos(k_Z Z)
//
// Nonhomogeneous Dirichlet boundary conditions are imposed weakly as
// psi = psi_exact on the plasma boundary.
//
// We use a preconditioned Jacobian-Free Newton-Krylov method for the
// nonlinear solve. GMRES is used for the inner Krylov solve, and HYPRE
// BoomerAMG is applied to the assembled DG diffusion operator as the
// preconditioner.
//
// Coordinate convention:
// x(0) = R
// x(1) = Z
// NOTE: With any chosen mesh, R should be > 0 for all coordinates along the
// mesh. Otherwise, the diffusion term can blow up due to a 1 / R dependency.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>

using namespace std;
using namespace mfem;


/**
 * @brief Jacobian-free operator used by the Jacobian-Free Newton-Krylov method.
 *
 * Computes J(psi)v through a first order finite-difference approximation of the
 * Jacobian action on a vector, avoiding explicit Jacobian assembly:
 * 
 * J(psi)v ≈ [R(psi + epsilon*v) - R(psi)] / epsilon.
 * 
 * J(psi)v is then utilized by a Krylov solver like GMRES.
 */
class JFNKOperator : public Operator
{
private:
   MPI_Comm comm;

   std::function<void(const Vector &, Vector &)> EvalResidual;

   // psi^(k) at Newton iteration k and its residual R(psi^(k))
   Vector psi_base;
   Vector res_base;

   // Scratch vectors representing the finite difference perturbation
   mutable Vector psi_pert;
   mutable Vector res_pert;

public:
   /**
    * @brief Construct the Jacobian-free operator.
    */
   JFNKOperator(
      MPI_Comm comm_,
      int size,
      std::function<void(const Vector &, Vector &)> eval_residual)
      : Operator(size),
        comm(comm_),
        EvalResidual(eval_residual),
        psi_base(size),
        res_base(size),
        psi_pert(size),
        res_pert(size)
   { }

   /**
    * @brief Set the current Newton linearization point.
    */
   void SetLinearizationPoint(const Vector &psi)
   {
      psi_base = psi;
      EvalResidual(psi_base, res_base);  // Cache R(psi_k)
   }

   /**
    * @brief Return the residual at the current Newton step.
    */
   const Vector &GetBaseResidual() const
   {
      return res_base;
   }

   /**
    * @brief Apply the Jacobian to a vector without explicitly forming it.
    *
    * GMRES calls this repeatedly with input: v and output: J(psi_k)*v during linear solve
    */
   void Mult(const Vector &v, Vector &Jv) const override
   {
      const real_t psi_norm = std::sqrt(InnerProduct(comm, psi_base, psi_base));
      const real_t v_norm = std::sqrt(InnerProduct(comm, v, v));

      // J*0 = 0
      if (v_norm == 0.0)
      {
         Jv = 0.0;
         return;
      }

      // Finite-difference perturbation
      const real_t eps_machine = std::numeric_limits<real_t>::epsilon();
      const real_t epsilon = std::sqrt(eps_machine) * (1.0 + psi_norm) / v_norm;

      // psi_pert = psi_k + epsilon*v
      psi_pert = psi_base;
      psi_pert.Add(epsilon, v);

      // R(psi_k + epsilon*v)
      EvalResidual(psi_pert, res_pert);

      // Jv = [R(psi_k + epsilon*v) - R(psi_k)] / epsilon
      Jv = res_pert;
      Jv -= res_base;
      Jv /= epsilon;
   }
};



/**
 * @brief Nonlinear manufactured source term from Section 7.1.2
 * of the DPG Grad-Shafranov paper.
 *
 * Evaluates the RHS f(R,Z,psi) in
 *
 *   -div[(1/R) grad(psi)] = f(R,Z,psi).
 */
class NonlinearGSSource : public Coefficient
{
private:
   const ParGridFunction &psi;

   real_t k_R;
   real_t k_Z;
   real_t R_0;

public:
   NonlinearGSSource(const ParGridFunction &psi_,
                     real_t k_R_,
                     real_t k_Z_,
                     real_t R_0_)
      : psi(psi_),
      k_R(k_R_),
      k_Z(k_Z_),
      R_0(R_0_)
   { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      Vector x(T.GetSpaceDim());
      T.Transform(ip, x);

      const real_t R = x(0);
      const real_t Z = x(1);

      MFEM_VERIFY(R > 0.0, "Grad-Shafranov mesh must lie entirely in R > 0.");

      // Current nonlinear iterate evaluated at this quadrature point
      const real_t psi_val = psi.GetValue(T, ip);

      const real_t arg_R = k_R * (R + R_0);
      const real_t arg_Z = k_Z * Z;

      // Manufactured exact solution q(R,Z)
      const real_t q =
         std::sin(arg_R) * std::cos(arg_Z);

      const real_t cos_term =
         std::cos(arg_R) * std::cos(arg_Z);

      return ((k_R*k_R + k_Z*k_Z) / R) * psi_val
             + (k_R / (R*R)) * cos_term
             + q*q
             - psi_val*psi_val
             + std::exp(-q)
             - std::exp(-psi_val);
   }
};



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
   std::string output_gf = "./solutions/solution_gf.gf";

   // Polynomial order and mesh refinement
   int ser_ref_levels = 0;
   int par_ref_levels = 1;
   int order = 1;

   // DG parameters
   real_t sigma = -1.0;
   real_t kappa = -1.0;

   // Newton solver parameters
   int max_newton_iter = 20;
   real_t newton_abs_tol = 1.0e-12;
   real_t newton_rel_tol = 1.0e-8;

   // GMRES solver parameters
   int max_gmres_iter = 500;
   int gmres_kdim = 10;
   real_t gmres_abs_tol = 0.0;
   real_t gmres_rel_tol = 1.0e-6;

   // Preconditioner
   std::string preconditioner = "amg";

   // Miscellaneous
   bool pa = false;
   bool visualization = true;
   bool save_as_one = false;
   // const char *device_config = "cpu";

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

   args.AddOption(&max_newton_iter, "-nmi", "--newton-max-it",
                  "Maximum number of Newton iterations.");

   args.AddOption(&newton_rel_tol, "-nrtol", "--newton-rtol",
                  "Relative nonlinear residual tolerance.");

   args.AddOption(&newton_abs_tol, "-natol", "--newton-atol",
                  "Absolute nonlinear residual tolerance.");

   args.AddOption(&max_gmres_iter, "-gmi", "--gmres-max-it",
                  "Maximum number of GMRES iterations per Newton iteration.");

   args.AddOption(&gmres_rel_tol, "-grtol", "--gmres-rtol",
                  "Relative GMRES convergence tolerance.");

   args.AddOption(&gmres_abs_tol, "-gatol", "--gmres-atol",
                  "Absolute GMRES convergence tolerance.");

   args.AddOption(&gmres_kdim, "-gk", "--gmres-kdim",
                  "GMRES Krylov subspace dimension before restart.");

   args.AddOption(&preconditioner, "-pc", "--preconditioner",
                  "Preconditioner to use: 'amg' or 'none'.");

   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");

   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   args.AddOption(&save_as_one, "-one", "--save-as-one", "-sep",
                  "--save-separately",
                  "Save parallel solution mesh and grid function as one file "
                  "or as separate files for each MPI rank.");

   // args.AddOption(&device_config, "-d", "--device",
   //                "Device configuration string, see Device::Configure().");

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

   MFEM_VERIFY(max_newton_iter > 0, "--newton-max-it must be positive.");
   MFEM_VERIFY(newton_abs_tol >= 0.0, "--newton-atol must be nonnegative.");
   MFEM_VERIFY(newton_rel_tol >= 0.0, "--newton-rtol must be nonnegative.");
   MFEM_VERIFY(max_gmres_iter > 0, "--gmres-max-it must be positive.");
   MFEM_VERIFY(gmres_abs_tol >= 0.0, "--gmres-atol must be nonnegative.");
   MFEM_VERIFY(gmres_rel_tol >= 0.0, "--gmres-rtol must be nonnegative.");
   MFEM_VERIFY(gmres_kdim > 0, "--gmres-kdim must be positive.");

   // TODO: add more preconditioners
   MFEM_VERIFY(preconditioner == "amg" || preconditioner == "none",
               "--preconditioner must be 'amg' or 'none'.");

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

   OperatorHandle A;
   std::unique_ptr<HypreBoomerAMG> amg;  // Algebraic Multigrid preconditioner

   // Partial assembly
   if (pa)
   {
      // Set the operator A to point at the bilinear form a, for GMRES later on  
      A.Reset(&a, false);

      if (preconditioner == "amg")
      {
         MFEM_ABORT("BoomerAMG requires the fully assembled HypreParMatrix in "
                  "the current implementation. Use --preconditioner none with "
                  "--partial-assembly.");
      }
   }

   // No partial assembly and preconditioner setup
   else
   {
      // Build HYPRE sparse matrix
      A.SetType(Operator::Hypre_ParCSR);
      a.ParallelAssemble(A);

      // Build Algebraic Multigrid preconditioner from A
      if (preconditioner == "amg")
      {
         amg.reset(new HypreBoomerAMG(*A.As<HypreParMatrix>()));
      }
   }

   /**************************************************************/
   // Linear Form/GS Source Term
   /**************************************************************/

   // Magnetic permeability
   // const real_t mu0 = 4.0*M_PI*1e-7;  // kept for later use (nonlinear source)

   // Coefficients from section 7.1.2 of DPG paper
   const real_t pi = std::acos(-1.0);
   const real_t k_R = 1.15 * pi;
   const real_t k_Z = 1.15;
   const real_t R_0 = -0.5;
   
   // Nonlinear source term from section 7.1.2 of DPG paper
   ParGridFunction psi_eval(&fespace);
   psi_eval = 0.0;
   NonlinearGSSource rhs(psi_eval, k_R, k_Z, R_0);

   // Exact solution: needed here because of boundary conditions
   FunctionCoefficient psi_exact([=](const Vector &x)
   {
      const real_t R = x(0);
      const real_t Z = x(1);

      return std::sin(k_R * (R + R_0)) * std::cos(k_Z * Z);
   });

   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(rhs));

   // Handle boundary conditions. In DG, boundary conditions are imposed weakly
   // through boundary integrals rather than by directly fixing boundary DOFs.
   // BCs are given by the exact solution in this case
   b.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(psi_exact, invR, sigma, kappa));

   /**************************************************************/
   // Newton Loop With Jacobian-Free Newton-Krylov
   /**************************************************************/

   // Calculate residual
   auto EvaluateResidual = [&](const Vector &u, Vector &res)
   {
      psi_eval = u;
      b.Assemble();

      A->Mult(u, res);
      res -= b;
   };

   MPI_Barrier(MPI_COMM_WORLD);
   double solve_start = MPI_Wtime();

   // Define solution vector psi and initialize as zero
   ParGridFunction psi(&fespace);
   psi = 0.0;

   // Construct the Jacobian-free operator
   JFNKOperator J(MPI_COMM_WORLD,A->Height(),EvaluateResidual);

   // Set GMRES solver parameters
   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetAbsTol(gmres_abs_tol);
   gmres.SetRelTol(gmres_rel_tol);
   gmres.SetMaxIter(max_gmres_iter);
   gmres.SetKDim(gmres_kdim);
   gmres.SetPrintLevel(1);
   gmres.SetOperator(J);  // Jacobian action on vector
   if (amg) { gmres.SetPreconditioner(*amg); }  // With partial assembly, there is no AMG preconditioner

   bool newton_converged = false;
   real_t initial_res_norm = 0.0;
   for (int k = 0; k <= max_newton_iter; k++)
   {
      J.SetLinearizationPoint(psi);
      const Vector &res = J.GetBaseResidual();

      // Check for convergence before new Newton iteration and break if already converged
      const real_t res_norm = std::sqrt(InnerProduct(MPI_COMM_WORLD, res, res));  // Residual norm
      if (k == 0) {initial_res_norm = res_norm;}
      const real_t rel_res = (initial_res_norm > 0.0) ? res_norm / initial_res_norm : 0.0;  // Relative residual
      if (Mpi::Root())
      {
         cout << "Newton iteration " << k
            << ": ||R|| = " << res_norm
            << ", relative residual = " << rel_res
            << endl;
      }
      if (res_norm <= newton_abs_tol || rel_res <= newton_rel_tol)
      {
         newton_converged = true;
         if (Mpi::Root()) {cout << "Newton converged after " << k << " iterations." << endl;}
         break;
      }
      if (k == max_newton_iter) {break;}

      // Initialize Newton correction term delta_psi
      Vector delta_psi(psi.Size());
      delta_psi = 0.0;

      // Form residual -R(psi_k)
      Vector newton_rhs(res);
      newton_rhs *= -1.0;

      // Solve J(psi_k) delta_psi = -R(psi_k) with GMRES
      gmres.Mult(newton_rhs, delta_psi);
      MFEM_VERIFY(gmres.GetConverged(), "GMRES failed during Newton iteration.");

      // Newton update step
      psi += delta_psi;
   }

   if (!newton_converged && Mpi::Root())
   {
      cout << "WARNING: Newton solver reached the maximum of "
         << max_newton_iter
         << " iterations without converging." << endl;
   }

   // Track solve time
   double local_solve_time = MPI_Wtime() - solve_start;
   double solve_time = 0.0;
   MPI_Reduce(&local_solve_time, &solve_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

   /**************************************************************/
   // Save Output and visualization
   /**************************************************************/
   
   // Save the refined parallel mesh and the solution
   // If input is --save-as-one, then the entire mesh and solution are saved as one file. Else,
   // if input is --save-separately, then separate mesh and solution files are saved for each
   // MPI rank.
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

   // Print time to solve and number of DOFs
   if (Mpi::Root()) 
   {
      cout << '\n';
      cout << "Number of DOFs: " << size << endl;
      cout << "Solve time:     " << solve_time << " seconds" << endl;
   }


   /**************************************************************/
   // Compare against analytic solution
   /**************************************************************/

   // Note: exact solution defined earlier in linear form section

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
