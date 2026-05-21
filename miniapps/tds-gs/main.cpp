/* 
   Compile with: make

   Sample runs:  
   ./main.o
   ./main.o -m meshes/gs_mesh.msh
   ./main.o -m meshes/gs_mesh.msh -o 2

   In order to visualize, run:
   glvis -m mesh.mesh -g sol.gf

   Description: 
   Solve the Grad-Shafranov equation using a Newton iteration:
   d_psi a(psi^k, v, phi^k) = l(I, v) - a(psi^k, v), for all v in V
   
   a = + int 1/(mu r) grad psi dot grad v dr dz  (term1)
       - int (r Sp + 1/(mu r) Sff) v dr dz       (term2)
       + int_Gamma 1/mu psi(x) N(x) v(x) dS(x)   (term3)
       + int_Gamma int_Gamma 1/(2 mu) (psi(x) - psi(y)) M(x, y) (v(x) - v(y)) dS(x) dS(y)  (term4)
   
   d_psi a = + int 1/(mu r) grad phi dot grad v dr dz           (term1')
             - int (r Sp' + 1/(mu r) Sff') d_psi psi_N v dr dz  (term2')
             + int_Gamma 1/mu phi(x) N(x) v(x) dS(x)            (term3')
             + int_Gamma int_Gamma 1/(2 mu) (phi(x) - phi(y)) M(x, y) (v(x) - v(y)) dS(x) dS(y)  (term4')
             
   l(I, v): coil_term:     coil contribution
   term1:   diff_operator: diffusion integrator
   term2:   plasma_term:   nonlinear contribution from plasma
   term3:   (contained inside of diff operator)
   term4:   
   term1':  diff_operator:      diffusion integrator (shared with term1)
   term2':  diff_plasma_term_i: derivative of nonlinear contribution from plasma (i=1,2,3)
   term3':  (contained inside of diff operator)
   term4':

   Mesh attributes:
   831:  r=0 boundary
   900:  far-field boundary
   1000: limiter
   2000: exterior
   everything else: coils

   TODO: double boundary integral

   need boundary of plasma term?
   derivative of plasma functions?
   exact mask?
*/

#include "mfem.hpp"
#include <set>
#include <limits>
#include <iostream>
#include <math.h>
#include "test.hpp"
#include "exact.hpp"
#include "initial_coefficient.hpp"
#include "plasma_model.hpp"
#include "sys_operator.hpp"
#include "boundary.hpp"
#include "diffusion_term.hpp"
#include "gs.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // All CLI-driven defaults are set by GSProblemConfig's in-class member
   // initializers. AddOption below points each flag at the matching field.
   GSProblemConfig cfg;
   int do_test = 0;  // not part of cfg; gates the unit-test path below

   OptionsParser args(argc, argv);

   // Files & discretization
   args.AddOption(&cfg.mesh_file,    "-m",   "--mesh",              "Mesh file to use.");
   args.AddOption(&cfg.initial_gf,   "-igf", "--initial_gf",        "initial grid function.");
   args.AddOption(&cfg.data_file,    "-d",   "--data_file",         "Plasma data file");
   args.AddOption(&cfg.order,        "-o",   "--order",             "Finite element polynomial degree");
   args.AddOption(&cfg.d_refine,     "-g",   "--refinement_factor", "Number of grid refinements");

   // Plasma model parameters
   args.AddOption(&cfg.model_choice, "-mo",  "--model",          "model (1: ff', 2: Taylor equilibrium)");
   args.AddOption(&cfg.alpha,        "-al",  "--alpha",          "alpha");
   args.AddOption(&cfg.beta,         "-be",  "--beta",           "beta");
   args.AddOption(&cfg.gamma,        "-ga",  "--gamma",          "gamma");
   args.AddOption(&cfg.mu,           "-mu",  "--mu",             "mu");
   args.AddOption(&cfg.r0,           "-rz",  "--r_zero",         "r0");
   args.AddOption(&cfg.Ip,           "-Ip",  "--plasma_current", "Ip");
   args.AddOption(&cfg.rho_gamma,    "-rg",  "--rho_gamma",      "rho_gamma");

   // External coil currents (5 poloidal field + 6 central solenoid)
   args.AddOption(&cfg.c1,  "-c1",  "--c1",  "coil 1 (PF)");
   args.AddOption(&cfg.c2,  "-c2",  "--c2",  "coil 2 (PF)");
   args.AddOption(&cfg.c3,  "-c3",  "--c3",  "coil 3 (PF)");
   args.AddOption(&cfg.c4,  "-c4",  "--c4",  "coil 4 (PF)");
   args.AddOption(&cfg.c5,  "-c5",  "--c5",  "coil 5 (PF)");
   args.AddOption(&cfg.c6,  "-c6",  "--c6",  "coil 6 (CS)");
   args.AddOption(&cfg.c7,  "-c7",  "--c7",  "coil 7 (CS)");
   args.AddOption(&cfg.c8,  "-c8",  "--c8",  "coil 8 (CS)");
   args.AddOption(&cfg.c9,  "-c9",  "--c9",  "coil 9 (CS)");
   args.AddOption(&cfg.c10, "-c10", "--c10", "coil 10 (CS)");
   args.AddOption(&cfg.c11, "-c11", "--c11", "coil 11 (CS)");

   // Mode flags
   args.AddOption(&do_test,                      "-t",  "--test",                     "Perform tests only");
   args.AddOption(&cfg.do_initial,               "-i",  "--initial",                  "solve for initial guess");
   args.AddOption(&cfg.do_control,               "-dc", "--do_control",               "solve the control problem");
   args.AddOption(&cfg.do_manufactured_solution, "-dm", "--do_manufactured_solution", "do manufactured solution");
   args.AddOption(&cfg.debug_output,             "-dbg", "--debug_output",            "write per-iteration debug/visualization files (0/1)");

   // Misc tuning
   args.AddOption(&cfg.PC_option, "-pc", "--pc_option", "preconditioner option");
   args.AddOption(&cfg.ur_coeff,  "-ur", "--ur_coeff",  "under relaxation coefficient");

   // Newton / FGMRES solver tuning (SolverParams)
   args.AddOption(&cfg.solver.max_newton_iter, "-mn", "--max_newton_iter", "maximum newton iterations");
   args.AddOption(&cfg.solver.max_krylov_iter, "-mk", "--max_krylov_iter", "maximum krylov iterations");
   args.AddOption(&cfg.solver.newton_tol,      "-nt", "--newton_tol",      "newton tolerance");
   args.AddOption(&cfg.solver.krylov_tol,      "-kt", "--krylov_tol",      "krylov tolerance");

   // Hypre BoomerAMG tuning (AMGParams)
   args.AddOption(&cfg.amg.amg_cycle_type,   "-ct",  "--amg_cycle_type",   "AMG cycle type (1: v, 2: w)");
   args.AddOption(&cfg.amg.amg_num_sweeps_a, "-nsa", "--amg_num_sweeps_a", "AMG num sweeps a");
   args.AddOption(&cfg.amg.amg_num_sweeps_b, "-nsb", "--amg_num_sweeps_b", "AMG num sweeps b");
   args.AddOption(&cfg.amg.amg_max_iter,     "-mi",  "--amg_max_iter",     "AMG max iterations");

   // Adaptive mesh refinement (AMROptions)
   args.AddOption(&cfg.amr.max_amr_levels, "-ml",  "--max_amr_levels", "max amr levels");
   args.AddOption(&cfg.amr.max_dofs,       "-md",  "--max_dofs",       "max amr dofs");
   args.AddOption(&cfg.amr.light_tol,      "-lt",  "--light_tol",      "light tolerance");
   args.AddOption(&cfg.amr.amr_frac_in,    "-afi", "--amr_frac_in",    "AMR fraction for limiter");
   args.AddOption(&cfg.amr.amr_frac_out,   "-afo", "--amr_frac_out",   "AMR fraction for outside limiter");

   // Objective / regularization tuning (ObjectiveParams)
   args.AddOption(&cfg.objective.N_control,        "-Nc", "--N_control",        "N_control");
   args.AddOption(&cfg.objective.obj_option,       "-oo", "--obj_option",       "objective option (0, 1, 2)");
   args.AddOption(&cfg.objective.obj_weight,       "-wo", "--weight_obj",       "weight of optimization");
   args.AddOption(&cfg.objective.weight_coils,     "-wc", "--weight_coils",     "weight of regularization");
   args.AddOption(&cfg.objective.weight_solenoids, "-ws", "--weight_solenoids", "weight of regularization");

   // Inexact-Newton tuning (InexactNewtonParams)
   args.AddOption(&cfg.inexact_newton.alpha_in, "-ai", "--alpha_in", "inexact newton param alpha");
   args.AddOption(&cfg.inexact_newton.gamma_in, "-gi", "--gamma_in", "inexact newton param gamma");

   args.ParseCheck();

   // When do_initial == 1, override the mesh file and skip the control
   // problem so we generate an initial guess for the free-boundary GS.
   if (cfg.do_initial == 1) {
     cout << "Solving for initial guess" << endl;
     cfg.mesh_file = "meshes/iter_gen_initial.msh";
   }

   // Unit tests
   if (do_test == 1) {
     cout << "" << endl;
     cout << "Configured for testing only--performing unit tests." << endl;
     cout << "" << endl;

     test();

     cout << "Testing complete--no issues detected." << endl;
   }

   // Run Grad-Shafranov solver
   else {
     gs(cfg);
   }

   return 0;
}
