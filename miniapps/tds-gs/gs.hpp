#ifndef GS
#define GS

#include "mfem.hpp"
#include <set>
#include <limits>
#include <iostream>
#include <math.h>
#include <stdio.h>

#include "test.hpp"
#include "exact.hpp"
#include "initial_coefficient.hpp"
#include "plasma_model.hpp"
#include "sys_operator.hpp"
#include "boundary.hpp"
#include "diffusion_term.hpp"

using namespace std;
using namespace mfem;

const int attr_r_eq_0_bdr = 900;
const int attr_ff_bdr = 831;
const int attr_lim = 1000;
const int attr_ext = 2000;
const int attr_vv  = 1001;

const int num_currents = 11;


// FGMRES + Newton solver tuning
struct SolverParams {
  int    kdim             = 10000;
  int    max_newton_iter  = 5;
  int    max_krylov_iter  = 1000;
  double newton_tol       = 1e-12;
  double krylov_tol       = 1e-12;
};

// Hypre BoomerAMG tuning
struct AMGParams {
  int amg_cycle_type    = 1;
  int amg_num_sweeps_a  = 1;
  int amg_num_sweeps_b  = 1;
  int amg_max_iter      = 1;
};

// Adaptive mesh refinement tuning
struct AMROptions {
  int    max_amr_levels = 8;
  int    max_dofs       = 100000;
  double light_tol      = 1e-5;
  double amr_frac_in    = 0.01;
  double amr_frac_out   = 0.3;
};

// Objective / regularization tuning
struct ObjectiveParams {
  int    N_control          = 10;
  int    obj_option         = 2;
  double obj_weight         = 1.0;
  double weight_coils       = 1e-5;
  double weight_solenoids   = 1e-5;
};

// Inexact-Newton η-update tuning
struct InexactNewtonParams {

  // Default alpha_in is the golden ratio
  double alpha_in = 0.5 * (1.0 + 2.2360679774997896);
  double gamma_in = 1.0;
};

// Top-level configuration for a single GS solve. Built by main.cpp from CLI args
// and passed by value to gs().
struct GSProblemConfig {

  // Files / discretization
  const char *mesh_file   = "meshes/iter_gen.msh";
  const char *initial_gf  = "initial/interpolated.gf";
  const char *data_file   = "separated_file.data";
  int    order            = 1;
  int    d_refine         = 0;

  // Plasma model
  int    model_choice     = 1;
  double alpha            = 1.0;
  double beta             = 2.0;
  double gamma            = 0.9;
  double mu               = 1.0;
  double r0               = 1.0;
  double Ip               = 1.5e+7;
  double rho_gamma        = 2.5;

  // External coil currents (5 PF + 6 CS = 11)
  double c1 = 0.0,  c2 = 3.0,  c3 = 1.0,  c4 = 1.0,  c5 = 1.0;
  double c6 = 1.0,  c7 = 1.0,  c8 = 1.0,  c9 = 1.0,  c10 = 1.0,  c11 = 1.0;

  // Mode flags
  int do_test                  = 0;
  int do_initial               = 0;
  int do_control               = 1;
  int do_manufactured_solution = 0;
  int debug_output             = 1;  // write per-iteration debug/visualization files

  // Misc tuning
  double ur_coeff   = 1.0;
  int    PC_option  = 6;

  // Sub-configs
  SolverParams        solver;
  AMGParams           amg;
  AMROptions          amr;
  ObjectiveParams     objective;
  InexactNewtonParams inexact_newton;
};


double gs(GSProblemConfig cfg);

#endif
