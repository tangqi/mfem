////////////////////////////////////////////////////////////////////////////////

// TODO: General List:
// 1) Review code and make any necessary style changes (function naming, formatting)
// 2) Add function descriptions and comment liberally
// 3) Namespaces: identify where classes come from (are they MFEM or user-defined? If
//    user-defined, where are they defined?)
// 4) Consider removing the code sections related to do_control == 0 and manufactured
//    solutions.
// 5) Modularize code: move sections into other files to enhance readability.
// 6) Address memory leakage: every time a "new" object is called, there is a risk of
//    memory leaks (no "delete" to clean up memory).
// 7) The makefile is forcing C++11--perhaps change this to a later C++ version?

#include "mfem.hpp"
#include "gs.hpp"
#include "gs_test_utils.hpp"
#include "boundary.hpp"
#include "amr.hpp"
#include "field.hpp"
#include "double_integrals.hpp"
#include <stdio.h>
#include <chrono>

using namespace std;
using namespace mfem;

// ---------------------------------------------------------------------------
// True-DOF conversion helpers for nonconforming mesh support
// ---------------------------------------------------------------------------

// Convert a VSize x VSize SparseMatrix to TrueVSize x TrueVSize
// via A_true = R * A * P  (conforming projection, R = P^T)
SparseMatrix* ToTrueDofs(const SparseMatrix &A, const FiniteElementSpace &fes) {

    // Galerkin projection to true-DOF space: P^T A P.
    const SparseMatrix *P = fes.GetConformingProlongation();

    if (!P) return new SparseMatrix(A);

    SparseMatrix *PT = Transpose(*P);
    SparseMatrix *PTA = mfem::Mult(*PT, A);
    SparseMatrix *result = mfem::Mult(*PTA, *P);

    delete PT;
    delete PTA;

    return result;
}

// Convert a VSize Vector (dual / RHS convention: e.g. LinearForm integrals)
// to TrueVSize by v_true = P^T v_full.  Again: do NOT use ConformingRestriction
// here — that picker matrix drops slave entries instead of accumulating them
// into the master rows.
void ToTrueDofs(const Vector &v_full, Vector &v_true, const FiniteElementSpace &fes) {
    const SparseMatrix *P = fes.GetConformingProlongation();
    if (!P) { v_true = v_full; return; }
    P->MultTranspose(v_full, v_true);
}

// Prolong a TrueVSize Vector back to VSize
void ToFullDofs(const Vector &v_true, Vector &v_full, const FiniteElementSpace &fes) {
    const SparseMatrix *P = fes.GetConformingProlongation();
    if (!P) { v_full = v_true; return; }
    P->Mult(v_true, v_full);
}

// ---------------------------------------------------------------------------


void WriteSparseMatrixToFile(FILE *fp, SparseMatrix *Mat) {
  /**
    * Write matrix to a text file in a MATLAB-readable CSR format.
    *
    * @param[in] fp   The output filepath.
    * @param[in] Mat  The matrix to be written.
  */
  int *I = Mat->GetI();
  int *J = Mat->GetJ();
  double *A = Mat->GetData();
  int height = Mat->Height();

  int i, j;
  for (i = 0; i < height; ++i) {
    for (j = I[i]; j < I[i+1]; ++j) {
      fprintf(fp, "%d %d %.3e\n", i, J[j], A[j]);
    }
  }
}


// TODO: for each of the inputs to DefineRHS, consider adding namespaces to make
// clear where each input type comes from. Example:
//
// void DefineRHS(
//     double & rho_gamma,
//     physics::PlasmaModelBase & model,
//     mfem::Mesh & mesh, 
//     physics::ExactCoefficient & exact_coefficient,
//     physics::ExactForcingCoefficient & exact_forcing_coeff,
//     mfem::LinearForm & coil_term,
//     mfem::SparseMatrix * F
//   ) {
//
// This way it is clear where each object comes from (MFEM, user-defined, built-in C++ type, etc.)
//
// Consider removing manufactured solution components.

void DefineRHS(
    PlasmaModelBase &model,
    double &rho_gamma,
    Mesh &mesh, 
    ExactCoefficient &exact_coefficient,
    ExactForcingCoefficient &exact_forcing_coeff,
    LinearForm &coil_term,
    SparseMatrix *F
  ) {
  /**
  * Build the right-hand side of the Grad-Shafronov finite element system, representing
  * the contributions from the external coil currents. Modifies F matrix and coil_term
  * vector in-place.
  *
  * @param[in] model                PlasmaModel object containing constants used in plasma.
  * @param[in] rho_gamma            Scalar parameter used in boundary coefficients.
  * @param[in] mesh                 MFEM finite element mesh object.
  * @param[in] exact_coefficient    Analytic solution for "manufactured solution" tests.
  * @param[in] exact_forcing_coeff  Analytic right-hand side for "manufactured solution" tests.
  * @param[in,out] coil_term        Right-hand side of GS finite element system.
  * @param[in,out] F                Coil geometry matrix.
  */

  // Retrieve the finite element space and number of DOFs for the RHS term
  FiniteElementSpace *fespace = coil_term.FESpace();
  int ndof = fespace->GetNDofs();

  // Create a constant function of ones over the domain
  GridFunction ones(fespace);
  ones = 1.0;

  // Get the unique element attributes used by the mesh -- each attribute corresponds to a region
  // in the domain that the element belongs to.
  Array<int> attribs(mesh.attributes);

  int counter = 0;
  for (int i = 0; i < attribs.Size(); ++i) {
    int attrib = attribs[i];

    // TODO: consider rewriting this switch block as if/else if/else statements
    // If cases are true, skip. Otherwise, run default code block
    switch(attrib) {
      case attr_ext:
        break;
      case attr_vv:  // exterior domain
        break;
      case attr_lim:  // limiter domain
        break;
      case 1100:  // What does this correspond to?
        break;
      default:
        
        // Create a piecewise constant coefficient that is 1 on the current region (attrib) and 0 everywhere else
        Vector pw_vector(attribs.Max());
        pw_vector = 0.0;
        pw_vector(attrib - 1) = 1.0;  // index starts from 1, so attrib - 1 gets us the correct attribute
        PWConstCoefficient pw_coeff(pw_vector);  // TODO: what is the namespace for PWConstCoefficient?

        // Assemble the linear form
        LinearForm lf(fespace);
        lf.AddDomainIntegrator(new DomainLFIntegrator(pw_coeff));
        lf.Assemble();

        // Normalize and insert into F
        double area = lf(ones);
        for (int j = 0; j < ndof; ++j) {
          if (lf[j] != 0) {
            F->Set(j, counter, lf[j] / area);
          }
        } 
        ++counter;
    }
  }
  F->Finalize();

  // Manufactured (analytical) solution forcing -- has no effect when manufactured solution is turned off
  // Assemble the RHS term
  coil_term.AddDomainIntegrator(new DomainLFIntegrator(exact_forcing_coeff));
  coil_term.Assemble();

  // magnetic permittivity
  double mu = model.get_mu();

  // N(x): Green's function for far-field boundary
  auto N_lambda = [&rho_gamma, &mu](const Vector &x) -> double {
    return N_coefficient(x, rho_gamma, mu);  // N_coefficient comes from boundary.cpp
  };

  // M(x, y): Green's function for far-field boundary
  auto M_lambda = [&mu](const Vector &x, const Vector &y) -> double {
    return M_coefficient(x, y, mu);  // M_coefficient comes from boundary.cpp
  };

  // Incorporate far-field boundary
  BilinearForm b(fespace);
  FunctionCoefficient first_boundary_coeff(N_lambda);
  b.AddBoundaryIntegrator(new MassIntegrator(first_boundary_coeff));
  DoubleBoundaryBFIntegrator i(M_lambda);
  b.Assemble();
  AssembleDoubleBoundaryIntegrator(b, i, attr_ff_bdr);
  b.Finalize(); // is this needed?

  // Project manufactured (analytical/exact) solution and add: coil_term += b * u_ex
  // If manufactured solution is turned off, then u_ex is set to 0, and this term has no effect
  GridFunction u_ex(fespace);
  u_ex.ProjectCoefficient(exact_coefficient);
  b.AddMult(u_ex, coil_term);
}


void DefineLHS(PlasmaModelBase &model, double rho_gamma, BilinearForm &diff_operator) {
  /**
  * Build the bilinear form of the Grad-Shafranov finite element system, containing
  * contributions from the diffusion operator, plasma terms, and far-field boundary
  * terms. Modifies diff_operator in-place.
  *
  * @param[in] model              PlasmaModel object containing constants used in plasma.
  * @param[in] rho_gamma          Scalar parameter used in boundary coefficients.
  * @param[in,out] diff_operator  Left-hand side of GS finite element system.
  */

  // Diffusion operator contribution
  DiffusionIntegratorCoefficient diff_op_coeff(&model);
  diff_operator.AddDomainIntegrator(new DiffusionIntegrator(diff_op_coeff));

  // Plasma terms contribution
  Vector pw_vector_(2000);
  pw_vector_ = 0.0;
  pw_vector_(1100 - 1) = 1.0;  // What does 1100 correspond to? A particular coil?
  PWConstCoefficient pw_coeff(pw_vector_);  // TODO: what is the namespace for PWConstCoefficient?
  ConstantCoefficient one(1.0);  // TODO: is this needed?
  diff_operator.AddDomainIntegrator(new MassIntegrator(pw_coeff));

  // magnetic permittivity
  double mu = model.get_mu();

  // N(x): Green's function for far-field boundary
  auto N_lambda = [&rho_gamma, &mu](const Vector &x) -> double {
    return N_coefficient(x, rho_gamma, mu);  // N_coefficient comes from boundary.cpp
  };

  // M(x, y): Green's function for far-field boundary
  auto M_lambda = [&mu](const Vector &x, const Vector &y) -> double {
    return M_coefficient(x, y, mu);  // M_coefficient comes from boundary.cpp
  };

  // Far-field boundary contribution
  FunctionCoefficient first_boundary_coeff(N_lambda);
  diff_operator.AddBoundaryIntegrator(new MassIntegrator(first_boundary_coeff));
  diff_operator.Assemble();  // TODO: compared to DefineRHS, this line and the following line are switched. How should this be fixed?
  DoubleBoundaryBFIntegrator i(M_lambda);
  AssembleDoubleBoundaryIntegrator(diff_operator, i, attr_ff_bdr);
  diff_operator.Finalize(); // is this needed?
}


HypreParMatrix *ConvertToHypre(SparseMatrix *P) {
  /**
  * Convert an MFEM sparse matrix into a Hypre-compatible parallel matrix.
  *
  * @param[in] P  Pointer to the MFEM SpareMatrix object to be converted.
  *
  * @return  Pointer to a HypreParMatrix object that references the data of P.
  */
  HYPRE_BigInt col_starts[2], row_starts[2];
  row_starts[0] = 0;
  row_starts[1] = P->Height();
  col_starts[0] = 0;
  col_starts[1] = P->Height();

  return new HypreParMatrix(
    MPI_COMM_WORLD,
    P->Height(),
    (HYPRE_BigInt) P->Height(),
    (HYPRE_BigInt) P->Width(),
    P->GetI(),
    P->GetJ(),
    P->GetData(),
    row_starts,
    col_starts
  ); 
}


void Solve(
  FiniteElementSpace &fespace,
  PlasmaModelBase *model,
  GridFunction &x,
  int &kdim,
  int &max_newton_iter,
  int &max_krylov_iter,
  double &newton_tol,
  double &krylov_tol, 
  double &Ip,
  int N_control,
  int do_control,
  int obj_option,
  double &obj_weight,
  double &rho_gamma,
  Mesh *mesh,
  ExactForcingCoefficient *exact_forcing_coeff,
  ExactCoefficient *exact_coefficient,
  InitialCoefficient *init_coeff,
  bool include_plasma,
  double &weight_coils,
  double &weight_solenoids,
  Vector *uv,  // External coil currents [I_1, I_2, ..., I_N]
  double &alpha,
  int &PC_option,
  int &max_amr_levels,  // Renamed from max_levels
  int &max_dofs,
  double &light_tol,
  double &alpha_in,
  double &gamma_in,
  int amg_cycle_type,
  int amg_num_sweeps_a,
  int amg_num_sweeps_b,
  int amg_max_iter,
  double amr_frac_in,
  double amr_frac_out
) {

  // Initialize MPI and Hypre so we can use AMG
  Mpi::Init();
  Hypre::Init();

  // Initialize containers for magnetic flux psi, magnetic field B, and toroidal magnetic field function f
  GridFunction psi_r(&fespace);
  GridFunction psi_z(&fespace);
  FieldCoefficient BrCoeff(&x, &psi_r, &psi_z, model, fespace, 0);  // FieldCoefficient is defined in field.cpp
  FieldCoefficient BpCoeff(&x, &psi_r, &psi_z, model, fespace, 1);
  FieldCoefficient BzCoeff(&x, &psi_r, &psi_z, model, fespace, 2);
  GridFunction Br_field(&fespace);
  GridFunction Bp_field(&fespace);
  GridFunction Bz_field(&fespace);
  GridFunction f(&fespace);

  // Save data in the VisIt format for visualization
  char outname[60];
  sprintf(outname, "out/gs_model%d_pc%d_cyc%d_it%d", model->get_model_choice(), PC_option, amg_cycle_type, amg_max_iter);
  VisItDataCollection visit_dc(outname, fespace.GetMesh());
  visit_dc.RegisterField("psi", &x);
  visit_dc.RegisterField("Br", &Br_field);
  visit_dc.RegisterField("Bp", &Bp_field);
  visit_dc.RegisterField("Bz", &Bz_field);

  // NOTE: typically you finalize saving the data to the VisIt format by visit_dc.save();.
  // This line is later in the code, between lines 1000-1200.

  double psi_x;  // Psi at X-point
  double f_x;  // Constant set by the vacuum toroidal field

  // Solve the optimization problem of determining currents to fit the desired plasma shape
  if (do_control) {

    // Initialize log for plasma current, alpha, X-point, and magnetic axis values per iteration in out_iter/ directory
    FILE *fp;
    char filename[60];
    sprintf(filename, "out_iter/iters_model%d_pc%d_cyc%d_it%d.txt", model->get_model_choice(), PC_option, amg_cycle_type, amg_max_iter);
    fp = fopen(filename, "w");

    // Print initial currents
    printf("Initial currents: [");
    for (int i = 0; i < uv->Size(); ++i) { printf("%.3e ", (*uv)[i]); };  // uv: list of external coil currents
    printf("]\n\n");

    // Initialize Lagrange multipliers pv and lv (corresponding to p and \lambda in the paper, respectively)
    GridFunction pv(&fespace);
    pv = 0.0;
    double lv = 0.0;

    // Initialize residuals and regularization
    GridFunction eq_res(&fespace);  // Residual of the GS equation \Delta^* \psi - J_{\phi}(\psi)
    Vector reg_res(uv->Size());  // Regularization term R(u) = \frac{1}{2} u^T H u, where u is a vector of external coil currents I_j, j = 1, ... , N
    GridFunction opt_res(&fespace);  // Optimization residual \psi - \psi_{target}

    // Initialize b1, b2, b3: elements of RHS vector. The other elements b4 and b5 are initialized later in the Newton iteration.
    GridFunction b1(&fespace);
    Vector b2(uv->Size());
    GridFunction b3(&fespace);
    b1 = 0.0;
    b2 = 0.0;
    b3 = 0.0;

    // Define error estimator for AMR
    // MFEM uses ErrorEstimator objects to determine where the mesh should be refined in AMR
    // The "x" parameter is the current finite element solution
    DiffusionIntegratorCoefficient diff_op_coeff(model);  // DiffusionIntegratorCoefficient is defined in diffusion_term.cpp
    DiffusionIntegrator *integ = new DiffusionIntegrator(diff_op_coeff);
    ErrorEstimator *estimator{nullptr};
    estimator = new LSZienkiewiczZhuEstimator(*integ, x);  // Estimator: Zienkiewicz-Zhu error estimation
    RegionalThresholdRefiner refiner(*estimator);  // MFEM mesh refiner that uses the estimator to decide which elements to refine

    // Initialize time tracker
    auto t_init = std::chrono::high_resolution_clock::now();

    // ============================================================================
    // AMR loop
    // ============================================================================

    // Initialize vectors storing magnetic axis, X-point, and cpasma_vals (TODO: what is cpasma_vals? The plasma domain?)
    vector<double> psi_ma_vals, psi_x_vals, cpasma_vals;

    for (int it_amr = 0; it_amr <= max_amr_levels; ++it_amr) {
      int total_gmres = 0;
      int cdofs = fespace.GetTrueVSize();

      // printf("Number of unknowns: %d\n", cdofs);

      // Save per-iteration mesh snapshot (matches the mesh used for this iteration's Newton solve)
      char name_mesh[60];
      sprintf(name_mesh, "gf/mesh_amr%d_model%d_pc%d_cyc%d_it%d.mesh", it_amr, model->get_model_choice(), PC_option, amg_cycle_type, amg_max_iter);
      mesh->Save(name_mesh);

      // ============================================================================
      // Define and assemble PDE operator components
      // ============================================================================

      // Initialize the RHS forcing term for the GS equation due to coil currents u
      LinearForm coil_term(&fespace);

      // Initialize the coefficient matrix F -- maps coil currents -> PDE forcing
      SparseMatrix *F = new SparseMatrix(fespace.GetNDofs(), num_currents);

      // Initialize the elliptic PDE operator (LHS of GS equation + plasma contributions + far-field BCs)
      BilinearForm diff_operator(&fespace);

      // Assemble PDE operators
      DefineRHS(*model, rho_gamma, *mesh, *exact_coefficient, *exact_forcing_coeff, coil_term, F);
      DefineLHS(*model, rho_gamma, diff_operator);

      // ============================================================================
      // Define objective function components
      // ============================================================================

      // Precompute quadrature point data for objective and constraints
      init_coeff->compute_QP(N_control, mesh, &fespace);  // TODO: where does init_coeff come from?

      // Compute gradient w.r.t. ψ
      Vector g_ = init_coeff->compute_g();
      
      // Compute Hessian w.r.t. ψ
      SparseMatrix *K_ = init_coeff->compute_K();

      // Quadrature point weighting coefficients and indices
      std::vector<Vector>     *alpha_coeffs = init_coeff->get_alpha();
      std::vector<Array<int>> *J_inds       = init_coeff->get_J();

      // Regularization matrix H -- R(u) = 1/2 uᵀ H u (L2 regularization)
      SparseMatrix *H = new SparseMatrix(num_currents, num_currents);
      for (int i = 0; i < num_currents; ++i) {
        if (i < 5) {  // TODO: this is a hard-coded assumption that there are exactly 5 "coil" currents, and the rest are "solenoid" currents. This should be fixed.
          H->Set(i, i, weight_coils);
        }
        else {
          H->Set(i, i, weight_solenoids);
        }
      }
      H->Finalize();

      // ============================================================================
      // SysOperator and KKT system
      // ============================================================================

      // Define system operator
      SysOperator op(&diff_operator, &coil_term, model, &fespace, mesh, attr_lim, &x, F, uv, H, K_, &g_, alpha_coeffs, J_inds, &alpha, include_plasma);  // This is in sys_operator.cpp. TODO: does SysOperator clean up memory?
      op.set_i_option(obj_option);
      op.set_obj_weight(obj_weight);  

      // Set size of blocks in equation (use TrueVSize for hanging node support)
      int tdof = fespace.GetTrueVSize();
      Array<int> row_offsets(3);
      row_offsets[0] = 0;
      row_offsets[1] = tdof;
      row_offsets[2] = 2 * tdof;

      // ============================================================================
      // Newton loop
      // ============================================================================

      // TODO: memory leak potential in this Newton loop. Every time a "new" object is created, there is a memory leak risk if no "delete" is added afterwards. No "delete" lines are present in the code.

      // Inexact newton settings
      double eta_last = 0.0;
      double sg_threshold = 0.1;
      double lin_rtol_max = krylov_tol;
      double eta = krylov_tol;

      double error_old;
      double error;

      for (int i = 0; i <= max_newton_iter; ++i) {

        // Debugging
        MFEM_VERIFY(x.Size() > 0, "x vector is empty!");
        MFEM_VERIFY(uv->Size() > 0, "uv vector is empty!");

        // Compute vector and matrix components of the block Newton system
        op.NonlinearEquationRes(x, uv, alpha);

        // Get operators
        SparseMatrix By = op.get_By();
        double C = op.get_plasma_current();
        double Ca = op.get_Ca();
        Vector Cy = op.get_Cy();
        Vector Ba = op.get_Ba();
        SparseMatrix *AMat = op.compute_hess_obj(x);
        Vector g = op.compute_grad_obj(x);

        // Print plasma current
        printf("plasma_current = %10.8e\n", C / op.get_mu());
        printf("alpha = %10.8e\n", alpha);

        // Log plasma current in out_iter/ directory
        fprintf(fp, "plasma_current = %10.8e\n", C / op.get_mu());
        fprintf(fp, "alpha = %10.8e\n", alpha);

        // Locate X-point (psi_x) and magnetic axis (psi_ma)
        psi_x = op.get_psi_x();
        double* x_x = op.get_x_x();
        double* x_ma = op.get_x_ma();
        double psi_ma = op.get_psi_ma();
        BrCoeff.set_psi_vals(psi_x, psi_ma);
        BpCoeff.set_psi_vals(psi_x, psi_ma);
        BzCoeff.set_psi_vals(psi_x, psi_ma);

        psi_ma_vals.push_back(psi_ma);
        psi_x_vals.push_back(psi_x);
        cpasma_vals.push_back(C / op.get_mu());

        // Print X-point and magnetic axis
        printf("psi_x = %10.8e; r_x = %10.8e; z_x = %10.8e\n", psi_x, x_x[0], x_x[1]);
        printf("psi_ma = %10.8e; r_ma = %10.8e; z_ma = %10.8e\n", psi_ma, x_ma[0], x_ma[1]);

        // Log X-point and magnetic axis in out_iter/ directory
        fprintf(fp, "psi_x = %10.8e; r_x = %10.8e; z_x = %10.8e\n", psi_x, x_x[0], x_x[1]);
        fprintf(fp, "psi_ma = %10.8e; r_ma = %10.8e; z_ma = %10.8e\n", psi_ma, x_ma[0], x_ma[1]);


        // Compute RHS vector components (equation 4.11 from paper)

        // -b1 = Gy + By^T p + Cy lambda
        opt_res = g;
        By.AddMultTranspose(pv, opt_res);
        add(opt_res, lv, Cy, opt_res);
        b1 = opt_res;
        b1 *= -1.0;

        // -b2 = reg_res = H u^n - F^T p^n
        H->Mult(*uv, reg_res);
        F->AddMultTranspose(pv, reg_res, -1.0);
        b2 = reg_res;
        b2 *= -1.0;

        // -b3 = eq_res = B(y^n) - F u^n
        eq_res = op.get_res();
        b3 = eq_res;
        b3 *= -1.0;

        // -b4 = B_a^T p^n + C_a l^n
        double b4 = Ba * pv + Ca * lv;
        b4 *= -1.0;

        // -b5 = C - Ip * mu
        double b5 = C - Ip * op.get_mu();  // TODO: mu is not present in the calculation of b5 in equation 4.11 of the paper--check if this is a bug here.
        b5 *= -1.0;

        // Save equilibrium residual (b3)
        char name_eq_res[60];
        sprintf(name_eq_res, "gf/eq_res_amr%d_i%d.gf", it_amr, i);
        eq_res.Save(name_eq_res);

        // Get max errors for residuals
        Vector eq_res_true(fespace.GetTrueVSize());
        ToTrueDofs(eq_res, eq_res_true, fespace);
        error = GetMaxError(eq_res_true);
        printf("newton error (eq_res tdof): %.3e  (newton_tol=%.1e)\n",
               error, newton_tol);
        // double max_opt_res = op.get_mu() * GetMaxError(opt_res);
        // double max_reg_res = GetMaxError(reg_res) / op.get_mu();

        // Inexact Newton. Adjust the relative tolerance eta as the Newton iteration converges
        if (i > 0) {
          eta = gamma_in * pow(error / error_old, alpha_in);  // alpha_in and gamma_in: inexact Newton parameters inputted as arguments through main.cpp
          double sg_eta = gamma_in * pow(eta_last, alpha_in); // safeguard to prevent eta from becoming too small too quickly
          if (sg_eta > sg_threshold) {  // if eta is above a predefined threshold, truncate eta
            eta = max(eta, sg_eta);
          }
          eta = min(eta, lin_rtol_max);
          eta_last = eta;  // for the next iteration
        }

        printf("inexact newton rtol: %.2e\n", eta);
        printf("\n");

        // Newton iteration stopping conditions
        if (error < newton_tol) {
          break;
        }
        if (i == max_newton_iter) {
          break;
        }

        // Save current residual for the next iteration
        error_old = error;

        // Compute the C matrix in the block linear system: CMat

        // invH is diagonal inverse of H (regularization on coil currents)
        const int nu = uv->Size();  // number of coil currents (control dofs)
        mfem::SparseMatrix* invH = new mfem::SparseMatrix(nu, nu);
        for (int j = 0; j < nu; ++j) {
            const double hjj = (*H)(j, j);
            MFEM_VERIFY(hjj > 0.0, "H must be positive diagonal.");
            invH->Set(j, j, 1.0 / hjj);
        }
        invH->Finalize();

        // Calculations to form CMat
        SparseMatrix *FT = Transpose(*F);
        SparseMatrix *mF = Add(-1.0, *F, 0.0, *F);  // This is just calculating -F
        SparseMatrix *invHFT = Mult(*invH, *FT);
        SparseMatrix *mFinvHFT = Mult(*mF, *invHFT);
        SparseMatrix *FinvH = Mult(*F, *invH);
        SparseMatrix *mMuFinvHFT = Add(op.get_mu(), *mFinvHFT, 0.0, *mFinvHFT);
        SparseMatrix *MuFinvH = Add(op.get_mu(), *FinvH, 0.0, *FinvH);
        double scale = 1.0 / sqrt(mMuFinvHFT->MaxNorm());

        // Form CMat
        SparseMatrix *CMat = Add(scale * scale, *mMuFinvHFT, 0.0, *mMuFinvHFT);

        // Compute the B and B^T matrices in the block linear system: BMat and BTMat

        // Compute 1 / Ca Cy Ba^T (in VSize space)
        int vsize = fespace.GetVSize();
        SparseMatrix *CyBa = new SparseMatrix(vsize, vsize);
        for (int j = 0; j < vsize; ++j) {
          for (int k = 0; k < vsize; ++k) {
            if (Ba(j) * Cy(k) != 0.0) {
              CyBa->Set(j, k, Cy(j) * Ba(k) / Ca);
            }
          }
        }
        CyBa->Finalize();

        SparseMatrix *ByT = Transpose(By);
        SparseMatrix *ScaleByT = Add(scale, *ByT, 0.0, *CyBa);
        SparseMatrix *ScaleBy = Transpose(*ScaleByT);

        // Form BMat and BTMat (in VSize space)
        SparseMatrix *BMat = Add(scale, *ByT, -scale, *CyBa);
        SparseMatrix *BTMat = Transpose(*BMat);

        // ============================================================================
        // Convert all operators to true-DOF space for hanging node support
        // A_true = P^T * A * P,  v_true = P^T * v
        // ============================================================================
        SparseMatrix *AMat_t   = ToTrueDofs(*AMat,     fespace);
        SparseMatrix *BMat_t   = ToTrueDofs(*BMat,     fespace);
        SparseMatrix *BTMat_t  = ToTrueDofs(*BTMat,    fespace);
        SparseMatrix *CMat_t   = ToTrueDofs(*CMat,     fespace);

        // Convert vectors used in preconditioner to true-DOF space
        Vector Ba_t(tdof), Cy_t(tdof);
        ToTrueDofs(Ba, Ba_t, fespace);
        ToTrueDofs(Cy, Cy_t, fespace);

        // Build block linear system (in true-DOF space)
        BlockOperator BlockSystem(row_offsets);
        BlockVector rhs(row_offsets);

        int ind_x, ind_p;

        // If we have a non-symmetric block matrix
        if (PC_option == 0) {  // PC_option: preconditioner option
          ind_x = 0;
          ind_p = 1;
        }

        // If we have a symmetric block matrix
        else {
          ind_x = 1;
          ind_p = 0;
        }

        // Form block matrix (true-DOF space)
        BlockSystem.SetBlock(0, ind_x, AMat_t);
        BlockSystem.SetBlock(0, ind_p, BMat_t);
        BlockSystem.SetBlock(1, ind_x, BTMat_t);
        BlockSystem.SetBlock(1, ind_p, CMat_t);

        // Write contents of matrices to text files in CSR format
        FILE *fp_spy;
        char filename_spy[60];
        sprintf(filename_spy, "spys/spy_model%d_amr%d.txt", model->get_model_choice(), it_amr);
        fp_spy = fopen(filename_spy, "w");
        fprintf(fp_spy, "AMat\n");
        WriteSparseMatrixToFile(fp_spy, AMat_t);
        fprintf(fp_spy, "\nBMat\n");
        WriteSparseMatrixToFile(fp_spy, BMat_t);
        fprintf(fp_spy, "\nCMat\n");
        WriteSparseMatrixToFile(fp_spy, CMat_t);

        // Define RHS of the block linear system (equation 5.3 of paper)
        // Compute in VSize first, then convert to true-DOF
        // c1 = b1 - Cy b4 / Ca
        // c2 = b3 + mu F H^{-1} b_2 - Ba b5 / Ca
        Vector rhs0_full(vsize), rhs1_full(vsize);
        add(1.0, b1, -b4 / Ca, Cy, rhs0_full);  // c1 (VSize)
        MuFinvH->Mult(b2, rhs1_full);  // c2 (VSize)
        rhs1_full += b3;
        add(1.0, rhs1_full, - b5 / Ca, Ba, rhs1_full);
        rhs1_full *= scale;

        // Convert RHS to true-DOF space
        rhs = 0;
        ToTrueDofs(rhs0_full, rhs.GetBlock(0), fespace);
        ToTrueDofs(rhs1_full, rhs.GetBlock(1), fespace);

        // Configure the FGMRES solver
        FGMRESSolver solver;
        solver.SetAbsTol(1e-12);
        solver.SetRelTol(eta);
        solver.SetMaxIter(max_krylov_iter);
        solver.SetOperator(BlockSystem);
        solver.SetKDim(kdim);
        solver.SetPrintLevel(-1);

        // Initialize solution guess dx to zero (true-DOF space)
        BlockVector dx(row_offsets);
        dx = 0.0;

        double dalpha, dlv;

        // ============================================================================
        // Preconditioners (all in true-DOF space)
        // ============================================================================

        // Preconditioning: 0 = block diagonal PC, 5 = block upper triangular PC, 6 = block lower triangular PC
        if (PC_option == 0 || PC_option == 5 || PC_option == 6) {

          Solver *inv_BT, *inv_B;

          // Build AMG solvers for B_y and B_y^T using Hypre (true-DOF matrices)
          SparseMatrix *ScaleByT_t = ToTrueDofs(*ScaleByT, fespace);
          SparseMatrix *ScaleBy_t  = ToTrueDofs(*ScaleBy,  fespace);
          HypreParMatrix *B_Hypre = ConvertToHypre(ScaleByT_t);
          HypreParMatrix *BT_Hypre = ConvertToHypre(ScaleBy_t);
          HypreBoomerAMG *B_AMG = new HypreBoomerAMG(*B_Hypre);
          HypreBoomerAMG *BT_AMG = new HypreBoomerAMG(*BT_Hypre);

          // Configure AMG parameters for AMG(B_y)
          B_AMG->SetPrintLevel(0);
          B_AMG->SetCycleType(amg_cycle_type);  // choose V-cycle/W-cycle
          B_AMG->SetCycleNumSweeps(amg_num_sweeps_a, amg_num_sweeps_b);
          B_AMG->SetMaxIter(amg_max_iter);

          // Configure AMG parameters for AMG(B_y^T)
          BT_AMG->SetPrintLevel(0);
          BT_AMG->SetCycleType(amg_cycle_type);
          BT_AMG->SetCycleNumSweeps(amg_num_sweeps_a, amg_num_sweeps_b);
          BT_AMG->SetMaxIter(amg_max_iter);

          inv_B = B_AMG;
          inv_BT = BT_AMG;

          // Block diagonal preconditioner: equation 5.4 from paper
          if (PC_option == 0) {
            BlockDiagonalPreconditioner BlockPrec(row_offsets);
            BlockPrec.SetDiagonalBlock(0, inv_B);
            BlockPrec.SetDiagonalBlock(1, inv_BT);
            solver.SetPreconditioner(BlockPrec);

            solver.Mult(rhs, dx);
            fprintf(fp, "amr=%d newton=%d iters=%d\n", it_amr, i, solver.GetNumIterations());
          }

          // Block upper triangular preconditioner: equation 5.5 from paper
          else if (PC_option == 5) {
            SchurPC *SCPC = new SchurPC(AMat_t, CMat_t, inv_B, inv_BT, &Ba_t, &Cy_t, Ca, 1);
            solver.SetPreconditioner(*SCPC);

            solver.Mult(rhs, dx);
            fprintf(fp, "amr=%d newton=%d iters=%d\n", it_amr, i, solver.GetNumIterations());
          }

          // Block lower triangular preconditioner: equation 5.6 from paper
          else if (PC_option == 6) {
            SchurPC *SCPC = new SchurPC(AMat_t, CMat_t, inv_B, inv_BT, &Ba_t, &Cy_t, Ca, 2);
            solver.SetPreconditioner(*SCPC);

            solver.Mult(rhs, dx);
            fprintf(fp, "amr=%d newton=%d iters=%d\n", it_amr, i, solver.GetNumIterations());
          }
        }
        
        // Terminate if preconditioner option is unsupported
        else {
          fprintf(stderr, "ERROR: Unsupported PC_option=%d (currently supported: 0, 5, 6)\n", PC_option);
          MFEM_ABORT("Unsupported PC_option");
        }

        // Check for convergence
        if (solver.GetConverged()) {
          printf("GMRES converged in %d iterations with a residual norm of %e\n", solver.GetNumIterations(), solver.GetFinalNorm());
        }
        else {
          printf("GMRES did not converge in %d iterations. Residual norm is %e\n", solver.GetNumIterations(), solver.GetFinalNorm());
        }
        total_gmres += solver.GetNumIterations();

        if (solver.GetNumIterations() == -1) {
          printf("failure...\n");
          return;
        }
        dx.GetBlock(ind_p) *= scale;

        // Prolong solution increments from true-DOF space back to VSize
        Vector dx_x_full(vsize), dx_p_full(vsize);
        ToFullDofs(dx.GetBlock(ind_x), dx_x_full, fespace);
        ToFullDofs(dx.GetBlock(ind_p), dx_p_full, fespace);

        // Update solution in VSize space
        x += dx_x_full;
        pv += dx_p_full;

        // Update coil currents (invHFT and invH operate on coil-space vectors, not DOF vectors)
        invHFT->AddMult(dx_p_full, *uv);
        invH->AddMult(b2, *uv);

        // Compute alpha and lambda increments using VSize vectors
        dalpha = (b5 - (Cy * dx_x_full)) / Ca;
        dlv = (b4 - (Ba * dx_p_full)) / Ca;
        alpha += dalpha;
        lv += dlv;

        // ============================================================================
        // Calculate residuals for the Newton system after solve
        // ============================================================================

        // Residual = RHS of 2 x 2 block system (eq. 4.13 in paper)

        // Residuals are assembled in VSize (to match the VSize operators and
        // RHS) and then projected to true-DOF via P^T before norming. On a
        // non-conforming mesh the VSize residual has non-zero slave-row
        // entries that the true-DOF solve never had to zero; the raw VSize
        // L-inf would misrepresent them as Newton non-convergence. On a
        // conforming mesh P==nullptr and the projection is a no-op.
        const int tvsize = fespace.GetTrueVSize();

        // First block row in RHS
        Vector res1(vsize);
        res1 = 0.0;
        AMat->AddMult(dx_x_full, res1);
        ByT->AddMult(dx_p_full, res1);
        add(res1, dlv, Cy, res1);
        add(res1, -1.0, b1, res1);
        Vector res1_true(tvsize);
        ToTrueDofs(res1, res1_true, fespace);
        printf("res_1: %.2e\n", GetMaxError(res1_true));

        // Second block row in RHS
        Vector res2(vsize);
        res2 = 0.0;
        mMuFinvHFT->AddMult(dx_p_full, res2);
        MuFinvH->Mult(b2, res2);
        res2 *= -1.0;
        By.AddMult(dx_x_full, res2);
        mMuFinvHFT->AddMult(dx_p_full, res2);
        add(res2, dalpha, Ba, res2);
        add(res2, -1.0, b3, res2);
        Vector res2_true(tvsize);
        ToTrueDofs(res2, res2_true, fespace);
        printf("res_2: %.2e\n", GetMaxError(res2_true));

        // ============================================================================
        // Save/print/update parameters post-solve
        // ============================================================================

        // Print currents
        printf("Currents: [");
        for (int i = 0; i < uv->Size(); ++i) {printf("%.3e ", (*uv)[i]);}
        printf("]\n");

        // Save grid function
        char name_[60];
        sprintf(name_, "gf/xtmp_amr%d.gf", it_amr);
        x.Save(name_);
        char name[60];
        sprintf(name, "gf/xtmp_amr%d_i%d.gf", it_amr, i);
        x.Save(name);

        // Save magnetic field components
        Br_field.Save("gf/Br.gf");
        Bp_field.Save("gf/Bp.gf");
        Bz_field.Save("gf/Bz.gf");

        // Compute psi_r and psi_z
        x.GetDerivative(1, 0, psi_r);
        x.GetDerivative(1, 1, psi_z);

        //
        Br_field.ProjectCoefficient(BrCoeff);
        Bp_field.ProjectCoefficient(BpCoeff);
        Bz_field.ProjectCoefficient(BzCoeff);

        visit_dc.Save();
      }
      
      // Stopping condition: max AMR levels
      if (it_amr >= max_amr_levels) {
        printf("Reached the maximum number of refinement levels\n");
        break;
      }

      // Stopping condition: max number of DOFs
      if (cdofs > max_dofs) {
        cout << "Reached the maximum number of dofs. Stop." << endl;
        break;
      }

      // Stopping condition:
      // Logic is in amr.cpp: RegionalThresholdRefiner::ApplyRef. Stop conditions are:
      // 1) number of elements exceeds max_elements, 2) total estimator norm below total_err_goal,
      // 3) no elements marked for refinement
      f = op.get_f();
      refiner.ApplyRef(*mesh, 1000, amr_frac_in, amr_frac_out);
      if (refiner.Stop()) {
        cout << "Stopping criterion satisfied. Stop." << endl;
        break;
      }
      
      else {
        printf("Refining mesh, AMR iteration %d\n", it_amr+1);
      }

      // update variables due to refinement
      fespace.Update();
      x.Update();
      psi_r.Update();
      psi_z.Update();
      Br_field.Update();
      Bp_field.Update();
      Bz_field.Update();
      pv.Update();
      eq_res.Update();
      opt_res.Update();
      b1.Update();
      b3.Update();
      
      printf("Number of true DOFs: %d\n", fespace.GetTrueVSize());
    }

    // Save final mesh (matches the mesh that x lives on after the AMR loop)
    mesh->Save("meshes/mesh_refine.mesh");

    // Print elapsed time to convergence
    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = t_end - t_init;
    printf("time elapsed: %f seconds\n", ms_double.count() / 1000.0);

    // Compute f at X-point
    f_x = model->get_f_x();

    // Write final value of psi_ma to a different file for GEQDSK
    system("mkdir -p ../gslib/GEQDSK");
    ofstream file("../gslib/GEQDSK/GEQDSK_simagx_sibdry_cpasma.txt");
    file << scientific << setprecision(9)
      << setw(16) << psi_ma_vals.back() << "\n"
      << setw(16) << psi_x_vals.back() << "\n"
      << setw(16) << cpasma_vals.back() << "\n";
      file.close();
  }
  
  // do_control == 0
  // Given currents, solve the GS equation
  else {
    GridFunction dx(&fespace);
    dx = 0.0;

    // ============================================================================
    // Define and assemble PDE operator components
    // ============================================================================

    // Initialize the RHS forcing term for the GS equation due to coil currents u
    LinearForm coil_term(&fespace);

    // Initialize the coefficient matrix F -- maps coil currents -> PDE forcing
    SparseMatrix *F = new SparseMatrix(fespace.GetNDofs(), num_currents);

    // Initialize the elliptic PDE operator (LHS of GS equation + plasma contributions + far-field BCs)
    BilinearForm diff_operator(&fespace);

    // Assemble PDE operators
    DefineRHS(*model, rho_gamma, *mesh, *exact_coefficient, *exact_forcing_coeff, coil_term, F);
    DefineLHS(*model, rho_gamma, diff_operator);

    // ============================================================================
    // Define objective function components
    // ============================================================================

    // Precompute quadrature point data for objective and constraints
    init_coeff->compute_QP(N_control, mesh, &fespace);

    // Compute gradient w.r.t. ψ
    Vector g_ = init_coeff->compute_g();

    // Compute Hessian w.r.t. ψ
    SparseMatrix *K_ = init_coeff->compute_K();

    // Quadrature point weighting coefficients and indices
    std::vector<Vector>     *alpha_coeffs = init_coeff->get_alpha();
    std::vector<Array<int>> *J_inds       = init_coeff->get_J();

    // Regularization matrix H -- R(u) = 1/2 uᵀ H u (L2 regularization)
    SparseMatrix *H = new SparseMatrix(num_currents, num_currents);
    for (int i = 0; i < num_currents; ++i) {
      if (i < 5) {  // TODO: this is a hard-coded assumption that there are exactly 5 "coil" currents, and the rest are "solenoid" currents. This should be fixed.
        H->Set(i, i, weight_coils);
      }
      else {
        H->Set(i, i, weight_solenoids);
      }
    }
    H->Finalize();
    
    // ============================================================================
    // SysOperator and KKT system
    // ============================================================================

    // Define system operator
    SysOperator op(&diff_operator, &coil_term, model, &fespace, mesh, attr_lim, &x, F, uv, H, K_, &g_, alpha_coeffs, J_inds, &alpha, include_plasma);
    op.set_i_option(obj_option);
    op.set_obj_weight(obj_weight);

    // ============================================================================
    // Newton loop
    // ============================================================================

    GridFunction eq_res(&fespace);   // Residual of the GS equation
    GridFunction b3(&fespace);
    b3 = 0.0;
    LinearForm out_vec(&fespace);

    double error_old;
    double error;
    for (int i = 0; i < max_newton_iter; ++i) {

      // Compute vector and matrix components of the block Newton system
      op.NonlinearEquationRes(x, uv, alpha);

      // eq_res = B(y^n) - F u^n
      eq_res = op.get_res();
      b3 = eq_res;
      // b3 *= -1.0;  // TODO: is this supposed to be here?

      // Track Newton error
      Vector eq_res_true(fespace.GetTrueVSize());
      ToTrueDofs(eq_res, eq_res_true, fespace);
      error = GetMaxError(eq_res_true);
      if (i == 0) {
        printf("\n i: %3d, max residual: %.3e\n", i, error);
      }
      else {
        printf("\n i: %3d, max residual: %.3e, ratio %.3e\n", i, error, error_old / error);
      }
      error_old = error;

      // Stop if Newton has converged
      if (error < newton_tol) {
        break;
      }

      dx = 0.0;

      // Get operator
      SparseMatrix By = op.get_By();

      // Build a preconditioner for linear solve
      Solver *inv_B;
      HypreParMatrix *B_Hypre = ConvertToHypre(&By);
      HypreBoomerAMG *B_AMG = new HypreBoomerAMG(*B_Hypre);
      B_AMG->SetPrintLevel(0);
      B_AMG->SetCycleType(1);
      B_AMG->SetCycleNumSweeps(1, 1);
      B_AMG->SetMaxIter(1);
      inv_B = B_AMG;

      // Solve Newton system with GMRES
      int gmres_iter = max_krylov_iter;
      double gmres_tol = krylov_tol;
      int gmres_kdim = kdim;
      GMRES(By, dx, b3, *inv_B, gmres_iter, gmres_kdim, gmres_tol, 0.0, 0);

      printf("gmres iters: %d, gmres err: %e\n", gmres_iter, gmres_tol);

      // Newton update step
      x -= dx;

      // Save current iteration
      x.Save("gf/xtmp.gf");
      GridFunction err(&fespace);
      err = out_vec;
      err.Save("gf/res.gf");

      // ParaView
      visit_dc.Save();
    }

    // Final residual check
    op.Mult(x, out_vec);
    Vector out_vec_true(fespace.GetTrueVSize());
    ToTrueDofs(out_vec, out_vec_true, fespace);
    error = GetMaxError(out_vec_true);

    printf("\n\n********************************\n");
    printf("final max residual: %.3e, ratio %.3e\n", error, error_old / error);
    printf("********************************\n\n");
  }

  // Export to GEQDSK file
  system("mkdir -p ../gslib/GEQDSK"); 
  ofstream NewFile("../gslib/GEQDSK/GEQDSK_alpha_f_x_psi_x.txt");
  NewFile << alpha << "\n" << f_x << "\n" << psi_x << "\n"; 
  NewFile.close();
}


double gs(const char *mesh_file, const char *initial_gf, const char *data_file, int order, int d_refine,
          int model_choice,
          double & alpha, double & beta, double & gamma, double & mu, double & Ip,
          double & r0, double & rho_gamma, int max_krylov_iter, int max_newton_iter,
          double & krylov_tol, double & newton_tol,
          double & c1, double & c2, double & c3, double & c4, double & c5, double & c6, double & c7,
          double & c8, double & c9, double & c10, double & c11,
          double & ur_coeff,
          int do_control, int N_control, double & weight_solenoids, double & weight_coils,
          double & weight_obj, int obj_option,
          bool do_manufactured_solution, bool do_initial,
          int & PC_option, int & max_amr_levels, int & max_dofs, double & light_tol,
          double & alpha_in, double & gamma_in,
          int amg_cycle_type, int amg_num_sweeps_a, int amg_num_sweeps_b, int amg_max_iter,
          double amr_frac_in, double amr_frac_out) {

  // External currents
  Vector uv_currents(num_currents);
  uv_currents[0] = c1;
  uv_currents[1] = c2;
  uv_currents[2] = c3;
  uv_currents[3] = c4;
  uv_currents[4] = c5;
  uv_currents[5] = c6;
  uv_currents[6] = c7;
  uv_currents[7] = c8;
  uv_currents[8] = c9;
  uv_currents[9] = c10;
  uv_currents[10] = c11;

  // Solver options
  int kdim = 10000;

  // ============================================================================
  // Process inputs
  // ============================================================================

  // Create a new Mesh object named mesh by reading in the mesh data from the filepath "mesh_file".
  Mesh mesh(mesh_file);
  
  // Save options in model: alpha: multiplier in \bar{S}_{ff'} term, beta: multiplier for S_{p'} term, gamma: multiplier for S_{ff'} term
  const char *data_file_ = "data/fpol_pres_ffprim_pprime.data";  // TODO: what is the difference between this data_file_ versus data_file, which is an argument passed into main.cpp?
  PlasmaModelFile model(mu, data_file_, alpha, beta, gamma, model_choice);

  // Define a finite element space on the mesh. Here we use H1 continuous high-order Lagrange finite elements of the given order.
  H1_FECollection fec(order, mesh.Dimension());
  FiniteElementSpace fespace(&mesh, &fec);
  cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

  // Exact solution
  double r0_ = 1.0;
  double z0_ = 0.0;
  double L_ = 0.35;
  double k_ = M_PI/(2.0*L_);
  ExactForcingCoefficient exact_forcing_coeff(r0_, z0_, k_, model, do_manufactured_solution);
  ExactCoefficient exact_coefficient(r0_, z0_, k_, do_manufactured_solution);

  // ============================================================================
  // Solve
  // ============================================================================

  // Remove control point optimization to solve fixed-boundary GS in order to get initial guesses for the free-boundary GS cases.
  if (do_initial) {
    do_control = false;
  }
  
  // Define the solution x as a finite element grid function in fespace. Set
  // the initial guess to zero, which also sets the boundary conditions.
  GridFunction u(&fespace);
  
  InitialCoefficient init_coeff = read_data_file(data_file);  // data_file is the plasma data file--not sure what that is, but I assume it defines certain plasma parameters?

  // I think that do_manufactured_solution == 1 is used for comparing the GS solver against a known analytical solution
  if (do_manufactured_solution) {

  // Project exact solution onto your finite element function u and save
    u.ProjectCoefficient(exact_coefficient);
    u.Save("gf/exact.gf");
  }
   
  else {

    // If not solving for initial guess
    if (!do_initial) {

      // Load initial GridFunction from file
      ifstream ifs(initial_gf);
      GridFunction lgf(&mesh, ifs);
      lgf.SetSpace(&fespace);
      u = lgf;
    }
    u.Save("gf/initial.gf");
  }

  // Perform uniform mesh refinement and save mesh post-refinement
  for (int i = 0; i < d_refine; ++i) {
    mesh.UniformRefinement();
    fespace.Update();
    u.Update();
  }
  mesh.Save("meshes/mesh.mesh");

  // Save initial solution mesh
  if (do_initial) {
    mesh.Save("meshes/initial.mesh");
  }

  GridFunction x(&fespace);
  x = u;

  // The include_plasma parameter determines if the nonlinear plasma contribution term in the GS equation is included
  // in the solve. When set to false, solver is solving just the diffusion operator + coil contributions.
  bool include_plasma = true;
  if (do_initial) {
    include_plasma = false;
  }

  cout << "Beginning GS Solve." << endl;

  Solve(fespace, &model, x, kdim, max_newton_iter, max_krylov_iter, newton_tol, krylov_tol,
        Ip, N_control, do_control,
        obj_option, weight_obj,
        rho_gamma,
        &mesh,
        &exact_forcing_coeff,
        &exact_coefficient,
        &init_coeff,
        include_plasma,
        weight_coils,
        weight_solenoids,
        &uv_currents,
        alpha,
        PC_option, max_amr_levels, max_dofs, light_tol,
        alpha_in, gamma_in,
        amg_cycle_type, amg_num_sweeps_a, amg_num_sweeps_b, amg_max_iter,
        amr_frac_in, amr_frac_out);

  // Save mesh and solution for initial solve
  if (do_initial) {
    char name_gf_out[60];
    char name_mesh_out[60];
    sprintf(name_gf_out, "initial/initial_guess_g%d.gf", d_refine);
    sprintf(name_mesh_out, "initial/initial_mesh_g%d.mesh", d_refine);

    x.Save(name_gf_out);
    mesh.Save(name_mesh_out);
    printf("Saved solution to %s\n", name_gf_out);
    printf("Saved mesh to %s\n", name_mesh_out);
    printf("glvis -m %s -g %s\n", name_mesh_out, name_gf_out);
  }
   
  // Save mesh and solution for free-boundary solve
  else {
    char name_gf_out[60];
    sprintf(name_gf_out, "gf/final_model%d_pc%d_cyc%d_it%d.gf", model.get_model_choice(), PC_option, amg_cycle_type, amg_max_iter);
    x.Save(name_gf_out);

    printf("glvis -m meshes/mesh_refine.mesh -g %s\n", name_gf_out);

    // Paraview
    ParaViewDataCollection paraview_dc("gs", &mesh);
    paraview_dc.SetPrefixPath("ParaView");
    paraview_dc.SetLevelsOfDetail(order);
    paraview_dc.SetCycle(0);
    paraview_dc.SetDataFormat(VTKFormat::BINARY);
    paraview_dc.SetHighOrderOutput(true);
    paraview_dc.SetTime(0.0); // set the time
    paraview_dc.RegisterField("psi",&x);
    paraview_dc.Save();
  }

  // Exact solution test path
  if (do_manufactured_solution) {
    GridFunction diff(&fespace);
    add(x, -1.0, u, diff);
    double num_error = GetMaxError(diff);
    diff.Save("gf/error.gf");
    double L2_error = x.ComputeL2Error(exact_coefficient);
    printf("\n\n********************************\n");
    printf("Numerical error: %.3e\n", num_error);
    printf("L2 error: %.3e\n", L2_error);
    printf("********************************\n\n");

    return L2_error;
  }
  
  else {
    return 0.0;
  }
}
