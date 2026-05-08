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
// File overview
// ---------------------------------------------------------------------------
//
// Implements the Newton-based free-boundary Grad-Shafranov solver. The only
// public entry point is `gs(...)` (declared in gs.hpp); all other functions
// in this file are static.
//
// The major functions, in roughly the order they're called:
//
//   gs
//       Public entry point. Loads the mesh + FE space, builds the plasma
//       model and initial guess, applies uniform refinement, then dispatches
//       to SolveControlProblem (free-boundary) or SolveFixedBoundaryProblem
//       (fixed-boundary) based on do_control.
//
//   SolveControlProblem
//       Free-boundary GS solve. Outer AMR loop containing a Newton iteration
//       that builds and solves the reduced 2x2 KKT block system at each step.
//
//   SolveFixedBoundaryProblem
//       Fixed-boundary GS solve. A single Newton iteration with a simple AMG
//       preconditioner; no AMR, no KKT.
//
//   ComputeNewtonRHS
//       Compute b1..b5 of the Newton-system RHS (eq. 4.11 of the paper).
//
//   BuildReducedKKTBlocks
//       Build the reduced 2x2 block KKT matrices (LHS of eq. 4.13).
//
//   BuildBlockPreconditioner
//       Build the block preconditioner used by FGMRES on the reduced KKT
//       system: block diagonal (eq. 5.4), block upper-triangular Schur
//       (eq. 5.5), or block lower-triangular Schur (eq. 5.6), selected by
//       PC_option.
//
// ---------------------------------------------------------------------------


// True-DOF conversion helpers for nonconforming mesh support
// Convert a VSize x VSize SparseMatrix to TrueVSize x TrueVSize
// via A_true = R * A * P  (conforming projection, R = P^T)
static SparseMatrix* ToTrueDofs(const SparseMatrix &A, const FiniteElementSpace &fes) {

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
// to TrueVSize by v_true = P^T v_full
static void ToTrueDofs(const Vector &v_full, Vector &v_true, const FiniteElementSpace &fes) {
    const SparseMatrix *P = fes.GetConformingProlongation();
    if (!P) { v_true = v_full; return; }
    P->MultTranspose(v_full, v_true);
}

// Prolong a TrueVSize Vector back to VSize
static void ToFullDofs(const Vector &v_true, Vector &v_full, const FiniteElementSpace &fes) {
    const SparseMatrix *P = fes.GetConformingProlongation();
    if (!P) { v_full = v_true; return; }
    P->Mult(v_true, v_full);
}


static void WriteSparseMatrixToFile(FILE *fp, SparseMatrix *Mat) {
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


static HypreParMatrix *ConvertToHypre(SparseMatrix *P) {
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


static void DefineRHS(
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

    if (
      attrib == attr_ext ||
      attrib == attr_vv  ||  // exterior domain
      attrib == attr_lim ||  // limiter domain
      attrib == 1100         // TODO: what does attrib == 1100 correspond to?
    ) {
      // do nothing
    }

    else {
      
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


static void DefineLHS(PlasmaModelBase &model, double rho_gamma, BilinearForm &diff_operator) {
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
  PWConstCoefficient pw_coeff(pw_vector_);
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


// Assemble the LHS bilinear form, the RHS linear form, and the F coil-current
// matrix that maps coil currents to a forcing on each PDE region.
static void AssemblePDEOperators(
  PlasmaModelBase &model, double rho_gamma, Mesh &mesh,
  FiniteElementSpace &fespace,
  ExactCoefficient &exact_coefficient,
  ExactForcingCoefficient &exact_forcing_coeff,
  LinearForm &coil_term,
  BilinearForm &diff_operator,
  std::unique_ptr<SparseMatrix> &F)
{
  F.reset(new SparseMatrix(fespace.GetNDofs(), num_currents));
  DefineRHS(model, rho_gamma, mesh, exact_coefficient, exact_forcing_coeff, coil_term, F.get());
  DefineLHS(model, rho_gamma, diff_operator);
}


// Bundle of objective-function terms:
//   H            — coil-current regularization matrix (owned)
//   K            — quadratic-form matrix for the target-shape penalty,
//                  returned by InitialCoefficient::compute_K() as a fresh
//                  allocation. Stored as a raw pointer here; currently not freed.
//   g            — gradient of the objective w.r.t. ψ
//   alpha_coeffs,
//   J_inds       — quadrature weights and dof indices, owned by InitialCoefficient
struct ObjectiveTerms {
  std::unique_ptr<SparseMatrix> H;
  SparseMatrix *K;
  Vector g;
  std::vector<Vector>     *alpha_coeffs;
  std::vector<Array<int>> *J_inds;
};


// Assemble the objective and regularization terms
static void AssembleObjectiveTerms(
  InitialCoefficient &init_coeff,
  int N_control, Mesh &mesh, FiniteElementSpace &fespace,
  double weight_coils, double weight_solenoids,
  ObjectiveTerms &terms)
{
  init_coeff.compute_QP(N_control, &mesh, &fespace);
  terms.g = init_coeff.compute_g();
  terms.K = init_coeff.compute_K();
  terms.alpha_coeffs = init_coeff.get_alpha();
  terms.J_inds = init_coeff.get_J();

  terms.H.reset(new SparseMatrix(num_currents, num_currents));
  for (int i = 0; i < num_currents; ++i) {
    if (i < 5) {  // TODO: hard-coded assumption that the first 5 entries are PF coils and the rest are CS solenoids.
      terms.H->Set(i, i, weight_coils);
    } else {
      terms.H->Set(i, i, weight_solenoids);
    }
  }
  terms.H->Finalize();
}


// Open a per-iteration log file
static std::unique_ptr<FILE, int(*)(FILE*)> OpenIterationLog(
  int model_choice, int PC_option, int amg_cycle_type, int amg_max_iter)
{
  char filename[60];
  sprintf(filename, "out_iter/iters_model%d_pc%d_cyc%d_it%d.txt",
          model_choice, PC_option, amg_cycle_type, amg_max_iter);
  return std::unique_ptr<FILE, int(*)(FILE*)>(fopen(filename, "w"), &fclose);
}


// Reduced 2x2 block KKT system from equation 4.13 of the paper
struct KKTBlocks {
  std::unique_ptr<SparseMatrix> invH, FT, mF, invHFT, mFinvHFT, FinvH;
  std::unique_ptr<SparseMatrix> mMuFinvHFT, MuFinvH, CMat, CyBa, ByT;
  std::unique_ptr<SparseMatrix> ScaleByT, ScaleBy, BMat, BTMat;
  std::unique_ptr<SparseMatrix> AMat_t, BMat_t, BTMat_t, CMat_t;
  Vector Ba_t, Cy_t;
  double scale = 0.0;
  int ind_x = 0, ind_p = 0;
};


// Build the reduced KKT block matrices and assemble them into BlockSystem.
// This is the LHS of equation 4.13 from paper.
static void BuildReducedKKTBlocks(
  SysOperator &op, FiniteElementSpace &fespace,
  SparseMatrix *AMat, const SparseMatrix &By,
  const SparseMatrix &F, const SparseMatrix &H,
  const Vector &Ba, const Vector &Cy, double Ca,
  const Vector &uv, int PC_option,
  BlockOperator &BlockSystem,
  KKTBlocks &kkt)
{
  // invH is the diagonal inverse of H (regularization on coil currents).
  const int nu = uv.Size();
  kkt.invH.reset(new SparseMatrix(nu, nu));
  for (int j = 0; j < nu; ++j) {
    const double hjj = H(j, j);
    MFEM_VERIFY(hjj > 0.0, "H must be positive diagonal.");
    kkt.invH->Set(j, j, 1.0 / hjj);
  }
  kkt.invH->Finalize();

  // Calculations to form CMat
  kkt.FT.reset(Transpose(F));
  kkt.mF.reset(Add(-1.0, F, 0.0, F));  // -F
  kkt.invHFT.reset(Mult(*kkt.invH, *kkt.FT));
  kkt.mFinvHFT.reset(Mult(*kkt.mF, *kkt.invHFT));
  kkt.FinvH.reset(Mult(F, *kkt.invH));
  kkt.mMuFinvHFT.reset(Add(op.get_mu(), *kkt.mFinvHFT, 0.0, *kkt.mFinvHFT));
  kkt.MuFinvH.reset(Add(op.get_mu(), *kkt.FinvH, 0.0, *kkt.FinvH));
  kkt.scale = 1.0 / sqrt(kkt.mMuFinvHFT->MaxNorm());

  // Form CMat
  kkt.CMat.reset(Add(kkt.scale * kkt.scale, *kkt.mMuFinvHFT, 0.0, *kkt.mMuFinvHFT));

  // Compute (1/Ca) Cy Ba^T (in VSize space)
  const int vsize = fespace.GetVSize();
  kkt.CyBa.reset(new SparseMatrix(vsize, vsize));
  for (int j = 0; j < vsize; ++j) {
    for (int k = 0; k < vsize; ++k) {
      if (Ba(j) * Cy(k) != 0.0) {
        kkt.CyBa->Set(j, k, Cy(j) * Ba(k) / Ca);
      }
    }
  }
  kkt.CyBa->Finalize();

  kkt.ByT.reset(Transpose(By));
  kkt.ScaleByT.reset(Add(kkt.scale, *kkt.ByT, 0.0, *kkt.CyBa));
  kkt.ScaleBy.reset(Transpose(*kkt.ScaleByT));

  // Form BMat and BTMat (in VSize space)
  kkt.BMat.reset(Add(kkt.scale, *kkt.ByT, -kkt.scale, *kkt.CyBa));
  kkt.BTMat.reset(Transpose(*kkt.BMat));

  // Convert all operators to true-DOF space for hanging-node support
  kkt.AMat_t.reset(ToTrueDofs(*AMat, fespace));
  kkt.BMat_t.reset(ToTrueDofs(*kkt.BMat, fespace));
  kkt.BTMat_t.reset(ToTrueDofs(*kkt.BTMat, fespace));
  kkt.CMat_t.reset(ToTrueDofs(*kkt.CMat, fespace));

  // Convert vectors used in preconditioner to true-DOF space
  const int tdof = fespace.GetTrueVSize();
  kkt.Ba_t.SetSize(tdof);
  kkt.Cy_t.SetSize(tdof);
  ToTrueDofs(Ba, kkt.Ba_t, fespace);
  ToTrueDofs(Cy, kkt.Cy_t, fespace);

  // Block index ordering: PC_option == 0 is non-symmetric block matrix.
  if (PC_option == 0) {
    kkt.ind_x = 0;
    kkt.ind_p = 1;
  }
  else {
    kkt.ind_x = 1;
    kkt.ind_p = 0;
  }

  BlockSystem.SetBlock(0, kkt.ind_x, kkt.AMat_t.get());
  BlockSystem.SetBlock(0, kkt.ind_p, kkt.BMat_t.get());
  BlockSystem.SetBlock(1, kkt.ind_x, kkt.BTMat_t.get());
  BlockSystem.SetBlock(1, kkt.ind_p, kkt.CMat_t.get());
}


// Diagnostic dump of A/B/C blocks of the reduced KKT system in CSR text format
static void WriteSpyMatrices(int it_amr, int model_choice,
                             SparseMatrix *AMat, SparseMatrix *BMat, SparseMatrix *CMat)
{
  char filename_spy[60];
  sprintf(filename_spy, "spys/spy_model%d_amr%d.txt", model_choice, it_amr);
  std::unique_ptr<FILE, int(*)(FILE*)> spy_guard(fopen(filename_spy, "w"), &fclose);
  FILE *fp_spy = spy_guard.get();
  fprintf(fp_spy, "AMat\n");
  WriteSparseMatrixToFile(fp_spy, AMat);
  fprintf(fp_spy, "\nBMat\n");
  WriteSparseMatrixToFile(fp_spy, BMat);
  fprintf(fp_spy, "\nCMat\n");
  WriteSparseMatrixToFile(fp_spy, CMat);
}


// Bundle of objects keeping the active preconditioner and its dependencies
// alive across the FGMRES solve. `prec` is a non-owning pointer into
// either `bdiag` or `schur` depending on PC_option.
struct BlockPreconditionerBundle {
  std::unique_ptr<HypreParMatrix> B_Hypre, BT_Hypre;
  std::unique_ptr<SparseMatrix>   ScaleByT_t, ScaleBy_t;
  std::unique_ptr<HypreBoomerAMG> B_AMG, BT_AMG;
  std::unique_ptr<SchurPC>        schur;
  std::unique_ptr<BlockDiagonalPreconditioner> bdiag;
  Solver *prec = nullptr;
};


// Build the block preconditioner for the reduced KKT system:
//   PC_option == 0 -> block-diagonal       (eq 5.4)
//   PC_option == 5 -> block upper-triangular Schur PC (eq 5.5)
//   PC_option == 6 -> block lower-triangular Schur PC (eq 5.6)
static void BuildBlockPreconditioner(
  KKTBlocks &kkt, FiniteElementSpace &fespace, double Ca,
  int PC_option,
  const AMGParams &amg,
  const Array<int> &row_offsets,
  BlockPreconditionerBundle &bundle)
{
  // AMG inner solvers for B_y and B_y^T (true-DOF matrices)
  bundle.ScaleByT_t.reset(ToTrueDofs(*kkt.ScaleByT, fespace));
  bundle.ScaleBy_t.reset(ToTrueDofs(*kkt.ScaleBy, fespace));
  bundle.B_Hypre.reset(ConvertToHypre(bundle.ScaleByT_t.get()));
  bundle.BT_Hypre.reset(ConvertToHypre(bundle.ScaleBy_t.get()));
  bundle.B_AMG.reset(new HypreBoomerAMG(*bundle.B_Hypre));
  bundle.BT_AMG.reset(new HypreBoomerAMG(*bundle.BT_Hypre));

  bundle.B_AMG->SetPrintLevel(0);
  bundle.B_AMG->SetCycleType(amg.amg_cycle_type);
  bundle.B_AMG->SetCycleNumSweeps(amg.amg_num_sweeps_a, amg.amg_num_sweeps_b);
  bundle.B_AMG->SetMaxIter(amg.amg_max_iter);

  bundle.BT_AMG->SetPrintLevel(0);
  bundle.BT_AMG->SetCycleType(amg.amg_cycle_type);
  bundle.BT_AMG->SetCycleNumSweeps(amg.amg_num_sweeps_a, amg.amg_num_sweeps_b);
  bundle.BT_AMG->SetMaxIter(amg.amg_max_iter);

  Solver *inv_B  = bundle.B_AMG.get();
  Solver *inv_BT = bundle.BT_AMG.get();

  // Block diagonal preconditioner (eq 5.4)
  if (PC_option == 0) {
    bundle.bdiag.reset(new BlockDiagonalPreconditioner(row_offsets));
    bundle.bdiag->SetDiagonalBlock(0, inv_B);
    bundle.bdiag->SetDiagonalBlock(1, inv_BT);
    bundle.prec = bundle.bdiag.get();
  }

  // Block upper-triangular Schur preconditioner (eq 5.5)
  else if (PC_option == 5) {
    bundle.schur.reset(new SchurPC(kkt.AMat_t.get(), kkt.CMat_t.get(),
                                   inv_B, inv_BT,
                                   &kkt.Ba_t, &kkt.Cy_t, Ca, 1));
    bundle.prec = bundle.schur.get();
  }

  // Block lower-triangular Schur preconditioner (eq 5.6)
  else if (PC_option == 6) {
    bundle.schur.reset(new SchurPC(kkt.AMat_t.get(), kkt.CMat_t.get(),
                                   inv_B, inv_BT,
                                   &kkt.Ba_t, &kkt.Cy_t, Ca, 2));
    bundle.prec = bundle.schur.get();
  }

  // Unsupported preconditioner option
  else {
    fprintf(stderr, "ERROR: Unsupported PC_option=%d (currently supported: 0, 5, 6)\n", PC_option);
    MFEM_ABORT("Unsupported PC_option");
  }
}


// Compute the Newton-system RHS pieces (equation 4.11 in the paper):
//   -b1 = Gy + By^T p + Cy lambda
//   -b2 = H u^n - F^T p^n
//   -b3 = B(y^n) - F u^n
//   -b4 = B_a^T p^n + C_a l^n
//   -b5 = C - Ip * mu
static void ComputeNewtonRHS(
  SysOperator &op,
  const Vector &g, const SparseMatrix &By,
  const GridFunction &pv, double lv,
  const Vector &Cy, const Vector &Ba, double Ca, double C,
  SparseMatrix &H, const Vector &uv, const SparseMatrix &F,
  double Ip,
  GridFunction &opt_res, Vector &reg_res, GridFunction &eq_res,
  GridFunction &b1, Vector &b2, GridFunction &b3,
  double &b4, double &b5)
{
  // -b1 = Gy + By^T p + Cy lambda
  opt_res = g;
  By.AddMultTranspose(pv, opt_res);
  add(opt_res, lv, Cy, opt_res);
  b1 = opt_res;
  b1 *= -1.0;

  // -b2 = reg_res = H u^n - F^T p^n
  H.Mult(uv, reg_res);
  F.AddMultTranspose(pv, reg_res, -1.0);
  b2 = reg_res;
  b2 *= -1.0;

  // -b3 = eq_res = B(y^n) - F u^n
  eq_res = op.get_res();
  b3 = eq_res;
  b3 *= -1.0;

  // -b4 = B_a^T p^n + C_a l^n
  b4 = Ba * pv + Ca * lv;
  b4 *= -1.0;

  // -b5 = C - Ip * mu
  b5 = C - Ip * op.get_mu();  // TODO: mu is not present in the calculation of b5 in equation 4.11 of the paper--check if this is a bug here.
  b5 *= -1.0;
}


// Per-Newton-iteration diagnostic writes: snapshot ψ, recompute and project the
// magnetic-field components onto Br/Bp/Bz, and emit a VisIt frame.
static void WritePerIterationDiagnostics(
  int it_amr, int newton_it,
  GridFunction &x, GridFunction &psi_r, GridFunction &psi_z,
  GridFunction &Br_field, GridFunction &Bp_field, GridFunction &Bz_field,
  FieldCoefficient &BrCoeff, FieldCoefficient &BpCoeff, FieldCoefficient &BzCoeff,
  VisItDataCollection &visit_dc)
{
  char name_[60];
  sprintf(name_, "gf/xtmp_amr%d.gf", it_amr);
  x.Save(name_);
  char name[60];
  sprintf(name, "gf/xtmp_amr%d_i%d.gf", it_amr, newton_it);
  x.Save(name);

  Br_field.Save("gf/Br.gf");
  Bp_field.Save("gf/Bp.gf");
  Bz_field.Save("gf/Bz.gf");

  x.GetDerivative(1, 0, psi_r);
  x.GetDerivative(1, 1, psi_z);

  Br_field.ProjectCoefficient(BrCoeff);
  Bp_field.ProjectCoefficient(BpCoeff);
  Bz_field.ProjectCoefficient(BzCoeff);

  visit_dc.Save();
}


// Inexact-Newton tolerance update: shrink the linear-solve relative tolerance
// `eta` as the Newton iteration converges, with a safeguard preventing it
// from collapsing too quickly. No-op for the very first Newton iteration.
static void ApplyInexactNewtonTolerance(
  int newton_it,
  double error,
  double error_old,
  double alpha_in,
  double gamma_in,
  double sg_threshold,
  double lin_rtol_max,
  double &eta,
  double &eta_last)
{
  if (newton_it > 0) {
    eta = gamma_in * pow(error / error_old, alpha_in);  // alpha_in and gamma_in: inexact Newton parameters

    // safeguard to prevent eta from becoming too small too quickly
    double sg_eta = gamma_in * pow(eta_last, alpha_in);

    // if eta is above a predefined threshold, truncate eta
    if (sg_eta > sg_threshold) {
      eta = max(eta, sg_eta);
    }

    eta = min(eta, lin_rtol_max);
    eta_last = eta;
  }
}


// Compute and print the two block-row residuals of the post-solve Newton
// system (RHS of eq. 4.13 in the paper). Used as a sanity check that the
// FGMRES solve actually drove the block residuals near zero.
static void CheckPostSolveResiduals(
  FiniteElementSpace &fespace,
  KKTBlocks &kkt,
  SparseMatrix &By,
  SparseMatrix *AMat,
  const GridFunction &b1,
  const Vector &b2,
  const GridFunction &b3,
  const Vector &Cy,
  const Vector &Ba,
  const Vector &dx_x_full,
  const Vector &dx_p_full,
  double dlv,
  double dalpha)
{

  // Residuals are assembled in VSize and then projected to true-DOF via P^T before norming
  const int vsize  = fespace.GetVSize();
  const int tvsize = fespace.GetTrueVSize();

  // First block row in RHS
  Vector res1(vsize);
  res1 = 0.0;
  AMat->AddMult(dx_x_full, res1);
  kkt.ByT->AddMult(dx_p_full, res1);
  add(res1, dlv, Cy, res1);
  add(res1, -1.0, b1, res1);
  Vector res1_true(tvsize);
  ToTrueDofs(res1, res1_true, fespace);
  printf("res_1: %.2e\n", GetMaxError(res1_true));

  // Second block row in RHS
  Vector res2(vsize);
  res2 = 0.0;
  kkt.mMuFinvHFT->AddMult(dx_p_full, res2);
  kkt.MuFinvH->Mult(b2, res2);
  res2 *= -1.0;
  By.AddMult(dx_x_full, res2);
  kkt.mMuFinvHFT->AddMult(dx_p_full, res2);
  add(res2, dalpha, Ba, res2);
  add(res2, -1.0, b3, res2);
  Vector res2_true(tvsize);
  ToTrueDofs(res2, res2_true, fespace);
  printf("res_2: %.2e\n", GetMaxError(res2_true));
}


// Solve the free-boundary GS problem (do_control == 1)
static void SolveControlProblem(
  FiniteElementSpace &fespace,
  PlasmaModelBase *model,
  GridFunction &x,
  Mesh *mesh,
  ExactCoefficient *exact_coefficient,
  ExactForcingCoefficient *exact_forcing_coeff,
  InitialCoefficient *init_coeff,
  Vector *uv,
  double &alpha,
  bool include_plasma,
  double rho_gamma, double Ip,
  int PC_option,
  const SolverParams &sp,
  const AMGParams &amg,
  const AMROptions &amr,
  const ObjectiveParams &obj_p,
  const InexactNewtonParams &inp
) {
    const int    kdim             = sp.kdim;
    const int    max_newton_iter  = sp.max_newton_iter;
    const int    max_krylov_iter  = sp.max_krylov_iter;
    const double newton_tol       = sp.newton_tol;
    const double krylov_tol       = sp.krylov_tol;
    const int    amg_cycle_type   = amg.amg_cycle_type;
    const int    amg_max_iter     = amg.amg_max_iter;
    const int    max_amr_levels   = amr.max_amr_levels;
    const int    max_dofs         = amr.max_dofs;
    const double amr_frac_in      = amr.amr_frac_in;
    const double amr_frac_out     = amr.amr_frac_out;
    const int    N_control        = obj_p.N_control;
    const int    obj_option       = obj_p.obj_option;
    const double obj_weight       = obj_p.obj_weight;
    const double weight_coils     = obj_p.weight_coils;
    const double weight_solenoids = obj_p.weight_solenoids;
    const double alpha_in         = inp.alpha_in;
    const double gamma_in         = inp.gamma_in;

    // Containers for magnetic flux psi, magnetic field B, and toroidal magnetic field function f
    GridFunction psi_r(&fespace);
    GridFunction psi_z(&fespace);
    FieldCoefficient BrCoeff(&x, &psi_r, &psi_z, model, fespace, 0);
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

    double psi_x;  // Psi at X-point
    double f_x;    // Constant set by the vacuum toroidal field

    auto log_guard = OpenIterationLog(
      model->get_model_choice(),
      PC_option,
      amg_cycle_type,
      amg_max_iter
    );
    
    FILE *fp = log_guard.get();

    // Print initial currents
    printf("Initial currents: [");
    for (int i = 0; i < uv->Size(); ++i) { printf("%.3e ", (*uv)[i]); };  // uv: list of external coil currents
    printf("]\n\n");

    // Initialize Lagrange multipliers pv and lv (corresponding to p and \lambda in the paper, respectively)
    GridFunction pv(&fespace);
    pv = 0.0;
    double lv = 0.0;

    // Initialize residuals and regularization
    // eq_res: residual of the GS equation \Delta^* \psi - J_{\phi}(\psi)
    // reg_res: regularization term R(u) = \frac{1}{2} u^T H u, where u is a vector of external coil currents I_j, j = 1, ... , N
    // opt_res: optimization residual \psi - \psi_{target}
    GridFunction eq_res(&fespace);
    Vector reg_res(uv->Size());
    GridFunction opt_res(&fespace);

    // Initialize b1, b2, b3: elements of RHS vector. The other elements b4 and b5 are initialized later in the Newton iteration.
    GridFunction b1(&fespace);
    Vector b2(uv->Size());
    GridFunction b3(&fespace);
    b1 = 0.0;
    b2 = 0.0;
    b3 = 0.0;

    // Define Zienkiewicz-Zhu error estimator for AMR
    DiffusionIntegratorCoefficient diff_op_coeff(model);  // DiffusionIntegratorCoefficient is defined in diffusion_term.cpp
    std::unique_ptr<DiffusionIntegrator> integ(new DiffusionIntegrator(diff_op_coeff));
    std::unique_ptr<ErrorEstimator> estimator(new LSZienkiewiczZhuEstimator(*integ, x));
    RegionalThresholdRefiner refiner(*estimator);

    // Initialize time tracker
    auto t_init = std::chrono::high_resolution_clock::now();

    // Histories of psi at the magnetic axis, psi at the X-point, and total plasma current per Newton iteration
    vector<double> psi_ma_vals, psi_x_vals, cpasma_vals;

    // ============================================================================
    // AMR loop
    // ============================================================================

    for (int it_amr = 0; it_amr <= max_amr_levels; ++it_amr) {
      int cdofs = fespace.GetTrueVSize();

      // Save per-iteration mesh snapshot
      char name_mesh[60];
      sprintf(name_mesh, "gf/mesh_amr%d_model%d_pc%d_cyc%d_it%d.mesh", it_amr,
              model->get_model_choice(), PC_option, amg_cycle_type, amg_max_iter);
      mesh->Save(name_mesh);

      // ============================================================================
      // Define and assemble PDE operator components
      // ============================================================================

      LinearForm coil_term(&fespace);
      BilinearForm diff_operator(&fespace);
      std::unique_ptr<SparseMatrix> F;
      AssemblePDEOperators(
        *model,
        rho_gamma,
        *mesh,
        fespace,
        *exact_coefficient,
        *exact_forcing_coeff,
        coil_term,
        diff_operator,
        F
      );

      // ============================================================================
      // Define objective function components
      // ============================================================================

      ObjectiveTerms obj;
      AssembleObjectiveTerms(
        *init_coeff,
        N_control,
        *mesh,
        fespace,
        weight_coils,
        weight_solenoids,
        obj
      );

      // ============================================================================
      // SysOperator and KKT system
      // ============================================================================

      SysOperator op(&diff_operator, &coil_term, model, &fespace, mesh, attr_lim,
                    &x, F.get(), uv, obj.H.get(), obj.K, &obj.g, obj.alpha_coeffs, 
                    obj.J_inds, &alpha, include_plasma);

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

      // Inexact newton settings
      double eta_last = 0.0;
      double sg_threshold = 0.1;
      double lin_rtol_max = krylov_tol;
      double eta = krylov_tol;
      double error_old;
      double error;

      for (int i = 0; i <= max_newton_iter; ++i) {

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

        // Print and log plasma current
        printf("plasma_current = %10.8e\n", C / op.get_mu());
        printf("alpha = %10.8e\n", alpha);
        fprintf(fp, "plasma_current = %10.8e\n", C / op.get_mu());
        fprintf(fp, "alpha = %10.8e\n", alpha);

        // Locate X-point (x_x) and magnetic axis (x_ma) and get corresponding psi values
        double* x_x = op.get_x_x();
        double* x_ma = op.get_x_ma();
        psi_x = op.get_psi_x();
        double psi_ma = op.get_psi_ma();

        // Project X- and O-points to magnetic field
        BrCoeff.set_psi_vals(psi_x, psi_ma);
        BpCoeff.set_psi_vals(psi_x, psi_ma);
        BzCoeff.set_psi_vals(psi_x, psi_ma);

        psi_ma_vals.push_back(psi_ma);
        psi_x_vals.push_back(psi_x);
        cpasma_vals.push_back(C / op.get_mu());

        // Print and log X-point and magnetic axis
        printf("psi_x = %10.8e; r_x = %10.8e; z_x = %10.8e\n", psi_x, x_x[0], x_x[1]);
        printf("psi_ma = %10.8e; r_ma = %10.8e; z_ma = %10.8e\n", psi_ma, x_ma[0], x_ma[1]);
        fprintf(fp, "psi_x = %10.8e; r_x = %10.8e; z_x = %10.8e\n", psi_x, x_x[0], x_x[1]);
        fprintf(fp, "psi_ma = %10.8e; r_ma = %10.8e; z_ma = %10.8e\n", psi_ma, x_ma[0], x_ma[1]);

        // Compute the Newton-system RHS pieces (equation 4.11 in paper)
        double b4, b5;
        ComputeNewtonRHS(op, g, By, pv, lv, Cy, Ba, Ca, C, *obj.H, *uv, *F,
                         Ip, opt_res, reg_res, eq_res, b1, b2, b3, b4, b5);

        // Save equilibrium residual (b3)
        char name_eq_res[60];
        sprintf(name_eq_res, "gf/eq_res_amr%d_i%d.gf", it_amr, i);
        eq_res.Save(name_eq_res);

        // Get max errors for residuals
        Vector eq_res_true(fespace.GetTrueVSize());
        ToTrueDofs(eq_res, eq_res_true, fespace);
        error = GetMaxError(eq_res_true);
        printf("newton error (eq_res tdof): %.3e  (newton_tol=%.1e)\n", error, newton_tol);

        // double max_opt_res = op.get_mu() * GetMaxError(opt_res);
        // double max_reg_res = GetMaxError(reg_res) / op.get_mu();

        // Inexact Newton. Adjust the relative tolerance eta as the Newton iteration converges
        ApplyInexactNewtonTolerance(i, error, error_old, alpha_in, gamma_in,
                                    sg_threshold, lin_rtol_max, eta, eta_last);

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

        // Build the reduced 2x2 KKT block system (LHS of equation 4.13 of paper)
        BlockOperator BlockSystem(row_offsets);
        BlockVector rhs(row_offsets);
        KKTBlocks kkt;
        BuildReducedKKTBlocks(op, fespace, AMat, By, *F, *obj.H,
                              Ba, Cy, Ca, *uv, PC_option, BlockSystem, kkt);

        const int vsize = fespace.GetVSize();

        // Logging of reduced KKT system in CSR text format
        WriteSpyMatrices(
          it_amr,
          model->get_model_choice(),
          kkt.AMat_t.get(),
          kkt.BMat_t.get(),
          kkt.CMat_t.get()
        );

        // Define RHS of the block linear system (RHS of equation 4.13 of paper)
        // Compute in VSize first, then convert to true-DOF
        // c1 = b1 - Cy b4 / Ca
        // c2 = b3 + mu F H^{-1} b_2 - Ba b5 / Ca
        Vector rhs0_full(vsize), rhs1_full(vsize);
        add(1.0, b1, -b4 / Ca, Cy, rhs0_full);  // c1 (VSize)
        kkt.MuFinvH->Mult(b2, rhs1_full);  // c2 (VSize)
        rhs1_full += b3;
        add(1.0, rhs1_full, - b5 / Ca, Ba, rhs1_full);
        rhs1_full *= kkt.scale;

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

        // Build preconditioner and run the FGMRES solve.
        BlockPreconditionerBundle prec;
        BuildBlockPreconditioner(kkt, fespace, Ca, PC_option,
                                 amg, row_offsets, prec);
        solver.SetPreconditioner(*prec.prec);
        solver.Mult(rhs, dx);

        fprintf(fp, "amr=%d newton=%d iters=%d\n", it_amr, i, solver.GetNumIterations());

        // Check for convergence
        if (solver.GetConverged()) {
          printf("GMRES converged in %d iterations with a residual norm of %e\n", solver.GetNumIterations(), solver.GetFinalNorm());
        }
        else {
          printf("GMRES did not converge in %d iterations. Residual norm is %e\n", solver.GetNumIterations(), solver.GetFinalNorm());
        }

        // Failed to converge
        if (solver.GetNumIterations() == -1) {
          printf("failure...\n");
          return;
        }
        dx.GetBlock(kkt.ind_p) *= kkt.scale;

        // Prolong solution increments from true-DOF space back to VSize
        Vector dx_x_full(vsize), dx_p_full(vsize);
        ToFullDofs(dx.GetBlock(kkt.ind_x), dx_x_full, fespace);
        ToFullDofs(dx.GetBlock(kkt.ind_p), dx_p_full, fespace);

        // Update solution in VSize space
        x += dx_x_full;
        pv += dx_p_full;

        // Update coil currents (invHFT and invH operate on coil-space vectors, not DOF vectors)
        kkt.invHFT->AddMult(dx_p_full, *uv);
        kkt.invH->AddMult(b2, *uv);

        // Compute alpha and lambda increments using VSize vectors
        dalpha = (b5 - (Cy * dx_x_full)) / Ca;
        dlv = (b4 - (Ba * dx_p_full)) / Ca;
        alpha += dalpha;
        lv += dlv;

        // Sanity-check the post-solve block residuals (RHS of eq. 4.13 in paper)
        CheckPostSolveResiduals(fespace, kkt, By, AMat, b1, b2, b3, Cy, Ba,
                                dx_x_full, dx_p_full, dlv, dalpha);

        // ============================================================================
        // Save/print/update parameters post-solve
        // ============================================================================

        // Print currents
        printf("Currents: [");
        for (int i = 0; i < uv->Size(); ++i) {printf("%.3e ", (*uv)[i]);}
        printf("]\n");

        WritePerIterationDiagnostics(it_amr, i, x, psi_r, psi_z,
                                     Br_field, Bp_field, Bz_field,
                                     BrCoeff, BpCoeff, BzCoeff, visit_dc);
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

    // Write final values for GEQDSK
    system("mkdir -p ../gslib/GEQDSK");
    ofstream simagx_file("../gslib/GEQDSK/GEQDSK_simagx_sibdry_cpasma.txt");
    simagx_file << scientific << setprecision(9)
      << setw(16) << psi_ma_vals.back() << "\n"
      << setw(16) << psi_x_vals.back() << "\n"
      << setw(16) << cpasma_vals.back() << "\n";
    simagx_file.close();

    ofstream alpha_file("../gslib/GEQDSK/GEQDSK_alpha_f_x_psi_x.txt");
    alpha_file << alpha << "\n" << f_x << "\n" << psi_x << "\n";
    alpha_file.close();
}


// Given currents, solve the GS equation (do_control == 0)
static void SolveFixedBoundaryProblem(
  FiniteElementSpace &fespace,
  PlasmaModelBase *model,
  GridFunction &x,
  Mesh *mesh,
  ExactCoefficient *exact_coefficient,
  ExactForcingCoefficient *exact_forcing_coeff,
  InitialCoefficient *init_coeff,
  Vector *uv,
  double &alpha,
  bool include_plasma,
  double rho_gamma,
  int PC_option,
  const SolverParams &sp,
  const AMGParams &amg,
  const ObjectiveParams &obj_p
) {
    const int    kdim             = sp.kdim;
    const int    max_newton_iter  = sp.max_newton_iter;
    const int    max_krylov_iter  = sp.max_krylov_iter;
    const double newton_tol       = sp.newton_tol;
    const double krylov_tol       = sp.krylov_tol;
    const int    amg_cycle_type   = amg.amg_cycle_type;
    const int    amg_max_iter     = amg.amg_max_iter;
    const int    N_control        = obj_p.N_control;
    const int    obj_option       = obj_p.obj_option;
    const double obj_weight       = obj_p.obj_weight;
    const double weight_coils     = obj_p.weight_coils;
    const double weight_solenoids = obj_p.weight_solenoids;

    // Magnetic-field GridFunctions registered with visit_dc to match the do_control==1
    // VisIt output schema. These are never updated in the fixed-boundary path.
    GridFunction Br_field(&fespace);
    GridFunction Bp_field(&fespace);
    GridFunction Bz_field(&fespace);

    char outname[60];
    sprintf(outname, "out/gs_model%d_pc%d_cyc%d_it%d", model->get_model_choice(), PC_option, amg_cycle_type, amg_max_iter);
    VisItDataCollection visit_dc(outname, fespace.GetMesh());
    visit_dc.RegisterField("psi", &x);
    visit_dc.RegisterField("Br", &Br_field);
    visit_dc.RegisterField("Bp", &Bp_field);
    visit_dc.RegisterField("Bz", &Bz_field);

    GridFunction dx(&fespace);
    dx = 0.0;

    // ============================================================================
    // Define and assemble PDE operator components
    // ============================================================================

    LinearForm coil_term(&fespace);
    BilinearForm diff_operator(&fespace);
    std::unique_ptr<SparseMatrix> F;
    AssemblePDEOperators(
      *model,
      rho_gamma,
      *mesh,
      fespace,
      *exact_coefficient,
      *exact_forcing_coeff,
      coil_term,
      diff_operator,
      F
    );

    // ============================================================================
    // Define objective function components
    // ============================================================================

    ObjectiveTerms obj;
    AssembleObjectiveTerms(
      *init_coeff,
      N_control,
      *mesh,
      fespace,
      weight_coils,
      weight_solenoids,
      obj
    );

    // ============================================================================
    // SysOperator and KKT system
    // ============================================================================

    SysOperator op(&diff_operator, &coil_term, model, &fespace, mesh, attr_lim,
                   &x, F.get(), uv, obj.H.get(), obj.K, &obj.g, obj.alpha_coeffs, 
                   obj.J_inds, &alpha, include_plasma);

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

      eq_res = op.get_res();  // eq_res = B(y^n) - F u^n
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
      std::unique_ptr<HypreParMatrix> B_Hypre(ConvertToHypre(&By));
      std::unique_ptr<HypreBoomerAMG> B_AMG(new HypreBoomerAMG(*B_Hypre));
      B_AMG->SetPrintLevel(0);
      B_AMG->SetCycleType(1);
      B_AMG->SetCycleNumSweeps(1, 1);
      B_AMG->SetMaxIter(1);
      Solver *inv_B = B_AMG.get();

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

      // Save VisIt frame
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


// Public entry point for the Grad-Shafranov solver.
//
// Sets up the discretization (mesh, FE space, plasma model, initial guess),
// applies any uniform refinement, then dispatches to either
//   SolveControlProblem        (free-boundary, do_control == 1)
// or
//   SolveFixedBoundaryProblem  (fixed-boundary, do_control == 0).
//
// Saves the final solution GridFunction (and, for the free-boundary path,
// a ParaView dataset). Returns the manufactured-solution L2 error when
// do_manufactured_solution == 1, else 0.0.
double gs(GSProblemConfig cfg) {

  const char *mesh_file   = cfg.mesh_file;
  const char *initial_gf  = cfg.initial_gf;
  const char *data_file   = cfg.data_file;
  const int    order            = cfg.order;
  const int    d_refine         = cfg.d_refine;
  const int    model_choice     = cfg.model_choice;
  double      &alpha            = cfg.alpha;
  double       beta             = cfg.beta;
  double       gamma            = cfg.gamma;
  double      &mu               = cfg.mu;
  const double Ip               = cfg.Ip;
  const double rho_gamma        = cfg.rho_gamma;
  const bool   do_manufactured_solution = cfg.do_manufactured_solution;
  const bool   do_initial               = cfg.do_initial;
  const int    PC_option                = cfg.PC_option;
  int          do_control               = cfg.do_control;

  // ============================================================================
  // Preprocessing
  // ============================================================================

  // Pack the external coil currents (c1..c11) passed individually as CLI
  // arguments into the single Vector that downstream code expects.
  Vector uv_currents(num_currents);
  uv_currents[0]  = cfg.c1;
  uv_currents[1]  = cfg.c2;
  uv_currents[2]  = cfg.c3;
  uv_currents[3]  = cfg.c4;
  uv_currents[4]  = cfg.c5;
  uv_currents[5]  = cfg.c6;
  uv_currents[6]  = cfg.c7;
  uv_currents[7]  = cfg.c8;
  uv_currents[8]  = cfg.c9;
  uv_currents[9]  = cfg.c10;
  uv_currents[10] = cfg.c11;

  // Define mesh, finite element space (H1), and solution vector u
  Mesh mesh(mesh_file);
  H1_FECollection fec(order, mesh.Dimension());
  FiniteElementSpace fespace(&mesh, &fec);
  GridFunction u(&fespace);

  cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

  // Save options in model: alpha: multiplier in \bar{S}_{ff'} term, beta: multiplier for S_{p'} term, gamma: multiplier for S_{ff'} term
  // TODO: what is the difference between this data_file_ versus data_file, which is an argument passed into main.cpp?
  const char *data_file_ = "data/fpol_pres_ffprim_pprime.data";
  PlasmaModelFile model(mu, data_file_, alpha, beta, gamma, model_choice);

  // Exact solution (for do_manufactured_solution == 1)
  double r0_ = 1.0;
  double z0_ = 0.0;
  double L_ = 0.35;
  double k_ = M_PI/(2.0*L_);
  ExactForcingCoefficient exact_forcing_coeff(r0_, z0_, k_, model, do_manufactured_solution);
  ExactCoefficient exact_coefficient(r0_, z0_, k_, do_manufactured_solution);

  // Remove control point optimization to solve fixed-boundary GS in order to get initial guesses for the free-boundary GS cases.
  if (do_initial) {
    do_control = false;
  }
  
  // Plasma profile data used to build the target shape and objective terms.
  InitialCoefficient init_coeff = read_data_file(data_file);

  // Project exact solution onto u and save (used for analytical-solution tests)
  if (do_manufactured_solution) {
    u.ProjectCoefficient(exact_coefficient);
    u.Save("gf/exact.gf");
  }
   
  else {
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

  // Initialize MPI and Hypre so we can use AMG
  Mpi::Init();
  Hypre::Init();

  // ============================================================================
  // Solve
  // ============================================================================

  cout << "Beginning GS Solve." << endl;

  // Free-boundary solve
  if (do_control) {
    SolveControlProblem(fespace, &model, x, &mesh, &exact_coefficient, &exact_forcing_coeff,
                        &init_coeff, &uv_currents, alpha, include_plasma,
                        rho_gamma, Ip, PC_option,
                        cfg.solver, cfg.amg, cfg.amr, cfg.objective, cfg.inexact_newton);
  }

  // Fixed-boundary solve
  else {
    SolveFixedBoundaryProblem(fespace, &model, x, &mesh, &exact_coefficient, &exact_forcing_coeff,
                              &init_coeff, &uv_currents, alpha, include_plasma,
                              rho_gamma, PC_option,
                              cfg.solver, cfg.amg, cfg.objective);
  }

  // ============================================================================
  // Postprocessing
  // ============================================================================

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
    sprintf(
      name_gf_out,
      "gf/final_model%d_pc%d_cyc%d_it%d.gf",
      model.get_model_choice(),
      PC_option,
      cfg.amg.amg_cycle_type,
      cfg.amg.amg_max_iter
    );
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
