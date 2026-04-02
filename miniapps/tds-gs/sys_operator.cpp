#include "mfem.hpp"
#include "sys_operator.hpp"
using namespace mfem;
using namespace std;


double GetMaxError(LinearForm &res) {
  double *resv = res.GetData();
  int size = res.FESpace()->GetTrueVSize();

  double max_val = - numeric_limits<double>::infinity();
  for (int i = 0; i < size; ++i) {
    if (abs(resv[i]) > max_val) {
      max_val = abs(resv[i]);
    }
  }
  return max_val;
}

double GetMaxError(GridFunction &res) {
  double *resv = res.GetData();
  int size = res.FESpace()->GetTrueVSize();

  double max_val = - numeric_limits<double>::infinity();
  for (int i = 0; i < size; ++i) {
    if (abs(resv[i]) > max_val) {
      max_val = abs(resv[i]);
    }
  }
  return max_val;
}

double GetMaxError(Vector &res) {
  int size = res.Size();

  double max_val = - numeric_limits<double>::infinity();
  for (int i = 0; i < size; ++i) {
    if (abs(res[i]) > max_val) {
      max_val = abs(res[i]);
    }
  }
  return max_val;
}

void Print(const Vector &y) {
  for (int i = 0; i < y.Size(); ++i) {
    printf("%d %.14e\n", i+1, y[i]);
  }
}

void Print(const LinearForm &y) {
  double *yv = y.GetData();
  int size = y.FESpace()->GetTrueVSize();
  for (int i = 0; i < size; ++i) {
    printf("%d %.14e\n", i+1, yv[i]);
  }
}



void SysOperator::compute_psi_N_derivs(const GridFunction &psi, int k, double * psi_N,
                                       GridFunction * grad_psi_N,
                                       SparseMatrix * hess_psi_N) {

  int dof = (*alpha_coeffs)[0].Size();
  double psi_interp;
  double psi_x = psi[ind_x];
  double psi_ma = psi[ind_ma];
  
  psi_interp = 0;
  for (int m = 0; m < dof; ++m) {
    int i = (*J_inds)[k][m];
    psi_interp += (*alpha_coeffs)[k][m] * psi[i];
  }

  (*psi_N) = (psi_interp - psi_ma) / (psi_x - psi_ma);

  for (int m = 0; m < dof; ++m) {
    int i = (*J_inds)[k][m];
    (*grad_psi_N)[i] += (*alpha_coeffs)[k][m] / (psi_x - psi_ma);
  }
  (*grad_psi_N)[ind_ma] += ((*psi_N) - 1.0) / (psi_x - psi_ma);
  (*grad_psi_N)[ind_x] += - (*psi_N) / (psi_x - psi_ma);

  int i, j;
  for (int m = 0; m < dof+2; ++m) {
    if (m < dof) {
      i = (*J_inds)[k][m];
    } else if (m == dof) {
      i = ind_ma;
    } else {
      i = ind_x;
    }

    for (int n = 0; n < dof+2; ++n) {
      if (n < dof) {
        j = (*J_inds)[k][n];
      } else if (n == dof) {
        j = ind_ma;
      } else {
        j = ind_x;
      }

      hess_psi_N->Add(i, j, 0.0);
      if (j == ind_x) {
        hess_psi_N->Add(i, j, - (*grad_psi_N)[i] / (psi_x - psi_ma));
      }
      if (j == ind_ma) {
        hess_psi_N->Add(i, j, (*grad_psi_N)[i] / (psi_x - psi_ma));
      }
      if (i == ind_x) {
        hess_psi_N->Add(i, j, - (*grad_psi_N)[j] / (psi_x - psi_ma));
      }
      if (i == ind_ma) {
        hess_psi_N->Add(i, j, (*grad_psi_N)[j] / (psi_x - psi_ma));
      }
    }
  }
  
  hess_psi_N->Finalize(0);
}


double SysOperator::compute_obj(const GridFunction &psi) {

  int dof = (*alpha_coeffs)[0].Size();
  double psi_interp, psi_interp_0, psi_N;
  double obj = 0;
  double psi_x = psi[ind_x];
  double psi_ma = psi[ind_ma];
  double weight = obj_weight;

  if (i_option == 0) {
    // obj = sum_{n=1}^{N} (psi - psi_0) ^ 2

    int k = 0;
    psi_interp_0 = 0;
    for (int m = 0; m < dof; ++m) {
      int i = (*J_inds)[k][m];
      psi_interp_0 += (*alpha_coeffs)[k][m] * psi[i];
    }
      
    for (int k = 1; k < N_control; ++k) {
      psi_interp = 0;
      for (int m = 0; m < dof; ++m) {
        int i = (*J_inds)[k][m];
        psi_interp += (*alpha_coeffs)[k][m] * psi[i];
      }

      obj += 0.5 * (psi_interp - psi_interp_0) * (psi_interp - psi_interp_0);

    }

  } else if (i_option == 1) {
    // obj = sum_{n=1}^{N} (psi_N - 1) ^ 2

    for (int k = 0; k < N_control; ++k) {
      psi_interp = 0;
      for (int m = 0; m < dof; ++m) {
        int i = (*J_inds)[k][m];
        psi_interp += (*alpha_coeffs)[k][m] * psi[i];
      }

      psi_N = (psi_interp - psi_ma) / (psi_x - psi_ma);
      obj += 0.5 * (psi_N - 1.0) * (psi_N - 1.0);
    }
  } else {
    // obj = sum_{n=1}^{N} (psi - psi_x) ^ 2

    for (int k = 0; k < N_control; ++k) {
      psi_interp = 0;
      for (int m = 0; m < dof; ++m) {
        int i = (*J_inds)[k][m];
        psi_interp += (*alpha_coeffs)[k][m] * psi[i];
      }

      obj += 0.5 * (psi_interp - psi_x) * (psi_interp - psi_x);
    }
  }

  obj *= weight;
  return obj;
}


GridFunction SysOperator::compute_grad_obj(const GridFunction &psi) {
  
  double psi_x = psi[ind_x];
  FiniteElementSpace fespace = *(psi.FESpace());
  // int ndof = psi.Size();
  int ndof = fespace.GetNDofs();
  double weight = obj_weight;
  
  GridFunction g_(&fespace);
  g_ = 0.0;
  if (i_option == 0) {
    K->Mult(psi, g_);

  } else if (i_option == 1) {
    int dof = (*alpha_coeffs)[0].Size();
    double psi_N;

    GridFunction * grad_psi_N;
    grad_psi_N = new GridFunction(&fespace);
    SparseMatrix * hess_psi_N;
    for (int k = 0; k < N_control; ++k) {

      hess_psi_N = new SparseMatrix(ndof, ndof);

      *grad_psi_N = 0.0;
      compute_psi_N_derivs(psi, k, &psi_N, grad_psi_N, hess_psi_N);

      for (int m = 0; m < dof; ++m) {
        int i = (*J_inds)[k][m];
        if ((i == ind_ma) || (i == ind_x)) {
          continue;
        }
        g_[i] += (psi_N - 1.0) * (*grad_psi_N)[i];
      }

      g_[ind_ma] += (psi_N - 1.0) * (*grad_psi_N)[ind_ma];
      g_[ind_x] += (psi_N - 1.0) * (*grad_psi_N)[ind_x];
    }
    delete hess_psi_N;


    return g_;
  } else {
    int dof = (*alpha_coeffs)[0].Size();
    double psi_interp;
    for (int k = 0; k < N_control; ++k) {
      psi_interp = 0;
      for (int m = 0; m < dof; ++m) {
        int i = (*J_inds)[k][m];
        psi_interp += (*alpha_coeffs)[k][m] * psi[i];
      }

      for (int m = 0; m < dof; ++m) {
        int i = (*J_inds)[k][m];
        g_[i] += (psi_interp - psi_x) * (*alpha_coeffs)[k][m];
      }
      g_[ind_x] += - (psi_interp - psi_x);
    }

  }

  g_ *= weight;
  return g_;
  
}



SparseMatrix* SysOperator::compute_hess_obj(const GridFunction &psi) {

  // int ndof = psi.Size();
  FiniteElementSpace fespace = *(psi.FESpace());
  int ndof = fespace.GetNDofs();
  double weight = obj_weight;

  if (i_option == 0) {
    *K *= weight;
    return K;
  } else if (i_option == 1) {
  
    SparseMatrix * K_;
    K_ = new SparseMatrix(ndof, ndof);

    int dof = (*alpha_coeffs)[0].Size();
    double psi_N;

    GridFunction * grad_psi_N;
    grad_psi_N = new GridFunction(&fespace);
      
    SparseMatrix * hess_psi_N;

    int i, j;
    for (int k = 0; k < N_control; ++k) {

      hess_psi_N = new SparseMatrix(ndof, ndof);
      
      *grad_psi_N = 0.0;
      compute_psi_N_derivs(psi, k, &psi_N, grad_psi_N, hess_psi_N);

      for (int m = 0; m < dof+2; ++m) {
        if (m < dof) {
          i = (*J_inds)[k][m];
          if ((i == ind_ma) || (i == ind_x)) {
            continue;
          }
        } else if (m == dof) {
          i = ind_ma;
        } else {
          i = ind_x;
        }

        for (int n = 0; n < dof+2; ++n) {
          if (n < dof) {
            j = (*J_inds)[k][n];
            if ((j == ind_ma) || (j == ind_x)) {
              continue;
            }
          } else if (n == dof) {
            j = ind_ma;
          } else {
            j = ind_x;
          }
          K_->Add(i, j, (*grad_psi_N)[i] * (*grad_psi_N)[j] + (*hess_psi_N)(i, j) * (psi_N - 1));
        }
      }
      delete hess_psi_N;

    }
    K_->Finalize();
    *K_ *= weight;
    return K_;
  } else {
    SparseMatrix * K_;
    K_ = new SparseMatrix(ndof, ndof);
    int dof = (*alpha_coeffs)[0].Size();
    int i, j;
    double a, b;
    for (int k = 0; k < N_control; ++k) {

      for (int m = 0; m < dof+1; ++m) {
        if (m < dof) {
          i = (*J_inds)[k][m];
          a = (*alpha_coeffs)[k][m];
        } else {
          i = ind_x;
          a = -1.0;
        }

        for (int n = 0; n < dof+1; ++n) {
          if (n < dof) {
            j = (*J_inds)[k][n];
            b = (*alpha_coeffs)[k][n];
          } else {
            j = ind_x;
            b = -1.0;
          }
          
          K_->Add(i, j, a * b);
          // K_->Add(i, j, (*alpha_coeffs)[k][m] * (*alpha_coeffs)[k][n]);
          
        }
      }

    }
    
    K_->Finalize();
    *K_ *= weight;
    return K_;
  }
  
}


double SysOperator::get_plasma_current(GridFunction &x, double &alpha) {

  model->set_alpha_bar(alpha);

  double val_ma, val_x;
  int iprint = 0;
  set<int> plasma_inds_;
  compute_plasma_points(&x, *mesh, vertex_map, plasma_inds_, ind_ma, ind_x, val_ma, val_x, iprint);
  psi_x = val_x;
  psi_ma = val_ma;
  plasma_inds = plasma_inds_;

  double* x_ma_ = mesh->GetVertex(ind_ma);
  double* x_x_ = mesh->GetVertex(ind_x);
  x_ma = x_ma_;
  x_x = x_x_;

  NonlinearGridCoefficient nlgcoeff1(model, 1, &x, val_ma, val_x, plasma_inds, attr_lim);

  LinearForm plasma_term(fespace);
  plasma_term.AddDomainIntegrator(new DomainLFIntegrator(nlgcoeff1));
  plasma_term.Assemble();

  Plasma_Current = plasma_term(ones);
  Plasma_Current *= -1.0;

  return Plasma_Current;
}


void SysOperator::NonlinearEquationRes(GridFunction &psi, Vector *currents, double &alpha) {
  // Assemble the necessary constraints and Jacobians needed to solve the linearized system of equations
  // solved by Newton for the PDE-constrained optimization.
  // Specific operators/vectors assembled:
  // 1) Grad-Shafranov PDE constraint: B(y, alpha) - F u = 0
  // 2) Total plasma current constraint: C(y, alpha) = I_p
  // 3) Jacobians needed in the linearized system of equations solved by Newton: B_y, B_alpha, C_y, C_alpha

  GridFunction x(fespace);
  x = psi;

  // Set alpha in the plasma model to current alpha value
  model->set_alpha_bar(alpha);

  // Compute the X-point and magnetic axis and determine plasma region
  double val_ma, val_x;
  int iprint = 0;
  set<int> plasma_inds_;
  compute_plasma_points(&x, *mesh, vertex_map, plasma_inds_, ind_ma, ind_x, val_ma, val_x, iprint);

  psi_x = val_x;
  psi_ma = val_ma;
  plasma_inds = plasma_inds_;

  // Get vertices associated with X-point and magnetic axis
  double* x_ma_ = mesh->GetVertex(ind_ma);
  double* x_x_ = mesh->GetVertex(ind_x);
  x_ma = x_ma_;
  x_x = x_x_;
 
  // Create coefficient object to evaluate plasma source terms (GS RHS) at any point
  NonlinearGridCoefficient nlgcoeff0(model, 0, &x, val_ma, val_x, plasma_inds, attr_lim);
  GridFunction f_(fespace);
  f_.ProjectCoefficient(nlgcoeff0);
  f_.Save("gf/f.gf");

  f = f_;

  // ----------------------------------------------------------------------------------------
  // Compute the residual: res = B(y, alpha) - F u
  // ----------------------------------------------------------------------------------------

  // Assemble contribution from plasma source terms in GS
  NonlinearGridCoefficient nlgcoeff1(model, 1, &x, val_ma, val_x, plasma_inds, attr_lim);
  LinearForm plasma_term(fespace);
  plasma_term.AddDomainIntegrator(new DomainLFIntegrator(nlgcoeff1));
  plasma_term.Assemble();

  // Start by setting res as the diffusion operator contribution
  diff_operator->Mult(psi, res);
  
  // Subtract coil term (and plasma term if enabled) from res -> this is B(y, alpha)
  add(res, -1.0, *coil_term, res);
  if (include_plasma) {
    add(res, -1.0, plasma_term, res);
  }

  // Debugging/visualization
  GridFunction ffp(fespace);
  ffp = 0;
  add(ffp, 1.0, plasma_term, ffp);
  ffp.Save("gf/plasma_term.gf");

  // Debugging/visualization
  GridFunction outres(fespace);
  diff_operator->Mult(psi, outres);
  outres.Save("gf/diff_operator.gf");

  // Debugging/visualization
  GridFunction plascoeff(fespace);
  plascoeff.ProjectCoefficient(nlgcoeff1);
  plascoeff.Save("gf/plascoeff.gf");

  // Include contribution from currents to res -> This is F u
  F->AddMult(*currents, res, -model->get_mu());

  // Enforce Dirichlet boundary conditions
  Vector u_b_exact, u_tmp, u_b;
  psi.GetSubVector(boundary_tdofs, u_b);
  u_tmp = u_b;
  u_boundary->GetSubVector(boundary_tdofs, u_b_exact);
  u_tmp -= u_b_exact;
  res.SetSubVector(boundary_tdofs, u_tmp);

  // Zero-out residual where we are interpolating from a guess
  res *= hat;

  // ----------------------------------------------------------------------------------------
  // Compute Jacobians: B_y, B_alpha, C_y, C_alpha
  // ----------------------------------------------------------------------------------------

  // Bilinear operator corresponding to the Jacobian contribution from plasma source term (Gateaux semiderivative--Eq. 3.10 in the paper)
  NonlinearGridCoefficient nlgcoeff_2(model, 2, &x, psi_ma, psi_x, plasma_inds, attr_lim);
  BilinearForm diff_plasma_term_2(fespace);
  diff_plasma_term_2.AddDomainIntegrator(new MassIntegrator(nlgcoeff_2));
  diff_plasma_term_2.Assemble();

  // Jacobian contribution associated with the magnetic axis: build column in Jacobian corresponding to the magnetic axis DOF
  NonlinearGridCoefficient nlgcoeff_3(model, 3, &x, psi_ma, psi_x, plasma_inds, attr_lim);
  LinearForm diff_plasma_term_3(fespace);
  diff_plasma_term_3.AddDomainIntegrator(new DomainLFIntegrator(nlgcoeff_3));
  diff_plasma_term_3.Assemble();

  // Jacobian contribution associated with the X-point: build column in Jacobian corresponding to the X-point DOF
  NonlinearGridCoefficient nlgcoeff_4(model, 4, &x, psi_ma, psi_x, plasma_inds, attr_lim);
  LinearForm diff_plasma_term_4(fespace);
  diff_plasma_term_4.AddDomainIntegrator(new DomainLFIntegrator(nlgcoeff_4));
  diff_plasma_term_4.Assemble();

  // Turn diff_operator and diff_plasma_term_2 into sparse matrices in CSR format
  SparseMatrix diff_operator_sp_mat = diff_operator->SpMat();
  SparseMatrix psi_coeff_sp_mat = diff_plasma_term_2.SpMat();
  diff_operator_sp_mat.Finalize();
  psi_coeff_sp_mat.Finalize();

  // Create a new sparse matrix that will later combine all terms to form the Jacobian
  // Use GetVSize() so all operators are consistently in full-DOF space.
  // The true-DOF conversion (for hanging node constraints) happens in gs.cpp
  // before the block system is solved.
  const int m = fespace->GetVSize();
  SparseMatrix *psi_x_psi_ma_coeff_sp_mat = new SparseMatrix(m, m);

  // Build the two columns of the Jacobian that correspond to the magnetic axis and X-point
  for (int k = 0; k < m; ++k) {
    psi_x_psi_ma_coeff_sp_mat->Add(k, ind_ma, -diff_plasma_term_3[k]);
    psi_x_psi_ma_coeff_sp_mat->Add(k, ind_x, -diff_plasma_term_4[k]);
  }
  psi_x_psi_ma_coeff_sp_mat->Finalize();

  // Build a preliminary Jacobian that will eventually become B_y
  SparseMatrix *Mat_Prelim;
  if (!include_plasma) {
    Mat_Prelim = Add(1.0, diff_operator_sp_mat, 0.0, diff_operator_sp_mat);  // Jacobian is just the diffusion operator
  }
  else {
    Mat_Prelim = Add(1.0, diff_operator_sp_mat, -1.0, psi_coeff_sp_mat);  // Jacobian is the diffusion operator minus the Gateaux semiderivative of the plasma source
  }

  // Apply Dirichlet boundary conditions to the preliminary Jacobian
  for (int k = 0; k < boundary_tdofs.Size(); ++k) {
    Mat_Prelim->EliminateRow((boundary_tdofs)[k], DIAG_ONE);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Create a copy of Mat_Prelim to produce a symmetric B_y (no magnetic axis or X-point contributions). Used elsewhere for preconditioning
  By_symmetric = Add(1.0, *Mat_Prelim, 0.0, *Mat_Prelim);

  
  // Produce B_y
  if (!include_plasma) {
    By = Add(1.0, *Mat_Prelim, 0.0, *Mat_Prelim);
  }
  else {
    By = Add(*Mat_Prelim, *psi_x_psi_ma_coeff_sp_mat);
  }
  
  // Compute B_alpha
  NonlinearGridCoefficient nlgcoeff_5(model, 5, &x, psi_ma, psi_x, plasma_inds, attr_lim);
  LinearForm diff_plasma_term_5(fespace);
  diff_plasma_term_5.AddDomainIntegrator(new DomainLFIntegrator(nlgcoeff_5));
  diff_plasma_term_5.Assemble();
  Ba = diff_plasma_term_5;
  Ba *= -1.0;

  // Assemble Jacobian of plasma operator w.r.t. ψ
  SparseMatrix *Mat_Plasma = Add(-1.0, *psi_x_psi_ma_coeff_sp_mat, 1.0, psi_coeff_sp_mat);

  // Compute plasma current I_p
  plasma_current = plasma_term(ones);
  plasma_current *= -1.0;
  
  // Compute C_y (derivative of I_p w.r.t. y)
  Vector Plasma_Vec_(m);
  Mat_Plasma->MultTranspose(ones, Plasma_Vec_);
  Plasma_Vec_ *= -1.0;
  Cy = Plasma_Vec_;

  // Compute C_alpha (derivative of I_p w.r.t. alpha)
  Ca = diff_plasma_term_5(ones);
  Ca *= -1.0;
}


void SysOperator::Mult(const Vector &psi, Vector &y) const {
}


void ByMinusRankOnePerturbation::Mult(const Vector &k, Vector &y) const {
  // By - 1/Ca Ba Cy^T
  double inner = k * (*Cy); // inner product
  add(- inner / ca, *Ba, 0.0, *Ba, y);
  By->AddMult(k, y, 1.0);
};


void ByMinusRankOnePerturbation::MultTranspose(const Vector &k, Vector &y) const {
  // By^T - 1/Ca Cy Ba^T
  double inner = k * (*Ba); // inner product
  add(- inner / ca, *Cy, 0.0, *Cy, y);
  By->AddMultTranspose(k, y, 1.0);
};


void SchurComplement::Mult(const Vector &x, Vector &y) const {
  // D - C A^{-1} B

  B->Mult(x, a);
  b = 0.0;
  a *= -1.0;
  solver.Mult(a, b);
  C->Mult(b, y);

  D->AddMult(x, y);

  total_calls += 1;
  total_iterations += solver.GetNumIterations();
}


void SchurComplementInverse::Mult(const Vector &x, Vector &y) const {
  // (D - C A^{-1} B)^{-1}

  b = 0.0;
  solver.Mult(x, b);
  y = b;

  total_calls += 1;
  total_iterations += solver.GetNumIterations();
}


void WoodburyInverse::Mult(const Vector &x, Vector &y) const {
  // (A + uv^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
  //
  // A^{-1} x = a
  // A^{-1} u = b

  // A_prec->Mult(x, a);
  a = 0.0;
  solver.Mult(x, a);

  add(1.0, a, - scale * ((*V) * a) / (1.0 + scale * dot), b, y);

  total_calls += 1;
  total_iterations += solver.GetNumIterations();
};


void WoodburyInversePC::Mult(const Vector &x, Vector &y) const {
  // (A + uv^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
  //
  // A^{-1} x = a
  // A^{-1} u = b

  // A_prec->Mult(x, a);
  a = 0.0;
  A_prec->Mult(x, a);

  add(1.0, a, - scale * ((*V) * a) / (1.0 + scale * dot), b, y);
};
