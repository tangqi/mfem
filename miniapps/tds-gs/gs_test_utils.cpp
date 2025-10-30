////////////////////////////////////////////////////////////////////////////////

/**
  * @file gs_test_utils.cpp
  * @brief Debugging and verification utilities for gs.cpp
  *
  * This file provides some developer-side tools for verifying analytic derivatives
  * implemented in the Grad-Shafranov solver, as well as printing vectors and
  * matrices. These functions are meant for internal debugging only. Currently not used.
*/

#include "mfem.hpp"
#include "gs.hpp"
#include "gs_test_utils.hpp"
#include <stdio.h>

using namespace std;
using namespace mfem;

void Print_(const Vector &y) {
  /**
    * Print the entries of an MFEM Vector y to 14 digits of precision.
    * Intended for debugging purposes.
    *
    * @param[in] y  Input vector.
  */
  for (int i = 0; i < y.Size(); ++i) {
    printf("%d %.14e\n", i+1, y[i]);
  }
}


void CompareSparseMatrices(SparseMatrix *Mat, SparseMatrix *M1, SparseMatrix *M2) {
  /**
    * Print the element-wise difference between the sparse matrices M1 and M2, stored in CSR format.
    * Flag any large differences with "***".
    *
    * @param[in] Mat  The difference matrix (usually Mat = M1 - M2).
    * @param[in] M1   The analytic Jacobian matrix.
    * @param[in] M2   The finite-difference approximation.
  */

  int *I = Mat->GetI();
  int *J = Mat->GetJ();
  double *A = Mat->GetData();
  int height = Mat->Height();

  // Define a tolerance
  double tol = 1e-5;
  
  int i, j;
  for (i = 0; i < height; ++i) {
    for (j = I[i]; j < I[i+1]; ++j) {

      // Only consider entries in the difference matrix whose magnitude exceeds the set tolerance
      if (abs(A[j]) > tol) {
        double m1 = 0.0;
        double m2 = 0.0;

        // Get corresponding entry in M1
        for (int k = M1->GetI()[i]; k < M1->GetI()[i+1]; ++k) {
          if (M1->GetJ()[k] == J[j]) {
            m1 = M1->GetData()[k];
            break;
          }
        }

        // Get corresponding entry in M2
        for (int k = M2->GetI()[i]; k < M2->GetI()[i+1]; ++k) {
          if (M2->GetJ()[k] == J[j]) {
            m2 = M2->GetData()[k];
            break;
          }
        }
        
        // Print the analytic Jacobian (J) and finite difference (FD) values, along with their differences,
        // and flag with "***" when the relative difference is greater than 1e-4.
        printf("i=%d, j=%d, J=%10.3e, FD=%10.3e, diff=%10.3e ", i, J[j], m1, m2, A[j] / max(m1, m2));
        if (abs(A[j] / max(m1, m2)) > 1e-4) {
          printf("***");
        }
        printf("\n");
      }
    }
  }
}


void TestGrad(SysOperator *op, GridFunction x, FiniteElementSpace fespace) {
  /**
    * Perform finite-difference check against the analytic derivatives
    * implemented in the solver. Verify that (d C) / (d \alpha), (d B) / (d \alpha),
    * (d C) / (d y), and (d B) / (d y) all match their finite-difference approximations.
    * Intended for debugging purposes.
    *
    * @param[in] op       Nonlinear operator.
    * @param[in] x        Current finite element field solution.
    * @param[in] fespace  Discretized function space.
  */
  LinearForm y1(&fespace);
  LinearForm y2(&fespace);
  LinearForm fy(&fespace);

  int size = y1.Size();

  Vector *currents = op->get_uv();
  GridFunction res_1(&fespace);
  GridFunction res_2(&fespace);
  GridFunction res_3(&fespace);
  GridFunction res_4(&fespace);
  GridFunction Cy(&fespace);
  GridFunction Ba(&fespace);
  SparseMatrix By;
  double plasma_current_1, plasma_current_2;
  double Ca;

  // *********************************
  // Test Ca and Ba
  double alpha = 1.0;
  double eps = 1e-4;

  alpha += eps;
  op->NonlinearEquationRes(x, currents, alpha);
  plasma_current_1 = op->get_plasma_current();
  res_1 = op->get_res();

  alpha -= eps;
  op->NonlinearEquationRes(x, currents, alpha);
  plasma_current_2 = op->get_plasma_current();
  Ca = op->get_Ca();

  printf("Ca: %e\n", Ca);
  
  Ba = op->get_Ba();
  res_2 = op->get_res();

  double Ca_FD = (plasma_current_1 - plasma_current_2) / eps;

  printf("\ndC/dalpha\n");
  printf("Ca: code=%e, FD=%e, Diff=%e\n", Ca, Ca_FD, Ca-Ca_FD);

  GridFunction Ba_FD(&fespace);
  add(1.0 / eps, res_1, -1.0 / eps, res_2, Ba_FD);
  printf("\ndB/dalpha\n");
  for (int i = 0; i < size; ++i) {
    if ((Ba_FD[i] != 0) || (Ba[i] != 0)) {
      printf("%d: code=%e, FD=%e, Diff=%e\n", i, Ba[i], Ba_FD[i], Ba[i]-Ba_FD[i]);
    }
  }

  // *********************************
  // Test Cy and By, ind_x
  int ind_x = op->get_ind_x();
  int ind_ma = op->get_ind_ma();

  int ind = ind_x;
  while (true) {
    x[ind] += eps;
    op->NonlinearEquationRes(x, currents, alpha);
    plasma_current_1 = op->get_plasma_current();
    res_3 = op->get_res();

    x[ind] -= eps;
    op->NonlinearEquationRes(x, currents, alpha);
    plasma_current_2 = op->get_plasma_current();
    Cy = op->get_Cy();
    By = op->get_By();
    res_4 = op->get_res();

    double Cy_FD = (plasma_current_1 - plasma_current_2) / eps;
  
    printf("\ndC/dy\n");
    printf("Cy: code=%e, FD=%e, Diff=%e\n", Cy[ind], Cy_FD, Cy[ind]-Cy_FD);

    printf("\ndB/dy\n");
    GridFunction By_FD(&fespace);
    add(1.0 / eps, res_3, -1.0 / eps, res_4, By_FD);

    int *I = By.GetI();
    int *J = By.GetJ();
    double *A = By.GetData();
    int height = By.Height();
    for (int i = 0; i < height; ++i) {
      for (int j = I[i]; j < I[i+1]; ++j) {
        if ((J[j] == ind)) {
          printf("%d %d: code=%e, FD=%e, Diff=%e\n", i, J[j], A[j], By_FD[i], A[j]-By_FD[i]);
        }
      }
    }

    if (ind == ind_x) {
      ind = ind_ma;
    } else {
      break;
    }

  }

  if (true) {
    return;
  }
  
  // *********************************
  // Test grad_obj and hess_obj
  double obj1, obj2, grad_obj_FD;

  op->set_i_option(2);
  
  GridFunction grad_obj(&fespace);
  GridFunction grad_obj_1(&fespace);
  GridFunction grad_obj_2(&fespace);
  GridFunction grad_obj_3(&fespace);
  GridFunction grad_obj_4(&fespace);
  grad_obj = op->compute_grad_obj(x);
  printf("\ndf/dy\n");
  for (int i = 0; i < size; ++i) {
    x[i] += eps;
    obj1 = op->compute_obj(x);
    x[i] -= eps;
    obj2 = op->compute_obj(x);

    grad_obj_FD = (obj1 - obj2) / eps;
    if ((grad_obj[i] != 0) || (grad_obj_FD != 0)) {
      printf("%d: code=%e, FD=%e, Diff=%e\n", i, grad_obj[i], grad_obj_FD, grad_obj[i]-grad_obj_FD);
    }
  }

  x[ind_x] += eps;
  grad_obj_1 = op->compute_grad_obj(x);

  x[ind_x] -= eps;
  grad_obj_2 = op->compute_grad_obj(x);

  x[ind_ma] += eps;
  grad_obj_3 = op->compute_grad_obj(x);

  x[ind_ma] -= eps;
  grad_obj_4 = op->compute_grad_obj(x);

  GridFunction K_FD_x(&fespace);
  GridFunction K_FD_ma(&fespace);
  add(1.0 / eps, grad_obj_1, -1.0 / eps, grad_obj_2, K_FD_x);
  add(1.0 / eps, grad_obj_3, -1.0 / eps, grad_obj_4, K_FD_ma);
  
  SparseMatrix * K = op->compute_hess_obj(x);

  printf("ind_x =%d\n", ind_x);
  printf("ind_ma=%d\n", ind_ma);
  
  printf("\nd2f/dy2\n");
  int *I_K = K->GetI();
  int *J_K = K->GetJ();
  double *A_K = K->GetData();
  int height_K = K->Height();
  double TOL = 1e-15;
  for (int i = 0; i < height_K; ++i) {
    for (int j = I_K[i]; j < I_K[i+1]; ++j) {
      if ((J_K[j] == ind_x) && ((abs(A_K[j]) > TOL) || (abs(K_FD_x[i]) > TOL))) {
        printf("%d %d: code=%e, FD=%e, Diff=%e\n", i, J_K[j], A_K[j], K_FD_x[i], A_K[j]-K_FD_x[i]);
      }
      if ((J_K[j] == ind_ma) && ((abs(A_K[j]) > TOL) || (abs(K_FD_ma[i]) > TOL))) {
        printf("%d %d: code=%e, FD=%e, Diff=%e\n", i, J_K[j], A_K[j], K_FD_ma[i], A_K[j]-K_FD_ma[i]);
      }
    }
  }
}
