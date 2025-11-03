#include "mfem.hpp"
#include "field.hpp"
#include "plasma_model.hpp"

using namespace std;
using namespace mfem;

double FieldCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip) {
  /**
  * Evaluate a magnetic field component (B_r, B_{\phi}, or B_z) at a given integration point.
  *
  * @param[in] T   Mapping between reference element coordinates to physical coordinates.
  * @param[in] ip  Integration (quadrature) point within the reference element.
  *
  * @return The scalar value of the selected magnetic field component at the given point.
  */
  double x_[3];
  Vector x(x_, 3);
  T.Transform(ip, x);  // Compute the physical coordinates of ip
  double r(x(0));

  int Component = 0;
   
  // Radial component B_r
  if (comp == 0) {
     return psi_z->GetValue(T, ip, Component) / r;  // B_r = \psi_z / r
  }

  // Toroidal component B_{\phi}
  else if (comp == 1) {
     double alpha = model->get_alpha_bar();  // model is from plasma_model.cpp
     double f_x = model->get_f_x();
     double psi_val = psi->GetValue(T, ip, Component);

     return (f_x + alpha * (psi_x - psi_val)) / r;  // B_{\phi} = \frac{f_x + \alpha (\psi_x - \psi)}{r}
  }

  // Vertical component B_z
  else {
     return - psi_r->GetValue(T, ip, Component) / r;  // B_z = - \psi_r / r
  }
}
