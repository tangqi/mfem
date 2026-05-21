#include "mfem.hpp"
#include "diffusion_term.hpp"

using namespace mfem;
using namespace std;

// Evaluate the diffusion coefficient 1/r for the diffusion operator at a given quadrature point
double DiffusionIntegratorCoefficient::Eval(
  ElementTransformation &T,
  const IntegrationPoint &ip
) {
  // Transform integration (quadrature point) from reference to physical coordinates (r, z)
  double x_[3];
  Vector x(x_, 3);
  T.Transform(ip, x);

  // Extract r component
  double ri(x(0));

  if (T.Attribute != 1100) {  // What is 1100? A coil or a region in the domain?
    return 1.0 / (ri);
  }

  else {
    return 0.0;
  }
}
