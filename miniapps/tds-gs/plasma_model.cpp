#include "mfem.hpp"
#include "plasma_model.hpp"
#include <iostream>
#include <set>
#include <list>
using namespace mfem;
using namespace std;

// ***************************************************
// These functions involve a simplified plasma model
double PlasmaModel::S_p_prime(double & psi_N) const
{
  // return zero when derivative is singular
  if ((gamma < 1.0) && (psi_N == 1.0)) {
    return 0.0;
  }
  if ((alpha < 1.0) && (psi_N == 0.0)) {
    return 0.0;
  }
  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }
  return lambda * beta * pow(1.0 - pow(psi_N, alpha), gamma) / r0;
}


double PlasmaModel::S_prime_p_prime(double & psi_N) const
{
  // return zero when derivative is singular
  if ((gamma < 1.0) && (psi_N == 1.0)) {
    return 0.0;
  }
  if ((alpha < 1.0) && (psi_N == 0.0)) {
    return 0.0;
  }
  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }
  return - alpha * gamma * lambda * beta
    * pow(1.0 - pow(psi_N, alpha), gamma - 1.0)
    * pow(psi_N, alpha - 1.0) / r0;
}


double PlasmaModel::S_ff_prime(double & psi_N) const
{
  // return zero when derivative is singular
  if ((gamma < 1.0) && (psi_N == 1.0)) {
    return 0.0;
  }
  if ((alpha < 1.0) && (psi_N == 0.0)) {
    return 0.0;
  }
  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }
  return lambda * (1.0 - beta) * mu0 * r0 * pow(1.0 - pow(psi_N, alpha), gamma);
}


double PlasmaModel::S_prime_ff_prime(double & psi_N) const
{
  // return zero when derivative is singular
  if ((gamma < 1.0) && (psi_N == 1.0)) {
    return 0.0;
  }
  if ((alpha < 1.0) && (psi_N == 0.0)) {
    return 0.0;
  }
  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }

  return - alpha * gamma * lambda * (1.0 - beta) * mu0 * r0
    * pow(1.0 - pow(psi_N, alpha), gamma - 1.0)
    * pow(psi_N, alpha - 1.0);
}


// ***************************************************
// These functions involve a plasma model loaded from a file
double PlasmaModelFile::S_p_prime(double & psi_N) const
{
  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }
  int index = (int) (psi_N / dx);
  double alpha = (psi_N - index * dx) / dx;
  
  return alpha * pprime_vector[index+1] + (1 - alpha) * pprime_vector[index];
}


double PlasmaModelFile::S_prime_p_prime(double & psi_N) const
{
  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }
  int index = (int) (psi_N / dx);

  
  return (pprime_vector[index+1] - pprime_vector[index]) / dx;
}


double PlasmaModelFile::S_ff_prime(double & psi_N) const
{
  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }
  int index = (int) (psi_N / dx);
  double alpha = (psi_N - index * dx) / dx;
  
  return alpha * ffprime_vector[index+1] + (1 - alpha) * ffprime_vector[index];
}


double PlasmaModelFile::S_prime_ff_prime(double & psi_N) const
{
  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }

  int index = (int) (psi_N / dx);
 

  return (ffprime_vector[index+1] - ffprime_vector[index]) / dx;
}


double PlasmaModelFile::f_bar(double & psi_N) const
{
  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }

  int index = (int) (psi_N / dx);
  double alpha = (psi_N - index * dx) / dx;
  
  return alpha * fpol_bar_vector[index+1] + (1 - alpha) * fpol_bar_vector[index];
}


double PlasmaModelFile::f_bar_prime(double & psi_N) const
{
  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }

  int index = (int) (psi_N / dx);

  return (fpol_bar_vector[index+1] - fpol_bar_vector[index]) / dx;
  // return alpha * fpol_bar_prime_vector[index+1] + (1 - alpha) * fpol_bar_prime_vector[index];
}


double PlasmaModelFile::f_bar_double_prime(double & psi_N) const
{
  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }

  return 0.0;
  // return alpha * fpol_bar_double_prime_vector[index+1] + (1 - alpha) * fpol_bar_double_prime_vector[index];
}


double normalized_psi(double & psi, double & psi_max, double & psi_bdp)
{
  if (false) {  // Dead code?
    return psi;
  }
  return (psi - psi_max) / (psi_bdp - psi_max);
}


// Pointwise evaluator for the nonlinear plasma source coefficient.
// Whenever a DomainLFIntegrator or MassIntegrator needs the coefficient at a quadrature point, this function gets called.
double NonlinearGridCoefficient::Eval(ElementTransformation & T, const IntegrationPoint & ip) {

  double f_x = model->get_f_x();
  double alpha_bar = model->get_alpha_bar();
  double alpha = alpha_bar;  // alpha: multiplier in \bar{S}_{ff'} term
  double beta = model->get_beta();  // beta: multiplier for S_{p'} term
  double gamma = model->get_gamma();  // gamma: multiplier for S_{ff'} term
  double mu = model->get_mu();
  double coeff_u2 = model->get_coeff_u2();
  double r0 = 6.2;
  double alpha_0 = 2.0;
  double beta_0 = 0.5978;
  double gamma_0 = 1.395;

  // Restrict to limiter region
  if (T.Attribute != attr_lim) {
    return 0.0;
  }

  // Restrict to plasma elements
  const int *v = T.mesh->GetElement(T.ElementNo)->GetVertices();
  set<int>::iterator plasma_inds_it;
  for (int i = 0; i < 3; ++i) {
    plasma_inds_it = plasma_inds.find(v[i]);
    if (plasma_inds_it == plasma_inds.end()) {
      return 0.0;
    }
  }

  // Transform from reference to physical coordinates
  double x_[3];
  Vector x(x_, 3);
  T.Transform(ip, x);
  double ri(x(0));

  // Get psi and normalized psi at quadrature point
  int Component = 0;
  double psi_val;
  psi_val = psi->GetValue(T, ip, Component);
  double psi_N = normalized_psi(psi_val, psi_max, psi_bdp);

  // Choose model  TODO: probably best to refactor all this switch block stuff to remove it
  int model_choice = model->get_model_choice();
  double switch_beta = 0.0;
  double switch_taylor = 1.0;
  double switch_ff = 0.0;
  double switch_lb = 0.0;

  // Luxon and Brown variant
  if (model_choice == 1) {
    switch_beta = 1.0;
    switch_taylor = 0.0;
    switch_ff = 0.0;
    switch_lb = 0.0;
  }
  
  // Taylor state
  else if (model_choice == 2) {
    switch_beta = 0.0;
    switch_taylor = - 1.0; // 8/31/22 DAS - sign error...
    switch_ff = 0.0;
    switch_lb = 0.0;
  }
  
  // 15MA ITER baseline
  else if (model_choice == 3) {
    switch_beta = 0.0;
    switch_taylor = 0.0;
    switch_ff = 1.0;
    switch_lb = 0.0;
  }
  
  // Luxon and Brown
  else if (model_choice == 4) {
    switch_beta = 0.0;
    switch_taylor = 0.0;
    switch_ff = 0.0;
    switch_lb = 1.0;
  }

  // Return f(psi). This is the Taylor-state formula for f(psi)
  if (option == 0) {
    return f_x + alpha * (psi_bdp - psi_val);
    
  }
  
  // Return the full plasma-source coefficient, depending on the model choice used (1, 2, 3, 4)
  else if (option == 1) {  // POSSIBLE BUG HERE for Taylor state
  
    // Compute the integrand of:
    // int_{\Omega_p(\psi)} (r S_{p'}(\psi_N) + S_{ff'}(\psi_N) / (\mu r) + \bar{S}_{ff'}(\psi)) v dr dz

    double S_bar_ffprime = switch_beta * alpha * (f_x + alpha * (model->f_bar(psi_N))) * (model->f_bar_prime(psi_N)) / (psi_bdp - psi_max)
      + switch_taylor * alpha * (- f_x + alpha * (psi_bdp - psi_val))
      + switch_ff * alpha * (model->S_ff_prime(psi_N))
      + (((psi_N > 0.0) & (psi_N < 1.0)) ? switch_lb * alpha * (1.0 - beta_0) * r0 * pow(1.0 - pow(psi_N, alpha_0), gamma_0) : 0.0);

    return
      beta * ri * (model->S_p_prime(psi_N)) * mu
      + gamma * (model->S_ff_prime(psi_N)) / (ri)
      + S_bar_ffprime / (ri)
      + coeff_u2 * pow(psi_val, 2.0)
      + (((psi_N > 0.0) & (psi_N < 1.0)) ? switch_lb * ri * mu * alpha * beta_0 / r0 * pow(1.0 - pow(psi_N, alpha_0), gamma_0) : 0.0);
  }
  
  // Return the derivative of the plasma source coefficient w.r.t. alpha
  else if (option == 5) {    
    return
      switch_beta * (f_x + alpha * (model->f_bar(psi_N))) * (model->f_bar_prime(psi_N)) / (psi_bdp - psi_max) / (ri)
      + switch_beta * alpha * (model->f_bar(psi_N)) * (model->f_bar_prime(psi_N)) / (psi_bdp - psi_max) / (ri)
      + switch_taylor * (- f_x + 2.0 * alpha * (psi_bdp - psi_val)) / (ri)
      + switch_ff * (model->S_ff_prime(psi_N)) / (ri)
      + (((psi_N > 0.0) & (psi_N < 1.0)) ? switch_lb * (1.0 - beta_0) * r0 * pow(1.0 - pow(psi_N, alpha_0), gamma_0) / ri : 0.0)
      + (((psi_N > 0.0) & (psi_N < 1.0)) ? switch_lb * ri * mu * beta_0 / r0 * pow(1.0 - pow(psi_N, alpha_0), gamma_0) : 0.0);
  }
  
  // Jacobian contribution w.r.t. psi for the nonlinear plasma source. Includes contributions from O- and X-points.
  // This is eq. 3.10 from the paper--the Gateaux semiderivative.
  else {
    double coeff;

    // Compute the integrand of:
    // int_{\Omega_p(\psi)} ( (  r S_{p'}'(\psi_N)
    //                         + S_{ff'}'(\psi_N) / (\mu r) ) d_{\psi} \psi_N(\psi, \phi) v
    //                       + d_{\psi} \bar{S}_{ff'}'(\psi, \phi) v ) dr dz
    // = 
    // int_{\Omega_p(\psi)} ( (  r S_{p'}'(\psi_N)
    //                         + S_{ff'}'(\psi_N) / (\mu r)
    //                         + A                          ) d_{\psi} \psi_N(\psi, \phi) v
    //                       + B phi_x v
    //                       + C phi_ma v) dr dz

    double psi_N_multiplier = beta * ri * (model->S_prime_p_prime(psi_N)) * mu
      + gamma * (model->S_prime_ff_prime(psi_N)) / (ri)
      + switch_beta * alpha * alpha * pow(model->f_bar_prime(psi_N), 2.0) / (psi_bdp - psi_max) / (ri)
      + switch_beta * alpha * (f_x + alpha * (model->f_bar(psi_N))) * (model->f_bar_double_prime(psi_N)) / (psi_bdp - psi_max) / (ri)
      + switch_ff * alpha * (model->S_prime_ff_prime(psi_N)) / (ri)
      - (((psi_N > 0.0) & (psi_N < 1.0)) ? switch_lb * alpha * (1.0 - beta_0) * r0 * gamma_0 * pow(1.0 - pow(psi_N, alpha_0), gamma_0 - 1.0) * alpha_0 * pow(psi_N, alpha_0 - 1.0) / ri : 0.0)
      - (((psi_N > 0.0) & (psi_N < 1.0)) ? switch_lb * alpha * ri * mu * beta_0 / r0 * gamma_0 * pow(1.0 - pow(psi_N, alpha_0), gamma_0 - 1.0) * alpha_0 * pow(psi_N, alpha_0 - 1.0) : 0.0);

    double other = - switch_beta * alpha * (f_x + alpha * (model->f_bar(psi_N))) * (model->f_bar_prime(psi_N))
      / (psi_bdp - psi_max) / (psi_bdp - psi_max) / (ri);

    // Coefficient for phi in d_psi psi_N
    if (option == 2) {
      coeff = 1.0 / (psi_bdp - psi_max) * psi_N_multiplier - switch_taylor * alpha * alpha / (ri);
    }
    
    // Coefficient for phi_ma in d_psi psi_N
    else if (option == 3) {
      coeff = - (1.0 - psi_N) / (psi_bdp - psi_max) * psi_N_multiplier - other;
    }
    
    // Coefficient for phi_x in d_psi psi_N
    else if (option == 4) {
      coeff = - 1.0 * psi_N / (psi_bdp - psi_max) * psi_N_multiplier + other + switch_taylor * alpha * alpha / (ri);
    }

    else {
      MFEM_ABORT("Invalid option in NonlinearGridCoefficient::Eval");
    }

    return coeff + coeff_u2 * 2.0 * psi_val;
  }
}


map<int, vector<int>> compute_vertex_map(Mesh & mesh, int with_attrib) {
  // get map between vertices and neighboring vertices
  map<int, vector<int>> vertex_map;
  for (int i = 0; i < mesh.GetNE(); i++) {
    const int *v = mesh.GetElement(i)->GetVertices();
    const int ne = mesh.GetElement(i)->GetNEdges();
    const int attrib = mesh.GetElement(i)->GetAttribute();

    if ((with_attrib == -1) || (attrib == with_attrib)) {
      for (int j = 0; j < ne; j++) {
        const int *e = mesh.GetElement(i)->GetEdgeVertices(j);
        vertex_map[v[e[0]]].push_back(v[e[1]]);
      }
    }
  }
  return vertex_map;
}


void compute_plasma_points(GridFunction * z, const Mesh & mesh,
                           const map<int, vector<int>> & vertex_map,
                           set<int> & plasma_inds,
                           int &ind_min, int &ind_max, double &min_val, double & max_val,
                           int iprint) {

  // mag ax point: global minimum in z
  // saddle point: closest saddle point to mag ax point, otherwise maximum on limiter boundary

  // keep track of elements inside of plasma region
  
   Vector nval;
   z->GetNodalValues(nval);

   min_val = + numeric_limits<double>::infinity();
   max_val = - numeric_limits<double>::infinity();
   ind_min = 0;
   ind_max = 0;

   vector<int> candidate_x_points;
     
   int count = 0;
   // **********************************************************************************
   // loop through vertices and check neighboring vertices to see if we found a saddle point
   for(int iv = 0; iv < mesh.GetNV(); ++iv) {

     // ensure point is in vertex map
     vector<int> adjacent;
     try {
       adjacent = vertex_map.at(iv);
     } catch (...) {
       continue;
     }

     // min/max checker
     if (nval[iv] < min_val) {
       min_val = nval[iv];
       ind_min = iv;
     }
     if (nval[iv] > max_val) {
       max_val = nval[iv];
       ind_max = iv;
     }
     
     // saddle point checker
     int j = 0;
     const double* x0 = mesh.GetVertex(iv);


     map<double, double> clock;
     set<double> ordered_angs;
     for (j = 0; j < static_cast<int>( adjacent.size()); ++j) {
       const int jv = adjacent[j];
       const double* b = mesh.GetVertex(jv);
       double diff = nval[jv] - nval[iv];
       // cout << b[0] << ", " << b[1] << endl;



       double bx = b[0]-x0[0];
       double by = b[1]-x0[1];

       double ang = atan2(by, bx);
       clock[ang] = diff;
       ordered_angs.insert(ang);
     }

     int sign_changes = 0;
     set<double>::iterator it = ordered_angs.begin();
     double init = clock[*it];
     double prev = clock[*it];
     ++it;
     for (; it != ordered_angs.end(); ++it) {
       if (clock[*it] * prev < 0.0) {
         ++sign_changes;
       }
       prev = clock[*it];
     }
     if (prev * init < 0.0) {
       ++sign_changes;
     }

     if (sign_changes >= 4) {
       if (iprint) {
         printf("Found saddle at (%9.6f, %9.6f), val=%9.6f\n", x0[0], x0[1], nval[iv]);
       }
       candidate_x_points.push_back(iv);
       ++count;
     } 
   }

   int ind_x = ind_max;
   double x_val = max_val;
   for (int i = 0; i < static_cast<int>(candidate_x_points.size()); ++i) {
     int iv = candidate_x_points[i];
     if (nval[iv] < x_val) {
       x_val = nval[iv];
       ind_x = iv;
     }
   }

   const double* x_min = mesh.GetVertex(ind_min);
   const double* x_max = mesh.GetVertex(ind_max);
   const double* x_x = mesh.GetVertex(ind_x);
   
   // cout << "total saddles found: " << count << endl;
   if (iprint) {
     printf("  min of %9.6f at (%9.6f, %9.6f), ind %d\n", min_val, x_min[0], x_min[1], ind_min);
     printf("  max of %9.6f at (%9.6f, %9.6f), ind %d\n", max_val, x_max[0], x_max[1], ind_max);
     printf("x_val of %9.6f at (%9.6f, %9.6f), ind %d\n", x_val, x_x[0], x_x[1], ind_x);
   }

   // DAS: we need to return the x_val, not the max_val.
   // TODO, refactor to make less confusing...
   max_val = x_val;
   ind_max = ind_x;

   // **********************************************************************************
   // start at x_min and mark vertices that are in the plasma
   list<int> queue;
   set<int>::iterator plasma_inds_it;
   queue.push_back(ind_min);
   plasma_inds.insert(ind_min);
   plasma_inds.insert(ind_x);
   while (!queue.empty()) {
     // get a point that is already in the plasma region
     int iv = queue.front();
     queue.pop_front();

     // check for neighboring points
     vector<int> adjacent;
     try {
       adjacent = vertex_map.at(iv);
     } catch (...) {
       continue;
     }

     // check if neighboring points are in the plasma
     for (int i = 0; i < static_cast<int>(adjacent.size()); ++i) {
       double val = nval[adjacent[i]];
       plasma_inds_it = plasma_inds.find(adjacent[i]);
       // check that found vertex is not already accounted for
       if (plasma_inds_it == plasma_inds.end()) {
         // point is not yet in plasma inds
         if ((val >= min_val) && (val <= x_val)) {
           // value at this point is between min and max, it is part of plasma
           // add the point to the queue
           queue.push_back(adjacent[i]);
           plasma_inds.insert(adjacent[i]);
         } else {
           // value at point is not between min and max, it is adjacent to the plasma
           plasma_inds.insert(adjacent[i]);
         }
       }
     }
   }
   
   
}
