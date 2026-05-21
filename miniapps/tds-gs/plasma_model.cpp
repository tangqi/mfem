#include "mfem.hpp"
#include "plasma_model.hpp"
#include <iostream>
#include <set>
#include <list>
using namespace mfem;
using namespace std;

// ***************************************************
// These functions involve a simplified plasma model
double PlasmaModel::S_p_prime(double & psi_N) const {

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


double PlasmaModel::S_prime_p_prime(double & psi_N) const {

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


double PlasmaModel::S_ff_prime(double & psi_N) const {

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


double PlasmaModel::S_prime_ff_prime(double & psi_N) const {

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


double PlasmaModelFile::S_prime_p_prime(double & psi_N) const {

  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }
  int index = (int) (psi_N / dx);

  
  return (pprime_vector[index+1] - pprime_vector[index]) / dx;
}


double PlasmaModelFile::S_ff_prime(double & psi_N) const {

  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }
  int index = (int) (psi_N / dx);
  double alpha = (psi_N - index * dx) / dx;
  
  return alpha * ffprime_vector[index+1] + (1 - alpha) * ffprime_vector[index];
}


double PlasmaModelFile::S_prime_ff_prime(double & psi_N) const {

  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }

  int index = (int) (psi_N / dx);

  return (ffprime_vector[index+1] - ffprime_vector[index]) / dx;
}


double PlasmaModelFile::f_bar(double & psi_N) const {

  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }

  int index = (int) (psi_N / dx);
  double alpha = (psi_N - index * dx) / dx;
  
  return alpha * fpol_bar_vector[index+1] + (1 - alpha) * fpol_bar_vector[index];
}


double PlasmaModelFile::f_bar_prime(double & psi_N) const {

  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }

  int index = (int) (psi_N / dx);

  return (fpol_bar_vector[index+1] - fpol_bar_vector[index]) / dx;
}


double PlasmaModelFile::f_bar_double_prime(double & psi_N) const {

  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }

  return 0.0;
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
  const int nv = T.mesh->GetElement(T.ElementNo)->GetNVertices();
  set<int>::iterator plasma_inds_it;
  for (int i = 0; i < nv; ++i) {
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


/**
* Build a vertex adjacency map for a mesh. For each vertex in the mesh, record
* the set of neighboring vertices that share an element with it. Optionally,
* the adjacency can be restricted to elements with a specific attribute.
*
* @param mesh        MFEM mesh containing the elements and vertices.
* @param with_attrib Element attribute filter. If -1, all elements are used;
*                    otherwise only elements whose attribute matches this
*                    value contribute to the adjacency map.
*
* @return A map from vertex index -> vector of neighboring vertex indices.
*/
map<int, vector<int>> compute_vertex_map(Mesh &mesh, int with_attrib) {
  map<int, vector<int>> vertex_map;

  for (int i = 0; i < mesh.GetNE(); i++) {

    // Get element node indices, number of nodes in the element, and element attributes
    const int *v = mesh.GetElement(i)->GetVertices();
    const int nv = mesh.GetElement(i)->GetNVertices();
    const int attrib = mesh.GetElement(i)->GetAttribute();

    // with_attrib = -1: accept all elements. Otherwise, only accept elements that match with_attrib input
    if ((with_attrib == -1) || (attrib == with_attrib)) {

      // For each edge in an element, store every pair of vertices in vertex_map
      for (int a = 0; a < nv; ++a) {
        for (int b = 0; b < nv; ++b) {
          if (a == b) continue;
          vertex_map[v[a]].push_back(v[b]);
        }
      }
    }
  }

  // Remove duplicates in vertex_map
  for (auto &p : vertex_map)
  {
      auto &nbrs = p.second;
      std::sort(nbrs.begin(), nbrs.end());
      nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
  }

  return vertex_map;
}


/**
* @brief Identify magnetic axis, X-point, and plasma region for a given flux field ψ.
*
* This routine analyzes the nodal values of the Grad–Shafranov solution ψ
* stored in the MFEM GridFunction `z` and performs three tasks:
*
* 1. **Magnetic axis detection (O-point)**  
*    The magnetic axis is defined as the global minimum of ψ over all mesh vertices.
*
* 2. **X-point (saddle point) detection**  
*    For each vertex, neighboring vertices are sorted by polar angle around
*    the candidate vertex. The differences are examined along the circular ordering
*    of neighbors. A vertex is classified as a saddle point if at least four sign changes
*    occur. Among all detected saddles, the one with the smallest ψ value is
*    selected as the X-point. If no saddles are detected, the vertex with
*    maximum ψ is used as a fallback.
*
* 3. **Plasma region identification**  
*    Starting from the magnetic axis, a breadth-first search (BFS) over the
*    vertex adjacency graph marks vertices belonging to the plasma region.
*    A vertex is classified as inside the plasma if ψ_axis ≤ ψ ≤ ψ_X
*    where ψ_axis is the flux at the magnetic axis and ψ_X is the flux at
*    the X-point.
*
* The vertex adjacency information is provided by `vertex_map`, which lists
* neighboring vertices for each vertex in the mesh.
*
* @param[in]  z
*      GridFunction containing the nodal values of ψ on the mesh.
*
* @param[in]  mesh
*      MFEM mesh containing vertex coordinates and element connectivity.
*
* @param[in]  vertex_map
*      Map from vertex index → vector of neighboring vertex indices used for
*      saddle detection and BFS traversal.
*
* @param[out] plasma_inds
*      Set of vertex indices classified as belonging to the plasma region.
*
* @param[out] ind_min
*      Index of the magnetic axis vertex (global minimum of ψ).
*
* @param[out] ind_x
*      Index of the detected X-point vertex.
*
* @param[out] min_val
*      Value of ψ at the magnetic axis.
*
* @param[out] val_x
*      Value of ψ at the X-point.
*
* @param[in]  iprint
*      Verbosity flag. If non-zero, diagnostic information about detected
*      extrema and saddle points is printed.
*/
void compute_plasma_points(
  GridFunction *z,
  const Mesh &mesh,
  const map<int,
  vector<int>> &vertex_map,
  set<int> &plasma_inds,
  int &ind_min,
  int &ind_x,
  double &min_val,
  double &val_x,
  int iprint,
  const mfem::SparseMatrix *cP
) {
  Vector nval;
  z->GetNodalValues(nval);

  // Running min/max trackers for the vertex sweep below. The magnetic axis
  // is the global minimum, so min_val / ind_min are the function's outputs
  // for it directly. The running max is only used as the fallback when no
  // saddle is detected; the final X-point output (val_x / ind_x) is set
  // from the saddle search further down.
  min_val = + numeric_limits<double>::infinity();
  ind_min = 0;
  double running_max_val = - numeric_limits<double>::infinity();
  int running_max_idx = 0;

  // Slave (hanging-node) DOFs are not independent degrees of freedom —
  // their values are an affine combination of their masters — so the
  // axis/X-point extremum search would pick near-equal slave vertices
  // differently from iteration to iteration and stall Newton in a limit
  // cycle. Exclude them from the argmin/argmax and saddle-candidate set.
  // On a conforming mesh cP==nullptr and this reduces to the old behavior.
  auto is_slave = [&](int iv) -> bool {
    if (cP == nullptr) return false;
    const int *I = cP->GetI();
    const int *J = cP->GetJ();
    const double *V = cP->GetData();
    const int nnz = I[iv+1] - I[iv];
    if (nnz != 1) return true;
    return (J[I[iv]] != iv) || (V[I[iv]] != 1.0);
  };

  vector<int> candidate_x_points;
  int saddle_pt_count = 0;

  // Loop over all vertices to determine: global minimum, global maximum, candidate saddle points
  for(int iv = 0; iv < mesh.GetNV(); ++iv) {

    // Get neighbors of vertex iv from adjacency map
    vector<int> adjacent;
    try {
      adjacent = vertex_map.at(iv);
    } catch (...) {
      continue;
    }

    const bool iv_is_slave = is_slave(iv);

    // Find global minimum and maximum values and indices of z
    if (!iv_is_slave) {
      if (nval[iv] < min_val) {
        min_val = nval[iv];
        ind_min = iv;
      }
      if (nval[iv] > running_max_val) {
        running_max_val = nval[iv];
        running_max_idx = iv;
      }
    }

    // -----------------------------------------------------------------------
    // Detect saddle point candidates using neighbor sign changes
    // -----------------------------------------------------------------------

    const double* x0 = mesh.GetVertex(iv);
    map<double, double> clock;
    set<double> ordered_angs;

    // For each node iv, sort adjacent nodes by angular order
    int j = 0;
    for (j = 0; j < static_cast<int>(adjacent.size()); ++j) {

      const int jv = adjacent[j];
      const double *b = mesh.GetVertex(jv);

      // Difference in z between center node iv and adjacent node
      double diff = nval[jv] - nval[iv];

      // Compute polar angle w.r.t x-axis
      double bx = b[0]-x0[0];
      double by = b[1]-x0[1];
      double ang = atan2(by, bx);

      // Store polar angles and their associated differences
      clock[ang] = diff;
      ordered_angs.insert(ang);
    }

    // For each node iv, loop through adjacent nodes to see if iv is a saddle point.
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
    if (prev * init < 0.0) {  // Complete the loop: last adjacent node to first adjacent node
      ++sign_changes;
    }

    // If 4 or more sign changes, save node iv as a saddle point.
    // Skip slaves: their ψ value is an affine combination of masters and
    // picking one as the X-point destabilizes Newton across iterations.
    if (sign_changes >= 4 && !iv_is_slave) {
      if (iprint) {
        printf("Found saddle at (%9.6f, %9.6f), val=%9.6f\n", x0[0], x0[1], nval[iv]);
      }

      cout << "Found saddle at (" << x0[0] << ", " << x0[1] << "), val=" << nval[iv] << endl;  // Debugging: remove later

      candidate_x_points.push_back(iv);
      ++saddle_pt_count;
    }
  }

  // Determine which saddle point is the X-point. Initial fallback is the
  // running global max (used when no saddle survives the filter below).
  ind_x = running_max_idx;
  val_x = running_max_val;
  const double axis_tol = 1e-2 * std::max(1.0, std::abs(min_val));  // Tolerance to avoid spurious candidates
  for (int i = 0; i < static_cast<int>(candidate_x_points.size()); ++i) {
    int iv = candidate_x_points[i];
    if (iv == ind_min)                               { continue; }
    if (std::abs(nval[iv] - min_val) < axis_tol)     { continue; }
    if (nval[iv] < val_x) {
      val_x = nval[iv];
      ind_x = iv;
    }
  }

  const double* x_min = mesh.GetVertex(ind_min);
  const double* x_max = mesh.GetVertex(running_max_idx);
  const double* x_x = mesh.GetVertex(ind_x);

  cout << "total saddles found: " << saddle_pt_count << endl;  // Debugging: remove later

  if (iprint) {
    printf("  min of %9.6f at (%9.6f, %9.6f), ind %d\n", min_val, x_min[0], x_min[1], ind_min);
    printf("  max of %9.6f at (%9.6f, %9.6f), ind %d\n", running_max_val, x_max[0], x_max[1], running_max_idx);
    printf("x_val of %9.6f at (%9.6f, %9.6f), ind %d\n", val_x, x_x[0], x_x[1], ind_x);
  }

  // ---------------------------------------------------------------------------
  // Plasma region identification
  // ---------------------------------------------------------------------------

  // Initialize queue for BFS and set to hold vertices classified as part of the plasma region
  list<int> queue;
  set<int>::iterator plasma_inds_it;

  // Start BFS from minimum vertex index
  queue.push_back(ind_min);
  plasma_inds.insert(ind_min);
  plasma_inds.insert(ind_x);
  while (!queue.empty()) {

    // Get a point that is already in the plasma region
    int iv = queue.front();
    queue.pop_front();

    // Check for neighboring points and store in adjacent
    vector<int> adjacent;
    try {
      adjacent = vertex_map.at(iv);
    } catch (...) {
      continue;
    }

    // Check if the neighboring points are in the plasma region
    for (int i = 0; i < static_cast<int>(adjacent.size()); ++i) {
      double val = nval[adjacent[i]];
      plasma_inds_it = plasma_inds.find(adjacent[i]);

      // Check that found vertex is not already accounted for
      if (plasma_inds_it == plasma_inds.end()) {

        // If the value at this vertex is between min and X-point vals, then add to plasma region
        if ((val >= min_val) && (val <= val_x)) {
          queue.push_back(adjacent[i]);
          plasma_inds.insert(adjacent[i]);
        }
         
        else {
          // If the value at this vertex is not between min and X-point vals, don't add but mark as visited.
          plasma_inds.insert(adjacent[i]);
        }
      }
    }
  }
}
