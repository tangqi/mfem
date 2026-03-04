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





//
double normalized_psi(double & psi, double & psi_max, double & psi_bdp)
{
  if (false) {
    return psi;
  }
  return (psi - psi_max) / (psi_bdp - psi_max);
}

double NonlinearGridCoefficient::Eval(ElementTransformation & T,
                                      const IntegrationPoint & ip)
{

  if (true) {
    // check that we are in the limiter region
    if (T.Attribute != attr_lim) {
      return 0.0;
    }

    // check to see if integration point is inside an element that is
    // part of the plasma region
    const int *v = T.mesh->GetElement(T.ElementNo)->GetVertices();
    set<int>::iterator plasma_inds_it;
    for (int i = 0; i < 3; ++i) {
      plasma_inds_it = plasma_inds.find(v[i]);
      if (plasma_inds_it == plasma_inds.end()) {
        return 0.0;
      }
    }
  }

  double x_[3];
  Vector x(x_, 3);
  T.Transform(ip, x);
  double ri(x(0));
  if (false) {
    // check to see if we're inside the exact plasma region
    double r0_ = 1.0;
    double z0_ = 0.0;
    double L_ = 0.35;
    double zi(x(1));
    if (abs(ri - r0_) + abs(zi - z0_) > L_) {
      return 0.0;
    }
  }

  // alpha: multiplier in \bar{S}_{ff'} term
  // beta: multiplier for S_{p'} term
  // gamma: multiplier for S_{ff'} term

  double f_x = model->get_f_x();

  double alpha_bar = model->get_alpha_bar();
  double alpha = alpha_bar;
  double beta = model->get_beta();
  double gamma = model->get_gamma();

  double psi_val;

  int Component = 0;

  psi_val = psi->GetValue(T, ip, Component);
  
  double psi_N = normalized_psi(psi_val, psi_max, psi_bdp);
  double mu = model->get_mu();
  double coeff_u2 = model->get_coeff_u2();

  int model_choice = model->get_model_choice();
  double switch_beta = 0.0;
  double switch_taylor = 1.0;
  double switch_ff = 0.0;
  double switch_lb = 0.0;
  double r0 = 6.2;
  double alpha_0 = 2.0;
  double beta_0 = 0.5978;
  double gamma_0 = 1.395;
  if (model_choice == 1) {
    switch_beta = 1.0;
    switch_taylor = 0.0;
    switch_ff = 0.0;
    switch_lb = 0.0;
  } else if (model_choice == 2) {
    switch_beta = 0.0;
    switch_taylor = - 1.0; // 8/31/22 DAS - sign error...
    switch_ff = 0.0;
    switch_lb = 0.0;
  } else if (model_choice == 3) {
    switch_beta = 0.0;
    switch_taylor = 0.0;
    switch_ff = 1.0;
    switch_lb = 0.0;
  } else if (model_choice == 4) {
    switch_beta = 0.0;
    switch_taylor = 0.0;
    switch_ff = 0.0;
    switch_lb = 1.0;
  }

  if (option == 0) {
    // return "f"
    return f_x + alpha * (psi_bdp - psi_val);
    
  } else if (option == 1) {
    // integrand of
    // int_{\Omega_p(\psi)} (  r S_{p'}(\psi_N)
    //                       + S_{ff'}(\psi_N) / (\mu r)
    //                       + \bar{S}_{ff'}(\psi)       ) v dr dz

    double S_bar_ffprime =
      + switch_beta * alpha * (f_x + alpha * (model->f_bar(psi_N))) * (model->f_bar_prime(psi_N)) / (psi_bdp - psi_max)
      + switch_taylor * alpha * (- f_x + alpha * (psi_bdp - psi_val))
      + switch_ff * alpha * (model->S_ff_prime(psi_N))
      + (((psi_N > 0.0) & (psi_N < 1.0)) ? switch_lb * alpha * (1.0 - beta_0) * r0 * pow(1.0 - pow(psi_N, alpha_0), gamma_0) : 0.0);

    return
      + beta * ri * (model->S_p_prime(psi_N)) * mu
      + gamma * (model->S_ff_prime(psi_N)) / (ri)
      + S_bar_ffprime / (ri)
      + coeff_u2 * pow(psi_val, 2.0)
      + (((psi_N > 0.0) & (psi_N < 1.0)) ? switch_lb * ri * mu * alpha * beta_0 / r0 * pow(1.0 - pow(psi_N, alpha_0), gamma_0) : 0.0);
  } else if (option == 5) {
    // derivative with respect to alpha
      
    return
      + switch_beta * (f_x + alpha * (model->f_bar(psi_N))) * (model->f_bar_prime(psi_N)) / (psi_bdp - psi_max) / (ri)
      + switch_beta * alpha * (model->f_bar(psi_N)) * (model->f_bar_prime(psi_N)) / (psi_bdp - psi_max) / (ri)
      + switch_taylor * (- f_x + 2.0 * alpha * (psi_bdp - psi_val)) / (ri)
      + switch_ff * (model->S_ff_prime(psi_N)) / (ri)
      + (((psi_N > 0.0) & (psi_N < 1.0)) ? switch_lb * (1.0 - beta_0) * r0 * pow(1.0 - pow(psi_N, alpha_0), gamma_0) / ri : 0.0)
      + (((psi_N > 0.0) & (psi_N < 1.0)) ? switch_lb * ri * mu * beta_0 / r0 * pow(1.0 - pow(psi_N, alpha_0), gamma_0) : 0.0);

  } else {
    // integrand of
    // int_{\Omega_p(\psi)} ( (  r S_{p'}'(\psi_N)
    //                         + S_{ff'}'(\psi_N) / (\mu r) ) d_{\psi} \psi_N(\psi, \phi) v
    //                       + d_{\psi} \bar{S}_{ff'}'(\psi, \phi) v ) dr dz
    // = 
    // int_{\Omega_p(\psi)} ( (  r S_{p'}'(\psi_N)
    //                         + S_{ff'}'(\psi_N) / (\mu r)
    //                         + A                          ) d_{\psi} \psi_N(\psi, \phi) v
    //                       + B phi_x v
    //                       + C phi_ma v) dr dz
    // 
    
    double coeff;

    double psi_N_multiplier = \
      + beta * ri * (model->S_prime_p_prime(psi_N)) * mu
      + gamma * (model->S_prime_ff_prime(psi_N)) / (ri)
      + switch_beta * alpha * alpha * pow(model->f_bar_prime(psi_N), 2.0) / (psi_bdp - psi_max) / (ri)
      + switch_beta * alpha * (f_x + alpha * (model->f_bar(psi_N))) * (model->f_bar_double_prime(psi_N)) / (psi_bdp - psi_max) / (ri)
      + switch_ff * alpha * (model->S_prime_ff_prime(psi_N)) / (ri)
      - (((psi_N > 0.0) & (psi_N < 1.0)) ? switch_lb * alpha * (1.0 - beta_0) * r0 * gamma_0 * pow(1.0 - pow(psi_N, alpha_0), gamma_0 - 1.0) * alpha_0 * pow(psi_N, alpha_0 - 1.0) / ri : 0.0)
      - (((psi_N > 0.0) & (psi_N < 1.0)) ? switch_lb * alpha * ri * mu * beta_0 / r0 * gamma_0 * pow(1.0 - pow(psi_N, alpha_0), gamma_0 - 1.0) * alpha_0 * pow(psi_N, alpha_0 - 1.0) : 0.0);

    double other =
      - switch_beta * alpha * (f_x + alpha * (model->f_bar(psi_N))) * (model->f_bar_prime(psi_N))
      / (psi_bdp - psi_max) / (psi_bdp - psi_max) / (ri);
    
    // double other = 0.0;
    if (option == 2) {
      // coefficient for phi in d_psi psi_N
      coeff = 1.0 / (psi_bdp - psi_max) * psi_N_multiplier
        - switch_taylor * alpha * alpha / (ri);
    } else if (option == 3) {
      // coefficient for phi_ma in d_psi psi_N
      coeff = - (1.0 - psi_N) / (psi_bdp - psi_max) * psi_N_multiplier
        - other;
    } else if (option == 4) {
      // coefficient for phi_x in d_psi psi_N
      coeff = - 1.0 * psi_N / (psi_bdp - psi_max) * psi_N_multiplier
        + other
        + switch_taylor * alpha * alpha / (ri);
    } 

    return
      + coeff
      + coeff_u2 * 2.0 * psi_val;
  }
}


map<int, vector<int>> compute_vertex_map(Mesh &mesh, int with_attrib) {
    /**
  * Build a vertex adjacency map for a mesh. For each vertex in the mesh, record
  * the set of neighboring vertices that share an element edge with it. Optionally,
  * the adjacency can be restricted to elements with a specific attribute.
  *
  * @param mesh        MFEM mesh containing the elements and vertices.
  * @param with_attrib Element attribute filter. If -1, all elements are used;
  *                    otherwise only elements whose attribute matches this
  *                    value contribute to the adjacency map.
  *
  * @return A map from vertex index -> vector of neighboring vertex indices.
  */

  // Initialize dictionary for adjacent nodes
  map<int, vector<int>> vertex_map;

  for (int i = 0; i < mesh.GetNE(); i++) {

    // Get element node indices, number of edges in the element, and element attributes
    const int *v = mesh.GetElement(i)->GetVertices();
    const int ne = mesh.GetElement(i)->GetNEdges();
    const int attrib = mesh.GetElement(i)->GetAttribute();

    // with_attrib = -1: accept all elements. Otherwise, only accept elements that match with_attrib input
    if ((with_attrib == -1) || (attrib == with_attrib)) {

      // For each edge in an element, get the nodes that form the edge and store them in vertex_map
      for (int j = 0; j < ne; j++) {
        const int *e = mesh.GetElement(i)->GetEdgeVertices(j);

        vertex_map[v[e[0]]].push_back(v[e[1]]);
        vertex_map[v[e[1]]].push_back(v[e[0]]);
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


void compute_plasma_points(
  GridFunction *z,
  const Mesh &mesh,
  const map<int,
  vector<int>> &vertex_map,
  set<int> &plasma_inds,
  int &ind_min,
  int &ind_max,
  double &min_val,
  double &max_val,
  int iprint
) {

   // mag ax point: global minimum in z
   // saddle point: closest saddle point to mag ax point, otherwise maximum on limiter boundary
   // keep track of elements inside of plasma region
  
   Vector nval;
   z->GetNodalValues(nval);

   // Initialize global extrema trackers
   min_val = + numeric_limits<double>::infinity();
   max_val = - numeric_limits<double>::infinity();
   ind_min = 0;
   ind_max = 0;

   // Candidate saddle points (X-points)
   vector<int> candidate_x_points;
     
   int count = 0;

   //////////////////////////////////////////////////////////////////////////////////////////////////
   // DEBUGGING
   int min_deg = 1e9, max_deg = 0;
   std::vector<int> degree_count;
   for (auto &p : vertex_map) {
     int deg = (int)p.second.size();
     min_deg = std::min(min_deg, deg);
     max_deg = std::max(max_deg, deg);
    if (deg >= degree_count.size())
        degree_count.resize(deg + 1);
    degree_count[deg]++;
   }
   std::cout << "vertex_map size=" << vertex_map.size()
             << " min_deg=" << min_deg
             << " max_deg=" << max_deg << "\n";
   for (int d = 0; d < degree_count.size(); ++d) {
       if (degree_count[d] > 0)
           std::cout << "degree " << d << " : " << degree_count[d] << " nodes\n";
   }
   //////////////////////////////////////////////////////////////////////////////////////////////////

   // DEBUGGING: find max number of sign changes
   int max_sc_seen = 0;

   // Loop over all vertices to determine: global minimum, global maximum, candidate saddle points
   for(int iv = 0; iv < mesh.GetNV(); ++iv) {

     // Get neighbors of vertex iv from adjacency map
     vector<int> adjacent;
     try {
       adjacent = vertex_map.at(iv);
     } catch (...) {
       continue;
     }

     // Update global minimum and maximum values and indices of z
     if (nval[iv] < min_val) {
       min_val = nval[iv];
       ind_min = iv;
     }
     if (nval[iv] > max_val) {
       max_val = nval[iv];
       ind_max = iv;
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

     // DEBUGGING: find max number of sign changes
     max_sc_seen = std::max(max_sc_seen, sign_changes);

     // If 4 or more sign changes, save node iv as a saddle point.
     if (sign_changes >= 4) {
       if (iprint) {
         printf("Found saddle at (%9.6f, %9.6f), val=%9.6f\n", x0[0], x0[1], nval[iv]);
       }

       // DEBUGGING
       cout << "Found saddle at (" << x0[0] << ", " << x0[1] << "), val=" << nval[iv] << endl;

       candidate_x_points.push_back(iv);
       ++count;
     } 
   }

   // Determine which saddle point is the X-point.
   // X-point is the saddle point with the minimum value of z. If no saddle points were found,
   // X-point is the max value of z.
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
   
   cout << "total saddles found: " << count << endl;  // <-- TODO: useful to un-comment out

   // DEBUGGING: find max number of sign changes
   std::cout << "max sign_changes observed = " << max_sc_seen << "\n";

   if (iprint) {
     printf("  min of %9.6f at (%9.6f, %9.6f), ind %d\n", min_val, x_min[0], x_min[1], ind_min);
     printf("  max of %9.6f at (%9.6f, %9.6f), ind %d\n", max_val, x_max[0], x_max[1], ind_max);
     printf("x_val of %9.6f at (%9.6f, %9.6f), ind %d\n", x_val, x_x[0], x_x[1], ind_x);
   }

   // DAS: we need to return the x_val, not the max_val.
   // TODO, refactor to make less confusing...
   max_val = x_val;  // max_val used to be the maximum z value, now it is the X-point value
   ind_max = ind_x;  // ind_max used to be the node index with maximum z, now it is the index of the X-point node

   // ---------------------------------------------------------------------------
   // Flood-fill from magnetic axis to mark plasma region using
   // breadth-first search starting from the minimum vertex. A vertex is
   // considered inside the plasma if its value lies between min_val and x_val.
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
         if ((val >= min_val) && (val <= x_val)) {
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
