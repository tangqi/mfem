#include <fstream>
#include <sstream>
#include <ostream>
#include <string>
#include <vector>
#include <memory>

#include <mpi.h>

#include "elag.hpp"
#include "utilities/MPIDataTypes.hpp"

using namespace mfem;
using namespace parelag;
using namespace std;

void solfunc(const Vector &, Vector &);
void rhsfunc(const Vector &, Vector &);

int main(int argc, char *argv[])
{
   // Initialize MPI.
   mpi_session session(argc, argv);
   MPI_Comm comm = MPI_COMM_WORLD;
   int num_ranks, myid;
   MPI_Comm_size(comm, &num_ranks);
   MPI_Comm_rank(comm, &myid);

   if (!myid)
      cout << "-- This is an example of using a geometric-like multilevel hierarchy, constructed by ParElag,\n"
              "-- to solve a finite element H(curl) form.\n\n";

   // Get basic parameters from command line.
   const char *xml_file_c = "";
   OptionsParser args(argc, argv);
   args.AddOption(&xml_file_c, "-f", "--xml-file", "XML parameter list (an XML file with detailed parameters).");
   args.Parse();
   if (!args.Good())
   {
      if (!myid)
         args.PrintUsage(cout);
      return EXIT_FAILURE;
   }
   args.PrintOptions(cout);
   string xml_file(xml_file_c);

   // Read and parse the detailed parameter list from file.
   unique_ptr<ParameterList> master_list;
   ifstream xml_in(xml_file);
   if (!xml_in.good())
   {
      if (!myid)
         cerr << "ERROR: Cannot read from input file: " << xml_file << ".\n";
      return EXIT_FAILURE;
   }
   SimpleXMLParameterListReader reader;
   master_list = reader.GetParameterList(xml_in);
   xml_in.close();

   // General parameters for the problem.
   ParameterList& prob_list = master_list->Sublist("Problem parameters", true);

   // The file from which to read the mesh.
   const string meshfile = prob_list.Get("Mesh file", "");

   // The number of times to refine in serial.
   // Negative means refine until mesh is big enough to distribute, i.e.,
   // until the number of elements is 6 times the number of processes.
   int ser_ref_levels = prob_list.Get("Serial refinement levels", -1);

   // The number of times to refine in parallel. This determines the
   // number of levels in the AMGe hierarchy.
   const int par_ref_levels = prob_list.Get("Parallel refinement levels", 2);

   // The order of the finite elements on the finest level.
   const int feorder = prob_list.Get("Finite element order", 0);

   // The order of the polynomials to include in the coarse spaces
   // (after interpolating them onto the fine space).
   const int upscalingOrder = prob_list.Get("Upscaling order", 0);

   // The list of solvers to invoke.
   auto list_of_solvers = prob_list.Get<list<string>>("List of linear solvers");

   ConstantCoefficient alpha(1.0);
   ConstantCoefficient beta(1.0);

   ostringstream mesh_msg;
   if (!myid)
   {
      mesh_msg << '\n' << string(50, '*') << '\n'
               << "*                     Mesh: " << meshfile << "\n*\n"
               << "*                 FE order: " << feorder << '\n'
               << "*          Upscaling order: " << upscalingOrder << "\n*\n";
   }

   // Read the (serial) mesh from the given mesh file and uniformly refine it.
   shared_ptr<ParMesh> pmesh;
   {
      if (!myid)
         cout << "Reading and refining serial mesh...\n";

      ifstream imesh(meshfile);
      if (!imesh)
      {
         if (!myid)
            cerr << "ERROR: Cannot open mesh file: " << meshfile << ".\n";
         return EXIT_FAILURE;
      }

      auto mesh = make_unique<Mesh>(imesh, true, true);
      imesh.close();

      for (int l = 0; l < ser_ref_levels; ++l)
         mesh->UniformRefinement();

      if (ser_ref_levels < 0)
      {
         ser_ref_levels = 0;
         for (; mesh->GetNE() < 6 * num_ranks; ++ser_ref_levels)
            mesh->UniformRefinement();
      }

      if (!myid)
      {
         cout << "Times refined mesh in serial: " << ser_ref_levels << ".\n";
         cout << "Building and refining parallel mesh...\n";
         mesh_msg << "*    Serial refinements: " << ser_ref_levels << '\n'
                  << "*      Coarse mesh size: " << mesh->GetNE() << "\n*\n";
      }

      pmesh = make_shared<ParMesh>(comm, *mesh);
   }

   // Mark all boundary attributes as essential.
   vector<Array<int>> ess_attr(1);
   ess_attr[0].SetSize(pmesh->bdr_attributes.Max());
   ess_attr[0] = 1;

   // Refine the mesh in parallel.
   const int nDimensions = pmesh->Dimension();

   // This is mainly because AMS (at least the way ParElag uses them)
   // is bound to be used in 3D. Note that, for the purpose of demonstration,
   // some of the code below is still constructed in a way that is applicable in
   // 2D as well, taking into account that case as well. Also, in 2D, ParElag
   // defaults to H(div) interpretation of form 1.
   MFEM_ASSERT(nDimensions == 3, "Only 3D problems are supported.");

   const int nLevels = par_ref_levels + 1;
   vector<int> level_nElements(nLevels);
   for (int l = 0; l < par_ref_levels; ++l)
   {
      level_nElements[par_ref_levels - l] = pmesh->GetNE();
      pmesh->UniformRefinement();
   }
   level_nElements[0] = pmesh->GetNE();

   if (!myid)
      cout << "Times refined mesh in parallel: " << par_ref_levels << ".\n";

   {
      size_t local_num_elmts = pmesh->GetNE(), global_num_elmts;
      MPI_Reduce(&local_num_elmts, &global_num_elmts, 1, GetMPIType<size_t>(0),
                 MPI_SUM, 0, comm);
      if (!myid)
      {
         mesh_msg << "*  Parallel refinements: " << par_ref_levels << '\n'
                  << "*        Fine mesh size: " << global_num_elmts << '\n'
                  << "*          Total levels: " << nLevels << '\n'
                  << string(50, '*') << '\n';
      }
   }

   if (!myid)
      cout << mesh_msg.str();
   pmesh->ReorientTetMesh();

   // Obtain the hierarchy of agglomerate topologies.
   if (!myid)
      cout << "Agglomerating topology for " << nLevels << " levels...\n";

   constexpr auto AT_elem = AgglomeratedTopology::ELEMENT;
   // This partitioner simply geometrically coarsens the mesh by recovering the geometric
   // coarse elements as agglomerate elements. That is, it reverts the MFEM uniform refinement
   // procedure to provide agglomeration.
   MFEMRefinedMeshPartitioner partitioner(nDimensions);
   vector<shared_ptr<AgglomeratedTopology>> topology(nLevels);

   topology[0] = make_shared<AgglomeratedTopology>(pmesh, nDimensions);
   for(int l = 0; l < nLevels - 1; ++l)
   {
      Array<int> partitioning(topology[l]->GetNumberLocalEntities(AT_elem));
      partitioner.Partition(topology[l]->GetNumberLocalEntities(AT_elem),
                            level_nElements[l + 1], partitioning);
      topology[l + 1] = topology[l]->CoarsenLocalPartitioning(partitioning, false, false);
   }

   // Construct the hierarchy of spaces, thus forming a hierarchy of (partial) de Rham sequences.
   if (!myid)
      cout << "Building the fine-level de Rham sequence...\n";

   vector<shared_ptr<DeRhamSequence>> sequence(topology.size());

   const int nForms = nDimensions + 1;
   const int jform = 1; // This is the H(curl) form.
   if(nDimensions == 3)
      sequence[0] = make_shared<DeRhamSequence3D_FE>(topology[0], pmesh.get(), feorder);
   else
      MFEM_ABORT("No H(curl) 2D interpretation of form 1 is implemented.");

   // To build H(curl) (form 1 in 3D), it is needed to obtain all forms and spaces with larger
   // indices. To use the so called "Hiptmair smoothers", a one form lower is needed
   // (H1, form 0). Anyway, to use AMS all forms and spaces to H1 (0 form) are needed.
   // Therefore, the entire de Rham complex is constructed.
   sequence[0]->SetjformStart(0);

   DeRhamSequenceFE *DRSequence_FE = sequence[0]->FemSequence();
   MFEM_ASSERT(DRSequence_FE, "Failed to obtain the fine-level de Rham sequence.");
   DRSequence_FE->ReplaceMassIntegrator(AT_elem, jform, make_unique<VectorFEMassIntegrator>(beta), false);
   if(nDimensions == 3)
      DRSequence_FE->ReplaceMassIntegrator(AT_elem, jform + 1, make_unique<VectorFEMassIntegrator>(alpha), true);
   else
      DRSequence_FE->ReplaceMassIntegrator(AT_elem, jform + 1, make_unique<MassIntegrator>(alpha), true);

   if (!myid)
      cout << "Interpolating and setting polynomial targets...\n";

   vector<unique_ptr<MultiVector>> targets(nForms);

   Array<Coefficient *> L2coeff;
   Array<VectorCoefficient *> Hcurlcoeff;
   Array<Coefficient *> H1coeff;
   fillVectorCoefficientArray(nDimensions, upscalingOrder, Hcurlcoeff);
   fillCoefficientArray(nDimensions, upscalingOrder, L2coeff);
   fillCoefficientArray(nDimensions, upscalingOrder + 1, H1coeff);

   targets[0] = DRSequence_FE->InterpolateScalarTargets(0, H1coeff);
   if(nDimensions == 3)
      targets[jform - 1] = DRSequence_FE->InterpolateVectorTargets(jform - 1, Hcurlcoeff);
   targets[jform] = DRSequence_FE->InterpolateVectorTargets(jform, Hcurlcoeff);
   targets[jform + 1] = DRSequence_FE->InterpolateScalarTargets(jform + 1, L2coeff);

   freeCoeffArray(L2coeff);
   freeCoeffArray(Hcurlcoeff);
   freeCoeffArray(H1coeff);

   Array<MultiVector *> targets_in(targets.size());
   for (int i = 0; i < targets_in.Size(); ++i)
      targets_in[i] = targets[i].get();
   sequence[0]->SetTargets(targets_in);

   if (!myid)
      cout << "Building the coarse-level de Rham sequences...\n";

   for(int l = 0; l < nLevels - 1; ++l)
   {
      const double tolSVD = 1e-9;
      sequence[l]->SetSVDTol(tolSVD);
      sequence[l + 1] = sequence[l]->Coarsen();
   }

   if (!myid)
      cout << "Assembling the fine-level system...\n";

   VectorFunctionCoefficient rhscoeff(nDimensions, rhsfunc);
   VectorFunctionCoefficient solcoeff(nDimensions, solfunc);

   // Take the vector FE space and construct a RHS linear form on it.
   // Then, move the linear form to a vector. This is local, i.e. on all known dofs for the process.
   FiniteElementSpace *fespace = DRSequence_FE->GetFeSpace(jform);
   auto rhsform = make_unique<LinearForm>(fespace);
   rhsform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(rhscoeff));
   rhsform->Assemble();
   unique_ptr<Vector> rhs = move(rhsform);

   // Obtain the boundary data. This is local, i.e. on all known dofs for the process.
   auto solgf = make_unique<GridFunction>(fespace);
   solgf->ProjectCoefficient(solcoeff);
   unique_ptr<Vector> sol = move(solgf);

   // Create the parallel linear system.
   const SharingMap& hcurl_dofTrueDof = sequence[0]->GetDofHandler(jform)->GetDofTrueDof();

   // System RHS, B. It is defined on the true dofs owned by the process.
   Vector B(hcurl_dofTrueDof.GetTrueLocalSize());

   // System matrix, A.
   shared_ptr<HypreParMatrix> A;
   {
      // Get the mass and derivative operators.
      // M1 represents the form (beta u, v) on H(curl) vector fields.
      // M2 represents the form (alpha u, v) on H(div) vector fields, in 3D.
      // D1 is the curl operator from H(curl) vector fields to H(div) vector fields, in 3D.
      // In 2D, instead of considering H(div) vector fields, L2 scalar fields are to be considered.
      // Thus, D1^T * M2 * D1 represents the form (alpha curl u, curl v) on H(curl) vector fields.
      auto M1 = sequence[0]->ComputeMassOperator(jform),
           M2 = sequence[0]->ComputeMassOperator(jform + 1);
      auto D1 = sequence[0]->GetDerivativeOperator(jform);

      // spA = D1^T * M2 * D1 + M1 represents the form (alpha curl u, curl v) + (beta u, v)
      // on H(curl) vector fields. This is local, i.e. on all known dofs for the process.
      auto spA = ToUnique(Add(*M1, *ToUnique(RAP(*D1, *M2, *D1))));

      // Eliminate the boundary conditions
      Array<int> marker(spA->Height());
      marker = 0;
      sequence[0]->GetDofHandler(jform)->MarkDofsOnSelectedBndr(ess_attr[0], marker);

      for(int i = 0; i < spA->Height(); ++i)
         if(marker[i])
            spA->EliminateRowCol(i, sol->Elem(i), *rhs);

      A = Assemble(hcurl_dofTrueDof, *spA, hcurl_dofTrueDof);
      hcurl_dofTrueDof.Assemble(*rhs, B);
   }
   if (!myid)
   {
      cout << "A size: " << A->GetGlobalNumRows() << 'x' << A->GetGlobalNumCols() << '\n'
           << " A NNZ: " << A->NNZ() << '\n';
   }
   MFEM_ASSERT(B.Size() == A->Height(), "Matrix and vector size are incompatible.");

   // Perform the solves.
   if (!myid)
      cout << "\nRunning fine-level solvers...\n\n";

   // Create the solver library.
   auto lib = SolverLibrary::CreateLibrary(master_list->Sublist("Preconditioner Library"));

   // Loop through the solvers.
   for (const auto& solver_name : list_of_solvers)
   {
      // Get the solver factory.
      auto solver_factory = lib->GetSolverFactory(solver_name);
      auto solver_state = solver_factory->GetDefaultState();
      solver_state->SetDeRhamSequence(sequence[0]);
      solver_state->SetBoundaryLabels(ess_attr);
      solver_state->SetForms({jform});

      // Build the silver.
      if (!myid)
         cout << "Building solver \"" << solver_name << "\"...\n";
      unique_ptr<Solver> solver = solver_factory->BuildSolver(A, *solver_state);

      // Run the solver.
      if (!myid)
         cout << "Solving system with \"" << solver_name << "\"...\n";

      // Note that X is on true dofs owned by the process, while x is on local dofs that are known to the process.
      Vector X(A->Width()), x(sequence[0]->GetNumberOfDofs(jform));
      X=0.0;

      {
         Vector tmp(A->Height());
         A->Mult(X, tmp);
         tmp *= -1.0;
         tmp += B;

         double local_norm = tmp.Norml2() * tmp.Norml2();
         double global_norm;
         MPI_Reduce(&local_norm, &global_norm, 1, GetMPIType(local_norm), MPI_SUM, 0, comm);
         if (!myid)
            cout << "Initial residual l2 norm: " << sqrt(global_norm) << '\n';
      }

      // Perform the solve.
      solver->Mult(B, X);

      {
         Vector tmp(A->Height());
         A->Mult(X, tmp);
         tmp *= -1.0;
         tmp += B;

         double local_norm = tmp.Norml2() * tmp.Norml2();
         double global_norm;
         MPI_Reduce(&local_norm, &global_norm, 1, GetMPIType(local_norm), MPI_SUM, 0, comm);
         if (!myid)
            cout << "Final residual l2 norm: " << sqrt(global_norm) << '\n';
      }

      if (!myid)
         cout << "Solver \"" << solver_name << "\" finished.\n";

      // Visualize the solution.
      hcurl_dofTrueDof.Distribute(X, x);
      MultiVector tmp(x.GetData(), 1, x.Size());
      sequence[0]->show(jform, tmp);
   }

   if (!myid)
      cout << "Finished.\n";

   return EXIT_SUCCESS;
}

// Solution vector field. Used for setting boundary conditions.
void solfunc(const Vector &p, Vector &F)
{
   int dim = p.Size();

   F(0) = 0.0;
   F(1) = 0.0;
   if (dim == 3)
   {
      F(2) = 0.0;
   }
}

// The right hand side.
void rhsfunc(const Vector &p, Vector &f)
{
   int dim = p.Size();

   f(0) = 0.0;
   f(1) = 0.0;
   if (dim == 3)
   {
      f(2) = 1.0;
   }
}

