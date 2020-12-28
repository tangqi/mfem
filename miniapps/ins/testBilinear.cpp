#include "mfem.hpp"
#include "InsIntegrator.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double u_exact(const Vector &x)
{
   return x(0)*x(0)*x(0); // run with -o 2 or higher
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 3;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 4. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   FiniteElementSpace *fespace  = new FiniteElementSpace(mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // Initial function
   GridFunction u(fespace);
   FunctionCoefficient u_func(u_exact);
   u.ProjectCoefficient(u_func);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << u
               << "plot_caption 'Initial Function (u)'" << flush;
   }

   // Mass and stiffness matrices
   BilinearForm *M = new BilinearForm(fespace);
   M->AddDomainIntegrator(new MassIntegrator);           //  M matrix
   M->Assemble();
   BilinearForm *K = new BilinearForm(fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator);      //  K matrix
   K->AddBdrFaceIntegrator(new BoundaryGradIntegrator);  // -B matrix
   K->Assemble();

   // z = - ( K - B ) u
   Vector z(u.Size());
   K->Mult(u, z);
   z.Neg();

   // J = M^{-1}(- K u + B u)
   GridFunction J(fespace);
   GSSmoother GS(M->SpMat());
   PCG(M->SpMat(), GS, z, J, 1, 200, 1e-18, 0.0);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << J
               << "plot_caption 'Projected Laplacian (J)'" << flush;
   }

   // Free the used memory.
   delete K;
   delete M;
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}
