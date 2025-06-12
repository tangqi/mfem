// Compile with: make field-interp2
//
// Sample runs:
//   field-interp2
//   field-interp2 -m1 ../tds-gs/meshes/mesh_refine.mesh -s1 ../tds-gs/gf/final_model2_pc5_cyc1_it5.gf -m2 ../meshing/square01.mesh
//   field-interp2 -m1 ../tds-gs/meshes/mesh_refine.mesh -s1 ../tds-gs/gf/final_model2_pc5_cyc1_it5.gf -m2 ../tds-gs/meshes/mesh_taylor1_trimmer.mesh

#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

// Scalar function to project
double scalar_func(const Vector &x)
{
   const int dim = x.Size();
   double res = 0.0;
   for (int d = 0; d < dim; d++) { res += x(d) * x(d); }
   return res;
}

int main (int argc, char *argv[])
{
   // Set the method's default parameters.
   const char *src_mesh_file = "../meshing/square01.mesh";
   const char *tar_mesh_file = "../../data/inline-tri.mesh";
   const char *src_sltn_file = "must_be_provided_by_the_user.gf";
   int src_fieldtype   = 0;
   int src_gf_ordering = 0;
   int ref_levels      = 0;
   int fieldtype       = 0;
   int order           = 1;
   bool visualization  = true;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&src_mesh_file, "-m1", "--mesh1",
                  "Mesh file for the starting solution.");
   args.AddOption(&tar_mesh_file, "-m2", "--mesh2",
                  "Mesh file for interpolation.");
   args.AddOption(&src_sltn_file, "-s1", "--solution1",
                  "(optional) GridFunction file compatible with src_mesh_file."
                  "Set src_fieldtype to -1 if this option is used.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of refinements of the interpolation mesh.");
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

   // If a gridfunction is specified, set src_fieldtype to -1
   if (strcmp(src_sltn_file, "must_be_provided_by_the_user.gf") != 0)
   {
      src_fieldtype = -1;
   }

   // Input meshes.
   Mesh mesh_1(src_mesh_file, 1, 1, false);
   Mesh mesh_2(tar_mesh_file, 1, 1, false);
   const int dim = mesh_1.Dimension();
   MFEM_ASSERT(dim == mesh_2.Dimension(), "Source and target meshes "
               "must be in the same dimension.");
   MFEM_VERIFY(dim > 1, "GSLIB requires a 2D or a 3D mesh" );

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh_2.UniformRefinement();
   }

   if (mesh_1.GetNodes() == NULL) { mesh_1.SetCurvature(1); }
   if (mesh_2.GetNodes() == NULL) { mesh_2.SetCurvature(1); }
   const int mesh_poly_deg =
      mesh_2.GetNodes()->FESpace()->GetElementOrder(0);
   cout << "Source mesh curvature: "
        << mesh_1.GetNodes()->OwnFEC()->Name() << endl
        << "Target mesh curvature: "
        << mesh_2.GetNodes()->OwnFEC()->Name() << endl;

   int src_vdim = 1;
   FiniteElementCollection *src_fec = NULL;
   FiniteElementSpace *src_fes = NULL;
   GridFunction *func_source = NULL;
   if (src_fieldtype < 0) // use src_sltn_file
   {
      ifstream mat_stream_1(src_sltn_file);
      func_source = new GridFunction(&mesh_1, mat_stream_1);
      src_vdim = func_source->FESpace()->GetVDim();
      src_fes = func_source->FESpace();
   }
   else if (src_fieldtype == 0)
   {
      src_fec = new H1_FECollection(order, dim);
   }
   else
   {
      MFEM_ABORT("Invalid FECollection type.");
   }

   if (src_fieldtype > -1)
   {
      src_fes = new FiniteElementSpace(&mesh_1, src_fec, 1, src_gf_ordering);
      func_source = new GridFunction(src_fes);
      // Project the grid function using VectorFunctionCoefficient.
      FunctionCoefficient F(scalar_func);
      func_source->ProjectCoefficient(F);
   }

   // Display the starting mesh and the field.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sout1;
      sout1.open(vishost, visport);
      if (!sout1)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
      }
      else
      {
         sout1.precision(8);
         sout1 << "solution\n" << mesh_1 << *func_source
               << "window_title 'Source mesh and solution'"
               << "window_geometry 0 0 600 600";
         if (dim == 2) { sout1 << "keys RmjAc"; }
         if (dim == 3) { sout1 << "keys mA\n"; }
         sout1 << flush;
      }
   }

   const Geometry::Type gt = mesh_2.GetNodalFESpace()->GetFE(0)->GetGeomType();
   MFEM_VERIFY(gt != Geometry::PRISM, "Wedge elements are not currently "
               "supported.");
   MFEM_VERIFY(mesh_2.GetNumGeometries(mesh_2.Dimension()) == 1, "Mixed meshes"
               "are not currently supported.");

   // Ensure the source grid function can be transferred using GSLIB-FindPoints.
   const FiniteElementCollection *fec_in = func_source->FESpace()->FEColl();
   std::cout << "Source FE collection: " << fec_in->Name() << std::endl;

   if (src_fieldtype < 0)
   {
      const H1_FECollection *fec_h1 = dynamic_cast<const H1_FECollection *>(fec_in);
      src_fieldtype = 0;
   }

   // Setup the FiniteElementSpace and GridFunction on the target mesh.
   FiniteElementCollection *tar_fec = NULL;
   FiniteElementSpace *tar_fes = NULL;

   int tar_vdim = src_vdim;
   if (fieldtype == 0)
   {
      tar_fec = new H1_FECollection(order, dim);
      tar_vdim = (src_fieldtype > 1) ? dim : src_vdim;
   }
   else
   {
      MFEM_ABORT("GridFunction type not supported.");
   }
   std::cout << "Target FE collection: " << tar_fec->Name() << std::endl;
   tar_fes = new FiniteElementSpace(&mesh_2, tar_fec, tar_vdim,
                                    src_fes->GetOrdering());
   GridFunction func_target(tar_fes);

   const int NE = mesh_2.GetNE(),
             nsp = tar_fes->GetFE(0)->GetNodes().GetNPoints(),
             tar_ncomp = func_target.VectorDim();

   // Generate list of points where the grid function will be evaluated.
   Vector vxyz;
   int point_ordering;
   if (fieldtype == 0 && order == mesh_poly_deg)
   {
      vxyz = *mesh_2.GetNodes();
      point_ordering = mesh_2.GetNodes()->FESpace()->GetOrdering();
   }
   else
   {
      MFEM_ABORT("GridFunction type not supported."); 
   }
   const int nodes_cnt = vxyz.Size() / dim;

   // Evaluate source grid function.
   Vector interp_vals(nodes_cnt*tar_ncomp);
   FindPointsGSLIB finder;
   finder.Setup(mesh_1);
   finder.Interpolate(vxyz, *func_source, interp_vals, point_ordering);

   // Project the interpolated values to the target FiniteElementSpace.
   func_target = interp_vals;

   // Visualize the transferred solution.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sout1;
      sout1.open(vishost, visport);
      if (!sout1)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
      }
      else
      {
         sout1.precision(8);
         sout1 << "solution\n" << mesh_2 << func_target
               << "window_title 'Target mesh and solution'"
               << "window_geometry 600 0 600 600";
         if (dim == 2) { sout1 << "keys RmjAc"; }
         if (dim == 3) { sout1 << "keys mA\n"; }
         sout1 << flush;
      }
   }

   // Output the target mesh with the interpolated solution.
   ostringstream rho_name;
   rho_name  << "interpolated.gf";
   ofstream rho_ofs(rho_name.str().c_str());
   rho_ofs.precision(8);
   func_target.Save(rho_ofs);
   rho_ofs.close();

   // Free the internal gslib data.
   finder.FreeData();

   // Delete remaining memory.
   if (func_source->OwnFEC())
   {
      delete func_source;
   }
   else
   {
      delete func_source;
      delete src_fes;
      delete src_fec;
   }
   delete tar_fes;
   delete tar_fec;

   return 0;
}
