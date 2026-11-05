// Samples a saved DG Grad-Shafranov solution and the analytic Solov'ev solution
// on a per-element refinement of the mesh, for 2D error plots.
//
// Reads the mesh and grid function written by DG_GS_fixed_boundary_linear, either
// as one file (-one) or as per-rank pieces <base>.0 .. <base>.N-1 (-sep, use -np N).
// Points are kept per element so DG discontinuities are preserved.
//
// Binary output layout (native endianness):
//   int64 npts, ntri, nseg
//   float64 R[npts], Z[npts], psi_h[npts], psi_exact[npts]
//   int32   tri[ntri*3]      (point indices of each sub-triangle)
//   int32   rank[ntri]       (piece/rank that owns each sub-triangle)
//   float64 seg[nseg*4]      (R0, Z0, R1, Z1 of each piece-boundary edge)

#include "mfem.hpp"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;
using namespace mfem;

// Exact solution from Section 7.1.1 of the DPG paper; must match the
// coefficients and formula in DG_GS_fixed_boundary_linear.cpp.
static real_t psi_exact(real_t R, real_t Z)
{
   const real_t d1 =  0.075385029660066;
   const real_t d2 = -0.206294962187880;
   const real_t d3 = -0.031433707280533;

   const real_t R2 = R*R;
   const real_t R4 = R2*R2;

   return R4/8.0 + d1 + d2*R2 + d3*(R4 - 4.0*R2*Z*Z);
}

int main(int argc, char *argv[])
{
   const char *mesh_base = "";
   const char *gf_base = "";
   const char *out_file = "samples.bin";
   int num_pieces = 0;
   int ref = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_base, "-m", "--mesh", "Mesh file (or base name with -np).");
   args.AddOption(&gf_base, "-g", "--gridfunction",
                  "Grid function file (or base name with -np).");
   args.AddOption(&num_pieces, "-np", "--num-pieces",
                  "Number of per-rank pieces <base>.0..N-1; 0 = single file.");
   args.AddOption(&ref, "-r", "--refine",
                  "Per-element subdivision; <= 0 picks max(order, enough for ~1e5 triangles).");
   args.AddOption(&out_file, "-o", "--output", "Binary output file.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }

   // Load the mesh and grid function of every piece
   const int npieces = max(num_pieces, 1);
   vector<unique_ptr<Mesh>> meshes;
   vector<unique_ptr<GridFunction>> gfs;
   long long total_elems = 0;
   for (int k = 0; k < npieces; k++)
   {
      string mname = mesh_base, gname = gf_base;
      if (num_pieces > 0)
      {
         mname += "." + to_string(k);
         gname += "." + to_string(k);
      }
      // Load as GLVis does (no refinement marking, no orientation fix-up): either
      // can reorder element vertices and break the saved DOF ordering.
      meshes.emplace_back(new Mesh(mname, 1, 0, false));
      ifstream gf_ifs(gname);
      MFEM_VERIFY(gf_ifs.good(), "Could not open grid function: " << gname);
      gfs.emplace_back(new GridFunction(meshes.back().get(), gf_ifs));
      total_elems += meshes.back()->GetNE();
   }

   // Pick the subdivision: at least the polynomial order, refined further on
   // coarse meshes so the plot still resolves the high-order shape.
   const int order = gfs[0]->FESpace()->GetMaxElementOrder();
   if (ref <= 0)
   {
      ref = max(order, 1);
      while (total_elems*ref*ref < 100000) { ref++; }
   }

   vector<double> R, Z, ph, pe, seg;
   vector<int32_t> tri, rank;
   double max_err = 0.0;

   for (int k = 0; k < npieces; k++)
   {
      Mesh &mesh = *meshes[k];
      GridFunction &gf = *gfs[k];

      for (int e = 0; e < mesh.GetNE(); e++)
      {
         RefinedGeometry *RG = GlobGeometryRefiner.Refine(mesh.GetElementBaseGeometry(e), ref);
         ElementTransformation *T = mesh.GetElementTransformation(e);
         const int offset = (int) R.size();

         Vector x(2);
         for (int i = 0; i < RG->RefPts.GetNPoints(); i++)
         {
            const IntegrationPoint &ip = RG->RefPts.IntPoint(i);
            T->SetIntPoint(&ip);
            T->Transform(ip, x);
            const double v = gf.GetValue(e, ip);
            const double ve = psi_exact(x(0), x(1));
            R.push_back(x(0));
            Z.push_back(x(1));
            ph.push_back(v);
            pe.push_back(ve);
            max_err = max(max_err, fabs(v - ve));
         }

         for (int j = 0; j < RG->RefGeoms.Size(); j += 3)
         {
            for (int l = 0; l < 3; l++) { tri.push_back(offset + RG->RefGeoms[j+l]); }
            rank.push_back(k);
         }
      }

      // Edges with only one element in this piece outline the piece (domain
      // boundary plus, for per-rank pieces, the partition interfaces)
      for (int f = 0; f < mesh.GetNumFaces(); f++)
      {
         int e1, e2;
         mesh.GetFaceElements(f, &e1, &e2);
         if (e2 >= 0) { continue; }
         Array<int> v;
         mesh.GetFaceVertices(f, v);
         const real_t *a = mesh.GetVertex(v[0]);
         const real_t *b = mesh.GetVertex(v[1]);
         seg.insert(seg.end(), {a[0], a[1], b[0], b[1]});
      }
   }

   const int64_t npts = R.size(), ntri = rank.size(), nseg = seg.size()/4;
   ofstream ofs(out_file, ios::binary);
   MFEM_VERIFY(ofs.good(), "Could not open output file: " << out_file);
   ofs.write((const char*) &npts, sizeof(npts));
   ofs.write((const char*) &ntri, sizeof(ntri));
   ofs.write((const char*) &nseg, sizeof(nseg));
   for (const vector<double> *a : {&R, &Z, &ph, &pe})
   {
      ofs.write((const char*) a->data(), a->size()*sizeof(double));
   }
   ofs.write((const char*) tri.data(), tri.size()*sizeof(int32_t));
   ofs.write((const char*) rank.data(), rank.size()*sizeof(int32_t));
   ofs.write((const char*) seg.data(), seg.size()*sizeof(double));

   cout << "pieces " << npieces << "  elements " << total_elems << "  order " << order
        << "  ref " << ref << "  points " << npts << "  triangles " << ntri
        << "  max|psi_h - psi_exact| " << max_err << endl;

   return 0;
}
