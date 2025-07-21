#include "mfem.hpp"
#include <fstream>
using namespace mfem;
using namespace std;

void ExtractContourLine(const Mesh &mesh, const GridFunction &u, double level,
                        const std::string &filename)
{
    MFEM_VERIFY(mesh.Dimension() == 2, "Only 2D meshes are supported.");
    MFEM_VERIFY(u.FESpace()->GetVDim() == 1, "Only scalar fields are supported.");

    const FiniteElementSpace &fes = *u.FESpace();
    const GridFunction *nodes = mesh.GetNodes();
    MFEM_VERIFY(nodes, "Mesh must be in nodal form (use high-order mesh).");

    ofstream out(filename);

    const int nedges = mesh.GetNEdges();
    const int dim = 2;

    ofstream NewFile(filename);

    for (int e = 0; e < nedges; ++e)
    {
        // Get the vertex indices of this edge
        Array<int> ev;
        mesh.GetEdgeVertices(e, ev);

        // Get coordinates of endpoints
        const double *coords_i = mesh.GetVertex(ev[0]);
        const double *coords_j = mesh.GetVertex(ev[1]);

        Array<int> dofs_i, dofs_j;

        // Evaluate u at the vertices
        fes.GetVertexDofs(ev[0], dofs_i);
        fes.GetVertexDofs(ev[1], dofs_j);
        double ui = u(dofs_i[0]);
        double uj = u(dofs_j[0]);
        // Check if contour level crosses this edge
        if ((ui - level) * (uj - level) < 0.0)
        {
            cout<<ui<<" "<<uj<<" ";
            // Linear interpolation to find contour crossing
            double alpha = (level - ui) / (uj - ui);
            Vector pt(dim);
            for (int d = 0; d < dim; d++)
            {
                pt[d] = coords_i[d] + alpha * (coords_j[d] - coords_i[d]);
            }

            NewFile << pt[0] << " " << pt[1] << "\n";
        }
    }

    NewFile.close();
}


int main() {
    Mesh my_mesh("../tds-gs/meshes/mesh_refine.mesh");
    my_mesh.EnsureNodes();
    FiniteElementCollection *fec = new H1_FECollection(1, my_mesh.Dimension()); // P1 elements
    FiniteElementSpace fespace(&my_mesh, fec);

    // Create GridFunction
    ifstream ifs("../tds-gs/gf/final_model2_pc5_cyc1_it5.gf");
    GridFunction lgf(&my_mesh, ifs);

    // Now call your function
    ExtractContourLine(my_mesh, lgf, -8.602077472e-02, "contour_line.txt");

    delete fec;
    return 0;
}
