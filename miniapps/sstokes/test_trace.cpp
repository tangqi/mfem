#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <math.h>
#include "myIntegrator.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   //this define a single QUADRILATERAL element on the domain of [0, 2]x[0, 3]
   //See CrouzeixRaviartQuadFiniteElement::CalcShape for bases in the reference space
   //Mesh *mesh = new Mesh(1, 1, Element::QUADRILATERAL, 1, 2.0, 3.0);
   Mesh *mesh = new Mesh(1, 2, Element::QUADRILATERAL, 2, 2.0, 3.0);
   //Mesh *mesh = new Mesh(1, 1, Element::TRIANGLE, 1, 2.0, 3.0);
   int dim = 2;

   FiniteElementCollection *vel_fec, *trace_fec, *rt_fec;
   vel_fec = new CrouzeixRaviartFECollection();
   rt_fec = new RT_FECollection(0, dim);
   trace_fec = new DG_Interface_FECollection(0, dim);   //I guess this L2 basis is just constant of 1

   FiniteElementSpace vel_fes(mesh, vel_fec, dim);
   FiniteElementSpace trace_fes(mesh, trace_fec);
   FiniteElementSpace rt_fes(mesh, rt_fec);

   cout << "***********************************************************\n";
   cout << "Dofs in CR = " << vel_fes.GetVSize() << "\n";
   cout << "Dofs in RT = " << rt_fes.GetVSize() << "\n";
   cout << "Dofs in trace = " << trace_fes.GetVSize() << "\n";
   cout << "Eles in mesh = " << mesh->GetNE() << "\n";
   cout << "***********************************************************\n";

   MixedBilinearForm mform(&trace_fes, &rt_fes);
   mform.AddTraceFaceIntegrator(new NormalTraceIntegrator());
   mform.Assemble();
   mform.Finalize();

   cout<<"M matrix before abs diagonal"<<endl;
   mform.SpMat().Print();

   int msize=mform.SpMat().Size();
   Vector diag, rdiag(msize);
   mform.SpMat().GetDiag(diag);
   //set the matrix being positive diagonal
   for (int i=0; i<msize; i++)
   {
       rdiag(i)=1./fabs(diag(i));
       if (diag(i)>0)
       {
           continue;
       }
       else
       {
         mform.SpMat()._Set_(i,i,-diag(i));
       }

   }
   cout<<"diag of mass matrix is"<<endl;
   diag.Print();
   cout<<"reciprocal diag of mass matrix is"<<endl;
   rdiag.Print();

   MixedBilinearForm pform(&trace_fes, &vel_fes);
   pform.AddTraceFaceIntegrator(new NormalVectorTraceIntegrator());
   pform.Assemble();
   pform.Finalize();

   cout<<"**M matrix is"<<endl;
   mform.SpMat().Print();

   cout<<"**P matrix is"<<endl;
   pform.SpMat().Print();

   cout<<"**B=PM^{-1} is\n";
   SparseMatrix Bmat=pform.SpMat();
   Bmat.ScaleColumns(rdiag);
   Bmat.Print();

   BilinearForm massform(&rt_fes);
   massform.AddDomainIntegrator(new VectorFEMassIntegrator());
   massform.Assemble();
   massform.Finalize();
   SparseMatrix Mmat=massform.SpMat();
   cout<<"**Mass matrix for RT is\n";
   Mmat.Print();

   SparseMatrix *BM, *Bt, *BMBt;
   BM=Mult(Bmat, Mmat);
   Bt=Transpose(Bmat);
   BMBt=Mult(*BM, *Bt);
   cout<<"**BMBt (new mass matrix for CR) is\n";
   BMBt->Print();

   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);

   delete Bt;
   delete BM;
   delete BMBt;
   delete mesh;
   delete rt_fec;
   delete vel_fec;
   delete trace_fec;

   return 0;
}

/*
   const char *mesh_file = "../../data/inline-quad.mesh";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   */
