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

   cout<<"mass matrix before slip diagonal"<<endl;
   mform.SpMat().Print();

   Vector diag;
   mform.SpMat().GetDiag(diag);
   diag.Print();
   //set the matrix being positive diagonal
   for (int i=0; i<mform.SpMat().Size(); i++)
   {
       if (diag(i)>0)
        {continue;}
       else
       {
         mform.SpMat()._Set_(i,i,-diag(i));
       }

   }

   //this matrix needs to be transformed!!
   MixedBilinearForm pform(&trace_fes, &vel_fes);
   pform.AddTraceFaceIntegrator(new NormalVectorTraceIntegrator());
   pform.Assemble();
   pform.Finalize();

   cout<<"mass matrix is"<<endl;
   mform.SpMat().Print();

   cout<<"projection matrix is"<<endl;
   pform.SpMat().Print();

   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);

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
