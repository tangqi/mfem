// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
void u_ex(const Vector & x, Vector & u);
double p_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
void velocity_function(const Vector &x, Vector &v);

int main(int argc, char *argv[])
{

    // Parse command-line options.
    const char *mesh_file = "../data/square.msh";
    int order = 2;
    const char *device_config = "cpu";

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree).");
    args.AddOption(&device_config, "-d", "--device",
                   "Device configuration string, see Device::Configure().");

    args.Parse();
    if (!args.Good())
    {
        //args.PrintUsage(cout);
        return 1;
    }
    //args.PrintOptions(cout);
    // Read in mesh from the given mesh file.
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();
    //mesh->UniformRefinement();



    Device device(device_config);
    //device.Print();

    // Define a finite element space on the mesh. Here we use the
    // Taylor Hood finite elements of the specified order.
    FiniteElementCollection *quad_coll(new H1_FECollection(order, dim));
    FiniteElementCollection *lin_coll(new H1_FECollection(order-1, dim));

    FiniteElementSpace *Xh_space = new FiniteElementSpace(mesh, quad_coll, dim);
    FiniteElementSpace *Ph_space = new FiniteElementSpace(mesh, lin_coll, dim - 1);


    int Xh_size = Xh_space->GetTrueVSize();
    int Ph_size = Ph_space->GetTrueVSize();


    //    Define the BlockStructure of the problem, i.e. define the array of
    //    offsets for each variable. The last component of the Array is the sum
    //    of the dimensions of each block.
    Array<int> block_offsets(3); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = Xh_space->GetVSize();
    block_offsets[2] = Ph_space->GetVSize();
    block_offsets.PartialSum();

    Array<int> ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 1;
    Array<int> ess_dof;
    Array<int> ess_tdof_list;
    Xh_space->GetEssentialVDofs(ess_bdr, ess_dof); // returns 0 and -1's
    Xh_space->GetEssentialTrueDofs(ess_bdr, ess_tdof_list); // returns actually location of DOFs


    // std::cout << "***********************************************************\n";
    // std::cout << "dim(Xh) = " << block_offsets[1] - block_offsets[0] << "\n";
    // std::cout << "dim(Ph) = " << block_offsets[2] - block_offsets[1] << "\n";
    // std::cout << "dim(Xh + Ph) = " << block_offsets.Last() << "\n";
    // std::cout << "***********************************************************\n";


    //Define the coefficients, analytical solution, and rhs of the PDE.
    VectorFunctionCoefficient fcoeff(dim, fFun);
    VectorFunctionCoefficient ucoeff(dim, u_ex);
    FunctionCoefficient pcoeff(p_ex);

    ///////////////////////////////////////////////
    //Solve Stokes to generate an initial condition
    ///////////////////////////////////////////////

    //    Allocate memory (x, rhs) for the analytical solution and the right hand
    //    side.  Define the GridFunction u,p for the finite element solution and
    //    linear forms fform for the right hand side.

    MemoryType mt = device.GetMemoryType();
    // BlockVector x(block_offsets, mt), rhs(block_offsets, mt); //Initialize block vectors, all blocks default to 0.

    // GridFunction x_bdr(Xh_space);
    // x_bdr.MakeRef(Xh_space, x.GetBlock(0), 0); // Make x_bdr point to the velocity block
    // x_bdr.ProjectCoefficient(ucoeff); //project the boundary condition


    // //Construct the righthand side of the system
    // LinearForm *fform(new LinearForm);
    // fform->Update(Xh_space, rhs.GetBlock(0), 0);
    // fform->AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));
    // fform->Assemble();
    // fform->SyncAliasMemory(rhs);

    // //  Assemble the finite element matrices for the Stokes operator 
    // BilinearForm *aVar(new BilinearForm(Xh_space));
    // MixedBilinearForm *bVar(new MixedBilinearForm(Xh_space, Ph_space));
    // BilinearForm *cVar(new BilinearForm(Ph_space));


    // aVar->AddDomainIntegrator(new VectorDiffusionIntegrator());
    // aVar->Assemble();
    // aVar->EliminateEssentialBC(ess_bdr, x_bdr, rhs.GetBlock(0));
    // aVar->Finalize();

    // ConstantCoefficient k(-1.0);
    // bVar->AddDomainIntegrator(new VectorDivergenceIntegrator(k));
    // bVar->Assemble();
    // bVar->EliminateTrialDofs(ess_bdr, x, rhs.GetBlock(1));
    // bVar->Finalize();


    // ConstantCoefficient pen(1e-16);
    // cVar->AddDomainIntegrator(new MassIntegrator(pen));
    // cVar->Assemble();
    // cVar->Finalize();

    // BlockMatrix stokesOp(block_offsets);



    // SparseMatrix &A(aVar->SpMat());
    // SparseMatrix &B(bVar->SpMat());
    // SparseMatrix &C(cVar->SpMat()); 
    // SparseMatrix *Bt = Transpose(B);

    // stokesOp.SetBlock(0,0, &A);
    // stokesOp.SetBlock(0,1, Bt);
    // stokesOp.SetBlock(1,0, &B);
    // stokesOp.SetBlock(1,1, &C);
    // SparseMatrix *sparse_stokes = stokesOp.CreateMonolithic(); //Converts BlockMatrix to SparseMatrix. Note this approach will not work in parallel.

    // //Zero out the last last row and column of the matrix and then sets the last entry in the matrix to 1 and the rhs to 0.
    // //This is equivalent to setting the pressure at that point equal to 0.
    // sparse_stokes->EliminateRowCol(block_offsets.Last()-1,0.0,rhs);

    // //  Solve the linear system A X = B.
    // UMFPackSolver umf_solver;
    // umf_solver.SetOperator(*sparse_stokes);
    // umf_solver.Mult(rhs, x);

    // Recover the solutions into finite element grid functions.
    
    // u.MakeRef(Xh_space, x.GetBlock(0), 0);
    // p.MakeRef(Ph_space, x.GetBlock(1), 0);

    ///////////////////////////////////////////////
    //Steady NSE solve
    ///////////////////////////////////////////////
    //Take the initial condtion to be the solution of the stokes problem

   BlockVector x_NSE(block_offsets, mt), rhs_NSE(block_offsets, mt); //Initialize block vectors, all blocks default to 0
   GridFunction u, p;
   u.MakeRef(Xh_space, x_NSE.GetBlock(0), 0);
   p.MakeRef(Ph_space, x_NSE.GetBlock(1), 0);


    GridFunction x_bdr0(Xh_space); //Grid function for the NSE solve. Since the initial condition satisfies the BC, each iteration of Picard/Newton should be 0
    //VectorFunctionCoefficient u0coeff(dim, u_0);
    x_bdr0.MakeRef(Xh_space, x_NSE.GetBlock(0), 0); // Make x_bdr point to the velocity block

    
    //VectorFunctionCoefficient u_test(dim, u_ex);
    
    u *= 0.0;
    p *= 0.0;
    UMFPackSolver umf_solver_NSE;
    BlockMatrix NSEOp(block_offsets);
    ConstantCoefficient k_NSE(-1.0);
    ConstantCoefficient pen_NSE(1e-16);
    double err_u, err_p;
    GridFunction u_test(Xh_space);
    VectorFunctionCoefficient u0(dim, u_ex);
    x_bdr0.ProjectCoefficient(u0); //project the boundary condition
    //u_test.ProjectCoefficient(velocity_function);
    //VectorFunctionCoefficient u_test1(dim,velocity_function);
    //SparseMatrix *Non;


    for(int ti = 0; ti < 1; ti++ )
    {
        rhs_NSE *= 0.0;


        //Construct the righthand side of the system
        LinearForm *fform_NSE(new LinearForm);
        fform_NSE->Update(Xh_space, rhs_NSE.GetBlock(0), 0);
        fform_NSE->AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));
        fform_NSE->Assemble();
        fform_NSE->SyncAliasMemory(rhs_NSE);

        BilinearForm *aVar_NSE(new BilinearForm(Xh_space));
        MixedBilinearForm *bVar_NSE(new MixedBilinearForm(Xh_space, Ph_space));
        BilinearForm *cVar_NSE(new BilinearForm(Ph_space));

        NonlinearForm *N_NSE(new NonlinearForm(Xh_space));
        N_NSE->AddDomainIntegrator(new ConvectiveVectorConvectionNLFIntegrator());
        




        
        SparseMatrix *Non = dynamic_cast<SparseMatrix*>(&N_NSE->GetGradient(u));
        Non->PrintMatlab();
        for (int i = 0; i < ess_tdof_list.Size(); i++)
            {
               Non->EliminateRowColDiag(ess_tdof_list[i], 0.0); //Should probably figure out how to do this with dpolicy call to be consistent with what is going on in getGradient approach
            }
       
       
        aVar_NSE->AddDomainIntegrator(new VectorDiffusionIntegrator()); //add diffusion term
        aVar_NSE->Assemble();
        aVar_NSE->EliminateEssentialBC(ess_bdr, x_bdr0, rhs_NSE.GetBlock(0));
        aVar_NSE->Finalize();

        bVar_NSE->AddDomainIntegrator(new VectorDivergenceIntegrator(k_NSE));
        bVar_NSE->Assemble();
        bVar_NSE->EliminateTrialDofs(ess_bdr, x_NSE, rhs_NSE.GetBlock(1));
        bVar_NSE->Finalize();

    
        //Penalty block, ideally want to set single point for pressure to 0 and not use this.
        cVar_NSE->AddDomainIntegrator(new MassIntegrator(pen_NSE));
        cVar_NSE->Assemble();
        cVar_NSE->Finalize();



        SparseMatrix &A_NSE(aVar_NSE->SpMat());
        SparseMatrix &B_NSE(bVar_NSE->SpMat());
        SparseMatrix &C_NSE(cVar_NSE->SpMat()); 
        SparseMatrix *Bt_NSE = Transpose(B_NSE);

        //Non->PrintMatlab();
        //A_NSE.Add(1.0,*Non);// += *Non;
        //A_NSE += *Non;
        //A_NSE.PrintMatlab();

        NSEOp.SetBlock(0,0, &A_NSE);
        NSEOp.SetBlock(0,1, Bt_NSE);
        NSEOp.SetBlock(1,0, &B_NSE);
        NSEOp.SetBlock(1,1, &C_NSE); 
        SparseMatrix *sparse_NSE = NSEOp.CreateMonolithic(); //Converts BlockMatrix to SparseMatrix. Note this approach will not work in parallel.

        //Zero out the last last row and column of the matrix and then sets the last entry in the matrix to 1 and the rhs to 0.
        //This is equivalent to setting the pressure at that point equal to 0.
        sparse_NSE->EliminateRowCol(block_offsets.Last()-1,0.0,rhs_NSE);


        //  Solve the linear system A X = B.
        umf_solver_NSE.SetOperator(*sparse_NSE);
        umf_solver_NSE.Mult(rhs_NSE, x_NSE);

        //Reference the new solution
        u.MakeRef(Xh_space, x_NSE.GetBlock(0), 0);
        p.MakeRef(Ph_space, x_NSE.GetBlock(1), 0);

        //Calculate the L2 error
        err_u  = u.ComputeL2Error(ucoeff);
        err_p  = p.ComputeL2Error(pcoeff);

        //std::cout << "|| u_h - u_ex ||  = " <<setprecision(10) << err_u  << "\n";
        //std::cout << "|| p_h - p_ex ||  = " <<setprecision(10) << err_p  << "\n";


        //std::cout << Non << std::endl;
        //std::cout << N_NSE << std::endl;

        //delete Non;
        delete fform_NSE;
        delete aVar_NSE;
        delete bVar_NSE;
        delete cVar_NSE; 
        delete N_NSE;
        delete Bt_NSE;
        delete sparse_NSE;
        
        //std::cout << Non << std::endl;

        
    }
    




    // Recover the solutions into finite element grid functions.


    //Plotting. This is pretty sloppy right now, fix later.
    ParaViewDataCollection *pdu = NULL;
    pdu = new ParaViewDataCollection("NSE_manufacture_velocity", mesh);
    pdu->SetPrefixPath("ParaView");
    pdu->RegisterField("velocity", &u);
    pdu->SetLevelsOfDetail(order);
    pdu->SetDataFormat(VTKFormat::BINARY);
    pdu->SetHighOrderOutput(true);
    pdu->Save();

    ParaViewDataCollection *pdp = NULL;
    pdp = new ParaViewDataCollection("NSE_manufacture_pressure", mesh);
    pdp->SetPrefixPath("ParaView");
    pdp->RegisterField("pressure", &p);
    pdp->SetLevelsOfDetail(order);
    pdp->SetDataFormat(VTKFormat::BINARY);
    pdp->SetHighOrderOutput(true);
    pdp->Save();


    

    // Free the used memory.
    delete Xh_space;
    delete Ph_space;
    delete quad_coll;
    delete lin_coll;
    delete pdp;
    delete pdu;
    delete mesh;

    return 0;
}


void u_ex(const Vector & x, Vector & u)
{
    double xi(x(0));
    double yi(x(1));


    u(0) = cos(xi)*sin(yi);
    u(1) = -1*sin(xi)*cos(yi);


}


double p_ex(const Vector & x)
{
    double xi(x(0));
    double yi(x(1));

    return 0.0;
    //return -1./4 * (cos(2*xi) + cos(2*yi));
}

void fFun(const Vector & x, Vector & f)
{
    double xi(x(0));
    double yi(x(1));

    f(0) = 2.0*sin(yi)*cos(xi);
    f(1) = -2.0*sin(xi)*cos(yi);

    //f(0) = (1.0*sin(xi) + 2.0*sin(yi))*cos(xi);
    //f(1) = (-2.0*sin(xi) + 1.0*sin(yi))*cos(yi);

}

void velocity_function(const Vector &x, Vector &v)
{

   
    v(0) = 1.0;
    v(1) = 1.0;
}
}

void NonlinearFormIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Tr,
   const Vector &elfun, Vector &elvect)
{
   mfem_error("NonlinearFormIntegrator::AssembleElementVector"
              " is not overloaded!");
}

void NonlinearFormIntegrator::AssembleFaceVector(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect)
{
   mfem_error("NonlinearFormIntegrator::AssembleFaceVector"
              " is not overloaded!");
}

void NonlinearFormIntegrator::AssembleElementGrad(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
   DenseMatrix &elmat)
{
   mfem_error("NonlinearFormIntegrator::AssembleElementGrad"
              " is not overloaded!");
}

void NonlinearFormIntegrator::AssembleFaceGrad(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, const Vector &elfun,
   DenseMatrix &elmat)
{
   mfem_error("NonlinearFormIntegrator::AssembleFaceGrad"
              " is not overloaded!");
}

double NonlinearFormIntegrator::GetElementEnergy(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun)
{
   mfem_error("NonlinearFormIntegrator::GetElementEnergy"
              " is not overloaded!");
   return 0.0;
}


void BlockNonlinearFormIntegrator::AssembleElementVector(
   const Array<const FiniteElement *> &el,
   ElementTransformation &Tr,
   const Array<const Vector *> &elfun,
   const Array<Vector *> &elvec)
{
   mfem_error("BlockNonlinearFormIntegrator::AssembleElementVector"
              " is not overloaded!");
}

void BlockNonlinearFormIntegrator::AssembleFaceVector(
   const Array<const FiniteElement *> &el1,
   const Array<const FiniteElement *> &el2,
   FaceElementTransformations &Tr,
   const Array<const Vector *> &elfun,
   const Array<Vector *> &elvect)
{
   mfem_error("BlockNonlinearFormIntegrator::AssembleFaceVector"
              " is not overloaded!");
}

void BlockNonlinearFormIntegrator::AssembleElementGrad(
   const Array<const FiniteElement*> &el,
   ElementTransformation &Tr,
   const Array<const Vector *> &elfun,
   const Array2D<DenseMatrix *> &elmats)
{
   mfem_error("BlockNonlinearFormIntegrator::AssembleElementGrad"
              " is not overloaded!");
}

void BlockNonlinearFormIntegrator::AssembleFaceGrad(
   const Array<const FiniteElement *>&el1,
   const Array<const FiniteElement *>&el2,
   FaceElementTransformations &Tr,
   const Array<const Vector *> &elfun,
   const Array2D<DenseMatrix *> &elmats)
{
   mfem_error("BlockNonlinearFormIntegrator::AssembleFaceGrad"
              " is not overloaded!");
}

double BlockNonlinearFormIntegrator::GetElementEnergy(
   const Array<const FiniteElement *>&el,
   ElementTransformation &Tr,
   const Array<const Vector *>&elfun)
{
   mfem_error("BlockNonlinearFormIntegrator::GetElementEnergy"
              " is not overloaded!");
   return 0.0;
}


double InverseHarmonicModel::EvalW(const DenseMatrix &J) const
{
   Z.SetSize(J.Width());
   CalcAdjugateTranspose(J, Z);
   return 0.5*(Z*Z)/J.Det();
}

void InverseHarmonicModel::EvalP(const DenseMatrix &J, DenseMatrix &P) const
{
   int dim = J.Width();
   double t;

   Z.SetSize(dim);
   S.SetSize(dim);
   CalcAdjugateTranspose(J, Z);
   MultAAt(Z, S);
   t = 0.5*S.Trace();
   for (int i = 0; i < dim; i++)
   {
      S(i,i) -= t;
   }
   t = J.Det();
   S *= -1.0/(t*t);
   Mult(S, Z, P);
}

void InverseHarmonicModel::AssembleH(
   const DenseMatrix &J, const DenseMatrix &DS, const double weight,
   DenseMatrix &A) const
{
   int dof = DS.Height(), dim = DS.Width();
   double t;

   Z.SetSize(dim);
   S.SetSize(dim);
   G.SetSize(dof, dim);
   C.SetSize(dof, dim);

   CalcAdjugateTranspose(J, Z);
   MultAAt(Z, S);

   t = 1.0/J.Det();
   Z *= t;  // Z = J^{-t}
   S *= t;  // S = |J| (J.J^t)^{-1}
   t = 0.5*S.Trace();

   MultABt(DS, Z, G);  // G = DS.J^{-1}
   Mult(G, S, C);

   // 1.
   for (int i = 0; i < dof; i++)
      for (int j = 0; j <= i; j++)
      {
         double a = 0.0;
         for (int d = 0; d < dim; d++)
         {
            a += G(i,d)*G(j,d);
         }
         a *= weight;
         for (int k = 0; k < dim; k++)
            for (int l = 0; l <= k; l++)
            {
               double b = a*S(k,l);
               A(i+k*dof,j+l*dof) += b;
               if (i != j)
               {
                  A(j+k*dof,i+l*dof) += b;
               }
               if (k != l)
               {
                  A(i+l*dof,j+k*dof) += b;
                  if (i != j)
                  {
                     A(j+l*dof,i+k*dof) += b;
                  }
               }
            }
      }

   // 2.
   for (int i = 1; i < dof; i++)
      for (int j = 0; j < i; j++)
      {
         for (int k = 1; k < dim; k++)
            for (int l = 0; l < k; l++)
            {
               double a =
                  weight*(C(i,l)*G(j,k) - C(i,k)*G(j,l) +
                          C(j,k)*G(i,l) - C(j,l)*G(i,k) +
                          t*(G(i,k)*G(j,l) - G(i,l)*G(j,k)));

               A(i+k*dof,j+l*dof) += a;
               A(j+l*dof,i+k*dof) += a;

               A(i+l*dof,j+k*dof) -= a;
               A(j+k*dof,i+l*dof) -= a;
            }
      }
}


inline void NeoHookeanModel::EvalCoeffs() const
{
   mu = c_mu->Eval(*Ttr, Ttr->GetIntPoint());
   K = c_K->Eval(*Ttr, Ttr->GetIntPoint());
   if (c_g)
   {
      g = c_g->Eval(*Ttr, Ttr->GetIntPoint());
   }
}

double NeoHookeanModel::EvalW(const DenseMatrix &J) const
{
   int dim = J.Width();

   if (have_coeffs)
   {
      EvalCoeffs();
   }

   double dJ = J.Det();
   double sJ = dJ/g;
   double bI1 = pow(dJ, -2.0/dim)*(J*J); // \bar{I}_1

   return 0.5*(mu*(bI1 - dim) + K*(sJ - 1.0)*(sJ - 1.0));
}

void NeoHookeanModel::EvalP(const DenseMatrix &J, DenseMatrix &P) const
{
   int dim = J.Width();

   if (have_coeffs)
   {
      EvalCoeffs();
   }

   Z.SetSize(dim);
   CalcAdjugateTranspose(J, Z);

   double dJ = J.Det();
   double a  = mu*pow(dJ, -2.0/dim);
   double b  = K*(dJ/g - 1.0)/g - a*(J*J)/(dim*dJ);

   P = 0.0;
   P.Add(a, J);
   P.Add(b, Z);
}

void NeoHookeanModel::AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                                const double weight, DenseMatrix &A) const
{
   int dof = DS.Height(), dim = DS.Width();

   if (have_coeffs)
   {
      EvalCoeffs();
   }

   Z.SetSize(dim);
   G.SetSize(dof, dim);
   C.SetSize(dof, dim);

   double dJ = J.Det();
   double sJ = dJ/g;
   double a  = mu*pow(dJ, -2.0/dim);
   double bc = a*(J*J)/dim;
   double b  = bc - K*sJ*(sJ - 1.0);
   double c  = 2.0*bc/dim + K*sJ*(2.0*sJ - 1.0);

   CalcAdjugateTranspose(J, Z);
   Z *= (1.0/dJ); // Z = J^{-t}

   MultABt(DS, J, C); // C = DS J^t
   MultABt(DS, Z, G); // G = DS J^{-1}

   a *= weight;
   b *= weight;
   c *= weight;

   // 1.
   for (int i = 0; i < dof; i++)
      for (int k = 0; k <= i; k++)
      {
         double s = 0.0;
         for (int d = 0; d < dim; d++)
         {
            s += DS(i,d)*DS(k,d);
         }
         s *= a;

         for (int d = 0; d < dim; d++)
         {
            A(i+d*dof,k+d*dof) += s;
         }

         if (k != i)
            for (int d = 0; d < dim; d++)
            {
               A(k+d*dof,i+d*dof) += s;
            }
      }

   a *= (-2.0/dim);

   // 2.
   for (int i = 0; i < dof; i++)
      for (int j = 0; j < dim; j++)
         for (int k = 0; k < dof; k++)
            for (int l = 0; l < dim; l++)
            {
               A(i+j*dof,k+l*dof) +=
                  a*(C(i,j)*G(k,l) + G(i,j)*C(k,l)) +
                  b*G(i,l)*G(k,j) + c*G(i,j)*G(k,l);
            }
}


double HyperelasticNLFIntegrator::GetElementEnergy(const FiniteElement &el,
                                                   ElementTransformation &Ttr,
                                                   const Vector &elfun)
{
   int dof = el.GetDof(), dim = el.GetDim();
   double energy;

   DSh.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpr.SetSize(dim);
   Jpt.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);

   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
   }

   energy = 0.0;
   model->SetTransformation(Ttr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);
      CalcInverse(Ttr.Jacobian(), Jrt);

      el.CalcDShape(ip, DSh);
      MultAtB(PMatI, DSh, Jpr);
      Mult(Jpr, Jrt, Jpt);

      energy += ip.weight * Ttr.Weight() * model->EvalW(Jpt);
   }

   return energy;
}

void HyperelasticNLFIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Ttr,
   const Vector &elfun, Vector &elvect)
{
   int dof = el.GetDof(), dim = el.GetDim();

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpt.SetSize(dim);
   P.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elvect.SetSize(dof*dim);
   PMatO.UseExternalData(elvect.GetData(), dof, dim);

   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
   }

   elvect = 0.0;
   model->SetTransformation(Ttr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);
      CalcInverse(Ttr.Jacobian(), Jrt);

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS);
      MultAtB(PMatI, DS, Jpt);

      model->EvalP(Jpt, P);

      P *= ip.weight * Ttr.Weight();
      AddMultABt(DS, P, PMatO);
   }
}

void HyperelasticNLFIntegrator::AssembleElementGrad(const FiniteElement &el,
                                                    ElementTransformation &Ttr,
                                                    const Vector &elfun,
                                                    DenseMatrix &elmat)
{
   int dof = el.GetDof(), dim = el.GetDim();

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpt.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elmat.SetSize(dof*dim);

   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
   }

   elmat = 0.0;
   model->SetTransformation(Ttr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);
      CalcInverse(Ttr.Jacobian(), Jrt);

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS);
      MultAtB(PMatI, DS, Jpt);

      model->AssembleH(Jpt, DS, ip.weight * Ttr.Weight(), elmat);
   }
}

double IncompressibleNeoHookeanIntegrator::GetElementEnergy(
   const Array<const FiniteElement *>&el,
   ElementTransformation &Tr,
   const Array<const Vector *>&elfun)
{
   if (el.Size() != 2)
   {
      mfem_error("IncompressibleNeoHookeanIntegrator::GetElementEnergy"
                 " has incorrect block finite element space size!");
   }

   int dof_u = el[0]->GetDof();
   int dim = el[0]->GetDim();

   DSh_u.SetSize(dof_u, dim);
   J0i.SetSize(dim);
   J1.SetSize(dim);
   J.SetSize(dim);
   PMatI_u.UseExternalData(elfun[0]->GetData(), dof_u, dim);

   int intorder = 2*el[0]->GetOrder() + 3; // <---
   const IntegrationRule &ir = IntRules.Get(el[0]->GetGeomType(), intorder);

   double energy = 0.0;
   double mu = 0.0;

   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);
      CalcInverse(Tr.Jacobian(), J0i);

      el[0]->CalcDShape(ip, DSh_u);
      MultAtB(PMatI_u, DSh_u, J1);
      Mult(J1, J0i, J);

      mu = c_mu->Eval(Tr, ip);

      energy += ip.weight*Tr.Weight()*(mu/2.0)*(J*J - 3);
   }

   return energy;
}

void IncompressibleNeoHookeanIntegrator::AssembleElementVector(
   const Array<const FiniteElement *> &el,
   ElementTransformation &Tr,
   const Array<const Vector *> &elfun,
   const Array<Vector *> &elvec)
{
   if (el.Size() != 2)
   {
      mfem_error("IncompressibleNeoHookeanIntegrator::AssembleElementVector"
                 " has finite element space of incorrect block number");
   }

   int dof_u = el[0]->GetDof();
   int dof_p = el[1]->GetDof();

   int dim = el[0]->GetDim();
   int spaceDim = Tr.GetSpaceDim();

   if (dim != spaceDim)
   {
      mfem_error("IncompressibleNeoHookeanIntegrator::AssembleElementVector"
                 " is not defined on manifold meshes");
   }


   DSh_u.SetSize(dof_u, dim);
   DS_u.SetSize(dof_u, dim);
   J0i.SetSize(dim);
   F.SetSize(dim);
   FinvT.SetSize(dim);
   P.SetSize(dim);
   PMatI_u.UseExternalData(elfun[0]->GetData(), dof_u, dim);
   elvec[0]->SetSize(dof_u*dim);
   PMatO_u.UseExternalData(elvec[0]->GetData(), dof_u, dim);

   Sh_p.SetSize(dof_p);
   elvec[1]->SetSize(dof_p);

   int intorder = 2*el[0]->GetOrder() + 3; // <---
   const IntegrationRule &ir = IntRules.Get(el[0]->GetGeomType(), intorder);

   *elvec[0] = 0.0;
   *elvec[1] = 0.0;

   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);
      CalcInverse(Tr.Jacobian(), J0i);

      el[0]->CalcDShape(ip, DSh_u);
      Mult(DSh_u, J0i, DS_u);
      MultAtB(PMatI_u, DS_u, F);

      el[1]->CalcShape(ip, Sh_p);

      double pres = Sh_p * *elfun[1];
      double mu = c_mu->Eval(Tr, ip);
      double dJ = F.Det();

      CalcInverseTranspose(F, FinvT);

      P = 0.0;
      P.Add(mu * dJ, F);
      P.Add(-1.0 * pres * dJ, FinvT);
      P *= ip.weight*Tr.Weight();

      AddMultABt(DS_u, P, PMatO_u);

      elvec[1]->Add(ip.weight * Tr.Weight() * (dJ - 1.0), Sh_p);
   }

}

void IncompressibleNeoHookeanIntegrator::AssembleElementGrad(
   const Array<const FiniteElement*> &el,
   ElementTransformation &Tr,
   const Array<const Vector *> &elfun,
   const Array2D<DenseMatrix *> &elmats)
{
   int dof_u = el[0]->GetDof();
   int dof_p = el[1]->GetDof();

   int dim = el[0]->GetDim();

   elmats(0,0)->SetSize(dof_u*dim, dof_u*dim);
   elmats(0,1)->SetSize(dof_u*dim, dof_p);
   elmats(1,0)->SetSize(dof_p, dof_u*dim);
   elmats(1,1)->SetSize(dof_p, dof_p);

   *elmats(0,0) = 0.0;
   *elmats(0,1) = 0.0;
   *elmats(1,0) = 0.0;
   *elmats(1,1) = 0.0;

   DSh_u.SetSize(dof_u, dim);
   DS_u.SetSize(dof_u, dim);
   J0i.SetSize(dim);
   F.SetSize(dim);
   FinvT.SetSize(dim);
   Finv.SetSize(dim);
   P.SetSize(dim);
   PMatI_u.UseExternalData(elfun[0]->GetData(), dof_u, dim);
   Sh_p.SetSize(dof_p);

   int intorder = 2*el[0]->GetOrder() + 3; // <---
   const IntegrationRule &ir = IntRules.Get(el[0]->GetGeomType(), intorder);

   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);
      CalcInverse(Tr.Jacobian(), J0i);

      el[0]->CalcDShape(ip, DSh_u);
      Mult(DSh_u, J0i, DS_u);
      MultAtB(PMatI_u, DS_u, F);

      el[1]->CalcShape(ip, Sh_p);
      double pres = Sh_p * *elfun[1];
      double mu = c_mu->Eval(Tr, ip);
      double dJ = F.Det();
      double dJ_FinvT_DS;

      CalcInverseTranspose(F, FinvT);

      // u,u block
      for (int i_u = 0; i_u < dof_u; ++i_u)
      {
         for (int i_dim = 0; i_dim < dim; ++i_dim)
         {
            for (int j_u = 0; j_u < dof_u; ++j_u)
            {
               for (int j_dim = 0; j_dim < dim; ++j_dim)
               {

                  // m = j_dim;
                  // k = i_dim;

                  for (int n=0; n<dim; ++n)
                  {
                     for (int l=0; l<dim; ++l)
                     {
                        (*elmats(0,0))(i_u + i_dim*dof_u, j_u + j_dim*dof_u) +=
                           dJ * (mu * F(i_dim, l) - pres * FinvT(i_dim,l)) *
                           FinvT(j_dim,n) * DS_u(i_u,l) * DS_u(j_u, n) *
                           ip.weight * Tr.Weight();

                        if (j_dim == i_dim && n==l)
                        {
                           (*elmats(0,0))(i_u + i_dim*dof_u, j_u + j_dim*dof_u) +=
                              dJ * mu * DS_u(i_u, l) * DS_u(j_u,n) *
                              ip.weight * Tr.Weight();
                        }

                        // a = n;
                        // b = m;
                        (*elmats(0,0))(i_u + i_dim*dof_u, j_u + j_dim*dof_u) +=
                           dJ * pres * FinvT(i_dim, n) *
                           FinvT(j_dim,l) * DS_u(i_u,l) * DS_u(j_u,n) *
                           ip.weight * Tr.Weight();
                     }
                  }
               }
            }
         }
      }

      // u,p and p,u blocks
      for (int i_p = 0; i_p < dof_p; ++i_p)
      {
         for (int j_u = 0; j_u < dof_u; ++j_u)
         {
            for (int dim_u = 0; dim_u < dim; ++dim_u)
            {
               for (int l=0; l<dim; ++l)
               {
                  dJ_FinvT_DS = dJ * FinvT(dim_u,l) * DS_u(j_u, l) * Sh_p(i_p) *
                                ip.weight * Tr.Weight();
                  (*elmats(1,0))(i_p, j_u + dof_u * dim_u) += dJ_FinvT_DS;
                  (*elmats(0,1))(j_u + dof_u * dim_u, i_p) -= dJ_FinvT_DS;

               }
            }
         }
      }
   }

}

const IntegrationRule&
VectorConvectionNLFIntegrator::GetRule(const FiniteElement &fe,
                                       ElementTransformation &T)
{
   const int order = 2 * fe.GetOrder() + T.OrderGrad(&fe);
   return IntRules.Get(fe.GetGeomType(), order);
}

void VectorConvectionNLFIntegrator::AssembleElementVector(
   const FiniteElement &el,
   ElementTransformation &T,
   const Vector &elfun,
   Vector &elvect)
{
   const int nd = el.GetDof();
   const int dim = el.GetDim();

   shape.SetSize(nd);
   dshape.SetSize(nd, dim);
   elvect.SetSize(nd * dim);
   gradEF.SetSize(dim);

   EF.UseExternalData(elfun.GetData(), nd, dim);
   ELV.UseExternalData(elvect.GetData(), nd, dim);

   Vector vec1(dim), vec2(dim);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, T);
   ELV = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      T.SetIntPoint(&ip);
      el.CalcShape(ip, shape);
      el.CalcPhysDShape(T, dshape);
      double w = ip.weight * T.Weight();
      if (Q) { w *= Q->Eval(T, ip); }
      MultAtB(EF, dshape, gradEF); // grad u
      EF.MultTranspose(shape, vec1); // u
      gradEF.Mult(vec1, vec2); // (u \cdot \grad u
      vec2 *= w;
      AddMultVWt(shape, vec2, ELV); // (u \cdot \grad u,v)
   }
}

void VectorConvectionNLFIntegrator::AssembleElementGrad(
   const FiniteElement &el,
   ElementTransformation &trans,
   const Vector &elfun,
   DenseMatrix &elmat)
{
   int nd = el.GetDof();
   int dim = el.GetDim();

   shape.SetSize(nd);
   dshape.SetSize(nd, dim);
   dshapex.SetSize(nd, dim);
   elmat.SetSize(nd * dim);
   elmat_comp.SetSize(nd);
   gradEF.SetSize(dim);

   EF.UseExternalData(elfun.GetData(), nd, dim);

   double w;
   Vector vec1(dim), vec2(dim), vec3(nd);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      el.CalcShape(ip, shape);
      el.CalcDShape(ip, dshape);

      Mult(dshape, trans.InverseJacobian(), dshapex);

      w = ip.weight;

      if (Q)
      {
         w *= Q->Eval(trans, ip);
      }

      MultAtB(EF, dshapex, gradEF);
      EF.MultTranspose(shape, vec1);

      trans.AdjugateJacobian().Mult(vec1, vec2);

      vec2 *= w;
      dshape.Mult(vec2, vec3);
      MultVWt(shape, vec3, elmat_comp);

      for (int i = 0; i < dim; i++)
      {
         elmat.AddMatrix(elmat_comp, i * nd, i * nd);
      }

      MultVVt(shape, elmat_comp);
      w = ip.weight * trans.Weight();
      if (Q)
      {
         w *= Q->Eval(trans, ip);
      }
      for (int i = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++)
         {
            elmat.AddMatrix(w * gradEF(i, j), elmat_comp, i * nd, j * nd);
         }
      }
   }
}

const IntegrationRule&
ConvectiveVectorConvectionNLFIntegrator::GetRule(const FiniteElement &fe,
                                       ElementTransformation &T)
{
   const int order = 2 * fe.GetOrder() + T.OrderGrad(&fe);
   return IntRules.Get(fe.GetGeomType(), order);
}

void ConvectiveVectorConvectionNLFIntegrator::AssembleElementGrad(
   const FiniteElement &el,
   ElementTransformation &trans,
   const Vector &elfun,
   DenseMatrix &elmat)
{
   int nd = el.GetDof();
   int dim = el.GetDim();

   shape.SetSize(nd);
   dshape.SetSize(nd, dim);
   dshapex.SetSize(nd, dim);
   elmat.SetSize(nd * dim);
   elmat_comp.SetSize(nd);
   gradEF.SetSize(dim);

   EF.UseExternalData(elfun.GetData(), nd, dim);

   elfun.Print();

   // for(int i = 0; i < nd; i++)
   // {
   //    for(int j = 0; j < dim; j++)
   //    {

   //    }
   // }

   double w;
   Vector vec1(dim), vec2(dim), vec3(nd);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      el.CalcShape(ip, shape);
      el.CalcDShape(ip, dshape);

      Mult(dshape, trans.InverseJacobian(), dshapex);

      w = ip.weight;

      if (Q)
      {
         w *= Q->Eval(trans, ip);
      }

      //MultAtB(EF, dshapex, gradEF);
      EF.MultTranspose(shape, vec1); // u^n

      trans.AdjugateJacobian().Mult(vec1, vec2);

      vec2 *= w;
      dshape.Mult(vec2, vec3); // (u^n \cdot \ grad u^{n+1})
      MultVWt(shape, vec3, elmat_comp); // (u^n \cdot \ grad u^{n+1},v)

      for (int i = 0; i < dim; i++)
      {
         elmat.AddMatrix(elmat_comp, i * nd, i * nd);
      }

      
   }
}

}
