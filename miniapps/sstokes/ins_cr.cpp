//                                MFEM mixed INS solver
//
// Compile with: make ins_cr
//
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


class INSOperator : public TimeDependentOperator
{
protected:
   FiniteElementSpace &cr, &l2;
   Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c. (need to be updated if we do Dirichelt)

   BilinearForm *M, *K, *Dx, *Dy;

   SparseMatrix Mmat, Kmat, Dxmat, Dymat;
   SparseMatrix *DxT, *DyT, *T, *S; // T = M + dt K
   double current_dt;

   MINRESSolver *solver;
   BlockDiagonalPreconditioner *prec;

   DSmoother *Tinv;

   mutable Vector z; // auxiliary vector

public:
   INSOperator(FiniteElementSpace &cr_, FiniteElementSpace &l2_);

   virtual void Mult(const Vector &u, Vector &du_dt) const;

   /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   virtual void ImplicitSolve(const double dt, const Vector &u, Vector &k);

   virtual ~INSOperator();
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int ref_levels = 2;
   int ode_solver_type = 1;
   double t_final = 0.5;
   double dt = 1.0e-2;
   bool visualization = true;
   bool visit = false;
   int vis_steps = 5;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3.\n");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    explicit Runge-Kutta methods are available.
   ODESolver *ode_solver;
   switch (ode_solver_type)
   {
      // Implicit L-stable methods
      case 1:  ode_solver = new BackwardEulerSolver; break;
      case 2:  ode_solver = new SDIRK23Solver(2); break;
      case 3:  ode_solver = new SDIRK33Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 23: ode_solver = new SDIRK23Solver; break;
      case 24: ode_solver = new SDIRK34Solver; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         delete mesh;
         return 3;
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define finite element space
   CrouzeixRaviartFECollection  cr_coll;
   L2_FECollection l2_coll(0, dim);
   FiniteElementSpace cr_fespace(mesh, &cr_coll);
   FiniteElementSpace l2_fespace(mesh, &l2_coll);

   if (dim!=2)
   {         
       cout << "only 2D is supported for now" << '\n';
       delete mesh;
       return 3;
   }

   Array<int> block_offsets(4); // number of variables + 1
   block_offsets[0] = 0;
   block_offsets[1] = cr_space->GetTrueVSize();
   block_offsets[2] = cr_space->GetTrueVSize();
   block_offsets[3] = l2_space->GetTrueVSize();
   block_offsets.PartialSum();

   std::cout << "***********************************************************\n";
   std::cout << "TrueVSize in CR = " << block_offsets[1] - block_offsets[0] << "\n";
   std::cout << "TrueVSize in L2 = " << block_offsets[3] - block_offsets[2] << "\n";
   std::cout << "Total TrueVSize (2*CR+L2) = " << block_offsets.Last() << "\n";
   std::cout << "***********************************************************\n";

   GridFunction u_gf(&fespace);

   // 7. Initialize the conduction operator and the visualization.
   INSOperator oper(cr_space, l2_space);

   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << *mesh << u_gf;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // 8. Perform time-integration (looping over the time iterations, ti, with a
   //    time-step dt).
   ode_solver->Init(oper);
   double t = 0.0;

   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }

      ode_solver->Step(u, t, dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         cout << "step " << ti << ", t = " << t << endl;

         u_gf.SetFromTrueDofs(u);
         if (visualization)
         {
            sout << "solution\n" << *mesh << u_gf << flush;
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }
      }
   }

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m ex16.mesh -g ex16-final.gf".
   {
      ofstream osol("ex16-final.gf");
      osol.precision(precision);
      u_gf.Save(osol);
   }

   // 10. Free the used memory.
   delete ode_solver;
   delete mesh;

   return 0;
}

//TODO: double check and define a block operator
//this is where the operator M, K, Dx, Dy should be assembled
INSOperator::INSOperator(FiniteElementSpace &cr_, FiniteElementSpace &l2_)
   : TimeDependentOperator(f.GetTrueVSize(), 0.0), cr(cr_), l2(l2_), 
     M(NULL), K(NULL), T(NULL), DxT(NULL), DyT(NULL), Minv(NULL)
     current_dt(0.0), z(height)
{
   M = new BilinearForm(&cr);
   M->AddDomainIntegrator(new MassIntegrator());
   M->Assemble();
   M->FormSystemMatrix(ess_tdof_list, Mmat);    //ess_tdof_list is empty for now!!

   K = new BilinearForm(&cr);
   K->AddDomainIntegrator(new DiffusionIntegrator());
   K->Assemble();
   K->FormSystemMatrix(ess_tdof_list, Kmat);

   ConstantCoefficient one(1.0);

   Dx = new MixedBilinearForm(&cr, &l2);
   Dx->AddDomainIntegrator(new DerivativeIntegrator(one, 0));
   Dx->Assemble();
   Dx->FormSystemMatrix(ess_tdof_list, Dxmat);

   Dy = new MixedBilinearForm(&cr, &l2);
   Dy->AddDomainIntegrator(new DerivativeIntegrator(one, 1));
   Dy->Assemble();
   Dy->FormSystemMatrix(ess_tdof_list, Dymat);

   DxT = Transpose(Dxmat);
   DyT = Transpose(Dymat);

    /*
    * the block operator should be 
    * [ M+dtK          DxT ]  
    * [        M+dtK   DyT ]
    * [  Dx     Dy         ]
    * but M+kK needs to be updated on the fly 
    * (or maybe we can fix time step with backward Euler for now)
    */
   blockOp = new BlockOperator(block_offsets);
   blockOp->SetBlock(0,2, DxT);
   blockOp->SetBlock(1,2, DyT);
   blockOp->SetBlock(2,0, Dx);
   blockOp->SetBlock(2,1, Dy);
   //blockOp (0,0) and (1,1) will be updated on the fly
   
   /*
    * the preconditioenr should be 
    * [ diag(M+kK)                              ]  
    * [            diag(M+kK)                   ]
    * [                        B diag(M+kK) B^T ]
    * which needs to be updated on the fly
    */
   prec = new BlockDiagonalPreconditioner(block_offsets);

   solver->SetAbsTol(1e-10);
   solver->SetRelTol(1e-6);
   solver->SetMaxIter(10000);
   solver->SetOperator(blockOp); //this needs to be updated every implicitSolve
   solver->SetPreconditioner(prec);
   solver->SetPrintLevel(1);

}

//this should never be called
void INSOperator::Mult(const Vector &u, Vector &du_dt) const
{
    MFEM_ABORT("No explicit integrator should be called");
}

//TODO
//this is where du/dt is computed, 
//we can simply solved for u^{n+1}, then let du/dt = (u^{n+1}-u^{n})/dt
void INSOperator::ImplicitSolve(const double dt,
                                       const Vector &u, Vector &du_dt)
{
   //first update blockOp and prec

   // Define T = M + dt K (it is updated in the first call of implicitSolve)
   if (!T)
   {
      T = Add(1.0, Mmat, dt, Kmat);
      current_dt = dt;

      blockOp->SetBlock(0,0, T);
      blockOp->SetBlock(1,1, T);
      solver->SetOperator(blockOp); 

      Vector Td(M->Height());
      T.GetDiag(Td);
      Tinv = new DSmoother(T);

      //need to work out the block preconditioner
      //I believe the schur complement is  Dx Td^-1 DxT + Dy Td^-1 DyT
      //But we need to double check
      for (int i = 0; i < Td.Size(); i++)
      {
         DxT->ScaleRow(i, 1./Td(i));
         DyT->ScaleRow(i, 1./Td(i));
      }
      SparseMatrix *Stmp;
      S = Mult(Dx, *DxT);
      Stmp = Mult(Dy, *DyT);
      S += Stmp;
      delete Stmp;
      prec->SetBlock(0,0,Tinv);
      prec->SetBlock(1,1,Tinv);
      prec->SetBlock(2,2,S);
   }
   MFEM_VERIFY(dt == current_dt, ""); // SDIRK methods use the same dt

   // update RHS = [u, v]+[f, v] (both are vector and the third component is 0)
   // use block vector

   // solve the system
   solver->Mult(z, du_dt);
}

INSOperator::~INSOperator()
{
   delete T;
   delete M;
   delete K;
   delete Dx;
   delete Dy;
   delete DxT;
   delete DyT;
   delete Tinv;
}


