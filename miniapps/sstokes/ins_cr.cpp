//                                MFEM mixed INS solver
//
// Compile with: make ins_cr
//
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "ortho_solver.hpp"

using namespace mfem::navier;
using namespace std;
using namespace mfem;


class INSOperator : public TimeDependentOperator
{
protected:
   FiniteElementSpace &U_space, &P_space;
   Array<int> &ess_bdr, &block_offsets, vel_ess_tdof_list, pres_ess_tdof_list;    

   BilinearForm *M, *Mrhs, *K, *D;
   MixedBilinearForm *dform;
   LinearForm *fform;
   SparseMatrix Mmat, Kmat, Dmat;
   SparseMatrix *DmatT, *T, *S; 
   double current_dt, viscosity;
   VectorFunctionCoefficient *u_coeff;
   VectorFunctionCoefficient *force_coeff;

   MINRESSolver *solver;
   BlockOperator *blockOp;
   BlockDiagonalPreconditioner *prec;
   Solver *invS, *invT;

   mutable BlockVector z, rhs; // auxiliary BlockVector
   mutable GridFunction u_bdr;

public:
   INSOperator(FiniteElementSpace &vel_fes, FiniteElementSpace &pres_fes, double visc, 
               Array<int> &ess_bdr_, Array<int> &block_offsets_);

   virtual void Mult(const Vector &u, Vector &du_dt) const;

   /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   virtual void ImplicitSolve(const double dt, const Vector &up, Vector &dup_dt);

   virtual ~INSOperator();
};


double visc=1.0;

void vel_ex(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);
   double Tt = 1.+t/2.+t*t/3.;

   u(0) = (xi*xi+2*xi*yi+yi*yi)*Tt;
   u(1) = (xi*xi-2*xi*yi-yi*yi)*Tt;
}

double pres_ex(const Vector &x, double t)
{
   double xi = x(0);
   double yi = x(1);
   double Tt = 1.+t/2.+t*t/3.;

   //this pressure has 0 average in [0, 1]^2
   return (xi*xi+xi*yi*4/3.+yi*yi-1)*Tt;
}

void forcefun(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);
   double Tt = 1.+t/2.+t*t/3.;
   double dTt= 0.5+2.*t/3.;

   //f = du/dt + grad p - nu Delta u
   u(0) = (xi*xi+2*xi*yi+yi*yi)*dTt + (2*xi+yi*4/3.)*Tt - visc*4.*Tt ;
   u(1) = (xi*xi-2*xi*yi-yi*yi)*dTt + (xi*4/3.+2*yi)*Tt;
}


int main(int argc, char *argv[])
{
   const char *mesh_file = "../../data/inline-quad.mesh";
   int ref_levels = 2;
   int ode_solver_type = 1;
   double t_final = 0.5;
   double dt = 1.0e-2;
   bool visualization = false;
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
   args.AddOption(&visc, "-visc", "--viscosity", "Viscosity.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
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

   if (dim!=2)
   {         
       cout << "only 2D (tri and quad) is supported for now" << '\n';
       delete mesh;
       return 3;
   }

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

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // Define a finite element space on the mesh.
   FiniteElementCollection *vel_fec, *pres_fec;
   vel_fec = new CrouzeixRaviartFECollection();
   pres_fec = new L2_FECollection(0, dim);
   FiniteElementSpace *vel_fes = new FiniteElementSpace(mesh, vel_fec, dim);
   FiniteElementSpace *pres_fes = new FiniteElementSpace(mesh, pres_fec);

   Array<int> block_offsets(3); 
   block_offsets[0] = 0;
   block_offsets[1] = vel_fes->GetVSize();
   block_offsets[2] = pres_fes->GetVSize();
   block_offsets.PartialSum();

   std::cout << "***********************************************************\n";
   std::cout << "Dofs in CR = " << block_offsets[1] - block_offsets[0] << "\n";
   std::cout << "Dofs in L2 = " << block_offsets[2] - block_offsets[1] << "\n";
   std::cout << "Total Dofs = " << block_offsets.Last() << "\n";
   std::cout << "***********************************************************\n";

   // Create arrays for the essential boundary conditions.
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;

   // Create BlockVector and GridFunctions
   BlockVector up(block_offsets);

   GridFunction u_gf, p_gf;
   u_gf.MakeRef(vel_fes,  up.GetBlock(0), 0);
   p_gf.MakeRef(pres_fes, up.GetBlock(1), 0);

   // initial conditions
   VectorFunctionCoefficient v0coeff(dim, vel_ex);
   v0coeff.SetTime(0.);
   u_gf.ProjectCoefficient(v0coeff);

   INSOperator oper(*vel_fes, *pres_fes, visc, ess_bdr, block_offsets);

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

      ode_solver->Step(up, t, dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         cout << "step " << ti << ", t = " << t << endl;

         if (visualization)
         {
            sout << "solution\n" << *mesh << u_gf << flush;
         }

      }
   }

   if (true)
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("u.gf"), p_ofs("p.gf");
      sol_ofs.precision(8);
      p_ofs.precision(8);
      u_gf.Save(sol_ofs);
      p_gf.Save(p_ofs);
   }
  
   delete ode_solver;
   delete mesh;
   delete vel_fec;
   delete pres_fec;
   delete vel_fes;
   delete pres_fes;

   return 0;
}

INSOperator::INSOperator(FiniteElementSpace &vel_fes, 
                         FiniteElementSpace &pres_fes, double visc, 
                         Array<int> &ess_bdr_, 
                         Array<int> &block_offsets_)
   : TimeDependentOperator(vel_fes.GetVSize()+pres_fes.GetVSize(), 0.0), 
     U_space(vel_fes), P_space(pres_fes), 
     ess_bdr(ess_bdr_), block_offsets(block_offsets_),
     M(NULL), K(NULL), D(NULL), DmatT(NULL), T(NULL), S(NULL),
     invS(NULL), invT(NULL), blockOp(NULL),
     current_dt(0.0), viscosity(visc), u_coeff(NULL), force_coeff(NULL),
     z(block_offsets), rhs(block_offsets), u_bdr(&vel_fes)
{
   U_space.GetEssentialTrueDofs(ess_bdr, vel_ess_tdof_list);
   //note pres_ess_tdof_list remains empty

   M = new BilinearForm(&U_space);
   Mrhs = new BilinearForm(&U_space);

   ConstantCoefficient visc_coeff(viscosity);
   K = new BilinearForm(&U_space);
   K->AddDomainIntegrator(new VectorDiffusionIntegrator(visc_coeff));
   K->Assemble();
   K->FormSystemMatrix(vel_ess_tdof_list, Kmat);

   dform = new MixedBilinearForm(&U_space, &P_space);
   dform->AddDomainIntegrator(new VectorDivergenceIntegrator);
   dform->Assemble();
   dform->FormRectangularSystemMatrix(vel_ess_tdof_list, pres_ess_tdof_list, Dmat);
    
   DmatT = Transpose(Dmat);

   u_coeff = new VectorFunctionCoefficient(2, vel_ex);  //this assumes 2D
   force_coeff = new VectorFunctionCoefficient(2, forcefun);

   fform = new LinearForm(&U_space);
   fform->AddDomainIntegrator(new VectorDomainLFIntegrator(*force_coeff));

   /*
   * the block operator should be 
   * [ M/dt+K  -DmatT ]  
   * [ -D       0     ]
   * but M/dt+K needs to be updated on the fly 
   */
   blockOp = new BlockOperator(block_offsets);
   blockOp->SetBlock(0,1, DmatT, -1.);
   blockOp->SetBlock(1,0, &Dmat, -1.);
   
   /*
    * the preconditioenr should be 
    * [ diag(M/dt+K)                  ]  
    * [              B diag(M+kK) B^T ]
    * which needs to be updated on the fly
    */
   prec = new BlockDiagonalPreconditioner(block_offsets);

   solver = new MINRESSolver;
   solver->SetAbsTol(0.);
   solver->SetRelTol(1e-8);
   solver->SetMaxIter(10000);
   solver->SetPrintLevel(1);
   solver->iterative_mode = false;  //this might be okay to use true
}

//this should never be called
void INSOperator::Mult(const Vector &u, Vector &du_dt) const
{
    MFEM_ABORT("No explicit integrator should be called");
}

//backward Euler update
void INSOperator::ImplicitSolve(const double dt,
                                const Vector &up, Vector &dup_dt)
{
   // Define T = M/dt + K and so on (it is updated in the first call of implicitSolve)
   if (!T)
   {
      ConstantCoefficient rdt_coeff(1./dt);
      M->AddDomainIntegrator(new VectorMassIntegrator(rdt_coeff));
      M->Assemble();
      M->FormSystemMatrix(vel_ess_tdof_list, Mmat);    

      //we do not eliminate boundary in Mrhs
      Mrhs->AddDomainIntegrator(new VectorMassIntegrator(rdt_coeff));
      Mrhs->Assemble();

      T = Add(Mmat, Kmat);
      current_dt = dt;

      blockOp->SetBlock(0,0, T);

      Vector Td(M->Height());
      T->GetDiag(Td);
      invT = new DSmoother(*T);
      invT->iterative_mode = false;

      SparseMatrix *Mtmp = new SparseMatrix(*DmatT);         //deep copy DmatT
      for (int i = 0; i < Td.Size(); i++)
      {
         Mtmp->ScaleRow(i, 1./Td(i));
      }
      S = mfem::Mult(Dmat, *Mtmp);  //Here mfem is needed otherwise it will pick up INSOperator::Mult
      delete Mtmp;

#ifndef MFEM_USE_SUITESPARSE
      invS = new GSSmoother(*S);
#else
      invS = new UMFPackSolver(*S);
#endif
      invS->iterative_mode = false;

      prec->SetDiagonalBlock(0,invT);
      prec->SetDiagonalBlock(1,invS);

      solver->SetOperator(*blockOp); 
      solver->SetPreconditioner(*prec);
   }
   MFEM_VERIFY(dt == current_dt, "It only supports fixed dt for now."); 

   // update RHS  
   int sc = block_offsets[1]-block_offsets[0];
   Vector uold(up.GetData() + 0, sc);

   double time=GetTime();   //this will return current time

   //update boundary and force with the coefficients
   u_coeff->SetTime(time);
   u_bdr.ProjectBdrCoefficient(*u_coeff, ess_bdr);

   force_coeff->SetTime(time);
   fform->Assemble();

   // Compute rhs[0]=M/dt*uold+fform
   rhs=0.;
   Mrhs->Mult(uold, rhs.GetBlock(0));
   rhs.GetBlock(0)+=(*fform);

   SparseMatrix Mdummy;
   Vector X, B;

   //apply boundary condition
   M->FormLinearSystem(vel_ess_tdof_list, u_bdr, rhs.GetBlock(0), Mdummy, X, B);
   rhs.GetBlock(0)=B;

   K->FormLinearSystem(vel_ess_tdof_list, u_bdr, rhs.GetBlock(0), Mdummy, X, B);
   rhs.GetBlock(0)=B;

   // The wrong FormRectangularSystemMatrix will be called here, likely a bug
   // So we use FormColLinearSystem instead
   dform->FormColLinearSystem(vel_ess_tdof_list, u_bdr, rhs.GetBlock(1), Mdummy, X, B);
   rhs.GetBlock(1)=B;
   rhs.GetBlock(1)*=-1.0;

   // solve the system (dup_dt is used to hold up_new here)
   //solver->Mult(rhs, dup_dt);
   OrthoSolver::Mult(rhs,dup_dt);

   // upate dup_dt = (up_new-up_old)/dt
   dup_dt-=up;
   dup_dt/=dt;
}

INSOperator::~INSOperator()
{
   delete T;
   delete M;
   delete Mrhs;
   delete D;
   delete K;
   delete dform;
   delete fform;
   delete S; 
   delete DmatT;
   delete solver;
   delete blockOp;
   delete prec;
   delete invS;
   delete invT;
   delete u_coeff;
   delete force_coeff;
}


OrthoSolver::OrthoSolver() : Solver(0, true) {}

void OrthoSolver::SetOperator(const Operator &op)
{
   oper = &op;
}

void OrthoSolver::Mult(const Vector &b, Vector &x) const
{
   // Orthogonalize input
   Orthogonalize(b, b_ortho);

   // Apply operator
   oper->Mult(b_ortho, x);

   // Orthogonalize output
   Orthogonalize(x, x);
}

void OrthoSolver::Orthogonalize(const Vector &v, Vector &v_ortho) const
{
   double global_sum = v.Sum();
   int global_size = v.Size();

   double ratio = global_sum / static_cast<double>(global_size);
   v_ortho.SetSize(v.Size());
   for (int i = 0; i < v_ortho.Size(); ++i)
   {
      v_ortho(i) = v(i) - ratio;
   }
}
