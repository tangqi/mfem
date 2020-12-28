#include "mfem.hpp"

using namespace std;
using namespace mfem;

// Integrator for the boundary gradient integral from the Laplacian operator
// this is used in the auxiliary variable where the boundary condition is not needed
class BoundaryGradIntegrator: public BilinearFormIntegrator
{
private:
   Vector shape1, dshape_dn, nor;
   DenseMatrix dshape, dshapedxt;

public:
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

void BoundaryGradIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int i, j, ndof1;
   int dim, order;
   double w;

   dim = el1.GetDim();
   ndof1 = el1.GetDof();

   // set to this for now, integration includes rational terms
   order = 2*el1.GetOrder() + 1;

   nor.SetSize(dim);
   shape1.SetSize(ndof1);
   dshape_dn.SetSize(ndof1);
   dshape.SetSize(ndof1,dim);
   dshapedxt.SetSize(ndof1,dim);

   elmat.SetSize(ndof1);
   elmat = 0.0;

   const IntegrationRule *ir = &IntRules.Get(Trans.FaceGeom, order);
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1;
      Trans.Loc1.Transform(ip, eip1);
      el1.CalcShape(eip1, shape1);
      //d of shape function, evaluated at eip1
      el1.CalcDShape(eip1, dshape);

      Trans.Elem1->SetIntPoint(&eip1);

      // cout<<"J size = "<<Trans.Elem1->Jacobian().Height()<<" "<<Trans.Elem1->Jacobian().Width()<<endl;
      // let dshapedxt = grad phi* J^-1 where J^-1=adj(J)/det(J)
      // need to distinguis Trans.Weight from Trans.Elem1->Weight 
      w = Trans.Elem1->Weight();
      Mult(dshape, Trans.Elem1->AdjugateJacobian(), dshapedxt); 
      dshapedxt*=(1./w);

      //get normal vector (note nor contains the face weight, so we do not need to multiply it to final result)
      Trans.Face->SetIntPoint(&ip);
      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), nor);
      }
      nor.Print();

      // multiply weight into normal, make answer negative
      // (boundary integral is subtracted)
      nor *= -ip.weight;

      dshapedxt.Mult(nor, dshape_dn);

      for (i = 0; i < ndof1; i++)
         for (j = 0; j < ndof1; j++)
         {
            elmat(i, j) += shape1(i)*dshape_dn(j);
         }
   }
}

// Boundary integral for the boundary condition
// <curl u, n x grad v> 
// u in H1^d, v in H1
class CurlNormalCrossGradIntegrator: public BilinearFormIntegrator
{
private:
   Vector nor;
   DenseMatrix dshape, dshapedxt, 
               dshape_trial, dshapedxt_trial, 
               test_shape, curl_trial;

public:
   virtual void AssembleFaceMatrix(const FiniteElement &trial_el,
                                   const FiniteElement &test_el,
                                   const FiniteElement &test_dummy,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

void CurlNormalCrossGradIntegrator::AssembleFaceMatrix(
   const FiniteElement &trial_el, const FiniteElement &test_el, const FiniteElement &test_dummy,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int i, j, dim, order;
   int trial_nd = trial_el.GetDof(), test_nd = test_el.GetDof();
   double w;

   dim = trial_el.GetDim();
   nor.SetSize(dim);

   // set to this for now
   order = 2*Trans.OrderGrad(&test_el);
   const IntegrationRule *ir = &IntRules.Get(Trans.FaceGeom, order);

   dshape.SetSize(test_nd,dim);
   dshape_trial.SetSize(trial_nd,dim);
   dshapedxt.SetSize(test_nd,dim);
   dshapedxt_trial.SetSize(trial_nd,dim);

   test_shape.SetSize(test_nd, dim*(dim-1)/2);
   curl_trial.SetSize(trial_nd*dim, dim*(dim-1)/2);

   elmat.SetSize(test_nd, trial_nd);
   elmat = 0.0;

   if (dim==1)
   {
       mfem_error("CurlNormalCrossGradIntegrator::AssembleFaceMatrix(...)\n"
                  "1D is not supported.");
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      //new
      //Trans.SetAllIntPoints(&ip);
      //const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

      //old
      IntegrationPoint eip1;
      Trans.Loc1.Transform(ip, eip1);
      Trans.Elem1->SetIntPoint(&eip1);

      //d of shape function, evaluated at eip1
      test_el.CalcDShape(eip1, dshape);
      w = Trans.Elem1->Weight();
      Mult(dshape, Trans.Elem1->AdjugateJacobian(), dshapedxt); 
      dshapedxt*=(1./w);

      //get normal vector (nor contain Trans.Frac weight)
      Trans.Face->SetIntPoint(&ip);
      CalcOrtho(Trans.Face->Jacobian(), nor);

      // multiply weight into normal 
      nor *= ip.weight;

      nor.Print();
      cout<<"weights ="<<w<<endl;

      //compute n x grad v
      if (dim==3)
      {
        for (int j=0; j<test_nd; j++)
        {
           test_shape(j,0) = dshapedxt(j,2) * nor(1) -
                             dshapedxt(j,1) * nor(2);
           test_shape(j,1) = dshapedxt(j,0) * nor(2) -
                             dshapedxt(j,2) * nor(0);
           test_shape(j,2) = dshapedxt(j,1) * nor(0) -
                             dshapedxt(j,0) * nor(1);
        }
      }
      else{
        for (int j=0; j<test_nd; j++)
        {
           test_shape(j,0) = dshapedxt(j,1) * nor(0) -
                             dshapedxt(j,0) * nor(1);
        }
      }

      trial_el.CalcDShape(eip1, dshape_trial);
      w = Trans.Elem1->Weight();
      Mult(dshape_trial, Trans.Elem1->AdjugateJacobian(), dshapedxt_trial); 
      dshapedxt_trial*=(1./w);
 
      dshapedxt_trial.GradToCurl(curl_trial);

      AddMult_a_ABt(1.0, test_shape, curl_trial, elmat);
  }
}


