#include "mfem.hpp"

using namespace std;
using namespace mfem;

class NormalTraceIntegrator : public BilinearFormIntegrator
{
private:
   Vector face_shape, normal, norN, shape1_n;
   DenseMatrix shape1, shape2;

public:
   NormalTraceIntegrator() { }
   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                                   const FiniteElement &test_fe1,
                                   const FiniteElement &test_fe2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};


void NormalTraceIntegrator::AssembleFaceMatrix(
   const FiniteElement &trial_face_fe, const FiniteElement &test_fe1,
   const FiniteElement &test_fe2, FaceElementTransformations &Trans,
   DenseMatrix &elmat)
{
   int i, j, face_ndof, ndof1, ndof2, dim;
   int order;
   double face_weight;

   MFEM_VERIFY(trial_face_fe.GetMapType() == FiniteElement::VALUE, "");

   face_ndof = trial_face_fe.GetDof();
   ndof1 = test_fe1.GetDof();
   dim = test_fe1.GetDim();

   face_shape.SetSize(face_ndof);
   normal.SetSize(dim);
   norN.SetSize(dim);
   shape1.SetSize(ndof1,dim);
   shape1_n.SetSize(ndof1);

   ndof2 = 0;
   elmat.SetSize(ndof1 + ndof2, face_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      
      order = test_fe1.GetOrder() - 1;
      order += trial_face_fe.GetOrder();

      if (order>1)
      {
        mfem_error("this should never happen!");
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   //cout<<"Trans.weight="<<Trans.Weight();

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1, eip2;

      // Trace finite element shape function
      trial_face_fe.CalcShape(ip, face_shape);

      // compute a normalized normal
      Trans.Loc1.Transf.SetIntPoint(&ip);
      CalcOrtho(Trans.Loc1.Transf.Jacobian(), normal);
      face_weight = normal.Norml2();
      norN.Set(1.0/face_weight, normal);

      // Side 1 finite element shape function
      Trans.Loc1.Transform(ip, eip1);
      test_fe1.CalcVShape(eip1, shape1);
      shape1.Mult(norN, shape1_n);

      // Here Trans.Weight() should not matter as long as it is consistent both sides!
      face_shape *= ip.weight*Trans.Weight();
      for (i = 0; i < ndof1; i++)
         for (j = 0; j < face_ndof; j++)
         {
            elmat(i, j) -= shape1_n(i) * face_shape(j);
         }
      
   }
}

//this is for Vector of Scalar component when CalcVShape is not available
class NormalVectorTraceIntegrator : public BilinearFormIntegrator
{
private:
   Vector face_shape, normal, norN, shape1;
   DenseMatrix shape2, partelmat;

public:
   NormalVectorTraceIntegrator() { }
   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                                   const FiniteElement &test_fe1,
                                   const FiniteElement &test_fe2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};


void NormalVectorTraceIntegrator::AssembleFaceMatrix(
   const FiniteElement &trial_face_fe, const FiniteElement &test_fe1,
   const FiniteElement &test_fe2, FaceElementTransformations &Trans,
   DenseMatrix &elmat)
{
   int i, j, face_ndof, te_ndof, dim;
   int order, vdim;
   double norm, face_weight;

   vdim = Trans.GetSpaceDim();
   dim = test_fe1.GetDim();

   MFEM_VERIFY(dim == vdim, "");
   MFEM_VERIFY(trial_face_fe.GetMapType() == FiniteElement::VALUE, "");

   face_ndof = trial_face_fe.GetDof();
   te_ndof = test_fe1.GetDof();

   face_shape.SetSize(face_ndof);
   normal.SetSize(dim);
   norN.SetSize(dim);
   shape1.SetSize(te_ndof);

   partelmat.SetSize(te_ndof, face_ndof);
   elmat.SetSize(te_ndof*vdim, face_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      order = test_fe1.GetOrder() - 1;
      order += trial_face_fe.GetOrder();

      if (order>1)
      {
        mfem_error("this should never happen!");
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1, eip2;

      // Trace finite element shape function
      trial_face_fe.CalcShape(ip, face_shape);

      // compute a normalized normal
      Trans.Loc1.Transf.SetIntPoint(&ip);
      CalcOrtho(Trans.Loc1.Transf.Jacobian(), normal);
      face_weight = normal.Norml2();
      norN.Set(1.0/face_weight, normal);

      // Side 1 finite element shape function
      Trans.Loc1.Transform(ip, eip1);
      test_fe1.CalcShape(eip1, shape1);

      norm = ip.weight*Trans.Weight();
      MultVWt(shape1, face_shape, partelmat);

      for (int k = 0; k < vdim; k++)
      {
         //cout<<"k="<<k<<" weigth="<<norm*normal(k)<<endl;
         elmat.AddMatrix(norm*norN(k), partelmat, te_ndof*k, 0);
      }
      
   }
   
   /*
   cout<<"vdim="<<vdim<<" elmat:\n";
   elmat.Print();
   cout<<"partelmat:\n";
   partelmat.Print();
   cout<<"normal:\n";
   normal.Print();
   */
}
