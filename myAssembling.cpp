/*********************************************************************/
/* File:   myAssembling.cpp                                          */
/* Author: Joachim Schoeberl                                         */
/* Date:   3. May. 2010                                              */
/*********************************************************************/


/*

Assembling the matrix

*/

#include <solve.hpp>

using namespace ngsolve;

namespace myAssembling
{
  class NumProcMyAssembling : public NumProc
  {
  protected:
    GridFunction *gfu;

  public:
    
    NumProcMyAssembling (PDE & apde, const Flags & flags)
      : NumProc (apde)
    { 
      cout << "We assemble matrix and rhs vector" << endl;

      gfu = pde.GetGridFunction (flags.GetStringFlag ("gridfunction", "u"));
    }
  
    static NumProc * Create (PDE & pde, const Flags & flags)
    {
      return new NumProcMyAssembling (pde, flags);
    }
  
    virtual string GetClassName () const
    {
      return "MyAssembling";
    }


    virtual void Do (LocalHeap & lh)
    {
      const FESpace & fes = gfu -> GetFESpace();

      int ndof = fes.GetNDof();
      int ne = GetMeshAccess().GetNE();
    

      // generate sparse matrix

      Array<int> dnums;
      Array<int> cnt(ne);
      cnt = 0;
    
      for (int i = 0; i < ne; i++)
	{
	  fes.GetDofNrs (i, dnums);
	  for (int j = 0; j < dnums.Size(); j++)
            cnt[i]++;
	}	  
      
      Table<int> el2dof(cnt);

      cnt = 0;
      for (int i = 0; i < ne; i++)
	{
	  fes.GetDofNrs (i, dnums);
	  for (int j = 0; j < dnums.Size(); j++)
	    el2dof[i][cnt[i]++] = dnums[j];
	}
      
      MatrixGraph * graph = new MatrixGraph (el2dof, true);
      SparseMatrixSymmetric<double> & mat = *new SparseMatrixSymmetric<double> (*graph, true);


      VVector<double> vecf (fes.GetNDof());




      LaplaceIntegrator<2> laplace (new ConstantCoefficientFunction (1));
      SourceIntegrator<2> source (new ConstantCoefficientFunction (1));

      ElementTransformation eltrans;

      mat = 0.0;
      vecf = 0.0;

      for (int i = 0; i < ne; i++)   // loop over elements
	{  
	  HeapReset hr(lh); 

	  ma.GetElementTransformation (i, eltrans, lh);
	  
	  fes.GetDofNrs (i, dnums);
	  const FiniteElement & fel =  fes.GetFE (i, lh);
	  
	  FlatMatrix<> elmat (dnums.Size(), lh);
	  laplace.CalcElementMatrix (fel, eltrans, elmat, lh);
	  mat.AddElementMatrix (dnums, elmat);

	  FlatVector<> elvec (dnums.Size(), lh);
	  source.CalcElementVector (fel, eltrans, elvec, lh);
	  vecf.AddIndirect (dnums, elvec);
	} 


      *testout << "mat = " << mat << endl;
      *testout << "vecf = " << vecf << endl;


      BaseMatrix * inv = mat.InverseMatrix (fes.GetFreeDofs());

      gfu -> GetVector() = (*inv) * vecf;
    }
  };




    
  class Init
  { 
  public: 
    Init ();
  };
  
  Init::Init()
  {
    GetNumProcs().AddNumProc ("myassembling", NumProcMyAssembling::Create);
  }
  
  Init init;
}