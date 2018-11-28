/*********************************************************************/
/* File:   myFESpace.cpp                                             */
/* Author: Joachim Schoeberl                                         */
/* Date:   26. Apr. 2009                                             */
/*********************************************************************/

/*

My own FESpace for linear and quadratic triangular elements.

A fe-space provides the connection between the local reference
element, and the global mesh.

*/

#include <comp.hpp> // provides FESpace, ...
#include <h1lofe.hpp>
#include <regex>
#include <python_ngstd.hpp>
#include "myElement.hpp"
#include "myFESpace.hpp"

namespace ngcomp
{

MyFESpace ::MyFESpace(shared_ptr<MeshAccess> ama, const Flags &flags)
    : FESpace(ama, flags)
{
  cout << "Constructor of MyFESpace" << endl;
  cout << "Flags = " << flags << endl;

  order = flags.GetNumFlag("order", 1);

  cout << "You have chosen element order " << order << endl;

  // needed for symbolic integrators and to draw solution
  evaluator[VOL] = make_shared<T_DifferentialOperator<DiffOpId<2>>>();
  flux_evaluator[VOL] = make_shared<T_DifferentialOperator<DiffOpGradient<2>>>();
  evaluator[BND] = make_shared<T_DifferentialOperator<DiffOpIdBoundary<2>>>();

  // (still) needed to draw solution
  integrator[VOL] = GetIntegrators().CreateBFI("mass", ma->GetDimension(),
                                               make_shared<ConstantCoefficientFunction>(1));
}

DocInfo MyFESpace ::GetDocu()
{
  auto docu = FESpace::GetDocu();
  docu.Arg("order") = "int = 1\n"
                      "order of basis functions";
  return docu;
}

void MyFESpace ::Update(LocalHeap &lh)
{
  // some global update:
  cout << "Update MyFESpace, #vert = " << ma->GetNV()
       << ", #edge = " << ma->GetNEdges() << endl;

  // number of vertices
  nvert = ma->GetNV();

  // number of dofs:
  ndof = nvert;
  switch (order)
  {
  case 1:
    ndof = ma->GetNV();
    break;
  case 2:
    ndof = ma->GetNV() + ma->GetNEdges();
    break;
  case 3:
    ndof = ma->GetNV() + 2 * ma->GetNEdges() + ma->GetNE();
    break;
  }
}

void MyFESpace ::GetDofNrs(ElementId ei, Array<DofId> &dnums) const
{
  // returns dofs of element ei
  // may be a volume triangle or boundary segment
  dnums.SetSize(0);

  auto element = ma->GetElement(ei);

  switch (order)
  {
  case 1:
    for (auto vertex : element.Vertices())
    {
      dnums.Append(vertex);
    }
    break;
  case 2:
    for (auto vertex : element.Vertices())
    {
      dnums.Append(vertex);
    }
    for (auto edge : element.Edges())
    {
      dnums.Append(ma->GetNV() + edge);
    }
    break;
  case 3:
    for (auto vertex : element.Vertices())
    {
      dnums.Append(vertex);
    }

    Array<int> edgeElements;
    for (auto edge : element.Edges())
    {
      ma->GetEdgeElements(edge, edgeElements);
      if (edgeElements.Size() == 1 || ei.Nr() <= edgeElements[0] && ei.Nr() <= edgeElements[1])
      {
        dnums.Append(ma->GetNV() + 2 * edge);
        dnums.Append(ma->GetNV() + 2 * edge + 1);
      }
      else
      {
        dnums.Append(ma->GetNV() + 2 * edge + 1);
        dnums.Append(ma->GetNV() + 2 * edge);
      }
    }

    if (ei.IsVolume())
    {
      dnums.Append(ma->GetNV() + ma->GetNEdges() * 2 + ei.Nr());
    }
    break;
  }
}

FiniteElement &MyFESpace ::GetFE(ElementId ei, Allocator &alloc) const
{
  if (ei.IsVolume())
  {
    switch (order)
    {
    case 1:
      switch (ma->GetElVertices(ei).Size())
      {
      case 3:
        return *new (alloc) MyLinearTrig;
      case 4:
        return *new (alloc) BilinearQuadElement;
      }
      break;
    case 2:
      return *new (alloc) MyQuadraticTrig;
    case 3:
      return *new (alloc) ThirdOrderTriangleElement;
    }
  }
  else
  {
    switch (order)
    {
    case 1:
      return *new (alloc) FE_Segm1;
    case 2:
      return *new (alloc) FE_Segm2;
    case 3:
      return *new (alloc) ThirdOrderLineSegment;
    }
  }

  throw "could not determine a finite element";
}

/*
    register fe-spaces
    Object of type MyFESpace can be defined in the pde-file via
    "define fespace v -type=myfespace"
  */

static RegisterFESpace<MyFESpace> initifes("myfespace");
} // namespace ngcomp

void ExportMyFESpace(py::module m)
{
  using namespace ngcomp;
  /*
    We just export the class here and use the FESpace constructor to create our space.
    This has the advantage, that we do not need to specify all the flags to parse (like
    dirichlet, definedon,...), but we can still append new functions only for that space.
  */
  auto docu = MyFESpace::GetDocu();
  auto myfes = py::class_<MyFESpace, shared_ptr<MyFESpace>, FESpace>(m, "MyFESpace", (docu.short_docu + "\n\n" + docu.long_docu).c_str());
  myfes
      /*
       this is optional, if you don't write an init function, you can create your fespace
       with FESpace("myfes",mesh,...), but it's nicer to write MyFESpace(mesh,...) ;)
    */
      .def(py::init([myfes](shared_ptr<MeshAccess> ma, py::kwargs kwa) {
             py::list info;
             info.append(ma);
             auto flags = CreateFlagsFromKwArgs(myfes, kwa, info);
             auto fes = make_shared<MyFESpace>(ma, flags);
             LocalHeap glh(100000000, "init-fes-lh");
             fes->Update(glh);
             fes->FinalizeUpdate(glh);
             return fes;
           }),
           py::arg("mesh"))
      /*
      this is, so that we do not get an 'undocumented flag' warning
    */
      .def_static("__flags_doc__", [docu]() {
        auto doc = py::cast<py::dict>(py::module::import("ngsolve").attr("FESpace").attr("__flags_doc__")());
        for (auto &flagdoc : docu.arguments)
          doc[get<0>(flagdoc).c_str()] = get<1>(flagdoc);
        return doc;
      })
      .def("GetNVert", &MyFESpace::GetNVert);
}
