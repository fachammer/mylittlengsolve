/*********************************************************************/
/* File:   myElement.cpp                                             */
/* Author: Joachim Schoeberl                                         */
/* Date:   26. Apr. 2009                                             */
/*********************************************************************/

/*

My own simple first and second order triangular finite elements

*/

#include <fem.hpp>
#include <python_ngstd.hpp>
#include "myElement.hpp"

namespace ngfem
{

MyLinearTrig ::MyLinearTrig()
    /*
    Call constructor for base class:
    geometry is ET_TRIG, number of dofs is 3, maximal order is 1
   */
    : ScalarFiniteElement<2>(3, 1)
{
  ;
}

void MyLinearTrig ::CalcShape(const IntegrationPoint &ip,
                              BareSliceVector<> shape) const
{
  // coordinates in reference elements
  double x = ip(0);
  double y = ip(1);

  /*
      Vertex coordinates have been defined to be (1,0), (0,1), (0,0)
      see ElementTopology::GetVertices(ET_TRIG)
     */

  // define shape functions
  shape(0) = x;
  shape(1) = y;
  shape(2) = 1 - x - y;
}

void MyLinearTrig ::CalcDShape(const IntegrationPoint &ip,
                               BareSliceMatrix<> dshape) const

{
  // matrix of derivatives:

  dshape(0, 0) = 1;
  dshape(0, 1) = 0;
  dshape(1, 0) = 0;
  dshape(1, 1) = 1;
  dshape(2, 0) = -1;
  dshape(2, 1) = -1;
}

MyQuadraticTrig ::MyQuadraticTrig()
    : ScalarFiniteElement<2>(6, 2)
{
  ;
}

void MyQuadraticTrig ::CalcShape(const IntegrationPoint &ip,
                                 BareSliceVector<> shape) const
{
  // now, use barycentric coordinates x, y, 1-x-y:
  double lam[3] = {ip(0), ip(1), 1 - ip(0) - ip(1)};

  // vertex basis functions:
  for (int i = 0; i < 3; i++)
    shape(i) = lam[i] * (2 * lam[i] - 1);

  // edge basis functions:

  const EDGE *edges = ElementTopology::GetEdges(ET_TRIG);
  // table provides connection of edges and vertices
  // i-th edge is between vertex edges[i][0] and edges[i][1]

  for (int i = 0; i < 3; i++)
    shape(3 + i) = 4 * lam[edges[i][0]] * lam[edges[i][1]];
}

void MyQuadraticTrig ::CalcDShape(const IntegrationPoint &ip,
                                  BareSliceMatrix<> dshape) const

{
  // Use automatic (exact !) differentiation with overloaded data-types

  AutoDiff<2> x(ip(0), 0); // value of x, gradient is 0-th unit vector (1,0)
  AutoDiff<2> y(ip(1), 1); // value of y, gradient is 1-th unit vector (0,1)

  AutoDiff<2> lam[3] = {x, y, 1 - x - y};

  // vertex basis functions:
  for (int i = 0; i < 3; i++)
  {
    AutoDiff<2> shape = lam[i] * (2 * lam[i] - 1);
    dshape(i, 0) = shape.DValue(0); // x-derivative
    dshape(i, 1) = shape.DValue(1); // y-derivative
  }

  // edge basis functions:
  const EDGE *edges = ElementTopology::GetEdges(ET_TRIG);

  for (int i = 0; i < 3; i++)
  {
    AutoDiff<2> shape = 4 * lam[edges[i][0]] * lam[edges[i][1]];
    dshape(3 + i, 0) = shape.DValue(0); // x-derivative
    dshape(3 + i, 1) = shape.DValue(1); // y-derivative
  }
}

BilinearQuadElement ::BilinearQuadElement() : ScalarFiniteElement<2>(4, 1) {}

void BilinearQuadElement::CalcShape(const IntegrationPoint &ip, BareSliceVector<> shape) const
{
  double x = ip(0);
  double y = ip(1);

  shape(0) = (1 - x) * (1 - y); // (0,0)
  shape(1) = x * (1 - y);       // (1,0)
  shape(2) = x * y;             // (1,1)
  shape(3) = (1 - x) * y;       // (0,1)
}

void BilinearQuadElement::CalcDShape(const IntegrationPoint &ip, BareSliceMatrix<> dshape) const
{
  double x = ip(0);
  double y = ip(1);

  dshape(0, 0) = -1 * (1 - y);
  dshape(0, 1) = (1 - x) * (-1);
  dshape(1, 0) = 1 - y;
  dshape(1, 1) = x * (-1);
  dshape(2, 0) = y;
  dshape(2, 1) = x;
  dshape(3, 0) = (-1) * y;
  dshape(3, 1) = (1 - x);
}

ThirdOrderTriangleElement::ThirdOrderTriangleElement() : ScalarFiniteElement<2>(10, 3) {}

std::array<AutoDiff<2>, 10> GetThirdOrderTriangleBasisFunctions(const IntegrationPoint &ip) {
  AutoDiff<2> x(ip(0), 0);
  AutoDiff<2> y(ip(1), 1);

  AutoDiff<2> b[3] = { x, y, 1 - x - y };
  return {
    (9.0/2) * b[0] * (1.0/3 - b[0]) * (2.0/3 - b[0]),
    (9.0/2) * b[1] * (1.0/3 - b[1]) * (2.0/3 - b[1]),
    (9.0/2) * b[2] * (1.0/3 - b[2]) * (2.0/3 - b[2]),
    (27.0/2) * b[2] * b[0] * (b[2] - 1.0/3),
    (27.0/2) * b[2] * b[0] * (b[0] - 1.0/3),
    (27.0/2) * b[1] * b[2] * (b[1] - 1.0/3),
    (27.0/2) * b[1] * b[2] * (b[2] - 1.0/3),
    (27.0/2) * b[0] * b[1] * (b[0] - 1.0/3),
    (27.0/2) * b[0] * b[1] * (b[1] - 1.0/3),
    27.0 * b[0] * b[1] * b[2]
  };
}

void ThirdOrderTriangleElement::CalcShape(const IntegrationPoint &ip, BareSliceVector<> shape) const
{
  auto basis = GetThirdOrderTriangleBasisFunctions(ip);
  for(int i = 0; i < 10; i++)
    shape(i) = basis[i].Value();
}

void ThirdOrderTriangleElement::CalcDShape(const IntegrationPoint &ip, BareSliceMatrix<> dshape) const
{
  auto basis = GetThirdOrderTriangleBasisFunctions(ip);
  for(int i = 0; i < 10; i++)
  {
    dshape(i, 0) = basis[i].DValue(0);
    dshape(i, 1) = basis[i].DValue(1);
  }
}

} // namespace ngfem

void ExportMyElement(py::module m)
{
  using namespace ngfem;
  /*
    Our Trig is derived
    from the classes ScalarFiniteElement<2> -> BaseScalarFiniteElement -> FiniteElement.
    Only BaseScalarFiniteElement and FiniteElement are exported to python
    (see ngsolve/fem/python_fem.cpp), so we derive from BaseScalarFiniteElement (which derives from
    FiniteElement).
    If we only want to use it in our FESpace we do not need the Python hierarchy, but it's nice for
    debugging :)
  */
  py::class_<MyLinearTrig, shared_ptr<MyLinearTrig>, BaseScalarFiniteElement>(m, "MyLinearTrig", "My new linear Trig")
      .def(py::init<>());
}
