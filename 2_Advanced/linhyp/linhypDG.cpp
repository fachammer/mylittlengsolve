
/*
  
  Solver for the linear hyperbolic equation

  du/dt  +  div (b u) = 0

  by an explicit time-stepping method

*/

#include <solve.hpp>
#include <python_ngstd.hpp>
#include <general/seti.hpp>

using namespace ngsolve;
using ngfem::ELEMENT_TYPE;

template <int D>
class Convection
{
protected:
  shared_ptr<L2HighOrderFESpace> finiteElementSpace;
  shared_ptr<CoefficientFunction> flowCoefficientFunction;

  class FacetData
  {
  public:
    size_t adjacentElementNumbers[2];
    int elementLocalFacetNumber[2];
    int facetId;
    Array<int> adjacentRanks;
    Vector<> InnerProductOfFlowWithNormalVectorAtIntegrationPoint;
    FacetData() = default;
    FacetData(int facetId, size_t numberOfIntegrationPoints) : facetId(facetId), InnerProductOfFlowWithNormalVectorAtIntegrationPoint(numberOfIntegrationPoints) { ; }
  };

  class ElementData
  {
  public:
    MatrixFixWidth<D> flowAtIntegrationPoint;
    ElementData() = default;
    ElementData(size_t ndof, size_t numberOfIntegrationPoints) : flowAtIntegrationPoint(numberOfIntegrationPoints) { ; }
  };

  Array<int> adjacentRanks;
  Array<Array<FacetData>> sharedFacetsDataByRank;
  Array<FacetData> boundaryFacetsData;
  Array<FacetData> innerFacetsData;
  Array<ElementData> elementsData;

public:
  Convection(shared_ptr<FESpace> finiteElementSpace, shared_ptr<CoefficientFunction> aflow)
      : finiteElementSpace(dynamic_pointer_cast<L2HighOrderFESpace>(finiteElementSpace)), flowCoefficientFunction(aflow)
  {
    LocalHeap localHeap(1000000);
    shared_ptr<MeshAccess> meshAccess = this->finiteElementSpace->GetMeshAccess();
    elementsData.SetAllocSize(meshAccess->GetNE());

    if (!this->finiteElementSpace->AllDofsTogether())
      throw Exception("mlngs-Convection needs 'all_dofs_together=True' for L2-FESpace");

    for (auto elementId : meshAccess->Elements())
    {
      HeapReset heapReset(localHeap);

      auto &finiteElement = dynamic_cast<const DGFiniteElement<D> &>(this->finiteElementSpace->GetFE(elementId, localHeap));
      const IntegrationRule integrationRule(finiteElement.ElementType(), 2 * finiteElement.Order());

      const_cast<DGFiniteElement<D> &>(finiteElement).PrecomputeShapes(integrationRule);
      const_cast<DGFiniteElement<D> &>(finiteElement).PrecomputeTrace();

      auto &elementTransform = meshAccess->GetTrafo(elementId, localHeap);
      MappedIntegrationRule<D, D> mappedIntegrationRule(integrationRule, elementTransform, localHeap);

      ElementData elementData(finiteElement.GetNDof(), integrationRule.Size());
      flowCoefficientFunction->Evaluate(mappedIntegrationRule, elementData.flowAtIntegrationPoint);
      for (size_t j = 0; j < integrationRule.Size(); j++)
      {
        Vec<D> flow = mappedIntegrationRule[j].GetJacobianInverse() * elementData.flowAtIntegrationPoint.Row(j);
        flow *= mappedIntegrationRule[j].GetWeight(); // weight times Jacobian
        elementData.flowAtIntegrationPoint.Row(j) = flow;
      }

      elementsData.Append(move(elementData));
    }

    Array<int> facetElementNumbers, facetNumbers, vertexNumbers;
    Array<FacetData> facetsData(meshAccess->GetNFacets());

    for (auto i : Range(meshAccess->GetNFacets()))
    {
      HeapReset heapReset(localHeap);

      const DGFiniteElement<D - 1> &finiteElementFacet =
          dynamic_cast<const DGFiniteElement<D - 1> &>(this->finiteElementSpace->GetFacetFE(i, localHeap));
      IntegrationRule facetIntegrationRule(finiteElementFacet.ElementType(), 2 * finiteElementFacet.Order());
      const_cast<DGFiniteElement<D - 1> &>(finiteElementFacet).PrecomputeShapes(facetIntegrationRule);

      FacetData facetData(i, facetIntegrationRule.Size());

      meshAccess->GetFacetElements(i, facetElementNumbers);

      facetData.adjacentElementNumbers[1] = -1;
      for (size_t j : Range(facetElementNumbers))
      {
        facetData.adjacentElementNumbers[j] = facetElementNumbers[j];
        auto facetNumbers = meshAccess->GetElFacets(ElementId(VOL, facetElementNumbers[j]));
        facetData.elementLocalFacetNumber[j] = facetNumbers.Pos(i);
      }

      ELEMENT_TYPE elementType = meshAccess->GetElType(ElementId(VOL, facetElementNumbers[0]));

      vertexNumbers = meshAccess->GetElVertices(ElementId(VOL, facetElementNumbers[0]));
      Facet2ElementTrafo transform(elementType, vertexNumbers);
      FlatVec<D> referenceNormal = ElementTopology::GetNormals(elementType)[facetData.elementLocalFacetNumber[0]];

      size_t numberOfIntegrationPoints = facetIntegrationRule.Size();

      // transform facet coordinates to element coordinates
      IntegrationRule &elementIntegrationRule = transform(facetData.elementLocalFacetNumber[0], facetIntegrationRule, localHeap);
      MappedIntegrationRule<D, D> mappedElementIntegrationRule(elementIntegrationRule, meshAccess->GetTrafo(ElementId(VOL, facetElementNumbers[0]), localHeap), localHeap);

      FlatMatrixFixWidth<D> flowAtMappedElementIntegrationPoints(numberOfIntegrationPoints, localHeap);
      flowCoefficientFunction->Evaluate(mappedElementIntegrationRule, flowAtMappedElementIntegrationPoints);

      for (size_t j = 0; j < numberOfIntegrationPoints; j++)
      {
        Vec<D> normal = Trans(mappedElementIntegrationRule[j].GetJacobianInverse()) * referenceNormal;

        facetData.InnerProductOfFlowWithNormalVectorAtIntegrationPoint(j) = InnerProduct(normal, flowAtMappedElementIntegrationPoints.Row(j));
        facetData.InnerProductOfFlowWithNormalVectorAtIntegrationPoint(j) *= facetIntegrationRule[j].Weight() * mappedElementIntegrationRule[j].GetJacobiDet();
      }
      
      meshAccess->GetDistantProcs(NodeId(NT_EDGE, i), facetData.adjacentRanks);

      facetsData.Append(move(facetData));
    }

    Array<FacetData> sharedFacetsData;
    sharedFacetsData.SetAllocSize(NumberOfSharedFacets(facetsData));
    FindSharedFacets(facetsData, sharedFacetsData);
    adjacentRanks = FindAdjacentRanks(sharedFacetsData);
    Array<int> numberOfSharedFacetsByRank;
    numberOfSharedFacetsByRank.SetSize(adjacentRanks.Size());
    for(auto i : Range(numberOfSharedFacetsByRank.Size()))
    {
      numberOfSharedFacetsByRank[i] = 0;
    }
    FindNumberOfSharedFacetsByRank(sharedFacetsData, adjacentRanks, numberOfSharedFacetsByRank);
    sharedFacetsDataByRank.SetSize(numberOfSharedFacetsByRank.Size());
    for(auto i: Range(numberOfSharedFacetsByRank.Size()))
    {
      sharedFacetsDataByRank[i] = Array<FacetData>();
      sharedFacetsDataByRank[i].SetSize(numberOfSharedFacetsByRank[i]);
    }
    FindSharedFacetsByRank(sharedFacetsData, adjacentRanks, sharedFacetsDataByRank);
    innerFacetsData.SetAllocSize(NumberOfInnerFacets(facetsData));
    FindInnerFacets(facetsData, innerFacetsData);
    boundaryFacetsData.SetAllocSize(NumberOfBoundaryFacets(facetsData));
    FindBoundaryFacets(facetsData, boundaryFacetsData);
  }

  void Apply(BaseVector &_vecu, BaseVector &_conv)
  {
    static Timer timer("Convection::Apply");
    RegionTimer reg(timer);
    LocalHeap localHeap(1000 * 1000);

    auto concentration = _vecu.FV<double>();
    auto convection = _conv.FV<double>();

    auto meshAccess = this->finiteElementSpace->GetMeshAccess();

    ParallelFor(Range(meshAccess->GetNE()), [&](size_t i) {
      LocalHeap threadLocalHeap = localHeap.Split();

      auto &finiteElement = static_cast<const ScalarFiniteElement<D> &>(this->finiteElementSpace->GetFE(ElementId(VOL, i), threadLocalHeap));
      const IntegrationRule integrationRule(finiteElement.ElementType(), 2 * finiteElement.Order());

      FlatMatrixFixWidth<D> flowAtIntegrationPoint = elementsData[i].flowAtIntegrationPoint;

      /*
         // use this for time-dependent flow (updated version not yet tested)
         MappedIntegrationRule<D,D> mappedIntegrationRule(integrationRule, meshAccess->GetTrafo (i, 0, threadLocalHeap), threadLocalHeap);
         FlatMatrixFixWidth<D> flowAtIntegrationPoint(mappedIntegrationRule.Size(), threadLocalHeap);
         flowCoefficientFunction -> Evaluate (mappedIntegrationRule, flowAtIntegrationPoint);
         for (size_t j = 0; j < integrationRule.Size(); j++)
         {
         Vec<D> flow = mappedIntegrationRule[j].GetJacobianInverse() * flowAtIntegrationPoint.Row(j);
         flow *= mappedIntegrationRule[j].GetWeight();		
         flowAtIntegrationPoint.Row(j) = flow;
         }
         */

      IntRange elementDofs = this->finiteElementSpace->GetElementDofs(i);

      size_t numberOfIntegrationPoints = integrationRule.Size();
      FlatVector<> concentrationAtElementIntegrationPoints(numberOfIntegrationPoints, threadLocalHeap);
      FlatMatrixFixWidth<D> flowTimesConcentrationAtElementIntegrationPoints(numberOfIntegrationPoints, threadLocalHeap);

      finiteElement.Evaluate(integrationRule, concentration.Range(elementDofs), concentrationAtElementIntegrationPoints);

      for (auto k : Range(numberOfIntegrationPoints))
        flowTimesConcentrationAtElementIntegrationPoints.Row(k) = concentrationAtElementIntegrationPoints(k) * flowAtIntegrationPoint.Row(k);

      finiteElement.EvaluateGradTrans(integrationRule, flowTimesConcentrationAtElementIntegrationPoints, convection.Range(elementDofs));
    });

    static mutex add_mutex;

    ParallelFor(Range(innerFacetsData.Size()), [&](size_t i) {
      LocalHeap threadLocalHeap = localHeap.Split();
      const FacetData& facetData = innerFacetsData[i];
      
      // internal facet
      const DGFiniteElement<D> &finiteElement1 =
          static_cast<const DGFiniteElement<D> &>(this->finiteElementSpace->GetFE(ElementId(VOL, facetData.adjacentElementNumbers[0]), threadLocalHeap));
      const DGFiniteElement<D> &finiteElement2 =
          static_cast<const DGFiniteElement<D> &>(this->finiteElementSpace->GetFE(ElementId(VOL, facetData.adjacentElementNumbers[1]), threadLocalHeap));
      
      const DGFiniteElement<D - 1> &finiteElementFacet =
          static_cast<const DGFiniteElement<D - 1> &>(this->finiteElementSpace->GetFacetFE(facetData.facetId, threadLocalHeap));


      IntRange dofNumbers1 = this->finiteElementSpace->GetElementDofs(facetData.adjacentElementNumbers[0]);
      IntRange dofNumbers2 = this->finiteElementSpace->GetElementDofs(facetData.adjacentElementNumbers[1]);


      size_t numberOfFacetDofs = finiteElementFacet.GetNDof();
      size_t numberOfDofsIn1 = finiteElement1.GetNDof();
      size_t numberOfDofsIn2 = finiteElement2.GetNDof();

      FlatVector<> convectionCoefficients1(numberOfDofsIn1, threadLocalHeap), convectionCoefficients2(numberOfDofsIn2, threadLocalHeap);
      FlatVector<> traceCoefficients1(numberOfFacetDofs, threadLocalHeap), traceCoefficients2(numberOfFacetDofs, threadLocalHeap);

      finiteElement1.GetTrace(facetData.elementLocalFacetNumber[0], concentration.Range(dofNumbers1), traceCoefficients1);
      finiteElement2.GetTrace(facetData.elementLocalFacetNumber[1], concentration.Range(dofNumbers2), traceCoefficients2);

      IntegrationRule facetIntegrationRule(finiteElementFacet.ElementType(), 2 * finiteElementFacet.Order());
      size_t numberOfIntegrationPoints = facetIntegrationRule.Size();

      FlatVector<> InnerProductOfFlowWithNormalVectorAtIntegrationPoint = facetData.InnerProductOfFlowWithNormalVectorAtIntegrationPoint;

      FlatVector<> traceAtIntegrationPoints1(numberOfIntegrationPoints, threadLocalHeap), traceAtIntegrationPoints2(numberOfIntegrationPoints, threadLocalHeap);
      FlatVector<> upwindTraceAtIntegrationPoints(numberOfIntegrationPoints, threadLocalHeap);

      finiteElementFacet.Evaluate(facetIntegrationRule, traceCoefficients1, traceAtIntegrationPoints1);
      finiteElementFacet.Evaluate(facetIntegrationRule, traceCoefficients2, traceAtIntegrationPoints2);

      for (size_t j = 0; j < numberOfIntegrationPoints; j++)
      {
        upwindTraceAtIntegrationPoints(j) = InnerProductOfFlowWithNormalVectorAtIntegrationPoint(j) * ((InnerProductOfFlowWithNormalVectorAtIntegrationPoint(j) > 0) ? traceAtIntegrationPoints1(j) : traceAtIntegrationPoints2(j));
      }

      finiteElementFacet.EvaluateTrans(facetIntegrationRule, upwindTraceAtIntegrationPoints, traceCoefficients1);
      finiteElement1.GetTraceTrans(facetData.elementLocalFacetNumber[0], traceCoefficients1, convectionCoefficients1);
      finiteElement2.GetTraceTrans(facetData.elementLocalFacetNumber[1], traceCoefficients1, convectionCoefficients2);

      {
        lock_guard<mutex> guard(add_mutex);
        convection.Range(dofNumbers1) -= convectionCoefficients1;
        convection.Range(dofNumbers2) += convectionCoefficients2;
      }
    });

    ParallelFor(Range(boundaryFacetsData.Size()), [&](size_t i) {
      LocalHeap threadLocalHeap = localHeap.Split();
      const FacetData& facetData = boundaryFacetsData[i];
      // boundary facet
      const DGFiniteElement<D> &finiteElement =
          dynamic_cast<const DGFiniteElement<D> &>(this->finiteElementSpace->GetFE(ElementId(VOL, facetData.adjacentElementNumbers[0]), threadLocalHeap));
      const DGFiniteElement<D - 1> &finiteElementFacet =
          dynamic_cast<const DGFiniteElement<D - 1> &>(this->finiteElementSpace->GetFacetFE(facetData.facetId, threadLocalHeap));

      IntRange dofNumbers = this->finiteElementSpace->GetElementDofs(facetData.adjacentElementNumbers[0]);

      size_t numberOfFacetDofs = finiteElementFacet.GetNDof();
      size_t numberOfElementDofs = finiteElement.GetNDof();

      FlatVector<> convectionCoefficients(numberOfElementDofs, threadLocalHeap);
      FlatVector<> traceCoefficients(numberOfFacetDofs, threadLocalHeap);

      finiteElement.GetTrace(facetData.elementLocalFacetNumber[0], concentration.Range(dofNumbers), traceCoefficients);

      IntegrationRule facetIntegrationRule(finiteElementFacet.ElementType(), 2 * finiteElementFacet.Order());
      size_t numberOfIntegrationPoints = facetIntegrationRule.Size();

      FlatVector<> InnerProductOfFlowWithNormalVectorAtIntegrationPoint = facetData.InnerProductOfFlowWithNormalVectorAtIntegrationPoint;
      FlatVector<> traceAtIntegrationPoints(numberOfIntegrationPoints, threadLocalHeap), upwindTraceAtIntegrationPoints(numberOfIntegrationPoints, threadLocalHeap);

      finiteElementFacet.Evaluate(facetIntegrationRule, traceCoefficients, traceAtIntegrationPoints);

      for (size_t j = 0; j < numberOfIntegrationPoints; j++)
        upwindTraceAtIntegrationPoints(j) = InnerProductOfFlowWithNormalVectorAtIntegrationPoint(j) * ((InnerProductOfFlowWithNormalVectorAtIntegrationPoint(j) > 0) ? traceAtIntegrationPoints(j) : 0);

      finiteElementFacet.EvaluateTrans(facetIntegrationRule, upwindTraceAtIntegrationPoints, traceCoefficients);
      finiteElement.GetTraceTrans(facetData.elementLocalFacetNumber[0], traceCoefficients, convectionCoefficients);

      {
        lock_guard<mutex> guard(add_mutex);
        convection.Range(dofNumbers) -= convectionCoefficients;
      }
    });
  }

  private:

    static bool IsInnerFacet(const FacetData& facet)
    {
      return facet.adjacentRanks.Size() == 0 && facet.adjacentElementNumbers[1] != -1;
    }

    static bool NumberOfInnerFacets(const Array<FacetData>& facets)
    {
      int numberOfInnerFacets = 0;
      for(auto facet: facets)
      {
        if(IsInnerFacet(facet))
        {
          numberOfInnerFacets++;
        }
      }
      return numberOfInnerFacets;
    }

    static void FindInnerFacets(const Array<FacetData>& facets, Array<FacetData>& innerFacets)
    {
      for(auto facet: facets)
      {
        if(IsInnerFacet(facet))
        {
          innerFacets.Append(move(facet));
        }
      }
    }

    static bool IsBoundaryFacet(const FacetData& facet)
    {
      return facet.adjacentRanks.Size() == 0 && facet.adjacentElementNumbers[1] == -1;
    }

    static int NumberOfBoundaryFacets(const Array<FacetData>& facets) 
    {
      int numberOfBoundaryFacets = 0;
      for(auto facet: facets)
      {
        if(IsBoundaryFacet(facet))
        {
          numberOfBoundaryFacets++;
        }
      }
      return numberOfBoundaryFacets;
    }

    static void FindBoundaryFacets(const Array<FacetData>& facets, Array<FacetData>& boundaryFacets)
    {
      for(auto facet: facets)
      {
        if(IsBoundaryFacet(facet))
        {
          boundaryFacets.Append(move(facet));
        }
      }
    }

    static bool IsSharedFacet(const FacetData& facet)
    {
      return facet.adjacentRanks.Size() > 0;
    }

    static int NumberOfSharedFacets(const Array<FacetData>& facets)
    {
      int numberOfSharedFacets = 0;
      for(auto facet: facets)
      {
        if(IsSharedFacet(facet))
        {
          numberOfSharedFacets++;
        }
      }
      return numberOfSharedFacets;
    }

    static void FindSharedFacets(const Array<FacetData>& facets, Array<FacetData>& sharedFacets)
    {
      for(auto facet : facets)
      {
        if(IsSharedFacet(facet))
        {
          sharedFacets.Append(move(facet));
        }
      }
    }

    Array<int> FindAdjacentRanks(const Array<FacetData>& sharedFacets)
    {
      netgen::IndexSet adjacentRanksSet(MyMPI_GetNTasks());
      for(auto facet : sharedFacets)
      {
        adjacentRanksSet.Add(facet.adjacentRanks[0]);
      }
      const netgen::Array<int>& a = adjacentRanksSet.GetArray();
      Array<int> adjacentRanks;
      adjacentRanks.SetAllocSize(a.Size());
      for(int i = 0; i < a.Size(); i++)
      {
        adjacentRanks.Append(a[i]);
      }
      return adjacentRanks;
    }

    void FindNumberOfSharedFacetsByRank(const Array<FacetData>& sharedFacets, const Array<int>& adjacentRanks, Array<int>& numberOfSharedFacets)
    {
      for(auto facetData: sharedFacets)
      {
        int position = adjacentRanks.Pos(facetData.adjacentRanks[0]);
        numberOfSharedFacets[position]++;
      }
    }

    void FindSharedFacetsByRank(const Array<FacetData>& sharedFacets, const Array<int>& adajcentRanks, Array<Array<FacetData>>& sharedFacetsByRank)
    {
      for(auto facetData: sharedFacets)
      {
        int position = adjacentRanks.Pos(facetData.adjacentRanks[0]);
        sharedFacetsByRank[position].Append(move(facetData));
      }
    }
};

PYBIND11_MODULE(liblinhyp, m)
{
  py::class_<Convection<2>>(m, "Convection")
      .def(py::init<shared_ptr<FESpace>, shared_ptr<CoefficientFunction>>())
      .def("Apply", &Convection<2>::Apply);
}
