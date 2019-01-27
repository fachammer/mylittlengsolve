#include <utility>


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

template<int D>
class Convection {
protected:
    shared_ptr<L2HighOrderFESpace> finiteElementSpace;
    shared_ptr<CoefficientFunction> flowCoefficientFunction;

    class FacetData {
    public:
        int adjacentElementNumbers[2];
        int elementLocalFacetNumber[2];
        int facetId;
        Array<int> adjacentRanks;
        Vector<> innerProductOfFlowWithNormalVectorAtIntegrationPoint;

        FacetData() = default;

        FacetData(int facetId, size_t numberOfIntegrationPoints) : facetId(facetId),
                                                                   innerProductOfFlowWithNormalVectorAtIntegrationPoint(
                                                                           numberOfIntegrationPoints) { ; }
    };

    class ElementData {
    public:
        MatrixFixWidth<D> flowAtIntegrationPoint;

        ElementData() = default;

        ElementData(size_t numberOfIntegrationPoints) : flowAtIntegrationPoint(numberOfIntegrationPoints) { ; }
    };

    Array<int> adjacentRanks;
    Array<Array<FacetData>> sharedFacetsDataByRank;
    Array<Array<double>> receivedData;
    Array<FacetData> boundaryFacetsData;
    Array<FacetData> innerFacetsData;
    Array<ElementData> elementsData;
    shared_ptr<MeshAccess> meshAccess;

public:
    Convection(const shared_ptr<FESpace>& finiteElementSpace, shared_ptr<CoefficientFunction> aflow)
            : finiteElementSpace(dynamic_pointer_cast<L2HighOrderFESpace>(finiteElementSpace)),
              flowCoefficientFunction(aflow),
              meshAccess(finiteElementSpace->GetMeshAccess()) {
        LocalHeap localHeap(1000000);
        elementsData.SetAllocSize(meshAccess->GetNE());

        if (!this->finiteElementSpace->AllDofsTogether())
            throw Exception("mlngs-Convection needs 'all_dofs_together=True' for L2-FESpace");

        PrecomputeElements(localHeap);
        PrecomputeFacets(localHeap);
    }

private:
    void PrecomputeElements(LocalHeap& localHeap) {
        for (auto elementId : meshAccess->Elements()) {
            HeapReset heapReset(localHeap);

            auto& finiteElement = dynamic_cast<const DGFiniteElement<D>&>(finiteElementSpace->GetFE(elementId,
                                                                                                    localHeap));
            const IntegrationRule integrationRule(finiteElement.ElementType(), 2 * finiteElement.Order());

            const_cast<DGFiniteElement<D>&>(finiteElement).PrecomputeShapes(integrationRule);
            const_cast<DGFiniteElement<D>&>(finiteElement).PrecomputeTrace();

            auto& elementTransform = meshAccess->GetTrafo(elementId, localHeap);
            MappedIntegrationRule<D, D> mappedIntegrationRule(integrationRule, elementTransform, localHeap);

            ElementData elementData(integrationRule.Size());
            flowCoefficientFunction->Evaluate(mappedIntegrationRule, elementData.flowAtIntegrationPoint);
            for (size_t j = 0; j < integrationRule.Size(); j++) {
                Vec<D> flow =
                        mappedIntegrationRule[j].GetJacobianInverse() * elementData.flowAtIntegrationPoint.Row(j);
                flow *= mappedIntegrationRule[j].GetWeight(); // weight times Jacobian
                elementData.flowAtIntegrationPoint.Row(j) = flow;
            }

            elementsData.Append(move(elementData));
        }
    }

    void PrecomputeFacets(LocalHeap& localHeap) {
        Array<int> facetElementNumbers, facetNumbers, vertexNumbers;
        Array<FacetData> facetsData(meshAccess->GetNFacets());

        for (auto i : Range(meshAccess->GetNFacets())) {
            HeapReset heapReset(localHeap);

            const auto& finiteElementFacet =
                    dynamic_cast<const DGFiniteElement<D - 1>&>(finiteElementSpace->GetFacetFE(i, localHeap));
            IntegrationRule facetIntegrationRule = GetFacetIntegrationRule(finiteElementFacet);
            const_cast<DGFiniteElement<D - 1>&>(finiteElementFacet).PrecomputeShapes(facetIntegrationRule);

            FacetData facetData(i, facetIntegrationRule.Size());
            meshAccess->GetFacetElements(i, facetElementNumbers);

            facetData.adjacentElementNumbers[1] = -1;
            for (size_t j : Range(facetElementNumbers)) {
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
            IntegrationRule& elementIntegrationRule = transform(facetData.elementLocalFacetNumber[0],
                                                                facetIntegrationRule, localHeap);
            MappedIntegrationRule<D, D> mappedElementIntegrationRule(elementIntegrationRule, meshAccess->GetTrafo(
                    ElementId(VOL, facetElementNumbers[0]), localHeap), localHeap);

            FlatMatrixFixWidth<D> flowAtMappedElementIntegrationPoints(numberOfIntegrationPoints, localHeap);
            flowCoefficientFunction->Evaluate(mappedElementIntegrationRule, flowAtMappedElementIntegrationPoints);

            for (size_t j = 0; j < numberOfIntegrationPoints; j++) {
                Vec<D> normal = Trans(mappedElementIntegrationRule[j].GetJacobianInverse()) * referenceNormal;

                facetData.innerProductOfFlowWithNormalVectorAtIntegrationPoint(j) = InnerProduct(normal,
                                                                                                 flowAtMappedElementIntegrationPoints.Row(
                                                                                                         j));
                facetData.innerProductOfFlowWithNormalVectorAtIntegrationPoint(j) *=
                        facetIntegrationRule[j].Weight() * mappedElementIntegrationRule[j].GetJacobiDet();
            }

            meshAccess->GetDistantProcs(NodeId(StdNodeType(NT_FACET, meshAccess->GetDimension()), i), facetData.adjacentRanks);

            facetsData.Append(move(facetData));
        }

        Array<FacetData> sharedFacetsData;
        sharedFacetsData.SetAllocSize(NumberOfSharedFacets(facetsData));
        FindSharedFacets(facetsData, sharedFacetsData);
        adjacentRanks = FindAdjacentRanks(sharedFacetsData);
        Array<size_t> numberOfSharedFacetsByRank(adjacentRanks.Size());
        FindNumberOfSharedFacetsByRank(sharedFacetsData, adjacentRanks, numberOfSharedFacetsByRank);
        sharedFacetsDataByRank.SetSize(numberOfSharedFacetsByRank.Size());
        for (auto i: Range(numberOfSharedFacetsByRank.Size())) {
            sharedFacetsDataByRank[i].SetAllocSize(numberOfSharedFacetsByRank[i]);
        }
        FindSharedFacetsByRank(sharedFacetsData, adjacentRanks, sharedFacetsDataByRank);
        innerFacetsData.SetAllocSize(NumberOfInnerFacets(facetsData));
        FindInnerFacets(facetsData, innerFacetsData);
        boundaryFacetsData.SetAllocSize(NumberOfBoundaryFacets(facetsData));
        FindBoundaryFacets(facetsData, boundaryFacetsData);
    }


    static bool IsInnerFacet(const FacetData& facet) {
        return facet.adjacentRanks.Size() == 0 && facet.adjacentElementNumbers[1] != -1;
    }

    static size_t NumberOfInnerFacets(const Array<FacetData>& facets) {
        size_t numberOfInnerFacets = 0;
        for (auto facet: facets) {
            if (IsInnerFacet(facet)) {
                numberOfInnerFacets++;
            }
        }
        return numberOfInnerFacets;
    }

    static void FindInnerFacets(const Array<FacetData>& facets, Array<FacetData>& innerFacets) {
        for (auto facet: facets) {
            if (IsInnerFacet(facet)) {
                innerFacets.Append(move(facet));
            }
        }
    }

    static bool IsBoundaryFacet(const FacetData& facet) {
        return facet.adjacentRanks.Size() == 0 && facet.adjacentElementNumbers[1] == -1;
    }

    static size_t NumberOfBoundaryFacets(const Array<FacetData>& facets) {
        size_t numberOfBoundaryFacets = 0;
        for (auto facet: facets) {
            if (IsBoundaryFacet(facet)) {
                numberOfBoundaryFacets++;
            }
        }
        return numberOfBoundaryFacets;
    }

    static void FindBoundaryFacets(const Array<FacetData>& facets, Array<FacetData>& boundaryFacets) {
        for (auto facet: facets) {
            if (IsBoundaryFacet(facet)) {
                boundaryFacets.Append(move(facet));
            }
        }
    }

    static bool IsSharedFacet(const FacetData& facet) {
        return facet.adjacentRanks.Size() > 0;
    }

    static size_t NumberOfSharedFacets(const Array<FacetData>& facets) {
        size_t numberOfSharedFacets = 0;
        for (auto facet: facets) {
            if (IsSharedFacet(facet)) {
                numberOfSharedFacets++;
            }
        }
        return numberOfSharedFacets;
    }

    static void FindSharedFacets(const Array<FacetData>& facets, Array<FacetData>& sharedFacets) {
        for (auto facet : facets) {
            if (IsSharedFacet(facet)) {
                sharedFacets.Append(move(facet));
            }
        }
    }

    Array<size_t> FindAdjacentRanks(const Array<FacetData>& sharedFacets) {
        netgen::IndexSet adjacentRanksSet(MyMPI_GetNTasks());
        for (auto facet : sharedFacets) {
            adjacentRanksSet.Add(facet.adjacentRanks[0]);
        }
        const netgen::Array<int>& a = adjacentRanksSet.GetArray();
        Array<size_t> adjacentRanks;
        adjacentRanks.SetAllocSize(a.Size());
        for (int i = 0; i < a.Size(); i++) {
            adjacentRanks.Append(static_cast<size_t&&>(a[i]));
        }
        return adjacentRanks;
    }

    void FindNumberOfSharedFacetsByRank(const Array<FacetData>& sharedFacets, const Array<int>& adjacentRanks,
                                        Array<size_t>& numberOfSharedFacets) {
        for (auto i : Range(numberOfSharedFacets.Size())) {
            numberOfSharedFacets[i] = 0;
        }

        for (auto facetData: sharedFacets) {
            size_t position = adjacentRanks.Pos(facetData.adjacentRanks[0]);
            numberOfSharedFacets[position]++;
        }
    }

    void FindSharedFacetsByRank(const Array<FacetData>& sharedFacets, const Array<int>& adajcentRanks,
                                Array<Array<FacetData>>& sharedFacetsByRank) {
        for (auto facetData: sharedFacets) {
            size_t position = adjacentRanks.Pos(facetData.adjacentRanks[0]);
            sharedFacetsByRank[position].Append(move(facetData));
        }
    }

public:
    void Apply(BaseVector& _vecu, BaseVector& _conv) {
        static Timer timer("Convection::Apply");
        RegionTimer reg(timer);
        LocalHeap localHeap(1000 * 1000);

        auto concentration = _vecu.FV<double>();
        auto convection = _conv.FV<double>();

        Array<MPI_Request> mpiRequests;
        mpiRequests.SetAllocSize(adjacentRanks.Size());
        SendSharedFacetData(concentration, mpiRequests, localHeap);
        ApplyToElements(concentration, convection, localHeap);
        ApplyToInnerFacets(concentration, convection, localHeap);
        ApplyToBoundaryFacets(concentration, convection, localHeap);
        ApplyToSharedFacets(mpiRequests, concentration, convection, localHeap);
    }

private:
    void SendSharedFacetData(const FlatVector<double>& concentration,
                             Array<MPI_Request>& mpiRequests,
                             const LocalHeap& localHeap) {
        Array<MPI_Request> sendRequests(adjacentRanks.Size());
        receivedData.SetSize(adjacentRanks.Size());
        ParallelFor(Range(adjacentRanks.Size()), [&](size_t i) {
            LocalHeap threadLocalHeap = localHeap.Split();
            int destination = adjacentRanks[i];
            Array<double> data;
            for (auto j: Range(sharedFacetsDataByRank[i].Size())) {
                const FacetData& sharedFacet = sharedFacetsDataByRank[i][j];
                FlatVector<> trace = GetTraceValuesOnFacetFromElementCoefficients(
                        concentration,
                        sharedFacet.adjacentElementNumbers[0],
                        sharedFacet.elementLocalFacetNumber[0],
                        threadLocalHeap);
                data.Append(FlatVectorToFlatArray(trace, threadLocalHeap));
            }
            MPI_Request sendRequest = MyMPI_ISend(data, destination);
            sendRequests[i] = sendRequest;
            receivedData[i].SetSize(data.Size());
            MPI_Request receiveRequest = MyMPI_IRecv(receivedData[i], destination);
            mpiRequests.Append(receiveRequest);
        });

        // MyMPI_WaitAll(sendRequests);
    }

    void ApplyToBoundaryFacets(const FlatVector<double>& concentration,
                               const FlatVector<double>& convection,
                               const LocalHeap& localHeap) const {
        static mutex add_mutex;
        ParallelFor(Range(boundaryFacetsData.Size()), [&](size_t i) {
            LocalHeap threadLocalHeap = localHeap.Split();

            const FacetData& facetData = boundaryFacetsData[i];
            FlatVector<> traceOnFacet = GetTraceValuesOnFacetFromElementCoefficients(concentration,
                                                                                     facetData.adjacentElementNumbers[0],
                                                                                     facetData.elementLocalFacetNumber[0],
                                                                                     threadLocalHeap);
            FlatVector<> upwindTraceAtIntegrationPoints = GetUpwindTrace(traceOnFacet,
                                                                         [](size_t j) { return 0; },
                                                                         facetData.innerProductOfFlowWithNormalVectorAtIntegrationPoint,
                                                                         threadLocalHeap);

            FlatVector<> convectionCoefficients = GetElementCoefficientsFromTraceValuesOnFacet(
                    upwindTraceAtIntegrationPoints,
                    facetData.adjacentElementNumbers[0],
                    facetData.elementLocalFacetNumber[0],
                    threadLocalHeap);
            IntRange dofNumbers = finiteElementSpace->GetElementDofs(
                    static_cast<size_t>(facetData.elementLocalFacetNumber[0]));
            {
                lock_guard<mutex> guard(add_mutex);
                convection.Range(dofNumbers) -= convectionCoefficients;
            }
        });
    }

    void ApplyToInnerFacets(const FlatVector<double>& concentration,
                            const FlatVector<double>& convection,
                            const LocalHeap& localHeap) const {
        static mutex add_mutex;
        ParallelFor(Range(innerFacetsData.Size()), [&](size_t i) {
            LocalHeap threadLocalHeap = localHeap.Split();
            const FacetData& facetData = innerFacetsData[i];
            FlatVector<> traceAtIntegrationPoints0 = GetTraceValuesOnFacetFromElementCoefficients(
                    concentration,
                    facetData.adjacentElementNumbers[0],
                    facetData.elementLocalFacetNumber[0],
                    threadLocalHeap);
            FlatVector<> traceAtIntegrationPoints1 = GetTraceValuesOnFacetFromElementCoefficients(
                    concentration,
                    facetData.adjacentElementNumbers[1],
                    facetData.elementLocalFacetNumber[1],
                    threadLocalHeap);

            FlatVector<> upwindTraceAtIntegrationPoints = GetUpwindTrace(
                    traceAtIntegrationPoints0,
                    traceAtIntegrationPoints1,
                    facetData.innerProductOfFlowWithNormalVectorAtIntegrationPoint,
                    threadLocalHeap);

            FlatVector<> upwindTraceCoefficients = GetFacetCoefficientsFromTraceValuesOnFacet(
                    upwindTraceAtIntegrationPoints,
                    facetData.adjacentElementNumbers[0],
                    facetData.elementLocalFacetNumber[0],
                    threadLocalHeap);

            FlatVector<> convectionCoefficients0 = GetElementCoefficientsFromTraceCoefficients(
                    upwindTraceCoefficients,
                    facetData.adjacentElementNumbers[0],
                    facetData.elementLocalFacetNumber[0],
                    threadLocalHeap);
            FlatVector<> convectionCoefficients1 = GetElementCoefficientsFromTraceCoefficients(
                    upwindTraceCoefficients,
                    facetData.adjacentElementNumbers[1],
                    facetData.elementLocalFacetNumber[1],
                    threadLocalHeap);

            IntRange dofNumbers0 = finiteElementSpace->GetElementDofs(
                    static_cast<size_t>(facetData.adjacentElementNumbers[0]));
            IntRange dofNumbers1 = finiteElementSpace->GetElementDofs(
                    static_cast<size_t>(facetData.adjacentElementNumbers[1]));
            {
                lock_guard<mutex> guard(add_mutex);
                convection.Range(dofNumbers0) -= convectionCoefficients0;
                convection.Range(dofNumbers1) += convectionCoefficients1;
            }
        });
    }

    void ApplyToElements(const FlatVector<double>& concentration,
                         const FlatVector<double>& convection,
                         const LocalHeap& localHeap) const {
        ParallelFor(Range(meshAccess->GetNE()), [&](size_t i) {
            LocalHeap threadLocalHeap = localHeap.Split();

            auto& finiteElement = static_cast<const ScalarFiniteElement<D>&>(finiteElementSpace->GetFE(
                    ElementId(VOL, i), threadLocalHeap));
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

            IntRange elementDofs = finiteElementSpace->GetElementDofs(i);

            size_t numberOfIntegrationPoints = integrationRule.Size();
            FlatVector<> concentrationAtElementIntegrationPoints(numberOfIntegrationPoints, threadLocalHeap);
            FlatMatrixFixWidth<D> flowTimesConcentrationAtElementIntegrationPoints(numberOfIntegrationPoints,
                                                                                   threadLocalHeap);

            finiteElement.Evaluate(integrationRule, concentration.Range(elementDofs),
                                   concentrationAtElementIntegrationPoints);

            for (auto k : Range(numberOfIntegrationPoints))
                flowTimesConcentrationAtElementIntegrationPoints.Row(k) =
                        concentrationAtElementIntegrationPoints(k) * flowAtIntegrationPoint.Row(k);

            finiteElement.EvaluateGradTrans(integrationRule, flowTimesConcentrationAtElementIntegrationPoints,
                                            convection.Range(elementDofs));
        });
    }

    void ApplyToSharedFacets(const Array<MPI_Request>& mpiRequests,
                             const FlatVector<>& concentration,
                             const FlatVector<>& convection,
                             const LocalHeap& localHeap) {
        static mutex add_mutex;
        ParallelFor(Range(mpiRequests.Size()), [&](size_t i) {
            LocalHeap threadLocalHeap = localHeap.Split();
            MPI_Request receiveRequest = mpiRequests[i];
            int receivedIndex = MyMPI_WaitAny(mpiRequests);
            size_t startFacetIndex = 0;
            for (auto j: Range(sharedFacetsDataByRank[receivedIndex].Size())) {
                const FacetData& facetData = sharedFacetsDataByRank[receivedIndex][j];
                FlatVector<> thisTraceAtIntegrationPoints = GetTraceValuesOnFacetFromElementCoefficients(
                        concentration,
                        facetData.adjacentElementNumbers[0],
                        facetData.elementLocalFacetNumber[0],
                        threadLocalHeap);
                FlatVector<> otherTraceAtIntegrationPoints(thisTraceAtIntegrationPoints.Size(), threadLocalHeap);
                otherTraceAtIntegrationPoints = FlatArrayToFlatVector(
                        receivedData[receivedIndex].Range(startFacetIndex, startFacetIndex + thisTraceAtIntegrationPoints.Size()), threadLocalHeap);

                FlatVector<> upwindTraceAtIntegrationPoints = GetUpwindTrace(
                        thisTraceAtIntegrationPoints,
                        otherTraceAtIntegrationPoints,
                        facetData.innerProductOfFlowWithNormalVectorAtIntegrationPoint,
                        threadLocalHeap);

                FlatVector<> convectionCoefficients = GetElementCoefficientsFromTraceValuesOnFacet(
                        upwindTraceAtIntegrationPoints,
                        facetData.adjacentElementNumbers[0],
                        facetData.elementLocalFacetNumber[0],
                        threadLocalHeap);

                IntRange dofNumbers = finiteElementSpace->GetElementDofs(
                        static_cast<size_t>(facetData.adjacentElementNumbers[0]));
                {
                    lock_guard<mutex> guard(add_mutex);
                    convection.Range(dofNumbers) -= convectionCoefficients;
                }

                startFacetIndex += thisTraceAtIntegrationPoints.Size();
            }
        });
    }

    const DGFiniteElement<D>& GetElement(int elementId, LocalHeap& localHeap) const {
        return dynamic_cast<const DGFiniteElement<D>&>(
                finiteElementSpace->GetFE(ElementId(VOL, elementId), localHeap));
    }

    const DGFiniteElement<D - 1>& GetFacetElement(int elementId, int localFacetId, LocalHeap& localHeap) const {
        const auto globalFacetId = meshAccess->GetElFacets(ElementId(VOL, elementId))[localFacetId];
        return dynamic_cast<const DGFiniteElement<D - 1>&>(
                finiteElementSpace->GetFacetFE(globalFacetId, localHeap));
    }

    FlatVector<> GetTraceValuesOnFacetFromElementCoefficients(
            const FlatVector<>& coefficients,
            int elementId,
            int localFacetId,
            LocalHeap& localHeap) const {
        const auto& finiteElement = GetElement(elementId, localHeap);
        const auto& finiteElementFacet = GetFacetElement(elementId, localFacetId, localHeap);

        const auto numberOfFacetDofs = static_cast<size_t>(finiteElementFacet.GetNDof());
        FlatVector<> traceCoefficients(numberOfFacetDofs, localHeap);
        IntRange dofNumbers = finiteElementSpace->GetElementDofs(elementId);
        finiteElement.GetTrace(localFacetId, coefficients.Range(dofNumbers), traceCoefficients);

        IntegrationRule facetIntegrationRule = GetFacetIntegrationRule(finiteElementFacet);
        size_t numberOfIntegrationPoints = facetIntegrationRule.Size();

        FlatVector<> traceAtIntegrationPoints(numberOfIntegrationPoints, localHeap);
        finiteElementFacet.Evaluate(facetIntegrationRule, traceCoefficients, traceAtIntegrationPoints);
        return traceAtIntegrationPoints;
    }

    IntegrationRule GetFacetIntegrationRule(const DGFiniteElement<D - 1>& facet) const {
        return IntegrationRule(facet.ElementType(), 2 * facet.Order());
    }

    FlatVector<> GetElementCoefficientsFromTraceValuesOnFacet(const FlatVector<>& traceValues,
                                                              int elementId,
                                                              int localFacetId,
                                                              LocalHeap& localHeap) const {
        FlatVector<> traceCoefficients = GetFacetCoefficientsFromTraceValuesOnFacet(traceValues, elementId,
                                                                                    localFacetId, localHeap);
        return GetElementCoefficientsFromTraceCoefficients(traceCoefficients, elementId, localFacetId, localHeap);
    }

    FlatVector<>
    GetElementCoefficientsFromTraceCoefficients(const FlatVector<>& traceCoefficients,
                                                int elementId,
                                                int localFacetId,
                                                LocalHeap& localHeap) const {
        const auto& element = GetElement(elementId, localHeap);
        auto numberOfElementDofs = static_cast<const size_t>(element.GetNDof());
        FlatVector<> elementCoefficients(numberOfElementDofs, localHeap);
        element.GetTraceTrans(localFacetId, traceCoefficients, elementCoefficients);
        return elementCoefficients;
    }

    FlatVector<>
    GetFacetCoefficientsFromTraceValuesOnFacet(const FlatVector<>& traceValues, int elementId, int localFacetId,
                                               LocalHeap& localHeap) const {
        const auto& facet = GetFacetElement(elementId, localFacetId, localHeap);
        const auto facetIntegrationRule = GetFacetIntegrationRule(facet);
        FlatVector<> traceCoefficients(facetIntegrationRule.Size(), localHeap);
        facet.EvaluateTrans(GetFacetIntegrationRule(facet), traceValues, traceCoefficients);
        return traceCoefficients;
    }

    FlatVector<> GetUpwindTrace(
            const function<double(size_t)>& thisTraceOnFacet,
            const function<double(size_t)>& otherTraceOnFacet,
            const FlatVector<>& thisInnerProductOfFlowWithNormalVectorAtIntegrationPoint,
            LocalHeap& localHeap) const {
        FlatVector<> upwindTraceAtIntegrationPoints(thisInnerProductOfFlowWithNormalVectorAtIntegrationPoint.Size(),
                                                    localHeap);
        for (auto j: Range(thisInnerProductOfFlowWithNormalVectorAtIntegrationPoint.Size())) {
            upwindTraceAtIntegrationPoints(j) = thisInnerProductOfFlowWithNormalVectorAtIntegrationPoint(j) *
                                                ((thisInnerProductOfFlowWithNormalVectorAtIntegrationPoint(j) > 0)
                                                 ? thisTraceOnFacet(j) : otherTraceOnFacet(j));
        }
        return upwindTraceAtIntegrationPoints;
    }

    template<typename T>
    FlatVector<T> FlatArrayToFlatVector(FlatArray<T> flatArray, LocalHeap& localHeap) {
        FlatVector<T> vector(flatArray.Size(), localHeap);
        for (auto i: Range(flatArray.Size())) {
            vector(i) = flatArray[i];
        }
        return vector;
    }

    template<typename T>
    FlatArray<T> FlatVectorToFlatArray(FlatVector<T> flatVector, LocalHeap& localHeap) {
        FlatArray<T> array(flatVector.Size(), localHeap);
        for (auto i: Range(flatVector.Size())) {
            array[i] = flatVector(i);
        }
        return array;
    }
};

PYBIND11_MODULE(liblinhyp, m) {
    py::class_<Convection<2>>(m, "Convection")
            .def(py::init<shared_ptr<FESpace>, shared_ptr<CoefficientFunction>>())
            .def("Apply", &Convection<2>::Apply);
}
