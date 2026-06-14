cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(afqmcPhaseless C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

find_package(HDF5 COMPONENTS C CXX HL REQUIRED)
find_package(AFQMCLAB REQUIRED)

set(CMAKE_CXX_COMPILER ${AFQMCLAB_COMPILER})
include_directories(${AFQMCLAB_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${AFQMCLAB_FLAGS}")
add_definitions(${AFQMCLAB_DEFINITIONS})

set(SOURCE_HEADERS
        include/utils.h
        include/afqmcPhaseless.h
        include/afqmcPhaselessDefine.h
        include/afqmcPhaselessMethod.h
        include/afqmcWalkerPop.h
	include/HubbardRealSpaceSOC_icf/HubbardSOC_icf.h
	include/HubbardRealSpaceSOC_icf/HubbardSOC_icfAll.h
	include/HubbardRealSpaceSOC_icf/HubbardSOC_icfMeasureCommuteSDSD.h
	include/HubbardRealSpaceSOC_icf/HubbardSOC_icfMeasureCommuteSD2sSD2s.h
	include/HubbardRealSpaceSOC_icf/HubbardSOC_icfMeasureFixedSDSD.h
	include/HubbardRealSpaceSOC_icf/HubbardSOC_icfMeasureObserveMetroChainsSD.h
	include/HubbardRealSpaceSOC_icf/HubbardSOC_icfMeasureObserveMetroChains2sSD2s.h
	include/HubbardRealSpaceSOC_icf/HubbardSOC_icfMeasureObserveSDSD.h
	include/HubbardRealSpaceSOC_icf/HubbardSOC_icfMeasureObserveSD2sSD2s.h
	include/HubbardRealSpaceSOC_icf/HubbardSOC_icfSDOperation.h
        include/generalHamiltonian_icf/generalHamiltonian_icfAll.h
        include/generalHamiltonian_icf/generalHamiltonian_icf.h
        include/generalHamiltonian_icf/tensor_hao_mpi_shared.h
        include/generalHamiltonian_icf/generalHamiltonian_sym_icf.h
        include/generalHamiltonian_icf/generalHamiltonian_icfSDOperation.h
        include/generalHamiltonian_icf/generalHamiltonian_icfSD2sOperation.h
        include/generalHamiltonian_icf/generalHamiltonian_icfMeasureObserveSDSD.h
        include/generalHamiltonian_icf/generalHamiltonian_icfMeasureObserveSD2sSD2s.h
        include/generalHamiltonian_icf/generalHamiltonian_sym_icfMeasureObserveSDSD.h
        include/generalHamiltonian_icf/generalHamiltonian_icfMeasureObserveMetroChainsSD.h
        include/generalHamiltonian_icf/generalHamiltonian_icfMeasureObserveMetroChains2sSD2s.h
        include/oneBodyOperator_icf/logHopSDOperation_icf.h
        include/oneBodyOperator_icf/logHop2sSD2sOperation.h
        include/oneBodyOperator_icf/logHop2s.h
        include/twoBodyOperator_icf/svd.h
        include/twoBodyOperator_icf/svd_sym.h
        include/twoBodyOperator_icf/svdAux.h
        include/twoBodyOperator_icf/svdForce.h
        include/twoBodyOperator_icf/svdSample.h
        include/twoBodyWalkerOperation_icf/svdSDOperation.h
        include/twoBodyWalkerOperation_icf/svdSD2sOperation.h
        include/twoBodyWalkerOperation_icf/NiupNidnSD2sOperation.h
        lib_icf_oneBodyOperator_AuxMatrix/include/hop_AuxMatrix.h
        lib_icf_twoBodyOperator_AuxMatrix/include/AuxMatrix.h
        lib_icf_twoBodyOperator_AuxMatrix/include/AuxMatrixAux.h
        lib_icf_twoBodyOperator_AuxMatrix/include/AuxMatrixForce.h
        lib_icf_twoBodyOperator_AuxMatrix/include/AuxMatrixSample.h
        lib_icf_twoBodyOperator_AuxMatrix/include/AuxMatrixSample2s.h
        include/MetroChains/metroChainsAll.h
        include/MetroChains/multDET.h
        include/MetroChains/metroChains.h
        include/MetroChains/metroChainsOperation.h
        include/MetroChains/SDSDOperation_icf.h
        include/Metropolis/LeapFrog.h
        include/Metropolis/JastrowProjector.h
        include/Metropolis/metropolis.h
        include/Metropolis/metropolisInfo.h
        include/Metropolis/metropolisDefine.h
        include/Metropolis/metropolisMethod.h
        include/MetroChains2s/multDET2s.h
        include/MetroChains2s/metroChains2sAll.h
        include/MetroChains2s/metroChains2s.h
        include/MetroChains2s/metroChains2sOperation.h
        include/MetroChains2s/SD2sSD2sOperation_icf.h
        include/Metropolis2s/JastrowProjector2s.h
        include/Metropolis2s/metropolis2s.h
        include/Metropolis2s/metropolis2sInfo.h
        include/Metropolis2s/metropolis2sDefine.h
        include/Metropolis2s/metropolis2sMethod.h
        populationControl/include/populationControl_icf.h
        )

set(SOURCE_FILES
        source/afqmcPhaseless.cpp
        source/afqmcPhaselessMeasure.cpp
        source/afqmcPhaselessMethod.cpp
        source/afqmcPhaselessProjection.cpp
        source/afqmcPhaselessSetWalker.cpp
        source/afqmcWalkerPop.cpp
	source/HubbardRealSpaceSOC_icf/HubbardSOC_icf.cpp
	source/HubbardRealSpaceSOC_icf/HubbardSOC_icfMeasureCommuteSDSD.cpp
	source/HubbardRealSpaceSOC_icf/HubbardSOC_icfMeasureCommuteSD2sSD2s.cpp
	source/HubbardRealSpaceSOC_icf/HubbardSOC_icfMeasureFixedSDSD.cpp
	source/HubbardRealSpaceSOC_icf/HubbardSOC_icfMeasureObserveMetroChainsSD.cpp
	source/HubbardRealSpaceSOC_icf/HubbardSOC_icfMeasureObserveMetroChains2sSD2s.cpp
	source/HubbardRealSpaceSOC_icf/HubbardSOC_icfMeasureObserveSDSD.cpp
	source/HubbardRealSpaceSOC_icf/HubbardSOC_icfMeasureObserveSD2sSD2s.cpp
	source/HubbardRealSpaceSOC_icf/HubbardSOC_icfSDOperation.cpp
        source/generalHamiltonian_icf/generalHamiltonian_icf.cpp
        source/generalHamiltonian_icf/generalHamiltonian_sym_icf.cpp
        source/generalHamiltonian_icf/generalHamiltonian_icfSDOperation.cpp
        source/generalHamiltonian_icf/generalHamiltonian_icfSD2sOperation.cpp
        source/generalHamiltonian_icf/generalHamiltonian_icfMeasureObserveSDSD.cpp
        source/generalHamiltonian_icf/generalHamiltonian_icfMeasureObserveSD2sSD2s.cpp
        source/generalHamiltonian_icf/generalHamiltonian_sym_icfMeasureObserveSDSD.cpp
        source/generalHamiltonian_icf/generalHamiltonian_icfMeasureObserveMetroChainsSD.cpp
        source/generalHamiltonian_icf/generalHamiltonian_icfMeasureObserveMetroChains2sSD2s.cpp
        source/oneBodyOperator_icf/logHopSDOperation_icf.cpp
        source/oneBodyOperator_icf/logHop2sSD2sOperation.cpp
        source/oneBodyOperator_icf/logHop2s.cpp
        source/twoBodyOperator_icf/svd.cpp
        source/twoBodyOperator_icf/svd_sym.cpp
        source/twoBodyWalkerOperation_icf/NiupNidnSD2sOperation.cpp
        lib_icf_oneBodyOperator_AuxMatrix/source/hop_AuxMatrix.cpp
        lib_icf_twoBodyOperator_AuxMatrix/source/AuxMatrix.cpp
        lib_icf_twoBodyOperator_AuxMatrix/source/AuxMatrixAux.cpp
        lib_icf_twoBodyOperator_AuxMatrix/source/AuxMatrixForce.cpp
        lib_icf_twoBodyOperator_AuxMatrix/source/AuxMatrixSample.cpp
        lib_icf_twoBodyOperator_AuxMatrix/source/AuxMatrixSample2s.cpp
        source/MetroChains/multDET.cpp
        source/MetroChains/metroChains.cpp
        source/MetroChains/metroChainsOperation.cpp
        source/MetroChains/SDSDOperation_icf.cpp
        source/Metropolis/LeapFrog.cpp
        source/Metropolis/JastrowProjector.cpp
        source/Metropolis/metropolis.cpp
        source/Metropolis/metropolisInfo.cpp
        source/Metropolis/metropolisMethod.cpp
        source/Metropolis/metropolisSetWalkerAndField.cpp
        source/Metropolis/metropolisUpdate.cpp
        source/MetroChains2s/multDET2s.cpp
        source/MetroChains2s/metroChains2s.cpp
        source/MetroChains2s/metroChains2sOperation.cpp
        source/MetroChains2s/SD2sSD2sOperation_icf.cpp
        source/Metropolis2s/JastrowProjector2s.cpp
        source/Metropolis2s/metropolis2s.cpp
        source/Metropolis2s/metropolis2sInfo.cpp
        source/Metropolis2s/metropolis2sMethod.cpp
        source/Metropolis2s/metropolis2sSetWalkerAndField.cpp
        source/Metropolis2s/metropolis2sUpdate.cpp
        populationControl/source/populationControlAux_icf.cpp
        populationControl/source/populationControlConfiguration_icf.cpp
        source/utils.cpp
        source/main.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_HEADERS} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${AFQMCLAB_LIBRARIES})

add_custom_target(rmdat COMMAND rm -rf *dat)