//
// Created by boruoshihao on 6/23/17.
//
// USE_SD: 1 use SD, 0 use SD2s
#define USE_SD 1
// USE_generalHamiltonian: 1 use generalHamiltonian, 0 use Hubbard Hamiltonian
#define USE_generalHamiltonian 1 

#ifndef AFQMCLAB_AFQMCPHASELESSDEFINE_H
#define AFQMCLAB_AFQMCPHASELESSDEFINE_H

#include "afqmclab.h"

// #include "twoBodyOperator_icf/svd.h"
// #include "twoBodyOperator_icf/svdAux.h"
// #include "twoBodyOperator_icf/svdForce.h"
// #include "twoBodyOperator_icf/svdSample.h"

// #include "twoBodyWalkerOperation_icf/svdSDOperation.h"

#include "MetroChains/metroChainsAll.h"
#include "MetroChains2s/metroChains2sAll.h"

#include "HubbardRealSpaceSOC_icf/HubbardSOC_icfAll.h"
#include "generalHamiltonian_icf/generalHamiltonian_icfAll.h"

#include "twoBodyWalkerOperation_icf/NiupNidnSD2sOperation.h"

#include "../populationControl/include/populationControl_icf.h"

typedef SVD       TwoBody;
typedef SVDAux    TwoBodyAux;
typedef SVDForce  TwoBodyForce;

#ifdef USE_SD
    typedef SVDSample TwoBodySample;
#else
    typedef SVDSample2s TwoBodySample;
#endif

typedef GeneralHamiltonian_icf Model;

// typedef NiupNidn       TwoBody;
// typedef NiupNidnAux    TwoBodyAux;
// typedef NiupNidnForce  TwoBodyForce;
// typedef NiupNidnSample TwoBodySample;

// typedef HubbardSOC_icf Model;

//////////////////////////////////////////////////////
//Jastrow
//////////////////////////////////////////////////////
#ifdef USE_SD
// SD
// Make sure MetroChains is changed to SD
// 
typedef multDET  multDET_AFQMC;
typedef JastrowProjector  JastrowProjector_AFQMC;
typedef MetropolisMethod  MetropolisMethod_AFQMC;
typedef Hop OneBody;
typedef MetroChains  WalkerLeft;
typedef SD  WalkerRight;
typedef HopSDOperation OneBodyWalkerRightOperation;
typedef SVDSampleSDOperation TwoBodySampleWalkerRightOperation;
// typedef NiupNidnSampleSDOperation TwoBodySampleWalkerRightOperation;
typedef MetroChainsOperation WalkerWalkerOperation;
typedef SDSDOperation WalkerRightWalkerRightOperation;
typedef GeneralHamiltonian_icfMeasureObserveMetroChainsSD ModelMeasureMixed;
typedef GeneralHamiltonian_icfMeasureObserveSDSD ModelMeasurePure_walkerWalker;
// typedef HubbardSOC_icfMeasureObserveMetroChainsSD ModelMeasureMixed;
// typedef HubbardSOC_icfMeasureObserveSDSD ModelMeasurePure_walkerWalker;
#else
//SD2s
// Make sure MetroChains is changed to SD2s
// 
typedef multDET2s  multDET_AFQMC;
typedef JastrowProjector2s  JastrowProjector_AFQMC;
typedef Metropolis2sMethod  MetropolisMethod_AFQMC;
typedef Hop2s OneBody;
typedef MetroChains2s  WalkerLeft;
typedef SD2s  WalkerRight;
typedef Hop2sSD2sOperation OneBodyWalkerRightOperation;
typedef SVDSample2sSD2sOperation TwoBodySampleWalkerRightOperation;
// typedef NiupNidnSampleSD2sOperation TwoBodySampleWalkerRightOperation;
typedef MetroChains2sOperation WalkerWalkerOperation;
typedef SD2sSD2sOperation WalkerRightWalkerRightOperation;
typedef GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s ModelMeasureMixed;
typedef GeneralHamiltonian_icfMeasureObserveSD2sSD2s ModelMeasurePure_walkerWalker;
// typedef HubbardSOC_icfMeasureObserveMetroChains2sSD2s ModelMeasureMixed;
// typedef HubbardSOC_icfMeasureObserveSD2sSD2s ModelMeasurePure_walkerWalker;
#endif


#endif //AFQMCLAB_AFQMCPHASELESSDEFINE_H
