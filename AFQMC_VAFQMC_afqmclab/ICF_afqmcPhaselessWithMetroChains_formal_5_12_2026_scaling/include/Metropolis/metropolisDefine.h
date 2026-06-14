//
// Created by boruoshihao on 1/15/17.
//

#ifndef AFQMCLAB_AFQMCMETROPOLISDEFINE_H
#define AFQMCLAB_AFQMCMETROPOLISDEFINE_H

#include "afqmclab.h"

#include "../oneBodyOperator_icf/logHop2s.h"
#include "../twoBodyWalkerOperation_icf/NiupNidnSD2sOperation.h"
#include "../twoBodyWalkerOperation_icf/svdSDOperation.h"

#include "../twoBodyOperator_icf/svd.h"
#include "../twoBodyOperator_icf/svd_sym.h"
#include "../twoBodyOperator_icf/svdAux.h"
#include "../twoBodyOperator_icf/svdForce.h"
#include "../twoBodyOperator_icf/svdSample.h"

#include "../../lib_icf_oneBodyOperator_AuxMatrix/include/hop_AuxMatrix.h"
#include "../../lib_icf_twoBodyOperator_AuxMatrix/include/AuxMatrix.h"
#include "../../lib_icf_twoBodyOperator_AuxMatrix/include/AuxMatrixAux.h"
#include "../../lib_icf_twoBodyOperator_AuxMatrix/include/AuxMatrixForce.h"
#include "../../lib_icf_twoBodyOperator_AuxMatrix/include/AuxMatrixSample.h"


#include "../HubbardRealSpaceSOC_icf/HubbardSOC_icf.h"

#include "../generalHamiltonian_icf/generalHamiltonian_icf.h"
#include "../generalHamiltonian_icf/generalHamiltonian_sym_icf.h"
// #include "../generalHamiltonian_icf/generalHamiltonian_icfSDOperation.h"
// #include "../generalHamiltonian_icf/generalHamiltonian_icfMeasureObserveSDSD.h"
// #include "../generalHamiltonian_icf/generalHamiltonian_sym_icfMeasureObserveSDSD.h"

//SD
typedef SDSDOperation WalkerWalkerOperation_Jastrow;
//Jastrow
typedef GeneralHamiltonian_icf Model_Jastrow;
// typedef GeneralHamiltonian_sym_icf Model_Jastrow;
typedef Hop       OneBody_Jastrow;
typedef SVD   TwoBody_Jastrow;
// typedef SVD_sym   TwoBody_Jastrow;
typedef SVDAux    TwoBodyAux_Jastrow;
typedef SVDForce  TwoBodyForce_Jastrow;
typedef SVDSample TwoBodySample_Jastrow;

typedef HubbardSOC_icf     Model_BP_Jastrow;
typedef Hop                OneBody_BP_Jastrow;
typedef NiupNidn           TwoBody_BP_Jastrow;
typedef NiupNidnAux        TwoBodyAux_BP_Jastrow;
typedef NiupNidnForce      TwoBodyForce_BP_Jastrow;
typedef NiupNidnSample     TwoBodySample_BP_Jastrow;

typedef Hop                 OneBody_AM_Jastrow;
typedef AuxMatrix           TwoBody_AM_Jastrow;
typedef AuxMatrixAux        TwoBodyAux_AM_Jastrow;
typedef AuxMatrixForce      TwoBodyForce_AM_Jastrow;
typedef AuxMatrixSample     TwoBodySample_AM_Jastrow;
typedef Hop                 TwoBodySample_AM_Jastrow_Hop;
typedef LogHop              TwoBodySample_AM_Jastrow_LogHop;
typedef LogHop              OneBody_AM_Jastrow_LogHop;

typedef SD Walker;
typedef HopSDOperation HopWalkerOperation;
typedef SVDSampleSDOperation SVDSampleWalkerOperation;
typedef NiupNidnSampleSDOperation NiupNidnSampleWalkerOperation;
typedef LogHopSDOperation LogHopWalkerOperation;

//SD2s
// typedef SD2sSD2sOperation WalkerWalkerOperation_Jastrow;
// // typedef GeneralHamiltonian_sym_icf Model_Jastrow;
// typedef Hop2s       OneBody_Jastrow;
// typedef SVD_sym     TwoBody_Jastrow;
// typedef SVDAux      TwoBodyAux_Jastrow;
// typedef SVDForce    TwoBodyForce_Jastrow;
// typedef SVDSample2s TwoBodySample_Jastrow;

// // typedef HubbardSOC_icf     Model_BP_Jastrow;
// typedef Hop2s              OneBody_BP_Jastrow;
// typedef NiupNidn           TwoBody_BP_Jastrow;
// typedef NiupNidnAux        TwoBodyAux_BP_Jastrow;
// typedef NiupNidnForce      TwoBodyForce_BP_Jastrow;
// typedef NiupNidnSample     TwoBodySample_BP_Jastrow;

// typedef Hop2s                 OneBody_AM_Jastrow;
// typedef AuxMatrix             TwoBody_AM_Jastrow;
// typedef AuxMatrixAux          TwoBodyAux_AM_Jastrow;
// typedef AuxMatrixForce        TwoBodyForce_AM_Jastrow;
// typedef AuxMatrixSample2s       TwoBodySample_AM_Jastrow;
// typedef Hop2s                   TwoBodySample_AM_Jastrow_Hop;
// typedef LogHop2s               TwoBodySample_AM_Jastrow_LogHop;
// typedef LogHop2s               OneBody_AM_Jastrow_LogHop;

// typedef SD2s Walker;
// typedef Hop2sSD2sOperation HopWalkerOperation;
// typedef SVDSample2sSD2sOperation SVDSampleWalkerOperation;
// typedef NiupNidnSampleSD2sOperation NiupNidnSampleWalkerOperation;
// typedef LogHop2sSD2sOperation LogHopWalkerOperation;


#endif //AFQMCLAB_AFQMCMETROPOLISDEFINE_H
