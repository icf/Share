//
// Created by boruoshihao on 1/15/17.
//

#ifndef AFQMCLAB_AFQMCMETROPOLIS2SDEFINE_H
#define AFQMCLAB_AFQMCMETROPOLIS2SDEFINE_H

#include "afqmclab.h"

#include "../oneBodyOperator_icf/logHop2s.h"
#include "../oneBodyOperator_icf/logHop2sSD2sOperation.h"
#include "../twoBodyWalkerOperation_icf/NiupNidnSD2sOperation.h"
#include "../twoBodyWalkerOperation_icf/svdSD2sOperation.h"

#include "../twoBodyOperator_icf/svd.h"
#include "../twoBodyOperator_icf/svd_sym.h"
#include "../twoBodyOperator_icf/svdAux.h"
#include "../twoBodyOperator_icf/svdForce.h"
#include "../twoBodyOperator_icf/svdSample2s.h"

#include "../../lib_icf_oneBodyOperator_AuxMatrix/include/hop_AuxMatrix.h"
#include "../../lib_icf_twoBodyOperator_AuxMatrix/include/AuxMatrix.h"
#include "../../lib_icf_twoBodyOperator_AuxMatrix/include/AuxMatrixAux.h"
#include "../../lib_icf_twoBodyOperator_AuxMatrix/include/AuxMatrixForce.h"
#include "../../lib_icf_twoBodyOperator_AuxMatrix/include/AuxMatrixSample2s.h"


#include "../HubbardRealSpaceSOC_icf/HubbardSOC_icf.h"

#include "../generalHamiltonian_icf/generalHamiltonian_icf.h"
#include "../generalHamiltonian_icf/generalHamiltonian_sym_icf.h"
// #include "../generalHamiltonian_icf/generalHamiltonian_icfSDOperation.h"
// #include "../generalHamiltonian_icf/generalHamiltonian_icfMeasureObserveSDSD.h"
// #include "../generalHamiltonian_icf/generalHamiltonian_sym_icfMeasureObserveSDSD.h"

// SD2s
typedef SD2sSD2sOperation WalkerWalkerOperation_Jastrow2s;
typedef GeneralHamiltonian_icf Model_Jastrow2s;
// typedef GeneralHamiltonian_sym_icf Model_Jastrow2s;
typedef Hop2s       OneBody_Jastrow2s;
typedef SVD     TwoBody_Jastrow2s;
// typedef SVD_sym     TwoBody_Jastrow2s;
typedef SVDAux      TwoBodyAux_Jastrow2s;
typedef SVDForce    TwoBodyForce_Jastrow2s;
typedef SVDSample2s TwoBodySample_Jastrow2s;

typedef HubbardSOC_icf     Model_BP_Jastrow2s;
typedef Hop2s              OneBody_BP_Jastrow2s;
typedef NiupNidn           TwoBody_BP_Jastrow2s;
typedef NiupNidnAux        TwoBodyAux_BP_Jastrow2s;
typedef NiupNidnForce      TwoBodyForce_BP_Jastrow2s;
typedef NiupNidnSample     TwoBodySample_BP_Jastrow2s;

typedef Hop2s                 OneBody_AM_Jastrow2s;
typedef AuxMatrix             TwoBody_AM_Jastrow2s;
typedef AuxMatrixAux          TwoBodyAux_AM_Jastrow2s;
typedef AuxMatrixForce        TwoBodyForce_AM_Jastrow2s;
typedef AuxMatrixSample2s       TwoBodySample_AM_Jastrow2s;
typedef LogHop2s               TwoBodySample_AM_Jastrow_LogHop2s;

typedef SD2s Walker2s;
typedef Hop2sSD2sOperation HopWalkerOperation2s;
typedef SVDSample2sSD2sOperation SVDSampleWalkerOperation2s;
typedef NiupNidnSampleSD2sOperation NiupNidnSampleWalkerOperation2s;
typedef LogHop2sSD2sOperation LogHopWalkerOperation2s;


#endif //AFQMCLAB_AFQMCMETROPOLIS2SDEFINE_H
