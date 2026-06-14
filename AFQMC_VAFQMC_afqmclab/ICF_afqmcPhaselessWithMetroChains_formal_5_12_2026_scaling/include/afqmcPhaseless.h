//
// Created by boruoshihao on 7/8/17.
//

#ifndef AFQMCLAB_AFQMCPHASELESS_H
#define AFQMCLAB_AFQMCPHASELESS_H

#include "afqmcPhaselessDefine.h"
#include "afqmcPhaselessMethod.h"


class AfqmcPhaseless
{
 private:
    AfqmcPhaselessMethod method;
    Model model;
    //
    MetropolisMethod_AFQMC method_Jastrow;
    //JastrowProjector for propagation
    JastrowProjector_AFQMC jastrowProjector;
    //JastrowProjector for release measurement
   //  JastrowProjector_AFQMC jastrowProjectorExtended;
    //
    OneBody expMinusDtK, expMinusHalfDtK, expHalfDtK;
    TwoBody expMinusDtV;
    TwoBodyForce dynamicForce, constForce;

    OneBodyWalkerRightOperation oneBodyWalkerRightOperation;
    TwoBodySampleWalkerRightOperation twoBodySampleWalkerRightOperation;

    TwoBodyAux twoBodyAux;
    TwoBodySample twoBodySample;

    std::vector< WalkerLeft > phiT;
    multDET_AFQMC phiT_multDet_initial;

    std::vector<bool> walkerIsAlive;
    std::vector<std::complex<double>> logOverlapBackup;

    WalkerWalkerOperation walkerWalkerOperation;
    WalkerRightWalkerRightOperation walkerRightWalkerRightOperation;
    ModelMeasureMixed mixedMeasure;
    ModelMeasureMixed pureMeasure;
    ModelMeasurePure_walkerWalker pureMeasure_walkerWalker;

    bool ifRelease;

 public:
    AfqmcPhaseless();
    ~AfqmcPhaseless();

    void run();
    void initialParameters();
    void initialMeasure();
    void initialExpOneBody();
    void estimateMemory();
    void measureWithoutProjection();
    void measureWithProjection();
    void prepareStop();

 private:
    void initialPhiT();
    void initialWalker();
    void writeWalkers();
    void checkOverlap(WalkerRight &oneWalker);
    void initialMgsAndPopControl();

    void projectExpHalfDtK();
    void projectExpMinusHalfDtK();
    void projectExpMinusDtKExpMinusDtV();
    void updateMetroChainsWalkers(size_t i);
    void updateMetroChainsWalkers();
    void projectOneStep(size_t &MetroUpdateSkipIndex, size_t &mgsIndex, size_t &popControlIndex);
    void modifyGM();
    void popControl(double popWeightCap=0.2);
    void checkAndResetWalkerIsAlive();
    void resetLogOverlapBackup();

    void addMixedMeasurement_energy();
   //  void addPureMeasure_SDSD();
    void addMixedMeasurement();
    void addMixedMeasurement_timer();
    void writeAndResetMeasurement();
    void adjustETThenResetMeasurement();
    void adjustETAndBackGroundThenResetMeasurement();
};

#endif //AFQMCLAB_AFQMCPHASELESS_H