//
// Created by boruoshihao on 1/13/17.
//

#ifndef AFQMCLAB_HUBBARDSOC_ICFMetroChainsSDMEASUREOBSERVE_H
#define AFQMCLAB_HUBBARDSOC_ICFMetroChainsSDMEASUREOBSERVE_H

#include "afqmclab.h"
#include "HubbardSOC_icf.h"
#include "../MetroChains/metroChainsOperation.h"

class HubbardSOC_icfMeasureObserveMetroChainsSD
{
 private:
   const HubbardSOC_icf * hubbardSOC_icf;
   std::complex<double> den;
   std::complex<double> HNum, KNum, VNum, RNum;
   tensor_hao::TensorHao<std::complex<double>, 2> greenMatrixNum;
   tensor_hao::TensorHao<std::complex<double>, 2> sqsmqNum;
    tensor_hao::TensorHao<std::complex<double>, 1> densityNum;
    tensor_hao::TensorHao<std::complex<double>, 1> szszNum,sxsxNum,sysyNum;
    tensor_hao::TensorHao<std::complex<double>, 2> densityDensityNum;
   //  tensor_hao::TensorHao<std::complex<double>, 2> splusSminusNum;
   //  tensor_hao::TensorHao<std::complex<double>, 2> sminusSplusNum;
   //  tensor_hao::TensorHao<std::complex<double>, 2> spairSpairNum;

 public:
    HubbardSOC_icfMeasureObserveMetroChainsSD();
    HubbardSOC_icfMeasureObserveMetroChainsSD(const HubbardSOC_icf &hubbardSOC_);
    ~HubbardSOC_icfMeasureObserveMetroChainsSD();

    const HubbardSOC_icf *getHubbardSOC_icf() const;

    void initModelNullptr();
    void setModel(const HubbardSOC_icf &hubbardSOC_);
    void reSet();
    std::complex<double> returnEnergy();
    void addMeasurement(MetroChainsOperation &metroChainsOperation, std::complex<double> denIncrement);
    void addMeasurement_timer(MetroChainsOperation &metroChainsOperation, std::complex<double> denIncrement);
    void addMeasurement_energy(MetroChainsOperation &metroChainsOperation, std::complex<double> denIncrement);
    NiupNidnForce getForce(const NiupNidn &niupNidn, MetroChainsOperation &metroChainsOperation, double cap=1.0);
    
    void addEnergy(MetroChainsOperation &metroChainsOperation, std::complex<double> denIncrement);

    void write() const;
    void write(std::string postfix) const;
    double getMemory() const;

 private:
    void addGreenMatrix(MetroChainsOperation &metroChainsOperation, std::complex<double> denIncrement);
    void addSqSmqNum(MetroChainsOperation &metroChainsOperation, std::complex<double> denIncrement);
    void addSSNum(MetroChainsOperation &metroChainsOperation, std::complex<double> denIncrement);
    std::complex<double> localCdaggerCCdaggerC(MetroChainsOperation &metroChainsOperation, size_t i, size_t j, size_t k, size_t l);
    std::complex<double> localCdaggerCCdaggerC_bracket(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrix_bracket, size_t i, size_t j, size_t k, size_t l);
    void addDensityDensity(MetroChainsOperation &metroChainsOperation, std::complex<double> denIncrement);
   //  void addSplusSminus(MetroChainsOperation &metroChainsOperation, std::complex<double> denIncrement);
   //  void addSminusSplus(MetroChainsOperation &metroChainsOperation, std::complex<double> denIncrement);
   //  void addSpairSpair(MetroChainsOperation &metroChainsOperation, std::complex<double> denIncrement);

    HubbardSOC_icfMeasureObserveMetroChainsSD(const HubbardSOC_icfMeasureObserveMetroChainsSD& x);
    HubbardSOC_icfMeasureObserveMetroChainsSD & operator  = (const HubbardSOC_icfMeasureObserveMetroChainsSD& x);
};

#endif //AFQMCLAB_HUBBARDSOC_ICFMetroChainsSDMEASUREOBSERVE_H