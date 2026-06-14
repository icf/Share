//
// Created by Hao Shi on 8/1/17.
//

#ifndef AFQMCLAB_HUBBARDSOC_ICFMEASUREFIXEDSDSD_H
#define AFQMCLAB_HUBBARDSOC_ICFMEASUREFIXEDSDSD_H

#include "HubbardSOC_icf.h"
#include "afqmclab.h"

class HubbardSOC_icfMeasureFixedSDSD
{
 private:
    const HubbardSOC_icf * hubbardSOC_icf;
    const SD *walkerLeft;

    std::complex<double> den;
    std::complex<double> KNum, VNum, RNum, HNum;
    tensor_hao::TensorHao<std::complex<double>, 1> NupNum, NdnNum, SplusNum, SminusNum;
    std::complex<double> NupTotNum, NdnTotNum, SplusTotNum, SminusTotNum;

    tensor_hao::TensorHao<std::complex<double>,2> wfDaggerK;

 public:
    HubbardSOC_icfMeasureFixedSDSD();
    HubbardSOC_icfMeasureFixedSDSD(const HubbardSOC_icf& hubbardSOC_, const SD& walkerLeft_);
    ~HubbardSOC_icfMeasureFixedSDSD();

    void initModelWalkerNullptr();
    void setModelWalker(const HubbardSOC_icf& hubbardSOC_, const SD& walkerLeft_);
    void reSet();
    std::complex<double> returnEnergy();
    void addMeasurement(SDSDOperation &sdsdOperation, std::complex<double> denIncrement);
    NiupNidnForce getForce(const NiupNidn &niupNidn, SDSDOperation &sdsdOperation, double cap=1.0);

    void write() const;
    double getMemory() const;

 private:
    HubbardSOC_icfMeasureFixedSDSD(const HubbardSOC_icfMeasureFixedSDSD& x);
    HubbardSOC_icfMeasureFixedSDSD & operator = (const HubbardSOC_icfMeasureFixedSDSD& x);

    void initWfDaggerK();
    void checkWalkerLeft(const SDSDOperation &sdsdOperation);
    void addEnergy(SDSDOperation &sdsdOperation, std::complex<double> denIncrement);
    void addNupNdn(SDSDOperation &sdsdOperation, std::complex<double> denIncrement);
    void addSplusSminus(SDSDOperation &sdsdOperation, std::complex<double> denIncrement);
};


#endif //AFQMCLAB_HUBBARDSOC_ICFMEASUREFIXEDSDSD_H
