//
// Created by boruoshihao on 1/13/17.
//

#ifndef AFQMCLAB_HUBBARDSOC_ICFSDSDMEASURECOMMUTE_H
#define AFQMCLAB_HUBBARDSOC_ICFSDSDMEASURECOMMUTE_H

#include "HubbardSOC_icf.h"
#include "afqmclab.h"

class HubbardSOC_icfMeasureCommuteSDSD
{
 private:
    const HubbardSOC_icf * hubbardSOC_icf;
    std::complex<double> den;
    std::complex<double> HNum, KNum, VNum, RNum;

 public:
    HubbardSOC_icfMeasureCommuteSDSD();
    HubbardSOC_icfMeasureCommuteSDSD(const HubbardSOC_icf &hubbardSOC_);
    ~HubbardSOC_icfMeasureCommuteSDSD();

    const HubbardSOC_icf *getHubbardSOC_icf() const;

    void initModelNullptr();
    void setModel(const HubbardSOC_icf &hubbardSOC_);
    void reSet();
    std::complex<double> returnEnergy();
    void addMeasurement(SDSDOperation &sdsdOperation, std::complex<double> denIncrement);
    NiupNidnForce getForce(const NiupNidn &niupNidn, SDSDOperation &sdsdOperation, double cap=1.0);

    void write() const;
    void write(std::string postfix) const;
    void writeKNumVumRum() const;
    double getMemory() const;

 private:
    HubbardSOC_icfMeasureCommuteSDSD(const HubbardSOC_icfMeasureCommuteSDSD& x);
    HubbardSOC_icfMeasureCommuteSDSD & operator  = (const HubbardSOC_icfMeasureCommuteSDSD& x);
    void addEnergy(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrix, std::complex<double> denIncrement);
};

#endif //AFQMCLAB_HUBBARDSOC_ICFSDSDMEASURECOMMUTE_H