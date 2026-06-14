//
// Created by boruoshihao on 1/13/17.
//

#ifndef AFQMCLAB_HUBBARDSOC_ICFSDSDMEASUREOBSERVE_H
#define AFQMCLAB_HUBBARDSOC_ICFSDSDMEASUREOBSERVE_H

#include "HubbardSOC_icfMeasureCommuteSDSD.h"

class HubbardSOC_icfMeasureObserveSDSD : public HubbardSOC_icfMeasureCommuteSDSD
{
 private:
   tensor_hao::TensorHao<std::complex<double>, 1> szszNum;
   tensor_hao::TensorHao<std::complex<double>, 1> sxsxNum;
   tensor_hao::TensorHao<std::complex<double>, 1> sysyNum;
    tensor_hao::TensorHao<std::complex<double>, 2> greenMatrixNum;
    tensor_hao::TensorHao<std::complex<double>, 2> densityDensityNum;
    tensor_hao::TensorHao<std::complex<double>, 2> splusSminusNum;
    tensor_hao::TensorHao<std::complex<double>, 2> sminusSplusNum;
    tensor_hao::TensorHao<std::complex<double>, 2> spairSpairNum;

 public:
    HubbardSOC_icfMeasureObserveSDSD();
    HubbardSOC_icfMeasureObserveSDSD(const HubbardSOC_icf &hubbardSOC_);
    ~HubbardSOC_icfMeasureObserveSDSD();

    void reSet();
    void addMeasurement(SDSDOperation &sdsdOperation, std::complex<double> denIncrement);
    void addMeasurement_energy(SDSDOperation &sdsdOperation, std::complex<double> denIncrement);
    void write() const;
    void write(std::string postfix) const;
    double getMemory() const;

 private:
    void addSS(SDSDOperation &sdsdOperation, std::complex<double> denIncrement);
    std::complex<double> localCdaggerCCdaggerC(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrix, size_t i, size_t j, size_t k, size_t l);
    void addGreenMatrix(SDSDOperation &sdsdOperation, std::complex<double> denIncrement);
    void addDensityDensity(SDSDOperation &sdsdOperation, std::complex<double> denIncrement);
    void addSplusSminus(SDSDOperation &sdsdOperation, std::complex<double> denIncrement);
    void addSminusSplus(SDSDOperation &sdsdOperation, std::complex<double> denIncrement);
    void addSpairSpair(SDSDOperation &sdsdOperation, std::complex<double> denIncrement);

    HubbardSOC_icfMeasureObserveSDSD(const HubbardSOC_icfMeasureObserveSDSD& x);
    HubbardSOC_icfMeasureObserveSDSD & operator  = (const HubbardSOC_icfMeasureObserveSDSD& x);
};

#endif //AFQMCLAB_HUBBARDSOC_ICFSDSDMEASUREOBSERVE_H