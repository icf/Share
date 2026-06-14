//
// Created by boruoshihao on 1/10/17.
// Modefied by ICF on 2019-9-29.
//

#ifndef AFQMCLAB_METROCHAINSOPERATION_H
#define AFQMCLAB_METROCHAINSOPERATION_H

#include "afqmclab.h"
#include "metroChains.h"

class MetroChainsOperation
{
    const MetroChains* walkerLeft; //ICF: why using address instead of noraml class? static?

    std::vector< MetroChainsWalkerWalkerOperation > walkerwalkerOperationList;

    bool logOverlapIsCalculated;
 public:
    std::complex<double> logTotalPhase;

    MetroChainsOperation();
    MetroChainsOperation(const MetroChains &walkerLeft_);
    ~MetroChainsOperation();

    const MetroChains *getWalkerLeft() const;
    
    void set(const MetroChains &walkerLeft_);
    void reSet();
    
    const tensor_hao::TensorHao<std::complex<double>, 2> &returnGreenMatrix(size_t bracket);
    const tensor_hao::LUDecomp< std::complex<double> > &returnLUOverlap(size_t chain);
    const tensor_hao::TensorHao< std::complex<double>, 2> &returnTheta_T(size_t chain);
    std::complex<double> returnLogw();
    std::complex<double> returnLogTotalPhase();
    std::complex<double> returnLogTotalPhase_fromCurrentOverlap();
    std::complex<double> returnLogPhase(size_t bracket);
    std::complex<double> returnLogPhase_fromCurrentOverlap(size_t bracket);
    std::complex<double> returnSignRatio();

    double getMemory() const;

    double logTruncation;

 private:
    MetroChainsOperation(const MetroChainsOperation& x);
    MetroChainsOperation & operator  = (const MetroChainsOperation& x);

    void calculateLUOverlap();
    void calculateTheta_T();
};

void setWalkerFromPhiT(std::vector<SD> &walker, std::vector<bool> &walkerIsAlive, const MetroChains& phiT);

#endif //AFQMCLAB_METROCHAINSOPERATION_H