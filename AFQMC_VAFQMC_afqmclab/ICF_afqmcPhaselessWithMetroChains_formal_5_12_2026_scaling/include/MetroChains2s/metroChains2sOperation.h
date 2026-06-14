//
// Created by boruoshihao on 1/10/17.
// Modefied by ICF on 2019-9-29.
//

#ifndef AFQMCLAB_METROCHAINS2SOPERATION_H
#define AFQMCLAB_METROCHAINS2SOPERATION_H

#include "afqmclab.h"
#include "metroChains2s.h"

class MetroChains2sOperation
{
    const MetroChains2s* walkerLeft; //ICF: why using address instead of noraml class? static?

    std::vector< MetroChains2sWalkerWalkerOperation > walkerwalkerOperationList;

    bool logOverlapIsCalculated;
 public:
    std::complex<double> logTotalPhase;

    MetroChains2sOperation();
    MetroChains2sOperation(const MetroChains2s &walkerLeft_);
    ~MetroChains2sOperation();

    const MetroChains2s *getWalkerLeft() const;
    
    void set(const MetroChains2s &walkerLeft_);
    void reSet();
    
    const tensor_hao::TensorHao<std::complex<double>, 2> &returnGreenMatrixUp(size_t bracket);
    const tensor_hao::TensorHao<std::complex<double>, 2> &returnGreenMatrixDn(size_t bracket);
    const tensor_hao::LUDecomp< std::complex<double> > &returnLUOverlapUp(size_t chain);
    const tensor_hao::LUDecomp< std::complex<double> > &returnLUOverlapDn(size_t chain);
    const tensor_hao::TensorHao< std::complex<double>, 2> &returnThetaUp_T(size_t chain);
    const tensor_hao::TensorHao< std::complex<double>, 2> &returnThetaDn_T(size_t chain);
    std::complex<double> returnLogw();
    std::complex<double> returnLogTotalPhase();
    std::complex<double> returnLogTotalPhase_fromCurrentOverlap();
    std::complex<double> returnLogPhase(size_t bracket);
    std::complex<double> returnLogPhase_fromCurrentOverlap(size_t bracket);
    std::complex<double> returnSignRatio();

    double getMemory() const;

    double logTruncation;

 private:
    MetroChains2sOperation(const MetroChains2sOperation& x);
    MetroChains2sOperation & operator  = (const MetroChains2sOperation& x);

    void calculateLUOverlap();
    void calculateTheta_T();
};

void setWalkerFromPhiT(std::vector<SD> &walker, std::vector<bool> &walkerIsAlive, const MetroChains2s& phiT);

#endif //AFQMCLAB_METROCHAINS2SOPERATION_H