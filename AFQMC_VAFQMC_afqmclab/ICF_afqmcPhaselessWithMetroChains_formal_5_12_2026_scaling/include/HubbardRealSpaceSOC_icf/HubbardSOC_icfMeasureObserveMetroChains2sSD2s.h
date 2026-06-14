//
// Created by boruoshihao on 1/13/17.
//

#ifndef AFQMCLAB_HUBBARDSOC_ICFMetroChains2SSD2SMEASUREOBSERVE_H
#define AFQMCLAB_HUBBARDSOC_ICFMetroChains2SSD2SMEASUREOBSERVE_H

#include "afqmclab.h"
#include "HubbardSOC_icf.h"
#include "../MetroChains2s/metroChains2sOperation.h"

class HubbardSOC_icfMeasureObserveMetroChains2sSD2s
{
 private:
   const HubbardSOC_icf * hubbardSOC_icf;
   std::complex<double> den;
   std::complex<double> HNum, KNum, VNum, RNum;
   tensor_hao::TensorHao<std::complex<double>, 2> greenMatrixUpNum, greenMatrixDnNum;
   tensor_hao::TensorHao<std::complex<double>, 2> sqsmqNum;
    tensor_hao::TensorHao<std::complex<double>, 1> densityNum;
    tensor_hao::TensorHao<std::complex<double>, 1> szszNum,sxsxNum,sysyNum,DDNum;
    tensor_hao::TensorHao<std::complex<double>, 2> densityDensityNum;
   //  tensor_hao::TensorHao<std::complex<double>, 2> splusSminusNum;
   //  tensor_hao::TensorHao<std::complex<double>, 2> sminusSplusNum;
   //  tensor_hao::TensorHao<std::complex<double>, 2> spairSpairNum;

 public:
    HubbardSOC_icfMeasureObserveMetroChains2sSD2s();
    HubbardSOC_icfMeasureObserveMetroChains2sSD2s(const HubbardSOC_icf &hubbardSOC_);
    ~HubbardSOC_icfMeasureObserveMetroChains2sSD2s();

    const HubbardSOC_icf *getHubbardSOC_icf() const;

    void initModelNullptr();
    void setModel(const HubbardSOC_icf &hubbardSOC_);
    void reSet();
    std::complex<double> returnEnergy();
    void addMeasurement(MetroChains2sOperation &metroChains2sOperation, std::complex<double> denIncrement);
    void addMeasurement_timer(MetroChains2sOperation &metroChains2sOperation, std::complex<double> denIncrement);
    void addMeasurement_energy(MetroChains2sOperation &metroChains2sOperation, std::complex<double> denIncrement);
    NiupNidnForce getForce(const NiupNidn &niupNidn, MetroChains2sOperation &metroChains2sOperation, double cap=1.0);
    
    void addEnergy(MetroChains2sOperation &metroChains2sOperation, std::complex<double> denIncrement);

    void write() const;
    void write(std::string postfix) const;
    double getMemory() const;

 private:
    void addGreenMatrix(MetroChains2sOperation &metroChains2sOperation, std::complex<double> denIncrement);
    void addSqSmqNum(MetroChains2sOperation &metroChains2sOperation, std::complex<double> denIncrement);
    void addSSNum(MetroChains2sOperation &metroChains2sOperation, std::complex<double> denIncrement);
    void addDwavePairingNum(MetroChains2sOperation &metroChains2sOperation, std::complex<double> denIncrement);
    std::complex<double> localCdaggerCCdaggerC(MetroChains2sOperation &metroChains2sOperation, size_t i, size_t j, size_t k, size_t l);
    std::complex<double> localCdaggerCCdaggerC_bracket(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixUp_bracket, size_t i, size_t j, size_t k, size_t l);
    std::complex<double> localCdaggerLCLCdaggerLCL_bracket(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixDn_bracket,size_t i, size_t j, size_t k, size_t l);
    std::complex<double> localCdaggerLCLCdaggerC_bracket(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixUp_bracket, const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixDn_bracket,size_t i, size_t j, size_t k, size_t l);
    std::complex<double> localCdaggerCCdaggerLCL_bracket(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixUp_bracket, const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixDn_bracket,size_t i, size_t j, size_t k, size_t l);
    std::complex<double> localCdaggerCLCdaggerLC_bracket(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixUp_bracket, const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixDn_bracket,size_t i, size_t j, size_t k, size_t l);
    std::complex<double> localCdaggerLCCdaggerCL_bracket(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixUp_bracket, const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixDn_bracket,size_t i, size_t j, size_t k, size_t l);

    std::complex<double> localCdaggerLCdaggerCCL_bracket(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixUp_bracket, const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixDn_bracket,size_t i, size_t j, size_t k, size_t l);
    std::complex<double> localCdaggerLCdaggerCLC_bracket(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixUp_bracket, const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixDn_bracket,size_t i, size_t j, size_t k, size_t l);
    std::complex<double> localCdaggerCdaggerLCCL_bracket(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixUp_bracket, const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixDn_bracket,size_t i, size_t j, size_t k, size_t l);
    std::complex<double> localCdaggerCdaggerLCLC_bracket(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixUp_bracket, const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixDn_bracket,size_t i, size_t j, size_t k, size_t l);

    void addDensityDensity(MetroChains2sOperation &metroChains2sOperation, std::complex<double> denIncrement);
   //  void addSplusSminus(MetroChains2sOperation &metroChains2sOperation, std::complex<double> denIncrement);
   //  void addSminusSplus(MetroChains2sOperation &metroChains2sOperation, std::complex<double> denIncrement);
   //  void addSpairSpair(MetroChains2sOperation &metroChains2sOperation, std::complex<double> denIncrement);

    HubbardSOC_icfMeasureObserveMetroChains2sSD2s(const HubbardSOC_icfMeasureObserveMetroChains2sSD2s& x);
    HubbardSOC_icfMeasureObserveMetroChains2sSD2s & operator  = (const HubbardSOC_icfMeasureObserveMetroChains2sSD2s& x);
};

#endif //AFQMCLAB_HUBBARDSOC_ICFMetroChains2SSD2SMEASUREOBSERVE_H