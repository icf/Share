//

#ifndef AFQMCLAB_HUBBARDSOC_ICFSD2SSD2SMEASUREOBSERVE_H
#define AFQMCLAB_HUBBARDSOC_ICFSD2SSD2SMEASUREOBSERVE_H

#include "HubbardSOC_icfMeasureCommuteSD2sSD2s.h"

class HubbardSOC_icfMeasureObserveSD2sSD2s : public HubbardSOC_icfMeasureCommuteSD2sSD2s
{
 private:
   tensor_hao::TensorHao<std::complex<double>, 1> szszNum;
   tensor_hao::TensorHao<std::complex<double>, 1> sxsxNum;
   tensor_hao::TensorHao<std::complex<double>, 1> sysyNum;
   tensor_hao::TensorHao<std::complex<double>, 1>  DDNum;
    tensor_hao::TensorHao<std::complex<double>, 2> greenMatrixUpNum, greenMatrixDnNum;
    tensor_hao::TensorHao<std::complex<double>, 2> densityDensityNum;
    tensor_hao::TensorHao<std::complex<double>, 2> splusSminusNum;
    tensor_hao::TensorHao<std::complex<double>, 2> sminusSplusNum;
    tensor_hao::TensorHao<std::complex<double>, 2> spairSpairNum;

 public:
    HubbardSOC_icfMeasureObserveSD2sSD2s();
    HubbardSOC_icfMeasureObserveSD2sSD2s(const HubbardSOC_icf &hubbardSOC_);
    ~HubbardSOC_icfMeasureObserveSD2sSD2s();

    void reSet();
    void addMeasurement(SD2sSD2sOperation &sd2ssd2sOperation, std::complex<double> denIncrement);
    void addMeasurement_energy(SD2sSD2sOperation &sd2ssd2sOperation, std::complex<double> denIncrement);
    void write() const;
    void write(std::string postfix) const;
    double getMemory() const;

 private:
    void addSS(SD2sSD2sOperation &sd2ssd2sOperation, std::complex<double> denIncrement);
    void addDwavePairingNum(SD2sSD2sOperation &sd2ssd2sOperation, std::complex<double> denIncrement);
    std::complex<double> localCdaggerCCdaggerC(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixUp, size_t i, size_t j, size_t k, size_t l);
    std::complex<double> localCdaggerLCLCdaggerLCL(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixDn, size_t i, size_t j, size_t k, size_t l);
    std::complex<double> localCdaggerLCLCdaggerC(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixUp, const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixDn, size_t i, size_t j, size_t k, size_t l);
    std::complex<double> localCdaggerCCdaggerLCL(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixUp, const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixDn, size_t i, size_t j, size_t k, size_t l);
    std::complex<double> localCdaggerLCCdaggerCL(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixUp, const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixDn, size_t i, size_t j, size_t k, size_t l);
    std::complex<double> localCdaggerCLCdaggerLC(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixUp, const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixDn, size_t i, size_t j, size_t k, size_t l);

    std::complex<double> localCdaggerLCdaggerCCL(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixUp, const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixDn, size_t i, size_t j, size_t k, size_t l);
    std::complex<double> localCdaggerLCdaggerCLC(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixUp, const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixDn, size_t i, size_t j, size_t k, size_t l);
    std::complex<double> localCdaggerCdaggerLCCL(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixUp, const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixDn, size_t i, size_t j, size_t k, size_t l);
    std::complex<double> localCdaggerCdaggerLCLC(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixUp, const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixDn, size_t i, size_t j, size_t k, size_t l);

    void addGreenMatrix(SD2sSD2sOperation &sd2ssd2sOperation, std::complex<double> denIncrement);
    void addDensityDensity(SD2sSD2sOperation &sd2ssd2sOperation, std::complex<double> denIncrement);
    void addSplusSminus(SD2sSD2sOperation &sd2ssd2sOperation, std::complex<double> denIncrement);
    void addSminusSplus(SD2sSD2sOperation &sd2ssd2sOperation, std::complex<double> denIncrement);
    void addSpairSpair(SD2sSD2sOperation &sd2ssd2sOperation, std::complex<double> denIncrement);

    HubbardSOC_icfMeasureObserveSD2sSD2s(const HubbardSOC_icfMeasureObserveSD2sSD2s& x);
    HubbardSOC_icfMeasureObserveSD2sSD2s & operator  = (const HubbardSOC_icfMeasureObserveSD2sSD2s& x);
};

#endif //AFQMCLAB_HUBBARDSOC_ICFSD2SSD2SMEASUREOBSERVE_H