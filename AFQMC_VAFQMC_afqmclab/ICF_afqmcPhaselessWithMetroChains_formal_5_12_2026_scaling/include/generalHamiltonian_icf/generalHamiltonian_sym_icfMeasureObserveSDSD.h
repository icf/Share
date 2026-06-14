//
// Created by icf on 4/18/2024.
//

#ifndef AFQMCLAB_GENERALHAMILTONIAN_SYM_ICFMEASUREOBSERVESDSD_H
#define AFQMCLAB_GENERALHAMILTONIAN_SYM_ICFMEASUREOBSERVESDSD_H

#include "afqmclab.h"
#include "generalHamiltonian_sym_icf.h"

class GeneralHamiltonian_sym_icfMeasureObserveSDSD
{
 public:
    const GeneralHamiltonian_sym_icf *generalHamiltonian_icf;

    std::complex<double> den;
    std::complex<double> TNum;
    tensor_hao::TensorHao<std::complex<double>, 1> svdBgNum;
    tensor_hao::TensorHao<std::complex<double>, 1> svdExNum;
    tensor_hao::TensorHao<std::complex<double>, 1> svdNormalNum;
    std::complex<double> HNum;
    tensor_hao::TensorHao<std::complex<double>, 2> greenNum;
    tensor_hao::TensorHao<std::complex<double>, 1> szszNum;
    tensor_hao::TensorHao<std::complex<double>, 1> sxsxNum;
    tensor_hao::TensorHao<std::complex<double>, 1> sysyNum;

    tensor_hao::TensorHao<std::complex<double>,2> wfDaggerK;
    tensor_hao::TensorHao<std::complex<double>,3> wfDaggerSVDVecs;
    tensor_hao::TensorHao<std::complex<double>,3> wfDaggerSVDVecSquares;

    GeneralHamiltonian_sym_icfMeasureObserveSDSD();
    GeneralHamiltonian_sym_icfMeasureObserveSDSD(const GeneralHamiltonian_sym_icf& generalHamiltonian_icf_);
    ~GeneralHamiltonian_sym_icfMeasureObserveSDSD();

    void initModelNullptr();
    void setModel(const GeneralHamiltonian_sym_icf& generalHamiltonian_icf_);
    void reSet();
    std::complex<double> returnEnergy();
    std::complex<double> returnKEnergy();
    tensor_hao::TensorHao<std::complex<double>, 1> returnSVDBg();
    void addMeasurement(SDSDOperation &sdSDOperation, std::complex<double> denIncrement);
    void addMeasurement_energy(SDSDOperation &sdSDOperation, std::complex<double> denIncrement);
    SVDForce getForce(const SVD_sym &svd, SDSDOperation &sdSDOperation, double cap=1.0);

    void write() const;
    void write(std::string postfix ) const;
    double getMemory() const;

    GeneralHamiltonian_sym_icfMeasureObserveSDSD(const GeneralHamiltonian_sym_icfMeasureObserveSDSD& x);
    GeneralHamiltonian_sym_icfMeasureObserveSDSD & operator  = (const GeneralHamiltonian_sym_icfMeasureObserveSDSD& x);

    void checkWalkerWithModel(const SDSDOperation &sdSDOperation);
    void initWfDaggerK(SDSDOperation &sdSDOperation);
    void initWfDaggerSVDVecs(SDSDOperation &sdSDOperation);
    void initWfDaggerSVDVecSquares(SDSDOperation &sdSDOperation);
    void addEnergy(SDSDOperation &sdSDOperation, std::complex<double> denIncrement);
    void addGreen(SDSDOperation &sdSDOperation, std::complex<double> denIncrement);
    void addSS(SDSDOperation &sdSDOperation, std::complex<double> denIncrement);
    std::complex<double> localCdaggerCCdaggerC(SDSDOperation &sdSDOperation, size_t i, size_t j, size_t k, size_t l);
    std::complex<double> calculateKenergy(SDSDOperation &sdSDOperation);
    tensor_hao::TensorHao<std::complex<double>, 1> calculateSVDBg(SDSDOperation &sdSDOperation);
    tensor_hao::TensorHao<std::complex<double>, 1> calculateSVDEx(SDSDOperation &sdSDOperation);
    tensor_hao::TensorHao<std::complex<double>, 1> calculateSVDNormal(SDSDOperation &sdSDOperation);
};

#endif //AFQMCLAB_GENERALHAMILTONIAN_SYM_ICFMEASUREOBSERVESDSD_H