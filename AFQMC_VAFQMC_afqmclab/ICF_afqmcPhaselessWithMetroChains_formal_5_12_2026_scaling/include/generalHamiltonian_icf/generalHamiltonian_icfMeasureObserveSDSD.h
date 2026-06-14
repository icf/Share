//
// Created by icf on 4/18/2024.
//

#ifndef AFQMCLAB_GENERALHAMILTONIAN_ICFMEASUREOBSERVESDSD_H
#define AFQMCLAB_GENERALHAMILTONIAN_ICFMEASUREOBSERVESDSD_H

#include "afqmclab.h"
#include "generalHamiltonian_icf.h"

class GeneralHamiltonian_icfMeasureObserveSDSD
{
 public:
    const GeneralHamiltonian_icf *generalHamiltonian_icf;

    std::complex<double> den;
    std::complex<double> TNum;
    tensor_hao::TensorHao<std::complex<double>, 1> svdBgNum;
    tensor_hao::TensorHao<std::complex<double>, 1> svdExNum;
    tensor_hao::TensorHao<std::complex<double>, 1> svdNormalNum;
    std::complex<double> HNum;
    tensor_hao::TensorHao<std::complex<double>, 2> greenNum;

    tensor_hao::TensorHao<std::complex<double>,2> wfDaggerK;
    tensor_hao::TensorHao<std::complex<double>,3> wfDaggerSVDVecs;
    tensor_hao::TensorHao<std::complex<double>,3> wfDaggerSVDVecSquares;

    GeneralHamiltonian_icfMeasureObserveSDSD();
    GeneralHamiltonian_icfMeasureObserveSDSD(const GeneralHamiltonian_icf& generalHamiltonian_icf_);
    ~GeneralHamiltonian_icfMeasureObserveSDSD();

    void initModelNullptr();
    void setModel(const GeneralHamiltonian_icf& generalHamiltonian_icf_);
    void reSet();
    std::complex<double> returnEnergy();
    std::complex<double> returnKEnergy();
    tensor_hao::TensorHao<std::complex<double>, 1> returnSVDBg();
    void addMeasurement(SDSDOperation &sdSDOperation, std::complex<double> denIncrement);
    SVDForce getForce(const SVD &svd, SDSDOperation &sdSDOperation, double cap=1.0);

    void write() const;
    void write(std::string postfix ) const;
    double getMemory() const;

    GeneralHamiltonian_icfMeasureObserveSDSD(const GeneralHamiltonian_icfMeasureObserveSDSD& x);
    GeneralHamiltonian_icfMeasureObserveSDSD & operator  = (const GeneralHamiltonian_icfMeasureObserveSDSD& x);

    void checkWalkerWithModel(const SDSDOperation &sdSDOperation);
    void initWfDaggerK(SDSDOperation &sdSDOperation);
    void initWfDaggerSVDVecs(SDSDOperation &sdSDOperation);
    void initWfDaggerSVDVecSquares(SDSDOperation &sdSDOperation);
    void addEnergy(SDSDOperation &sdSDOperation, std::complex<double> denIncrement);
    void addGreen(SDSDOperation &sdSDOperation, std::complex<double> denIncrement);
    std::complex<double> calculateKenergy(SDSDOperation &sdSDOperation);
    tensor_hao::TensorHao<std::complex<double>, 1> calculateSVDBg(SDSDOperation &sdSDOperation);
    tensor_hao::TensorHao<std::complex<double>, 1> calculateSVDEx(SDSDOperation &sdSDOperation);
    tensor_hao::TensorHao<std::complex<double>, 1> calculateSVDNormal(SDSDOperation &sdSDOperation);
};

#endif //AFQMCLAB_GENERALHAMILTONIAN_ICFMEASUREOBSERVESDSD_H