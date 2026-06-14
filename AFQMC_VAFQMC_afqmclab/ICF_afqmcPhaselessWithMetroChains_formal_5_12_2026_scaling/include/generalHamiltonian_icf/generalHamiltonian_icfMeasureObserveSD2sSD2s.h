//
// Created by icf on 4/18/2024.
//

#ifndef AFQMCLAB_GENERALHAMILTONIAN_ICFMEASUREOBSERVESD2SSD2S_H
#define AFQMCLAB_GENERALHAMILTONIAN_ICFMEASUREOBSERVESD2SSD2S_H

#include "afqmclab.h"
#include "generalHamiltonian_icf.h"

class GeneralHamiltonian_icfMeasureObserveSD2sSD2s
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

    tensor_hao::TensorHao<std::complex<double>,2> wfUpDaggerK, wfDnDaggerK;
    // tensor_hao::TensorHao<std::complex<double>,3> wfUpDaggerSVDVecs, wfDnDaggerSVDVecs;
    // tensor_hao::TensorHao<std::complex<double>,3> wfUpDaggerSVDVecSquares, wfDnDaggerSVDVecSquares;

    GeneralHamiltonian_icfMeasureObserveSD2sSD2s();
    GeneralHamiltonian_icfMeasureObserveSD2sSD2s(const GeneralHamiltonian_icf& generalHamiltonian_icf_);
    ~GeneralHamiltonian_icfMeasureObserveSD2sSD2s();

    void initModelNullptr();
    void setModel(const GeneralHamiltonian_icf& generalHamiltonian_icf_);
    void reSet();
    std::complex<double> returnEnergy();
    std::complex<double> returnKEnergy();
    tensor_hao::TensorHao<std::complex<double>, 1> returnSVDBg();
    void addMeasurement(SD2sSD2sOperation &sd2ssd2sOperation, std::complex<double> denIncrement);
    SVDForce getForce(const SVD &svd, SD2sSD2sOperation &sd2ssd2sOperation, double cap=1.0);

    void write() const;
    void write(std::string postfix ) const;
    double getMemory() const;

    GeneralHamiltonian_icfMeasureObserveSD2sSD2s(const GeneralHamiltonian_icfMeasureObserveSD2sSD2s& x);
    GeneralHamiltonian_icfMeasureObserveSD2sSD2s & operator  = (const GeneralHamiltonian_icfMeasureObserveSD2sSD2s& x);

    void checkWalkerWithModel(const SD2sSD2sOperation &sd2ssd2sOperation);
    void initWfDaggerK(SD2sSD2sOperation &sd2ssd2sOperation);
    // void initWfDaggerSVDVecs(SD2sSD2sOperation &sd2ssd2sOperation);
    // void initWfDaggerSVDVecSquares(SD2sSD2sOperation &sd2ssd2sOperation);
    void addEnergy(SD2sSD2sOperation &sd2ssd2sOperation, std::complex<double> denIncrement);
    void addGreen(SD2sSD2sOperation &sd2ssd2sOperation, std::complex<double> denIncrement);
    std::complex<double> calculateKenergy(SD2sSD2sOperation &sd2ssd2sOperation);
    tensor_hao::TensorHao<std::complex<double>, 1> calculateSVDBg(SD2sSD2sOperation &sd2ssd2sOperation);
    tensor_hao::TensorHao<std::complex<double>, 1> calculateSVDEx(SD2sSD2sOperation &sd2ssd2sOperation);
    tensor_hao::TensorHao<std::complex<double>, 1> calculateSVDNormal(SD2sSD2sOperation &sd2ssd2sOperation);
};

#endif //AFQMCLAB_GENERALHAMILTONIAN_ICFMEASUREOBSERVESD2SSD2S_H