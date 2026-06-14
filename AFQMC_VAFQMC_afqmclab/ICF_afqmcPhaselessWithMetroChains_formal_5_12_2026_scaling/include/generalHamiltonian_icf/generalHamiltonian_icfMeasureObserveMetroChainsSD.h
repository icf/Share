//
// Created by icf on 4/18/2024.
//

#ifndef AFQMCLAB_GENERALHAMILTONIAN_ICFMEASUREOBSERVEMETROCHAINSSD_H
#define AFQMCLAB_GENERALHAMILTONIAN_ICFMEASUREOBSERVEMETROCHAINSSD_H

#include "afqmclab.h"
#include "generalHamiltonian_icf.h"
#include "tensor_hao_mpi_shared.h"
#include "../MetroChains/metroChainsOperation.h"

class GeneralHamiltonian_icfMeasureObserveMetroChainsSD
{ 
 public:
    const GeneralHamiltonian_icf *generalHamiltonian_icf;

    std::complex<double> den;
    std::complex<double> TNum;
    tensor_hao::TensorHao<std::complex<double>, 1> svdBgNum;
    tensor_hao::TensorHao<std::complex<double>, 1> svdExNum;
    tensor_hao::TensorHao<std::complex<double>, 1> svdNormalNum;
    std::complex<double> HNum;

    //memory for speed
    tensor_hao::TensorHao<std::complex<double>,2> KP;
    // 
    tensor_hao::TensorHao<std::complex<double>,2> A0;
    tensor_hao::TensorHao<std::complex<double>,2> A1;
    tensor_hao::TensorHao<std::complex<double>,2> B0;
    // A0Lgamma: [svdNumber][N_e x L]
    // B0Lgamma: [svdNumber][Dtruncated x L]
    std::vector<tensor_hao::TensorHaoMPIRef<std::complex<double>, 2>> A0Lgamma, B0Lgamma;

    // 
    GeneralHamiltonian_icfMeasureObserveMetroChainsSD();
    GeneralHamiltonian_icfMeasureObserveMetroChainsSD(const GeneralHamiltonian_icf& generalHamiltonian_icf_, MetroChains & walkerLeft);
    ~GeneralHamiltonian_icfMeasureObserveMetroChainsSD();

    void initModelNullptr();
    void setModel_withPhiTConst(const GeneralHamiltonian_icf& generalHamiltonian_icf_, MetroChains & walkerLeft);
    void reSet();
    std::complex<double> returnEnergy();
    std::complex<double> returnKEnergy();
    tensor_hao::TensorHao<std::complex<double>, 1> returnSVDBg();
    tensor_hao::TensorHao<std::complex<double>, 1> returnSVDBgReal();
    void addMeasurement(MetroChains & walkerLeft, std::complex<double> denIncrement);
    void addMeasurement_timer(MetroChains & walkerLeft, std::complex<double> denIncrement);
    void addMeasurement(MetroChainsOperation &metroChainsOperation, std::complex<double> denIncrement);
    void addMeasurement_timer(MetroChainsOperation &metroChainsOperation, std::complex<double> denIncrement);
    SVDForce getForce(const SVD &svd, MetroChainsOperation &metroChainsOperation, double cap=1.0);
    SVDForce getForce_fast(const SVD &svd, MetroChains & walkerLeftInput, double cap=1.0);

    void write() const;
    void write(std::string postfix ) const;
    double getMemory() const;

    GeneralHamiltonian_icfMeasureObserveMetroChainsSD(const GeneralHamiltonian_icfMeasureObserveMetroChainsSD& x);
    GeneralHamiltonian_icfMeasureObserveMetroChainsSD & operator  = (const GeneralHamiltonian_icfMeasureObserveMetroChainsSD& x);

    void checkWalkerWithModel(const MetroChainsOperation &metroChainsOperation);
    void checkWalkerWithModel(const MetroChains &walkerLeft);
    void addEnergy_fast(MetroChains & walkerLeft, std::complex<double> denIncrement);
    void addEnergy_compact(MetroChainsOperation &metroChainsOperation, std::complex<double> denIncrement);
    void addGreen(MetroChainsOperation &metroChainsOperation, std::complex<double> denIncrement);
    void get_A0LgammaWR_B0LgammaWR(tensor_hao::TensorHao<std::complex<double>,2> WR, std::vector<tensor_hao::TensorHao<std::complex<double>, 2>> & A0LgammaWR_vec, std::vector<tensor_hao::TensorHao<std::complex<double>, 2>> & B0LgammaWR_vec);
};

#endif //AFQMCLAB_GENERALHAMILTONIAN_ICFMEASUREOBSERVEMETROCHAINSSD_H