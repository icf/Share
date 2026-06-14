//
// Created by icf on 4/18/2024.
//

#ifndef AFQMCLAB_GENERALHAMILTONIAN_ICFMEASUREOBSERVEMETROCHAINS2SSD2S_H
#define AFQMCLAB_GENERALHAMILTONIAN_ICFMEASUREOBSERVEMETROCHAINS2SSD2S_H

#include "afqmclab.h"
#include "generalHamiltonian_icf.h"
#include "../MetroChains2s/metroChains2sOperation.h"

class GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s
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
    tensor_hao::TensorHao<std::complex<double>,2> KPup, KPdn;
    // std::vector<tensor_hao::TensorHao<std::complex<double>, 2>> svdVecComplexUp, svdVecComplexDn;
    // 
    tensor_hao::TensorHao<std::complex<double>,2> A0up, A0dn;
    tensor_hao::TensorHao<std::complex<double>,2> A1up, A1dn;
    tensor_hao::TensorHao<std::complex<double>,2> B0up, B0dn;
    // A0upLgamma: [svdNumber][N_eup x L]
    // A0dnLgamma: [svdNumber][N_edn x L]
    // B0upLgamma: [svdNumber][Dtruncated x L]
    std::vector<tensor_hao::TensorHaoMPIRef<std::complex<double>, 2>> A0upLgamma, A0dnLgamma, B0upLgamma;
    // std::vector<tensor_hao::TensorHao<std::complex<double>, 2>> A0upLgamma_BK, A0dnLgamma_BK, B0upLgamma_BK;
    // 
    const std::vector<tensor_hao::TensorHaoMPIRef<std::complex<double>, 2>> & B0dnLgamma = B0upLgamma;
    // const std::vector<tensor_hao::TensorHao<std::complex<double>, 2>> & B0dnLgamma_BK = B0upLgamma_BK;
    // 
    GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s();
    GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s(const GeneralHamiltonian_icf& generalHamiltonian_icf_, MetroChains2s & walkerLeft);
    ~GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s();

    void initModelNullptr();
    void setModel_withPhiTConst(const GeneralHamiltonian_icf& generalHamiltonian_icf_, MetroChains2s & walkerLeft);
    void reSet();
    std::complex<double> returnEnergy();
    std::complex<double> returnKEnergy();
    tensor_hao::TensorHao<std::complex<double>, 1> returnSVDBg();
    tensor_hao::TensorHao<std::complex<double>, 1> returnSVDBgReal();
    void addMeasurement(MetroChains2s & walkerLeft, std::complex<double> denIncrement);
    void addMeasurement_timer(MetroChains2s & walkerLeft, std::complex<double> denIncrement);
    void addMeasurement(MetroChains2sOperation &metroChains2sOperation, std::complex<double> denIncrement);
    void addMeasurement_timer(MetroChains2sOperation &metroChains2sOperation, std::complex<double> denIncrement);
    SVDForce getForce(const SVD &svd, MetroChains2sOperation &metroChains2sOperation, double cap=1.0);
    SVDForce getForce_fast(const SVD &svd, MetroChains2s & walkerLeftInput, double cap=1.0);

    void write() const;
    void write(std::string postfix ) const;
    double getMemory() const;

    GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s(const GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s& x);
    GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s & operator  = (const GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s& x);

    void checkWalkerWithModel(const MetroChains2sOperation &metroChains2sOperation);
    void checkWalkerWithModel(const MetroChains2s &walkerLeft);
    // void initWfDaggerKList(MetroChains2sOperation &metroChains2sOperation);
    // void initWfDaggerSVDVecsList(MetroChains2sOperation &metroChains2sOperation);
    // void initWfDaggerSVDTVecsList(MetroChains2sOperation &metroChains2sOperation);
    // void initWfDaggerSVDTVecSquaresList(MetroChains2sOperation &metroChains2sOperation);
    // void addEnergy(MetroChains2sOperation &metroChains2sOperation, std::complex<double> denIncrement);
    void addEnergy_fast(MetroChains2s & walkerLeft, std::complex<double> denIncrement);
    void addEnergy_compact(MetroChains2sOperation &metroChains2sOperation, std::complex<double> denIncrement);
    void addGreen(MetroChains2sOperation &metroChains2sOperation, std::complex<double> denIncrement);
    // std::complex<double> calculateKenergy(MetroChains2sOperation &metroChains2sOperation);
    // tensor_hao::TensorHao<std::complex<double>, 1> calculateSVDBg(size_t chain, MetroChains2sOperation &metroChains2sOperation);
    // tensor_hao::TensorHao<std::complex<double>, 1> calculateSVDEx(size_t chain, MetroChains2sOperation &metroChains2sOperation);
    // tensor_hao::TensorHao<std::complex<double>, 1> calculateSVDNormal(size_t chain, MetroChains2sOperation &metroChains2sOperation);
    void get_A0LgammaWR_B0LgammaWR(tensor_hao::TensorHao<std::complex<double>,2> WRup, tensor_hao::TensorHao<std::complex<double>,2> WRdn, std::vector<tensor_hao::TensorHao<std::complex<double>, 2>> & A0upLgammaWRup_vec, std::vector<tensor_hao::TensorHao<std::complex<double>, 2>> & A0dnLgammaWRdn_vec, std::vector<tensor_hao::TensorHao<std::complex<double>, 2>> & B0upLgammaWRup_vec, std::vector<tensor_hao::TensorHao<std::complex<double>, 2>> & B0dnLgammaWRdn_vec);
};

#endif //AFQMCLAB_GENERALHAMILTONIAN_ICFMEASUREOBSERVEMETROCHAINS2SSD2S_H