//
// Created by boruoshihao on 11/14/18.
//

#ifndef AFQMCLAB_SD2SSD2SOperation_ICF_H
#define AFQMCLAB_SD2SSD2SOperation_ICF_H

#include "afqmclab.h"

enum class SD2sSD2sOperation_icf_State
{
    VOID=0,
    LUOVERLAP,
    THETA_T
};

class SD2sSD2sOperation_icf
{
    SD2sSD2sOperation_icf_State state;
    const SD2s *walkerLeft;
    const SD2s *walkerRight;

    tensor_hao::LUDecomp< std::complex<double> > LUOverlapUp, LUOverlapDn;
    tensor_hao::TensorHao< std::complex<double>, 2 > thetaUp_T, thetaDn_T;

    std::complex<double> logOverlap; bool logOverlapIsCalculated;
    tensor_hao::TensorHao< std::complex<double>, 2 > greenMatrixUp, greenMatrixDn; bool greenMatrixUpIsCalculated, greenMatrixDnIsCalculated;
    tensor_hao::TensorHao< std::complex<double>, 1 > greenDiagonalUp, greenDiagonalDn; bool greenDiagonalUpIsCalculated, greenDiagonalDnIsCalculated;
 public:
    SD2sSD2sOperation_icf();
    SD2sSD2sOperation_icf(const SD2s &walkerLeft_, const SD2s &walkerRight_);
    ~SD2sSD2sOperation_icf();

    SD2sSD2sOperation_icf_State getState() const;
    const SD2s *getWalkerLeft() const;
    const SD2s *getWalkerRight() const;
    
    void set(const SD2s &walkerLeft_, const SD2s &walkerRight_);
    void reSet();

    const tensor_hao::LUDecomp<std::complex<double>> &returnLUOverlapUp();
    const tensor_hao::LUDecomp<std::complex<double>> &returnLUOverlapDn();
    const tensor_hao::TensorHao<std::complex<double>, 2> &returnThetaUp_T();
    const tensor_hao::TensorHao<std::complex<double>, 2> &returnThetaDn_T();

    std::complex<double> returnLogOverlap();
    const tensor_hao::TensorHao< std::complex<double>, 2 > &returnGreenMatrixUp();
    const tensor_hao::TensorHao< std::complex<double>, 2 > &returnGreenMatrixDn();
    const tensor_hao::TensorHao< std::complex<double>, 1 > &returnGreenDiagonalUp();
    const tensor_hao::TensorHao< std::complex<double>, 1 > &returnGreenDiagonalDn();

    double getMemory() const;

    SD2sSD2sOperation_icf(const SD2sSD2sOperation_icf& x);
 
 private:
    SD2sSD2sOperation_icf & operator  = (const SD2sSD2sOperation_icf& x);

    void calculateLUOverlap();
    void calculateTheta_T();
};

void setWalkerFromPhiT(std::vector<SD2s> &walker, std::vector<bool> &walkerIsAlive, const SD2s &phiT);

#endif //AFQMCLAB_SD2SSD2SOperation_ICF_H
