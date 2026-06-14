//
// Created by boruoshihao on 1/10/17.
//

#ifndef AFQMCLAB_SDSDOPERATION_ICF_H
#define AFQMCLAB_SDSDOPERATION_ICF_H

#include "afqmclab.h"

enum class SDSDOperation_icfState
{
    VOID=0,
    LUOVERLAP,
    THETA_T
};

class SDSDOperation_icf
{
    SDSDOperation_icfState state;
    const SD *walkerLeft;
    const SD *walkerRight;

    tensor_hao::LUDecomp< std::complex<double> > LUOverlap;
    tensor_hao::TensorHao< std::complex<double>, 2 > theta_T;

    std::complex<double> logOverlap; bool logOverlapIsCalculated;
    tensor_hao::TensorHao< std::complex<double>, 2 > greenMatrix; bool greenMatrixIsCalculated;
    tensor_hao::TensorHao< std::complex<double>, 1 > greenDiagonal; bool greenDiagonalIsCalculated;
    tensor_hao::TensorHao< std::complex<double>, 1 > greenOffDiagonal; bool greenOffDiagonalIsCalculated;
 public:
    SDSDOperation_icf();
    SDSDOperation_icf(const SD &walkerLeft_, const SD &walkerRight_);
    ~SDSDOperation_icf();

    SDSDOperation_icfState getState() const;
    const SD *getWalkerLeft() const;
    const SD *getWalkerRight() const;
    
    void set(const SD &walkerLeft_, const SD &walkerRight_);
    void reSet();
    
    const tensor_hao::LUDecomp<std::complex<double>> &returnLUOverlap();
    const tensor_hao::TensorHao<std::complex<double>, 2> &returnTheta_T();
    std::complex<double> returnLogOverlap();
    const tensor_hao::TensorHao< std::complex<double>, 2 > &returnGreenMatrix();
    const tensor_hao::TensorHao< std::complex<double>, 1 > &returnGreenDiagonal();
    const tensor_hao::TensorHao< std::complex<double>, 1 > &returnGreenOffDiagonal();

    double getMemory() const;

    SDSDOperation_icf(const SDSDOperation_icf& x);

 private:
    SDSDOperation_icf & operator  = (const SDSDOperation_icf& x);

    void calculateLUOverlap();
    void calculateTheta_T();
};


#endif //AFQMCLAB_SDSDOPERATION_ICF_H