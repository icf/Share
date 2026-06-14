//
// Created by boruoshihao on 11/14/18.
//

#include "../../include/MetroChains2s/SD2sSD2sOperation_icf.h"

using namespace std;
using namespace tensor_hao;

SD2sSD2sOperation_icf::SD2sSD2sOperation_icf(): walkerLeft(nullptr), walkerRight(nullptr)
{
    reSet();
}

SD2sSD2sOperation_icf::SD2sSD2sOperation_icf(const SD2s &walkerLeft_, const SD2s &walkerRight_)
{
    set(walkerLeft_, walkerRight_);
}

SD2sSD2sOperation_icf::~SD2sSD2sOperation_icf() { }

SD2sSD2sOperation_icf_State SD2sSD2sOperation_icf::getState() const { return state; }

const SD2s *SD2sSD2sOperation_icf::getWalkerLeft() const { return walkerLeft; }

const SD2s *SD2sSD2sOperation_icf::getWalkerRight() const { return walkerRight; }

void SD2sSD2sOperation_icf::set(const SD2s &walkerLeft_, const SD2s &walkerRight_)
{
    walkerLeft  = &walkerLeft_;
    walkerRight = &walkerRight_;

    size_t L = walkerLeft->getL(); size_t Nup = walkerLeft->getNup(); size_t Ndn = walkerLeft->getNdn();
    if( L != walkerRight->getL() || Nup != walkerRight->getNup() || Ndn != walkerRight->getNdn() )
    {
        cout<<"Error!!! Find Inconsistency between walkerLeft and walkerRight!"<<endl;
        exit(1);
    }

    reSet();
}

void SD2sSD2sOperation_icf::reSet()
{
    state = SD2sSD2sOperation_icf_State::VOID;
    logOverlapIsCalculated = false;
    greenMatrixUpIsCalculated = false;
    greenMatrixDnIsCalculated = false;
    greenDiagonalUpIsCalculated = false;
    greenDiagonalDnIsCalculated = false;
}

const LUDecomp<complex<double>> &SD2sSD2sOperation_icf::returnLUOverlapUp()
{
    calculateLUOverlap();
    return LUOverlapUp;
}

const LUDecomp<complex<double>> &SD2sSD2sOperation_icf::returnLUOverlapDn()
{
    calculateLUOverlap();
    return LUOverlapDn;
}

const TensorHao<complex<double>, 2> &SD2sSD2sOperation_icf::returnThetaUp_T()
{
    calculateLUOverlap();
    calculateTheta_T();
    return thetaUp_T;
}

const TensorHao<complex<double>, 2> &SD2sSD2sOperation_icf::returnThetaDn_T()
{
    calculateLUOverlap();
    calculateTheta_T();
    return thetaDn_T;
}

complex<double> SD2sSD2sOperation_icf::returnLogOverlap()
{
    if(logOverlapIsCalculated) return logOverlap;
    
    calculateLUOverlap();
    logOverlap =conj(walkerLeft->getLogw())+walkerRight->getLogw()+logDeterminant(LUOverlapUp)+logDeterminant(LUOverlapDn);
    
    logOverlapIsCalculated = true;
    
    return logOverlap;
}

const TensorHao<complex<double>, 2> &SD2sSD2sOperation_icf::returnGreenMatrixUp()
{
    if(greenMatrixUpIsCalculated) return greenMatrixUp;
    
    calculateLUOverlap();
    calculateTheta_T();

    size_t L = walkerLeft->getL();
    greenMatrixUp.resize(L,L);
    BL_NAME(gmm)( conj( walkerLeft->getWfUp() ), thetaUp_T, greenMatrixUp );

    greenMatrixUpIsCalculated = true;
    
    return greenMatrixUp;
}

const TensorHao<complex<double>, 2> &SD2sSD2sOperation_icf::returnGreenMatrixDn()
{
    if(greenMatrixDnIsCalculated) return greenMatrixDn;
    
    calculateLUOverlap();
    calculateTheta_T();

    size_t L = walkerLeft->getL();
    greenMatrixDn.resize(L,L);
    BL_NAME(gmm)( conj( walkerLeft->getWfDn() ), thetaDn_T, greenMatrixDn );

    greenMatrixDnIsCalculated = true;
    
    return greenMatrixDn;
}

const TensorHao<complex<double>, 1> &SD2sSD2sOperation_icf::returnGreenDiagonalUp()
{
    if(greenDiagonalUpIsCalculated) return greenDiagonalUp;
    
    calculateLUOverlap();
    calculateTheta_T();

    size_t L = walkerLeft->getL(); size_t Nup = walkerLeft->getNup();
    const TensorHao<complex<double>, 2> &wfLeftUp = walkerLeft->getWfUp();
    greenDiagonalUp.resize(L); greenDiagonalUp = complex<double>(0,0);
    for(size_t j = 0; j < Nup; ++j)
    {
        for(size_t i = 0; i < L; ++i)
        {
            greenDiagonalUp(i) += conj( wfLeftUp(i, j) ) * thetaUp_T(j,i);
        }
    }
    
    greenDiagonalUpIsCalculated= true;
    
    return greenDiagonalUp;
}

const TensorHao<complex<double>, 1> &SD2sSD2sOperation_icf::returnGreenDiagonalDn()
{
    if(greenDiagonalDnIsCalculated) return greenDiagonalDn;
    
    calculateLUOverlap();
    calculateTheta_T();

    size_t L = walkerLeft->getL(); size_t Ndn = walkerLeft->getNdn();
    const TensorHao<complex<double>, 2> &wfLeftDn = walkerLeft->getWfDn();
    greenDiagonalDn.resize(L); greenDiagonalDn = complex<double>(0,0);
    for(size_t j = 0; j < Ndn; ++j)
    {
        for(size_t i = 0; i < L; ++i)
        {
            greenDiagonalDn(i) += conj( wfLeftDn(i, j) ) * thetaDn_T(j,i);
        }
    }
    
    greenDiagonalDnIsCalculated = true;
    
    return greenDiagonalDn;
}

double SD2sSD2sOperation_icf::getMemory() const
{
    return 8.0*2+LUOverlapUp.A.getMemory()+LUOverlapUp.ipiv.getMemory()
                +LUOverlapDn.A.getMemory()+LUOverlapDn.ipiv.getMemory()
                +thetaUp_T.getMemory()+thetaDn_T.getMemory()
                +16.0+1.0+greenMatrixUp.getMemory()+1.0+greenMatrixDn.getMemory()+1.0
                +greenDiagonalUp.getMemory()+1.0+greenDiagonalDn.getMemory()+1.0;
}

SD2sSD2sOperation_icf::SD2sSD2sOperation_icf(const SD2sSD2sOperation_icf &x) { }

SD2sSD2sOperation_icf &SD2sSD2sOperation_icf::operator=(const SD2sSD2sOperation_icf &x) { return *this; }

void SD2sSD2sOperation_icf::calculateLUOverlap()
{
    if( state >= SD2sSD2sOperation_icf_State ::LUOVERLAP ) return;

    size_t Nup = walkerLeft->getNup(); size_t Ndn = walkerLeft->getNdn();

    TensorHao<complex<double>,2> overlapMatrixUp(Nup, Nup), overlapMatrixDn(Ndn, Ndn);
    BL_NAME(gmm)( walkerLeft->getWfUp(), walkerRight->getWfUp(), overlapMatrixUp, 'C' );
    BL_NAME(gmm)( walkerLeft->getWfDn(), walkerRight->getWfDn(), overlapMatrixDn, 'C' );

    LUOverlapUp = BL_NAME(LUconstruct)( move(overlapMatrixUp) );
    LUOverlapDn = BL_NAME(LUconstruct)( move(overlapMatrixDn) );

    state = SD2sSD2sOperation_icf_State ::LUOVERLAP;
}

void SD2sSD2sOperation_icf::calculateTheta_T()
{
    if( state >= SD2sSD2sOperation_icf_State ::THETA_T ) return;

    thetaUp_T =  BL_NAME(solve_lineq)( LUOverlapUp, trans(walkerRight->getWfUp()), 'T' );
    thetaDn_T =  BL_NAME(solve_lineq)( LUOverlapDn, trans(walkerRight->getWfDn()), 'T' );

    state = SD2sSD2sOperation_icf_State ::THETA_T;
}
