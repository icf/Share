//
// Created by boruoshihao on 1/10/17.
// Modefied by ICF on 2019-9-29.
//

#include <math.h>
#include "../../include/MetroChains2s/metroChains2sOperation.h"

using namespace std;
using namespace tensor_hao;

MetroChains2sOperation::MetroChains2sOperation() : walkerLeft(nullptr)
{
    reSet();
}

MetroChains2sOperation::MetroChains2sOperation(const MetroChains2s &walkerLeft_)
{
    set(walkerLeft_);
}

MetroChains2sOperation::~MetroChains2sOperation() { }

const MetroChains2s *MetroChains2sOperation::getWalkerLeft() const { return walkerLeft; }

void MetroChains2sOperation::set(const MetroChains2s &walkerLeft_)
{
    walkerLeft  = &walkerLeft_;

    walkerwalkerOperationList.resize(walkerLeft->getNumOfBrackets() * walkerLeft->getNumOfChains());
    for(int i=1-1; i<=walkerLeft->getNumOfBrackets()* walkerLeft->getNumOfChains()-1; i++){
        walkerwalkerOperationList[i].set(walkerLeft->metroLeft[i],walkerLeft->metroRight[i]);
    }
    
    reSet();
}

void MetroChains2sOperation::reSet()
{
    logOverlapIsCalculated = false;
}

const LUDecomp< complex<double> > &MetroChains2sOperation::returnLUOverlapUp(size_t bracket)
{
    return walkerwalkerOperationList[bracket].returnLUOverlapUp();
}

const LUDecomp< complex<double> > &MetroChains2sOperation::returnLUOverlapDn(size_t bracket)
{
    return walkerwalkerOperationList[bracket].returnLUOverlapDn();
}

const TensorHao<complex<double>, 2> &MetroChains2sOperation::returnThetaUp_T(size_t bracket)
{
    return walkerwalkerOperationList[bracket].returnThetaUp_T();
}

const TensorHao<complex<double>, 2> &MetroChains2sOperation::returnThetaDn_T(size_t bracket)
{
    return walkerwalkerOperationList[bracket].returnThetaDn_T();
}

const TensorHao<complex<double>, 2> &MetroChains2sOperation::returnGreenMatrixUp(size_t bracket)
{
    return walkerwalkerOperationList[bracket].returnGreenMatrixUp();
}

const TensorHao<complex<double>, 2> &MetroChains2sOperation::returnGreenMatrixDn(size_t bracket)
{
    return walkerwalkerOperationList[bracket].returnGreenMatrixDn();
}

complex<double> MetroChains2sOperation::returnSignRatio()
{
    return 0.0;
}

complex<double> MetroChains2sOperation::returnLogw()
{
    return walkerLeft->getLogw();
}

complex<double> MetroChains2sOperation::returnLogTotalPhase()
{
    if(logOverlapIsCalculated) return logTotalPhase;
    // 
    bool exitFlag = false;
    // 
    complex <double> totalPhase(0,0);
    for(int i=1-1; i<=walkerLeft->getNumOfBrackets()* walkerLeft->getNumOfChains()-1; i++){
        totalPhase += exp(walkerwalkerOperationList[i].returnLogOverlap());
        if( abs(abs(exp(walkerwalkerOperationList[i].returnLogOverlap())) - 1.0) >= 10e-7){
            cout<<"Error, returnLogTotalPhase is not phase: "<<exp(walkerwalkerOperationList[i].returnLogOverlap())<<"  "<<walkerwalkerOperationList[i].returnLogOverlap()<<"  "<<walkerwalkerOperationList[i].getWalkerLeft()->getLogw()<<"  "<<walkerwalkerOperationList[i].getWalkerRight()->getLogw()<<endl;
            cout<<"MPIRank(): "<<MPIRank()<<" chain "<<i<<" totalPhase: "<<totalPhase<<endl;
            exitFlag = true;
        }
    }
    if( exitFlag){
        exit(1);
    }

    logTotalPhase = log(totalPhase);
    logOverlapIsCalculated = true;
    return logTotalPhase;
}

complex<double> MetroChains2sOperation::returnLogTotalPhase_fromCurrentOverlap()
{
    bool exitFlag = false;
    // 
    if(walkerLeft->getNumOfBrackets() != 1){
        cout<<"Error, returnLogTotalPhase_fromCurrentOverlap is not implemented for more than one bracket"<<endl;
        exit(1);
    }
    // 
    complex <double> totalPhase(0,0);
    for(int i=1-1; i<=walkerLeft->getNumOfChains()-1; i++){
        totalPhase += exp(complex<double>(0.0,walkerLeft->metropolis2sVec[i].metropolis2sInfo.currentLogOverlap.imag()));
    }
    if( exitFlag){
        exit(1);
    }

    logTotalPhase = log(totalPhase);
    return logTotalPhase;
}


complex<double> MetroChains2sOperation::returnLogPhase(size_t bracket)
{
    if( abs(abs(exp(walkerwalkerOperationList[bracket].returnLogOverlap())) - 1.0) >= 10e-7){
        cout<<"Error, returnLogPhase is not phase"<<endl;
    }
    return walkerwalkerOperationList[bracket].returnLogOverlap();
}

complex<double> MetroChains2sOperation::returnLogPhase_fromCurrentOverlap(size_t bracket)
{
    if(walkerLeft->getNumOfBrackets() != 1){
        cout<<"Error, returnLogPhase_fromCurrentOverlap is not implemented for more than one bracket"<<endl;
        exit(1);
    }
    // 
    return complex<double>(0.0,walkerLeft->metropolis2sVec[bracket].metropolis2sInfo.currentLogOverlap.imag());
}


double MetroChains2sOperation::getMemory() const
{
    return 0.0;
}

MetroChains2sOperation::MetroChains2sOperation(const MetroChains2sOperation &x) { }

MetroChains2sOperation &MetroChains2sOperation::operator=(const MetroChains2sOperation &x) { return *this; }

void setWalkerFromPhiT(vector<SD> &walker, vector<bool> &walkerIsAlive, const MetroChains2s &phiT)
{

}