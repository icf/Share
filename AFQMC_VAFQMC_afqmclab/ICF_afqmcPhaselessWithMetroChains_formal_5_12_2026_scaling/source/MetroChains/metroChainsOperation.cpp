//
// Created by boruoshihao on 1/10/17.
// Modefied by ICF on 2019-9-29.
//

#include <math.h>
#include "../../include/MetroChains/metroChainsOperation.h"

using namespace std;
using namespace tensor_hao;

MetroChainsOperation::MetroChainsOperation() : walkerLeft(nullptr)
{
    reSet();
}

MetroChainsOperation::MetroChainsOperation(const MetroChains &walkerLeft_)
{
    set(walkerLeft_);
}

MetroChainsOperation::~MetroChainsOperation() { }

const MetroChains *MetroChainsOperation::getWalkerLeft() const { return walkerLeft; }

void MetroChainsOperation::set(const MetroChains &walkerLeft_)
{
    walkerLeft  = &walkerLeft_;

    walkerwalkerOperationList.resize(walkerLeft->getNumOfBrackets() * walkerLeft->getNumOfChains());
    for(int i=1-1; i<=walkerLeft->getNumOfBrackets()* walkerLeft->getNumOfChains()-1; i++){
        walkerwalkerOperationList[i].set(walkerLeft->metroLeft[i],walkerLeft->metroRight[i]);
    }
    
    reSet();
}

void MetroChainsOperation::reSet()
{
    logOverlapIsCalculated = false;
}

const LUDecomp< complex<double> > &MetroChainsOperation::returnLUOverlap(size_t bracket)
{
    return walkerwalkerOperationList[bracket].returnLUOverlap();
}

const TensorHao<complex<double>, 2> &MetroChainsOperation::returnTheta_T(size_t bracket)
{
    return walkerwalkerOperationList[bracket].returnTheta_T();
}

const TensorHao<complex<double>, 2> &MetroChainsOperation::returnGreenMatrix(size_t bracket)
{
    return walkerwalkerOperationList[bracket].returnGreenMatrix();
}


complex<double> MetroChainsOperation::returnSignRatio()
{
    return 0.0;
}

complex<double> MetroChainsOperation::returnLogw()
{
    return walkerLeft->getLogw();
}

complex<double> MetroChainsOperation::returnLogTotalPhase()
{
    if(logOverlapIsCalculated) return logTotalPhase;

    complex <double> totalPhase(0,0);
    for(int i=1-1; i<=walkerLeft->getNumOfBrackets()* walkerLeft->getNumOfChains()-1; i++){
        totalPhase += exp(walkerwalkerOperationList[i].returnLogOverlap());
        if( abs(abs(exp(walkerwalkerOperationList[i].returnLogOverlap())) - 1.0) >= 10e-7){
            cout<<"Error, returnLogTotalPhase is not phase: "<<exp(walkerwalkerOperationList[i].returnLogOverlap())<<"  "<<walkerwalkerOperationList[i].getWalkerLeft()->getLogw()<<"  "<<walkerwalkerOperationList[i].getWalkerRight()->getLogw()<<endl;
        }
    }

    logTotalPhase = log(totalPhase);
    logOverlapIsCalculated = true;
    return logTotalPhase;
}


complex<double> MetroChainsOperation::returnLogTotalPhase_fromCurrentOverlap()
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
        totalPhase += exp(complex<double>(0.0,walkerLeft->metropolisVec[i].metropolisInfo.currentLogOverlap.imag()));
    }
    if( exitFlag){
        exit(1);
    }

    logTotalPhase = log(totalPhase);
    return logTotalPhase;
}

complex<double> MetroChainsOperation::returnLogPhase(size_t bracket)
{
    if( abs(abs(exp(walkerwalkerOperationList[bracket].returnLogOverlap())) - 1.0) >= 10e-7){
        cout<<"Error, returnLogPhase is not phase"<<endl;
    }
    return walkerwalkerOperationList[bracket].returnLogOverlap();
}


complex<double> MetroChainsOperation::returnLogPhase_fromCurrentOverlap(size_t bracket)
{
    if(walkerLeft->getNumOfBrackets() != 1){
        cout<<"Error, returnLogPhase_fromCurrentOverlap is not implemented for more than one bracket"<<endl;
        exit(1);
    }
    // 
    return complex<double>(0.0,walkerLeft->metropolisVec[bracket].metropolisInfo.currentLogOverlap.imag());
}

double MetroChainsOperation::getMemory() const
{
    return 0.0;
}

MetroChainsOperation::MetroChainsOperation(const MetroChainsOperation &x) { }

MetroChainsOperation &MetroChainsOperation::operator=(const MetroChainsOperation &x) { return *this; }

void setWalkerFromPhiT(vector<SD> &walker, vector<bool> &walkerIsAlive, const MetroChains &phiT)
{

}