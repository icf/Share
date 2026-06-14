//
// Created by boruoshihao on 11/14/18.
//

#include "../../include/generalHamiltonian_icf/generalHamiltonian_icfMeasureObserveSDSD.h"
#include "afqmclab.h"
#include <math.h>

using namespace std;
using namespace tensor_hao;

GeneralHamiltonian_icfMeasureObserveSDSD::GeneralHamiltonian_icfMeasureObserveSDSD()
{
    initModelNullptr();
    reSet();
}

GeneralHamiltonian_icfMeasureObserveSDSD::GeneralHamiltonian_icfMeasureObserveSDSD(const GeneralHamiltonian_icf &generalHamiltonian_icf_)
{
    setModel(generalHamiltonian_icf_);
    reSet();
}

GeneralHamiltonian_icfMeasureObserveSDSD::~GeneralHamiltonian_icfMeasureObserveSDSD()
{

}

void GeneralHamiltonian_icfMeasureObserveSDSD::initModelNullptr()
{
    generalHamiltonian_icf = nullptr;
}

void GeneralHamiltonian_icfMeasureObserveSDSD::setModel(const GeneralHamiltonian_icf &generalHamiltonian_icf_)
{
    generalHamiltonian_icf = &generalHamiltonian_icf_;
}

void GeneralHamiltonian_icfMeasureObserveSDSD::reSet()
{
    complex<double> zero(0,0);

    den = zero;
    TNum = zero;
    svdBgNum = zero;
    svdExNum = zero;
    HNum = zero;
    greenNum = zero;
}

complex<double> GeneralHamiltonian_icfMeasureObserveSDSD::returnEnergy()
{
    complex<double> Htot   = MPISum(HNum);
    complex<double> denTot = MPISum(den);
    complex<double> energy;
    if( MPIRank() == 0 ) energy = Htot/denTot;
    MPIBcast(energy);
    return energy;
}

complex<double> GeneralHamiltonian_icfMeasureObserveSDSD::returnKEnergy()
{
    complex<double> Ktot   = MPISum(TNum);
    complex<double> denTot = MPISum(den);
    complex<double> Kenergy;
    if( MPIRank() == 0 ) Kenergy = Ktot/denTot;
    MPIBcast(Kenergy);
    return Kenergy;
}

TensorHao<complex<double>,1> GeneralHamiltonian_icfMeasureObserveSDSD::returnSVDBg()
{
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();

    TensorHao<complex<double>, 1> svdBgTot( svdNumber );
    MPISum( svdNumber, svdBgNum.data(), svdBgTot.data() );

    complex<double> denTot = MPISum(den);

    TensorHao<complex<double>,1> svdBg( svdNumber );
    for(size_t i = 0; i < svdNumber; ++i) svdBg(i) = svdBgTot(i)/denTot;
    MPIBcast(svdBg);

    return svdBg;
}

void GeneralHamiltonian_icfMeasureObserveSDSD::addMeasurement(SDSDOperation &sdSDOperation, complex<double> denIncrement)
{
    checkWalkerWithModel(sdSDOperation);
    initWfDaggerK(sdSDOperation);
    initWfDaggerSVDVecSquares(sdSDOperation);

    complex<double> denSave; denSave=den;
    den += denIncrement;

    addEnergy(sdSDOperation, denIncrement);
    // addGreen(sdSDOperation, denIncrement);

    //Reset matrix, incase we use them for other SDSDOperation
    //This is a temporary change, better way is to avoid using these matrix all the time.
    wfDaggerK.resize(0, 0);
    wfDaggerSVDVecs.resize(0, 0, 0); 
    wfDaggerSVDVecSquares.resize(0, 0, 0); 

}


SVDForce GeneralHamiltonian_icfMeasureObserveSDSD::getForce(const SVD &svd,
                                                                      SDSDOperation &sdSDOperation,
                                                                      double cap)
{
    checkWalkerWithModel(sdSDOperation);
    initWfDaggerSVDVecs(sdSDOperation);

    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();
    const TensorHao<complex<double>,1> & currentBg = generalHamiltonian_icf->getSVDBg();
    complex<double> sqrtMinusDt = svd.getSqrtMinusDt();

    TensorHao<complex<double>, 1> svdBg = calculateSVDBg(sdSDOperation);
    SVDForce force(svdNumber); complex<double> oneForce;
    for(size_t i = 0; i < svdNumber; ++i)
    {
        oneForce = (svdBg(i)-currentBg(i)) * sqrtMinusDt;

        if( abs(oneForce) > cap ) force(i) = oneForce*cap/abs(oneForce);
        else force(i) = oneForce;
    }

    //Reset matrix, incase we use them for other SDSDOperation
    //This is a temporary change, better way is to avoid using these matrix all the time.
    wfDaggerSVDVecs.resize(0, 0, 0); 

    return force;
}

void GeneralHamiltonian_icfMeasureObserveSDSD::write() const
{
    writeThreadSum(den, "den.dat", ios::app);
    writeThreadSum(TNum, "TNum.dat", ios::app);
    writeThreadSum(svdBgNum.size(), svdBgNum.data(), "svdBgNum.dat", ios::app);
    writeThreadSum(svdExNum.size(), svdExNum.data(), "svdExNum.dat", ios::app);
    writeThreadSum(HNum, "HNum.dat", ios::app);
    //writeThreadSum(greenNum.size(), greenNum.data(), "greenNum.dat", ios::app);
}

void GeneralHamiltonian_icfMeasureObserveSDSD::write(std::string postfix ) const
{
    std::string denName="den"+postfix+".dat";
    std::string HNumName="HNum"+postfix+".dat";
    std::string greenNumName="greenNum"+postfix+".dat";
    writeThreadSum(den, denName, ios::app);
    writeThreadSum(HNum, HNumName, ios::app);
    writeThreadSum(greenNum.size(), greenNum.data(), greenNumName, ios::app);
}

double GeneralHamiltonian_icfMeasureObserveSDSD::getMemory() const
{
    double mem(0.0);
    mem += 8.0;
    mem += 16.0*3; //den, TNum, HNum
    mem += svdBgNum.getMemory()+svdExNum.getMemory();
    mem += greenNum.getMemory();
    mem += wfDaggerK.getMemory();
    mem += wfDaggerSVDVecs.getMemory() ;
    return mem;
}

GeneralHamiltonian_icfMeasureObserveSDSD::GeneralHamiltonian_icfMeasureObserveSDSD(const GeneralHamiltonian_icfMeasureObserveSDSD &x)
{

}

GeneralHamiltonian_icfMeasureObserveSDSD & GeneralHamiltonian_icfMeasureObserveSDSD::operator=(const GeneralHamiltonian_icfMeasureObserveSDSD &x)
{
    return *this;
}

void GeneralHamiltonian_icfMeasureObserveSDSD::checkWalkerWithModel(const SDSDOperation &sdSDOperation)
{
    const SD *walkerLeft = sdSDOperation.getWalkerLeft();
    const SD *walkerRight = sdSDOperation.getWalkerRight();

    if(generalHamiltonian_icf->getHamiltonian_spin_flag() == true){
        if( generalHamiltonian_icf->getL() != walkerLeft->getL() ) {
            cout<<"Model L does not consistent with walker left L!"<<endl; cout<<generalHamiltonian_icf->getL()<<"  "<<walkerLeft->getL()<<endl; exit(1);
        }
    }else{
        if( 2*generalHamiltonian_icf->getL() != walkerLeft->getL() ) {
            cout<<"Model L does not consistent with walker left L!"<<endl; cout<<generalHamiltonian_icf->getL()<<"  "<<walkerLeft->getL()<<endl; exit(1);
        }
    }

    if( generalHamiltonian_icf->getN() != walkerLeft->getN() ) {cout<<"Model N does not consistent with walker left N!"<<endl; exit(1);}

    if( generalHamiltonian_icf->getN() != walkerRight->getN() ) {cout<<"Model N does not consistent with walker right N!"<<endl; exit(1);}
}

void GeneralHamiltonian_icfMeasureObserveSDSD::initWfDaggerK(SDSDOperation &sdSDOperation)
{
    const SD  *walkerLeft = sdSDOperation.getWalkerLeft();

    size_t L = walkerLeft->getL(); size_t N = walkerLeft->getN();

    wfDaggerK.resize(N, L);

    const TensorHao<complex<double>,2> & K = generalHamiltonian_icf->getK();
    TensorHao<complex<double>,2> K_temp(L,L); K_temp = 0.0;
    for(int i=1-1; i<=L/2-1; i++){
    for(int j=1-1; j<=L/2-1; j++){
        K_temp(i,j)=K(i,j);
        K_temp(i+L/2,j+L/2)=K(i,j);
    }
    }

    BL_NAME(gmm)(walkerLeft->getWf(), K_temp, wfDaggerK, 'C');
}

void GeneralHamiltonian_icfMeasureObserveSDSD::initWfDaggerSVDVecs(SDSDOperation &sdSDOperation)
{
    const SD  *walkerLeft = sdSDOperation.getWalkerLeft();

    size_t L = walkerLeft->getL(); size_t N = walkerLeft->getN(); 
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();

    wfDaggerSVDVecs.resize(N, L, svdNumber); 

    const TensorHaoMPIRef<complex<double>,3> & svdVecs = generalHamiltonian_icf->getSVDVecs();
    TensorHao<complex<double>, 2> svdVecComplex(L,L); svdVecComplex = 0.0;
    TensorHaoRef<complex<double>, 2> wfDaggerSVDVec;
    for(size_t k = 0; k < svdNumber ; ++k)
    {
        for(size_t i = 0; i < L/2; ++i)
        {
            for(size_t j = 0; j < L/2; ++j){
                svdVecComplex(j,i) = svdVecs(j,i,k);
                svdVecComplex(j+L/2,i+L/2) = svdVecs(j,i,k);
            } 
        }
        wfDaggerSVDVec=wfDaggerSVDVecs[k];
        BL_NAME(gmm)(walkerLeft->getWf(), svdVecComplex, wfDaggerSVDVec, 'C');
    }
}

void GeneralHamiltonian_icfMeasureObserveSDSD::initWfDaggerSVDVecSquares(SDSDOperation &sdSDOperation)
{
    initWfDaggerSVDVecs(sdSDOperation);
    //
    const SD  *walkerLeft = sdSDOperation.getWalkerLeft();

    size_t L = walkerLeft->getL(); size_t N = walkerLeft->getN(); 
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();

    wfDaggerSVDVecSquares.resize(N, L, svdNumber); 

    const TensorHaoMPIRef<complex<double>,3> & svdVecs = generalHamiltonian_icf->getSVDVecs();
    TensorHao<complex<double>, 2> svdVecComplex(L,L);
    TensorHaoRef<complex<double>, 2> wfDaggerSVDVec;
    TensorHaoRef<complex<double>, 2> wfDaggerSVDVecSquare;
    for(size_t k = 0; k < svdNumber ; ++k)
    {
        for(size_t i = 0; i < L/2; ++i)
        {
            for(size_t j = 0; j < L/2; ++j){
                svdVecComplex(j,i) = svdVecs(j,i,k);
                svdVecComplex(j+L/2,i+L/2) = svdVecs(j,i,k);
            } 
        }
        wfDaggerSVDVec=wfDaggerSVDVecs[k];
        wfDaggerSVDVecSquare=wfDaggerSVDVecSquares[k];
        BL_NAME(gmm)(wfDaggerSVDVec, svdVecComplex, wfDaggerSVDVecSquare);
    }
}


void GeneralHamiltonian_icfMeasureObserveSDSD::addEnergy(SDSDOperation &sdSDOperation, complex<double> denIncrement)
{
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();

    if( svdBgNum.rank(0) != svdNumber ) { svdBgNum.resize(svdNumber); svdBgNum = complex<double>(0,0); }
    if( svdNormalNum.rank(0) != svdNumber ) { svdNormalNum.resize(svdNumber); svdNormalNum = complex<double>(0,0); }
    if( svdExNum.rank(0) != svdNumber ) { svdExNum.resize(svdNumber); svdExNum = complex<double>(0,0); }

    complex<double> Kenergy=calculateKenergy(sdSDOperation);
    TensorHao<complex<double>, 1> svdBg = calculateSVDBg(sdSDOperation);
    TensorHao<complex<double>, 1> svdNormal = calculateSVDNormal(sdSDOperation);
    TensorHao<complex<double>, 1> svdEx = calculateSVDEx(sdSDOperation);

    complex<double> Henergy(0.0);
    for(size_t i = 0; i < svdNumber; ++i)
    {
        Henergy += ( svdBg(i)*svdBg(i) + svdNormal(i)- svdEx(i) );
    }
    Henergy *= 0.5;
    Henergy += Kenergy;

    TNum += ( Kenergy * denIncrement );
    svdBgNum += ( svdBg * denIncrement );
    svdExNum += ( svdEx * denIncrement );
    HNum += ( Henergy * denIncrement );
}

void GeneralHamiltonian_icfMeasureObserveSDSD::addGreen(SDSDOperation &sdSDOperation, complex<double> denIncrement)
{
    // const SD  *walkerLeft = sdSDOperation.getWalkerLeft();
    // size_t L = generalHamiltonian_icf->getL();size_t N = walkerLeft->getN(); 
    // if( greenNum.rank(0) != L ) { greenNum.resize(L,L); greenNum = complex<double>(0,0); }

    // greenNum += ( sdSDOperation.returnGreenMatrix() * denIncrement );
}

complex<double> GeneralHamiltonian_icfMeasureObserveSDSD::calculateKenergy(SDSDOperation &sdSDOperation)
{
    // 
    size_t L; 
    if( generalHamiltonian_icf->getHamiltonian_spin_flag()){
        L = generalHamiltonian_icf->getL();
    }else{
        L = 2*generalHamiltonian_icf->getL();
    }
    // 
    size_t N = generalHamiltonian_icf->getN(); 
    const TensorHao<complex<double>, 2> &theta_T =  sdSDOperation.returnTheta_T();

    complex<double> Kenergy(0,0);
    for(size_t i = 0; i < L; ++i)
    {
        for(size_t j = 0; j < N; ++j) Kenergy += wfDaggerK(j,i) * theta_T(j,i);
    }
    return Kenergy;
}

TensorHao<complex<double>, 1> GeneralHamiltonian_icfMeasureObserveSDSD::calculateSVDBg(SDSDOperation &sdSDOperation)
{
    // 
    size_t L; 
    if( generalHamiltonian_icf->getHamiltonian_spin_flag()){
        L = generalHamiltonian_icf->getL();
    }else{
        L = 2*generalHamiltonian_icf->getL();
    }
    // 
    size_t N = generalHamiltonian_icf->getN(); 
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();
    const TensorHao<complex<double>, 2> &theta_T =  sdSDOperation.returnTheta_T();

    TensorHaoRef<complex<double>, 2> left(L*N, svdNumber);
    TensorHaoRef<complex<double>, 1> right(L*N);
    left.point( wfDaggerSVDVecs.data() );
    right.point( const_cast<complex<double>*>( theta_T.data() ) );

    TensorHao<complex<double>, 1> svdBg(svdNumber);
    BL_NAME(gemv)(left, right, svdBg, 'T' );

    return svdBg;
}


TensorHao<complex<double>, 1> GeneralHamiltonian_icfMeasureObserveSDSD::calculateSVDNormal(SDSDOperation &sdSDOperation)
{
    // 
    size_t L; 
    if( generalHamiltonian_icf->getHamiltonian_spin_flag()){
        L = generalHamiltonian_icf->getL();
    }else{
        L = 2*generalHamiltonian_icf->getL();
    }
    // 
    size_t N = generalHamiltonian_icf->getN(); 
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();
    const TensorHao<complex<double>, 2> &theta_T =  sdSDOperation.returnTheta_T();

    TensorHao<complex<double>, 1> svdNormal(svdNumber);
    TensorHaoRef<complex<double>, 2> wfDaggerSVDVecSquare;
    for(size_t k = 0; k < svdNumber; ++k)
    {
        wfDaggerSVDVecSquare = wfDaggerSVDVecSquares[k];
        for(size_t i = 0; i < L; ++i)
        {
            for(size_t j = 0; j < N; ++j) svdNormal(k) += wfDaggerSVDVecSquare(j,i) * theta_T(j,i);
        }
    }

    return svdNormal;
}

TensorHao<complex<double>, 1> GeneralHamiltonian_icfMeasureObserveSDSD::calculateSVDEx(SDSDOperation &sdSDOperation)
{
    size_t N = generalHamiltonian_icf->getN(); 
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();
    const TensorHao<complex<double>, 2> &theta_T =  sdSDOperation.returnTheta_T();

    TensorHao<complex<double>, 2> densityL(N, N);
    TensorHaoRef<complex<double>, 2> wfDaggerSVDVec;
    TensorHao<complex<double>, 1> svdEx(svdNumber);

    TensorHao<complex<double>, 2> theta;
    theta = trans(theta_T); 
    for(size_t k = 0; k < svdNumber; ++k)
    {
        wfDaggerSVDVec = wfDaggerSVDVecs[k];
        BL_NAME(gmm)( wfDaggerSVDVec, theta, densityL);

        svdEx(k)=0.0;
        for(size_t i = 0; i < N; ++i) { for(size_t j = 0; j < N; ++j) svdEx(k) += densityL(j,i) * densityL(i,j); }
    }

    return svdEx;
}