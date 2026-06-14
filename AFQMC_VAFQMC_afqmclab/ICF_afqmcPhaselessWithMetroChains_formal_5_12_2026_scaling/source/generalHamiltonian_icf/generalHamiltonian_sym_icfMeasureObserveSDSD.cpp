//
// Created by boruoshihao on 11/14/18.
//

#include "../../include/generalHamiltonian_icf/generalHamiltonian_sym_icfMeasureObserveSDSD.h"
#include "afqmclab.h"
#include <math.h>

using namespace std;
using namespace tensor_hao;

GeneralHamiltonian_sym_icfMeasureObserveSDSD::GeneralHamiltonian_sym_icfMeasureObserveSDSD()
{
    initModelNullptr();
    reSet();
}

GeneralHamiltonian_sym_icfMeasureObserveSDSD::GeneralHamiltonian_sym_icfMeasureObserveSDSD(const GeneralHamiltonian_sym_icf &generalHamiltonian_icf_)
{
    setModel(generalHamiltonian_icf_);
    reSet();
}

GeneralHamiltonian_sym_icfMeasureObserveSDSD::~GeneralHamiltonian_sym_icfMeasureObserveSDSD()
{

}

void GeneralHamiltonian_sym_icfMeasureObserveSDSD::initModelNullptr()
{
    generalHamiltonian_icf = nullptr;
}

void GeneralHamiltonian_sym_icfMeasureObserveSDSD::setModel(const GeneralHamiltonian_sym_icf &generalHamiltonian_icf_)
{
    generalHamiltonian_icf = &generalHamiltonian_icf_;
}

void GeneralHamiltonian_sym_icfMeasureObserveSDSD::reSet()
{
    complex<double> zero(0,0);

    den = zero;
    TNum = zero;
    svdBgNum = zero;
    svdExNum = zero;
    HNum = zero;
    greenNum = zero;
    szszNum = zero;
    sxsxNum = zero;
    sysyNum = zero;
}

complex<double> GeneralHamiltonian_sym_icfMeasureObserveSDSD::returnEnergy()
{
    complex<double> Htot   = MPISum(HNum);
    complex<double> denTot = MPISum(den);
    complex<double> energy;
    if( MPIRank() == 0 ) energy = Htot/denTot;
    MPIBcast(energy);
    return energy;
}

complex<double> GeneralHamiltonian_sym_icfMeasureObserveSDSD::returnKEnergy()
{
    complex<double> Ktot   = MPISum(TNum);
    complex<double> denTot = MPISum(den);
    complex<double> Kenergy;
    if( MPIRank() == 0 ) Kenergy = Ktot/denTot;
    MPIBcast(Kenergy);
    return Kenergy;
}

TensorHao<complex<double>,1> GeneralHamiltonian_sym_icfMeasureObserveSDSD::returnSVDBg()
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

void GeneralHamiltonian_sym_icfMeasureObserveSDSD::addMeasurement(SDSDOperation &sdSDOperation, complex<double> denIncrement)
{
    checkWalkerWithModel(sdSDOperation);
    initWfDaggerK(sdSDOperation);
    initWfDaggerSVDVecSquares(sdSDOperation);

    complex<double> denSave; denSave=den;
    den += denIncrement;

    addEnergy(sdSDOperation, denIncrement);
    // addGreen(sdSDOperation, denIncrement);
    addSS(sdSDOperation, denIncrement);

    //Reset matrix, incase we use them for other SDSDOperation
    //This is a temporary change, better way is to avoid using these matrix all the time.
    wfDaggerK.resize(0, 0);
    wfDaggerSVDVecs.resize(0, 0, 0); 
    wfDaggerSVDVecSquares.resize(0, 0, 0); 

}

void GeneralHamiltonian_sym_icfMeasureObserveSDSD::addMeasurement_energy(SDSDOperation &sdSDOperation, complex<double> denIncrement)
{
    checkWalkerWithModel(sdSDOperation);
    initWfDaggerK(sdSDOperation);
    initWfDaggerSVDVecSquares(sdSDOperation);

    complex<double> denSave; denSave=den;
    den += denIncrement;

    addEnergy(sdSDOperation, denIncrement);

    //Reset matrix, incase we use them for other SDSDOperation
    //This is a temporary change, better way is to avoid using these matrix all the time.
    wfDaggerK.resize(0, 0);
    wfDaggerSVDVecs.resize(0, 0, 0); 
    wfDaggerSVDVecSquares.resize(0, 0, 0); 

}


SVDForce GeneralHamiltonian_sym_icfMeasureObserveSDSD::getForce(const SVD_sym &svd,
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

void GeneralHamiltonian_sym_icfMeasureObserveSDSD::write() const
{
    writeThreadSum(den, "den.dat", ios::app);
    // writeThreadSum(TNum, "TNum.dat", ios::app);
    // writeThreadSum(svdBgNum.size(), svdBgNum.data(), "svdBgNum.dat", ios::app);
    // writeThreadSum(svdExNum.size(), svdExNum.data(), "svdExNum.dat", ios::app);
    writeThreadSum(HNum, "HNum.dat", ios::app);
    //writeThreadSum(greenNum.size(), greenNum.data(), "greenNum.dat", ios::app);
}

void GeneralHamiltonian_sym_icfMeasureObserveSDSD::write(std::string postfix ) const
{
    std::string denName="den"+postfix+".dat";
    std::string HNumName="HNum"+postfix+".dat";
    std::string szszNumName="szszNum"+postfix+".dat";
    std::string sxsxNumName="sxsxNum"+postfix+".dat";
    std::string sysyNumName="sysyNum"+postfix+".dat";
    // std::string greenNumName="greenNum"+postfix+".dat";
    writeThreadSum(den, denName, ios::app);
    writeThreadSum(HNum, HNumName, ios::app);
    // writeThreadSum(greenNum.size(), greenNum.data(), greenNumName, ios::app);
    writeThreadSum(szszNum.size(), szszNum.data(), szszNumName, ios::app);
    writeThreadSum(sxsxNum.size(), sxsxNum.data(), sxsxNumName, ios::app);
    writeThreadSum(sysyNum.size(), sysyNum.data(), sysyNumName, ios::app);
}

double GeneralHamiltonian_sym_icfMeasureObserveSDSD::getMemory() const
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

GeneralHamiltonian_sym_icfMeasureObserveSDSD::GeneralHamiltonian_sym_icfMeasureObserveSDSD(const GeneralHamiltonian_sym_icfMeasureObserveSDSD &x)
{

}

GeneralHamiltonian_sym_icfMeasureObserveSDSD & GeneralHamiltonian_sym_icfMeasureObserveSDSD::operator=(const GeneralHamiltonian_sym_icfMeasureObserveSDSD &x)
{
    return *this;
}

void GeneralHamiltonian_sym_icfMeasureObserveSDSD::checkWalkerWithModel(const SDSDOperation &sdSDOperation)
{
    const SD *walkerLeft = sdSDOperation.getWalkerLeft();
    const SD *walkerRight = sdSDOperation.getWalkerRight();

    if( generalHamiltonian_icf->getL() != walkerLeft->getL() ) {cout<<"Model L does not consistent with walker left L!"<<endl; cout<<generalHamiltonian_icf->getL()<<"  "<<walkerLeft->getL()<<endl; exit(1);}
    if( generalHamiltonian_icf->getN() != walkerLeft->getN() ) {cout<<"Model N does not consistent with walker left N!"<<endl; exit(1);}

    if( generalHamiltonian_icf->getL() != walkerRight->getL() ) {cout<<"Model L does not consistent with walker right L!"<<endl; exit(1);}
    if( generalHamiltonian_icf->getN() != walkerRight->getN() ) {cout<<"Model N does not consistent with walker right N!"<<endl; exit(1);}
}

void GeneralHamiltonian_sym_icfMeasureObserveSDSD::initWfDaggerK(SDSDOperation &sdSDOperation)
{
    const SD  *walkerLeft = sdSDOperation.getWalkerLeft();

    size_t L = walkerLeft->getL(); size_t N = walkerLeft->getN();

    wfDaggerK.resize(N, L);

    const TensorHao<complex<double>,2> & K = generalHamiltonian_icf->getK();
    TensorHao<complex<double>,2> K_temp=K;

    BL_NAME(gmm)(walkerLeft->getWf(), K_temp, wfDaggerK, 'C');
}

void GeneralHamiltonian_sym_icfMeasureObserveSDSD::initWfDaggerSVDVecs(SDSDOperation &sdSDOperation)
{
    const SD  *walkerLeft = sdSDOperation.getWalkerLeft();

    size_t L = walkerLeft->getL(); size_t N = walkerLeft->getN(); 
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();

    wfDaggerSVDVecs.resize(N, L, svdNumber); 

    const TensorHao<complex<double>,2> & svdVecs = generalHamiltonian_icf->getSVDVecs();
    TensorHao<complex<double>, 2> svdVecComplex(L,L); svdVecComplex=0.0;
    TensorHaoRef<complex<double>, 2> wfDaggerSVDVec;
    for(size_t k = 0; k < svdNumber ; ++k)
    {
        for(size_t i = 0; i < L; ++i)
        {
            svdVecComplex(i,i) = svdVecs(i,k);
        }
        wfDaggerSVDVec=wfDaggerSVDVecs[k];
        BL_NAME(gmm)(walkerLeft->getWf(), svdVecComplex, wfDaggerSVDVec, 'C');
    }
}

void GeneralHamiltonian_sym_icfMeasureObserveSDSD::initWfDaggerSVDVecSquares(SDSDOperation &sdSDOperation)
{
    initWfDaggerSVDVecs(sdSDOperation);
    //
    const SD  *walkerLeft = sdSDOperation.getWalkerLeft();

    size_t L = walkerLeft->getL(); size_t N = walkerLeft->getN(); 
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();

    wfDaggerSVDVecSquares.resize(N, L, svdNumber); 

    const TensorHao<complex<double>,2> & svdVecs = generalHamiltonian_icf->getSVDVecs();
    TensorHao<complex<double>, 2> svdVecComplex(L,L); svdVecComplex=0.0;
    TensorHaoRef<complex<double>, 2> wfDaggerSVDVec;
    TensorHaoRef<complex<double>, 2> wfDaggerSVDVecSquare;
    for(size_t k = 0; k < svdNumber ; ++k)
    {
        for(size_t i = 0; i < L; ++i)
        {
            svdVecComplex(i,i) = svdVecs(i,i,k);
        }
        wfDaggerSVDVec=wfDaggerSVDVecs[k];
        wfDaggerSVDVecSquare=wfDaggerSVDVecSquares[k];
        BL_NAME(gmm)(wfDaggerSVDVec, svdVecComplex, wfDaggerSVDVecSquare);
    }
}


void GeneralHamiltonian_sym_icfMeasureObserveSDSD::addEnergy(SDSDOperation &sdSDOperation, complex<double> denIncrement)
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

void GeneralHamiltonian_sym_icfMeasureObserveSDSD::addGreen(SDSDOperation &sdSDOperation, complex<double> denIncrement)
{
    const SD  *walkerLeft = sdSDOperation.getWalkerLeft();
    size_t L = generalHamiltonian_icf->getL();size_t N = walkerLeft->getN(); 
    if( greenNum.rank(0) != L ) { greenNum.resize(L,L); greenNum = complex<double>(0,0); }

    greenNum += ( sdSDOperation.returnGreenMatrix() * denIncrement );
}

void GeneralHamiltonian_sym_icfMeasureObserveSDSD::addSS(SDSDOperation &sdSDOperation, complex<double> denIncrement)
{
    size_t L = generalHamiltonian_icf->getL();
    TensorHao<complex<double>, 1> szsz(L); szsz=0.0;
    TensorHao<complex<double>, 1> sxsx(L); sxsx=0.0;
    TensorHao<complex<double>, 1> sysy(L); sysy=0.0;

        if( szszNum.rank(0) != L ) { szszNum.resize(L); szszNum = complex<double>(0,0); }
        if( sxsxNum.rank(0) != L ) { sxsxNum.resize(L); sxsxNum = complex<double>(0,0); }
        if( sysyNum.rank(0) != L ) { sysyNum.resize(L); sysyNum = complex<double>(0,0); }

        for(int j=1-1; j<=L-1; j++){
            int i=0;
            complex<double> localCdaggerCCdaggerC_iijj = localCdaggerCCdaggerC(sdSDOperation,i,i,j,j);
            complex<double> localCdaggerCCdaggerC_iLiLjLjL = localCdaggerCCdaggerC(sdSDOperation,i+L,i+L,j+L,j+L);
            complex<double> localCdaggerCCdaggerC_iLiLjj = localCdaggerCCdaggerC(sdSDOperation,i+L,i+L,j,j);
            complex<double> localCdaggerCCdaggerC_iijLjL = localCdaggerCCdaggerC(sdSDOperation,i,i,j+L,j+L);
            // 
            complex<double> localCdaggerCCdaggerC_iiLjjL = localCdaggerCCdaggerC(sdSDOperation,i,i+L,j,j+L);
            complex<double> localCdaggerCCdaggerC_iLijLj = localCdaggerCCdaggerC(sdSDOperation,i+L,i,j+L,j);
            complex<double> localCdaggerCCdaggerC_iiLjLj = localCdaggerCCdaggerC(sdSDOperation,i,i+L,j+L,j);
            complex<double> localCdaggerCCdaggerC_iLijjL = localCdaggerCCdaggerC(sdSDOperation,i+L,i,j,j+L);
            // 
            szsz(j) += 0.25 *(localCdaggerCCdaggerC_iijj + localCdaggerCCdaggerC_iLiLjLjL - localCdaggerCCdaggerC_iLiLjj - localCdaggerCCdaggerC_iijLjL);
            sxsx(j) += 0.25 *(localCdaggerCCdaggerC_iiLjjL + localCdaggerCCdaggerC_iLijLj + localCdaggerCCdaggerC_iiLjLj + localCdaggerCCdaggerC_iLijjL);
            sysy(j) += 0.25 * (-1.0) * (localCdaggerCCdaggerC_iiLjjL + localCdaggerCCdaggerC_iLijLj - localCdaggerCCdaggerC_iiLjLj - localCdaggerCCdaggerC_iLijjL);
        }
        
        szszNum += ( szsz * denIncrement );
        sxsxNum += ( sxsx * denIncrement );
        sysyNum += ( sysy * denIncrement );
}

complex<double> GeneralHamiltonian_sym_icfMeasureObserveSDSD::localCdaggerCCdaggerC(SDSDOperation &sdSDOperation, size_t i, size_t j, size_t k, size_t l)
{
    size_t L = generalHamiltonian_icf->getL();
    complex<double> local_ijkl = 0.0;
    complex<double> local_ij_kl = 0.0;
    complex<double> local_il_jk = 0.0;
    const TensorHao<complex<double>, 2> greenMatrix = sdSDOperation.returnGreenMatrix();
    local_ij_kl = greenMatrix(i,j) * greenMatrix(k,l);
    if(j != k){
        local_il_jk = greenMatrix(i,l) * (-1.0) * greenMatrix(k,j);
    }else{
        local_il_jk = greenMatrix(i,l) * (1.0 - greenMatrix(k,j));
    }
    local_ijkl = (local_ij_kl + local_il_jk);

    return local_ijkl;
}

complex<double> GeneralHamiltonian_sym_icfMeasureObserveSDSD::calculateKenergy(SDSDOperation &sdSDOperation)
{
    size_t L = generalHamiltonian_icf->getL(); size_t N = generalHamiltonian_icf->getN(); 
    const TensorHao<complex<double>, 2> &theta_T =  sdSDOperation.returnTheta_T();

    complex<double> Kenergy(0,0);
    for(size_t i = 0; i < L; ++i)
    {
        for(size_t j = 0; j < N; ++j) Kenergy += wfDaggerK(j,i) * theta_T(j,i);
    }
    return Kenergy;
}

TensorHao<complex<double>, 1> GeneralHamiltonian_sym_icfMeasureObserveSDSD::calculateSVDBg(SDSDOperation &sdSDOperation)
{
    size_t L = generalHamiltonian_icf->getL(); size_t N = generalHamiltonian_icf->getN(); 
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


TensorHao<complex<double>, 1> GeneralHamiltonian_sym_icfMeasureObserveSDSD::calculateSVDNormal(SDSDOperation &sdSDOperation)
{
    size_t L = generalHamiltonian_icf->getL(); size_t N = generalHamiltonian_icf->getN(); 
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

TensorHao<complex<double>, 1> GeneralHamiltonian_sym_icfMeasureObserveSDSD::calculateSVDEx(SDSDOperation &sdSDOperation)
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