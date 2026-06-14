//
// Created by boruoshihao on 11/14/18.
//

#include "../../include/generalHamiltonian_icf/generalHamiltonian_icfMeasureObserveSD2sSD2s.h"
#include "afqmclab.h"
#include <math.h>

using namespace std;
using namespace tensor_hao;

GeneralHamiltonian_icfMeasureObserveSD2sSD2s::GeneralHamiltonian_icfMeasureObserveSD2sSD2s()
{
    initModelNullptr();
    reSet();
}

GeneralHamiltonian_icfMeasureObserveSD2sSD2s::GeneralHamiltonian_icfMeasureObserveSD2sSD2s(const GeneralHamiltonian_icf &generalHamiltonian_icf_)
{
    setModel(generalHamiltonian_icf_);
    reSet();
}

GeneralHamiltonian_icfMeasureObserveSD2sSD2s::~GeneralHamiltonian_icfMeasureObserveSD2sSD2s()
{

}

void GeneralHamiltonian_icfMeasureObserveSD2sSD2s::initModelNullptr()
{
    generalHamiltonian_icf = nullptr;
}

void GeneralHamiltonian_icfMeasureObserveSD2sSD2s::setModel(const GeneralHamiltonian_icf &generalHamiltonian_icf_)
{
    generalHamiltonian_icf = &generalHamiltonian_icf_;
}

void GeneralHamiltonian_icfMeasureObserveSD2sSD2s::reSet()
{
    complex<double> zero(0,0);

    den = zero;
    TNum = zero;
    svdBgNum = zero;
    svdExNum = zero;
    HNum = zero;
    // greenNum = zero;
}

complex<double> GeneralHamiltonian_icfMeasureObserveSD2sSD2s::returnEnergy()
{
    complex<double> Htot   = MPISum(HNum);
    complex<double> denTot = MPISum(den);
    complex<double> energy;
    if( MPIRank() == 0 ) energy = Htot/denTot;
    MPIBcast(energy);
    return energy;
}

complex<double> GeneralHamiltonian_icfMeasureObserveSD2sSD2s::returnKEnergy()
{
    complex<double> Ktot   = MPISum(TNum);
    complex<double> denTot = MPISum(den);
    complex<double> Kenergy;
    if( MPIRank() == 0 ) Kenergy = Ktot/denTot;
    MPIBcast(Kenergy);
    return Kenergy;
}

TensorHao<complex<double>,1> GeneralHamiltonian_icfMeasureObserveSD2sSD2s::returnSVDBg()
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

void GeneralHamiltonian_icfMeasureObserveSD2sSD2s::addMeasurement(SD2sSD2sOperation &sd2ssd2sOperation, complex<double> denIncrement)
{
    checkWalkerWithModel(sd2ssd2sOperation);
    initWfDaggerK(sd2ssd2sOperation);
    // initWfDaggerSVDVecs(sd2ssd2sOperation);
    // initWfDaggerSVDVecSquares(sd2ssd2sOperation); // Removed: compute on-the-fly in calculateSVDNormal

    complex<double> denSave; denSave=den;
    den += denIncrement;

    addEnergy(sd2ssd2sOperation, denIncrement);
    // addGreen(sd2ssd2sOperation, denIncrement);

    //Reset matrix, incase we use them for other SD2sSD2sOperation
    //This is a temporary change, better way is to avoid using these matrix all the time.
    wfUpDaggerK.resize(0, 0);
    wfDnDaggerK.resize(0, 0);
    // wfUpDaggerSVDVecs.resize(0, 0, 0); 
    // wfDnDaggerSVDVecs.resize(0, 0, 0);
    // wfUpDaggerSVDVecSquares.resize(0, 0, 0); // Removed: no longer stored
    // wfDnDaggerSVDVecSquares.resize(0, 0, 0); // Removed: no longer stored

}


SVDForce GeneralHamiltonian_icfMeasureObserveSD2sSD2s::getForce(const SVD &svd,
                                                                      SD2sSD2sOperation &sd2ssd2sOperation,
                                                                      double cap)
{
    checkWalkerWithModel(sd2ssd2sOperation);
    // initWfDaggerSVDVecs(sd2ssd2sOperation);

    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();
    const TensorHao<complex<double>,1> & currentBg = generalHamiltonian_icf->getSVDBg();
    complex<double> sqrtMinusDt = svd.getSqrtMinusDt();

    TensorHao<complex<double>, 1> svdBg = calculateSVDBg(sd2ssd2sOperation);
    SVDForce force(svdNumber); complex<double> oneForce;
    for(size_t i = 0; i < svdNumber; ++i)
    {
        oneForce = (svdBg(i)-currentBg(i)) * sqrtMinusDt;

        if( abs(oneForce) > cap ) force(i) = oneForce*cap/abs(oneForce);
        else force(i) = oneForce;
    }

    //Reset matrix, incase we use them for other SD2sSD2sOperation
    //This is a temporary change, better way is to avoid using these matrix all the time.
    // wfUpDaggerSVDVecs.resize(0, 0, 0); 
    // wfDnDaggerSVDVecs.resize(0, 0, 0); 

    return force;
}

void GeneralHamiltonian_icfMeasureObserveSD2sSD2s::write() const
{
    writeThreadSum(den, "den.dat", ios::app);
    writeThreadSum(TNum, "TNum.dat", ios::app);
    writeThreadSum(svdBgNum.size(), svdBgNum.data(), "svdBgNum.dat", ios::app);
    writeThreadSum(svdExNum.size(), svdExNum.data(), "svdExNum.dat", ios::app);
    writeThreadSum(HNum, "HNum.dat", ios::app);
    //writeThreadSum(greenNum.size(), greenNum.data(), "greenNum.dat", ios::app);
}

void GeneralHamiltonian_icfMeasureObserveSD2sSD2s::write(std::string postfix ) const
{
    std::string denName="den"+postfix+".dat";
    std::string HNumName="HNum"+postfix+".dat";
    // std::string greenNumName="greenNum"+postfix+".dat";
    writeThreadSum(den, denName, ios::app);
    writeThreadSum(HNum, HNumName, ios::app);
    // writeThreadSum(greenNum.size(), greenNum.data(), greenNumName, ios::app);
}

double GeneralHamiltonian_icfMeasureObserveSD2sSD2s::getMemory() const
{
    double mem(0.0);
    mem += 8.0;
    mem += 16.0*3; //den, TNum, HNum
    mem += svdBgNum.getMemory()+svdExNum.getMemory();
    // mem += greenNum.getMemory();
    mem += wfUpDaggerK.getMemory();
    mem += wfDnDaggerK.getMemory();
    // mem += wfUpDaggerSVDVecs.getMemory() ;
    // mem += wfDnDaggerSVDVecs.getMemory() ;
    return mem;
}

GeneralHamiltonian_icfMeasureObserveSD2sSD2s::GeneralHamiltonian_icfMeasureObserveSD2sSD2s(const GeneralHamiltonian_icfMeasureObserveSD2sSD2s &x)
{

}

GeneralHamiltonian_icfMeasureObserveSD2sSD2s & GeneralHamiltonian_icfMeasureObserveSD2sSD2s::operator=(const GeneralHamiltonian_icfMeasureObserveSD2sSD2s &x)
{
    return *this;
}

void GeneralHamiltonian_icfMeasureObserveSD2sSD2s::checkWalkerWithModel(const SD2sSD2sOperation &sd2ssd2sOperation)
{
    const SD2s *walkerLeft = sd2ssd2sOperation.getWalkerLeft();
    const SD2s *walkerRight = sd2ssd2sOperation.getWalkerRight();

    if(generalHamiltonian_icf->getHamiltonian_spin_flag() == true){
        if( generalHamiltonian_icf->getL() != 2*walkerLeft->getL() ) {cout<<"Model L does not consistent with walker left L!"<<endl; exit(1);}
    }else{
        if( generalHamiltonian_icf->getL() != walkerLeft->getL() ) {cout<<"Model L does not consistent with walker left L!"<<endl; exit(1);}
    }
    if( generalHamiltonian_icf->getNup() != walkerLeft->getNup() ) {cout<<"Model Nup does not consistent with walker left Nup!"<<endl; exit(1);}
    if( generalHamiltonian_icf->getNdn() != walkerLeft->getNdn() ) {cout<<"Model Ndn does not consistent with walker left Ndn!"<<endl; exit(1);}

    if(generalHamiltonian_icf->getHamiltonian_spin_flag() == true){
        if( generalHamiltonian_icf->getL() != 2*walkerRight->getL() ) {cout<<"Model L does not consistent with walker right L!"<<endl; exit(1);}
    }else{
        if( generalHamiltonian_icf->getL() != walkerRight->getL() ) {cout<<"Model L does not consistent with walker right L!"<<endl; exit(1);}
    }
    if( generalHamiltonian_icf->getNup() != walkerRight->getNup() ) {cout<<"Model Nup does not consistent with walker right Nup!"<<endl; exit(1);}
    if( generalHamiltonian_icf->getNdn() != walkerRight->getNdn() ) {cout<<"Model Ndn does not consistent with walker right Ndn!"<<endl; exit(1);}
}

void GeneralHamiltonian_icfMeasureObserveSD2sSD2s::initWfDaggerK(SD2sSD2sOperation &sd2ssd2sOperation)
{
    const SD2s  *walkerLeft = sd2ssd2sOperation.getWalkerLeft();

    size_t halfL = walkerLeft->getL(); size_t Nup = walkerLeft->getNup(); size_t Ndn = walkerLeft->getNdn();

    wfUpDaggerK.resize(Nup, halfL);
    wfDnDaggerK.resize(Ndn, halfL);

    bool Hamiltonian_spin_flag = generalHamiltonian_icf->getHamiltonian_spin_flag();
    const TensorHao<complex<double>,2> & K = generalHamiltonian_icf->getK();

    TensorHao<complex<double>,2> Kup(halfL, halfL), Kdn(halfL, halfL);
    if(Hamiltonian_spin_flag){
        for(int i=1-1; i<=halfL-1; i++){
        for(int j=1-1; j<=halfL-1; j++){
            Kup(i,j) = K(i,j);
            Kdn(i,j) = K(i + halfL,j + halfL);
        }
        }
    }else{
        for(int i=1-1; i<=halfL-1; i++){
        for(int j=1-1; j<=halfL-1; j++){
            Kup(i,j) = K(i,j);
            Kdn(i,j) = K(i,j);
        }
        }
    }

    BL_NAME(gmm)(walkerLeft->getWfUp(), Kup, wfUpDaggerK, 'C');
    BL_NAME(gmm)(walkerLeft->getWfDn(), Kdn, wfDnDaggerK, 'C');
}

// void GeneralHamiltonian_icfMeasureObserveSD2sSD2s::initWfDaggerSVDVecs(SD2sSD2sOperation &sd2ssd2sOperation)
// {
//     const SD2s  *walkerLeft = sd2ssd2sOperation.getWalkerLeft();

//     size_t halfL = walkerLeft->getL(); size_t Nup = walkerLeft->getNup(); size_t Ndn = walkerLeft->getNdn();
//     size_t svdNumber = generalHamiltonian_icf->getSVDNumber();

//     wfUpDaggerSVDVecs.resize(Nup, halfL, svdNumber); 
//     wfDnDaggerSVDVecs.resize(Ndn, halfL, svdNumber); 

//     bool Hamiltonian_spin_flag = generalHamiltonian_icf->getHamiltonian_spin_flag();
//     const TensorHaoMPIRef<complex<double>,3> & svdVecs = generalHamiltonian_icf->getSVDVecs();
//     TensorHao<complex<double>, 2> svdVecComplexUp(halfL,halfL), svdVecComplexDn(halfL,halfL);
//     TensorHaoRef<complex<double>, 2> wfUpDaggerSVDVec, wfDnDaggerSVDVec;
//     for(size_t k = 0; k < svdNumber ; ++k)
//     {
//         if(Hamiltonian_spin_flag){
//             for(size_t i = 0; i < halfL; ++i)
//             {
//                 for(size_t j = 0; j < halfL; ++j) svdVecComplexUp(j,i) = svdVecs(j,i,k);
//             }
//             for(size_t i = 0; i < halfL; ++i)
//             {
//                 for(size_t j = 0; j < halfL; ++j) svdVecComplexDn(j,i) = svdVecs(j+halfL,i+halfL,k);
//             }
//         }else{
//             for(size_t i = 0; i < halfL; ++i)
//             {
//                 for(size_t j = 0; j < halfL; ++j) svdVecComplexUp(j,i) = svdVecs(j,i,k);
//             }
//             for(size_t i = 0; i < halfL; ++i)
//             {
//                 for(size_t j = 0; j < halfL; ++j) svdVecComplexDn(j,i) = svdVecs(j,i,k);
//             }
//         }
//         wfUpDaggerSVDVec=wfUpDaggerSVDVecs[k];
//         BL_NAME(gmm)(walkerLeft->getWfUp(), svdVecComplexUp, wfUpDaggerSVDVec, 'C');
//         wfDnDaggerSVDVec=wfDnDaggerSVDVecs[k];
//         BL_NAME(gmm)(walkerLeft->getWfDn(), svdVecComplexDn, wfDnDaggerSVDVec, 'C');
//     }
// }

// void GeneralHamiltonian_icfMeasureObserveSD2sSD2s::initWfDaggerSVDVecSquares(SD2sSD2sOperation &sd2ssd2sOperation)
// {
//     initWfDaggerSVDVecs(sd2ssd2sOperation);
//     //
//     const SD2s  *walkerLeft = sd2ssd2sOperation.getWalkerLeft();

//     size_t halfL = walkerLeft->getL(); size_t Nup = walkerLeft->getNup(); size_t Ndn = walkerLeft->getNdn();
//     size_t svdNumber = generalHamiltonian_icf->getSVDNumber();

//     wfUpDaggerSVDVecSquares.resize(Nup, halfL, svdNumber); 
//     wfDnDaggerSVDVecSquares.resize(Ndn, halfL, svdNumber); 

//     bool Hamiltonian_spin_flag = generalHamiltonian_icf->getHamiltonian_spin_flag();
//     const TensorHaoMPIRef<complex<double>,3> & svdVecs = generalHamiltonian_icf->getSVDVecs();
//     TensorHao<complex<double>, 2> svdVecComplexUp(halfL,halfL), svdVecComplexDn(halfL,halfL);
//     TensorHaoRef<complex<double>, 2> wfUpDaggerSVDVec, wfDnDaggerSVDVec;
//     TensorHaoRef<complex<double>, 2> wfUpDaggerSVDVecSquare, wfDnDaggerSVDVecSquare;
//     for(size_t k = 0; k < svdNumber ; ++k)
//     {
//         if(Hamiltonian_spin_flag){
//             for(size_t i = 0; i < halfL; ++i)
//             {
//                 for(size_t j = 0; j < halfL; ++j) svdVecComplexUp(j,i) = svdVecs(j,i,k);
//             }
//             for(size_t i = 0; i < halfL; ++i)
//             {
//                 for(size_t j = 0; j < halfL; ++j) svdVecComplexDn(j,i) = svdVecs(j+halfL,i+halfL,k);
//             }
//         }else{
//             for(size_t i = 0; i < halfL; ++i)
//             {
//                 for(size_t j = 0; j < halfL; ++j) svdVecComplexUp(j,i) = svdVecs(j,i,k);
//             }
//             for(size_t i = 0; i < halfL; ++i)
//             {
//                 for(size_t j = 0; j < halfL; ++j) svdVecComplexDn(j,i) = svdVecs(j,i,k);
//             }
//         }
//         wfUpDaggerSVDVec=wfUpDaggerSVDVecs[k];
//         wfDnDaggerSVDVec=wfDnDaggerSVDVecs[k];
//         wfUpDaggerSVDVecSquare=wfUpDaggerSVDVecSquares[k];
//         wfDnDaggerSVDVecSquare=wfDnDaggerSVDVecSquares[k];
//         BL_NAME(gmm)(wfUpDaggerSVDVec, svdVecComplexUp, wfUpDaggerSVDVecSquare);
//         BL_NAME(gmm)(wfDnDaggerSVDVec, svdVecComplexDn, wfDnDaggerSVDVecSquare);
//     }
// }


void GeneralHamiltonian_icfMeasureObserveSD2sSD2s::addEnergy(SD2sSD2sOperation &sd2ssd2sOperation, complex<double> denIncrement)
{
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();

    if( svdBgNum.rank(0) != svdNumber ) { svdBgNum.resize(svdNumber); svdBgNum = complex<double>(0,0); }
    if( svdNormalNum.rank(0) != svdNumber ) { svdNormalNum.resize(svdNumber); svdNormalNum = complex<double>(0,0); }
    if( svdExNum.rank(0) != svdNumber ) { svdExNum.resize(svdNumber); svdExNum = complex<double>(0,0); }

    complex<double> Kenergy=calculateKenergy(sd2ssd2sOperation);
    TensorHao<complex<double>, 1> svdBg = calculateSVDBg(sd2ssd2sOperation);
    TensorHao<complex<double>, 1> svdNormal = calculateSVDNormal(sd2ssd2sOperation);
    TensorHao<complex<double>, 1> svdEx = calculateSVDEx(sd2ssd2sOperation);

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

void GeneralHamiltonian_icfMeasureObserveSD2sSD2s::addGreen(SD2sSD2sOperation &sd2ssd2sOperation, complex<double> denIncrement)
{
    // const SD2s  *walkerLeft = sd2ssd2sOperation.getWalkerLeft();
    // size_t L = generalHamiltonian_icf->getL();size_t N = walkerLeft->getN(); 
    // if( greenNum.rank(0) != L ) { greenNum.resize(L,L); greenNum = complex<double>(0,0); }

    // greenNum += ( sd2ssd2sOperation.returnGreenMatrix() * denIncrement );
}

complex<double> GeneralHamiltonian_icfMeasureObserveSD2sSD2s::calculateKenergy(SD2sSD2sOperation &sd2ssd2sOperation)
{
    const SD2s  *walkerLeft = sd2ssd2sOperation.getWalkerLeft();

    size_t halfL = walkerLeft->getL(); size_t Nup = walkerLeft->getNup(); size_t Ndn = walkerLeft->getNdn();
    const TensorHao<complex<double>, 2> &thetaUp_T =  sd2ssd2sOperation.returnThetaUp_T();
    const TensorHao<complex<double>, 2> &thetaDn_T =  sd2ssd2sOperation.returnThetaDn_T();

    complex<double> Kenergy(0,0);
    for(size_t i = 0; i < halfL; ++i)
    {
        for(size_t j = 0; j < Nup; ++j) Kenergy += wfUpDaggerK(j,i) * thetaUp_T(j,i);
        for(size_t j = 0; j < Ndn; ++j) Kenergy += wfDnDaggerK(j,i) * thetaDn_T(j,i);
    }
    return Kenergy;
}

TensorHao<complex<double>, 1> GeneralHamiltonian_icfMeasureObserveSD2sSD2s::calculateSVDBg(SD2sSD2sOperation &sd2ssd2sOperation)
{
    const SD2s  *walkerLeft = sd2ssd2sOperation.getWalkerLeft();

    size_t halfL = walkerLeft->getL(); size_t Nup = walkerLeft->getNup(); size_t Ndn = walkerLeft->getNdn();
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();
    const TensorHao<complex<double>, 2> &thetaUp_T =  sd2ssd2sOperation.returnThetaUp_T();
    const TensorHao<complex<double>, 2> &thetaDn_T =  sd2ssd2sOperation.returnThetaDn_T();

    // TensorHaoRef<complex<double>, 2> leftUp(halfL*Nup, svdNumber);
    // TensorHaoRef<complex<double>, 1> rightUp(halfL*Nup);
    // leftUp.point( wfUpDaggerSVDVecs.data() );
    // rightUp.point( const_cast<complex<double>*>( thetaUp_T.data() ) );

    // TensorHao<complex<double>, 1> svdBgUp(svdNumber);
    // BL_NAME(gemv)(leftUp, rightUp, svdBgUp, 'T' );

    // TensorHaoRef<complex<double>, 2> leftDn(halfL*Ndn, svdNumber);
    // TensorHaoRef<complex<double>, 1> rightDn(halfL*Ndn);
    // leftDn.point( wfDnDaggerSVDVecs.data() );
    // rightDn.point( const_cast<complex<double>*>( thetaDn_T.data() ) );

    // TensorHao<complex<double>, 1> svdBgDn(svdNumber);
    // BL_NAME(gemv)(leftDn, rightDn, svdBgDn, 'T' );
    
    // TensorHao<complex<double>, 1> svdBg(svdNumber);
    // svdBg = svdBgUp + svdBgDn;
    /////////////////
    TensorHao<complex<double>, 1> svdBg(svdNumber);
    // 
    bool Hamiltonian_spin_flag = generalHamiltonian_icf->getHamiltonian_spin_flag();
    const TensorHaoMPIRef<complex<double>,3> & svdVecs = generalHamiltonian_icf->getSVDVecs();
    TensorHao<complex<double>, 2> svdVecComplexUp(halfL,halfL), svdVecComplexDn(halfL,halfL);
    TensorHao<complex<double>, 2> wfUpDaggerSVDVec(Nup,halfL), wfDnDaggerSVDVec(Ndn,halfL);
    for(size_t k = 0; k < svdNumber ; ++k)
    {
        if(Hamiltonian_spin_flag){
            for(size_t i = 0; i < halfL; ++i)
            {
                for(size_t j = 0; j < halfL; ++j) svdVecComplexUp(j,i) = svdVecs(j,i,k);
            }
            for(size_t i = 0; i < halfL; ++i)
            {
                for(size_t j = 0; j < halfL; ++j) svdVecComplexDn(j,i) = svdVecs(j+halfL,i+halfL,k);
            }
        }else{
            for(size_t i = 0; i < halfL; ++i)
            {
                for(size_t j = 0; j < halfL; ++j) svdVecComplexUp(j,i) = svdVecs(j,i,k);
            }
            for(size_t i = 0; i < halfL; ++i)
            {
                for(size_t j = 0; j < halfL; ++j) svdVecComplexDn(j,i) = svdVecs(j,i,k);
            }
        }
        BL_NAME(gmm)(walkerLeft->getWfUp(), svdVecComplexUp, wfUpDaggerSVDVec, 'C');
        BL_NAME(gmm)(walkerLeft->getWfDn(), svdVecComplexDn, wfDnDaggerSVDVec, 'C');
        
        complex<double> svdBgUp_k(0,0), svdBgDn_k(0,0);
        for(size_t i = 0; i < halfL; ++i)
        {
            for(size_t j = 0; j < Nup; ++j){
                svdBgUp_k += wfUpDaggerSVDVec(j,i) * thetaUp_T(j,i);
            }
            for(size_t j = 0; j < Ndn; ++j){
                svdBgDn_k += wfDnDaggerSVDVec(j,i) * thetaDn_T(j,i);
            }
        }
        svdBg(k) = svdBgUp_k + svdBgDn_k;
    }

    return svdBg;
}


TensorHao<complex<double>, 1> GeneralHamiltonian_icfMeasureObserveSD2sSD2s::calculateSVDNormal(SD2sSD2sOperation &sd2ssd2sOperation)
{
    const SD2s  *walkerLeft = sd2ssd2sOperation.getWalkerLeft();

    size_t halfL = walkerLeft->getL(); size_t Nup = walkerLeft->getNup(); size_t Ndn = walkerLeft->getNdn();
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();
    const TensorHao<complex<double>, 2> &thetaUp_T =  sd2ssd2sOperation.returnThetaUp_T();
    const TensorHao<complex<double>, 2> &thetaDn_T =  sd2ssd2sOperation.returnThetaDn_T();

    bool Hamiltonian_spin_flag = generalHamiltonian_icf->getHamiltonian_spin_flag();
    const TensorHaoMPIRef<complex<double>,3> & svdVecs = generalHamiltonian_icf->getSVDVecs();

    TensorHao<complex<double>, 1> svdNormal(svdNumber);

    TensorHao<complex<double>, 2> svdVecComplexUp(halfL, halfL), svdVecComplexDn(halfL, halfL);
    // TensorHaoRef<complex<double>, 2> wfUpDaggerSVDVec, wfDnDaggerSVDVec;
    TensorHao<complex<double>, 2> wfUpDaggerSVDVec(Nup, halfL), wfDnDaggerSVDVec(Ndn, halfL);
    TensorHao<complex<double>, 2> wfUpDaggerSVDVecSquare(Nup, halfL), wfDnDaggerSVDVecSquare(Ndn, halfL);

    for(size_t k = 0; k < svdNumber; ++k)
    {
        if(Hamiltonian_spin_flag){
            for(size_t i = 0; i < halfL; ++i)
                for(size_t j = 0; j < halfL; ++j) {
                    svdVecComplexUp(j,i) = svdVecs(j,i,k);
                    svdVecComplexDn(j,i) = svdVecs(j+halfL,i+halfL,k);
                }
        }else{
            for(size_t i = 0; i < halfL; ++i)
                for(size_t j = 0; j < halfL; ++j) {
                    svdVecComplexUp(j,i) = svdVecs(j,i,k);
                    svdVecComplexDn(j,i) = svdVecs(j,i,k);
                }
        }
        ////////////////
        if(Hamiltonian_spin_flag){
            for(size_t i = 0; i < halfL; ++i)
            {
                for(size_t j = 0; j < halfL; ++j) svdVecComplexUp(j,i) = svdVecs(j,i,k);
            }
            for(size_t i = 0; i < halfL; ++i)
            {
                for(size_t j = 0; j < halfL; ++j) svdVecComplexDn(j,i) = svdVecs(j+halfL,i+halfL,k);
            }
        }else{
            for(size_t i = 0; i < halfL; ++i)
            {
                for(size_t j = 0; j < halfL; ++j) svdVecComplexUp(j,i) = svdVecs(j,i,k);
            }
            for(size_t i = 0; i < halfL; ++i)
            {
                for(size_t j = 0; j < halfL; ++j) svdVecComplexDn(j,i) = svdVecs(j,i,k);
            }
        }
        BL_NAME(gmm)(walkerLeft->getWfUp(), svdVecComplexUp, wfUpDaggerSVDVec, 'C');
        BL_NAME(gmm)(walkerLeft->getWfDn(), svdVecComplexDn, wfDnDaggerSVDVec, 'C');
        // wfUpDaggerSVDVec=wfUpDaggerSVDVecs[k];
        // wfDnDaggerSVDVec=wfDnDaggerSVDVecs[k];
        ////////////////

        BL_NAME(gmm)(wfUpDaggerSVDVec, svdVecComplexUp, wfUpDaggerSVDVecSquare);
        BL_NAME(gmm)(wfDnDaggerSVDVec, svdVecComplexDn, wfDnDaggerSVDVecSquare);

        for(size_t i = 0; i < halfL; ++i)
        {
            for(size_t j = 0; j < Nup; ++j) svdNormal(k) += wfUpDaggerSVDVecSquare(j,i) * thetaUp_T(j,i);
            for(size_t j = 0; j < Ndn; ++j) svdNormal(k) += wfDnDaggerSVDVecSquare(j,i) * thetaDn_T(j,i);
        }
    }

    return svdNormal;
}

TensorHao<complex<double>, 1> GeneralHamiltonian_icfMeasureObserveSD2sSD2s::calculateSVDEx(SD2sSD2sOperation &sd2ssd2sOperation)
{
    const SD2s  *walkerLeft = sd2ssd2sOperation.getWalkerLeft();
    bool Hamiltonian_spin_flag = generalHamiltonian_icf->getHamiltonian_spin_flag();

    size_t halfL = walkerLeft->getL(); size_t Nup = walkerLeft->getNup(); size_t Ndn = walkerLeft->getNdn();
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();
    const TensorHao<complex<double>, 2> &thetaUp_T =  sd2ssd2sOperation.returnThetaUp_T();
    const TensorHao<complex<double>, 2> &thetaDn_T =  sd2ssd2sOperation.returnThetaDn_T();

    const TensorHaoMPIRef<complex<double>,3> & svdVecs = generalHamiltonian_icf->getSVDVecs();
    TensorHao<complex<double>, 2> svdVecComplexUp(halfL, halfL), svdVecComplexDn(halfL, halfL);
    TensorHao<complex<double>, 2> densityUpL(Nup, Nup), densityDnL(Ndn, Ndn);
    // TensorHaoRef<complex<double>, 2> wfUpDaggerSVDVec, wfDnDaggerSVDVec;
    TensorHao<complex<double>, 2> wfUpDaggerSVDVec(Nup, halfL), wfDnDaggerSVDVec(Ndn, halfL);
    TensorHao<complex<double>, 1> svdEx(svdNumber);

    TensorHao<complex<double>, 2> thetaUp, thetaDn;
    thetaUp = trans(thetaUp_T); 
    thetaDn = trans(thetaDn_T); 
    for(size_t k = 0; k < svdNumber; ++k)
    {

        ////////////////
        if(Hamiltonian_spin_flag){
            for(size_t i = 0; i < halfL; ++i)
            {
                for(size_t j = 0; j < halfL; ++j) svdVecComplexUp(j,i) = svdVecs(j,i,k);
            }
            for(size_t i = 0; i < halfL; ++i)
            {
                for(size_t j = 0; j < halfL; ++j) svdVecComplexDn(j,i) = svdVecs(j+halfL,i+halfL,k);
            }
        }else{
            for(size_t i = 0; i < halfL; ++i)
            {
                for(size_t j = 0; j < halfL; ++j) svdVecComplexUp(j,i) = svdVecs(j,i,k);
            }
            for(size_t i = 0; i < halfL; ++i)
            {
                for(size_t j = 0; j < halfL; ++j) svdVecComplexDn(j,i) = svdVecs(j,i,k);
            }
        }
        BL_NAME(gmm)(walkerLeft->getWfUp(), svdVecComplexUp, wfUpDaggerSVDVec, 'C');
        BL_NAME(gmm)(walkerLeft->getWfDn(), svdVecComplexDn, wfDnDaggerSVDVec, 'C');
        // wfUpDaggerSVDVec=wfUpDaggerSVDVecs[k];
        // wfDnDaggerSVDVec=wfDnDaggerSVDVecs[k];
        ////////////////

        BL_NAME(gmm)( wfUpDaggerSVDVec, thetaUp, densityUpL);
        BL_NAME(gmm)( wfDnDaggerSVDVec, thetaDn, densityDnL);

        svdEx(k)=0.0;
        for(size_t i = 0; i < Nup; ++i) { for(size_t j = 0; j < Nup; ++j) svdEx(k) += densityUpL(j,i) * densityUpL(i,j); }
        for(size_t i = 0; i < Ndn; ++i) { for(size_t j = 0; j < Ndn; ++j) svdEx(k) += densityDnL(j,i) * densityDnL(i,j); }
    }

    return svdEx;
}