//
// Created by boruoshihao on 1/13/17.
//
#define _USE_MATH_DEFINES
#include <chrono>
#include <cmath>
#include "../../include/HubbardRealSpaceSOC_icf/HubbardSOC_icfMeasureObserveMetroChainsSD.h"
#include "afqmclab.h"

using namespace std;
using namespace tensor_hao;

HubbardSOC_icfMeasureObserveMetroChainsSD::HubbardSOC_icfMeasureObserveMetroChainsSD()
{
    initModelNullptr();
    reSet();
}

HubbardSOC_icfMeasureObserveMetroChainsSD::HubbardSOC_icfMeasureObserveMetroChainsSD(const HubbardSOC_icf &hubbardSOC_)
{
    setModel( hubbardSOC_ );
    reSet();
}

HubbardSOC_icfMeasureObserveMetroChainsSD::~HubbardSOC_icfMeasureObserveMetroChainsSD()
{

}

const HubbardSOC_icf *HubbardSOC_icfMeasureObserveMetroChainsSD::getHubbardSOC_icf() const
{
    return hubbardSOC_icf;
}

void HubbardSOC_icfMeasureObserveMetroChainsSD::initModelNullptr()
{
    hubbardSOC_icf = nullptr;
}

void HubbardSOC_icfMeasureObserveMetroChainsSD::setModel(const HubbardSOC_icf &hubbardSOC_)
{
    hubbardSOC_icf = &hubbardSOC_;
}

void HubbardSOC_icfMeasureObserveMetroChainsSD::reSet()
{
    complex<double> zero(0,0);
    den = zero;
    HNum = zero;
    KNum = zero;
    VNum = zero;
    RNum = zero;
    greenMatrixNum = zero;
    densityNum = zero;
    szszNum = zero;
    sxsxNum = zero;
    sysyNum = zero;
    // sqsmqNum = zero;
    densityDensityNum = zero;
    // splusSminusNum = zero;
    // sminusSplusNum = zero;
    // spairSpairNum = zero;
}

complex<double> HubbardSOC_icfMeasureObserveMetroChainsSD::returnEnergy()
{
    complex<double> Htot   = MPISum(HNum);
    complex<double> denTot = MPISum(den);
    complex<double> energy;
    if( MPIRank() == 0 ) energy = Htot/denTot;
    MPIBcast(energy);
    return energy;
}

void HubbardSOC_icfMeasureObserveMetroChainsSD::addMeasurement(MetroChainsOperation &metroChainsOperation, complex<double> denIncrement)
{
    den += denIncrement;

    addEnergy(metroChainsOperation, denIncrement);
    // addGreenMatrix(metroChainsOperation, denIncrement);
    addSSNum(metroChainsOperation, denIncrement);
    // addDensityDensity(metroChainsOperation, denIncrement);
    // addSqSmqNum(metroChainsOperation, denIncrement);
    // addSplusSminus(metroChainsOperation, denIncrement);
    // addSminusSplus(metroChainsOperation, denIncrement);
    // addSpairSpair(metroChainsOperation, denIncrement);
}

void HubbardSOC_icfMeasureObserveMetroChainsSD::addMeasurement_timer(MetroChainsOperation &metroChainsOperation, complex<double> denIncrement)
{
    den += denIncrement;

    /////////////////
    //Timer
    /////////////////
    auto begin = std::chrono::high_resolution_clock::now();
    addEnergy(metroChainsOperation, denIncrement);
    auto end_addEnergy = std::chrono::high_resolution_clock::now();
    addSSNum(metroChainsOperation, denIncrement);
    auto end_addSSNum = std::chrono::high_resolution_clock::now();
    /////////////////
    //Timer
    /////////////////
    auto elapsed_addEnergy = std::chrono::duration_cast<std::chrono::nanoseconds>(end_addEnergy - begin);
    auto elapsed_addSSNum = std::chrono::duration_cast<std::chrono::nanoseconds>(end_addSSNum - end_addEnergy);
    if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
    if(MPIRank() == 0)printf("Time measured for addEnergy: %.8f seconds.\n", elapsed_addEnergy.count() * 1e-9);
    if(MPIRank() == 0)printf("Time measured for addSSNum: %.8f seconds.\n", elapsed_addSSNum.count() * 1e-9);
    if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
}

void HubbardSOC_icfMeasureObserveMetroChainsSD::addMeasurement_energy(MetroChainsOperation &metroChainsOperation, complex<double> denIncrement)
{
    den += denIncrement;

    addEnergy(metroChainsOperation, denIncrement);
    // addGreenMatrix(metroChainsOperation, denIncrement);
    // addSSNum(metroChainsOperation, denIncrement);
    // addDensityDensity(metroChainsOperation, denIncrement);
    // addSqSmqNum(metroChainsOperation, denIncrement);
    // addSplusSminus(metroChainsOperation, denIncrement);
    // addSminusSplus(metroChainsOperation, denIncrement);
    // addSpairSpair(metroChainsOperation, denIncrement);
}

NiupNidnForce HubbardSOC_icfMeasureObserveMetroChainsSD::getForce(const NiupNidn &niupNidn, MetroChainsOperation &metroChainsOperation, double cap)
{
    size_t halfL = niupNidn.getL(); const string &decompType = niupNidn.getDecompType();
    const MetroChains  *walkerLeft = metroChainsOperation.getWalkerLeft();
    size_t numOfBrackets = walkerLeft->getNumOfBrackets();
    size_t numOfChains = walkerLeft->getNumOfChains();

    TensorHao<complex<double>, 2> greenMatrix; greenMatrix.resize(walkerLeft->getL(), walkerLeft->getL());greenMatrix=0.0;
    for(int bracket=1-1; bracket <= numOfBrackets*numOfChains-1; bracket++){
        const TensorHao<complex<double>, 2> greenMatrix_bracket = metroChainsOperation.returnGreenMatrix(bracket);
        greenMatrix += greenMatrix_bracket * exp(metroChainsOperation.returnLogPhase(bracket)-metroChainsOperation.returnLogTotalPhase());
    }

    TensorHao< complex<double>, 1 > backGround(halfL);
    if( decompType == "densityCharge" )
    {
        for(size_t i = 0; i < halfL; ++i) backGround(i) = greenMatrix(i,i) + greenMatrix(i+halfL, i+halfL) -1.0;
    }
    else if( decompType == "densitySpin" )
    {
        for(size_t i = 0; i < halfL; ++i) backGround(i) = greenMatrix(i,i) - greenMatrix(i+halfL, i+halfL);
    }
    else
    {
        cout<<"Error! Can not find the matched decompType! "<<decompType<<endl;
        exit(1);
    }

    NiupNidnForce force(halfL);
    const TensorHao<complex<double>, 1> &gamma = niupNidn.getGamma();
    for (size_t i = 0; i < halfL; ++i)
    {
        force(i) = ( gamma(i) * backGround(i) ).real();
        if( force(i) >  cap ) force(i) =  cap;
        if( force(i) < -cap ) force(i) = -cap;
    }

    return force;
}

void HubbardSOC_icfMeasureObserveMetroChainsSD::write() const
{
    writeThreadSum(den, "den.dat", ios::app);
    writeThreadSum(HNum, "HNum.dat", ios::app);
    // writeKNumVumRum();
    // writeThreadSum(greenMatrixNum.size(), greenMatrixNum.data(), "greenMatrixNum.dat", ios::app);
    writeThreadSum(densityNum.size(), densityNum.data(), "densityNum.dat", ios::app);
    // writeThreadSum(sqsmqNum.size(), sqsmqNum.data(), "sqsmqNum.dat", ios::app);
    writeThreadSum(szszNum.size(), szszNum.data(), "szszNum.dat", ios::app);
    writeThreadSum(sxsxNum.size(), sxsxNum.data(), "sxsxNum.dat", ios::app);
    writeThreadSum(sysyNum.size(), sysyNum.data(), "sysyNum.dat", ios::app);
    // writeThreadSum(densityDensityNum.size(), densityDensityNum.data(), "densityDensityNum.dat", ios::app);
    // writeThreadSum(splusSminusNum.size(), splusSminusNum.data(), "splusSminusNum.dat", ios::app);
    // writeThreadSum(sminusSplusNum.size(), sminusSplusNum.data(), "sminusSplusNum.dat", ios::app);
    // writeThreadSum(spairSpairNum.size(), spairSpairNum.data(), "spairSpairNum.dat", ios::app);
}

void HubbardSOC_icfMeasureObserveMetroChainsSD::write(std::string postfix ) const
{
    std::string denName="den"+postfix+".dat";
    std::string HNumName="HNum"+postfix+".dat";
    writeThreadSum(den, denName, ios::app);
    writeThreadSum(HNum, HNumName, ios::app);
    // writeKNumVumRum();
    std::string greenMatrixNumName="greenMatrixNum"+postfix+".dat";
    std::string densityNumName="densityNum"+postfix+".dat";
    writeThreadSum(greenMatrixNum.size(), greenMatrixNum.data(), greenMatrixNumName, ios::app);
    writeThreadSum(densityNum.size(), densityNum.data(), densityNumName, ios::app);

    std::string szszNumName="szszNum"+postfix+".dat";
    std::string sxsxNumName="sxsxNum"+postfix+".dat";
    std::string sysyNumName="sysyNum"+postfix+".dat";
    writeThreadSum(szszNum.size(), szszNum.data(), szszNumName, ios::app);
    writeThreadSum(sxsxNum.size(), sxsxNum.data(), sxsxNumName, ios::app);
    writeThreadSum(sysyNum.size(), sysyNum.data(), sysyNumName, ios::app);
    
    // std::string densityDensityNumName="densityDensityNum"+postfix+".dat";
    // writeThreadSum(densityDensityNum.size(), densityDensityNum.data(), densityDensityNumName, ios::app);
}

double HubbardSOC_icfMeasureObserveMetroChainsSD::getMemory() const
{
    double mem(0.0);
    mem += 8.0+16.0*5;
    mem += greenMatrixNum.getMemory();
    mem += densityNum.getMemory();
    // mem += sqsmqNum.getMemory();
    mem += szszNum.getMemory();
    mem += sxsxNum.getMemory();
    mem += sysyNum.getMemory();
    mem += densityDensityNum.getMemory();
    // mem += splusSminusNum.getMemory();
    // mem += sminusSplusNum.getMemory();
    // mem += spairSpairNum.getMemory();

    return mem;
}

void HubbardSOC_icfMeasureObserveMetroChainsSD::addGreenMatrix(MetroChainsOperation &metroChainsOperation, complex<double> denIncrement)
{
    size_t L2 = getHubbardSOC_icf()->getL() * 2;
    const MetroChains  *walkerLeft = metroChainsOperation.getWalkerLeft();
    size_t numOfBrackets = walkerLeft->getNumOfBrackets();
    size_t numOfChains = walkerLeft->getNumOfChains();

    TensorHao<complex<double>, 2> greenMatrix; greenMatrix.resize(L2, L2);greenMatrix=0.0;
    for(int bracket=1-1; bracket <= numOfBrackets*numOfChains-1; bracket++){
        const TensorHao<complex<double>, 2> greenMatrix_bracket = metroChainsOperation.returnGreenMatrix(bracket);
        greenMatrix += greenMatrix_bracket * exp(metroChainsOperation.returnLogPhase(bracket)-metroChainsOperation.returnLogTotalPhase());
    }

    if( greenMatrixNum.rank(0) != L2 ) { greenMatrixNum.resize(L2, L2); greenMatrixNum = complex<double>(0,0); }
    if( densityNum.rank(0) != L2 ) { densityNum.resize(L2); densityNum = complex<double>(0,0); }

    greenMatrixNum += ( greenMatrix * denIncrement );

    for(int i=1-1; i<=L2-1; i++){
        densityNum(i) += ( greenMatrix(i,i) * denIncrement );
    }
}

void HubbardSOC_icfMeasureObserveMetroChainsSD::addSSNum(MetroChainsOperation &metroChainsOperation, complex<double> denIncrement)
{
    size_t L = getHubbardSOC_icf()->getL();
    const MetroChains  *walkerLeft = metroChainsOperation.getWalkerLeft();
    size_t numOfBrackets = walkerLeft->getNumOfBrackets();
    size_t numOfChains = walkerLeft->getNumOfChains();
    TensorHao<complex<double>, 1> szsz(L); szsz=0.0;
    TensorHao<complex<double>, 1> sxsx(L); sxsx=0.0;
    TensorHao<complex<double>, 1> sysy(L); sysy=0.0;

    if( szszNum.rank(0) != L ) { szszNum.resize(L); szszNum = complex<double>(0,0); }
    if( sxsxNum.rank(0) != L ) { sxsxNum.resize(L); sxsxNum = complex<double>(0,0); }
    if( sysyNum.rank(0) != L ) { sysyNum.resize(L); sysyNum = complex<double>(0,0); }
    int L_x=0, L_y=0;
    if(L==8*8){
        L_x=8;
        L_y=8;
    }else if(L==16*16){
        L_x=16;
        L_y=16;
    }
    //
    if(L_x !=0 && L_y!=0){
        complex<double> logTotalPhase = metroChainsOperation.returnLogTotalPhase();
        ///////////////////////////////////////////////
        for(int bracket=1-1; bracket <= numOfBrackets*numOfChains-1; bracket++){  
            complex<double> local_iijj, local_iLiLjLjL;     
            complex<double> local_ii_jj, local_ij_ij, local_iLiL_jLjL, local_iLjL_iLjL, local_iLiL_jj, local_ii_jLjL;
            // ATTENTION: the following copy process of greenMatrix_bracket takes significant resources, should minimize this process in loops
            const TensorHao<complex<double>, 2> greenMatrix_bracket = metroChainsOperation.returnGreenMatrix(bracket);
            for(int i=1-1; i<=L-1; i++){
                int i_x=i%L_x;
                int i_y=i/L_y;
                for(int j=1-1; j<=L-1; j++){
                    int j_x=j%L_x;
                    int j_y=j/L_y;
                    ///////////////////////////////
                    complex<double> localCdaggerCCdaggerC_iijj = localCdaggerCCdaggerC_bracket(greenMatrix_bracket,i,i,j,j);
                    complex<double> localCdaggerCCdaggerC_iLiLjLjL = localCdaggerCCdaggerC_bracket(greenMatrix_bracket,i+L,i+L,j+L,j+L);
                    complex<double> localCdaggerCCdaggerC_iLiLjj = localCdaggerCCdaggerC_bracket(greenMatrix_bracket,i+L,i+L,j,j);
                    complex<double> localCdaggerCCdaggerC_iijLjL = localCdaggerCCdaggerC_bracket(greenMatrix_bracket,i,i,j+L,j+L);
                    // 
                    complex<double> localCdaggerCCdaggerC_iiLjjL = localCdaggerCCdaggerC_bracket(greenMatrix_bracket,i,i+L,j,j+L);
                    complex<double> localCdaggerCCdaggerC_iLijLj = localCdaggerCCdaggerC_bracket(greenMatrix_bracket,i+L,i,j+L,j);
                    complex<double> localCdaggerCCdaggerC_iiLjLj = localCdaggerCCdaggerC_bracket(greenMatrix_bracket,i,i+L,j+L,j);
                    complex<double> localCdaggerCCdaggerC_iLijjL = localCdaggerCCdaggerC_bracket(greenMatrix_bracket,i+L,i,j,j+L);
                    // 
                    ///////////////////////////////
                    int distance = ( (j_x - i_x + L_x)%L_x )*L_y + (j_y - i_y + L_y)%L_y;
                    szsz(distance) += 0.25/double(L) *(localCdaggerCCdaggerC_iijj + localCdaggerCCdaggerC_iLiLjLjL - localCdaggerCCdaggerC_iLiLjj - localCdaggerCCdaggerC_iijLjL) * exp(metroChainsOperation.returnLogPhase(bracket) - logTotalPhase);
                    sxsx(distance) += 0.25/double(L) *(localCdaggerCCdaggerC_iiLjjL + localCdaggerCCdaggerC_iLijLj + localCdaggerCCdaggerC_iiLjLj + localCdaggerCCdaggerC_iLijjL) * exp(metroChainsOperation.returnLogPhase(bracket) - logTotalPhase);
                    sysy(distance) += 0.25/double(L) * (-1.0) * (localCdaggerCCdaggerC_iiLjjL + localCdaggerCCdaggerC_iLijLj - localCdaggerCCdaggerC_iiLjLj - localCdaggerCCdaggerC_iLijjL) * exp(metroChainsOperation.returnLogPhase(bracket) - logTotalPhase);
                    ///////////////////////////////
                }
            }
        }
    }
    // 
    szszNum += ( szsz * denIncrement );
    sxsxNum += ( sxsx * denIncrement );
    sysyNum += ( sysy * denIncrement );
}

void HubbardSOC_icfMeasureObserveMetroChainsSD::addSqSmqNum(MetroChainsOperation &metroChainsOperation, complex<double> denIncrement)
{
    size_t L = getHubbardSOC_icf()->getL();
    // const MetroChains  *walkerLeft = metroChainsOperation.getWalkerLeft();
    // size_t numOfBrackets = walkerLeft->getNumOfBrackets();
    // size_t numOfChains = walkerLeft->getNumOfChains();
    // TensorHao<complex<double>, 2> sqsmq; 

    // if( L == 16){
    //     if( sqsmqNum.rank(0) != 4 ) { sqsmqNum.resize(4,4); sqsmqNum = complex<double>(0,0); }

    //     sqsmq.resize(4, 4);sqsmq=0.0;

    //     for(int q_x_order=1-1; q_x_order<=4-1; q_x_order++){
    //     for(int q_y_order=1-1; q_y_order<=4-1; q_y_order++){
    //         double q_x = 2.0 * M_PI * double(q_x_order)/double(4);
    //         double q_y = 2.0 * M_PI * double(q_y_order)/double(4);
    //         complex<double> szqszmq = 0.0;
    //         complex<double> sxqsxmq = 0.0;
    //         complex<double> syqsymq = 0.0;
    //         for(int i_x=1-1; i_x<=4-1; i_x++){
    //         for(int i_y=1-1; i_y<=4-1; i_y++){
    //             for(int j_x=1-1; j_x<=4-1; j_x++){
    //             for(int j_y=1-1; j_y<=4-1; j_y++){
    //                 int i=i_y + 4*i_x;
    //                 int j=j_y + 4*j_x; 
    //                 szqszmq += exp(complex<double>(0.0,1.0) * (q_x*double(i_x-j_x) + q_y*double(i_y-j_y)) ) * 0.25 *(localCdaggerCCdaggerC(metroChainsOperation,i,i,j,j) + localCdaggerCCdaggerC(metroChainsOperation,i+L,i+L,j+L,j+L) - localCdaggerCCdaggerC(metroChainsOperation,i+L,i+L,j,j) - localCdaggerCCdaggerC(metroChainsOperation,i,i,j+L,j+L));
    //                 sxqsxmq += exp(complex<double>(0.0,1.0) * (q_x*double(i_x-j_x) + q_y*double(i_y-j_y)) ) * 0.25 *(localCdaggerCCdaggerC(metroChainsOperation,i,i+L,j,j+L) + localCdaggerCCdaggerC(metroChainsOperation,i+L,i,j+L,j) + localCdaggerCCdaggerC(metroChainsOperation,i,i+L,j+L,j) + localCdaggerCCdaggerC(metroChainsOperation,i+L,i,j,j+L));
    //                 syqsymq += exp(complex<double>(0.0,1.0) * (q_x*double(i_x-j_x) + q_y*double(i_y-j_y)) ) * 0.25 * (-1.0) * (localCdaggerCCdaggerC(metroChainsOperation,i,i+L,j,j+L) + localCdaggerCCdaggerC(metroChainsOperation,i+L,i,j+L,j) - localCdaggerCCdaggerC(metroChainsOperation,i,i+L,j+L,j) - localCdaggerCCdaggerC(metroChainsOperation,i+L,i,j,j+L));
    //             }
    //             }
    //         }
    //         }
    //         sqsmq(q_x_order, q_y_order) += szqszmq + sxqsxmq + syqsymq;
    //     }
    //     }
    // }

    // sqsmqNum += ( sqsmq * denIncrement );
}

complex<double> HubbardSOC_icfMeasureObserveMetroChainsSD::localCdaggerCCdaggerC(MetroChainsOperation &metroChainsOperation, size_t i, size_t j, size_t k, size_t l)
{
    const MetroChains  *walkerLeft = metroChainsOperation.getWalkerLeft();
    size_t numOfBrackets = walkerLeft->getNumOfBrackets();
    size_t numOfChains = walkerLeft->getNumOfChains();

    complex<double> local_ijkl = 0.0;
    for(int bracket=1-1; bracket <= numOfBrackets*numOfChains-1; bracket++){
        complex<double> local_ij_kl = 0.0;
        complex<double> local_il_jk = 0.0;
        const TensorHao<complex<double>, 2> greenMatrix_bracket = metroChainsOperation.returnGreenMatrix(bracket);
        local_ij_kl = greenMatrix_bracket(i,j) * greenMatrix_bracket(k,l);
        if(j != k){
            local_il_jk = greenMatrix_bracket(i,l) * (-1.0) * greenMatrix_bracket(k,j);
        }else{
            local_il_jk = greenMatrix_bracket(i,l) * (1.0 - greenMatrix_bracket(k,j));
        }
        local_ijkl += (local_ij_kl + local_il_jk) * exp(metroChainsOperation.returnLogPhase(bracket));
    }
    local_ijkl = local_ijkl / exp(metroChainsOperation.returnLogTotalPhase());

    return local_ijkl;
}

complex<double> HubbardSOC_icfMeasureObserveMetroChainsSD::localCdaggerCCdaggerC_bracket(const TensorHao<complex<double>, 2> &greenMatrix_bracket, size_t i, size_t j, size_t k, size_t l)
{
    complex<double> local_ijkl = 0.0;
    complex<double> local_ij_kl = 0.0;
    complex<double> local_il_jk = 0.0;
    local_ij_kl = greenMatrix_bracket(i,j) * greenMatrix_bracket(k,l);
    if(j != k){
        local_il_jk = greenMatrix_bracket(i,l) * (-1.0) * greenMatrix_bracket(k,j);
    }else{
        local_il_jk = greenMatrix_bracket(i,l) * (1.0 - greenMatrix_bracket(k,j));
    }
    local_ijkl += (local_ij_kl + local_il_jk);

    return local_ijkl;
}

void HubbardSOC_icfMeasureObserveMetroChainsSD::addDensityDensity(MetroChainsOperation &metroChainsOperation, complex<double> denIncrement)
{
    //<ni nj>
    size_t L = getHubbardSOC_icf()->getL();
    const MetroChains  *walkerLeft = metroChainsOperation.getWalkerLeft();
    size_t numOfBrackets = walkerLeft->getNumOfBrackets();
    size_t numOfChains = walkerLeft->getNumOfChains();

    complex<double> temp;
    if( densityDensityNum.rank(0) != 2*L ) { densityDensityNum.resize(2*L, 2*L); densityDensityNum = complex<double>(0,0); }

    for(int i=1-1; i<=2*L-1; i++){
        for(int j=1-1; j<=2*L-1; j++){
            temp = localCdaggerCCdaggerC(metroChainsOperation,i,i,j,j);
            densityDensityNum(i,j) += temp * denIncrement;
        }
    }
}

// void HubbardSOC_icfMeasureObserveMetroChainsSD::addSplusSminus(MetroChainsOperation &metroChainsOperation, complex<double> denIncrement)
// {
//     const TensorHao< complex<double>, 2 > &greenMatrix = metroChainsOperation.returnGreenMatrix();

//     size_t L = getHubbardSOC_icf()->getL();

//     if( splusSminusNum.rank(0) != L ) { splusSminusNum.resize(L, L); splusSminusNum = complex<double>(0,0); }

//     complex<double> temp;
//     for(size_t j = 0; j < L; ++j)
//     {
//         for(size_t i = 0; i < L; ++i)
//         {
//             if( i==j )
//             {
//                 temp  = greenMatrix(i,i);
//                 temp -= ( greenMatrix(i,i)*greenMatrix(i+L,i+L)-greenMatrix(i,i+L)*greenMatrix(i+L,i) );
//             }
//             else
//             {
//                 temp = -greenMatrix(i,j)*greenMatrix(j+L,i+L)+greenMatrix(i,i+L)*greenMatrix(j+L, j);
//             }
//             splusSminusNum(i,j) += ( temp * denIncrement );
//         }
//     }
// }

// void HubbardSOC_icfMeasureObserveMetroChainsSD::addSminusSplus(MetroChainsOperation &metroChainsOperation, complex<double> denIncrement)
// {
//     const TensorHao< complex<double>, 2 > &greenMatrix = metroChainsOperation.returnGreenMatrix();

//     size_t L = getHubbardSOC_icf()->getL();

//     if( sminusSplusNum.rank(0) != L ) { sminusSplusNum.resize(L, L); sminusSplusNum = complex<double>(0,0); }

//     complex<double> temp;
//     for(size_t j = 0; j < L; ++j)
//     {
//         for(size_t i = 0; i < L; ++i)
//         {
//             if( i==j )
//             {
//                 temp  = greenMatrix(i+L, i+L);
//                 temp -= ( greenMatrix(i,i)*greenMatrix(i+L,i+L)-greenMatrix(i,i+L)*greenMatrix(i+L,i) );
//             }
//             else
//             {
//                 temp = -greenMatrix(i+L,j+L)*greenMatrix(j,i)+greenMatrix(i+L,i)*greenMatrix(j, j+L);
//             }
//             sminusSplusNum(i,j) += ( temp * denIncrement );
//         }
//     }
// }

// void HubbardSOC_icfMeasureObserveMetroChainsSD::addSpairSpair(MetroChainsOperation &metroChainsOperation, complex<double> denIncrement)
// {
//     const TensorHao< complex<double>, 2 > &greenMatrix = metroChainsOperation.returnGreenMatrix();

//     size_t L = getHubbardSOC_icf()->getL();

//     if( spairSpairNum.rank(0) != L ) { spairSpairNum.resize(L, L); spairSpairNum = complex<double>(0,0); }

//     complex<double> temp;
//     for(size_t j=0; j<L; j++)
//     {
//         for(size_t i=0; i<L; i++)
//         {
//             temp = greenMatrix(i,j)*greenMatrix(i+L,j+L)-greenMatrix(i,j+L)*greenMatrix(i+L,j);
//             spairSpairNum(i,j) += ( temp * denIncrement );
//         }
//     }
// }

void HubbardSOC_icfMeasureObserveMetroChainsSD::addEnergy(MetroChainsOperation &metroChainsOperation, complex<double> denIncrement)
{
    complex<double> Kenergy(0,0), Venergy(0,0), Renergy(0,0);

    size_t L  = hubbardSOC_icf->getL(); size_t L2 = L*2;
    const MetroChains  *walkerLeft = metroChainsOperation.getWalkerLeft();
    size_t numOfBrackets = walkerLeft->getNumOfBrackets();
    size_t numOfChains = walkerLeft->getNumOfChains();
    TensorHao<complex<double>, 2> greenMatrix; greenMatrix.resize(walkerLeft->getL(), walkerLeft->getL());
    
    //Add U
    const TensorHao< double, 1> &U = hubbardSOC_icf->getU();
    for(int bracket=1-1; bracket <= numOfBrackets*numOfChains-1; bracket++){
        const TensorHao<complex<double>, 2> greenMatrix_bracket = metroChainsOperation.returnGreenMatrix(bracket);
        greenMatrix += greenMatrix_bracket * exp(metroChainsOperation.returnLogPhase(bracket)-metroChainsOperation.returnLogTotalPhase());
        // 
        complex<double> Venergy_bracket(0,0);
        for(size_t i = 0; i < L; ++i)
        { 
            Venergy_bracket += U(i) * ( greenMatrix_bracket(i,i)*greenMatrix_bracket(i+L,i+L) - greenMatrix_bracket(i, i+L)*greenMatrix_bracket(i+L, i) );
        }
        Venergy += Venergy_bracket * exp(metroChainsOperation.returnLogPhase(bracket)-metroChainsOperation.returnLogTotalPhase());
    }

    //Add K
    const TensorHao< complex<double>, 2 > &K = hubbardSOC_icf->getK();
    for(size_t i = 0; i < L2; ++i)
    {
        for(size_t j = 0; j < L2; ++j)
        {
            Kenergy += K(j,i) * greenMatrix(j,i);
        }
    }

    //Add U
    // const TensorHao< double, 1> &U = hubbardSOC_icf->getU();
    // for(size_t bracket=1-1; bracket <=numOfBrackets*numOfChains - 1; bracket ++ ){
    //     const TensorHao<complex<double>, 2> greenMatrix_bracket = metroChainsOperation.returnGreenMatrix(bracket);  
    //     complex<double> Venergy_bracket(0,0);
    //     for(size_t i = 0; i < L; ++i)
    //     { 
    //         Venergy_bracket += U(i) * ( greenMatrix_bracket(i,i)*greenMatrix_bracket(i+L,i+L) - greenMatrix_bracket(i, i+L)*greenMatrix_bracket(i+L, i) );
    //     }
    //     Venergy += Venergy_bracket * exp(metroChainsOperation.returnLogPhase(bracket)-metroChainsOperation.returnLogTotalPhase());
    // }

    //Add mu and pinning field
    const TensorHao< double, 1> &mu = hubbardSOC_icf->getMu();
    const TensorHao< double, 1> &hx = hubbardSOC_icf->getHx();
    const TensorHao< double, 1> &hy = hubbardSOC_icf->getHy();
    const TensorHao< double, 1> &hz = hubbardSOC_icf->getHz();
    for(size_t i = 0; i < L; ++i)
    {
        Renergy += ( -mu(i) + hz(i)*0.5 ) * greenMatrix(i,i);
        Renergy += ( -mu(i) - hz(i)*0.5 ) * greenMatrix(i+L,i+L);
        Renergy += complex<double>( hx(i)*0.5, -hy(i)*0.5 ) * greenMatrix(i, i+L);
        Renergy += complex<double>( hx(i)*0.5,  hy(i)*0.5 ) * greenMatrix(i+L, i);
    }

    HNum += ( ( Kenergy + Venergy + Renergy ) * denIncrement );
    KNum += ( Kenergy * denIncrement );
    VNum += ( Venergy * denIncrement );
    RNum += ( Renergy * denIncrement );
}

HubbardSOC_icfMeasureObserveMetroChainsSD::HubbardSOC_icfMeasureObserveMetroChainsSD(const HubbardSOC_icfMeasureObserveMetroChainsSD &x)
{

}

HubbardSOC_icfMeasureObserveMetroChainsSD &HubbardSOC_icfMeasureObserveMetroChainsSD::operator=(const HubbardSOC_icfMeasureObserveMetroChainsSD &x)
{
    return *this;
}