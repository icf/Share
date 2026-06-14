//
// Created by boruoshihao on 11/14/18.
//

#include <chrono>
#include "../../include/utils.h"
#include "../../include/generalHamiltonian_icf/generalHamiltonian_icfMeasureObserveMetroChainsSD.h"
#include "afqmclab.h"
#include <math.h>

using namespace std;
using namespace tensor_hao;

GeneralHamiltonian_icfMeasureObserveMetroChainsSD::GeneralHamiltonian_icfMeasureObserveMetroChainsSD()
{
    initModelNullptr();
    reSet();
}

GeneralHamiltonian_icfMeasureObserveMetroChainsSD::GeneralHamiltonian_icfMeasureObserveMetroChainsSD(const GeneralHamiltonian_icf &generalHamiltonian_icf_, MetroChains & walkerLeft)
{
    setModel_withPhiTConst(generalHamiltonian_icf_, walkerLeft);
    reSet();
}

GeneralHamiltonian_icfMeasureObserveMetroChainsSD::~GeneralHamiltonian_icfMeasureObserveMetroChainsSD()
{

}

void GeneralHamiltonian_icfMeasureObserveMetroChainsSD::initModelNullptr()
{
    generalHamiltonian_icf = nullptr;
}

void GeneralHamiltonian_icfMeasureObserveMetroChainsSD::setModel_withPhiTConst(const GeneralHamiltonian_icf &generalHamiltonian_icf_, MetroChains & walkerLeft)
{
    generalHamiltonian_icf = &generalHamiltonian_icf_;
    //
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();
    int L = generalHamiltonian_icf->getL();
    bool Hamiltonian_spin_flag = generalHamiltonian_icf->getHamiltonian_spin_flag();
    if(Hamiltonian_spin_flag){
        // 
        A0Lgamma.resize(svdNumber);
        B0Lgamma.resize(svdNumber);
        // 
        const TensorHao<complex<double>,2> & K = generalHamiltonian_icf->getK();
        const TensorHaoMPIRef<complex<double>,3> & svdVecs = generalHamiltonian_icf->getSVDVecs();
        KP.resize(L, L);
        TensorHao<complex<double>, 2> svdVecComplexTemp(L,L);
        TensorHao<complex<double>, 2> svdTVecComplexSquare(L,L);
        svdTVecComplexSquare = 0.0;
        TensorHao<complex<double>, 2> svdTVecComplexSquareTemp(L,L);
        TensorHao<complex<double>, 2> svdTVecComplexSquareTemp2(L,L);
        // 
        A0 = walkerLeft.metropolisVec[0].metropolisInfo.A0;
        // 
        A1 = walkerLeft.metropolisVec[0].metropolisInfo.A1;
        // 
        B0 = walkerLeft.metropolisVec[0].metropolisInfo.B0;
        // 
#ifdef MPI_HAO
        int ranks_per_node = 24;
        char* cpus_per_node = getenv("SLURM_JOB_CPUS_PER_NODE");
        if(cpus_per_node) ranks_per_node = atoi(cpus_per_node);
        int node_id = MPIRank() / ranks_per_node;
        int local_rank = MPIRank() % ranks_per_node;
        bool is_local_root = (local_rank == 0);
        
        MPI_Comm node_comm;
        MPI_Comm_split(MPI_COMM_WORLD, node_id, MPIRank(), &node_comm);
#else
        bool is_local_root = true;
#endif
        
        size_t dimA0[2] = {A0.rank(0), (size_t)svdVecComplexTemp.rank(1)};
        size_t dimB0[2] = {B0.rank(0), (size_t)svdVecComplexTemp.rank(1)};
        
        for(size_t k = 0; k < svdNumber ; ++k)
        {
            if(is_local_root) {
                A0Lgamma[k].createSharedMemory(dimA0, 0, node_comm);
                B0Lgamma[k].createSharedMemory(dimB0, 0, node_comm);
            }
            else {
                A0Lgamma[k].createSharedMemoryView(0, node_comm);
                A0Lgamma[k].attachToSharedMemory(dimA0, 0, node_comm);
                B0Lgamma[k].createSharedMemoryView(0, node_comm);
                B0Lgamma[k].attachToSharedMemory(dimB0, 0, node_comm);
            }
        }
        
        // Create temporary TensorHao as intermediary for gmm calculation
        TensorHao<complex<double>, 2> tempA0Lgamma(A0.rank(0), svdVecComplexTemp.rank(1));
        TensorHao<complex<double>, 2> tempB0Lgamma(B0.rank(0), svdVecComplexTemp.rank(1));
        
        for(size_t k = 0; k < svdNumber ; ++k)
        {
            for(int i=1-1; i<=L-1; i++){
                for(int j=1-1; j<=L-1; j++){
                    svdVecComplexTemp(i,j) = svdVecs(i,j,k);
                }
            }
            // 
            BL_NAME(gmm)(svdVecComplexTemp, svdVecComplexTemp, svdTVecComplexSquareTemp, 'T', 'T');
            svdTVecComplexSquareTemp2 = svdTVecComplexSquare + svdTVecComplexSquareTemp;
            svdTVecComplexSquare = svdTVecComplexSquareTemp2;
            // 
            if(is_local_root) {
                // Use TensorHao as intermediary for gmm calculation
                BL_NAME(gmm)(A0, svdVecComplexTemp, tempA0Lgamma);
                BL_NAME(gmm)(B0, svdVecComplexTemp, tempB0Lgamma);
                
                // Copy from TensorHao to TensorHaoMPIRef
                for(size_t i = 0; i < tempA0Lgamma.rank(0); ++i)
                    for(size_t j = 0; j < tempA0Lgamma.rank(1); ++j)
                        A0Lgamma[k](i, j) = tempA0Lgamma(i, j);
                
                for(size_t i = 0; i < tempB0Lgamma.rank(0); ++i)
                    for(size_t j = 0; j < tempB0Lgamma.rank(1); ++j)
                        B0Lgamma[k](i, j) = tempB0Lgamma(i, j);
            }
        }
#ifdef MPI_HAO
        MPIBarrier(node_comm);
#endif
        // 
        for(int i=1-1; i<=L-1; i++){
        for(int j=1-1; j<=L-1; j++){
            KP(i,j) = K(i,j) + 0.5 * svdTVecComplexSquare(i,j);
        }
        }
    }else{
        // 
        A0Lgamma.resize(svdNumber);
        B0Lgamma.resize(svdNumber);
        // 
        const TensorHao<complex<double>,2> & K = generalHamiltonian_icf->getK();
        const TensorHaoMPIRef<complex<double>,3> & svdVecs = generalHamiltonian_icf->getSVDVecs();
        KP.resize(2*L, 2*L);
        TensorHao<complex<double>, 2> svdVecComplexTemp(2*L,2*L);
        TensorHao<complex<double>, 2> svdTVecComplexSquare(2*L,2*L);
        svdTVecComplexSquare = 0.0;
        TensorHao<complex<double>, 2> svdTVecComplexSquareTemp(2*L,2*L);
        TensorHao<complex<double>, 2> svdTVecComplexSquareTemp2(2*L,2*L);
        // 
        A0 = walkerLeft.metropolisVec[0].metropolisInfo.A0;
        // 
        A1 = walkerLeft.metropolisVec[0].metropolisInfo.A1;
        // 
        B0 = walkerLeft.metropolisVec[0].metropolisInfo.B0;
        // 
#ifdef MPI_HAO
        int ranks_per_node = 24;
        char* cpus_per_node = getenv("SLURM_JOB_CPUS_PER_NODE");
        if(cpus_per_node) ranks_per_node = atoi(cpus_per_node);
        int node_id = MPIRank() / ranks_per_node;
        int local_rank = MPIRank() % ranks_per_node;
        bool is_local_root = (local_rank == 0);
        
        MPI_Comm node_comm;
        MPI_Comm_split(MPI_COMM_WORLD, node_id, MPIRank(), &node_comm);
#else
        bool is_local_root = true;
#endif
        
        size_t dimA0[2] = {A0.rank(0), (size_t)svdVecComplexTemp.rank(1)};
        size_t dimB0[2] = {B0.rank(0), (size_t)svdVecComplexTemp.rank(1)};
        
        for(size_t k = 0; k < svdNumber ; ++k)
        {
            if(is_local_root) {
                A0Lgamma[k].createSharedMemory(dimA0, 0, node_comm);
                B0Lgamma[k].createSharedMemory(dimB0, 0, node_comm);
            }
            else {
                A0Lgamma[k].createSharedMemoryView(0, node_comm);
                A0Lgamma[k].attachToSharedMemory(dimA0, 0, node_comm);
                B0Lgamma[k].createSharedMemoryView(0, node_comm);
                B0Lgamma[k].attachToSharedMemory(dimB0, 0, node_comm);
            }
        }
        
        // Create temporary TensorHao as intermediary for gmm calculation
        TensorHao<complex<double>, 2> tempA0Lgamma(A0.rank(0), svdVecComplexTemp.rank(1));
        TensorHao<complex<double>, 2> tempB0Lgamma(B0.rank(0), svdVecComplexTemp.rank(1));
        
        for(size_t k = 0; k < svdNumber ; ++k)
        {
            svdVecComplexTemp = 0.0;
            for(int i=1-1; i<=L-1; i++){
                for(int j=1-1; j<=L-1; j++){
                    svdVecComplexTemp(i,j) = svdVecs(i,j,k);
                    svdVecComplexTemp(i+L,j+L) = svdVecs(i,j,k);
                }
            }
            // 
            BL_NAME(gmm)(svdVecComplexTemp, svdVecComplexTemp, svdTVecComplexSquareTemp, 'T', 'T');
            svdTVecComplexSquareTemp2 = svdTVecComplexSquare + svdTVecComplexSquareTemp;
            svdTVecComplexSquare = svdTVecComplexSquareTemp2;
            // 
            if(is_local_root) {
                // Use TensorHao as intermediary for gmm calculation
                BL_NAME(gmm)(A0, svdVecComplexTemp, tempA0Lgamma);
                BL_NAME(gmm)(B0, svdVecComplexTemp, tempB0Lgamma);
                
                // Copy from TensorHao to TensorHaoMPIRef
                for(size_t i = 0; i < tempA0Lgamma.rank(0); ++i)
                    for(size_t j = 0; j < tempA0Lgamma.rank(1); ++j)
                        A0Lgamma[k](i, j) = tempA0Lgamma(i, j);
                
                for(size_t i = 0; i < tempB0Lgamma.rank(0); ++i)
                    for(size_t j = 0; j < tempB0Lgamma.rank(1); ++j)
                        B0Lgamma[k](i, j) = tempB0Lgamma(i, j);
            }
        }
#ifdef MPI_HAO
        MPIBarrier(node_comm);
#endif
        // 
        for(int i=1-1; i<=L-1; i++){
        for(int j=1-1; j<=L-1; j++){
            KP(i,j) = K(i,j) + 0.5 * svdTVecComplexSquare(i,j);
            KP(i+L,j+L) = K(i,j) + 0.5 * svdTVecComplexSquare(i+L,j+L);
        }
        }
    }
    // 
}

void GeneralHamiltonian_icfMeasureObserveMetroChainsSD::reSet()
{
    complex<double> zero(0,0);

    den = zero;
    TNum = zero;
    svdBgNum = zero;
    svdExNum = zero;
    HNum = zero;
    // greenNum = zero;
}

complex<double> GeneralHamiltonian_icfMeasureObserveMetroChainsSD::returnEnergy()
{
    complex<double> Htot   = MPISum(HNum);
    complex<double> denTot = MPISum(den);
    complex<double> energy;
    if( MPIRank() == 0 ) energy = Htot/denTot;
    MPIBcast(energy);
    return energy;
}

complex<double> GeneralHamiltonian_icfMeasureObserveMetroChainsSD::returnKEnergy()
{
    complex<double> Ktot   = MPISum(TNum);
    complex<double> denTot = MPISum(den);
    complex<double> Kenergy;
    if( MPIRank() == 0 ) Kenergy = Ktot/denTot;
    MPIBcast(Kenergy);
    return Kenergy;
}

TensorHao<complex<double>,1> GeneralHamiltonian_icfMeasureObserveMetroChainsSD::returnSVDBg()
{
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();

    TensorHao<complex<double>, 1> svdBgTot( svdNumber );
    copy(svdBgNum.data(), svdBgNum.data() + svdNumber, svdBgTot.data());
    complex<double> denTot = den;

    TensorHao<complex<double>,1> svdBg( svdNumber );
    for(size_t i = 0; i < svdNumber; ++i) svdBg(i) = svdBgTot(i)/denTot;

    return svdBg;
}

TensorHao<complex<double>,1> GeneralHamiltonian_icfMeasureObserveMetroChainsSD::returnSVDBgReal()
{
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();

    TensorHao<complex<double>, 1> svdBgTot( svdNumber );
    copy(svdBgNum.data(), svdBgNum.data() + svdNumber, svdBgTot.data());
    complex<double> denTot = den;

    TensorHao<complex<double>,1> svdBg( svdNumber );
    for(size_t i = 0; i < svdNumber; ++i) svdBg(i) = (svdBgTot(i)/denTot).real();

    return svdBg;
}

void GeneralHamiltonian_icfMeasureObserveMetroChainsSD::addMeasurement(MetroChains & walkerLeft, complex<double> denIncrement)
{
    checkWalkerWithModel(walkerLeft);

    complex<double> denSave; denSave=den;
    den += denIncrement;
    addEnergy_fast(walkerLeft, denIncrement);
}

void GeneralHamiltonian_icfMeasureObserveMetroChainsSD::addMeasurement_timer(MetroChains & walkerLeft, complex<double> denIncrement)
{
    checkWalkerWithModel(walkerLeft);

    complex<double> denSave; denSave=den;
    den += denIncrement;
    addEnergy_fast(walkerLeft, denIncrement);
}

///////////////////////////////////////////////////////
void GeneralHamiltonian_icfMeasureObserveMetroChainsSD::addMeasurement(MetroChainsOperation &metroChainsOperation, complex<double> denIncrement)
{
    // // wfUpDaggerKList.resize(0);
    // // wfDnDaggerKList.resize(0);
    // // wfUpDaggerSVDVecsList.resize(0);
    // // wfDnDaggerSVDVecsList.resize(0);
    // // 
    // checkWalkerWithModel(metroChainsOperation);

    // // initWfDaggerKList(metroChainsOperation);
    // // initWfDaggerSVDVecsList(metroChainsOperation);

    // complex<double> denSave; denSave=den;
    // den += denIncrement;
    // addEnergy_compact(metroChainsOperation, denIncrement);

    // // wfUpDaggerKList.resize(0);
    // // wfDnDaggerKList.resize(0);
    // // wfUpDaggerSVDVecsList.resize(0);
    // // wfDnDaggerSVDVecsList.resize(0);
}

void GeneralHamiltonian_icfMeasureObserveMetroChainsSD::addMeasurement_timer(MetroChainsOperation &metroChainsOperation, complex<double> denIncrement)
{
    // // wfUpDaggerKList.resize(0);
    // // wfDnDaggerKList.resize(0);
    // // wfUpDaggerSVDVecsList.resize(0);
    // // wfDnDaggerSVDVecsList.resize(0);
    // // 
    // checkWalkerWithModel(metroChainsOperation);

    // // initWfDaggerKList(metroChainsOperation);
    // // initWfDaggerSVDVecsList(metroChainsOperation);

    // complex<double> denSave; denSave=den;
    // den += denIncrement;
    // addEnergy_compact(metroChainsOperation, denIncrement);

    // // wfUpDaggerKList.resize(0);
    // // wfDnDaggerKList.resize(0);
    // // wfUpDaggerSVDVecsList.resize(0);
    // // wfDnDaggerSVDVecsList.resize(0);
}
///////////////////////////////////////////////////////

SVDForce GeneralHamiltonian_icfMeasureObserveMetroChainsSD::getForce_fast(const SVD &svd,
                                                                      MetroChains & walkerLeft,
                                                                      double cap)
{
    checkWalkerWithModel(walkerLeft);

    size_t N = walkerLeft.getN();
    size_t numOfBrackets = walkerLeft.getNumOfBrackets();
    size_t numOfChains = walkerLeft.getNumOfChains();
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();
    const TensorHao<complex<double>,1> & currentBg = generalHamiltonian_icf->getSVDBg();
    complex<double> sqrtMinusDt = svd.getSqrtMinusDt();

    TensorHao<complex<double>, 1> svdBg(svdNumber); svdBg=0.0;
    //////////////////////////////////////////////////////////
    if(walkerLeft.metropolisVec[0].method.BPMetroUpdateType == "global_fast"){
        // 
        TensorHao<complex<double>, 2> matrixInv_Sum; 
        matrixInv_Sum.resize(N,N);
        matrixInv_Sum = 0.0;
        // 
        TensorHao<complex<double>, 2> matrixInvA1ExpS_Sum;
        matrixInvA1ExpS_Sum.resize(N,walkerLeft.metropolisVec[0].metropolisInfo.A1.rank(1)); 
        matrixInvA1ExpS_Sum = 0.0; 
        // 
        TensorHao<complex<double>, 2> overlapMatrixInvA1ExpS;
        // 
        for(int bracket=1-1; bracket <= numOfBrackets*numOfChains-1; bracket++){
            complex<double> scale = exp(walkerLeft.returnLogPhase_fromCurrentOverlap(bracket)-walkerLeft.returnLogTotalPhase_fromCurrentOverlap());
            ////////////////////////////////////////////
            // makesure overlapMatrix_inv is correct before using
            walkerLeft.metropolisVec[bracket].updateOverlapMatrix_inv_fromOverlapMatrix();
            ////////////////////////////////////////////
            matrixInv_Sum = matrixInv_Sum + walkerLeft.metropolisVec[bracket].metropolisInfo.overlapMatrix_inv * scale;
            // matrixInvA1ExpD_Sum = matrixInvA1ExpD_Sum + matrixInv @ A1 @ (S + 0.5 * S @ C0 @ S);
            walkerLeft.metropolisVec[bracket].getOverlapMatrixInvA1ExpS(overlapMatrixInvA1ExpS);
            matrixInvA1ExpS_Sum = matrixInvA1ExpS_Sum + overlapMatrixInvA1ExpS * scale;
        }
        // 
        TensorHao<complex<double>, 2> matrix(N,N);
        vector<TensorHao<complex<double>, 2>> A0LgammaWR_vec, B0LgammaWR_vec;
        get_A0LgammaWR_B0LgammaWR(walkerLeft.metropolisVec[0].metropolisInfo.walkerRightInBlock[0].getWf(), A0LgammaWR_vec, B0LgammaWR_vec);
        for(size_t k = 0; k < svdNumber ; ++k)
        {
            // get const matrix for bracket
            // TensorHao<complex<double>, 2> A0PWR, B0PWR;
            // walkerLeft.metropolisVec[0].get_A0PWR_B0PWR(svdVecComplex[k], A0PWR, B0PWR);
            // A0PWR_matrixInv_Sum
            TensorHao<complex<double>, 2> matrixInv_Sum_A0LgammaWR(matrixInv_Sum.rank(0), A0LgammaWR_vec[k].rank(1));
            BL_NAME(gmm)(A0LgammaWR_vec[k], matrixInv_Sum, matrixInv_Sum_A0LgammaWR);
            // 
            // matrixInvA1ExpS_Sum_B0PWR = matrixInvA1ExpS_Sum @ B0LgammaWR
            TensorHao<complex<double>, 2> matrixInvA1ExpS_Sum_B0LgammaWR(matrixInvA1ExpS_Sum.rank(0), B0LgammaWR_vec[k].rank(1));

            BL_NAME(gmm)(matrixInvA1ExpS_Sum, B0LgammaWR_vec[k], matrixInvA1ExpS_Sum_B0LgammaWR);
            // matrix = matrixInv_Sum @ A0LgammaWR + matrixInvA1ExpS_Sum @ B0LgammaWR;
            matrix = matrixInv_Sum_A0LgammaWR + matrixInvA1ExpS_Sum_B0LgammaWR;
            // 
            complex<double> diagSum = 0.0;
            for(size_t i = 0; i < N; ++i) {
                diagSum += matrix(i,i);
            }
            svdBg(k) = diagSum;
            // 
        }
        // 
    }
    //////////////////////////////////////////////////////////
    // 
    SVDForce force(svdNumber); complex<double> oneForce;
    for(size_t i = 0; i < svdNumber; ++i)
    {
        oneForce = (svdBg(i)-currentBg(i)) * sqrtMinusDt;

        if( abs(oneForce) > cap ) force(i) = oneForce*cap/abs(oneForce);
        else force(i) = oneForce;
    }
    
    return force;   
}

SVDForce GeneralHamiltonian_icfMeasureObserveMetroChainsSD::getForce(const SVD &svd,
                                                                      MetroChainsOperation &metroChainsOperation,
                                                                      double cap)
{
    // checkWalkerWithModel(metroChainsOperation);

    // const MetroChains  *walkerLeft = metroChainsOperation.getWalkerLeft();
    // size_t halfL = walkerLeft->getL(); size_t Nup = walkerLeft->getNup(); size_t Ndn = walkerLeft->getNdn(); 
    // size_t numOfBrackets = walkerLeft->getNumOfBrackets();
    // size_t numOfChains = walkerLeft->getNumOfChains();
    // size_t svdNumber = generalHamiltonian_icf->getSVDNumber();
    // const TensorHao<complex<double>,1> & currentBg = generalHamiltonian_icf->getSVDBg();
    // complex<double> sqrtMinusDt = svd.getSqrtMinusDt();

    // TensorHao<complex<double>, 1> svdBg(svdNumber); svdBg=0.0;
    // TensorHao<complex<double>, 2> wfUpDaggerSVDVec(Nup,halfL), wfDnDaggerSVDVec(Ndn,halfL);
    // // 
    // TensorHao<complex<double>, 2> matrixUp_temp(Nup,Nup), matrixDn_temp(Ndn,Ndn);
    // TensorHao<complex<double>, 2> thetaUp(halfL,Nup), thetaDn(halfL,Ndn);
    // for(int bracket=1-1; bracket <= numOfBrackets*numOfChains-1; bracket++){
    //     TensorHao<complex<double>, 1> svdBgChain(svdNumber); svdBgChain=0.0;
    //     const TensorHao<complex<double>, 2> &thetaUp_T =  metroChainsOperation.returnThetaUp_T(bracket);
    //     const TensorHao<complex<double>, 2> &thetaDn_T =  metroChainsOperation.returnThetaDn_T(bracket);
    //     thetaUp = trans(thetaUp_T); 
    //     thetaDn = trans(thetaDn_T); 
    //     for(size_t k = 0; k < svdNumber ; ++k)
    //     {
    //         TensorHao<complex<double>,2> wfUpTemp=walkerLeft->metroLeft[bracket].getWfUp();
    //         BL_NAME(gmm)(wfUpTemp, svdVecComplexUp[k], wfUpDaggerSVDVec, 'C');
    //         // 
    //         TensorHao<complex<double>,2> wfDnTemp=walkerLeft->metroLeft[bracket].getWfDn();
    //         BL_NAME(gmm)(wfDnTemp, svdVecComplexDn[k], wfDnDaggerSVDVec, 'C');
    //         ////////////////////////////////////////////////////////////
    //         // svdBgChain
    //         ////////////////////////////////////////////////////////////
    //         BL_NAME(gmm)(wfUpDaggerSVDVec, thetaUp, matrixUp_temp);
    //         for(size_t i = 0; i < Nup; ++i){
    //             svdBgChain(k) += matrixUp_temp(i,i);
    //         }
    //         // 
    //         BL_NAME(gmm)(wfDnDaggerSVDVec, thetaDn, matrixDn_temp);
    //         for(size_t i = 0; i < Ndn; ++i){
    //             svdBgChain(k) += matrixDn_temp(i,i);
    //         }
    //     }
    //     svdBg += svdBgChain * exp(metroChainsOperation.returnLogPhase(bracket)-metroChainsOperation.returnLogTotalPhase());
    // }
    // // 
    // SVDForce force(svdNumber); complex<double> oneForce;
    // for(size_t i = 0; i < svdNumber; ++i)
    // {
    //     oneForce = (svdBg(i)-currentBg(i)) * sqrtMinusDt;

    //     if( abs(oneForce) > cap ) force(i) = oneForce*cap/abs(oneForce);
    //     else force(i) = oneForce;
    // }
    

    // return force;
}

void GeneralHamiltonian_icfMeasureObserveMetroChainsSD::write() const
{
    writeThreadSum(den, "den.dat", ios::app);
    writeThreadSum(TNum, "TNum.dat", ios::app);
    // writeThreadSum(svdBgNum.size(), svdBgNum.data(), "svdBgNum.dat", ios::app);
    // writeThreadSum(svdExNum.size(), svdExNum.data(), "svdExNum.dat", ios::app);
    writeThreadSum(HNum, "HNum.dat", ios::app);
    // writeThreadSum(greenNum.size(), greenNum.data(), "greenNum.dat", ios::app);
}

void GeneralHamiltonian_icfMeasureObserveMetroChainsSD::write(std::string postfix ) const
{
    std::string denName="den"+postfix+".dat";
    std::string HNumName="HNum"+postfix+".dat";
    writeThreadSum(den, denName, ios::app);
    writeThreadSum(HNum, HNumName, ios::app);
}

double GeneralHamiltonian_icfMeasureObserveMetroChainsSD::getMemory() const
{
    double mem(0.0);
    
    // Basic primitive types: den, TNum, HNum
    mem += 3 * sizeof(std::complex<double>);
    
    // Tensor objects
    mem += svdBgNum.getMemory();
    mem += svdExNum.getMemory();
    mem += svdNormalNum.getMemory();
    mem += KP.getMemory();
    
    // Additional tensor objects
    mem += A0.getMemory();
    mem += A1.getMemory();
    mem += B0.getMemory();
    
    // Vectors of tensors and their contents
    for(const auto& tensor : A0Lgamma) {
        mem += tensor.getMemory();
    }
    for(const auto& tensor : B0Lgamma) {
        mem += tensor.getMemory();
    }

    return mem;
}

GeneralHamiltonian_icfMeasureObserveMetroChainsSD::GeneralHamiltonian_icfMeasureObserveMetroChainsSD(const GeneralHamiltonian_icfMeasureObserveMetroChainsSD &x)
{

}

GeneralHamiltonian_icfMeasureObserveMetroChainsSD & GeneralHamiltonian_icfMeasureObserveMetroChainsSD::operator=(const GeneralHamiltonian_icfMeasureObserveMetroChainsSD &x)
{
    return *this;
}

void GeneralHamiltonian_icfMeasureObserveMetroChainsSD::checkWalkerWithModel(const MetroChainsOperation &metroChainsOperation)
{
    const MetroChains *walkerLeft = metroChainsOperation.getWalkerLeft();

    if(generalHamiltonian_icf->getHamiltonian_spin_flag() == true){
        if( generalHamiltonian_icf->getL() != walkerLeft->getL() ) {
            cout<<"checkWalkerWithModel metroChainsOperation: Model L does not consistent with walker left L!"<<endl; 
            cout<<"generalHamiltonian_icf->getHamiltonian_spin_flag(): "<<generalHamiltonian_icf->getHamiltonian_spin_flag()<<endl; 
            cout<<"generalHamiltonian_icf->getL(): "<<generalHamiltonian_icf->getL()<<endl; 
            cout<<"walkerLeft->getL(): "<<walkerLeft->getL()<<endl; 
            exit(1);
        }
    }else{
        if( 2*generalHamiltonian_icf->getL() != walkerLeft->getL() ) {
            cout<<"checkWalkerWithModel metroChainsOperation: Model L does not consistent with walker left L!"<<endl; 
            cout<<"generalHamiltonian_icf->getHamiltonian_spin_flag(): "<<generalHamiltonian_icf->getHamiltonian_spin_flag()<<endl; 
            cout<<"generalHamiltonian_icf->getL(): "<<generalHamiltonian_icf->getL()<<endl; 
            cout<<"walkerLeft->getL(): "<<walkerLeft->getL()<<endl; 
            exit(1);
        }
    }
    if( generalHamiltonian_icf->getN() != walkerLeft->getN() ) {cout<<"Model N does not consistent with walker left N!"<<endl; exit(1);}
}

void GeneralHamiltonian_icfMeasureObserveMetroChainsSD::checkWalkerWithModel(const MetroChains &walkerLeft)
{

    if(generalHamiltonian_icf->getHamiltonian_spin_flag() == true){
        if( generalHamiltonian_icf->getL() != walkerLeft.getL() ) {
            cout<<"checkWalkerWithModel walkerLeft: Model L does not consistent with walker left L!"<<endl; 
            cout<<"generalHamiltonian_icf->getHamiltonian_spin_flag(): "<<generalHamiltonian_icf->getHamiltonian_spin_flag()<<endl; 
            cout<<"generalHamiltonian_icf->getL(): "<<generalHamiltonian_icf->getL()<<endl; 
            cout<<"walkerLeft.getL(): "<<walkerLeft.getL()<<endl; 
            exit(1);
        }
    }else{
        if( 2*generalHamiltonian_icf->getL() != walkerLeft.getL() ) {
            cout<<"checkWalkerWithModel walkerLeft: Model L does not consistent with walker left L!"<<endl; 
            cout<<"generalHamiltonian_icf->getHamiltonian_spin_flag(): "<<generalHamiltonian_icf->getHamiltonian_spin_flag()<<endl; 
            cout<<"generalHamiltonian_icf->getL(): "<<generalHamiltonian_icf->getL()<<endl; 
            cout<<"walkerLeft.getL(): "<<walkerLeft.getL()<<endl;  
            exit(1);
        }
    }
    if( generalHamiltonian_icf->getN() != walkerLeft.getN() ) {cout<<"Model N does not consistent with walker left N!"<<endl; exit(1);}
}

void GeneralHamiltonian_icfMeasureObserveMetroChainsSD::addEnergy_fast(MetroChains & walkerLeft, complex<double> denIncrement)
{
    size_t N = walkerLeft.getN(); 
    size_t numOfBrackets = walkerLeft.getNumOfBrackets();
    size_t numOfChains = walkerLeft.getNumOfChains();
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();

    if( svdBgNum.rank(0) != svdNumber ) { svdBgNum.resize(svdNumber); svdBgNum = complex<double>(0,0); }
    if( svdNormalNum.rank(0) != svdNumber ) { svdNormalNum.resize(svdNumber); svdNormalNum = complex<double>(0,0); }
    if( svdExNum.rank(0) != svdNumber ) { svdExNum.resize(svdNumber); svdExNum = complex<double>(0,0); }

    complex<double> Henergy(0.0);
    complex<double> Kenergy(0.0);
    TensorHao<complex<double>, 1> svdBg(svdNumber);svdBg=0.0;
    TensorHao<complex<double>, 1> svdEx(svdNumber);svdEx=0.0;
    //////////////////////////////////////////////////////////////
    vector<complex<double>> scale_chain(numOfBrackets*numOfChains);
    vector<TensorHao<complex<double>, 2>> A1ExpS_chain(numOfBrackets*numOfChains);
    // 
    if(walkerLeft.metropolisVec[0].method.BPMetroUpdateType == "global_fast"){
        ///////////////
        // Timer
        ///////////////
        auto begin = std::chrono::high_resolution_clock::now();
        ///////////////
        ////////////////////////////////////////////
        for(int bracket=1-1; bracket <= numOfBrackets*numOfChains-1; bracket++){
            walkerLeft.metropolisVec[bracket].updateOverlapMatrix_inv_fromOverlapMatrix();
        }
        ////////////////////////////////////////////
        ///////////////
        // Timer
        ///////////////
        auto updateOverlapMatrix_inv_end = std::chrono::high_resolution_clock::now();
        ///////////////
        complex<double> scaleSum = 0.0;
        // 
        TensorHao<complex<double>, 2> matrixInv_Sum; 
        matrixInv_Sum.resize(N,N); 
        matrixInv_Sum = 0.0;
        // 
        TensorHao<complex<double>, 2> matrixInvA1ExpS_Sum;
        matrixInvA1ExpS_Sum.resize(N,walkerLeft.metropolisVec[0].metropolisInfo.A1.rank(1)); 
        matrixInvA1ExpS_Sum = 0.0; 
        // 
        vector<TensorHao<complex<double>, 2> > overlapMatrixInvA1ExpS_vec;
        overlapMatrixInvA1ExpS_vec.resize(numOfBrackets*numOfChains);
        // 
        for(int bracket=1-1; bracket <= numOfBrackets*numOfChains-1; bracket++){
            scale_chain[bracket] = exp(walkerLeft.returnLogPhase_fromCurrentOverlap(bracket)-walkerLeft.returnLogTotalPhase_fromCurrentOverlap());
            scaleSum += scale_chain[bracket];
            ////////////////////////////////////////////
            matrixInv_Sum = matrixInv_Sum + walkerLeft.metropolisVec[bracket].metropolisInfo.overlapMatrix_inv * scale_chain[bracket];
            // matrixInvA1ExpD_Sum = matrixInvA1ExpD_Sum + matrixInv @ A1 @ (S + 0.5 * S @ C0 @ S);
            walkerLeft.metropolisVec[bracket].getOverlapMatrixInvA1ExpS(overlapMatrixInvA1ExpS_vec[bracket]);
            matrixInvA1ExpS_Sum = matrixInvA1ExpS_Sum + overlapMatrixInvA1ExpS_vec[bracket] * scale_chain[bracket];
        }
        /////////////////////////////////
        ///////////////
        // Timer
        ///////////////
        auto getOverlapMatrixInvA1ExpS_A1ExpS_end = std::chrono::high_resolution_clock::now();
        ///////////////
        // 
        //////////////////////////////////////////////////////////////
        // Kenergy
        //////////////////////////////////////////////////////////////
        TensorHao<complex<double>, 2> matrixK(N,N);
        // get const matrix for bracket
        TensorHao<complex<double>, 2> A0PWR, B0PWR;
        walkerLeft.metropolisVec[0].get_A0PWR_B0PWR(KP, A0PWR, B0PWR);
        // A0PWR_matrixInv_Sum
        TensorHao<complex<double>, 2> A0PWR_matrixInv_Sum(A0PWR.rank(0), matrixInv_Sum.rank(1));
        BL_NAME(gmm)(A0PWR, matrixInv_Sum, A0PWR_matrixInv_Sum);
        // 
        // matrixInvA1ExpS_Sum_B0PWR = matrixInvA1ExpS_Sum @ B0PWR
        TensorHao<complex<double>, 2> matrixInvA1ExpS_Sum_B0PWR(matrixInvA1ExpS_Sum.rank(0), B0PWR.rank(1));

        BL_NAME(gmm)(matrixInvA1ExpS_Sum, B0PWR, matrixInvA1ExpS_Sum_B0PWR);

        // matrixK = A0PWR @ matrixInv_Sum + matrixInvA1ExpS_Sum @ B0PWR;
        matrixK = A0PWR_matrixInv_Sum + matrixInvA1ExpS_Sum_B0PWR;
        // 
        for(size_t i = 0; i < N; ++i){
            Kenergy += matrixK(i,i);
        }
        ///////////////
        // Timer
        ///////////////
        auto Kenergy_end = std::chrono::high_resolution_clock::now();
        ///////////////
        //////////////////////////////////////////////////////////////
        // svdBg*svdBg + svdEx
        //////////////////////////////////////////////////////////////
        vector<TensorHao<complex<double>, 2>> A0LgammaWR_vec, B0LgammaWR_vec;
        get_A0LgammaWR_B0LgammaWR(walkerLeft.metropolisVec[0].metropolisInfo.walkerRightInBlock[0].getWf(), A0LgammaWR_vec, B0LgammaWR_vec);
        ///////////////
        // Timer
        ///////////////
        auto get_A0PWR_B0PWR_end = std::chrono::high_resolution_clock::now();
        ///////////////
        // 
        TensorHao<complex<double>, 2> matrixTemp(walkerLeft.metropolisVec[0].metropolisInfo.overlapMatrix_inv.rank(0), A0LgammaWR_vec[0].rank(1));
        TensorHao<complex<double>, 2> matrixsvdBgChain(matrixTemp[0].rank(0), walkerLeft.metropolisVec[0].metropolisInfo.overlapMatrix_inv.rank(1));
        ////////////////////////////////////////////////////////////
        for(size_t bracket=1-1; bracket <=numOfBrackets*numOfChains - 1; bracket ++ ){
            TensorHao<complex<double>, 1> svdBgChain(svdNumber); svdBgChain=0.0;
            TensorHao<complex<double>, 1> svdExChain(svdNumber); svdBgChain=0.0;
            complex<double> HenergyChain(0.0);
            // 
            for(size_t k = 0; k < svdNumber ; ++k)
            {
                ////////////////////////////////////////////////////////////
                // svdBg*svdBg + svdEX
                ////////////////////////////////////////////////////////////
                
                BL_NAME(gmm)( walkerLeft.metropolisVec[bracket].metropolisInfo.overlapMatrix_inv, A0LgammaWR_vec[k], matrixTemp);
                // 
                BL_NAME(gmm)( overlapMatrixInvA1ExpS_vec[bracket], B0LgammaWR_vec[k], matrixsvdBgChain);
                // 
                matrixsvdBgChain = matrixTemp + matrixsvdBgChain;
                // 
                for(size_t i = 0; i < N; ++i){
                    svdBgChain(k) += matrixsvdBgChain(i,i);
                }
                // svdEx
                for(size_t i = 0; i < N; ++i) { for(size_t j = 0; j < N; ++j) svdExChain(k) += matrixsvdBgChain(j,i) * matrixsvdBgChain(i,j); }
                ////////////////////////////////////////////////////////////
                // svdNormalChain --> in KPup and KPdn
                ////////////////////////////////////////////////////////////
                // 
                HenergyChain += svdBgChain(k)*svdBgChain(k) - svdExChain(k);
            }
            //
            Henergy += HenergyChain * scale_chain[bracket];
            svdBg += svdBgChain * scale_chain[bracket];
            svdEx += svdExChain * scale_chain[bracket];
        }
        ///////////////
        // Timer
        ///////////////
        auto HenergyChain_end = std::chrono::high_resolution_clock::now();
        ///////////////
        // ////////////////////////////////////////////
        auto elapsed_updateOverlapMatrix_inv_end = std::chrono::duration_cast<std::chrono::nanoseconds>(updateOverlapMatrix_inv_end - begin);
        auto elapsed_getOverlapMatrixInvA1ExpS_A1ExpS_end = std::chrono::duration_cast<std::chrono::nanoseconds>(getOverlapMatrixInvA1ExpS_A1ExpS_end - updateOverlapMatrix_inv_end);
        auto elapsed_Kenergy_end = std::chrono::duration_cast<std::chrono::nanoseconds>(Kenergy_end - getOverlapMatrixInvA1ExpS_A1ExpS_end);
        auto elapsed_get_A0PWR_B0PWR_end = std::chrono::duration_cast<std::chrono::nanoseconds>(get_A0PWR_B0PWR_end - Kenergy_end);
        auto elapsed_HenergyChain_end = std::chrono::duration_cast<std::chrono::nanoseconds>(HenergyChain_end - get_A0PWR_B0PWR_end);
        // if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
        // if(MPIRank() == 0)printf("Time measured for elapsed_updateOverlapMatrix_inv_end: %.8f seconds.\n", elapsed_updateOverlapMatrix_inv_end.count() * 1e-9);
        // if(MPIRank() == 0)printf("Time measured for elapsed_getOverlapMatrixInvA1ExpS_A1ExpS_end: %.8f seconds.\n", elapsed_getOverlapMatrixInvA1ExpS_A1ExpS_end.count() * 1e-9);
        // if(MPIRank() == 0)printf("Time measured for elapsed_Kenergy_end: %.8f seconds.\n", elapsed_Kenergy_end.count() * 1e-9);
        // if(MPIRank() == 0)printf("Time measured for elapsed_get_A0PWR_B0PWR_end: %.8f seconds.\n", elapsed_get_A0PWR_B0PWR_end.count() * 1e-9);
        // if(MPIRank() == 0)printf("Time measured for elapsed_HenergyChain_end: %.8f seconds.\n", elapsed_HenergyChain_end.count() * 1e-9);
        // if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
        // exit(1);
        //////////////////////////////////////////
        //////////////////////////////////////////////////////////////
    }
    ////////////////////////////////////////////////////////////
    Henergy *= 0.5;
    Henergy += Kenergy;

    TNum += ( Kenergy * denIncrement );
    svdBgNum += ( svdBg * denIncrement );
    svdExNum += ( svdEx * denIncrement );
    HNum += ( Henergy * denIncrement );

}



void GeneralHamiltonian_icfMeasureObserveMetroChainsSD::addEnergy_compact(MetroChainsOperation &metroChainsOperation, complex<double> denIncrement)
{
    // const MetroChains  *walkerLeft = metroChainsOperation.getWalkerLeft();
    // size_t halfL = walkerLeft->getL(); size_t Nup = walkerLeft->getNup(); size_t Ndn = walkerLeft->getNdn(); 
    // size_t numOfBrackets = walkerLeft->getNumOfBrackets();
    // size_t numOfChains = walkerLeft->getNumOfChains();
    // size_t svdNumber = generalHamiltonian_icf->getSVDNumber();
    // const TensorHao<complex<double>,3> &svdVecs = generalHamiltonian_icf->getSVDVecs();
    // const TensorHao<complex<double>,2> & K = generalHamiltonian_icf->getK();

    // if( svdBgNum.rank(0) != svdNumber ) { svdBgNum.resize(svdNumber); svdBgNum = complex<double>(0,0); }
    // if( svdNormalNum.rank(0) != svdNumber ) { svdNormalNum.resize(svdNumber); svdNormalNum = complex<double>(0,0); }
    // if( svdExNum.rank(0) != svdNumber ) { svdExNum.resize(svdNumber); svdExNum = complex<double>(0,0); }

    // complex<double> Henergy(0.0);
    // complex<double> Kenergy(0.0);
    // TensorHao<complex<double>, 1> svdBg(svdNumber);svdBg=0.0;
    // TensorHao<complex<double>, 1> svdEx(svdNumber);svdEx=0.0;
    
    // // complex<double> Kenergy=calculateKenergy_compact(metroChainsOperation);
    // TensorHao<complex<double>, 2> matrixUp_temp(Nup,Nup), matrixDn_temp(Ndn,Ndn);
    // TensorHao<complex<double>, 2> wfUpDaggerK(Nup, halfL), wfDnDaggerK(Ndn, halfL);
    // TensorHao<complex<double>, 2> wfUpDaggerSVDVec(Nup, halfL), wfDnDaggerSVDVec(Ndn, halfL);
    // TensorHao<complex<double>, 2> wfUpDaggerSVDTVec(Nup, halfL), wfDnDaggerSVDTVec(Ndn, halfL);
    // TensorHao<complex<double>, 2> wfUpDaggerSVDTVecSquare(Nup, halfL), wfDnDaggerSVDTVecSquare(Ndn, halfL);
    // TensorHao<complex<double>, 2> densityUpL(Nup, Nup), densityDnL(Ndn, Ndn);
    // TensorHao<complex<double>, 2> thetaUp(halfL,Nup), thetaDn(halfL,Ndn);
    // // 
    // for(size_t bracket=1-1; bracket <=numOfBrackets*numOfChains - 1; bracket ++ ){
    //     //
    //     const TensorHao<complex<double>, 2> &thetaUp_T =  metroChainsOperation.returnThetaUp_T(bracket);
    //     const TensorHao<complex<double>, 2> &thetaDn_T =  metroChainsOperation.returnThetaDn_T(bracket);
    //     TensorHao<complex<double>,2> wfUpTemp=walkerLeft->metroLeft[bracket].getWfUp();
    //     TensorHao<complex<double>,2> wfDnTemp=walkerLeft->metroLeft[bracket].getWfDn();
    //     thetaUp = trans(thetaUp_T); 
    //     thetaDn = trans(thetaDn_T); 
    //     ////////////////////////////////////////////////////////////
    //     // KenergyChain
    //     ////////////////////////////////////////////////////////////
    //     complex<double> KenergyChain(0,0);
    //     BL_NAME(gmm)(wfUpTemp, KPup, wfUpDaggerK, 'C');
    //     BL_NAME(gmm)(wfDnTemp, KPdn, wfDnDaggerK, 'C');
    //     for(size_t i = 0; i < halfL; ++i)
    //     {
    //         for(size_t j = 0; j < Nup; ++j) KenergyChain += wfUpDaggerK(j,i) * thetaUp_T(j,i);
    //         for(size_t j = 0; j < Ndn; ++j) KenergyChain += wfDnDaggerK(j,i) * thetaDn_T(j,i);
    //     }
    //     KenergyChain *= exp(metroChainsOperation.returnLogPhase(bracket)-metroChainsOperation.returnLogTotalPhase());
    //     Kenergy += KenergyChain;
    //     ////////////////////////////////////////////////////////////
    //     // TensorHao<complex<double>, 1> svdBgChain = calculateSVDBg(bracket, metroChainsOperation);
    //     // TensorHao<complex<double>, 1> svdExChain = calculateSVDEx(bracket, metroChainsOperation);
    //     // TensorHao<complex<double>, 1> svdNormalChain = calculateSVDNormal(bracket, metroChainsOperation);
    //     ////////////////////////////////////////////////////////////
    //     TensorHao<complex<double>, 1> svdBgChain(svdNumber); svdBgChain=0.0;
    //     TensorHao<complex<double>, 1> svdExChain(svdNumber); svdBgChain=0.0;
    //     // TensorHao<complex<double>, 1> svdNormalChain(svdNumber); svdBgChain=0.0;
    //     // 
    //     for(size_t k = 0; k < svdNumber ; ++k)
    //     {   
    //         ////////////////////////////////////////////////////////////
    //         BL_NAME(gmm)(wfUpTemp, svdVecComplexUp[k], wfUpDaggerSVDVec, 'C');
    //         BL_NAME(gmm)(wfDnTemp, svdVecComplexDn[k], wfDnDaggerSVDVec, 'C');
    //         ////////////////////////////////////////////////////////////
    //         // svdBgChain
    //         ////////////////////////////////////////////////////////////
    //         BL_NAME(gmm)(wfUpDaggerSVDVec, thetaUp, matrixUp_temp);
    //         for(size_t i = 0; i < Nup; ++i){
    //             svdBgChain(k) += matrixUp_temp(i,i);
    //         }
    //         // 
    //         BL_NAME(gmm)(wfDnDaggerSVDVec, thetaDn, matrixDn_temp);
    //         for(size_t i = 0; i < Ndn; ++i){
    //             svdBgChain(k) += matrixDn_temp(i,i);
    //         }
    //         ////////////////////////////////////////////////////////////
    //         // svdExChain
    //         ////////////////////////////////////////////////////////////
    //         BL_NAME(gmm)( wfUpDaggerSVDVec, thetaUp, densityUpL);
    //         BL_NAME(gmm)( wfDnDaggerSVDVec, thetaDn, densityDnL);
    //         for(size_t i = 0; i < Nup; ++i) { for(size_t j = 0; j < Nup; ++j) svdExChain(k) += densityUpL(j,i) * densityUpL(i,j); }
    //         for(size_t i = 0; i < Ndn; ++i) { for(size_t j = 0; j < Ndn; ++j) svdExChain(k) += densityDnL(j,i) * densityDnL(i,j); }
    //         ////////////////////////////////////////////////////////////
    //         // svdNormalChain --> in KPup and KPdn
    //         ////////////////////////////////////////////////////////////
    //         // BL_NAME(gmm)(wfUpTemp, svdVecComplexUp[k], wfUpDaggerSVDTVec, 'C', 'T');
    //         // BL_NAME(gmm)(wfUpDaggerSVDTVec, svdVecComplexUp[k], wfUpDaggerSVDTVecSquare, 'N', 'T');

    //         // BL_NAME(gmm)(wfDnTemp, svdVecComplexDn[k], wfDnDaggerSVDTVec, 'C', 'T');
    //         // BL_NAME(gmm)(wfDnDaggerSVDTVec, svdVecComplexDn[k], wfDnDaggerSVDTVecSquare, 'N', 'T');

    //         // for(size_t i = 0; i < halfL; ++i)
    //         // {
    //         //     for(size_t j = 0; j < Nup; ++j) svdNormalChain(k) += wfUpDaggerSVDTVecSquare(j,i) * thetaUp_T(j,i);
    //         //     for(size_t j = 0; j < Ndn; ++j) svdNormalChain(k) += wfDnDaggerSVDTVecSquare(j,i) * thetaDn_T(j,i);
    //         // }
    //     }
    //     ////////////////////////////////////////////////////////////
    //     // 
    //     complex<double> HenergyChain(0.0);
    //     for(size_t i = 0; i < svdNumber; ++i)
    //     {
    //         HenergyChain += ( svdBgChain(i)*svdBgChain(i) - svdExChain(i) );
    //     }
    //     Henergy += HenergyChain * exp(metroChainsOperation.returnLogPhase(bracket)-metroChainsOperation.returnLogTotalPhase());
    //     svdBg += svdBgChain*exp(metroChainsOperation.returnLogPhase(bracket)-metroChainsOperation.returnLogTotalPhase());
    //     svdEx += svdExChain*exp(metroChainsOperation.returnLogPhase(bracket)-metroChainsOperation.returnLogTotalPhase());
    // }

    // Henergy *= 0.5;
    // Henergy += Kenergy;

    // TNum += ( Kenergy * denIncrement );
    // svdBgNum += ( svdBg * denIncrement );
    // svdExNum += ( svdEx * denIncrement );
    // HNum += ( Henergy * denIncrement );
}

void GeneralHamiltonian_icfMeasureObserveMetroChainsSD::addGreen(MetroChainsOperation &metroChainsOperation, complex<double> denIncrement)
{
    // const MetroChains  *walkerLeft = metroChainsOperation.getWalkerLeft();
    // size_t numOfBrackets = walkerLeft->getNumOfBrackets();
    // size_t numOfChains = walkerLeft->getNumOfChains();
    // size_t L = generalHamiltonian_icf->getL();size_t N = walkerLeft->getN(); 
    // if( greenNum.rank(0) != L ) { greenNum.resize(L,L); greenNum = complex<double>(0,0); }

    // TensorHao<complex<double>, 2> greenChainNum(L,L);
    // for(size_t bracket=1-1; bracket <=numOfBrackets*numOfChains - 1; bracket ++ ){
    //     TensorHao<complex<double>, 2> greenChain = metroChainsOperation.returnGreenMatrix(bracket);

    //     greenChainNum += greenChain * exp(metroChainsOperation.returnLogPhase(bracket)-metroChainsOperation.returnLogTotalPhase());
    // }

    // greenNum += ( greenChainNum * denIncrement );
}


void GeneralHamiltonian_icfMeasureObserveMetroChainsSD::get_A0LgammaWR_B0LgammaWR(TensorHao<std::complex<double>, 2> WR, vector<TensorHao<std::complex<double>, 2>> & A0LgammaWR_vec, vector<TensorHao<std::complex<double>, 2>> & B0LgammaWR_vec)
{
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();
    // 
    A0LgammaWR_vec.resize(svdNumber);
    B0LgammaWR_vec.resize(svdNumber);
    
    // Create temporary TensorHao for reading from TensorHaoMPIRef
    TensorHao<complex<double>, 2> tempA0Lgamma(A0Lgamma[0].rank(0), A0Lgamma[0].rank(1));
    TensorHao<complex<double>, 2> tempB0Lgamma(B0Lgamma[0].rank(0), B0Lgamma[0].rank(1));
    
    for( int k = 0; k < svdNumber; ++k )
    {
        // Copy from TensorHaoMPIRef to temporary TensorHao
        for(size_t i = 0; i < tempA0Lgamma.rank(0); ++i)
            for(size_t j = 0; j < tempA0Lgamma.rank(1); ++j)
                tempA0Lgamma(i, j) = A0Lgamma[k](i, j);
        
        for(size_t i = 0; i < tempB0Lgamma.rank(0); ++i)
            for(size_t j = 0; j < tempB0Lgamma.rank(1); ++j)
                tempB0Lgamma(i, j) = B0Lgamma[k](i, j);
        
        // Use temporary TensorHao for gmm calculation
        TensorHao<complex<double>, 2> A0LgammaWR(tempA0Lgamma.rank(0), WR.rank(1));
        BL_NAME(gmm)(tempA0Lgamma, WR, A0LgammaWR);
        A0LgammaWR_vec[k] = A0LgammaWR;
        // 
        TensorHao<complex<double>, 2> B0LgammaWR(tempB0Lgamma.rank(0), WR.rank(1));
        BL_NAME(gmm)(tempB0Lgamma, WR, B0LgammaWR);
        B0LgammaWR_vec[k] = B0LgammaWR;
    }

}
////////////////////////////////////////////////////////////