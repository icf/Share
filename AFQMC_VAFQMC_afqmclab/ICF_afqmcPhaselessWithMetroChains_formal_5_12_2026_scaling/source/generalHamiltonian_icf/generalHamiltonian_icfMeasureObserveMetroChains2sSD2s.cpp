//
// Created by boruoshihao on 11/14/18.
//

#include <chrono>
#include "../../include/utils.h"
#include "../../include/generalHamiltonian_icf/generalHamiltonian_icfMeasureObserveMetroChains2sSD2s.h"
#include "afqmclab.h"
#include <math.h>

using namespace std;
using namespace tensor_hao;

GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s()
{
    initModelNullptr();
    reSet();
}

GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s(const GeneralHamiltonian_icf &generalHamiltonian_icf_, MetroChains2s & walkerLeft)
{
    setModel_withPhiTConst(generalHamiltonian_icf_, walkerLeft);
    reSet();
}

GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::~GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s()
{

}

void GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::initModelNullptr()
{
    generalHamiltonian_icf = nullptr;
}

void GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::setModel_withPhiTConst(const GeneralHamiltonian_icf &generalHamiltonian_icf_, MetroChains2s & walkerLeft)
{
    generalHamiltonian_icf = &generalHamiltonian_icf_;
    //
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();
    int L = generalHamiltonian_icf->getL();
    int SD2sL = generalHamiltonian_icf->getSD2sL();
    bool Hamiltonian_spin_flag = generalHamiltonian_icf->getHamiltonian_spin_flag();
    // 
    A0upLgamma.resize(svdNumber);
    A0dnLgamma.resize(svdNumber);
    B0upLgamma.resize(svdNumber);
    // B0dnLgamma.resize(svdNumber);
    // 
    const TensorHao<complex<double>,2> & K = generalHamiltonian_icf->getK();
    const TensorHaoMPIRef<complex<double>,3> & svdVecs = generalHamiltonian_icf->getSVDVecs();
    KPup.resize(SD2sL, SD2sL);
    KPdn.resize(SD2sL, SD2sL);
    // svdVecComplexUp.resize(svdNumber);
    // svdVecComplexDn.resize(svdNumber);
    TensorHao<complex<double>, 2> svdVecComplexUpTemp(SD2sL,SD2sL), svdVecComplexDnTemp(SD2sL,SD2sL);
    TensorHao<complex<double>, 2> svdTVecComplexUpSquare(SD2sL,SD2sL), svdTVecComplexDnSquare(SD2sL,SD2sL);
    svdTVecComplexUpSquare = 0.0; svdTVecComplexDnSquare = 0.0;
    TensorHao<complex<double>, 2> svdTVecComplexSquareTemp(SD2sL,SD2sL);
    TensorHao<complex<double>, 2> svdTVecComplexSquareTemp2(SD2sL,SD2sL);
    // 
    A0up = walkerLeft.metropolis2sVec[0].metropolis2sInfo.A0up;
    A0dn = walkerLeft.metropolis2sVec[0].metropolis2sInfo.A0dn;
    // 
    A1up = walkerLeft.metropolis2sVec[0].metropolis2sInfo.A1up;
    A1dn = walkerLeft.metropolis2sVec[0].metropolis2sInfo.A1dn;
    // 
    B0up = walkerLeft.metropolis2sVec[0].metropolis2sInfo.B0up;
    B0dn = walkerLeft.metropolis2sVec[0].metropolis2sInfo.B0dn;
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

    size_t dimA0up[2] = {A0up.rank(0), (size_t)svdVecComplexUpTemp.rank(1)};
    size_t dimA0dn[2] = {A0dn.rank(0), (size_t)svdVecComplexDnTemp.rank(1)};
    size_t dimB0up[2] = {B0up.rank(0), (size_t)svdVecComplexUpTemp.rank(1)};

    for(size_t k = 0; k < svdNumber ; ++k)
    {
        if(is_local_root) {
            A0upLgamma[k].createSharedMemory(dimA0up, 0, node_comm);
            A0dnLgamma[k].createSharedMemory(dimA0dn, 0, node_comm);
            B0upLgamma[k].createSharedMemory(dimB0up, 0, node_comm);
        }
        else {
            A0upLgamma[k].createSharedMemoryView(0, node_comm);
            A0upLgamma[k].attachToSharedMemory(dimA0up, 0, node_comm);
            A0dnLgamma[k].createSharedMemoryView(0, node_comm);
            A0dnLgamma[k].attachToSharedMemory(dimA0dn, 0, node_comm);
            B0upLgamma[k].createSharedMemoryView(0, node_comm);
            B0upLgamma[k].attachToSharedMemory(dimB0up, 0, node_comm);
        }
    }
    
    // Create temporary TensorHao as intermediary for gmm calculation
    TensorHao<complex<double>, 2> tempA0upLgamma(A0up.rank(0), svdVecComplexUpTemp.rank(1));
    TensorHao<complex<double>, 2> tempA0dnLgamma(A0dn.rank(0), svdVecComplexDnTemp.rank(1));
    TensorHao<complex<double>, 2> tempB0upLgamma(B0up.rank(0), svdVecComplexUpTemp.rank(1));
    
    for(size_t k = 0; k < svdNumber ; ++k)
    {
        if(Hamiltonian_spin_flag){
            for(size_t i = 0; i < SD2sL; ++i)
            {
                for(size_t j = 0; j < SD2sL; ++j) svdVecComplexUpTemp(j,i) = svdVecs(j,i,k);
            }
            for(size_t i = 0; i < SD2sL; ++i)
            {
                for(size_t j = 0; j < SD2sL; ++j) svdVecComplexDnTemp(j,i) = svdVecs(j+SD2sL,i+SD2sL,k);
            }
        }else{
            for(size_t i = 0; i < SD2sL; ++i)
            {
                for(size_t j = 0; j < SD2sL; ++j) svdVecComplexUpTemp(j,i) = svdVecs(j,i,k);
            }
            for(size_t i = 0; i < SD2sL; ++i)
            {
                for(size_t j = 0; j < SD2sL; ++j) svdVecComplexDnTemp(j,i) = svdVecs(j,i,k);
            }
        }
        // 
        BL_NAME(gmm)(svdVecComplexUpTemp, svdVecComplexUpTemp, svdTVecComplexSquareTemp, 'T', 'T');
        svdTVecComplexSquareTemp2 = svdTVecComplexUpSquare + svdTVecComplexSquareTemp;
        svdTVecComplexUpSquare = svdTVecComplexSquareTemp2;
        BL_NAME(gmm)(svdVecComplexDnTemp, svdVecComplexDnTemp, svdTVecComplexSquareTemp, 'T', 'T');
        svdTVecComplexSquareTemp2 = svdTVecComplexDnSquare + svdTVecComplexSquareTemp;
        svdTVecComplexDnSquare = svdTVecComplexSquareTemp2;
        // 
        if(is_local_root) {
            // Use TensorHao as intermediary for gmm calculation
            BL_NAME(gmm)(A0up, svdVecComplexUpTemp, tempA0upLgamma);
            BL_NAME(gmm)(A0dn, svdVecComplexDnTemp, tempA0dnLgamma);
            BL_NAME(gmm)(B0up, svdVecComplexUpTemp, tempB0upLgamma);
            
            // Copy from TensorHao to TensorHaoMPIRef
            for(size_t i = 0; i < tempA0upLgamma.rank(0); ++i)
                for(size_t j = 0; j < tempA0upLgamma.rank(1); ++j)
                    A0upLgamma[k](i, j) = tempA0upLgamma(i, j);
            
            for(size_t i = 0; i < tempA0dnLgamma.rank(0); ++i)
                for(size_t j = 0; j < tempA0dnLgamma.rank(1); ++j)
                    A0dnLgamma[k](i, j) = tempA0dnLgamma(i, j);
            
            for(size_t i = 0; i < tempB0upLgamma.rank(0); ++i)
                for(size_t j = 0; j < tempB0upLgamma.rank(1); ++j)
                    B0upLgamma[k](i, j) = tempB0upLgamma(i, j);
        }
    }
#ifdef MPI_HAO
    MPIBarrier(node_comm);
#endif
    // 
     
    if(Hamiltonian_spin_flag){
        for(int i=1-1; i<=SD2sL-1; i++){
        for(int j=1-1; j<=SD2sL-1; j++){
            KPup(i,j) = K(i,j) + 0.5 * svdTVecComplexUpSquare(i,j);
            KPdn(i,j) = K(i + SD2sL,j + SD2sL) + 0.5 * svdTVecComplexDnSquare(i,j);
        }
        }
    }else{
        for(int i=1-1; i<=SD2sL-1; i++){
        for(int j=1-1; j<=SD2sL-1; j++){
            KPup(i,j) = K(i,j) + 0.5 * svdTVecComplexUpSquare(i,j);
            KPdn(i,j) = K(i,j) + 0.5 * svdTVecComplexDnSquare(i,j);
        }
        }
    }
}

// void GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::setModel_withPhiTConst(const GeneralHamiltonian_icf &generalHamiltonian_icf_, MetroChains2s & walkerLeft)
// {
//     generalHamiltonian_icf = &generalHamiltonian_icf_;
//     //
//     size_t svdNumber = generalHamiltonian_icf->getSVDNumber();
//     int L = generalHamiltonian_icf->getL();
//     int SD2sL = generalHamiltonian_icf->getSD2sL();
//     bool Hamiltonian_spin_flag = generalHamiltonian_icf->getHamiltonian_spin_flag();
//     // 
//     A0upLgamma_BK.resize(svdNumber);
//     A0dnLgamma_BK.resize(svdNumber);
//     B0upLgamma_BK.resize(svdNumber);
//     // B0dnLgamma_BK.resize(svdNumber);
//     // 
//     const TensorHao<complex<double>,2> & K = generalHamiltonian_icf->getK();
//     const TensorHaoMPIRef<complex<double>,3> & svdVecs = generalHamiltonian_icf->getSVDVecs();
//     KPup.resize(SD2sL, SD2sL);
//     KPdn.resize(SD2sL, SD2sL);
//     // svdVecComplexUp.resize(svdNumber);
//     // svdVecComplexDn.resize(svdNumber);
//     TensorHao<complex<double>, 2> svdVecComplexUpTemp(SD2sL,SD2sL), svdVecComplexDnTemp(SD2sL,SD2sL);
//     TensorHao<complex<double>, 2> svdTVecComplexUpSquare(SD2sL,SD2sL), svdTVecComplexDnSquare(SD2sL,SD2sL);
//     svdTVecComplexUpSquare = 0.0; svdTVecComplexDnSquare = 0.0;
//     TensorHao<complex<double>, 2> svdTVecComplexSquareTemp(SD2sL,SD2sL);
//     TensorHao<complex<double>, 2> svdTVecComplexSquareTemp2(SD2sL,SD2sL);
//     // 
//     A0up = walkerLeft.metropolis2sVec[0].metropolis2sInfo.A0up;
//     A0dn = walkerLeft.metropolis2sVec[0].metropolis2sInfo.A0dn;
//     // 
//     A1up = walkerLeft.metropolis2sVec[0].metropolis2sInfo.A1up;
//     A1dn = walkerLeft.metropolis2sVec[0].metropolis2sInfo.A1dn;
//     // 
//     B0up = walkerLeft.metropolis2sVec[0].metropolis2sInfo.B0up;
//     B0dn = walkerLeft.metropolis2sVec[0].metropolis2sInfo.B0dn;
//     // 
//     for(size_t k = 0; k < svdNumber ; ++k)
//     {
//         if(Hamiltonian_spin_flag){
//             for(size_t i = 0; i < SD2sL; ++i)
//             {
//                 for(size_t j = 0; j < SD2sL; ++j) svdVecComplexUpTemp(j,i) = svdVecs(j,i,k);
//             }
//             for(size_t i = 0; i < SD2sL; ++i)
//             {
//                 for(size_t j = 0; j < SD2sL; ++j) svdVecComplexDnTemp(j,i) = svdVecs(j+SD2sL,i+SD2sL,k);
//             }
//         }else{
//             for(size_t i = 0; i < SD2sL; ++i)
//             {
//                 for(size_t j = 0; j < SD2sL; ++j) svdVecComplexUpTemp(j,i) = svdVecs(j,i,k);
//             }
//             for(size_t i = 0; i < SD2sL; ++i)
//             {
//                 for(size_t j = 0; j < SD2sL; ++j) svdVecComplexDnTemp(j,i) = svdVecs(j,i,k);
//             }
//         }
//         // svdVecComplexUp[k] = svdVecComplexUpTemp;
//         // svdVecComplexDn[k] = svdVecComplexDnTemp;
//         // 
//         BL_NAME(gmm)(svdVecComplexUpTemp, svdVecComplexUpTemp, svdTVecComplexSquareTemp, 'T', 'T');
//         svdTVecComplexSquareTemp2 = svdTVecComplexUpSquare + svdTVecComplexSquareTemp;
//         svdTVecComplexUpSquare = svdTVecComplexSquareTemp2;
//         BL_NAME(gmm)(svdVecComplexDnTemp, svdVecComplexDnTemp, svdTVecComplexSquareTemp, 'T', 'T');
//         svdTVecComplexSquareTemp2 = svdTVecComplexDnSquare + svdTVecComplexSquareTemp;
//         svdTVecComplexDnSquare = svdTVecComplexSquareTemp2;
//         // 
//         A0upLgamma_BK[k].resize(A0up.rank(0),svdVecComplexUpTemp.rank(1));
//         A0dnLgamma_BK[k].resize(A0dn.rank(0),svdVecComplexDnTemp.rank(1));
//         B0upLgamma_BK[k].resize(B0up.rank(0),svdVecComplexUpTemp.rank(1));
//         // B0dnLgamma_BK[k].resize(B0dn.rank(0),svdVecComplexDnTemp.rank(1));
//         BL_NAME(gmm)(A0up, svdVecComplexUpTemp, A0upLgamma_BK[k]);
//         BL_NAME(gmm)(A0dn, svdVecComplexDnTemp, A0dnLgamma_BK[k]);
//         BL_NAME(gmm)(B0up, svdVecComplexUpTemp, B0upLgamma_BK[k]);
//         // BL_NAME(gmm)(B0dn, svdVecComplexDnTemp, B0dnLgamma_BK[k]);
//     }
//     // 
    
//     if(Hamiltonian_spin_flag){
//         for(int i=1-1; i<=SD2sL-1; i++){
//         for(int j=1-1; j<=SD2sL-1; j++){
//             KPup(i,j) = K(i,j) + 0.5 * svdTVecComplexUpSquare(i,j);
//             KPdn(i,j) = K(i + SD2sL,j + SD2sL) + 0.5 * svdTVecComplexDnSquare(i,j);
//         }
//         }
//     }else{
//         for(int i=1-1; i<=SD2sL-1; i++){
//         for(int j=1-1; j<=SD2sL-1; j++){
//             KPup(i,j) = K(i,j) + 0.5 * svdTVecComplexUpSquare(i,j);
//             KPdn(i,j) = K(i,j) + 0.5 * svdTVecComplexDnSquare(i,j);
//         }
//         }
//     }
// }


void GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::reSet()
{
    complex<double> zero(0,0);

    den = zero;
    TNum = zero;
    svdBgNum = zero;
    svdExNum = zero;
    HNum = zero;
    // greenNum = zero;
}

complex<double> GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::returnEnergy()
{
    complex<double> Htot   = MPISum(HNum);
    complex<double> denTot = MPISum(den);
    complex<double> energy;
    if( MPIRank() == 0 ) energy = Htot/denTot;
    MPIBcast(energy);
    return energy;
}

complex<double> GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::returnKEnergy()
{
    complex<double> Ktot   = MPISum(TNum);
    complex<double> denTot = MPISum(den);
    complex<double> Kenergy;
    if( MPIRank() == 0 ) Kenergy = Ktot/denTot;
    MPIBcast(Kenergy);
    return Kenergy;
}

TensorHao<complex<double>,1> GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::returnSVDBg()
{
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();

    TensorHao<complex<double>, 1> svdBgTot( svdNumber );
    copy(svdBgNum.data(), svdBgNum.data() + svdNumber, svdBgTot.data());
    complex<double> denTot = den;

    TensorHao<complex<double>,1> svdBg( svdNumber );
    for(size_t i = 0; i < svdNumber; ++i) svdBg(i) = svdBgTot(i)/denTot;

    return svdBg;
}

TensorHao<complex<double>,1> GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::returnSVDBgReal()
{
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();

    TensorHao<complex<double>, 1> svdBgTot( svdNumber );
    copy(svdBgNum.data(), svdBgNum.data() + svdNumber, svdBgTot.data());
    complex<double> denTot = den;

    TensorHao<complex<double>,1> svdBg( svdNumber );
    for(size_t i = 0; i < svdNumber; ++i) svdBg(i) = (svdBgTot(i)/denTot).real();

    return svdBg;
}

void GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::addMeasurement(MetroChains2s & walkerLeft, complex<double> denIncrement)
{
    checkWalkerWithModel(walkerLeft);

    complex<double> denSave; denSave=den;
    den += denIncrement;
    addEnergy_fast(walkerLeft, denIncrement);
}

void GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::addMeasurement_timer(MetroChains2s & walkerLeft, complex<double> denIncrement)
{
    checkWalkerWithModel(walkerLeft);

    complex<double> denSave; denSave=den;
    den += denIncrement;
    addEnergy_fast(walkerLeft, denIncrement);
}

///////////////////////////////////////////////////////
void GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::addMeasurement(MetroChains2sOperation &metroChains2sOperation, complex<double> denIncrement)
{
    // // wfUpDaggerKList.resize(0);
    // // wfDnDaggerKList.resize(0);
    // // wfUpDaggerSVDVecsList.resize(0);
    // // wfDnDaggerSVDVecsList.resize(0);
    // // 
    // checkWalkerWithModel(metroChains2sOperation);

    // // initWfDaggerKList(metroChains2sOperation);
    // // initWfDaggerSVDVecsList(metroChains2sOperation);

    // complex<double> denSave; denSave=den;
    // den += denIncrement;
    // addEnergy_compact(metroChains2sOperation, denIncrement);

    // // wfUpDaggerKList.resize(0);
    // // wfDnDaggerKList.resize(0);
    // // wfUpDaggerSVDVecsList.resize(0);
    // // wfDnDaggerSVDVecsList.resize(0);
}

void GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::addMeasurement_timer(MetroChains2sOperation &metroChains2sOperation, complex<double> denIncrement)
{
    // // wfUpDaggerKList.resize(0);
    // // wfDnDaggerKList.resize(0);
    // // wfUpDaggerSVDVecsList.resize(0);
    // // wfDnDaggerSVDVecsList.resize(0);
    // // 
    // checkWalkerWithModel(metroChains2sOperation);

    // // initWfDaggerKList(metroChains2sOperation);
    // // initWfDaggerSVDVecsList(metroChains2sOperation);

    // complex<double> denSave; denSave=den;
    // den += denIncrement;
    // addEnergy_compact(metroChains2sOperation, denIncrement);

    // // wfUpDaggerKList.resize(0);
    // // wfDnDaggerKList.resize(0);
    // // wfUpDaggerSVDVecsList.resize(0);
    // // wfDnDaggerSVDVecsList.resize(0);
}
///////////////////////////////////////////////////////

SVDForce GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::getForce_fast(const SVD &svd,
                                                                      MetroChains2s & walkerLeft,
                                                                      double cap)
{
    checkWalkerWithModel(walkerLeft);

    size_t Nup = walkerLeft.getNup(); size_t Ndn = walkerLeft.getNdn(); 
    size_t numOfBrackets = walkerLeft.getNumOfBrackets();
    size_t numOfChains = walkerLeft.getNumOfChains();
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();
    const TensorHao<complex<double>,1> & currentBg = generalHamiltonian_icf->getSVDBg();
    complex<double> sqrtMinusDt = svd.getSqrtMinusDt();

    TensorHao<complex<double>, 1> svdBg(svdNumber); svdBg=0.0;
    //////////////////////////////////////////////////////////
    if(walkerLeft.metropolis2sVec[0].method.BPMetroUpdateType == "global_fast"){
        // 
        TensorHao<complex<double>, 2> matrixInvUp_Sum, matrixInvDn_Sum; 
        matrixInvUp_Sum.resize(Nup,Nup); matrixInvDn_Sum.resize(Ndn,Ndn);
        matrixInvUp_Sum = 0.0; matrixInvDn_Sum = 0.0;
        // 
        TensorHao<complex<double>, 2> matrixInvA1ExpSUp_Sum, matrixInvA1ExpSDn_Sum;
        matrixInvA1ExpSUp_Sum.resize(Nup,walkerLeft.metropolis2sVec[0].metropolis2sInfo.A1up.rank(1)); matrixInvA1ExpSDn_Sum.resize(Ndn,walkerLeft.metropolis2sVec[0].metropolis2sInfo.A1dn.rank(1));
        matrixInvA1ExpSUp_Sum = 0.0; matrixInvA1ExpSDn_Sum = 0.0;
        // 
        TensorHao<complex<double>, 2> overlapMatrixInvA1ExpSUp, overlapMatrixInvA1ExpSDn;
        // 
        for(int bracket=1-1; bracket <= numOfBrackets*numOfChains-1; bracket++){
            complex<double> scale = exp(walkerLeft.returnLogPhase_fromCurrentOverlap(bracket)-walkerLeft.returnLogTotalPhase_fromCurrentOverlap());
            ////////////////////////////////////////////
            // makesure overlapMatrix_inv is correct before using
            walkerLeft.metropolis2sVec[bracket].updateOverlapMatrix_inv_fromOverlapMatrix();
            ////////////////////////////////////////////
            matrixInvUp_Sum = matrixInvUp_Sum + walkerLeft.metropolis2sVec[bracket].metropolis2sInfo.overlapMatrixUp_inv * scale;
            matrixInvDn_Sum = matrixInvDn_Sum + walkerLeft.metropolis2sVec[bracket].metropolis2sInfo.overlapMatrixDn_inv * scale;
            // matrixInvA1ExpD_Sum = matrixInvA1ExpD_Sum + matrixInv @ A1 @ (S + 0.5 * S @ C0 @ S);
            walkerLeft.metropolis2sVec[bracket].getOverlapMatrixInvA1ExpS(overlapMatrixInvA1ExpSUp, overlapMatrixInvA1ExpSDn);
            matrixInvA1ExpSUp_Sum = matrixInvA1ExpSUp_Sum + overlapMatrixInvA1ExpSUp * scale;
            matrixInvA1ExpSDn_Sum = matrixInvA1ExpSDn_Sum + overlapMatrixInvA1ExpSDn * scale;
        }
        // 
        TensorHao<complex<double>, 2> matrixUp(Nup,Nup), matrixDn(Ndn,Ndn);
        vector<TensorHao<complex<double>, 2>> A0upLgammaWRup_vec, A0dnLgammaWRdn_vec, B0upLgammaWRup_vec, B0dnLgammaWRdn_vec;
        get_A0LgammaWR_B0LgammaWR(walkerLeft.metropolis2sVec[0].metropolis2sInfo.walkerRightInBlock[0].getWfUp(), walkerLeft.metropolis2sVec[0].metropolis2sInfo.walkerRightInBlock[0].getWfDn(), A0upLgammaWRup_vec, A0dnLgammaWRdn_vec, B0upLgammaWRup_vec, B0dnLgammaWRdn_vec);
        for(size_t k = 0; k < svdNumber ; ++k)
        {
            // get const matrix for bracket
            // TensorHao<complex<double>, 2> A0PWRup, A0PWRdn, B0PWRup, B0PWRdn;
            // walkerLeft.metropolis2sVec[0].get_A0PWR_B0PWR(svdVecComplexUp[k], svdVecComplexDn[k], A0PWRup, A0PWRdn, B0PWRup, B0PWRdn);
            // A0PWR_matrixInv_Sum
            TensorHao<complex<double>, 2> matrixInvUp_Sum_A0upLgammaWRup(matrixInvUp_Sum.rank(0), A0upLgammaWRup_vec[k].rank(1));
            TensorHao<complex<double>, 2> matrixInvDn_Sum_A0dnLgammaWRdn(matrixInvDn_Sum.rank(0), A0dnLgammaWRdn_vec[k].rank(1));
            BL_NAME(gmm)(A0upLgammaWRup_vec[k], matrixInvUp_Sum, matrixInvUp_Sum_A0upLgammaWRup);
            BL_NAME(gmm)(A0dnLgammaWRdn_vec[k], matrixInvDn_Sum, matrixInvDn_Sum_A0dnLgammaWRdn);
            // 
            // matrixInvA1ExpS_Sum_B0PWR = matrixInvA1ExpS_Sum @ B0LgammaWR
            TensorHao<complex<double>, 2> matrixInvA1ExpSUp_Sum_B0upLgammaWRup(matrixInvA1ExpSUp_Sum.rank(0), B0upLgammaWRup_vec[k].rank(1));
            TensorHao<complex<double>, 2> matrixInvA1ExpSDn_Sum_B0dnLgammaWRdn(matrixInvA1ExpSDn_Sum.rank(0), B0dnLgammaWRdn_vec[k].rank(1));

            BL_NAME(gmm)(matrixInvA1ExpSUp_Sum, B0upLgammaWRup_vec[k], matrixInvA1ExpSUp_Sum_B0upLgammaWRup);
            BL_NAME(gmm)(matrixInvA1ExpSDn_Sum, B0dnLgammaWRdn_vec[k], matrixInvA1ExpSDn_Sum_B0dnLgammaWRdn);
            // matrix = matrixInv_Sum @ A0LgammaWR + matrixInvA1ExpS_Sum @ B0LgammaWR;
            matrixUp = matrixInvUp_Sum_A0upLgammaWRup + matrixInvA1ExpSUp_Sum_B0upLgammaWRup;
            matrixDn = matrixInvDn_Sum_A0dnLgammaWRdn + matrixInvA1ExpSDn_Sum_B0dnLgammaWRdn;
            // 
            complex<double> diagSumUp = 0.0, diagSumDn = 0.0;
            for(size_t i = 0; i < Nup; ++i) {
                diagSumUp += matrixUp(i,i);
            }
            for(size_t i = 0; i < Ndn; ++i) {
                diagSumDn += matrixDn(i,i);
            }
            svdBg(k) = diagSumUp + diagSumDn;
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

SVDForce GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::getForce(const SVD &svd,
                                                                      MetroChains2sOperation &metroChains2sOperation,
                                                                      double cap)
{
    // checkWalkerWithModel(metroChains2sOperation);

    // const MetroChains2s  *walkerLeft = metroChains2sOperation.getWalkerLeft();
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
    //     const TensorHao<complex<double>, 2> &thetaUp_T =  metroChains2sOperation.returnThetaUp_T(bracket);
    //     const TensorHao<complex<double>, 2> &thetaDn_T =  metroChains2sOperation.returnThetaDn_T(bracket);
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
    //     svdBg += svdBgChain * exp(metroChains2sOperation.returnLogPhase(bracket)-metroChains2sOperation.returnLogTotalPhase());
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

void GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::write() const
{
    writeThreadSum(den, "den.dat", ios::app);
    writeThreadSum(TNum, "TNum.dat", ios::app);
    // writeThreadSum(svdBgNum.size(), svdBgNum.data(), "svdBgNum.dat", ios::app);
    // writeThreadSum(svdExNum.size(), svdExNum.data(), "svdExNum.dat", ios::app);
    writeThreadSum(HNum, "HNum.dat", ios::app);
    // writeThreadSum(greenNum.size(), greenNum.data(), "greenNum.dat", ios::app);
}

void GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::write(std::string postfix ) const
{
    std::string denName="den"+postfix+".dat";
    std::string HNumName="HNum"+postfix+".dat";
    writeThreadSum(den, denName, ios::app);
    writeThreadSum(HNum, HNumName, ios::app);
}

double GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::getMemory() const
{
    double mem(0.0);
    
    // Basic primitive types: den, TNum, HNum
    mem += 3 * sizeof(std::complex<double>);
    
    // Tensor objects
    mem += svdBgNum.getMemory();
    mem += svdExNum.getMemory();
    mem += svdNormalNum.getMemory();
    mem += KPup.getMemory();
    mem += KPdn.getMemory();
    
    // Vector of tensors and their contents
    // for(const auto& tensor : svdVecComplexUp) {
    //     mem += tensor.getMemory();
    // }
    // for(const auto& tensor : svdVecComplexDn) {
    //     mem += tensor.getMemory();
    // }
    
    // Additional tensor objects
    mem += A0up.getMemory();
    mem += A0dn.getMemory();
    mem += A1up.getMemory();
    mem += A1dn.getMemory();
    mem += B0up.getMemory();
    mem += B0dn.getMemory();
    
    // Vectors of tensors and their contents
    for(const auto& tensor : A0upLgamma) {
        mem += tensor.getMemory();
    }
    for(const auto& tensor : A0dnLgamma) {
        mem += tensor.getMemory();
    }
    for(const auto& tensor : B0upLgamma) {
        mem += tensor.getMemory();
    }
    // for(const auto& tensor : B0dnLgamma) {
    //     mem += tensor.getMemory();
    // }

    return mem;
}

GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s(const GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s &x)
{

}

GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s & GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::operator=(const GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s &x)
{
    return *this;
}

void GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::checkWalkerWithModel(const MetroChains2sOperation &metroChains2sOperation)
{
    const MetroChains2s *walkerLeft = metroChains2sOperation.getWalkerLeft();

    if(generalHamiltonian_icf->getHamiltonian_spin_flag() == true){
        if( generalHamiltonian_icf->getL() != 2*walkerLeft->getL() ) {cout<<"Model L does not consistent with walker left L!"<<endl; exit(1);}
    }else{
        if( generalHamiltonian_icf->getL() != walkerLeft->getL() ) {cout<<"Model L does not consistent with walker left L!"<<endl; exit(1);}
    }
    if( generalHamiltonian_icf->getNup() != walkerLeft->getNup() ) {cout<<"Model Nup does not consistent with walker left Nup!"<<endl; exit(1);}
    if( generalHamiltonian_icf->getNdn() != walkerLeft->getNdn() ) {cout<<"Model Ndn does not consistent with walker left Ndn!"<<endl; exit(1);}
}

void GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::checkWalkerWithModel(const MetroChains2s &walkerLeft)
{

    if(generalHamiltonian_icf->getHamiltonian_spin_flag() == true){
        if( generalHamiltonian_icf->getL() != 2*walkerLeft.getL() ) {cout<<"Model L does not consistent with walker left L!"<<endl; exit(1);}
    }else{
        if( generalHamiltonian_icf->getL() != walkerLeft.getL() ) {cout<<"Model L does not consistent with walker left L!"<<endl; exit(1);}
    }
    if( generalHamiltonian_icf->getNup() != walkerLeft.getNup() ) {cout<<"Model Nup does not consistent with walker left Nup!"<<endl; exit(1);}
    if( generalHamiltonian_icf->getNdn() != walkerLeft.getNdn() ) {cout<<"Model Ndn does not consistent with walker left Ndn!"<<endl; exit(1);}
}

void GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::addEnergy_fast(MetroChains2s & walkerLeft, complex<double> denIncrement)
{
    size_t halfL = walkerLeft.getL(); size_t Nup = walkerLeft.getNup(); size_t Ndn = walkerLeft.getNdn(); 
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
    vector<TensorHao<complex<double>, 2>> A1ExpSUp_chain(numOfBrackets*numOfChains), A1ExpSDn_chain(numOfBrackets*numOfChains);
    // 
    if(walkerLeft.metropolis2sVec[0].method.BPMetroUpdateType == "global_fast"){
        ///////////////
        // Timer
        ///////////////
        auto begin = std::chrono::high_resolution_clock::now();
        ///////////////
        ////////////////////////////////////////////
        for(int bracket=1-1; bracket <= numOfBrackets*numOfChains-1; bracket++){
            walkerLeft.metropolis2sVec[bracket].updateOverlapMatrix_inv_fromOverlapMatrix();
        }
        ////////////////////////////////////////////
        ///////////////
        // Timer
        ///////////////
        auto updateOverlapMatrix_inv_end = std::chrono::high_resolution_clock::now();
        ///////////////
        complex<double> scaleSum = 0.0;
        // 
        TensorHao<complex<double>, 2> matrixInvUp_Sum, matrixInvDn_Sum; 
        matrixInvUp_Sum.resize(Nup,Nup); matrixInvDn_Sum.resize(Ndn,Ndn);
        matrixInvUp_Sum = 0.0; matrixInvDn_Sum = 0.0;
        // 
        TensorHao<complex<double>, 2> matrixInvA1ExpSUp_Sum, matrixInvA1ExpSDn_Sum;
        matrixInvA1ExpSUp_Sum.resize(Nup,walkerLeft.metropolis2sVec[0].metropolis2sInfo.A1up.rank(1)); matrixInvA1ExpSDn_Sum.resize(Ndn,walkerLeft.metropolis2sVec[0].metropolis2sInfo.A1dn.rank(1));
        matrixInvA1ExpSUp_Sum = 0.0; matrixInvA1ExpSDn_Sum = 0.0;
        // 
        vector<TensorHao<complex<double>, 2> > overlapMatrixInvA1ExpSUp_vec, overlapMatrixInvA1ExpSDn_vec;
        overlapMatrixInvA1ExpSUp_vec.resize(numOfBrackets*numOfChains);
        overlapMatrixInvA1ExpSDn_vec.resize(numOfBrackets*numOfChains);
        // 
        for(int bracket=1-1; bracket <= numOfBrackets*numOfChains-1; bracket++){
            scale_chain[bracket] = exp(walkerLeft.returnLogPhase_fromCurrentOverlap(bracket)-walkerLeft.returnLogTotalPhase_fromCurrentOverlap());
            scaleSum += scale_chain[bracket];
            ////////////////////////////////////////////
            // makesure overlapMatrix_inv is correct before using
            // walkerLeft.metropolis2sVec[bracket].updateOverlapMatrix_inv_fromOverlapMatrix();
            ////////////////////////////////////////////
            matrixInvUp_Sum = matrixInvUp_Sum + walkerLeft.metropolis2sVec[bracket].metropolis2sInfo.overlapMatrixUp_inv * scale_chain[bracket];
            matrixInvDn_Sum = matrixInvDn_Sum + walkerLeft.metropolis2sVec[bracket].metropolis2sInfo.overlapMatrixDn_inv * scale_chain[bracket];
            // matrixInvA1ExpD_Sum = matrixInvA1ExpD_Sum + matrixInv @ A1 @ (S + 0.5 * S @ C0 @ S);
            walkerLeft.metropolis2sVec[bracket].getOverlapMatrixInvA1ExpS(overlapMatrixInvA1ExpSUp_vec[bracket], overlapMatrixInvA1ExpSDn_vec[bracket]);
            matrixInvA1ExpSUp_Sum = matrixInvA1ExpSUp_Sum + overlapMatrixInvA1ExpSUp_vec[bracket] * scale_chain[bracket];
            matrixInvA1ExpSDn_Sum = matrixInvA1ExpSDn_Sum + overlapMatrixInvA1ExpSDn_vec[bracket] * scale_chain[bracket];
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
        TensorHao<complex<double>, 2> matrixKUp(Nup,Nup), matrixKDn(Ndn,Ndn);
        // get const matrix for bracket
        TensorHao<complex<double>, 2> A0PWRup, A0PWRdn, B0PWRup, B0PWRdn;
        walkerLeft.metropolis2sVec[0].get_A0PWR_B0PWR(KPup, KPdn, A0PWRup, A0PWRdn, B0PWRup, B0PWRdn);
        // A0PWR_matrixInv_Sum
        TensorHao<complex<double>, 2> A0PWRup_matrixInvUp_Sum(A0PWRup.rank(0), matrixInvUp_Sum.rank(1));
        TensorHao<complex<double>, 2> A0PWRdn_matrixInvDn_Sum(A0PWRdn.rank(0), matrixInvDn_Sum.rank(1));
        BL_NAME(gmm)(A0PWRup, matrixInvUp_Sum, A0PWRup_matrixInvUp_Sum);
        BL_NAME(gmm)(A0PWRdn, matrixInvDn_Sum, A0PWRdn_matrixInvDn_Sum);
        // 
        // matrixInvA1ExpS_Sum_B0PWR = matrixInvA1ExpS_Sum @ B0PWR
        TensorHao<complex<double>, 2> matrixInvA1ExpSUp_Sum_B0PWRup(matrixInvA1ExpSUp_Sum.rank(0), B0PWRup.rank(1));
        TensorHao<complex<double>, 2> matrixInvA1ExpSDn_Sum_B0PWRdn(matrixInvA1ExpSDn_Sum.rank(0), B0PWRdn.rank(1));

        BL_NAME(gmm)(matrixInvA1ExpSUp_Sum, B0PWRup, matrixInvA1ExpSUp_Sum_B0PWRup);
        BL_NAME(gmm)(matrixInvA1ExpSDn_Sum, B0PWRdn, matrixInvA1ExpSDn_Sum_B0PWRdn);

        // matrixKUp = A0PWRup @ matrixInvUp_Sum + matrixInvA1ExpSUp_Sum @ B0PWRUp;
        matrixKUp = A0PWRup_matrixInvUp_Sum + matrixInvA1ExpSUp_Sum_B0PWRup;
        // matrixKDn = A0PWRdn @ matrixInvDn_Sum + matrixInvA1ExpSDn_Sum @ B0PWRDn;
        matrixKDn = A0PWRdn_matrixInvDn_Sum + matrixInvA1ExpSDn_Sum_B0PWRdn;
        // 
        for(size_t i = 0; i < Nup; ++i){
            Kenergy += matrixKUp(i,i);
        }
        for(size_t i = 0; i < Ndn; ++i){
            Kenergy += matrixKDn(i,i);
        }
        ///////////////
        // Timer
        ///////////////
        auto Kenergy_end = std::chrono::high_resolution_clock::now();
        ///////////////
        //////////////////////////////////////////////////////////////
        // svdBg*svdBg + svdEx
        //////////////////////////////////////////////////////////////
        vector<TensorHao<complex<double>, 2>> A0upLgammaWRup_vec, A0dnLgammaWRdn_vec, B0upLgammaWRup_vec, B0dnLgammaWRdn_vec;
        get_A0LgammaWR_B0LgammaWR(walkerLeft.metropolis2sVec[0].metropolis2sInfo.walkerRightInBlock[0].getWfUp(), walkerLeft.metropolis2sVec[0].metropolis2sInfo.walkerRightInBlock[0].getWfDn(), A0upLgammaWRup_vec, A0dnLgammaWRdn_vec, B0upLgammaWRup_vec, B0dnLgammaWRdn_vec);
        ///////////////
        // Timer
        ///////////////
        auto get_A0PWR_B0PWR_end = std::chrono::high_resolution_clock::now();
        ///////////////
        // 
        TensorHao<complex<double>, 2> matrixTempUp(walkerLeft.metropolis2sVec[0].metropolis2sInfo.overlapMatrixUp_inv.rank(0), A0upLgammaWRup_vec[0].rank(1));
        TensorHao<complex<double>, 2> matrixTempDn(walkerLeft.metropolis2sVec[0].metropolis2sInfo.overlapMatrixDn_inv.rank(0), A0dnLgammaWRdn_vec[0].rank(1));
        TensorHao<complex<double>, 2> matrixsvdBgChainUp(matrixTempUp[0].rank(0), walkerLeft.metropolis2sVec[0].metropolis2sInfo.overlapMatrixUp_inv.rank(1));
        TensorHao<complex<double>, 2> matrixsvdBgChainDn(matrixTempDn[0].rank(0), walkerLeft.metropolis2sVec[0].metropolis2sInfo.overlapMatrixDn_inv.rank(1));
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
                
                BL_NAME(gmm)( walkerLeft.metropolis2sVec[bracket].metropolis2sInfo.overlapMatrixUp_inv, A0upLgammaWRup_vec[k], matrixTempUp);
                BL_NAME(gmm)( walkerLeft.metropolis2sVec[bracket].metropolis2sInfo.overlapMatrixDn_inv, A0dnLgammaWRdn_vec[k], matrixTempDn);
                // 
                BL_NAME(gmm)( overlapMatrixInvA1ExpSUp_vec[bracket], B0upLgammaWRup_vec[k], matrixsvdBgChainUp);
                BL_NAME(gmm)( overlapMatrixInvA1ExpSDn_vec[bracket], B0dnLgammaWRdn_vec[k], matrixsvdBgChainDn);
                // 
                matrixsvdBgChainUp = matrixTempUp + matrixsvdBgChainUp;
                matrixsvdBgChainDn = matrixTempDn + matrixsvdBgChainDn;
                // 
                for(size_t i = 0; i < Nup; ++i){
                    svdBgChain(k) += matrixsvdBgChainUp(i,i);
                }
                for(size_t i = 0; i < Ndn; ++i){
                    svdBgChain(k) += matrixsvdBgChainDn(i,i);
                }
                // svdEx
                for(size_t i = 0; i < Nup; ++i) { for(size_t j = 0; j < Nup; ++j) svdExChain(k) += matrixsvdBgChainUp(j,i) * matrixsvdBgChainUp(i,j); }
                for(size_t i = 0; i < Ndn; ++i) { for(size_t j = 0; j < Ndn; ++j) svdExChain(k) += matrixsvdBgChainDn(j,i) * matrixsvdBgChainDn(i,j); }
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



void GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::addEnergy_compact(MetroChains2sOperation &metroChains2sOperation, complex<double> denIncrement)
{
    // const MetroChains2s  *walkerLeft = metroChains2sOperation.getWalkerLeft();
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
    
    // // complex<double> Kenergy=calculateKenergy_compact(metroChains2sOperation);
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
    //     const TensorHao<complex<double>, 2> &thetaUp_T =  metroChains2sOperation.returnThetaUp_T(bracket);
    //     const TensorHao<complex<double>, 2> &thetaDn_T =  metroChains2sOperation.returnThetaDn_T(bracket);
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
    //     KenergyChain *= exp(metroChains2sOperation.returnLogPhase(bracket)-metroChains2sOperation.returnLogTotalPhase());
    //     Kenergy += KenergyChain;
    //     ////////////////////////////////////////////////////////////
    //     // TensorHao<complex<double>, 1> svdBgChain = calculateSVDBg(bracket, metroChains2sOperation);
    //     // TensorHao<complex<double>, 1> svdExChain = calculateSVDEx(bracket, metroChains2sOperation);
    //     // TensorHao<complex<double>, 1> svdNormalChain = calculateSVDNormal(bracket, metroChains2sOperation);
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
    //     Henergy += HenergyChain * exp(metroChains2sOperation.returnLogPhase(bracket)-metroChains2sOperation.returnLogTotalPhase());
    //     svdBg += svdBgChain*exp(metroChains2sOperation.returnLogPhase(bracket)-metroChains2sOperation.returnLogTotalPhase());
    //     svdEx += svdExChain*exp(metroChains2sOperation.returnLogPhase(bracket)-metroChains2sOperation.returnLogTotalPhase());
    // }

    // Henergy *= 0.5;
    // Henergy += Kenergy;

    // TNum += ( Kenergy * denIncrement );
    // svdBgNum += ( svdBg * denIncrement );
    // svdExNum += ( svdEx * denIncrement );
    // HNum += ( Henergy * denIncrement );
}

void GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::addGreen(MetroChains2sOperation &metroChains2sOperation, complex<double> denIncrement)
{
    // const MetroChains2s  *walkerLeft = metroChains2sOperation.getWalkerLeft();
    // size_t numOfBrackets = walkerLeft->getNumOfBrackets();
    // size_t numOfChains = walkerLeft->getNumOfChains();
    // size_t L = generalHamiltonian_icf->getL();size_t N = walkerLeft->getN(); 
    // if( greenNum.rank(0) != L ) { greenNum.resize(L,L); greenNum = complex<double>(0,0); }

    // TensorHao<complex<double>, 2> greenChainNum(L,L);
    // for(size_t bracket=1-1; bracket <=numOfBrackets*numOfChains - 1; bracket ++ ){
    //     TensorHao<complex<double>, 2> greenChain = metroChains2sOperation.returnGreenMatrix(bracket);

    //     greenChainNum += greenChain * exp(metroChains2sOperation.returnLogPhase(bracket)-metroChains2sOperation.returnLogTotalPhase());
    // }

    // greenNum += ( greenChainNum * denIncrement );
}


void GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::get_A0LgammaWR_B0LgammaWR(TensorHao<std::complex<double>, 2> WRup, TensorHao<std::complex<double>, 2> WRdn, vector<TensorHao<std::complex<double>, 2>> & A0upLgammaWRup_vec, vector<TensorHao<std::complex<double>, 2>> & A0dnLgammaWRdn_vec, vector<TensorHao<std::complex<double>, 2>> & B0upLgammaWRup_vec, vector<TensorHao<std::complex<double>, 2>> & B0dnLgammaWRdn_vec)
{
    size_t svdNumber = generalHamiltonian_icf->getSVDNumber();
    // 
    A0upLgammaWRup_vec.resize(svdNumber);
    A0dnLgammaWRdn_vec.resize(svdNumber);
    B0upLgammaWRup_vec.resize(svdNumber);
    B0dnLgammaWRdn_vec.resize(svdNumber);
    
    // Create temporary TensorHao to read from TensorHaoMPIRef
    TensorHao<complex<double>, 2> tempA0upLgamma(A0upLgamma[0].rank(0), A0upLgamma[0].rank(1));
    TensorHao<complex<double>, 2> tempA0dnLgamma(A0dnLgamma[0].rank(0), A0dnLgamma[0].rank(1));
    TensorHao<complex<double>, 2> tempB0upLgamma(B0upLgamma[0].rank(0), B0upLgamma[0].rank(1));
    TensorHao<complex<double>, 2> tempB0dnLgamma(B0dnLgamma[0].rank(0), B0dnLgamma[0].rank(1));
    
    for( int k = 0; k < svdNumber; ++k )
    {
        // Copy from TensorHaoMPIRef to temporary TensorHao
        for(size_t i = 0; i < tempA0upLgamma.rank(0); ++i)
            for(size_t j = 0; j < tempA0upLgamma.rank(1); ++j)
                tempA0upLgamma(i, j) = A0upLgamma[k](i, j);
        
        for(size_t i = 0; i < tempA0dnLgamma.rank(0); ++i)
            for(size_t j = 0; j < tempA0dnLgamma.rank(1); ++j)
                tempA0dnLgamma(i, j) = A0dnLgamma[k](i, j);
        
        for(size_t i = 0; i < tempB0upLgamma.rank(0); ++i)
            for(size_t j = 0; j < tempB0upLgamma.rank(1); ++j)
                tempB0upLgamma(i, j) = B0upLgamma[k](i, j);
        
        for(size_t i = 0; i < tempB0dnLgamma.rank(0); ++i)
            for(size_t j = 0; j < tempB0dnLgamma.rank(1); ++j)
                tempB0dnLgamma(i, j) = B0dnLgamma[k](i, j); 
        
        // Use temporary TensorHao for gmm calculation
        TensorHao<complex<double>, 2> A0upLgammaWRup(tempA0upLgamma.rank(0), WRup.rank(1));
        TensorHao<complex<double>, 2> A0dnLgammaWRdn(tempA0dnLgamma.rank(0), WRdn.rank(1));
        BL_NAME(gmm)(tempA0upLgamma, WRup, A0upLgammaWRup);
        BL_NAME(gmm)(tempA0dnLgamma, WRdn, A0dnLgammaWRdn);
        A0upLgammaWRup_vec[k] = A0upLgammaWRup;
        A0dnLgammaWRdn_vec[k] = A0dnLgammaWRdn;
        // 
        TensorHao<complex<double>, 2> B0upLgammaWRup(tempB0upLgamma.rank(0), WRup.rank(1));
        TensorHao<complex<double>, 2> B0dnLgammaWRdn(tempB0dnLgamma.rank(0), WRdn.rank(1));
        BL_NAME(gmm)(tempB0upLgamma, WRup, B0upLgammaWRup);
        BL_NAME(gmm)(tempB0dnLgamma, WRdn, B0dnLgammaWRdn);
        B0upLgammaWRup_vec[k] = B0upLgammaWRup;
        B0dnLgammaWRdn_vec[k] = B0dnLgammaWRdn;
    }

}

// void GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::get_A0LgammaWR_B0LgammaWR(TensorHao<std::complex<double>, 2> WRup, TensorHao<std::complex<double>, 2> WRdn, vector<TensorHao<std::complex<double>, 2>> & A0upLgammaWRup_vec, vector<TensorHao<std::complex<double>, 2>> & A0dnLgammaWRdn_vec, vector<TensorHao<std::complex<double>, 2>> & B0upLgammaWRup_vec, vector<TensorHao<std::complex<double>, 2>> & B0dnLgammaWRdn_vec)
// {
//     size_t svdNumber = generalHamiltonian_icf->getSVDNumber();
//     // 
//     A0upLgammaWRup_vec.resize(svdNumber);
//     A0dnLgammaWRdn_vec.resize(svdNumber);
//     B0upLgammaWRup_vec.resize(svdNumber);
//     B0dnLgammaWRdn_vec.resize(svdNumber);
//     for( int k = 0; k < svdNumber; ++k )
//     {
//         TensorHao<complex<double>, 2> A0upLgammaWRup(A0upLgamma_BK[k].rank(0), WRup.rank(1));
//         TensorHao<complex<double>, 2> A0dnLgammaWRdn(A0dnLgamma_BK[k].rank(0), WRdn.rank(1));
//         BL_NAME(gmm)(A0upLgamma_BK[k], WRup, A0upLgammaWRup);
//         BL_NAME(gmm)(A0dnLgamma_BK[k], WRdn, A0dnLgammaWRdn);
//         A0upLgammaWRup_vec[k] = A0upLgammaWRup;
//         A0dnLgammaWRdn_vec[k] = A0dnLgammaWRdn;
//         // 
//         TensorHao<complex<double>, 2> B0upLgammaWRup(B0upLgamma_BK[k].rank(0), WRup.rank(1));
//         TensorHao<complex<double>, 2> B0dnLgammaWRdn(B0dnLgamma_BK[k].rank(0), WRdn.rank(1));
//         BL_NAME(gmm)(B0upLgamma_BK[k], WRup, B0upLgammaWRup);
//         BL_NAME(gmm)(B0dnLgamma_BK[k], WRdn, B0dnLgammaWRdn);
//         B0upLgammaWRup_vec[k] = B0upLgammaWRup;
//         B0dnLgammaWRdn_vec[k] = B0dnLgammaWRdn;
//     }

// }










// complex<double> GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::calculateKenergy(MetroChains2sOperation &metroChains2sOperation)
// {
//     const MetroChains2s  *walkerLeft = metroChains2sOperation.getWalkerLeft();

//     size_t numOfBrackets = walkerLeft->getNumOfBrackets();
//     size_t numOfChains = walkerLeft->getNumOfChains();
//     size_t halfL = walkerLeft->getL(); size_t Nup = walkerLeft->getNup(); size_t Ndn = walkerLeft->getNdn();

//     complex<double> Kenergy(0,0);
//     for(int bracket=1-1; bracket <= numOfBrackets*numOfChains-1; bracket++){
//         complex<double> KenergyChain(0,0);
//         const TensorHao<complex<double>, 2> &thetaUp_T =  metroChains2sOperation.returnThetaUp_T(bracket);
//         const TensorHao<complex<double>, 2> &thetaDn_T =  metroChains2sOperation.returnThetaDn_T(bracket);

//         for(size_t i = 0; i < halfL; ++i)
//         {
//             for(size_t j = 0; j < Nup; ++j) KenergyChain += wfUpDaggerKList[bracket](j,i) * thetaUp_T(j,i);
//             for(size_t j = 0; j < Ndn; ++j) KenergyChain += wfDnDaggerKList[bracket](j,i) * thetaDn_T(j,i);
//         }

//         KenergyChain *= exp(metroChains2sOperation.returnLogPhase(bracket)-metroChains2sOperation.returnLogTotalPhase());
//         Kenergy += KenergyChain;
//     }

//     return Kenergy;
// }

// TensorHao<complex<double>, 1> GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::calculateSVDBg(size_t bracket, MetroChains2sOperation &metroChains2sOperation)
// {
//     const MetroChains2s  *walkerLeft = metroChains2sOperation.getWalkerLeft();

//     size_t halfL = walkerLeft->getL(); size_t Nup = walkerLeft->getNup(); size_t Ndn = walkerLeft->getNdn();
//     size_t svdNumber = generalHamiltonian_icf->getSVDNumber();

//     //
//     TensorHao<complex<double>, 1> svdBg(svdNumber); svdBg=0.0;
//     //
//     const TensorHao<complex<double>, 2> &thetaUp_T =  metroChains2sOperation.returnThetaUp_T(bracket);
//     const TensorHao<complex<double>, 2> &thetaDn_T =  metroChains2sOperation.returnThetaDn_T(bracket);
//     const TensorHao<complex<double>,3> & svdVecs = generalHamiltonian_icf->getSVDVecs();
//     TensorHao<complex<double>, 2> thetaUp(halfL,Nup), thetaDn(halfL,Ndn);
//     thetaUp = trans(thetaUp_T); 
//     thetaDn = trans(thetaDn_T); 
//     //
//     for(size_t k = 0; k < svdNumber ; ++k)
//     {
//         TensorHao<complex<double>, 2> matrixUp_temp(Nup,Nup);
//         BL_NAME(gmm)(wfUpDaggerSVDVecsList[bracket][k], thetaUp, matrixUp_temp);
//         for(size_t i = 0; i < Nup; ++i){
//             svdBg(k) += matrixUp_temp(i,i);
//         }

//         TensorHao<complex<double>, 2> matrixDn_temp(Ndn,Ndn);
//         BL_NAME(gmm)(wfDnDaggerSVDVecsList[bracket][k], thetaDn, matrixDn_temp);
//         for(size_t i = 0; i < Ndn; ++i){
//             svdBg(k) += matrixDn_temp(i,i);
//         }
//     }
//     //
//     return svdBg;
// }


// TensorHao<complex<double>, 1> GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::calculateSVDNormal(size_t bracket, MetroChains2sOperation &metroChains2sOperation)
// {
//     const MetroChains2s  *walkerLeft = metroChains2sOperation.getWalkerLeft();

//     size_t halfL = walkerLeft->getL(); size_t Nup = walkerLeft->getNup(); size_t Ndn = walkerLeft->getNdn();
//     size_t svdNumber = generalHamiltonian_icf->getSVDNumber();

//     TensorHao<complex<double>, 1> svdNormal(svdNumber); svdNormal=0.0;
//     //
//     const TensorHao<complex<double>,3> & svdVecs = generalHamiltonian_icf->getSVDVecs();
//     const TensorHao<complex<double>, 2> &thetaUp_T =  metroChains2sOperation.returnThetaUp_T(bracket);
//     const TensorHao<complex<double>, 2> &thetaDn_T =  metroChains2sOperation.returnThetaDn_T(bracket);

//     // TensorHao<complex<double>, 2> wfUpDaggerSVDTVec(Nup,halfL);
//     // TensorHao<complex<double>, 2> wfDnDaggerSVDTVec(Ndn,halfL);
//     TensorHao<complex<double>, 2> wfUpDaggerSVDTVecSquare(Nup,halfL);
//     TensorHao<complex<double>, 2> wfDnDaggerSVDTVecSquare(Ndn,halfL);
//     for(size_t k = 0; k < svdNumber ; ++k)
//     {
//         TensorHao<complex<double>,2> wfUpTemp=walkerLeft->metroLeft[bracket].getWfUp();
//         // BL_NAME(gmm)(wfUpTemp, svdTVecComplexUp, wfUpDaggerSVDTVec, 'C');
//         // BL_NAME(gmm)(wfUpDaggerSVDTVec, svdTVecComplexUp, wfUpDaggerSVDTVecSquare);
//         BL_NAME(gmm)(wfUpTemp, svdTVecComplexUpSquare[k], wfUpDaggerSVDTVecSquare, 'C');

//         TensorHao<complex<double>,2> wfDnTemp=walkerLeft->metroLeft[bracket].getWfDn();
//         // BL_NAME(gmm)(wfDnTemp, svdTVecComplexDn, wfDnDaggerSVDTVec, 'C');
//         // BL_NAME(gmm)(wfDnDaggerSVDTVec, svdTVecComplexDn, wfDnDaggerSVDTVecSquare);
//         BL_NAME(gmm)(wfDnTemp, svdTVecComplexDnSquare[k], wfDnDaggerSVDTVecSquare, 'C');

//         for(size_t i = 0; i < halfL; ++i)
//         {
//             for(size_t j = 0; j < Nup; ++j) svdNormal(k) += wfUpDaggerSVDTVecSquare(j,i) * thetaUp_T(j,i);
//             for(size_t j = 0; j < Ndn; ++j) svdNormal(k) += wfDnDaggerSVDTVecSquare(j,i) * thetaDn_T(j,i);
//         }
//     }
//     //

//     return svdNormal;
// }

// TensorHao<complex<double>, 1> GeneralHamiltonian_icfMeasureObserveMetroChains2sSD2s::calculateSVDEx(size_t bracket, MetroChains2sOperation &metroChains2sOperation)
// {
//     const MetroChains2s  *walkerLeft = metroChains2sOperation.getWalkerLeft();

//     size_t halfL = walkerLeft->getL(); size_t Nup = walkerLeft->getNup(); size_t Ndn = walkerLeft->getNdn(); 
//     size_t svdNumber = generalHamiltonian_icf->getSVDNumber();

//     TensorHao<complex<double>, 1> svdEx(svdNumber); svdEx=0.0;
//     //
//     const TensorHao<complex<double>, 2> &thetaUp_T =  metroChains2sOperation.returnThetaUp_T(bracket);
//     const TensorHao<complex<double>, 2> &thetaDn_T =  metroChains2sOperation.returnThetaDn_T(bracket);
//     const TensorHao<complex<double>,3> & svdVecs = generalHamiltonian_icf->getSVDVecs();
//     //
//     TensorHao<complex<double>, 2> densityUpL(Nup, Nup), densityDnL(Ndn, Ndn);
//     TensorHao<complex<double>, 2> thetaUp, thetaDn;
//     thetaUp = trans(thetaUp_T); 
//     thetaDn = trans(thetaDn_T); 
//     for(size_t k = 0; k < svdNumber ; ++k)
//     {
//         BL_NAME(gmm)( wfUpDaggerSVDVecsList[bracket][k], thetaUp, densityUpL);

//         BL_NAME(gmm)( wfDnDaggerSVDVecsList[bracket][k], thetaDn, densityDnL);

//         for(size_t i = 0; i < Nup; ++i) { for(size_t j = 0; j < Nup; ++j) svdEx(k) += densityUpL(j

////////////////////////////////////////////////////////////