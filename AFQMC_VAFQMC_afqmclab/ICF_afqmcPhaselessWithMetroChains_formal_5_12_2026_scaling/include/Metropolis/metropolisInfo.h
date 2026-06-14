//
// Created by boruoshihao on 1/15/17.
//

#ifndef AFQMCLAB_METROPOLISINFO_H
#define AFQMCLAB_METROPOLISINFO_H

#include "afqmclab.h"
#include "metropolisDefine.h"
#include "JastrowProjector.h"

#ifdef MPI_HAO
class MetropolisInfo;
void MPIBcast(MetropolisInfo &buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
MetropolisInfo MPIAllgather(MetropolisInfo &buffer, int root=0, const MPI_Comm& comm=MPI_COMM_WORLD);
#endif

class MetropolisInfo
{
 public:
    //MetropolisInfo + method + model can reproduce exactly the same Metropolis2s  
    //Distributd by MPI
    int BPMetroTimesliceBlockSize;
    //
    int inBlockIndex;
    std::complex<double> currentLogOverlap;
    //
    std::vector<TwoBodyAux_Jastrow> auxiliaryFields;
    // std::vector<TwoBodyAux_BP_Jastrow2s> auxiliaryFields_BP;
    std::vector<TwoBodyForce_Jastrow> dynamicForceFields;
    // std::vector<TwoBodyForce_BP_Jastrow2s> dynamicForceFields_BP;
    //
    std::vector<Walker> walkerRightInBlock;
    std::vector<Walker> walkerLeftInBlock;
    std::vector<std::complex<double>> logWeightRightInBlock;
    std::vector<std::complex<double>> logWeightLeftInBlock;
    //
    tensor_hao::TensorHao<std::complex<double>,2> overlapMatrix_inv;
    tensor_hao::TensorHao<std::complex<double>,2> B;
    /////////////////////////////////////////
    int truncatedD;
    tensor_hao::TensorHao<std::complex<double>,2> overlapMatrix;
    tensor_hao::TensorHao<std::complex<double>,2> A0;
    tensor_hao::TensorHao<std::complex<double>,2> A0WR;
    tensor_hao::TensorHao<std::complex<double>,2> A1;
    tensor_hao::TensorHao<std::complex<double>,2> B0;
    tensor_hao::TensorHao<std::complex<double>,2> B0WR;
    tensor_hao::TensorHao<std::complex<double>,2> C0;
    tensor_hao::TensorHao<std::complex<double>,2> D;
    bool globalFastInitialized;
    bool globalFastUpdated;
    /////////////////////////////////////////
    // Jastrow projector related data
    std::vector<std::string> variableName_vec;
    // 
    std::vector<TwoBodyForce_Jastrow>  constForce_Jastrow;
    std::vector<std::string> KVorder;
    //
    int numOfJastrow;
    std::vector<int> JastrowSlice;
    std::vector<int> JastrowExpM;
    /////////////////////////////////////////

    //
    MetropolisInfo();
    MetropolisInfo(const MetropolisInfo& x);
    MetropolisInfo(MetropolisInfo&& x);
    ~MetropolisInfo();

    MetropolisInfo & operator  = (const MetropolisInfo& x);
    MetropolisInfo & operator  = (MetropolisInfo&& x);

    void initial_JastrowProjectorRelated(JastrowProjector &jastrowProjector_);
    void initial_localUpdate(int L, int N);
    void initial_globalFastUpdate(int L, int N, int truncatedD);
    void readTwoJastrowAuxiliaryFields(int i);
    void readAuxiliaryFields(int i);
    void takeLeftHalf();
    void extendMetroChainToRight( MetropolisInfo metropolisInfo_input);

    int returnNbuf() const;
    double getMemory() const;
#ifdef MPI_HAO
    friend void MPIBcast(MetropolisInfo &buffer, int root,  const MPI_Comm& comm);
    void pack( std::vector<char> &buf,  int &posit ) const;
    void unpack( const std::vector<char> &buf, int &posit );
#endif

 private:
    void copy_deep(const MetropolisInfo &x);
    void move_deep(MetropolisInfo &x);
};

#endif //AFQMCLAB_METROPOLISINFO_H