//
// Created by boruoshihao on 1/15/17.
//

#ifndef AFQMCLAB_METROPOLIS2SINFO_H
#define AFQMCLAB_METROPOLIS2SINFO_H

#include "afqmclab.h"
#include "metropolis2sDefine.h"
#include "JastrowProjector2s.h"

#ifdef MPI_HAO
class Metropolis2sInfo;
void MPIBcast(Metropolis2sInfo &buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
Metropolis2sInfo MPIAllgather(Metropolis2sInfo &buffer, int root=0, const MPI_Comm& comm=MPI_COMM_WORLD);
#endif

class Metropolis2sInfo
{
 public:
    //Metropolis2sInfo + method + model can reproduce exactly the same Metropolis2s  
    //Distributd by MPI
    int BPMetroTimesliceBlockSize;
    //
    int inBlockIndex;
    std::complex<double> currentLogOverlap;
    //
    std::vector<TwoBodyAux_Jastrow2s> auxiliaryFields;
    // std::vector<TwoBodyAux_BP_Jastrow2s> auxiliaryFields_BP;
    std::vector<TwoBodyForce_Jastrow2s> dynamicForceFields;
    // std::vector<TwoBodyForce_BP_Jastrow2s> dynamicForceFields_BP;
    //
    std::vector<Walker2s> walkerRightInBlock;
    std::vector<Walker2s> walkerLeftInBlock;
    std::vector<std::complex<double>> logWeightRightInBlock;
    std::vector<std::complex<double>> logWeightLeftInBlock;
    //
    tensor_hao::TensorHao<std::complex<double>,2> overlapMatrixUp_inv;
    tensor_hao::TensorHao<std::complex<double>,2> overlapMatrixDn_inv;
    tensor_hao::TensorHao<std::complex<double>,2> Bup;
    tensor_hao::TensorHao<std::complex<double>,2> Bdn;
    /////////////////////////////////////////
    int truncatedDup;
    int truncatedDdn;
    tensor_hao::TensorHao<std::complex<double>,2> overlapMatrixUp;
    tensor_hao::TensorHao<std::complex<double>,2> overlapMatrixDn;
    tensor_hao::TensorHao<std::complex<double>,2> A0up, A0dn;
    tensor_hao::TensorHao<std::complex<double>,2> A0WRup, A0WRdn;
    tensor_hao::TensorHao<std::complex<double>,2> A1up, A1dn;
    tensor_hao::TensorHao<std::complex<double>,2> B0up;
    const tensor_hao::TensorHao<std::complex<double>,2> & B0dn = B0up;
    tensor_hao::TensorHao<std::complex<double>,2> B0WRup, B0WRdn;
    tensor_hao::TensorHao<std::complex<double>,2> C0up;
    const tensor_hao::TensorHao<std::complex<double>,2> & C0dn = C0up;
    tensor_hao::TensorHao<std::complex<double>,2> Dup;
    const tensor_hao::TensorHao<std::complex<double>,2> & Ddn = Dup;
    bool globalFastInitialized;
    bool globalFastUpdated;
    /////////////////////////////////////////
    // Jastrow projector related data
    std::vector<std::string> variableName_vec;
    // 
    std::vector<TwoBodyForce_Jastrow2s>  constForce_Jastrow;
    std::vector<std::string> KVorder;
    //
    int numOfJastrow;
    std::vector<int> JastrowSlice;
    std::vector<int> JastrowExpM;
    /////////////////////////////////////////

    //
    Metropolis2sInfo();
    Metropolis2sInfo(const Metropolis2sInfo& x);
    Metropolis2sInfo(Metropolis2sInfo&& x);
    ~Metropolis2sInfo();

    Metropolis2sInfo & operator  = (const Metropolis2sInfo& x);
    Metropolis2sInfo & operator  = (Metropolis2sInfo&& x);

    void initial_JastrowProjectorRelated(JastrowProjector2s &jastrowProjector_);
    void initial_localUpdate(int L, int Nup, int Ndn);
    void initial_globalFastUpdate(int L, int Nup, int Ndn, int truncatedDup, int truncatedDdn);
    void readTwoJastrowAuxiliaryFields(int i);
    void readAuxiliaryFields(int i);
    void takeLeftHalf();
    void extendMetroChainToRight( Metropolis2sInfo metropolisInfo_input);

    int returnNbuf() const;
    double getMemory() const;
#ifdef MPI_HAO
    friend void MPIBcast(Metropolis2sInfo &buffer, int root,  const MPI_Comm& comm);
    void pack( std::vector<char> &buf,  int &posit ) const;
    void unpack( const std::vector<char> &buf, int &posit );
#endif

 private:
    void copy_deep(const Metropolis2sInfo &x);
    void move_deep(Metropolis2sInfo &x);
};

#endif //AFQMCLAB_METROPOLIS2SINFO_H