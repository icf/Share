//
// Created by boruoshihao on 12/25/16.
// Modified by Icf on 2019-9-29
//

#ifndef AFQMCLAB_METROCHAINS2S_H
#define AFQMCLAB_METROCHAINS2S_H

#include "afqmclab.h"
#include "metroChains2sDefine.h"

//Single Determinant.

#ifdef MPI_HAO
class MetroChains2s;
void MPIBcast(MetroChains2s &buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
//MetroChains2s MPIAllgather(MetroChains2s &buffer, int root=0, const MPI_Comm& comm=MPI_COMM_WORLD);
#endif

class MetroChains2s
{
 private:
    bool isInitialMetroChains;
    size_t numOfChains;
    //std::vector< MetroChains2sMetropolis > metropolis2sVec;
    //std::vector< MetroChains2sMetropolisInfo > metropolisInfoVec;
    size_t numOfBrackets;
    // std::vector< MetroChains2sWalker > metroLeft;
    // std::vector< MetroChains2sWalker > metroRight;
    std::complex<double> logw;
    //
    size_t numOfReleasedSlice;
    // std::vector< MetroChains2sWalker > MetroChainsWalkerVec;
    // std::vector< tensor_hao::TensorHao<std::complex<double>, 1> > twoBodyForceVec;
    // std::vector< tensor_hao::TensorHao<std::complex<double>, 1> > twoBodyAuxVec;

 public:
    std::vector< MetroChains2sMetropolis > metropolis2sVec;
    std::vector< MetroChains2sWalker > metroLeft;
    std::vector< MetroChains2sWalker > metroRight;
    //For MPI optimization, which won't be distributed in MPI
    std::vector<OneBody_Jastrow2s> expMinusDtK_Jastrow_vec;
    std::vector<TwoBody_Jastrow2s> expMinusDtV_Jastrow_vec;
    // For release
    // std::vector< MetroChains2sWalker > walkerRightVec;
    // std::vector< tensor_hao::TensorHao<std::complex<double>, 1> > twoBodyForceVec;
    // std::vector< tensor_hao::TensorHao<double, 1> > twoBodyForceVec_BP;
    // std::vector< tensor_hao::TensorHao<std::complex<double>, 1> > twoBodyAuxVec;
    // std::vector< tensor_hao::TensorHao<int, 1> > twoBodyAuxVec_BP;

    MetroChains2s();
    MetroChains2s(const MetroChains2s& x);
    MetroChains2s(MetroChains2s&& x);
    ~MetroChains2s();

    MetroChains2s & operator  = (const MetroChains2s& x);
    MetroChains2s & operator  = (MetroChains2s&& x);

    MetroChains2sWalker getMetroRight(size_t n);
    size_t getNumOfChains() const;
    size_t getNumOfBrackets() const;
    size_t getBPMetroTimesliceBlockSize(size_t n) const;
    size_t getBPMetroTimesliceBlockSize() const;

    size_t getL() const;
    size_t getN() const;
    size_t getNup() const;
    size_t getNdn() const;

    const std::complex<double> getLogw() const;
    std::complex<double> &logwRef();

    int returnNbuf() const;
    double getMemory() const;

    void initialMetroChains(int L, int Nup, int Ndn, size_t numOfReleasedSlice_input, size_t numOfChains_input, size_t numOfBrackets_input, JastrowProjector2s &jastrowProjector_);
    void MetroChainsInitialField(MetroChains2sWalker &MetroChainsLeftInitial, MetroChains2sWalker &MetroChainsWalkerInitial);
    void MetroChainsInitialField_readAuxFields(MetroChains2sWalker &MetroChainsWalkerLeftInitial, MetroChains2sWalker &MetroChainsWalkerRightInitial);
    //
    void MetroChainsTwoJastrowInitialField_readAuxFields(multDET2s &multDetInitial, MetroChains2sWalker &MetroChainsWalkerInitial);
    //
    void initialMetroChainsTwoJastrow(int L, int Nup, int Ndn, size_t numOfReleasedSlice_input, size_t numOfChains_input, size_t numOfBrackets_input, JastrowProjector2s &jastrowProjector_);
    void MetroChainsTwoJastrowInitialField(multDET2s &multDetInitial, MetroChains2sWalker &MetroChainsWalkerInitial);

    //
    void copyMetroChains_FromMetroChainsTwoJastrow(int L, int Nup, int Ndn, size_t numOfReleasedSlice_input, size_t numOfChains_input, size_t numOfBrackets_input, MetroChains2s phiT_twoJastrow_input, JastrowProjector2s &jastrowProjector_);
    void updateMetroChainsWithNewCket_MetroAuxUpdate_prop_withSave(MetroChains2sWalker MetroChainsWalkerInitial, int numOfThermalSweeps, std::complex<double> phase_dy, MetroChains2sWalker MetroChainsWalker_lastStep, tensor_hao::TensorHao<std::complex<double>, 1> dynamicForce, tensor_hao::TensorHao<std::complex<double>, 1> twoBodyAux, std::vector<int> flip_i_vec);
    void updateMetroChainsWithNewCket_MetroAuxUpdate(int numOfThermalSweeps, std::vector<std::vector<int>> flip_i_vec);
    void updateMetroChains(int chain, int numOfSweeps, std::vector<int> flip_i_vec, bool ifFrozen);
    void updateMetroChains_allChains(int numOfSweeps, std::vector<int> flip_i_vec, bool ifFrozen);
    void addAndPopBracket(int chain, int n_left, int n_right);
    void addAndPopBracket_1Slice_walkerRightMetro(int chain);

    //
    void setJastrowProjector(JastrowProjector2s &jastrowProjector_);
    void extendMetroChainInfoForRelease();
    void updateMetroChains_allChainsAndAddAndPopBracketForRelease(size_t numOfBrackets_input);
    void updateMetroChains_allChainsAndAddAndPopBracket(size_t numOfBrackets_input);

    void stabilize();

    std::complex<double> returnLogPhase_fromCurrentOverlap(size_t chain);
    std::complex<double> returnLogTotalPhase_fromCurrentOverlap();

#ifdef MPI_HAO
    friend void MPIBcast(MetroChains2s &buffer, int root,  const MPI_Comm& comm);
    void pack( std::vector<char> &buf,  int &posit ) const;
    void unpack( const std::vector<char> &buf, int &posit );
#endif

 private:
    void copy_deep(const MetroChains2s &x);
    void move_deep(MetroChains2s &x);
};

#endif //AFQMCLAB_METROCHAINS2S_H
