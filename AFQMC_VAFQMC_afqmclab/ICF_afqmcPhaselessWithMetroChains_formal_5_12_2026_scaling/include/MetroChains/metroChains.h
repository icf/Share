//
// Created by boruoshihao on 12/25/16.
// Modified by Icf on 2019-9-29
//

#ifndef AFQMCLAB_METROCHAINS_H
#define AFQMCLAB_METROCHAINS_H

#include "afqmclab.h"
#include "metroChainsDefine.h"

//Single Determinant.

#ifdef MPI_HAO
class MetroChains;
void MPIBcast(MetroChains &buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
//MetroChains MPIAllgather(MetroChains &buffer, int root=0, const MPI_Comm& comm=MPI_COMM_WORLD);
#endif

class MetroChains
{
 private:
    bool isInitialMetroChains;
    size_t numOfChains;
    //std::vector< MetroChainsMetropolis > metropolisVec;
    //std::vector< MetroChainsMetropolisInfo > metropolisInfoVec;
    size_t numOfBrackets;
    // std::vector< MetroChainsWalker > metroLeft;
    // std::vector< MetroChainsWalker > metroRight;
    std::complex<double> logw;
    //
    size_t numOfReleasedSlice;
    // std::vector< MetroChainsWalker > MetroChainsWalkerVec;
    // std::vector< tensor_hao::TensorHao<std::complex<double>, 1> > twoBodyForceVec;
    // std::vector< tensor_hao::TensorHao<std::complex<double>, 1> > twoBodyAuxVec;

 public:
    std::vector< MetroChainsMetropolis > metropolisVec;
    std::vector< MetroChainsWalker > metroLeft;
    std::vector< MetroChainsWalker > metroRight;
    //For MPI optimization, which won't be distributed in MPI
    std::vector<OneBody_Jastrow> expMinusDtK_Jastrow_vec;
    std::vector<TwoBody_Jastrow> expMinusDtV_Jastrow_vec;
    // For release
    // std::vector< MetroChainsWalker > walkerRightVec;
    // std::vector< tensor_hao::TensorHao<std::complex<double>, 1> > twoBodyForceVec;
    // std::vector< tensor_hao::TensorHao<double, 1> > twoBodyForceVec_BP;
    // std::vector< tensor_hao::TensorHao<std::complex<double>, 1> > twoBodyAuxVec;
    // std::vector< tensor_hao::TensorHao<int, 1> > twoBodyAuxVec_BP;

    MetroChains();
    MetroChains(const MetroChains& x);
    MetroChains(MetroChains&& x);
    ~MetroChains();

    MetroChains & operator  = (const MetroChains& x);
    MetroChains & operator  = (MetroChains&& x);

    MetroChainsWalker getMetroRight(size_t n);
    size_t getNumOfChains() const;
    size_t getNumOfBrackets() const;
    size_t getBPMetroTimesliceBlockSize(size_t n) const;
    size_t getBPMetroTimesliceBlockSize() const;

    size_t getL() const;
    size_t getN() const;

    const std::complex<double> getLogw() const;
    std::complex<double> &logwRef();

    int returnNbuf() const;
    double getMemory() const;

    void initialMetroChains(int L, int N, size_t numOfReleasedSlice_input, size_t numOfChains_input, size_t numOfBrackets_input, JastrowProjector &jastrowProjector_);
    void MetroChainsInitialField(MetroChainsWalker &MetroChainsLeftInitial, MetroChainsWalker &MetroChainsWalkerInitial);
    void MetroChainsInitialField_readAuxFields(MetroChainsWalker &MetroChainsWalkerLeftInitial, MetroChainsWalker &MetroChainsWalkerRightInitial);
    //
    void MetroChainsTwoJastrowInitialField_readAuxFields(multDET &multDetInitial, MetroChainsWalker &MetroChainsWalkerInitial);
    //
    void initialMetroChainsTwoJastrow(int L, int N, size_t numOfReleasedSlice_input, size_t numOfChains_input, size_t numOfBrackets_input, JastrowProjector &jastrowProjector_);
    void MetroChainsTwoJastrowInitialField(multDET &multDetInitial, MetroChainsWalker &MetroChainsWalkerInitial);

    //
    void copyMetroChains_FromMetroChainsTwoJastrow(int L, int N, size_t numOfReleasedSlice_input, size_t numOfChains_input, size_t numOfBrackets_input, MetroChains phiT_twoJastrow_input, JastrowProjector &jastrowProjector_);
    void updateMetroChainsWithNewCket_MetroAuxUpdate_prop_withSave(MetroChainsWalker MetroChainsWalkerInitial, int numOfThermalSweeps, std::complex<double> phase_dy, MetroChainsWalker MetroChainsWalker_lastStep, tensor_hao::TensorHao<std::complex<double>, 1> dynamicForce, tensor_hao::TensorHao<std::complex<double>, 1> twoBodyAux, std::vector<int> flip_i_vec);
    void updateMetroChainsWithNewCket_MetroAuxUpdate(int numOfThermalSweeps, std::vector<std::vector<int>> flip_i_vec);
    void updateMetroChains(int chain, int numOfSweeps, std::vector<int> flip_i_vec, bool ifFrozen);
    void updateMetroChains_allChains(int numOfSweeps, std::vector<int> flip_i_vec, bool ifFrozen);
    void addAndPopBracket(int chain, int n_left, int n_right);
    void addAndPopBracket_1Slice_walkerRightMetro(int chain);

    //
    void setJastrowProjector(JastrowProjector &jastrowProjector_);
    void extendMetroChainInfoForRelease();
    void updateMetroChains_allChainsAndAddAndPopBracketForRelease(size_t numOfBrackets_input);
    void updateMetroChains_allChainsAndAddAndPopBracket(size_t numOfBrackets_input);

    void stabilize();

    std::complex<double> returnLogPhase_fromCurrentOverlap(size_t chain);
    std::complex<double> returnLogTotalPhase_fromCurrentOverlap();

#ifdef MPI_HAO
    friend void MPIBcast(MetroChains &buffer, int root,  const MPI_Comm& comm);
    void pack( std::vector<char> &buf,  int &posit ) const;
    void unpack( const std::vector<char> &buf, int &posit );
#endif

 private:
    void copy_deep(const MetroChains &x);
    void move_deep(MetroChains &x);
};

#endif //AFQMCLAB_METROCHAINS_H
