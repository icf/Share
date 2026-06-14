//
// Created by boruoshihao on 1/15/17.
//

#ifndef AFQMCLAB_METROPOLIS_H
#define AFQMCLAB_METROPOLIS_H

#include "metropolisDefine.h"
#include "metropolisMethod.h"
#include "JastrowProjector.h"
#include "metropolisInfo.h"

#ifdef MPI_HAO
class Metropolis;
void MPIBcast(Metropolis &buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
#endif

class Metropolis
{
public:
    ////////////////////////////////////////////////////////////
    //objects do depends on sampling  -- > MPI distributed
    ////////////////////////////////////////////////////////////
    //MetropolisInfo is the only object that shared and distributd by MPI
    MetropolisInfo metropolisInfo;
    ////////////////////////////////////////////////////////////
    //objects do not depends on sampling -- > MPI not distributed
    ////////////////////////////////////////////////////////////
    JastrowProjector *jastrowProjector;
    JastrowProjector *jastrowProjectorExtended;
    
    std::vector<OneBody_Jastrow> *expMinusDtK_Jastrow_vec;
    std::vector<TwoBody_Jastrow> *expMinusDtV_Jastrow_vec;
    //
    MetropolisMethod method;
    ////////////////////////////////////////////////////////////
    //No need to be distributed but restore after popcontrol
    HopWalkerOperation oneBodyWalkerRightOperation;
    HopWalkerOperation oneBodyWalkerLeftOperation;
    SVDSampleWalkerOperation twoBodySampleWalkerRightOperation;
    SVDSampleWalkerOperation twoBodySampleWalkerLeftOperation;

    Metropolis();
    Metropolis(const Metropolis& x);
    Metropolis(Metropolis &&x);
    ~Metropolis();

    Metropolis & operator  = (const Metropolis& x);
    Metropolis & operator  = (Metropolis&& x);

    void initJastrowProjectorNullptr();
    void setJastrowProjector(JastrowProjector &jastrowProjector_);
    void setJastrowProjectorExtended(JastrowProjector &jastrowProjectorExtended_);
    void setExpMinusDt_KV_Jastrow_vec(std::vector<OneBody_Jastrow> &expMinusDtK_Jastrow_vec_, std::vector<TwoBody_Jastrow> &expMinusDtV_Jastrow_vec_);
    void initialParameters(int L, int N, JastrowProjector &jastrowProjector_, std::vector<OneBody_Jastrow> &expMinusDtK_Jastrow_vec_, std::vector<TwoBody_Jastrow> &expMinusDtV_Jastrow_vec_);
    void initialParametersTwoJastrow(int L, int N, JastrowProjector &jastrowProjector_, std::vector<OneBody_Jastrow> &expMinusDtK_Jastrow_vec_, std::vector<TwoBody_Jastrow> &expMinusDtV_Jastrow_vec_);

    void extendMetroChainInfoToRight(MetropolisInfo metropolisInfo_input);
    void initialField(Walker &walkerLeft, Walker &walkerRight);
    void initialField_again(Walker &walkerRight);
    void copyField(MetropolisInfo metropolisInfo_input);

    void addLeftLogw(std::complex<double> addLogw);
    void addLogWeightLeftInBlockFinal(std::complex<double> addLogw);
    void balanceLogWeightLeftInBlockFinal(double numOfChains);
    size_t getBPMetroTimesliceBlockSize() const;
    const Walker& getWalkerLeftInBlock(int n) const;
    const Walker& getWalkerRightInBlock(int n) const;
    const Walker& getWalkerLeftInBlockFinal() const;
    Walker& walkerRightInBlockInitialRef();
    std::complex<double>& logWeightRightInBlockInitialRef();

    void updateOneSweep(std::vector<int> flip_i_vec);
    void updateToRightOneSweep(std::vector<int> flip_i_vec, bool ifFrozen);
    void updateToLeftOneSweep(std::vector<int> flip_i_vec, bool ifFrozen);
    void updateToRightOneStep(Walker & walkerLeft, std::complex<double> &logWeight, std::vector<int> flip_i_vec, bool ifFrozen);
    void updateToLeftOneStep(Walker & walkerRight, std::complex<double> &logWeight, std::vector<int> flip_i_vec, bool ifFrozen);
    //
    //////////////////////////////////////////////////////////
    void initial_globalFast();
    void update_globalFast();
    void getAuxMatrix_D(TwoBodyAux_Jastrow auxNew, size_t j_Jastrow);
    tensor_hao::TensorHao<std::complex<double>, 2> getAuxMatrix_D_exp_x(TwoBodyAux_Jastrow auxNew, size_t j_Jastrow);
    std::complex<double> getLogOverlapFromAux(TwoBodyAux_Jastrow auxNew, size_t expM, tensor_hao::TensorHao<std::complex<double>, 2> &overlapMatrix);
    void getOverlapMatrixInvA1ExpS(tensor_hao::TensorHao<std::complex<double>, 2> & overlapMatrixInvA1ExpS);
    void getOverlapMatrixInvA1ExpS_A1ExpS(tensor_hao::TensorHao<std::complex<double>, 2> & overlapMatrixInvA1ExpS, tensor_hao::TensorHao<std::complex<double>, 2> & A1ExpS_12);
    void get_PWR(tensor_hao::TensorHao<std::complex<double>, 2> P_matrix, tensor_hao::TensorHao<std::complex<double>, 2> & PWR);
    void get_A0PWR_B0PWR(tensor_hao::TensorHao<std::complex<double>, 2> P_matrix, tensor_hao::TensorHao<std::complex<double>, 2> & A0PWR, tensor_hao::TensorHao<std::complex<double>, 2> & B0PWR);
    void get_A0_A1ExpS12B0(tensor_hao::TensorHao<std::complex<double>, 2> & A1ExpS, tensor_hao::TensorHao<std::complex<double>, 2> & A0_A1ExpS12B0);
    //////////////////////////////////////////////////////////
    void updateWalkerRight(Walker walkerRight);
    void checkLocalUpdateData(Walker walkerRight_input);
    // local update related
    void updateDirectB();
    void updateDirectOverlapMatrix_inv(Walker & walkerRight, 
                                      tensor_hao::TensorHao<std::complex<double>, 2> &overlapMatrix_inv);
    //   
    void updateOverlapMatrix_inv_fromOverlapMatrix();
    // 
    std::complex<double> calculateTargetOverlapRatioFastUpdate(tensor_hao::TensorHao<std::complex<double>, 2> U, 
                                                              tensor_hao::TensorHao<std::complex<double>, 2> Vdagger,
                                                              tensor_hao::TensorHao<std::complex<double>, 2> & A_inv);
    void updateOverlapMatrixInvWithSMW(tensor_hao::TensorHao<std::complex<double>, 2> & A_inv,
                                       tensor_hao::TensorHao<std::complex<double>, 2> U,
                                       tensor_hao::TensorHao<std::complex<double>, 2> Vdagger);
    void calculateUpdatedB(tensor_hao::TensorHao<std::complex<double>, 2> B, 
                          tensor_hao::TensorHao<std::complex<double>, 2> U_s,
                          tensor_hao::TensorHao<std::complex<double>, 2> Vdagger_s,
                          tensor_hao::TensorHao<std::complex<double>, 2> &updated_B);
    void updateOverlapMatricesAndRatios(std::vector<tensor_hao::TensorHao<std::complex<double>, 2>> &U_vec,
                                                std::vector<tensor_hao::TensorHao<std::complex<double>, 2>> &Vdagger_vec,
                                                tensor_hao::TensorHao<std::complex<double>, 2> &updated_overlapMatrix_inv,
                                                std::complex<double> &overlapRatio_fastUpdate);

    std::complex<double> performLocalUpdateWithFastOverlapRatio(int j_Jastrow, 
                                                int inBlockIndex,
                                                std::vector<int> flip_i_vec,
                                                tensor_hao::TensorHao<std::complex<double>, 1> auxNew,
                                                tensor_hao::TensorHao<std::complex<double>, 2> & B_proposed, 
                                                tensor_hao::TensorHao<std::complex<double>, 2> & overlapMatrix_inv_proposed);
                                    
    void updateWalkerLeftMetro();
    const Walker& getWalkerLeftMetro();
    //////////////////////////////////////////////////////////
    //
    int returnNbuf() const;
    double getMemory() const;
#ifdef MPI_HAO
    friend void MPIBcast(Metropolis &buffer, int root,  const MPI_Comm& comm);
    void pack( std::vector<char> &buf,  int &posit ) const;
    void unpack( const std::vector<char> &buf, int &posit );
#endif

    void copy_deep(const Metropolis &x);
    void move_deep(Metropolis &x);

};

#endif //AFQMCLAB_METROPOLIS_H

/////////////////////////////////////////////////
/////////////////////////////////////////////////