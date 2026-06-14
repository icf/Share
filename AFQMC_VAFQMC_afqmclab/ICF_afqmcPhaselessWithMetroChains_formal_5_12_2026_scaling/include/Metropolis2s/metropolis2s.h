//
// Created by boruoshihao on 1/15/17.
//

#ifndef AFQMCLAB_METROPOLIS2S_H
#define AFQMCLAB_METROPOLIS2S_H

#include "metropolis2sDefine.h"
#include "metropolis2sMethod.h"
#include "JastrowProjector2s.h"
#include "metropolis2sInfo.h"

#ifdef MPI_HAO
class Metropolis2s;
void MPIBcast(Metropolis2s &buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
#endif

class Metropolis2s
{
public:
    ////////////////////////////////////////////////////////////
    //objects do depends on sampling  -- > MPI distributed
    ////////////////////////////////////////////////////////////
    //Metropolis2sInfo is the only object that shared and distributd by MPI
    Metropolis2sInfo metropolis2sInfo;
    ////////////////////////////////////////////////////////////
    //objects do not depends on sampling -- > MPI not distributed
    ////////////////////////////////////////////////////////////
    JastrowProjector2s *jastrowProjector;
    JastrowProjector2s *jastrowProjectorExtended;
    
    std::vector<OneBody_Jastrow2s> *expMinusDtK_Jastrow_vec;
    std::vector<TwoBody_Jastrow2s> *expMinusDtV_Jastrow_vec;
    //
    Metropolis2sMethod method;
    ////////////////////////////////////////////////////////////
    //No need to be distributed but restore after popcontrol
    HopWalkerOperation2s oneBodyWalkerRightOperation;
    HopWalkerOperation2s oneBodyWalkerLeftOperation;
    SVDSampleWalkerOperation2s twoBodySampleWalkerRightOperation;
    SVDSampleWalkerOperation2s twoBodySampleWalkerLeftOperation;

    Metropolis2s();
    Metropolis2s(const Metropolis2s& x);
    Metropolis2s(Metropolis2s &&x);
    ~Metropolis2s();

    Metropolis2s & operator  = (const Metropolis2s& x);
    Metropolis2s & operator  = (Metropolis2s&& x);

    void initJastrowProjectorNullptr();
    void setJastrowProjector(JastrowProjector2s &jastrowProjector_);
    void setJastrowProjectorExtended(JastrowProjector2s &jastrowProjectorExtended_);
    void setExpMinusDt_KV_Jastrow_vec(std::vector<OneBody_Jastrow2s> &expMinusDtK_Jastrow_vec_, std::vector<TwoBody_Jastrow2s> &expMinusDtV_Jastrow_vec_);
    void initialParameters(int L, int Nup, int Ndn, JastrowProjector2s &jastrowProjector_, std::vector<OneBody_Jastrow2s> &expMinusDtK_Jastrow_vec_, std::vector<TwoBody_Jastrow2s> &expMinusDtV_Jastrow_vec_);
    void initialParametersTwoJastrow(int L, int Nup, int Ndn, JastrowProjector2s &jastrowProjector_, std::vector<OneBody_Jastrow2s> &expMinusDtK_Jastrow_vec_, std::vector<TwoBody_Jastrow2s> &expMinusDtV_Jastrow_vec_);

    void extendMetroChainInfoToRight(Metropolis2sInfo metropolisInfo_input);
    void initialField(Walker2s &walkerLeft, Walker2s &walkerRight);
    void initialField_again(Walker2s &walkerRight);
    void copyField(Metropolis2sInfo metropolisInfo_input);

    void addLeftLogw(std::complex<double> addLogw);
    void addLogWeightLeftInBlockFinal(std::complex<double> addLogw);
    void balanceLogWeightLeftInBlockFinal(double numOfChains);
    size_t getBPMetroTimesliceBlockSize() const;
    const Walker2s& getWalkerLeftInBlock(int n) const;
    const Walker2s& getWalkerRightInBlock(int n) const;
    const Walker2s& getWalkerLeftInBlockFinal() const;
    Walker2s& walkerRightInBlockInitialRef();
    std::complex<double>& logWeightRightInBlockInitialRef();

    void updateOneSweep(std::vector<int> flip_i_vec);
    void updateToRightOneSweep(std::vector<int> flip_i_vec, bool ifFrozen);
    void updateToLeftOneSweep(std::vector<int> flip_i_vec, bool ifFrozen);
    void updateToRightOneStep(Walker2s & walkerLeft, std::complex<double> &logWeight, std::vector<int> flip_i_vec, bool ifFrozen);
    void updateToLeftOneStep(Walker2s & walkerRight, std::complex<double> &logWeight, std::vector<int> flip_i_vec, bool ifFrozen);
    //
    //////////////////////////////////////////////////////////
    void initial_globalFast();
    void update_globalFast();
    void getAuxMatrix_Dup(TwoBodyAux_Jastrow2s auxNew, size_t j_Jastrow);
    void getAuxMatrix_Ddn(TwoBodyAux_Jastrow2s auxNew, size_t j_Jastrow);
    tensor_hao::TensorHao<std::complex<double>, 2> getAuxMatrix_Dup_exp_x(TwoBodyAux_Jastrow2s auxNew, size_t j_Jastrow);
    tensor_hao::TensorHao<std::complex<double>, 2> getAuxMatrix_Ddn_exp_x(TwoBodyAux_Jastrow2s auxNew, size_t j_Jastrow);
    std::complex<double> getLogOverlapFromAux(TwoBodyAux_Jastrow2s auxNew, size_t expM, tensor_hao::TensorHao<std::complex<double>, 2> &overlapMatrixup, tensor_hao::TensorHao<std::complex<double>, 2> &overlapMatrixdn);
    void getOverlapMatrixInvA1ExpS(tensor_hao::TensorHao<std::complex<double>, 2> & overlapMatrixInvA1ExpSUp, tensor_hao::TensorHao<std::complex<double>, 2> & overlapMatrixInvA1ExpSDn);
    void getOverlapMatrixInvA1ExpS_A1ExpS(tensor_hao::TensorHao<std::complex<double>, 2> & overlapMatrixInvA1ExpSUp, tensor_hao::TensorHao<std::complex<double>, 2> & overlapMatrixInvA1ExpSDn, tensor_hao::TensorHao<std::complex<double>, 2> & A1ExpS_12up, tensor_hao::TensorHao<std::complex<double>, 2> & A1ExpS_12dn);
    void get_PWR(tensor_hao::TensorHao<std::complex<double>, 2> Pup_matrix, tensor_hao::TensorHao<std::complex<double>, 2> Pdn_matrix, tensor_hao::TensorHao<std::complex<double>, 2> & PWRup, tensor_hao::TensorHao<std::complex<double>, 2> & PWRdn);
    void get_A0PWR_B0PWR(tensor_hao::TensorHao<std::complex<double>, 2> Pup_matrix, tensor_hao::TensorHao<std::complex<double>, 2> Pdn_matrix, tensor_hao::TensorHao<std::complex<double>, 2> & A0PWRup, tensor_hao::TensorHao<std::complex<double>, 2> & A0PWRdn, tensor_hao::TensorHao<std::complex<double>, 2> & B0PWRup, tensor_hao::TensorHao<std::complex<double>, 2> & B0PWRdn);
    void get_A0_A1ExpS12B0(tensor_hao::TensorHao<std::complex<double>, 2> & A1ExpSUp, tensor_hao::TensorHao<std::complex<double>, 2> & A1ExpSDn, tensor_hao::TensorHao<std::complex<double>, 2> & A0up_A1ExpS12B0up, tensor_hao::TensorHao<std::complex<double>, 2> & A0dn_A1ExpS12B0dn);
    //////////////////////////////////////////////////////////
    void updateWalkerRight(Walker2s walkerRight);
    void checkLocalUpdateData(Walker2s walkerRight_input);
    // local update related
    void updateDirectB();
    void updateDirectOverlapMatrix_inv(Walker2s & walkerRight, 
                                      tensor_hao::TensorHao<std::complex<double>, 2> &overlapMatrixUp_inv, 
                                      tensor_hao::TensorHao<std::complex<double>, 2> &overlapMatrixDn_inv);
    //   
    void updateOverlapMatrix_inv_fromOverlapMatrix();
    // 
    std::complex<double> calculateTargetOverlapRatioFastUpdate(tensor_hao::TensorHao<std::complex<double>, 2> Uup, 
                                                              tensor_hao::TensorHao<std::complex<double>, 2> Vdaggerup,
                                                              tensor_hao::TensorHao<std::complex<double>, 2> & A_inv);
    void updateOverlapMatrixInvWithSMW(tensor_hao::TensorHao<std::complex<double>, 2> & A_inv,
                                       tensor_hao::TensorHao<std::complex<double>, 2> Uup,
                                       tensor_hao::TensorHao<std::complex<double>, 2> Vdaggerup);
    void calculateUpdatedB(tensor_hao::TensorHao<std::complex<double>, 2> Bup, 
                          tensor_hao::TensorHao<std::complex<double>, 2> Bdn,
                          tensor_hao::TensorHao<std::complex<double>, 2> Uup_s,
                          tensor_hao::TensorHao<std::complex<double>, 2> Vdaggerup_s,
                          tensor_hao::TensorHao<std::complex<double>, 2> Udn_s,
                          tensor_hao::TensorHao<std::complex<double>, 2> Vdaggerdn_s,
                          tensor_hao::TensorHao<std::complex<double>, 2> &updated_Bup,
                          tensor_hao::TensorHao<std::complex<double>, 2> &updated_Bdn);
    void updateOverlapMatricesAndRatios(std::vector<tensor_hao::TensorHao<std::complex<double>, 2>> &Uup_vec,
                                                std::vector<tensor_hao::TensorHao<std::complex<double>, 2>> &Udn_vec,
                                                std::vector<tensor_hao::TensorHao<std::complex<double>, 2>> &Vdaggerup_vec,
                                                std::vector<tensor_hao::TensorHao<std::complex<double>, 2>> &Vdaggerdn_vec,
                                                tensor_hao::TensorHao<std::complex<double>, 2> &updated_overlapMatrixUp_inv,
                                                tensor_hao::TensorHao<std::complex<double>, 2> &updated_overlapMatrixDn_inv,
                                                std::complex<double> &overlapRatioUp_fastUpdate,
                                                std::complex<double> &overlapRatioDn_fastUpdate);

    std::complex<double> performLocalUpdateWithFastOverlapRatio(int j_Jastrow, 
                                                int inBlockIndex,
                                                std::vector<int> flip_i_vec,
                                                tensor_hao::TensorHao<std::complex<double>, 1> auxNew,
                                                tensor_hao::TensorHao<std::complex<double>, 2> & Bup_proposed, 
                                                tensor_hao::TensorHao<std::complex<double>, 2> & Bdn_proposed, 
                                                tensor_hao::TensorHao<std::complex<double>, 2> & overlapMatrixUp_inv_proposed, 
                                                tensor_hao::TensorHao<std::complex<double>, 2> & overlapMatrixDn_inv_proposed);
                                    
    void updateWalkerLeftMetro();
    const Walker2s& getWalkerLeftMetro();
    //////////////////////////////////////////////////////////
    //
    int returnNbuf() const;
    double getMemory() const;
#ifdef MPI_HAO
    friend void MPIBcast(Metropolis2s &buffer, int root,  const MPI_Comm& comm);
    void pack( std::vector<char> &buf,  int &posit ) const;
    void unpack( const std::vector<char> &buf, int &posit );
#endif

    void copy_deep(const Metropolis2s &x);
    void move_deep(Metropolis2s &x);

};

#endif //AFQMCLAB_METROPOLIS2S_H

/////////////////////////////////////////////////
/////////////////////////////////////////////////