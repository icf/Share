//
// Created by boruoshihao on 5/23/17.
//

#ifndef AFQMCLAB_SVD_H
#define AFQMCLAB_SVD_H

#include "svdAux.h"
#include "svdForce.h"
#include "svdSample.h"
#include "svdSample2s.h"
#include "../generalHamiltonian_icf/tensor_hao_mpi_shared.h"

class SVD
{
 private:
    double dt;
    double SVD_cutoff = 1e-8;
    std::complex<double> sqrtMinusDt;
    size_t svdNumber;
     tensor_hao::TensorHaoMPIRef<std::complex<double>, 3> sqrtMinusDtSVDVecs;
     tensor_hao::TensorHaoMPIRef<std::complex<double>, 3> sqrtMinusDtSVDVecsUp;
     const tensor_hao::TensorHaoMPIRef<std::complex<double>, 3> & sqrtMinusDtSVDVecsDn = sqrtMinusDtSVDVecsUp;
   //  
    tensor_hao::TensorHao<std::complex<double>, 2> U0up, Vdagger0up;
    tensor_hao::TensorHao<std::complex<double>, 2> U0dn, Vdagger0dn;
    tensor_hao::TensorHao<std::complex<double>, 2> U0, Vdagger0;
    tensor_hao::TensorHao<std::complex<double>, 3> sqrtMinusDtSVDVecs_Dup;
    const tensor_hao::TensorHao<std::complex<double>, 3> & sqrtMinusDtSVDVecs_Ddn = sqrtMinusDtSVDVecs_Dup;
   //  
    tensor_hao::TensorHao<std::complex<double>, 3> sqrtMinusDtSVDVecs_D;
   //  
   // For expM2, globalFastUpdate
   //  tensor_hao::TensorHao<std::complex<double>, 3> sqrtMinusDtSVDVecs_DC0D;
   //  tensor_hao::TensorHao<std::complex<double>, 3> sqrtMinusDtSVDVecs_DupC0upDup, sqrtMinusDtSVDVecs_DdnC0dnDdn;
   //  const tensor_hao::TensorHao<std::complex<double>, 1> *svdBg;
   tensor_hao::TensorHao<std::complex<double>, 1> svdBg;
   // 
   bool Hamiltonian_spin_flag;

 public:
    SVD();
    SVD(double dt,
                 const tensor_hao::TensorHaoMPIRef<std::complex<double>, 3> &svdVecs,
                 const tensor_hao::TensorHao<std::complex<double>, 2> &svdVecs_U0,
                 const tensor_hao::TensorHao<std::complex<double>, 3> &svdVecs_D,
                 const tensor_hao::TensorHao<std::complex<double>, 2> &svdVecs_Vdagger0,
                 const tensor_hao::TensorHao<std::complex<double>, 1> &svdBg,
                 bool Hamiltonian_spin_flag_input,
                 bool cutoffFlag=false);
    SVD(double dt,
                 const tensor_hao::TensorHaoMPIRef<std::complex<double>, 3> &svdVecs,
                 const tensor_hao::TensorHao<std::complex<double>, 2> &svdVecs_U0,
                 const tensor_hao::TensorHao<std::complex<double>, 3> &svdVecs_D,
                 const tensor_hao::TensorHao<std::complex<double>, 2> &svdVecs_Vdagger0,
                 const tensor_hao::TensorHao<std::complex<double>, 1> &svdBg, 
                 bool Hamiltonian_spin_flag_input,
                 int conjugate,
                 bool cutoffFlag=false);
    SVD(const SVD& x);
    SVD(SVD&& x);
    ~SVD();

    SVD & operator  = (const SVD& x);
    SVD & operator  = (SVD&& x);

    double getDt() const;
    const tensor_hao::TensorHaoMPIRef<std::complex<double>, 3> &getSqrtMinusDtSVDVecs() const;
    const tensor_hao::TensorHao<std::complex<double>, 3> &getSqrtMinusDtSVDVecs_Dup() const;
    const tensor_hao::TensorHao<std::complex<double>, 3> &getSqrtMinusDtSVDVecs_Ddn() const;
    const tensor_hao::TensorHao<std::complex<double>, 3> &getSqrtMinusDtSVDVecs_D() const;
    tensor_hao::TensorHao<std::complex<double>,2> getSqrtMinusDtSVDVecsMatrix(int i) const;
    tensor_hao::TensorHao<std::complex<double>,2> getSqrtMinusDtSVDVecsMatrixUp(int i) const;
    tensor_hao::TensorHao<std::complex<double>,2> getSqrtMinusDtSVDVecsMatrixDn(int i) const;
    tensor_hao::TensorHao<std::complex<double>,2> getSqrtMinusDtSVDVecsMatrix_SVD(int i, std::string name) const;
   //  const tensor_hao::TensorHao<std::complex<double>, 1> *getSVDBg() const;
    const tensor_hao::TensorHao<std::complex<double>, 1> getSVDBg() const;
    void updateBG(const tensor_hao::TensorHao<std::complex<double>, 1> &bg);
    size_t getSVDNumber() const;
   // ///////////////////////
   size_t getSVD_U0upRank0() const;
   size_t getSVD_U0dnRank0() const;
   size_t getSVD_U0Rank0() const;
   size_t getSVD_Vdagger0upRank1() const;
   size_t getSVD_Vdagger0dnRank1() const;
   size_t getSVD_Vdagger0Rank1() const;
   size_t getSVD_DupRank() const;
   size_t getSVD_DdnRank() const;
   size_t getSVD_DRank() const;
   // ///////////////////////
    const std::complex<double> &getSqrtMinusDt() const;
    size_t returnBasisSize() const;

    SVDForce readForce(const std::string &filename) const;
    SVDAux sampleAuxFromForce(const SVDForce &force) const;
    SVDAux sampleAuxFromForce_localUpdate(const SVDForce &force, const SVDAux &auxInput, std::vector<int> flip_i) const;
    std::complex<double> logProbOfAuxFromForce(const SVDAux &aux, const SVDForce &force) const;
    SVDSample getTwoBodySampleFromAux(const SVDAux &aux) const;
    SVDSample2s getTwoBodySampleFromAux2s(const SVDAux &aux) const;
    std::complex<double> getTwoBodySample_logw_FromAux2s(const SVDAux &aux) const;
    std::complex<double> getTwoBodySample_logw_FromAux(const SVDAux &aux) const;
    //
    SVDSample getTwoBodySampleFromAux_test(const SVDAux &aux) const;
    //
    SVDSample getTwoBodySampleFromAux_icf_fixedForce(const SVDAux &aux, const SVDForce &force) const;
    SVDSample getTwoBodySampleFromAux_icf_ConstForce(const SVDAux &aux) const;
    SVDSample getTwoBodySampleFromAuxForce(const SVDAux &aux, const SVDForce &force) const;

    SVDSample2s getTwoBodySampleFromAux_icf_fixedForce2s(const SVDAux &aux, const SVDForce &force) const;
    SVDSample2s getTwoBodySampleFromAux_icf_ConstForce2s(const SVDAux &aux) const;
    SVDSample2s getTwoBodySampleFromAuxForce2s(const SVDAux &aux, const SVDForce &force) const;
    double getMemory() const;
 private:
    void copy_deep(const SVD &x);
    void move_deep(SVD &x);
    void initialSqrtMinusDtSVDVecs(const tensor_hao::TensorHaoMPIRef<std::complex<double>, 3> &svdVecs, const tensor_hao::TensorHao<std::complex<double>, 2> &svdVecs_U0, const tensor_hao::TensorHao<std::complex<double>, 3> &svdVecs_D, const tensor_hao::TensorHao<std::complex<double>, 2> &svdVecs_Vdagger0, bool cutoffFlag=false);
    void setTwoBodySampleMatrix(SVDSample &svdSample, const SVDAux &aux) const;
    void setTwoBodySampleMatrix(SVDSample2s &svdSample2s, const SVDAux &aux) const;
};

#endif //AFQMCLAB_SVD_H