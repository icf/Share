//
// Created by boruoshihao on 5/23/17.
//

#ifndef AFQMCLAB_SVD_SYM_H
#define AFQMCLAB_SVD_SYM_H

#include "svdAux.h"
#include "svdForce.h"
#include "svdSample.h"
#include "svdSample2s.h"

class SVD_sym
{
 private:
    double dt;
    std::complex<double> sqrtMinusDt;
    size_t svdNumber;
    tensor_hao::TensorHao<std::complex<double>, 2> sqrtMinusDtSVDVecs;
   //  const tensor_hao::TensorHao<std::complex<double>, 1> *svdBg;
   tensor_hao::TensorHao<std::complex<double>, 1> svdBg;

 public:
    SVD_sym();
    SVD_sym(double dt,
                 const tensor_hao::TensorHao<std::complex<double>, 2> &svdVecs,
                 const tensor_hao::TensorHao<std::complex<double>, 1> &svdBg );
    SVD_sym(double dt,
                 const tensor_hao::TensorHao<std::complex<double>, 2> &svdVecs,
                 const tensor_hao::TensorHao<std::complex<double>, 1> &svdBg, 
                 int conjugate);
    SVD_sym(const SVD_sym& x);
    SVD_sym(SVD_sym&& x);
    ~SVD_sym();

    SVD_sym & operator  = (const SVD_sym& x);
    SVD_sym & operator  = (SVD_sym&& x);

    double getDt() const;
    const tensor_hao::TensorHao<std::complex<double>, 2> &getSqrtMinusDtSVDVecs() const;
    tensor_hao::TensorHao<std::complex<double>,2> getSqrtMinusDtSVDVecsMatrix(int i);
   //  const tensor_hao::TensorHao<std::complex<double>, 1> *getSVDBg() const;
    const tensor_hao::TensorHao<std::complex<double>, 1> getSVDBg() const;
    size_t getSVDNumber() const;
    const std::complex<double> &getSqrtMinusDt() const;
    size_t returnBasisSize() const;

    SVDForce readForce(const std::string &filename) const;
    SVDAux sampleAuxFromForce(const SVDForce &force) const;
    std::complex<double> logProbOfAuxFromForce(const SVDAux &aux, const SVDForce &force) const;
    SVDSample getTwoBodySampleFromAux(const SVDAux &aux) const;
    SVDSample2s getTwoBodySampleFromAux2s(const SVDAux &aux) const;
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
    void copy_deep(const SVD_sym &x);
    void move_deep(SVD_sym &x);
    void initialSqrtMinusDtSVDVecs(const tensor_hao::TensorHao<std::complex<double>, 2> &svdVecs);
    void setTwoBodySampleMatrix(SVDSample &svdSample, const SVDAux &aux) const;
    void setTwoBodySampleMatrix(SVDSample2s &svdSample2s, const SVDAux &aux) const;
};

#endif //AFQMCLAB_SVD_SYM_H