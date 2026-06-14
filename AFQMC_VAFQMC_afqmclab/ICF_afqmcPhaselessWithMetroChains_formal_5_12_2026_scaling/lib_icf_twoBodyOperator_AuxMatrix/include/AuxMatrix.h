//
// Created by Hao Shi on 1/12/18.
//

#ifndef AFQMC_HEISENBERG_AuxMatrix_H
#define AFQMC_HEISENBERG_AuxMatrix_H

#include "AuxMatrixAux.h"
#include "AuxMatrixForce.h"
#include "AuxMatrixSample.h"

class AuxMatrix
{
 private:
    size_t L, NumberOfAuxMatrix;
    tensor_hao::TensorHao<int,1> site_i, site_j;

    std::string decompType;  
    std::vector<std::complex<double> > Qij_vec;

 public:
    AuxMatrix();
    AuxMatrix(const std::string &filename);
    AuxMatrix(const AuxMatrix& x);
    AuxMatrix(AuxMatrix&& x);
    ~AuxMatrix();

    AuxMatrix & operator  = (const AuxMatrix& x);
    AuxMatrix & operator  = (AuxMatrix&& x);

    void defaultModel();
    void readModel(const std::string &filename);

    const size_t &getNumberOfAuxMatrix() const;
    const std::string &getDecompType() const;
    const std::complex<double> &getQij_vec(size_t i) const;

    std::complex<double> calculateAuxForce(const AuxMatrixAux &aux, const AuxMatrixForce &force);
    AuxMatrixAux sampleAuxFromForce(const AuxMatrixForce &force) const;
    std::complex<double> logProbOfAuxFromForce(const AuxMatrixAux &aux, const AuxMatrixForce &force) const;
    AuxMatrixSample getTwoBodySampleFromAux(const AuxMatrixAux &aux) const;
    AuxMatrixSample getTwoBodySampleFromAuxForce(const AuxMatrixAux &aux, const AuxMatrixForce &force) const;
    //
    const LogHop getTwoBodySampleFromAuxForce_LogHopType(const AuxMatrixAux &aux, const AuxMatrixForce &force) const;
    const LogHop getTwoBodySampleFromAux_LogHopType(const AuxMatrixAux &aux) const;

    double getMemory();
 private:
    void copy_deep(const AuxMatrix &x);
    void move_deep(AuxMatrix &x);

    void setGamma();

    void setTwoBodySampleMatrix(AuxMatrixSample &twoBodySample, const AuxMatrixAux &aux) const;

};

#endif //AFQMC_HEISENBERG_AuxMatrix_H
