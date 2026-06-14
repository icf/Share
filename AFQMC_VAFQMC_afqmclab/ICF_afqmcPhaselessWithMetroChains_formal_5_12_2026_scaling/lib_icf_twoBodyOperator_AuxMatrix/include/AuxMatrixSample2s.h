//
// Created by Hao Shi on 1/13/18.
//

#ifndef AFQMC_HEISENBERG_AuxMatrixSample2s_H
#define AFQMC_HEISENBERG_AuxMatrixSample2s_H

#include "afqmclab.h"

class AuxMatrixSample2s
{
 public:
    std::complex<double> logw;
    tensor_hao::TensorHao<std::complex<double>,2> matrixUp, matrixDn;

    AuxMatrixSample2s();
    AuxMatrixSample2s(size_t NumberOfAuxMatrix, size_t L);
    AuxMatrixSample2s(const AuxMatrixSample2s &x);
    AuxMatrixSample2s(AuxMatrixSample2s &&x);
    ~AuxMatrixSample2s();

    AuxMatrixSample2s & operator  = (const AuxMatrixSample2s& x);
    AuxMatrixSample2s & operator  = (AuxMatrixSample2s&& x);

    size_t getNumberOfAuxMatrix() const;
    size_t getL() const;
    double getMemory() const;

 private:
    void copy_deep(const AuxMatrixSample2s &x);
    void move_deep(AuxMatrixSample2s &x);
};

#endif //AFQMC_HEISENBERG_AuxMatrixSample2s_H