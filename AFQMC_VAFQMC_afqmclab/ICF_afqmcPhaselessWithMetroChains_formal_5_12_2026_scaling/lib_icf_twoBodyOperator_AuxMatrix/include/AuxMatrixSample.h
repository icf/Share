//
// Created by Hao Shi on 1/13/18.
//

#ifndef AFQMC_HEISENBERG_AuxMatrixSample_H
#define AFQMC_HEISENBERG_AuxMatrixSample_H

#include "afqmclab.h"

class AuxMatrixSample
{
 public:
    std::complex<double> logw;
    tensor_hao::TensorHao<std::complex<double>,2> matrix;

    AuxMatrixSample();
    AuxMatrixSample(size_t NumberOfAuxMatrix, size_t L);
    AuxMatrixSample(const AuxMatrixSample &x);
    AuxMatrixSample(AuxMatrixSample &&x);
    ~AuxMatrixSample();

    AuxMatrixSample & operator  = (const AuxMatrixSample& x);
    AuxMatrixSample & operator  = (AuxMatrixSample&& x);

    size_t getNumberOfAuxMatrix() const;
    size_t getL() const;
    double getMemory() const;

 private:
    void copy_deep(const AuxMatrixSample &x);
    void move_deep(AuxMatrixSample &x);
};

#endif //AFQMC_HEISENBERG_AuxMatrixSample_H