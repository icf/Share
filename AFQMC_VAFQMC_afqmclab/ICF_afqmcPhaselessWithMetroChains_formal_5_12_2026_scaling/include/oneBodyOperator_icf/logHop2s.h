//
// Created by Zhi-Yu 
//

#ifndef AFQMCLAB_LOGHOP2S_H
#define AFQMCLAB_LOGHOP2S_H

#include "afqmclab.h"

//One body operator: two identical spin species.
//The operator is exp( M ) = 1 + M + 1/(2!) M^2 + 1/(3!) M^3 + ...
//M matrix is stored in LogHop class.

class LogHop2s
{
 public:
    std::complex<double> logw;
    tensor_hao::TensorHao<std::complex<double>,2> matrixUp, matrixDn;

    LogHop2s();
    LogHop2s(size_t L);
    LogHop2s(const LogHop2s& x);
    LogHop2s(LogHop2s&& x);
    ~LogHop2s();

    LogHop2s & operator  = (const LogHop2s& x);
    LogHop2s & operator  = (LogHop2s&& x);

    size_t getL() const;
    double getMemory() const;

 private:
    void copy_deep(const LogHop2s &x);
    void move_deep(LogHop2s &x);
};


#endif //AFQMCLAB_LOGHOP2S_H
