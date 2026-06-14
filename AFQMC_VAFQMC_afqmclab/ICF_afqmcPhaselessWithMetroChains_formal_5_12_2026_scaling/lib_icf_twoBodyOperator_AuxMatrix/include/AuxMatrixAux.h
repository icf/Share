//
// Created by Hao Shi on 1/12/18.
//

#ifndef AFQMC_HEISENBERG_AuxMatrixAux_H
#define AFQMC_HEISENBERG_AuxMatrixAux_H

#include "afqmclab.h"

class AuxMatrixAux
{
 public:
    tensor_hao::TensorHao<int, 1> QijAux;

    AuxMatrixAux();
    AuxMatrixAux(size_t NumberOfAuxMatrix);
    AuxMatrixAux(const AuxMatrixAux& x);
    AuxMatrixAux(AuxMatrixAux&& x);
    ~AuxMatrixAux();

    AuxMatrixAux & operator  = (const AuxMatrixAux& x);
    AuxMatrixAux & operator  = (AuxMatrixAux&& x);

    size_t getNumberOfAuxMatrix() const;
    double getMemory() const;

    int returnNbuf() const;
#ifdef MPI_HAO
    void pack( std::vector<char> &buf,  int &posit ) const;
    void unpack( const std::vector<char> &buf, int &posit );
#endif

private:
    void copy_deep(const AuxMatrixAux &x);
    void move_deep(AuxMatrixAux &x);
};


#endif //AFQMC_HEISENBERG_AuxMatrixAux_H
