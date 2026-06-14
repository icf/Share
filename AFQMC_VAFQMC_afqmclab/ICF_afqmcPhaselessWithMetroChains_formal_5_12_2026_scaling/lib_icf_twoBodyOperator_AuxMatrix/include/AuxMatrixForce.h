//
// Created by Hao Shi on 1/12/18.
//

#ifndef AFQMC_HEISENBERG_AuxMatrixForce_H
#define AFQMC_HEISENBERG_AuxMatrixForce_H

#include "afqmclab.h"

class AuxMatrixForce
{
 public:
    tensor_hao::TensorHao<double, 1> QijForce;

    AuxMatrixForce();
    AuxMatrixForce(size_t NumberOfAuxMatrix);
    AuxMatrixForce(const AuxMatrixForce& x);
    AuxMatrixForce(AuxMatrixForce&& x);
    ~AuxMatrixForce();

    AuxMatrixForce & operator  = (const AuxMatrixForce& x);
    AuxMatrixForce & operator  = (AuxMatrixForce&& x);

    size_t getNumberOfAuxMatrix() const;
    double getMemory() const;

    int returnNbuf() const;
#ifdef MPI_HAO
    void pack( std::vector<char> &buf,  int &posit ) const;
    void unpack( const std::vector<char> &buf, int &posit );
#endif

 private:
    void copy_deep(const AuxMatrixForce &x);
    void move_deep(AuxMatrixForce &x);
};

#endif //AFQMC_HEISENBERG_AuxMatrixForce_H
