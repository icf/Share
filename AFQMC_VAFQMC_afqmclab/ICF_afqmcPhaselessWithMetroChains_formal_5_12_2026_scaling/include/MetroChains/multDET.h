//
// Created by boruoshihao on 12/25/16.
// Modified by Icf on 2019-9-29
//

#ifndef AFQMCLAB_MULTDETERMINANT_H
#define AFQMCLAB_MULTDETERMINANT_H

#include "afqmclab.h"

//Single Determinant.

#ifdef MPI_HAO
class multDET;
void MPIBcast(multDET &buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
//multDET MPIAllgather(multDET &buffer, int root=0, const MPI_Comm& comm=MPI_COMM_WORLD);
#endif

class multDET
{
 private:
    int numOfDET;
    // std::vector< std::complex<double> > logw;
    // std::vector< tensor_hao::TensorHao<std::complex<double>,2> > wf;

    std::vector< SD > sdVec;
 public:
    multDET();
    multDET(size_t numOfDETTemp);   
    multDET(const multDET& x);
    multDET(multDET&& x);
    ~multDET();

    multDET & operator  = (const multDET& x);
    multDET & operator  = (multDET&& x);

    void reset();
    void setSD( size_t n, SD sdTemp);
    SD &getSD( size_t n);
    void addDET( tensor_hao::TensorHao<std::complex<double>, 2> wfTemp, std::complex<double> LogWeightTemp);
    void addDET( SD sdTemp);
    void pop_front(size_t M);

    size_t getNumOfDET() const;
    const std::complex<double> &getLogw(size_t n) const;
    const tensor_hao::TensorHao<std::complex<double>, 2> &getWf(size_t n) const;
    std::complex<double> &logwRef(size_t n);
    tensor_hao::TensorHao<std::complex<double>, 2> &wfRef(size_t n);
    size_t getL() const;
    size_t getN() const;

    void resizeAll(size_t numOfDETTemp);
    void stabilize(size_t n);
    void addLogw(size_t n, std::complex<double> logw_add);

    void readAddDET(const std::string &filename);
    int returnNbuf() const;
    double getMemory() const;

#ifdef MPI_HAO
    friend void MPIBcast(multDET &buffer, int root,  const MPI_Comm& comm);
    void pack( std::vector<char> &buf,  int &posit ) const;
    void unpack( const std::vector<char> &buf, int &posit );
#endif

 private:
    void copy_deep(const multDET &x);
    void move_deep(multDET &x);
};

#endif //AFQMCLAB_MULTDETERMINANT_H
