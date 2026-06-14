//
// Created by boruoshihao on 12/25/16.
// Modified by Icf on 2019-9-29
//

#ifndef AFQMCLAB_MULTDETERMINANT2S_H
#define AFQMCLAB_MULTDETERMINANT2S_H

#include "afqmclab.h"

//Single Determinant.

#ifdef MPI_HAO
class multDET2s;
void MPIBcast(multDET2s &buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
//multDET2s MPIAllgather(multDET2s &buffer, int root=0, const MPI_Comm& comm=MPI_COMM_WORLD);
#endif

class multDET2s
{
 private:
    int numOfDET;

    std::vector< SD2s > sd2sVec;
 public:
    multDET2s();
    multDET2s(size_t numOfDETTemp);   
    multDET2s(const multDET2s& x);
    multDET2s(multDET2s&& x);
    ~multDET2s();

    multDET2s & operator  = (const multDET2s& x);
    multDET2s & operator  = (multDET2s&& x);

    void reset();
    void setSD( size_t n, SD2s sdTemp);
    SD2s &getSD( size_t n);
    void addDET( tensor_hao::TensorHao<std::complex<double>, 2> wfUpTemp, tensor_hao::TensorHao<std::complex<double>, 2> wfDnTemp, std::complex<double> LogWeightTemp);
    void addDET( SD2s sdTemp);
    void pop_front(size_t M);

    size_t getNumOfDET() const;
    const std::complex<double> &getLogw(size_t n) const;
    const tensor_hao::TensorHao<std::complex<double>, 2> &getWfUp(size_t n) const;
    const tensor_hao::TensorHao<std::complex<double>, 2> &getWfDn(size_t n) const;
    std::complex<double> &logwRef(size_t n);
    tensor_hao::TensorHao<std::complex<double>, 2> &wfUpRef(size_t n);
    tensor_hao::TensorHao<std::complex<double>, 2> &wfDnRef(size_t n);
    size_t getL() const;
    size_t getN() const;
    size_t getNup() const;
    size_t getNdn() const;

    void resizeAll(size_t numOfDETTemp);
    void stabilize(size_t n);
    void addLogw(size_t n, std::complex<double> logw_add);

    void readAddDET(const std::string &filename);
    int returnNbuf() const;
    double getMemory() const;

#ifdef MPI_HAO
    friend void MPIBcast(multDET2s &buffer, int root,  const MPI_Comm& comm);
    void pack( std::vector<char> &buf,  int &posit ) const;
    void unpack( const std::vector<char> &buf, int &posit );
#endif

 private:
    void copy_deep(const multDET2s &x);
    void move_deep(multDET2s &x);
};

#endif //AFQMCLAB_MULTDETERMINANT2S_H
