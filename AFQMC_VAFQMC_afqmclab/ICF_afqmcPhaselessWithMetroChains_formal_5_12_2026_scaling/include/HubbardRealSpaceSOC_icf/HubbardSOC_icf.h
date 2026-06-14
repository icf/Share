//
// Created by boruoshihao on 1/11/17.
//

#ifndef AFQMCLAB_HUBBARDREALSPACESOC_ICF_H
#define AFQMCLAB_HUBBARDREALSPACESOC_ICF_H

#include "afqmclab.h"

#ifdef MPI_HAO
class HubbardSOC_icf;
void MPIBcast(HubbardSOC_icf &buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
#endif

class HubbardSOC_icf
{
 private:
    size_t L, N;
    tensor_hao::TensorHao< std::complex<double>, 2 > K;
    tensor_hao::TensorHao< double, 1> mu, hx, hy, hz;
    tensor_hao::TensorHao< double, 1> U;

    bool KEigenStatus;
    tensor_hao::TensorHao< double, 1 > KEigenValue;
    tensor_hao::TensorHao< std::complex<double>, 2 > KEigenVector;

 public:
    HubbardSOC_icf();
    HubbardSOC_icf(const std::string &filename);
    ~HubbardSOC_icf();

    size_t getL() const;
    size_t getN() const;
    const tensor_hao::TensorHao<std::complex<double>, 2> &getK() const;
    const tensor_hao::TensorHao<double, 1> &getMu() const;
    const tensor_hao::TensorHao<double, 1> &getHx() const;
    const tensor_hao::TensorHao<double, 1> &getHy() const;
    const tensor_hao::TensorHao<double, 1> &getHz() const;
    const tensor_hao::TensorHao<double, 1> &getU() const;
    bool getKEigenStatus() const;
    const tensor_hao::TensorHao<double, 1> &getKEigenValue() const;
    const tensor_hao::TensorHao<std::complex<double>, 2> &getKEigenVector() const;

    void read(const std::string &filename);
    void write(const std::string &filename) const;

#ifdef MPI_HAO
    friend void MPIBcast(HubbardSOC_icf &buffer, int root,  const MPI_Comm& comm);
#endif

    void setKEigenValueAndVector();
    Hop returnExpMinusAlphaK(double alpha);
    Hop2s returnExpMinusAlphaK2s(double alpha);
    NiupNidn returnExpMinusAlphaV(double alpha, const std::string &decompType);

    double getMemory() const;

//  private:
    HubbardSOC_icf(const HubbardSOC_icf& x);
    HubbardSOC_icf & operator  = (const HubbardSOC_icf& x);

};

#endif //AFQMCLAB_HUBBARDREALSPACESOC_ICF_H