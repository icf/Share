//
// Created by icf on 4/18/24.
//

#ifndef AFQMCLAB_GENERALHAMILTONIAN_SYM_ICF_H
#define AFQMCLAB_GENERALHAMILTONIAN_SYM_ICF_H

#include "afqmclab.h"
#include "../twoBodyOperator_icf/svd_sym.h"
#include "../oneBodyOperator_icf/logHopSDOperation_icf.h"

//K is the non-interacting matrix.
//K is K - svdVecs*svdVecs.
//Kp is K + svdBg*svdVecs.

#ifdef MPI_HAO
class GeneralHamiltonian_sym_icf;
void MPIBcast(GeneralHamiltonian_sym_icf &buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
#endif

class GeneralHamiltonian_sym_icf
{
 private:
    //size_t k_l_x, k_l_y;
    size_t L, Nup, Ndn, N;
    size_t svdNumber;
    tensor_hao::TensorHao<std::complex<double>,2> K;
    tensor_hao::TensorHao<std::complex<double>,2> svdVecs;
    tensor_hao::TensorHao<std::complex<double>,1> svdBg,initialBg;
    
    size_t KpEigenStatus; //0: void, 1: calculated Kp, 2: calculated Kp Eigens.
    tensor_hao::TensorHao<std::complex<double>,2> Kp;
    tensor_hao::TensorHao<double,1> KpEigenValue;
    tensor_hao::TensorHao<std::complex<double>,2> KpEigenVector;

 public:
    GeneralHamiltonian_sym_icf();
    GeneralHamiltonian_sym_icf(const std::string &filename);
    ~GeneralHamiltonian_sym_icf();

    size_t getL() const;
    size_t getKlx() const;
    size_t getKly() const;
    size_t getN() const;
    size_t getNup() const;
    size_t getNdn() const;
    size_t getSVDNumber() const;
    const tensor_hao::TensorHao<std::complex<double>,2> &getK() const;
    const tensor_hao::TensorHao<std::complex<double>,2> &getSVDVecs() const;
    const tensor_hao::TensorHao<std::complex<double>,1> &getSVDBg() const;
    const tensor_hao::TensorHao<std::complex<double>,1> &getInitialBg() const;
    size_t getKpEigenStatus() const;
    const tensor_hao::TensorHao<std::complex<double>,2> &getKp() const;
    const tensor_hao::TensorHao<double,1> &getKpEigenValue() const;
    const tensor_hao::TensorHao<std::complex<double>,2> &getKpEigenVector() const;

    void read(const std::string &filename);
    void read_conj(const std::string &filename);
    void write(const std::string &filename) const;
    
#ifdef MPI_HAO
    friend void MPIBcast(GeneralHamiltonian_sym_icf &buffer, int root,  const MPI_Comm& comm);
#endif

    void writeBackGround(const std::string &filename) const;
    void updateBackGround(const tensor_hao::TensorHao<std::complex<double>,1> &background);
    void updateBackGround(tensor_hao::TensorHao<std::complex<double>,1> &&background);

    Hop returnExpMinusAlphaK(double alpha);
    Hop2s returnExpMinusAlphaK2s(double alpha);
    LogHop returnLogExpMinusAlphaK(double alpha);
    LogHop2s returnLogExpMinusAlphaK2s(double alpha);
    Hop returnExpMinusAlphaK_nonHermitian(double alpha, std::string flag="dynamicOrder", size_t taylorOrder=0, double accuracy=1e-10, size_t baseTaylorOrder=3);
    Hop2s returnExpMinusAlphaK2s_nonHermitian(double alpha, std::string flag="dynamicOrder", size_t taylorOrder=0, double accuracy=1e-10, size_t baseTaylorOrder=3);
    SVD_sym returnExpMinusAlphaV(double alpha);
    SVD_sym returnExpMinusAlphaV_daggerSqrtDt(double alpha);

    void setKp();
    void setKpEigenValueAndVector();

    double getMemory() const;

    GeneralHamiltonian_sym_icf(const GeneralHamiltonian_sym_icf& x);

    GeneralHamiltonian_sym_icf & operator  = (const GeneralHamiltonian_sym_icf& x);
    // GeneralHamiltonian_sym_icf & operator  = (GeneralHamiltonian_sym_icf&& x);
};

#endif //AFQMCLAB_GENERALHAMILTONIAN_SYM_ICF_H