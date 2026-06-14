//
// Created by icf on 4/18/24.
//

#ifndef AFQMCLAB_GENERALHAMILTONIAN_ICF_H
#define AFQMCLAB_GENERALHAMILTONIAN_ICF_H

#include "afqmclab.h"
#include "../twoBodyOperator_icf/svd.h"
#include "../oneBodyOperator_icf/logHopSDOperation_icf.h"
#include "../oneBodyOperator_icf/logHop2sSD2sOperation.h"
#include "tensor_hao_mpi_shared.h"

//K is the non-interacting matrix.
//K is K - svdVecs*svdVecs.
//Kp is K + svdBg*svdVecs.

#ifdef MPI_HAO
class GeneralHamiltonian_icf;
// void MPIBcast(GeneralHamiltonian_icf &buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
void MPIBcastShared(GeneralHamiltonian_icf &buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
#endif

class GeneralHamiltonian_icf
{
 private:
    //size_t k_l_x, k_l_y;
    size_t SD2sL;
    bool Hamiltonian_spin_flag;
    size_t L, Nup, Ndn, N;
    size_t truncatedDup, truncatedDdn, truncatedD;
    size_t svdNumber, truncatedSvdNumber;
    tensor_hao::TensorHao<std::complex<double>,2> K, K_U, K_D, K_Vdagger;
    tensor_hao::TensorHao<std::complex<double>,2> svdVecs_U0, svdVecs_Vdagger0;
// #ifdef MPI_HAO
    tensor_hao::TensorHaoMPIRef<std::complex<double>,3> svdVecs;
// #else
//     tensor_hao::TensorHao<std::complex<double>,3> svdVecs;
// #endif
    tensor_hao::TensorHao<std::complex<double>,3> svdVecs_D;
    tensor_hao::TensorHao<std::complex<double>,1> svdBg, initialBg;
    
    size_t KpEigenStatus; //0: void, 1: calculated Kp, 2: calculated Kp Eigens.
    tensor_hao::TensorHao<std::complex<double>,2> Kp;
    tensor_hao::TensorHao<double,1> KpEigenValue;
    tensor_hao::TensorHao<std::complex<double>,2> KpEigenVector;

 public:
    GeneralHamiltonian_icf();
    GeneralHamiltonian_icf(const std::string &filename);
    ~GeneralHamiltonian_icf();

    size_t getL() const;
    size_t getSD2sL() const;
    bool getHamiltonian_spin_flag() const;
    
    size_t getTruncatedD() const;
    size_t getTruncatedDup() const;
    size_t getTruncatedDdn() const;
    size_t getKlx() const;
    size_t getKly() const;
    size_t getN() const;
    size_t getNup() const;
    size_t getNdn() const;
    size_t getSVDNumber() const;
    size_t getTruncatedSVDNumber() const;
    const tensor_hao::TensorHao<std::complex<double>,2> &getK() const;
    const tensor_hao::TensorHaoMPIRef<std::complex<double>,3> &getSVDVecs() const;
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
    // friend void MPIBcast(GeneralHamiltonian_icf &buffer, int root,  const MPI_Comm& comm);
    friend void MPIBcastShared(GeneralHamiltonian_icf &buffer, int root,  const MPI_Comm& comm);
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
    SVD returnExpMinusAlphaV(double alpha, bool cutoffFlag=false);
    SVD returnExpMinusAlphaV_daggerSqrtDt(double alpha, bool cutoffFlag=false);

    void setKp();
    void setKpEigenValueAndVector();

    double getMemory() const;

    GeneralHamiltonian_icf(const GeneralHamiltonian_icf& x);

    GeneralHamiltonian_icf & operator  = (const GeneralHamiltonian_icf& x);
    // GeneralHamiltonian_icf & operator  = (GeneralHamiltonian_icf&& x);
};

#endif //AFQMCLAB_GENERALHAMILTONIAN_ICF_H