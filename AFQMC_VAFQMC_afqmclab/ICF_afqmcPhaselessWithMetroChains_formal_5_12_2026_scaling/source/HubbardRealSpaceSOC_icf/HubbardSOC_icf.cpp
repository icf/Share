//
// Created by boruoshihao on 1/11/17.
//

#include "../../include/HubbardRealSpaceSOC_icf/HubbardSOC_icf.h"
#include "afqmclab.h"

using namespace std;
using namespace tensor_hao;

HubbardSOC_icf::HubbardSOC_icf():L(0),N(0),KEigenStatus(false) { }

HubbardSOC_icf::HubbardSOC_icf(const string &filename) { read(filename); }

HubbardSOC_icf::~HubbardSOC_icf() { }

size_t HubbardSOC_icf::getL() const { return L; }

size_t HubbardSOC_icf::getN() const { return N; }

const TensorHao<complex<double>, 2> &HubbardSOC_icf::getK() const { return K; }

const TensorHao<double, 1> &HubbardSOC_icf::getMu() const { return mu; }

const TensorHao<double, 1> &HubbardSOC_icf::getHx() const { return hx; }

const TensorHao<double, 1> &HubbardSOC_icf::getHy() const { return hy; }

const TensorHao<double, 1> &HubbardSOC_icf::getHz() const { return hz; }

const TensorHao<double, 1> &HubbardSOC_icf::getU() const  { return U; }

bool HubbardSOC_icf::getKEigenStatus() const { return KEigenStatus; }

const TensorHao<double, 1> &HubbardSOC_icf::getKEigenValue() const { return KEigenValue; }

const TensorHao<complex<double>, 2> &HubbardSOC_icf::getKEigenVector() const { return KEigenVector; }

void HubbardSOC_icf::read(const string &filename)
{
    ifstream file;
    file.open(filename, ios::in);
    if ( ! file.is_open() ) {cout << "Error opening file in File!!! "<<filename<<endl; exit(1);}

    readFile( L, file );
    readFile( N, file );
    K.resize(2*L,2*L); readFile( K.size(),  K.data(),  file );
    mu.resize(L);      readFile( mu.size(), mu.data(), file );
    hx.resize(L);      readFile( hx.size(), hx.data(), file );
    hy.resize(L);      readFile( hy.size(), hy.data(), file );
    hz.resize(L);      readFile( hz.size(), hz.data(), file );
    U.resize(L);       readFile( U.size(),  U.data(),  file );

    file.close();

    KEigenStatus = false;
    KEigenValue.resize( static_cast<size_t>(0) );
    KEigenVector.resize( 0, 0 );
}

void HubbardSOC_icf::write(const string &filename) const
{
    ofstream file;
    file.open(filename, ios::out|ios::trunc);
    if ( ! file.is_open() ) {cout << "Error opening file in File!!! "<<filename<<endl; exit(1);}

    writeFile( L, file );
    writeFile( N, file );
    writeFile( K.size(),  K.data(),  file );
    writeFile( mu.size(), mu.data(), file );
    writeFile( hx.size(), hx.data(), file );
    writeFile( hy.size(), hy.data(), file );
    writeFile( hz.size(), hz.data(), file );
    writeFile( U.size(),  U.data(),  file );

    file.close();
}

#ifdef MPI_HAO
void MPIBcast(HubbardSOC_icf &buffer, int root, MPI_Comm const &comm)
{
    MPIBcast( buffer.L, root, comm  );
    MPIBcast( buffer.N, root, comm  );
    MPIBcast( buffer.K, root, comm  );
    MPIBcast( buffer.mu, root, comm );
    MPIBcast( buffer.hx, root, comm );
    MPIBcast( buffer.hy, root, comm );
    MPIBcast( buffer.hz, root, comm );
    MPIBcast( buffer.U, root, comm  );

    MPIBcast( buffer.KEigenStatus, root, comm );
    MPIBcast( buffer.KEigenValue, root, comm );
    MPIBcast( buffer.KEigenVector, root, comm );
}
#endif

void HubbardSOC_icf::setKEigenValueAndVector()
{
    if( KEigenStatus ) return;

    checkHermitian(K);
    KEigenVector = K;
    KEigenValue.resize(2*L);
    BL_NAME(eigen)(KEigenVector, KEigenValue);

    KEigenStatus = true;
}

Hop HubbardSOC_icf::returnExpMinusAlphaK(double alpha)
{
    setKEigenValueAndVector();

    Hop expAlphaK(2*L);

    BL_NAME(gmm)( KEigenVector, dMultiMatrix( exp(-alpha*KEigenValue), conjtrans(KEigenVector) ), expAlphaK.matrix);

    return expAlphaK;
}

Hop2s HubbardSOC_icf::returnExpMinusAlphaK2s(double alpha)
{
    setKEigenValueAndVector();

    Hop2s expAlphaK2s(L);
    TensorHao<complex<double>,2> matrix(2*L,2*L); 

    BL_NAME(gmm)( KEigenVector, dMultiMatrix( exp(-alpha*KEigenValue), conjtrans(KEigenVector) ), matrix);

    for(size_t i = 0; i < L; ++i)
    {
        for(size_t j = 0; j < L; ++j) expAlphaK2s.matrixUp(j,i) = matrix(j,i);
    }
    for(size_t i = 0; i < L; ++i)
    {
        for(size_t j = 0; j < L; ++j) expAlphaK2s.matrixDn(j,i) = matrix(j+L,i+L);
    }

    return expAlphaK2s;
}

NiupNidn HubbardSOC_icf::returnExpMinusAlphaV(double alpha, const std::string &decompType)
{
    return NiupNidn(alpha, decompType, U, mu, hx, hy, hz);
}

double HubbardSOC_icf::getMemory() const
{
    double mem(0.0);
    mem += 8.0+8.0;
    mem += K.getMemory();
    mem += mu.getMemory()+hx.getMemory()+hy.getMemory()+hz.getMemory()+U.getMemory();
    mem += 1.0+ KEigenValue.getMemory()+KEigenVector.getMemory();
    return mem;
}

HubbardSOC_icf::HubbardSOC_icf(const HubbardSOC_icf &x) { }

HubbardSOC_icf &HubbardSOC_icf::operator=(const HubbardSOC_icf &x) { return *this; }