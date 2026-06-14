//
// Created by boruoshihao on 12/28/16.
//

#include "../include/hop_AuxMatrix.h"

using namespace std;
using namespace tensor_hao;

Hop_AuxMatrix::Hop_AuxMatrix() { }

Hop_AuxMatrix::Hop_AuxMatrix(const string &filename) 
{ 
    //read parameters
    readModel(filename);
}

Hop_AuxMatrix::Hop_AuxMatrix(const Hop_AuxMatrix &x) { copy_deep(x); }

Hop_AuxMatrix::Hop_AuxMatrix(Hop_AuxMatrix &&x) { move_deep(x); }

Hop_AuxMatrix::~Hop_AuxMatrix() { }

Hop_AuxMatrix &Hop_AuxMatrix::operator=(const Hop_AuxMatrix &x)  { copy_deep(x); return *this; }

Hop_AuxMatrix &Hop_AuxMatrix::operator=(Hop_AuxMatrix &&x) { move_deep(x); return *this; }

size_t Hop_AuxMatrix::getL() const { return matrix.rank(0); }


void Hop_AuxMatrix::readModel(const string &filename)
{
    int L;
    int N;
    int NumberOfAuxMatrix;
    std::string decompType;
    TensorHao<std::complex<double>,2> t;
    //
    ifstream file;
    file.open(filename, ios::in);
    if ( ! file.is_open() ) {cout << "Error opening file in File!!! "<<filename<<endl; exit(1);}

    readFile( L, file );
    readFile( N, file );
    readFile(NumberOfAuxMatrix, file);
    readFile(decompType, file);

    t.resize(2*L,2*L); readFile( t.size(),  t.data(),  file );

    file.close();

    //
    matrix.resize(2*L,2*L);
    matrix=t;
    logw=0.0;
}

LogHop Hop_AuxMatrix::getLogHop()
{
    LogHop loghop(matrix.rank(0));
    loghop.matrix=matrix;
    loghop.logw=logw;
    //
    return loghop;
}

Hop Hop_AuxMatrix::getHop(std::string flag, size_t taylorOrder, double accuracy, size_t baseTaylorOrder)
{
    LogHop loghop(matrix.rank(0));
    loghop.matrix=matrix;
    loghop.logw=logw;
    //
    int L=matrix.rank(0);
    //
    SD walkerIdentity(L,L);
    TensorHao<complex<double>,2> identity(L,L); identity=0.0;
    for(size_t i = 0; i < L; ++i)
    {
        identity(i,i) = 1.0; 
    }
    walkerIdentity.wfRef()=identity;
    SD walkerTemp=walkerIdentity;
    //
    Hop hop(matrix.rank(0));
    //
    LogHopSDOperation oneBodyWalkerOperation;
    oneBodyWalkerOperation.reset(flag, taylorOrder, accuracy, baseTaylorOrder);
    oneBodyWalkerOperation.applyToRight(loghop, walkerIdentity, walkerTemp);
    //
    hop.matrix = walkerTemp.getWf();
    hop.logw = walkerTemp.getLogw();
    //
    return hop;
}

Hop Hop_AuxMatrix::getHop()
{
    LogHop loghop(matrix.rank(0));
    loghop.matrix=matrix;
    loghop.logw=logw;
    //
    int L=matrix.rank(0);
    //
    SD walkerIdentity(L,L);
    TensorHao<complex<double>,2> identity(L,L); identity=0.0;
    for(size_t i = 0; i < L; ++i)
    {
        identity(i,i) = 1.0; 
    }
    walkerIdentity.wfRef()=identity;
    SD walkerTemp=walkerIdentity;
    //
    Hop hop(matrix.rank(0));
    //
    LogHopSDOperation oneBodyWalkerOperation;
    oneBodyWalkerOperation.applyToRight(loghop, walkerIdentity, walkerTemp);
    //
    hop.matrix = walkerTemp.getWf();
    hop.logw = walkerTemp.getLogw();
    //
    return hop;
}


double Hop_AuxMatrix::getMemory() const
{
    return 16.0+matrix.getMemory();
}

void Hop_AuxMatrix::copy_deep(const Hop_AuxMatrix &x)
{
    logw = x.logw;
    matrix = x.matrix;
}

void Hop_AuxMatrix::move_deep(Hop_AuxMatrix &x)
{
    logw = x.logw;
    matrix = move( x.matrix );
}
