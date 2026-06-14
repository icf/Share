//
// Created by Hao Shi on 1/12/18.
//

#include "afqmclab.h"
#include "../include/AuxMatrix.h"

using namespace std;
using namespace tensor_hao;

#define pi 3.14159265358979324

//e^{-dt q_{i_up,j_dn}}: q_{i_up,j_dn}=-c^dagger_{i,up} c^dagger_{j,dn} c_{i,dn} c_{j,up}
AuxMatrix::AuxMatrix() {}

AuxMatrix::AuxMatrix(const string &filename)
{
    //read parameters
    readModel(filename);
    //defaultModel();
}

AuxMatrix::AuxMatrix(const AuxMatrix &x) { copy_deep(x); }

AuxMatrix::AuxMatrix(AuxMatrix&&x) { move_deep(x); }

AuxMatrix::~AuxMatrix() { }

AuxMatrix &AuxMatrix::operator=(const AuxMatrix &x)  { copy_deep(x); return *this; }

AuxMatrix &AuxMatrix::operator=(AuxMatrix &&x) { move_deep(x); return *this; }

const size_t &AuxMatrix::getNumberOfAuxMatrix() const { return NumberOfAuxMatrix; }

const complex<double> &AuxMatrix::getQij_vec(size_t i) const { return Qij_vec[i]; }

const string &AuxMatrix::getDecompType() const { return decompType; }

std::complex<double> AuxMatrix::calculateAuxForce(const AuxMatrixAux &aux, const AuxMatrixForce &force)
{
    complex<double> auxForce(0,0);
    return auxForce;
}

AuxMatrixAux AuxMatrix::sampleAuxFromForce(const AuxMatrixForce &force) const
{
    AuxMatrixAux aux(NumberOfAuxMatrix);
    double expPlus, expMinus, prob;

    for(size_t i = 0; i < NumberOfAuxMatrix; ++i)
    {
        expMinus = exp( -force.QijForce(i) );
        expPlus  = exp(  force.QijForce(i) );
        prob = expMinus / (expMinus + expPlus);

        if( uniformHao() < prob ) aux.QijAux(i) = -1;
        else aux.QijAux(i)=1;

        //icf test
        //aux.QijAux(i) = -1;
    }

    return aux;
}

complex<double> AuxMatrix::logProbOfAuxFromForce(const AuxMatrixAux &aux, const AuxMatrixForce &force) const
{
    complex<double> logProb(0, 0);

    //U: exp( x*force ) / ( exp(force)+exp(-force) )
    for(size_t i=0; i<L; i++)
    {
        logProb += aux.QijAux(i)*force.QijForce(i) - log( exp(force.QijForce(i)) + exp(-force.QijForce(i)) );
    }

    return logProb;
}

AuxMatrixSample AuxMatrix::getTwoBodySampleFromAux(const AuxMatrixAux &aux) const
{
    AuxMatrixSample twoBodySample(NumberOfAuxMatrix, 2*L);

    complex<double> logw=0.0;
    logw = log(0.5)*L;
    // for(int i=1-1; i<=NumberOfAuxMatrix-1; i++){
    //     logw += -1.0 * log( 2 );
    // }

    twoBodySample.logw = logw;

    setTwoBodySampleMatrix(twoBodySample, aux);

    return twoBodySample;
}

const LogHop AuxMatrix::getTwoBodySampleFromAuxForce_LogHopType(const AuxMatrixAux &aux, const AuxMatrixForce &force) const
{
    AuxMatrixSample twoBodySample = getTwoBodySampleFromAux(aux);
    twoBodySample.logw = twoBodySample.logw - logProbOfAuxFromForce(aux, force);
    //
    LogHop twoBodySample_LogHop;
    twoBodySample_LogHop.matrix=twoBodySample.matrix;
    twoBodySample_LogHop.logw=twoBodySample.logw;

    return twoBodySample_LogHop;
}

const LogHop AuxMatrix::getTwoBodySampleFromAux_LogHopType(const AuxMatrixAux &aux) const
{
    AuxMatrixSample twoBodySample = getTwoBodySampleFromAux(aux);
    //
    LogHop twoBodySample_LogHop;
    twoBodySample_LogHop.matrix=twoBodySample.matrix;
    twoBodySample_LogHop.logw=twoBodySample.logw;

    return twoBodySample_LogHop;
}

AuxMatrixSample AuxMatrix::getTwoBodySampleFromAuxForce(const AuxMatrixAux &aux, const AuxMatrixForce &force) const
{
    AuxMatrixSample twoBodySample = getTwoBodySampleFromAux(aux);

    twoBodySample.logw = twoBodySample.logw - logProbOfAuxFromForce(aux, force);

    return twoBodySample;
}

double AuxMatrix::getMemory()
{
    double mem(0.0);
    return mem;
}

void AuxMatrix::copy_deep(const AuxMatrix &x)
{
    L = x.L;
    NumberOfAuxMatrix = x.NumberOfAuxMatrix;
    site_i = x.site_i;
    site_j = x.site_j;
    //
    Qij_vec = x.Qij_vec;
    //
    decompType = x.decompType;
}

void AuxMatrix::move_deep(AuxMatrix &x)
{
    L = move( x.L );
    NumberOfAuxMatrix = move( x.NumberOfAuxMatrix );
    site_i = move( x.site_i );
    site_j = move( x.site_j );
    //
    Qij_vec = move( x.Qij_vec );
    //
    decompType = move( x.decompType );
}

void AuxMatrix::defaultModel()
{
    L=10;
    decompType="densitySpin";

    site_i.resize(NumberOfAuxMatrix);
    site_j.resize(NumberOfAuxMatrix);

    Qij_vec.resize(NumberOfAuxMatrix); 
}

void AuxMatrix::readModel(const string &filename)
{
    int N;
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
    //
    site_i.resize(NumberOfAuxMatrix); readFile(site_i.size(), site_i.data(), file);
    site_j.resize(NumberOfAuxMatrix); readFile(site_j.size(), site_j.data(), file);
    //
    Qij_vec.resize(NumberOfAuxMatrix); readFile(Qij_vec.size(), Qij_vec.data(), file);
    //

    file.close();

}



void AuxMatrix::setTwoBodySampleMatrix(AuxMatrixSample &twoBodySample, const AuxMatrixAux &aux) const
{
    if( decompType == "densitySpin" )
    {
        //Set zero
        TensorHao<complex<double>,2> &matrix = twoBodySample.matrix;
        matrix = complex<double>(0.0, 0.0);
        TensorHao<complex<double>,2> logMatrix(2*L,2*L);
        TensorHao<complex<double>,2> matrixTemp(2*L,2*L);matrixTemp=0.0;
        logMatrix = complex<double>(0.0, 0.0);
        //
        for(size_t i = 0; i < NumberOfAuxMatrix; ++i)
        {
            int ki=site_i(i);
            int kj=site_j(i);
            logMatrix(ki,kj) += (double(aux.QijAux(i))*Qij_vec[i]); // - log(cosh(Qij_vec[i]));
            logMatrix(ki+L,kj+L) += (double(aux.QijAux(i))*(-1.0)*Qij_vec[i]); // - log(cosh(Qij_vec[i]));
            //matrix(ki,kj) = exp(double(aux.QijAux(i))*Qij_vec[i]); // - log(cosh(Qij_vec[i]));
            //matrix(ki+L,kj+L) = exp(double(aux.QijAux(i))*(-1.0)*Qij_vec[i]); // - log(cosh(Qij_vec[i]));
        }
        //
        // TensorHao<std::complex<double>,2> KpEigenVector = logMatrix;
        // TensorHao<double,1> KpEigenValue(2*L);
        // BL_NAME(eigen)(KpEigenVector, KpEigenValue);

        // //
        // for(int i=1-1; i<=2*L-1; i++){
        //     matrixTemp(i,i)=exp(KpEigenValue(i));
        // }
        // TensorHao<std::complex<double>,2> matrixTemp2(2*L,2*L); matrixTemp2=0.0;
        // BL_NAME(gmm)( matrixTemp, KpEigenVector, matrixTemp2, 'N', 'C' );
        // BL_NAME(gmm)( KpEigenVector, matrixTemp2, matrixTemp);

        matrix=logMatrix;

    }
}