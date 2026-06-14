//
// Created by boruoshihao on 5/23/17.
//
#include "afqmclab.h"
#include "../../include/twoBodyOperator_icf/svd_sym.h"

using namespace std;
using namespace tensor_hao;

#define pi 3.14159265358979324

// SVD_sym::SVD_sym():dt(0), sqrtMinusDt(0), svdNumber(0),svdBg(nullptr) { }
SVD_sym::SVD_sym():dt(0), sqrtMinusDt(0), svdNumber(0) { }

SVD_sym::SVD_sym(double dt,
                           const TensorHao<complex<double>, 2> &svdVecs,
                           const TensorHao<complex<double>, 1> &svdBg)
{
    SVD_sym::dt = dt;
    sqrtMinusDt = sqrt( -dt*complex<double>(1.0, 0.0) );
    svdNumber = svdVecs.rank(1);
    initialSqrtMinusDtSVDVecs(svdVecs);
    SVD_sym::svdBg = svdBg;
}

SVD_sym::SVD_sym(double dt,
                           const TensorHao<complex<double>, 2> &svdVecs,
                           const TensorHao<complex<double>, 1> &svdBg,
                           int conjugate)
{
    SVD_sym::dt = dt;
    if( conjugate==-1 ){
        sqrtMinusDt = conj(sqrt( -dt*complex<double>(1.0, 0.0) ));
    }else{
        sqrtMinusDt = sqrt( -dt*complex<double>(1.0, 0.0) );
    }
    svdNumber = svdVecs.rank(1);
    initialSqrtMinusDtSVDVecs(svdVecs);
    SVD_sym::svdBg = svdBg;
}

SVD_sym::SVD_sym(const SVD_sym &x) { copy_deep(x); }

SVD_sym::SVD_sym(SVD_sym &&x) { move_deep(x); }

SVD_sym::~SVD_sym() { }

SVD_sym &SVD_sym::operator = (const SVD_sym &x) { copy_deep(x); return *this; }

SVD_sym &SVD_sym::operator = (SVD_sym &&x) { move_deep(x); return *this; }

double SVD_sym::getDt() const { return dt; }

const TensorHao<complex<double>, 2> &SVD_sym::getSqrtMinusDtSVDVecs() const { return sqrtMinusDtSVDVecs; }

const TensorHao<complex<double>, 1> SVD_sym::getSVDBg() const { return svdBg; }

size_t SVD_sym::getSVDNumber() const { return svdNumber; }

const complex<double> &SVD_sym::getSqrtMinusDt() const { return sqrtMinusDt; }

size_t SVD_sym::returnBasisSize() const { return sqrtMinusDtSVDVecs.rank(0); }

SVDForce SVD_sym::readForce(const std::string &filename) const
{
    SVDForce force(svdNumber);

    if( !checkFile(filename) ) force = 0.0;
    else readFile( force.size(), force.data(), filename );

    return force;
}

SVDAux SVD_sym::sampleAuxFromForce(const SVDForce &force) const
{
    if( svdNumber != force.size() ) { cout<<"Error!!! Force size does not consistent with svdNumber! "<<force.size()<<endl; exit(1); }

    SVDAux svdAux( svdNumber );
    for(size_t i = 0; i < svdNumber; ++i)  svdAux(i) = gaussianHao() + force(i);

    return svdAux;
}

complex<double> SVD_sym::logProbOfAuxFromForce(const SVDAux &aux, const SVDForce &force) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<aux.size()<<endl; exit(1); }
    if( svdNumber != force.size() ) { cout<<"Error!!! Force size does not consistent with svdNumber! "<<force.size()<<endl; exit(1); }

    // Product: 1/sqrt(2.0*Pi) * Exp( -(x-force)^2 / 2.0 )
    complex<double> logProb(0,0), auxMinusForceSquare(0,0), tmp;
    for(size_t i = 0; i < svdNumber; ++i)
    {
        tmp = aux(i)-force(i);
        auxMinusForceSquare += tmp*tmp;
    }
    logProb = -0.5*log(2.0*pi)*svdNumber - 0.5*auxMinusForceSquare;

    return logProb;
}

SVDSample SVD_sym::getTwoBodySampleFromAux_test(const SVDAux &aux) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<aux.size()<<endl; exit(1); }

    SVDSample svdSample( sqrtMinusDtSVDVecs.rank(0) );

    // Product: 1/sqrt(2.0*Pi) * Exp( -x^2 / 2.0 ) * Exp( -sqrt(-dt) x*B )
    complex<double> aux2Sum(0,0), auxBSum(0,0);
    cout<<"setTwoBodySampleMatrix sss"<<endl;
    for(size_t i = 0; i < svdNumber; ++i)
    {
        aux2Sum += aux(i)*aux(i);
        cout<<i<<endl;
        cout<<"svdBg->operator()(i): "<<svdBg(i)<<endl;
        auxBSum += aux(i) * svdBg(i);
        cout<<i<<" end"<<endl;
    }
    cout<<"setTwoBodySampleMatrix"<<endl;
    svdSample.logw = -0.5*log(2.0*pi)*svdNumber - 0.5*aux2Sum - sqrtMinusDt*auxBSum;
    
    setTwoBodySampleMatrix(svdSample, aux);

    return svdSample;
}


SVDSample SVD_sym::getTwoBodySampleFromAux(const SVDAux &aux) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<aux.size()<<endl; exit(1); }

    SVDSample svdSample( sqrtMinusDtSVDVecs.rank(0) );

    // Product: 1/sqrt(2.0*Pi) * Exp( -x^2 / 2.0 ) * Exp( -sqrt(-dt) x*B )
    complex<double> aux2Sum(0,0), auxBSum(0,0);
    for(size_t i = 0; i < svdNumber; ++i)
    {
        aux2Sum += aux(i)*aux(i);
        auxBSum += aux(i) * svdBg(i);
    }
    svdSample.logw = -0.5*log(2.0*pi)*svdNumber - 0.5*aux2Sum - sqrtMinusDt*auxBSum;
    
    setTwoBodySampleMatrix(svdSample, aux);

    return svdSample;
}

SVDSample2s SVD_sym::getTwoBodySampleFromAux2s(const SVDAux &aux) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<aux.size()<<endl; exit(1); }
    
    size_t L = sqrtMinusDtSVDVecs.rank(0); 
    if(L != 2 * (L/2) ){
        cout<<"Error: L != 2 * (L/2) --> getTwoBodySampleFromAux is not allowed for SVDSample2s"<<endl;
        exit(1);
    }
    SVDSample2s svdSample2s( L/2 );

    // Product: 1/sqrt(2.0*Pi) * Exp( -x^2 / 2.0 ) * Exp( -sqrt(-dt) x*B )
    complex<double> aux2Sum(0,0), auxBSum(0,0);
    for(size_t i = 0; i < svdNumber; ++i)
    {
        aux2Sum += aux(i)*aux(i);
        auxBSum += aux(i) * svdBg(i);
    }
    svdSample2s.logw = -0.5*log(2.0*pi)*svdNumber - 0.5*aux2Sum - sqrtMinusDt*auxBSum;
    
    setTwoBodySampleMatrix(svdSample2s, aux);

    return svdSample2s;
}

/////////////////////////////////////////////////
TensorHao<complex<double>,2> SVD_sym::getSqrtMinusDtSVDVecsMatrix(int i)
{
    size_t L = sqrtMinusDtSVDVecs.rank(0); 
    TensorHao<complex<double>,2> matrix(L,L); matrix=0.0;
    //
    for(int j=1-1; j<=L-1; j++){
        matrix(j,j) = sqrtMinusDtSVDVecs(j,i);
    }
    return matrix;
}
/////////////////////////////////////////////////

SVDSample SVD_sym::getTwoBodySampleFromAux_icf_fixedForce(const SVDAux &aux, const SVDForce &force) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<aux.size()<<endl; exit(1); }

    SVDSample svdSample( sqrtMinusDtSVDVecs.rank(0) );

    // Product: 1/sqrt(2.0*Pi) * Exp( -x^2 / 2.0 ) * Exp( -sqrt(-dt) x*B )
    complex<double> aux2Sum(0,0), auxBSum(0,0);
    for(size_t i = 0; i < svdNumber; ++i)
    {
        aux2Sum += aux(i)*aux(i)-abs(aux(i)-force(i))*abs(aux(i)-force(i));
        auxBSum += aux(i) * svdBg(i);
    }
    svdSample.logw = - 0.5*aux2Sum - sqrtMinusDt*auxBSum;

    setTwoBodySampleMatrix(svdSample, aux);

    return svdSample;
}

SVDSample2s SVD_sym::getTwoBodySampleFromAux_icf_fixedForce2s(const SVDAux &aux, const SVDForce &force) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<aux.size()<<endl; exit(1); }
    
    size_t L = sqrtMinusDtSVDVecs.rank(0); 
    if(L != 2 * (L/2) ){
        cout<<"Error: L != 2 * (L/2) --> getTwoBodySampleFromAux_icf_fixedForce is not allowed for SVDSample2s"<<endl;
        exit(1);
    }
    SVDSample2s svdSample2s( L/2 );

    // Product: 1/sqrt(2.0*Pi) * Exp( -x^2 / 2.0 ) * Exp( -sqrt(-dt) x*B )
    complex<double> aux2Sum(0,0), auxBSum(0,0);
    for(size_t i = 0; i < svdNumber; ++i)
    {
        aux2Sum += aux(i)*aux(i)-abs(aux(i)-force(i))*abs(aux(i)-force(i));
        auxBSum += aux(i) * svdBg(i);
    }
    svdSample2s.logw = - 0.5*aux2Sum - sqrtMinusDt*auxBSum;

    setTwoBodySampleMatrix(svdSample2s, aux);

    return svdSample2s;
}

SVDSample SVD_sym::getTwoBodySampleFromAux_icf_ConstForce(const SVDAux &aux) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<aux.size()<<endl; exit(1); }

    SVDSample svdSample( sqrtMinusDtSVDVecs.rank(0) );

    // Product: 1/sqrt(2.0*Pi) * Exp( -x^2 / 2.0 ) * Exp( -sqrt(-dt) x*B )
    complex<double> aux2Sum(0,0), auxBSum(0,0);
    for(size_t i = 0; i < svdNumber; ++i)
    {
        aux2Sum += 0.0;
        auxBSum += aux(i) * svdBg(i);
    }
    svdSample.logw = - 0.5*aux2Sum - sqrtMinusDt*auxBSum;

    setTwoBodySampleMatrix(svdSample, aux);

    return svdSample;
}

SVDSample2s SVD_sym::getTwoBodySampleFromAux_icf_ConstForce2s(const SVDAux &aux) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<aux.size()<<endl; exit(1); }
    
    size_t L = sqrtMinusDtSVDVecs.rank(0); 
    if(L != 2 * (L/2) ){
        cout<<"Error: L != 2 * (L/2) --> getTwoBodySampleFromAux_icf_ConstForce is not allowed for SVDSample2s"<<endl;
        exit(1);
    }
    SVDSample2s svdSample2s( L/2 );

    // Product: 1/sqrt(2.0*Pi) * Exp( -x^2 / 2.0 ) * Exp( -sqrt(-dt) x*B )
    complex<double> aux2Sum(0,0), auxBSum(0,0);
    for(size_t i = 0; i < svdNumber; ++i)
    {
        aux2Sum += 0.0;
        auxBSum += aux(i) * svdBg(i);
    }
    svdSample2s.logw = - 0.5*aux2Sum - sqrtMinusDt*auxBSum;

    setTwoBodySampleMatrix(svdSample2s, aux);

    return svdSample2s;
}

SVDSample SVD_sym::getTwoBodySampleFromAuxForce(const SVDAux &aux, const SVDForce &force) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<aux.size()<<endl; exit(1); }
    if( svdNumber != force.size() ) { cout<<"Error!!! Force size does not consistent with svdNumber! "<<force.size()<<endl; exit(1); }

    SVDSample svdSample( sqrtMinusDtSVDVecs.rank(0) );

    // Product: Exp( force^2/2 - x*force ) Exp( -sqrt(-dt) x*B )
    complex<double> force2Sum(0,0), auxForce(0,0),auxBSum(0,0);
    for(size_t i = 0; i < svdNumber; ++i)
    {
        force2Sum += force(i)*force(i);
        auxForce += aux(i)*force(i);
        auxBSum += aux(i) * svdBg(i);
    }
    svdSample.logw =  0.5*force2Sum - auxForce - sqrtMinusDt*auxBSum;

    setTwoBodySampleMatrix(svdSample, aux);

    return svdSample;
}

SVDSample2s SVD_sym::getTwoBodySampleFromAuxForce2s(const SVDAux &aux, const SVDForce &force) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<aux.size()<<endl; exit(1); }
    if( svdNumber != force.size() ) { cout<<"Error!!! Force size does not consistent with svdNumber! "<<force.size()<<endl; exit(1); }

    size_t L = sqrtMinusDtSVDVecs.rank(0); 
    if(L != 2 * (L/2) ){
        cout<<"Error: L != 2 * (L/2) --> getTwoBodySampleFromAuxForce is not allowed for SVDSample2s"<<endl;
        exit(1);
    }
    SVDSample2s svdSample2s( L/2 );

    // Product: Exp( force^2/2 - x*force ) Exp( -sqrt(-dt) x*B )
    complex<double> force2Sum(0,0), auxForce(0,0),auxBSum(0,0);
    for(size_t i = 0; i < svdNumber; ++i)
    {
        force2Sum += force(i)*force(i);
        auxForce += aux(i)*force(i);
        auxBSum += aux(i) * svdBg(i);
    }
    svdSample2s.logw =  0.5*force2Sum - auxForce - sqrtMinusDt*auxBSum;

    setTwoBodySampleMatrix(svdSample2s, aux);

    return svdSample2s;
}

double SVD_sym::getMemory() const
{
    return 8.0+16.0+8.0+sqrtMinusDtSVDVecs.getMemory()+8.0;
}

void SVD_sym::copy_deep(const SVD_sym &x)
{
    dt = x.dt;
    sqrtMinusDt = x.sqrtMinusDt;
    svdNumber = x.svdNumber;
    sqrtMinusDtSVDVecs = x.sqrtMinusDtSVDVecs;
    svdBg = x.svdBg;
}

void SVD_sym::move_deep(SVD_sym &x)
{
    dt = x.dt;
    sqrtMinusDt = x.sqrtMinusDt;
    svdNumber = x.svdNumber;
    sqrtMinusDtSVDVecs = move( x.sqrtMinusDtSVDVecs );
    svdBg = move( x.svdBg );
}

void SVD_sym::initialSqrtMinusDtSVDVecs(const tensor_hao::TensorHao<complex<double>, 2> &svdVecs)
{
    sqrtMinusDtSVDVecs.resize( svdVecs.getRank() );
    complex<double> *p0 = sqrtMinusDtSVDVecs.data();
    const complex<double> *p1 = svdVecs.data();

    //There is a cubic scaling, can be faster by OpenMP
    size_t svdVecsSize = svdVecs.size();
    for(size_t i = 0; i < svdVecsSize; ++i) p0[i] = p1[i] * sqrtMinusDt;
}

void SVD_sym::setTwoBodySampleMatrix(SVDSample &svdSample, const SVDAux &aux) const
{
    //Calculate aux * sqrtMinusDt * svdVecs
    size_t L = sqrtMinusDtSVDVecs.rank(0); 
    size_t choleskyNum = sqrtMinusDtSVDVecs.rank(1); 

    TensorHao<complex<double>, 2> matrix_temp(L,L); matrix_temp=0.0;
    for(int i=1-1; i<=choleskyNum-1; i++){
        for(int j=1-1; j<=L-1; j++){
            matrix_temp(j,j) +=  sqrtMinusDtSVDVecs(j,i) * aux(i);
        }
    }
    svdSample.matrix = matrix_temp;
}

void SVD_sym::setTwoBodySampleMatrix(SVDSample2s &svdSample2s, const SVDAux &aux) const
{
    //Calculate aux * sqrtMinusDt * svdVecs
    size_t L = sqrtMinusDtSVDVecs.rank(0); 
    size_t choleskyNum = sqrtMinusDtSVDVecs.rank(1); 

    if(L != 2 * (L/2) ){
        cout<<"Error: L != 2 * (L/2) --> setTwoBodySampleMatrix is not allowed for SVDSample2s"<<endl;
        exit(1);
    }
    TensorHao<complex<double>, 2> matrixUp_temp(L/2,L/2); matrixUp_temp=0.0;
    TensorHao<complex<double>, 2> matrixDn_temp(L/2,L/2); matrixDn_temp=0.0;
    for(int i=1-1; i<=choleskyNum-1; i++){
        for(int j=1-1; j<=L/2-1; j++){
            matrixUp_temp(j,j) +=  sqrtMinusDtSVDVecs(j,i) * aux(i);
            matrixDn_temp(j,j) +=  sqrtMinusDtSVDVecs(j + L/2,i) * aux(i);
        }
    }
    svdSample2s.matrixUp = matrixUp_temp;
    svdSample2s.matrixDn = matrixDn_temp;
}
