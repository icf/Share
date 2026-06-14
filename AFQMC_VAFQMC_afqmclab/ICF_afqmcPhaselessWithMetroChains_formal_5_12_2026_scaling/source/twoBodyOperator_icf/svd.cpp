//
// Created by boruoshihao on 5/23/17.
//
#include "afqmclab.h"
#include "../../include/twoBodyOperator_icf/svd.h"
#include "../../include/afqmcPhaselessDefine.h"
#include <unordered_map>

using namespace std;
using namespace tensor_hao;

#define pi 3.14159265358979324

// SVD::SVD():dt(0), sqrtMinusDt(0), svdNumber(0),svdBg(nullptr) { }
SVD::SVD():dt(0), sqrtMinusDt(0), svdNumber(0) { }

SVD::SVD(double dt,
                           const TensorHaoMPIRef<complex<double>, 3> &svdVecs,
                           const TensorHao<complex<double>, 2> &svdVecs_U0,
                           const TensorHao<complex<double>, 3> &svdVecs_D,
                           const TensorHao<complex<double>, 2> &svdVecs_Vdagger0,
                           const TensorHao<complex<double>, 1> &svdBg, 
                           bool Hamiltonian_spin_flag_input,
                           bool cutOffFlag)
{
    Hamiltonian_spin_flag = Hamiltonian_spin_flag_input;
    SVD::dt = dt;
    sqrtMinusDt = sqrt( -dt*complex<double>(1.0, 0.0) );
    svdNumber = svdVecs.rank(2);
    initialSqrtMinusDtSVDVecs(svdVecs, svdVecs_U0, svdVecs_D, svdVecs_Vdagger0, cutOffFlag);
    
    SVD::svdBg = svdBg;
}

SVD::SVD(double dt,
                           const TensorHaoMPIRef<complex<double>, 3> &svdVecs,
                           const TensorHao<complex<double>, 2> &svdVecs_U0,
                           const TensorHao<complex<double>, 3> &svdVecs_D,
                           const TensorHao<complex<double>, 2> &svdVecs_Vdagger0,
                           const TensorHao<complex<double>, 1> &svdBg,
                           bool Hamiltonian_spin_flag_input,
                           int conjugate, 
                           bool cutOffFlag)
{
    Hamiltonian_spin_flag = Hamiltonian_spin_flag_input;
    SVD::dt = dt;
    if( conjugate==-1 ){
        sqrtMinusDt = conj(sqrt( -dt*complex<double>(1.0, 0.0) ));
    }else{
        sqrtMinusDt = sqrt( -dt*complex<double>(1.0, 0.0) );
    }
    svdNumber = svdVecs.rank(2);
    initialSqrtMinusDtSVDVecs(svdVecs, svdVecs_U0, svdVecs_D, svdVecs_Vdagger0, cutOffFlag);
    SVD::svdBg = svdBg;
}

SVD::SVD(const SVD &x) { copy_deep(x); }

SVD::SVD(SVD &&x) { move_deep(x); }

SVD::~SVD() { }

SVD &SVD::operator = (const SVD &x) { copy_deep(x); return *this; }

SVD &SVD::operator = (SVD &&x) { move_deep(x); return *this; }

double SVD::getDt() const { return dt; }

const TensorHaoMPIRef<complex<double>, 3> &SVD::getSqrtMinusDtSVDVecs() const { return sqrtMinusDtSVDVecs; }
const TensorHao<complex<double>, 3> &SVD::getSqrtMinusDtSVDVecs_Dup() const { return sqrtMinusDtSVDVecs_Dup; }
const TensorHao<complex<double>, 3> &SVD::getSqrtMinusDtSVDVecs_Ddn() const { return sqrtMinusDtSVDVecs_Ddn; }
const TensorHao<complex<double>, 3> &SVD::getSqrtMinusDtSVDVecs_D() const { return sqrtMinusDtSVDVecs_D; }

const TensorHao<complex<double>, 1> SVD::getSVDBg() const { return svdBg; }

void SVD::updateBG(const TensorHao<complex<double>, 1> &bg) {
    if(bg.size() != svdNumber) {
        cout<<"Error!!! Background size is not svdNumber!"<<endl;
        exit(1);
    }
    svdBg = bg;
}

size_t SVD::getSVDNumber() const { return svdNumber; }
// ///////////////////////
size_t SVD::getSVD_U0upRank0() const { return U0up.rank(1); }
size_t SVD::getSVD_U0dnRank0() const { return U0dn.rank(1); }
size_t SVD::getSVD_U0Rank0() const { return U0.rank(1); }
size_t SVD::getSVD_Vdagger0upRank1() const { return Vdagger0up.rank(1); }
size_t SVD::getSVD_Vdagger0dnRank1() const { return Vdagger0dn.rank(1); }
size_t SVD::getSVD_Vdagger0Rank1() const { return Vdagger0.rank(1); }
size_t SVD::getSVD_DupRank() const { return sqrtMinusDtSVDVecs_Dup.rank(1); }
size_t SVD::getSVD_DdnRank() const { return sqrtMinusDtSVDVecs_Ddn.rank(1); }
size_t SVD::getSVD_DRank() const { return sqrtMinusDtSVDVecs_D.rank(1); }
// ///////////////////////
const complex<double> &SVD::getSqrtMinusDt() const { return sqrtMinusDt; }

size_t SVD::returnBasisSize() const { 
    #ifdef USE_SD
        return sqrtMinusDtSVDVecs.rank(0); 
    #else
        return sqrtMinusDtSVDVecsUp.rank(0); 
    #endif
}

SVDForce SVD::readForce(const std::string &filename) const
{
    SVDForce force(svdNumber);

    if( !checkFile(filename) ) force = 0.0;
    else readFile( force.size(), force.data(), filename );

    return force;
}

SVDAux SVD::sampleAuxFromForce(const SVDForce &force) const
{
    if( svdNumber != force.size() ) { cout<<"Error!!! Force size does not consistent with svdNumber! "<<svdNumber<<" "<<force.size()<<endl; exit(1); }

    SVDAux svdAux( svdNumber );
    for(size_t i = 0; i < svdNumber; ++i)  svdAux(i) = gaussianHao() + force(i);

    return svdAux;
}

SVDAux SVD::sampleAuxFromForce_localUpdate(const SVDForce &force, const SVDAux &auxInput, vector<int> flip_i_vec) const
{
    if( svdNumber != force.size() ) { cout<<"Error!!! Force size does not consistent with svdNumber! "<<svdNumber<<" "<<force.size()<<endl; exit(1); }

    SVDAux svdAux( svdNumber );
    svdAux = auxInput;
    for(int flip_i: flip_i_vec){
        svdAux(flip_i) = gaussianHao() + force(flip_i);
    }

    return svdAux;
}

complex<double> SVD::logProbOfAuxFromForce(const SVDAux &aux, const SVDForce &force) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<svdNumber<<" "<<aux.size()<<endl; exit(1); }
    if( svdNumber != force.size() ) { cout<<"Error!!! Force size does not consistent with svdNumber! "<<svdNumber<<" "<<force.size()<<endl; exit(1); }

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

SVDSample SVD::getTwoBodySampleFromAux_test(const SVDAux &aux) const
{
    // if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<aux.size()<<endl; exit(1); }

    // SVDSample svdSample( sqrtMinusDtSVDVecs.rank(0) );

    // // Product: 1/sqrt(2.0*Pi) * Exp( -x^2 / 2.0 ) * Exp( -sqrt(-dt) x*B )
    // complex<double> aux2Sum(0,0), auxBSum(0,0);
    // cout<<"setTwoBodySampleMatrix sss"<<endl;
    // for(size_t i = 0; i < svdNumber; ++i)
    // {
    //     aux2Sum += aux(i)*aux(i);
    //     cout<<i<<endl;
    //     cout<<"svdBg->operator()(i): "<<svdBg(i)<<endl;
    //     auxBSum += aux(i) * svdBg(i);
    //     cout<<i<<" end"<<endl;
    // }
    // cout<<"setTwoBodySampleMatrix"<<endl;
    // svdSample.logw = -0.5*log(2.0*pi)*svdNumber - 0.5*aux2Sum - sqrtMinusDt*auxBSum;
    
    // setTwoBodySampleMatrix(svdSample, aux);

    // return svdSample;
}


SVDSample SVD::getTwoBodySampleFromAux(const SVDAux &aux) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<aux.size()<<endl; exit(1); }

    int L = sqrtMinusDtSVDVecs.rank(0);
    if(!Hamiltonian_spin_flag){
        L = 2*L;
    }

    SVDSample svdSample( L );

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

complex<double> SVD::getTwoBodySample_logw_FromAux(const SVDAux &aux) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<aux.size()<<endl; exit(1); }
    
    size_t L = sqrtMinusDtSVDVecsUp.rank(0); 

    // Product: 1/sqrt(2.0*Pi) * Exp( -x^2 / 2.0 ) * Exp( -sqrt(-dt) x*B )
    complex<double> logw;
    complex<double> aux2Sum(0,0), auxBSum(0,0);
    for(size_t i = 0; i < svdNumber; ++i)
    {
        aux2Sum += aux(i)*aux(i);
        auxBSum += aux(i) * svdBg(i);
    }
    logw = -0.5*log(2.0*pi)*svdNumber - 0.5*aux2Sum - sqrtMinusDt*auxBSum;

    return logw;
}

complex<double> SVD::getTwoBodySample_logw_FromAux2s(const SVDAux &aux) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<aux.size()<<endl; exit(1); }
    
    size_t L = sqrtMinusDtSVDVecsUp.rank(0); 

    // Product: 1/sqrt(2.0*Pi) * Exp( -x^2 / 2.0 ) * Exp( -sqrt(-dt) x*B )
    complex<double> logw;
    complex<double> aux2Sum(0,0), auxBSum(0,0);
    for(size_t i = 0; i < svdNumber; ++i)
    {
        aux2Sum += aux(i)*aux(i);
        auxBSum += aux(i) * svdBg(i);
    }
    logw = -0.5*log(2.0*pi)*svdNumber - 0.5*aux2Sum - sqrtMinusDt*auxBSum;

    return logw;
}

SVDSample2s SVD::getTwoBodySampleFromAux2s(const SVDAux &aux) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<aux.size()<<endl; exit(1); }
    
    size_t L = sqrtMinusDtSVDVecsUp.rank(0); 
    SVDSample2s svdSample2s( L );

    // Product: 1/sqrt(2.0*Pi) * Exp( -x^2 / 2.0 ) * Exp( -sqrt(-dt) x*B )
    complex<double> aux2Sum(0,0), auxBSum(0,0);
    for(size_t i = 0; i < svdNumber; ++i)
    {
        aux2Sum += aux(i)*aux(i);
        auxBSum += aux(i) * svdBg(i);
    }
    svdSample2s.logw = -0.5*log(2.0*pi)*svdNumber - 0.5*aux2Sum - sqrtMinusDt*auxBSum;
    //
    setTwoBodySampleMatrix(svdSample2s, aux);

    return svdSample2s;
}

/////////////////////////////////////////////////
TensorHao<complex<double>,2> SVD::getSqrtMinusDtSVDVecsMatrix(int i) const
{
    size_t L = sqrtMinusDtSVDVecs.rank(0); 
    size_t N = sqrtMinusDtSVDVecs.rank(2);  // 获取第三维大小用于边界检查
    
    if(i < 0 || i >= static_cast<int>(N)) {
        cout << "Error!!! SVD::getSqrtMinusDtSVDVecsMatrix: Invalid index i = " << i 
             << " for matrix with size " << N << endl;
        exit(1);
    }
    
    TensorHao<complex<double>,2> matrix(L, L);
    
    // 使用指针访问优化内存访问
    complex<double> *matrix_data = matrix.data();
    const complex<double> *source_data = sqrtMinusDtSVDVecs.data();
    size_t slice_offset = i * L * L;
    
    for(size_t j = 0; j < L; j++) {
        for(size_t k = 0; k < L; k++) {
            size_t source_idx = slice_offset + j * L + k;
            matrix_data[j * L + k] = source_data[source_idx];
        }
    }
    
    return matrix;
}

TensorHao<complex<double>,2> SVD::getSqrtMinusDtSVDVecsMatrixUp(int i) const
{
    size_t L = sqrtMinusDtSVDVecsUp.rank(0); 
    size_t N = sqrtMinusDtSVDVecsUp.rank(2);  // 获取第三维大小用于边界检查
    
    if(i < 0 || i >= static_cast<int>(N)) {
        cout << "Error!!! SVD::getSqrtMinusDtSVDVecsMatrixUp: Invalid index i = " << i 
             << " for matrix with size " << N << endl;
        exit(1);
    }
    
    TensorHao<complex<double>,2> matrix(L, L);
    
    // 使用指针访问优化内存访问
    complex<double> *matrix_data = matrix.data();
    const complex<double> *source_data = sqrtMinusDtSVDVecsUp.data();
    size_t slice_offset = i * L * L;
    
    for(size_t j = 0; j < L; j++) {
        for(size_t k = 0; k < L; k++) {
            size_t source_idx = slice_offset + j * L + k;
            matrix_data[j * L + k] = source_data[source_idx];
        }
    }
    
    return matrix;
}

TensorHao<complex<double>,2> SVD::getSqrtMinusDtSVDVecsMatrixDn(int i) const
{
    size_t L = sqrtMinusDtSVDVecsDn.rank(0); 
    size_t N = sqrtMinusDtSVDVecsDn.rank(2);  // 获取第三维大小用于边界检查
    
    if(i < 0 || i >= static_cast<int>(N)) {
        cout << "Error!!! SVD::getSqrtMinusDtSVDVecsMatrixDn: Invalid index i = " << i 
             << " for matrix with size " << N << endl;
        exit(1);
    }
    
    TensorHao<complex<double>,2> matrix(L, L);
    
    // 使用指针访问优化内存访问
    complex<double> *matrix_data = matrix.data();
    const complex<double> *source_data = sqrtMinusDtSVDVecsDn.data();
    size_t slice_offset = i * L * L;
    
    for(size_t j = 0; j < L; j++) {
        for(size_t k = 0; k < L; k++) {
            size_t source_idx = slice_offset + j * L + k;
            matrix_data[j * L + k] = source_data[source_idx];
        }
    }
    
    return matrix;
}

TensorHao<complex<double>,2> SVD::getSqrtMinusDtSVDVecsMatrix_SVD(int i, string name) const
{
    static const unordered_map<string, int> nameMap = {
        {"U0up", 0}, {"Vdagger0up", 1}, {"U0dn", 2}, {"Vdagger0dn", 3},
        {"U0", 4}, {"Vdagger0", 5}, {"Dup", 6}, {"Ddn", 7}, {"D", 8}
    };

    auto it = nameMap.find(name);
    if (it == nameMap.end()) {
        cout << "Error!!! SVD::getSqrtMinusDtSVDVecsMatrix_SVD: name = " << name << " is not defined!" << endl;
        exit(1);
    }

    TensorHao<complex<double>,2> matrix;
    
    switch (it->second) {
        case 0: // "U0up"
            matrix = U0up;
            break;
        case 1: // "Vdagger0up"
            matrix = Vdagger0up;
            break;
        case 2: // "U0dn"
            matrix = U0dn;
            break;
        case 3: // "Vdagger0dn"
            matrix = Vdagger0dn;
            break;
        case 4: // "U0"
            matrix = U0;
            break;
        case 5: // "Vdagger0"
            matrix = Vdagger0;
            break;
        case 6: // "Dup"
            if(i >= 0 && i < static_cast<int>(sqrtMinusDtSVDVecs_Dup.rank(2))) {
                matrix.resize(sqrtMinusDtSVDVecs_Dup.rank(0), sqrtMinusDtSVDVecs_Dup.rank(1));
                // 使用指针访问优化内存访问模式
                complex<double> *matrix_data = matrix.data();
                const complex<double> *source_data = sqrtMinusDtSVDVecs_Dup.data();
                size_t slice_offset = i * sqrtMinusDtSVDVecs_Dup.rank(0) * sqrtMinusDtSVDVecs_Dup.rank(1);
                
                for(size_t j = 0; j < sqrtMinusDtSVDVecs_Dup.rank(0); j++) {
                    for(size_t k = 0; k < sqrtMinusDtSVDVecs_Dup.rank(1); k++) {
                        size_t source_idx = slice_offset + j * sqrtMinusDtSVDVecs_Dup.rank(1) + k;
                        matrix_data[j * matrix.rank(1) + k] = source_data[source_idx];
                    }
                }
            } else {
                cout << "Error!!! SVD::getSqrtMinusDtSVDVecsMatrix_SVD: Invalid index i = " << i 
                     << " for matrix Dup with size " << sqrtMinusDtSVDVecs_Dup.rank(2) << endl;
                exit(1);
            }
            break;
        case 7: // "Ddn"
            if(i >= 0 && i < static_cast<int>(sqrtMinusDtSVDVecs_Ddn.rank(2))) {
                matrix.resize(sqrtMinusDtSVDVecs_Ddn.rank(0), sqrtMinusDtSVDVecs_Ddn.rank(1));
                // 使用指针访问优化内存访问模式
                complex<double> *matrix_data = matrix.data();
                const complex<double> *source_data = sqrtMinusDtSVDVecs_Ddn.data();
                size_t slice_offset = i * sqrtMinusDtSVDVecs_Ddn.rank(0) * sqrtMinusDtSVDVecs_Ddn.rank(1);
                
                for(size_t j = 0; j < sqrtMinusDtSVDVecs_Ddn.rank(0); j++) {
                    for(size_t k = 0; k < sqrtMinusDtSVDVecs_Ddn.rank(1); k++) {
                        size_t source_idx = slice_offset + j * sqrtMinusDtSVDVecs_Ddn.rank(1) + k;
                        matrix_data[j * matrix.rank(1) + k] = source_data[source_idx];
                    }
                }
            } else {
                cout << "Error!!! SVD::getSqrtMinusDtSVDVecsMatrix_SVD: Invalid index i = " << i 
                     << " for matrix Ddn with size " << sqrtMinusDtSVDVecs_Ddn.rank(2) << endl;
                exit(1);
            }
            break;
        case 8: // "D"
            if(i >= 0 && i < static_cast<int>(sqrtMinusDtSVDVecs_D.rank(2))) {
                matrix.resize(sqrtMinusDtSVDVecs_D.rank(0), sqrtMinusDtSVDVecs_D.rank(1));
                // 使用指针访问优化内存访问模式
                complex<double> *matrix_data = matrix.data();
                const complex<double> *source_data = sqrtMinusDtSVDVecs_D.data();
                size_t slice_offset = i * sqrtMinusDtSVDVecs_D.rank(0) * sqrtMinusDtSVDVecs_D.rank(1);
                
                for(size_t j = 0; j < sqrtMinusDtSVDVecs_D.rank(0); j++) {
                    for(size_t k = 0; k < sqrtMinusDtSVDVecs_D.rank(1); k++) {
                        size_t source_idx = slice_offset + j * sqrtMinusDtSVDVecs_D.rank(1) + k;
                        matrix_data[j * matrix.rank(1) + k] = source_data[source_idx];
                    }
                }
            } else {
                cout << "Error!!! SVD::getSqrtMinusDtSVDVecsMatrix_SVD: Invalid index i = " << i 
                     << " for matrix D with size " << sqrtMinusDtSVDVecs_D.rank(2) << endl;
                exit(1);
            }
            break;
    }
    
    return matrix;
}
/////////////////////////////////////////////////

SVDSample SVD::getTwoBodySampleFromAux_icf_fixedForce(const SVDAux &aux, const SVDForce &force) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<aux.size()<<endl; exit(1); }

    int L = sqrtMinusDtSVDVecs.rank(0);
    if(!Hamiltonian_spin_flag){
        L = 2*L;
    }
    SVDSample svdSample( L );

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

SVDSample2s SVD::getTwoBodySampleFromAux_icf_fixedForce2s(const SVDAux &aux, const SVDForce &force) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<aux.size()<<endl; exit(1); }

    SVDSample2s svdSample2s( sqrtMinusDtSVDVecsUp.rank(0) );

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


SVDSample SVD::getTwoBodySampleFromAux_icf_ConstForce(const SVDAux &aux) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<aux.size()<<endl; exit(1); }

    int L = sqrtMinusDtSVDVecs.rank(0);
    if(!Hamiltonian_spin_flag){
        L = 2*L;
    }

    SVDSample svdSample( L );

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

SVDSample2s SVD::getTwoBodySampleFromAux_icf_ConstForce2s(const SVDAux &aux) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<aux.size()<<endl; exit(1); }

    SVDSample2s svdSample2s( sqrtMinusDtSVDVecsUp.rank(0) );

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

SVDSample SVD::getTwoBodySampleFromAuxForce(const SVDAux &aux, const SVDForce &force) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<svdNumber<<" "<<aux.size()<<endl; exit(1); }
    if( svdNumber != force.size() ) { cout<<"Error!!! Force size does not consistent with svdNumber! "<<svdNumber<<" "<<force.size()<<endl; exit(1); }

    int L = sqrtMinusDtSVDVecs.rank(0);
    if(!Hamiltonian_spin_flag){
        L = 2*L;
    }

    SVDSample svdSample( L );

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

SVDSample2s SVD::getTwoBodySampleFromAuxForce2s(const SVDAux &aux, const SVDForce &force) const
{
    if( svdNumber != aux.size() ) { cout<<"Error!!! Aux size does not consistent with svdNumber! "<<svdNumber<<" "<<aux.size()<<endl; exit(1); }
    if( svdNumber != force.size() ) { cout<<"Error!!! Force size does not consistent with svdNumber! "<<svdNumber<<" "<<force.size()<<endl; exit(1); }

    SVDSample2s svdSample2s( sqrtMinusDtSVDVecsUp.rank(0) );

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

double SVD::getMemory() const
{
    double mem(0.0);
    //Hamiltonian_spin_flag
    mem += sizeof(Hamiltonian_spin_flag);
    // Basic primitive types
    mem += sizeof(dt);                    // double
    mem += sizeof(SVD_cutoff);            // double 
    mem += sizeof(sqrtMinusDt);           // std::complex<double>
    mem += sizeof(svdNumber);             // size_t
    
    // Tensor objects declared in header
    mem += sqrtMinusDtSVDVecs.getMemory();
    mem += sqrtMinusDtSVDVecsUp.getMemory();
    // mem += sqrtMinusDtSVDVecsDn.getMemory();
    mem += U0up.getMemory();
    mem += Vdagger0up.getMemory();
    mem += U0dn.getMemory();
    mem += Vdagger0dn.getMemory();
    mem += U0.getMemory();
    mem += Vdagger0.getMemory();
    mem += sqrtMinusDtSVDVecs_Dup.getMemory();
    // mem += sqrtMinusDtSVDVecs_Ddn.getMemory();
    mem += sqrtMinusDtSVDVecs_D.getMemory();
    mem += svdBg.getMemory();

    return mem;
}

void SVD::copy_deep(const SVD &x)
{
    Hamiltonian_spin_flag = x.Hamiltonian_spin_flag;
    dt = x.dt;
    sqrtMinusDt = x.sqrtMinusDt;
    svdNumber = x.svdNumber;
    sqrtMinusDtSVDVecs = x.sqrtMinusDtSVDVecs;
    sqrtMinusDtSVDVecsUp = x.sqrtMinusDtSVDVecsUp;
    // sqrtMinusDtSVDVecsDn = x.sqrtMinusDtSVDVecsDn;

    U0up =  x.U0up;
    Vdagger0up =  x.Vdagger0up;

    U0dn =  x.U0dn;
    Vdagger0dn =  x.Vdagger0dn;

    U0 =  x.U0;
    Vdagger0 =  x.Vdagger0;

    sqrtMinusDtSVDVecs_Dup =  x.sqrtMinusDtSVDVecs_Dup;
    // sqrtMinusDtSVDVecs_Ddn =  x.sqrtMinusDtSVDVecs_Ddn;
    sqrtMinusDtSVDVecs_D =  x.sqrtMinusDtSVDVecs_D;

    svdBg = x.svdBg;
}

void SVD::move_deep(SVD &x)
{
    Hamiltonian_spin_flag = x.Hamiltonian_spin_flag;
    dt = x.dt;
    sqrtMinusDt = x.sqrtMinusDt;
    svdNumber = x.svdNumber;
    sqrtMinusDtSVDVecs = move( x.sqrtMinusDtSVDVecs );
    sqrtMinusDtSVDVecsUp = move( x.sqrtMinusDtSVDVecsUp );
    // sqrtMinusDtSVDVecsDn = move( x.sqrtMinusDtSVDVecsDn );

    U0up =  move(x.U0up);
    Vdagger0up =   move(x.Vdagger0up);

    U0dn =   move(x.U0dn);
    Vdagger0dn =   move(x.Vdagger0dn);

    U0 =  move(x.U0);
    Vdagger0 =   move(x.Vdagger0);

    sqrtMinusDtSVDVecs_Dup =  move(x.sqrtMinusDtSVDVecs_Dup);
    // sqrtMinusDtSVDVecs_Ddn =   move(x.sqrtMinusDtSVDVecs_Ddn);
    sqrtMinusDtSVDVecs_D =   move(x.sqrtMinusDtSVDVecs_D);

    svdBg = move( x.svdBg );
}

void SVD::initialSqrtMinusDtSVDVecs(const tensor_hao::TensorHaoMPIRef<complex<double>, 3> &svdVecs, const tensor_hao::TensorHao<complex<double>, 2> &svdVecs_U0, const tensor_hao::TensorHao<complex<double>, 3> &svdVecs_D, const tensor_hao::TensorHao<complex<double>, 2> &svdVecs_Vdagger0, bool cutOffFlag)
{
    sqrtMinusDtSVDVecs.resize(0,0,0);
    sqrtMinusDtSVDVecsUp.resize(0,0,0);
    sqrtMinusDtSVDVecs_D.resize(0,0,0);
    sqrtMinusDtSVDVecs_Dup.resize(0,0,0);
    // 
    U0up.resize(0,0); U0dn.resize(0,0);
    Vdagger0up.resize(0,0); Vdagger0dn.resize(0,0);
    U0.resize(0,0);
    Vdagger0.resize(0,0);
    // 
    size_t L = svdVecs.rank(0); 
    size_t truncatedD = svdVecs_D.rank(1); 
    size_t truncatedSVDnumber = svdVecs_D.rank(2);
    // 
    #ifdef MPI_HAO
    int ranks_per_node = 24;
    char* cpus_per_node = getenv("SLURM_JOB_CPUS_PER_NODE");
    if(cpus_per_node) {
        ranks_per_node = atoi(cpus_per_node);
    }
    int node_id = MPIRank() / ranks_per_node;
    int local_rank = MPIRank() % ranks_per_node;
    bool is_local_root = (local_rank == 0);
    
    MPI_Comm node_comm;
    MPI_Comm_split(MPI_COMM_WORLD, node_id, MPIRank(), &node_comm);
    #endif
    // 
    if(Hamiltonian_spin_flag){
        #ifdef USE_SD
            #ifdef MPI_HAO
            {
                size_t dims[3] = {L, L, svdNumber};
                if(is_local_root) {
                    sqrtMinusDtSVDVecs.createSharedMemory(dims, 0, node_comm);
                }
                else {
                    sqrtMinusDtSVDVecs.createSharedMemoryView(0, node_comm);
                    sqrtMinusDtSVDVecs.attachToSharedMemory(dims, 0, node_comm);
                }
            }
            #endif

            if(is_local_root) {
                sqrtMinusDtSVDVecs.resize( L, L, svdNumber );
                complex<double> *p0 = sqrtMinusDtSVDVecs.data();
                const complex<double> *p1 = svdVecs.data();
                size_t svdVecsSize = svdVecs.size();
                for(size_t i = 0; i < svdVecsSize; ++i) p0[i] = p1[i] * sqrtMinusDt;
            }
            MPIBarrier();
            //get SVD decomposition:
            if(truncatedD > 0){
                U0=svdVecs_U0;
                Vdagger0=svdVecs_Vdagger0;
                sqrtMinusDtSVDVecs_D.resize(truncatedD,truncatedD,truncatedSVDnumber);
                complex<double> *p0 = sqrtMinusDtSVDVecs_D.data();
                const complex<double> *p1 = svdVecs_D.data();
                size_t svdVecsSize = svdVecs_D.size();
                for(size_t i = 0; i < svdVecsSize; ++i) p0[i] = p1[i] * sqrtMinusDt;
            }else{
                U0.resize(0,0); 
                Vdagger0.resize(0,0); 
                sqrtMinusDtSVDVecs_D.resize(0,0,0);   
            }
        #else
            #ifdef MPI_HAO
            {
                size_t dimsUp[3] = {L/2, L/2, svdNumber};
                if(is_local_root) {
                    sqrtMinusDtSVDVecsUp.createSharedMemory(dimsUp, 0, node_comm);
                }
                else {
                    sqrtMinusDtSVDVecsUp.createSharedMemoryView(0, node_comm);
                    sqrtMinusDtSVDVecsUp.attachToSharedMemory(dimsUp, 0, node_comm);
                }
            }
            #endif

            if(is_local_root) {
                for(int i=1-1; i<=svdNumber-1; i++){
                    for(int j=1-1; j<=L/2-1; j++){
                    for(int k=1-1; k<=L/2-1; k++){
                        sqrtMinusDtSVDVecsUp(j,k,i) +=  svdVecs(j,k,i) * sqrtMinusDt;
                    }
                    }
                }
            }
            MPIBarrier();

            // 
            if(sqrtMinusDtSVDVecsUp.rank(0) != sqrtMinusDtSVDVecsDn.rank(0) ){
                cout<<"Error: sqrtMinusDtSVDVecsUp.rank(0) != sqrtMinusDtSVDVecsDn.rank(0) --> initialSqrtMinusDtSVDVecs is not allowed for SVDSample2s"<<endl;
                exit(1);
            }
            //get SVD decomposition:
            if(truncatedD > 0){
                U0up.resize(L/2,truncatedD/2); U0dn.resize(L/2,truncatedD/2);
                Vdagger0up.resize(truncatedD/2,L/2); Vdagger0dn.resize(truncatedD/2,L/2);
                sqrtMinusDtSVDVecs_Dup.resize(truncatedD/2, truncatedD/2, truncatedSVDnumber);
                for(int i=1-1; i<=L/2-1; i++){
                for(int j=1-1; j<=truncatedD/2-1; j++){
                    U0up(i,j)=svdVecs_U0(i,j);
                    U0dn(i,j)=svdVecs_U0(i+L/2,j+truncatedD/2);
                }
                }
                for(int i=1-1; i<=truncatedD/2-1; i++){
                for(int j=1-1; j<=L/2-1; j++){
                    Vdagger0up(i,j)=svdVecs_Vdagger0(i,j);
                    Vdagger0dn(i,j)=svdVecs_Vdagger0(i+truncatedD/2,j+L/2);
                }
                }
                for(int i=1-1; i<=truncatedSVDnumber-1; i++){
                    for(int j=1-1; j<=truncatedD/2-1; j++){
                    for(int k=1-1; k<=truncatedD/2-1; k++){
                        sqrtMinusDtSVDVecs_Dup(j,k,i) +=  svdVecs_D(j,k,i) * sqrtMinusDt;
                    }
                    }
                }
                // 
                // C0 = Vdagger0 @ U0
                TensorHao<std::complex<double>, 2> C0up(truncatedD/2, truncatedD/2);
                BL_NAME(gmm)(Vdagger0up, U0up, C0up);
                TensorHao<std::complex<double>, 2> C0dn(truncatedD/2, truncatedD/2);
                BL_NAME(gmm)(Vdagger0dn, U0dn, C0dn);
            }else{
                U0up.resize(0,0); U0dn.resize(0,0);
                Vdagger0up.resize(0,0); Vdagger0dn.resize(0,0);
                sqrtMinusDtSVDVecs_Dup.resize(0,0,0);   
            }
        #endif
    }else{
        #ifdef USE_SD
            #ifdef MPI_HAO
            {
                size_t dims[3] = {L, L, svdNumber};
                if(is_local_root) {
                    sqrtMinusDtSVDVecs.createSharedMemory(dims, 0, node_comm);
                }
                else {
                    sqrtMinusDtSVDVecs.createSharedMemoryView(0, node_comm);
                    sqrtMinusDtSVDVecs.attachToSharedMemory(dims, 0, node_comm);
                }
            }
            #endif

            if(is_local_root) {
                sqrtMinusDtSVDVecs.resize( L, L, svdNumber );
                complex<double> *p0 = sqrtMinusDtSVDVecs.data();
                const complex<double> *p1 = svdVecs.data();
                size_t svdVecsSize = svdVecs.size();
                for(size_t i = 0; i < svdVecsSize; ++i) p0[i] = p1[i] * sqrtMinusDt;
            }
            MPIBarrier();

            //get SVD decomposition:
            if(truncatedD > 0){
                U0=svdVecs_U0;
                Vdagger0=svdVecs_Vdagger0;
                sqrtMinusDtSVDVecs_D.resize(truncatedD,truncatedD,truncatedSVDnumber);
                complex<double> *p0 = sqrtMinusDtSVDVecs_D.data();
                const complex<double> *p1 = svdVecs_D.data();
                size_t svdVecsSize = svdVecs_D.size();
                for(size_t i = 0; i < svdVecsSize; ++i) p0[i] = p1[i] * sqrtMinusDt;
            }else{
                U0.resize(0,0); 
                Vdagger0.resize(0,0); 
                sqrtMinusDtSVDVecs_D.resize(0,0,0);   
            }
        #else
            #ifdef MPI_HAO
            {
                size_t dimsUp[3] = {L, L, svdNumber};
                if(is_local_root) {
                    sqrtMinusDtSVDVecsUp.createSharedMemory(dimsUp, 0, node_comm);
                }
                else {
                    sqrtMinusDtSVDVecsUp.createSharedMemoryView(0, node_comm);
                    sqrtMinusDtSVDVecsUp.attachToSharedMemory(dimsUp, 0, node_comm);
                }
            }
            #endif

            if(is_local_root) {
                // 
                for(int i=1-1; i<=svdNumber-1; i++){
                    for(int j=1-1; j<=L-1; j++){
                    for(int k=1-1; k<=L-1; k++){
                        sqrtMinusDtSVDVecsUp(j,k,i) +=  svdVecs(j,k,i) * sqrtMinusDt;
                        // sqrtMinusDtSVDVecsDn(j,k,i) +=  svdVecs(j,k,i) * sqrtMinusDt;
                    }
                    }
                }
            }
            MPIBarrier();
            
            // 
            if(sqrtMinusDtSVDVecsUp.rank(0) != sqrtMinusDtSVDVecsDn.rank(0) ){
                cout<<"Error: sqrtMinusDtSVDVecsUp.rank(0) != sqrtMinusDtSVDVecsDn.rank(0) --> initialSqrtMinusDtSVDVecs is not allowed for SVDSample2s"<<endl;
                exit(1);
            }
            //get SVD decomposition:
            if(truncatedD > 0){
                U0up.resize(L,truncatedD); U0dn.resize(L,truncatedD);
                Vdagger0up.resize(truncatedD,L); Vdagger0dn.resize(truncatedD,L);
                sqrtMinusDtSVDVecs_Dup.resize(truncatedD,truncatedD,truncatedSVDnumber); 
                // sqrtMinusDtSVDVecs_Ddn.resize(truncatedD,truncatedD,truncatedSVDnumber);
                for(int i=1-1; i<=L-1; i++){
                for(int j=1-1; j<=truncatedD-1; j++){
                    U0up(i,j)=svdVecs_U0(i,j);
                    U0dn(i,j)=svdVecs_U0(i,j);
                }
                }
                for(int i=1-1; i<=truncatedD-1; i++){
                for(int j=1-1; j<=L-1; j++){
                    Vdagger0up(i,j)=svdVecs_Vdagger0(i,j);
                    Vdagger0dn(i,j)=svdVecs_Vdagger0(i,j);
                }
                }
                for(int i=1-1; i<=truncatedSVDnumber-1; i++){
                    for(int j=1-1; j<=truncatedD-1; j++){
                    for(int k=1-1; k<=truncatedD-1; k++){
                        sqrtMinusDtSVDVecs_Dup(j,k,i) +=  svdVecs_D(j,k,i) * sqrtMinusDt;
                        // sqrtMinusDtSVDVecs_Ddn(j,k,i) +=  svdVecs_D(j,k,i) * sqrtMinusDt;
                    }
                    }
                }
                // 
                // C0 = Vdagger0 @ U0
                TensorHao<std::complex<double>, 2> C0up(truncatedD, truncatedD);
                BL_NAME(gmm)(Vdagger0up, U0up, C0up);
                TensorHao<std::complex<double>, 2> C0dn(truncatedD, truncatedD);
                BL_NAME(gmm)(Vdagger0dn, U0dn, C0dn);
                // 
            }else{
                U0up.resize(0,0); U0dn.resize(0,0);
                Vdagger0up.resize(0,0); Vdagger0dn.resize(0,0);
                sqrtMinusDtSVDVecs_Dup.resize(0,0,0); 
                // sqrtMinusDtSVDVecs_Ddn.resize(0,0,0);    
            }
        #endif
    }
}

void SVD::setTwoBodySampleMatrix(SVDSample &svdSample, const SVDAux &aux) const
{
    //Calculate aux * sqrtMinusDt * svdVecs
    if(Hamiltonian_spin_flag){
        size_t L = sqrtMinusDtSVDVecs.rank(0); size_t L2 = L * L;
        TensorHaoRef<complex<double>, 1> vecsAux(L2);
        TensorHaoRef<complex<double>, 2> vecs(L2, svdNumber);
        vecsAux.point( svdSample.matrix.data() );
        vecs.point( const_cast<complex<double>*> ( sqrtMinusDtSVDVecs.data() ) );
        BL_NAME(gemv)(vecs, aux, vecsAux);
    }else{
        size_t L = sqrtMinusDtSVDVecs.rank(0); size_t L2 = L * L;
        TensorHao<complex<double>, 2> matrixTemp(L, L); matrixTemp = 0.0;
        // 
        TensorHaoRef<complex<double>, 1> vecsAux(L2);
        TensorHaoRef<complex<double>, 2> vecs(L2, svdNumber);
        vecsAux.point( matrixTemp.data() );
        vecs.point( const_cast<complex<double>*> ( sqrtMinusDtSVDVecs.data() ) );
        BL_NAME(gemv)(vecs, aux, vecsAux);
        //
        svdSample.matrix = 0.0;
        for(int i=1-1; i<=L-1; i++){
            for(int j=1-1; j<=L-1; j++){
                svdSample.matrix(i,j) = matrixTemp(i,j);
                svdSample.matrix(i+L,j+L) = matrixTemp(i,j);
            }
        }
        //
    }
}

void SVD::setTwoBodySampleMatrix(SVDSample2s &svdSample2s, const SVDAux &aux) const
{
    if(Hamiltonian_spin_flag){
        cout<<"Error: setTwoBodySampleMatrix is not allowed for SVDSample2s"<<endl;
        exit(1);
    }
    //Calculate aux * sqrtMinusDt * svdVecs
    size_t L = sqrtMinusDtSVDVecsUp.rank(0);
    size_t L2 = (L) * (L);

    if(sqrtMinusDtSVDVecsUp.rank(0) != sqrtMinusDtSVDVecsDn.rank(0) ){
        cout<<"Error: sqrtMinusDtSVDVecsUp.rank(0) != sqrtMinusDtSVDVecsDn.rank(0) --> setTwoBodySampleMatrix is not allowed for SVDSample2s"<<endl;
        exit(1);
    }

    TensorHaoRef<complex<double>, 1> vecsAuxUp(L2);
    TensorHaoRef<complex<double>, 2> vecsUp(L2, svdNumber);
    vecsAuxUp.point( svdSample2s.matrixUp.data() );
    vecsUp.point( const_cast<complex<double>*> ( sqrtMinusDtSVDVecsUp.data() ) );
    BL_NAME(gemv)(vecsUp, aux, vecsAuxUp);

    // TensorHaoRef<complex<double>, 1> vecsAuxDn(L2);
    // TensorHaoRef<complex<double>, 2> vecsDn(L2, svdNumber);
    // vecsAuxDn.point( svdSample2s.matrixDn.data() );
    // vecsDn.point( const_cast<complex<double>*> ( sqrtMinusDtSVDVecsDn.data() ) );
    // BL_NAME(gemv)(vecsDn, aux, vecsAuxDn);
    // 
    svdSample2s.matrixDn = svdSample2s.matrixUp;
}