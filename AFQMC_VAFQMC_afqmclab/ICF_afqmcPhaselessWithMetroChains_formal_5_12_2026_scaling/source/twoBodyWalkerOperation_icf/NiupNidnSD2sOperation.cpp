//
// Created by Zhi-Yu

#include "../../include/twoBodyWalkerOperation_icf/NiupNidnSD2sOperation.h"
#include "afqmclab.h"

using namespace std;
using namespace tensor_hao;

NiupNidnSampleSD2sOperation::NiupNidnSampleSD2sOperation()  { }

NiupNidnSampleSD2sOperation::~NiupNidnSampleSD2sOperation() { }

void NiupNidnSampleSD2sOperation::applyToRight(const NiupNidnSample &oneBody, const SD2s &walker, SD2s &walkerNew) const
{
    size_t L = walker.getL(); size_t Nup = walker.getNup(); size_t Ndn = walker.getNdn(); 

    if( L != oneBody.getL() ) { cout<<"Error!!! NiupNidnSample size is not consistent with walker!"<<endl; exit(1); }
    if( walkerNew.getL() != L  ||  walkerNew.getNup() != Nup || walkerNew.getNdn() != Ndn ) {
        walkerNew.wfUpRef().resize( L, Nup );
        walkerNew.wfDnRef().resize( L, Ndn );
    }

    const TensorHao<complex<double>,2> &wfUp = walker.getWfUp();
    const TensorHao<complex<double>,2> &wfDn = walker.getWfDn();
    TensorHao<complex<double>,2> &wfUpNew = walkerNew.wfUpRef();
    TensorHao<complex<double>,2> &wfDnNew = walkerNew.wfDnRef();

    const TensorHao<complex<double>,1> &diag00 = oneBody.diag00;
    const TensorHao<complex<double>,1> &diag10 = oneBody.diag10;
    const TensorHao<complex<double>,1> &diag01 = oneBody.diag01;
    const TensorHao<complex<double>,1> &diag11 = oneBody.diag11;

    for(size_t j = 0; j < Nup; ++j)
    {
        for(size_t i = 0; i < L; ++i)
        {
            wfUpNew(i,j)        = diag00(i) * wfUp(i,j);
        }
    }

    for(size_t j = 0; j < Ndn; ++j)
    {
        for(size_t i = 0; i < L; ++i)
        {
            wfDnNew(i, j) = diag11(i)*wfDn(i, j);
        }
    }

    walkerNew.logwRef() = oneBody.logw + walker.getLogw();

}

void NiupNidnSampleSD2sOperation::applyToLeft(const NiupNidnSample &oneBody, const SD2s &walker, SD2s &walkerNew) const
{
    size_t L = walker.getL(); size_t Nup = walker.getNup(); size_t Ndn = walker.getNdn(); 

    if( L != oneBody.getL() ) { cout<<"Error!!! NiupNidnSample size is not consistent with walker!"<<endl; exit(1); }
    if( walkerNew.getL() != L  ||  walkerNew.getNup() != Nup || walkerNew.getNdn() != Ndn ) {
        walkerNew.wfUpRef().resize( L, Nup );
        walkerNew.wfDnRef().resize( L, Ndn );
    }

    const TensorHao<complex<double>,2> &wfUp = walker.getWfUp();
    const TensorHao<complex<double>,2> &wfDn = walker.getWfDn();
    TensorHao<complex<double>,2> &wfUpNew = walkerNew.wfUpRef();
    TensorHao<complex<double>,2> &wfDnNew = walkerNew.wfDnRef();

    const TensorHao<complex<double>,1> diag00 = conj( oneBody.diag00 );
    const TensorHao<complex<double>,1> diag10 = conj( oneBody.diag01 );
    const TensorHao<complex<double>,1> diag01 = conj( oneBody.diag10 );
    const TensorHao<complex<double>,1> diag11 = conj( oneBody.diag11 );

    for(size_t j = 0; j < Nup; ++j)
    {
        for(size_t i = 0; i < L; ++i)
        {
            wfUpNew(i,j)        = diag00(i) * wfUp(i,j);
        }
    }

    for(size_t j = 0; j < Ndn; ++j)
    {
        for(size_t i = 0; i < L; ++i)
        {
            wfDnNew(i, j) = diag11(i)*wfDn(i, j);
        }
    }

    walkerNew.logwRef() = conj( oneBody.logw ) + walker.getLogw();
}
