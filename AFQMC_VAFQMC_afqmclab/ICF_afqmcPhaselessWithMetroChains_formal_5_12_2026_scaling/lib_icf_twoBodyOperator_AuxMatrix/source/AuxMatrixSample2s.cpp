//
// Created by boruoshihao on 1/9/17.
//

#include "../include/AuxMatrixSample2s.h"

using namespace std;
using namespace tensor_hao;

AuxMatrixSample2s::AuxMatrixSample2s():logw(0.0) { }

AuxMatrixSample2s::AuxMatrixSample2s(size_t NumberOfAuxMatrix, size_t L)
{
    logw=0.0;
    //
    if(NumberOfAuxMatrix != 2*L*L){
        cout<<"Error in set AuxMatrixSample2s: NumberOfAuxMatrix !=  2*L*L "<<endl;
    }
    //
    matrixUp.resize(L,L);
    matrixDn.resize(L,L);
}

AuxMatrixSample2s::AuxMatrixSample2s(const AuxMatrixSample2s &x) { copy_deep(x);}

AuxMatrixSample2s::AuxMatrixSample2s(AuxMatrixSample2s &&x) { move_deep(x); }

AuxMatrixSample2s::~AuxMatrixSample2s() { }

AuxMatrixSample2s &AuxMatrixSample2s::operator=(const AuxMatrixSample2s &x) { copy_deep(x); return *this; }

AuxMatrixSample2s &AuxMatrixSample2s::operator=(AuxMatrixSample2s &&x) { move_deep(x); return *this; }

size_t AuxMatrixSample2s::getNumberOfAuxMatrix() const { return matrixUp.size()+matrixDn.size(); }
size_t AuxMatrixSample2s::getL() const { return matrixUp.rank(0); }

double AuxMatrixSample2s::getMemory() const
{
    return 16.0+matrixUp.getMemory()+matrixDn.getMemory();
}

void AuxMatrixSample2s::copy_deep(const AuxMatrixSample2s &x)
{
    logw = x.logw;
    matrixUp = x.matrixUp;
    matrixDn = x.matrixDn;
}

void AuxMatrixSample2s::move_deep(AuxMatrixSample2s &x)
{
    logw = x.logw;
    matrixUp = move( x.matrixUp );
    matrixDn = move( x.matrixDn );
}
