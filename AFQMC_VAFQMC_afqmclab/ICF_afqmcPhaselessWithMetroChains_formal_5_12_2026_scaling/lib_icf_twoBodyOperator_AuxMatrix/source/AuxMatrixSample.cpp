//
// Created by boruoshihao on 1/9/17.
//

#include "../include/AuxMatrixSample.h"

using namespace std;
using namespace tensor_hao;

AuxMatrixSample::AuxMatrixSample():logw(0.0) { }

AuxMatrixSample::AuxMatrixSample(size_t NumberOfAuxMatrix, size_t L)
{
    logw=0.0;
    //
    matrix.resize(L,L);
}

AuxMatrixSample::AuxMatrixSample(const AuxMatrixSample &x) { copy_deep(x);}

AuxMatrixSample::AuxMatrixSample(AuxMatrixSample &&x) { move_deep(x); }

AuxMatrixSample::~AuxMatrixSample() { }

AuxMatrixSample &AuxMatrixSample::operator=(const AuxMatrixSample &x) { copy_deep(x); return *this; }

AuxMatrixSample &AuxMatrixSample::operator=(AuxMatrixSample &&x) { move_deep(x); return *this; }

size_t AuxMatrixSample::getNumberOfAuxMatrix() const { return matrix.size(); }
size_t AuxMatrixSample::getL() const { return matrix.rank(0); }

double AuxMatrixSample::getMemory() const
{
    return 16.0+matrix.getMemory();
}

void AuxMatrixSample::copy_deep(const AuxMatrixSample &x)
{
    logw = x.logw;
    matrix = x.matrix;
}

void AuxMatrixSample::move_deep(AuxMatrixSample &x)
{
    logw = x.logw;
    matrix = move( x.matrix );
}
