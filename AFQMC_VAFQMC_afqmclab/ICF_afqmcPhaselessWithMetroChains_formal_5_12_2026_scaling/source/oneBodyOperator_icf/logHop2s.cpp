//
// Created by Hao Shi on 1/13/18.
//

#include "../../include/oneBodyOperator_icf/logHop2s.h"

using namespace std;
using namespace tensor_hao;

LogHop2s::LogHop2s():logw(0) { }

LogHop2s::LogHop2s(size_t L):logw(0) { matrixUp.resize(L, L); matrixDn.resize(L, L); }

LogHop2s::LogHop2s(const LogHop2s &x) { copy_deep(x); }

LogHop2s::LogHop2s(LogHop2s &&x) { move_deep(x); }

LogHop2s::~LogHop2s() { }

LogHop2s &LogHop2s::operator=(const LogHop2s &x)  { copy_deep(x); return *this; }

LogHop2s &LogHop2s::operator=(LogHop2s &&x) { move_deep(x); return *this; }

size_t LogHop2s::getL() const { return matrixUp.rank(0); }

double LogHop2s::getMemory() const
{
    return 16.0+matrixUp.getMemory()+matrixDn.getMemory();
}

void LogHop2s::copy_deep(const LogHop2s &x)
{
    logw = x.logw;
    matrixUp = x.matrixUp;
    matrixDn = x.matrixDn;
}

void LogHop2s::move_deep(LogHop2s &x)
{
    logw = x.logw;
    matrixUp = move( x.matrixUp );
    matrixDn = move( x.matrixDn );
}