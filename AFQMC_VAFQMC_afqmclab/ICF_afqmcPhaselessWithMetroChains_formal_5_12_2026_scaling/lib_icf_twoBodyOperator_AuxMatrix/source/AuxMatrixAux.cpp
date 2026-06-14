//
// Created by Hao Shi on 1/12/18.
//

#include "../include/AuxMatrixAux.h"

using namespace std;

AuxMatrixAux::AuxMatrixAux() { }

AuxMatrixAux::AuxMatrixAux(size_t NumberOfAuxMatrix)
{
    QijAux.resize(NumberOfAuxMatrix);
}

AuxMatrixAux::AuxMatrixAux(const AuxMatrixAux &x) { copy_deep(x); }

AuxMatrixAux::AuxMatrixAux(AuxMatrixAux&&x) { move_deep(x); }

AuxMatrixAux::~AuxMatrixAux() { }

AuxMatrixAux &AuxMatrixAux::operator=(const AuxMatrixAux &x)  { copy_deep(x); return *this; }

AuxMatrixAux &AuxMatrixAux::operator=(AuxMatrixAux &&x) { move_deep(x); return *this; }

size_t AuxMatrixAux::getNumberOfAuxMatrix() const  { return QijAux.size(); }

double AuxMatrixAux::getMemory() const
{
    double mem(0.0);
    mem += QijAux.getMemory();
    return mem;
}

void AuxMatrixAux::copy_deep(const AuxMatrixAux &x)
{
    QijAux = x.QijAux;
}

void AuxMatrixAux::move_deep(AuxMatrixAux &x)
{
    QijAux = move( x.QijAux );
}

////////////////////////////////////////////////
//MPI
////////////////////////////////////////////////

int AuxMatrixAux::returnNbuf() const
{
    return 4*QijAux.size();
}

#ifdef MPI_HAO
void AuxMatrixAux::pack(vector<char> &buf, int &posit) const     
{
    if(QijAux.size() > 0)MPI_Pack(QijAux.data(), QijAux.size(), MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
}

void AuxMatrixAux::unpack(const vector<char> &buf, int &posit)
{
    if(QijAux.size() > 0)MPI_Unpack(buf.data(), buf.size(), &posit, QijAux.data(), QijAux.size(), MPI_INT, MPI_COMM_WORLD);
}
#endif