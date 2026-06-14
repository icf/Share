//
// Created by Hao Shi on 1/12/18.
//

#include "../include/AuxMatrixForce.h"

using namespace std;

AuxMatrixForce::AuxMatrixForce()  { }

AuxMatrixForce::AuxMatrixForce(size_t NumberOfAuxMatrix)
{
    QijForce.resize(NumberOfAuxMatrix);
    if(NumberOfAuxMatrix >=1 ){
        for(int i=1-1; i<=NumberOfAuxMatrix-1; i++){
            QijForce(i)=0.0;
        }
    }
}

AuxMatrixForce::AuxMatrixForce(const AuxMatrixForce &x) { copy_deep(x); }

AuxMatrixForce::AuxMatrixForce(AuxMatrixForce &&x) { move_deep(x); }

AuxMatrixForce::~AuxMatrixForce() { }

AuxMatrixForce &AuxMatrixForce::operator=(const AuxMatrixForce &x)  { copy_deep(x); return *this; }

AuxMatrixForce &AuxMatrixForce::operator=(AuxMatrixForce &&x) { move_deep(x); return *this; }

size_t AuxMatrixForce::getNumberOfAuxMatrix() const  { return QijForce.size(); }

double AuxMatrixForce::getMemory() const
{
    double mem(0.0);
    mem += QijForce.getMemory();
    return mem;
}

void AuxMatrixForce::copy_deep(const AuxMatrixForce &x)
{
    QijForce = x.QijForce;
}

void AuxMatrixForce::move_deep(AuxMatrixForce &x)
{
    QijForce = move( x.QijForce );
}

////////////////////////////////////////////////
//MPI
////////////////////////////////////////////////

int AuxMatrixForce::returnNbuf() const
{
    return 8*QijForce.size();
}

#ifdef MPI_HAO
void AuxMatrixForce::pack(vector<char> &buf, int &posit) const     
{
    if(QijForce.size() > 0)MPI_Pack(QijForce.data(), QijForce.size(), MPI_DOUBLE, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
}

void AuxMatrixForce::unpack(const vector<char> &buf, int &posit)
{
    if(QijForce.size() > 0)MPI_Unpack(buf.data(), buf.size(), &posit, QijForce.data(), QijForce.size(), MPI_DOUBLE, MPI_COMM_WORLD);
}
#endif
