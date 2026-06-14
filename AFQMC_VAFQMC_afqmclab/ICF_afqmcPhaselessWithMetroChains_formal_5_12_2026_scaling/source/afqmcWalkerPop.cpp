//
// Created by boruoshihao on 7/9/17.
//

#include "../include/afqmcWalkerPop.h"

using namespace std;

AfqmcWalkerPop::AfqmcWalkerPop()
{

}

AfqmcWalkerPop::AfqmcWalkerPop(WalkerLeft &walkerLeft_in)
{
    walkerLeft = &walkerLeft_in;
    setNbuf();
}

AfqmcWalkerPop::~AfqmcWalkerPop()
{

}

int AfqmcWalkerPop::getNbuf() const
{
    return Nbuf;
}

AfqmcWalkerPop &AfqmcWalkerPop::operator=(const AfqmcWalkerPop &x)
{
    *walkerLeft=*(x.walkerLeft);
    return *this;
}

#ifdef MPI_HAO

vector<char> AfqmcWalkerPop::pack() const
{
    vector<char> buf(Nbuf);

    int posit=0;
    walkerLeft->pack( buf, posit );
    if(posit!=Nbuf) {cout<<"ERROR in pack!!! posit does not equal Nbuf! "<<endl; exit(1);}

    return buf;
}

void AfqmcWalkerPop::unpack(const vector<char>& buf)
{
    int bufSize=buf.size();
    if(bufSize!=Nbuf) {cout<<"ERROR in unpack!!! Size of input buf does not equal Nbuf! "<<"  bufSize: "<<bufSize<<"  Nbuf: "<<Nbuf<<endl; exit(1);}

    int posit=0;
    walkerLeft->unpack( buf, posit );
    if(posit!=Nbuf) {cout<<"ERROR in unpack!!! posit does not equal Nbuf! "<<endl; exit(1);}
}

#endif

void AfqmcWalkerPop::setNbuf()
{
    Nbuf = walkerLeft->returnNbuf();
}
