//
// Created by boruoshihao on 12/25/16.
// Modified by Icf on 2019-9-29
//

#include "../../include/MetroChains2s/multDET2s.h"
#include "afqmclab.h"

using namespace std;
using namespace tensor_hao;

multDET2s::multDET2s() {numOfDET = 0; }

multDET2s::multDET2s(size_t numOfDETTemp)
{ 
    numOfDET=numOfDETTemp;
    sd2sVec.resize(numOfDET); 
}

multDET2s::multDET2s(const multDET2s &x) { copy_deep(x); }

multDET2s::multDET2s(multDET2s &&x) { move_deep(x); }

multDET2s::~multDET2s() { }

multDET2s &multDET2s::operator=(const multDET2s &x) { copy_deep(x); return *this; }

multDET2s &multDET2s::operator=(multDET2s &&x) { move_deep(x); return *this; }

void multDET2s::reset(){
    numOfDET=0;
    sd2sVec.resize(0);
}

void multDET2s::setSD( size_t n, SD2s sdTemp){
    sd2sVec[n] =  sdTemp;
}

SD2s &multDET2s::getSD( size_t n)
{
    return sd2sVec[n];
}

void multDET2s::addDET( tensor_hao::TensorHao<std::complex<double>, 2> wfUpTemp, tensor_hao::TensorHao<std::complex<double>, 2> wfDnTemp, std::complex<double> LogWeightTemp){
    numOfDET=numOfDET+1;
    SD2s sd2s_save;
    sd2s_save.wfUpRef()=wfUpTemp;  
    sd2s_save.wfDnRef()=wfDnTemp;  
    sd2s_save.logwRef()=LogWeightTemp;
    sd2sVec.push_back(sd2s_save);  
}

void multDET2s::addDET( SD2s sdTemp){
    numOfDET=numOfDET+1;
    sd2sVec.push_back(sdTemp);  
}

void multDET2s::pop_front(size_t M){
    if(numOfDET < M){
        cout<<"Error: can't pop_front as numOfDET < M"<<endl;
    }else{
        numOfDET=numOfDET-M;
        sd2sVec.erase(sd2sVec.begin(),sd2sVec.begin()+M);
    }

}

size_t multDET2s::getNumOfDET() const { return numOfDET; }

const complex<double> &multDET2s::getLogw(size_t n) const { return sd2sVec[n].getLogw(); }

const TensorHao<complex<double>, 2> &multDET2s::getWfUp(size_t n) const { return sd2sVec[n].getWfUp(); }
const TensorHao<complex<double>, 2> &multDET2s::getWfDn(size_t n) const { return sd2sVec[n].getWfDn(); }

complex<double> &multDET2s::logwRef(size_t n) { return sd2sVec[n].logwRef(); }

TensorHao<complex<double>, 2> &multDET2s::wfUpRef(size_t n) { return sd2sVec[n].wfUpRef(); }
TensorHao<complex<double>, 2> &multDET2s::wfDnRef(size_t n) { return sd2sVec[n].wfDnRef(); }

size_t multDET2s::getL() const { return sd2sVec[0].getWfUp().rank(0); }

size_t multDET2s::getN() const { return sd2sVec[0].getWfUp().rank(1) + sd2sVec[0].getWfDn().rank(1); }
size_t multDET2s::getNup() const { return sd2sVec[0].getWfUp().rank(1); }
size_t multDET2s::getNdn() const { return sd2sVec[0].getWfDn().rank(1); }

void multDET2s::resizeAll(size_t numOfDETTemp)
{ 
    numOfDET=numOfDETTemp;
    sd2sVec.resize(numOfDET); 
}

void multDET2s::stabilize(size_t n)
{
    sd2sVec[n].stabilize();
}

void multDET2s::addLogw(size_t n, std::complex<double> logw_add)
{
    complex<double> logwTemp = sd2sVec[n].getLogw();
    sd2sVec[n].logwRef() = logwTemp + logw_add;
}

void multDET2s::readAddDET(const string &filename)
{
    /////////////////////////////
    //SD2s
    //////////////////////////////
    ifstream file;
    file.open(filename, ios::in);
    if ( ! file.is_open() ) { cout << "Error opening file in File!!! "<<filename<<endl; exit(1); }
    complex <double> logwTemp;
    TensorHao< complex <double>,2> wfUpTemp;
    TensorHao< complex <double>,2> wfDnTemp;
    readFile(logwTemp, file);
    wfUpTemp.read(file);
    wfDnTemp.read(file);
    file.close();

    numOfDET=numOfDET+1;
    SD2s sd2s_save;
    sd2s_save.logwRef()=logwTemp;
    sd2s_save.wfUpRef()=wfUpTemp; 
    sd2s_save.wfDnRef()=wfDnTemp; 
    sd2sVec.push_back(sd2s_save); 
}

int multDET2s::returnNbuf() const
{
    int Nbuf=4;
    for(size_t i=1-1; i<=sd2sVec.size()-1; i++){
        Nbuf += sd2sVec[i].returnNbuf();
    }
    return Nbuf;
}

double multDET2s::getMemory() const
{
    double memory=4.0;
    for(size_t i=1-1; i<=sd2sVec.size()-1; i++){
        memory += sd2sVec[i].getMemory();
    }
    return memory;
}

#ifdef MPI_HAO
void MPIBcast(multDET2s &buffer, int root, MPI_Comm const &comm)
{
    MPIBcast( buffer.numOfDET, root, comm );
    for(size_t i=1-1; i<= buffer.numOfDET-1; i++){
        MPIBcast( buffer.sd2sVec[i], root, comm );
    }
}

void multDET2s::pack(vector<char> &buf, int &posit) const     
{
    MPI_Pack(&numOfDET, 1, MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    for(size_t i=1-1; i<= numOfDET-1; i++){
        sd2sVec[i].pack(buf, posit);
    }
}

void multDET2s::unpack(const vector<char> &buf, int &posit)
{
    MPI_Unpack(buf.data(), buf.size(), &posit, &numOfDET, 1, MPI_INT, MPI_COMM_WORLD);
    for(size_t i=1-1; i<= numOfDET-1; i++){
        sd2sVec[i].unpack(buf, posit);
    }
}
#endif

void multDET2s::copy_deep(const multDET2s &x)
{
    numOfDET = x.numOfDET;
    sd2sVec.resize(numOfDET);
    for(size_t i=1-1; i<= numOfDET-1; i++){
        sd2sVec[i] = x.sd2sVec[i];
    }
}

void multDET2s::move_deep(multDET2s &x)
{
    numOfDET = x.numOfDET;
    sd2sVec.resize(numOfDET);
    for(size_t i=1-1; i<= numOfDET-1; i++){
        sd2sVec[i] = move(x.sd2sVec[i]);
    }
}

// SD2s multDET2s::SDToSD2s(const SD &sd, int Nup, int Ndn)
// {
//     int L=sd.getL()/2; int N=sd.getN();   
//     if(N != Nup + Ndn){
//         cout<<"Error: in SDToSD2s, N != Nup + Ndn     !!!!!!!!!!"<<endl;
//         exit(1);
//     }

//     SD2s sd2s(L, Nup, Ndn);

//     TensorHao< complex<double>, 2 > wfMatrix(2*L,Nup+Ndn);
//     TensorHao< complex<double>, 2 > wfUpMatrix(L,Nup);
//     TensorHao< complex<double>, 2 > wfDnMatrix(L,Ndn);

//     wfMatrix=sd.getWf(); 

//     double constTemp=0.0;
//     for(int i=1-1; i<=L-1; i++){
//         for(int j=1-1; j<=Nup-1; j++){
//             constTemp += abs(wfMatrix(i+L,j));
//         }
//         for(int j=1-1; j<=Ndn-1; j++){
//             constTemp += abs(wfMatrix(i,j+Nup));
//         }
//     }
//     if(constTemp >= 10e-9){
//         cout<<"Error: in SDToSD2s, SD has non-zero off dia terms     !!!!!!!!!!"<<endl;
//         exit(1);
//     }

//     for(size_t i=1-1; i<=L-1; i++){
//         for(size_t j=1-1; j<=Nup-1; j++){
//             wfUpMatrix(i,j)=wfMatrix(i,j);
//         }
//         for(size_t j=1-1; j<=Ndn-1; j++){
//             wfDnMatrix(i,j)=wfMatrix(i+L,j+Nup);
//         }
//     }

//     sd2s.wfUpRef()=wfUpMatrix;
//     sd2s.wfDnRef()=wfDnMatrix;
//     sd2s.logwRef()=sd.getLogw();

//     return sd2s;
// }

// SD multDET2s::SD2sToSD(const SD2s &sd2s)
// {
//     int L=sd2s.getL(); int Nup=sd2s.getNup(); int Ndn=sd2s.getNdn();

//     SD sd(L, Nup+Ndn);

//     TensorHao< complex<double>, 2 > wfMatrix(2*L,Nup+Ndn);
//     TensorHao< complex<double>, 2 > wfUpMatrix(L,Nup);
//     TensorHao< complex<double>, 2 > wfDnMatrix(L,Ndn);

//     wfUpMatrix=sd2s.getWfUp(); wfDnMatrix=sd2s.getWfDn();
//     for(size_t i=1-1; i<=L-1; i++){
//         for(size_t j=1-1; j<=Nup-1; j++){
//             wfMatrix(i,j)=wfUpMatrix(i,j);
//         }
//         for(size_t j=1-1; j<=Ndn-1; j++){
//             wfMatrix(i+L,j+Nup)=wfDnMatrix(i,j);
//         }
//     }
//     sd.wfRef()=wfMatrix;
//     sd.logwRef()=sd2s.getLogw();

//     return sd;
// }



