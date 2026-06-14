//
// Created by boruoshihao on 12/25/16.
// Modified by Icf on 2019-9-29
//

#include "../../include/MetroChains/multDET.h"
#include "afqmclab.h"

using namespace std;
using namespace tensor_hao;

multDET::multDET() {numOfDET = 0; }

multDET::multDET(size_t numOfDETTemp)
{ 
    numOfDET=numOfDETTemp;
    sdVec.resize(numOfDET); 
}

multDET::multDET(const multDET &x) { copy_deep(x); }

multDET::multDET(multDET &&x) { move_deep(x); }

multDET::~multDET() { }

multDET &multDET::operator=(const multDET &x) { copy_deep(x); return *this; }

multDET &multDET::operator=(multDET &&x) { move_deep(x); return *this; }

void multDET::reset(){
    numOfDET=0;
    sdVec.resize(0);
}

void multDET::setSD( size_t n, SD sdTemp){
    sdVec[n] =  sdTemp;
}

SD &multDET::getSD( size_t n)
{
    return sdVec[n];
}

void multDET::addDET( tensor_hao::TensorHao<std::complex<double>, 2> wfTemp, std::complex<double> LogWeightTemp){
    numOfDET=numOfDET+1;
    SD sd_save;
    sd_save.wfRef()=wfTemp;  
    sd_save.logwRef()=LogWeightTemp;
    sdVec.push_back(sd_save);  
}

void multDET::addDET( SD sdTemp){
    numOfDET=numOfDET+1;
    sdVec.push_back(sdTemp);  
}

void multDET::pop_front(size_t M){
    if(numOfDET < M){
        cout<<"Error: can't pop_front as numOfDET < M"<<endl;
    }else{
        numOfDET=numOfDET-M;
        sdVec.erase(sdVec.begin(),sdVec.begin()+M);
    }

}

size_t multDET::getNumOfDET() const { return numOfDET; }

const complex<double> &multDET::getLogw(size_t n) const { return sdVec[n].getLogw(); }

const TensorHao<complex<double>, 2> &multDET::getWf(size_t n) const { return sdVec[n].getWf(); }

complex<double> &multDET::logwRef(size_t n) { return sdVec[n].logwRef(); }

TensorHao<complex<double>, 2> &multDET::wfRef(size_t n) { return sdVec[n].wfRef(); }

size_t multDET::getL() const { return sdVec[0].getWf().rank(0); }

size_t multDET::getN() const { return sdVec[0].getWf().rank(1); }

void multDET::resizeAll(size_t numOfDETTemp)
{ 
    numOfDET=numOfDETTemp;
    sdVec.resize(numOfDET); 
}

void multDET::stabilize(size_t n)
{
    sdVec[n].stabilize();
}

void multDET::addLogw(size_t n, std::complex<double> logw_add)
{
    complex<double> logwTemp = sdVec[n].getLogw();
    sdVec[n].logwRef() = logwTemp + logw_add;
}

void multDET::readAddDET(const string &filename)
{
    ifstream file;
    file.open(filename, ios::in);
    if ( ! file.is_open() ) { cout << "Error opening file in File!!! "<<filename<<endl; exit(1); }
    complex <double> logwTemp;
    TensorHao< complex <double>,2> wfTemp;
    readFile(logwTemp, file);
    wfTemp.read(file);
    file.close();

    numOfDET=numOfDET+1;
    SD sd_save;
    sd_save.wfRef()=wfTemp;  
    sd_save.logwRef()=logwTemp;
    sdVec.push_back(sd_save); 
}

int multDET::returnNbuf() const
{
    int Nbuf=4;
    for(size_t i=1-1; i<=sdVec.size()-1; i++){
        Nbuf += sdVec[i].returnNbuf();
    }
    return Nbuf;
}

double multDET::getMemory() const
{
    double memory=4.0;
    for(size_t i=1-1; i<=sdVec.size()-1; i++){
        memory += sdVec[i].getMemory();
    }
    return memory;
}

#ifdef MPI_HAO
void MPIBcast(multDET &buffer, int root, MPI_Comm const &comm)
{
    MPIBcast( buffer.numOfDET, root, comm );
    // for(size_t i = 1-1; i <= buffer.numOfDET-1; ++i){
    //     MPIBcast( buffer.logw[i], root, comm );
    //     MPIBcast( buffer.wf[i], root, comm );
    // }
    for(size_t i=1-1; i<= buffer.numOfDET-1; i++){
        MPIBcast( buffer.sdVec[i], root, comm );
    }
}

// multDET MPIAllgather(multDET &buffer, int root, MPI_Comm const &comm)
// {
//     //int totalDet=MPISum( buffer.numOfDET, root, comm );
//     int totalDet=buffer.getNumOfDET()*MPISize();
//     int L=buffer.getL();
//     int N=buffer.getN();
//     multDET rbuf(totalDet,L, N);
//     //
//     MPI_Allgather(buffer.logwRef().data(), buffer.logwRef().size(), MPI_DOUBLE_COMPLEX, rbuf.logwRef().data(), buffer.logwRef().size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
//     //
//     for(int i=1-1; i<=buffer.getNumOfDET()-1; i++){
//         vector<complex<double>> vec_temp(L*N);
//         vector<complex<double>> rbuf_vec_temp(MPISize()*L*N);
//         for(int j=1-1; j<=L-1; j++){
//         for(int k=1-1; k<=N-1; k++){
//             vec_temp[j*N+k]=buffer.wfRef(i)(j,k);
//         }  
//         }  
//         MPI_Allgather(vec_temp.data(), vec_temp.size(), MPI_DOUBLE_COMPLEX, rbuf_vec_temp.data(), vec_temp.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
//         for(int p=1-1; p<=MPISize()-1; p++){
//         for(int j=1-1; j<=L-1; j++){
//         for(int k=1-1; k<=N-1; k++){
//             rbuf.wfRef(i+p*buffer.getNumOfDET())(j,k) = rbuf_vec_temp[j*N+k+p*L*N];
//         }  
//         } 
//         }
//     }
//     //

//     return rbuf;
// }

void multDET::pack(vector<char> &buf, int &posit) const     
{
    MPI_Pack(&numOfDET, 1, MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    // MPI_Pack(logw.data(), logw.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    // for(int i=1-1; i<=wf.size()-1; i++){
    //     MPI_Pack(wf[i].data(), wf[i].size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    // }
    for(size_t i=1-1; i<= numOfDET-1; i++){
        sdVec[i].pack(buf, posit);
    }
}

void multDET::unpack(const vector<char> &buf, int &posit)
{
    MPI_Unpack(buf.data(), buf.size(), &posit, &numOfDET, 1, MPI_INT, MPI_COMM_WORLD);
    // MPI_Unpack(buf.data(), buf.size(), &posit, logw.data(), logw.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    // for(int i=1-1; i<=wf.size()-1; i++){
    //     MPI_Unpack(buf.data(), buf.size(), &posit, wf[i].data(), wf[i].size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    // }
    for(size_t i=1-1; i<= numOfDET-1; i++){
        sdVec[i].unpack(buf, posit);
    }
}
#endif

void multDET::copy_deep(const multDET &x)
{
    numOfDET = x.numOfDET;
    // logw = x.logw;
    // wf = x.wf;
    sdVec.resize(numOfDET);
    for(size_t i=1-1; i<= numOfDET-1; i++){
        sdVec[i] = x.sdVec[i];
    }
}

void multDET::move_deep(multDET &x)
{
    numOfDET = x.numOfDET;
    // logw = x.logw;
    // wf = move( x.wf );
    sdVec.resize(numOfDET);
    for(size_t i=1-1; i<= numOfDET-1; i++){
        sdVec[i] = move(x.sdVec[i]);
    }
}

// SD2s multDET::SDToSD2s(const SD &sd, int Nup, int Ndn)
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

// SD multDET::SD2sToSD(const SD2s &sd2s)
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



