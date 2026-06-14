//
// Created by boruoshihao on 1/15/17.
//

#ifndef AFQMCLAB_AFQMCMETROPOLIS2SMETHOD_H
#define AFQMCLAB_AFQMCMETROPOLIS2SMETHOD_H

#include "metropolis2sDefine.h"

// #ifdef MPI_HAO
// class Metropolis2sMethod;
// void MPIBcast(Metropolis2sMethod &buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
// #endif

class Metropolis2sMethod
{
 public:

    Metropolis2sMethod();    
    Metropolis2sMethod(const Metropolis2sMethod& x);
    Metropolis2sMethod(Metropolis2sMethod&& x);
    ~Metropolis2sMethod();

    Metropolis2sMethod & operator  = (const Metropolis2sMethod& x);
    Metropolis2sMethod & operator  = (Metropolis2sMethod&& x);
    //
    int numOfJastrow;
    std::vector<int> JastrowSlice;
    std::vector<std::string> JastrowName;
    std::vector<int> JastrowExpM;
    //
    double HMC_dt;
    double HMC_length;
    //
    double BPMetroSampleCap;
    size_t BPMetroStabilizeStep;
    size_t BPMetroTimesliceBlockSize;
    size_t blockNum;
    
    std::string BPMetroForceType;
    std::string BPMetroInitialAuxiliaryFlag; //"dynamicForceInitial", "constForceInitial", "readFromFile"
    std::string BPMetroUpdateType;          //"global", "local"
    int seed;  // -1. read file, 0. random, else is seeds
    
    double Metro_dtET;

    void read(const std::string& filename);
    void setDefault();
    void print();

#ifdef MPI_HAO
    //friend void MPIBcast(Metropolis2sMethod &buffer, int root,  const MPI_Comm& comm);
    void pack( std::vector<char> &buf,  int &posit ) const;
    void unpack( const std::vector<char> &buf, int &posit );
#endif

 private:
    void copy_deep(const Metropolis2sMethod &x);
    void move_deep(Metropolis2sMethod &x);
};

#endif //AFQMCLAB_AFQMCMETROPOLIS2SMETHOD_H
