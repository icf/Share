//
// Created by boruoshihao on 1/15/17.
//

#ifndef AFQMCLAB_AFQMCMETROPOLISMETHOD_H
#define AFQMCLAB_AFQMCMETROPOLISMETHOD_H

#include "metropolisDefine.h"

// #ifdef MPI_HAO
// class MetropolisMethod;
// void MPIBcast(MetropolisMethod &buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
// #endif

class MetropolisMethod
{
 public:

    MetropolisMethod();    
    MetropolisMethod(const MetropolisMethod& x);
    MetropolisMethod(MetropolisMethod&& x);
    ~MetropolisMethod();

    MetropolisMethod & operator  = (const MetropolisMethod& x);
    MetropolisMethod & operator  = (MetropolisMethod&& x);
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
    //friend void MPIBcast(MetropolisMethod &buffer, int root,  const MPI_Comm& comm);
    void pack( std::vector<char> &buf,  int &posit ) const;
    void unpack( const std::vector<char> &buf, int &posit );
#endif

 private:
    void copy_deep(const MetropolisMethod &x);
    void move_deep(MetropolisMethod &x);
};

#endif //AFQMCLAB_AFQMCMETROPOLISMETHOD_H
