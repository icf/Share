//
// Created by boruoshihao on 6/23/17.
//

#ifndef AFQMCLAB_AFQMCPHASELESSMETHOD_H
#define AFQMCLAB_AFQMCPHASELESSMETHOD_H

#include "afqmcPhaselessDefine.h"

#ifdef MPI_HAO
class AfqmcPhaselessMethod;
void MPIBcast(AfqmcPhaselessMethod &buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
#endif

class AfqmcPhaselessMethod
{
 public:
    double dt;
    size_t thermalSize;
    size_t writeNumber;
    size_t measureNumberPerWrite;
    size_t measureSkipStep;

    int walkerSizePerThread;
    int walkerSize;

    std::string forceType;   // "dynamicForce", "constForce"
    double forceCap;
    std::string initialPhiTFlag;   //"setFromModel", "setRandomly", "readFromFile"
    std::string initialWalkerFlag; //"setFromModel", "setRandomly", "sampleFromPhiT", "readFromFile", "readAllWalkers"
    std::string initialAuxiliaryFieldFlag; //"readFromFile", "sampleFromMCMC"
    
    size_t mgsStep;
    size_t popControlStep;
    size_t initPopControlMaxSize;

    double logEnergyCap;

    double ET;
    std::string backGroundInit; //"readFromFile", "EstimateFromPhiTWalker"
    size_t ETAdjustStep;
    size_t ETAdjustMaxSize;
    size_t ETAndBackGroundGrowthEstimateStep;
    size_t ETAndBackGroundGrowthEstimateMaxSize;

    int seed;  // -1. read file, 0. random, else is seeds

    int numOfReleasedSlice;

    int numOfChains;
    int numOfBrackets;
    int numOfSweeps;
    int numOfInitialThermalSweeps;
    int numOfThermalSweeps;
    int numOfMeasureSweeps;

    int MetroUpdateSkip;
    int numOfSweepMeasurements;

    AfqmcPhaselessMethod();
    ~AfqmcPhaselessMethod();

    void read(const std::string& filename);
    void write(const std::string& filename);
    void print();

#ifdef MPI_HAO
    friend void MPIBcast(AfqmcPhaselessMethod &buffer, int root,  const MPI_Comm& comm);
#endif

 private:
    void setDefault();
    void analysis();
};

#endif //AFQMCLAB_AFQMCPHASELESSMETHOD_H