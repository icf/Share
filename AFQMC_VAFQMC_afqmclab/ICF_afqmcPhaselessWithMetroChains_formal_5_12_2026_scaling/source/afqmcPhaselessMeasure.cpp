//
// Created by boruoshihao on 7/8/17.
//
#include <chrono>
#include <cmath>
#include "../include/afqmcPhaseless.h"

using namespace std;
using namespace tensor_hao;

void AfqmcPhaseless::addMixedMeasurement_energy()
{
    complex<double> logw;
    for(int i = 0; i < method.walkerSizePerThread; ++i)
    {
        if( walkerIsAlive[i] )
        {
            logw = phiT[i].getLogw();
            mixedMeasure.addMeasurement(phiT[i], exp(logw + phiT[i].returnLogTotalPhase_fromCurrentOverlap()) );
        }
    }
}

void AfqmcPhaseless::addMixedMeasurement()
{   
    complex<double> logOverlap, logw;
    double logDiff;
    WalkerRight walkerTemp;
    double phase, beforePhase;
    std::vector<int> vec_temp{};

    for(int i = 0; i < method.walkerSizePerThread; ++i)
    {
        if( walkerIsAlive[i] )
        {
            ///////////////////////////////////////////
            if(ifRelease){
            }else{
                // icf: make sure the constrained phase be restored in measurement --> this process can reduce bias significantly
                for(int chain =1-1; chain <= method.numOfChains-1; chain++){
                    phiT[i].updateMetroChains(chain, method.numOfThermalSweeps, vec_temp, false);
                }
                for(int counter=1-1; counter<=method.numOfSweepMeasurements-1; counter++){
                    phiT[i].updateMetroChains_allChainsAndAddAndPopBracket( 1 );   //measure after each update to save memory
                    logw = phiT[i].getLogw();
                    mixedMeasure.addMeasurement(phiT[i], exp(logw + phiT[i].returnLogTotalPhase_fromCurrentOverlap()) );
                }     
            }
        }
    }
}

///////////////////////////////////////////////////
void AfqmcPhaseless::addMixedMeasurement_timer()
{   
    complex<double> logOverlap, logw;
    double logDiff;
    WalkerRight walkerTemp;
    double phase, beforePhase;
    std::vector<int> vec_temp{};

    for(int i = 0; i < method.walkerSizePerThread; ++i)
    {
        if( walkerIsAlive[i] )
        {
            /////////////////
            //Timer
            /////////////////
            auto begin = std::chrono::high_resolution_clock::now();
            ///////////////////////////////////////////
            for(int chain =1-1; chain <= method.numOfChains-1; chain++){
                phiT[i].updateMetroChains(chain, method.numOfThermalSweeps, vec_temp, false);
            }
            auto end_updateMetroChains = std::chrono::high_resolution_clock::now();
            // 
            for(int counter=1-1; counter<=method.numOfSweepMeasurements-1; counter++){
                phiT[i].updateMetroChains_allChainsAndAddAndPopBracket( 1 );   //measure after each update to save memory
                logw = phiT[i].getLogw();
                mixedMeasure.addMeasurement_timer( phiT[i], exp(logw + phiT[i].returnLogTotalPhase_fromCurrentOverlap()) );
            }     
            // 
            auto end_addMeasurement = std::chrono::high_resolution_clock::now();
            ///////////////////////////////////////////
            /////////////////
            //Timer
            /////////////////
            auto end = std::chrono::high_resolution_clock::now();
            auto updateMetroChains_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end_updateMetroChains - begin);
            auto addMeasurement_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end_addMeasurement - end_updateMetroChains);
            auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            if(MPIRank() == 0 && i == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
            if(MPIRank() == 0 && i == 0)printf("Time measured per walker for updateMetroChains: %.8f seconds.\n", updateMetroChains_elapsed.count() * 1e-9);
            if(MPIRank() == 0 && i == 0)printf("Time measured per walker for addMeasurement: %.8f seconds.\n", addMeasurement_elapsed.count() * 1e-9);
            if(MPIRank() == 0 && i == 0)printf("Time measured per walker for addMixedMeasurement_timer: %.8f seconds.\n", elapsed.count() * 1e-9);
            if(MPIRank() == 0 && i == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
        }
    }
}
///////////////////////////////////////////////////

void AfqmcPhaseless::writeAndResetMeasurement()
{
    mixedMeasure.write();
    mixedMeasure.reSet();
    // 
    // pureMeasure.write("_pure");
    // pureMeasure.reSet();
}

void AfqmcPhaseless::adjustETThenResetMeasurement()
{                
    method.ET = ( mixedMeasure.returnEnergy() ).real();
    // double energy_pure = ( pureMeasure.returnEnergy() ).real();
    if( MPIRank()==0 )
    {
        cout<<"\nAdjust trial energy: "<<method.ET<<endl;
    }

    mixedMeasure.reSet();
    // pureMeasure.reSet();
    // pureMeasure_SDSD.reSet();
}

void AfqmcPhaseless::adjustETAndBackGroundThenResetMeasurement()
{
    method.ET = ( mixedMeasure.returnEnergy() ).real();
    model.updateBackGround(mixedMeasure.returnSVDBgReal());
    
    if( MPIRank()==0 )
    {
        cout<<"\nAdjust trial energy: "<<method.ET<<endl;
        cout<<"ASTTENTION: background is forced to be real:"<<endl;
        cout<<"Adjust background: "<<model.getSVDBg()<<"\n"<<endl;
    }

    mixedMeasure.reSet();
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////