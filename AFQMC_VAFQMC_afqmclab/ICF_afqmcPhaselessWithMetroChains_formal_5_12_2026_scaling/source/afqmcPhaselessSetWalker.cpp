    //
// Created by boruoshihao on 7/8/17.
//
#include <chrono>
#include "../include/afqmcPhaseless.h"
#include "../include/utils.h"

using namespace std;
using namespace tensor_hao;

void AfqmcPhaseless::initialPhiT()
{
    if(method.initialPhiTFlag == "setFromModel")
    {
        exit(1);
    }
    else if(method.initialPhiTFlag == "setRandomly")
    {
        exit(1);
    }
    else if(method.initialPhiTFlag == "readFromFile")
    {
        size_t L = model.getL(); size_t N = model.getN(); 
        if(MPIRank()==0)cout<<"phiT_multDet_initial"<<endl;
        for (int i=1-1; i <= method.numOfChains-1; i++){
            phiT_multDet_initial.readAddDET("phiT_"+to_string(i)+".dat");
        }

    }
    else if(method.initialPhiTFlag == "readFromFile_phiT_0")
    {
        size_t L = model.getL(); size_t N = model.getN(); 
        if(MPIRank()==0)cout<<"readFromFile_phiT_0"<<endl;
        for (int i=1-1; i <= method.numOfChains-1; i++){
            phiT_multDet_initial.readAddDET("phiT_0.dat");
        }

    }
    else
    {
        cout<<"Error!!! Do not recognize initialPhiTFlag!"<<endl;
        exit(1);
    }
    //
}

void AfqmcPhaseless::initialWalker()
{
    walkerIsAlive.resize(method.walkerSizePerThread);
    WalkerRight phi_save;
    std::vector<int> vec_temp{};

    if(method.initialWalkerFlag == "readFromFile")
    {
        if( MPIRank()==0 ) phi_save.read("phi.dat");
        if( MPIRank()==0 ) checkOverlap( phi_save );
        MPIBcast(phi_save);
    }
    else
    {
        cout<<"Error!!! Do not recognize initialWalkerFlag!"<<endl;
        exit(1);
    }

    complex<double> energy_temp;
    complex<double> Kenergy_temp;
    complex<double> logOverlap;
    // 
   if(method.initialAuxiliaryFieldFlag == "readFromFile"){
        phiT.resize(method.walkerSizePerThread);
        if(MPIRank() == 0)cout<<"method.initialAuxiliaryFieldFlag == readFromFile: "<<endl;
        for(int i = 0; i < method.walkerSizePerThread; ++i){
            #ifdef USE_SD
                phiT[i].initialMetroChains(model.getL(), phi_save.getN(), method.numOfReleasedSlice, method.numOfChains, method.numOfBrackets, jastrowProjector);
            #else
                phiT[i].initialMetroChains(model.getL(), phi_save.getNup(), phi_save.getNdn(), method.numOfReleasedSlice, method.numOfChains, method.numOfBrackets, jastrowProjector);
            #endif
            // phiT[i].MetroChainsInitialField_readAuxFields(phi_save, phi_save);
            phiT[i].MetroChainsInitialField(phi_save, phi_save);
        }
    }else if(method.initialAuxiliaryFieldFlag == "sampleFromMCMC"){
        vector< WalkerLeft > phiT_twoJastrow;
        // 
        if(MPIRank() == 0){
            cout<<"set Model before"<<endl;
        }
        // 
        pureMeasure_walkerWalker.reSet();
        pureMeasure_walkerWalker.setModel(model);
        //
        if(MPIRank() == 0){
            cout<<"set Model"<<endl;
        }
        walkerRightWalkerRightOperation.set(phi_save, phi_save);
        logOverlap = walkerRightWalkerRightOperation.returnLogOverlap();
        pureMeasure_walkerWalker.addMeasurement( walkerRightWalkerRightOperation, exp(logOverlap)/abs(exp(logOverlap)) );
        energy_temp=pureMeasure_walkerWalker.returnEnergy();
        if(MPIRank() == 0){
            cout<<"phi_save energy: "<<setprecision(10)<<energy_temp<<endl;
        }
        /////////////////////////////////////////////////////////////////////////////////////////
        JastrowProjector_AFQMC jastrowProjectorTwoJastrow;
        jastrowProjectorTwoJastrow.initialParametersTwoJastrow(method_Jastrow);
        // 
        //Metropolis initialization
        phiT_twoJastrow.resize(method.walkerSizePerThread);
        for(int i = 0; i < method.walkerSizePerThread; ++i){
            #ifdef USE_SD
                phiT_twoJastrow[i].initialMetroChainsTwoJastrow(model.getL(), phi_save.getN(), method.numOfReleasedSlice, method.numOfChains, method.numOfBrackets, jastrowProjectorTwoJastrow);
            #else
                phiT_twoJastrow[i].initialMetroChainsTwoJastrow(model.getL(), phi_save.getNup(), phi_save.getNdn(), method.numOfReleasedSlice, method.numOfChains, method.numOfBrackets, jastrowProjectorTwoJastrow);
            #endif
            /////////////////////////////
            /////////////////////////////
            /////////////////////////////
            if(MPIRank() == 0 && i==0){
                #ifdef USE_SD
                    cout<<"start with "<<phiT_twoJastrow[0].metropolisVec[0].method.BPMetroForceType<<" initialization"<<endl;
                #else
                    cout<<"start with "<<phiT_twoJastrow[0].metropolis2sVec[0].method.BPMetroForceType<<" initialization"<<endl;
                #endif
            }
            ///////////////////////////// 
            /////////////////////////////
            /////////////////////////////
            phiT_twoJastrow[i].MetroChainsTwoJastrowInitialField(phiT_multDet_initial, phi_save);
            // phiT_twoJastrow[i].MetroChainsTwoJastrowInitialField_readAuxFields(phiT_multDet_initial, phi_save);
            //
            for(int j=1-1; j<=method.numOfChains-1; j++){
                // phiT_twoJastrow[i].updateMetroChains(j, method.numOfInitialThermalSweeps, -1, false);
                for(int counter =1-1; counter <= method.numOfBrackets-1; counter++){
                    phiT_twoJastrow[i].addAndPopBracket(j, phiT_twoJastrow[i].getBPMetroTimesliceBlockSize()/2, phiT_twoJastrow[i].getBPMetroTimesliceBlockSize()/2);
                }
            }
        }
        phiT_multDet_initial.reset();

        // 
        if(MPIRank()==0){
            long rss = get_memory_usage_linux();
            if (rss >= 0) {
                cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
                std::cout << "current Memory " << rss << " KB (" << rss / 1024.0 << " MB)" << std::endl;
                cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
            }
        }
        // 
        if(MPIRank() == 0){
            cout<<"Initialization "<<endl;
        }
        ////////////////
        //Measure
        ////////////////
        pureMeasure_walkerWalker.reSet();
        pureMeasure_walkerWalker.setModel(model);
        for(int i = 0; i < method.walkerSizePerThread; ++i){
            for(int j = 0; j < phiT_twoJastrow[i].metroLeft.size(); ++j){
                if(j == 0){
                    walkerRightWalkerRightOperation.set(phiT_twoJastrow[i].metroLeft[j], phiT_twoJastrow[i].metroRight[j]);
                    //
                    complex<double> logOverlap = walkerRightWalkerRightOperation.returnLogOverlap();
                    pureMeasure_walkerWalker.addMeasurement( walkerRightWalkerRightOperation, exp(logOverlap)/abs(exp(logOverlap)) );
                }
            }
        }
        energy_temp=pureMeasure_walkerWalker.returnEnergy();

        // 
        if(MPIRank()==0){
            long rss = get_memory_usage_linux();
            if (rss >= 0) {
                cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
                std::cout << "current Memory " << rss << " KB (" << rss / 1024.0 << " MB)" << std::endl;
                cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
            }
        }
        // 
        if(MPIRank() == 0){
            cout<<"MetroTwoJastrow energy at Initialization: "<<setprecision(10)<<energy_temp<<endl;
        }
        pureMeasure_walkerWalker.reSet();
        ////////////////
        for(int counter_temp2 =1-1; counter_temp2 <= 1-1; counter_temp2++){
            /////////////////
            //Timer
            /////////////////
            auto begin = std::chrono::high_resolution_clock::now();
            //
            pureMeasure_walkerWalker.setModel(model);
            //Thermalization
            if(MPIRank() == 0){
                cout<<"MetroTwoJastrow thermalization: "<<setprecision(10)<<method.numOfInitialThermalSweeps<<endl;
            }
            for(int i = 0; i < method.walkerSizePerThread; ++i){
                for(int chain =1-1; chain <= method.numOfChains-1; chain++){
                    phiT_twoJastrow[i].updateMetroChains(chain, method.numOfInitialThermalSweeps, vec_temp, false);
                }
            }
            //
            for(int counter_temp =1-1; counter_temp <= 1-1; counter_temp++){
                for(int i = 0; i < method.walkerSizePerThread; ++i){
                    for(int chain =1-1; chain <= method.numOfChains-1; chain++){
                    for(int counter =1-1; counter <= method.numOfBrackets-1; counter++){
                        phiT_twoJastrow[i].updateMetroChains(chain, method.numOfThermalSweeps, vec_temp, false);
                        phiT_twoJastrow[i].addAndPopBracket(chain, phiT_twoJastrow[i].getBPMetroTimesliceBlockSize()/2, phiT_twoJastrow[i].getBPMetroTimesliceBlockSize()/2);
                    }
                    }
                }
                for(int i = 0; i < method.walkerSizePerThread; ++i){
                    for(int j = 0; j < phiT_twoJastrow[i].metroLeft.size(); ++j){
                        if(j==0){
                            walkerRightWalkerRightOperation.set(phiT_twoJastrow[i].metroLeft[j], phiT_twoJastrow[i].metroRight[j]);
                            complex<double> logOverlap = walkerRightWalkerRightOperation.returnLogOverlap();
                            pureMeasure_walkerWalker.addMeasurement( walkerRightWalkerRightOperation, exp(logOverlap)/abs(exp(logOverlap)) );
                        }
                    }
                }
            }
            //
            energy_temp=pureMeasure_walkerWalker.returnEnergy();

            // 
            if(MPIRank()==0){
                long rss = get_memory_usage_linux();
                if (rss >= 0) {
                    cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
                    std::cout << "current Memory " << rss << " KB (" << rss / 1024.0 << " MB)" << std::endl;
                    cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
                }
            }
            // 
            if(MPIRank() == 0){
                cout<<"MetroTwoJastrow energy: "<<energy_temp<<endl;
            }
            pureMeasure_walkerWalker.reSet();
            /////////////////
            //Timer
            /////////////////
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            if(MPIRank() == 0 && counter_temp2 == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
            if(MPIRank() == 0 && counter_temp2 == 0)printf("Time measured for one step of thermalization: %.8f seconds.\n", elapsed.count() * 1e-9);
            if(MPIRank() == 0 && counter_temp2 == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
        }
        ////////////////
        phiT.resize(method.walkerSizePerThread);
        if(MPIRank() == 0)cout<<"MetroTwoJastrow copyMetroChains_FromMetroChainsTwoJastrow: "<<endl;
        for(int i = 0; i < method.walkerSizePerThread; ++i){
            #ifdef USE_SD
                phiT[i].copyMetroChains_FromMetroChainsTwoJastrow(model.getL(), phi_save.getN(), method.numOfReleasedSlice, method.numOfChains, method.numOfBrackets, phiT_twoJastrow[i], jastrowProjector);
            #else
                phiT[i].copyMetroChains_FromMetroChainsTwoJastrow(model.getL(), phi_save.getNup(), phi_save.getNdn(), method.numOfReleasedSlice, method.numOfChains, method.numOfBrackets, phiT_twoJastrow[i], jastrowProjector);
            #endif
        }
        //
        phiT_twoJastrow.resize(0);
    }
    /////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////
    // if(MPIRank() == 0)cout<<"updateWalkerRight: "<<endl;
    // for(int i = 0; i < method.walkerSizePerThread; ++i){
    //     //update currentoverlap to the middle of the Metro chain
    //     for(int chain =1-1; chain <= method.numOfChains-1; chain++){
    //         #ifdef USE_SD
    //             // walkerRightWalkerRightOperation.set( phiT[i].metropolisVec[chain].getWalkerLeftInBlockFinal(), phiT[i].metropolisVec[chain].getWalkerRightInBlock(0));
    //             // phiT[i].metropolisVec[chain].metropolisInfo.currentLogOverlap = walkerRightWalkerRightOperation.returnLogOverlap();
    //             phiT[i].metropolisVec[chain].updateWalkerRight(phiT[i].metropolisVec[chain].getWalkerRightInBlock(0));
    //         #else
    //             phiT[i].metropolis2sVec[chain].updateWalkerRight(phiT[i].metropolis2sVec[chain].getWalkerRightInBlock(0));
    //         #endif
    //     }
    // }
    /////////////////////////////////////////////////////////////////////////////////////////
    //Thermalization
    // pureMeasure_walkerWalker.setModel(model);
    // for(int i = 0; i < method.walkerSizePerThread; ++i){
    //     for(int chain =1-1; chain <= method.numOfChains-1; chain++){
    //         for(int counter =1-1; counter <= method.numOfBrackets-1; counter++){
    //             phiT[i].addAndPopBracket_1Slice_walkerRightMetro(chain);
    //         }
    //     }
    //     for(int j = 1-1; j <= phiT[i].metroLeft.size()-1; ++j){
    //         walkerRightWalkerRightOperation.set(phiT[i].metroLeft[j], phiT[i].metroRight[j]);
    //         complex<double> logOverlap = walkerRightWalkerRightOperation.returnLogOverlap();
    //         if( abs(abs(exp(logOverlap)) - 1.0) >= 10e-7){
    //             cout<<"Error, returnLogPhase from MetroCopy is not phase: "<<exp(logOverlap)<<endl;
    //         }
    //         pureMeasure_walkerWalker.addMeasurement( walkerRightWalkerRightOperation, exp(logOverlap)/abs(exp(logOverlap)) );
    //     }
    // }
    // //
    // energy_temp=pureMeasure_walkerWalker.returnEnergy();
    // if(MPIRank() == 0){
    //     cout<<"MetroTwoJastrow initial energy: "<<energy_temp<<endl;
    // }
    pureMeasure_walkerWalker.reSet();
    // 
    /////////////////
    //Timer
    /////////////////
    auto begin = std::chrono::high_resolution_clock::now();
    /////////////////
    for(int i = 0; i < method.walkerSizePerThread; ++i){
        for(int chain =1-1; chain <= method.numOfChains-1; chain++){
            phiT[i].updateMetroChains(chain, method.numOfInitialThermalSweeps, vec_temp, false);
        }
    }
    /////////////////
    //Timer
    /////////////////
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
    if(MPIRank() == 0)printf("Time measured for updateMetroChains: %.8f seconds.\n", elapsed.count() * 1e-9);
    if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
    /////////////////
    for(int counter = 1-1; counter<=1-1; counter++){
        pureMeasure_walkerWalker.setModel(model);
        for(int i = 0; i < method.walkerSizePerThread; ++i){
            for(int chain =1-1; chain <= method.numOfChains-1; chain++){
                phiT[i].updateMetroChains(chain, method.numOfInitialThermalSweeps, vec_temp, false);
            }
        }
        //
        for(int i = 0; i < method.walkerSizePerThread; ++i){
            for(int chain =1-1; chain <= method.numOfChains-1; chain++){
                for(int counter =1-1; counter <= method.numOfBrackets-1; counter++){
                    phiT[i].updateMetroChains(chain, 1, vec_temp, false);
                    phiT[i].addAndPopBracket_1Slice_walkerRightMetro(chain);
                }
            }
            for(int j = 1-1; j <= phiT[i].metroLeft.size()-1; ++j){
                if(j==0){
                    walkerRightWalkerRightOperation.set(phiT[i].metroLeft[j], phiT[i].metroRight[j]);
                    complex<double> logOverlap = walkerRightWalkerRightOperation.returnLogOverlap();
                    if( abs(abs(exp(logOverlap)) - 1.0) >= 10e-7){
                        cout<<"Error, returnLogPhase from MetroCopy is not phase: "<<exp(logOverlap)<<endl;
                    }
                    pureMeasure_walkerWalker.addMeasurement( walkerRightWalkerRightOperation, exp(logOverlap)/abs(exp(logOverlap)) );
                }
            }
        }
        //
        energy_temp=pureMeasure_walkerWalker.returnEnergy();
        if(MPIRank() == 0){
            cout<<"Metro phiT initial energy: "<<energy_temp<<endl;
        }
        pureMeasure_walkerWalker.reSet();
    }
    ////////////////////////////////////////////////////
    for(int i = 0; i < method.walkerSizePerThread; ++i){
        walkerIsAlive[i] = true;
    }
    // 
    initialMgsAndPopControl();
}

void AfqmcPhaseless::writeWalkers()
{
}

void AfqmcPhaseless::checkOverlap(WalkerRight &oneWalker)
{
}

void AfqmcPhaseless::initialMgsAndPopControl()
{
    modifyGM();
    popControl(1.01);
}