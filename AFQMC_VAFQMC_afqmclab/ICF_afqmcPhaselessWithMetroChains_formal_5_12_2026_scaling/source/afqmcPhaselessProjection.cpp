//
// Created by boruoshihao on 7/8/17.
//
#include <chrono>
#include "../include/afqmcPhaseless.h"
#include "../include/afqmcWalkerPop.h"
#include "../include/utils.h"

using namespace std;
using namespace tensor_hao;

void AfqmcPhaseless::projectExpHalfDtK()
{
}

void AfqmcPhaseless::projectExpMinusHalfDtK()
{
}

void AfqmcPhaseless::projectExpMinusDtKExpMinusDtV()
{
    complex<double> logw;
    double logDiff;
    double phase;
    WalkerRight walkerTemp, walkerTemp2;
    complex<double> auxForce;
    //
    // MPIBarrier();
    
    for(int i = 0; i < method.walkerSizePerThread; ++i)
    {
        if( walkerIsAlive[i] )
        {
            ///////////////
            // Timer
            ///////////////
            // auto begin = std::chrono::high_resolution_clock::now();
            ///////////////
            logw = phiT[i].getLogw();
            phase = logw.imag();
            if (cos(phase) <= 0.0) { 
                walkerIsAlive[i] = false; 
                cout<<"walker: "<<i<<"In propgation remove phase: "<<cos(phase)<<endl;
                // continue; 
            }
            phiT[i].logwRef() = phiT[i].getLogw() + log(cos(phase)) - complex<double>(0, phase);
            for(int counter=1-1; counter<=phiT[i].metroRight.size()-1; counter++){
                complex<double> constTemp=phiT[i].metroRight[counter].getLogw();
                phiT[i].metroRight[counter].logwRef() = constTemp - complex<double>(0, phase);
                #ifdef USE_SD
                    phiT[i].metropolisVec[counter].metropolisInfo.walkerRightInBlock[0].logwRef() = phiT[i].metropolisVec[counter].metropolisInfo.walkerRightInBlock[0].getLogw()  - complex<double>(0, phase);
                    phiT[i].metropolisVec[counter].metropolisInfo.currentLogOverlap = phiT[i].metropolisVec[counter].metropolisInfo.currentLogOverlap  - complex<double>(0, phase);
                #else
                    phiT[i].metropolis2sVec[counter].metropolis2sInfo.walkerRightInBlock[0].logwRef() = phiT[i].metropolis2sVec[counter].metropolis2sInfo.walkerRightInBlock[0].getLogw()  - complex<double>(0, phase);
                    phiT[i].metropolis2sVec[counter].metropolis2sInfo.currentLogOverlap = phiT[i].metropolis2sVec[counter].metropolis2sInfo.currentLogOverlap  - complex<double>(0, phase);
                #endif
            }
            
            //////////////////////////////////////////////////////////////////////////
            if (method.forceType == "constForce")
            {
                dynamicForce = constForce;
                twoBodyAux = expMinusDtV.sampleAuxFromForce(constForce);
                #ifdef USE_SD
                    twoBodySample = expMinusDtV.getTwoBodySampleFromAuxForce(twoBodyAux, constForce);
                #else
                    twoBodySample = expMinusDtV.getTwoBodySampleFromAuxForce2s(twoBodyAux, constForce);
                #endif
                // auxForce = 0.0; for(size_t k = 0; k < model.getSVDNumber() ; ++k) auxForce += twoBodyAux(k)*constForce(k);
            }
            else if (method.forceType == "dynamicForce")
            {
                // Timer
                ///////////////
                // auto timer_getForce_fast_start = std::chrono::high_resolution_clock::now();
                ///////////////
                dynamicForce = mixedMeasure.getForce_fast(expMinusDtV, phiT[i], method.forceCap);
                ///////////////
                // Timer
                ///////////////
                // auto timer_getForce_fast_end = std::chrono::high_resolution_clock::now();
                ///////////////
                // WalkerWalkerOperation walkerWalkerOperation(phiT[i]);
                // TwoBodyForce dynamicForce_Test = mixedMeasure.getForce(expMinusDtV, walkerWalkerOperation, method.forceCap);
                ///////////////
                // Timer
                ///////////////
                // auto timer_getForce_end = std::chrono::high_resolution_clock::now();
                ///////////////
                //Test 
                ///////////////
                // for(size_t k = 0; k < model.getSVDNumber() ; ++k){
                //     if( abs(dynamicForce(k) - dynamicForce_Test(k)) >= 10e-8){
                //         cout<<"Error in dynamicForce: "<<dynamicForce(k)<<"  "<<dynamicForce_Test(k)<<endl;
                //         exit(1);
                //     } 
                // }
                ///////////////
                twoBodyAux = expMinusDtV.sampleAuxFromForce(dynamicForce);
                
                #ifdef USE_SD
                    twoBodySample = expMinusDtV.getTwoBodySampleFromAuxForce(twoBodyAux, dynamicForce);
                #else
                    twoBodySample = expMinusDtV.getTwoBodySampleFromAuxForce2s(twoBodyAux, dynamicForce);
                #endif
                
                ////////////////////////////////////////////
                // For a good enough Background != 0 --> auxForce never a small number --> auxForce.imag ~= -1.0 * phase_AF
                // And phase_AF is condidered in updateMetroChainsWithNewCket_MetroAuxUpdate_prop_withSave
                // If we set Background = 0 --> auxForce is never a small number --> auxForce.imag != -1.0 * phase_AF
                ////////////////////////////////////////////
                // We no need to apply phaseless approximation based on auxForce
                // Instead, in our framework, we can directly calculate phase_AF for a correct phaseless approximation   
                ////////////////////////////////////////////
                auxForce = 0.0; for(size_t k = 0; k < model.getSVDNumber() ; ++k) auxForce += twoBodyAux(k)*dynamicForce(k);
                // ////////////////////////////////////////////
                // auto elapsed_timer_getForce_fast = std::chrono::duration_cast<std::chrono::nanoseconds>(timer_getForce_fast_end - timer_getForce_fast_start);
                // auto elapsed_timer_getForce = std::chrono::duration_cast<std::chrono::nanoseconds>(timer_getForce_end - timer_getForce_fast_start);
                // if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
                // if(MPIRank() == 0)printf("Time measured for elapsed_timer_getForce_fast: %.8f seconds.\n", elapsed_timer_getForce_fast.count() * 1e-9);
                // if(MPIRank() == 0)printf("Time measured for elapsed_timer_getForce: %.8f seconds.\n", elapsed_timer_getForce.count() * 1e-9);
                // if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
                // //////////////////////////////////////////
            }
            else
            {
                cout << "Error!!! Do not know method.forceType " << method.forceType << endl;
                exit(1);
            }
            
            ///////////////
            // Timer
            ///////////////
            auto timer_propagateRight_start = std::chrono::high_resolution_clock::now();
            // Phaseless cos projection
            // HubbardSOC, no phaseless approximation here
            phase = auxForce.imag();
            if (cos(phase) <= 0.0) { 
                // ensure the phase fluctuation is small enough
                walkerIsAlive[i] = false; 
                cout<<"walker: "<<i<<"In propgation remove phase2: "<<cos(phase)<<endl;
                // continue; 
            }
            // 
            //////////////////////////////////////////////////////////////////////////
            // In our framework, we can directly calculate phase_AF for an exact phaseless approximation, instead of using approximated auxForce   
            phase = 0.0;
            //////////////////////////////////////////////////////////////////////////
            
            oneBodyWalkerRightOperation.applyToRight(expMinusHalfDtK, phiT[i].getMetroRight(0), walkerTemp2);
            twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerTemp2, walkerTemp);
            walkerTemp.addLogw(method.dt * method.ET);
            oneBodyWalkerRightOperation.applyToRight(expMinusHalfDtK, walkerTemp, walkerTemp2);
            
            ////////////////////////////////////////////////////////////////////////// 
            ///////////////
            // Timer
            ///////////////
            // auto timer_propagateRight = std::chrono::high_resolution_clock::now();
            ///////////////
            // For H2chian in sto-3g only
            int correlated_L = 1;
            vector<int> flip_i_vec; flip_i_vec.resize(correlated_L);
            flip_i_vec[0]=0;
            
            phiT[i].updateMetroChainsWithNewCket_MetroAuxUpdate_prop_withSave(walkerTemp2, method.numOfThermalSweeps, phase, phiT[i].getMetroRight(0), dynamicForce, twoBodyAux, flip_i_vec);
            
            // 
            /////////////////////////////////////////////////////////////////////////
            string BPMetroUpdateType;
            #ifdef USE_SD
                BPMetroUpdateType = phiT[i].metropolisVec[0].method.BPMetroUpdateType;
            #else
                BPMetroUpdateType = phiT[i].metropolis2sVec[0].method.BPMetroUpdateType;
            #endif
            if (BPMetroUpdateType == "local"){
                int jastrowModel_SVDNumber;
                #ifdef USE_SD
                    jastrowModel_SVDNumber = phiT[i].metropolisVec[0].jastrowProjector->model_Jastrow[0].getSVDNumber();
                #else
                    jastrowModel_SVDNumber = phiT[i].metropolis2sVec[0].jastrowProjector->model_Jastrow[0].getSVDNumber();
                #endif
                vector<vector<int>> flip_i_vec_vec; flip_i_vec_vec.resize(jastrowModel_SVDNumber/correlated_L-1);
                if(jastrowModel_SVDNumber != correlated_L*jastrowModel_SVDNumber/correlated_L){
                    cout<<"Error!!! Do not know jastrowModel_SVDNumber != correlated_L*jastrowModel_SVDNumber/correlated_L"<<endl;
                    cout<<"jastrowModel_SVDNumber/correlated_L: "<<jastrowModel_SVDNumber/correlated_L<<endl;
                    cout<<"correlated_L: "<<correlated_L<<endl;
                    cout<<"jastrowModel_SVDNumber: "<<jastrowModel_SVDNumber<<endl;
                    exit(1);
                }
                for(int counter = 1; counter <= jastrowModel_SVDNumber/correlated_L - 1; counter++){ 
                    flip_i_vec[0]=counter;
                    flip_i_vec_vec[counter-1] = flip_i_vec;
                }
                // 
                phiT[i].updateMetroChainsWithNewCket_MetroAuxUpdate(method.numOfThermalSweeps, flip_i_vec_vec);
            }
            /////////////////////////////////////////////////////////////////////////
            ///////////////
            // Timer
            ///////////////
            // auto timer_updateMetroChains = std::chrono::high_resolution_clock::now();
            ///////////////
            //
            logw = phiT[i].getLogw();
            phase = logw.imag();
            if (cos(phase) <= 0.0) { 
                walkerIsAlive[i] = false; 
                cout<<"walker: "<<i<<"In propgation remove phase after: "<<cos(phase)<<endl;
            }
            //
            phiT[i].logwRef() = phiT[i].getLogw() + log(cos(phase)) - complex<double>(0, phase);
            //
            double phase_rotate=log(exp(phiT[i].returnLogTotalPhase_fromCurrentOverlap())/abs(exp(phiT[i].returnLogTotalPhase_fromCurrentOverlap()))).imag();
            for(int counter=1-1; counter<=phiT[i].metroRight.size()-1; counter++){
                //prevent the totoal phase to retate during the propagation, important for measurement
                // in our code, we always take  phiT[i].metroRight[counter] == phiT[i].metropolis2sVec[counter].metropolis2sInfo.walkerRightInBlock[0], which should be optimized later
                phiT[i].metroRight[counter].logwRef() = phiT[i].metroRight[counter].getLogw() - complex<double>(0, phase) - complex<double>(0, phase_rotate);
                #ifdef USE_SD
                    phiT[i].metropolisVec[counter].metropolisInfo.walkerRightInBlock[0].logwRef() = phiT[i].metropolisVec[counter].metropolisInfo.walkerRightInBlock[0].getLogw() - complex<double>(0, phase) - complex<double>(0, phase_rotate);
                    phiT[i].metropolisVec[counter].metropolisInfo.currentLogOverlap = phiT[i].metropolisVec[counter].metropolisInfo.currentLogOverlap - complex<double>(0, phase) - complex<double>(0, phase_rotate);
                #else
                    phiT[i].metropolis2sVec[counter].metropolis2sInfo.walkerRightInBlock[0].logwRef() = phiT[i].metropolis2sVec[counter].metropolis2sInfo.walkerRightInBlock[0].getLogw() - complex<double>(0, phase) - complex<double>(0, phase_rotate);
                    phiT[i].metropolis2sVec[counter].metropolis2sInfo.currentLogOverlap = phiT[i].metropolis2sVec[counter].metropolis2sInfo.currentLogOverlap - complex<double>(0, phase) - complex<double>(0, phase_rotate);
                #endif
            }
            ///////////////
            // Timer
            ///////////////
            // auto timer_end = std::chrono::high_resolution_clock::now();
            ///////////////
            // auto elapsed_timer_force = std::chrono::duration_cast<std::chrono::nanoseconds>(timer_propagateRight_start - begin);
            // auto elapsed_timer_propagateRight = std::chrono::duration_cast<std::chrono::nanoseconds>(timer_propagateRight - timer_propagateRight_start);
            // auto elapsed_timer_updateMetroChains = std::chrono::duration_cast<std::chrono::nanoseconds>(timer_updateMetroChains - timer_propagateRight);
            // auto elapsed_timer_end = std::chrono::duration_cast<std::chrono::nanoseconds>(timer_end - begin);
            // if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
            // if(MPIRank() == 0)printf("Time measured for elapsed_timer_force: %.8f seconds.\n", elapsed_timer_force.count() * 1e-9);
            // if(MPIRank() == 0)printf("Time measured for elapsed_timer_propagateRight: %.8f seconds.\n", elapsed_timer_propagateRight.count() * 1e-9);
            // if(MPIRank() == 0)printf("Time measured for elapsed_timer_updateMetroChainsWithNewCket_MetroAuxUpdate: %.8f seconds.\n", elapsed_timer_updateMetroChains.count() * 1e-9);
            // if(MPIRank() == 0)printf("Time measured for elapsed_timer_end: %.8f seconds.\n", elapsed_timer_end.count() * 1e-9);
            // if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
            // exit(1);
            ///////////////
        }
    }
}

void AfqmcPhaseless::projectOneStep(size_t &MetroUpdateSkipIndex, size_t &mgsIndex, size_t &popControlIndex)
{
    projectExpMinusDtKExpMinusDtV();
    
    mgsIndex++;
    if (mgsIndex == method.mgsStep)
    {
        modifyGM();
        mgsIndex = 0;
    }

    popControlIndex++;
    if (popControlIndex == method.popControlStep)
    {
        popControl();
        popControlIndex = 0;
    }
}

void AfqmcPhaseless::modifyGM()
{
    for(int i = 0; i < method.walkerSizePerThread; ++i)
    {
        if( walkerIsAlive[i] ) {
            phiT[i].stabilize();
        }
    }
}

void AfqmcPhaseless::popControl(double popWeightCap)
{
    vector<double> weightPerThread( method.walkerSizePerThread );
    complex<double> logw; double wReal;
    double logDiff;
    for(int i = 0; i < method.walkerSizePerThread; ++i)
    {

        if( walkerIsAlive[i] )
        {
            logw = phiT[i].getLogw();

            //Cap weight real
            // if( logOverlapBackup.size() > 0 )
            // {
            //     logDiff = ( logw - logOverlapBackup[i] ).real();
            //     if( (logDiff-method.logEnergyCap)>1e-12  )
            //     {
            //         cout<<"Cap weight in projection before popControl: "
            //             <<setw(25)<<i+MPIRank()*method.walkerSizePerThread
            //             <<setw(25)<<logDiff<<setw(25)<<method.logEnergyCap<<endl;
            //        phiT[i].logwRef() = phiT[i].getLogw() + method.logEnergyCap  - logDiff ;
            //         logw += ( method.logEnergyCap - logDiff );
            //     }
            // }

            wReal = (exp(logw)).real();
            if (wReal <= 0.0) { weightPerThread[i] = 0.0; walkerIsAlive[i] = false; }
            else { weightPerThread[i] = wReal; }

            phiT[i].logwRef() = logw - log(wReal);
        }
        else
        {
            weightPerThread[i] = 0.0;
        }
    }

    checkAndResetWalkerIsAlive();
    resetLogOverlapBackup();

    // 
    vector<double> weight;
#ifdef MPI_HAO
    if( MPIRank()==0 ) weight.resize( method.walkerSize );
    MPI_Gather( weightPerThread.data(), method.walkerSizePerThread, MPI_DOUBLE_PRECISION,
                weight.data(), method.walkerSizePerThread, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD );
#else
    weight = move( weightPerThread );
#endif


    if( MPIRank() == 0 ) popCheck_icf(weight);
    if( MPIRank() == 0 ) capWeight_icf(weight, popWeightCap);

    vector<int> table;
    if( MPIRank()==0 ) table=popConfiguration_icf( MPISize(), weight );
    //
    vector<AfqmcWalkerPop> walkerPop;
    walkerPop.reserve( method.walkerSizePerThread );
    for(int i=0; i<method.walkerSizePerThread; i++) walkerPop.push_back( AfqmcWalkerPop(phiT[i]) );
    //         // // icf Test
    //         MPIBarrier();
    //         for(int i=0; i<method.walkerSizePerThread; i++){
    //             WalkerWalkerOperation walkerWalkerOperation(phiT[i]);
    //             complex<double> tempConst = walkerWalkerOperation.returnLogTotalPhase();
    //         }
    //         MPIBarrier();
    //         // // 
    // // 
    // if( MPIRank() == 0 )
    // {
    //     for(int i=1-1; i<= table.size()-1; i++){
    //         table[i] = 0;
    //     }
    //     cout<<" change this pop"<<endl;
    //     for(int i=1-1; i<= table.size()-1; i++)cout<<table[i]<<" ";
    //     cout<<endl;
    // }
    // if( MPIRank() == 0 )
    // {
    //     cout<<" Start this pop"<<endl;
    // }
    // 
    MPIBarrier();
    
    // DEBUG: Check memory before popControl
    long rss_before = get_memory_usage_linux();
    long total_rss_before;
    MPI_Reduce(&rss_before, &total_rss_before, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if(MPIRank()==0) {
        cout<<"!!! DEBUG: Memory before popControl - Rank0: "<<rss_before<<" KB, Total: "<<total_rss_before<<" KB ("<<total_rss_before/1024.0<<" MB)"<<endl;
    }
    MPIBarrier();
    
    populationControl_icf(walkerPop, table);
    
    // DEBUG: Check memory after popControl returns
    long rss_after = get_memory_usage_linux();
    long total_rss_after;
    MPI_Reduce(&rss_after, &total_rss_after, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if(MPIRank()==0) {
        cout<<"!!! DEBUG: Memory after popControl - Rank0: "<<rss_after<<" KB, Total: "<<total_rss_after<<" KB ("<<total_rss_after/1024.0<<" MB)"<<endl;
    }
    MPIBarrier();
    // 
    // if( MPIRank() == 0 )
    // {
    //     cout<<" after this pop"<<endl;
    // }
            // // // icf Test
            // MPIBarrier();
            // for(int i=0; i<method.walkerSizePerThread; i++){
            //     WalkerWalkerOperation walkerWalkerOperation(phiT[i]);
            //     complex<double> tempConst = walkerWalkerOperation.returnLogTotalPhase();
            // }
            // MPIBarrier();
            // // // 
    if( MPIRank() == 0 )
    {
        cout<<" End this pop"<<endl;
    }
}

void AfqmcPhaseless::checkAndResetWalkerIsAlive()
{
    size_t aliveWalkerPerThread(0);
    for (int i = 0; i < method.walkerSizePerThread; ++i)
    {
        if( walkerIsAlive[i] ) aliveWalkerPerThread++;
    }

    size_t aliveWalker = MPISum(aliveWalkerPerThread);

    if( MPIRank()==0 )
    {
        cout<<"Total number of walker is "<<method.walkerSize<<"."<<endl;
        cout<<"Currently "<<aliveWalker<<" walkers are still alive."<<endl;
        cout<<"Currently "<<method.walkerSize-aliveWalker<<" walkers are killed."<<endl;
    }

    for (int i = 0; i < method.walkerSizePerThread ; ++i)
    {
        walkerIsAlive[i] = true;
    }
}

void AfqmcPhaseless::resetLogOverlapBackup()
{
    logOverlapBackup.resize(method.walkerSizePerThread);
    for (int i = 0; i < method.walkerSizePerThread ; ++i)
    {
        logOverlapBackup[i] = 0.0;
    }
}


//////////////////////////////////////////////////

//////////////////////////////////////////////////

//////////////////////////////////////////////////

//////////////////////////////////////////////////