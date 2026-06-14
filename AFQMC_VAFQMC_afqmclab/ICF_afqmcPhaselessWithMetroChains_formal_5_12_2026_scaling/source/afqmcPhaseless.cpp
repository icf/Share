//
// Created by boruoshihao on 7/8/17.
//

#include <chrono>
#include "../include/afqmcPhaseless.h"
#include "../include/utils.h"

using namespace std;
using namespace tensor_hao;

AfqmcPhaseless::AfqmcPhaseless() { }

AfqmcPhaseless::~AfqmcPhaseless() { }

void AfqmcPhaseless::run()
{
    if(MPIRank()==0){
        cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
        #ifdef USE_SD
            cout<<"ATTENTION: this code USE_SD"<<endl;
        #else
            cout<<"ATTENTION: this code USE_SD2s"<<endl;
        #endif
        cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
        cout<<"ATTENTION: this code is only work for: VAFQMC with 1 slice"<<endl;
        cout<<"==========================================================================="<<endl;
        cout<<"ATTENTION: fast local update only support one Jastrow and 1 slice !"<<endl;
        cout<<"==========================================================================="<<endl;
        cout<<"ATTENTION: for real material, generalHamiltonian_icf input request L^dagger = L = L^T in the measurement of energy"<<endl; 
        cout<<"we leave ChainsSD and SDSD with different way to evaluate --> calculateSVDNormal which is the same when L = L^T"<<endl;
        cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
    }
    initialParameters();
    initialPhiT();
    initialWalker();
    initialMeasure();

    if( std::abs(method.dt) < 1e-12 )
    {
        estimateMemory();

        measureWithoutProjection();
    }
    else
    {
        initialExpOneBody();
        estimateMemory();
        measureWithProjection();
    }

    prepareStop();
}

void AfqmcPhaseless::initialParameters()
{
    if( MPIRank()==0 ) method.read("afqmc_param");
    if( MPIRank()==0 ) cout<<"============== Method AFQMC =============="<<endl;
    if( MPIRank()==0 ) method.print();
    if( MPIRank()==0 ) cout<<"============== ============ =============="<<endl;
    MPIBcast(method);

    randomHaoInit(method.seed, 1);
    if( method.seed != -1 ) randomHaoSave();

    if( MPIRank()==0 ) cout<<"============== Read model_param =============="<<endl;
    model.read("model_param");
    if( MPIRank()==0 ) cout<<"model.getL(): "<<model.getL()<<endl;
    if( MPIRank()==0 ) cout<<"model.getNup(): "<<model.getNup()<<endl;
    if( MPIRank()==0 ) cout<<"model.getNdn(): "<<model.getNdn()<<endl;
    if( MPIRank()==0 ) cout<<"model.getSVDNumber(): "<<model.getSVDNumber()<<endl;
    if( MPIRank()==0 ) cout<<"============== ============ =============="<<endl;

    // For generalHamiltonian
    expMinusDtV = model.returnExpMinusAlphaV( method.dt );
    // For HubbardSOC
    // expMinusDtV = model.returnExpMinusAlphaV( method.dt, "densityCharge" );
    // expMinusDtV = model.returnExpMinusAlphaV( method.dt, "densitySpin" );
    constForce = expMinusDtV.readForce("constForce_param");
    if( MPIRank()==0 ) cout<<"============== ============ =============="<<endl;

    //To do: MPIBcast
    method_Jastrow.read("afqmc_param_Metro");
    if( MPIRank()==0 ) cout<<"============== Method Jastrow =============="<<endl;
    if( MPIRank()==0 ) method_Jastrow.print();
    if( MPIRank()==0 ) cout<<"============== ============== =============="<<endl;
    jastrowProjector.initialParameters(method_Jastrow);
    // jastrowProjectorExtended.initialParameters(method_Jastrow);
    //
    if( MPIRank()==0 ) cout<<"============== ============== =============="<<endl;
}

void AfqmcPhaseless::initialMeasure()
{
    mixedMeasure.setModel_withPhiTConst(model, phiT[0]);
    // pureMeasure.setModel(model);
    // pureMeasure_walkerWalker.setModel(model);
}

void AfqmcPhaseless::initialExpOneBody()
{
    //Init backGround to init exp( OneBoy )
    if( method.backGroundInit == "readFromFile" )
    {
        //Already read from the model.
    }
    else if( method.backGroundInit == "EstimateFromPhiTWalker")
    {
        // addPureMeasure();
        // adjustETAndBackGroundThenResetMeasurement();
    }
    else
    {
        cout<<"Error!!! do not know the type of backGroundInit!"<<endl;
        exit(1);
    }

    #ifdef USE_SD
        expMinusDtK     = model.returnExpMinusAlphaK(  method.dt     );
        expMinusHalfDtK = model.returnExpMinusAlphaK(  method.dt*0.5 );
        expHalfDtK      = model.returnExpMinusAlphaK( -method.dt*0.5 );
    #else
        expMinusDtK     = model.returnExpMinusAlphaK2s(  method.dt     );
        expMinusHalfDtK = model.returnExpMinusAlphaK2s(  method.dt*0.5 );
        expHalfDtK      = model.returnExpMinusAlphaK2s( -method.dt*0.5 );
    #endif

    
    // For generalHamiltonian
    // expMinusDtV = model.returnExpMinusAlphaV( method.dt );
    expMinusDtV.updateBG(model.getSVDBg());
    // For HubbardSOC
    // expMinusDtV = model.returnExpMinusAlphaV( method.dt, "densityCharge" );
    // expMinusDtV = model.returnExpMinusAlphaV( method.dt, "densitySpin");
}

void AfqmcPhaseless::estimateMemory()
{
    double mem(0.0);
    double memTemp(0.0);
    mem += model.getMemory();
    if(MPIRank()==0)
    {
        cout<<"Memory need for -->[ model ]<-- is roughly: ("<<mem/1e9<<") per process."<<endl;
    }

    memTemp += expMinusHalfDtK.getMemory()+expHalfDtK.getMemory()+expMinusDtK.getMemory();
    memTemp += expMinusDtV.getMemory();
    memTemp += constForce.getMemory()*2.0;

    twoBodyAux = expMinusDtV.sampleAuxFromForce(constForce);
    memTemp += twoBodyAux.getMemory();

    #ifdef USE_SD
        twoBodySample = expMinusDtV.getTwoBodySampleFromAuxForce(twoBodyAux, constForce);
    #else
        twoBodySample = expMinusDtV.getTwoBodySampleFromAuxForce2s(twoBodyAux, constForce);
    #endif
    memTemp += twoBodySample.getMemory();

    if(MPIRank()==0)
    {
        cout<<"Memory need for -->[ AFQMC propagation ]<-- is roughly: ("<<memTemp/1e9<<") per process."<<endl;
    }

    mem += memTemp;
    memTemp = 0.0;
    // 
    memTemp += (phiT[0].getMemory()+1.0 ) * method.walkerSizePerThread;

    if(MPIRank()==0)
    {
        cout<<"Memory need for -->[ Metropolis chains ]<-- is roughly: ("<<memTemp/1e9<<") per process."<<endl;
    }

    mem += memTemp;
    memTemp = 0.0;
    // 
    mixedMeasure.addMeasurement_timer(phiT[0], 1.0);
    memTemp += mixedMeasure.getMemory();

    if(MPIRank()==0)
    {
        cout<<"Memory need for -->[ Measurements ]<-- is roughly: ("<<memTemp/1e9<<") per process."<<endl;
    }

    mem += memTemp;
    // 
    //Make a slightly big estimation for uncounted memory.
    mem*=1.2;
    if(MPIRank()==0)
    {
        cout<<"Memory need for this program is roughly: "<<mem/1e9<<"G per process."<<endl;
        cout<<"Please make sure available memory is larger than this.\n"<<endl;
    }


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
    mixedMeasure.reSet();
}

void AfqmcPhaseless::measureWithoutProjection()
{
    if( MPIRank() == 0 ) cout<<"Measure without projection."<<endl;

    addMixedMeasurement_energy();
    writeAndResetMeasurement();
}

void AfqmcPhaseless::measureWithProjection()
{
    if( MPIRank() == 0 ) cout<<"Start the projection..."<<endl;

    double beta;

    beta = (method.thermalSize+method.writeNumber*method.measureNumberPerWrite*method.measureSkipStep)*method.dt;
    if( MPIRank() == 0 ) cout<<"Total beta will be "<<beta<<endl;

    // projectExpMinusHalfDtK();

    size_t MetroUpdateSkipIndex(0), mgsIndex(0), popControlIndex(0); 
    ifRelease = false;
    
    if( MPIRank() == 0 ) cout<<"\nThermalize..."<<endl;

    size_t numberOfGrowthMeasure = (method.ETAndBackGroundGrowthEstimateMaxSize-method.ETAdjustMaxSize)/method.ETAndBackGroundGrowthEstimateStep;

    for (size_t i = 0; i < method.thermalSize; ++i)
    {
        if ( i<method.ETAdjustMaxSize )
        {
            if ( i%method.ETAdjustStep == 0 )
            {
                addMixedMeasurement_timer();
                // 
                adjustETAndBackGroundThenResetMeasurement();

                #ifdef USE_SD
                    expMinusDtK     = model.returnExpMinusAlphaK(  method.dt     );
                    expMinusHalfDtK = model.returnExpMinusAlphaK(  method.dt*0.5 );
                    expHalfDtK      = model.returnExpMinusAlphaK( -method.dt*0.5 );
                #else
                    expMinusDtK     = model.returnExpMinusAlphaK2s(  method.dt     );
                    expMinusHalfDtK = model.returnExpMinusAlphaK2s(  method.dt*0.5 );
                    expHalfDtK      = model.returnExpMinusAlphaK2s( -method.dt*0.5 );
                #endif

                // For generalHamiltonian
                expMinusDtV.updateBG(model.getSVDBg());
                // For HubbardSOC
                // expMinusDtV = model.returnExpMinusAlphaV( method.dt, "densityCharge" );
                // expMinusDtV = model.returnExpMinusAlphaV( method.dt, "densitySpin" );
            }
            
        }

        if (i < method.ETAndBackGroundGrowthEstimateMaxSize && i >= method.ETAdjustMaxSize)
        {
            if ( (i + 1 - method.ETAdjustMaxSize) % method.ETAndBackGroundGrowthEstimateStep == 0 )
            {
                addMixedMeasurement_timer();
            }

            if ( i == (method.ETAndBackGroundGrowthEstimateMaxSize - 1) )
            {
                if(numberOfGrowthMeasure>0)
                {
                    adjustETAndBackGroundThenResetMeasurement();

                    #ifdef USE_SD
                        expMinusDtK     = model.returnExpMinusAlphaK(  method.dt     );
                        expMinusHalfDtK = model.returnExpMinusAlphaK(  method.dt*0.5 );
                        expHalfDtK      = model.returnExpMinusAlphaK( -method.dt*0.5 );
                    #else
                        expMinusDtK     = model.returnExpMinusAlphaK2s(  method.dt     );
                        expMinusHalfDtK = model.returnExpMinusAlphaK2s(  method.dt*0.5 );
                        expHalfDtK      = model.returnExpMinusAlphaK2s( -method.dt*0.5 );
                    #endif

                    // For generalHamiltonian
                    expMinusDtV.updateBG(model.getSVDBg());
                    // For HubbardSOC
                    // expMinusDtV = model.returnExpMinusAlphaV( method.dt, "densityCharge" );
                    // expMinusDtV = model.returnExpMinusAlphaV( method.dt, "densitySpin" );

                    if(ifRelease && method.numOfReleasedSlice >=1){
                        ///////////////////////////////////////////////////////////////
                        // JastrowProjector_AFQMC jastrowProjector_input;
                        // //
                        // jastrowProjector_input.variableName_vec.resize(0);
                        // jastrowProjector_input.variableName_vec.push_back("HubbardSOC");
                        // jastrowProjector_input.model_Jastrow.resize(1);
                        // jastrowProjector_input.model_BP_Jastrow.resize(0);
                        // jastrowProjector_input.model_BP_Jastrow.push_back(model);
                        // jastrowProjector_input.expMinusDtV_Jastrow_vec.resize(1);
                        // jastrowProjector_input.expMinusDtV_BP_Jastrow_vec.resize(0);
                        // jastrowProjector_input.expMinusDtV_BP_Jastrow_vec.push_back(expMinusDtV);
                        // jastrowProjector_input.expMinusDtV_AM_Jastrow_vec.resize(1);
                        // jastrowProjector_input.expMinusDtK_Jastrow_vec.resize(1);
                        // jastrowProjector_input.expMinusDtK_BP_Jastrow_vec.resize(0);
                        // jastrowProjector_input.expMinusDtK_BP_Jastrow_vec.push_back(expMinusHalfDtK);
                        // jastrowProjector_input.expMinusDtK_AM_Jastrow_vec.resize(1);
                        // jastrowProjector_input.constForce_Jastrow.resize(1);
                        // jastrowProjector_input.constForce_BP_Jastrow.resize(0);
                        // jastrowProjector_input.constForce_BP_Jastrow.push_back(constForce);
                        // jastrowProjector_input.constForce_AM_Jastrow.resize(1);
                        // jastrowProjector_input.KVorder.resize(0);
                        // jastrowProjector_input.KVorder.push_back("KVK");
                        // //
                        // jastrowProjector_input.numOfJastrow = 1;
                        // jastrowProjector_input.JastrowSlice.push_back(method.numOfReleasedSlice);
                        // jastrowProjector_input.JastrowExpM.push_back(0);  //Dynamic expm is adopted in KVK automatically
                        // ///////////////////////////////////////////////////////////////
                        // jastrowProjectorExtended=jastrowProjector;
                        // jastrowProjectorExtended.extendMetroChainToRight(jastrowProjector_input);
                    }
                }
            }
        }

        if( MPIRank() == 0 ) cout<<i*method.dt<<endl;

        if( i<method.initPopControlMaxSize )
        {
            if( method.popControlStep>0 ) popControlIndex=method.popControlStep-1;
            else popControlIndex=0;
        }
        /////////////////
        //Timer
        /////////////////
        if(i <= 50 && i%10==0){
            MPIBarrier();
        }
        auto begin = std::chrono::high_resolution_clock::now();
        // 
        projectOneStep(MetroUpdateSkipIndex, mgsIndex, popControlIndex);
        /////////////////
        //Timer
        /////////////////
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        double local_time = elapsed.count();
        // if(MPIRank() == 0 && i <= 50 && i%10==0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
        // if(MPIRank() == 0 && i <= 50 && i%10==0)printf("Time measured for projectOneStep: %f seconds.\n", local_time * 1e-9);
        // if(MPIRank() == 0 && i <= 50 && i%10==0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
        if(i <= 50 && i%10==0){
            double global_max_time, global_min_time, global_sum_time;
            // 
            MPIBarrier();
            MPI_Reduce(&local_time, &global_max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_time, &global_min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_time, &global_sum_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            // 
            if (MPIRank() == 0) {
                int world_size;
                MPI_Comm_size(MPI_COMM_WORLD, &world_size);
                double average_time = global_sum_time / world_size;
                printf("Performance Summary:\n");
                printf("world_size: %d\n", world_size);
                printf("  Max time: %f seconds (load imbalance indicator)\n", global_max_time * 1e-9);
                printf("  Min time: %f seconds\n", global_min_time * 1e-9);
                printf("  Avg time: %f seconds\n", average_time * 1e-9);
                printf("  Load imbalance: %f%%\n", (global_max_time - global_min_time) / global_max_time * 100);
            }
        }
        // 
        if(i==method.numOfReleasedSlice){
            if(method.numOfReleasedSlice >=1 ){
                ifRelease = false;
            }else{
                ifRelease = false;
            }
            //extend jastrowProjector to jastrowProjectorExtended for release
            if(ifRelease  && method.numOfReleasedSlice >=1){
                ///////////////////////////////////////////////////////////////
                // JastrowProjector_AFQMC jastrowProjector_input;
                // //
                // jastrowProjector_input.variableName_vec.resize(0);
                // jastrowProjector_input.variableName_vec.push_back("HubbardSOC");
                // jastrowProjector_input.model_Jastrow.resize(1);
                // jastrowProjector_input.model_BP_Jastrow.resize(0);
                // jastrowProjector_input.model_BP_Jastrow.push_back(model);
                // jastrowProjector_input.expMinusDtV_Jastrow_vec.resize(1);
                // jastrowProjector_input.expMinusDtV_BP_Jastrow_vec.resize(0);
                // jastrowProjector_input.expMinusDtV_BP_Jastrow_vec.push_back(expMinusDtV);
                // jastrowProjector_input.expMinusDtV_AM_Jastrow_vec.resize(1);
                // jastrowProjector_input.expMinusDtK_Jastrow_vec.resize(1);
                // jastrowProjector_input.expMinusDtK_BP_Jastrow_vec.resize(0);
                // jastrowProjector_input.expMinusDtK_BP_Jastrow_vec.push_back(expMinusHalfDtK);
                // jastrowProjector_input.expMinusDtK_AM_Jastrow_vec.resize(1);
                // jastrowProjector_input.constForce_Jastrow.resize(1);
                // jastrowProjector_input.constForce_BP_Jastrow.resize(0);
                // jastrowProjector_input.constForce_BP_Jastrow.push_back(constForce);
                // jastrowProjector_input.constForce_AM_Jastrow.resize(1);
                // jastrowProjector_input.KVorder.resize(0);
                // jastrowProjector_input.KVorder.push_back("KVK");
                // //
                // jastrowProjector_input.numOfJastrow = 1;
                // jastrowProjector_input.JastrowSlice.push_back(method.numOfReleasedSlice);
                // jastrowProjector_input.JastrowExpM.push_back(0);  //Dynamic expm is adopted in KVK automatically
                // ///////////////////////////////////////////////////////////////
                // jastrowProjectorExtended=jastrowProjector;
                // jastrowProjectorExtended.extendMetroChainToRight(jastrowProjector_input);
            }
        }
    }

    if( MPIRank() == 0 ) cout<<"\nMeasure..."<<endl;

    for (size_t i = 0; i < method.writeNumber; ++i)
    {
        for (size_t j = 0; j < method.measureNumberPerWrite; ++j)
        {
            addMixedMeasurement();

            for (size_t k = 0; k < method.measureSkipStep; ++k)
            {
                beta = ( method.thermalSize+k+j*method.measureSkipStep
                         +i*method.measureSkipStep*method.measureNumberPerWrite)*method.dt;
                if (MPIRank() == 0) cout << beta << endl;
                projectOneStep(MetroUpdateSkipIndex, mgsIndex, popControlIndex);
            }
        }

        beta = ( method.thermalSize+(i+0.5)*method.measureNumberPerWrite*method.measureSkipStep-0.5 )*method.dt;
        if (MPIRank() == 0) writeFile( beta, "beta.dat", ios::app);
        writeAndResetMeasurement();
    }

    // projectExpHalfDtK();
}

void AfqmcPhaseless::prepareStop()
{
    if( MPIRank()==0 ) method.write("afqmc_param");
    writeWalkers();
    randomHaoSave();
}