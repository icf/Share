//
// Created by boruoshihao on 1/17/17.
//
#include <cmath>
#include <chrono>
#include "../../include/Metropolis/metropolis.h"
#include "../../include/utils.h"

using namespace std;
using namespace tensor_hao;
                                               
void Metropolis::updateOneSweep(vector<int> flip_i_vec)
{
    updateToRightOneSweep(flip_i_vec, false);
    updateToLeftOneSweep(flip_i_vec, false);
}

void Metropolis::updateToRightOneSweep(vector<int> flip_i_vec, bool ifFrozen)
{
    Walker walkerLeft = metropolisInfo.walkerLeftInBlock[0]; complex<double>logWeight=0.0;
    Walker walkerRight = metropolisInfo.walkerRightInBlock[0];
    //
    for(metropolisInfo.inBlockIndex = (metropolisInfo.BPMetroTimesliceBlockSize-1); metropolisInfo.inBlockIndex > -1 ; --metropolisInfo.inBlockIndex)
    {
        updateToRightOneStep(walkerLeft, logWeight, flip_i_vec, ifFrozen);
        // 
        metropolisInfo.walkerLeftInBlock[metropolisInfo.BPMetroTimesliceBlockSize-metropolisInfo.inBlockIndex]=walkerLeft;
        metropolisInfo.logWeightLeftInBlock[metropolisInfo.BPMetroTimesliceBlockSize-metropolisInfo.inBlockIndex]=logWeight;
    }
}

void Metropolis::updateToLeftOneSweep(vector<int> flip_i_vec, bool ifFrozen)
{
    Walker walkerLeft = metropolisInfo.walkerLeftInBlock[0];
    Walker walkerRight = metropolisInfo.walkerRightInBlock[0];complex<double>logWeight=0.0;
    for(metropolisInfo.inBlockIndex = 1-1; metropolisInfo.inBlockIndex <=metropolisInfo.BPMetroTimesliceBlockSize-1 ; metropolisInfo.inBlockIndex++)
    {
        // auto begin = std::chrono::high_resolution_clock::now();
        // 
        updateToLeftOneStep(walkerRight, logWeight, flip_i_vec, ifFrozen);
        // 
        // auto end = std::chrono::high_resolution_clock::now();
        // auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        // printf("Time measured for updateToLeftOneStep: %.8f seconds.\n", elapsed.count() * 1e-9);
        // exit(1);
        // 
        metropolisInfo.walkerRightInBlock[metropolisInfo.inBlockIndex+1]=walkerRight;
        metropolisInfo.logWeightRightInBlock[metropolisInfo.inBlockIndex+1]=logWeight;
    }
}

void Metropolis::updateToRightOneStep(Walker &walkerLeft, complex<double> &logWeight, vector<int> flip_i_vec, bool ifFrozen)
{

    ///////////////
    // Timer
    ///////////////
    // auto begin = std::chrono::high_resolution_clock::now();
    // auto globalFast_begin = std::chrono::high_resolution_clock::now();
    // auto globalFast_end = std::chrono::high_resolution_clock::now();
    // auto end = std::chrono::high_resolution_clock::now();
    ///////////////
    //
    int j_Jastrow=metropolisInfo.numOfJastrow-1;
    int tempConst=0;
    for(int i= metropolisInfo.numOfJastrow-1; i>= 0; i--){
        tempConst += metropolisInfo.JastrowSlice[i];
        if(metropolisInfo.inBlockIndex>=tempConst){
            j_Jastrow--;
        }
    }
    //
    TwoBodyAux_Jastrow auxNew;
    Walker walkerSaver=walkerLeft;
    Walker walkerLeftTemp;
    WalkerWalkerOperation_Jastrow walkerWalkerOperation;
    TwoBodyForce_Jastrow tempForce;

    complex<double> logProbNew=0.0;
    complex<double> currentLogProb=0.0;

    complex<double> logOverlapNew,currentLogOverlapSave;
    //
    TensorHao<complex<double>,2> B_proposed(walkerLeft.getL(),walkerLeft.getL());
    TensorHao<complex<double>,2> overlapMatrix_proposed(walkerLeft.getN(),walkerLeft.getN());
    TensorHao<complex<double>,2> overlapMatrix_inv_proposed(walkerLeft.getN(),walkerLeft.getN());
    // 
    complex<double> old_logSampleweight, new_logSampleweight;
    complex<double> overlapRatio_fastUpdate;
    double alpha;
    //
    metropolisInfo.currentLogOverlap += metropolisInfo.logWeightRightInBlock[metropolisInfo.inBlockIndex+1];
    currentLogOverlapSave=metropolisInfo.currentLogOverlap;
    if(metropolisInfo.inBlockIndex< method.blockNum || ifFrozen){
        alpha=0.0;
    }else{
        if(metropolisInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_HAFQMC_icf"){
            ////////////////////////////////////////////////////////////
            walkerLeftTemp = walkerLeft;
            if( method.BPMetroForceType == "constForce" )
            {
                if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
                    auxNew = expMinusDtV_Jastrow_vec->at(j_Jastrow).sampleAuxFromForce(metropolisInfo.constForce_Jastrow[j_Jastrow]);
                }else if(flip_i_vec.size() > 0 && method.BPMetroUpdateType == "local"){
                    auxNew = expMinusDtV_Jastrow_vec->at(j_Jastrow).sampleAuxFromForce_localUpdate(metropolisInfo.constForce_Jastrow[j_Jastrow], metropolisInfo.auxiliaryFields[metropolisInfo.inBlockIndex], flip_i_vec);
                }else{
                    cout<<"Error: UNKNOW BPMetroUpdateType: "<<method.BPMetroUpdateType<<endl;
                    exit(1);
                }
                // 
                if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
                    tempForce = metropolisInfo.constForce_Jastrow[j_Jastrow];
                    //
                    currentLogProb = expMinusDtV_Jastrow_vec->at(j_Jastrow).logProbOfAuxFromForce(metropolisInfo.auxiliaryFields[metropolisInfo.inBlockIndex], metropolisInfo.dynamicForceFields[metropolisInfo.inBlockIndex]); 
                    logProbNew = expMinusDtV_Jastrow_vec->at(j_Jastrow).logProbOfAuxFromForce(auxNew, tempForce); 
                    if(method.BPMetroUpdateType == "global_fast" && metropolisInfo.numOfJastrow == 1 && metropolisInfo.JastrowSlice[0] == 1){
                        // In global_fast, we only need to update aux related objects and we don't update walkerRight/walkerLeft in the middle of chain
                    }else{
                        TwoBodySample_Jastrow twoBodySample;
                        twoBodySample = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySampleFromAux(auxNew);
                        if(metropolisInfo.KVorder[j_Jastrow]=="VK"){
                            twoBodySampleWalkerLeftOperation.reset("fixedOrder", metropolisInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                            twoBodySampleWalkerLeftOperation.applyToLeft(twoBodySample, walkerLeft, walkerLeftTemp);
                            oneBodyWalkerLeftOperation.applyToLeft(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerLeftTemp, walkerLeft);
                        }else if(metropolisInfo.KVorder[j_Jastrow]=="K^daggerV"){
                            twoBodySampleWalkerLeftOperation.reset("fixedOrder", metropolisInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                            oneBodyWalkerLeftOperation.applyToLeft(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerLeft, walkerLeftTemp);
                            twoBodySampleWalkerLeftOperation.applyToLeft(twoBodySample, walkerLeftTemp, walkerLeft);
                        }else if(metropolisInfo.KVorder[j_Jastrow]=="KVK"){
                            twoBodySampleWalkerLeftOperation.reset("dynamicOrder");
                            oneBodyWalkerLeftOperation.applyToLeft(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerLeft, walkerLeftTemp);
                            twoBodySampleWalkerLeftOperation.applyToLeft(twoBodySample, walkerLeftTemp, walkerLeft);
                            oneBodyWalkerLeftOperation.applyToLeft(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerLeft, walkerLeftTemp);
                            walkerLeft = walkerLeftTemp;
                        }else{
                            cout<<"Error: UNKNOW metropolisInfo.KVorder[j_Jastrow]: "<<metropolisInfo.KVorder[j_Jastrow]<<endl;
                            exit(1);
                        }
                    }
                }else if(flip_i_vec.size() > 0 && method.BPMetroUpdateType == "local"){
                    tempForce = metropolisInfo.constForce_Jastrow[j_Jastrow];
                    currentLogProb = expMinusDtV_Jastrow_vec->at(j_Jastrow).logProbOfAuxFromForce(metropolisInfo.auxiliaryFields[metropolisInfo.inBlockIndex], metropolisInfo.dynamicForceFields[metropolisInfo.inBlockIndex]); 
                    logProbNew = expMinusDtV_Jastrow_vec->at(j_Jastrow).logProbOfAuxFromForce(auxNew, tempForce); 
                    // 
                    old_logSampleweight = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySample_logw_FromAux(metropolisInfo.auxiliaryFields[metropolisInfo.inBlockIndex]);
                    new_logSampleweight = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySample_logw_FromAux(auxNew);
                    overlapRatio_fastUpdate = performLocalUpdateWithFastOverlapRatio(j_Jastrow, metropolisInfo.inBlockIndex, flip_i_vec, auxNew, B_proposed, overlapMatrix_inv_proposed);
                }
                //
                ////////////////////////////////////////////////////////////
            }else{
                    cout<<"Error: UNKNOW BPMetroForceType: "<<method.BPMetroForceType<<endl;
                    exit(1);
            }
        }else{
            cout<<"Error: UNKNOW metropolisInfo.variableName_vec[j_Jastrow] in update: "<<metropolisInfo.variableName_vec[j_Jastrow]<<endl;
            exit(1);
        }
        //
        // walkerLeft.addLogw(method.Metro_dtET );
        //
        if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
            ///////////////////////////////////////////////
            if(method.BPMetroUpdateType == "global_fast" && metropolisInfo.numOfJastrow == 1 && metropolisInfo.JastrowSlice[0] == 1){
                //////////////////////////////////////////////////////
                // globalFast_begin = std::chrono::high_resolution_clock::now();
                //////////////////////////////////////////////////////
                logOverlapNew = getLogOverlapFromAux(auxNew, 2, overlapMatrix_proposed);            
                //////////////////////////////////////////////////////
                // globalFast_end = std::chrono::high_resolution_clock::now();
                //////////////////////////////////////////////////////
            }else{
                walkerWalkerOperation.set( walkerLeft, metropolisInfo.walkerRightInBlock[metropolisInfo.inBlockIndex] );
                logOverlapNew = walkerWalkerOperation.returnLogOverlap();
            }
            ///////////////////////////////////////////////
            // 
            alpha = abs( exp(logOverlapNew-logProbNew-metropolisInfo.currentLogOverlap+currentLogProb) );
        }else if(flip_i_vec.size() > 0 && method.BPMetroUpdateType == "local"){
            // fast evaluation of alpha through Matrix Determinant Lemma
            alpha = abs(exp(log(overlapRatio_fastUpdate)+new_logSampleweight-old_logSampleweight-logProbNew+currentLogProb)  );
            logOverlapNew = log(overlapRatio_fastUpdate)+ new_logSampleweight - old_logSampleweight + metropolisInfo.currentLogOverlap;
        }

        if(metropolisInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_K_HAFQMC_icf" && abs(alpha-1.0) >= 10e-8 ){
            cout<<"Error: abs(alpha-1.0) >= 10e-8 in generalHamiltonian_K_HAFQMC_icf to left: "<<alpha<<endl;
            exit(1);
        }
    }

    if( uniformHao() < alpha )
    {
        if(metropolisInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_HAFQMC_icf"){
            metropolisInfo.auxiliaryFields[metropolisInfo.inBlockIndex] = move(auxNew);
            metropolisInfo.dynamicForceFields[metropolisInfo.inBlockIndex]=tempForce;
            // 
            if(flip_i_vec.size() > 0 && method.BPMetroUpdateType == "local"){
                metropolisInfo.B = B_proposed;
                metropolisInfo.overlapMatrix_inv = overlapMatrix_inv_proposed;
            }else if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
                metropolisInfo.overlapMatrix = overlapMatrix_proposed;
            }
        }else{
            cout<<"Error: UNKNOW metropolisInfo.variableName_vec[j_Jastrow]: "<<metropolisInfo.variableName_vec[j_Jastrow]<<endl;
            exit(1);
        }
        //
        metropolisInfo.currentLogOverlap=logOverlapNew;
        /////////////////////////////
    }
    else
    {
        if(metropolisInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_HAFQMC_icf"){
            if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
                if(method.BPMetroUpdateType == "global_fast" && metropolisInfo.numOfJastrow == 1 && metropolisInfo.JastrowSlice[0] == 1){
                }else{
                    TwoBodySample_Jastrow twoBodySample;
                    if( method.BPMetroForceType == "constForce"){
                        twoBodySample = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySampleFromAux(metropolisInfo.auxiliaryFields[metropolisInfo.inBlockIndex]);
                        // 
                    }
                    else{
                        cout<<"Error: UNKNOW BPMetroForceType: "<<method.BPMetroForceType<<endl;
                        exit(1);
                    }
                    if(metropolisInfo.KVorder[j_Jastrow]=="VK"){
                        twoBodySampleWalkerLeftOperation.reset("fixedOrder", metropolisInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                        twoBodySampleWalkerLeftOperation.applyToLeft(twoBodySample, walkerSaver, walkerLeftTemp);
                        oneBodyWalkerLeftOperation.applyToLeft(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerLeftTemp, walkerLeft);
                    }else if(metropolisInfo.KVorder[j_Jastrow]=="K^daggerV"){
                        twoBodySampleWalkerLeftOperation.reset("fixedOrder", metropolisInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                        oneBodyWalkerLeftOperation.applyToLeft(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerSaver, walkerLeftTemp);
                        twoBodySampleWalkerLeftOperation.applyToLeft(twoBodySample, walkerLeftTemp, walkerLeft);
                    }else if(metropolisInfo.KVorder[j_Jastrow]=="KVK"){
                        twoBodySampleWalkerLeftOperation.reset("dynamicOrder");
                        oneBodyWalkerLeftOperation.applyToLeft(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerSaver, walkerLeftTemp);
                        twoBodySampleWalkerLeftOperation.applyToLeft(twoBodySample, walkerLeftTemp, walkerLeft);
                        oneBodyWalkerLeftOperation.applyToLeft(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerLeft, walkerLeftTemp);
                        walkerLeft = walkerLeftTemp;
                    }else{
                        cout<<"Error: UNKNOW metropolisInfo.KVorder[j_Jastrow]: "<<metropolisInfo.KVorder[j_Jastrow]<<endl;
                        exit(1);
                    }
                }
            }
            //
        }
        else{
            cout<<"Error: UNKNOW metropolisInfo.variableName_vec[j_Jastrow] in update save: "<<metropolisInfo.variableName_vec[j_Jastrow]<<endl;
            exit(1);
        }
        // walkerLeft.addLogw(method.Metro_dtET );

        metropolisInfo.currentLogOverlap=currentLogOverlapSave;
    }
    //
    logWeight = 0.0;
    // if( (method.BPMetroStabilizeStep - metropolisInfo.inBlockIndex) % method.BPMetroStabilizeStep == 0 )
    // {
        // walkerLeft.stabilize();
    // }

    ////////////////////////////////////////
    // keep track of the walkerLeft
    ////////////////////////////////////////
    // complex <double> logWeightTemp = walkerLeft.getLogw();
    // logWeight = real(walkerLeft.getLogw());
    // /////////////////////////////////
    // // logWeight = 0.0;
    // /////////////////////////////////
    // walkerLeft.logwRef()= logWeightTemp - logWeight;  
    // metropolisInfo.currentLogOverlap -= conj(logWeight);
    ////////////////////////////////////////
    // logWeight = expMinusDtK_Jastrow_vec->at(j_Jastrow).logw.real() + expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySample_logw_FromAux(metropolisInfo.auxiliaryFields[metropolisInfo.inBlockIndex]).real();
    // metropolisInfo.currentLogOverlap -= conj(logWeight);
    ////////////////////////////////////////
    // end = std::chrono::high_resolution_clock::now();
    // if(method.BPMetroUpdateType == "global_fast" && metropolisInfo.numOfJastrow == 1 && metropolisInfo.JastrowSlice[0] == 1){
    //     auto elapsed_updateToRightOneStep = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    //     auto elapsed_getLogOverlapFromAux = std::chrono::duration_cast<std::chrono::nanoseconds>(globalFast_end - globalFast_begin);
    //     if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
    //     if(MPIRank() == 0)printf("Time measured for elapsed_updateToRightOneStep: %.8f seconds.\n", elapsed_updateToRightOneStep.count() * 1e-9);
    //     if(MPIRank() == 0)printf("Time measured for elapsed_getLogOverlapFromAux: %.8f seconds.\n", elapsed_getLogOverlapFromAux.count() * 1e-9);
    //     if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
    //     exit(1);
    // }
    ////////////////////////////////////////

}

void Metropolis::updateToLeftOneStep(Walker &walkerRight, complex<double> &logWeight, vector<int> flip_i_vec, bool ifFrozen)
{
    //
    int j_Jastrow=metropolisInfo.numOfJastrow-1;
    int tempConst=0;
    for(int i= metropolisInfo.numOfJastrow-1; i>= 0; i--){
        tempConst += metropolisInfo.JastrowSlice[i];
        if(metropolisInfo.inBlockIndex>=tempConst){
            j_Jastrow--;
        }
    }
    //
    TwoBodyAux_Jastrow auxNew;
    Walker walkerSaver=walkerRight;
    Walker walkerRightTemp;
    WalkerWalkerOperation_Jastrow walkerWalkerOperation;
    TwoBodyForce_Jastrow tempForce;
    
    complex<double> logProbNew=0.0;
    complex<double> currentLogProb=0.0;

    complex<double> logOverlapNew,currentLogOverlapSave;
    //
    TensorHao<complex<double>,2> B_proposed(walkerRight.getL(),walkerRight.getL());
    TensorHao<complex<double>,2> overlapMatrix_proposed(walkerRight.getN(),walkerRight.getN());
    TensorHao<complex<double>,2> overlapMatrix_inv_proposed(walkerRight.getN(),walkerRight.getN());
    // 
    complex<double> old_logSampleweight, new_logSampleweight;
    complex<double> overlapRatio_fastUpdate;
    double alpha;

    metropolisInfo.currentLogOverlap += conj(metropolisInfo.logWeightLeftInBlock[metropolisInfo.BPMetroTimesliceBlockSize-metropolisInfo.inBlockIndex]);
    currentLogOverlapSave=metropolisInfo.currentLogOverlap;
    // 
    if(metropolisInfo.inBlockIndex< method.blockNum || ifFrozen){
        alpha=0.0;
    }else{
        if(metropolisInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_HAFQMC_icf"){
            ////////////////////////////////////////////////////////////
            walkerRightTemp=walkerRight;
            if( method.BPMetroForceType == "constForce" )
            {
                // auto begin = std::chrono::high_resolution_clock::now();
                if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
                    auxNew = expMinusDtV_Jastrow_vec->at(j_Jastrow).sampleAuxFromForce(metropolisInfo.constForce_Jastrow[j_Jastrow]);
                }else if(flip_i_vec.size() > 0 && method.BPMetroUpdateType == "local"){
                    auxNew = expMinusDtV_Jastrow_vec->at(j_Jastrow).sampleAuxFromForce_localUpdate(metropolisInfo.constForce_Jastrow[j_Jastrow], metropolisInfo.auxiliaryFields[metropolisInfo.inBlockIndex], flip_i_vec);
                }else{
                    cout<<"Error: UNKNOW BPMetroUpdateType: "<<method.BPMetroUpdateType<<endl;
                    exit(1);
                }

                if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
                    tempForce = metropolisInfo.constForce_Jastrow[j_Jastrow];
                    // 
                    currentLogProb = expMinusDtV_Jastrow_vec->at(j_Jastrow).logProbOfAuxFromForce(metropolisInfo.auxiliaryFields[metropolisInfo.inBlockIndex], metropolisInfo.dynamicForceFields[metropolisInfo.inBlockIndex]); 
                    logProbNew = expMinusDtV_Jastrow_vec->at(j_Jastrow).logProbOfAuxFromForce(auxNew, tempForce); 
                    // 
                    if(method.BPMetroUpdateType == "global_fast" && metropolisInfo.numOfJastrow == 1 && metropolisInfo.JastrowSlice[0] == 1){
                        // In global_fast, we only need to update aux related objects and we don't update walkerRight/walkerLeft in the middle of chain
                    }else{
                        TwoBodySample_Jastrow twoBodySample;
                        twoBodySample = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySampleFromAux(auxNew);
                        if(metropolisInfo.KVorder[j_Jastrow]=="VK"){
                            twoBodySampleWalkerRightOperation.reset("fixedOrder", metropolisInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                            oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRight, walkerRightTemp);
                            twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp, walkerRight);
                        }else if(metropolisInfo.KVorder[j_Jastrow]=="K^daggerV"){
                            twoBodySampleWalkerRightOperation.reset("fixedOrder", metropolisInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                            twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRight, walkerRightTemp);
                            oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRight);
                        }else if(metropolisInfo.KVorder[j_Jastrow]=="KVK"){
                            twoBodySampleWalkerRightOperation.reset("dynamicOrder");
                            oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRight, walkerRightTemp);
                            twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp, walkerRight);
                            oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRight, walkerRightTemp);
                            walkerRight = walkerRightTemp;
                        }else{
                            cout<<"Error: UNKNOW metropolisInfo.KVorder[j_Jastrow]: "<<metropolisInfo.KVorder[j_Jastrow]<<endl;
                            exit(1);
                        }
                    }
                }else if(flip_i_vec.size() > 0 && method.BPMetroUpdateType == "local"){
                    tempForce = metropolisInfo.constForce_Jastrow[j_Jastrow];
                    currentLogProb = expMinusDtV_Jastrow_vec->at(j_Jastrow).logProbOfAuxFromForce(metropolisInfo.auxiliaryFields[metropolisInfo.inBlockIndex], metropolisInfo.dynamicForceFields[metropolisInfo.inBlockIndex]); 
                    logProbNew = expMinusDtV_Jastrow_vec->at(j_Jastrow).logProbOfAuxFromForce(auxNew, tempForce); 
                    // 
                    old_logSampleweight = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySample_logw_FromAux(metropolisInfo.auxiliaryFields[metropolisInfo.inBlockIndex]);
                    new_logSampleweight = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySample_logw_FromAux(auxNew);
                    // 
                    overlapRatio_fastUpdate = performLocalUpdateWithFastOverlapRatio(j_Jastrow, metropolisInfo.inBlockIndex, flip_i_vec, auxNew, B_proposed, overlapMatrix_inv_proposed);
                }
                ////////////////////////////////////////////////////////////
            }else{
                cout<<"Error: UNKNOW BPMetroForceType: "<<method.BPMetroForceType<<endl;
                exit(1);
            }
        }
        else{
            cout<<"Error: UNKNOW metropolisInfo.variableName_vec[j_Jastrow] in update: "<<metropolisInfo.variableName_vec[j_Jastrow]<<endl;
            exit(1);
        }
        //
        // walkerRight.addLogw(method.Metro_dtET );
        //
        if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
            ///////////////////////////////////////////////
            if(method.BPMetroUpdateType == "global_fast" && metropolisInfo.numOfJastrow == 1 && metropolisInfo.JastrowSlice[0] == 1){
                logOverlapNew = getLogOverlapFromAux(auxNew, 2, overlapMatrix_proposed);
            }else{
                walkerWalkerOperation.set( metropolisInfo.walkerLeftInBlock[metropolisInfo.BPMetroTimesliceBlockSize-metropolisInfo.inBlockIndex-1], walkerRight );
                logOverlapNew = walkerWalkerOperation.returnLogOverlap();
            }
            ///////////////////////////////////////////////
            //
            alpha = abs( exp(logOverlapNew-logProbNew-metropolisInfo.currentLogOverlap+currentLogProb) );
        }else if(flip_i_vec.size() > 0 && method.BPMetroUpdateType == "local"){
            // fast evaluation of alpha through Matrix Determinant Lemma
            logOverlapNew = log(overlapRatio_fastUpdate)+ new_logSampleweight - old_logSampleweight + metropolisInfo.currentLogOverlap;
            alpha = abs(exp(log(overlapRatio_fastUpdate)+new_logSampleweight-old_logSampleweight-logProbNew+currentLogProb)  );
        }

        if(metropolisInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_K_HAFQMC_icf" && abs(alpha-1.0) >= 10e-8 ){
            cout<<"Error: abs(alpha-1.0) >= 10e-8 in generalHamiltonian_K_HAFQMC_icf to right: "<<alpha<<endl;
            exit(1);
        }
    }

    if( uniformHao() < alpha )
    {
        if(metropolisInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_HAFQMC_icf"){
            metropolisInfo.auxiliaryFields[metropolisInfo.inBlockIndex] = move(auxNew);
            metropolisInfo.dynamicForceFields[metropolisInfo.inBlockIndex]=tempForce;
            // 
            if(flip_i_vec.size() > 0 && method.BPMetroUpdateType == "local"){
                metropolisInfo.B = B_proposed;
                metropolisInfo.overlapMatrix_inv = overlapMatrix_inv_proposed;
            }else if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
                metropolisInfo.overlapMatrix = overlapMatrix_proposed;
            }
        }else{
            cout<<"Error: UNKNOW metropolisInfo.variableName_vec[j_Jastrow]: "<<metropolisInfo.variableName_vec[j_Jastrow]<<endl;
            exit(1);
        }
        //
        metropolisInfo.currentLogOverlap=logOverlapNew;
    }
    else
    {
        if(metropolisInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_HAFQMC_icf"){
            if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
                if(method.BPMetroUpdateType == "global_fast" && metropolisInfo.numOfJastrow == 1 && metropolisInfo.JastrowSlice[0] == 1){
                }else{
                    TwoBodySample_Jastrow twoBodySample;
                    if( method.BPMetroForceType == "constForce" ){
                        twoBodySample = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySampleFromAux(metropolisInfo.auxiliaryFields[metropolisInfo.inBlockIndex]);    
                        // 
                    }
                    else{
                        cout<<"Error: UNKNOW BPMetroForceType: "<<method.BPMetroForceType<<endl;
                        exit(1);
                    }
                    if(metropolisInfo.KVorder[j_Jastrow]=="VK"){
                        twoBodySampleWalkerRightOperation.reset("fixedOrder", metropolisInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                        oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerSaver, walkerRightTemp);
                        twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp, walkerRight);
                    }else if(metropolisInfo.KVorder[j_Jastrow]=="K^daggerV"){
                        twoBodySampleWalkerRightOperation.reset("fixedOrder", metropolisInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                        twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerSaver, walkerRightTemp);
                        oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRight);
                    }else if(metropolisInfo.KVorder[j_Jastrow]=="KVK"){
                        twoBodySampleWalkerRightOperation.reset("dynamicOrder");
                        oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerSaver, walkerRightTemp);
                        twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp, walkerRight);
                        oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRight, walkerRightTemp);
                        walkerRight = walkerRightTemp;
                    }else{
                        cout<<"Error: UNKNOW metropolisInfo.KVorder[j_Jastrow]: "<<metropolisInfo.KVorder[j_Jastrow]<<endl;
                        exit(1);
                    }
                }
            }
            //
        }
        else{
            cout<<"Error: UNKNOW metropolisInfo.variableName_vec[j_Jastrow] in update save: "<<metropolisInfo.variableName_vec[j_Jastrow]<<endl;
            exit(1);
        }
        /////////////////////////////////
        // walkerRight.addLogw(method.Metro_dtET );
        /////////////////////////////////
        
        metropolisInfo.currentLogOverlap=currentLogOverlapSave;
    }

    logWeight = 0.0;
    // if( (method.BPMetroStabilizeStep - metropolisInfo.inBlockIndex) % method.BPMetroStabilizeStep == 0 )
    // {
        // walkerRight.stabilize();
    // }

    ////////////////////////////////////////
    // keep track of the walkerLeft
    ////////////////////////////////////////
    // complex <double> logWeightTemp = walkerRight.getLogw();
    // logWeight = real(walkerRight.getLogw());
    // /////////////////////////////////
    // //logWeight = 0.0;
    // /////////////////////////////////
    // walkerRight.logwRef()= logWeightTemp - logWeight;   
    // metropolisInfo.currentLogOverlap -= logWeight;
    ////////////////////////////////////////
    // logWeight = expMinusDtK_Jastrow_vec->at(j_Jastrow).logw.real() + expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySample_logw_FromAux(metropolisInfo.auxiliaryFields[metropolisInfo.inBlockIndex]).real();
    // metropolisInfo.currentLogOverlap -= logWeight;
}

////////////////////////////////////////////////////////////////////
void Metropolis::initial_globalFast()
{
    // 
    Walker walkerRightTemp;
    if(metropolisInfo.KVorder[0]=="VK"){
        // 
        // A0 = walkerLeft0Wfdagger @ K
        metropolisInfo.A0.resize(metropolisInfo.walkerLeftInBlock[0].getWf().rank(1), expMinusDtK_Jastrow_vec->at(0).matrix.rank(1));
        BL_NAME(gmm)(trans(conj(metropolisInfo.walkerLeftInBlock[0].getWf())), expMinusDtK_Jastrow_vec->at(0).matrix, metropolisInfo.A0);
        // A0WR = A0 @ walkerRight0Wf
        metropolisInfo.A0WR.resize(metropolisInfo.A0.rank(0), metropolisInfo.walkerRightInBlock[0].getWf().rank(1));
        BL_NAME(gmm)(metropolisInfo.A0, metropolisInfo.walkerRightInBlock[0].getWf(), metropolisInfo.A0WR);
        // A1 = walkerLeft0Wfdagger @ U0
        metropolisInfo.A1.resize(metropolisInfo.walkerLeftInBlock[0].getWf().rank(1), expMinusDtV_Jastrow_vec->at(0).getSVD_DRank());
        BL_NAME(gmm)(trans(conj(metropolisInfo.walkerLeftInBlock[0].getWf())), expMinusDtV_Jastrow_vec->at(0).getSqrtMinusDtSVDVecsMatrix_SVD(0, "U0"), metropolisInfo.A1);
        // B0 = Vdagger0 @ K
        metropolisInfo.B0.resize(expMinusDtV_Jastrow_vec->at(0).getSVD_DRank(), expMinusDtK_Jastrow_vec->at(0).matrix.rank(1));
        BL_NAME(gmm)(expMinusDtV_Jastrow_vec->at(0).getSqrtMinusDtSVDVecsMatrix_SVD(0, "Vdagger0"), expMinusDtK_Jastrow_vec->at(0).matrix, metropolisInfo.B0);
        // B0WR = B0 @ walkerRight0Wf
        metropolisInfo.B0WR.resize(metropolisInfo.B0.rank(0), metropolisInfo.walkerRightInBlock[0].getWf().rank(1));
        BL_NAME(gmm)(metropolisInfo.B0, metropolisInfo.walkerRightInBlock[0].getWf(), metropolisInfo.B0WR);
        // C0 = Vdagger0 @ U0
        metropolisInfo.C0.resize(expMinusDtV_Jastrow_vec->at(0).getSVD_DRank(), expMinusDtV_Jastrow_vec->at(0).getSVD_DRank());
        BL_NAME(gmm)(expMinusDtV_Jastrow_vec->at(0).getSqrtMinusDtSVDVecsMatrix_SVD(0, "Vdagger0"), expMinusDtV_Jastrow_vec->at(0).getSqrtMinusDtSVDVecsMatrix_SVD(0, "U0"), metropolisInfo.C0);
        // 
        metropolisInfo.D.resize(expMinusDtV_Jastrow_vec->at(0).getSVD_DRank(),expMinusDtV_Jastrow_vec->at(0).getSVD_DRank());
    }else{
        cout<<"Error: UNKNOW metropolisInfo.KVorder[0]: "<<metropolisInfo.KVorder[0]<<endl;
        exit(1);
    }

    metropolisInfo.globalFastInitialized = true;
    metropolisInfo.globalFastUpdated = true;

}

void Metropolis::update_globalFast()
{
    size_t N = metropolisInfo.walkerLeftInBlock[metropolisInfo.BPMetroTimesliceBlockSize].getN();
    TensorHao<complex<double>,2> overlapMatrix(N, N);
    BL_NAME(gmm)( metropolisInfo.walkerLeftInBlock[metropolisInfo.BPMetroTimesliceBlockSize].getWf(), metropolisInfo.walkerRightInBlock[0].getWf(), overlapMatrix, 'C' );
    // 
    metropolisInfo.overlapMatrix = overlapMatrix;
    // 
    if(metropolisInfo.KVorder[0]=="VK"){
        Walker walkerRightTemp;
        oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(0), metropolisInfo.walkerRightInBlock[0], walkerRightTemp);
        // 
        // A0WR = A0 @ walkerRight0Wf
        metropolisInfo.A0WR.resize(metropolisInfo.A0.rank(0), metropolisInfo.walkerRightInBlock[0].getWf().rank(1));
        BL_NAME(gmm)(metropolisInfo.A0, metropolisInfo.walkerRightInBlock[0].getWf(), metropolisInfo.A0WR);
        // B0WR = B0 @ walkerRight0Wf
        metropolisInfo.B0WR.resize(metropolisInfo.B0.rank(0), metropolisInfo.walkerRightInBlock[0].getWf().rank(1));
        BL_NAME(gmm)(metropolisInfo.B0, metropolisInfo.walkerRightInBlock[0].getWf(), metropolisInfo.B0WR);
    }else{
        cout<<"Error: UNKNOW metropolisInfo.KVorder[0]: "<<metropolisInfo.KVorder[0]<<endl;
        exit(1);
    }
    // 
    metropolisInfo.globalFastUpdated = true;
}

void Metropolis::getAuxMatrix_D(TwoBodyAux_Jastrow auxNew, size_t j_Jastrow)
{
    //  directly calculate the overlap matrix from aux and predefined matrix
    int truncatedD = expMinusDtV_Jastrow_vec->at(j_Jastrow).getSVD_DRank();
    metropolisInfo.D = 0.0;
    // 
    //Calculate aux * sqrtMinusDt * svdVecs
    TensorHaoRef<complex<double>, 1> vecsAux(truncatedD*truncatedD);
    TensorHaoRef<complex<double>, 2> vecs(truncatedD*truncatedD, expMinusDtV_Jastrow_vec->at(j_Jastrow).getSVDNumber());
    vecsAux.point( metropolisInfo.D.data() );
    vecs.point( const_cast<complex<double>*> ( expMinusDtV_Jastrow_vec->at(j_Jastrow).getSqrtMinusDtSVDVecs_D().data() ) );
    BL_NAME(gemv)(vecs, auxNew, vecsAux);
}


TensorHao<complex<double>, 2> Metropolis::getAuxMatrix_D_exp_x(TwoBodyAux_Jastrow auxNew, size_t j_Jastrow)
{
    ///////////////
    // Timer
    ///////////////
    // auto begin = std::chrono::high_resolution_clock::now();
    // auto getAuxMatrix_Dup_begin = std::chrono::high_resolution_clock::now();
    // auto getAuxMatrix_Dup_end = std::chrono::high_resolution_clock::now();
    // auto end = std::chrono::high_resolution_clock::now();
    ///////////////
    // getAuxMatrix_Dup_begin = std::chrono::high_resolution_clock::now();
    /////////////////////////////////////////////////
    getAuxMatrix_D(auxNew, 0);
    /////////////////////////////////////////////////
    // getAuxMatrix_Dup_end = std::chrono::high_resolution_clock::now();
    /////////////////////////////////////////////////
    // 
    // Dup_exp_x = Sup + 0.5 * Sup @ C0 @ Sup
    TensorHao<complex<double>, 2> SC0(metropolisInfo.D.rank(0), metropolisInfo.C0.rank(1));
    BL_NAME(gmm)(metropolisInfo.D, metropolisInfo.C0, SC0);
    TensorHao<complex<double>, 2> D_exp_x(metropolisInfo.D.rank(0), metropolisInfo.D.rank(1));
    BL_NAME(gmm)(SC0, metropolisInfo.D, D_exp_x);
    for(int j=1-1; j<=D_exp_x.rank(1)-1; j++){
        for(int i=1-1; i<=D_exp_x.rank(0)-1; i++){
            D_exp_x(i,j) = metropolisInfo.D(i,j) + 0.5 * D_exp_x(i,j);
        }
    }   
    /////////////////////////////////////////////////
    // end = std::chrono::high_resolution_clock::now();
    // auto elapsed_whole = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    // auto elapsed_getAuxMatrix_Dup = std::chrono::duration_cast<std::chrono::nanoseconds>(getAuxMatrix_Dup_end - getAuxMatrix_Dup_begin);
    // if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
    // if(MPIRank() == 0)printf("Time measured for elapsed_whole: %.8f seconds.\n", elapsed_whole.count() * 1e-9);
    // if(MPIRank() == 0)printf("Time measured for elapsed_getAuxMatrix_Dup: %.8f seconds.\n", elapsed_getAuxMatrix_Dup.count() * 1e-9);
    // if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
    /////////////////////////////////////////////////
    // 
    return D_exp_x;
}

complex<double> Metropolis::getLogOverlapFromAux(TwoBodyAux_Jastrow auxNew, size_t expM, TensorHao<std::complex<double>, 2> & overlapMatrix)
{
    ///////////////
    // Timer
    ///////////////
    // auto begin = std::chrono::high_resolution_clock::now();
    // auto end = std::chrono::high_resolution_clock::now();
    ///////////////
    if(!metropolisInfo.globalFastInitialized){
        initial_globalFast();
    }
    if(!metropolisInfo.globalFastUpdated){
        update_globalFast();
    }
    //  directly calculate the overlap matrix from aux and predefined matrix
    if(expM == 2 && metropolisInfo.globalFastInitialized && metropolisInfo.globalFastUpdated){
        // 
        int N = metropolisInfo.B0WR.rank(1);
        int truncatedD = metropolisInfo.truncatedD;
        // 
        if(metropolisInfo.B0WR.rank(0) > metropolisInfo.B0WR.rank(1)){
            // cost: O(N_e M^2)
            getAuxMatrix_D(auxNew, 0);
            // 
            // SB0WR = S @ B0WR
            TensorHao<complex<double>, 2> SB0WR(truncatedD, N);
            BL_NAME(gmm)(metropolisInfo.D, metropolisInfo.B0WR, SB0WR);
            // C0SB0WR = C0 @ SB0WR
            TensorHao<complex<double>, 2> C0SB0WR(truncatedD, N);
            BL_NAME(gmm)(metropolisInfo.C0, SB0WR, C0SB0WR);
            // SC0SB0WR = S @ C0SB0WR
            TensorHao<complex<double>, 2> SC0SB0WR(truncatedD, N);
            BL_NAME(gmm)(metropolisInfo.D, C0SB0WR, SC0SB0WR);
            // temp = SB0WR + 0.5 * SC0SB0WR
            TensorHao<complex<double>, 2>& temp = SB0WR;
            for(int j=1-1; j<=temp.rank(1)-1; j++){
                for(int i=1-1; i<=temp.rank(0)-1; i++){
                    temp(i,j) = SB0WR(i,j) + 0.5 * SC0SB0WR(i,j);
                }
            }   
            // A1temp = A1 @ temp
            TensorHao<complex<double>, 2> A1temp(N, N);
            BL_NAME(gmm)(metropolisInfo.A1, temp, A1temp);
            overlapMatrix = metropolisInfo.A0WR + A1temp;
        }else{
        ////////////////////////////////////////////////////
            // cost: O(M^3)
            TensorHao<complex<double>, 2> D_exp_x = getAuxMatrix_D_exp_x(auxNew, 0);
            // A1D_exp_x = A1 @ D_exp_x
            // A1D_exp_xB0WR = A1D_exp_x @ B0WR
            TensorHao<complex<double>, 2> A1D_exp_x(N, truncatedD);
            TensorHao<complex<double>, 2> A1D_exp_xB0WR(N, N);
            BL_NAME(gmm)(metropolisInfo.A1, D_exp_x, A1D_exp_x);
            BL_NAME(gmm)(A1D_exp_x, metropolisInfo.B0WR, A1D_exp_xB0WR);

            overlapMatrix = metropolisInfo.A0WR + A1D_exp_xB0WR;
            /////////////////////////////
        }
    }else{
        cout<<"Error: UNKNOW expM or no globalFastInitialized in getLogOverlapFromAux. expM: "<<expM<<" globalFastInitialized: "<<metropolisInfo.globalFastInitialized<<" globalFastUpdated: "<<metropolisInfo.globalFastUpdated<<endl;
        exit(1);
    }
    TensorHao<complex<double>, 2> overlapMatrixTemp = overlapMatrix;
    complex<double> logOverlap = logDeterminant(BL_NAME(LUconstruct)( move(overlapMatrixTemp) )) ;
    logOverlap += conj(metropolisInfo.walkerLeftInBlock[0].getLogw()) + metropolisInfo.walkerRightInBlock[0].getLogw();
    logOverlap += expMinusDtV_Jastrow_vec->at(0).getTwoBodySample_logw_FromAux(auxNew);
    logOverlap += expMinusDtK_Jastrow_vec->at(0).logw;

    /////////////////////////////////////////////////
    // end = std::chrono::high_resolution_clock::now();
    // auto elapsed_whole = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    // if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
    // if(MPIRank() == 0)printf("Time measured for elapsed_whole: %.8f seconds.\n", elapsed_whole.count() * 1e-9);
    // if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
    /////////////////////////////////////////////////

    return logOverlap;
}


void Metropolis::getOverlapMatrixInvA1ExpS(TensorHao<std::complex<double>, 2> & overlapMatrixInvA1ExpS)
{
    if(!metropolisInfo.globalFastInitialized){
        initial_globalFast();
    }
    if(!metropolisInfo.globalFastUpdated){
        update_globalFast();
    }
    // 
    //  directly calculate the overlap matrix from aux and predefined matrix
    if(metropolisInfo.globalFastInitialized && metropolisInfo.globalFastUpdated){
        int N = metropolisInfo.B0WR.rank(1);
        int truncatedD = metropolisInfo.truncatedD;
        // 
        if(metropolisInfo.B0WR.rank(0) > metropolisInfo.B0WR.rank(1)){
            // cost: O(Ne M^2)
            getAuxMatrix_D(metropolisInfo.auxiliaryFields[0], 0);
            // A1S = A1 @ S
            TensorHao<complex<double>, 2> A1S(N, truncatedD);
            BL_NAME(gmm)(metropolisInfo.A1, metropolisInfo.D, A1S);
            // A1SC0 = A1S @ C0
            TensorHao<complex<double>, 2> A1SC0(N, truncatedD);
            BL_NAME(gmm)(A1S, metropolisInfo.C0, A1SC0);
            // A1SC0S = A1SC0 @ S
            TensorHao<complex<double>, 2> A1SC0S(N, truncatedD);
            BL_NAME(gmm)(A1SC0, metropolisInfo.D, A1SC0S);
            // tempUp = A1S + 0.5 * A1SC0S
            TensorHao<complex<double>, 2> temp(N, truncatedD);
            for(int i=1-1; i<=A1S.rank(0)-1; i++){
                for(int j=1-1; j<=A1S.rank(1)-1; j++){
                    temp(i,j) = A1S(i,j) + 0.5 * A1SC0S(i,j);
                }
            }
            overlapMatrixInvA1ExpS.resize(N, truncatedD);
            BL_NAME(gmm)(metropolisInfo.overlapMatrix_inv, temp, overlapMatrixInvA1ExpS);
        }else{
            // cost: O(M^3)
            TensorHao<complex<double>, 2> D_exp_x = getAuxMatrix_D_exp_x(metropolisInfo.auxiliaryFields[0], 0);
            ////////////////////////////////////////////////
            // A1D_exp_x = A1 @ D_exp_x
            TensorHao<complex<double>, 2> A1D_exp_x(N, D_exp_x.rank(1));
            BL_NAME(gmm)(metropolisInfo.A1, D_exp_x, A1D_exp_x);
            // 
            overlapMatrixInvA1ExpS.resize(metropolisInfo.overlapMatrix_inv.rank(0), A1D_exp_x.rank(1));
            BL_NAME(gmm)(metropolisInfo.overlapMatrix_inv, A1D_exp_x, overlapMatrixInvA1ExpS);
            /////////////////////////////
        }
        /////////////////////////////////////////////////////
    }else{
        cout<<"Error: UNKNOW expM or no globalFastInitialized in getOverlapMatrixInvA1ExpS.: "<<" globalFastInitialized: "<<metropolisInfo.globalFastInitialized<<" globalFastUpdated: "<<metropolisInfo.globalFastUpdated<<endl;
        exit(1);
    }
}

void Metropolis::getOverlapMatrixInvA1ExpS_A1ExpS(TensorHao<std::complex<double>, 2> & overlapMatrixInvA1ExpS, TensorHao<std::complex<double>, 2> & A1ExpS_12)
{
    if(!metropolisInfo.globalFastInitialized){
        initial_globalFast();
    }
    if(!metropolisInfo.globalFastUpdated){
        update_globalFast();
    }
    // 
    //  directly calculate the overlap matrix from aux and predefined matrix
    int N = metropolisInfo.B0WR.rank(1);
    int truncatedD = metropolisInfo.truncatedD;
    // 
    if(metropolisInfo.globalFastInitialized && metropolisInfo.globalFastUpdated){
        getAuxMatrix_D(metropolisInfo.auxiliaryFields[0], 0);
        ////////////////////////////////////////////////
        // A0 = walkerLeft0Wfdagger @ K
        // A0WR = A0 @ walkerRight0Wf
        // A1 = walkerLeft0Wfdagger @ U0
        // B0 = Vdagger0 @ K
        // B0WR = B0 @ walkerRight0Wf
        // C0 = Vdagger0 @ U0
        // 
        // A1S = A1 @ S
        TensorHao<complex<double>, 2> A1S(N, truncatedD);
        BL_NAME(gmm)(metropolisInfo.A1, metropolisInfo.D, A1S);
        // C0S = C0 @ S
        TensorHao<complex<double>, 2> C0S(truncatedD, truncatedD);
        BL_NAME(gmm)(metropolisInfo.C0, metropolisInfo.D, C0S);
        // A1SC0S = A1S @ C0S 
        TensorHao<complex<double>, 2> A1SC0S(N, truncatedD);
        BL_NAME(gmm)(A1S, C0S, A1SC0S);
        // A1ExpS_12 = A1S + halfA1SC0S
        A1ExpS_12.resize(N, truncatedD);
        for(int i=1-1; i<=A1ExpS_12.rank(0)-1; i++){
            for(int j=1-1; j<=A1ExpS_12.rank(1)-1; j++){
                A1ExpS_12(i,j) = A1S(i,j) + 0.5 * A1SC0S(i,j);
            }
        }
        // 
        overlapMatrixInvA1ExpS.resize(N, truncatedD);
        BL_NAME(gmm)(metropolisInfo.overlapMatrix_inv, A1ExpS_12, overlapMatrixInvA1ExpS);
        
        /////////////////////////////////////////////////////
    }else{
        cout<<"Error: UNKNOW expM or no globalFastInitialized in getOverlapMatrixInvA1ExpS.: "<<" globalFastInitialized: "<<metropolisInfo.globalFastInitialized<<" globalFastUpdated: "<<metropolisInfo.globalFastUpdated<<endl;
        exit(1);
    }
}

void Metropolis::get_PWR(TensorHao<std::complex<double>, 2> P_matrix, TensorHao<std::complex<double>, 2> & PWR)
{
    if(!metropolisInfo.globalFastInitialized){
        initial_globalFast();
    }
    if(!metropolisInfo.globalFastUpdated){
        update_globalFast();
    }
    //  directly calculate the overlap matrix from aux and predefined matrix
    if(metropolisInfo.globalFastInitialized && metropolisInfo.globalFastUpdated){
        ////////////////////////////////////////////////
        // A0 = walkerLeft0Wfdagger @ K
        // A0WR = A0 @ walkerRight0Wf
        // A1 = walkerLeft0Wfdagger @ U0
        // B0 = Vdagger0 @ K
        // B0WR = B0 @ walkerRight0Wf
        // C0 = Vdagger0 @ U0
        // 
        // PWR = P @ walkerRight0Wf
        PWR.resize(P_matrix.rank(0), metropolisInfo.walkerRightInBlock[0].getWf().rank(1));
        BL_NAME(gmm)(P_matrix, metropolisInfo.walkerRightInBlock[0].getWf(), PWR);
        /////////////////////////////////////////////////////
    }else{
        cout<<"Error: UNKNOW expM or no globalFastInitialized in getLogOverlapFromAux.: "<<" globalFastInitialized: "<<metropolisInfo.globalFastInitialized<<" globalFastUpdated: "<<metropolisInfo.globalFastUpdated<<endl;
        exit(1);
    }
}

void Metropolis::get_A0PWR_B0PWR(TensorHao<std::complex<double>, 2> P_matrix, TensorHao<std::complex<double>, 2> & A0PWR, TensorHao<std::complex<double>, 2> & B0PWR)
{
    if(!metropolisInfo.globalFastInitialized){
        initial_globalFast();
    }
    if(!metropolisInfo.globalFastUpdated){
        update_globalFast();
    }
    //  directly calculate the overlap matrix from aux and predefined matrix
    if(metropolisInfo.globalFastInitialized && metropolisInfo.globalFastUpdated){
        ////////////////////////////////////////////////
        // A0 = walkerLeft0Wfdagger @ K
        // A0WR = A0 @ walkerRight0Wf
        // A1 = walkerLeft0Wfdagger @ U0
        // B0 = Vdagger0 @ K
        // B0WR = B0 @ walkerRight0Wf
        // C0 = Vdagger0 @ U0
        // 
        // PWR = P @ walkerRight0Wf
        TensorHao<complex<double>, 2> PWR(P_matrix.rank(0), metropolisInfo.walkerRightInBlock[0].getWf().rank(1));
        BL_NAME(gmm)(P_matrix, metropolisInfo.walkerRightInBlock[0].getWf(), PWR);
        // A0PWR = A0 @ PWR
        A0PWR.resize(metropolisInfo.A0.rank(0), PWR.rank(1));
        BL_NAME(gmm)(metropolisInfo.A0, PWR, A0PWR);
        // B0PWR = B0 @ PWR
        B0PWR.resize(metropolisInfo.B0.rank(0), PWR.rank(1));
        BL_NAME(gmm)(metropolisInfo.B0, PWR, B0PWR);
        /////////////////////////////////////////////////////
    }else{
        cout<<"Error: UNKNOW expM or no globalFastInitialized in getLogOverlapFromAux.: "<<" globalFastInitialized: "<<metropolisInfo.globalFastInitialized<<" globalFastUpdated: "<<metropolisInfo.globalFastUpdated<<endl;
        exit(1);
    }
}

void Metropolis::get_A0_A1ExpS12B0(TensorHao<std::complex<double>, 2> & A1ExpS, TensorHao<std::complex<double>, 2> & A0_A1ExpS12B0)
{
    if(!metropolisInfo.globalFastInitialized){
        initial_globalFast();
    }
    if(!metropolisInfo.globalFastUpdated){
        update_globalFast();
    }
    //  directly calculate the overlap matrix from aux and predefined matrix
    if(metropolisInfo.globalFastInitialized && metropolisInfo.globalFastUpdated){
        ////////////////////////////////////////////////
        // A0 = walkerLeft0Wfdagger @ K
        // A0WR = A0 @ walkerRight0Wf
        // A1 = walkerLeft0Wfdagger @ U0
        // B0 = Vdagger0 @ K
        // B0WR = B0 @ walkerRight0Wf
        // C0 = Vdagger0 @ U0
        // 
        // PWR = P @ walkerRight0Wf
        // A1S = A1 @ S
        // 
        TensorHao<complex<double>, 2> A1ExpS12B0(A1ExpS.rank(0), metropolisInfo.B0.rank(1));
        BL_NAME(gmm)(A1ExpS, metropolisInfo.B0, A1ExpS12B0);
        // 
        A0_A1ExpS12B0.resize(A1ExpS.rank(0), metropolisInfo.B0.rank(1));
        A0_A1ExpS12B0 = metropolisInfo.A0 + A1ExpS12B0;
        /////////////////////////////////////////////////////
    }else{
        cout<<"Error: UNKNOW expM or no globalFastInitialized in getLogOverlapFromAux.: "<<" globalFastInitialized: "<<metropolisInfo.globalFastInitialized<<" globalFastUpdated: "<<metropolisInfo.globalFastUpdated<<endl;
        exit(1);
    }
}

///////////////////////////////////////////////////////////////////
void Metropolis::updateDirectB()
{
    TwoBodySample_Jastrow twoBodySample = expMinusDtV_Jastrow_vec->at(0).getTwoBodySampleFromAux(metropolisInfo.auxiliaryFields[0]);
    // update B
    metropolisInfo.B = twoBodySample.matrix;
}


void Metropolis::updateDirectOverlapMatrix_inv(Walker &walkerRight,
                                                TensorHao<std::complex<double>, 2> &overlapMatrix_inv)
{
    overlapMatrix_inv.resize(walkerRight.getWf().rank(1), walkerRight.getWf().rank(1));
    BL_NAME(gmm)(trans(conj(metropolisInfo.walkerLeftInBlock[metropolisInfo.BPMetroTimesliceBlockSize].getWf())), walkerRight.getWf(), overlapMatrix_inv);
    BL_NAME(inverse)(overlapMatrix_inv);
}

void Metropolis::updateOverlapMatrix_inv_fromOverlapMatrix()
{
    metropolisInfo.overlapMatrix_inv = metropolisInfo.overlapMatrix;
    BL_NAME(inverse)(metropolisInfo.overlapMatrix_inv);
}
///////////////////////////////////////////////////////////////////

void Metropolis::updateOverlapMatrixInvWithSMW(TensorHao<std::complex<double>, 2> &A_inv,
                                               TensorHao<std::complex<double>, 2> U,
                                               TensorHao<std::complex<double>, 2> Vdagger)
{
    // Using Sherman-Morrison-Woodbury formula to update A^{-1}:
    // (A + UV^T)^{-1} = A^{-1} - A^{-1}U(I + V^T A^{-1} U)^{-1} V^T A^{-1}
    
    // Step 1: Calculate A^{-1}U
    TensorHao<std::complex<double>, 2> Ainv_U(A_inv.rank(0), U.rank(1));
    BL_NAME(gmm)(A_inv, U, Ainv_U);
    
    // Step 2: Calculate V^T A^{-1}
    TensorHao<std::complex<double>, 2> V_T_Ainv(Vdagger.rank(0), A_inv.rank(1));
    BL_NAME(gmm)(Vdagger, A_inv, V_T_Ainv);
    
    // Step 3: Calculate I + V^T A^{-1} U
    TensorHao<std::complex<double>, 2> I_plus_VTAinvU(V_T_Ainv.rank(0), U.rank(1));
    BL_NAME(gmm)(V_T_Ainv, U, I_plus_VTAinvU);
    
    // Add identity matrix
    for(size_t i = 0; i < I_plus_VTAinvU.rank(0); ++i) {
        for(size_t j = 0; j < I_plus_VTAinvU.rank(1); ++j) {
            if(i == j) {
                I_plus_VTAinvU(i, j) += 1.0;
            }
        }
    }
    
    // Step 4: Calculate (I + V^T A^{-1} U)^{-1}
    LUDecomp<std::complex<double>> LU_I_plus = BL_NAME(LUconstruct)(move(I_plus_VTAinvU));
    TensorHao<std::complex<double>, 2> inv_I_plus_VTAinvU = BL_NAME(inverse)(LU_I_plus);
    
    // Step 5: Calculate A^{-1} U (I + V^T A^{-1} U)^{-1}
    TensorHao<std::complex<double>, 2> AinvU_invIplus(Vdagger.rank(1), inv_I_plus_VTAinvU.rank(1));
    BL_NAME(gmm)(Ainv_U, inv_I_plus_VTAinvU, AinvU_invIplus);
    
    // Step 6: Calculate A^{-1} U (I + V^T A^{-1} U)^{-1} V^T A^{-1}
    TensorHao<std::complex<double>, 2> update_term(AinvU_invIplus.rank(0), V_T_Ainv.rank(1));
    BL_NAME(gmm)(AinvU_invIplus, V_T_Ainv, update_term);
    
    // Step 7: Update A^{-1} = A^{-1} - A^{-1} U (I + V^T A^{-1} U)^{-1} V^T A^{-1}
    for(size_t i = 0; i < A_inv.rank(0); ++i) {
        for(size_t j = 0; j < A_inv.rank(1); ++j) {
            A_inv(i, j) -= update_term(i, j);
        }
    }
}

std::complex<double> Metropolis::calculateTargetOverlapRatioFastUpdate(TensorHao<std::complex<double>, 2> U, 
                                                                       TensorHao<std::complex<double>, 2> Vdagger,
                                                                       TensorHao<std::complex<double>, 2> &overlapMatrix_inv)
{
    // Calculate det(I+V^T*A^{-1}*U) using fast update method
    TensorHao<std::complex<double>, 2> temp1(overlapMatrix_inv.rank(0), U.rank(1));
    BL_NAME(gmm)(overlapMatrix_inv, U, temp1);
    TensorHao<std::complex<double>, 2> temp2(Vdagger.rank(0), temp1.rank(1));
    BL_NAME(gmm)(Vdagger, temp1, temp2);
    
    // Add identity matrix
    for(size_t i_temp = 0; i_temp <= temp2.rank(0)-1; i_temp++) {
        temp2(i_temp, i_temp) += 1.0;
    }
    LUDecomp<complex<double>> temp2_LUOverlap = BL_NAME(LUconstruct)( move(temp2) );
    complex<double> targetOverlapRatio_fastUpdate = determinant(temp2_LUOverlap);
    
    return targetOverlapRatio_fastUpdate;
}

void Metropolis::calculateUpdatedB(TensorHao<std::complex<double>, 2> B, 
                                   TensorHao<std::complex<double>, 2> U_s,
                                   TensorHao<std::complex<double>, 2> Vdagger_s,
                                   TensorHao<std::complex<double>, 2> &updated_B)
{
    // updated_B = B + U_s * Vdagger_s;
    TensorHao<complex<double>,2> temp_UV(U_s.rank(0), Vdagger_s.rank(1));
    BL_NAME(gmm)(U_s, Vdagger_s, temp_UV);
    updated_B = B + temp_UV;
}

void Metropolis::updateOverlapMatricesAndRatios(std::vector<TensorHao<std::complex<double>, 2>> &U_vec,
                                                std::vector<TensorHao<std::complex<double>, 2>> &Vdagger_vec,
                                                TensorHao<std::complex<double>, 2> &updated_overlapMatrix_inv,
                                                std::complex<double> &overlapRatio_fastUpdate)
{
    for(int temp_i=1-1; temp_i<=U_vec.size()-1; temp_i++){
        TensorHao<complex<double>,2> U = U_vec[temp_i];
        TensorHao<complex<double>,2> Vdagger = Vdagger_vec[temp_i];
        // det(I+U^TA^{−1}V)
        overlapRatio_fastUpdate = overlapRatio_fastUpdate * calculateTargetOverlapRatioFastUpdate(U, Vdagger, updated_overlapMatrix_inv);
        updateOverlapMatrixInvWithSMW(updated_overlapMatrix_inv, U, Vdagger);
    }
}

complex<double> Metropolis::performLocalUpdateWithFastOverlapRatio(int j_Jastrow, 
                                                    int inBlockIndex,               
                                                std::vector<int> flip_i_vec,
                                                tensor_hao::TensorHao<std::complex<double>, 1> auxNew,
                                                tensor_hao::TensorHao<std::complex<double>, 2> & B_proposed, 
                                                tensor_hao::TensorHao<std::complex<double>, 2> & overlapMatrix_inv_proposed)
{
    if(inBlockIndex != 0 || j_Jastrow != 0){
        cout<<"Error: Metropolis::performLocalUpdateWithFastOverlapRatio() inBlockIndex != 0 || j_Jastrow != 0: "<<inBlockIndex<<"  "<<j_Jastrow<<endl;
        exit(1);
    }
    Walker walkerRightTemp;
    // 
    TensorHao<std::complex<double>, 2> U_s = expMinusDtV_Jastrow_vec->at(j_Jastrow).getSqrtMinusDtSVDVecsMatrix_SVD(flip_i_vec[0], "U");
    U_s = U_s * (auxNew(flip_i_vec[0])-metropolisInfo.auxiliaryFields[inBlockIndex](flip_i_vec[0]));
    TensorHao<std::complex<double>, 2> Vdagger_s = expMinusDtV_Jastrow_vec->at(j_Jastrow).getSqrtMinusDtSVDVecsMatrix_SVD(flip_i_vec[0], "Vdagger");
    // 
    TensorHao<std::complex<double>, 2> k_matrix = expMinusDtK_Jastrow_vec->at(j_Jastrow).matrix;
    TensorHao<std::complex<double>, 2> wfnRight_matrix = metropolisInfo.walkerRightInBlock[0].getWf();
    TensorHao<std::complex<double>, 2> wfnLeft_matrix = metropolisInfo.walkerLeftInBlock[0].getWf();

    /////////////////////////////////////
    //get U V matrix
    /////////////////////////////////////
    vector<TensorHao<complex<double>,2>> U_vec, Vdagger_vec;
    get_updatedUV_1s(U_vec, Vdagger_vec, metropolisInfo.JastrowExpM[j_Jastrow], metropolisInfo.KVorder[j_Jastrow], wfnLeft_matrix, U_s, Vdagger_s, metropolisInfo.B, k_matrix, wfnRight_matrix);
    // 
    /////////////////////////////////////
    // 
    complex<double> overlapRatio_fastUpdate=1.0;
    // 
    TensorHao<complex<double>,2> updated_B = metropolisInfo.B;
    TensorHao<complex<double>,2> updated_overlapMatrix_inv = metropolisInfo.overlapMatrix_inv;
    // 
    if(metropolisInfo.KVorder[j_Jastrow]=="VK"){
        calculateUpdatedB(metropolisInfo.B, U_s, Vdagger_s, updated_B);
        // 
        updateOverlapMatricesAndRatios(U_vec, Vdagger_vec, updated_overlapMatrix_inv, overlapRatio_fastUpdate);
    }  
    B_proposed = updated_B;
    overlapMatrix_inv_proposed = updated_overlapMatrix_inv;
    
    return overlapRatio_fastUpdate;
}

void Metropolis::updateWalkerLeftMetro(){
    if(method.BPMetroUpdateType == "global_fast" && metropolisInfo.KVorder[0]=="VK" && metropolisInfo.JastrowExpM[0] == 2){
        ////////////////////////////////////////////////
        if(!metropolisInfo.globalFastInitialized){
            initial_globalFast();
        }
        if(!metropolisInfo.globalFastUpdated){
            update_globalFast();
        }
        int L = metropolisInfo.B0.rank(1);

        int N = metropolisInfo.B0WR.rank(1);
        int truncatedD = metropolisInfo.truncatedD;
        ////////////////////////////////////////////////
        // 
        // A0 = walkerLeft0Wfdagger @ K
        // A1 = walkerLeft0Wfdagger @ U0
        // B0 = Vdagger0 @ K
        // C0 = Vdagger0 @ U0
        // 
        getAuxMatrix_D(metropolisInfo.auxiliaryFields[0], 0);
        // A1S = A1 @ S
        TensorHao<complex<double>, 2> A1S(N, truncatedD);
        BL_NAME(gmm)(metropolisInfo.A1, metropolisInfo.D, A1S);
        // C0S = C0 @ S
        TensorHao<complex<double>, 2> C0S(truncatedD, truncatedD);
        BL_NAME(gmm)(metropolisInfo.C0, metropolisInfo.D, C0S);
        // A1SB0 = A1S @ B0
        TensorHao<complex<double>, 2> A1SB0(N, L);
        BL_NAME(gmm)(A1S, metropolisInfo.B0, A1SB0);
        // A1SC0S = A1S @ C0S
        TensorHao<complex<double>, 2> A1SC0S(N, truncatedD);
        BL_NAME(gmm)(A1S, C0S, A1SC0S);
        // A1SC0SB0 = A1SC0S @ B0
        TensorHao<complex<double>, 2> A1SC0SB0(N, L);
        BL_NAME(gmm)(A1SC0S, metropolisInfo.B0, A1SC0SB0);
        for(int i=1-1; i<=A1SC0SB0.rank(0)-1; i++){
            for(int j=1-1; j<=A1SC0SB0.rank(1)-1; j++){
                A1SC0SB0(i,j) = 0.5 * A1SC0SB0(i,j);
            }
        }
        // 
        // metropolisInfo.walkerLeftInBlock[metropolisInfo.BPMetroTimesliceBlockSize] = A0 + A1SB0 + 0.5 @ A1SC0SB0
        // 
        metropolisInfo.walkerLeftInBlock[metropolisInfo.BPMetroTimesliceBlockSize].wfRef() = conj(trans(metropolisInfo.A0 + A1SB0 + A1SC0SB0));
        metropolisInfo.walkerLeftInBlock[metropolisInfo.BPMetroTimesliceBlockSize].logwRef() = metropolisInfo.walkerLeftInBlock[0].getLogw() + conj(expMinusDtK_Jastrow_vec->at(0).logw) + conj(expMinusDtV_Jastrow_vec->at(0).getTwoBodySample_logw_FromAux(metropolisInfo.auxiliaryFields[0]));
        // 
        ///////////////////////////////////
        // 
        ////////////////////////////////////////////////
    }else if(method.BPMetroUpdateType == "local" && metropolisInfo.KVorder[0]=="VK" && metropolisInfo.JastrowExpM[0] == 2){ 
        cout<<"Error: method.BPMetroUpdateType == local is not supported yet!"<<endl;
    }
}

const Walker& Metropolis::getWalkerLeftMetro()
{
    updateWalkerLeftMetro();
    return metropolisInfo.walkerLeftInBlock[metropolisInfo.BPMetroTimesliceBlockSize];  
}

////////////////////////////////////////////////
///////////////////