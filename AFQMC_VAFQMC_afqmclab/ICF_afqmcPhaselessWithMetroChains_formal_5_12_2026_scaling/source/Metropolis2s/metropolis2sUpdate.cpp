//
// Created by boruoshihao on 1/17/17.
//
#include <cmath>
#include <chrono>
#include "../../include/Metropolis2s/metropolis2s.h"
#include "../../include/utils.h"

using namespace std;
using namespace tensor_hao;
                                               
void Metropolis2s::updateOneSweep(vector<int> flip_i_vec)
{
    updateToRightOneSweep(flip_i_vec, false);
    updateToLeftOneSweep(flip_i_vec, false);
}

void Metropolis2s::updateToRightOneSweep(vector<int> flip_i_vec, bool ifFrozen)
{
    Walker2s walkerLeft = metropolis2sInfo.walkerLeftInBlock[0]; complex<double>logWeight=0.0;
    Walker2s walkerRight = metropolis2sInfo.walkerRightInBlock[0];
    //
    for(metropolis2sInfo.inBlockIndex = (metropolis2sInfo.BPMetroTimesliceBlockSize-1); metropolis2sInfo.inBlockIndex > -1 ; --metropolis2sInfo.inBlockIndex)
    {
        updateToRightOneStep(walkerLeft, logWeight, flip_i_vec, ifFrozen);
        // 
        metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize-metropolis2sInfo.inBlockIndex]=walkerLeft;
        metropolis2sInfo.logWeightLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize-metropolis2sInfo.inBlockIndex]=logWeight;
    }
}

void Metropolis2s::updateToLeftOneSweep(vector<int> flip_i_vec, bool ifFrozen)
{
    Walker2s walkerLeft = metropolis2sInfo.walkerLeftInBlock[0];
    Walker2s walkerRight = metropolis2sInfo.walkerRightInBlock[0];complex<double>logWeight=0.0;
    for(metropolis2sInfo.inBlockIndex = 1-1; metropolis2sInfo.inBlockIndex <=metropolis2sInfo.BPMetroTimesliceBlockSize-1 ; metropolis2sInfo.inBlockIndex++)
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
        metropolis2sInfo.walkerRightInBlock[metropolis2sInfo.inBlockIndex+1]=walkerRight;
        metropolis2sInfo.logWeightRightInBlock[metropolis2sInfo.inBlockIndex+1]=logWeight;
    }
}

void Metropolis2s::updateToRightOneStep(Walker2s &walkerLeft, complex<double> &logWeight, vector<int> flip_i_vec, bool ifFrozen)
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
    int j_Jastrow=metropolis2sInfo.numOfJastrow-1;
    int tempConst=0;
    for(int i= metropolis2sInfo.numOfJastrow-1; i>= 0; i--){
        tempConst += metropolis2sInfo.JastrowSlice[i];
        if(metropolis2sInfo.inBlockIndex>=tempConst){
            j_Jastrow--;
        }
    }
    //
    TwoBodyAux_Jastrow2s auxNew;
    // TwoBodyAux_BP_Jastrow2s auxNew_BP;
    Walker2s walkerSaver=walkerLeft;
    Walker2s walkerLeftTemp;
    WalkerWalkerOperation_Jastrow2s walkerWalkerOperation;
    TwoBodyForce_Jastrow2s tempForce;
    // TwoBodyForce_BP_Jastrow2s tempForce_BP;

    complex<double> logProbNew=0.0;
    complex<double> currentLogProb=0.0;

    complex<double> logOverlapNew,currentLogOverlapSave;
    //
    TensorHao<complex<double>,2> Bup_proposed(walkerLeft.getL(),walkerLeft.getL()), Bdn_proposed(walkerLeft.getL(),walkerLeft.getL());
    TensorHao<complex<double>,2> overlapMatrixUp_proposed(walkerLeft.getNup(),walkerLeft.getNup()), overlapMatrixDn_proposed(walkerLeft.getNdn(), walkerLeft.getNdn());
    TensorHao<complex<double>,2> overlapMatrixUp_inv_proposed(walkerLeft.getNup(),walkerLeft.getNup()), overlapMatrixDn_inv_proposed(walkerLeft.getNdn(), walkerLeft.getNdn());
    // 
    complex<double> old_logSampleweight, new_logSampleweight;
    complex<double> overlapRatio_fastUpdate;
    double alpha;
    //
    metropolis2sInfo.currentLogOverlap += metropolis2sInfo.logWeightRightInBlock[metropolis2sInfo.inBlockIndex+1];
    currentLogOverlapSave=metropolis2sInfo.currentLogOverlap;
    if(metropolis2sInfo.inBlockIndex< method.blockNum || ifFrozen){
        alpha=0.0;
    }else{
        if(metropolis2sInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_HAFQMC_icf"){
            ////////////////////////////////////////////////////////////
            walkerLeftTemp = walkerLeft;
            if( method.BPMetroForceType == "constForce" )
            {
                if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
                    // auxNew = expMinusDtV_Jastrow_vec->at(j_Jastrow).sampleAuxFromForce(metropolis2sInfo.constForce_Jastrow[j_Jastrow]);
                    auxNew = expMinusDtV_Jastrow_vec->at(j_Jastrow).sampleAuxFromForce(metropolis2sInfo.constForce_Jastrow[j_Jastrow]);
                }else if(flip_i_vec.size() > 0 && method.BPMetroUpdateType == "local"){
                    auxNew = expMinusDtV_Jastrow_vec->at(j_Jastrow).sampleAuxFromForce_localUpdate(metropolis2sInfo.constForce_Jastrow[j_Jastrow], metropolis2sInfo.auxiliaryFields[metropolis2sInfo.inBlockIndex], flip_i_vec);
                }else{
                    cout<<"Error: UNKNOW BPMetroUpdateType: "<<method.BPMetroUpdateType<<endl;
                    exit(1);
                }
                // 
                if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
                    tempForce = metropolis2sInfo.constForce_Jastrow[j_Jastrow];
                    //
                    currentLogProb = expMinusDtV_Jastrow_vec->at(j_Jastrow).logProbOfAuxFromForce(metropolis2sInfo.auxiliaryFields[metropolis2sInfo.inBlockIndex], metropolis2sInfo.dynamicForceFields[metropolis2sInfo.inBlockIndex]); 
                    logProbNew = expMinusDtV_Jastrow_vec->at(j_Jastrow).logProbOfAuxFromForce(auxNew, tempForce); 
                    if(method.BPMetroUpdateType == "global_fast" && metropolis2sInfo.numOfJastrow == 1 && metropolis2sInfo.JastrowSlice[0] == 1){
                        // In global_fast, we only need to update aux related objects and we don't update walkerRight/walkerLeft in the middle of chain
                    }else{
                        TwoBodySample_Jastrow2s twoBodySample;
                        twoBodySample = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySampleFromAux2s(auxNew);
                        if(metropolis2sInfo.KVorder[j_Jastrow]=="VK"){
                            twoBodySampleWalkerLeftOperation.reset("fixedOrder", metropolis2sInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                            twoBodySampleWalkerLeftOperation.applyToLeft(twoBodySample, walkerLeft, walkerLeftTemp);
                            oneBodyWalkerLeftOperation.applyToLeft(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerLeftTemp, walkerLeft);
                        }else if(metropolis2sInfo.KVorder[j_Jastrow]=="K^daggerV"){
                            twoBodySampleWalkerLeftOperation.reset("fixedOrder", metropolis2sInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                            oneBodyWalkerLeftOperation.applyToLeft(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerLeft, walkerLeftTemp);
                            twoBodySampleWalkerLeftOperation.applyToLeft(twoBodySample, walkerLeftTemp, walkerLeft);
                        }else if(metropolis2sInfo.KVorder[j_Jastrow]=="KVK"){
                            twoBodySampleWalkerLeftOperation.reset("dynamicOrder");
                            oneBodyWalkerLeftOperation.applyToLeft(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerLeft, walkerLeftTemp);
                            twoBodySampleWalkerLeftOperation.applyToLeft(twoBodySample, walkerLeftTemp, walkerLeft);
                            oneBodyWalkerLeftOperation.applyToLeft(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerLeft, walkerLeftTemp);
                            walkerLeft = walkerLeftTemp;
                        }else{
                            cout<<"Error: UNKNOW metropolis2sInfo.KVorder[j_Jastrow]: "<<metropolis2sInfo.KVorder[j_Jastrow]<<endl;
                            exit(1);
                        }
                    }
                }else if(flip_i_vec.size() > 0 && method.BPMetroUpdateType == "local"){
                    tempForce = metropolis2sInfo.constForce_Jastrow[j_Jastrow];
                    currentLogProb = expMinusDtV_Jastrow_vec->at(j_Jastrow).logProbOfAuxFromForce(metropolis2sInfo.auxiliaryFields[metropolis2sInfo.inBlockIndex], metropolis2sInfo.dynamicForceFields[metropolis2sInfo.inBlockIndex]); 
                    logProbNew = expMinusDtV_Jastrow_vec->at(j_Jastrow).logProbOfAuxFromForce(auxNew, tempForce); 
                    // 
                    old_logSampleweight = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySample_logw_FromAux2s(metropolis2sInfo.auxiliaryFields[metropolis2sInfo.inBlockIndex]);
                    new_logSampleweight = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySample_logw_FromAux2s(auxNew);
                    overlapRatio_fastUpdate = performLocalUpdateWithFastOverlapRatio(j_Jastrow, metropolis2sInfo.inBlockIndex, flip_i_vec, auxNew, Bup_proposed, Bdn_proposed, overlapMatrixUp_inv_proposed, overlapMatrixDn_inv_proposed);
                }
                //
                ////////////////////////////////////////////////////////////
            }else{
                    cout<<"Error: UNKNOW BPMetroForceType: "<<method.BPMetroForceType<<endl;
                    exit(1);
            }
        }else{
            cout<<"Error: UNKNOW metropolis2sInfo.variableName_vec[j_Jastrow] in update: "<<metropolis2sInfo.variableName_vec[j_Jastrow]<<endl;
            exit(1);
        }
        //
        // walkerLeft.addLogw(method.Metro_dtET );
        //
        if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
            ///////////////////////////////////////////////
            if(method.BPMetroUpdateType == "global_fast" && metropolis2sInfo.numOfJastrow == 1 && metropolis2sInfo.JastrowSlice[0] == 1){
                //////////////////////////////////////////////////////
                // globalFast_begin = std::chrono::high_resolution_clock::now();
                //////////////////////////////////////////////////////
                logOverlapNew = getLogOverlapFromAux(auxNew, 2, overlapMatrixUp_proposed, overlapMatrixDn_proposed);            
                //////////////////////////////////////////////////////
                // globalFast_end = std::chrono::high_resolution_clock::now();
                //////////////////////////////////////////////////////
            }else{
                walkerWalkerOperation.set( walkerLeft, metropolis2sInfo.walkerRightInBlock[metropolis2sInfo.inBlockIndex] );
                logOverlapNew = walkerWalkerOperation.returnLogOverlap();
            }
            ///////////////////////////////////////////////
            // 
            alpha = abs( exp(logOverlapNew-logProbNew-metropolis2sInfo.currentLogOverlap+currentLogProb) );
        }else if(flip_i_vec.size() > 0 && method.BPMetroUpdateType == "local"){
            // fast evaluation of alpha through Matrix Determinant Lemma
            alpha = abs(exp(log(overlapRatio_fastUpdate)+new_logSampleweight-old_logSampleweight-logProbNew+currentLogProb)  );
            logOverlapNew = log(overlapRatio_fastUpdate)+ new_logSampleweight - old_logSampleweight + metropolis2sInfo.currentLogOverlap;
        }

        if(metropolis2sInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_K_HAFQMC_icf" && abs(alpha-1.0) >= 10e-8 ){
            cout<<"Error: abs(alpha-1.0) >= 10e-8 in generalHamiltonian_K_HAFQMC_icf to left: "<<alpha<<endl;
            exit(1);
        }
    }

    if( uniformHao() < alpha )
    {
        if(metropolis2sInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_HAFQMC_icf"){
            metropolis2sInfo.auxiliaryFields[metropolis2sInfo.inBlockIndex] = move(auxNew);
            metropolis2sInfo.dynamicForceFields[metropolis2sInfo.inBlockIndex]=tempForce;
            // 
            if(flip_i_vec.size() > 0 && method.BPMetroUpdateType == "local"){
                metropolis2sInfo.Bup = Bup_proposed;
                metropolis2sInfo.Bdn = Bdn_proposed;
                metropolis2sInfo.overlapMatrixUp_inv = overlapMatrixUp_inv_proposed;
                metropolis2sInfo.overlapMatrixDn_inv = overlapMatrixDn_inv_proposed;
            }else if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
                metropolis2sInfo.overlapMatrixUp = overlapMatrixUp_proposed;
                metropolis2sInfo.overlapMatrixDn = overlapMatrixDn_proposed;
            }
        }else{
            cout<<"Error: UNKNOW metropolis2sInfo.variableName_vec[j_Jastrow]: "<<metropolis2sInfo.variableName_vec[j_Jastrow]<<endl;
            exit(1);
        }
        //
        metropolis2sInfo.currentLogOverlap=logOverlapNew;
        /////////////////////////////
    }
    else
    {
        if(metropolis2sInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_HAFQMC_icf"){
            if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
                if(method.BPMetroUpdateType == "global_fast" && metropolis2sInfo.numOfJastrow == 1 && metropolis2sInfo.JastrowSlice[0] == 1){
                }else{
                    TwoBodySample_Jastrow2s twoBodySample;
                    if( method.BPMetroForceType == "constForce"){
                        twoBodySample = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySampleFromAux2s(metropolis2sInfo.auxiliaryFields[metropolis2sInfo.inBlockIndex]);
                        // 
                        // if(flip_i_vec.size() > 0 && method.BPMetroUpdateType == "local"){
                        //     // update B
                        //     metropolis2sInfo.Bup = twoBodySample.matrixUp;
                        //     metropolis2sInfo.Bdn = twoBodySample.matrixDn;
                        // }
                    }
                    else{
                        cout<<"Error: UNKNOW BPMetroForceType: "<<method.BPMetroForceType<<endl;
                        exit(1);
                    }
                    if(metropolis2sInfo.KVorder[j_Jastrow]=="VK"){
                        twoBodySampleWalkerLeftOperation.reset("fixedOrder", metropolis2sInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                        twoBodySampleWalkerLeftOperation.applyToLeft(twoBodySample, walkerSaver, walkerLeftTemp);
                        oneBodyWalkerLeftOperation.applyToLeft(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerLeftTemp, walkerLeft);
                    }else if(metropolis2sInfo.KVorder[j_Jastrow]=="K^daggerV"){
                        twoBodySampleWalkerLeftOperation.reset("fixedOrder", metropolis2sInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                        oneBodyWalkerLeftOperation.applyToLeft(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerSaver, walkerLeftTemp);
                        twoBodySampleWalkerLeftOperation.applyToLeft(twoBodySample, walkerLeftTemp, walkerLeft);
                    }else if(metropolis2sInfo.KVorder[j_Jastrow]=="KVK"){
                        twoBodySampleWalkerLeftOperation.reset("dynamicOrder");
                        oneBodyWalkerLeftOperation.applyToLeft(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerSaver, walkerLeftTemp);
                        twoBodySampleWalkerLeftOperation.applyToLeft(twoBodySample, walkerLeftTemp, walkerLeft);
                        oneBodyWalkerLeftOperation.applyToLeft(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerLeft, walkerLeftTemp);
                        walkerLeft = walkerLeftTemp;
                    }else{
                        cout<<"Error: UNKNOW metropolis2sInfo.KVorder[j_Jastrow]: "<<metropolis2sInfo.KVorder[j_Jastrow]<<endl;
                        exit(1);
                    }
                }
            }
            //
        }
        else{
            cout<<"Error: UNKNOW metropolis2sInfo.variableName_vec[j_Jastrow] in update save: "<<metropolis2sInfo.variableName_vec[j_Jastrow]<<endl;
            exit(1);
        }
        // walkerLeft.addLogw(method.Metro_dtET );

        metropolis2sInfo.currentLogOverlap=currentLogOverlapSave;
    }
    //
    logWeight = 0.0;
    // if( (method.BPMetroStabilizeStep - metropolis2sInfo.inBlockIndex) % method.BPMetroStabilizeStep == 0 )
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
    // metropolis2sInfo.currentLogOverlap -= conj(logWeight);
    ////////////////////////////////////////
    // logWeight = expMinusDtK_Jastrow_vec->at(j_Jastrow).logw.real() + expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySample_logw_FromAux2s(metropolis2sInfo.auxiliaryFields[metropolis2sInfo.inBlockIndex]).real();
    // metropolis2sInfo.currentLogOverlap -= conj(logWeight);
    ////////////////////////////////////////
    // end = std::chrono::high_resolution_clock::now();
    // if(method.BPMetroUpdateType == "global_fast" && metropolis2sInfo.numOfJastrow == 1 && metropolis2sInfo.JastrowSlice[0] == 1){
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

void Metropolis2s::updateToLeftOneStep(Walker2s &walkerRight, complex<double> &logWeight, vector<int> flip_i_vec, bool ifFrozen)
{
    //
    int j_Jastrow=metropolis2sInfo.numOfJastrow-1;
    int tempConst=0;
    for(int i= metropolis2sInfo.numOfJastrow-1; i>= 0; i--){
        tempConst += metropolis2sInfo.JastrowSlice[i];
        if(metropolis2sInfo.inBlockIndex>=tempConst){
            j_Jastrow--;
        }
    }
    //
    TwoBodyAux_Jastrow2s auxNew;
    // TwoBodyAux_BP_Jastrow2s auxNew_BP;
    Walker2s walkerSaver=walkerRight;
    Walker2s walkerRightTemp;
    WalkerWalkerOperation_Jastrow2s walkerWalkerOperation;
    TwoBodyForce_Jastrow2s tempForce;
    // TwoBodyForce_BP_Jastrow2s tempForce_BP;
    
    complex<double> logProbNew=0.0;
    complex<double> currentLogProb=0.0;

    complex<double> logOverlapNew,currentLogOverlapSave;
    //
    TensorHao<complex<double>,2> Bup_proposed(walkerRight.getL(),walkerRight.getL()), Bdn_proposed(walkerRight.getL(),walkerRight.getL());
    TensorHao<complex<double>,2> overlapMatrixUp_proposed(walkerRight.getNup(),walkerRight.getNup()), overlapMatrixDn_proposed(walkerRight.getNdn(), walkerRight.getNdn());
    TensorHao<complex<double>,2> overlapMatrixUp_inv_proposed(walkerRight.getNup(),walkerRight.getNup()), overlapMatrixDn_inv_proposed(walkerRight.getNdn(), walkerRight.getNdn());
    // 
    complex<double> old_logSampleweight, new_logSampleweight;
    complex<double> overlapRatio_fastUpdate;
    double alpha;

    metropolis2sInfo.currentLogOverlap += conj(metropolis2sInfo.logWeightLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize-metropolis2sInfo.inBlockIndex]);
    currentLogOverlapSave=metropolis2sInfo.currentLogOverlap;
    // 
    if(metropolis2sInfo.inBlockIndex< method.blockNum || ifFrozen){
        alpha=0.0;
    }else{
        if(metropolis2sInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_HAFQMC_icf"){
            ////////////////////////////////////////////////////////////
            walkerRightTemp=walkerRight;
            if( method.BPMetroForceType == "constForce" )
            {
                // auto begin = std::chrono::high_resolution_clock::now();
                if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
                    auxNew = expMinusDtV_Jastrow_vec->at(j_Jastrow).sampleAuxFromForce(metropolis2sInfo.constForce_Jastrow[j_Jastrow]);
                }else if(flip_i_vec.size() > 0 && method.BPMetroUpdateType == "local"){
                    auxNew = expMinusDtV_Jastrow_vec->at(j_Jastrow).sampleAuxFromForce_localUpdate(metropolis2sInfo.constForce_Jastrow[j_Jastrow], metropolis2sInfo.auxiliaryFields[metropolis2sInfo.inBlockIndex], flip_i_vec);
                }else{
                    cout<<"Error: UNKNOW BPMetroUpdateType: "<<method.BPMetroUpdateType<<endl;
                    exit(1);
                }

                if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
                    tempForce = metropolis2sInfo.constForce_Jastrow[j_Jastrow];
                    // 
                    currentLogProb = expMinusDtV_Jastrow_vec->at(j_Jastrow).logProbOfAuxFromForce(metropolis2sInfo.auxiliaryFields[metropolis2sInfo.inBlockIndex], metropolis2sInfo.dynamicForceFields[metropolis2sInfo.inBlockIndex]); 
                    logProbNew = expMinusDtV_Jastrow_vec->at(j_Jastrow).logProbOfAuxFromForce(auxNew, tempForce); 
                    // 
                    if(method.BPMetroUpdateType == "global_fast" && metropolis2sInfo.numOfJastrow == 1 && metropolis2sInfo.JastrowSlice[0] == 1){
                        // In global_fast, we only need to update aux related objects and we don't update walkerRight/walkerLeft in the middle of chain
                    }else{
                        TwoBodySample_Jastrow2s twoBodySample;
                        twoBodySample = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySampleFromAux2s(auxNew);
                        if(metropolis2sInfo.KVorder[j_Jastrow]=="VK"){
                            twoBodySampleWalkerRightOperation.reset("fixedOrder", metropolis2sInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                            oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRight, walkerRightTemp);
                            twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp, walkerRight);
                        }else if(metropolis2sInfo.KVorder[j_Jastrow]=="K^daggerV"){
                            twoBodySampleWalkerRightOperation.reset("fixedOrder", metropolis2sInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                            twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRight, walkerRightTemp);
                            oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRight);
                        }else if(metropolis2sInfo.KVorder[j_Jastrow]=="KVK"){
                            twoBodySampleWalkerRightOperation.reset("dynamicOrder");
                            oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRight, walkerRightTemp);
                            twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp, walkerRight);
                            oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRight, walkerRightTemp);
                            walkerRight = walkerRightTemp;
                        }else{
                            cout<<"Error: UNKNOW metropolis2sInfo.KVorder[j_Jastrow]: "<<metropolis2sInfo.KVorder[j_Jastrow]<<endl;
                            exit(1);
                        }
                    }
                }else if(flip_i_vec.size() > 0 && method.BPMetroUpdateType == "local"){
                    tempForce = metropolis2sInfo.constForce_Jastrow[j_Jastrow];
                    currentLogProb = expMinusDtV_Jastrow_vec->at(j_Jastrow).logProbOfAuxFromForce(metropolis2sInfo.auxiliaryFields[metropolis2sInfo.inBlockIndex], metropolis2sInfo.dynamicForceFields[metropolis2sInfo.inBlockIndex]); 
                    logProbNew = expMinusDtV_Jastrow_vec->at(j_Jastrow).logProbOfAuxFromForce(auxNew, tempForce); 
                    // 
                    old_logSampleweight = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySample_logw_FromAux2s(metropolis2sInfo.auxiliaryFields[metropolis2sInfo.inBlockIndex]);
                    new_logSampleweight = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySample_logw_FromAux2s(auxNew);
                    // 
                    overlapRatio_fastUpdate = performLocalUpdateWithFastOverlapRatio(j_Jastrow, metropolis2sInfo.inBlockIndex, flip_i_vec, auxNew, Bup_proposed, Bdn_proposed, overlapMatrixUp_inv_proposed, overlapMatrixDn_inv_proposed);
                }
                ////////////////////////////////////////////////////////////
            }else{
                cout<<"Error: UNKNOW BPMetroForceType: "<<method.BPMetroForceType<<endl;
                exit(1);
            }
        }
        else{
            cout<<"Error: UNKNOW metropolis2sInfo.variableName_vec[j_Jastrow] in update: "<<metropolis2sInfo.variableName_vec[j_Jastrow]<<endl;
            exit(1);
        }
        //
        // walkerRight.addLogw(method.Metro_dtET );
        //
        if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
            ///////////////////////////////////////////////
            if(method.BPMetroUpdateType == "global_fast" && metropolis2sInfo.numOfJastrow == 1 && metropolis2sInfo.JastrowSlice[0] == 1){
                logOverlapNew = getLogOverlapFromAux(auxNew, 2, overlapMatrixUp_proposed, overlapMatrixDn_proposed);
            }else{
                walkerWalkerOperation.set( metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize-metropolis2sInfo.inBlockIndex-1], walkerRight );
                logOverlapNew = walkerWalkerOperation.returnLogOverlap();
            }
            ///////////////////////////////////////////////
            //
            alpha = abs( exp(logOverlapNew-logProbNew-metropolis2sInfo.currentLogOverlap+currentLogProb) );
        }else if(flip_i_vec.size() > 0 && method.BPMetroUpdateType == "local"){
            // fast evaluation of alpha through Matrix Determinant Lemma
            logOverlapNew = log(overlapRatio_fastUpdate)+ new_logSampleweight - old_logSampleweight + metropolis2sInfo.currentLogOverlap;
            alpha = abs(exp(log(overlapRatio_fastUpdate)+new_logSampleweight-old_logSampleweight-logProbNew+currentLogProb)  );
        }

        if(metropolis2sInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_K_HAFQMC_icf" && abs(alpha-1.0) >= 10e-8 ){
            cout<<"Error: abs(alpha-1.0) >= 10e-8 in generalHamiltonian_K_HAFQMC_icf to right: "<<alpha<<endl;
            exit(1);
        }
    }

    if( uniformHao() < alpha )
    {
        if(metropolis2sInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_HAFQMC_icf"){
            metropolis2sInfo.auxiliaryFields[metropolis2sInfo.inBlockIndex] = move(auxNew);
            metropolis2sInfo.dynamicForceFields[metropolis2sInfo.inBlockIndex]=tempForce;
            // 
            if(flip_i_vec.size() > 0 && method.BPMetroUpdateType == "local"){
                metropolis2sInfo.Bup = Bup_proposed;
                metropolis2sInfo.Bdn = Bdn_proposed;
                metropolis2sInfo.overlapMatrixUp_inv = overlapMatrixUp_inv_proposed;
                metropolis2sInfo.overlapMatrixDn_inv = overlapMatrixDn_inv_proposed;
            }else if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
                metropolis2sInfo.overlapMatrixUp = overlapMatrixUp_proposed;
                metropolis2sInfo.overlapMatrixDn = overlapMatrixDn_proposed;
            }
        }else{
            cout<<"Error: UNKNOW metropolis2sInfo.variableName_vec[j_Jastrow]: "<<metropolis2sInfo.variableName_vec[j_Jastrow]<<endl;
            exit(1);
        }
        //
        metropolis2sInfo.currentLogOverlap=logOverlapNew;
    }
    else
    {
        if(metropolis2sInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_HAFQMC_icf"){
            if(flip_i_vec.size() <= 0 || method.BPMetroUpdateType == "global" || method.BPMetroUpdateType == "global_fast"){
                if(method.BPMetroUpdateType == "global_fast" && metropolis2sInfo.numOfJastrow == 1 && metropolis2sInfo.JastrowSlice[0] == 1){
                }else{
                    TwoBodySample_Jastrow2s twoBodySample;
                    if( method.BPMetroForceType == "constForce" ){
                        twoBodySample = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySampleFromAux2s(metropolis2sInfo.auxiliaryFields[metropolis2sInfo.inBlockIndex]);    
                        // 
                        // if(flip_i_vec.size() > 0 && method.BPMetroUpdateType == "local"){
                        //     // update B
                        //     metropolis2sInfo.Bup = twoBodySample.matrixUp;
                        //     metropolis2sInfo.Bdn = twoBodySample.matrixDn;
                        // }
                    }
                    else{
                        cout<<"Error: UNKNOW BPMetroForceType: "<<method.BPMetroForceType<<endl;
                        exit(1);
                    }
                    if(metropolis2sInfo.KVorder[j_Jastrow]=="VK"){
                        twoBodySampleWalkerRightOperation.reset("fixedOrder", metropolis2sInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                        oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerSaver, walkerRightTemp);
                        twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp, walkerRight);
                    }else if(metropolis2sInfo.KVorder[j_Jastrow]=="K^daggerV"){
                        twoBodySampleWalkerRightOperation.reset("fixedOrder", metropolis2sInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                        twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerSaver, walkerRightTemp);
                        oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRight);
                    }else if(metropolis2sInfo.KVorder[j_Jastrow]=="KVK"){
                        twoBodySampleWalkerRightOperation.reset("dynamicOrder");
                        oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerSaver, walkerRightTemp);
                        twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp, walkerRight);
                        oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRight, walkerRightTemp);
                        walkerRight = walkerRightTemp;
                    }else{
                        cout<<"Error: UNKNOW metropolis2sInfo.KVorder[j_Jastrow]: "<<metropolis2sInfo.KVorder[j_Jastrow]<<endl;
                        exit(1);
                    }
                }
            }
            //
        }
        else{
            cout<<"Error: UNKNOW metropolis2sInfo.variableName_vec[j_Jastrow] in update save: "<<metropolis2sInfo.variableName_vec[j_Jastrow]<<endl;
            exit(1);
        }
        /////////////////////////////////
        // walkerRight.addLogw(method.Metro_dtET );
        /////////////////////////////////
        
        metropolis2sInfo.currentLogOverlap=currentLogOverlapSave;
    }

    logWeight = 0.0;
    // if( (method.BPMetroStabilizeStep - metropolis2sInfo.inBlockIndex) % method.BPMetroStabilizeStep == 0 )
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
    // metropolis2sInfo.currentLogOverlap -= logWeight;
    ////////////////////////////////////////
    // logWeight = expMinusDtK_Jastrow_vec->at(j_Jastrow).logw.real() + expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySample_logw_FromAux2s(metropolis2sInfo.auxiliaryFields[metropolis2sInfo.inBlockIndex]).real();
    // metropolis2sInfo.currentLogOverlap -= logWeight;
}

////////////////////////////////////////////////////////////////////
void Metropolis2s::initial_globalFast()
{
    // 
    Walker2s walkerRightTemp;
    if(metropolis2sInfo.KVorder[0]=="VK"){
        // oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(0), metropolis2sInfo.walkerRightInBlock[0], walkerRightTemp);
        // 
        // A0 = walkerLeft0Wfdagger @ K
        metropolis2sInfo.A0up.resize(metropolis2sInfo.walkerLeftInBlock[0].getWfUp().rank(1), expMinusDtK_Jastrow_vec->at(0).matrixUp.rank(1));
        BL_NAME(gmm)(trans(conj(metropolis2sInfo.walkerLeftInBlock[0].getWfUp())), expMinusDtK_Jastrow_vec->at(0).matrixUp, metropolis2sInfo.A0up);
        metropolis2sInfo.A0dn.resize(metropolis2sInfo.walkerLeftInBlock[0].getWfDn().rank(1), expMinusDtK_Jastrow_vec->at(0).matrixDn.rank(1));
        BL_NAME(gmm)(trans(conj(metropolis2sInfo.walkerLeftInBlock[0].getWfDn())), expMinusDtK_Jastrow_vec->at(0).matrixDn, metropolis2sInfo.A0dn);
        // A0WR = A0 @ walkerRight0Wf
        metropolis2sInfo.A0WRup.resize(metropolis2sInfo.A0up.rank(0), metropolis2sInfo.walkerRightInBlock[0].getWfUp().rank(1));
        BL_NAME(gmm)(metropolis2sInfo.A0up, metropolis2sInfo.walkerRightInBlock[0].getWfUp(), metropolis2sInfo.A0WRup);
        metropolis2sInfo.A0WRdn.resize(metropolis2sInfo.A0dn.rank(0), metropolis2sInfo.walkerRightInBlock[0].getWfDn().rank(1));
        BL_NAME(gmm)(metropolis2sInfo.A0dn, metropolis2sInfo.walkerRightInBlock[0].getWfDn(), metropolis2sInfo.A0WRdn);
        // A1 = walkerLeft0Wfdagger @ U0
        metropolis2sInfo.A1up.resize(metropolis2sInfo.walkerLeftInBlock[0].getWfUp().rank(1), expMinusDtV_Jastrow_vec->at(0).getSVD_DupRank());
        BL_NAME(gmm)(trans(conj(metropolis2sInfo.walkerLeftInBlock[0].getWfUp())), expMinusDtV_Jastrow_vec->at(0).getSqrtMinusDtSVDVecsMatrix_SVD(0, "U0up"), metropolis2sInfo.A1up);
        metropolis2sInfo.A1dn.resize(metropolis2sInfo.walkerLeftInBlock[0].getWfDn().rank(1), expMinusDtV_Jastrow_vec->at(0).getSVD_DdnRank());
        BL_NAME(gmm)(trans(conj(metropolis2sInfo.walkerLeftInBlock[0].getWfDn())), expMinusDtV_Jastrow_vec->at(0).getSqrtMinusDtSVDVecsMatrix_SVD(0, "U0dn"), metropolis2sInfo.A1dn);
        // B0 = Vdagger0 @ K
        metropolis2sInfo.B0up.resize(expMinusDtV_Jastrow_vec->at(0).getSVD_DupRank(), expMinusDtK_Jastrow_vec->at(0).matrixUp.rank(1));
        BL_NAME(gmm)(expMinusDtV_Jastrow_vec->at(0).getSqrtMinusDtSVDVecsMatrix_SVD(0, "Vdagger0up"), expMinusDtK_Jastrow_vec->at(0).matrixUp, metropolis2sInfo.B0up);
        // metropolis2sInfo.B0dn.resize(expMinusDtV_Jastrow_vec->at(0).getSVD_DdnRank(), expMinusDtK_Jastrow_vec->at(0).matrixDn.rank(1));
        // BL_NAME(gmm)(expMinusDtV_Jastrow_vec->at(0).getSqrtMinusDtSVDVecsMatrix_SVD(0, "Vdagger0dn"), expMinusDtK_Jastrow_vec->at(0).matrixDn, metropolis2sInfo.B0dn);
        // B0WR = B0 @ walkerRight0Wf
        metropolis2sInfo.B0WRup.resize(metropolis2sInfo.B0up.rank(0), metropolis2sInfo.walkerRightInBlock[0].getWfUp().rank(1));
        BL_NAME(gmm)(metropolis2sInfo.B0up, metropolis2sInfo.walkerRightInBlock[0].getWfUp(), metropolis2sInfo.B0WRup);
        metropolis2sInfo.B0WRdn.resize(metropolis2sInfo.B0dn.rank(0), metropolis2sInfo.walkerRightInBlock[0].getWfDn().rank(1));
        BL_NAME(gmm)(metropolis2sInfo.B0dn, metropolis2sInfo.walkerRightInBlock[0].getWfDn(), metropolis2sInfo.B0WRdn);
        // C0 = Vdagger0 @ U0
        metropolis2sInfo.C0up.resize(expMinusDtV_Jastrow_vec->at(0).getSVD_DupRank(), expMinusDtV_Jastrow_vec->at(0).getSVD_DupRank());
        BL_NAME(gmm)(expMinusDtV_Jastrow_vec->at(0).getSqrtMinusDtSVDVecsMatrix_SVD(0, "Vdagger0up"), expMinusDtV_Jastrow_vec->at(0).getSqrtMinusDtSVDVecsMatrix_SVD(0, "U0up"), metropolis2sInfo.C0up);
        // metropolis2sInfo.C0dn.resize(expMinusDtV_Jastrow_vec->at(0).getSVD_DdnRank(), expMinusDtV_Jastrow_vec->at(0).getSVD_DdnRank());
        // BL_NAME(gmm)(expMinusDtV_Jastrow_vec->at(0).getSqrtMinusDtSVDVecsMatrix_SVD(0, "Vdagger0dn"), expMinusDtV_Jastrow_vec->at(0).getSqrtMinusDtSVDVecsMatrix_SVD(0, "U0dn"), metropolis2sInfo.C0dn);
        // 
        metropolis2sInfo.Dup.resize(expMinusDtV_Jastrow_vec->at(0).getSVD_DupRank(),expMinusDtV_Jastrow_vec->at(0).getSVD_DupRank());
        // metropolis2sInfo.Ddn.resize(expMinusDtV_Jastrow_vec->at(0).getSVD_DdnRank(),expMinusDtV_Jastrow_vec->at(0).getSVD_DdnRank());
    }else{
        cout<<"Error: UNKNOW metropolis2sInfo.KVorder[0]: "<<metropolis2sInfo.KVorder[0]<<endl;
        exit(1);
    }

    metropolis2sInfo.globalFastInitialized = true;
    metropolis2sInfo.globalFastUpdated = true;

}

void Metropolis2s::update_globalFast()
{
    size_t Nup = metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize].getNup(); size_t Ndn = metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize].getNdn();
    TensorHao<complex<double>,2> overlapMatrixUp(Nup, Nup), overlapMatrixDn(Ndn, Ndn);
    BL_NAME(gmm)( metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize].getWfUp(), metropolis2sInfo.walkerRightInBlock[0].getWfUp(), overlapMatrixUp, 'C' );
    BL_NAME(gmm)( metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize].getWfDn(), metropolis2sInfo.walkerRightInBlock[0].getWfDn(), overlapMatrixDn, 'C' );
    // 
    metropolis2sInfo.overlapMatrixUp = overlapMatrixUp;
    metropolis2sInfo.overlapMatrixDn = overlapMatrixDn;
    // 
    if(metropolis2sInfo.KVorder[0]=="VK"){
        Walker2s walkerRightTemp;
        oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(0), metropolis2sInfo.walkerRightInBlock[0], walkerRightTemp);
        // 
        // A0WR = A0 @ walkerRight0Wf
        metropolis2sInfo.A0WRup.resize(metropolis2sInfo.A0up.rank(0), metropolis2sInfo.walkerRightInBlock[0].getWfUp().rank(1));
        BL_NAME(gmm)(metropolis2sInfo.A0up, metropolis2sInfo.walkerRightInBlock[0].getWfUp(), metropolis2sInfo.A0WRup);
        metropolis2sInfo.A0WRdn.resize(metropolis2sInfo.A0dn.rank(0), metropolis2sInfo.walkerRightInBlock[0].getWfDn().rank(1));
        BL_NAME(gmm)(metropolis2sInfo.A0dn, metropolis2sInfo.walkerRightInBlock[0].getWfDn(), metropolis2sInfo.A0WRdn);
        // B0WR = B0 @ walkerRight0Wf
        metropolis2sInfo.B0WRup.resize(metropolis2sInfo.B0up.rank(0), metropolis2sInfo.walkerRightInBlock[0].getWfUp().rank(1));
        BL_NAME(gmm)(metropolis2sInfo.B0up, metropolis2sInfo.walkerRightInBlock[0].getWfUp(), metropolis2sInfo.B0WRup);
        metropolis2sInfo.B0WRdn.resize(metropolis2sInfo.B0dn.rank(0), metropolis2sInfo.walkerRightInBlock[0].getWfDn().rank(1));
        BL_NAME(gmm)(metropolis2sInfo.B0dn, metropolis2sInfo.walkerRightInBlock[0].getWfDn(), metropolis2sInfo.B0WRdn);
    }else{
        cout<<"Error: UNKNOW metropolis2sInfo.KVorder[0]: "<<metropolis2sInfo.KVorder[0]<<endl;
        exit(1);
    }
    // 
    metropolis2sInfo.globalFastUpdated = true;
}

void Metropolis2s::getAuxMatrix_Dup(TwoBodyAux_Jastrow2s auxNew, size_t j_Jastrow)
{
    //  directly calculate the overlap matrix from aux and predefined matrix
    int truncatedDup = expMinusDtV_Jastrow_vec->at(j_Jastrow).getSVD_DupRank();
    metropolis2sInfo.Dup = 0.0;
    // 
    //Calculate aux * sqrtMinusDt * svdVecs
    TensorHaoRef<complex<double>, 1> vecsAuxUp(truncatedDup*truncatedDup);
    TensorHaoRef<complex<double>, 2> vecsUp(truncatedDup*truncatedDup, expMinusDtV_Jastrow_vec->at(j_Jastrow).getSVDNumber());
    vecsAuxUp.point( metropolis2sInfo.Dup.data() );
    vecsUp.point( const_cast<complex<double>*> ( expMinusDtV_Jastrow_vec->at(j_Jastrow).getSqrtMinusDtSVDVecs_Dup().data() ) );
    BL_NAME(gemv)(vecsUp, auxNew, vecsAuxUp);
}

void Metropolis2s::getAuxMatrix_Ddn(TwoBodyAux_Jastrow2s auxNew, size_t j_Jastrow)
{
    //  directly calculate the overlap matrix from aux and predefined matrix
    // int truncatedDdn = expMinusDtV_Jastrow_vec->at(j_Jastrow).getSVD_DdnRank();
    // metropolis2sInfo.Ddn = 0.0;
    // //Calculate aux * sqrtMinusDt * svdVecs
    // TensorHaoRef<complex<double>, 1> vecsAuxDn(truncatedDdn*truncatedDdn);
    // TensorHaoRef<complex<double>, 2> vecsDn(truncatedDdn*truncatedDdn, expMinusDtV_Jastrow_vec->at(j_Jastrow).getSVDNumber());
    // vecsAuxDn.point( metropolis2sInfo.Ddn.data() );
    // vecsDn.point( const_cast<complex<double>*> ( expMinusDtV_Jastrow_vec->at(j_Jastrow).getSqrtMinusDtSVDVecs_Ddn().data() ) );
    // BL_NAME(gemv)(vecsDn, auxNew, vecsAuxDn);
}

TensorHao<complex<double>, 2> Metropolis2s::getAuxMatrix_Dup_exp_x(TwoBodyAux_Jastrow2s auxNew, size_t j_Jastrow)
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
    getAuxMatrix_Dup(auxNew, 0);
    /////////////////////////////////////////////////
    // getAuxMatrix_Dup_end = std::chrono::high_resolution_clock::now();
    /////////////////////////////////////////////////
    // 
    // Dup_exp_x = Sup + 0.5 * Sup @ C0 @ Sup
    TensorHao<complex<double>, 2> SupC0up(metropolis2sInfo.Dup.rank(0), metropolis2sInfo.C0up.rank(1));
    BL_NAME(gmm)(metropolis2sInfo.Dup, metropolis2sInfo.C0up, SupC0up);
    TensorHao<complex<double>, 2> Dup_exp_x(metropolis2sInfo.Dup.rank(0), metropolis2sInfo.Dup.rank(1));
    BL_NAME(gmm)(SupC0up, metropolis2sInfo.Dup, Dup_exp_x);
    for(int j=1-1; j<=Dup_exp_x.rank(1)-1; j++){
        for(int i=1-1; i<=Dup_exp_x.rank(0)-1; i++){
            Dup_exp_x(i,j) = metropolis2sInfo.Dup(i,j) + 0.5 * Dup_exp_x(i,j);
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
    return Dup_exp_x;
}
TensorHao<complex<double>, 2> Metropolis2s::getAuxMatrix_Ddn_exp_x(TwoBodyAux_Jastrow2s auxNew, size_t j_Jastrow)
{
    // getAuxMatrix_Ddn(auxNew, 0);
    // // 
    // // Ddn_exp_x = Sdn + 0.5 * Sdn @ C0 @ Sdn
    // TensorHao<complex<double>, 2> SdnC0dn(metropolis2sInfo.Ddn.rank(0), metropolis2sInfo.C0dn.rank(1));
    // BL_NAME(gmm)(metropolis2sInfo.Ddn, metropolis2sInfo.C0dn, SdnC0dn);
    // TensorHao<complex<double>, 2> Ddn_exp_x(metropolis2sInfo.Ddn.rank(0), metropolis2sInfo.Ddn.rank(1));
    // BL_NAME(gmm)(SdnC0dn, metropolis2sInfo.Ddn, Ddn_exp_x);
    // for(int i=1-1; i<=Ddn_exp_x.rank(0)-1; i++){
    //     for(int j=1-1; j<=Ddn_exp_x.rank(1)-1; j++){
    //         Ddn_exp_x(i,j) = metropolis2sInfo.Ddn(i,j) + 0.5 * Ddn_exp_x(i,j);
    //     }
    // }
    // // 
    // return Ddn_exp_x;
}

complex<double> Metropolis2s::getLogOverlapFromAux(TwoBodyAux_Jastrow2s auxNew, size_t expM, TensorHao<std::complex<double>, 2> & overlapMatrixup, TensorHao<std::complex<double>, 2> & overlapMatrixdn)
{
    ///////////////
    // Timer
    ///////////////
    // auto begin = std::chrono::high_resolution_clock::now();
    // auto end = std::chrono::high_resolution_clock::now();
    ///////////////
    if(!metropolis2sInfo.globalFastInitialized){
        initial_globalFast();
    }
    if(!metropolis2sInfo.globalFastUpdated){
        update_globalFast();
    }
    //  directly calculate the overlap matrix from aux and predefined matrix
    if(expM == 2 && metropolis2sInfo.globalFastInitialized && metropolis2sInfo.globalFastUpdated){
        // 
        if(metropolis2sInfo.B0WRup.rank(0) != metropolis2sInfo.B0WRdn.rank(0)){
            cout<<"Error: metropolis2sInfo.B0WRup.rank(0) != metropolis2sInfo.B0WRdn.rank(0) in getLogOverlapFromAux: "<<metropolis2sInfo.B0WRup.rank(0)<<" "<<metropolis2sInfo.B0WRdn.rank(0)<<endl;
            exit(1);
        }
        int Nup = metropolis2sInfo.B0WRup.rank(1);
        int Ndn = metropolis2sInfo.B0WRdn.rank(1);
        int truncatedDup = metropolis2sInfo.truncatedDup;
        int truncatedDdn = metropolis2sInfo.truncatedDdn;
        // 
        if(metropolis2sInfo.B0WRup.rank(0) + metropolis2sInfo.B0WRdn.rank(0) > metropolis2sInfo.B0WRup.rank(1) + metropolis2sInfo.B0WRdn.rank(1)){
            // cost: O(N_e M^2)
            getAuxMatrix_Dup(auxNew, 0);
            // getAuxMatrix_Ddn(auxNew, 0);
            // 
            // SB0WR = S @ B0WR
            TensorHao<complex<double>, 2> SupB0WRup(truncatedDup, Nup);
            BL_NAME(gmm)(metropolis2sInfo.Dup, metropolis2sInfo.B0WRup, SupB0WRup);
            TensorHao<complex<double>, 2> SdnB0WRdn(truncatedDdn, Ndn);
            BL_NAME(gmm)(metropolis2sInfo.Ddn, metropolis2sInfo.B0WRdn, SdnB0WRdn);
            // C0SB0WR = C0 @ SB0WR
            TensorHao<complex<double>, 2> C0SupB0WRup(truncatedDup, Nup);
            BL_NAME(gmm)(metropolis2sInfo.C0up, SupB0WRup, C0SupB0WRup);
            TensorHao<complex<double>, 2> C0SdnB0WRdn(truncatedDdn, Ndn);
            BL_NAME(gmm)(metropolis2sInfo.C0dn, SdnB0WRdn, C0SdnB0WRdn);
            // SC0SB0WR = S @ C0SB0WR
            TensorHao<complex<double>, 2> SupC0SupB0WRup(truncatedDup, Nup);
            BL_NAME(gmm)(metropolis2sInfo.Dup, C0SupB0WRup, SupC0SupB0WRup);
            TensorHao<complex<double>, 2> SdnC0SdnB0WRdn(truncatedDdn, Ndn);
            BL_NAME(gmm)(metropolis2sInfo.Ddn, C0SdnB0WRdn, SdnC0SdnB0WRdn);
            // temp = SB0WR + 0.5 * SC0SB0WR
            TensorHao<complex<double>, 2>& tempUp = SupB0WRup;
            for(int j=1-1; j<=tempUp.rank(1)-1; j++){
                for(int i=1-1; i<=tempUp.rank(0)-1; i++){
                    tempUp(i,j) = SupB0WRup(i,j) + 0.5 * SupC0SupB0WRup(i,j);
                }
            }   
            TensorHao<complex<double>, 2> & tempDn = SdnB0WRdn;
            for(int j=1-1; j<=tempDn.rank(1)-1; j++){
                for(int i=1-1; i<=tempDn.rank(0)-1; i++){
                    tempDn(i,j) = SdnB0WRdn(i,j) + 0.5 * SdnC0SdnB0WRdn(i,j);
                }
            }   
            // A1temp = A1 @ temp
            TensorHao<complex<double>, 2> A1tempUp(Nup, Nup);
            BL_NAME(gmm)(metropolis2sInfo.A1up, tempUp, A1tempUp);
            TensorHao<complex<double>, 2> A1tempDn(Ndn, Ndn);
            BL_NAME(gmm)(metropolis2sInfo.A1dn, tempDn, A1tempDn);
            // overlapMatrixup = metropolis2sInfo.A0WRup + A1temp;
            // TensorHao<std::complex<double>, 2> overlapMatrixup_Test = metropolis2sInfo.A0WRup + A1tempUp;
            // TensorHao<std::complex<double>, 2> overlapMatrixdn_Test = metropolis2sInfo.A0WRdn + A1tempDn;
            overlapMatrixup = metropolis2sInfo.A0WRup + A1tempUp;
            overlapMatrixdn = metropolis2sInfo.A0WRdn + A1tempDn;
        }else{
        ////////////////////////////////////////////////////
            // cost: O(M^3)
            TensorHao<complex<double>, 2> Dup_exp_x = getAuxMatrix_Dup_exp_x(auxNew, 0);
            // TensorHao<complex<double>, 2> Ddn_exp_x = getAuxMatrix_Ddn_exp_x(auxNew, 0);
            TensorHao<complex<double>, 2> & Ddn_exp_x = Dup_exp_x;
            // A1D_exp_x = A1 @ D_exp_x
            // A1D_exp_xB0WR = A1D_exp_x @ B0WR
            TensorHao<complex<double>, 2> A1upDup_exp_x(Nup, truncatedDup);
            TensorHao<complex<double>, 2> A1upDup_exp_xB0WRup(Nup, Nup);
            BL_NAME(gmm)(metropolis2sInfo.A1up, Dup_exp_x, A1upDup_exp_x);
            BL_NAME(gmm)(A1upDup_exp_x, metropolis2sInfo.B0WRup, A1upDup_exp_xB0WRup);
            // 
            TensorHao<complex<double>, 2> A1dnDdn_exp_x(Ndn, truncatedDdn);
            TensorHao<complex<double>, 2> A1dnDdn_exp_xB0WRdn(Ndn, Ndn);
            BL_NAME(gmm)(metropolis2sInfo.A1dn, Ddn_exp_x, A1dnDdn_exp_x);
            BL_NAME(gmm)(A1dnDdn_exp_x, metropolis2sInfo.B0WRdn, A1dnDdn_exp_xB0WRdn);

            overlapMatrixup = metropolis2sInfo.A0WRup + A1upDup_exp_xB0WRup;
            overlapMatrixdn = metropolis2sInfo.A0WRdn + A1dnDdn_exp_xB0WRdn;
            /////////////////////////////
            // checkMatrixDiff(overlapMatrixup, overlapMatrixup_Test, 10e-8);
            // checkMatrixDiff(overlapMatrixdn, overlapMatrixdn_Test, 10e-8);
        }
    }else{
        cout<<"Error: UNKNOW expM or no globalFastInitialized in getLogOverlapFromAux. expM: "<<expM<<" globalFastInitialized: "<<metropolis2sInfo.globalFastInitialized<<" globalFastUpdated: "<<metropolis2sInfo.globalFastUpdated<<endl;
        exit(1);
    }
    TensorHao<complex<double>, 2> overlapMatrixupTemp = overlapMatrixup;
    TensorHao<complex<double>, 2> overlapMatrixdnTemp = overlapMatrixdn;
    complex<double> logOverlap = logDeterminant(BL_NAME(LUconstruct)( move(overlapMatrixupTemp) )) + logDeterminant(BL_NAME(LUconstruct)( move(overlapMatrixdnTemp) ));
    logOverlap += conj(metropolis2sInfo.walkerLeftInBlock[0].getLogw()) + metropolis2sInfo.walkerRightInBlock[0].getLogw();
    logOverlap += expMinusDtV_Jastrow_vec->at(0).getTwoBodySample_logw_FromAux2s(auxNew);
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


void Metropolis2s::getOverlapMatrixInvA1ExpS(TensorHao<std::complex<double>, 2> & overlapMatrixInvA1ExpSUp, TensorHao<std::complex<double>, 2> & overlapMatrixInvA1ExpSDn)
{
    auto initial_start = std::chrono::high_resolution_clock::now();
    if(!metropolis2sInfo.globalFastInitialized){
        initial_globalFast();
    }
    if(!metropolis2sInfo.globalFastUpdated){
        update_globalFast();
    }
    auto initial_end = std::chrono::high_resolution_clock::now();
    // 
    //  directly calculate the overlap matrix from aux and predefined matrix
    auto calcu_start = std::chrono::high_resolution_clock::now();
    if(metropolis2sInfo.globalFastInitialized && metropolis2sInfo.globalFastUpdated){
        int Nup = metropolis2sInfo.B0WRup.rank(1);
        int Ndn = metropolis2sInfo.B0WRdn.rank(1);
        int truncatedDup = metropolis2sInfo.truncatedDup;
        int truncatedDdn = metropolis2sInfo.truncatedDdn;
        // 
        if(metropolis2sInfo.B0WRup.rank(0) + metropolis2sInfo.B0WRdn.rank(0) > metropolis2sInfo.B0WRup.rank(1) + metropolis2sInfo.B0WRdn.rank(1)){
            // cost: O(Ne M^2)
            auto calcu_Dup_start = std::chrono::high_resolution_clock::now();
            getAuxMatrix_Dup(metropolis2sInfo.auxiliaryFields[0], 0);
            auto calcu_Dup_end = std::chrono::high_resolution_clock::now();
            // getAuxMatrix_Ddn(metropolis2sInfo.auxiliaryFields[0], 0);
            // A1S = A1 @ S
            TensorHao<complex<double>, 2> A1upSup(Nup, truncatedDup);
            BL_NAME(gmm)(metropolis2sInfo.A1up, metropolis2sInfo.Dup, A1upSup);
            TensorHao<complex<double>, 2> A1dnSdn(Ndn, truncatedDdn);
            BL_NAME(gmm)(metropolis2sInfo.A1dn, metropolis2sInfo.Ddn, A1dnSdn); 
            // A1SC0 = A1S @ C0
            TensorHao<complex<double>, 2> A1upSupC0up(Nup, truncatedDup);
            BL_NAME(gmm)(A1upSup, metropolis2sInfo.C0up, A1upSupC0up);
            TensorHao<complex<double>, 2> A1dnSdnC0dn (Ndn, truncatedDdn);
            BL_NAME(gmm)(A1dnSdn, metropolis2sInfo.C0dn, A1dnSdnC0dn);
            // A1SC0S = A1SC0 @ S
            TensorHao<complex<double>, 2> A1upSupC0Sup(Nup, truncatedDup);
            BL_NAME(gmm)(A1upSupC0up, metropolis2sInfo.Dup, A1upSupC0Sup);
            TensorHao<complex<double>, 2> A1dnSdnC0Sdn(Ndn, truncatedDdn);
            BL_NAME(gmm)(A1dnSdnC0dn, metropolis2sInfo.Ddn, A1dnSdnC0Sdn);
            // tempUp = A1S + 0.5 * A1SC0S
            TensorHao<complex<double>, 2> tempUp(Nup, truncatedDup);
            TensorHao<complex<double>, 2> tempDn(Ndn, truncatedDdn);
            for(int i=1-1; i<=A1upSup.rank(0)-1; i++){
                for(int j=1-1; j<=A1upSup.rank(1)-1; j++){
                    tempUp(i,j) = A1upSup(i,j) + 0.5 * A1upSupC0Sup(i,j);
                }
            }
            for(int i=1-1; i<=A1dnSdn.rank(0)-1; i++){
                for(int j=1-1; j<=A1dnSdn.rank(1)-1; j++){
                    tempDn(i,j) = A1dnSdn(i,j) + 0.5 * A1dnSdnC0Sdn(i,j);
                }
            }
            overlapMatrixInvA1ExpSUp.resize(Nup, truncatedDup);
            overlapMatrixInvA1ExpSDn.resize(Ndn, truncatedDdn);
            BL_NAME(gmm)(metropolis2sInfo.overlapMatrixUp_inv, tempUp, overlapMatrixInvA1ExpSUp);
            BL_NAME(gmm)(metropolis2sInfo.overlapMatrixDn_inv, tempDn, overlapMatrixInvA1ExpSDn);
            // 
            auto elapsed_calcu_Dup = std::chrono::duration_cast<std::chrono::nanoseconds>(calcu_Dup_start - calcu_Dup_end);
            // if(MPIRank() == 0)printf("Time measured for elapsed_calcu_Dup: %.8f seconds.\n", elapsed_calcu_Dup.count() * 1e-9);
        }else{
            // cost: O(M^3)
            TensorHao<complex<double>, 2> Dup_exp_x = getAuxMatrix_Dup_exp_x(metropolis2sInfo.auxiliaryFields[0], 0);
            // TensorHao<complex<double>, 2> Ddn_exp_x = getAuxMatrix_Ddn_exp_x(metropolis2sInfo.auxiliaryFields[0], 0);
            TensorHao<complex<double>, 2> & Ddn_exp_x = Dup_exp_x;
            ////////////////////////////////////////////////
            // A1D_exp_x = A1 @ D_exp_x
            TensorHao<complex<double>, 2> A1upDup_exp_x(Nup, Dup_exp_x.rank(1));
            BL_NAME(gmm)(metropolis2sInfo.A1up, Dup_exp_x, A1upDup_exp_x);
            TensorHao<complex<double>, 2> A1dnDdn_exp_x(Ndn, Ddn_exp_x.rank(1));
            BL_NAME(gmm)(metropolis2sInfo.A1dn, Ddn_exp_x, A1dnDdn_exp_x);
            // 
            overlapMatrixInvA1ExpSUp.resize(metropolis2sInfo.overlapMatrixUp_inv.rank(0), A1upDup_exp_x.rank(1));
            overlapMatrixInvA1ExpSDn.resize(metropolis2sInfo.overlapMatrixDn_inv.rank(0), A1dnDdn_exp_x.rank(1));
            BL_NAME(gmm)(metropolis2sInfo.overlapMatrixUp_inv, A1upDup_exp_x, overlapMatrixInvA1ExpSUp);
            BL_NAME(gmm)(metropolis2sInfo.overlapMatrixDn_inv, A1dnDdn_exp_x, overlapMatrixInvA1ExpSDn);
            /////////////////////////////
            // checkMatrixDiff(overlapMatrixInvA1ExpSUp, overlapMatrixInvA1ExpSUp_Test, 10e-8);
            // checkMatrixDiff(overlapMatrixInvA1ExpSDn, overlapMatrixInvA1ExpSDn_Test, 10e-8);
        }
        /////////////////////////////////////////////////////
    }else{
        cout<<"Error: UNKNOW expM or no globalFastInitialized in getOverlapMatrixInvA1ExpS.: "<<" globalFastInitialized: "<<metropolis2sInfo.globalFastInitialized<<" globalFastUpdated: "<<metropolis2sInfo.globalFastUpdated<<endl;
        exit(1);
    }
    auto calcu_end = std::chrono::high_resolution_clock::now();
    // 
    auto elapsed_initial = std::chrono::duration_cast<std::chrono::nanoseconds>(initial_end - initial_start);
    auto elapsed_calcu = std::chrono::duration_cast<std::chrono::nanoseconds>(calcu_end - calcu_start);
    // if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
    // if(MPIRank() == 0)printf("Time measured for elapsed_initial: %.8f seconds.\n", elapsed_initial.count() * 1e-9);
    // if(MPIRank() == 0)printf("Time measured for elapsed_calcu: %.8f seconds.\n", elapsed_calcu.count() * 1e-9);
    // if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
}

void Metropolis2s::getOverlapMatrixInvA1ExpS_A1ExpS(TensorHao<std::complex<double>, 2> & overlapMatrixInvA1ExpSUp, TensorHao<std::complex<double>, 2> & overlapMatrixInvA1ExpSDn, TensorHao<std::complex<double>, 2> & A1ExpS_12up, TensorHao<std::complex<double>, 2> & A1ExpS_12dn)
{
    if(!metropolis2sInfo.globalFastInitialized){
        initial_globalFast();
    }
    if(!metropolis2sInfo.globalFastUpdated){
        update_globalFast();
    }
    // 
    //  directly calculate the overlap matrix from aux and predefined matrix
    int Nup = metropolis2sInfo.B0WRup.rank(1);
    int Ndn = metropolis2sInfo.B0WRdn.rank(1);
    int truncatedDup = metropolis2sInfo.truncatedDup;
    int truncatedDdn = metropolis2sInfo.truncatedDdn;
    // 
    if(metropolis2sInfo.globalFastInitialized && metropolis2sInfo.globalFastUpdated){
        getAuxMatrix_Dup(metropolis2sInfo.auxiliaryFields[0], 0);
        // getAuxMatrix_Ddn(metropolis2sInfo.auxiliaryFields[0], 0);
        ////////////////////////////////////////////////
        // A0 = walkerLeft0Wfdagger @ K
        // A0WR = A0 @ walkerRight0Wf
        // A1 = walkerLeft0Wfdagger @ U0
        // B0 = Vdagger0 @ K
        // B0WR = B0 @ walkerRight0Wf
        // C0 = Vdagger0 @ U0
        // 
        // A1S = A1 @ S
        TensorHao<complex<double>, 2> A1Sup(Nup, truncatedDup);
        BL_NAME(gmm)(metropolis2sInfo.A1up, metropolis2sInfo.Dup, A1Sup);
        TensorHao<complex<double>, 2> A1Sdn(Ndn, truncatedDdn);
        BL_NAME(gmm)(metropolis2sInfo.A1dn, metropolis2sInfo.Ddn, A1Sdn);
        // C0S = C0 @ S
        TensorHao<complex<double>, 2> C0Sup(truncatedDup, truncatedDup);
        BL_NAME(gmm)(metropolis2sInfo.C0up, metropolis2sInfo.Dup, C0Sup);
        TensorHao<complex<double>, 2> & C0Sdn = C0Sup;
        // TensorHao<complex<double>, 2> C0Sdn(truncatedDdn, truncatedDdn);
        // BL_NAME(gmm)(metropolis2sInfo.C0dn, metropolis2sInfo.Ddn, C0Sdn);
        // A1SC0S = A1S @ C0S 
        TensorHao<complex<double>, 2> A1SC0Sup(Nup, truncatedDup);
        BL_NAME(gmm)(A1Sup, C0Sup, A1SC0Sup);
        TensorHao<complex<double>, 2> A1SC0Sdn(Ndn, truncatedDdn);
        BL_NAME(gmm)(A1Sdn, C0Sdn, A1SC0Sdn);
        // A1ExpS_12 = A1S + halfA1SC0S
        A1ExpS_12up.resize(Nup, truncatedDup);
        A1ExpS_12dn.resize(Ndn, truncatedDdn);
        for(int i=1-1; i<=A1ExpS_12up.rank(0)-1; i++){
            for(int j=1-1; j<=A1ExpS_12up.rank(1)-1; j++){
                A1ExpS_12up(i,j) = A1Sup(i,j) + 0.5 * A1SC0Sup(i,j);
            }
        }
        for(int i=1-1; i<=A1ExpS_12dn.rank(0)-1; i++){
            for(int j=1-1; j<=A1ExpS_12dn.rank(1)-1; j++){
                A1ExpS_12dn(i,j) = A1Sdn(i,j) + 0.5 * A1SC0Sdn(i,j);
            }
        }
        // 
        overlapMatrixInvA1ExpSUp.resize(Nup, truncatedDup);
        overlapMatrixInvA1ExpSDn.resize(Ndn, truncatedDdn);
        BL_NAME(gmm)(metropolis2sInfo.overlapMatrixUp_inv, A1ExpS_12up, overlapMatrixInvA1ExpSUp);
        BL_NAME(gmm)(metropolis2sInfo.overlapMatrixDn_inv, A1ExpS_12dn, overlapMatrixInvA1ExpSDn);
        
        /////////////////////////////////////////////////////
    }else{
        cout<<"Error: UNKNOW expM or no globalFastInitialized in getOverlapMatrixInvA1ExpS.: "<<" globalFastInitialized: "<<metropolis2sInfo.globalFastInitialized<<" globalFastUpdated: "<<metropolis2sInfo.globalFastUpdated<<endl;
        exit(1);
    }
}

void Metropolis2s::get_PWR(TensorHao<std::complex<double>, 2> Pup_matrix, TensorHao<std::complex<double>, 2> Pdn_matrix, TensorHao<std::complex<double>, 2> & PWRup, TensorHao<std::complex<double>, 2> & PWRdn)
{
    if(!metropolis2sInfo.globalFastInitialized){
        initial_globalFast();
    }
    if(!metropolis2sInfo.globalFastUpdated){
        update_globalFast();
    }
    //  directly calculate the overlap matrix from aux and predefined matrix
    if(metropolis2sInfo.globalFastInitialized && metropolis2sInfo.globalFastUpdated){
        ////////////////////////////////////////////////
        // A0 = walkerLeft0Wfdagger @ K
        // A0WR = A0 @ walkerRight0Wf
        // A1 = walkerLeft0Wfdagger @ U0
        // B0 = Vdagger0 @ K
        // B0WR = B0 @ walkerRight0Wf
        // C0 = Vdagger0 @ U0
        // 
        // PWR = P @ walkerRight0Wf
        PWRup.resize(Pup_matrix.rank(0), metropolis2sInfo.walkerRightInBlock[0].getWfUp().rank(1));
        PWRdn.resize(Pdn_matrix.rank(0), metropolis2sInfo.walkerRightInBlock[0].getWfDn().rank(1));
        BL_NAME(gmm)(Pup_matrix, metropolis2sInfo.walkerRightInBlock[0].getWfUp(), PWRup);
        BL_NAME(gmm)(Pdn_matrix, metropolis2sInfo.walkerRightInBlock[0].getWfDn(), PWRdn);
        /////////////////////////////////////////////////////
    }else{
        cout<<"Error: UNKNOW expM or no globalFastInitialized in getLogOverlapFromAux.: "<<" globalFastInitialized: "<<metropolis2sInfo.globalFastInitialized<<" globalFastUpdated: "<<metropolis2sInfo.globalFastUpdated<<endl;
        exit(1);
    }
}

void Metropolis2s::get_A0PWR_B0PWR(TensorHao<std::complex<double>, 2> Pup_matrix, TensorHao<std::complex<double>, 2> Pdn_matrix, TensorHao<std::complex<double>, 2> & A0PWRup, TensorHao<std::complex<double>, 2> & A0PWRdn, TensorHao<std::complex<double>, 2> & B0PWRup, TensorHao<std::complex<double>, 2> & B0PWRdn)
{
    if(!metropolis2sInfo.globalFastInitialized){
        initial_globalFast();
    }
    if(!metropolis2sInfo.globalFastUpdated){
        update_globalFast();
    }
    //  directly calculate the overlap matrix from aux and predefined matrix
    if(metropolis2sInfo.globalFastInitialized && metropolis2sInfo.globalFastUpdated){
        ////////////////////////////////////////////////
        // A0 = walkerLeft0Wfdagger @ K
        // A0WR = A0 @ walkerRight0Wf
        // A1 = walkerLeft0Wfdagger @ U0
        // B0 = Vdagger0 @ K
        // B0WR = B0 @ walkerRight0Wf
        // C0 = Vdagger0 @ U0
        // 
        // PWR = P @ walkerRight0Wf
        TensorHao<complex<double>, 2> PWRup(Pup_matrix.rank(0), metropolis2sInfo.walkerRightInBlock[0].getWfUp().rank(1));
        TensorHao<complex<double>, 2> PWRdn(Pdn_matrix.rank(0), metropolis2sInfo.walkerRightInBlock[0].getWfDn().rank(1));
        BL_NAME(gmm)(Pup_matrix, metropolis2sInfo.walkerRightInBlock[0].getWfUp(), PWRup);
        BL_NAME(gmm)(Pdn_matrix, metropolis2sInfo.walkerRightInBlock[0].getWfDn(), PWRdn);
        // A0PWR = A0 @ PWR
        A0PWRup.resize(metropolis2sInfo.A0up.rank(0), PWRup.rank(1));
        A0PWRdn.resize(metropolis2sInfo.A0dn.rank(0), PWRdn.rank(1));
        BL_NAME(gmm)(metropolis2sInfo.A0up, PWRup, A0PWRup);
        BL_NAME(gmm)(metropolis2sInfo.A0dn, PWRdn, A0PWRdn);
        // B0PWR = B0 @ PWR
        B0PWRup.resize(metropolis2sInfo.B0up.rank(0), PWRup.rank(1));
        B0PWRdn.resize(metropolis2sInfo.B0dn.rank(0), PWRdn.rank(1));
        BL_NAME(gmm)(metropolis2sInfo.B0up, PWRup, B0PWRup);
        BL_NAME(gmm)(metropolis2sInfo.B0dn, PWRdn, B0PWRdn);
        /////////////////////////////////////////////////////
    }else{
        cout<<"Error: UNKNOW expM or no globalFastInitialized in getLogOverlapFromAux.: "<<" globalFastInitialized: "<<metropolis2sInfo.globalFastInitialized<<" globalFastUpdated: "<<metropolis2sInfo.globalFastUpdated<<endl;
        exit(1);
    }
}

void Metropolis2s::get_A0_A1ExpS12B0(TensorHao<std::complex<double>, 2> & A1ExpSUp, TensorHao<std::complex<double>, 2> & A1ExpSDn, TensorHao<std::complex<double>, 2> & A0up_A1ExpS12B0up, TensorHao<std::complex<double>, 2> & A0dn_A1ExpS12B0dn)
{
    if(!metropolis2sInfo.globalFastInitialized){
        initial_globalFast();
    }
    if(!metropolis2sInfo.globalFastUpdated){
        update_globalFast();
    }
    //  directly calculate the overlap matrix from aux and predefined matrix
    if(metropolis2sInfo.globalFastInitialized && metropolis2sInfo.globalFastUpdated){
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
        TensorHao<complex<double>, 2> A1ExpS12B0up(A1ExpSUp.rank(0), metropolis2sInfo.B0up.rank(1));
        TensorHao<complex<double>, 2> A1ExpS12B0dn(A1ExpSDn.rank(0), metropolis2sInfo.B0dn.rank(1));
        BL_NAME(gmm)(A1ExpSUp, metropolis2sInfo.B0up, A1ExpS12B0up);
        BL_NAME(gmm)(A1ExpSDn, metropolis2sInfo.B0dn, A1ExpS12B0dn);
        // 
        A0up_A1ExpS12B0up.resize(A1ExpSUp.rank(0), metropolis2sInfo.B0up.rank(1));
        A0dn_A1ExpS12B0dn.resize(A1ExpSDn.rank(0), metropolis2sInfo.B0dn.rank(1));
        A0up_A1ExpS12B0up = metropolis2sInfo.A0up + A1ExpS12B0up;
        A0dn_A1ExpS12B0dn = metropolis2sInfo.A0dn + A1ExpS12B0dn;
        /////////////////////////////////////////////////////
    }else{
        cout<<"Error: UNKNOW expM or no globalFastInitialized in getLogOverlapFromAux.: "<<" globalFastInitialized: "<<metropolis2sInfo.globalFastInitialized<<" globalFastUpdated: "<<metropolis2sInfo.globalFastUpdated<<endl;
        exit(1);
    }
}

///////////////////////////////////////////////////////////////////
void Metropolis2s::updateDirectB()
{
    TwoBodySample_Jastrow2s twoBodySample = expMinusDtV_Jastrow_vec->at(0).getTwoBodySampleFromAux2s(metropolis2sInfo.auxiliaryFields[0]);
    // update B
    metropolis2sInfo.Bup = twoBodySample.matrixUp;
    metropolis2sInfo.Bdn = twoBodySample.matrixDn;
}


void Metropolis2s::updateDirectOverlapMatrix_inv(Walker2s &walkerRight,
                                                TensorHao<std::complex<double>, 2> &overlapMatrixUp_inv,
                                                TensorHao<std::complex<double>, 2> &overlapMatrixDn_inv)
{
    overlapMatrixUp_inv.resize(walkerRight.getWfUp().rank(1), walkerRight.getWfUp().rank(1));
    BL_NAME(gmm)(trans(conj(metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize].getWfUp())), walkerRight.getWfUp(), overlapMatrixUp_inv);
    BL_NAME(inverse)(overlapMatrixUp_inv);
    
    overlapMatrixDn_inv.resize(walkerRight.getWfDn().rank(1), walkerRight.getWfDn().rank(1));
    BL_NAME(gmm)(trans(conj(metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize].getWfDn())), walkerRight.getWfDn(), overlapMatrixDn_inv);
    BL_NAME(inverse)(overlapMatrixDn_inv);
}

void Metropolis2s::updateOverlapMatrix_inv_fromOverlapMatrix()
{
    metropolis2sInfo.overlapMatrixUp_inv = metropolis2sInfo.overlapMatrixUp;
    metropolis2sInfo.overlapMatrixDn_inv = metropolis2sInfo.overlapMatrixDn;
    BL_NAME(inverse)(metropolis2sInfo.overlapMatrixUp_inv);
    BL_NAME(inverse)(metropolis2sInfo.overlapMatrixDn_inv);
}
///////////////////////////////////////////////////////////////////

void Metropolis2s::updateOverlapMatrixInvWithSMW(TensorHao<std::complex<double>, 2> &A_inv,
                                               TensorHao<std::complex<double>, 2> Uup,
                                               TensorHao<std::complex<double>, 2> Vdaggerup)
{
    // Using Sherman-Morrison-Woodbury formula to update A^{-1}:
    // (A + UV^T)^{-1} = A^{-1} - A^{-1}U(I + V^T A^{-1} U)^{-1} V^T A^{-1}
    
    // Step 1: Calculate A^{-1}U
    TensorHao<std::complex<double>, 2> Ainv_U(A_inv.rank(0), Uup.rank(1));
    BL_NAME(gmm)(A_inv, Uup, Ainv_U);
    
    // Step 2: Calculate V^T A^{-1}
    TensorHao<std::complex<double>, 2> V_T_Ainv(Vdaggerup.rank(0), A_inv.rank(1));
    BL_NAME(gmm)(Vdaggerup, A_inv, V_T_Ainv);
    
    // Step 3: Calculate I + V^T A^{-1} U
    TensorHao<std::complex<double>, 2> I_plus_VTAinvU(V_T_Ainv.rank(0), Uup.rank(1));
    BL_NAME(gmm)(V_T_Ainv, Uup, I_plus_VTAinvU);
    
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
    TensorHao<std::complex<double>, 2> AinvU_invIplus(Vdaggerup.rank(1), inv_I_plus_VTAinvU.rank(1));
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

std::complex<double> Metropolis2s::calculateTargetOverlapRatioFastUpdate(TensorHao<std::complex<double>, 2> Uup, 
                                                                       TensorHao<std::complex<double>, 2> Vdaggerup,
                                                                       TensorHao<std::complex<double>, 2> &overlapMatrixUp_inv)
{
    // Calculate det(I+V^T*A^{-1}*U) using fast update method
    TensorHao<std::complex<double>, 2> temp1(overlapMatrixUp_inv.rank(0), Uup.rank(1));
    BL_NAME(gmm)(overlapMatrixUp_inv, Uup, temp1);
    TensorHao<std::complex<double>, 2> temp2(Vdaggerup.rank(0), temp1.rank(1));
    BL_NAME(gmm)(Vdaggerup, temp1, temp2);
    
    // Add identity matrix
    for(size_t i_temp = 0; i_temp <= temp2.rank(0)-1; i_temp++) {
        temp2(i_temp, i_temp) += 1.0;
    }
    LUDecomp<complex<double>> temp2_LUOverlap = BL_NAME(LUconstruct)( move(temp2) );
    complex<double> targetOverlapRatio_fastUpdate = determinant(temp2_LUOverlap);
    
    return targetOverlapRatio_fastUpdate;
}

void Metropolis2s::calculateUpdatedB(TensorHao<std::complex<double>, 2> Bup, 
                                   TensorHao<std::complex<double>, 2> Bdn,
                                   TensorHao<std::complex<double>, 2> Uup_s,
                                   TensorHao<std::complex<double>, 2> Vdaggerup_s,
                                   TensorHao<std::complex<double>, 2> Udn_s,
                                   TensorHao<std::complex<double>, 2> Vdaggerdn_s,
                                   TensorHao<std::complex<double>, 2> &updated_Bup,
                                   TensorHao<std::complex<double>, 2> &updated_Bdn)
{
    // updated_Bup = Bup + Uup_s * Vdaggerup_s;
    TensorHao<complex<double>,2> temp_UV_up(Uup_s.rank(0), Vdaggerup_s.rank(1));
    BL_NAME(gmm)(Uup_s, Vdaggerup_s, temp_UV_up);
    updated_Bup = Bup + temp_UV_up;
    
    // updated_Bdn = Bdn + Udn_s * Vdaggerdn_s;
    TensorHao<complex<double>,2> temp_UV_dn(Udn_s.rank(0), Vdaggerdn_s.rank(1));
    BL_NAME(gmm)(Udn_s, Vdaggerdn_s, temp_UV_dn);
    updated_Bdn = Bdn + temp_UV_dn;
}

void Metropolis2s::updateOverlapMatricesAndRatios(std::vector<TensorHao<std::complex<double>, 2>> &Uup_vec,
                                                std::vector<TensorHao<std::complex<double>, 2>> &Udn_vec,
                                                std::vector<TensorHao<std::complex<double>, 2>> &Vdaggerup_vec,
                                                std::vector<TensorHao<std::complex<double>, 2>> &Vdaggerdn_vec,
                                                TensorHao<std::complex<double>, 2> &updated_overlapMatrixUp_inv,
                                                TensorHao<std::complex<double>, 2> &updated_overlapMatrixDn_inv,
                                                std::complex<double> &overlapRatioUp_fastUpdate,
                                                std::complex<double> &overlapRatioDn_fastUpdate)
{
    for(int temp_i=1-1; temp_i<=Uup_vec.size()-1; temp_i++){
        TensorHao<complex<double>,2> Uup = Uup_vec[temp_i];
        TensorHao<complex<double>,2> Udn = Udn_vec[temp_i];
        TensorHao<complex<double>,2> Vdaggerup = Vdaggerup_vec[temp_i];
        TensorHao<complex<double>,2> Vdaggerdn = Vdaggerdn_vec[temp_i];
        // det(I+U^TA^{−1}V)
        overlapRatioUp_fastUpdate = overlapRatioUp_fastUpdate * calculateTargetOverlapRatioFastUpdate(Uup, Vdaggerup, updated_overlapMatrixUp_inv);
        overlapRatioDn_fastUpdate = overlapRatioDn_fastUpdate * calculateTargetOverlapRatioFastUpdate(Udn, Vdaggerdn, updated_overlapMatrixDn_inv);
        updateOverlapMatrixInvWithSMW(updated_overlapMatrixUp_inv, Uup, Vdaggerup);
        updateOverlapMatrixInvWithSMW(updated_overlapMatrixDn_inv, Udn, Vdaggerdn);
    }
}

complex<double> Metropolis2s::performLocalUpdateWithFastOverlapRatio(int j_Jastrow, 
                                                    int inBlockIndex,               
                                                std::vector<int> flip_i_vec,
                                                tensor_hao::TensorHao<std::complex<double>, 1> auxNew,
                                                tensor_hao::TensorHao<std::complex<double>, 2> & Bup_proposed, 
                                                tensor_hao::TensorHao<std::complex<double>, 2> & Bdn_proposed, 
                                                tensor_hao::TensorHao<std::complex<double>, 2> & overlapMatrixUp_inv_proposed, 
                                                tensor_hao::TensorHao<std::complex<double>, 2> & overlapMatrixDn_inv_proposed)
{
    if(inBlockIndex != 0 || j_Jastrow != 0){
        cout<<"Error: Metropolis2s::performLocalUpdateWithFastOverlapRatio() inBlockIndex != 0 || j_Jastrow != 0: "<<inBlockIndex<<"  "<<j_Jastrow<<endl;
        exit(1);
    }
    Walker2s walkerRightTemp;
    // 
    TensorHao<std::complex<double>, 2> Uup_s = expMinusDtV_Jastrow_vec->at(j_Jastrow).getSqrtMinusDtSVDVecsMatrix_SVD(flip_i_vec[0], "Uup");
    Uup_s = Uup_s * (auxNew(flip_i_vec[0])-metropolis2sInfo.auxiliaryFields[inBlockIndex](flip_i_vec[0]));
    TensorHao<std::complex<double>, 2> Udn_s = expMinusDtV_Jastrow_vec->at(j_Jastrow).getSqrtMinusDtSVDVecsMatrix_SVD(flip_i_vec[0], "Udn");
    Udn_s = Udn_s * (auxNew(flip_i_vec[0])-metropolis2sInfo.auxiliaryFields[inBlockIndex](flip_i_vec[0]));
    TensorHao<std::complex<double>, 2> Vdaggerup_s = expMinusDtV_Jastrow_vec->at(j_Jastrow).getSqrtMinusDtSVDVecsMatrix_SVD(flip_i_vec[0], "Vdaggerup");
    TensorHao<std::complex<double>, 2> Vdaggerdn_s = expMinusDtV_Jastrow_vec->at(j_Jastrow).getSqrtMinusDtSVDVecsMatrix_SVD(flip_i_vec[0], "Vdaggerdn");
    // 
    TensorHao<std::complex<double>, 2> kup_matrix = expMinusDtK_Jastrow_vec->at(j_Jastrow).matrixUp;
    TensorHao<std::complex<double>, 2> kdn_matrix = expMinusDtK_Jastrow_vec->at(j_Jastrow).matrixDn;
    TensorHao<std::complex<double>, 2> wfnUpRight_matrix = metropolis2sInfo.walkerRightInBlock[0].getWfUp();
    TensorHao<std::complex<double>, 2> wfnDnRight_matrix = metropolis2sInfo.walkerRightInBlock[0].getWfDn();
    TensorHao<std::complex<double>, 2> wfnUpLeft_matrix = metropolis2sInfo.walkerLeftInBlock[0].getWfUp();
    TensorHao<std::complex<double>, 2> wfnDnLeft_matrix = metropolis2sInfo.walkerLeftInBlock[0].getWfDn();

    /////////////////////////////////////
    //get U V matrix
    /////////////////////////////////////
    // TensorHao<std::complex<double>, 2> deltaMatrixUp, deltaMatrixDn;
    // get_deltaMatrix_1s(deltaMatrixUp, 1, metropolis2sInfo.KVorder[j_Jastrow], wfnUpLeft_matrix, Uup_s, Vdaggerup_s, old_Bup, kup_matrix, wfnUpRight_matrix);
    // get_deltaMatrix_1s(deltaMatrixDn, 1, metropolis2sInfo.KVorder[j_Jastrow], wfnDnLeft_matrix, Udn_s, Vdaggerdn_s, old_Bdn, kdn_matrix, wfnDnRight_matrix);
    // 
    vector<TensorHao<complex<double>,2>> Uup_vec, Vdaggerup_vec;
    vector<TensorHao<complex<double>,2>> Udn_vec, Vdaggerdn_vec;
    get_updatedUV_1s(Uup_vec, Vdaggerup_vec, metropolis2sInfo.JastrowExpM[j_Jastrow], metropolis2sInfo.KVorder[j_Jastrow], wfnUpLeft_matrix, Uup_s, Vdaggerup_s, metropolis2sInfo.Bup, kup_matrix, wfnUpRight_matrix);
    get_updatedUV_1s(Udn_vec, Vdaggerdn_vec, metropolis2sInfo.JastrowExpM[j_Jastrow], metropolis2sInfo.KVorder[j_Jastrow], wfnDnLeft_matrix, Udn_s, Vdaggerdn_s, metropolis2sInfo.Bdn, kdn_matrix, wfnDnRight_matrix);
    // 
    // TensorHao<std::complex<double>, 2> tempMatrixUp(Uup.rank(0),Vdaggerup.rank(1));
    // BL_NAME(gmm)(Uup, Vdaggerup, tempMatrixUp);
    // for(int temp_i=1-1; temp_i<=tempMatrixUp.rank(0)-1; temp_i++){
    // for(int temp_j=1-1; temp_j<=tempMatrixUp.rank(1)-1; temp_j++){
    //     if(abs( tempMatrixUp(temp_i, temp_j) - deltaMatrixUp(temp_i, temp_j) )>=10e-8)cout <<"ATTENTION: "<<tempMatrixUp(temp_i, temp_j) - deltaMatrixUp(temp_i, temp_j) << endl;
    // }
    // }
    /////////////////////////////////////
    // 
    complex<double> overlapRatioUp_fastUpdate=1.0;
    complex<double> overlapRatioDn_fastUpdate=1.0;
    // 
    TensorHao<complex<double>,2> updated_Bup = metropolis2sInfo.Bup;
    TensorHao<complex<double>,2> updated_Bdn = metropolis2sInfo.Bdn;
    TensorHao<complex<double>,2> updated_overlapMatrixUp_inv = metropolis2sInfo.overlapMatrixUp_inv;
    TensorHao<complex<double>,2> updated_overlapMatrixDn_inv = metropolis2sInfo.overlapMatrixDn_inv;
    // 
    if(metropolis2sInfo.KVorder[j_Jastrow]=="VK"){
        calculateUpdatedB(metropolis2sInfo.Bup, metropolis2sInfo.Bdn, Uup_s, Vdaggerup_s, Udn_s, Vdaggerdn_s, updated_Bup, updated_Bdn);
        // 
        updateOverlapMatricesAndRatios(Uup_vec, Udn_vec, Vdaggerup_vec, Vdaggerdn_vec, updated_overlapMatrixUp_inv, updated_overlapMatrixDn_inv, overlapRatioUp_fastUpdate, overlapRatioDn_fastUpdate);
    }  
    Bup_proposed = updated_Bup;
    Bdn_proposed = updated_Bdn;
    overlapMatrixUp_inv_proposed = updated_overlapMatrixUp_inv;
    overlapMatrixDn_inv_proposed = updated_overlapMatrixDn_inv;
    
    return overlapRatioUp_fastUpdate*overlapRatioDn_fastUpdate;
}

void Metropolis2s::updateWalkerLeftMetro(){
    if(method.BPMetroUpdateType == "global_fast" && metropolis2sInfo.KVorder[0]=="VK" && metropolis2sInfo.JastrowExpM[0] == 2){
        ////////////////////////////////////////////////
        if(!metropolis2sInfo.globalFastInitialized){
            initial_globalFast();
        }
        if(!metropolis2sInfo.globalFastUpdated){
            update_globalFast();
        }
        int L = metropolis2sInfo.B0up.rank(1);
        if(metropolis2sInfo.B0up.rank(1) != metropolis2sInfo.B0dn.rank(1)){
            cout<<"Error: Metropolis2s::updateWalkerLeftMetro() metropolis2sInfo.B0up.rank(1) != metropolis2sInfo.B0dn.rank(1): "<<metropolis2sInfo.B0up.rank(1)<<"  "<<metropolis2sInfo.B0dn.rank(1)<<endl;
            exit(1);
        }
        int Nup = metropolis2sInfo.B0WRup.rank(1);
        int Ndn = metropolis2sInfo.B0WRdn.rank(1);
        int truncatedDup = metropolis2sInfo.truncatedDup;
        int truncatedDdn = metropolis2sInfo.truncatedDdn;
        ////////////////////////////////////////////////
        // 
        // A0 = walkerLeft0Wfdagger @ K
        // A1 = walkerLeft0Wfdagger @ U0
        // B0 = Vdagger0 @ K
        // C0 = Vdagger0 @ U0
        // 
        getAuxMatrix_Dup(metropolis2sInfo.auxiliaryFields[0], 0);
        // getAuxMatrix_Ddn(metropolis2sInfo.auxiliaryFields[0], 0);
        // A1S = A1 @ S
        TensorHao<complex<double>, 2> A1Sup(Nup, truncatedDup);
        BL_NAME(gmm)(metropolis2sInfo.A1up, metropolis2sInfo.Dup, A1Sup);
        TensorHao<complex<double>, 2> A1Sdn(Ndn, truncatedDdn);
        BL_NAME(gmm)(metropolis2sInfo.A1dn, metropolis2sInfo.Ddn, A1Sdn);
        // C0S = C0 @ S
        TensorHao<complex<double>, 2> C0Sup(truncatedDup, truncatedDup);
        BL_NAME(gmm)(metropolis2sInfo.C0up, metropolis2sInfo.Dup, C0Sup);
        TensorHao<complex<double>, 2> & C0Sdn = C0Sup;
        // TensorHao<complex<double>, 2> C0Sdn(truncatedDdn, truncatedDdn);
        // BL_NAME(gmm)(metropolis2sInfo.C0dn, metropolis2sInfo.Ddn, C0Sdn);
        // A1SB0 = A1S @ B0
        TensorHao<complex<double>, 2> A1SB0up(Nup, L);
        BL_NAME(gmm)(A1Sup, metropolis2sInfo.B0up, A1SB0up);
        TensorHao<complex<double>, 2> A1SB0dn(Ndn, L);
        BL_NAME(gmm)(A1Sdn, metropolis2sInfo.B0dn, A1SB0dn);
        // A1SC0S = A1S @ C0S
        TensorHao<complex<double>, 2> A1SC0Sup(Nup, truncatedDup);
        BL_NAME(gmm)(A1Sup, C0Sup, A1SC0Sup);
        TensorHao<complex<double>, 2> A1SC0Sdn(Ndn, truncatedDdn);
        BL_NAME(gmm)(A1Sdn, C0Sdn, A1SC0Sdn);
        // A1SC0SB0 = A1SC0S @ B0
        TensorHao<complex<double>, 2> A1SC0SB0up(Nup, L);
        BL_NAME(gmm)(A1SC0Sup, metropolis2sInfo.B0up, A1SC0SB0up);
        TensorHao<complex<double>, 2> A1SC0SB0dn(Ndn, L);
        BL_NAME(gmm)(A1SC0Sdn, metropolis2sInfo.B0dn, A1SC0SB0dn);
        for(int i=1-1; i<=A1SC0SB0up.rank(0)-1; i++){
            for(int j=1-1; j<=A1SC0SB0up.rank(1)-1; j++){
                A1SC0SB0up(i,j) = 0.5 * A1SC0SB0up(i,j);
            }
        }
        for(int i=1-1; i<=A1SC0SB0dn.rank(0)-1; i++){
            for(int j=1-1; j<=A1SC0SB0dn.rank(1)-1; j++){
                A1SC0SB0dn(i,j) = 0.5 * A1SC0SB0dn(i,j);
            }
        }
        // 
        // metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize] = A0 + A1SB0 + 0.5 @ A1SC0SB0
        // 
        metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize].wfUpRef() = conj(trans(metropolis2sInfo.A0up + A1SB0up + A1SC0SB0up));
        metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize].wfDnRef() = conj(trans(metropolis2sInfo.A0dn + A1SB0dn + A1SC0SB0dn));
        metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize].logwRef() = metropolis2sInfo.walkerLeftInBlock[0].getLogw() + conj(expMinusDtK_Jastrow_vec->at(0).logw) + conj(expMinusDtV_Jastrow_vec->at(0).getTwoBodySample_logw_FromAux2s(metropolis2sInfo.auxiliaryFields[0]));
        // 
        ///////////////////////////////////
        // Test
        ///////////////////////////////////
        // TwoBodySample_Jastrow2s twoBodySample = expMinusDtV_Jastrow_vec->at(0).getTwoBodySampleFromAux2s(metropolis2sInfo.auxiliaryFields[0]);
        // //
        // TensorHao<complex<double>,2> exp_Bup_proposed = expMatrix(twoBodySample.matrixUp, metropolis2sInfo.JastrowExpM[0]);
        // TensorHao<complex<double>,2> exp_Bdn_proposed = expMatrix(twoBodySample.matrixDn, metropolis2sInfo.JastrowExpM[0]);
        // // 
        // TensorHao<complex<double>,2> matrixUp_temp(metropolis2sInfo.walkerLeftInBlock[0].getWfUp().rank(0), metropolis2sInfo.walkerLeftInBlock[0].getWfUp().rank(1));
        // TensorHao<complex<double>,2> matrixUp_temp2(metropolis2sInfo.walkerLeftInBlock[0].getWfUp().rank(0), metropolis2sInfo.walkerLeftInBlock[0].getWfUp().rank(1));
        // TensorHao<complex<double>,2> matrixDn_temp2(metropolis2sInfo.walkerLeftInBlock[0].getWfDn().rank(0), metropolis2sInfo.walkerLeftInBlock[0].getWfDn().rank(1)); 
        // TensorHao<complex<double>,2> matrixDn_temp(metropolis2sInfo.walkerLeftInBlock[0].getWfDn().rank(0), metropolis2sInfo.walkerLeftInBlock[0].getWfDn().rank(1)); 
        // // 
        // BL_NAME(gmm)( exp_Bup_proposed, metropolis2sInfo.walkerLeftInBlock[0].getWfUp(), matrixUp_temp, 'C' );
        // BL_NAME(gmm)( exp_Bdn_proposed, metropolis2sInfo.walkerLeftInBlock[0].getWfDn(), matrixDn_temp, 'C' );
        // // 
        // BL_NAME(gmm)( expMinusDtK_Jastrow_vec->at(0).matrixUp, matrixUp_temp, matrixUp_temp2, 'C' );
        // BL_NAME(gmm)( expMinusDtK_Jastrow_vec->at(0).matrixDn, matrixDn_temp, matrixDn_temp2, 'C' );
        // // 
        // // cout<<"metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize].getWfUp(): "<<metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize].getWfUp().rank(0)<<" "<<metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize].getWfUp().rank(1)<<endl;
        // // cout<<"matrixUp_temp2: "<<matrixUp_temp2.rank(0)<<" "<<matrixUp_temp2.rank(1)<<endl;
        // checkMatrixDiff(metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize].getWfUp(), matrixUp_temp2, 10e-8);
        // checkMatrixDiff(metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize].getWfDn(), matrixDn_temp2, 10e-8);
        // // 
        ///////////////////////////////////
        // 
        ////////////////////////////////////////////////
    }else if(method.BPMetroUpdateType == "local" && metropolis2sInfo.KVorder[0]=="VK" && metropolis2sInfo.JastrowExpM[0] == 2){ 
        ////////////////////////////////////////////////
        TensorHao<complex<double>,2> exp_Bup_proposed = expMatrix(metropolis2sInfo.Bup, metropolis2sInfo.JastrowExpM[0]);
        TensorHao<complex<double>,2> exp_Bdn_proposed = expMatrix(metropolis2sInfo.Bdn, metropolis2sInfo.JastrowExpM[0]);
        // 
        TensorHao<complex<double>,2> matrixUp_temp(metropolis2sInfo.walkerLeftInBlock[0].getWfUp().rank(0), metropolis2sInfo.walkerLeftInBlock[0].getWfUp().rank(1));
        TensorHao<complex<double>,2> matrixUp_temp2(metropolis2sInfo.walkerLeftInBlock[0].getWfUp().rank(0), metropolis2sInfo.walkerLeftInBlock[0].getWfUp().rank(1));
        TensorHao<complex<double>,2> matrixDn_temp2(metropolis2sInfo.walkerLeftInBlock[0].getWfDn().rank(0), metropolis2sInfo.walkerLeftInBlock[0].getWfDn().rank(1)); 
        TensorHao<complex<double>,2> matrixDn_temp(metropolis2sInfo.walkerLeftInBlock[0].getWfDn().rank(0), metropolis2sInfo.walkerLeftInBlock[0].getWfDn().rank(1)); 
        // 
        BL_NAME(gmm)( exp_Bup_proposed, metropolis2sInfo.walkerLeftInBlock[0].getWfUp(), matrixUp_temp, 'C' );
        BL_NAME(gmm)( exp_Bdn_proposed, metropolis2sInfo.walkerLeftInBlock[0].getWfDn(), matrixDn_temp, 'C' );
        // 
        BL_NAME(gmm)( expMinusDtK_Jastrow_vec->at(0).matrixUp, matrixUp_temp, matrixUp_temp2, 'C' );
        BL_NAME(gmm)( expMinusDtK_Jastrow_vec->at(0).matrixDn, matrixDn_temp, matrixDn_temp2, 'C' );
        // 
        metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize].wfUpRef() = matrixUp_temp2;
        metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize].wfDnRef() = matrixDn_temp2;
        metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize].logwRef() = metropolis2sInfo.walkerLeftInBlock[0].getLogw() + conj(expMinusDtK_Jastrow_vec->at(0).logw) + conj(expMinusDtV_Jastrow_vec->at(0).getTwoBodySample_logw_FromAux2s(metropolis2sInfo.auxiliaryFields[0]));
        ////////////////////////////////////////////////
    }
}

const Walker2s& Metropolis2s::getWalkerLeftMetro()
{
    updateWalkerLeftMetro();
    return metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize];  
}

////////////////////////////////////////////////
///////////////////