//
// Created by boruoshihao on 1/17/17.
//
#include "../../include/Metropolis/metropolis.h"

using namespace std;
using namespace tensor_hao;

void Metropolis::initialField(Walker &walkerLeft,Walker &walkerRight)
{
    //
    metropolisInfo.auxiliaryFields.resize( metropolisInfo.BPMetroTimesliceBlockSize );
    metropolisInfo.dynamicForceFields.resize( metropolisInfo.BPMetroTimesliceBlockSize );

    metropolisInfo.walkerRightInBlock.resize(metropolisInfo.BPMetroTimesliceBlockSize+1);
    metropolisInfo.logWeightRightInBlock.resize(metropolisInfo.BPMetroTimesliceBlockSize+1);

    metropolisInfo.walkerLeftInBlock.resize(metropolisInfo.BPMetroTimesliceBlockSize+1);
    metropolisInfo.logWeightLeftInBlock.resize(metropolisInfo.BPMetroTimesliceBlockSize+1);

    Walker walkerRightTemp,walkerRightTemp2;
    TwoBodySample_Jastrow twoBodySample;

    complex<double> logWeight(0,0);

    walkerRightTemp=walkerRight;
    walkerRightTemp2=walkerRight;
    
    metropolisInfo.walkerRightInBlock[0] = walkerRight;
    metropolisInfo.logWeightRightInBlock[0] = logWeight;
    metropolisInfo.walkerLeftInBlock[0] = walkerLeft;
    metropolisInfo.logWeightLeftInBlock[0] = logWeight;
    //
    int counter=-1;
    for(int j_Jastrow=metropolisInfo.numOfJastrow-1; j_Jastrow >= 0; j_Jastrow--){
    for(int j=1-1; j<=metropolisInfo.JastrowSlice[j_Jastrow]-1; j++){
            counter ++;
            if(metropolisInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_HAFQMC_icf"){
                ////////////////////////////////////////////////////////////
                if( method.BPMetroInitialAuxiliaryFlag == "constForceInitial" )
                {
                    metropolisInfo.auxiliaryFields[counter] = expMinusDtV_Jastrow_vec->at(j_Jastrow).sampleAuxFromForce(metropolisInfo.constForce_Jastrow[j_Jastrow]);
                    metropolisInfo.dynamicForceFields[counter]=metropolisInfo.constForce_Jastrow[j_Jastrow];
                    if( method.BPMetroForceType == "constForce" ){
                        twoBodySample = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySampleFromAux(metropolisInfo.auxiliaryFields[counter]);
                    }
                    //
                }else{
                    cout<<"Error: UNKNOW BPMetroInitialAuxiliaryFlag: "<< method.BPMetroInitialAuxiliaryFlag<<endl;
                    exit(1);
                }
                //
                    
                if(metropolisInfo.KVorder[j_Jastrow]=="VK"){
                    //
                    twoBodySampleWalkerRightOperation.reset("fixedOrder", metropolisInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp2, walkerRightTemp);
                }else if(metropolisInfo.KVorder[j_Jastrow]=="K^daggerV"){
                    twoBodySampleWalkerRightOperation.reset("fixedOrder", metropolisInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                    twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp, walkerRightTemp2);
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp2, walkerRightTemp);
                }else if(metropolisInfo.KVorder[j_Jastrow]=="KVK"){
                    twoBodySampleWalkerRightOperation.reset("dynamicOrder");
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp2, walkerRightTemp);
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    walkerRightTemp = walkerRightTemp2;
                }else{
                    cout<<"Error: UNKNOW metropolisInfo.KVorder[j_Jastrow]: "<<metropolisInfo.KVorder[j_Jastrow]<<endl;
                    exit(1);
                }
                //
                ////////////////////////////////////////////////////////////
            }
            else if(metropolisInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_K_HAFQMC_icf"){
                ////////////////////////////////////////////////////////////
                //default aux 
                metropolisInfo.auxiliaryFields[counter] = expMinusDtV_Jastrow_vec->at(j_Jastrow).sampleAuxFromForce(metropolisInfo.constForce_Jastrow[j_Jastrow]);
                metropolisInfo.dynamicForceFields[counter]=metropolisInfo.constForce_Jastrow[j_Jastrow];
                //
                if(metropolisInfo.KVorder[j_Jastrow]=="VK"){
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    walkerRightTemp=walkerRightTemp2;
                }else if(metropolisInfo.KVorder[j_Jastrow]=="K^daggerV"){
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    walkerRightTemp=walkerRightTemp2;
                }else if(metropolisInfo.KVorder[j_Jastrow]=="KVK"){
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp2, walkerRightTemp);
                }else{
                    cout<<"Error: UNKNOW metropolisInfo.KVorder[j_Jastrow]: "<<metropolisInfo.KVorder[j_Jastrow]<<endl;
                    exit(1);
                }
                //
                ////////////////////////////////////////////////////////////
            }
            else{
                cout<<"Error: UNKNOW metropolisInfo.variableName_vec[j_Jastrow]: "<<metropolisInfo.variableName_vec[j_Jastrow]<<endl;
                exit(1);
            }       
            //
            // walkerRightTemp.addLogw(method.Metro_dtET );
            // if( j % method.BPMetroStabilizeStep == 0 )
            // {
                walkerRightTemp.stabilize();
            // }
            complex <double> logWeightTemp = walkerRightTemp.getLogw();
            logWeight = real(walkerRightTemp.getLogw());
            //////////////////////////////////
            // logWeight = 0.0;
            //////////////////////////////////
            walkerRightTemp.logwRef()= logWeightTemp - logWeight;
            metropolisInfo.walkerRightInBlock[counter+1] = walkerRightTemp;
            metropolisInfo.logWeightRightInBlock[counter+1] = logWeight;
    }
    }
    //
    WalkerWalkerOperation_Jastrow walkerWalkerOperation;
    walkerWalkerOperation.set( metropolisInfo.walkerLeftInBlock[0], metropolisInfo.walkerRightInBlock[metropolisInfo.BPMetroTimesliceBlockSize] );
    metropolisInfo.currentLogOverlap = walkerWalkerOperation.returnLogOverlap();
    //
}


void Metropolis::copyField(MetropolisInfo metropolisInfo_input)
{
    // metropolisInfo = metropolisInfo_input;
    // 
    metropolisInfo.BPMetroTimesliceBlockSize = metropolisInfo_input.BPMetroTimesliceBlockSize;

    metropolisInfo.inBlockIndex = metropolisInfo_input.inBlockIndex;
    metropolisInfo.currentLogOverlap = metropolisInfo_input.currentLogOverlap;

    metropolisInfo.auxiliaryFields = metropolisInfo_input.auxiliaryFields;
    metropolisInfo.dynamicForceFields = metropolisInfo_input.dynamicForceFields;

    metropolisInfo.walkerRightInBlock = metropolisInfo_input.walkerRightInBlock;
    metropolisInfo.walkerLeftInBlock = metropolisInfo_input.walkerLeftInBlock;

    metropolisInfo.logWeightRightInBlock = metropolisInfo_input.logWeightRightInBlock;
    metropolisInfo.logWeightLeftInBlock = metropolisInfo_input.logWeightLeftInBlock;
    metropolisInfo.overlapMatrix_inv = metropolisInfo_input.overlapMatrix_inv;

    metropolisInfo.B = metropolisInfo_input.B;
    /////////////////////////////////////////
    metropolisInfo_input.truncatedD = metropolisInfo_input.truncatedD;
    metropolisInfo_input.overlapMatrix = metropolisInfo_input.overlapMatrix;
    metropolisInfo_input.A0 = metropolisInfo_input.A0;
    metropolisInfo_input.A0WR = metropolisInfo_input.A0WR;
    metropolisInfo_input.A1 = metropolisInfo_input.A1;
    metropolisInfo_input.B0 = metropolisInfo_input.B0;
    metropolisInfo_input.B0WR = metropolisInfo_input.B0WR;
    metropolisInfo_input.C0 = metropolisInfo_input.C0;
    metropolisInfo_input.D = metropolisInfo_input.D;
    metropolisInfo_input.globalFastInitialized = metropolisInfo_input.globalFastInitialized;
    metropolisInfo_input.globalFastUpdated = metropolisInfo_input.globalFastUpdated;
    /////////////////////////////////////////
}


void Metropolis::initialField_again(Walker &walkerRight)
{
    //
    Walker walkerRightTemp,walkerRightTemp2;
    TwoBodySample_Jastrow twoBodySample;

    complex<double> logWeight(0,0);

    walkerRightTemp=walkerRight;
    walkerRightTemp2=walkerRight;

    metropolisInfo.walkerRightInBlock[0] = walkerRight;
    metropolisInfo.logWeightRightInBlock[0] = logWeight;
    //
    int counter=-1;
    for(int j_Jastrow=metropolisInfo.numOfJastrow-1; j_Jastrow >= 0; j_Jastrow--){
    for(int j=1-1; j<=metropolisInfo.JastrowSlice[j_Jastrow]-1; j++){
            counter ++;
            if(metropolisInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_HAFQMC_icf"){
                ////////////////////////////////////////////////////////////
                if( method.BPMetroForceType == "constForce" ){
                    // twoBodySample = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySampleFromAux_icf_fixedForce(metropolisInfo.auxiliaryFields[counter],metropolisInfo.dynamicForceFields[counter]);
                    twoBodySample = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySampleFromAux(metropolisInfo.auxiliaryFields[counter]);
                    cout<<"j_Jastrow: "<<j_Jastrow<<" metropolisInfo.auxiliaryFields[counter]: "<<metropolisInfo.auxiliaryFields[counter]<<endl;
                }
                //
                if(metropolisInfo.KVorder[j_Jastrow]=="VK"){
                    twoBodySampleWalkerRightOperation.reset("fixedOrder", metropolisInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp2, walkerRightTemp);
                }else if(metropolisInfo.KVorder[j_Jastrow]=="K^daggerV"){
                    twoBodySampleWalkerRightOperation.reset("fixedOrder", metropolisInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                    twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp, walkerRightTemp2);
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp2, walkerRightTemp);
                }else if(metropolisInfo.KVorder[j_Jastrow]=="KVK"){
                    twoBodySampleWalkerRightOperation.reset("dynamicOrder");
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp2, walkerRightTemp);
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    walkerRightTemp = walkerRightTemp2;
                }else{
                    cout<<"Error: UNKNOW metropolisInfo.KVorder[j_Jastrow]: "<<metropolisInfo.KVorder[j_Jastrow]<<endl;
                    cout<<metropolisInfo.numOfJastrow<<endl;
                    cout<<metropolisInfo.KVorder.size()<<endl;
                    for(int temp_i=1-1; temp_i<=metropolisInfo.KVorder.size()-1; temp_i++ ){
                        cout<<metropolisInfo.KVorder[temp_i]<<endl;
                    }
                    exit(1);
                }
                //
                ////////////////////////////////////////////////////////////
            }
            else if(metropolisInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_K_HAFQMC_icf"){
                ////////////////////////////////////////////////////////////
                //default aux 
                metropolisInfo.auxiliaryFields[counter] = expMinusDtV_Jastrow_vec->at(j_Jastrow).sampleAuxFromForce(metropolisInfo.constForce_Jastrow[j_Jastrow]);
                metropolisInfo.dynamicForceFields[counter]=metropolisInfo.constForce_Jastrow[j_Jastrow];
                //
                if(metropolisInfo.KVorder[j_Jastrow]=="VK"){
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    walkerRightTemp=walkerRightTemp2;
                }else if(metropolisInfo.KVorder[j_Jastrow]=="K^daggerV"){
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    walkerRightTemp=walkerRightTemp2;
                }else if(metropolisInfo.KVorder[j_Jastrow]=="KVK"){
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp2, walkerRightTemp);
                }else{
                    cout<<"Error: UNKNOW metropolisInfo.KVorder[j_Jastrow]: "<<metropolisInfo.KVorder[j_Jastrow]<<endl;
                    cout<<metropolisInfo.numOfJastrow<<endl;
                    cout<<metropolisInfo.KVorder.size()<<endl;
                    for(int temp_i=1-1; temp_i<=metropolisInfo.KVorder.size()-1; temp_i++ ){
                        cout<<metropolisInfo.KVorder[temp_i]<<endl;
                    }
                    exit(1);
                }
                //
                ////////////////////////////////////////////////////////////
            }
            else{
                cout<<"Error: UNKNOW metropolisInfo.variableName_vec[j_Jastrow]: "<<metropolisInfo.variableName_vec[j_Jastrow]<<endl;
                exit(1);
            }
            //
            // walkerRightTemp.addLogw(method.Metro_dtET );
            // if( j % method.BPMetroStabilizeStep == 0 )
            // {
                walkerRightTemp.stabilize();
            // }
            complex <double> logWeightTemp = walkerRightTemp.getLogw();
            logWeight = real(walkerRightTemp.getLogw());
            //////////////////////////////////
            // logWeight = 0.0;
            //////////////////////////////////
            walkerRightTemp.logwRef()= logWeightTemp - logWeight;

            metropolisInfo.walkerRightInBlock[counter+1] = walkerRightTemp;
            metropolisInfo.logWeightRightInBlock[counter+1] = logWeight;
    }
    }
    WalkerWalkerOperation_Jastrow walkerWalkerOperation;
    walkerWalkerOperation.set( metropolisInfo.walkerLeftInBlock[0], metropolisInfo.walkerRightInBlock[metropolisInfo.BPMetroTimesliceBlockSize] );
    metropolisInfo.currentLogOverlap = walkerWalkerOperation.returnLogOverlap();
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
//////////////////////////////////////////////////