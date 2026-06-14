//
// Created by boruoshihao on 1/17/17.
//
#include "../../include/Metropolis2s/metropolis2s.h"

using namespace std;
using namespace tensor_hao;

void Metropolis2s::initialField(Walker2s &walkerLeft,Walker2s &walkerRight)
{
    //
    metropolis2sInfo.auxiliaryFields.resize( metropolis2sInfo.BPMetroTimesliceBlockSize );
    // metropolis2sInfo.auxiliaryFields_BP.resize( metropolis2sInfo.BPMetroTimesliceBlockSize );
    metropolis2sInfo.dynamicForceFields.resize( metropolis2sInfo.BPMetroTimesliceBlockSize );
    // metropolis2sInfo.dynamicForceFields_BP.resize( metropolis2sInfo.BPMetroTimesliceBlockSize );

    metropolis2sInfo.walkerRightInBlock.resize(metropolis2sInfo.BPMetroTimesliceBlockSize+1);
    metropolis2sInfo.logWeightRightInBlock.resize(metropolis2sInfo.BPMetroTimesliceBlockSize+1);

    metropolis2sInfo.walkerLeftInBlock.resize(metropolis2sInfo.BPMetroTimesliceBlockSize+1);
    metropolis2sInfo.logWeightLeftInBlock.resize(metropolis2sInfo.BPMetroTimesliceBlockSize+1);

    Walker2s walkerRightTemp,walkerRightTemp2;
    TwoBodySample_Jastrow2s twoBodySample;
    // TwoBodySample_BP_Jastrow2s twoBodySample_BP;

    complex<double> logWeight(0,0);

    walkerRightTemp=walkerRight;
    walkerRightTemp2=walkerRight;
    
    metropolis2sInfo.walkerRightInBlock[0] = walkerRight;
    metropolis2sInfo.logWeightRightInBlock[0] = logWeight;
    metropolis2sInfo.walkerLeftInBlock[0] = walkerLeft;
    metropolis2sInfo.logWeightLeftInBlock[0] = logWeight;
    //
    int counter=-1;
    for(int j_Jastrow=metropolis2sInfo.numOfJastrow-1; j_Jastrow >= 0; j_Jastrow--){
    for(int j=1-1; j<=metropolis2sInfo.JastrowSlice[j_Jastrow]-1; j++){
            counter ++;
            if(metropolis2sInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_HAFQMC_icf"){
                ////////////////////////////////////////////////////////////
                if( method.BPMetroInitialAuxiliaryFlag == "constForceInitial" )
                {
                    metropolis2sInfo.auxiliaryFields[counter] = expMinusDtV_Jastrow_vec->at(j_Jastrow).sampleAuxFromForce(metropolis2sInfo.constForce_Jastrow[j_Jastrow]);
                    metropolis2sInfo.dynamicForceFields[counter]=metropolis2sInfo.constForce_Jastrow[j_Jastrow];
                    if( method.BPMetroForceType == "constForce" ){
                        // twoBodySample = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySampleFromAux_icf_fixedForce(metropolis2sInfo.auxiliaryFields[counter],metropolis2sInfo.dynamicForceFields[counter]);
                        twoBodySample = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySampleFromAux2s(metropolis2sInfo.auxiliaryFields[counter]);
                    }
                    //
                }else{
                    cout<<"Error: UNKNOW BPMetroInitialAuxiliaryFlag: "<< method.BPMetroInitialAuxiliaryFlag<<endl;
                    exit(1);
                }
                //
                    
                if(metropolis2sInfo.KVorder[j_Jastrow]=="VK"){
                    //
                    twoBodySampleWalkerRightOperation.reset("fixedOrder", metropolis2sInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp2, walkerRightTemp);
                }else if(metropolis2sInfo.KVorder[j_Jastrow]=="K^daggerV"){
                    twoBodySampleWalkerRightOperation.reset("fixedOrder", metropolis2sInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                    twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp, walkerRightTemp2);
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp2, walkerRightTemp);
                }else if(metropolis2sInfo.KVorder[j_Jastrow]=="KVK"){
                    twoBodySampleWalkerRightOperation.reset("dynamicOrder");
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp2, walkerRightTemp);
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    walkerRightTemp = walkerRightTemp2;
                }else{
                    cout<<"Error: UNKNOW metropolis2sInfo.KVorder[j_Jastrow]: "<<metropolis2sInfo.KVorder[j_Jastrow]<<endl;
                    exit(1);
                }
                //
                ////////////////////////////////////////////////////////////
            }
            else if(metropolis2sInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_K_HAFQMC_icf"){
                ////////////////////////////////////////////////////////////
                //default aux 
                metropolis2sInfo.auxiliaryFields[counter] = expMinusDtV_Jastrow_vec->at(j_Jastrow).sampleAuxFromForce(metropolis2sInfo.constForce_Jastrow[j_Jastrow]);
                metropolis2sInfo.dynamicForceFields[counter]=metropolis2sInfo.constForce_Jastrow[j_Jastrow];
                //
                if(metropolis2sInfo.KVorder[j_Jastrow]=="VK"){
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    walkerRightTemp=walkerRightTemp2;
                }else if(metropolis2sInfo.KVorder[j_Jastrow]=="K^daggerV"){
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    walkerRightTemp=walkerRightTemp2;
                }else if(metropolis2sInfo.KVorder[j_Jastrow]=="KVK"){
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp2, walkerRightTemp);
                }else{
                    cout<<"Error: UNKNOW metropolis2sInfo.KVorder[j_Jastrow]: "<<metropolis2sInfo.KVorder[j_Jastrow]<<endl;
                    exit(1);
                }
                //
                ////////////////////////////////////////////////////////////
            }
            else if(metropolis2sInfo.variableName_vec[j_Jastrow]=="HubbardSOC"){
                ////////////////////////////////////////////////////////////
                // if( method.BPMetroInitialAuxiliaryFlag == "constForceInitial" )
                // {
                //     metropolis2sInfo.auxiliaryFields_BP[counter] = jastrowProjector->expMinusDtV_BP_Jastrow_vec[j_Jastrow].sampleAuxFromForce(jastrowProjector->constForce_BP_Jastrow[j_Jastrow]);
                //     metropolis2sInfo.dynamicForceFields_BP[counter]=jastrowProjector->constForce_BP_Jastrow[j_Jastrow];
                //     // twoBodySample_BP = jastrowProjector->expMinusDtV_BP_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAuxForce(metropolis2sInfo.auxiliaryFields_BP[counter],metropolis2sInfo.dynamicForceFields_BP[counter]);
                //     twoBodySample_BP = jastrowProjector->expMinusDtV_BP_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAux(metropolis2sInfo.auxiliaryFields_BP[counter]);
                //     //
                // }else{
                //     cout<<"Error: UNKNOW BPMetroInitialAuxiliaryFlag: "<< method.BPMetroInitialAuxiliaryFlag<<endl;
                //     exit(1);
                // }
                // // 
                // if(metropolis2sInfo.KVorder[j_Jastrow]=="KVK"){
                //     oneBodyWalkerRightOperation_BP.applyToRight(jastrowProjector->expMinusDtK_BP_Jastrow_vec[j_Jastrow], walkerRightTemp, walkerRightTemp2);
                //     twoBodySampleWalkerRightOperation_BP.applyToRight(twoBodySample_BP, walkerRightTemp2, walkerRightTemp);
                //     oneBodyWalkerRightOperation_BP.applyToRight(jastrowProjector->expMinusDtK_BP_Jastrow_vec[j_Jastrow], walkerRightTemp, walkerRightTemp2);
                //     walkerRightTemp = walkerRightTemp2;
                // }else{
                //     cout<<"Error: UNKNOW metropolis2sInfo.KVorder[j_Jastrow]: "<<metropolis2sInfo.KVorder[j_Jastrow]<<endl;
                //     exit(1);
                // }
                // //
                // ////////////////////////////////////////////////////////////
            }
            else{
                cout<<"Error: UNKNOW metropolis2sInfo.variableName_vec[j_Jastrow]: "<<metropolis2sInfo.variableName_vec[j_Jastrow]<<endl;
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
            metropolis2sInfo.walkerRightInBlock[counter+1] = walkerRightTemp;
            metropolis2sInfo.logWeightRightInBlock[counter+1] = logWeight;
    }
    }
    //
    WalkerWalkerOperation_Jastrow2s walkerWalkerOperation;
    walkerWalkerOperation.set( metropolis2sInfo.walkerLeftInBlock[0], metropolis2sInfo.walkerRightInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize] );
    metropolis2sInfo.currentLogOverlap = walkerWalkerOperation.returnLogOverlap();
    //
}


void Metropolis2s::copyField(Metropolis2sInfo metropolisInfo_input)
{
    // metropolis2sInfo = metropolisInfo_input;
    // 
    metropolis2sInfo.BPMetroTimesliceBlockSize = metropolisInfo_input.BPMetroTimesliceBlockSize;

    metropolis2sInfo.inBlockIndex = metropolisInfo_input.inBlockIndex;
    metropolis2sInfo.currentLogOverlap = metropolisInfo_input.currentLogOverlap;

    metropolis2sInfo.auxiliaryFields = metropolisInfo_input.auxiliaryFields;
    // auxiliaryFields_BP = x.auxiliaryFields_BP;
    metropolis2sInfo.dynamicForceFields = metropolisInfo_input.dynamicForceFields;
    // dynamicForceFields_BP = x.dynamicForceFields_BP;

    metropolis2sInfo.walkerRightInBlock = metropolisInfo_input.walkerRightInBlock;
    metropolis2sInfo.walkerLeftInBlock = metropolisInfo_input.walkerLeftInBlock;

    metropolis2sInfo.logWeightRightInBlock = metropolisInfo_input.logWeightRightInBlock;
    metropolis2sInfo.logWeightLeftInBlock = metropolisInfo_input.logWeightLeftInBlock;
    metropolis2sInfo.overlapMatrixUp_inv = metropolisInfo_input.overlapMatrixUp_inv;
    metropolis2sInfo.overlapMatrixDn_inv = metropolisInfo_input.overlapMatrixDn_inv;

    metropolis2sInfo.Bup = metropolisInfo_input.Bup;
    metropolis2sInfo.Bdn = metropolisInfo_input.Bdn;
    /////////////////////////////////////////
    metropolisInfo_input.truncatedDup = metropolisInfo_input.truncatedDup;
    metropolisInfo_input.truncatedDdn = metropolisInfo_input.truncatedDdn;
    metropolisInfo_input.overlapMatrixUp = metropolisInfo_input.overlapMatrixUp;
    metropolisInfo_input.overlapMatrixDn = metropolisInfo_input.overlapMatrixDn;
    metropolisInfo_input.A0up = metropolisInfo_input.A0up;
    metropolisInfo_input.A0dn = metropolisInfo_input.A0dn;
    metropolisInfo_input.A0WRup = metropolisInfo_input.A0WRup;
    metropolisInfo_input.A0WRdn = metropolisInfo_input.A0WRdn;
    metropolisInfo_input.A1up = metropolisInfo_input.A1up;
    metropolisInfo_input.A1dn = metropolisInfo_input.A1dn;
    metropolisInfo_input.B0up = metropolisInfo_input.B0up;
    // metropolisInfo_input.B0dn = metropolisInfo_input.B0dn;
    metropolisInfo_input.B0WRup = metropolisInfo_input.B0WRup;
    metropolisInfo_input.B0WRdn = metropolisInfo_input.B0WRdn;
    metropolisInfo_input.C0up = metropolisInfo_input.C0up;
    // metropolisInfo_input.C0dn = metropolisInfo_input.C0dn;
    metropolisInfo_input.Dup = metropolisInfo_input.Dup;
    // metropolisInfo_input.Ddn = metropolisInfo_input.Ddn;
    metropolisInfo_input.globalFastInitialized = metropolisInfo_input.globalFastInitialized;
    metropolisInfo_input.globalFastUpdated = metropolisInfo_input.globalFastUpdated;
    /////////////////////////////////////////
}


void Metropolis2s::initialField_again(Walker2s &walkerRight)
{
    //
    Walker2s walkerRightTemp,walkerRightTemp2;
    TwoBodySample_Jastrow2s twoBodySample;
    // TwoBodySample_BP_Jastrow2s twoBodySample_BP;

    complex<double> logWeight(0,0);

    walkerRightTemp=walkerRight;
    walkerRightTemp2=walkerRight;

    metropolis2sInfo.walkerRightInBlock[0] = walkerRight;
    metropolis2sInfo.logWeightRightInBlock[0] = logWeight;
    //
    int counter=-1;
    for(int j_Jastrow=metropolis2sInfo.numOfJastrow-1; j_Jastrow >= 0; j_Jastrow--){
    for(int j=1-1; j<=metropolis2sInfo.JastrowSlice[j_Jastrow]-1; j++){
            counter ++;
            if(metropolis2sInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_HAFQMC_icf"){
                ////////////////////////////////////////////////////////////
                if( method.BPMetroForceType == "constForce" ){
                    // twoBodySample = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySampleFromAux_icf_fixedForce(metropolis2sInfo.auxiliaryFields[counter],metropolis2sInfo.dynamicForceFields[counter]);
                    twoBodySample = expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySampleFromAux2s(metropolis2sInfo.auxiliaryFields[counter]);
                    cout<<"j_Jastrow: "<<j_Jastrow<<" metropolis2sInfo.auxiliaryFields[counter]: "<<metropolis2sInfo.auxiliaryFields[counter]<<endl;
                }
                //
                if(metropolis2sInfo.KVorder[j_Jastrow]=="VK"){
                    twoBodySampleWalkerRightOperation.reset("fixedOrder", metropolis2sInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp2, walkerRightTemp);
                }else if(metropolis2sInfo.KVorder[j_Jastrow]=="K^daggerV"){
                    twoBodySampleWalkerRightOperation.reset("fixedOrder", metropolis2sInfo.JastrowExpM[j_Jastrow], 10e-8, 10);
                    twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp, walkerRightTemp2);
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp2, walkerRightTemp);
                }else if(metropolis2sInfo.KVorder[j_Jastrow]=="KVK"){
                    twoBodySampleWalkerRightOperation.reset("dynamicOrder");
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRightTemp2, walkerRightTemp);
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    walkerRightTemp = walkerRightTemp2;
                }else{
                    cout<<"Error: UNKNOW metropolis2sInfo.KVorder[j_Jastrow]: "<<metropolis2sInfo.KVorder[j_Jastrow]<<endl;
                    cout<<metropolis2sInfo.numOfJastrow<<endl;
                    cout<<metropolis2sInfo.KVorder.size()<<endl;
                    for(int temp_i=1-1; temp_i<=metropolis2sInfo.KVorder.size()-1; temp_i++ ){
                        cout<<metropolis2sInfo.KVorder[temp_i]<<endl;
                    }
                    exit(1);
                }
                //
                ////////////////////////////////////////////////////////////
            }
            else if(metropolis2sInfo.variableName_vec[j_Jastrow]=="generalHamiltonian_K_HAFQMC_icf"){
                ////////////////////////////////////////////////////////////
                //default aux 
                metropolis2sInfo.auxiliaryFields[counter] = expMinusDtV_Jastrow_vec->at(j_Jastrow).sampleAuxFromForce(metropolis2sInfo.constForce_Jastrow[j_Jastrow]);
                metropolis2sInfo.dynamicForceFields[counter]=metropolis2sInfo.constForce_Jastrow[j_Jastrow];
                //
                if(metropolis2sInfo.KVorder[j_Jastrow]=="VK"){
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    walkerRightTemp=walkerRightTemp2;
                }else if(metropolis2sInfo.KVorder[j_Jastrow]=="K^daggerV"){
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    walkerRightTemp=walkerRightTemp2;
                }else if(metropolis2sInfo.KVorder[j_Jastrow]=="KVK"){
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp, walkerRightTemp2);
                    oneBodyWalkerRightOperation.applyToRight(expMinusDtK_Jastrow_vec->at(j_Jastrow), walkerRightTemp2, walkerRightTemp);
                }else{
                    cout<<"Error: UNKNOW metropolis2sInfo.KVorder[j_Jastrow]: "<<metropolis2sInfo.KVorder[j_Jastrow]<<endl;
                    cout<<metropolis2sInfo.numOfJastrow<<endl;
                    cout<<metropolis2sInfo.KVorder.size()<<endl;
                    for(int temp_i=1-1; temp_i<=metropolis2sInfo.KVorder.size()-1; temp_i++ ){
                        cout<<metropolis2sInfo.KVorder[temp_i]<<endl;
                    }
                    exit(1);
                }
                //
                ////////////////////////////////////////////////////////////
            }
            else if(metropolis2sInfo.variableName_vec[j_Jastrow]=="HubbardSOC"){
                // ////////////////////////////////////////////////////////////
                // if(metropolis2sInfo.KVorder[j_Jastrow]=="KVK"){
                //     // twoBodySample_BP = jastrowProjector->expMinusDtV_BP_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAuxForce(metropolis2sInfo.auxiliaryFields_BP[counter],metropolis2sInfo.dynamicForceFields_BP[counter]);
                //     twoBodySample_BP = jastrowProjector->expMinusDtV_BP_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAux(metropolis2sInfo.auxiliaryFields_BP[counter]);
                //     oneBodyWalkerRightOperation_BP.applyToRight(jastrowProjector->expMinusDtK_BP_Jastrow_vec[j_Jastrow], walkerRightTemp, walkerRightTemp2);
                //     twoBodySampleWalkerRightOperation_BP.applyToRight(twoBodySample_BP, walkerRightTemp2, walkerRightTemp);
                //     oneBodyWalkerRightOperation_BP.applyToRight(jastrowProjector->expMinusDtK_BP_Jastrow_vec[j_Jastrow], walkerRightTemp, walkerRightTemp2);
                //     walkerRightTemp = walkerRightTemp2;
                // }else{
                //     cout<<"Error: UNKNOW metropolis2sInfo.KVorder[j_Jastrow]: "<<metropolis2sInfo.KVorder[j_Jastrow]<<endl;
                //     exit(1);
                // }
                // //
                // ////////////////////////////////////////////////////////////
            }
            else{
                cout<<"Error: UNKNOW metropolis2sInfo.variableName_vec[j_Jastrow]: "<<metropolis2sInfo.variableName_vec[j_Jastrow]<<endl;
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

            metropolis2sInfo.walkerRightInBlock[counter+1] = walkerRightTemp;
            metropolis2sInfo.logWeightRightInBlock[counter+1] = logWeight;
    }
    }
    WalkerWalkerOperation_Jastrow2s walkerWalkerOperation;
    walkerWalkerOperation.set( metropolis2sInfo.walkerLeftInBlock[0], metropolis2sInfo.walkerRightInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize] );
    metropolis2sInfo.currentLogOverlap = walkerWalkerOperation.returnLogOverlap();
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
//////////////////////////////////////////////////