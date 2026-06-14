//

#include "../../include/Metropolis/JastrowProjector.h"
#include "afqmclab.h"

using namespace std;
using namespace tensor_hao;

JastrowProjector::JastrowProjector() {
}

JastrowProjector::JastrowProjector(JastrowProjector &x) { copy_deep(x); }
JastrowProjector &JastrowProjector::operator=(JastrowProjector &x) { return *this; }

JastrowProjector::~JastrowProjector() { }


void JastrowProjector::initialParameters(MetropolisMethod method)
{
    numOfJastrow = method.numOfJastrow;
    variableName_vec=method.JastrowName;
    JastrowSlice = method.JastrowSlice;
    JastrowExpM = method.JastrowExpM;
    /////////////////////////
    expMinusDtV_Jastrow_vec.resize(numOfJastrow);
    // expMinusDtV_BP_Jastrow_vec.resize(numOfJastrow);
    expMinusDtK_Jastrow_vec.resize(numOfJastrow);
    // expMinusDtK_BP_Jastrow_vec.resize(numOfJastrow);
    //
    model_Jastrow.resize(numOfJastrow);
    // model_BP_Jastrow.resize(numOfJastrow);
    constForce_Jastrow.resize(numOfJastrow);
    // constForce_BP_Jastrow.resize(numOfJastrow);
    /////////////////////////
    for(int jastrow=1-1; jastrow<=numOfJastrow-1; jastrow++){
        if(getVariableName(jastrow)=="generalHamiltonian_HAFQMC_icf"){
            const string stringTemp="model_Jastrow_param_"+to_string(jastrow);
            model_Jastrow[jastrow].read_conj(stringTemp);
            //
            //////////////////////////////////////////////////////////////////////////////////////
            expMinusDtV_Jastrow_vec[jastrow] = model_Jastrow[jastrow].returnExpMinusAlphaV_daggerSqrtDt( 1 , true);
            //////////////////////////////////////////////////////////////////////////////////////
            const string stringTemp2="constForce_param_"+to_string(jastrow);
            constForce_Jastrow[jastrow] = expMinusDtV_Jastrow_vec[jastrow].readForce(stringTemp2);
            //////////////////////////////////////////////////////////////////////////////////////
            expMinusDtK_Jastrow_vec[jastrow]= model_Jastrow[jastrow].returnExpMinusAlphaK_nonHermitian(  1, "fixedOrder", JastrowExpM[jastrow], 10e-8, 0  );
            //////////////////////////////////////////////////////////////////////////////////////
        }
        else if(getVariableName(jastrow)=="generalHamiltonian_K_HAFQMC_icf"){
            const string stringTemp="model_Jastrow_param_"+to_string(jastrow);
            model_Jastrow[jastrow].read_conj(stringTemp);
            //
            //////////////////////////////////////////////////////////////////////////////////////
            expMinusDtV_Jastrow_vec[jastrow] = model_Jastrow[jastrow].returnExpMinusAlphaV_daggerSqrtDt( 1 , true);
            //////////////////////////////////////////////////////////////////////////////////////
            const string stringTemp2="constForce_param_"+to_string(jastrow);
            constForce_Jastrow[jastrow] = expMinusDtV_Jastrow_vec[jastrow].readForce(stringTemp2);
            //////////////////////////////////////////////////////////////////////////////////////
            expMinusDtK_Jastrow_vec[jastrow]= model_Jastrow[jastrow].returnExpMinusAlphaK_nonHermitian(  1, "fixedOrder", JastrowExpM[jastrow], 10e-8, 0  );
            //////////////////////////////////////////////////////////////////////////////////////
        }
        else{
            cout<<"Error: UNKNOW getVariableName: "<<getVariableName(jastrow)<<endl;
            exit(1);
        }
    }
    //
    KVorder.resize( numOfJastrow );
    //
    for(int i=1-1; i<=numOfJastrow-1; i++){
        // KVorder[i]="K^daggerV";
        KVorder[i]="VK";
    }
    //
    if(MPIRank()==0){
        cout<<"For AFQMC trial, we take:"<<endl;
        string nameTemp;
        nameTemp="<phi0|";
        for(int i=1-1; i<=numOfJastrow-1; i++){
            nameTemp += KVorder[i];
        }
        nameTemp += "|AFQMC>";
        cout<<nameTemp<<endl;
        cout<<"================================"<<endl;
    }
}

void JastrowProjector::initialParametersTwoJastrow(MetropolisMethod method)
{
    //////////////////////////////////////////////////////////////
    numOfJastrow = 2*method.numOfJastrow;
    /////////////////////////
    JastrowSlice=method.JastrowSlice;
    for(int i=1-1; i<=method.numOfJastrow-1; i++){
        JastrowSlice.push_back(method.JastrowSlice[method.numOfJastrow-1-i]);
    }
    /////////////////////////
    expMinusDtV_Jastrow_vec.resize(numOfJastrow);
    // expMinusDtV_BP_Jastrow_vec.resize(numOfJastrow);
    expMinusDtK_Jastrow_vec.resize(numOfJastrow);
    // expMinusDtK_BP_Jastrow_vec.resize(numOfJastrow);
    //
    model_Jastrow.resize(numOfJastrow);
    // model_BP_Jastrow.resize(numOfJastrow);
    constForce_Jastrow.resize(numOfJastrow);
    // constForce_BP_Jastrow.resize(numOfJastrow);
    //
    vector<string> listTemp = method.JastrowName;
    for(int i=1-1; i<=numOfJastrow/2-1; i++){
        listTemp.push_back(method.JastrowName[numOfJastrow/2-1-i]);
    }
    method.JastrowName=listTemp;
    variableName_vec=method.JastrowName;
    /////////////////////////
    JastrowExpM=method.JastrowExpM;
    for(int i=1-1; i<=method.numOfJastrow-1; i++){
        JastrowExpM.push_back(method.JastrowExpM[method.numOfJastrow-1-i]);
    }
    /////////////////////////
    //<SD|J^\dagger_0 J^\dagger_1 ..
    /////////////////////////
    for(int jastrow=1-1; jastrow<=numOfJastrow/2-1; jastrow++){
        if(getVariableName(jastrow)=="generalHamiltonian_HAFQMC_icf"){
            const string stringTemp="model_Jastrow_param_"+to_string(jastrow);
            model_Jastrow[jastrow].read_conj(stringTemp);
            //
            expMinusDtV_Jastrow_vec[jastrow] = model_Jastrow[jastrow].returnExpMinusAlphaV_daggerSqrtDt( 1.0 , true);
            //
            const string stringTemp2="constForce_param_"+to_string(jastrow);
            constForce_Jastrow[jastrow] = expMinusDtV_Jastrow_vec[jastrow].readForce(stringTemp2);
            expMinusDtK_Jastrow_vec[jastrow]= model_Jastrow[jastrow].returnExpMinusAlphaK_nonHermitian(  1.0, "fixedOrder", JastrowExpM[jastrow], 10e-8, 0      );
            //
        }
        else if(getVariableName(jastrow)=="generalHamiltonian_K_HAFQMC_icf"){
            const string stringTemp="model_Jastrow_param_"+to_string(jastrow);
            model_Jastrow[jastrow].read_conj(stringTemp);
            //
            expMinusDtV_Jastrow_vec[jastrow] = model_Jastrow[jastrow].returnExpMinusAlphaV_daggerSqrtDt( 1.0 , true);
            //
            //
            const string stringTemp2="constForce_param_"+to_string(jastrow);
            constForce_Jastrow[jastrow] = expMinusDtV_Jastrow_vec[jastrow].readForce(stringTemp2);
            expMinusDtK_Jastrow_vec[jastrow]= model_Jastrow[jastrow].returnExpMinusAlphaK_nonHermitian(  1.0, "fixedOrder", JastrowExpM[jastrow], 10e-8, 0      );
            //
        }
        else{
            cout<<"Error: UNKNOW getVariableName: "<<getVariableName(jastrow)<<endl;
            exit(1);
        }
    }
    //
    /////////////////////////
    //... J_1 J_0|SD>
    /////////////////////////
    for(int jastrow=numOfJastrow/2; jastrow<=numOfJastrow-1; jastrow++){
        if(getVariableName(jastrow)=="generalHamiltonian_HAFQMC_icf"){
            const string stringTemp="model_Jastrow_param_"+to_string(numOfJastrow-1-jastrow);
            model_Jastrow[jastrow].read(stringTemp);
            //
            expMinusDtV_Jastrow_vec[jastrow] = model_Jastrow[jastrow].returnExpMinusAlphaV( 1.0 , true);
            const string stringTemp2="constForce_param_"+to_string(numOfJastrow-1-jastrow);
            constForce_Jastrow[jastrow] = expMinusDtV_Jastrow_vec[jastrow].readForce(stringTemp2);
            expMinusDtK_Jastrow_vec[jastrow]= model_Jastrow[jastrow].returnExpMinusAlphaK_nonHermitian(  1.0  , "fixedOrder", JastrowExpM[jastrow], 10e-8, 0 );
        }
        else if(getVariableName(jastrow)=="generalHamiltonian_K_HAFQMC_icf"){
            const string stringTemp="model_Jastrow_param_"+to_string(numOfJastrow-1-jastrow);
            model_Jastrow[jastrow].read(stringTemp);
            //
            expMinusDtV_Jastrow_vec[jastrow] = model_Jastrow[jastrow].returnExpMinusAlphaV( 1.0 , true);
            const string stringTemp2="constForce_param_"+to_string(numOfJastrow-1-jastrow);
            constForce_Jastrow[jastrow] = expMinusDtV_Jastrow_vec[jastrow].readForce(stringTemp2);
            expMinusDtK_Jastrow_vec[jastrow]= model_Jastrow[jastrow].returnExpMinusAlphaK_nonHermitian(  1.0 , "fixedOrder", JastrowExpM[jastrow], 10e-8, 0 );
        }
        else if(getVariableName(jastrow)=="AuxMatrix"){
            // const string stringTemp="model_Jastrow_param_"+to_string(numOfJastrow-1-jastrow);
            // //////////////////////////////////////////////////////////////////////////////////////
            // expMinusDtV_AM_Jastrow_vec[jastrow].readModel(stringTemp);
            // //////////////////////////////////////////////////////////////////////////////////////
            // TwoBodyForce_AM_Jastrow2s aux_AM_temp(expMinusDtV_AM_Jastrow_vec[jastrow].getNumberOfAuxMatrix());
            // constForce_AM_Jastrow[jastrow]=aux_AM_temp;
            // //////////////////////////////////////////////////////////////////////////////////////
            // Hop_AuxMatrix expMinusDtK_AM_Jastrow_Temp(stringTemp);
            // // expMinusDtK_AM_Jastrow_vec[jastrow]=expMinusDtK_AM_Jastrow_Temp.getHop("fixedOrder", JastrowExpM[jastrow], 10e-8, 0 );
            // expMinusDtK_AM_Jastrow_vec[jastrow]=expMinusDtK_AM_Jastrow_Temp.getHop();
            // //////////////////////////////////////////////////////////////////////////////////////
        }
        else{
            cout<<"Error: UNKNOW getVariableName: "<<getVariableName(jastrow)<<endl;
            exit(1);
        }
    }
    //
    KVorder.resize( numOfJastrow );
    //
    for(int i=1-1; i<=numOfJastrow/2-1; i++){
        // KVorder[i]="K^daggerV";
        // KVorder[i+numOfJastrow/2]="VK";
        KVorder[i]="VK";
        KVorder[i+numOfJastrow/2]="K^daggerV";
    }
    //
    if(MPIRank()==0){
        cout<<"To estimate trial energy, we take:"<<endl;
        string nameTemp;
        nameTemp="<phi0|";
        for(int i=1-1; i<=numOfJastrow-1; i++){
            nameTemp += KVorder[i];
        }
        nameTemp +="|phi0>";
        cout<<nameTemp<<endl;
        cout<<"================================"<<endl;
    }
    //////////////////////////////////////////////////////////////

}

string JastrowProjector::getVariableName( int i_temp)
{
    return variableName_vec[i_temp];
}

int JastrowProjector::getVariableSite( int i_temp)
{
    int i=i_temp;
    return i;
}

void JastrowProjector::extendMetroChainToRight( JastrowProjector jastrowProjector_input)
{
    // numOfJastrow += jastrowProjector_input.numOfJastrow;
    // for(int i=1-1; i<=jastrowProjector_input.numOfJastrow-1; i++){
    //     variableName_vec.push_back(jastrowProjector_input.variableName_vec[i]);
    //     model_Jastrow.push_back(jastrowProjector_input.model_Jastrow[i]);
    //     // model_BP_Jastrow.push_back(jastrowProjector_input.model_BP_Jastrow[i]);
    //     expMinusDtV_Jastrow_vec.push_back(jastrowProjector_input.expMinusDtV_Jastrow_vec[i]);
    //     // expMinusDtV_BP_Jastrow_vec.push_back(jastrowProjector_input.expMinusDtV_BP_Jastrow_vec[i]);
    //     expMinusDtK_Jastrow_vec.push_back(jastrowProjector_input.expMinusDtK_Jastrow_vec[i]);
    //     // expMinusDtK_BP_Jastrow_vec.push_back(jastrowProjector_input.expMinusDtK_BP_Jastrow_vec[i]);
    //     constForce_Jastrow.push_back(jastrowProjector_input.constForce_Jastrow[i]);
    //     // constForce_BP_Jastrow.push_back(jastrowProjector_input.constForce_BP_Jastrow[i]);
    //     KVorder.push_back(jastrowProjector_input.KVorder[i]);
    //     //
    //     JastrowSlice.push_back(jastrowProjector_input.JastrowSlice[i]);
    //     JastrowExpM.push_back(jastrowProjector_input.JastrowExpM[i]);
    // }
}


void JastrowProjector::copy_deep(JastrowProjector &x)
{
    variableName_vec = x.variableName_vec;
    model_Jastrow = x.model_Jastrow;
    // model_BP_Jastrow = x.model_BP_Jastrow;
    expMinusDtV_Jastrow_vec = x.expMinusDtV_Jastrow_vec;
    // expMinusDtV_BP_Jastrow_vec = x.expMinusDtV_BP_Jastrow_vec;
    expMinusDtK_Jastrow_vec = x.expMinusDtK_Jastrow_vec;
    // expMinusDtK_BP_Jastrow_vec = x.expMinusDtK_BP_Jastrow_vec;
    constForce_Jastrow = x.constForce_Jastrow;
    // constForce_BP_Jastrow = x.constForce_BP_Jastrow;
    KVorder = x.KVorder;
    //
    numOfJastrow = x.numOfJastrow;
    JastrowSlice = x.JastrowSlice;
    JastrowExpM = x.JastrowExpM;
}




