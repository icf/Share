//
// Created by boruoshihao on 1/15/17.
//

#include "../../include/Metropolis2s/metropolis2s.h"
#include "../../include/utils.h"

using namespace std;
using namespace tensor_hao;

Metropolis2s::Metropolis2s() { }
Metropolis2s::Metropolis2s(const Metropolis2s &x) { copy_deep(x); }
Metropolis2s::Metropolis2s(Metropolis2s &&x) { move_deep(x); }
Metropolis2s::~Metropolis2s() { }

Metropolis2s &Metropolis2s::operator=(const Metropolis2s &x) { copy_deep(x); return *this; }

Metropolis2s &Metropolis2s::operator=(Metropolis2s &&x) { move_deep(x); return *this; }

void Metropolis2s::initJastrowProjectorNullptr()
{
    jastrowProjector = nullptr;
}

void Metropolis2s::setExpMinusDt_KV_Jastrow_vec(std::vector<OneBody_Jastrow2s> &expMinusDtK_Jastrow_vec_, std::vector<TwoBody_Jastrow2s> &expMinusDtV_Jastrow_vec_)
{
    expMinusDtK_Jastrow_vec = &expMinusDtK_Jastrow_vec_;
    expMinusDtV_Jastrow_vec = &expMinusDtV_Jastrow_vec_;
}

void Metropolis2s::setJastrowProjector(JastrowProjector2s &jastrowProjector_)
{
    jastrowProjector = &jastrowProjector_;
}

void Metropolis2s::setJastrowProjectorExtended(JastrowProjector2s &jastrowProjectorExtended_)
{
    jastrowProjectorExtended = &jastrowProjectorExtended_;
}

void Metropolis2s::initialParameters(int L, int Nup, int Ndn, JastrowProjector2s &jastrowProjector_, std::vector<OneBody_Jastrow2s> &expMinusDtK_Jastrow_vec_, std::vector<TwoBody_Jastrow2s> &expMinusDtV_Jastrow_vec_)
{
    //each chain has independent method to hold different randomHaoInit
    method.read("afqmc_param_Metro");
    //
    randomHaoInit(method.seed, 1);
    if( method.seed != -1 ) randomHaoSave();
    //
    //set metropolis2sInfo to be distributed
    metropolis2sInfo.BPMetroTimesliceBlockSize=method.BPMetroTimesliceBlockSize;
    //
    setJastrowProjector(jastrowProjector_);
    setExpMinusDt_KV_Jastrow_vec(expMinusDtK_Jastrow_vec_, expMinusDtV_Jastrow_vec_);
    // 
    /////////////////////////////////////
    metropolis2sInfo.initial_JastrowProjectorRelated(jastrowProjector_);
    metropolis2sInfo.initial_localUpdate(0, 0, 0);
    metropolis2sInfo.initial_globalFastUpdate(0, 0, 0, 0, 0);
    if(method.BPMetroUpdateType == "local"){
        metropolis2sInfo.initial_localUpdate(L, Nup, Ndn);
    }else if(method.BPMetroUpdateType == "global_fast"){
        metropolis2sInfo.initial_globalFastUpdate(L, Nup, Ndn, jastrowProjector_.model_Jastrow[0].getTruncatedDup(), jastrowProjector_.model_Jastrow[0].getTruncatedDdn());
    }
    /////////////////////////////////////
}

void Metropolis2s::initialParametersTwoJastrow(int L, int Nup, int Ndn, JastrowProjector2s &jastrowProjector_, std::vector<OneBody_Jastrow2s> &expMinusDtK_Jastrow_vec_, std::vector<TwoBody_Jastrow2s> &expMinusDtV_Jastrow_vec_)
{
    //each chain has independent method to hold different randomHaoInit
    method.read("afqmc_param_Metro");
    //
    randomHaoInit(method.seed, 1);
    if( method.seed != -1 ) randomHaoSave();
    //
    //set metropolis2sInfo to be distributed
    metropolis2sInfo.BPMetroTimesliceBlockSize=2*method.BPMetroTimesliceBlockSize;
    //
    setJastrowProjector(jastrowProjector_);
    setExpMinusDt_KV_Jastrow_vec(expMinusDtK_Jastrow_vec_, expMinusDtV_Jastrow_vec_);
    // 
    metropolis2sInfo.initial_JastrowProjectorRelated(jastrowProjector_);
    metropolis2sInfo.initial_localUpdate(0, 0, 0);
    metropolis2sInfo.initial_globalFastUpdate(0, 0, 0, 0, 0);
}

void Metropolis2s::extendMetroChainInfoToRight(Metropolis2sInfo metropolisInfo_input)
{
    ///////////////////////////
    //with addLength slices
    ///////////////////////////
    //ATTENTION: order
    metropolis2sInfo.extendMetroChainToRight(metropolisInfo_input);
}

const Walker2s& Metropolis2s::getWalkerLeftInBlock(int n) const
{
    return metropolis2sInfo.walkerLeftInBlock[n];  
}

const Walker2s& Metropolis2s::getWalkerRightInBlock(int n) const
{
    return metropolis2sInfo.walkerRightInBlock[n];  
}

const Walker2s& Metropolis2s::getWalkerLeftInBlockFinal() const
{
    return metropolis2sInfo.walkerLeftInBlock[metropolis2sInfo.BPMetroTimesliceBlockSize];  
}

size_t Metropolis2s::getBPMetroTimesliceBlockSize() const
{
    return metropolis2sInfo.BPMetroTimesliceBlockSize;  
}

Walker2s& Metropolis2s::walkerRightInBlockInitialRef()
{
    return metropolis2sInfo.walkerRightInBlock[0];  
}

complex<double>& Metropolis2s::logWeightRightInBlockInitialRef()
{
    return metropolis2sInfo.logWeightRightInBlock[0];  
}

/////////////////////////////////////////////
//update walkerRight in metroChain
/////////////////////////////////////////////
void Metropolis2s::updateWalkerRight(Walker2s walkerRight_input)
{
    walkerRightInBlockInitialRef() = walkerRight_input;
    Walker2s walkerLeftSD2s = getWalkerLeftMetro();
    WalkerWalkerOperation_Jastrow2s walkerWalkerOperation( walkerLeftSD2s, walkerRight_input );
    metropolis2sInfo.currentLogOverlap = walkerWalkerOperation.returnLogOverlap();
    ///////////////////////////////////
    // For local update
    ///////////////////////////////////
    if(method.BPMetroUpdateType == "local"){
        updateDirectOverlapMatrix_inv(walkerRight_input, metropolis2sInfo.overlapMatrixUp_inv, metropolis2sInfo.overlapMatrixDn_inv); 
    }
    // 
    ///////////////////////////////////
    // For global fast update 
    ///////////////////////////////////
    if(method.BPMetroUpdateType == "global_fast"){
        metropolis2sInfo.globalFastUpdated = false;
        update_globalFast();
    }
}


void Metropolis2s::checkLocalUpdateData(Walker2s walkerRight_input)
{
    bool check = true;
    // 
    TensorHao<complex<double>, 2> overlapMatrixUp_inv_Temp, overlapMatrixDn_inv_Temp;
    updateDirectOverlapMatrix_inv(walkerRight_input, overlapMatrixUp_inv_Temp, overlapMatrixDn_inv_Temp);
    // 
    check = checkMatrixDiff(overlapMatrixUp_inv_Temp, metropolis2sInfo.overlapMatrixUp_inv, 10e-8);
    if(check != true){
        cout<<"Error in checkLocalUpdateData: overlapMatrixUp_inv"<<endl;
        exit(1);
    }
    check = checkMatrixDiff(overlapMatrixDn_inv_Temp, metropolis2sInfo.overlapMatrixDn_inv, 10e-8);
    if(check != true){
        cout<<"Error in checkLocalUpdateData: overlapMatrixDn_inv"<<endl;
        exit(1);
    }
    //
    // if(MPIRank()==0){
    //     cout<<"==========================================================================="<<endl;
    //     cout<<"ATTENTION: In this code, fast update only support one Jastrow and 1 slice !"<<endl;
    //     cout<<"==========================================================================="<<endl;
    // }
    int j_Jastrow = jastrowProjector->expMinusDtV_Jastrow_vec.size()-1;
    check = checkMatrixDiff(metropolis2sInfo.Bup, expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySampleFromAux2s(metropolis2sInfo.auxiliaryFields[0]).matrixUp, 10e-8);
    if(check != true){
        cout<<"Error in checkLocalUpdateData: Bup"<<endl;
        exit(1);
    }
    check = checkMatrixDiff(metropolis2sInfo.Bdn, expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySampleFromAux2s(metropolis2sInfo.auxiliaryFields[0]).matrixDn, 10e-8);
    if(check != true){
        cout<<"Error in checkLocalUpdateData: Bdn"<<endl;
        exit(1);
    }
}

/////////////////////////////////////////////
double Metropolis2s::getMemory() const
{
    return  metropolis2sInfo.getMemory(); 
}

int Metropolis2s::returnNbuf() const
{
    return  metropolis2sInfo.returnNbuf(); 
}

#ifdef MPI_HAO
void MPIBcast(Metropolis2s &buffer, int root, MPI_Comm const &comm)
{
    MPIBcast( buffer.metropolis2sInfo, root, comm );
}

void Metropolis2s::pack(vector<char> &buf, int &posit) const
{
    metropolis2sInfo.pack(buf, posit);
}

void Metropolis2s::unpack(const vector<char> &buf, int &posit)
{
    metropolis2sInfo.unpack(buf, posit);
}
#endif


void Metropolis2s::copy_deep(const Metropolis2s &x)
{

    metropolis2sInfo = x.metropolis2sInfo;

    /////////////////////////////////////////////
    jastrowProjector = x.jastrowProjector;
    //
    method = x.method;
    /////////////////////////////////////////////
}

void Metropolis2s::move_deep(Metropolis2s &x)
{
    
    metropolis2sInfo = move(x.metropolis2sInfo);

    /////////////////////////////////////////////
    jastrowProjector = x.jastrowProjector;
    //
    method = x.method;
    /////////////////////////////////////////////
}