//
// Created by boruoshihao on 1/15/17.
//

#include "../../include/Metropolis/metropolis.h"
#include "../../include/utils.h"

using namespace std;
using namespace tensor_hao;

Metropolis::Metropolis() { }
Metropolis::Metropolis(const Metropolis &x) { copy_deep(x); }
Metropolis::Metropolis(Metropolis &&x) { move_deep(x); }
Metropolis::~Metropolis() { }

Metropolis &Metropolis::operator=(const Metropolis &x) { copy_deep(x); return *this; }

Metropolis &Metropolis::operator=(Metropolis &&x) { move_deep(x); return *this; }

void Metropolis::initJastrowProjectorNullptr()
{
    jastrowProjector = nullptr;
}

void Metropolis::setExpMinusDt_KV_Jastrow_vec(std::vector<OneBody_Jastrow> &expMinusDtK_Jastrow_vec_, std::vector<TwoBody_Jastrow> &expMinusDtV_Jastrow_vec_)
{
    expMinusDtK_Jastrow_vec = &expMinusDtK_Jastrow_vec_;
    expMinusDtV_Jastrow_vec = &expMinusDtV_Jastrow_vec_;
}

void Metropolis::setJastrowProjector(JastrowProjector &jastrowProjector_)
{
    jastrowProjector = &jastrowProjector_;
}

void Metropolis::setJastrowProjectorExtended(JastrowProjector &jastrowProjectorExtended_)
{
    jastrowProjectorExtended = &jastrowProjectorExtended_;
}

void Metropolis::initialParameters(int L, int N, JastrowProjector &jastrowProjector_, std::vector<OneBody_Jastrow> &expMinusDtK_Jastrow_vec_, std::vector<TwoBody_Jastrow> &expMinusDtV_Jastrow_vec_)
{
    //each chain has independent method to hold different randomHaoInit
    method.read("afqmc_param_Metro");
    //
    randomHaoInit(method.seed, 1);
    if( method.seed != -1 ) randomHaoSave();
    //
    //set metropolisInfo to be distributed
    metropolisInfo.BPMetroTimesliceBlockSize=method.BPMetroTimesliceBlockSize;
    //
    setJastrowProjector(jastrowProjector_);
    setExpMinusDt_KV_Jastrow_vec(expMinusDtK_Jastrow_vec_, expMinusDtV_Jastrow_vec_);
    // 
    /////////////////////////////////////
    metropolisInfo.initial_JastrowProjectorRelated(jastrowProjector_);
    metropolisInfo.initial_localUpdate(0, 0);
    metropolisInfo.initial_globalFastUpdate(0, 0, 0);
    if(method.BPMetroUpdateType == "local"){
        metropolisInfo.initial_localUpdate(L, N);
    }else if(method.BPMetroUpdateType == "global_fast"){
        metropolisInfo.initial_globalFastUpdate(L, N, jastrowProjector_.model_Jastrow[0].getTruncatedD());
    }
    /////////////////////////////////////
}

void Metropolis::initialParametersTwoJastrow(int L, int N, JastrowProjector &jastrowProjector_, std::vector<OneBody_Jastrow> &expMinusDtK_Jastrow_vec_, std::vector<TwoBody_Jastrow> &expMinusDtV_Jastrow_vec_)
{
    //each chain has independent method to hold different randomHaoInit
    method.read("afqmc_param_Metro");
    //
    randomHaoInit(method.seed, 1);
    if( method.seed != -1 ) randomHaoSave();
    //
    //set metropolisInfo to be distributed
    metropolisInfo.BPMetroTimesliceBlockSize=2*method.BPMetroTimesliceBlockSize;
    //
    setJastrowProjector(jastrowProjector_);
    setExpMinusDt_KV_Jastrow_vec(expMinusDtK_Jastrow_vec_, expMinusDtV_Jastrow_vec_);
    // 
    metropolisInfo.initial_JastrowProjectorRelated(jastrowProjector_);
    metropolisInfo.initial_localUpdate(0, 0);
    metropolisInfo.initial_globalFastUpdate(0, 0, 0);
}

void Metropolis::extendMetroChainInfoToRight(MetropolisInfo metropolisInfo_input)
{
    ///////////////////////////
    //with addLength slices
    ///////////////////////////
    //ATTENTION: order
    metropolisInfo.extendMetroChainToRight(metropolisInfo_input);
}

const Walker& Metropolis::getWalkerLeftInBlock(int n) const
{
    return metropolisInfo.walkerLeftInBlock[n];  
}

const Walker& Metropolis::getWalkerRightInBlock(int n) const
{
    return metropolisInfo.walkerRightInBlock[n];  
}

const Walker& Metropolis::getWalkerLeftInBlockFinal() const
{
    return metropolisInfo.walkerLeftInBlock[metropolisInfo.BPMetroTimesliceBlockSize];  
}

size_t Metropolis::getBPMetroTimesliceBlockSize() const
{
    return metropolisInfo.BPMetroTimesliceBlockSize;  
}

Walker& Metropolis::walkerRightInBlockInitialRef()
{
    return metropolisInfo.walkerRightInBlock[0];  
}

complex<double>& Metropolis::logWeightRightInBlockInitialRef()
{
    return metropolisInfo.logWeightRightInBlock[0];  
}

/////////////////////////////////////////////
//update walkerRight in metroChain
/////////////////////////////////////////////
void Metropolis::updateWalkerRight(Walker walkerRight_input)
{
    walkerRightInBlockInitialRef() = walkerRight_input;
    Walker walkerLeftSD = getWalkerLeftMetro();
    WalkerWalkerOperation_Jastrow walkerWalkerOperation( walkerLeftSD, walkerRight_input );
    metropolisInfo.currentLogOverlap = walkerWalkerOperation.returnLogOverlap();
    ///////////////////////////////////
    // For local update
    ///////////////////////////////////
    if(method.BPMetroUpdateType == "local"){
        updateDirectOverlapMatrix_inv(walkerRight_input, metropolisInfo.overlapMatrix_inv); 
    }
    // 
    ///////////////////////////////////
    // For global fast update 
    ///////////////////////////////////
    if(method.BPMetroUpdateType == "global_fast"){
        metropolisInfo.globalFastUpdated = false;
        update_globalFast();
    }
}


void Metropolis::checkLocalUpdateData(Walker walkerRight_input)
{
    bool check = true;
    // 
    TensorHao<complex<double>, 2> overlapMatrix_inv_Temp;
    updateDirectOverlapMatrix_inv(walkerRight_input, overlapMatrix_inv_Temp);
    // 
    check = checkMatrixDiff(overlapMatrix_inv_Temp, metropolisInfo.overlapMatrix_inv, 10e-8);
    if(check != true){
        cout<<"Error in checkLocalUpdateData: overlapMatrix_inv"<<endl;
        exit(1);
    }
    //
    // if(MPIRank()==0){
    //     cout<<"==========================================================================="<<endl;
    //     cout<<"ATTENTION: In this code, fast update only support one Jastrow and 1 slice !"<<endl;
    //     cout<<"==========================================================================="<<endl;
    // }
    int j_Jastrow = jastrowProjector->expMinusDtV_Jastrow_vec.size()-1;
    check = checkMatrixDiff(metropolisInfo.B, expMinusDtV_Jastrow_vec->at(j_Jastrow).getTwoBodySampleFromAux(metropolisInfo.auxiliaryFields[0]).matrix, 10e-8);
    if(check != true){
        cout<<"Error in checkLocalUpdateData: B"<<endl;
        exit(1);
    }
}

/////////////////////////////////////////////
double Metropolis::getMemory() const
{
    return  metropolisInfo.getMemory(); 
}

int Metropolis::returnNbuf() const
{
    return  metropolisInfo.returnNbuf(); 
}

#ifdef MPI_HAO
void MPIBcast(Metropolis &buffer, int root, MPI_Comm const &comm)
{
    MPIBcast( buffer.metropolisInfo, root, comm );
}

void Metropolis::pack(vector<char> &buf, int &posit) const
{
    metropolisInfo.pack(buf, posit);
}

void Metropolis::unpack(const vector<char> &buf, int &posit)
{
    metropolisInfo.unpack(buf, posit);
}
#endif


void Metropolis::copy_deep(const Metropolis &x)
{

    metropolisInfo = x.metropolisInfo;

    /////////////////////////////////////////////
    jastrowProjector = x.jastrowProjector;
    //
    method = x.method;
    /////////////////////////////////////////////
}

void Metropolis::move_deep(Metropolis &x)
{
    
    metropolisInfo = move(x.metropolisInfo);

    /////////////////////////////////////////////
    jastrowProjector = x.jastrowProjector;
    //
    method = x.method;
    /////////////////////////////////////////////
}