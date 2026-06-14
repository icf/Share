//
// Created by boruoshihao on 1/15/17.
//

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "../../include/Metropolis/metropolisInfo.h"

using namespace std;
using namespace tensor_hao;

MetropolisInfo::MetropolisInfo(){
    BPMetroTimesliceBlockSize = 0;
}

MetropolisInfo::MetropolisInfo(const MetropolisInfo &x) { copy_deep(x); }

MetropolisInfo::MetropolisInfo(MetropolisInfo &&x) { move_deep(x); }

MetropolisInfo::~MetropolisInfo() { }

MetropolisInfo &MetropolisInfo::operator=(const MetropolisInfo &x) { copy_deep(x); return *this; }

MetropolisInfo &MetropolisInfo::operator=(MetropolisInfo &&x) { move_deep(x); return *this; }

void MetropolisInfo::initial_JastrowProjectorRelated(JastrowProjector &jastrowProjector_){
    // ATTENTION: we won't distribute these data by MPI as all threads shares the same data
    // optimizing the visiting of Jastrow projector as this part is time consuming for MPI to visit same address
    variableName_vec = jastrowProjector_.variableName_vec;
    // 
    constForce_Jastrow = jastrowProjector_.constForce_Jastrow;
    KVorder = jastrowProjector_.KVorder;
    // 
    numOfJastrow = jastrowProjector_.numOfJastrow;
    JastrowSlice = jastrowProjector_.JastrowSlice;
    JastrowExpM = jastrowProjector_.JastrowExpM;
}
void MetropolisInfo::initial_localUpdate(int L, int N){

    overlapMatrix_inv.resize(N,N);

    B.resize(L,L);
}

void MetropolisInfo::initial_globalFastUpdate(int L, int N, int truncatedD_input){

    truncatedD = truncatedD_input;

    overlapMatrix.resize(N,N);

    A0.resize(N,L);

    A0WR.resize(N,N);

    A1.resize(N,truncatedD);

    B0.resize(truncatedD,L);

    B0WR.resize(truncatedD,N);

    C0.resize(truncatedD,truncatedD);

    D.resize(truncatedD,truncatedD);

    globalFastInitialized = false;
    globalFastUpdated = false;
}

void MetropolisInfo::readTwoJastrowAuxiliaryFields(int i )
{
    if(auxiliaryFields.size() != 2){
        cout<<"Eror: auxiliaryFields.size() != 2, "<<auxiliaryFields.size()<<endl;
        exit(1);
    }
    //auxiliaryFields for left bra
    auxiliaryFields[0].read("HAFQMC_AuxFields/HAFQMC_auxiliary_field_Left_sample_list_"+to_string(i)+".HAFQMCdat");
    //auxiliaryFields for right ket
    auxiliaryFields[1].read("HAFQMC_AuxFields/HAFQMC_auxiliary_field_Right_sample_list_"+to_string(i)+".HAFQMCdat");
}

void MetropolisInfo::readAuxiliaryFields(int i )
{
    if(auxiliaryFields.size() != 1){
        cout<<"Eror: auxiliaryFields.size() != 1, "<<auxiliaryFields.size()<<endl;
        exit(1);
    }
    //auxiliaryFields for left bra
    auxiliaryFields[0].read("HAFQMC_AuxFields/HAFQMC_auxiliary_field_Left_sample_list_"+to_string(i)+".HAFQMCdat");
}

void MetropolisInfo::takeLeftHalf()
{
    BPMetroTimesliceBlockSize = BPMetroTimesliceBlockSize/2;

    currentLogOverlap = currentLogOverlap;

    auxiliaryFields.erase(auxiliaryFields.begin(),auxiliaryFields.begin()+BPMetroTimesliceBlockSize);
    dynamicForceFields.erase(dynamicForceFields.begin(),dynamicForceFields.begin()+BPMetroTimesliceBlockSize);

    walkerRightInBlock.erase(walkerRightInBlock.begin(),walkerRightInBlock.begin()+BPMetroTimesliceBlockSize);
    walkerLeftInBlock.erase(walkerLeftInBlock.begin()+BPMetroTimesliceBlockSize+1,walkerLeftInBlock.begin()+BPMetroTimesliceBlockSize*2+1);

    logWeightRightInBlock.erase(logWeightRightInBlock.begin(),logWeightRightInBlock.begin()+BPMetroTimesliceBlockSize);
    logWeightLeftInBlock.erase(logWeightLeftInBlock.begin()+BPMetroTimesliceBlockSize+1,logWeightLeftInBlock.begin()+BPMetroTimesliceBlockSize*2+1);
}


void MetropolisInfo::extendMetroChainToRight( MetropolisInfo metropolisInfo_input)
{
    // BPMetroTimesliceBlockSize += metropolisInfo_input.BPMetroTimesliceBlockSize;
    // //
    // inBlockIndex = metropolisInfo_input.BPMetroTimesliceBlockSize;
    // currentLogOverlap = currentLogOverlap;
    // //
    // for(int i=1-1; i<=metropolisInfo_input.BPMetroTimesliceBlockSize-1; i++){
    //     auxiliaryFields.push_back(metropolisInfo_input.auxiliaryFields[metropolisInfo_input.BPMetroTimesliceBlockSize-1-i]);
    //     rotate(auxiliaryFields.rbegin(), auxiliaryFields.rbegin() + 1, auxiliaryFields.rend());
    //     dynamicForceFields.push_back(metropolisInfo_input.dynamicForceFields[metropolisInfo_input.BPMetroTimesliceBlockSize-1-i]);
    //     rotate(dynamicForceFields.rbegin(), dynamicForceFields.rbegin() + 1, dynamicForceFields.rend());
    //     //
    //     // auxiliaryFields_BP.push_back(metropolisInfo_input.auxiliaryFields_BP[metropolisInfo_input.BPMetroTimesliceBlockSize-1-i]);
    //     // rotate(auxiliaryFields_BP.rbegin(), auxiliaryFields_BP.rbegin() + 1, auxiliaryFields_BP.rend());
    //     // dynamicForceFields_BP.push_back(metropolisInfo_input.dynamicForceFields_BP[metropolisInfo_input.BPMetroTimesliceBlockSize-1-i]);
    //     // rotate(dynamicForceFields_BP.rbegin(), dynamicForceFields_BP.rbegin() + 1, dynamicForceFields_BP.rend());
    //     //
    //     walkerRightInBlock.push_back(metropolisInfo_input.walkerRightInBlock[metropolisInfo_input.BPMetroTimesliceBlockSize-1-i]);
    //     rotate(walkerRightInBlock.rbegin(), walkerRightInBlock.rbegin() + 1, walkerRightInBlock.rend());
    //     logWeightRightInBlock.push_back(metropolisInfo_input.logWeightRightInBlock[metropolisInfo_input.BPMetroTimesliceBlockSize-1-i]);
    //     rotate(logWeightRightInBlock.rbegin(), logWeightRightInBlock.rbegin() + 1, logWeightRightInBlock.rend());
    //     //
    //     walkerLeftInBlock.push_back(metropolisInfo_input.walkerLeftInBlock[i]);
    //     logWeightLeftInBlock.push_back(metropolisInfo_input.logWeightLeftInBlock[i]);
    // }
}

double MetropolisInfo::getMemory() const
{
    // int BPMetroTimesliceBlockSize;
    double mem = 4.0;
    // int inBlockIndex;
    mem += 4.0;
    // complex<double> currentLogOverlap;
    mem += 16.0;

    for(size_t i=1-1; i<=auxiliaryFields.size()-1; i++){
        mem += 16*auxiliaryFields[i].size();
    }
    for(size_t i=1-1; i<=dynamicForceFields.size()-1; i++){
        mem += 16*dynamicForceFields[i].size();
    }

    for(size_t i=1-1; i<=walkerRightInBlock.size()-1; i++){
        mem += walkerRightInBlock[i].getMemory();
    }
    for(size_t i=1-1; i<=walkerLeftInBlock.size()-1; i++){
        mem += walkerLeftInBlock[i].getMemory();
    }

    mem += 16*logWeightRightInBlock.size();
    mem += 16*logWeightLeftInBlock.size();

    mem += overlapMatrix_inv.getMemory();

    mem += B.getMemory();

    /////////////////////////////////////////
    mem += overlapMatrix.getMemory();
    mem += A0.getMemory();
    mem += A0WR.getMemory();
    mem += A1.getMemory();
    mem += B0.getMemory();
    mem += B0WR.getMemory();
    mem += C0.getMemory();
    mem += D.getMemory();

    mem += 1;
    mem += 1;
    /////////////////////////////////////////
    mem += variableName_vec.size() * variableName_vec[0].size() * 16;

    mem += constForce_Jastrow.size() * constForce_Jastrow[0].size() * 16;
    mem += KVorder.size() * KVorder[0].size() * 16;

    mem += numOfJastrow;
    mem += JastrowSlice.size() * 4;
    mem += JastrowExpM.size() * 4;
    /////////////////////////////////////////

    return  mem; 
}

int MetropolisInfo::returnNbuf() const
{
    // int BPMetroTimesliceBlockSize;
    int Nbuf = 4;
    // int inBlockIndex;
    Nbuf += 4;
    // complex<double> currentLogOverlap;
    Nbuf += 16;

    for(size_t i=1-1; i<=auxiliaryFields.size()-1; i++){
        Nbuf += 16*auxiliaryFields[i].size();
    }
    for(size_t i=1-1; i<=dynamicForceFields.size()-1; i++){
        Nbuf += 16*dynamicForceFields[i].size();
    }

    for(size_t i=1-1; i<=walkerRightInBlock.size()-1; i++){
        Nbuf += walkerRightInBlock[i].returnNbuf();
    }
    for(size_t i=1-1; i<=walkerLeftInBlock.size()-1; i++){
        Nbuf += walkerLeftInBlock[i].returnNbuf();
    }

    Nbuf += 16*logWeightRightInBlock.size();
    Nbuf += 16*logWeightLeftInBlock.size();

    Nbuf += 16*overlapMatrix_inv.size();

    Nbuf += 16*B.size();

    /////////////////////////////////////////
    Nbuf += 4;
    Nbuf += 16*overlapMatrix.size();
    Nbuf += 16*A0.size();
    Nbuf += 16*A0WR.size();
    Nbuf += 16*A1.size();
    Nbuf += 16*B0.size();
    Nbuf += 16*B0WR.size();
    Nbuf += 16*C0.size();
    Nbuf += 16*D.size();
    Nbuf += 1;
    Nbuf += 1;
    /////////////////////////////////////////

    return  Nbuf; 
}

#ifdef MPI_HAO
void MPIBcast(MetropolisInfo &buffer, int root, MPI_Comm const &comm)
{
    MPIBcast( buffer.BPMetroTimesliceBlockSize, root, comm );
    MPIBcast( buffer.inBlockIndex, root, comm );
    MPIBcast( buffer.currentLogOverlap, root, comm );
    for(size_t i = 1-1; i <= buffer.auxiliaryFields.size()-1; ++i){
        MPIBcast( buffer.auxiliaryFields[i], root, comm );
    }
    for(size_t i = 1-1; i <= buffer.dynamicForceFields.size()-1; ++i){
        MPIBcast( buffer.dynamicForceFields[i], root, comm );
    }

    for(size_t i = 1-1; i <= buffer.walkerRightInBlock.size()-1; ++i){
        MPIBcast( buffer.walkerRightInBlock[i], root, comm );
    }
    for(size_t i = 1-1; i <= buffer.walkerLeftInBlock.size()-1; ++i){
        MPIBcast( buffer.walkerLeftInBlock[i], root, comm );
    }

    for(size_t i = 1-1; i <= buffer.logWeightRightInBlock.size()-1; ++i){
        MPIBcast( buffer.logWeightRightInBlock, root, comm );
    }
    for(size_t i = 1-1; i <= buffer.logWeightLeftInBlock.size()-1; ++i){
        MPIBcast( buffer.logWeightLeftInBlock, root, comm );
    }

    MPIBcast( buffer.overlapMatrix_inv, root, comm );

    MPIBcast( buffer.B, root, comm );

    /////////////////////////////////////////
    MPIBcast( buffer.truncatedD, root, comm );
    MPIBcast( buffer.overlapMatrix, root, comm );
    MPIBcast( buffer.A0, root, comm );
    MPIBcast( buffer.A0WR, root, comm );
    MPIBcast( buffer.A1, root, comm );
    MPIBcast( buffer.B0, root, comm );
    MPIBcast( buffer.B0WR, root, comm );
    MPIBcast( buffer.C0, root, comm );
    MPIBcast( buffer.D, root, comm );
    MPIBcast( buffer.globalFastInitialized, root, comm );
    MPIBcast( buffer.globalFastUpdated, root, comm );
    /////////////////////////////////////////
}

void MetropolisInfo::pack(vector<char> &buf, int &posit) const     
{    
    MPI_Pack(&BPMetroTimesliceBlockSize, 1, MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    //
    MPI_Pack(&inBlockIndex, 1, MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(&currentLogOverlap, 1, MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    //
    for(size_t i=1-1; i<=auxiliaryFields.size()-1; i++){
        MPI_Pack(auxiliaryFields[i].data(), auxiliaryFields[i].size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    }
    for(size_t i=1-1; i<=dynamicForceFields.size()-1; i++){
        MPI_Pack(dynamicForceFields[i].data(), dynamicForceFields[i].size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    }
    //
    for(size_t i=1-1; i<=walkerRightInBlock.size()-1; i++){
        walkerRightInBlock[i].pack(buf, posit);
    }
    for(size_t i=1-1; i<=walkerLeftInBlock.size()-1; i++){
        walkerLeftInBlock[i].pack(buf, posit);
    }
    //
    for(size_t i=1-1; i<=logWeightRightInBlock.size()-1; i++){
        MPI_Pack(&logWeightRightInBlock[i], 1, MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    }
    for(size_t i=1-1; i<=logWeightLeftInBlock.size()-1; i++){
        MPI_Pack(&logWeightLeftInBlock[i], 1, MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    }
    //
    // cout<<MPIRank()<<" pack: "<<overlapMatrixUp_inv.size()<<" "<<overlapMatrixDn_inv.size()<<" "<<Bup.size()<<" "<<Bdn.size()<<endl;
    // 
    MPI_Pack(overlapMatrix_inv.data(), overlapMatrix_inv.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    //
    MPI_Pack(B.data(), B.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);

    /////////////////////////////////////////
    MPI_Pack(&truncatedD, 1, MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(overlapMatrix.data(), overlapMatrix.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(A0.data(), A0.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(A0WR.data(), A0WR.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(A1.data(), A1.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(B0.data(), B0.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(B0WR.data(), B0WR.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(C0.data(), C0.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(D.data(), D.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(&globalFastInitialized, 1, MPI_CXX_BOOL, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(&globalFastUpdated, 1, MPI_CXX_BOOL, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    /////////////////////////////////////////
}

void MetropolisInfo::unpack(const vector<char> &buf, int &posit)
{
    MPI_Unpack(buf.data(), buf.size(), &posit, &BPMetroTimesliceBlockSize,1, MPI_INT, MPI_COMM_WORLD);
    //
    MPI_Unpack(buf.data(), buf.size(), &posit, &inBlockIndex,1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, &currentLogOverlap,1, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    //
    for(size_t i=1-1; i<=auxiliaryFields.size()-1; i++){
        MPI_Unpack(buf.data(), buf.size(), &posit, auxiliaryFields[i].data(), auxiliaryFields[i].size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    }
    for(size_t i=1-1; i<=dynamicForceFields.size()-1; i++){
        MPI_Unpack(buf.data(), buf.size(), &posit, dynamicForceFields[i].data(), dynamicForceFields[i].size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    }
    //
    for(size_t i=1-1; i<=walkerRightInBlock.size()-1; i++){
        walkerRightInBlock[i].unpack(buf, posit);
    }
    for(size_t i=1-1; i<=walkerLeftInBlock.size()-1; i++){
        walkerLeftInBlock[i].unpack(buf, posit);
    }
    //
    for(size_t i=1-1; i<=logWeightRightInBlock.size()-1; i++){
        MPI_Unpack(buf.data(), buf.size(), &posit, &logWeightRightInBlock[i],1, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    }
    for(size_t i=1-1; i<=logWeightLeftInBlock.size()-1; i++){
        MPI_Unpack(buf.data(), buf.size(), &posit, &logWeightLeftInBlock[i],1, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    }

    //
    // cout<<MPIRank()<<" unpack: "<<overlapMatrixUp_inv.size()<<" "<<overlapMatrixDn_inv.size()<<" "<<Bup.size()<<" "<<Bdn.size()<<endl;
    MPI_Unpack(buf.data(), buf.size(), &posit, overlapMatrix_inv.data(), overlapMatrix_inv.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);

    MPI_Unpack(buf.data(), buf.size(), &posit, B.data(), B.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);

    /////////////////////////////////////////
    MPI_Unpack(buf.data(), buf.size(), &posit, &truncatedD,1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, overlapMatrix.data(), overlapMatrix.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, A0.data(), A0.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, A0WR.data(), A0WR.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, A1.data(), A1.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, B0.data(), B0.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, B0WR.data(), B0WR.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, C0.data(), C0.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, D.data(), D.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, &globalFastInitialized,1, MPI_CXX_BOOL, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, &globalFastUpdated,1, MPI_CXX_BOOL, MPI_COMM_WORLD);
    /////////////////////////////////////////
}
#endif


void MetropolisInfo::copy_deep(const MetropolisInfo &x)
{
    BPMetroTimesliceBlockSize = x.BPMetroTimesliceBlockSize;

    inBlockIndex = x.inBlockIndex;
    currentLogOverlap = x.currentLogOverlap;

    auxiliaryFields = x.auxiliaryFields;
    dynamicForceFields = x.dynamicForceFields;

    walkerRightInBlock = x.walkerRightInBlock;
    walkerLeftInBlock = x.walkerLeftInBlock;

    logWeightRightInBlock = x.logWeightRightInBlock;
    logWeightLeftInBlock = x.logWeightLeftInBlock;

    overlapMatrix_inv = x.overlapMatrix_inv;

    B = x.B;

    /////////////////////////////////////////
    truncatedD = x.truncatedD;
    overlapMatrix = x.overlapMatrix;
    A0 = x.A0;
    A0WR = x.A0WR;
    A1 = x.A1;
    B0 = x.B0;
    B0WR = x.B0WR;
    C0 = x.C0;
    D = x.D;
    globalFastInitialized = x.globalFastInitialized;
    globalFastUpdated = x.globalFastUpdated;
    /////////////////////////////////////////
    variableName_vec = x.variableName_vec;
    // 
    constForce_Jastrow = x.constForce_Jastrow;
    KVorder = x.KVorder;
    // 
    numOfJastrow = x.numOfJastrow;
    JastrowSlice = x.JastrowSlice;
    JastrowExpM = x.JastrowExpM;
}

void MetropolisInfo::move_deep(MetropolisInfo &x)
{
    BPMetroTimesliceBlockSize = x.BPMetroTimesliceBlockSize;

    inBlockIndex = x.inBlockIndex;
    currentLogOverlap = x.currentLogOverlap;

    auxiliaryFields = move(x.auxiliaryFields);
    dynamicForceFields = move(x.dynamicForceFields);

    walkerRightInBlock = move(x.walkerRightInBlock);
    walkerLeftInBlock = move(x.walkerLeftInBlock);

    logWeightRightInBlock = move(x.logWeightRightInBlock);
    logWeightLeftInBlock = move(x.logWeightLeftInBlock);

    overlapMatrix_inv = move(x.overlapMatrix_inv);
    
    B = move(x.B);

    /////////////////////////////////////////
    truncatedD = x.truncatedD;
    overlapMatrix = move(x.overlapMatrix);
    A0 = move(x.A0);
    A0WR = move(x.A0WR);
    A1 = move(x.A1);
    B0 = move(x.B0);
    B0WR = move(x.B0WR);
    C0 = move(x.C0);
    // 
    D = move(x.D);
    globalFastInitialized = x.globalFastInitialized;
    globalFastUpdated = x.globalFastUpdated;
    /////////////////////////////////////////
    variableName_vec = x.variableName_vec;
    // 
    constForce_Jastrow = x.constForce_Jastrow;
    KVorder = x.KVorder;
    // 
    numOfJastrow = x.numOfJastrow;
    JastrowSlice = x.JastrowSlice;
    JastrowExpM = x.JastrowExpM;
}



