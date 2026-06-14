//
// Created by boruoshihao on 1/15/17.
//

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "../../include/Metropolis2s/metropolis2sInfo.h"

using namespace std;
using namespace tensor_hao;

Metropolis2sInfo::Metropolis2sInfo(){
    BPMetroTimesliceBlockSize = 0;
}

Metropolis2sInfo::Metropolis2sInfo(const Metropolis2sInfo &x) { copy_deep(x); }

Metropolis2sInfo::Metropolis2sInfo(Metropolis2sInfo &&x) { move_deep(x); }

Metropolis2sInfo::~Metropolis2sInfo() { }

Metropolis2sInfo &Metropolis2sInfo::operator=(const Metropolis2sInfo &x) { copy_deep(x); return *this; }

Metropolis2sInfo &Metropolis2sInfo::operator=(Metropolis2sInfo &&x) { move_deep(x); return *this; }

void Metropolis2sInfo::initial_JastrowProjectorRelated(JastrowProjector2s &jastrowProjector_){
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
void Metropolis2sInfo::initial_localUpdate(int L, int Nup, int Ndn){

    overlapMatrixUp_inv.resize(Nup,Nup);
    overlapMatrixDn_inv.resize(Ndn,Ndn);

    Bup.resize(L,L);
    Bdn.resize(L,L);
}

void Metropolis2sInfo::initial_globalFastUpdate(int L, int Nup, int Ndn, int truncatedDup_input, int truncatedDdn_input){

    truncatedDup = truncatedDup_input;
    truncatedDdn = truncatedDdn_input;

    if(truncatedDup != truncatedDdn){
        cout<<"Error: Metropolis2sInfo::initial_globalFastUpdate truncatedDup != truncatedDdn: "<<truncatedDup<<"  "<<truncatedDdn<<endl;
        exit(1);
    }

    overlapMatrixUp.resize(Nup,Nup);
    overlapMatrixDn.resize(Ndn,Ndn);

    A0up.resize(Nup,L);
    A0dn.resize(Ndn,L);

    A0WRup.resize(Nup,Nup);
    A0WRdn.resize(Ndn,Ndn);

    A1up.resize(Nup,truncatedDup);
    A1dn.resize(Ndn,truncatedDdn);

    B0up.resize(truncatedDup,L);
    // B0dn.resize(truncatedDdn,L);

    B0WRup.resize(truncatedDup,Nup);
    B0WRdn.resize(truncatedDdn,Ndn);

    C0up.resize(truncatedDup,truncatedDup);
    // C0dn.resize(truncatedDdn,truncatedDdn);

    Dup.resize(truncatedDup,truncatedDup);
    // Ddn.resize(truncatedDdn,truncatedDdn);

    globalFastInitialized = false;
    globalFastUpdated = false;
}

void Metropolis2sInfo::readTwoJastrowAuxiliaryFields(int i )
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

void Metropolis2sInfo::readAuxiliaryFields(int i )
{
    if(auxiliaryFields.size() != 1){
        cout<<"Eror: auxiliaryFields.size() != 1, "<<auxiliaryFields.size()<<endl;
        exit(1);
    }
    //auxiliaryFields for left bra
    auxiliaryFields[0].read("HAFQMC_AuxFields/HAFQMC_auxiliary_field_Left_sample_list_"+to_string(i)+".HAFQMCdat");
}

void Metropolis2sInfo::takeLeftHalf()
{
    BPMetroTimesliceBlockSize = BPMetroTimesliceBlockSize/2;

    currentLogOverlap = currentLogOverlap;

    auxiliaryFields.erase(auxiliaryFields.begin(),auxiliaryFields.begin()+BPMetroTimesliceBlockSize);
    // auxiliaryFields_BP.erase(auxiliaryFields_BP.begin(),auxiliaryFields_BP.begin()+BPMetroTimesliceBlockSize);
    dynamicForceFields.erase(dynamicForceFields.begin(),dynamicForceFields.begin()+BPMetroTimesliceBlockSize);
    // dynamicForceFields_BP.erase(dynamicForceFields_BP.begin(),dynamicForceFields_BP.begin()+BPMetroTimesliceBlockSize);

    walkerRightInBlock.erase(walkerRightInBlock.begin(),walkerRightInBlock.begin()+BPMetroTimesliceBlockSize);
    walkerLeftInBlock.erase(walkerLeftInBlock.begin()+BPMetroTimesliceBlockSize+1,walkerLeftInBlock.begin()+BPMetroTimesliceBlockSize*2+1);

    logWeightRightInBlock.erase(logWeightRightInBlock.begin(),logWeightRightInBlock.begin()+BPMetroTimesliceBlockSize);
    logWeightLeftInBlock.erase(logWeightLeftInBlock.begin()+BPMetroTimesliceBlockSize+1,logWeightLeftInBlock.begin()+BPMetroTimesliceBlockSize*2+1);
}


void Metropolis2sInfo::extendMetroChainToRight( Metropolis2sInfo metropolisInfo_input)
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

double Metropolis2sInfo::getMemory() const
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
    // for(size_t i=1-1; i<=auxiliaryFields_BP.size()-1; i++){
    //     mem += 4*auxiliaryFields_BP[i].size();
    // }
    for(size_t i=1-1; i<=dynamicForceFields.size()-1; i++){
        mem += 16*dynamicForceFields[i].size();
    }
    // for(size_t i=1-1; i<=dynamicForceFields_BP.size()-1; i++){
    //     mem += 8*dynamicForceFields_BP[i].size();
    // }

    for(size_t i=1-1; i<=walkerRightInBlock.size()-1; i++){
        mem += walkerRightInBlock[i].getMemory();
    }
    for(size_t i=1-1; i<=walkerLeftInBlock.size()-1; i++){
        mem += walkerLeftInBlock[i].getMemory();
    }

    mem += 16*logWeightRightInBlock.size();
    mem += 16*logWeightLeftInBlock.size();

    mem += overlapMatrixUp_inv.getMemory();
    mem += overlapMatrixDn_inv.getMemory();

    mem += Bup.getMemory();
    mem += Bdn.getMemory();

    /////////////////////////////////////////
    mem += overlapMatrixUp.getMemory();
    mem += overlapMatrixDn.getMemory();
    mem += A0up.getMemory() + A0dn.getMemory();
    mem += A0WRup.getMemory() + A0WRdn.getMemory();
    mem += A1up.getMemory() + A1dn.getMemory();
    mem += B0up.getMemory();
    // mem += B0dn.getMemory();
    mem += B0WRup.getMemory() + B0WRdn.getMemory();
    mem += C0up.getMemory();
    // mem += C0dn.getMemory();
    mem += Dup.getMemory();
    // mem += Ddn.getMemory();

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

int Metropolis2sInfo::returnNbuf() const
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
    // for(size_t i=1-1; i<=auxiliaryFields_BP.size()-1; i++){
    //     Nbuf += 4*auxiliaryFields_BP[i].size();
    // }
    for(size_t i=1-1; i<=dynamicForceFields.size()-1; i++){
        Nbuf += 16*dynamicForceFields[i].size();
    }
    // for(size_t i=1-1; i<=dynamicForceFields_BP.size()-1; i++){
    //     Nbuf += 8*dynamicForceFields_BP[i].size();
    // }

    for(size_t i=1-1; i<=walkerRightInBlock.size()-1; i++){
        Nbuf += walkerRightInBlock[i].returnNbuf();
    }
    for(size_t i=1-1; i<=walkerLeftInBlock.size()-1; i++){
        Nbuf += walkerLeftInBlock[i].returnNbuf();
    }

    Nbuf += 16*logWeightRightInBlock.size();
    Nbuf += 16*logWeightLeftInBlock.size();

    Nbuf += 16*overlapMatrixUp_inv.size();
    Nbuf += 16*overlapMatrixDn_inv.size();

    Nbuf += 16*Bup.size();
    Nbuf += 16*Bdn.size();

    /////////////////////////////////////////
    Nbuf += 4;
    Nbuf += 4;
    Nbuf += 16*overlapMatrixUp.size();
    Nbuf += 16*overlapMatrixDn.size();
    Nbuf += 16*A0up.size() + 16*A0dn.size();
    Nbuf += 16*A0WRup.size() + 16*A0WRdn.size();
    Nbuf += 16*A1up.size() + 16*A1dn.size();
    Nbuf += 16*B0up.size();
    // Nbuf += 16*B0dn.size();
    Nbuf += 16*B0WRup.size() + 16*B0WRdn.size();
    Nbuf += 16*C0up.size();
    // Nbuf += 16*C0dn.size();
    Nbuf += 16*Dup.size();
    // Nbuf += 16*Ddn.size();
    Nbuf += 1;
    Nbuf += 1;
    /////////////////////////////////////////

    return  Nbuf; 
}

#ifdef MPI_HAO
void MPIBcast(Metropolis2sInfo &buffer, int root, MPI_Comm const &comm)
{
    MPIBcast( buffer.BPMetroTimesliceBlockSize, root, comm );
    MPIBcast( buffer.inBlockIndex, root, comm );
    MPIBcast( buffer.currentLogOverlap, root, comm );
    for(size_t i = 1-1; i <= buffer.auxiliaryFields.size()-1; ++i){
        MPIBcast( buffer.auxiliaryFields[i], root, comm );
    }
    // for(size_t i = 1-1; i <= buffer.auxiliaryFields_BP.size()-1; ++i){
    //     MPIBcast( buffer.auxiliaryFields_BP[i], root, comm );
    // }
    for(size_t i = 1-1; i <= buffer.dynamicForceFields.size()-1; ++i){
        MPIBcast( buffer.dynamicForceFields[i], root, comm );
    }
    // for(size_t i = 1-1; i <= buffer.dynamicForceFields_BP.size()-1; ++i){
    //     MPIBcast( buffer.dynamicForceFields_BP[i], root, comm );
    // }

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

    MPIBcast( buffer.overlapMatrixUp_inv, root, comm );
    MPIBcast( buffer.overlapMatrixDn_inv, root, comm );

    MPIBcast( buffer.Bup, root, comm );
    MPIBcast( buffer.Bdn, root, comm );

    /////////////////////////////////////////
    MPIBcast( buffer.truncatedDup, root, comm );
    MPIBcast( buffer.truncatedDdn, root, comm );
    MPIBcast( buffer.overlapMatrixUp, root, comm );
    MPIBcast( buffer.overlapMatrixDn, root, comm );
    MPIBcast( buffer.A0up, root, comm );
    MPIBcast( buffer.A0dn, root, comm );
    MPIBcast( buffer.A0WRup, root, comm );
    MPIBcast( buffer.A0WRdn, root, comm );
    MPIBcast( buffer.A1up, root, comm );
    MPIBcast( buffer.A1dn, root, comm );
    MPIBcast( buffer.B0up, root, comm );
    // MPIBcast( buffer.B0dn, root, comm );
    MPIBcast( buffer.B0WRup, root, comm );
    MPIBcast( buffer.B0WRdn, root, comm );
    MPIBcast( buffer.C0up, root, comm );
    // MPIBcast( buffer.C0dn, root, comm );
    MPIBcast( buffer.Dup, root, comm );
    // MPIBcast( buffer.Ddn, root, comm );
    MPIBcast( buffer.globalFastInitialized, root, comm );
    MPIBcast( buffer.globalFastUpdated, root, comm );
    /////////////////////////////////////////
}

void Metropolis2sInfo::pack(vector<char> &buf, int &posit) const     
{    
    MPI_Pack(&BPMetroTimesliceBlockSize, 1, MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    //
    MPI_Pack(&inBlockIndex, 1, MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(&currentLogOverlap, 1, MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    //
    for(size_t i=1-1; i<=auxiliaryFields.size()-1; i++){
        MPI_Pack(auxiliaryFields[i].data(), auxiliaryFields[i].size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    }
    // for(size_t i=1-1; i<=auxiliaryFields_BP.size()-1; i++){
    //     MPI_Pack(auxiliaryFields_BP[i].data(), auxiliaryFields_BP[i].size(), MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    // }
    for(size_t i=1-1; i<=dynamicForceFields.size()-1; i++){
        MPI_Pack(dynamicForceFields[i].data(), dynamicForceFields[i].size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    }
    // for(size_t i=1-1; i<=dynamicForceFields_BP.size()-1; i++){
    //     MPI_Pack(dynamicForceFields_BP[i].data(), dynamicForceFields_BP[i].size(), MPI_DOUBLE, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    // }
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
    MPI_Pack(overlapMatrixUp_inv.data(), overlapMatrixUp_inv.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(overlapMatrixDn_inv.data(), overlapMatrixDn_inv.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    //
    MPI_Pack(Bup.data(), Bup.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(Bdn.data(), Bdn.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);

    /////////////////////////////////////////
    MPI_Pack(&truncatedDup, 1, MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(&truncatedDdn, 1, MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(overlapMatrixUp.data(), overlapMatrixUp.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(overlapMatrixDn.data(), overlapMatrixDn.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(A0up.data(), A0up.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(A0dn.data(), A0dn.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(A0WRup.data(), A0WRup.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(A0WRdn.data(), A0WRdn.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(A1up.data(), A1up.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(A1dn.data(), A1dn.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(B0up.data(), B0up.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    // MPI_Pack(B0dn.data(), B0dn.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(B0WRup.data(), B0WRup.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(B0WRdn.data(), B0WRdn.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(C0up.data(), C0up.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    // MPI_Pack(C0dn.data(), C0dn.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(Dup.data(), Dup.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    // MPI_Pack(Ddn.data(), Ddn.size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(&globalFastInitialized, 1, MPI_CXX_BOOL, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(&globalFastUpdated, 1, MPI_CXX_BOOL, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    /////////////////////////////////////////
}

void Metropolis2sInfo::unpack(const vector<char> &buf, int &posit)
{
    MPI_Unpack(buf.data(), buf.size(), &posit, &BPMetroTimesliceBlockSize,1, MPI_INT, MPI_COMM_WORLD);
    //
    MPI_Unpack(buf.data(), buf.size(), &posit, &inBlockIndex,1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, &currentLogOverlap,1, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    //
    for(size_t i=1-1; i<=auxiliaryFields.size()-1; i++){
        MPI_Unpack(buf.data(), buf.size(), &posit, auxiliaryFields[i].data(), auxiliaryFields[i].size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    }
    // for(size_t i=1-1; i<=auxiliaryFields_BP.size()-1; i++){
    //     MPI_Unpack(buf.data(), buf.size(), &posit, auxiliaryFields_BP[i].data(), auxiliaryFields_BP[i].size(), MPI_INT, MPI_COMM_WORLD);
    // }
    for(size_t i=1-1; i<=dynamicForceFields.size()-1; i++){
        MPI_Unpack(buf.data(), buf.size(), &posit, dynamicForceFields[i].data(), dynamicForceFields[i].size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    }
    // for(size_t i=1-1; i<=dynamicForceFields_BP.size()-1; i++){
    //     MPI_Unpack(buf.data(), buf.size(), &posit, dynamicForceFields_BP[i].data(), dynamicForceFields_BP[i].size(), MPI_DOUBLE, MPI_COMM_WORLD);
    // }
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
    MPI_Unpack(buf.data(), buf.size(), &posit, overlapMatrixUp_inv.data(), overlapMatrixUp_inv.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, overlapMatrixDn_inv.data(), overlapMatrixDn_inv.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);

    MPI_Unpack(buf.data(), buf.size(), &posit, Bup.data(), Bup.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, Bdn.data(), Bdn.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);

    /////////////////////////////////////////
    MPI_Unpack(buf.data(), buf.size(), &posit, &truncatedDup,1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, &truncatedDdn,1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, overlapMatrixUp.data(), overlapMatrixUp.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, overlapMatrixDn.data(), overlapMatrixDn.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, A0up.data(), A0up.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, A0dn.data(), A0dn.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, A0WRup.data(), A0WRup.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, A0WRdn.data(), A0WRdn.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, A1up.data(), A1up.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, A1dn.data(), A1dn.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, B0up.data(), B0up.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    // MPI_Unpack(buf.data(), buf.size(), &posit, B0dn.data(), B0dn.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, B0WRup.data(), B0WRup.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, B0WRdn.data(), B0WRdn.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, C0up.data(), C0up.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    // MPI_Unpack(buf.data(), buf.size(), &posit, C0dn.data(), C0dn.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, Dup.data(), Dup.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    // MPI_Unpack(buf.data(), buf.size(), &posit, Ddn.data(), Ddn.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, &globalFastInitialized,1, MPI_CXX_BOOL, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, &globalFastUpdated,1, MPI_CXX_BOOL, MPI_COMM_WORLD);
    /////////////////////////////////////////
}
#endif


void Metropolis2sInfo::copy_deep(const Metropolis2sInfo &x)
{
    BPMetroTimesliceBlockSize = x.BPMetroTimesliceBlockSize;

    inBlockIndex = x.inBlockIndex;
    currentLogOverlap = x.currentLogOverlap;

    auxiliaryFields = x.auxiliaryFields;
    // auxiliaryFields_BP = x.auxiliaryFields_BP;
    dynamicForceFields = x.dynamicForceFields;
    // dynamicForceFields_BP = x.dynamicForceFields_BP;

    walkerRightInBlock = x.walkerRightInBlock;
    walkerLeftInBlock = x.walkerLeftInBlock;

    logWeightRightInBlock = x.logWeightRightInBlock;
    logWeightLeftInBlock = x.logWeightLeftInBlock;

    overlapMatrixUp_inv = x.overlapMatrixUp_inv;
    overlapMatrixDn_inv = x.overlapMatrixDn_inv;

    Bup = x.Bup;
    Bdn = x.Bdn;

    /////////////////////////////////////////
    truncatedDup = x.truncatedDup;
    truncatedDdn = x.truncatedDdn;
    overlapMatrixUp = x.overlapMatrixUp;
    overlapMatrixDn = x.overlapMatrixDn;
    A0up = x.A0up;
    A0dn = x.A0dn;
    A0WRup = x.A0WRup;
    A0WRdn = x.A0WRdn;
    A1up = x.A1up;
    A1dn = x.A1dn;
    B0up = x.B0up;
    // B0dn = x.B0dn;
    B0WRup = x.B0WRup;
    B0WRdn = x.B0WRdn;
    C0up = x.C0up;
    // C0dn = x.C0dn;
    Dup = x.Dup;
    // Ddn = x.Ddn;
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

void Metropolis2sInfo::move_deep(Metropolis2sInfo &x)
{
    BPMetroTimesliceBlockSize = x.BPMetroTimesliceBlockSize;

    inBlockIndex = x.inBlockIndex;
    currentLogOverlap = x.currentLogOverlap;

    auxiliaryFields = move(x.auxiliaryFields);
    // auxiliaryFields_BP = move(x.auxiliaryFields_BP);
    dynamicForceFields = move(x.dynamicForceFields);
    // dynamicForceFields_BP = move(x.dynamicForceFields_BP);

    walkerRightInBlock = move(x.walkerRightInBlock);
    walkerLeftInBlock = move(x.walkerLeftInBlock);

    logWeightRightInBlock = move(x.logWeightRightInBlock);
    logWeightLeftInBlock = move(x.logWeightLeftInBlock);

    overlapMatrixUp_inv = move(x.overlapMatrixUp_inv);
    overlapMatrixDn_inv = move(x.overlapMatrixDn_inv);
    
    Bup = move(x.Bup);
    Bdn = move(x.Bdn);

    /////////////////////////////////////////
    truncatedDup = x.truncatedDup;
    truncatedDdn = x.truncatedDdn;
    overlapMatrixUp = move(x.overlapMatrixUp);
    overlapMatrixDn = move(x.overlapMatrixDn);
    A0up = move(x.A0up);
    A0dn = move(x.A0dn);
    A0WRup = move(x.A0WRup);
    A0WRdn = move(x.A0WRdn);
    A1up = move(x.A1up);
    A1dn = move(x.A1dn);
    B0up = move(x.B0up);
    // B0dn = move(x.B0dn);
    B0WRup = move(x.B0WRup);
    B0WRdn = move(x.B0WRdn);
    C0up = move(x.C0up);
    // C0dn = move(x.C0dn);
    // 
    Dup = move(x.Dup);
    // Ddn = move(x.Ddn);
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



