//
// Created by boruoshihao on 1/15/17.
//
#include "../../include/Metropolis/metropolisMethod.h"

using namespace std;

MetropolisMethod::MetropolisMethod()
{
    setDefault();
}

MetropolisMethod::MetropolisMethod(const MetropolisMethod &x) { copy_deep(x); }

MetropolisMethod::MetropolisMethod(MetropolisMethod &&x) { move_deep(x); }

MetropolisMethod::~MetropolisMethod()
{

}

MetropolisMethod &MetropolisMethod::operator=(const MetropolisMethod &x) { copy_deep(x); return *this; }

MetropolisMethod &MetropolisMethod::operator=(MetropolisMethod &&x) { move_deep(x); return *this; }

void MetropolisMethod::read(const string &filename)
{
    readBySearchString(numOfJastrow, "numOfJastrow", filename);
    JastrowSlice.resize(numOfJastrow);
    for(int i=1-1; i<=numOfJastrow-1; i++){
        readBySearchString(JastrowSlice[i], "JastrowSlice_"+to_string(i), filename);
    }
    JastrowName.resize(numOfJastrow);
    for(int i=1-1; i<=numOfJastrow-1; i++){
        readBySearchString(JastrowName[i], "JastrowName_"+to_string(i), filename);
    }
    JastrowExpM.resize(numOfJastrow);
    for(int i=1-1; i<=numOfJastrow-1; i++){
        readBySearchString(JastrowExpM[i], "JastrowExpM_"+to_string(i), filename);
    }
    //
    BPMetroTimesliceBlockSize=0;
    for(int i=1-1; i<=numOfJastrow-1; i++){
        BPMetroTimesliceBlockSize += JastrowSlice[i];
    }
    //
    readBySearchString(BPMetroSampleCap, "BPMetroSampleCap", filename);
    readBySearchString(BPMetroStabilizeStep, "BPMetroStabilizeStep", filename);
    readBySearchString(blockNum, "blockNum", filename);
    readBySearchString(BPMetroForceType, "BPMetroForceType", filename); 
    readBySearchString(BPMetroInitialAuxiliaryFlag, "BPMetroInitialAuxiliaryFlag", filename);  //"dynamicForceInitial" or "constForceInitial" 
    readBySearchString(BPMetroUpdateType, "BPMetroUpdateType", filename);
    readBySearchString(seed, "seed", filename);

    readBySearchString(Metro_dtET, "Metro_dtET", filename);
    readBySearchString(HMC_dt, "HMC_dt", filename);
    readBySearchString(HMC_length, "HMC_length", filename);
}

void MetropolisMethod::print()
{
    cout<<left<<endl;

    cout<<setw(36)<<"numOfJastrow "<<setw(26)<<numOfJastrow<<endl;
    for(int i=1-1; i<=numOfJastrow-1; i++){
        cout<<setw(36)<<"JastrowSlice_"+to_string(i)<<setw(26)<<JastrowSlice[i]<<endl;
    }
    for(int i=1-1; i<=numOfJastrow-1; i++){
        cout<<setw(36)<<"JastrowName_"+to_string(i)<<setw(26)<<JastrowName[i]<<endl;
    }
    for(int i=1-1; i<=numOfJastrow-1; i++){
        cout<<setw(36)<<"JastrowExpM_"+to_string(i)<<setw(26)<<JastrowExpM[i]<<endl;
    }
    cout<<setw(36)<<"BPMetroTimesliceBlockSize "<<setw(26)<<BPMetroTimesliceBlockSize<<endl;
    cout<<endl;

    cout<<setw(36)<<"BPMetroSampleCap "<<setw(26)<<BPMetroSampleCap<<endl;
    cout<<setw(36)<<"blockNum "<<setw(26)<<blockNum<<endl;
    cout<<setw(36)<<"BPMetroForceType "<<setw(26)<<BPMetroForceType<<endl;
    cout<<setw(36)<<"BPMetroInitialAuxiliaryFlag "<<setw(26)<<BPMetroInitialAuxiliaryFlag<<endl;
    cout<<setw(36)<<"BPMetroUpdateType "<<setw(26)<<BPMetroUpdateType<<endl;
    cout<<setw(36)<<"seed "<<setw(26)<<seed<<endl;
    cout<<endl;

    cout<<setw(36)<<"Metro_dtET "<<setw(26)<<Metro_dtET<<endl;
    cout<<setw(36)<<"HMC_dt "<<setw(26)<<HMC_dt<<endl;
    cout<<setw(36)<<"HMC_length "<<setw(26)<<HMC_length<<endl;
    cout<<setw(36)<<"HMC leaps for  "<<setw(26)<<int(HMC_length / HMC_dt)<<" steps"<<endl;
}

#ifdef MPI_HAO
void MPIBcast(MetropolisMethod &buffer, int root, MPI_Comm const &comm)
{
    MPIBcast(buffer.numOfJastrow, root, comm);
    MPIBcast(buffer.JastrowSlice, root, comm);
    //
    buffer.JastrowName.resize(buffer.numOfJastrow);
    for(int i=1-1; i<=buffer.numOfJastrow-1; i++){
        MPIBcast(buffer.JastrowName[i], root, comm);
    }
    MPIBcast(buffer.JastrowExpM, root, comm);
    //
    MPIBcast(buffer.BPMetroSampleCap, root, comm);
    MPIBcast(buffer.BPMetroStabilizeStep, root, comm);
    MPIBcast(buffer.BPMetroTimesliceBlockSize, root, comm);
    MPIBcast(buffer.blockNum, root, comm);
    MPIBcast(buffer.BPMetroForceType, root, comm);
    MPIBcast(buffer.BPMetroInitialAuxiliaryFlag, root, comm);
    MPIBcast(buffer.BPMetroUpdateType, root, comm);
    MPIBcast(buffer.seed, root, comm);

    MPIBcast(buffer.Metro_dtET, root, comm);
    MPIBcast(buffer.HMC_dt, root, comm);
    MPIBcast(buffer.HMC_length, root, comm);
}

void MetropolisMethod::pack(vector<char> &buf, int &posit) const
{
    MPI_Pack(&numOfJastrow, 1, MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    for(size_t i=1-1; i<=numOfJastrow-1; i++){
        MPI_Pack(&JastrowSlice[i], 1, MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    }
    for(size_t i=1-1; i<=numOfJastrow-1; i++){
        MPI_Pack(JastrowName[i].c_str(), JastrowName[i].size(), MPI_CHAR, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    }
    for(size_t i=1-1; i<=numOfJastrow-1; i++){
        MPI_Pack(&JastrowExpM[i], 1, MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    }
    MPI_Pack(&BPMetroSampleCap, 1, MPI_DOUBLE, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(&BPMetroStabilizeStep, 1, MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(&BPMetroTimesliceBlockSize, 1, MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(&blockNum, 1, MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);

    MPI_Pack(BPMetroForceType.c_str(), BPMetroForceType.size(), MPI_CHAR, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(BPMetroInitialAuxiliaryFlag.c_str(), BPMetroInitialAuxiliaryFlag.size(), MPI_CHAR, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(&seed, 1, MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(&Metro_dtET, 1, MPI_DOUBLE, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(&HMC_dt, 1, MPI_DOUBLE, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(&HMC_length, 1, MPI_DOUBLE, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
}

void MetropolisMethod::unpack(const vector<char> &buf, int &posit)
{
    MPI_Unpack(buf.data(), buf.size(), &posit, &numOfJastrow, 1, MPI_INT, MPI_COMM_WORLD);
    for(size_t i=1-1; i<=numOfJastrow-1; i++){
        MPI_Unpack(buf.data(), buf.size(), &posit, &JastrowSlice[i], 1, MPI_INT, MPI_COMM_WORLD);
    }
    for(size_t i=1-1; i<=numOfJastrow-1; i++){
        char* JastrowName_CharList;
        int JastrowName_CharList_size;
        MPI_Unpack(buf.data(), buf.size(), &posit, &JastrowName_CharList, JastrowName_CharList_size, MPI_CHAR, MPI_COMM_WORLD);
        JastrowName[i]=JastrowName_CharList;
    }
    for(size_t i=1-1; i<=numOfJastrow-1; i++){
        MPI_Unpack(buf.data(), buf.size(), &posit, &JastrowExpM[i], 1, MPI_INT, MPI_COMM_WORLD);
    }
    MPI_Unpack(buf.data(), buf.size(), &posit, &BPMetroSampleCap, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, &BPMetroStabilizeStep, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, &BPMetroTimesliceBlockSize, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, &blockNum, 1, MPI_INT, MPI_COMM_WORLD);

    char* BPMetroForceType_CharList;
    int BPMetroForceType_CharList_size;
    MPI_Unpack(buf.data(), buf.size(), &posit, &BPMetroForceType_CharList, BPMetroForceType_CharList_size, MPI_CHAR, MPI_COMM_WORLD);
    BPMetroForceType=BPMetroForceType_CharList;

    char* BPMetroInitialAuxiliaryFlag_CharList;
    int BPMetroInitialAuxiliaryFlag_CharList_size;
    MPI_Unpack(buf.data(), buf.size(), &posit, &BPMetroInitialAuxiliaryFlag_CharList, BPMetroInitialAuxiliaryFlag_CharList_size, MPI_CHAR, MPI_COMM_WORLD);
    BPMetroInitialAuxiliaryFlag=BPMetroInitialAuxiliaryFlag_CharList;

    char* BPMetroUpdateType_CharList;
    int BPMetroUpdateType_CharList_size;
    MPI_Unpack(buf.data(), buf.size(), &posit, &BPMetroUpdateType_CharList, BPMetroUpdateType_CharList_size, MPI_CHAR, MPI_COMM_WORLD);
    BPMetroUpdateType=BPMetroUpdateType_CharList;
    
    MPI_Unpack(buf.data(), buf.size(), &posit, &seed, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, &Metro_dtET, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, &HMC_dt, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, &HMC_length, 1, MPI_DOUBLE, MPI_COMM_WORLD);
}
#endif

void MetropolisMethod::setDefault()
{
    BPMetroSampleCap = 5;
    BPMetroStabilizeStep = 1;
    blockNum = 0;
    BPMetroForceType = "constForce";
    BPMetroInitialAuxiliaryFlag = "constForceInitial";
    BPMetroUpdateType = "global";
    seed = 0;
    Metro_dtET = 0.0;
    HMC_dt = 0.0;
    HMC_length = 0.0;
}


void MetropolisMethod::copy_deep(const MetropolisMethod &x)
{
    JastrowSlice = x.JastrowSlice;
    JastrowName = x.JastrowName;
    JastrowExpM = x.JastrowExpM;
    numOfJastrow = x.numOfJastrow;
    BPMetroSampleCap = x.BPMetroSampleCap;
    BPMetroStabilizeStep = x.BPMetroStabilizeStep;
    BPMetroTimesliceBlockSize = x.BPMetroTimesliceBlockSize;
    blockNum = x.blockNum;
    BPMetroForceType = x.BPMetroForceType;
    BPMetroInitialAuxiliaryFlag = x.BPMetroInitialAuxiliaryFlag;
    BPMetroUpdateType = x.BPMetroUpdateType;
    seed = x.seed;
    Metro_dtET = x.Metro_dtET;
    HMC_dt = x.HMC_dt;
    HMC_length = x.HMC_length;
}

void MetropolisMethod::move_deep(MetropolisMethod &x)
{
    JastrowSlice = x.JastrowSlice;
    JastrowName = x.JastrowName;
    JastrowExpM = x.JastrowExpM;
    numOfJastrow = x.numOfJastrow;
    BPMetroSampleCap = x.BPMetroSampleCap;
    BPMetroStabilizeStep = x.BPMetroStabilizeStep;
    BPMetroTimesliceBlockSize = x.BPMetroTimesliceBlockSize;
    blockNum = x.blockNum;
    BPMetroForceType = x.BPMetroForceType;
    BPMetroInitialAuxiliaryFlag = x.BPMetroInitialAuxiliaryFlag;
    BPMetroUpdateType = x.BPMetroUpdateType;
    seed = x.seed;
    Metro_dtET = x.Metro_dtET;
    HMC_dt = x.HMC_dt;
    HMC_length = x.HMC_length;
}
