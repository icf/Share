//
// Created by boruoshihao on 5/30/17.
//

#include "../../include/generalHamiltonian_icf/generalHamiltonian_icf.h"

#ifdef MPI_HAO
#include <mpi.h>
#include <algorithm>
#endif

using namespace std;
using namespace H5;
using namespace tensor_hao;

GeneralHamiltonian_icf::GeneralHamiltonian_icf():L(0), N(0) { }

GeneralHamiltonian_icf::GeneralHamiltonian_icf(const string &filename) { read(filename); }

GeneralHamiltonian_icf::~GeneralHamiltonian_icf() { }

size_t GeneralHamiltonian_icf::getL() const { return L; }

size_t GeneralHamiltonian_icf::getSD2sL() const { return SD2sL; }

bool GeneralHamiltonian_icf::getHamiltonian_spin_flag() const { return Hamiltonian_spin_flag; }

size_t GeneralHamiltonian_icf::getTruncatedDup() const { return truncatedDup; }

size_t GeneralHamiltonian_icf::getTruncatedDdn() const { return truncatedDdn; }

size_t GeneralHamiltonian_icf::getTruncatedD() const { return truncatedD; }

// size_t GeneralHamiltonian_icf::getKlx() const { return k_l_x; }

// size_t GeneralHamiltonian_icf::getKly() const { return k_l_y; }

size_t GeneralHamiltonian_icf::getN() const { return N; }

size_t GeneralHamiltonian_icf::getNup() const { return Nup; }

size_t GeneralHamiltonian_icf::getNdn() const { return Ndn; }

size_t GeneralHamiltonian_icf::getSVDNumber() const { return svdNumber; }

size_t GeneralHamiltonian_icf::getTruncatedSVDNumber() const { return truncatedSvdNumber; }

const TensorHao<complex<double>,2> &GeneralHamiltonian_icf::getK() const { return K; }

const TensorHaoMPIRef<complex<double>,3> &GeneralHamiltonian_icf::getSVDVecs() const { return svdVecs; }

const TensorHao<complex<double>,1> &GeneralHamiltonian_icf::getSVDBg() const { return svdBg; }

const TensorHao<complex<double>,1> &GeneralHamiltonian_icf::getInitialBg() const { return initialBg; }

size_t GeneralHamiltonian_icf::getKpEigenStatus() const { return KpEigenStatus; }

const TensorHao<complex<double>,2> &GeneralHamiltonian_icf::getKp() const { return Kp; }

const TensorHao<double,1> &GeneralHamiltonian_icf::getKpEigenValue() const { return KpEigenValue; }

const TensorHao<complex<double>, 2> &GeneralHamiltonian_icf::getKpEigenVector() const { return KpEigenVector; }

void GeneralHamiltonian_icf::read(const string &filename)
{
    H5File file(filename, H5F_ACC_RDONLY);
    // readFile(k_l_x, file, "k_l_x");
    // readFile(k_l_y, file, "k_l_y");
    readFile(L, file, "L");
    readFile(SD2sL, file, "SD2sL");
    readFile(truncatedDup, file, "truncatedDup");
    readFile(truncatedDdn, file, "truncatedDdn");
    readFile(Nup, file, "Nup");
    readFile(Ndn, file, "Ndn");
    readFile(N, file, "N");
    readFile(svdNumber, file, "svdNumber");
    readFile(truncatedSvdNumber, file, "truncatedSvdNumber");

    TensorHao<double,2> K_U_r,K_U_i;
    TensorHao<double,2> K_r,K_i;
    TensorHao<double,2> K_D_r,K_D_i;
    TensorHao<double,2> K_Vdagger_r,K_Vdagger_i;
    TensorHao<double,2> svdVecs_U0_r,svdVecs_U0_i;
    TensorHao<double,3> svdVecs_r,svdVecs_i;
    TensorHao<double,3> svdVecs_D_r,svdVecs_D_i;
    TensorHao<double,2> svdVecs_Vdagger0_r,svdVecs_Vdagger0_i;
    TensorHao<double,1> svdBg_r,svdBg_i;
    // 
    #ifdef MPI_HAO
    int ranks_per_node = 24;
    char* cpus_per_node = getenv("SLURM_JOB_CPUS_PER_NODE");
    if(cpus_per_node) {
        ranks_per_node = atoi(cpus_per_node);
    }
    int node_id = MPIRank() / ranks_per_node;
    int local_rank = MPIRank() % ranks_per_node;
    bool is_local_root = (local_rank == 0);
    
    MPI_Comm node_comm;
    MPI_Comm_split(MPI_COMM_WORLD, node_id, MPIRank(), &node_comm);
    #endif
    // 
    if(SD2sL == L/2){
        Hamiltonian_spin_flag = true;
        truncatedD = truncatedDup + truncatedDdn;
        if( SD2sL * 2 != L){
            cout<<"Error in GeneralHamiltonian_icf::read: L is not even!"<<endl;
            exit(1);
        }
    }else if(SD2sL == L){
        Hamiltonian_spin_flag = false;
        if(truncatedDup == truncatedDdn ){
            truncatedD = (truncatedDup + truncatedDdn)/2;
        }else{
            cout<<"Error in GeneralHamiltonian_icf::read: truncatedDup != truncatedDdn!"<<endl;
            exit(1);
        }
    }else{
        cout<<"Error in GeneralHamiltonian_icf::read: SD2sL is not equal to L or L/2!"<<endl;
        exit(1);
    }
    // 
    if(truncatedD > 0 && svdNumber==truncatedSvdNumber){
        K_U_r.resize(L, truncatedD); readFile( K_U_r.size(),  K_U_r.data(),  file, "K_U_r" );
        K_U_i.resize(L, truncatedD); readFile( K_U_i.size(),  K_U_i.data(),  file, "K_U_i" );
        K_D_r.resize(truncatedD, truncatedD); readFile( K_D_r.size(),  K_D_r.data(),  file, "K_D_r" );
        K_D_i.resize(truncatedD, truncatedD); readFile( K_D_i.size(),  K_D_i.data(),  file, "K_D_i" );
        K_Vdagger_r.resize(truncatedD, L); readFile( K_Vdagger_r.size(),  K_Vdagger_r.data(),  file, "K_Vdagger_r" );
        K_Vdagger_i.resize(truncatedD, L); readFile( K_Vdagger_i.size(),  K_Vdagger_i.data(),  file, "K_Vdagger_i" );
        // 
        svdVecs_U0_r.resize(L, truncatedD); readFile(svdVecs_U0_r.size(), svdVecs_U0_r.data(), file, "svdVecs_U0_r" );
        svdVecs_U0_i.resize(L, truncatedD); readFile(svdVecs_U0_i.size(), svdVecs_U0_i.data(), file, "svdVecs_U0_i" );
        svdVecs_D_r.resize(truncatedD, truncatedD, truncatedSvdNumber); readFile(svdVecs_D_r.size(), svdVecs_D_r.data(), file, "svdVecs_D_r" );
        svdVecs_D_i.resize(truncatedD, truncatedD, truncatedSvdNumber); readFile(svdVecs_D_i.size(), svdVecs_D_i.data(), file, "svdVecs_D_i" );
        svdVecs_Vdagger0_r.resize(truncatedD, L); readFile(svdVecs_Vdagger0_r.size(), svdVecs_Vdagger0_r.data(), file, "svdVecs_Vdagger0_r" );
        svdVecs_Vdagger0_i.resize(truncatedD, L); readFile(svdVecs_Vdagger0_i.size(), svdVecs_Vdagger0_i.data(), file, "svdVecs_Vdagger0_i" );
    }else{
        K_r.resize(L, L); readFile( K_r.size(),  K_r.data(),  file, "K_r" );
        K_i.resize(L, L); readFile( K_i.size(),  K_i.data(),  file, "K_i" );
        //Vq is not depends on k
        if(is_local_root){
            svdVecs_r.resize(L, L, svdNumber); readFile(svdVecs_r.size(), svdVecs_r.data(), file, "svdVecs_r" );
            svdVecs_i.resize(L, L, svdNumber); readFile(svdVecs_i.size(), svdVecs_i.data(), file, "svdVecs_i" );
        }
    }
    // 
    svdBg_r.resize(svdNumber); readFile(svdBg_r.size(), svdBg_r.data(), file, "svdBg_r" );
    svdBg_i.resize(svdNumber); readFile(svdBg_i.size(), svdBg_i.data(), file, "svdBg_i" );
    // 
    file.close();
    // 
    ///////////////////////////////////
    // 
    if(truncatedD > 0 && svdNumber==truncatedSvdNumber){
        K_U.resize(L, truncatedD);K_D.resize(truncatedD, truncatedD);K_Vdagger.resize(truncatedD, L);
        svdVecs_U0.resize(L, truncatedD);svdVecs_D.resize(truncatedD, truncatedD, truncatedSvdNumber);svdVecs_Vdagger0.resize(truncatedD, L);
        for(size_t i=1-1; i<=L-1; i++){
        for(size_t j=1-1; j<=truncatedD-1; j++){
            K_U(i,j)=K_U_r(i,j)+complex<double>(0.0,1.0)*K_U_i(i,j);
            svdVecs_U0(i,j)=svdVecs_U0_r(i,j)+complex<double>(0.0,1.0)*svdVecs_U0_i(i,j);
        } 
        } 
        for(size_t i=1-1; i<=truncatedD-1; i++){
        for(size_t j=1-1; j<=L-1; j++){
            K_Vdagger(i,j)=K_Vdagger_r(i,j)+complex<double>(0.0,1.0)*K_Vdagger_i(i,j);
            svdVecs_Vdagger0(i,j)=svdVecs_Vdagger0_r(i,j)+complex<double>(0.0,1.0)*svdVecs_Vdagger0_i(i,j);
        } 
        } 
        for(size_t i=1-1; i<=truncatedD-1; i++){
        for(size_t j=1-1; j<=truncatedD-1; j++){
            K_D(i,j)=K_D_r(i,j)+complex<double>(0.0,1.0)*K_D_i(i,j);
        } 
        } 
        for(size_t k=1-1; k<=truncatedSvdNumber-1; k++){
            for(size_t i=1-1; i<=truncatedD-1; i++){
            for(size_t j=1-1; j<=truncatedD-1; j++){
                svdVecs_D(i,j,k)=svdVecs_D_r(i,j,k)+complex<double>(0.0,1.0)*svdVecs_D_i(i,j,k);
            }
            }
        }
    }else{
        K_U.resize(0,0);K_D.resize(0,0);K_Vdagger.resize(0,0);
        svdVecs_U0.resize(0,0);svdVecs_D.resize(0,0,0);svdVecs_Vdagger0.resize(0,0);
    }
    /////////////////////////////////
    // Restore from UDV
    /////////////////////////////////
    K.resize(L, L);
    svdBg.resize(svdNumber); 
    // 
#ifdef MPI_HAO
    size_t dims[3] = {L, L, svdNumber};
    if(is_local_root) {
        svdVecs.createSharedMemory(dims, 0, node_comm);
    }
    else {
        svdVecs.createSharedMemoryView(0, node_comm);
        svdVecs.attachToSharedMemory(dims, 0, node_comm);
    }
#endif
    // 
    if(is_local_root) {
        if(truncatedD > 0 && svdNumber==truncatedSvdNumber){
            TensorHao<complex<double>, 2> svdVecs_D_matrix(truncatedD,truncatedD);
            TensorHao<complex<double>, 2> matrixTemp(L,truncatedD);
            TensorHao<complex<double>, 2> matrix(L,L);
            for(size_t k=1-1; k<=truncatedSvdNumber-1; k++){
                for(size_t i=1-1; i<=truncatedD-1; i++){
                for(size_t j=1-1; j<=truncatedD-1; j++){
                    svdVecs_D_matrix(i,j) = svdVecs_D(i,j,k);
                }
                }
                BL_NAME(gmm)(svdVecs_U0, svdVecs_D_matrix, matrixTemp);
                BL_NAME(gmm)(matrixTemp, svdVecs_Vdagger0, matrix);
                // 
                for(size_t i=1-1; i<=L-1; i++){
                for(size_t j=1-1; j<=L-1; j++){
                    svdVecs(i,j,k) = matrix(i,j);
                }
                }
            }
        }else{
            //get full svd from input, directly write to shared memory
            for(size_t k=1-1; k<=svdNumber-1; k++){
                for(size_t i=1-1; i<=L-1; i++){
                for(size_t j=1-1; j<=L-1; j++){
                    svdVecs(i,j,k)=svdVecs_r(i,j,k)+complex<double>(0.0,1.0)*svdVecs_i(i,j,k);
                }
                }
            }
        }
    }
    MPIBarrier();
    // 
    if(truncatedD > 0 && svdNumber==truncatedSvdNumber){
        TensorHao<complex<double>, 2> matrixTemp(L,truncatedD);
        TensorHao<complex<double>, 2> matrix(L,L);
        BL_NAME(gmm)(K_U, K_D, matrixTemp);
        BL_NAME(gmm)(matrixTemp, K_Vdagger, matrix);
        // 
        K = matrix;
    }else{
        for(size_t i=1-1; i<=L-1; i++){
        for(size_t j=1-1; j<=L-1; j++){
            K(i,j)=K_r(i,j)+complex<double>(0.0,1.0)*K_i(i,j);
        } 
        } 
    }
    //
    for(size_t k=1-1; k<=svdNumber-1; k++){
        svdBg(k)=svdBg_r(k)+complex<double>(0.0,1.0)*svdBg_i(k);
    }
    initialBg=svdBg;

    KpEigenStatus = 0;
    Kp.resize(0,0);
    KpEigenValue.resize( static_cast<size_t>(0) );
    KpEigenVector.resize( 0, 0 );
    #ifdef MPI_HAO
    MPI_Comm_free(&node_comm);
    #endif
}

void GeneralHamiltonian_icf::read_conj(const string &filename)
{
    H5File file(filename, H5F_ACC_RDONLY);
    // readFile(k_l_x, file, "k_l_x");
    // readFile(k_l_y, file, "k_l_y");
    readFile(L, file, "L");
    readFile(SD2sL, file, "SD2sL");
    readFile(truncatedDup, file, "truncatedDup");
    readFile(truncatedDdn, file, "truncatedDdn");
    truncatedD = truncatedDup + truncatedDdn;
    readFile(Nup, file, "Nup");
    readFile(Ndn, file, "Ndn");
    readFile(N, file, "N");
    readFile(svdNumber, file, "svdNumber");
    readFile(truncatedSvdNumber, file, "truncatedSvdNumber");
    // 
    if(SD2sL == L/2){
        Hamiltonian_spin_flag = true;
        truncatedD = truncatedDup + truncatedDdn;
        if( SD2sL * 2 != L){
            cout<<"Error in GeneralHamiltonian_icf::read: L is not even!"<<endl;
            exit(1);
        }
    }else if(SD2sL == L){
        Hamiltonian_spin_flag = false;
        if(truncatedDup == truncatedDdn ){
            truncatedD = (truncatedDup + truncatedDdn)/2;
        }else{
            cout<<"Error in GeneralHamiltonian_icf::read: truncatedDup != truncatedDdn!"<<endl;
            exit(1);
        }
    }else{
        cout<<"Error in GeneralHamiltonian_icf::read: SD2sL is not equal to L or L/2!"<<endl;
        exit(1);
    }
    // 

    TensorHao<double,2> K_U_r,K_U_i;
    TensorHao<double,2> K_r,K_i;
    TensorHao<double,2> K_D_r,K_D_i;
    TensorHao<double,2> K_Vdagger_r,K_Vdagger_i;
    TensorHao<double,2> svdVecs_U0_r,svdVecs_U0_i;
    TensorHao<double,3> svdVecs_r,svdVecs_i;
    TensorHao<double,3> svdVecs_D_r,svdVecs_D_i;
    TensorHao<double,2> svdVecs_Vdagger0_r,svdVecs_Vdagger0_i;
    TensorHao<double,1> svdBg_r,svdBg_i;
    // 
    #ifdef MPI_HAO
    int ranks_per_node = 24;
    char* cpus_per_node = getenv("SLURM_JOB_CPUS_PER_NODE");
    if(cpus_per_node) {
        ranks_per_node = atoi(cpus_per_node);
    }
    int node_id = MPIRank() / ranks_per_node;
    int local_rank = MPIRank() % ranks_per_node;
    bool is_local_root = (local_rank == 0);
    
    MPI_Comm node_comm;
    MPI_Comm_split(MPI_COMM_WORLD, node_id, MPIRank(), &node_comm);
    #endif
    // 
    if(truncatedD > 0 && svdNumber==truncatedSvdNumber){
        K_U_r.resize(L, truncatedD); readFile( K_U_r.size(),  K_U_r.data(),  file, "K_U_r" );
        K_U_i.resize(L, truncatedD); readFile( K_U_i.size(),  K_U_i.data(),  file, "K_U_i" );
        K_D_r.resize(truncatedD, truncatedD); readFile( K_D_r.size(),  K_D_r.data(),  file, "K_D_r" );
        K_D_i.resize(truncatedD, truncatedD); readFile( K_D_i.size(),  K_D_i.data(),  file, "K_D_i" );
        K_Vdagger_r.resize(truncatedD, L); readFile( K_Vdagger_r.size(),  K_Vdagger_r.data(),  file, "K_Vdagger_r" );
        K_Vdagger_i.resize(truncatedD, L); readFile( K_Vdagger_i.size(),  K_Vdagger_i.data(),  file, "K_Vdagger_i" );
        // 
        svdVecs_U0_r.resize(L, truncatedD); readFile(svdVecs_U0_r.size(), svdVecs_U0_r.data(), file, "svdVecs_U0_r" );
        svdVecs_U0_i.resize(L, truncatedD); readFile(svdVecs_U0_i.size(), svdVecs_U0_i.data(), file, "svdVecs_U0_i" );
        svdVecs_D_r.resize(truncatedD, truncatedD, truncatedSvdNumber); readFile(svdVecs_D_r.size(), svdVecs_D_r.data(), file, "svdVecs_D_r" );
        svdVecs_D_i.resize(truncatedD, truncatedD, truncatedSvdNumber); readFile(svdVecs_D_i.size(), svdVecs_D_i.data(), file, "svdVecs_D_i" );
        svdVecs_Vdagger0_r.resize(truncatedD, L); readFile(svdVecs_Vdagger0_r.size(), svdVecs_Vdagger0_r.data(), file, "svdVecs_Vdagger0_r" );
        svdVecs_Vdagger0_i.resize(truncatedD, L); readFile(svdVecs_Vdagger0_i.size(), svdVecs_Vdagger0_i.data(), file, "svdVecs_Vdagger0_i" );
    }else{
        K_r.resize(L, L); readFile( K_r.size(),  K_r.data(),  file, "K_r" );
        K_i.resize(L, L); readFile( K_i.size(),  K_i.data(),  file, "K_i" );
        //Vq is not depends on k
        if(is_local_root){
            svdVecs_r.resize(L, L, svdNumber); readFile(svdVecs_r.size(), svdVecs_r.data(), file, "svdVecs_r" );
            svdVecs_i.resize(L, L, svdNumber); readFile(svdVecs_i.size(), svdVecs_i.data(), file, "svdVecs_i" );
        }
    }
    //
    svdBg_r.resize(svdNumber); readFile(svdBg_r.size(), svdBg_r.data(), file, "svdBg_r" );
    svdBg_i.resize(svdNumber); readFile(svdBg_i.size(), svdBg_i.data(), file, "svdBg_i" );

    file.close();
    // 
    ///////////////////////////////////
    if(truncatedD > 0 && svdNumber==truncatedSvdNumber){
        K_U.resize(L, truncatedD);K_D.resize(truncatedD, truncatedD);K_Vdagger.resize(truncatedD, L);
        svdVecs_U0.resize(L, truncatedD);svdVecs_D.resize(truncatedD, truncatedD, truncatedSvdNumber);svdVecs_Vdagger0.resize(truncatedD, L);
        for(size_t i=1-1; i<=L-1; i++){
        for(size_t j=1-1; j<=truncatedD-1; j++){
            K_U(i,j)=K_Vdagger_r(j,i)+complex<double>(0.0,-1.0)*K_Vdagger_i(j,i);
            svdVecs_U0(i,j)=svdVecs_Vdagger0_r(j,i)+complex<double>(0.0,-1.0)*svdVecs_Vdagger0_i(j,i);
        } 
        } 
        for(size_t i=1-1; i<=truncatedD-1; i++){
        for(size_t j=1-1; j<=L-1; j++){
            K_Vdagger(i,j)=K_U_r(j,i)+complex<double>(0.0,-1.0)*K_U_i(j,i);
            svdVecs_Vdagger0(i,j)=svdVecs_U0_r(j,i)+complex<double>(0.0,-1.0)*svdVecs_U0_i(j,i);
        } 
        } 
        for(size_t i=1-1; i<=truncatedD-1; i++){
        for(size_t j=1-1; j<=truncatedD-1; j++){
            K_D(i,j)=K_D_r(j,i)+complex<double>(0.0,-1.0)*K_D_i(j,i);
        } 
        } 
        for(size_t k=1-1; k<=truncatedSvdNumber-1; k++){
            for(size_t i=1-1; i<=truncatedD-1; i++){
            for(size_t j=1-1; j<=truncatedD-1; j++){
                svdVecs_D(i,j,k)=svdVecs_D_r(j,i,truncatedSvdNumber-1-k)+complex<double>(0.0,-1.0)*svdVecs_D_i(j,i,truncatedSvdNumber-1-k);
            }
            }
        }
    }else{
        K_U.resize(0,0);K_D.resize(0,0);K_Vdagger.resize(0,0);
        svdVecs_U0.resize(0,0);svdVecs_D.resize(0,0,0);svdVecs_Vdagger0.resize(0,0);
    }
    /////////////////////////////////
    // Restore from UDV
    /////////////////////////////////
    K.resize(L, L);
    svdBg.resize(svdNumber); 

#ifdef MPI_HAO
    size_t dims[3] = {L, L, svdNumber};
    if(is_local_root) {
        svdVecs.createSharedMemory(dims, 0, node_comm);
    }
    else {
        svdVecs.createSharedMemoryView(0, node_comm);
        svdVecs.attachToSharedMemory(dims, 0, node_comm);
    }
#endif

    if(is_local_root) {
        if(truncatedD > 0 && svdNumber==truncatedSvdNumber){
            TensorHao<complex<double>, 2> svdVecs_D_matrix(truncatedD,truncatedD);
            TensorHao<complex<double>, 2> matrixTemp(L,truncatedD);
            TensorHao<complex<double>, 2> matrix(L,L);
            for(size_t k=1-1; k<=svdNumber-1; k++){
                for(size_t i=1-1; i<=truncatedD-1; i++){
                for(size_t j=1-1; j<=truncatedD-1; j++){
                    svdVecs_D_matrix(i,j) = svdVecs_D(i,j,k);
                }
                }
                BL_NAME(gmm)(svdVecs_U0, svdVecs_D_matrix, matrixTemp);
                BL_NAME(gmm)(matrixTemp, svdVecs_Vdagger0, matrix);
                // 
                for(size_t i=1-1; i<=L-1; i++){
                for(size_t j=1-1; j<=L-1; j++){
                    svdVecs(i,j,k) = matrix(i,j);
                }
                }
            }
        }else{
            //get full svd from input
            for(size_t k=1-1; k<=svdNumber-1; k++){
                for(size_t i=1-1; i<=L-1; i++){
                for(size_t j=1-1; j<=L-1; j++){
                    svdVecs(i,j,k)=svdVecs_r(j,i,svdNumber-1-k)+complex<double>(0.0,-1.0)*svdVecs_i(j,i,svdNumber-1-k);
                }
                }
            }
        }
    }
    MPIBarrier();
    // 
    if(truncatedD > 0 && svdNumber==truncatedSvdNumber){
        TensorHao<complex<double>, 2> matrixTemp(L,truncatedD);
        TensorHao<complex<double>, 2> matrix(L,L);
        BL_NAME(gmm)(K_U, K_D, matrixTemp);
        BL_NAME(gmm)(matrixTemp, K_Vdagger, matrix);
        // 
        K = matrix;
    }else{
        for(size_t i=1-1; i<=L-1; i++){
        for(size_t j=1-1; j<=L-1; j++){
            K(i,j)=K_r(j,i)+complex<double>(0.0,-1.0)*K_i(j,i);
        } 
        } 
    }
    //
    for(size_t k=1-1; k<=svdNumber-1; k++){
        svdBg(k)=svdBg_r(svdNumber-1-k)+complex<double>(0.0,-1.0)*svdBg_i(svdNumber-1-k);
    }
    initialBg=svdBg;

    KpEigenStatus = 0;
    Kp.resize(0,0);
    KpEigenValue.resize( static_cast<size_t>(0) );
    KpEigenVector.resize( 0, 0 );
    #ifdef MPI_HAO
    MPI_Comm_free(&node_comm);
    #endif
}

void GeneralHamiltonian_icf::write(const string &filename) const
{
    // //
    // tensor_hao::TensorHao<double,2> K_rTemp,K_iTemp;
    // K_rTemp.resize(L,L);K_iTemp.resize(L,L);
    // for(size_t i=1-1; i<=L-1; i++){
    // for(size_t j=1-1; j<=L-1; j++){
    //     K_rTemp(i,j)=real(K(i,j));
    //     K_iTemp(i,j)=imag(K(i,j));
    // }
    // }
    // //
    // tensor_hao::TensorHao<double,3> svdVecs_rTemp,svdVecs_iTemp;
    // svdVecs_rTemp.resize(L,L,svdNumber);svdVecs_iTemp.resize(L,L,svdNumber);
    // for(size_t i=1-1; i<=L-1; i++){
    // for(size_t j=1-1; j<=L-1; j++){
    // for(size_t kk=1-1; kk<=svdNumber-1; kk++){
    //     svdVecs_rTemp(i,j,kk)=real(svdVecs(i,j,kk));
    //     svdVecs_iTemp(i,j,kk)=imag(svdVecs(i,j,kk));
    // }
    // }
    // }
    // //
    // tensor_hao::TensorHao<double,1> svdBg_rTemp,svdBg_iTemp;
    // svdBg_rTemp.resize(svdNumber);svdBg_iTemp.resize(svdNumber);
    // for(size_t kk=1-1; kk<=svdNumber-1; kk++){
    //     svdBg_rTemp(kk)=real(svdBg(kk));
    //     svdBg_iTemp(kk)=imag(svdBg(kk));
    // }

    // H5File file(filename, H5F_ACC_TRUNC);

    // // writeFile( k_l_x, file, "k_l_x" );
    // // writeFile( k_l_y, file, "k_l_y" );
    // writeFile( halfL, file, "halfL" );
    // writeFile( L, file, "L" );
    // writeFile( Nup, file, "Nup" );
    // writeFile( Ndn, file, "Ndn" );
    // writeFile( N, file, "N" );
    // writeFile( svdNumber, file, "svdNumber" );
    // writeFile( K.size(),  K_rTemp.data(),  file, "K_r" );
    // writeFile( K.size(),  K_iTemp.data(),  file, "K_i" );
    // writeFile( svdVecs.size(), svdVecs_rTemp.data(), file, "svdVecs_r" );
    // writeFile( svdVecs.size(), svdVecs_iTemp.data(), file, "svdVecs_i" );
    // writeFile( svdBg.size(), svdBg_rTemp.data(), file, "svdBg_r" );
    // writeFile( svdBg.size(), svdBg_iTemp.data(), file, "svdBg_i" );

    // file.close();
}

#ifdef MPI_HAO
// void MPIBcast(GeneralHamiltonian_icf &buffer, int root, MPI_Comm const &comm)
// {
//     // MPIBcast(buffer.k_l_x, root, comm);
//     // MPIBcast(buffer.k_l_y, root, comm);
//     MPIBcast(buffer.L, root, comm);
//     MPIBcast(buffer.SD2sL, root, comm);
//     MPIBcast(buffer.Hamiltonian_spin_flag, root, comm);
//     MPIBcast(buffer.truncatedDup, root, comm);
//     MPIBcast(buffer.truncatedDdn, root, comm);
//     MPIBcast(buffer.truncatedD, root, comm);
//     MPIBcast(buffer.Nup, root, comm);
//     MPIBcast(buffer.Ndn, root, comm);
//     MPIBcast(buffer.N, root, comm);
//     MPIBcast(buffer.svdNumber, root, comm);
//     MPIBcast(buffer.truncatedSvdNumber, root, comm);
//     MPIBcast(buffer.K, root, comm);
//     MPIBcast(buffer.K_U, root, comm);
//     MPIBcast(buffer.K_D, root, comm);
//     MPIBcast(buffer.K_Vdagger, root, comm);
//     MPIBcast(buffer.svdVecs, root, comm);
//     MPIBcast(buffer.svdVecs_U0, root, comm);
//     MPIBcast(buffer.svdVecs_D, root, comm);
//     MPIBcast(buffer.svdVecs_Vdagger0, root, comm);
//     MPIBcast(buffer.svdBg, root, comm);
//     MPIBcast(buffer.initialBg, root, comm);
//     MPIBcast(buffer.KpEigenStatus, root, comm);
//     MPIBcast(buffer.Kp, root, comm);
//     MPIBcast(buffer.KpEigenValue, root, comm);
//     MPIBcast(buffer.KpEigenVector, root, comm);
// }

void MPIBcastShared(GeneralHamiltonian_icf &buffer, int root, MPI_Comm const &comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Broadcast scalar data
    MPIBcast(buffer.L, root, comm);
    MPIBcast(buffer.SD2sL, root, comm);
    MPIBcast(buffer.Hamiltonian_spin_flag, root, comm);
    MPIBcast(buffer.truncatedDup, root, comm);
    MPIBcast(buffer.truncatedDdn, root, comm);
    MPIBcast(buffer.truncatedD, root, comm);
    MPIBcast(buffer.Nup, root, comm);
    MPIBcast(buffer.Ndn, root, comm);
    MPIBcast(buffer.N, root, comm);
    MPIBcast(buffer.svdNumber, root, comm);
    MPIBcast(buffer.truncatedSvdNumber, root, comm);
    MPIBcast(buffer.KpEigenStatus, root, comm);

    // Broadcast small tensors
    MPIBcast(buffer.K, root, comm);
    MPIBcast(buffer.K_U, root, comm);
    MPIBcast(buffer.K_D, root, comm);
    MPIBcast(buffer.K_Vdagger, root, comm);
    MPIBcast(buffer.svdVecs_U0, root, comm);
    MPIBcast(buffer.svdVecs_Vdagger0, root, comm);
    MPIBcast(buffer.svdVecs_D, root, comm);
    MPIBcast(buffer.svdBg, root, comm);
    MPIBcast(buffer.initialBg, root, comm);
    MPIBcast(buffer.Kp, root, comm);
    MPIBcast(buffer.KpEigenValue, root, comm);
    MPIBcast(buffer.KpEigenVector, root, comm);

    // svdVecs is already in shared memory from read(), other ranks attach to it
    if(rank != root) {
        size_t dims[3] = {buffer.L, buffer.L, buffer.svdNumber};
        buffer.svdVecs.createSharedMemoryView(root, comm);
        buffer.svdVecs.attachToSharedMemory(dims, root, comm);
    }
}
#endif

void GeneralHamiltonian_icf::writeBackGround(const string &filename) const
{
    H5File file(filename, H5F_ACC_RDWR);
    writeFile( svdBg.size(), svdBg.data(), file, "svdBg" );
    file.close();
}

void GeneralHamiltonian_icf::updateBackGround(const TensorHao<complex<double>, 1> &background)
{
    if( background.size() != svdNumber ) {cout<<"Error!!! Background size is not svdNumber!"<<endl; exit(1);}
    KpEigenStatus = 0;
    svdBg = background;
}

void GeneralHamiltonian_icf::updateBackGround(TensorHao<complex<double>, 1> &&background)
{
    if( background.size() != svdNumber ) {cout<<"Error!!! Background size is not svdNumber!"<<endl; exit(1);}
    KpEigenStatus = 0;
    svdBg = move(background);
}

Hop GeneralHamiltonian_icf::returnExpMinusAlphaK(double alpha)
{
    setKpEigenValueAndVector();

    if(Hamiltonian_spin_flag==true){
        TensorHao<complex<double>,2> matrix(L,L);
        BL_NAME(gmm)( KpEigenVector, dMultiMatrix( exp(-alpha*KpEigenValue), conj(trans(KpEigenVector) )), matrix );

        Hop hop(L);
        complex<double> bg2(0.0); for(size_t i = 0; i < svdNumber; ++i) bg2 += svdBg(i) * svdBg(i);
        hop.logw = alpha*0.5*bg2;
        for(size_t i = 0; i < L; ++i)
        {
            for(size_t j = 0; j < L; ++j) hop.matrix(j,i) = matrix(j,i);
        }
        return hop;
    }else{ 
        TensorHao<complex<double>,2> matrix(L,L);
        BL_NAME(gmm)( KpEigenVector, dMultiMatrix( exp(-alpha*KpEigenValue), conj(trans(KpEigenVector) )), matrix );

        Hop hop(2*L);
        complex<double> bg2(0.0); for(size_t i = 0; i < svdNumber; ++i) bg2 += svdBg(i) * svdBg(i);
        hop.logw = alpha*0.5*bg2;
        hop.matrix = 0.0;
        for(size_t i = 0; i < L; ++i)
        {
            for(size_t j = 0; j < L; ++j){
                hop.matrix(j,i) = matrix(j,i);
                hop.matrix(j+L,i+L) = matrix(j,i);
            } 
        }
        return hop;
    }
}

Hop2s GeneralHamiltonian_icf::returnExpMinusAlphaK2s(double alpha)
{
    setKpEigenValueAndVector();

    TensorHao<complex<double>,2> matrix(L,L);
    BL_NAME(gmm)( KpEigenVector, dMultiMatrix( exp(-alpha*KpEigenValue), conj(trans(KpEigenVector) )), matrix );

    if(Hamiltonian_spin_flag==true){
        Hop2s hop2s(SD2sL);
        complex<double> bg2(0.0); for(size_t i = 0; i < svdNumber; ++i) bg2 += svdBg(i) * svdBg(i);
        hop2s.logw = alpha*0.5*bg2;
        for(size_t i = 0; i < SD2sL; ++i)
        {
            for(size_t j = 0; j < SD2sL; ++j) hop2s.matrixUp(j,i) = matrix(j,i);
            for(size_t j = 0; j < SD2sL; ++j) hop2s.matrixDn(j,i) = matrix(j+SD2sL,i+SD2sL);
        }
        return hop2s;
    }else{
        Hop2s hop2s(SD2sL);
        complex<double> bg2(0.0); for(size_t i = 0; i < svdNumber; ++i) bg2 += svdBg(i) * svdBg(i);
        hop2s.logw = alpha*0.5*bg2;
        for(size_t i = 0; i < SD2sL; ++i)
        {
            for(size_t j = 0; j < SD2sL; ++j) hop2s.matrixUp(j,i) = matrix(j,i);
            for(size_t j = 0; j < SD2sL; ++j) hop2s.matrixDn(j,i) = matrix(j,i);
        }
        return hop2s;
    }

}

Hop GeneralHamiltonian_icf::returnExpMinusAlphaK_nonHermitian(double alpha, std::string flag, size_t taylorOrder, double accuracy, size_t baseTaylorOrder)
{
    if(Hamiltonian_spin_flag==true){
        //generate Hop from LogHop for nonHermitian Kp
        LogHop logHop=returnLogExpMinusAlphaK(alpha);
        //Aux walker to store nonHermitian expm
        SD walkerIdentity(L,L);
        TensorHao<complex<double>,2> identity(L,L); identity=0.0;
        for(size_t i = 0; i < L; ++i)
        {
            identity(i,i) = 1.0; 
        }
        walkerIdentity.wfRef()=identity;
        SD walkerTemp=walkerIdentity;
        //
        Hop hop(L);
        //
        LogHopSDOperation_icf oneBodyWalkerOperation;
        oneBodyWalkerOperation.reset(flag, taylorOrder, accuracy, baseTaylorOrder);
        oneBodyWalkerOperation.applyToRight(logHop, walkerIdentity, walkerTemp);
        //
        hop.matrix = walkerTemp.getWf();
        hop.logw = walkerTemp.getLogw();
        //
        return hop;
    }else{
        //generate Hop from LogHop for nonHermitian Kp
        LogHop logHop=returnLogExpMinusAlphaK(alpha);
        //Aux walker to store nonHermitian expm
        SD walkerIdentity(L,L);
        TensorHao<complex<double>,2> identity(L,L); identity=0.0;
        for(size_t i = 0; i < L; ++i)
        {
            identity(i,i) = 1.0; 
        }
        walkerIdentity.wfRef()=identity;
        SD walkerTemp=walkerIdentity;
        //
        Hop hop(2*L);
        //
        LogHopSDOperation_icf oneBodyWalkerOperation;
        oneBodyWalkerOperation.reset(flag, taylorOrder, accuracy, baseTaylorOrder);
        oneBodyWalkerOperation.applyToRight(logHop, walkerIdentity, walkerTemp);
        //
        for(size_t i = 0; i < L; ++i)
        {
            for(size_t j = 0; j < L; ++j){
                hop.matrix(j,i) = walkerTemp.getWf()(j,i);
                hop.matrix(j+L,i+L) = walkerTemp.getWf()(j,i);
            } 
        }
        hop.logw = walkerTemp.getLogw();
        //
        return hop;
    }
}

Hop2s GeneralHamiltonian_icf::returnExpMinusAlphaK2s_nonHermitian(double alpha, std::string flag, size_t taylorOrder, double accuracy, size_t baseTaylorOrder)
{
    //generate Hop from LogHop for nonHermitian Kp
    LogHop2s logHop2s=returnLogExpMinusAlphaK2s(alpha);
    //Aux walker to store nonHermitian expm
    SD2s walkerIdentity(SD2sL,SD2sL,SD2sL);
    TensorHao<complex<double>,2> identity(SD2sL,SD2sL); identity=0.0;
    for(size_t i = 0; i < SD2sL; ++i)
    {
        identity(i,i) = 1.0; 
    }
    walkerIdentity.wfUpRef()=identity;
    walkerIdentity.wfDnRef()=identity;
    SD2s walkerTemp=walkerIdentity;
    //
    Hop2s hop2s(SD2sL);
    //
    LogHop2sSD2sOperation oneBodyWalkerOperation;
    oneBodyWalkerOperation.reset(flag, taylorOrder, accuracy, baseTaylorOrder);
    oneBodyWalkerOperation.applyToRight(logHop2s, walkerIdentity, walkerTemp);
    //
    hop2s.matrixUp = walkerTemp.getWfUp();
    hop2s.matrixDn = walkerTemp.getWfDn();
    hop2s.logw = walkerTemp.getLogw();
    //
    return hop2s;
}

LogHop GeneralHamiltonian_icf::returnLogExpMinusAlphaK(double alpha)
{
    setKp();

    if(Hamiltonian_spin_flag==true){
        LogHop logHop(L);
        complex<double> bg2(0.0); for(size_t i = 0; i < svdNumber; ++i) bg2 += svdBg(i) * svdBg(i);
        logHop.logw = alpha*0.5*bg2;
        for(size_t i = 0; i < L; ++i)
        {
            for(size_t j = 0; j < L; ++j) logHop.matrix(j,i) = -alpha*Kp(j,i);
        }
        return logHop;
    }else{
        LogHop logHop(2*L);
        complex<double> bg2(0.0); for(size_t i = 0; i < svdNumber; ++i) bg2 += svdBg(i) * svdBg(i);
        logHop.logw = alpha*0.5*bg2;
        for(size_t i = 0; i < L; ++i)
        {
            for(size_t j = 0; j < L; ++j){
                logHop.matrix(j,i) = -alpha*Kp(j,i);
                logHop.matrix(j+L,i+L) = -alpha*Kp(j,i);
            } 
        }
        return logHop;
    }
}

LogHop2s GeneralHamiltonian_icf::returnLogExpMinusAlphaK2s(double alpha)
{
    setKp();

    if(Hamiltonian_spin_flag==true){
        LogHop2s logHop2s(SD2sL);
        complex<double> bg2(0.0); for(size_t i = 0; i < svdNumber; ++i) bg2 += svdBg(i) * svdBg(i);
        logHop2s.logw = alpha*0.5*bg2;
        for(size_t i = 0; i < SD2sL; ++i)
        {
            for(size_t j = 0; j < SD2sL; ++j) logHop2s.matrixUp(j,i) = -alpha*Kp(j,i);
            for(size_t j = 0; j < SD2sL; ++j) logHop2s.matrixDn(j,i) = -alpha*Kp(j+SD2sL,i+SD2sL);
        }
        return logHop2s;
    }else{
        LogHop2s logHop2s(SD2sL);
        complex<double> bg2(0.0); for(size_t i = 0; i < svdNumber; ++i) bg2 += svdBg(i) * svdBg(i);
        logHop2s.logw = alpha*0.5*bg2;
        for(size_t i = 0; i < SD2sL; ++i)
        {
            for(size_t j = 0; j < SD2sL; ++j) logHop2s.matrixUp(j,i) = -alpha*Kp(j,i);
            for(size_t j = 0; j < SD2sL; ++j) logHop2s.matrixDn(j,i) = -alpha*Kp(j,i);
        }
        return logHop2s;
    }
}

SVD GeneralHamiltonian_icf::returnExpMinusAlphaV(double alpha, bool cutoffFlag)
{
    return SVD(alpha, svdVecs, svdVecs_U0, svdVecs_D, svdVecs_Vdagger0, svdBg, Hamiltonian_spin_flag, cutoffFlag);//ATTENTION: default cutoffFlag=false
}

SVD GeneralHamiltonian_icf::returnExpMinusAlphaV_daggerSqrtDt(double alpha, bool cutoffFlag)
{
    return SVD(alpha, svdVecs, svdVecs_U0, svdVecs_D, svdVecs_Vdagger0, svdBg, Hamiltonian_spin_flag, -1, cutoffFlag);//ATTENTION: extra -1 here for sqrt(-1*dt) in SVD
}


void GeneralHamiltonian_icf::setKp()
{
    if( KpEigenStatus >=1 ) return;

    size_t L2 = L*L;

    Kp = K;
    TensorHaoRef<complex<double>,2> vecs(L2, svdNumber); vecs.point( svdVecs.data() );
    TensorHaoRef<complex<double>,1> vecsBg(L2); vecsBg.point( Kp.data() );
    BL_NAME(gemv)(vecs, svdBg, vecsBg, 'N', 1.0, 1.0);
    
    KpEigenStatus=1;

    // 
    if(SD2sL == L/2){
        Hamiltonian_spin_flag = true;
        if( SD2sL * 2 != L){
            cout<<"Error in GeneralHamiltonian_icf::read: L is not even!"<<endl;
            exit(1);
        }
    }else if(SD2sL == L){
        Hamiltonian_spin_flag = false;
    }else{
        cout<<"Error in GeneralHamiltonian_icf::read: SD2sL is not equal to L or L/2!"<<endl;
        exit(1);
    }
    // 
}

void GeneralHamiltonian_icf::setKpEigenValueAndVector()
{
    if( KpEigenStatus >=2 ) return;

    setKp();
    checkHermitian(Kp, 1e-8);
    KpEigenVector = Kp;
    KpEigenValue.resize(L);
    BL_NAME(eigen)(KpEigenVector, KpEigenValue);

    KpEigenStatus = 2;
}

double GeneralHamiltonian_icf::getMemory() const
{
    double mem(0.0);

    mem += 8.0*10;
    mem += 1.0;
    // 
    mem += K.getMemory();
    mem += K_U.getMemory();
    mem += K_D.getMemory();
    mem += K_Vdagger.getMemory();
    // 
    mem += svdVecs.getMemory();
    mem += svdVecs_U0.getMemory();
    mem += svdVecs_D.getMemory();
    mem += svdVecs_Vdagger0.getMemory();
    mem += svdBg.getMemory();
    mem += initialBg.getMemory();
    // 
    mem += 8.0;
    mem += Kp.getMemory();
    mem += KpEigenValue.getMemory();
    mem += KpEigenVector.getMemory();

    return mem;
}

GeneralHamiltonian_icf::GeneralHamiltonian_icf(const GeneralHamiltonian_icf &x)  { }

GeneralHamiltonian_icf &GeneralHamiltonian_icf::operator=(const GeneralHamiltonian_icf &x) { return *this; }

////////////////////////////////////////////
////////////////////////////////////////////
///////////////////////////////////////////