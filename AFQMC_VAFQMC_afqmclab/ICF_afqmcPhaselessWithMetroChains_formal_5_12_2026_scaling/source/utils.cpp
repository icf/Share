#include <iostream>
#include <fstream>
#include <string>

#include "../include/utils.h"

using namespace std;
using namespace tensor_hao;

long get_memory_usage_linux() {
    std::ifstream status_file("/proc/self/status");
    std::string line;
    long rss_kb = 0;

    if (!status_file.is_open()) {
        std::cerr << "Failed to open /proc/self/status" << std::endl;
        return -1;
    }

    // 逐行解析，找到VmRSS字段
    while (std::getline(status_file, line)) {
        if (line.substr(0, 5) == "VmRSS") {
            // 格式：VmRSS:  12345 kB
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                rss_kb = std::stol(line.substr(pos + 1));
                break;
            }
        }
    }

    status_file.close();
    return rss_kb;
}

bool checkMatrixDiff(TensorHao<complex<double>,2> matrixInputA, TensorHao<complex<double>,2> matrixInputB, double accurancy){
    for(int i = 1-1; i <= matrixInputA.rank(0)-1; i++){
        for(int j = 1-1; j <= matrixInputA.rank(1)-1; j++){
            if( abs(matrixInputA(i,j) - matrixInputB(i,j)) >= accurancy ){
                cout<<"Error in checkMatrixDiff: "<<i<<" "<<j<<" "<<matrixInputA(i,j)<<" "<<matrixInputB(i,j)<<endl;
                return false;
            }
        }
    }
    return true;
}

TensorHao<std::complex<double>, 2> expMatrix(TensorHao<std::complex<double>, 2> A, int b) {
    if (b <= 0) {
        cout << "Error in expM: expansion order b must be positive" << endl;
        exit(1);
    }
    
    if (A.rank(0) != A.rank(1)) {
        cout << "Error in expM: matrix A must be square" << endl;
        exit(1);
    }
    
    size_t n = A.rank(0);
    
    // Create identity matrix
    TensorHao<std::complex<double>, 2> result(n, n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            result(i, j) = (i == j) ? 1.0 : 0.0;
        }
    }
    
    // Add A term
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            result(i, j) += A(i, j);
        }
    }
    
    if (b >= 2) {
        // Calculate A^2
        TensorHao<std::complex<double>, 2> A_power(n, n);
        BL_NAME(gmm)(A, A, A_power);
        
        // Add A^2/2! term
        double factor = 0.5;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                result(i, j) += factor * A_power(i, j);
            }
        }
        
        // For higher order terms
        for (int k = 3; k <= b; k++) {
            TensorHao<std::complex<double>, 2> A_power_next(n, n);
            BL_NAME(gmm)(A, A_power, A_power_next);
            A_power = A_power_next;
            
            // Calculate 1/k! factor
            factor /= k;
            
            // Add A^k/k! term
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    result(i, j) += factor * A_power(i, j);
                }
            }
        }
    }
    
    return result;
}

void get_deltaMatrix_1s(TensorHao<complex<double>,2> & deltaMatrix, int expM, string order, TensorHao<complex<double>,2> wfnLeft_matrix, TensorHao<complex<double>,2> U_s, TensorHao<complex<double>,2> Vdagger_s, TensorHao<complex<double>,2> old_A, TensorHao<complex<double>,2> k_matrix, TensorHao<complex<double>,2> wfnRight_matrix){
    if(expM != 1){
        cout<<"Error in get_deltaMatrix_1s: expM != 1"<<endl;
        exit(1);
    } 

    deltaMatrix.resize(wfnLeft_matrix.rank(1), wfnLeft_matrix.rank(1));
    
    
    if(order == "VK"){
        // Create temporary matrices for intermediate results
        TensorHao<complex<double>,2> temp_save(wfnLeft_matrix.rank(1), U_s.rank(1));
        TensorHao<complex<double>,2> temp_save2(wfnLeft_matrix.rank(1), Vdagger_s.rank(1));
        TensorHao<complex<double>,2> temp_save3(wfnLeft_matrix.rank(1), k_matrix.rank(1));
        BL_NAME(gmm)(trans(conj(wfnLeft_matrix)), U_s, temp_save);
        BL_NAME(gmm)(temp_save, Vdagger_s, temp_save2);
        BL_NAME(gmm)(temp_save2, k_matrix, temp_save3);
        BL_NAME(gmm)(temp_save3, wfnRight_matrix, deltaMatrix);
    }else if(order == "K^daggerV"){        
        // Create temporary matrices for intermediate results
        TensorHao<complex<double>,2> temp_save(wfnLeft_matrix.rank(1), U_s.rank(1));
        TensorHao<complex<double>,2> temp_save2(wfnLeft_matrix.rank(1), U_s.rank(1));
        TensorHao<complex<double>,2> temp_save3(wfnLeft_matrix.rank(1), Vdagger_s.rank(1));
        BL_NAME(gmm)(trans(conj(wfnLeft_matrix)), trans(conj(k_matrix)), temp_save);
        BL_NAME(gmm)(temp_save, U_s, temp_save2);
        BL_NAME(gmm)(temp_save2, Vdagger_s, temp_save3);
        BL_NAME(gmm)(temp_save3, wfnRight_matrix, deltaMatrix);
    }else{
        cout<<"Error in get_deltaMatrix_1s: order: "<<order<<endl;
        exit(1);
    }
}

void get_updatedUV_1s(vector<TensorHao<complex<double>,2>> & U_vec, vector<TensorHao<complex<double>,2>> & Vdagger_vec, int expM, string order, TensorHao<complex<double>,2> wfnLeft_matrix, TensorHao<complex<double>,2> U_s, TensorHao<complex<double>,2> Vdagger_s, TensorHao<complex<double>,2> old_B, TensorHao<complex<double>,2> k_matrix, TensorHao<complex<double>,2> wfnRight_matrix){
    if(expM != 1 && expM != 2){
        cout<<"Error in get_updatedUV_up: expM != 1 && expM != 2"<<endl;
        exit(1);
    } 

    if(expM == 1){
        U_vec.resize(1);
        Vdagger_vec.resize(1);
        U_vec[0].resize(wfnLeft_matrix.rank(1), U_s.rank(1));
        Vdagger_vec[0].resize(Vdagger_s.rank(0), wfnRight_matrix.rank(1));
        if(order == "VK"){
            // U = wfnLeft_matrix * U_s;
            BL_NAME(gmm)(trans(conj(wfnLeft_matrix)), U_s, U_vec[0]);
            
            // temp_save = k_matrix * wfnRight_matrix;
            TensorHao<complex<double>,2> temp_save(k_matrix.rank(0), wfnRight_matrix.rank(1));
            BL_NAME(gmm)(k_matrix, wfnRight_matrix, temp_save);
            
            // Vdagger = Vdaggerup_s * temp_save;
            BL_NAME(gmm)(Vdagger_s, temp_save, Vdagger_vec[0]);
        }else if(order == "K^daggerV"){

        }else{
            cout<<"Error in get_updatedUV_up: order: "<<order<<endl;
            exit(1);
        }
    }else if(expM == 2){
        U_vec.resize(2);
        Vdagger_vec.resize(2);
        // 
        TensorHao<complex<double>,2> k_wfnRight(k_matrix.rank(0), wfnRight_matrix.rank(1));
        BL_NAME(gmm)(k_matrix, wfnRight_matrix, k_wfnRight);
        // 
        for(int i=1-1; i<=2-1; i++){
            U_vec[i].resize(wfnLeft_matrix.rank(1), U_s.rank(1));
            Vdagger_vec[i].resize(Vdagger_s.rank(0), k_wfnRight.rank(1));
            if(order == "VK"){
                if( i == 0 ){
                    /////////////////////////////////////////
                    // U_vec[0] = wfnLeft_matrix * U_s;
                    /////////////////////////////////////////
                    BL_NAME(gmm)(trans(conj(wfnLeft_matrix)), U_s, U_vec[0]);
                    // 
                    /////////////////////////////////////////
                    // temp_save = I + 0.5 * ( U_s * Vdagger_s + old_B)
                    // temp_save2 = Vdagger_s * temp_save;
                    // Vdagger_vec[0] = temp_save2 * k_wfnRight
                    /////////////////////////////////////////
                    TensorHao<complex<double>,2> temp_save(U_s.rank(0), Vdagger_s.rank(1));
                    BL_NAME(gmm)(U_s, Vdagger_s, temp_save);
                    
                    // Add old_B to temp_save and multiply by 0.5
                    for(size_t i_elem = 0; i_elem < temp_save.size(); i_elem++) {
                        temp_save.data()[i_elem] = 0.5 * (temp_save.data()[i_elem] + old_B.data()[i_elem]);
                    }
                    
                    // Add identity matrix
                    for(size_t i_diag = 0; i_diag < min(temp_save.rank(0), temp_save.rank(1)); i_diag++) {
                        temp_save(i_diag, i_diag) += 1.0;
                    }
                    
                    // temp_save2 = Vdagger_s * temp_save
                    TensorHao<complex<double>,2> temp_save2(Vdagger_s.rank(0), temp_save.rank(1));
                    BL_NAME(gmm)(Vdagger_s, temp_save, temp_save2);
                    
                    // Vdagger_vec[0] = temp_save2 * k_wfnRight
                    BL_NAME(gmm)(temp_save2, k_wfnRight, Vdagger_vec[0]);
                }else if( i == 1 ){
                    /////////////////////////////////////////
                    // U_vec[1] = 0.5 * wfnLeft_matrix * old_B * U_s;
                    /////////////////////////////////////////
                    TensorHao<complex<double>,2> temp_oldB_U(old_B.rank(0), U_s.rank(1));
                    BL_NAME(gmm)(old_B, U_s, temp_oldB_U);
                    
                    TensorHao<complex<double>,2> temp_wfnLeft_oldB_U(wfnLeft_matrix.rank(1), temp_oldB_U.rank(1));
                    BL_NAME(gmm)(trans(conj(wfnLeft_matrix)), temp_oldB_U, temp_wfnLeft_oldB_U);
                    
                    // Multiply by 0.5
                    for(size_t i_elem = 0; i_elem < temp_wfnLeft_oldB_U.size(); i_elem++) {
                        temp_wfnLeft_oldB_U.data()[i_elem] *= 0.5;
                    }
                    U_vec[1] = temp_wfnLeft_oldB_U;
                    /////////////////////////////////////////
                    // Vdagger_vec[1] = Vdagger_s * k_wfnRight
                    /////////////////////////////////////////
                    BL_NAME(gmm)(Vdagger_s, k_wfnRight, Vdagger_vec[1]);
                }
            }else if(order == "K^daggerV"){
            }else{
                cout<<"Error in get_updatedUV_up: order: "<<order<<endl;
                exit(1);
            }
        }
    }
}