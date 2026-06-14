//

#ifndef LEAPFROG_H
#define LEAPFROG_H

#include "afqmclab.h"
#include "JastrowProjector.h"
#include "metropolisDefine.h"
#include "metropolisMethod.h"

class LeapFrog
{
 public:
    LeapFrog(SD walkerLeft_input_temp, SD walkerRight_input_temp, JastrowProjector *jastrowProjector_input, int j_Jastrow_input);
    ~LeapFrog();
    //
    SD walkerLeft_input, walkerRight_input;
    SD walkerLeft, walkerRight;
    //
    int Gamma; // # of SVD vectors 
    int L;     // size of systems
    int N;     // number of electrons
    int j_Jastrow;
    JastrowProjector *jastrowProjector;
    //
    TwoBodyAux_Jastrow x_initial,p_initial;
    TwoBodyAux_Jastrow x,p;
    TwoBodyAux_Jastrow x_final,p_final;
    //
    std::complex<double> U_x_initial, K_p_initial;
    std::complex<double> U_x, K_p;
    std::complex<double> U_x_final, K_p_final;
    //
    tensor_hao::TensorHao<std::complex<double>,1> grad_U_x_initial, grad_U_x, grad_U_x_final;
    // 
    //////////////////////////////////////////
    //To save computational cost
    // tensor_hao::TensorHao<std::complex<double>,2> svd_i; 
    TwoBodySample_Jastrow twoBodySample_x;
    std::vector<tensor_hao::TensorHao<std::complex<double>,2>> svd;
    tensor_hao::TensorHao<std::complex<double>,2> matrix_sum;
    tensor_hao::TensorHao<std::complex<double>,1> sqrtMinusDt_svdBg; 
    //////////////////////////////////////////
    //
    SDSDOperation walkerWalkerOperation;
    HopSDOperation oneBodyWalkerRightOperation;
    HopSDOperation oneBodyWalkerLeftOperation;
    SVDSampleSDOperation twoBodySampleWalkerRightOperation;
    SVDSampleSDOperation twoBodySampleWalkerLeftOperation;
    //
    void setInitial(TwoBodyAux_Jastrow aux_x_input, TwoBodyAux_Jastrow aux_p_input);
    void leap(double dt, int steps);
    void leap_Random(double dt, int steps);
    std::complex<double> get_logDet(tensor_hao::TensorHao<std::complex<double>,2> matrix_input);
    tensor_hao::TensorHao<std::complex<double>,1> get_grad_minus_logOverlap(TwoBodyAux_Jastrow aux_x);
    tensor_hao::TensorHao<std::complex<double>,1> get_grad_minus_logWeight(TwoBodyAux_Jastrow aux_x);
    tensor_hao::TensorHao<std::complex<double>,1> get_grad_det(TwoBodyAux_Jastrow aux_x);
    tensor_hao::TensorHao<std::complex<double>,1> get_grad_det_inv_det_expM_fast(TwoBodyAux_Jastrow aux_x);
    tensor_hao::TensorHao<std::complex<double>,1> get_grad_minus_logOverlap_num(TwoBodyAux_Jastrow aux_x);
    tensor_hao::TensorHao<std::complex<double>,1> get_grad_minus_logWeight_num(TwoBodyAux_Jastrow aux_x);
    tensor_hao::TensorHao<std::complex<double>,1> get_grad_det_num(TwoBodyAux_Jastrow aux_x);
    std::complex<double> get_minus_logOverlap(TwoBodyAux_Jastrow aux_x);
    std::complex<double> get_minus_logOverlap_fast();
    std::complex<double> get_minus_logWeight(TwoBodyAux_Jastrow aux_x);
    tensor_hao::TensorHao<std::complex<double>,2> get_expMatrix(TwoBodyAux_Jastrow aux_x);
    tensor_hao::TensorHao<std::complex<double>,2> get_expMatrix_fast();
    tensor_hao::TensorHao<std::complex<double>,2> get_secondMatrix_di(TwoBodyAux_Jastrow aux_x_input, int i);
    tensor_hao::TensorHao<std::complex<double>,2> get_secondSVD_di(TwoBodyAux_Jastrow aux_x_input, int i);
    //
    std::complex<double> get_square(TwoBodyAux_Jastrow aux_p_input);
    tensor_hao::TensorHao<std::complex<double>,2> matrixProduct(tensor_hao::TensorHao<std::complex<double>,2> matrix_L,  tensor_hao::TensorHao<std::complex<double>,2> matrix_M, tensor_hao::TensorHao<std::complex<double>,2> matrix_R, char L_s, char M_s, char R_s);
    // tensor_hao::TensorHao<std::complex<double>,2> matrixProduct(tensor_hao::TensorHao<std::complex<double>,2> matrix_L,  tensor_hao::TensorHao<std::complex<double>,2> matrix_M, tensor_hao::TensorHao<std::complex<double>,2> matrix_R, char L_s, char M_s, char M_s);
};

#endif //JASTROWPROJECTOR_H
