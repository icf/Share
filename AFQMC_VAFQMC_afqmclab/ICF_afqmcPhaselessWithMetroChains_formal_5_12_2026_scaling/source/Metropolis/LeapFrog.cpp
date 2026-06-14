//
//
#include <chrono>
#include "../../include/Metropolis/LeapFrog.h"

using namespace std;
using namespace tensor_hao;

#define pi 3.14159265358979324

LeapFrog::LeapFrog(SD walkerLeft_input_temp, SD walkerRight_input_temp, JastrowProjector *jastrowProjector_input, int j_Jastrow_input) {
    walkerLeft_input = walkerLeft_input_temp;
    walkerRight_input = walkerRight_input_temp;
    jastrowProjector = jastrowProjector_input;
    j_Jastrow = j_Jastrow_input;
    // 
    L = walkerLeft_input.getWf().rank(0);
    N = walkerLeft_input.getWf().rank(1);
    matrix_sum.resize(L,L);
    // 
    complex<double> sqrtMinusDt = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getSqrtMinusDt();
    TensorHao<complex<double>, 1> svdBg = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getSVDBg(); 
    //
    sqrtMinusDt_svdBg.resize(svdBg.size());
    for(int i=1-1; i<=svdBg.size()-1; i++){
        sqrtMinusDt_svdBg(i) = sqrtMinusDt * svdBg(i);
    }
}

LeapFrog::~LeapFrog() { }

void LeapFrog::setInitial(TwoBodyAux_Jastrow aux_x_input, TwoBodyAux_Jastrow aux_p_input)
{
    x_initial=aux_x_input;
    p_initial=aux_p_input;
    Gamma=x_initial.size();
    // 
    svd.resize(Gamma);
    for(int i=1-1; i<=Gamma-1; i++ ){
        svd[i] = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getSqrtMinusDtSVDVecsMatrix(i);
    }
    // 
    if(x_initial.size() != p_initial.size()){
        cout<<"Error: x_initial.size() != p_initial.size()"<<endl;
    }
    //
    if(jastrowProjector->KVorder[j_Jastrow]=="VK"){
        oneBodyWalkerRightOperation.applyToRight(jastrowProjector->expMinusDtK_Jastrow_vec[j_Jastrow], walkerRight_input, walkerRight);
        walkerLeft = walkerLeft_input;
    }else if(jastrowProjector->KVorder[j_Jastrow]=="K^daggerV"){
        oneBodyWalkerLeftOperation.applyToLeft(jastrowProjector->expMinusDtK_Jastrow_vec[j_Jastrow], walkerLeft_input, walkerLeft);
        walkerRight = walkerRight_input;
    }else if(jastrowProjector->KVorder[j_Jastrow]=="KVK"){
        oneBodyWalkerLeftOperation.applyToLeft(jastrowProjector->expMinusDtK_Jastrow_vec[j_Jastrow], walkerLeft_input, walkerLeft);
        oneBodyWalkerRightOperation.applyToRight(jastrowProjector->expMinusDtK_Jastrow_vec[j_Jastrow], walkerRight_input, walkerRight);
    }else{
        cout<<"Error: UNKNOW jastrowProjector->KVorder[j_Jastrow]: "<<jastrowProjector->KVorder[j_Jastrow]<<endl;
        exit(1);
    }
    // 
    ///////////////////////////////////////////
    //icf: ATTENTION: have to update twoBodySample_x once x is updated
    ///////////////////////////////////////////
    twoBodySample_x = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAux(x_initial);
    //
    // cout<<"get_grad_det: "<<get_grad_det(x_initial)<<endl;
    // cout<<"get_grad_det_fast: "<<get_grad_det_fast(x_initial)<<endl;
    // cout<<"get_grad_det_num: "<<get_grad_det_num(x_initial)<<endl;
    // exit(1);
    // 
    // TensorHao<complex<double>,1> grad_U_x_initial_num = get_grad_minus_logOverlap_num(x_initial);
    // auto begin = std::chrono::high_resolution_clock::now();
    grad_U_x_initial = get_grad_minus_logOverlap(x_initial);
    // auto end = std::chrono::high_resolution_clock::now();
    // auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    // printf("Time measured for get_grad_minus_logOverlap: %.8f seconds.\n", elapsed.count() * 1e-9);
    // cout<<"get_grad_minus_logOverlap: "<<grad_U_x_initial<<endl;
    // cout<<"get_grad_minus_logOverlap_num: "<<grad_U_x_initial_num<<endl;
    // exit(1);
    //
    // auto begin2 = std::chrono::high_resolution_clock::now();
    // U_x_initial = get_minus_logOverlap(x_initial);
    U_x_initial = get_minus_logOverlap_fast();
    // auto end2 = std::chrono::high_resolution_clock::now();
    // auto elapsed2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - begin2);
    // printf("Time measured for get_minus_logOverlap: %.8f seconds.\n", elapsed2.count() * 1e-9);
    //
    K_p_initial = get_square(p_initial) / 2.0;
}

void LeapFrog::leap(double dt, int steps)
{
    x = x_initial;
    p = p_initial;
    U_x = U_x_initial;
    K_p = K_p_initial;
    grad_U_x = grad_U_x_initial;
    //
    for(int j=1-1; j<=Gamma-1; j++){
        p(j) -= 0.5 * dt * grad_U_x(j); // first half p
    }
    // 
    // auto begin = std::chrono::high_resolution_clock::now();
    for(int i=1-1; i<=steps-1; i++){
        // x += dt * p;
        for(int j=1-1; j<=Gamma-1; j++){
            x(j) += dt * p(j);
        }
        ///////////////////////////////////////////
        //icf: ATTENTION: have to update twoBodySample_x once x is updated
        ///////////////////////////////////////////
        twoBodySample_x = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAux(x);
        // TensorHao<complex<double>,1> grad_U_x_check = get_grad_minus_logOverlap_num(x);
        grad_U_x = get_grad_minus_logOverlap(x);
        // cout<<"grad_U_x: "<<grad_U_x<<endl;
        // cout<<"grad_U_x_check: "<<grad_U_x_check<<endl;
        // p -= dt * grad_U_x;
        for(int j=1-1; j<=Gamma-1; j++){
            p(j) -= dt * grad_U_x(j); 
        }
    }
    // exit(1);
    // auto end = std::chrono::high_resolution_clock::now();
    // auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    // printf("=======================\n");
    // cout<<"steps: "<<steps<<endl;
    // printf("Time measured for leap steps: %.8f seconds.\n", elapsed.count() * 1e-9);
    // printf("=======================\n");
    // 
    // x += dt * p;
    for(int j=1-1; j<=Gamma-1; j++){
        x(j) += dt * p(j);
    }
    ///////////////////////////////////////////
    //icf: ATTENTION: have to update twoBodySample_x once x is updated
    ///////////////////////////////////////////
    twoBodySample_x = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAux(x);
    // TensorHao<complex<double>,1> grad_U_x_check = get_grad_minus_logOverlap_num(x);
    grad_U_x = get_grad_minus_logOverlap(x);
    // p -= 0.5 * dt * grad_U_x;
    for(int j=1-1; j<=Gamma-1; j++){
        p(j) -= 0.5 * dt * grad_U_x(j);
    }
    //
    K_p = get_square(p) / 2.0;
    // U_x = get_minus_logOverlap(x);
    U_x = get_minus_logOverlap_fast();
    //
    x_final=x;
    p_final=p;
    U_x_final=U_x;
    K_p_final=K_p;
    grad_U_x_final=grad_U_x;
}

void LeapFrog::leap_Random(double dt, int steps)
{
    x = x_initial;
    p = p_initial;
    U_x = U_x_initial;
    K_p = K_p_initial;
    grad_U_x = grad_U_x_initial;
    // 
    x = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].sampleAuxFromForce(jastrowProjector->constForce_Jastrow[j_Jastrow]);
    ///////////////////////////////////////////
    //icf: ATTENTION: have to update twoBodySample_x once x is updated
    ///////////////////////////////////////////
    twoBodySample_x = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAux(x);
    p = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].sampleAuxFromForce(jastrowProjector->constForce_Jastrow[j_Jastrow]);
    K_p = get_square(p) / 2.0;
    U_x = get_minus_logOverlap_fast();
    // U_x = get_minus_logOverlap(x);
    grad_U_x = get_grad_minus_logOverlap(x);
    //
    x_final=x;
    p_final=p;
    U_x_final=U_x;
    K_p_final=K_p;
    grad_U_x_final=grad_U_x;
}

TensorHao<complex<double>,1> LeapFrog::get_grad_minus_logWeight(TwoBodyAux_Jastrow aux_x)
{
    // TwoBodySample_Jastrow twoBodySample = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAux(aux_x);
    // twoBodySample_x = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAux(aux_x);
    //icf: make sure logw and logDet is stabilized to maintain numerical stability
    complex<double> logWeight = conj(walkerLeft.getLogw()) + twoBodySample_x.logw + walkerRight.getLogw();
    TensorHao<complex<double>,1> grad_minus_logWeight(Gamma); 
    // TensorHao<complex<double>,1> grad_logWeight(Gamma); 
    //
    for(int i=1-1; i<=Gamma-1; i++){
        // // icf: svdSample.logw = -0.5*log(2.0*pi)*svdNumber - 0.5*aux2Sum - sqrtMinusDt*auxBSum;
        // grad_logWeight(i) = (-1.0 * aux_x(i) - sqrtMinusDt * svdBg(i)) ;
        grad_minus_logWeight(i) = aux_x(i) + sqrtMinusDt_svdBg(i);
    }
    //
    return grad_minus_logWeight;
}

TensorHao<complex<double>,1> LeapFrog::get_grad_det(TwoBodyAux_Jastrow aux_x)
{
    TensorHao<complex<double>,1> grad_det(Gamma); 
    //
    TensorHao<complex<double>,2> matrix_expM=get_expMatrix(aux_x);
    //
    for(int i=1-1; i<=Gamma-1; i++){
        grad_det(i)=0.0;
        // TensorHao<complex<double>,2> svd_i = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getSqrtMinusDtSVDVecsMatrix(i);
        TensorHao<complex<double>,2> matrix_di = matrixProduct(walkerLeft.getWf(), svd[i], walkerRight.getWf(), 'C', 'N', 'N');
        TensorHao<complex<double>,2> matrix_second_di=get_secondMatrix_di(aux_x, i);
        // TensorHao<complex<double>,2> matrix_di = svd_i;
        //
        for(int j=1-1; j<=matrix_expM.rank(0)-1; j++){
            TensorHao<complex<double>,2> matrix_grad_expM_j = matrix_expM;
            //
            if(jastrowProjector->JastrowExpM[j_Jastrow] == 1){
                for( int k=1-1; k<=matrix_expM.rank(1)-1; k++){
                    //order 1
                    matrix_grad_expM_j(j,k) = matrix_di(j,k);
                }
            }else if(jastrowProjector->JastrowExpM[j_Jastrow] == 2){
                for( int k=1-1; k<=matrix_expM.rank(1)-1; k++){
                    //order 1
                    matrix_grad_expM_j(j,k) = matrix_di(j,k);
                    //order 2
                    matrix_grad_expM_j(j,k) += matrix_second_di(j,k);
                }
            }
            else{
                cout<<"Error: Only support jastrowProjector->JastrowExpM[j_Jastrow] == 2, jastrowProjector->JastrowExpM[j_Jastrow] here is "<<jastrowProjector->JastrowExpM[j_Jastrow]<<endl;
                exit(1);
            }
            //
            grad_det(i) += exp(get_logDet(matrix_grad_expM_j));
        }
    }
    //
    return grad_det;
}


TensorHao<complex<double>,1> LeapFrog::get_grad_det_inv_det_expM_fast(TwoBodyAux_Jastrow aux_x)
{
    ////////////////////////////////////////////////////
    // grad_det_inv_det_expM: d( ln( det( U * exp(SVD) * V ) ) )/dt = Trace( expM^{-1} d(expM)/dt ) = Trace( [ V * expM^{-1} * U ] d( exp(SVD) )/dt )
    // expM == U * exp(SVD) * V
    // matrix_V_expM_inv_U == [ V * expM^{-1} * U ]
    // SVD_grad_expM_di == d(exp(SVD))/dt
    ////////////////////////////////////////////////////
    TensorHao<complex<double>,1> grad_det_inv_det_expM(Gamma); grad_det_inv_det_expM=0.0;
    //
    TensorHao<complex<double>,2> matrix_expM_inv=get_expMatrix_fast();
    BL_NAME(inverse)( matrix_expM_inv );
    TensorHao<complex<double>,2> matrix_V_expM_inv_U = matrixProduct(walkerRight.getWf(), matrix_expM_inv, walkerLeft.getWf(),  'N', 'N', 'C');
    //
    TensorHao<complex<double>,2> SVD_grad_expM_di(L,L);
    TensorHao<complex<double>,2> matrix_temp(L, L);
    // 
    // auto begin2 = std::chrono::high_resolution_clock::now();
    // cout<<"Gamma: "<<Gamma<<endl;
    for(int i=1-1; i<=Gamma-1; i++){
        // auto begin_Gamma = std::chrono::high_resolution_clock::now();
        // auto begin_di = std::chrono::high_resolution_clock::now();
        /////////////////////////////
        if(jastrowProjector->JastrowExpM[j_Jastrow] == 1){
            SVD_grad_expM_di = svd[i];
        }else if(jastrowProjector->JastrowExpM[j_Jastrow] == 2){
            SVD_grad_expM_di = svd[i] + get_secondSVD_di(aux_x, i);
        }else{
            cout<<"Error: Only support jastrowProjector->JastrowExpM[j_Jastrow] == 2, jastrowProjector->JastrowExpM[j_Jastrow] here is "<<jastrowProjector->JastrowExpM[j_Jastrow]<<endl;
            exit(1);
        }
        /////////////////////////////
        // auto end_di = std::chrono::high_resolution_clock::now();
        // auto elapsed_di = std::chrono::duration_cast<std::chrono::nanoseconds>(end_di - begin_di);
        // printf("Time measured for get_secondSVD_di: %.8f seconds.\n", elapsed_di.count() * 1e-9);
        // 
        // auto begin_gmm = std::chrono::high_resolution_clock::now();
        /////////////////////////////
        BL_NAME(gmm)( matrix_V_expM_inv_U, SVD_grad_expM_di, matrix_temp);
        /////////////////////////////
        // auto end_gmm = std::chrono::high_resolution_clock::now();
        // auto elapsed_gmm = std::chrono::duration_cast<std::chrono::nanoseconds>(end_gmm - begin_gmm);
        // printf("Time measured for gmm: %.8f seconds.\n", elapsed_gmm.count() * 1e-9);
        // 
        /////////////////////////////
        for(int j=1-1; j<=L-1; j++){
            grad_det_inv_det_expM(i) += matrix_temp(j,j);
        }
        /////////////////////////////
        // 
        // auto end_Gamma = std::chrono::high_resolution_clock::now();
        // auto elapsed_Gamma = std::chrono::duration_cast<std::chrono::nanoseconds>(end_Gamma - begin_Gamma);
        // printf("Time measured for one gamma: %.8f seconds.\n", elapsed_Gamma.count() * 1e-9);
        // exit(1);
    }
    // auto end2 = std::chrono::high_resolution_clock::now();
    // auto elapsed2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - begin2);
    // printf("Time measured for Gamma: %.8f seconds.\n", elapsed2.count() * 1e-9);
    // 
    return grad_det_inv_det_expM;
}


TensorHao<complex<double>,1> LeapFrog::get_grad_minus_logOverlap(TwoBodyAux_Jastrow aux_x)
{
    // TwoBodySample_Jastrow twoBodySample = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAux(aux_x);
    // twoBodySample_x = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAux(aux_x);
    //icf: make sure logw and logDet is stabilized to maintain numerical stability
    //
    complex<double> sqrtMinusDt = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getSqrtMinusDt();
    complex<double> logWeight = conj(walkerLeft.getLogw()) + twoBodySample_x.logw + walkerRight.getLogw();
    complex<double> weight = exp(logWeight);
    //
    TensorHao<complex<double>,1> grad_minus_logWeight(Gamma);
    // 
    // auto begin_logWeight = std::chrono::high_resolution_clock::now();
    //////////////////////////
    grad_minus_logWeight = get_grad_minus_logWeight(aux_x);
    //////////////////////////
    // auto end_logWeight = std::chrono::high_resolution_clock::now();
    // auto elapsed_logWeight = std::chrono::duration_cast<std::chrono::nanoseconds>(end_logWeight - begin_logWeight);
    // printf("Time measured for get_grad_minus_logWeight: %.8f seconds.\n", elapsed_logWeight.count() * 1e-9);
    // 
    TensorHao<complex<double>,1> grad_det_inv_det_expM(Gamma); 
    // 
    // auto begin = std::chrono::high_resolution_clock::now();
    //////////////////////////
    grad_det_inv_det_expM = get_grad_det_inv_det_expM_fast(aux_x);
    //////////////////////////
    // auto end = std::chrono::high_resolution_clock::now();
    // auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    // printf("Time measured for get_grad_det_inv_det_expM_fast: %.8f seconds.\n", elapsed.count() * 1e-9);
    //
    TensorHao<complex<double>,1> grad_minus_logOverlap(Gamma); 
    //
    for(int i=1-1; i<=Gamma-1; i++){
        grad_minus_logOverlap(i) = grad_minus_logWeight(i) - grad_det_inv_det_expM(i);
    }
    //
    return grad_minus_logOverlap;
}

TensorHao<complex<double>,1> LeapFrog::get_grad_minus_logWeight_num(TwoBodyAux_Jastrow aux_x)
{
    TensorHao<complex<double>,1> grad_minus_logWeight(Gamma); 
    //
    TwoBodyAux_Jastrow aux_x_plus = aux_x; 
    TwoBodyAux_Jastrow aux_x_minus = aux_x; 
    //
    complex<double> gap = 0.001;
    for(int i=1-1; i<=Gamma-1; i++){
        aux_x_plus = aux_x;
        aux_x_plus(i) += gap;
        aux_x_minus = aux_x;
        aux_x_minus(i) -= gap;
        complex<double> minus_logWeight_i_plus = get_minus_logWeight(aux_x_plus); 
        complex<double> minus_logWeight_i_minus = get_minus_logWeight(aux_x_minus); 
        grad_minus_logWeight(i) = (minus_logWeight_i_plus - minus_logWeight_i_minus)/(2.0 * gap);
    }
    //
    return grad_minus_logWeight;
}

TensorHao<complex<double>,1> LeapFrog::get_grad_det_num(TwoBodyAux_Jastrow aux_x)
{
    TensorHao<complex<double>,1> grad_det_num(Gamma); 
    //
    TwoBodyAux_Jastrow aux_x_plus = aux_x; 
    TwoBodyAux_Jastrow aux_x_minus = aux_x; 
    //
    complex<double> gap = 0.00001;
    for(int i=1-1; i<=Gamma-1; i++){
        aux_x_plus = aux_x;
        aux_x_plus(i) += gap;
        aux_x_minus = aux_x;
        aux_x_minus(i) -= gap;
        TensorHao<complex<double>,2> expMatrix_i_plus = get_expMatrix(aux_x_plus);
        complex<double> det_i_plus = exp(get_logDet(expMatrix_i_plus)); 
        TensorHao<complex<double>,2> expMatrix_i_minus = get_expMatrix(aux_x_minus);
        complex<double> det_i_minus = exp(get_logDet(expMatrix_i_minus)); 
        grad_det_num(i) = (det_i_plus - det_i_minus)/(2.0 * gap);
        //
        TensorHao<complex<double>,2> det_grad_expMatrix(expMatrix_i_plus.rank(0), expMatrix_i_plus.rank(1)); det_grad_expMatrix=0.0;
        for(int j=1-1; j<=expMatrix_i_plus.rank(0)-1; j++){
        for(int k=1-1; k<=expMatrix_i_plus.rank(1)-1; k++){
            det_grad_expMatrix(j,k) = (expMatrix_i_plus(j,k) - expMatrix_i_minus(j,k))/(complex<double>(2.0,0.0) * gap);
        }
        }
    }
    //
    return grad_det_num;
}

TensorHao<complex<double>,1> LeapFrog::get_grad_minus_logOverlap_num(TwoBodyAux_Jastrow aux_x)
{
    TensorHao<complex<double>,1> grad_logOverlap(Gamma); 
    //
    // complex<double> logOverlap_0 = get_minus_logOverlap(aux_x); 
    TwoBodyAux_Jastrow aux_x_plus = aux_x; 
    TwoBodyAux_Jastrow aux_x_minus = aux_x; 
    //
    complex<double> gap = 0.001;
    for(int i=1-1; i<=Gamma-1; i++){
        aux_x_plus = aux_x;
        aux_x_plus(i) += gap;
        aux_x_minus = aux_x;
        aux_x_minus(i) -= gap;
        complex<double> logOverlap_i_plus = get_minus_logOverlap(aux_x_plus); 
        complex<double> logOverlap_i_minus = get_minus_logOverlap(aux_x_minus); 
        grad_logOverlap(i) = (logOverlap_i_plus - logOverlap_i_minus )/(2.0 * gap);
    }
    //
    return grad_logOverlap;
}

TensorHao<complex<double>,2> LeapFrog::get_expMatrix(TwoBodyAux_Jastrow aux_x_input)
{
    SD walkerRightTemp;
    // TwoBodySample_Jastrow twoBodySample = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAux(aux_x_input);
    twoBodySample_x = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAux(aux_x_input);
    // 
    twoBodySampleWalkerRightOperation.reset("fixedOrder", jastrowProjector->JastrowExpM[j_Jastrow], 10e-8, 10);
    twoBodySampleWalkerRightOperation.applyToRight(twoBodySample_x, walkerRight, walkerRightTemp);
    // 
    TensorHao<complex<double>,2> matrix_expM(walkerRightTemp.getWf().rank(1), walkerRightTemp.getWf().rank(1));
    BL_NAME(gmm)( walkerLeft.getWf(), walkerRightTemp.getWf(), matrix_expM, 'C', 'N');
    // 
    return matrix_expM;
}

TensorHao<complex<double>,2> LeapFrog::get_expMatrix_fast()
{
    SD walkerRightTemp;
    // TwoBodySample_Jastrow twoBodySample = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAux(aux_x_input);
    // twoBodySample_x = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAux(aux_x_input);
    // 
    twoBodySampleWalkerRightOperation.reset("fixedOrder", jastrowProjector->JastrowExpM[j_Jastrow], 10e-8, 10);
    twoBodySampleWalkerRightOperation.applyToRight(twoBodySample_x, walkerRight, walkerRightTemp);
    // 
    TensorHao<complex<double>,2> matrix_expM(walkerRightTemp.getWf().rank(1), walkerRightTemp.getWf().rank(1));
    BL_NAME(gmm)( walkerLeft.getWf(), walkerRightTemp.getWf(), matrix_expM, 'C', 'N');
    // 
    return matrix_expM;
}

TensorHao<complex<double>,2> LeapFrog::get_secondSVD_di(TwoBodyAux_Jastrow aux_x_input, int i)
{
    ///////////////////////////////////////////////////////
    // matrix_sum is initialized as global parameter
    ///////////////////////////////////////////////////////
    // svd is obtained in get_expMatrix
    // TwoBodySample_Jastrow twoBodySample = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAux(aux_x_input);
    ///////////////////////////////////////////////////////
    // svd is obtained in setInitial
    // TensorHao<complex<double>,2> svd_i = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getSqrtMinusDtSVDVecsMatrix(i);
    ///////////////////////////////////////////////////////
    // BL_NAME(gmm)( twoBodySample_x.matrix, svd[i], matrix_sum, 'N', 'N', 0.5);
    // BL_NAME(gmm)( svd[i], twoBodySample_x.matrix, matrix_sum, 'N', 'N', 0.5, 1.0);
    // 
    // TensorHao<complex<double>,3> svd_vec=jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getSqrtMinusDtSVDVecsMatrix();
    // 
    TensorHaoRef<complex<double>, 2> vecsAux(L, L);
    TensorHaoRef<complex<double>, 2> vecs(L, L);
    TensorHaoRef<complex<double>, 2> matrix_sum_pointer(L, L);
    vecsAux.point( twoBodySample_x.matrix.data() );
    vecs.point( const_cast<complex<double>*> ( svd[i].data() ) );
    matrix_sum_pointer.point( const_cast<complex<double>*> ( matrix_sum.data() ) );
    BL_NAME(gmm)(vecsAux, vecs, matrix_sum_pointer, 'N', 'N', 0.5);
    BL_NAME(gmm)(vecs, vecsAux, matrix_sum_pointer, 'N', 'N', 0.5, 1.0);
    // 
    return matrix_sum;
}

TensorHao<complex<double>,2> LeapFrog::get_secondMatrix_di(TwoBodyAux_Jastrow aux_x_input, int i)
{
    // TwoBodySample_Jastrow twoBodySample = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAux(aux_x_input);
    TensorHao<complex<double>,2> matrix_temp(L, L);
    TensorHao<complex<double>,2> matrix_temp2(L, L);
    TensorHao<complex<double>,2> matrix_temp3(L, L);
    // 
    // TensorHao<complex<double>,2> svd_i = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getSqrtMinusDtSVDVecsMatrix(i);
    // 
    BL_NAME(gmm)( twoBodySample_x.matrix, svd[i], matrix_temp);
    BL_NAME(gmm)( svd[i], twoBodySample_x.matrix, matrix_temp2);
    // 
    matrix_temp3 = matrix_temp + matrix_temp2;
    for(int j=1-1; j<=matrix_temp.rank(0)-1; j++ ){
        for(int k=1-1; k<=matrix_temp.rank(0)-1; k++ ){
            matrix_temp3(j,k) = 0.5 * matrix_temp3(j,k);
        }
    }
    // 
    TensorHao<complex<double>,2> secondMatrix_di = matrixProduct(walkerLeft.getWf(), matrix_temp3, walkerRight.getWf(), 'C', 'N', 'N');
    return secondMatrix_di;
}

complex<double> LeapFrog::get_minus_logWeight(TwoBodyAux_Jastrow aux_x_input)
{
    SD walkerRightTemp;
    // TwoBodySample_Jastrow twoBodySample = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAux(aux_x_input);
    complex<double> logWeight = conj(walkerLeft.getLogw()) + twoBodySample_x.logw + walkerRight.getLogw();
    //
    return -1.0 * logWeight;
}

complex<double> LeapFrog::get_minus_logOverlap(TwoBodyAux_Jastrow aux_x_input)
{
    SD walkerRightTemp;
    TwoBodySample_Jastrow twoBodySample = jastrowProjector->expMinusDtV_Jastrow_vec[j_Jastrow].getTwoBodySampleFromAux(aux_x_input);
    // 
    twoBodySampleWalkerRightOperation.reset("fixedOrder", jastrowProjector->JastrowExpM[j_Jastrow], 10e-8, 10);
    twoBodySampleWalkerRightOperation.applyToRight(twoBodySample, walkerRight, walkerRightTemp);
    // 
    walkerWalkerOperation.set( walkerLeft, walkerRightTemp );
    //
    complex<double> logOverlap = walkerWalkerOperation.returnLogOverlap();
    // 
    return -1.0 * logOverlap;
}

complex<double> LeapFrog::get_minus_logOverlap_fast()
{
    SD walkerRightTemp;
    twoBodySampleWalkerRightOperation.reset("fixedOrder", jastrowProjector->JastrowExpM[j_Jastrow], 10e-8, 10);
    twoBodySampleWalkerRightOperation.applyToRight(twoBodySample_x, walkerRight, walkerRightTemp);
    // 
    walkerWalkerOperation.set( walkerLeft, walkerRightTemp );
    //
    complex<double> logOverlap = walkerWalkerOperation.returnLogOverlap();
    // 
    return -1.0 * logOverlap;
}

complex<double> LeapFrog::get_logDet(TensorHao<complex<double>,2> matrix_input)
{
    return logDeterminant(BL_NAME(LUconstruct)( matrix_input ));
}

complex<double> LeapFrog::get_square(TwoBodyAux_Jastrow aux_p_input)
{
    complex<double> temp_sum=0.0;
    for(int i=1-1; i<=aux_p_input.size()-1; i++ ){
        temp_sum += aux_p_input(i)*aux_p_input(i);
    }
    return temp_sum;
}

// TensorHao<complex<double>,2> LeapFrog::matrixProduct(TensorHao<complex<double>,2> matrix_L,  TensorHao<complex<double>,2> matrix_M, TensorHao<complex<double>,2> matrix_R, char L_s, char M_s)
// {
//     int L, M, R;
//     if( L_s == 'C'){
//         L = matrix_L.rank(1);
//         M = matrix_L.rank(0);
//     }else if( L_s == 'N' ){
//         L = matrix_L.rank(0);
//         M = matrix_L.rank(1);
//     }else{
//         cout<<"Error in matrixProduct L_s: "<<L_s<<endl;
//     }

//     if( M_s == 'C'){
//         M = matrix_M.rank(1);
//         R = matrix_M.rank(0);
//     }else if( M_s == 'N' ){
//         M = matrix_M.rank(0);
//         R = matrix_M.rank(1);
//     }else{
//         cout<<"Error in matrixProduct M_s: "<<M_s<<endl;
//     }

//     TensorHao<complex<double>,2> matrix_Temp(L, R), matrix_output(L, matrix_R.rank(1));
//     BL_NAME(gmm)( matrix_L, matrix_M, matrix_Temp, L_s, M_s);
//     BL_NAME(gmm)( matrix_Temp, matrix_R, matrix_output);
//     return matrix_output;
// }

TensorHao<complex<double>,2> LeapFrog::matrixProduct(TensorHao<complex<double>,2> matrix_L,  TensorHao<complex<double>,2> matrix_M, TensorHao<complex<double>,2> matrix_R, char L_s = 'N', char M_s = 'N', char R_s = 'N')
{
    int L, M, R;
    int L2, M2, R2;
    if( L_s == 'C'){
        L = matrix_L.rank(1);
        M = matrix_L.rank(0);
    }else if( L_s == 'N' ){
        L = matrix_L.rank(0);
        M = matrix_L.rank(1);
    }else{
        cout<<"Error in matrixProduct L_s: "<<L_s<<endl;
    }

    if( M_s == 'C'){
        M = matrix_M.rank(1);
        R = matrix_M.rank(0);
    }else if( M_s == 'N' ){
        M = matrix_M.rank(0);
        R = matrix_M.rank(1);
    }else{
        cout<<"Error in matrixProduct M_s: "<<M_s<<endl;
    }

    if( R_s == 'C'){
        L2 = matrix_R.rank(1);
        R2 = matrix_R.rank(0);
    }else if( M_s == 'N' ){
        L2 = matrix_R.rank(0);
        R2 = matrix_R.rank(1);
    }else{
        cout<<"Error in matrixProduct R_s: "<<R_s<<endl;
    }

    TensorHao<complex<double>,2> matrix_Temp(L, R), matrix_output(L, R2);
    BL_NAME(gmm)( matrix_L, matrix_M, matrix_Temp, L_s, M_s);
    BL_NAME(gmm)( matrix_Temp, matrix_R, matrix_output, 'N', R_s);
    return matrix_output;
}