#include "afqmclab.h"

using namespace std;
using namespace tensor_hao;

long get_memory_usage_linux();

bool checkMatrixDiff(TensorHao<complex<double>,2> matrixInputA, TensorHao<complex<double>,2> matrixInputB, double accurancy);

TensorHao<std::complex<double>, 2> expMatrix(TensorHao<std::complex<double>, 2> A, int b);

void get_deltaMatrix_1s(TensorHao<complex<double>,2> & deltaMatrix, int expM, string order, TensorHao<complex<double>,2> wfnLeft_matrix, TensorHao<complex<double>,2> U_s, TensorHao<complex<double>,2> Vdagger_s, TensorHao<complex<double>,2> old_A, TensorHao<complex<double>,2> k_matrix, TensorHao<complex<double>,2> wfnRight_matrix);

void get_updatedUV_1s(vector<TensorHao<complex<double>,2>> & U_vec, vector<TensorHao<complex<double>,2>> & Vdagger_vec, int expM, string order, TensorHao<complex<double>,2> wfnLeft_matrix, TensorHao<complex<double>,2> U_s, TensorHao<complex<double>,2> Vdagger_s, TensorHao<complex<double>,2> old_B, TensorHao<complex<double>,2> k_matrix, TensorHao<complex<double>,2> wfnRight_matrix);
