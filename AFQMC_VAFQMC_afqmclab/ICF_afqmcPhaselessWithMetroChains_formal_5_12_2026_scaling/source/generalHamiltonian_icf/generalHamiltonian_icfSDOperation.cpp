//
// Created by boruoshihao on 7/8/17.
//

#include "../../include/generalHamiltonian_icf/generalHamiltonian_icfSDOperation.h"

using namespace std;
using namespace tensor_hao;

void fillWalkerRandomly(SD &walker, const GeneralHamiltonian_icf &model)
{
    size_t L = model.getL(); size_t N = model.getN(); 
    walker.resize(L, N);
    walker.randomFill();
}

void fillWalkerFromModel(SD &walker, GeneralHamiltonian_icf &model)
{
    size_t L = model.getL(); size_t N = model.getN(); 
    walker.resize(L, N);

    model.setKpEigenValueAndVector();
    const TensorHao<complex<double>,2> &KpEigenVector = model.getKpEigenVector();

    TensorHao<complex<double>,2> &wf = walker.wfRef();
    for(size_t i = 0; i < N; ++i)
    {
        for(size_t j = 0; j < L; ++j) wf(j,i) = KpEigenVector(j,i);
    }

}