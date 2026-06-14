//
// Created by boruoshihao on 12/28/16.
//

#ifndef AFQMCLAB_HOPAUXMATRIX_H
#define AFQMCLAB_HOPAUXMATRIX_H

#include "afqmclab.h"

//One body operator.

class Hop_AuxMatrix
{
 public:
    std::complex<double> logw;
    tensor_hao::TensorHao<std::complex<double>,2> matrix;

    Hop_AuxMatrix(); 
    Hop_AuxMatrix(const std::string &filename); 
    Hop_AuxMatrix(const Hop_AuxMatrix& x);
    Hop_AuxMatrix(Hop_AuxMatrix&& x);
    ~Hop_AuxMatrix();

    Hop_AuxMatrix & operator  = (const Hop_AuxMatrix& x);
    Hop_AuxMatrix & operator  = (Hop_AuxMatrix&& x);
    
    void readModel(const std::string &filename);
    
    LogHop getLogHop();
    Hop getHop(std::string flag, size_t taylorOrder, double accuracy, size_t baseTaylorOrder);
    Hop getHop();
    size_t getL() const;
    double getMemory() const;

 private:
    void copy_deep(const Hop_AuxMatrix &x);
    void move_deep(Hop_AuxMatrix &x);
};

#endif //AFQMCLAB_HOPAUXMATRIX_H