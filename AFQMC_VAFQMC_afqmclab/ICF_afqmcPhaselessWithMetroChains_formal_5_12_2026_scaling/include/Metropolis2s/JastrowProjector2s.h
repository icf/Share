//

#ifndef JASTROWPROJECTOR2S_H
#define JASTROWPROJECTOR2S_H

#include "afqmclab.h"
#include "metropolis2sDefine.h"
#include "metropolis2sMethod.h"

class JastrowProjector2s
{
 public:
 
    JastrowProjector2s();
    JastrowProjector2s(JastrowProjector2s& x);
    JastrowProjector2s & operator  = (JastrowProjector2s& x);
    ~JastrowProjector2s();
    //
    std::vector<std::string> variableName_vec;
    //
    std::vector<Model_Jastrow2s> model_Jastrow;
    // std::vector<Model_BP_Jastrow2s> model_BP_Jastrow;
    std::vector<TwoBody_Jastrow2s> expMinusDtV_Jastrow_vec;
    // std::vector<TwoBody_BP_Jastrow2s> expMinusDtV_BP_Jastrow_vec;
    std::vector<OneBody_Jastrow2s> expMinusDtK_Jastrow_vec;
    // std::vector<OneBody_BP_Jastrow2s> expMinusDtK_BP_Jastrow_vec;
    std::vector<TwoBodyForce_Jastrow2s>  constForce_Jastrow;
    // std::vector<TwoBodyForce_BP_Jastrow2s>  constForce_BP_Jastrow;
    std::vector<std::string> KVorder;
    //
    int numOfJastrow;
    std::vector<int> JastrowSlice;
    std::vector<int> JastrowExpM;
    //
    void initialParameters(Metropolis2sMethod method);
    void initialParametersTwoJastrow(Metropolis2sMethod method);
    std::string getVariableName( int i_temp);
    int getVariableSite( int i_temp);
    void extendMetroChainToRight( JastrowProjector2s jastrowProjector_input);

    void copy_deep(JastrowProjector2s &x);
};

#endif //JASTROWPROJECTOR2S_H
