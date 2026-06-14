//

#ifndef JASTROWPROJECTOR_H
#define JASTROWPROJECTOR_H

#include "afqmclab.h"
#include "metropolisDefine.h"
#include "metropolisMethod.h"

class JastrowProjector
{
 public:
 
    JastrowProjector();
    JastrowProjector(JastrowProjector& x);
    JastrowProjector & operator  = (JastrowProjector& x);
    ~JastrowProjector();
    //
    std::vector<std::string> variableName_vec;
    //
    std::vector<Model_Jastrow> model_Jastrow;
    std::vector<TwoBody_Jastrow> expMinusDtV_Jastrow_vec;
    std::vector<OneBody_Jastrow> expMinusDtK_Jastrow_vec;
    std::vector<TwoBodyForce_Jastrow>  constForce_Jastrow;
    std::vector<std::string> KVorder;
    //
    int numOfJastrow;
    std::vector<int> JastrowSlice;
    std::vector<int> JastrowExpM;
    //
    void initialParameters(MetropolisMethod method);
    void initialParametersTwoJastrow(MetropolisMethod method);
    std::string getVariableName( int i_temp);
    int getVariableSite( int i_temp);
    void extendMetroChainToRight( JastrowProjector jastrowProjector_input);

    void copy_deep(JastrowProjector &x);
};

#endif //JASTROWPROJECTOR_H
