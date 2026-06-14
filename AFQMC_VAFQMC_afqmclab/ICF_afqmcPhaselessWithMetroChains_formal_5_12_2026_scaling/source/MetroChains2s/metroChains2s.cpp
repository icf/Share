//
// Created by boruoshihao on 12/25/16.
// Modified by Icf on 2019-9-29
//

#include <chrono>
#include "../../include/MetroChains2s/metroChains2s.h"
#include "afqmclab.h"

using namespace std;
using namespace tensor_hao;

MetroChains2s::MetroChains2s() {
    numOfReleasedSlice = 0; numOfChains = -1; logw = 0.0;
    isInitialMetroChains = false;
    numOfBrackets = -1;

    metropolis2sVec.resize(0);
    metroLeft.resize(0);
    metroRight.resize(0);

    expMinusDtK_Jastrow_vec.resize(0);
    expMinusDtV_Jastrow_vec.resize(0);
}

MetroChains2s::MetroChains2s(const MetroChains2s &x) { copy_deep(x); }

MetroChains2s::MetroChains2s(MetroChains2s &&x) { move_deep(x); }

MetroChains2s::~MetroChains2s() { }

MetroChains2s &MetroChains2s::operator=(const MetroChains2s &x) { copy_deep(x); return *this; }

MetroChains2s &MetroChains2s::operator=(MetroChains2s &&x) { move_deep(x); return *this; }

size_t MetroChains2s::getNumOfChains() const { return numOfChains; }
size_t MetroChains2s::getNumOfBrackets() const { return numOfBrackets; }

size_t MetroChains2s::getBPMetroTimesliceBlockSize(size_t n) const { return metropolis2sVec[n].getBPMetroTimesliceBlockSize(); }
size_t MetroChains2s::getBPMetroTimesliceBlockSize() const { return metropolis2sVec[0].getBPMetroTimesliceBlockSize(); }

MetroChains2sWalker MetroChains2s::getMetroRight(size_t n) { return metroRight[n]; }
size_t MetroChains2s::getL() const { return metroRight[0].getL(); }
size_t MetroChains2s::getN() const { return metroRight[0].getNup() + metroRight[0].getNdn(); }
size_t MetroChains2s::getNup() const { return metroRight[0].getNup(); }
size_t MetroChains2s::getNdn() const { return metroRight[0].getNdn(); }

const complex<double> MetroChains2s::getLogw() const { return logw; }
complex<double> &MetroChains2s::logwRef() { return logw; }

/////////////////////////////////////////////////////////////////////
//general wf and logw from initial wf, jastrowProjector and Force, Aux
/////////////////////////////////////////////////////////////////////
void MetroChains2s::initialMetroChains(int L, int Nup, int Ndn, size_t numOfReleasedSlice_input, size_t numOfChains_input, size_t numOfBrackets_input, JastrowProjector2s &jastrowProjector_)
{
    logw = 0.0;
    //
    isInitialMetroChains = true;
    numOfChains = numOfChains_input;
    numOfBrackets = numOfBrackets_input;
    numOfReleasedSlice = numOfReleasedSlice_input;
    // 
    expMinusDtK_Jastrow_vec = jastrowProjector_.expMinusDtK_Jastrow_vec;
    expMinusDtV_Jastrow_vec = jastrowProjector_.expMinusDtV_Jastrow_vec;
    //
    metropolis2sVec.resize(numOfChains);
    for(int i=1-1; i<= numOfChains-1; i++){
        metropolis2sVec[i].initialParameters(L, Nup, Ndn, jastrowProjector_, expMinusDtK_Jastrow_vec, expMinusDtV_Jastrow_vec);
    }

}

void MetroChains2s::initialMetroChainsTwoJastrow(int L, int Nup, int Ndn, size_t numOfReleasedSlice_input, size_t numOfChains_input, size_t numOfBrackets_input, JastrowProjector2s &jastrowProjector_)
{
    logw = 0.0;
    //
    isInitialMetroChains = true;
    numOfChains = numOfChains_input;
    numOfBrackets = numOfBrackets_input;
    numOfReleasedSlice = numOfReleasedSlice_input;
    // 
    expMinusDtK_Jastrow_vec = jastrowProjector_.expMinusDtK_Jastrow_vec;
    expMinusDtV_Jastrow_vec = jastrowProjector_.expMinusDtV_Jastrow_vec;
    //
    metropolis2sVec.resize(numOfChains);
    for(int i=1-1; i<= numOfChains-1; i++){
        metropolis2sVec[i].initialParametersTwoJastrow(L, Nup, Ndn, jastrowProjector_, expMinusDtK_Jastrow_vec, expMinusDtV_Jastrow_vec);
    }
}
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
void MetroChains2s::setJastrowProjector(JastrowProjector2s &jastrowProjector_)
{
    expMinusDtK_Jastrow_vec = jastrowProjector_.expMinusDtK_Jastrow_vec;
    expMinusDtV_Jastrow_vec = jastrowProjector_.expMinusDtV_Jastrow_vec;
    for(int i=1-1; i<= numOfChains-1; i++){
        metropolis2sVec[i].setJastrowProjector(jastrowProjector_);
    }
}

// void MetroChains2s::MetroChainsUpdateBackGround(std::vector<tensor_hao::TensorHao<std::complex<double>,1>> backgroundVec)
// {
//     for(int i=1-1; i<= numOfChains-1; i++){
//         metropolis2sVec[i].updateBackGround(backgroundVec);
//     }
// }

void MetroChains2s::MetroChainsInitialField(MetroChains2sWalker &walkerLeftInitial, MetroChains2sWalker &walkerRightInitial)
{
    std::vector<int> vec_temp{};
    for(int i=1-1; i<= numOfChains-1; i++){
        // MetroChains2sWalker walkerLeftInitial=multDetInitial.getSD(i);
        metropolis2sVec[i].initialField(walkerLeftInitial, walkerRightInitial);
        metropolis2sVec[i].updateToRightOneSweep(vec_temp, true);
    }
}

void MetroChains2s::MetroChainsTwoJastrowInitialField(MetroChains2sWalkerRead &multDetInitial, MetroChains2sWalker &walkerRightInitial)
{
    std::vector<int> vec_temp{};
    for(int i=1-1; i<= numOfChains-1; i++){
        MetroChains2sWalker walkerLeftInitial=multDetInitial.getSD(i);
        metropolis2sVec[i].initialField(walkerLeftInitial, walkerRightInitial);
        metropolis2sVec[i].updateToRightOneSweep(vec_temp, true);
    }
}


/////////////////////////////////////////////////////////////////////

void MetroChains2s::MetroChainsTwoJastrowInitialField_readAuxFields(MetroChains2sWalkerRead &multDetInitial, MetroChains2sWalker &walkerRightInitial)
{
    std::vector<int> vec_temp{};
    for(int i=1-1; i<= numOfChains-1; i++){
        MetroChains2sWalker walkerLeftInitial=multDetInitial.getSD(i);
        metropolis2sVec[i].initialField(walkerLeftInitial, walkerRightInitial);
        //
        metropolis2sVec[i].metropolis2sInfo.readTwoJastrowAuxiliaryFields(i);
        metropolis2sVec[i].initialField_again(walkerRightInitial);
        //
        metropolis2sVec[i].updateToRightOneSweep(vec_temp, true);
    }
}

void MetroChains2s::MetroChainsInitialField_readAuxFields(MetroChains2sWalker &walkerLeftInitial, MetroChains2sWalker &walkerRightInitial)
{
    std::vector<int> vec_temp{};
    for(int i=1-1; i<= numOfChains-1; i++){
        metropolis2sVec[i].initialField(walkerLeftInitial, walkerRightInitial);
        //
        metropolis2sVec[i].metropolis2sInfo.readAuxiliaryFields(i+MPIRank()*numOfChains);
        metropolis2sVec[i].initialField_again(walkerRightInitial);
        //
        metropolis2sVec[i].updateToRightOneSweep(vec_temp, true);
    }
}
/////////////////////////////////////////////////////////////////////

void MetroChains2s::copyMetroChains_FromMetroChainsTwoJastrow(int L, int Nup, int Ndn, size_t numOfReleasedSlice_input, size_t numOfChains_input, size_t numOfBrackets_input, MetroChains2s phiT_twoJastrow_input, JastrowProjector2s &jastrowProjector_)
{
    logw = 0.0;
    //
    isInitialMetroChains = true;
    numOfChains = numOfChains_input;
    numOfBrackets = numOfBrackets_input;
    numOfReleasedSlice = numOfReleasedSlice_input;
    //
    metropolis2sVec.resize(numOfChains);
    expMinusDtK_Jastrow_vec = jastrowProjector_.expMinusDtK_Jastrow_vec;
    expMinusDtV_Jastrow_vec = jastrowProjector_.expMinusDtV_Jastrow_vec;
    // 
    phiT_twoJastrow_input.metropolis2sVec[0].metropolis2sInfo.takeLeftHalf();  
    for(int i=1-1; i<= numOfChains-1; i++){
        metropolis2sVec[i].initialParameters(L, Nup, Ndn, jastrowProjector_, expMinusDtK_Jastrow_vec, expMinusDtV_Jastrow_vec);
        metropolis2sVec[i].copyField(phiT_twoJastrow_input.metropolis2sVec[0].metropolis2sInfo);
        ///////////////////////////////
        // update local and global fast
        if(metropolis2sVec[i].method.BPMetroUpdateType == "local"){
            metropolis2sVec[i].updateDirectB();
        }
        metropolis2sVec[i].updateWalkerRight(metropolis2sVec[i].getWalkerRightInBlock(0));
        ///////////////////////////////
    }
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void MetroChains2s::extendMetroChainInfoForRelease()
{
    // std::vector<int> vec_temp{};
    // ///////////////////////////////////////////////////////////////
    // MetroChains2sMetropolisInfo metropolisInfo_input;

    // metropolisInfo_input.BPMetroTimesliceBlockSize = numOfReleasedSlice;

    // metropolisInfo_input.inBlockIndex = 0;
    // metropolisInfo_input.currentLogOverlap = 0.0;

    // metropolisInfo_input.auxiliaryFields = twoBodyAuxVec;
    // // metropolisInfo_input.auxiliaryFields_BP = twoBodyAuxVec_BP;
    // metropolisInfo_input.dynamicForceFields = twoBodyForceVec;
    // // metropolisInfo_input.dynamicForceFields_BP = twoBodyForceVec_BP;

    // metropolisInfo_input.walkerRightInBlock = walkerRightVec;
    // //
    // //Below three objects will be replaced by metropolis2sVec[i].initialField_again in  metropolis2sVec[i].extendMetroChainToRight.
    // vector< complex<double> > tempVec; tempVec.resize(numOfReleasedSlice); 
    // for(int i=1-1; i<=numOfReleasedSlice-1; i++){
    //     tempVec[i]=0.0;
    // }
    // metropolisInfo_input.walkerLeftInBlock = walkerRightVec;    //icf: doesn't matter the choice of walkerLeftInBlock as extendMetroChainInfoToRight will update it
    // metropolisInfo_input.logWeightRightInBlock = tempVec;
    // metropolisInfo_input.logWeightLeftInBlock = tempVec;
    // ///////////////////////////////////////////////////////////////
    // for(int i=1-1; i<= numOfChains-1; i++){
    //     metropolis2sVec[i].extendMetroChainInfoToRight(metropolisInfo_input);
    //     //
    //     metropolis2sVec[i].initialField_again(metropolisInfo_input.walkerRightInBlock[0]);
    //     metropolis2sVec[i].updateToRightOneSweep(vec_temp, true);
    // }
}

void MetroChains2s::updateMetroChains_allChainsAndAddAndPopBracketForRelease(size_t numOfBrackets_input)
{
    // numOfBrackets = numOfBrackets_input;
    // std::vector<int> vec_temp{};
    // ///////////////////////////////////////////////////////////////
    // for(int i=1-1; i<= numOfChains-1; i++){
    //     for(int counter=1-1; counter<= numOfBrackets-1; counter++){
    //         updateMetroChains(i, 1, vec_temp, false);
    //         addAndPopBracket(i, metropolis2sVec[i].getBPMetroTimesliceBlockSize()-numOfReleasedSlice, numOfReleasedSlice);
    //     }
    // }
}

void MetroChains2s::updateMetroChains_allChainsAndAddAndPopBracket(size_t numOfBrackets_input)
{
    numOfBrackets = numOfBrackets_input;
    std::vector<int> vec_temp{};
    ///////////////////////////////////////////////////////////////
    for(int i=1-1; i<= numOfChains-1; i++){
        for(int counter=1-1; counter<= numOfBrackets-1; counter++){
            updateMetroChains(i, 1, vec_temp, false);
            addAndPopBracket_1Slice_walkerRightMetro(i);
        }
    }
}
//////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

void MetroChains2s::updateMetroChainsWithNewCket_MetroAuxUpdate_prop_withSave(MetroChains2sWalker walkerRight_input, int numOfThermalSweeps, complex<double> phase_dy, MetroChains2sWalker walkerRight_lastStep, TensorHao<complex<double>, 1> dynamicForce_BP, TensorHao<complex<double>, 1> twoBodyAux_BP, vector<int> flip_i_vec)
{
    // 
    if(numOfBrackets != 1){
        cout<<"Error, updateMetroChainsWithNewCket_MetroAuxUpdate is not implemented for more than one bracket"<<endl;
        exit(1);
    }
    //
    if(isInitialMetroChains ){
        complex<double> totalPhaseDen = 0.0;
        vector<complex<double>> logRealPart_Metro(numOfChains);
        for(int i=1-1; i<= numOfChains-1; i++){
            totalPhaseDen += exp(complex<double>(0.0,metropolis2sVec[i].metropolis2sInfo.currentLogOverlap.imag()));
            logRealPart_Metro[i] = metropolis2sVec[i].metropolis2sInfo.currentLogOverlap.real();
        }
        /////////////////////////////////////////////
        //update walkerRight in metroChain
        /////////////////////////////////////////////
        complex<double> logWRealShift = walkerRight_input.getLogw().real();
        //the abs of walker coefficient won't matter in our calculations, for stablization, we set it to 1:
        walkerRight_input.logwRef() = walkerRight_input.getLogw() - walkerRight_input.getLogw().real();
        ///////////////
        // Timer
        ///////////////
        auto initial_begin = std::chrono::high_resolution_clock::now();
        ///////////////
        for(int chain=1-1; chain<= numOfChains-1; chain++){
            metropolis2sVec[chain].updateWalkerRight(walkerRight_input);
            if(metropolis2sVec[chain].method.BPMetroUpdateType == "local"){
                metropolis2sVec[chain].checkLocalUpdateData(walkerRight_input);
            }
        }
        auto initial_end = std::chrono::high_resolution_clock::now();
        complex<double> totalPhaseNum = 0.0;
        for(int i=1-1; i<= numOfChains-1; i++){
            // after walkerRight --> walkerRight_input, the currentLogOverlap is not a phase ( exp(i\theta)) anymore
            // updateMetroChains will set it to a phase ( exp(i\theta)) later
            totalPhaseNum += exp(logWRealShift + metropolis2sVec[i].metropolis2sInfo.currentLogOverlap - logRealPart_Metro[i]);
        }
        /////////////////////////////////////////////
        complex<double> phase_AF = log(totalPhaseNum/totalPhaseDen).imag();
        double AF_scale = abs(totalPhaseNum/totalPhaseDen);
        //
        ///////////////
        // Timer
        ///////////////
        auto begin = std::chrono::high_resolution_clock::now();
        ///////////////
        for(int chain=1-1; chain<= numOfChains-1; chain++){
            for(int i=1-1; i<= numOfBrackets-1; i++){
                updateMetroChains(chain, numOfThermalSweeps, flip_i_vec, false);
                addAndPopBracket_1Slice_walkerRightMetro(chain);
            }
        }
        ///////////////
        // Timer
        ///////////////
        auto timer_end = std::chrono::high_resolution_clock::now();
        ///////////////
        auto elapsed_updateMetroChains = std::chrono::duration_cast<std::chrono::nanoseconds>(timer_end - begin);
        auto elapsed_initial = std::chrono::duration_cast<std::chrono::nanoseconds>(initial_end - initial_begin);
        // if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
        // if(MPIRank() == 0)printf("Time measured for elapsed_initial: %.8f seconds.\n", elapsed_initial.count() * 1e-9);
        // if(MPIRank() == 0)printf("Time measured for elapsed_updateMetroChains: %.8f seconds.\n", elapsed_updateMetroChains.count() * 1e-9);
        // if(MPIRank() == 0)printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");
        ///////////////
        //
        complex<double> totalPhaseAfterNum = 0.0;
        for(int i=1-1; i<= numOfChains-1; i++){
            totalPhaseAfterNum += exp(complex<double>(0.0, metropolis2sVec[i].metropolis2sInfo.currentLogOverlap.imag()));
        }
        ///////////////////
        //update weight from AFQMC propagation
        logw = logw + log(AF_scale);
        //
        complex<double> phase_Metro = log(totalPhaseAfterNum/totalPhaseNum).imag();
        //  in principle phase_AF ~= - (auxForce.imag) if SVDBg is applied
        complex<double> phase_whole = (phase_AF) + phase_Metro;
        // 
        // cout<<"cos(phase_dy) = "<<cos(phase_dy)<<"  "<<phase_dy<<" cos(phase_AF) = "<<cos(phase_AF)<<"  "<<phase_AF<<" cos(phase_Metro) = "<<cos(phase_Metro)<<" cos(phase_whole) = "<<cos(phase_whole)<<endl;
        // standard phaseless BRW constraint (phase_AF) + 2nd order Metro constraint (phase_Metro)
        logw = logw + log( cos(phase_whole) );
        //
    }else{
        cout<<"Error: no isInitialMetroChains !"<<endl;
    }
}

void MetroChains2s::updateMetroChainsWithNewCket_MetroAuxUpdate(int numOfThermalSweeps, vector<vector<int>> flip_i_vec_vec)
{
    // 
    if(numOfBrackets != 1){
        cout<<"Error, updateMetroChainsWithNewCket_MetroAuxUpdate is not implemented for more than one bracket"<<endl;
        exit(1);
    }
    if(isInitialMetroChains ){
        // 
        for(vector<int> flip_i_vec: flip_i_vec_vec){
            ////////////////////////////////////////////////
            // local update 
            ////////////////////////////////////////////////
            complex<double> totalPhaseNum = 0.0;
            // before update
            for(int i=1-1; i<=numOfChains-1; i++){
                totalPhaseNum += exp(complex<double>(0.0,metropolis2sVec[i].metropolis2sInfo.currentLogOverlap.imag()));
            }
            // 
            for(int chain=1-1; chain<= numOfChains-1; chain++){
                // 
                for(int i=1-1; i<= numOfBrackets-1; i++){
                    updateMetroChains(chain, numOfThermalSweeps, flip_i_vec, false);
                }
            }
            complex<double> totalPhaseAfterNum = 0.0;
            //update logw
            for(int i=1-1; i<=numOfChains-1; i++){
                totalPhaseAfterNum += exp(complex<double>(0.0,metropolis2sVec[i].metropolis2sInfo.currentLogOverlap.imag()));
            }
            //
            complex<double> phase_Metro = log(totalPhaseAfterNum/totalPhaseNum).imag();
            complex<double> phase_whole = phase_Metro;
            // 2nd order Metro constraint
            logw = logw + log( cos(phase_whole) );
        }
        // 
        for(int chain=1-1; chain<= numOfChains-1; chain++){
            addAndPopBracket_1Slice_walkerRightMetro(chain);
        }
        //
        ////////////////////////////////////////////////
    }else{
        cout<<"Error: no isInitialMetroChains !"<<endl;
    }
}


void MetroChains2s::updateMetroChains(int chain, int numOfSweeps, vector<int> flip_i_vec, bool ifFrozen)
{
    if(isInitialMetroChains ){
            for(int i=1-1; i<= numOfSweeps-1; i++){
                metropolis2sVec[chain].updateToLeftOneSweep(flip_i_vec, ifFrozen);
                metropolis2sVec[chain].updateToRightOneSweep(flip_i_vec, ifFrozen);
            }
    }else{
        cout<<"Error: no isInitialMetroChains !"<<endl;
    }
}


void MetroChains2s::updateMetroChains_allChains(int numOfSweeps, vector<int> flip_i_vec, bool ifFrozen)
{
    if(isInitialMetroChains ){
        for(int i=1-1; i<= numOfChains-1; i++){
            for(int counter=1-1; counter<= numOfSweeps-1; counter++){
                updateMetroChains(i, 1, flip_i_vec, ifFrozen);
            }
        }
    }else{
        cout<<"Error: no isInitialMetroChains !"<<endl;
    }
}


void MetroChains2s::addAndPopBracket(int chain, int n_left, int n_right)
{
    if(isInitialMetroChains ){
        MetroChains2sWalker sd_temp;
            if(metroRight.size() <= numOfBrackets*numOfChains-1 ){
                //
                MetroChains2sWalkerWalkerOperation sdsdOperation(metropolis2sVec[chain].getWalkerLeftInBlock(n_left), metropolis2sVec[chain].getWalkerRightInBlock(n_right));
                complex<double> logOverlap = sdsdOperation.returnLogOverlap();
                //
                // if( abs(exp(logOverlap)) <= 10e-12){
                //     cout<<"Warnning, abs(exp(logOverlap)) is too small: "<<abs(exp(logOverlap))<<"  "<<metropolis2sVec[chain].getWalkerLeftInBlock(n_left).getLogw()<<"  "<<metropolis2sVec[chain].getWalkerRightInBlock(n_right).getLogw()<<endl;
                // }
                //
                sd_temp.wfUpRef()=metropolis2sVec[chain].getWalkerLeftInBlock(n_left).getWfUp();  
                sd_temp.wfDnRef()=metropolis2sVec[chain].getWalkerLeftInBlock(n_left).getWfDn();  
                sd_temp.logwRef()=metropolis2sVec[chain].getWalkerLeftInBlock(n_left).getLogw() - logOverlap.real();
                metroLeft.push_back(sd_temp); 
                //
                sd_temp.wfUpRef()=metropolis2sVec[chain].getWalkerRightInBlock(n_right).getWfUp();  
                sd_temp.wfDnRef()=metropolis2sVec[chain].getWalkerRightInBlock(n_right).getWfDn(); 
                sd_temp.logwRef()=metropolis2sVec[chain].getWalkerRightInBlock(n_right).getLogw();
                metroRight.push_back(sd_temp);   
            }else{
                size_t constSave=metroRight.size();
                metroRight.erase(metroRight.begin(),metroRight.begin()+constSave-numOfBrackets*numOfChains+1);
                metroLeft.erase(metroLeft.begin(),metroLeft.begin()+constSave-numOfBrackets*numOfChains+1);
                //
                MetroChains2sWalkerWalkerOperation sdsdOperation(metropolis2sVec[chain].getWalkerLeftInBlock(n_left), metropolis2sVec[chain].getWalkerRightInBlock(n_right));
                complex<double> logOverlap = sdsdOperation.returnLogOverlap();
                //
                // if( abs(exp(logOverlap)) <= 10e-12){
                //     cout<<"Warnning, abs(exp(logOverlap)) is too small: "<<abs(exp(logOverlap))<<"  "<<metropolis2sVec[chain].getWalkerLeftInBlock(n_left).getLogw()<<"  "<<metropolis2sVec[chain].getWalkerRightInBlock(n_right).getLogw()<<endl;
                // }
                //complex<double> logOverlap=0.0;
                //
                sd_temp.wfUpRef()=metropolis2sVec[chain].getWalkerLeftInBlock(n_left).getWfUp();  
                sd_temp.wfDnRef()=metropolis2sVec[chain].getWalkerLeftInBlock(n_left).getWfDn();  
                sd_temp.logwRef()=metropolis2sVec[chain].getWalkerLeftInBlock(n_left).getLogw() - logOverlap.real();
                metroLeft.push_back(sd_temp); 
                //
                sd_temp.wfUpRef()=metropolis2sVec[chain].getWalkerRightInBlock(n_right).getWfUp();  
                sd_temp.wfDnRef()=metropolis2sVec[chain].getWalkerRightInBlock(n_right).getWfDn();  
                sd_temp.logwRef()=metropolis2sVec[chain].getWalkerRightInBlock(n_right).getLogw();
                metroRight.push_back(sd_temp); 
            }
    }else{
        cout<<"Error: no isInitialMetroChains !"<<endl;
    }
}

void MetroChains2s::addAndPopBracket_1Slice_walkerRightMetro(int chain)
{
    if(isInitialMetroChains ){
        MetroChains2sWalker sd_temp_right = metropolis2sVec[chain].getWalkerRightInBlock(0);
            if(metroRight.size() <= numOfBrackets*numOfChains-1 ){
                // 
                MetroChains2sWalker sd_temp_left = metropolis2sVec[chain].getWalkerLeftMetro();
                //
                //////////////////////
                // Test
                //////////////////////
                // MetroChains2sWalkerWalkerOperation sdsdOperation(sd_temp_left, sd_temp_right);
                // complex<double> logOverlap = sdsdOperation.returnLogOverlap();
                // if(abs(cos(logOverlap.imag() - metropolis2sVec[chain].metropolis2sInfo.currentLogOverlap.imag())-1.0)>=10e-8){
                //     cout<<"Error: logOverlap.imag() - metropolis2sVec[chain].metropolis2sInfo.currentLogOverlap.imag())>=10e-8: "<<logOverlap<<"  "<<metropolis2sVec[chain].metropolis2sInfo.currentLogOverlap<<endl;
                // }
                //////////////////////
                //
                sd_temp_left.logwRef()=sd_temp_left.getLogw() - metropolis2sVec[chain].metropolis2sInfo.currentLogOverlap.real();
                metroLeft.push_back(sd_temp_left); 
                //
                metroRight.push_back(sd_temp_right);   
            }else{
                size_t constSave=metroRight.size();
                metroRight.erase(metroRight.begin(),metroRight.begin()+constSave-numOfBrackets*numOfChains+1);
                metroLeft.erase(metroLeft.begin(),metroLeft.begin()+constSave-numOfBrackets*numOfChains+1);
                // 
                MetroChains2sWalker sd_temp_left = metropolis2sVec[chain].getWalkerLeftMetro();
                //
                //////////////////////
                // Test
                //////////////////////
                // MetroChains2sWalkerWalkerOperation sdsdOperation(sd_temp_left, sd_temp_right);
                // complex<double> logOverlap = sdsdOperation.returnLogOverlap();
                // if(abs(cos(logOverlap.imag() - metropolis2sVec[chain].metropolis2sInfo.currentLogOverlap.imag())-1.0)>=10e-8){
                //     cout<<"Error: logOverlap.imag() - metropolis2sVec[chain].metropolis2sInfo.currentLogOverlap.imag())>=10e-8: "<<logOverlap<<"  "<<metropolis2sVec[chain].metropolis2sInfo.currentLogOverlap<<endl;
                // }
                //////////////////////
                //
                sd_temp_left.logwRef()=sd_temp_left.getLogw() - metropolis2sVec[chain].metropolis2sInfo.currentLogOverlap.real();
                metroLeft.push_back(sd_temp_left); 
                //
                metroRight.push_back(sd_temp_right); 
            }
    }else{
        cout<<"Error: no isInitialMetroChains !"<<endl;
    }
}

void MetroChains2s::stabilize()
{
    if(isInitialMetroChains ){
        for(int i=1-1; i<= metroRight.size()-1; i++){
            metroRight[i].stabilize();
        }
    }else{
        cout<<"Error: no isInitialMetroChains !"<<endl;
    }
}

/////////////////////////////////////////////
// get overlap matrix related objects
/////////////////////////////////////////////
complex<double> MetroChains2s::returnLogPhase_fromCurrentOverlap(size_t chain)
{
    if(getNumOfBrackets() != 1){
        cout<<"Error, returnLogPhase_fromCurrentOverlap is not implemented for more than one bracket"<<endl;
        exit(1);
    }
    // 
    return complex<double>(0.0, metropolis2sVec[chain].metropolis2sInfo.currentLogOverlap.imag());
}

complex<double> MetroChains2s::returnLogTotalPhase_fromCurrentOverlap()
{
    bool exitFlag = false;
    // 
    if(getNumOfBrackets() != 1){
        cout<<"Error, returnLogTotalPhase_fromCurrentOverlap is not implemented for more than one bracket"<<endl;
        exit(1);
    }
    // 
    complex <double> totalPhase(0,0);
    for(int i=1-1; i<=getNumOfChains()-1; i++){
        totalPhase += exp(complex<double>(0.0,metropolis2sVec[i].metropolis2sInfo.currentLogOverlap.imag()));
    }
    if( exitFlag){
        exit(1);
    }

    return log(totalPhase);
}
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

int MetroChains2s::returnNbuf() const
{
    int Nbuf=1 + 4;
    for(int i=1-1; i<=metropolis2sVec.size()-1; i++){
        Nbuf += metropolis2sVec[i].returnNbuf();
    }
    Nbuf += 4;

    for(int i=1-1; i<=metroRight.size()-1; i++){
        Nbuf += metroRight[i].returnNbuf();
    }
    for(int i=1-1; i<=metroLeft.size()-1; i++){
        Nbuf += metroLeft[i].returnNbuf();
    }

    Nbuf += 16;

    // size_t numOfReleasedSlice;
    Nbuf += 4;
    // std::vector< MetroChains2sWalker > walkerRightVec;
    // if( numOfReleasedSlice >= 1 && walkerRightVec.size() >= 1){
    //     for(int i=1-1; i<=walkerRightVec.size()-1; i++){
    //         Nbuf += walkerRightVec[i].returnNbuf();
    //     }
    // }
    // std::vector< TwoBodyForce > twoBodyForceVec;
    // if( numOfReleasedSlice >= 1 && twoBodyForceVec.size() >= 1){
    //     for(int i=1-1; i<=twoBodyForceVec.size()-1; i++){
    //         Nbuf += 16*twoBodyForceVec[i].size();
    //     }
    // }
    // if( numOfReleasedSlice >= 1 && twoBodyForceVec_BP.size() >= 1){
    //     for(int i=1-1; i<=twoBodyForceVec_BP.size()-1; i++){
    //         Nbuf += 8*twoBodyForceVec_BP[i].size();
    //     }
    // }
    // std::vector< TwoBodyAux > twoBodyAuxVec;
    // if( numOfReleasedSlice >= 1 && twoBodyAuxVec.size() >= 1){
    //     for(int i=1-1; i<=twoBodyAuxVec.size()-1; i++){
    //         Nbuf += 16*twoBodyAuxVec[i].size();
    //     }
    // }
    // if( numOfReleasedSlice >= 1 && twoBodyAuxVec_BP.size() >= 1){
    //     for(int i=1-1; i<=twoBodyAuxVec_BP.size()-1; i++){
    //         Nbuf += 4*twoBodyAuxVec_BP[i].size();
    //     }
    // }

    return  Nbuf; 
}

double MetroChains2s::getMemory() const
{
    double mem(0.0);
    
    // Basic primitive types
    mem += sizeof(isInitialMetroChains);  // bool
    mem += sizeof(numOfChains);           // size_t
    mem += sizeof(numOfBrackets);         // size_t
    mem += sizeof(logw);                  // std::complex<double>
    mem += sizeof(numOfReleasedSlice);    // size_t
    
    // Vectors and their contents
    mem += sizeof(metropolis2sVec);
    for(const auto& metropolis : metropolis2sVec) {
        mem += metropolis.getMemory();
    }
    
    mem += sizeof(metroLeft);
    for(const auto& walker : metroLeft) {
        mem += walker.getMemory();
    }
    
    mem += sizeof(metroRight);
    for(const auto& walker : metroRight) {
        mem += walker.getMemory();
    }
    
    // mem += sizeof(walkerRightVec);
    // for(const auto& walker : walkerRightVec) {
    //     mem += walker.getMemory();
    // }
    
    // mem += sizeof(twoBodyForceVec);
    // for(const auto& force : twoBodyForceVec) {
    //     mem += force.getMemory();
    // }
    
    // mem += sizeof(twoBodyAuxVec);
    // for(const auto& aux : twoBodyAuxVec) {
    //     mem += aux.getMemory();
    // }
    

    return mem;
}

#ifdef MPI_HAO
void MPIBcast(MetroChains2s &buffer, int root, MPI_Comm const &comm)
{
    MPIBcast( buffer.isInitialMetroChains, root, comm );
    MPIBcast( buffer.numOfChains, root, comm );
    MPIBcast(buffer.numOfBrackets, root, comm);

    for(int i=1-1; i<=buffer.metroLeft.size()-1; i++){
        MPIBcast(buffer.metroLeft[i], root, comm);
    }
    for(int i=1-1; i<=buffer.metroRight.size()-1; i++){
        MPIBcast(buffer.metroRight[i], root, comm);
    }

    MPIBcast(buffer.logw, root, comm);

    // size_t numOfReleasedSlice;
    MPIBcast(buffer.numOfReleasedSlice, root, comm);
    // std::vector< MetroChains2sWalker > walkerRightVec;
    // if( buffer.numOfReleasedSlice >= 1 && buffer.walkerRightVec.size() >= 1){
    //     for(int i=1-1; i<=buffer.walkerRightVec.size()-1; i++){
    //         MPIBcast(buffer.walkerRightVec[i], root, comm);
    //     }
    // }   
    // std::vector< TwoBodyForce > twoBodyForceVec;
    // if( buffer.numOfReleasedSlice >= 1 && buffer.twoBodyForceVec.size() >= 1){
    //     for(int i=1-1; i<=buffer.twoBodyForceVec.size()-1; i++){
    //         MPIBcast(buffer.twoBodyForceVec[i], root, comm);
    //     }
    // }
    // if( buffer.numOfReleasedSlice >= 1 && buffer.twoBodyForceVec_BP.size() >= 1){
    //     for(int i=1-1; i<=buffer.twoBodyForceVec_BP.size()-1; i++){
    //         MPIBcast(buffer.twoBodyForceVec_BP[i], root, comm);
    //     }
    // }
    // std::vector< TwoBodyAux > twoBodyAuxVec;
    // if( buffer.numOfReleasedSlice >= 1 && buffer.twoBodyAuxVec.size() >= 1){
    //     for(int i=1-1; i<=buffer.twoBodyAuxVec.size()-1; i++){
    //         MPIBcast(buffer.twoBodyAuxVec[i], root, comm);
    //     }
    // }
    // if( buffer.numOfReleasedSlice >= 1 && buffer.twoBodyAuxVec_BP.size() >= 1){
    //     for(int i=1-1; i<=buffer.twoBodyAuxVec_BP.size()-1; i++){
    //         MPIBcast(buffer.twoBodyAuxVec_BP[i], root, comm);
    //     }
    // }

    // Metropolis2s
    for(size_t i = 1-1; i <= buffer.numOfChains-1; ++i){
        MPIBcast( buffer.metropolis2sVec[i], root, comm );
    }
}

void MetroChains2s::pack(vector<char> &buf, int &posit) const     
{
    MPI_Pack(&isInitialMetroChains, 1, MPI_CXX_BOOL, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(&numOfChains, 1, MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    MPI_Pack(&numOfBrackets, 1, MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);

    if(metroLeft.size() != numOfChains*numOfBrackets){
        cout<<"Error: MetroChains2s::pack, metroLeft.size() != numOfChains*numOfBrackets: "<<metroLeft.size()<<" != "<<numOfChains<<" * "<<numOfBrackets<<endl;
        exit(1);
    }
    for(int i=1-1; i<=metroLeft.size()-1; i++){
        metroLeft[i].pack( buf,  posit );
    }
    if(metroRight.size() != numOfChains*numOfBrackets){
        cout<<"Error: MetroChains2s::pack, metroRight.size() != numOfChains*numOfBrackets: "<<metroRight.size()<<" != "<<numOfChains<<" * "<<numOfBrackets<<endl;
        exit(1);
    }
    for(int i=1-1; i<=metroRight.size()-1; i++){
        metroRight[i].pack( buf,  posit );
    }

    MPI_Pack(&logw, 1, MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);

    // size_t numOfReleasedSlice;
    MPI_Pack(&numOfReleasedSlice, 1, MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    // std::vector< MetroChains2sWalker > walkerRightVec;
    // if( numOfReleasedSlice >= 1 && walkerRightVec.size() >= 1){
    //     for(int i=1-1; i<=walkerRightVec.size()-1; i++){
    //         walkerRightVec[i].pack( buf,  posit );
    //     }
    // }
    // std::vector< TwoBodyForce > twoBodyForceVec;
    // if( numOfReleasedSlice >= 1 && twoBodyForceVec.size() >= 1){
    //     for(int i=1-1; i<=twoBodyForceVec.size()-1; i++){
    //         MPI_Pack(twoBodyForceVec[i].data(), twoBodyForceVec[i].size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    //     }
    // }
    // if( numOfReleasedSlice >= 1 && twoBodyForceVec_BP.size() >= 1){
    //     for(int i=1-1; i<=twoBodyForceVec_BP.size()-1; i++){
    //         MPI_Pack(twoBodyForceVec_BP[i].data(), twoBodyForceVec_BP[i].size(), MPI_DOUBLE, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    //     }
    // }
    // std::vector< TwoBodyAux > twoBodyAuxVec;
    // if( numOfReleasedSlice >= 1 && twoBodyAuxVec.size() >= 1){
    //     for(int i=1-1; i<=twoBodyAuxVec.size()-1; i++){
    //         MPI_Pack(twoBodyAuxVec[i].data(), twoBodyAuxVec[i].size(), MPI_DOUBLE_COMPLEX, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    //     }
    // }
    // if( numOfReleasedSlice >= 1 && twoBodyAuxVec_BP.size() >= 1){
    //     for(int i=1-1; i<=twoBodyAuxVec_BP.size()-1; i++){
    //         MPI_Pack(twoBodyAuxVec_BP[i].data(), twoBodyAuxVec_BP[i].size(), MPI_INT, buf.data(), buf.size(), &posit, MPI_COMM_WORLD);
    //     }
    // }

    // Metropolis2s
    for(int i=1-1; i<=numOfChains-1; i++){
        metropolis2sVec[i].pack(buf, posit);
    }
}

void MetroChains2s::unpack(const vector<char> &buf, int &posit)
{
    bool isInitialMetroChains_save = isInitialMetroChains;
    int numOfChains_save = numOfChains;
    int numOfBrackets_save = numOfBrackets;
    // 
    MPI_Unpack(buf.data(), buf.size(), &posit, &isInitialMetroChains, 1, MPI_CXX_BOOL, MPI_COMM_WORLD);
    MPI_Unpack(buf.data(), buf.size(), &posit, &numOfChains, 1, MPI_INT, MPI_COMM_WORLD);

    MPI_Unpack(buf.data(), buf.size(), &posit, &numOfBrackets, 1, MPI_INT, MPI_COMM_WORLD);
    // ///////////////////////////
    ///////////////////////////
    // int posit_save = posit;
    // for(int i=1-1; i<=metroLeft.size()-1; i++){
    //     std::complex<double> logw;
    //     tensor_hao::TensorHao<std::complex<double>,2> wfUp, wfDn;
    //     wfUp.resize(40, 20);
    //     wfDn.resize(40, 20);
    //     MPI_Unpack(buf.data(), buf.size(), &posit, &logw, 1, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    //     MPI_Unpack(buf.data(), buf.size(), &posit, wfUp.data(), wfUp.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    //     MPI_Unpack(buf.data(), buf.size(), &posit, wfDn.data(), wfDn.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    //     // 
    //     complex<double> tempSumUp=0.0;
    //     complex<double> tempSumDn=0.0;
    //     for(int i_temp=1-1; i_temp<=40-1; i_temp++){
    //     for(int j_temp=1-1; j_temp<=20-1; j_temp++){
    //         tempSumUp += abs(wfUp(i_temp,j_temp));
    //         tempSumDn += abs(wfDn(i_temp,j_temp));
    //     }
    //     }
    //     if(abs(tempSumUp) <= 10e-8){
    //         cout<<"MPIRank: "<<MPIRank()<<" i: "<<i<<"in MetroChains2s before, metroLeft[i] wfUp is zero: "<<tempSumUp<<endl;
    //     }
    //     if(abs(tempSumDn) <= 10e-8){
    //         cout<<"MPIRank: "<<MPIRank()<<" i: "<<i<<"in MetroChains2s before, metroLeft[i] wfDn is zero: "<<tempSumDn<<endl;
    //     }
    // }
    // for(int i=1-1; i<=metroRight.size()-1; i++){
    //     std::complex<double> logw;
    //     tensor_hao::TensorHao<std::complex<double>,2> wfUp, wfDn;
    //     wfUp.resize(40, 20);
    //     wfDn.resize(40, 20);
    //     MPI_Unpack(buf.data(), buf.size(), &posit, &logw, 1, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    //     MPI_Unpack(buf.data(), buf.size(), &posit, wfUp.data(), wfUp.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    //     MPI_Unpack(buf.data(), buf.size(), &posit, wfDn.data(), wfDn.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    //     // 
    //     complex<double> tempSumUp=0.0;
    //     complex<double> tempSumDn=0.0;
    //     for(int i_temp=1-1; i_temp<=40-1; i_temp++){
    //     for(int j_temp=1-1; j_temp<=20-1; j_temp++){
    //         tempSumUp += abs(wfUp(i_temp,j_temp));
    //         tempSumDn += abs(wfDn(i_temp,j_temp));
    //     }
    //     }
    //     if(abs(tempSumUp) <= 10e-8){
    //         cout<<"MPIRank: "<<MPIRank()<<" i: "<<i<<"in MetroChains2s before, metroRight[i] wfUp is zero: "<<tempSumUp<<endl;
    //     }
    //     if(abs(tempSumDn) <= 10e-8){
    //         cout<<"MPIRank: "<<MPIRank()<<" i: "<<i<<"in MetroChains2s before, metroRight[i] wfDn is zero: "<<tempSumDn<<endl;
    //     }
    // }
    // MPIBarrier();
    // posit = posit_save;
    ///////////////////////////
    // ///////////////////////////
    if(metroLeft.size() != numOfChains*numOfBrackets){
        cout<<"MPIRank: "<<MPIRank()<<" Error: MetroChains2s::unpack, metroLeft.size() != numOfChains*numOfBrackets: "<<metroLeft.size()<<" != "<<numOfChains<<" * "<<numOfBrackets<<" isInitialMetroChains: "<<isInitialMetroChains<<endl;
        cout<<"MPIRank: "<<MPIRank()<<" Error: MetroChains2s::unpack, metroLeft.size() != numOfChains*numOfBrackets: "<<metroLeft.size()<<" != "<<numOfChains_save<<" * "<<numOfBrackets_save<<" isInitialMetroChains_save: "<<isInitialMetroChains_save<<endl;
        exit(1);
    }
    if(metroRight.size() != numOfChains*numOfBrackets){
        cout<<"MPIRank: "<<MPIRank()<<"Error: MetroChains2s::unpack, metroRight.size() != numOfChains*numOfBrackets: "<<metroRight.size()<<" != "<<numOfChains<<" * "<<numOfBrackets<<" isInitialMetroChains: "<<isInitialMetroChains<<endl;
        exit(1);
    }

    for(int i=1-1; i<=metroLeft.size()-1; i++){
        metroLeft[i].unpack( buf,  posit );
    }
    // 
    for(int i=1-1; i<=metroRight.size()-1; i++){
        metroRight[i].unpack( buf,  posit );
    }
    // // ///////////////////////////
    // for(int i=1-1; i<=metroLeft.size()-1; i++){
    //     complex<double> tempSumUp=0.0;
    //     complex<double> tempSumDn=0.0;
    //     for(int i_temp=1-1; i_temp<=40-1; i_temp++){
    //     for(int j_temp=1-1; j_temp<=20-1; j_temp++){
    //         tempSumUp += abs(metroLeft[i].getWfUp()(i_temp,j_temp));
    //         tempSumDn += abs(metroLeft[i].getWfDn()(i_temp,j_temp));
    //     }
    //     }
    //     if(abs(tempSumUp) <= 10e-8){
    //         cout<<"MPIRank: "<<MPIRank()<<" i: "<<i<<"in MetroChains2s after, metroLeft[i] wfUp is zero: "<<tempSumUp<<endl;
    //         exit(1);
    //     }
    //     if(abs(tempSumDn) <= 10e-8){
    //         cout<<"MPIRank: "<<MPIRank()<<" i: "<<i<<"in MetroChains2s after, metroLeft[i] wfDn is zero: "<<tempSumDn<<endl;
    //         exit(1);
    //     }
    // }
    // // 
    // for(int i=1-1; i<=metroRight.size()-1; i++){
    //     complex<double> tempSumUp=0.0;
    //     complex<double> tempSumDn=0.0;
    //     for(int i_temp=1-1; i_temp<=40-1; i_temp++){
    //     for(int j_temp=1-1; j_temp<=20-1; j_temp++){
    //         tempSumUp += abs(metroRight[i].getWfUp()(i_temp,j_temp));
    //         tempSumDn += abs(metroRight[i].getWfDn()(i_temp,j_temp));
    //     }
    //     }
    //     if(abs(tempSumUp) <= 10e-8){
    //         cout<<"MPIRank: "<<MPIRank()<<" i: "<<i<<"in MetroChains2s after, metroRight[i] wfUp is zero: "<<tempSumUp<<endl;
    //         exit(1);
    //     }
    //     if(abs(tempSumDn) <= 10e-8){
    //         cout<<"MPIRank: "<<MPIRank()<<" i: "<<i<<"in MetroChains2s after, metroRight[i] wfDn is zero: "<<tempSumDn<<endl;
    //         exit(1);
    //     }
    // }
    // ///////////////////////////
    MPI_Unpack(buf.data(), buf.size(), &posit, &logw, 1, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);

    // size_t numOfReleasedSlice;
    MPI_Unpack(buf.data(), buf.size(), &posit, &numOfReleasedSlice, 1, MPI_INT, MPI_COMM_WORLD);
    // std::vector< MetroChains2sWalker > walkerRightVec;
    // if( numOfReleasedSlice >= 1 && walkerRightVec.size() >= 1){
    //     for(int i=1-1; i<=walkerRightVec.size()-1; i++){
    //         walkerRightVec[i].unpack( buf,  posit );
    //     }
    // }
    // std::vector< TwoBodyForce > twoBodyForceVec;
    // if( numOfReleasedSlice >= 1 && twoBodyForceVec.size() >= 1){
    //     for(int i=1-1; i<=twoBodyForceVec.size()-1; i++){
    //         MPI_Unpack(buf.data(), buf.size(), &posit, twoBodyForceVec[i].data(), twoBodyForceVec[i].size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    //     }
    // }
    // if( numOfReleasedSlice >= 1 && twoBodyForceVec_BP.size() >= 1){
    //     for(int i=1-1; i<=twoBodyForceVec_BP.size()-1; i++){
    //         MPI_Unpack(buf.data(), buf.size(), &posit, twoBodyForceVec_BP[i].data(), twoBodyForceVec_BP[i].size(), MPI_DOUBLE, MPI_COMM_WORLD);
    //     }
    // }
    // std::vector< TwoBodyAux > twoBodyAuxVec;
    // if( numOfReleasedSlice >= 1 && twoBodyAuxVec.size() >= 1){
    //     for(int i=1-1; i<=twoBodyAuxVec.size()-1; i++){
    //         MPI_Unpack(buf.data(), buf.size(), &posit, twoBodyAuxVec[i].data(), twoBodyAuxVec[i].size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    //     }
    // }
    // if( numOfReleasedSlice >= 1 && twoBodyAuxVec_BP.size() >= 1){
    //     for(int i=1-1; i<=twoBodyAuxVec_BP.size()-1; i++){
    //         MPI_Unpack(buf.data(), buf.size(), &posit, twoBodyAuxVec_BP[i].data(), twoBodyAuxVec_BP[i].size(), MPI_INT, MPI_COMM_WORLD);
    //     }
    // }

    if(metropolis2sVec.size() != numOfChains*numOfBrackets){
        cout<<"MPIRank: "<<MPIRank()<<" Error: MetroChains2s::unpack, metropolis2sVec.size() != numOfChains*numOfBrackets: "<<metropolis2sVec.size()<<" != "<<numOfChains<<" * "<<numOfBrackets<<" isInitialMetroChains: "<<isInitialMetroChains<<endl;
        cout<<"MPIRank: "<<MPIRank()<<" Error: MetroChains2s::unpack, metropolis2sVec.size() != numOfChains*numOfBrackets: "<<metropolis2sVec.size()<<" != "<<numOfChains_save<<" * "<<numOfBrackets_save<<" isInitialMetroChains_save: "<<isInitialMetroChains_save<<endl;
        exit(1);
    }
    for(int i=1-1; i<=numOfChains-1; i++){
        metropolis2sVec[i].unpack(buf, posit);
    }
}
#endif

void MetroChains2s::copy_deep(const MetroChains2s &x)
{
    isInitialMetroChains = x.isInitialMetroChains;
    numOfChains = x.numOfChains;
    metropolis2sVec = x.metropolis2sVec;
    numOfBrackets = x.numOfBrackets;
    //
    metroLeft.resize(numOfBrackets*numOfChains);
    for(int i=1-1; i<=metroLeft.size()-1; i++){
        metroLeft[i] = x.metroLeft[i];
    }
    metroRight.resize(numOfBrackets*numOfChains);
    for(int i=1-1; i<=metroRight.size()-1; i++){
        metroRight[i] = x.metroRight[i];
    }
    //
    logw = x.logw;

    // size_t numOfReleasedSlice;
    numOfReleasedSlice = x.numOfReleasedSlice;
    // std::vector< MetroChains2sWalker > walkerRightVec;
    // walkerRightVec.resize(x.walkerRightVec.size());
    // if( numOfReleasedSlice >= 1 && walkerRightVec.size() >= 1){
    //     for(int i=1-1; i<=walkerRightVec.size()-1; i++){
    //         walkerRightVec[i] = x.walkerRightVec[i];
    //     }
    // }
    // std::vector< TwoBodyForce > twoBodyForceVec;
    // twoBodyForceVec.resize(x.twoBodyForceVec.size());
    // if( numOfReleasedSlice >= 1 && twoBodyForceVec.size() >= 1){
    //     for(int i=1-1; i<=twoBodyForceVec.size()-1; i++){
    //         twoBodyForceVec[i] = x.twoBodyForceVec[i];
    //     }
    // }
    // twoBodyForceVec_BP.resize(x.twoBodyForceVec_BP.size());
    // if( numOfReleasedSlice >= 1 && twoBodyForceVec_BP.size() >= 1){
    //     for(int i=1-1; i<=twoBodyForceVec_BP.size()-1; i++){
    //         twoBodyForceVec_BP[i] = x.twoBodyForceVec_BP[i];
    //     }
    // }
    // std::vector< TwoBodyAux > twoBodyAuxVec;
    // twoBodyAuxVec.resize(x.twoBodyAuxVec.size());
    // if( numOfReleasedSlice >= 1 && twoBodyAuxVec.size() >= 1){
    //     for(int i=1-1; i<=twoBodyAuxVec.size()-1; i++){
    //         twoBodyAuxVec[i] = x.twoBodyAuxVec[i];
    //     }
    // }
    // twoBodyAuxVec_BP.resize(x.twoBodyAuxVec_BP.size());
    // if( numOfReleasedSlice >= 1 && twoBodyAuxVec_BP.size() >= 1){
    //     for(int i=1-1; i<=twoBodyAuxVec_BP.size()-1; i++){
    //         twoBodyAuxVec_BP[i] = x.twoBodyAuxVec_BP[i];
    //     }
    // }

    expMinusDtK_Jastrow_vec = x.expMinusDtK_Jastrow_vec;
    expMinusDtV_Jastrow_vec = x.expMinusDtV_Jastrow_vec;
}

void MetroChains2s::move_deep(MetroChains2s &x)
{
    isInitialMetroChains = x.isInitialMetroChains;
    numOfChains = x.numOfChains;
    metropolis2sVec = move(x.metropolis2sVec);
    numOfBrackets = x.numOfBrackets;
    //
    metroLeft.resize(numOfBrackets*numOfChains);
    for(int i=1-1; i<=metroLeft.size()-1; i++){
        metroLeft[i] = move(x.metroLeft[i]);
    }
    metroRight.resize(numOfBrackets*numOfChains);
    for(int i=1-1; i<=metroRight.size()-1; i++){
        metroRight[i] = move(x.metroRight[i]);
    }
    //
    logw = x.logw;

    // size_t numOfReleasedSlice;
    numOfReleasedSlice = x.numOfReleasedSlice;
    // std::vector< MetroChains2sWalker > walkerRightVec;
    // walkerRightVec.resize(x.walkerRightVec.size());
    // if( numOfReleasedSlice >= 1 && walkerRightVec.size() >= 1){
    //     for(int i=1-1; i<=walkerRightVec.size()-1; i++){
    //         walkerRightVec[i] = move(x.walkerRightVec[i]);
    //     }
    // }
    // std::vector< TwoBodyForce > twoBodyForceVec;
    // twoBodyForceVec.resize(x.twoBodyForceVec.size());
    // if( numOfReleasedSlice >= 1 && twoBodyForceVec.size() >= 1){
    //     for(int i=1-1; i<=twoBodyForceVec.size()-1; i++){
    //         twoBodyForceVec[i] = move(x.twoBodyForceVec[i]);
    //     }
    // }
    // twoBodyForceVec_BP.resize(x.twoBodyForceVec_BP.size());
    // if( numOfReleasedSlice >= 1 && twoBodyForceVec_BP.size() >= 1){
    //     for(int i=1-1; i<=twoBodyForceVec_BP.size()-1; i++){
    //         twoBodyForceVec_BP[i] = move(x.twoBodyForceVec_BP[i]);
    //     }
    // }
    // std::vector< TwoBodyAux > twoBodyAuxVec;
    // twoBodyAuxVec.resize(x.twoBodyAuxVec.size());
    // if( numOfReleasedSlice >= 1 && twoBodyAuxVec.size() >= 1){
    //     for(int i=1-1; i<=twoBodyAuxVec.size()-1; i++){
    //         twoBodyAuxVec[i] = move(x.twoBodyAuxVec[i]);
    //     }
    // }
    // twoBodyAuxVec_BP.resize(x.twoBodyAuxVec_BP.size());
    // if( numOfReleasedSlice >= 1 && twoBodyAuxVec_BP.size() >= 1){
    //     for(int i=1-1; i<=twoBodyAuxVec_BP.size()-1; i++){
    //         twoBodyAuxVec_BP[i] = move(x.twoBodyAuxVec_BP[i]);
    //     }
    // }

    expMinusDtK_Jastrow_vec = x.expMinusDtK_Jastrow_vec;
    expMinusDtV_Jastrow_vec = x.expMinusDtV_Jastrow_vec;
}

////////////////////////////////////////////////////
////////////////////////////////////////////////////
////////////////////////////////////////////////////