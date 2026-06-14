//
// Created by boruoshihao on 1/13/17.
//

#include "../../include/HubbardRealSpaceSOC_icf/HubbardSOC_icfMeasureObserveSD2sSD2s.h"
#include "afqmclab.h"

using namespace std;
using namespace tensor_hao;

HubbardSOC_icfMeasureObserveSD2sSD2s::HubbardSOC_icfMeasureObserveSD2sSD2s()
{
    initModelNullptr();
    reSet();
}

HubbardSOC_icfMeasureObserveSD2sSD2s::HubbardSOC_icfMeasureObserveSD2sSD2s(const HubbardSOC_icf &hubbardSOC_)
{
    setModel( hubbardSOC_ );
    reSet();
}

HubbardSOC_icfMeasureObserveSD2sSD2s::~HubbardSOC_icfMeasureObserveSD2sSD2s()
{

}

void HubbardSOC_icfMeasureObserveSD2sSD2s::reSet()
{
    HubbardSOC_icfMeasureCommuteSD2sSD2s::reSet();

    complex<double> zero(0,0);
    greenMatrixUpNum = zero;
    greenMatrixDnNum = zero;
    densityDensityNum = zero;
    splusSminusNum = zero;
    sminusSplusNum = zero;
    spairSpairNum = zero;
    szszNum = zero;
    sxsxNum = zero;
    sysyNum = zero;
    DDNum   = zero;
}

void HubbardSOC_icfMeasureObserveSD2sSD2s::addMeasurement(SD2sSD2sOperation &sd2ssd2sOperation, complex<double> denIncrement)
{
    HubbardSOC_icfMeasureCommuteSD2sSD2s::addMeasurement(sd2ssd2sOperation, denIncrement);

    addSS(sd2ssd2sOperation, denIncrement);
    // addDwavePairingNum(sd2ssd2sOperation, denIncrement);
    addGreenMatrix(sd2ssd2sOperation, denIncrement);
}

void HubbardSOC_icfMeasureObserveSD2sSD2s::addMeasurement_energy(SD2sSD2sOperation &sd2ssd2sOperation, complex<double> denIncrement)
{
    HubbardSOC_icfMeasureCommuteSD2sSD2s::addMeasurement(sd2ssd2sOperation, denIncrement);
}

void HubbardSOC_icfMeasureObserveSD2sSD2s::write() const
{
    HubbardSOC_icfMeasureCommuteSD2sSD2s::write();
}

void HubbardSOC_icfMeasureObserveSD2sSD2s::write(std::string postfix) const
{
    HubbardSOC_icfMeasureCommuteSD2sSD2s::write(postfix);
    std::string szszNumName="szszNum"+postfix+".dat";
    std::string sxsxNumName="sxsxNum"+postfix+".dat";
    std::string sysyNumName="sysyNum"+postfix+".dat";
    std::string greenMatrixUpNumName="greenMatrixUpNum"+postfix+".dat";
    std::string greenMatrixDnNumName="greenMatrixDnNum"+postfix+".dat";
    // std::string DDNumName="DDNum"+postfix+".dat";
    writeThreadSum(szszNum.size(), szszNum.data(), szszNumName, ios::app);
    writeThreadSum(sxsxNum.size(), sxsxNum.data(), sxsxNumName, ios::app);
    writeThreadSum(sysyNum.size(), sysyNum.data(), sysyNumName, ios::app);
    writeThreadSum(greenMatrixUpNum.size(), greenMatrixUpNum.data(), greenMatrixUpNumName, ios::app);
    writeThreadSum(greenMatrixDnNum.size(), greenMatrixDnNum.data(), greenMatrixDnNumName, ios::app);
    // writeThreadSum(DDNum.size(), DDNum.data(), DDNumName, ios::app);
}

double HubbardSOC_icfMeasureObserveSD2sSD2s::getMemory() const
{
    double mem(0.0);
    mem += HubbardSOC_icfMeasureCommuteSD2sSD2s::getMemory();
    mem += greenMatrixUpNum.getMemory();
    mem += greenMatrixDnNum.getMemory();
    // mem += densityDensityNum.getMemory();
    // mem += splusSminusNum.getMemory();
    // mem += sminusSplusNum.getMemory();
    // mem += spairSpairNum.getMemory();

    return mem;
}

void HubbardSOC_icfMeasureObserveSD2sSD2s::addSS(SD2sSD2sOperation &sd2ssd2sOperation, complex<double> denIncrement)
{
    size_t L = getHubbardSOC_icf()->getL();
    TensorHao<complex<double>, 1> szsz(L); szsz=0.0;
    TensorHao<complex<double>, 1> sxsx(L); sxsx=0.0;
    // TensorHao<complex<double>, 1> sysy(L); sysy=0.0;

        if( szszNum.rank(0) != L ) { szszNum.resize(L); szszNum = complex<double>(0,0); }
        if( sxsxNum.rank(0) != L ) { sxsxNum.resize(L); sxsxNum = complex<double>(0,0); }
        if( sysyNum.rank(0) != L ) { sysyNum.resize(L); sysyNum = complex<double>(0,0); }
        // 
        int L_x=0, L_y=0;
        if(L==8*8){
            L_x=8;
            L_y=8;
        }else if(L==10*10){
            L_x=10;
            L_y=10;
        }else if(L==16*16){
            L_x=16;
            L_y=16;
        }else if(L==24*24){
            L_x=24;
            L_y=24;
        }
        // 
        if(L_x !=0 && L_y!=0){  
            complex<double> local_iijj, local_iLiLjLjL;     
            complex<double> local_ii_jj, local_ij_ij, local_iLiL_jLjL, local_iLjL_iLjL, local_iLiL_jj, local_ii_jLjL;
            // ATTENTION: the following copy process of GreenMatrix takes resources, should minimize this process
            const TensorHao<complex<double>, 2> greenMatrixUp = sd2ssd2sOperation.returnGreenMatrixUp();
            const TensorHao<complex<double>, 2> greenMatrixDn = sd2ssd2sOperation.returnGreenMatrixDn();
            for(int i=1-1; i<=L-1; i++){
                int i_x=i%L_x;
                int i_y=i/L_x;
                for(int j=1-1; j<=L-1; j++){
                    int j_x=j%L_x;
                    int j_y=j/L_x;
                    ///////////////////////////////
                    complex<double> localCdaggerCCdaggerC_iijj = localCdaggerCCdaggerC(greenMatrixUp,i,i,j,j);
                    complex<double> localCdaggerCCdaggerC_iLiLjLjL = localCdaggerLCLCdaggerLCL(greenMatrixDn,i,i,j,j);
                    complex<double> localCdaggerCCdaggerC_iLiLjj = localCdaggerLCLCdaggerC(greenMatrixUp, greenMatrixDn,i,i,j,j);
                    complex<double> localCdaggerCCdaggerC_iijLjL = localCdaggerCCdaggerLCL(greenMatrixUp, greenMatrixDn,i,i,j,j);
                    // 
                    complex<double> localCdaggerCCdaggerC_iiLjLj = localCdaggerCLCdaggerLC(greenMatrixUp, greenMatrixDn,i,i,j,j);
                    complex<double> localCdaggerCCdaggerC_iLijjL = localCdaggerLCCdaggerCL(greenMatrixUp, greenMatrixDn,i,i,j,j);
                    ///////////////////////////////
                    int distance = ( (j_x - i_x + L_x)%L_x )*L_y + (j_y - i_y + L_y)%L_y;
                    szsz(distance) += 0.25/double(L) *(localCdaggerCCdaggerC_iijj + localCdaggerCCdaggerC_iLiLjLjL - localCdaggerCCdaggerC_iLiLjj - localCdaggerCCdaggerC_iijLjL);
                    sxsx(distance) += 0.25/double(L) *( localCdaggerCCdaggerC_iiLjLj + localCdaggerCCdaggerC_iLijjL);
                    ///////////////////////////////
                }
            }
        }
        
        szszNum += ( szsz * denIncrement );
        sxsxNum += ( sxsx * denIncrement );
        sysyNum += ( sxsx * denIncrement );
}

void HubbardSOC_icfMeasureObserveSD2sSD2s::addDwavePairingNum(SD2sSD2sOperation &sd2ssd2sOperation, complex<double> denIncrement)
{
    size_t L = getHubbardSOC_icf()->getL();
    TensorHao<complex<double>, 1> DD(L); DD=0.0;

        if( DDNum.rank(0) != L ) { DDNum.resize(L); DDNum = complex<double>(0,0); }
        // 
        int L_x=0, L_y=0;
        if(L==8*8){
            L_x=8;
            L_y=8;
        }else if(L==10*10){
            L_x=10;
            L_y=10;
        }else if(L==16*16){
            L_x=16;
            L_y=16;
        }else if(L==24*24){
            L_x=24;
            L_y=24;
        }
        // 
        if(L_x !=0 && L_y!=0){  
            complex<double> local_iijj, local_iLiLjLjL;     
            complex<double> local_ii_jj, local_ij_ij, local_iLiL_jLjL, local_iLjL_iLjL, local_iLiL_jj, local_ii_jLjL;
            // ATTENTION: the following copy process of GreenMatrix takes resources, should minimize this process
            const TensorHao<complex<double>, 2> greenMatrixUp = sd2ssd2sOperation.returnGreenMatrixUp();
            const TensorHao<complex<double>, 2> greenMatrixDn = sd2ssd2sOperation.returnGreenMatrixDn();
            for(int i=1-1; i<=L-1; i++){
            int i_x=i%L_x;
            int i_y=i/L_x;
            // for(int NNhop_i=0; NNhop_i<=4-1; NNhop_i++){
                int NNhop_i=2;
                int ip;
                if(NNhop_i == 0){
                    // hop +x
                    int ip_x=(i_x+1+L_x)%L_x;
                    int ip_y=i_y;
                    ip = ip_y*L_x + ip_x;
                }else if(NNhop_i == 1){
                    // hop -x
                    int ip_x=(i_x-1+L_x)%L_x;
                    int ip_y=i_y;
                    ip = ip_y*L_x + ip_x;
                }else if(NNhop_i == 2){
                    // hop y
                    int ip_x=i_x;
                    int ip_y=(i_y+1+L_y)%L_y;
                    ip = ip_y*L_x + ip_x;
                }else if(NNhop_i == 3){
                    // hop -y
                    int ip_x=i_x;
                    int ip_y=(i_y-1+L_y)%L_y;
                    ip = ip_y*L_x + ip_x;
                }
                for(int j=1-1; j<=L-1; j++){
                int j_x=j%L_x;
                int j_y=j/L_x;
                // for(int NNhop_j=0; NNhop_j<=4-1; NNhop_j++){
                    int NNhop_j=2;
                    int jp;
                    if(NNhop_j == 0){
                        // hop +x
                        int jp_x=(j_x+1+L_x)%L_x;
                        int jp_y=j_y;
                        jp = jp_y*L_x + jp_x;
                    }else if(NNhop_j == 1){
                        // hop -x
                        int jp_x=(j_x-1+L_x)%L_x;
                        int jp_y=j_y;
                        jp = jp_y*L_x + jp_x;
                    }else if(NNhop_j == 2){
                        // hop y
                        int jp_x=j_x;
                        int jp_y=(j_y+1+L_y)%L_y;
                        jp = jp_y*L_x + jp_x;
                    }else if(NNhop_j == 3){
                        // hop -y
                        int jp_x=j_x;
                        int jp_y=(j_y-1+L_y)%L_y;
                        jp = jp_y*L_x + jp_x;
                    }
                    ///////////////////////////////
                    // c^\dagger_{j',dn}c^\dagger_{i',up}c_{i,up}c_{j,dn}
                    complex<double> localCdaggerLCdaggerLCC_jpipij = localCdaggerLCdaggerCCL(greenMatrixUp, greenMatrixDn,jp,ip,i,j);
                    // - c^\dagger_{j',dn}c^\dagger_{i',up}c_{i,dn}c_{j,up}
                    complex<double> localCdaggerLCdaggerCLC_jpipij = localCdaggerLCdaggerCLC(greenMatrixUp, greenMatrixDn,jp,ip,i,j);
                    // - c^\dagger_{j',up}c^\dagger_{i',dn}c_{i,up}c_{j,dn}
                    complex<double> localCdaggerCdaggerLCCL_jpipij = localCdaggerCdaggerLCCL(greenMatrixUp, greenMatrixDn,jp,ip,i,j);
                    // c^\dagger_{j',up}c^\dagger_{i',dn}c_{i,dn}c_{j,up}
                    complex<double> localCdaggerCdaggerLCLC_jpipij = localCdaggerCdaggerLCLC(greenMatrixUp, greenMatrixDn,jp,ip,i,j);
                    // 
                    ///////////////////////////////
                    int distance = ( (j_x - i_x + L_x)%L_x )*L_y + (j_y - i_y + L_y)%L_y;
                    DD(distance) += 0.25/double(L) *(localCdaggerLCdaggerLCC_jpipij - localCdaggerLCdaggerCLC_jpipij - localCdaggerCdaggerLCCL_jpipij + localCdaggerCdaggerLCLC_jpipij);
                    ///////////////////////////////
                // }
                }
            // }
            }
        }
        // 
        DDNum += ( DD * denIncrement );
}

complex<double> HubbardSOC_icfMeasureObserveSD2sSD2s::localCdaggerCCdaggerC(const TensorHao<complex<double>, 2> &greenMatrixUp, size_t i, size_t j, size_t k, size_t l)
{
    size_t L = getHubbardSOC_icf()->getL();
    complex<double> local_ijkl = 0.0;
    complex<double> local_ij_kl = 0.0;
    complex<double> local_il_jk = 0.0;

    local_ij_kl = greenMatrixUp(i,j) * greenMatrixUp(k,l);
    if(j != k){
        local_il_jk = greenMatrixUp(i,l) * (-1.0) * greenMatrixUp(k,j);
    }else{
        local_il_jk = greenMatrixUp(i,l) * (1.0 - greenMatrixUp(k,j));
    }
    local_ijkl = (local_ij_kl + local_il_jk);

    return local_ijkl;
}

complex<double> HubbardSOC_icfMeasureObserveSD2sSD2s::localCdaggerLCLCdaggerLCL(const TensorHao<complex<double>, 2> &greenMatrixDn, size_t i, size_t j, size_t k, size_t l)
{
    size_t L = getHubbardSOC_icf()->getL();
    complex<double> local_ijkl = 0.0;
    complex<double> local_ij_kl = 0.0;
    complex<double> local_il_jk = 0.0;

    local_ij_kl = greenMatrixDn(i,j) * greenMatrixDn(k,l);
    if(j != k){
        local_il_jk = greenMatrixDn(i,l) * (-1.0) * greenMatrixDn(k,j);
    }else{
        local_il_jk = greenMatrixDn(i,l) * (1.0 - greenMatrixDn(k,j));
    }
    local_ijkl = (local_ij_kl + local_il_jk);

    return local_ijkl;
}

complex<double> HubbardSOC_icfMeasureObserveSD2sSD2s::localCdaggerLCLCdaggerC(const TensorHao<complex<double>, 2> &greenMatrixUp, const TensorHao<complex<double>, 2> &greenMatrixDn, size_t i, size_t j, size_t k, size_t l)
{
    size_t L = getHubbardSOC_icf()->getL();
    complex<double> local_ijkl = 0.0;
    complex<double> local_ij_kl = 0.0;

    local_ij_kl = greenMatrixDn(i,j) * greenMatrixUp(k,l);

    local_ijkl = (local_ij_kl );

    return local_ijkl;
}

complex<double> HubbardSOC_icfMeasureObserveSD2sSD2s::localCdaggerCCdaggerLCL(const TensorHao<complex<double>, 2> &greenMatrixUp, const TensorHao<complex<double>, 2> &greenMatrixDn, size_t i, size_t j, size_t k, size_t l)
{
    size_t L = getHubbardSOC_icf()->getL();
    complex<double> local_ijkl = 0.0;
    complex<double> local_ij_kl = 0.0;

    local_ij_kl = greenMatrixUp(i,j) * greenMatrixDn(k,l);

    local_ijkl = (local_ij_kl );

    return local_ijkl;
}

complex<double> HubbardSOC_icfMeasureObserveSD2sSD2s::localCdaggerCLCdaggerLC(const TensorHao<complex<double>, 2> &greenMatrixUp, const TensorHao<complex<double>, 2> &greenMatrixDn, size_t i, size_t j, size_t k, size_t l)
{
    size_t L = getHubbardSOC_icf()->getL();
    complex<double> local_ijkl = 0.0;
    complex<double> local_il_jk = 0.0;

    if(j != k){
        local_il_jk = greenMatrixUp(i,l) * (-1.0) * greenMatrixDn(k,j);
    }else{
        local_il_jk = greenMatrixUp(i,l) * (1.0 - greenMatrixDn(k,j));
    }
    local_ijkl = (local_il_jk);

    return local_ijkl;
}

complex<double> HubbardSOC_icfMeasureObserveSD2sSD2s::localCdaggerLCCdaggerCL(const TensorHao<complex<double>, 2> &greenMatrixUp, const TensorHao<complex<double>, 2> &greenMatrixDn, size_t i, size_t j, size_t k, size_t l)
{
    size_t L = getHubbardSOC_icf()->getL();
    complex<double> local_ijkl = 0.0;
    complex<double> local_il_jk = 0.0;

    if(j != k){
        local_il_jk = greenMatrixDn(i,l) * (-1.0) * greenMatrixUp(k,j);
    }else{
        local_il_jk = greenMatrixDn(i,l) * (1.0 - greenMatrixUp(k,j));
    }
    local_ijkl = (local_il_jk);

    return local_ijkl;
}

// DD_icf
complex<double> HubbardSOC_icfMeasureObserveSD2sSD2s::localCdaggerLCdaggerCCL(const TensorHao<complex<double>, 2> &greenMatrixUp, const TensorHao<complex<double>, 2> &greenMatrixDn, size_t i, size_t j, size_t k, size_t l)
{
    complex<double> local_ijkl = 0.0;
    complex<double> local_il_jk = 0.0;
    // 
    local_il_jk = greenMatrixDn(i,l) * greenMatrixUp(j,k);
    // 
    local_ijkl += ( local_il_jk);

    return local_ijkl;
}

complex<double> HubbardSOC_icfMeasureObserveSD2sSD2s::localCdaggerLCdaggerCLC(const TensorHao<complex<double>, 2> &greenMatrixUp, const TensorHao<complex<double>, 2> &greenMatrixDn, size_t i, size_t j, size_t k, size_t l)
{
    complex<double> local_ijkl = 0.0;
    complex<double> local_ik_jl = 0.0;
    // 
    local_ik_jl = greenMatrixDn(i,k) * greenMatrixUp(j,l);
    // 
    local_ijkl += ( -1.0 * local_ik_jl);

    return local_ijkl;
}

complex<double> HubbardSOC_icfMeasureObserveSD2sSD2s::localCdaggerCdaggerLCCL(const TensorHao<complex<double>, 2> &greenMatrixUp, const TensorHao<complex<double>, 2> &greenMatrixDn, size_t i, size_t j, size_t k, size_t l)
{
    complex<double> local_ijkl = 0.0;
    complex<double> local_ik_jl = 0.0;
    // 
    local_ik_jl = greenMatrixUp(i,k) * greenMatrixDn(j,l);
    // 
    local_ijkl += ( -1.0 * local_ik_jl);

    return local_ijkl;
}

complex<double> HubbardSOC_icfMeasureObserveSD2sSD2s::localCdaggerCdaggerLCLC(const TensorHao<complex<double>, 2> &greenMatrixUp, const TensorHao<complex<double>, 2> &greenMatrixDn, size_t i, size_t j, size_t k, size_t l)
{
    complex<double> local_ijkl = 0.0;
    complex<double> local_il_jk = 0.0;
    // 
    local_il_jk = greenMatrixUp(i,l) * greenMatrixDn(j,k);
    // 
    local_ijkl += ( local_il_jk);

    return local_ijkl;
}

void HubbardSOC_icfMeasureObserveSD2sSD2s::addGreenMatrix(SD2sSD2sOperation &sd2ssd2sOperation, complex<double> denIncrement)
{
    const TensorHao< complex<double>, 2 > &greenMatrixUp = sd2ssd2sOperation.returnGreenMatrixUp();
    const TensorHao< complex<double>, 2 > &greenMatrixDn = sd2ssd2sOperation.returnGreenMatrixDn();

    size_t L = getHubbardSOC_icf()->getL();

    if( greenMatrixUpNum.rank(0) != L ) { greenMatrixUpNum.resize(L, L); greenMatrixUpNum = complex<double>(0,0); }
    if( greenMatrixDnNum.rank(0) != L ) { greenMatrixDnNum.resize(L, L); greenMatrixDnNum = complex<double>(0,0); }

    greenMatrixUpNum += ( greenMatrixUp * denIncrement );
    greenMatrixDnNum += ( greenMatrixDn * denIncrement );
}

void HubbardSOC_icfMeasureObserveSD2sSD2s::addDensityDensity(SD2sSD2sOperation &sd2ssd2sOperation, complex<double> denIncrement)
{
    // const TensorHao< complex<double>, 2 > &greenMatrix = sd2ssd2sOperation.returnGreenMatrix();

    // size_t L2 = getHubbardSOC_icf()->getL() * 2;

    // if( densityDensityNum.rank(0) != L2 ) { densityDensityNum.resize(L2, L2); densityDensityNum = complex<double>(0,0); }

    // complex<double> temp;
    // for(size_t j = 0; j < L2; ++j)
    // {
    //     for(size_t i = 0; i < L2; ++i)
    //     {
    //         if( i==j ) temp = greenMatrix(i,i);
    //         else temp = greenMatrix(i,i) * greenMatrix(j,j) - greenMatrix(i,j)*greenMatrix(j,i);
    //         densityDensityNum(i, j) += ( temp * denIncrement );
    //     }
    // }
}

void HubbardSOC_icfMeasureObserveSD2sSD2s::addSplusSminus(SD2sSD2sOperation &sd2ssd2sOperation, complex<double> denIncrement)
{
    // const TensorHao< complex<double>, 2 > &greenMatrix = sd2ssd2sOperation.returnGreenMatrix();

    // size_t L = getHubbardSOC_icf()->getL();

    // if( splusSminusNum.rank(0) != L ) { splusSminusNum.resize(L, L); splusSminusNum = complex<double>(0,0); }

    // complex<double> temp;
    // for(size_t j = 0; j < L; ++j)
    // {
    //     for(size_t i = 0; i < L; ++i)
    //     {
    //         if( i==j )
    //         {
    //             temp  = greenMatrix(i,i);
    //             temp -= ( greenMatrix(i,i)*greenMatrix(i+L,i+L)-greenMatrix(i,i+L)*greenMatrix(i+L,i) );
    //         }
    //         else
    //         {
    //             temp = -greenMatrix(i,j)*greenMatrix(j+L,i+L)+greenMatrix(i,i+L)*greenMatrix(j+L, j);
    //         }
    //         splusSminusNum(i,j) += ( temp * denIncrement );
    //     }
    // }
}

void HubbardSOC_icfMeasureObserveSD2sSD2s::addSminusSplus(SD2sSD2sOperation &sd2ssd2sOperation, complex<double> denIncrement)
{
    // const TensorHao< complex<double>, 2 > &greenMatrix = sd2ssd2sOperation.returnGreenMatrix();

    // size_t L = getHubbardSOC_icf()->getL();

    // if( sminusSplusNum.rank(0) != L ) { sminusSplusNum.resize(L, L); sminusSplusNum = complex<double>(0,0); }

    // complex<double> temp;
    // for(size_t j = 0; j < L; ++j)
    // {
    //     for(size_t i = 0; i < L; ++i)
    //     {
    //         if( i==j )
    //         {
    //             temp  = greenMatrix(i+L, i+L);
    //             temp -= ( greenMatrix(i,i)*greenMatrix(i+L,i+L)-greenMatrix(i,i+L)*greenMatrix(i+L,i) );
    //         }
    //         else
    //         {
    //             temp = -greenMatrix(i+L,j+L)*greenMatrix(j,i)+greenMatrix(i+L,i)*greenMatrix(j, j+L);
    //         }
    //         sminusSplusNum(i,j) += ( temp * denIncrement );
    //     }
    // }
}

void HubbardSOC_icfMeasureObserveSD2sSD2s::addSpairSpair(SD2sSD2sOperation &sd2ssd2sOperation, complex<double> denIncrement)
{
    // const TensorHao< complex<double>, 2 > &greenMatrix = sd2ssd2sOperation.returnGreenMatrix();

    // size_t L = getHubbardSOC_icf()->getL();

    // if( spairSpairNum.rank(0) != L ) { spairSpairNum.resize(L, L); spairSpairNum = complex<double>(0,0); }

    // complex<double> temp;
    // for(size_t j=0; j<L; j++)
    // {
    //     for(size_t i=0; i<L; i++)
    //     {
    //         temp = greenMatrix(i,j)*greenMatrix(i+L,j+L)-greenMatrix(i,j+L)*greenMatrix(i+L,j);
    //         spairSpairNum(i,j) += ( temp * denIncrement );
    //     }
    // }
}

HubbardSOC_icfMeasureObserveSD2sSD2s::HubbardSOC_icfMeasureObserveSD2sSD2s(const HubbardSOC_icfMeasureObserveSD2sSD2s &x)
{

}

HubbardSOC_icfMeasureObserveSD2sSD2s &HubbardSOC_icfMeasureObserveSD2sSD2s::operator=(const HubbardSOC_icfMeasureObserveSD2sSD2s &x)
{
    return *this;
}