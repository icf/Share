//
// Created by boruoshihao on 1/13/17.
//

#include "../../include/HubbardRealSpaceSOC_icf/HubbardSOC_icfMeasureObserveSDSD.h"
#include "afqmclab.h"

using namespace std;
using namespace tensor_hao;

HubbardSOC_icfMeasureObserveSDSD::HubbardSOC_icfMeasureObserveSDSD()
{
    initModelNullptr();
    reSet();
}

HubbardSOC_icfMeasureObserveSDSD::HubbardSOC_icfMeasureObserveSDSD(const HubbardSOC_icf &hubbardSOC_)
{
    setModel( hubbardSOC_ );
    reSet();
}

HubbardSOC_icfMeasureObserveSDSD::~HubbardSOC_icfMeasureObserveSDSD()
{

}

void HubbardSOC_icfMeasureObserveSDSD::reSet()
{
    HubbardSOC_icfMeasureCommuteSDSD::reSet();

    complex<double> zero(0,0);
    greenMatrixNum = zero;
    densityDensityNum = zero;
    splusSminusNum = zero;
    sminusSplusNum = zero;
    spairSpairNum = zero;
    szszNum = zero;
    sxsxNum = zero;
    sysyNum = zero;
}

void HubbardSOC_icfMeasureObserveSDSD::addMeasurement(SDSDOperation &sdsdOperation, complex<double> denIncrement)
{
    HubbardSOC_icfMeasureCommuteSDSD::addMeasurement(sdsdOperation, denIncrement);

    addSS(sdsdOperation, denIncrement);
}

void HubbardSOC_icfMeasureObserveSDSD::addMeasurement_energy(SDSDOperation &sdsdOperation, complex<double> denIncrement)
{
    HubbardSOC_icfMeasureCommuteSDSD::addMeasurement(sdsdOperation, denIncrement);
}

void HubbardSOC_icfMeasureObserveSDSD::write() const
{
    HubbardSOC_icfMeasureCommuteSDSD::write();
}

void HubbardSOC_icfMeasureObserveSDSD::write(std::string postfix) const
{
    HubbardSOC_icfMeasureCommuteSDSD::write(postfix);
    std::string szszNumName="szszNum"+postfix+".dat";
    std::string sxsxNumName="sxsxNum"+postfix+".dat";
    std::string sysyNumName="sysyNum"+postfix+".dat";
    writeThreadSum(szszNum.size(), szszNum.data(), szszNumName, ios::app);
    writeThreadSum(sxsxNum.size(), sxsxNum.data(), sxsxNumName, ios::app);
    writeThreadSum(sysyNum.size(), sysyNum.data(), sysyNumName, ios::app);
}

double HubbardSOC_icfMeasureObserveSDSD::getMemory() const
{
    double mem(0.0);
    mem += HubbardSOC_icfMeasureCommuteSDSD::getMemory();
    // mem += greenMatrixNum.getMemory();
    // mem += densityDensityNum.getMemory();
    // mem += splusSminusNum.getMemory();
    // mem += sminusSplusNum.getMemory();
    // mem += spairSpairNum.getMemory();

    return mem;
}

void HubbardSOC_icfMeasureObserveSDSD::addSS(SDSDOperation &sdSDOperation, complex<double> denIncrement)
{
    size_t L = getHubbardSOC_icf()->getL();
    TensorHao<complex<double>, 1> szsz(L); szsz=0.0;
    TensorHao<complex<double>, 1> sxsx(L); sxsx=0.0;
    TensorHao<complex<double>, 1> sysy(L); sysy=0.0;

        if( szszNum.rank(0) != L ) { szszNum.resize(L); szszNum = complex<double>(0,0); }
        if( sxsxNum.rank(0) != L ) { sxsxNum.resize(L); sxsxNum = complex<double>(0,0); }
        if( sysyNum.rank(0) != L ) { sysyNum.resize(L); sysyNum = complex<double>(0,0); }
        // 
        int L_x=0, L_y=0;
        if(L==8*8){
            L_x=8;
            L_y=8;
        }else if(L==16*16){
            L_x=16;
            L_y=16;
        }
        // 
        if(L_x !=0 && L_y!=0){  
            complex<double> local_iijj, local_iLiLjLjL;     
            complex<double> local_ii_jj, local_ij_ij, local_iLiL_jLjL, local_iLjL_iLjL, local_iLiL_jj, local_ii_jLjL;
            // ATTENTION: the following copy process of GreenMatrix takes resources, should minimize this process
            const TensorHao<complex<double>, 2> greenMatrix = sdSDOperation.returnGreenMatrix();
            for(int i=1-1; i<=L-1; i++){
                int i_x=i%L_x;
                int i_y=i/L_y;
                for(int j=1-1; j<=L-1; j++){
                    int j_x=j%L_x;
                    int j_y=j/L_y;
                    ///////////////////////////////
                    complex<double> localCdaggerCCdaggerC_iijj = localCdaggerCCdaggerC(greenMatrix,i,i,j,j);
                    complex<double> localCdaggerCCdaggerC_iLiLjLjL = localCdaggerCCdaggerC(greenMatrix,i+L,i+L,j+L,j+L);
                    complex<double> localCdaggerCCdaggerC_iLiLjj = localCdaggerCCdaggerC(greenMatrix,i+L,i+L,j,j);
                    complex<double> localCdaggerCCdaggerC_iijLjL = localCdaggerCCdaggerC(greenMatrix,i,i,j+L,j+L);
                    // 
                    complex<double> localCdaggerCCdaggerC_iiLjjL = localCdaggerCCdaggerC(greenMatrix,i,i+L,j,j+L);
                    complex<double> localCdaggerCCdaggerC_iLijLj = localCdaggerCCdaggerC(greenMatrix,i+L,i,j+L,j);
                    complex<double> localCdaggerCCdaggerC_iiLjLj = localCdaggerCCdaggerC(greenMatrix,i,i+L,j+L,j);
                    complex<double> localCdaggerCCdaggerC_iLijjL = localCdaggerCCdaggerC(greenMatrix,i+L,i,j,j+L);
                    ///////////////////////////////
                    int distance = ( (j_x - i_x + L_x)%L_x )*L_y + (j_y - i_y + L_y)%L_y;
                    szsz(distance) += 0.25/double(L) *(localCdaggerCCdaggerC_iijj + localCdaggerCCdaggerC_iLiLjLjL - localCdaggerCCdaggerC_iLiLjj - localCdaggerCCdaggerC_iijLjL);
                    sxsx(distance) += 0.25/double(L) *(localCdaggerCCdaggerC_iiLjjL + localCdaggerCCdaggerC_iLijLj + localCdaggerCCdaggerC_iiLjLj + localCdaggerCCdaggerC_iLijjL);
                    sysy(distance) += 0.25/double(L) * (-1.0) * (localCdaggerCCdaggerC_iiLjjL + localCdaggerCCdaggerC_iLijLj - localCdaggerCCdaggerC_iiLjLj - localCdaggerCCdaggerC_iLijjL);
                    ///////////////////////////////
                }
            }
        }
        
        szszNum += ( szsz * denIncrement );
        sxsxNum += ( sxsx * denIncrement );
        sysyNum += ( sysy * denIncrement );
}

complex<double> HubbardSOC_icfMeasureObserveSDSD::localCdaggerCCdaggerC(const TensorHao<complex<double>, 2> &greenMatrix, size_t i, size_t j, size_t k, size_t l)
{
    size_t L = getHubbardSOC_icf()->getL();
    complex<double> local_ijkl = 0.0;
    complex<double> local_ij_kl = 0.0;
    complex<double> local_il_jk = 0.0;
    local_ij_kl = greenMatrix(i,j) * greenMatrix(k,l);
    if(j != k){
        local_il_jk = greenMatrix(i,l) * (-1.0) * greenMatrix(k,j);
    }else{
        local_il_jk = greenMatrix(i,l) * (1.0 - greenMatrix(k,j));
    }
    local_ijkl = (local_ij_kl + local_il_jk);

    return local_ijkl;
}

void HubbardSOC_icfMeasureObserveSDSD::addGreenMatrix(SDSDOperation &sdsdOperation, complex<double> denIncrement)
{
    const TensorHao< complex<double>, 2 > &greenMatrix = sdsdOperation.returnGreenMatrix();

    size_t L2 = getHubbardSOC_icf()->getL() * 2;

    if( greenMatrixNum.rank(0) != L2 ) { greenMatrixNum.resize(L2, L2); greenMatrixNum = complex<double>(0,0); }

    greenMatrixNum += ( greenMatrix * denIncrement );
}

void HubbardSOC_icfMeasureObserveSDSD::addDensityDensity(SDSDOperation &sdsdOperation, complex<double> denIncrement)
{
    const TensorHao< complex<double>, 2 > &greenMatrix = sdsdOperation.returnGreenMatrix();

    size_t L2 = getHubbardSOC_icf()->getL() * 2;

    if( densityDensityNum.rank(0) != L2 ) { densityDensityNum.resize(L2, L2); densityDensityNum = complex<double>(0,0); }

    complex<double> temp;
    for(size_t j = 0; j < L2; ++j)
    {
        for(size_t i = 0; i < L2; ++i)
        {
            if( i==j ) temp = greenMatrix(i,i);
            else temp = greenMatrix(i,i) * greenMatrix(j,j) - greenMatrix(i,j)*greenMatrix(j,i);
            densityDensityNum(i, j) += ( temp * denIncrement );
        }
    }
}

void HubbardSOC_icfMeasureObserveSDSD::addSplusSminus(SDSDOperation &sdsdOperation, complex<double> denIncrement)
{
    const TensorHao< complex<double>, 2 > &greenMatrix = sdsdOperation.returnGreenMatrix();

    size_t L = getHubbardSOC_icf()->getL();

    if( splusSminusNum.rank(0) != L ) { splusSminusNum.resize(L, L); splusSminusNum = complex<double>(0,0); }

    complex<double> temp;
    for(size_t j = 0; j < L; ++j)
    {
        for(size_t i = 0; i < L; ++i)
        {
            if( i==j )
            {
                temp  = greenMatrix(i,i);
                temp -= ( greenMatrix(i,i)*greenMatrix(i+L,i+L)-greenMatrix(i,i+L)*greenMatrix(i+L,i) );
            }
            else
            {
                temp = -greenMatrix(i,j)*greenMatrix(j+L,i+L)+greenMatrix(i,i+L)*greenMatrix(j+L, j);
            }
            splusSminusNum(i,j) += ( temp * denIncrement );
        }
    }
}

void HubbardSOC_icfMeasureObserveSDSD::addSminusSplus(SDSDOperation &sdsdOperation, complex<double> denIncrement)
{
    const TensorHao< complex<double>, 2 > &greenMatrix = sdsdOperation.returnGreenMatrix();

    size_t L = getHubbardSOC_icf()->getL();

    if( sminusSplusNum.rank(0) != L ) { sminusSplusNum.resize(L, L); sminusSplusNum = complex<double>(0,0); }

    complex<double> temp;
    for(size_t j = 0; j < L; ++j)
    {
        for(size_t i = 0; i < L; ++i)
        {
            if( i==j )
            {
                temp  = greenMatrix(i+L, i+L);
                temp -= ( greenMatrix(i,i)*greenMatrix(i+L,i+L)-greenMatrix(i,i+L)*greenMatrix(i+L,i) );
            }
            else
            {
                temp = -greenMatrix(i+L,j+L)*greenMatrix(j,i)+greenMatrix(i+L,i)*greenMatrix(j, j+L);
            }
            sminusSplusNum(i,j) += ( temp * denIncrement );
        }
    }
}

void HubbardSOC_icfMeasureObserveSDSD::addSpairSpair(SDSDOperation &sdsdOperation, complex<double> denIncrement)
{
    const TensorHao< complex<double>, 2 > &greenMatrix = sdsdOperation.returnGreenMatrix();

    size_t L = getHubbardSOC_icf()->getL();

    if( spairSpairNum.rank(0) != L ) { spairSpairNum.resize(L, L); spairSpairNum = complex<double>(0,0); }

    complex<double> temp;
    for(size_t j=0; j<L; j++)
    {
        for(size_t i=0; i<L; i++)
        {
            temp = greenMatrix(i,j)*greenMatrix(i+L,j+L)-greenMatrix(i,j+L)*greenMatrix(i+L,j);
            spairSpairNum(i,j) += ( temp * denIncrement );
        }
    }
}

HubbardSOC_icfMeasureObserveSDSD::HubbardSOC_icfMeasureObserveSDSD(const HubbardSOC_icfMeasureObserveSDSD &x)
{

}

HubbardSOC_icfMeasureObserveSDSD &HubbardSOC_icfMeasureObserveSDSD::operator=(const HubbardSOC_icfMeasureObserveSDSD &x)
{
    return *this;
}