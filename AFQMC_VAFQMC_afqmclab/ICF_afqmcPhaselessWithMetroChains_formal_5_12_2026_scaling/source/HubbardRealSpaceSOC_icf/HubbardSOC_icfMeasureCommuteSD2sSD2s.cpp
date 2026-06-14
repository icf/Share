//
// Created by boruoshihao on 1/13/17.
//

#include "../../include/HubbardRealSpaceSOC_icf/HubbardSOC_icfMeasureCommuteSD2sSD2s.h"
#include "afqmclab.h"

using namespace std;
using namespace tensor_hao;

HubbardSOC_icfMeasureCommuteSD2sSD2s::HubbardSOC_icfMeasureCommuteSD2sSD2s()
{
    initModelNullptr();
    reSet();
}

HubbardSOC_icfMeasureCommuteSD2sSD2s::HubbardSOC_icfMeasureCommuteSD2sSD2s(const HubbardSOC_icf &hubbardSOC_)
{
    setModel(hubbardSOC_);
    reSet();
}

HubbardSOC_icfMeasureCommuteSD2sSD2s::~HubbardSOC_icfMeasureCommuteSD2sSD2s()
{

}

const HubbardSOC_icf *HubbardSOC_icfMeasureCommuteSD2sSD2s::getHubbardSOC_icf() const
{
    return hubbardSOC_icf;
}

void HubbardSOC_icfMeasureCommuteSD2sSD2s::initModelNullptr()
{
    hubbardSOC_icf = nullptr;
}

void HubbardSOC_icfMeasureCommuteSD2sSD2s::setModel(const HubbardSOC_icf &hubbardSOC_)
{
    hubbardSOC_icf = &hubbardSOC_;
}

void HubbardSOC_icfMeasureCommuteSD2sSD2s::reSet()
{
    complex<double> zero(0,0);
    den = zero;
    HNum = zero;
    KNum = zero;
    VNum = zero;
    RNum = zero;
}

complex<double> HubbardSOC_icfMeasureCommuteSD2sSD2s::returnEnergy()
{
    complex<double> Htot   = MPISum(HNum);
    complex<double> denTot = MPISum(den);
    complex<double> energy;
    if( MPIRank() == 0 ) energy = Htot/denTot;
    MPIBcast(energy);
    return energy;
}

void HubbardSOC_icfMeasureCommuteSD2sSD2s::addMeasurement(SD2sSD2sOperation &sd2ssd2sOperation, complex<double> denIncrement)
{
    den += denIncrement;

    const TensorHao< complex<double>, 2 > &greenMatrixUp = sd2ssd2sOperation.returnGreenMatrixUp();
    const TensorHao< complex<double>, 2 > &greenMatrixDn = sd2ssd2sOperation.returnGreenMatrixDn();

    addEnergy(greenMatrixUp, greenMatrixDn, denIncrement);
}

NiupNidnForce HubbardSOC_icfMeasureCommuteSD2sSD2s::getForce(const NiupNidn &niupNidn, SD2sSD2sOperation &sd2ssd2sOperation, double cap)
{
    size_t halfL = niupNidn.getL(); const string &decompType = niupNidn.getDecompType();

    TensorHao< complex<double>, 1 > backGround(halfL);
    if( decompType == "densityCharge" )
    {
        const TensorHao< complex<double>, 1 > &greenDiagonalUp = sd2ssd2sOperation.returnGreenDiagonalUp();
        const TensorHao< complex<double>, 1 > &greenDiagonalDn = sd2ssd2sOperation.returnGreenDiagonalDn();
        for(size_t i = 0; i < halfL; ++i) backGround(i) = greenDiagonalUp(i) + greenDiagonalDn(i) -1.0;
    }
    else if( decompType == "densitySpin" )
    {
        const TensorHao< complex<double>, 1 > &greenDiagonalUp = sd2ssd2sOperation.returnGreenDiagonalUp();
        const TensorHao< complex<double>, 1 > &greenDiagonalDn = sd2ssd2sOperation.returnGreenDiagonalDn();
        for(size_t i = 0; i < halfL; ++i) backGround(i) = greenDiagonalUp(i) - greenDiagonalDn(i);
    }
    else
    {
        cout<<"Error! Can not find the matched decompType! "<<decompType<<endl;
        exit(1);
    }

    NiupNidnForce force(halfL);
    const TensorHao<complex<double>, 1> &gamma = niupNidn.getGamma();
    for (size_t i = 0; i < halfL; ++i)
    {
        force(i) = ( gamma(i) * backGround(i) ).real();
        if( force(i) >  cap ) force(i) =  cap;
        if( force(i) < -cap ) force(i) = -cap;
    }

    return force;
}

void HubbardSOC_icfMeasureCommuteSD2sSD2s::write() const
{
    writeThreadSum(den, "den.dat", ios::app);
    writeThreadSum(HNum, "HNum.dat", ios::app);
}

void HubbardSOC_icfMeasureCommuteSD2sSD2s::write(std::string postfix) const
{
    std::string denName="den"+postfix+".dat";
    std::string HNumName="HNum"+postfix+".dat";
    writeThreadSum(den, denName, ios::app);
    writeThreadSum(HNum, HNumName, ios::app);
}

void HubbardSOC_icfMeasureCommuteSD2sSD2s::writeKNumVumRum() const
{
    writeThreadSum(KNum, "KNum.dat", ios::app);
    writeThreadSum(VNum, "VNum.dat", ios::app);
    writeThreadSum(RNum, "RNum.dat", ios::app);
}

double HubbardSOC_icfMeasureCommuteSD2sSD2s::getMemory() const
{
    return 8.0+16.0*5;
}

HubbardSOC_icfMeasureCommuteSD2sSD2s::HubbardSOC_icfMeasureCommuteSD2sSD2s(const HubbardSOC_icfMeasureCommuteSD2sSD2s &x)
{

}

HubbardSOC_icfMeasureCommuteSD2sSD2s &HubbardSOC_icfMeasureCommuteSD2sSD2s::operator=(const HubbardSOC_icfMeasureCommuteSD2sSD2s &x)
{
    return *this;
}

void HubbardSOC_icfMeasureCommuteSD2sSD2s::addEnergy(const TensorHao<complex<double>, 2> &greenMatrixUp, const TensorHao<complex<double>, 2> &greenMatrixDn, complex<double> denIncrement)
{
    complex<double> Kenergy(0,0), Venergy(0,0), Renergy(0,0);

    size_t L  = hubbardSOC_icf->getL(); size_t L2 = L*2;

    //Add K
    const TensorHao< complex<double>, 2 > &K = hubbardSOC_icf->getK();
    for(size_t i = 0; i < L; ++i)
    {
        for(size_t j = 0; j < L; ++j)
        {
            Kenergy += K(j,i) * greenMatrixUp(j,i);
        }
    }

    for(size_t i = 0; i < L; ++i)
    {
        for(size_t j = 0; j < L; ++j)
        {
            Kenergy += K(j+L,i+L) * greenMatrixDn(j,i);
        }
    }

    //Add U
    const TensorHao< double, 1> &U = hubbardSOC_icf->getU();
    for(size_t i = 0; i < L; ++i)
    {
        // Venergy += U(i) * ( greenMatrix(i,i)*greenMatrix(i+L,i+L) - greenMatrix(i, i+L)*greenMatrix(i+L, i) );
        Venergy += U(i) * ( greenMatrixUp(i,i)*greenMatrixDn(i,i) );
    }

    //Add mu and pinning field
    const TensorHao< double, 1> &mu = hubbardSOC_icf->getMu();
    const TensorHao< double, 1> &hx = hubbardSOC_icf->getHx();
    const TensorHao< double, 1> &hy = hubbardSOC_icf->getHy();
    const TensorHao< double, 1> &hz = hubbardSOC_icf->getHz();
    for(size_t i = 0; i < L; ++i)
    {
        Renergy += ( -mu(i) + hz(i)*0.5 ) * greenMatrixUp(i,i);
        Renergy += ( -mu(i) - hz(i)*0.5 ) * greenMatrixDn(i,i);
        // Renergy += complex<double>( hx(i)*0.5, -hy(i)*0.5 ) * greenMatrix(i, i+L);
        // Renergy += complex<double>( hx(i)*0.5,  hy(i)*0.5 ) * greenMatrix(i+L, i);
    }

    HNum += ( ( Kenergy + Venergy + Renergy ) * denIncrement );
    KNum += ( Kenergy * denIncrement );
    VNum += ( Venergy * denIncrement );
    RNum += ( Renergy * denIncrement );
}