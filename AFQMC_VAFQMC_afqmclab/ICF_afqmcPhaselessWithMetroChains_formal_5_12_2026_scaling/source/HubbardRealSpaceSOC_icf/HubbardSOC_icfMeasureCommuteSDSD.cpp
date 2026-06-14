//
// Created by boruoshihao on 1/13/17.
//

#include "../../include/HubbardRealSpaceSOC_icf/HubbardSOC_icfMeasureCommuteSDSD.h"
#include "afqmclab.h"

using namespace std;
using namespace tensor_hao;

HubbardSOC_icfMeasureCommuteSDSD::HubbardSOC_icfMeasureCommuteSDSD()
{
    initModelNullptr();
    reSet();
}

HubbardSOC_icfMeasureCommuteSDSD::HubbardSOC_icfMeasureCommuteSDSD(const HubbardSOC_icf &hubbardSOC_)
{
    setModel(hubbardSOC_);
    reSet();
}

HubbardSOC_icfMeasureCommuteSDSD::~HubbardSOC_icfMeasureCommuteSDSD()
{

}

const HubbardSOC_icf *HubbardSOC_icfMeasureCommuteSDSD::getHubbardSOC_icf() const
{
    return hubbardSOC_icf;
}

void HubbardSOC_icfMeasureCommuteSDSD::initModelNullptr()
{
    hubbardSOC_icf = nullptr;
}

void HubbardSOC_icfMeasureCommuteSDSD::setModel(const HubbardSOC_icf &hubbardSOC_)
{
    hubbardSOC_icf = &hubbardSOC_;
}

void HubbardSOC_icfMeasureCommuteSDSD::reSet()
{
    complex<double> zero(0,0);
    den = zero;
    HNum = zero;
    KNum = zero;
    VNum = zero;
    RNum = zero;
}

complex<double> HubbardSOC_icfMeasureCommuteSDSD::returnEnergy()
{
    complex<double> Htot   = MPISum(HNum);
    complex<double> denTot = MPISum(den);
    complex<double> energy;
    if( MPIRank() == 0 ) energy = Htot/denTot;
    MPIBcast(energy);
    return energy;
}

void HubbardSOC_icfMeasureCommuteSDSD::addMeasurement(SDSDOperation &sdsdOperation, complex<double> denIncrement)
{
    den += denIncrement;

    const TensorHao< complex<double>, 2 > &greenMatrix = sdsdOperation.returnGreenMatrix();

    addEnergy(greenMatrix, denIncrement);
}

NiupNidnForce HubbardSOC_icfMeasureCommuteSDSD::getForce(const NiupNidn &niupNidn, SDSDOperation &sdsdOperation, double cap)
{
    size_t halfL = niupNidn.getL(); const string &decompType = niupNidn.getDecompType();

    TensorHao< complex<double>, 1 > backGround(halfL);
    if( decompType == "densityCharge" )
    {
        const TensorHao< complex<double>, 1 > &greenDiagonal = sdsdOperation.returnGreenDiagonal();
        for(size_t i = 0; i < halfL; ++i) backGround(i) = greenDiagonal(i) + greenDiagonal(i+halfL) -1.0;
    }
    else if( decompType == "densitySpin" )
    {
        const TensorHao< complex<double>, 1 > &greenDiagonal = sdsdOperation.returnGreenDiagonal();
        for(size_t i = 0; i < halfL; ++i) backGround(i) = greenDiagonal(i) - greenDiagonal(i+halfL);
    }
    else if( decompType == "hopCharge" )
    {
        const TensorHao< complex<double>, 1 > &greenOffDiagonal = sdsdOperation.returnGreenOffDiagonal();
        for(size_t i = 0; i < halfL; ++i) backGround(i) = greenOffDiagonal(i) + greenOffDiagonal(i+halfL);
    }
    else if( decompType == "hopSpin" )
    {
        const TensorHao< complex<double>, 1 > &greenOffDiagonal = sdsdOperation.returnGreenOffDiagonal();
        for(size_t i = 0; i < halfL; ++i) backGround(i) = greenOffDiagonal(i) - greenOffDiagonal(i+halfL);
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

void HubbardSOC_icfMeasureCommuteSDSD::write() const
{
    writeThreadSum(den, "den.dat", ios::app);
    writeThreadSum(HNum, "HNum.dat", ios::app);
}

void HubbardSOC_icfMeasureCommuteSDSD::write(std::string postfix) const
{
    std::string denName="den"+postfix+".dat";
    std::string HNumName="HNum"+postfix+".dat";
    writeThreadSum(den, denName, ios::app);
    writeThreadSum(HNum, HNumName, ios::app);
}

void HubbardSOC_icfMeasureCommuteSDSD::writeKNumVumRum() const
{
    writeThreadSum(KNum, "KNum.dat", ios::app);
    writeThreadSum(VNum, "VNum.dat", ios::app);
    writeThreadSum(RNum, "RNum.dat", ios::app);
}

double HubbardSOC_icfMeasureCommuteSDSD::getMemory() const
{
    return 8.0+16.0*5;
}

HubbardSOC_icfMeasureCommuteSDSD::HubbardSOC_icfMeasureCommuteSDSD(const HubbardSOC_icfMeasureCommuteSDSD &x)
{

}

HubbardSOC_icfMeasureCommuteSDSD &HubbardSOC_icfMeasureCommuteSDSD::operator=(const HubbardSOC_icfMeasureCommuteSDSD &x)
{
    return *this;
}

void HubbardSOC_icfMeasureCommuteSDSD::addEnergy(const TensorHao<complex<double>, 2> &greenMatrix, complex<double> denIncrement)
{
    complex<double> Kenergy(0,0), Venergy(0,0), Renergy(0,0);

    size_t L  = hubbardSOC_icf->getL(); size_t L2 = L*2;

    //Add K
    const TensorHao< complex<double>, 2 > &K = hubbardSOC_icf->getK();
    for(size_t i = 0; i < L2; ++i)
    {
        for(size_t j = 0; j < L2; ++j)
        {
            Kenergy += K(j,i) * greenMatrix(j,i);
        }
    }

    //Add U
    const TensorHao< double, 1> &U = hubbardSOC_icf->getU();
    for(size_t i = 0; i < L; ++i)
    {
        Venergy += U(i) * ( greenMatrix(i,i)*greenMatrix(i+L,i+L) - greenMatrix(i, i+L)*greenMatrix(i+L, i) );
    }

    //Add mu and pinning field
    const TensorHao< double, 1> &mu = hubbardSOC_icf->getMu();
    const TensorHao< double, 1> &hx = hubbardSOC_icf->getHx();
    const TensorHao< double, 1> &hy = hubbardSOC_icf->getHy();
    const TensorHao< double, 1> &hz = hubbardSOC_icf->getHz();
    for(size_t i = 0; i < L; ++i)
    {
        Renergy += ( -mu(i) + hz(i)*0.5 ) * greenMatrix(i,i);
        Renergy += ( -mu(i) - hz(i)*0.5 ) * greenMatrix(i+L,i+L);
        Renergy += complex<double>( hx(i)*0.5, -hy(i)*0.5 ) * greenMatrix(i, i+L);
        Renergy += complex<double>( hx(i)*0.5,  hy(i)*0.5 ) * greenMatrix(i+L, i);
    }

    HNum += ( ( Kenergy + Venergy + Renergy ) * denIncrement );
    KNum += ( Kenergy * denIncrement );
    VNum += ( Venergy * denIncrement );
    RNum += ( Renergy * denIncrement );
}