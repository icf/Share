//

#ifndef AFQMCLAB_HUBBARDSOC_ICFSD2SSD2SMEASURECOMMUTE_H
#define AFQMCLAB_HUBBARDSOC_ICFSD2SSD2SMEASURECOMMUTE_H

#include "HubbardSOC_icf.h"
#include "afqmclab.h"

class HubbardSOC_icfMeasureCommuteSD2sSD2s
{
 private:
    const HubbardSOC_icf * hubbardSOC_icf;
    std::complex<double> den;
    std::complex<double> HNum, KNum, VNum, RNum;

 public:
    HubbardSOC_icfMeasureCommuteSD2sSD2s();
    HubbardSOC_icfMeasureCommuteSD2sSD2s(const HubbardSOC_icf &hubbardSOC_);
    ~HubbardSOC_icfMeasureCommuteSD2sSD2s();

    const HubbardSOC_icf *getHubbardSOC_icf() const;

    void initModelNullptr();
    void setModel(const HubbardSOC_icf &hubbardSOC_);
    void reSet();
    std::complex<double> returnEnergy();
    void addMeasurement(SD2sSD2sOperation &sd2ssd2sOperation, std::complex<double> denIncrement);
    NiupNidnForce getForce(const NiupNidn &niupNidn, SD2sSD2sOperation &sd2ssd2sOperation, double cap=1.0);

    void write() const;
    void write(std::string postfix) const;
    void writeKNumVumRum() const;
    double getMemory() const;

 private:
    HubbardSOC_icfMeasureCommuteSD2sSD2s(const HubbardSOC_icfMeasureCommuteSD2sSD2s& x);
    HubbardSOC_icfMeasureCommuteSD2sSD2s & operator  = (const HubbardSOC_icfMeasureCommuteSD2sSD2s& x);
    void addEnergy(const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixUp, const tensor_hao::TensorHao<std::complex<double>, 2> &greenMatrixDn, std::complex<double> denIncrement);
};

#endif //AFQMCLAB_HUBBARDSOC_ICFSD2SSD2SMEASURECOMMUTE_H