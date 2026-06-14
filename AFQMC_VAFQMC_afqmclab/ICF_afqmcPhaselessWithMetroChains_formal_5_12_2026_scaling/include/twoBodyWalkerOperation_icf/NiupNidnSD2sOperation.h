//
// Created by Zhi-Yu
//

#ifndef AFQMCLAB_NIUPNIDNSD2SOPERATION_H
#define AFQMCLAB_NIUPNIDNSD2SOPERATION_H

#include <tuple>
#include "afqmclab.h"

class NiupNidnSampleSD2sOperation
{
 public:
    NiupNidnSampleSD2sOperation();
    ~NiupNidnSampleSD2sOperation();

    void applyToRight(const NiupNidnSample &oneBody, const SD2s &walker, SD2s &walkerNew) const;
    void applyToLeft(const NiupNidnSample &oneBody, const SD2s &walker, SD2s &walkerNew) const;
};
//
#endif //AFQMCLAB_NIUPNIDNSD2SOPERATION_H
