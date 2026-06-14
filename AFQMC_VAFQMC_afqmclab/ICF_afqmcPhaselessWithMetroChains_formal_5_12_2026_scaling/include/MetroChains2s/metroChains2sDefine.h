//
#ifndef AFQMCLAB_METROCHAINS2SDEFINE_H
#define AFQMCLAB_METROCHAINS2SDEFINE_H

#include "afqmclab.h"
#include "../Metropolis2s/metropolis2s.h"

#include "multDET2s.h"

#include "SD2sSD2sOperation_icf.h"

// SD2s
typedef multDET2s MetroChains2sWalkerRead;
typedef SD2s MetroChains2sWalker;
typedef SD2sSD2sOperation_icf MetroChains2sWalkerWalkerOperation;

// make sure Metropolis and MetropolisInfo recognize the switch between SD <--> SD2s
typedef Metropolis2s MetroChains2sMetropolis;
typedef Metropolis2sInfo  MetroChains2sMetropolisInfo;

#endif //AFQMCLAB_METROCHAINS2SDEFINE_H
