//
#ifndef AFQMCLAB_METROCHAINSDEFINE_H
#define AFQMCLAB_METROCHAINSDEFINE_H

#include "afqmclab.h"
#include "../Metropolis/metropolis.h"

#include "multDET.h"

#include "SDSDOperation_icf.h"

// SD
typedef multDET MetroChainsWalkerRead;
typedef SD MetroChainsWalker;
typedef SDSDOperation_icf MetroChainsWalkerWalkerOperation;

typedef Metropolis MetroChainsMetropolis;
typedef MetropolisInfo  MetroChainsMetropolisInfo;

#endif //AFQMCLAB_METROCHAINSDEFINE_H
