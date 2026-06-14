#ifndef AFQMCLAB_GENERALHAMILTONIAN_ICFSD2SOPERATION_H
#define AFQMCLAB_GENERALHAMILTONIAN_ICFSD2SOPERATION_H

#include "generalHamiltonian_icf.h"
#include "afqmclab.h"

void fillWalkerRandomly(SD2s &walker, const GeneralHamiltonian_icf &model);
void fillWalkerFromModel(SD2s &walker, GeneralHamiltonian_icf &model);

#endif //AFQMCLAB_GENERALHAMILTONIAN_ICFSD2SOPERATION_H
