#ifndef AFQMCLAB_GENERALHAMILTONIAN_ICFSDOPERATION_H
#define AFQMCLAB_GENERALHAMILTONIAN_ICFSDOPERATION_H

#include "generalHamiltonian_icf.h"
#include "afqmclab.h"

void fillWalkerRandomly(SD &walker, const GeneralHamiltonian_icf &model);
void fillWalkerFromModel(SD &walker, GeneralHamiltonian_icf &model);

#endif //AFQMCLAB_GENERALHAMILTONIAN_ICFSDOPERATION_H
