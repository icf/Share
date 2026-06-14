//
// Created by boruoshihao on 1/17/17.
//

#ifndef AFQMCLAB_HUBBARDSOC_ICFSDSD_H
#define AFQMCLAB_HUBBARDSOC_ICFSDSD_H

#include "HubbardSOC_icf.h"
#include "afqmclab.h"

void fillWalkerRandomly(SD &walker, const HubbardSOC_icf &model);
void fillWalkerFromModel(SD &walker, HubbardSOC_icf &model);
void fillWalkerFromModelIncludePinningField(SD &walker, HubbardSOC_icf &model);

#endif //AFQMCLAB_HUBBARDSOC_ICFSDSD_H
