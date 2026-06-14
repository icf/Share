//
// Created by Zhi-Yu
//

#ifndef AFQMCLAB_LOGHOP2SSD2SOPERATION_H
#define AFQMCLAB_LOGHOP2SSD2SOPERATION_H

#include "afqmclab.h"
#include "logHop2s.h"

// Three different type of operation.
// flag: fixedOrder, taylorOrder>0. Always use the input taylorOrder order.
// flag: fixedOrder, taylorOrder=0. Determine the taylor order from accuracy and baseTaylorOrder and always use it.
// flag: dynamicOrder. Determine the taylor order from accuracy and baseTaylorOrder in all the operations.

class LogHop2sSD2sOperation
{
 private:
    std::string flag; //"fixedOrder", "dynamicOrder"
    size_t taylorOrder;
    double accuracy;
    size_t baseTaylorOrder;

    size_t operationNumber;
    size_t minTaylorOrder, maxTaylorOrder, totalTaylorOrder;

    size_t currentOrder;
    tensor_hao::TensorHao<std::complex<double>,2> wfUpTempOld, wfDnTempOld, wfUpTempNew, wfDnTempNew;

 public:
    LogHop2sSD2sOperation(std::string flag="dynamicOrder", size_t taylorOrder=0, double accuracy=1e-10, size_t baseTaylorOrder=3);
    ~LogHop2sSD2sOperation();

    void reset(std::string flag="dynamicOrder", size_t taylorOrder=0, double accuracy=1e-10, size_t baseTaylorOrder=3);
    void applyToRight(const LogHop2s &oneBody, const SD2s &walker, SD2s &walkerNew);
    void applyToLeft(const LogHop2s &oneBody, const SD2s &walker, SD2s &walkerNew);
    void print();
    size_t getCurrentOrder() const;

 private:
    void checkAndResize(const LogHop2s &oneBody, const SD2s &walker, SD2s &walkerNew) const;

    void addOrders(const LogHop2s &oneBody, const SD2s &walker, SD2s &walkerNew, char TRANSOneBody);
    void determinantAndAddFixedOrders(const LogHop2s &oneBody, const SD2s &walker, SD2s &walkerNew, char TRANSOneBody);
    void initialAndAddFixedOrders(const LogHop2s &oneBody, const SD2s &walker, SD2s &walkerNew, size_t maxOrder, char TRANSOneBody);
    void initialAndAddDynamicOrders(const LogHop2s &oneBody, const SD2s &walker, SD2s &walkerNew, char TRANSOneBody);

    void initialZeroOrder(const LogHop2s &oneBody, const SD2s &walker, SD2s &walkerNew, char TRANSOneBody);
    void addFixedOrders(const LogHop2s &oneBody, const SD2s &walker, SD2s &walkerNew, size_t maxOrder, char TRANSOneBody);
    void addDynamicOrders(const LogHop2s &oneBody, const SD2s &walker, SD2s &walkerNew, char TRANSOneBody);
    bool isConverged();
    void clearWfTemp();

    LogHop2sSD2sOperation(const LogHop2sSD2sOperation& x);
    LogHop2sSD2sOperation & operator  = (const LogHop2sSD2sOperation& x);
};


#endif //AFQMCLAB_LOGHOP2SSD2SOPERATION_H