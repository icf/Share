//
// Created by Hao Shi on 1/13/18.
//

#include <climits>
#include "../../include/oneBodyOperator_icf/logHopSDOperation_icf.h"

using namespace std;
using namespace tensor_hao;

LogHopSDOperation_icf::LogHopSDOperation_icf(std::string flag, size_t taylorOrder, double accuracy, size_t baseTaylorOrder)
{
    reset(flag, taylorOrder, accuracy, baseTaylorOrder);
}

LogHopSDOperation_icf::~LogHopSDOperation_icf() { }

void LogHopSDOperation_icf::reset(string flag, size_t taylorOrder, double accuracy, size_t baseTaylorOrder)
{
    LogHopSDOperation_icf::flag = flag;
    LogHopSDOperation_icf::taylorOrder = taylorOrder;
    LogHopSDOperation_icf::accuracy = accuracy;
    LogHopSDOperation_icf::baseTaylorOrder = baseTaylorOrder;

    operationNumber  = 0;
    minTaylorOrder   = INT_MAX;
    maxTaylorOrder   = 0;
    totalTaylorOrder = 0;

    if( flag != "fixedOrder" && flag != "dynamicOrder" )
    {
        cout<<"Error!!! Do not know the input flag for LogHopSDOperation_icf: "<<flag<<endl;
        exit(1);
    }
}

void LogHopSDOperation_icf::applyToRight(const LogHop &oneBody, const SD &walker, SD &walkerNew)
{
    checkAndResize(oneBody, walker, walkerNew);

    char TRANSOneBody='N';
    addOrders(oneBody, walker, walkerNew, TRANSOneBody);
    walkerNew.logwRef() = oneBody.logw + walker.getLogw();
}

void LogHopSDOperation_icf::applyToLeft(const LogHop &oneBody, const SD &walker, SD &walkerNew)
{
    checkAndResize(oneBody, walker, walkerNew);
    // icf ATTENTION here: 
    char TRANSOneBody='C';
    //
    // char TRANSOneBody='N';
    addOrders(oneBody, walker, walkerNew, TRANSOneBody);
    walkerNew.logwRef() = conj( oneBody.logw ) + walker.getLogw();
    // walkerNew.logwRef() = oneBody.logw + walker.getLogw();
}

void LogHopSDOperation_icf::print()
{
    if( flag == "fixedOrder" )
    {
        if( MPIRank() == 0 )
        {
            if( taylorOrder == 0 ) cout<<"Use fixed order, the order is not determined yet."<<endl;
            else cout<<"Use fixedOrder, the order is "<<taylorOrder<<endl;
        }
    }
    else if( flag == "dynamicOrder" )
    {
        size_t globalNumber, globalMin, globalMax, globalTotal;
        globalNumber = MPISum( operationNumber );
#ifdef MPI_HAO
        MPIReduce(minTaylorOrder, globalMin, MPI_MIN);
        MPIReduce(maxTaylorOrder, globalMax, MPI_MAX);
#else
        globalMin = minTaylorOrder;
        globalMax = maxTaylorOrder;
#endif
        globalTotal  = MPISum( totalTaylorOrder );

        if( MPIRank() == 0 )
        {
            cout<<"Use dynamicOrder. "<<endl;
            cout<<"The base order is "<<baseTaylorOrder<<endl;
            cout<<"The min order is "<<globalMin<<endl;
            cout<<"The max order is "<<globalMax<<endl;
            cout<<"The average order is "<< globalTotal*1.0/globalNumber <<endl;
        }
    }
}

size_t LogHopSDOperation_icf::getCurrentOrder() const { return currentOrder; }

void LogHopSDOperation_icf::checkAndResize(const LogHop &oneBody, const SD &walker, SD &walkerNew) const
{
    size_t L = walker.getL(); size_t N = walker.getN();
    if( oneBody.getL() !=  L ) { cout<<"Error!!! LogHop size is not consistent with walker!"<<endl; exit(1); }
    if( walkerNew.getL() != L  ||  walkerNew.getN() != N ) walkerNew.resize( L, N );
}

void LogHopSDOperation_icf::addOrders(const LogHop &oneBody, const SD &walker, SD &walkerNew, char TRANSOneBody)
{
    if( flag == "fixedOrder" )
    {
        if( taylorOrder==0 ) determinantAndAddFixedOrders(oneBody, walker, walkerNew, TRANSOneBody);
        else initialAndAddFixedOrders(oneBody, walker, walkerNew, taylorOrder, TRANSOneBody);
        operationNumber++;
    }
    else if( flag == "dynamicOrder" )
    {
        initialAndAddDynamicOrders(oneBody, walker, walkerNew, TRANSOneBody);
        minTaylorOrder = min( minTaylorOrder, currentOrder );
        maxTaylorOrder = max( maxTaylorOrder, currentOrder );
        totalTaylorOrder += currentOrder;
        operationNumber++;
    }
    else
    {
        cout<<"Error!!! Do not know the input flag for LogHopSDOperation_icf: "<<flag<<endl;
        exit(1);
    }
    clearWfTemp();
}

void LogHopSDOperation_icf::determinantAndAddFixedOrders(const LogHop &oneBody, const SD &walker, SD &walkerNew, char TRANSOneBody)
{
    initialAndAddFixedOrders(oneBody, walker, walkerNew, baseTaylorOrder, TRANSOneBody);
    addDynamicOrders(oneBody, walker, walkerNew, TRANSOneBody);

#ifdef MPI_HAO
    MPIReduce(currentOrder, taylorOrder, MPI_MAX);
#else
    taylorOrder = currentOrder;
#endif
}

void LogHopSDOperation_icf::initialAndAddFixedOrders(const LogHop &oneBody, const SD &walker, SD &walkerNew, size_t maxOrder, char TRANSOneBody)
{
    initialZeroOrder(oneBody, walker, walkerNew, TRANSOneBody);
    addFixedOrders(oneBody, walker, walkerNew, maxOrder, TRANSOneBody);
}

void LogHopSDOperation_icf::initialAndAddDynamicOrders(const LogHop &oneBody, const SD &walker, SD &walkerNew, char TRANSOneBody)
{
    initialAndAddFixedOrders(oneBody, walker, walkerNew, baseTaylorOrder, TRANSOneBody);
    addDynamicOrders(oneBody, walker, walkerNew, TRANSOneBody);
}

void LogHopSDOperation_icf::initialZeroOrder(const LogHop &oneBody, const SD &walker, SD &walkerNew, char TRANSOneBody)
{
    size_t L = walker.getL(); size_t N = walker.getN();
    if( wfTempNew.rank(0) != L || wfTempNew.rank(1) != N ) wfTempNew.resize( L, N );
    if( wfTempOld.rank(0) != L || wfTempOld.rank(1) != N ) wfTempOld.resize( L, N );

    walkerNew.wfRef() = walker.getWf();
    wfTempOld = walker.getWf();

    currentOrder = 0;
}

void LogHopSDOperation_icf::addFixedOrders(const LogHop &oneBody, const SD &walker, SD &walkerNew, size_t maxOrder, char TRANSOneBody)
{
    while( currentOrder < maxOrder )
    {
        BL_NAME(gmm)( oneBody.matrix, wfTempOld, wfTempNew, TRANSOneBody, 'N', 1.0/(currentOrder+1.0) );
        walkerNew.wfRef() += wfTempNew;
        wfTempOld = move( wfTempNew );

        currentOrder++;
    }
}

void LogHopSDOperation_icf::addDynamicOrders(const LogHop &oneBody, const SD &walker, SD &walkerNew, char TRANSOneBody)
{
    do
    {
        BL_NAME(gmm)( oneBody.matrix, wfTempOld, wfTempNew, TRANSOneBody, 'N', 1.0/(currentOrder+1.0) );
        walkerNew.wfRef() += wfTempNew;
        wfTempOld = move( wfTempNew );

        currentOrder++;
    } while( ! isConverged() );
}

bool LogHopSDOperation_icf::isConverged()
{
    size_t L = wfTempOld.rank(0); size_t N = wfTempOld.rank(1);
    for(size_t i = 0; i < N; ++i)
    {
        for(size_t j = 0; j < L; ++j)
        {
            if( abs( wfTempOld(j,i) ) > accuracy  ) return false;
        }
    }

    return true;
}

void LogHopSDOperation_icf::clearWfTemp()
{
    wfTempOld.resize(0,0);
    wfTempNew.resize(0,0);
}

LogHopSDOperation_icf::LogHopSDOperation_icf(const LogHopSDOperation_icf &x) { }

LogHopSDOperation_icf &LogHopSDOperation_icf::operator=(const LogHopSDOperation_icf &x) { return *this; }