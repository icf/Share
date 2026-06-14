#include <iostream>
#include <iomanip>
#include <vector>
#include "afqmclab.h"

using namespace std;

void popCheck_icf(const std::vector<double> &weight)
{
    int L=weight.size();
    double avgW=0.0; for(int i=0; i<L; i++) {avgW+=weight[i];} avgW /= (L*1.0);

    double maxW(0.0), minW(1e20);
    for(int i = 0;  i< L; i++)
    {
        if( maxW < weight[i] ) maxW = weight[i];
        if( minW > weight[i] ) minW = weight[i];
    }

    cout<<"In Population control: "<<endl;
    cout<<"Min weight is "<< minW <<endl;
    cout<<"Max weight is "<< maxW <<endl;
    cout<<"Average weight is "<< avgW <<endl;
    if( avgW<1e-12 ) cout<<"Warning!!!!! Average weight is too small!"<<endl;
}

void capWeight_icf(std::vector<double> &weight, double cap)
{
    size_t L=weight.size();
    double totalWeight(0); for(size_t i = 0; i < L; ++i) totalWeight+=weight[i];
    double weightCap=totalWeight*cap;

    size_t LNormal(0); double totalWeightNormal(0.0);
    for(size_t i = 0; i < L; ++i)
    {
        if( weight[i] < weightCap )
        {
            totalWeightNormal += weight[i];
            LNormal += 1;
        }
    }

    totalWeightNormal *= ( L*1.0/LNormal );
    weightCap = totalWeightNormal*cap;
    for(size_t i = 0; i < L; ++i)
    {
        if( weight[i]> weightCap )
        {
            cout<<"Cap the large weight "<<setw(5)<<i<<setw(26)<<weight[i]<<setw(26)<<weightCap<<endl;
            weight[i] = weightCap;
        }
    }
}

vector<int> popConfiguration_icf(int size, const vector<double>& weight)
{
    //Set size
    int L=weight.size(); int L_chunk=L/size;
    if( (L_chunk*size) !=  L ) {cout<<"ERROR!!! Size of weight can not be divided by Size (suppose to be MPISize) "<<endl; exit(1);}

    //Probability function: normalized weight
    double sum=0.0; for(int i=0; i<L; i++) sum+=weight[i];
    vector<double> prob(L); for(int i=0; i<L; i++) prob[i]=weight[i]/sum;

    //Set num transfer
    vector<int> number(L,0);
    double prob_sum_new;
    int index_old=0; double prob_sum_old=prob[0];
    for(int i=0; i<L; i++)
    {
        prob_sum_new=( i+uniformHao() ) / (L*1.0);
        while( prob_sum_new>prob_sum_old )
        {
            index_old++;
            prob_sum_old+=prob[index_old]; 
        }
        number[index_old]++;
        //We used to only use following formula
        //table[i]=index_old; 
        //Currently change to vector<int> number to get high efficiency
    }

    //Initial table
    vector<int> table(L);
    for(int i=0; i<L; i++) { if(number[i] !=0 ) table[i]=i; }

    int zero_i, one_i, max_i;

    //Exchange between one thread
    for(int i=0; i<size; i++)
    {
        zero_i=L_chunk*i; one_i=L_chunk*i; max_i=L_chunk*(i+1);
        while(number[zero_i] !=0 && zero_i<max_i ) zero_i++; //number[zero_i]=0 or zero_i is out of range
        while(number[one_i]  <=1 && one_i<max_i  ) one_i++;  //number[one_i]>1  or one_i  is out of range

        while(zero_i<max_i && one_i<max_i)
        {
            table[zero_i]=one_i; 
            number[zero_i]++;
            number[one_i]--;
 
            while(number[zero_i] !=0 && zero_i<max_i ) zero_i++;
            while(number[one_i]  <=1 && one_i<max_i  ) one_i++;
        }
    }

    //Exchange between threads
    if(size>1)
    {
        zero_i=0; one_i=0; max_i=L;
        while(number[zero_i] !=0 && zero_i<max_i ) zero_i++; //number[zero_i]=0 or zero_i is out of range
        while(number[one_i]  <=1 && one_i<max_i  ) one_i++;  //number[one_i]>1  or one_i  is out of range

        while(zero_i<max_i && one_i<max_i )
        {
            table[zero_i]=one_i; 
            number[zero_i]++;
            number[one_i]--; 

            while(number[zero_i] !=0 && zero_i<max_i ) zero_i++;
            while(number[one_i]  <=1 && one_i<max_i  ) one_i++;
        }
    }

    //Check
    for(int i=0; i<L; i++)
    {
        if(number[i]!=1) {cout<<"ERROR!!!! After pop configuration, number of each walker is not 1!"<<endl; exit(1);}
    }

    return table;
}
