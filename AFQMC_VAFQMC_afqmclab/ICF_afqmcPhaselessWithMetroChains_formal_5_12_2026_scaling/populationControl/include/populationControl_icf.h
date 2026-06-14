#ifndef AFQMCLAB_POPULATIONCONTROL_ICF_H
#define AFQMCLAB_POPULATIONCONTROL_ICF_H

#include <iostream>
#include <vector>
#include <tuple>
#include <numeric>
#include "afqmclab.h"

using namespace std;

//size is MPISize(), only call pop_configuration for one thread
void popCheck_icf(const std::vector<double>& weight);
void capWeight_icf(std::vector<double>& weight, double cap=0.2);
std::vector<int> popConfiguration_icf(int size, const std::vector<double>& weight);

#ifdef MPI_HAO
void deterSendNumber_icf(int& send_number, std::vector<int>& send_number_list, const std::vector<int>& table);
void deterTableSend_icf(int send_number, const std::vector<int>& send_number_list, const std::vector<int>& table, std::vector<int>& table_send);
int deterBufSendNumber_icf(int L_chunk, int send_number, const std::vector<int>& table_send);
int deterBufRecvNumber_icf(const std::vector<int>& table_recv);
#endif

template<class T >
void populationControl_icf(std::vector<T> &walker, const std::vector<int> &table)
{
    int rank=MPIRank(); int size=MPISize(); int L_chunk=walker.size();
    int L=0; if(rank==0) L=table.size(); MPIBcast(L);
    if( (L_chunk*size) !=  L ) {std::cout<<"ERROR!!! Size of table can not be divided by MPISize."<<std::endl; exit(1);}
    if( L_chunk==0 ) return;


#ifdef MPI_HAO
    //Determine table_recv
    std::vector<int> table_recv(L_chunk);
    MPI_Scatter(table.data(), L_chunk, MPI_INT, table_recv.data(), L_chunk, MPI_INT, 0, MPI_COMM_WORLD);

    // CHECKPOINT 1: After MPI_Scatter
    MPIBarrier();
    if(rank == 0) {
        cout<<"!!! CHECKPOINT 1: After MPI_Scatter"<<endl;
        cout<<"!!! DEBUG: L="<<L<<", size="<<size<<", L_chunk="<<L_chunk<<endl;
        cout<<"!!! DEBUG: table: ";
        for(int i=0; i<L; i++) cout<<table[i]<<" ";
        cout<<endl;
    }
    MPIBarrier();

    // CHECKPOINT 2: Before deterSendNumber_icf
    MPIBarrier();
    if(rank == 0) cout<<"!!! CHECKPOINT 2: Before deterSendNumber_icf"<<endl;
    MPIBarrier();

    //Determine the number each thread will send out, send_number_list is the list only in main thread
    int send_number;
    std::vector<int> send_number_list;
    deterSendNumber_icf(send_number, send_number_list, table);

    // CHECKPOINT 3: After deterSendNumber_icf
    MPIBarrier();
    if(rank == 0) cout<<"!!! CHECKPOINT 3: After deterSendNumber_icf, send_number="<<send_number<<endl;
    MPIBarrier();

    //Determine_table_send
    std::vector<int> table_send; //Parent is 0~send_number, child is send_number~2*send_number
    deterTableSend_icf(send_number, send_number_list, table, table_send);

    // CHECKPOINT 4: After deterTableSend_icf
    MPIBarrier();
    if(rank == 0) cout<<"!!! CHECKPOINT 4: After deterTableSend_icf"<<endl;
    MPIBarrier();

    //Determine buf_send_number, and buf_recv_number for each thread.
    //Not all send_number are sent, use the property that receive thread and send thread are ascending
    int buf_send_number=deterBufSendNumber_icf(L_chunk, send_number, table_send);
    int buf_recv_number=deterBufRecvNumber_icf(table_recv);

    // CHECKPOINT 5: After buf calculations
    MPIBarrier();
    if(rank == 0) cout<<"!!! CHECKPOINT 5: After buf calculations"<<endl;
    MPIBarrier();

    //Some parameters for send and recv
    int count, send_index, recv_rank, send_index_bk, recv_rank_bk, send_rank, send_local_index, recv_local_index;
    int Nbuf=(walker[0]).getNbuf();
    
    // CHECKPOINT 6: After buffer allocation - with timeout detection
    MPIBarrier();
    if(rank==0) {
        cout<<"!!! CHECKPOINT 6: After buffer allocation"<<endl;
        cout<<"!!! DEBUG: buf_send_number="<<buf_send_number<<", buf_recv_number="<<buf_recv_number<<", Nbuf="<<Nbuf<<endl;
    }
    MPIBarrier();
    
    std::vector< std::vector<char>  >  buf_send(buf_send_number);
    MPIBarrier(); 
    if(rank==0) cout<<"!!! DEBUG popControl: After buf_send allocation"<<endl;
    MPIBarrier();
    
    std::vector<char> buf_recv_one(Nbuf);
    MPIBarrier();
    if(rank==0) cout<<"!!! DEBUG popControl: After buf_recv_one allocation, Nbuf="<<Nbuf<<endl;
    MPIBarrier();
    
    std::vector< MPI_Request > ISREQ(buf_send_number); 
    std::vector< MPI_Status  > ISSTA(buf_send_number);
    MPI_Status IRSTA;

    // CHECKPOINT 6.5: Before send loop - with timeout detection
    MPIBarrier();
    if(rank==0) {
        cout<<"!!! CHECKPOINT 6.5: After buffer allocation"<<endl;
    }
    MPIBarrier();
    //Start to send
    if(send_number == 1){
        count=0; 
        int i=0;
        send_index = table_send[i];
        recv_rank  = table_send[i+send_number]/L_chunk;
        send_local_index= send_index- rank*L_chunk;
        recv_local_index= table_send[i+send_number]-recv_rank*L_chunk;
        // 
        buf_send[count] = (walker[send_local_index]).pack();
        /////////////////////////////////////
        // Check buffer size before MPI_Isend
        size_t actual_buf_size = (buf_send[count]).size();
        if(actual_buf_size != Nbuf) {
            cout << "!!! ERROR rank " << rank << ": actual_buf_size=" << actual_buf_size << " != Nbuf=" << Nbuf << endl;
            cout << "!!! ERROR: send_index=" << send_index << ", recv_rank=" << recv_rank << ", send_local_index=" << send_local_index << endl;
        }
        /////////////////////////////////////
    }
    // CHECKPOINT 6.8: After pack - with timeout detection
    {
        MPI_Request barrier_req;
        MPI_Ibarrier(MPI_COMM_WORLD, &barrier_req);
        double start_time = MPI_Wtime();
        int flag = 0;
        bool printed_stuck = false;
        while(!flag) {
            double elapsed = MPI_Wtime() - start_time;
            if(elapsed > 600.0 && !printed_stuck) {
                cout<<"!!! STUCK rank "<<rank<<" at CHECKPOINT 6.8 after "<<elapsed<<"s: ";
                cout<<"count="<<count<<", send_number="<<send_number<<", buf_send_number="<<buf_send_number<<", table_recv[0]="<<table_recv[0]<<endl;
                printed_stuck = true;
            }
            MPI_Test(&barrier_req, &flag, MPI_STATUS_IGNORE);
#ifdef _WIN32
            Sleep(100);
#else
            usleep(100000);
#endif
        }
    }
    MPIBarrier();
    if(rank==0) cout<<"!!! CHECKPOINT 6.8: After pack loop"<<endl;
    MPIBarrier();

    ///////////////////////////////////////////
    // 准备当前 rank 的数据
    int my_data[6];
    my_data[0] = rank;
    my_data[1] = send_number;
    my_data[2] = buf_send_number;
    my_data[3] = Nbuf;
    if(send_number > 0){
        my_data[4] = recv_rank;
        my_data[5] = recv_local_index;
    } else {
        my_data[4] = -1; // No valid recv_rank
        my_data[5] = -1; // No valid recv_local_index
    }
    int my_bufsize;
    if(send_number > 0){
        my_bufsize = buf_send[0].size();
    } else {
        my_bufsize = -1;
    }
    // 收集所有 rank 的数据到 rank 0
    const int datasize = 7;  // 6个int + 1个bufsize
    std::vector<int> all_data(datasize * size);
    MPI_Gather(my_data, 6, MPI_INT, all_data.data(), 6, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 收集 bufsize 到 rank 0
    std::vector<int> all_bufsize(size);
    MPI_Gather(&my_bufsize, 1, MPI_INT, all_bufsize.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // rank 0 写入文件
    if(rank == 0){
        std::ofstream logfile("log_MPI.dat", std::ios::trunc);
        for(int r = 0; r < size; r++){
            int idx = r * 6;
            logfile << "rank=" << all_data[idx]
                    << " send_number=" << all_data[idx+1]
                    << " buf_send_number=" << all_data[idx+2]
                    << " Nbuf=" << all_data[idx+3]
                    << " recv_rank=" << all_data[idx+4]
                    << " recv_local_index=" << all_data[idx+5]
                    << " buf_send[0].size()=" << all_bufsize[r]
                    << std::endl;
        }
        logfile.close();
    }
    MPIBarrier();
    ////////////////////////////////////////////
    if(send_number == 1){
        count=0; 
        MPI_Isend( (buf_send[count]).data(), Nbuf, MPI_BYTE, recv_rank, recv_local_index, MPI_COMM_WORLD, ISREQ.data()+count);
    }
    // CHECKPOINT 6.9: After send loop - with timeout detection
    {
        MPI_Request barrier_req;
        MPI_Ibarrier(MPI_COMM_WORLD, &barrier_req);
        double start_time = MPI_Wtime();
        int flag = 0;
        bool printed_stuck = false;
        while(!flag) {
            double elapsed = MPI_Wtime() - start_time;
            if(elapsed > 600.0 && !printed_stuck) {
                cout<<"!!! STUCK rank "<<rank<<" at CHECKPOINT 6.9 after "<<elapsed<<"s: ";
                cout<<"count="<<count<<", send_number="<<send_number<<", buf_send_number="<<buf_send_number<<", table_recv[0]="<<table_recv[0]<<endl;
                printed_stuck = true;
            }
            MPI_Test(&barrier_req, &flag, MPI_STATUS_IGNORE);
#ifdef _WIN32
            Sleep(100);
#else
            usleep(100000);
#endif
        }
    }
    MPIBarrier();
    if(rank==0) cout<<"!!! CHECKPOINT 6.9: After send_number =1 send loop, sent "<<count<<" walkers"<<endl;
    MPIBarrier();
    if(send_number > 1){
        count=0; send_index_bk=-1, recv_rank_bk=-1;
        for(int i=0; i<send_number; i++)
        {
            send_index = table_send[i];
            recv_rank  = table_send[i+send_number]/L_chunk;
            if(send_index!=send_index_bk || recv_rank!=recv_rank_bk )
            {
                send_local_index= send_index- rank*L_chunk;
                recv_local_index= table_send[i+send_number]-recv_rank*L_chunk;

                buf_send[count] = (walker[send_local_index]).pack();
                /////////////////////////////////////
                // Check buffer size before MPI_Isend
                size_t actual_buf_size = (buf_send[count]).size();
                if(actual_buf_size != Nbuf) {
                    cout << "!!! ERROR rank " << rank << ": actual_buf_size=" << actual_buf_size << " != Nbuf=" << Nbuf << endl;
                    cout << "!!! ERROR: send_index=" << send_index << ", recv_rank=" << recv_rank << ", send_local_index=" << send_local_index << endl;
                }
                MPI_Isend( (buf_send[count]).data(), Nbuf, MPI_BYTE, recv_rank, recv_local_index, MPI_COMM_WORLD, ISREQ.data()+count);

                count++;
                send_index_bk   = send_index;
                recv_rank_bk    = recv_rank;
            }
        }
    }
    // CHECKPOINT 7: After send loop - with timeout detection
    {
        MPI_Request barrier_req;
        MPI_Ibarrier(MPI_COMM_WORLD, &barrier_req);
        double start_time = MPI_Wtime();
        int flag = 0;
        bool printed_stuck = false;
        while(!flag) {
            double elapsed = MPI_Wtime() - start_time;
            if(elapsed > 600.0 && !printed_stuck) {
                cout<<"!!! STUCK rank "<<rank<<" at CHECKPOINT 7 after "<<elapsed<<"s: ";
                cout<<"count="<<count<<", send_number="<<send_number<<", buf_send_number="<<buf_send_number<<", table_recv[0]="<<table_recv[0]<<endl;
                printed_stuck = true;
            }
            MPI_Test(&barrier_req, &flag, MPI_STATUS_IGNORE);
#ifdef _WIN32
            Sleep(100);
#else
            usleep(100000);
#endif
        }
    }
    MPIBarrier();
    if(rank==0) cout<<"!!! CHECKPOINT 7: After send loop, sent "<<count<<" walkers"<<endl;
    MPIBarrier();

    //Start to receive
    count=0; send_index_bk=-1; recv_local_index=-1;
    for(int i=0; i<L_chunk; i++)
    {
        send_index = table_recv[i];
        send_rank  = send_index/L_chunk;

        if(send_rank == rank) //receive from same rank
        {
            send_local_index=send_index-send_rank*L_chunk; 
            if(i!=send_local_index) walker[i]=walker[send_local_index];
        }
        else //receive from different rank
        {
            if(send_index == send_index_bk)   walker[i]=walker[recv_local_index]; //Use the last received index  
            else  //receive and unpack, make backup
            {
                MPI_Recv(buf_recv_one.data(), Nbuf, MPI_BYTE, send_rank, i, MPI_COMM_WORLD, &IRSTA);
                /////////////////////////////////////
                // icf Test
                /////////////////////////////////////
                // int bufSize=buf_recv_one.size();
                // if(bufSize!=Nbuf) {cout<<"ERROR in unpack!!! Size of input buf does not equal Nbuf! "<<"  bufSize: "<<bufSize<<"  Nbuf: "<<Nbuf<<endl; exit(1);}

                // int posit=0;
                // bool isInitialMetroChains;
                // int numOfChains;
                // int numOfBrackets;
                // MPI_Unpack(buf_recv_one.data(), buf_recv_one.size(), &posit, &isInitialMetroChains, 1, MPI_CXX_BOOL, MPI_COMM_WORLD);
                // MPI_Unpack(buf_recv_one.data(), buf_recv_one.size(), &posit, &numOfChains, 1, MPI_INT, MPI_COMM_WORLD);
                // MPI_Unpack(buf_recv_one.data(), buf_recv_one.size(), &posit, &numOfBrackets, 1, MPI_INT, MPI_COMM_WORLD);
                // // 
                // std::complex<double> logw;
                // tensor_hao::TensorHao<std::complex<double>,2> wfUp, wfDn;
                // wfUp.resize(40, 20);
                // wfDn.resize(40, 20);
                // MPI_Unpack(buf_recv_one.data(), buf_recv_one.size(), &posit, &logw, 1, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
                // MPI_Unpack(buf_recv_one.data(), buf_recv_one.size(), &posit, wfUp.data(), wfUp.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
                // MPI_Unpack(buf_recv_one.data(), buf_recv_one.size(), &posit, wfDn.data(), wfDn.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
                // complex<double> tempSumUp=0.0;
                // complex<double> tempSumDn=0.0;
                // for(int i_temp=1-1; i_temp<=40-1; i_temp++){
                // for(int j_temp=1-1; j_temp<=20-1; j_temp++){
                //     tempSumUp += abs(wfUp(i_temp,j_temp));
                //     tempSumDn += abs(wfDn(i_temp,j_temp));
                // }
                // }
                // if(abs(tempSumUp) <= 10e-8){
                //     cout<<rank<<" wfUp left is zero: "<<tempSumUp<<endl;
                // }
                // if(abs(tempSumDn) <= 10e-8){
                //     cout<<rank<<" wfDn left is zero: "<<tempSumDn<<endl;
                // }
                // // 
                // wfUp.resize(40, 20);
                // wfDn.resize(40, 20);
                // MPI_Unpack(buf_recv_one.data(), buf_recv_one.size(), &posit, &logw, 1, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
                // MPI_Unpack(buf_recv_one.data(), buf_recv_one.size(), &posit, wfUp.data(), wfUp.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
                // MPI_Unpack(buf_recv_one.data(), buf_recv_one.size(), &posit, wfDn.data(), wfDn.size(), MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
                // tempSumUp=0.0;
                // tempSumDn=0.0;
                // for(int i_temp=1-1; i_temp<=40-1; i_temp++){
                // for(int j_temp=1-1; j_temp<=20-1; j_temp++){
                //     tempSumUp += abs(wfUp(i_temp,j_temp));
                //     tempSumDn += abs(wfDn(i_temp,j_temp));
                // }
                // }
                // if(abs(tempSumUp) <= 10e-8){
                //     cout<<rank<<" wfUp right is zero: "<<tempSumUp<<endl;
                // }
                // if(abs(tempSumDn) <= 10e-8){
                //     cout<<rank<<" wfDn right is zero: "<<tempSumDn<<endl;
                // }
                // cout<<"get rank: "<<rank<<endl;
                ///////////////////////////////////// 
                (walker[i]).unpack(buf_recv_one);

                count++;
                send_index_bk=send_index;
                recv_local_index=i;
            }
        }
    }
    if(rank==0) cout<<"!!! CHECKPOINT 8: After recv loop, count="<<count<<", buf_recv_number="<<buf_recv_number<<endl;

    if(count != buf_recv_number) {std::cout<<"Something is wrong in population control, buf_recv_number not consistent! "<<std::endl; exit(1);}


    //Wait
    if(rank==0) cout<<"!!! CHECKPOINT 9: About to MPI_Waitall, buf_send_number="<<buf_send_number<<endl;
    if(buf_send_number>0) MPI_Waitall(buf_send_number, ISREQ.data(), ISSTA.data() );

    // CHECKPOINT 10: After MPI_Waitall
    MPIBarrier();
    if(rank==0) cout<<"!!! CHECKPOINT 10: After MPI_Waitall, about to exit"<<endl;
    MPIBarrier();
    if(rank==0) cout<<"!!! CHECKPOINT 11: Exiting function normally"<<endl;
#else
    int j;
    for(int i=0; i<L; i++)
    {
        j=table[i]; //j is the parent 
        if(i!=j) walker[i]=walker[j];
    }
#endif
}

#endif
