#ifndef TENSOR_HAO_MPI_SHARED_H
#define TENSOR_HAO_MPI_SHARED_H

#include <complex>
#include <mpi.h>
#include <cstdarg>
#include <vector>
#include <algorithm>
#include <iostream>

namespace tensor_hao
{

template<class T = double, size_t D = 1>
class TensorHaoMPIRef
{
private:
    size_t n[D];
    size_t nStep[D];
    size_t L;
    T* p;
    MPI_Win mpi_win;
    int* ref_count;

    void setNNstepL(const size_t* n_ptr)
    {
        std::copy(n_ptr, n_ptr + D, this->n);
        this->nStep[0] = 1;
        for(size_t i = 1; i < D; ++i) {
            this->nStep[i] = this->nStep[i-1] * this->n[i-1];
        }
        this->L = this->nStep[D-1] * this->n[D-1];
    }

public:
    TensorHaoMPIRef() : L(0), p(nullptr), mpi_win(MPI_WIN_NULL), ref_count(new int(0)) {
        for(size_t i = 0; i < D; ++i) {
            n[i] = 0;
            nStep[i] = 0;
        }
    }

    ~TensorHaoMPIRef() {
    }

    TensorHaoMPIRef(const TensorHaoMPIRef& x) : L(x.L), p(x.p), mpi_win(x.mpi_win), ref_count(x.ref_count) {
        std::copy(x.n, x.n + D, n);
        std::copy(x.nStep, x.nStep + D, nStep);
    }

    TensorHaoMPIRef(TensorHaoMPIRef&& x) : L(x.L), p(x.p), mpi_win(x.mpi_win), ref_count(x.ref_count) {
        std::copy(x.n, x.n + D, n);
        std::copy(x.nStep, x.nStep + D, nStep);
        x.p = nullptr;
        x.mpi_win = MPI_WIN_NULL;
        x.L = 0;
    }

    TensorHaoMPIRef& operator=(const TensorHaoMPIRef& x) {
        if (this != &x) {
            std::copy(x.n, x.n + D, n);
            std::copy(x.nStep, x.nStep + D, nStep);
            L = x.L;
            p = x.p;
            mpi_win = x.mpi_win;
        }
        return *this;
    }

    TensorHaoMPIRef& operator=(TensorHaoMPIRef&& x) {
        if (this != &x) {
            std::copy(x.n, x.n + D, n);
            std::copy(x.nStep, x.nStep + D, nStep);
            L = x.L;
            p = x.p;
            mpi_win = x.mpi_win;
            x.p = nullptr;
            x.mpi_win = MPI_WIN_NULL;
            x.L = 0;
        }
        return *this;
    }

    void createSharedMemory(const size_t* dims, int root, MPI_Comm comm) {
        if(mpi_win != MPI_WIN_NULL) {
            MPI_Win_free(&mpi_win);
            mpi_win = MPI_WIN_NULL;
        }
        setNNstepL(dims);
        MPI_Win_allocate_shared(L * sizeof(T), sizeof(T), MPI_INFO_NULL, comm, &p, &mpi_win);
    }

    void createSharedMemoryView(int root, MPI_Comm comm) {
        if(mpi_win != MPI_WIN_NULL) {
            MPI_Win_free(&mpi_win);
            mpi_win = MPI_WIN_NULL;
        }
        MPI_Win_allocate_shared(0, sizeof(T), MPI_INFO_NULL, comm, &p, &mpi_win);
    }

    void attachToSharedMemory(const size_t* dims, int root, MPI_Comm comm) {
        setNNstepL(dims);
        MPI_Aint win_size; int disp_unit;
        T* shared_ptr;
        MPI_Win_shared_query(mpi_win, root, &win_size, &disp_unit, &shared_ptr);
        p = shared_ptr;
    }

    void resize(const size_t* n_ptr) {
        size_t LBackup = L;
        setNNstepL(n_ptr);
        if(L != LBackup) {
            std::cout<<"Error! Resize L can not be different with original size in TensorHaoMPIRef!"<<std::endl;
            exit(1);
        }
    }

    template<typename... Values>
    void resize(size_t input, Values... inputs) {
        size_t len = sizeof...(Values) + 1;
        size_t vals[] = {input, static_cast<size_t>(inputs)...};
        if(len != D) {
            std::cout<<"Length of inputs number is not consistent with template class!!! "<<len<<" "<<D<<std::endl;
            exit(1);
        }
        resize(vals);
    }

    inline size_t size() const { return L; }

    inline const size_t* getRank() const { return n; }

    inline size_t rank(size_t i) const {
#ifndef NDEBUG
        if(i >= D || i < 0) {
            std::cout<<"Input i for rank() should be [0, D)!!! "<<i<<" "<<D<<std::endl;
            exit(1);
        }
#endif
        return n[i];
    }

    inline size_t rankStep(size_t i) const {
#ifndef NDEBUG
        if(i >= D || i < 0) {
            std::cout<<"Input i for rankStep() should be [0, D)!!! "<<i<<" "<<D<<std::endl;
            exit(1);
        }
#endif
        return nStep[i];
    }

    inline T* data() { return p; }
    inline const T* data() const { return p; }

    inline T operator()(size_t i0) const {
#ifndef NDEBUG
        if(D != 1) { std::cout<<"TensorHaoMPIRef::operator(size_t) only works for D=1 !!!"<<std::endl; exit(1); }
        if(i0 >= n[0] || i0 < 0) { std::cout<<"i0 is out of range in TensorHaoMPIRef::operator() !!!"<<std::endl; exit(1); }
#endif
        return p[i0];
    }

    inline T& operator()(size_t i0) {
#ifndef NDEBUG
        if(D != 1) { std::cout<<"TensorHaoMPIRef::operator(size_t) only works for D=1 !!!"<<std::endl; exit(1); }
        if(i0 >= n[0] || i0 < 0) { std::cout<<"i0 is out of range in TensorHaoMPIRef::operator() !!!"<<std::endl; exit(1); }
#endif
        return p[i0];
    }

    inline T operator()(size_t i0, size_t i1) const {
#ifndef NDEBUG
        if(D != 2) { std::cout<<"TensorHaoMPIRef::operator(size_t, size_t) only works for D=2 !!!"<<std::endl; exit(1); }
        if(i0 >= n[0] || i0 < 0) { std::cout<<"i0 is out of range in TensorHaoMPIRef::operator() !!!"<<std::endl; exit(1); }
        if(i1 >= n[1] || i1 < 0) { std::cout<<"i1 is out of range in TensorHaoMPIRef::operator() !!!"<<std::endl; exit(1); }
#endif
        return p[i0 + i1 * nStep[1]];
    }

    inline T& operator()(size_t i0, size_t i1) {
#ifndef NDEBUG
        if(D != 2) { std::cout<<"TensorHaoMPIRef::operator(size_t, size_t) only works for D=2 !!!"<<std::endl; exit(1); }
        if(i0 >= n[0] || i0 < 0) { std::cout<<"i0 is out of range in TensorHaoMPIRef::operator() !!!"<<std::endl; exit(1); }
        if(i1 >= n[1] || i1 < 0) { std::cout<<"i1 is out of range in TensorHaoMPIRef::operator() !!!"<<std::endl; exit(1); }
#endif
        return p[i0 + i1 * nStep[1]];
    }

    inline T operator()(size_t i0, size_t i1, size_t i2) const {
#ifndef NDEBUG
        if(D != 3) { std::cout<<"TensorHaoMPIRef::operator(size_t, size_t, size_t) only works for D=3 !!!"<<std::endl; exit(1); }
        if(i0 >= n[0] || i0 < 0) { std::cout<<"i0 is out of range in TensorHaoMPIRef::operator() !!!"<<std::endl; exit(1); }
        if(i1 >= n[1] || i1 < 0) { std::cout<<"i1 is out of range in TensorHaoMPIRef::operator() !!!"<<std::endl; exit(1); }
        if(i2 >= n[2] || i2 < 0) { std::cout<<"i2 is out of range in TensorHaoMPIRef::operator() !!!"<<std::endl; exit(1); }
#endif
        return p[i0 + i1 * nStep[1] + i2 * nStep[2]];
    }

    inline T& operator()(size_t i0, size_t i1, size_t i2) {
#ifndef NDEBUG
        if(D != 3) { std::cout<<"TensorHaoMPIRef::operator(size_t, size_t, size_t) only works for D=3 !!!"<<std::endl; exit(1); }
        if(i0 >= n[0] || i0 < 0) { std::cout<<"i0 is out of range in TensorHaoMPIRef::operator() !!!"<<std::endl; exit(1); }
        if(i1 >= n[1] || i1 < 0) { std::cout<<"i1 is out of range in TensorHaoMPIRef::operator() !!!"<<std::endl; exit(1); }
        if(i2 >= n[2] || i2 < 0) { std::cout<<"i2 is out of range in TensorHaoMPIRef::operator() !!!"<<std::endl; exit(1); }
#endif
        return p[i0 + i1 * nStep[1] + i2 * nStep[2]];
    }

    template<typename... Values>
    T operator()(size_t i0, size_t i1, size_t i2, size_t i3, Values... inputs) const {
        size_t vals[] = {i0, i1, i2, i3, inputs...};
#ifndef NDEBUG
        size_t len = sizeof...(Values);
        if(D != (len + 4)) {
            std::cout<<"TensorHaoMPIRef::operator(size_t...) not consistent with D !!!"<<std::endl;
            exit(1);
        }
        for(size_t i = 0; i < D; ++i) {
            if(vals[i] >= n[i] || vals[i] < 0) {
                std::cout<<"i... is out of range in TensorHaoMPIRef::operator() !!!"<<std::endl;
                exit(1);
            }
        }
#endif
        size_t index = 0;
        for(size_t i = 0; i < D; ++i) index += vals[i] * nStep[i];
        return p[index];
    }

    template<typename... Values>
    T& operator()(size_t i0, size_t i1, size_t i2, size_t i3, Values... inputs) {
        size_t vals[] = {i0, i1, i2, i3, inputs...};
#ifndef NDEBUG
        size_t len = sizeof...(Values);
        if(D != (len + 4)) {
            std::cout<<"TensorHaoMPIRef::operator(size_t...) not consistent with D !!!"<<std::endl;
            exit(1);
        }
        for(size_t i = 0; i < D; ++i) {
            if(vals[i] >= n[i] || vals[i] < 0) {
                std::cout<<"i... is out of range in TensorHaoMPIRef::operator() !!!"<<std::endl;
                exit(1);
            }
        }
#endif
        size_t index = 0;
        for(size_t i = 0; i < D; ++i) index += vals[i] * nStep[i];
        return p[index];
    }

    MPI_Win getWindow() const { return mpi_win; }

    double getMemory() const {
        double mem = 0.0;
        mem += 8.0 * D + 8.0 * D + 8.0 + 8.0;
        return mem;
    }
};

}

#endif
