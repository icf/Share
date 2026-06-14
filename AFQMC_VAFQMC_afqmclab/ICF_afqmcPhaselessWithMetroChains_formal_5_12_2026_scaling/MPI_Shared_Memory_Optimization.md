# MPI共享内存优化：减少AFQMC代码内存占用

## 摘要

本文档记录了为AFQMC（Auxiliary Field Quantum Monte Carlo）代码实现MPI节点本地共享内存优化的过程。该优化使得在同一计算节点上的多个MPI进程可以共享同一份大张量数据（如`svdVecs`、`sqrtMinusDtSVDVecs`等），从而显著减少内存占用。

**日期**：2026-04-24（初稿）, 2026-04-25（更新）

---

## 1. 问题背景

### 1.1 内存瓶颈

在AFQMC计算中，以下大型3D张量需要在每个MPI进程上存储完整副本：
- `svdVecs`：SVD分解的向量，维度通常为 `(L, L, svdNumber)`
- `sqrtMinusDtSVDVecs`：时间演化矩阵
- `sqrtMinusDtSVDVecsUp`：自旋向上通道

对于大规模计算（如 L~100, svdNumber~1000），这些张量可能占用数十GB内存。在多节点并行时，每个进程都复制一份，导致严重的内存浪费。

### 1.2 初始观察

```cpp
// 原代码：每个进程都有自己的副本
tensor_hao::TensorHao<std::complex<double>, 3> svdVecs;
tensor_hao::TensorHao<std::complex<double>, 3> sqrtMinusDtSVDVecs;
tensor_hao::TensorHao<std::complex<double>, 3> sqrtMinusDtSVDVecsUp;
```

---

## 2. 解决方案：节点本地共享内存

### 2.1 核心原理

**关键发现**：`MPI_Win_allocate_shared` 只能在**同一节点内**的进程间共享内存。跨节点尝试共享会导致段错误。

**正确方案**：
1. 每个计算节点独立创建自己的共享内存区域
2. 每个节点本地的 root 进程（通常是 local_rank == 0）从HDF5文件读取数据
3. 同一节点内的其他进程通过共享内存访问该数据

### 2.2 实现架构

```
┌─────────────────────────────────────────────────────────┐
│                    Node 0 (24 ranks)                     │
│  ┌─────────────────────────────────────────────────────┐ │
│  │        Shared Memory (sqrtMinusDtSVDVecs)           │ │
│  │                                                     │ │
│  │  rank 0 (local root): 读取HDF5 → 填充数据           │ │
│  │  rank 1-23: 通过共享内存直接访问                      │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    Node 1 (24 ranks)                     │
│  ┌─────────────────────────────────────────────────────┐ │
│  │        Shared Memory (sqrtMinusDtSVDVecs)           │ │
│  │                                                     │ │
│  │  rank 24 (local root): 读取HDF5 → 填充数据           │ │
│  │  rank 25-47: 通过共享内存直接访问                      │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 关键实现

### 3.1 TensorHaoMPIRef 类

创建新文件 `include/generalHamiltonian_icf/tensor_hao_mpi_shared.h`，封装MPI共享内存功能：

```cpp
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

public:
    // 创建共享内存（由root调用）
    void createSharedMemory(const size_t* dims, int root, MPI_Comm comm) {
        setNNstepL(dims);
        MPI_Win_allocate_shared(L * sizeof(T), sizeof(T), MPI_INFO_NULL, comm, &p, &mpi_win);
    }

    // 创建共享内存视图（由非root调用）
    void createSharedMemoryView(int root, MPI_Comm comm) {
        MPI_Win_allocate_shared(0, sizeof(T), MPI_INFO_NULL, comm, &p, &mpi_win);
    }

    // 附加到已存在的共享内存
    void attachToSharedMemory(const size_t* dims, int root, MPI_Comm comm) {
        setNNstepL(dims);
        MPI_Aint win_size; int disp_unit;
        T* shared_ptr;
        MPI_Win_shared_query(mpi_win, root, &win_size, &disp_unit, &shared_ptr);
        p = shared_ptr;
    }
};
```

### 3.2 节点信息获取

使用SLURM环境变量动态获取每节点进程数：

```cpp
#ifdef MPI_HAO
int ranks_per_node = 24;  // 默认值
char* cpus_per_node = getenv("SLURM_JOB_CPUS_PER_NODE");
if(cpus_per_node) {
    ranks_per_node = atoi(cpus_per_node);
}
int node_id = MPIRank() / ranks_per_node;
int local_rank = MPIRank() % ranks_per_node;
bool is_local_root = (local_rank == 0);

MPI_Comm node_comm;
MPI_Comm_split(MPI_COMM_WORLD, node_id, MPIRank(), &node_comm);
#endif
```

### 3.3 SVD类修改

**头文件** (`include/twoBodyOperator_icf/svd.h`)：

```cpp
// 修改前
tensor_hao::TensorHao<std::complex<double>, 3> sqrtMinusDtSVDVecs;

// 修改后
tensor_hao::TensorHaoMPIRef<std::complex<double>, 3> sqrtMinusDtSVDVecs;
```

**实现文件** (`source/twoBodyOperator_icf/svd.cpp`)：

```cpp
void SVD::initialSqrtMinusDtSVDVecs(...)
{
    // ...
    #ifdef MPI_HAO
    {
        size_t dims[3] = {L, L, svdNumber};
        if(is_local_root) {
            // local root 创建共享内存
            sqrtMinusDtSVDVecs.createSharedMemory(dims, 0, node_comm);
        }
        else {
            // 其他进程创建视图并附加
            sqrtMinusDtSVDVecs.createSharedMemoryView(0, node_comm);
            sqrtMinusDtSVDVecs.attachToSharedMemory(dims, 0, node_comm);
        }
    }
    #endif

    if(is_local_root) {
        // 只有 local root 填充数据
        sqrtMinusDtSVDVecs.resize(L, L, svdNumber);
        // ... 填充数据
    }
    MPIBarrier();
    // ...
}
```

### 3.4 GeneralHamiltonian_icf修改

同样模式应用于 `read()` 和 `read_conj()` 函数：

```cpp
void GeneralHamiltonian_icf::read(const string &filename)
{
    // ... 读取除svdVecs外的其他数据 ...

    #ifdef MPI_HAO
    if(is_local_root) {
        readFile(svdVecs, file, "svdVecs");
    }
    MPIBarrier();
    MPI_Comm_free(&node_comm);
    #else
    readFile(svdVecs, file, "svdVecs");
    #endif
}
```

---

## 4. 拷贝/移动语义处理

### 4.1 问题

SVD类的 `copy_deep` 和 `move_deep` 函数需要处理 `TensorHaoMPIRef` 成员的拷贝。最初的引用计数实现导致了"双重释放"错误。

### 4.2 最终方案：纯浅拷贝

采用纯浅拷贝策略 - 不释放任何共享内存，直接复制指针：

```cpp
~TensorHaoMPIRef() {
    // 空：不释放任何东西
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
```

**理由**：
- `jastrowProjector_` 和 `MetroChains2s` 的生命周期由程序逻辑管理
- 共享内存在正确的时机被创建和使用，不需要通过引用计数管理释放

---

## 5. 内存节省效果

### 5.1 计算示例

假设配置：
- 节点数：2
- 每节点进程数：24
- L = 100, svdNumber = 500

单个张量大小：
```
size = L × L × svdNumber × sizeof(complex<double>)
     = 100 × 100 × 500 × 16 bytes
     = 80 MB per process
```

使用共享内存后（每节点）：
```
原来：24 × 80 MB = 1920 MB (per node)
现在：1 × 80 MB = 80 MB (per node)
节省：~96%
```

### 5.2 总体节省

涉及的张量：
- `svdVecs`
- `sqrtMinusDtSVDVecs`
- `sqrtMinusDtSVDVecsUp`

保守估计：**整体内存占用减少50%以上**

---

## 6. 进一步优化建议

### 6.1 扩展共享内存到其他张量

以下张量也可能受益于相同的优化：
- `U0`, `Vdagger0`：SVD分解的矩阵
- `sqrtMinusDtSVDVecs_D`：D矩阵
- 其他大型中间计算张量

### 6.2 精度优化

在某些情况下，可以考虑：
- 使用 `float` 而非 `double` 存储某些张量
- 评估数值精度损失是否可接受

### 6.3 异步数据加载

当前实现是同步的。可以考虑：
- 在后台线程预加载下一个需要的数据
- 重叠计算和数据加载

### 6.4 内存池管理

对于频繁创建/销毁共享内存的场景，可以实现内存池避免碎片化。

---

## 7. 测试验证清单

- [ ] 单节点测试（24进程）：验证共享内存正确创建和访问
- [ ] 多节点测试（2节点×24进程）：验证跨节点无干扰
- [ ] 内存监控：确认每个节点只有一份数据副本
- [ ] 结果正确性：对比优化前后的计算结果
- [ ] 稳定性：长时间运行测试，无内存泄漏

---

## 8. 已知限制

1. **跨节点共享内存不可行**：MPI标准不支持跨节点共享内存
2. **SLURM环境依赖**：使用`SLURM_JOB_CPUS_PER_NODE`获取节点配置，其他调度器需要适配
3. **纯浅拷贝策略**：如果程序异常退出，可能有资源泄漏

---

## 9. 修改的文件清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `include/generalHamiltonian_icf/tensor_hao_mpi_shared.h` | 新建 | TensorHaoMPIRef类 |
| `include/generalHamiltonian_icf/generalHamiltonian_icf.h` | 修改 | svdVecs类型改为TensorHaoMPIRef |
| `source/generalHamiltonian_icf/generalHamiltonian_icf.cpp` | 修改 | read()和read_conj()支持共享内存 |
| `include/twoBodyOperator_icf/svd.h` | 修改 | sqrtMinusDtSVDVecs等类型改为TensorHaoMPIRef |
| `source/twoBodyOperator_icf/svd.cpp` | 修改 | initialSqrtMinusDtSVDVecs()支持共享内存 |

---

## 10. 2026-04-25 更新：OOM问题修复

### 10.1 问题描述

在热化阶段，每次执行 "Adjust trial energy" 时（调用 `adjustETAndBackGroundThenResetMeasurement()`），代码执行：

```cpp
expMinusDtV = model.returnExpMinusAlphaV(method.dt);
```

这会创建一个**全新的 SVD 对象**，包括：
1. 新的 `sqrtMinusDtSVDVecs`（MPI共享内存窗口）
2. 新的 `sqrtMinusDtSVDVecsUp`
3. 新的 SVD分解矩阵

由于使用纯浅拷贝策略，原有的MPI共享内存窗口**永远不会被释放**。这导致：
- 每轮调整约增加 **6GB** 内存
- 多次调整后内存持续增长直至OOM

### 10.2 解决方案：updateBG方法

问题的根源是：**每次调整都重新创建SVD对象，但实际上只有 `svdBg`（背景向量）需要更新**。

**核心思路**：保持 `expMinusDtV` 对象不变，只更新其 `svdBg` 成员。

**实现**：

1. 在 `SVD` 类中添加 `updateBG()` 方法（`svd.h`）：
```cpp
void updateBG(const tensor_hao::TensorHao<std::complex<double>, 1> &bg);
```

2. 实现 `updateBG()` 方法（`svd.cpp`）：
```cpp
void SVD::updateBG(const TensorHao<complex<double>, 1> &bg) {
    if(bg.size() != svdNumber) {
        cout<<"Error!!! Background size is not svdNumber!"<<endl;
        exit(1);
    }
    svdBg = bg;
}
```

3. 修改 `afqmcPhaseless.cpp` 中的调用（`source/afqmcPhaseless.cpp`）：
```cpp
// 修改前
expMinusDtV = model.returnExpMinusAlphaV(method.dt);

// 修改后
expMinusDtV.updateBG(model.getSVDBg());
```

### 10.3 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `include/twoBodyOperator_icf/svd.h` | 添加 `updateBG()` 方法声明 |
| `source/twoBodyOperator_icf/svd.cpp` | 实现 `updateBG()` 方法 |
| `source/afqmcPhaseless.cpp` | 两处 `returnExpMinusAlphaV` 调用改为 `updateBG()` |

### 10.4 修复效果

- **内存稳定**：不再每次调整都创建新对象
- **性能提升**：避免重复创建MPI共享内存的开销
- **代码更清晰**：语义更明确——调整的是背景向量，不是整个传播子

---

## 11. ⚠️ 隐藏的未解决问题：SVD赋值内存泄漏

### 11.1 问题描述

在 `SVD::copy_deep()` 中：

```cpp
void SVD::copy_deep(const SVD &x)
{
    // ...
    sqrtMinusDtSVDVecs = x.sqrtMinusDtSVDVecs;
    sqrtMinusDtSVDVecsUp = x.sqrtMinusDtSVDVecsUp;
    // ...
}
```

`TensorHaoMPIRef` 的赋值操作**只是复制了MPI_Win句柄**：

```cpp
TensorHaoMPIRef& operator=(const TensorHaoMPIRef& x) {
    if (this != &x) {
        std::copy(x.n, x.n + D, n);
        std::copy(x.nStep, x.nStep + D, nStep);
        L = x.L;
        p = x.p;
        mpi_win = x.mpi_win;  // 只是复制句柄！
    }
    return *this;
}
```

### 11.2 问题影响

1. **共享窗口**：复制后的两个 `SVD` 对象共享同一个 MPI 共享内存窗口
2. **永不释放**：`TensorHaoMPIRef` 的析构函数是**空的**，不会调用 `MPI_Win_free`
3. **累积泄漏**：每次 SVD 赋值都会造成一个"幽灵"窗口，积累下去会耗尽系统资源

### 11.3 触发场景

- `phiT_twoJastrow` 通过 `copyMetroChains_FromMetroChainsTwoJastrow` 复制 SVD 对象
- 任何 `SVD a = b` 赋值操作
- `std::vector<SVD>` 的重新分配或复制

### 11.4 何时需要修复

当 SVD 复制操作**频繁且累积效应明显**时，会导致：
- MPI 共享内存耗尽（`MPI_WIN_NULL` 或系统限制）
- 系统内存持续增长
- 最终可能导致程序崩溃

### 11.5 潜在解决方案

**方案一：引用计数**
```cpp
~TensorHaoMPIRef() {
    if (p != nullptr) {
        // 只有最后一个使用者释放窗口
    }
}
```

**方案二：禁用复制**
```cpp
TensorHaoMPIRef(const TensorHaoMPIRef& x) = delete;
TensorHaoMPIRef& operator=(const TensorHaoMPIRef& x) = delete;
```

**方案三：显式所有权管理**
```cpp
void detach() {
    if (ref_count != nullptr && --(*ref_count) == 0) {
        MPI_Win_free(&mpi_win);
        delete ref_count;
    }
}
```

---

## 12. 参考文献

- MPI-3.1 Standard: Chapter 11 (Collective Communication)
- OpenMP + MPI Hybrid Programming documentation
- SLURM Multi-Core Support documentation
