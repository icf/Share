"""
AFQMC configuration for FeMoco BS8_5 with observable calculation.
This example uses UHF trial wavefunction for both trial and walker initialization.
"""

import os
import sys

sys.path.insert(0, "/home/zzkj/Zhi-Yu_Xiao/lib/tethys-master/tethys-master")

from hafqmc.afqmc import AFQMCConfig, afqmc_energy_from_pickle, load_hamiltonian

# ============================================================================
# 环境配置
# ============================================================================
os.environ["HAMIL_TYPE"] = "molecular"  # 使用分子哈密顿顿（而非周期边界条件）

# 指定使用的 GPU 设备，0-7 表示 8 块 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# ============================================================================
# 基础配置
# ============================================================================
cfg = AFQMCConfig.stochastic_example()  # 加载随机试验的默认配置

cfg.propagation.multi_gpu = True  # 启用多 GPU 并行模式

cfg.seed = 111  # 随机种子，保证计算可复现

cfg.trial_type = "stochastic"  # 使用随机试验波函数（而非单行列式）

# ============================================================================
# 诊断输出配置 - 控制计算过程中输出哪些观测量
# ============================================================================
cfg.diagnostics.print_s2 = True  # 输出 S² 期望值（用于检测自旋污染）
cfg.diagnostics.print_spin = True  # 输出自旋密度（轨道分辨）
cfg.diagnostics.print_rdm = True  # 输出 1-RDM（用于分析电子分布）

cfg.output.dir = "afqmc_output"  # 输出文件存放目录

# ============================================================================
# 传播参数 - 控制 AFQMC 的核心动力学
# ============================================================================
cfg.propagation.dt = 0.005  # 时间步长
# - 太小：收敛慢，计算量大
# - 太大：精度差，可能不稳定
# 典型值：0.001-0.01

cfg.propagation.n_walkers = 320  # walker 数量（每 GPU）
# - 影响统计误差：walker 越多，误差越小
# - 影响内存：walker 越多，内存需求越大
# 总 walker 数 = n_walkers * GPU数

cfg.propagation.n_block_steps = 50  # 每个 block 的传播步数
# block 是能量测量的基本单位

cfg.propagation.n_ene_measurements = 5  # 每个 block 内的能量测量次数
# 多次测量取平均可降低统计误差

cfg.propagation.n_blocks = 2000  # 总 block 数
# - 影响最终统计精度
# - 更多 block = 更精确的能量估计

cfg.propagation.n_eq_steps = 100  # 平衡化步骤数
# walker 初始化后需要一段时间达到平衡
# 这部分的数据通常不用于最终统计

cfg.propagation.ortho_freq = 1  # Gram-Schmidt 正交化频率
# =1 表示每步都做正交化
# >1 表示每隔这么多次才正交化一次
# 正交化防止 walker 之间线性相关

# ============================================================================
# 日志配置 - 控制屏幕和文件输出
# ============================================================================
cfg.log.enabled = True  # 启用日志输出
cfg.log.block_freq = 1  # 每隔多少 block 输出一次日志
cfg.log.equil_freq = 1  # 每隔多少步输出平衡化信息
cfg.log.equil_n_print = 5  # 平衡化时打印的 walker 统计信息数量
cfg.log.pop_control_stats = False  # 是否输出群体控制统计信息

# ============================================================================
# 输出文件配置
# ============================================================================
cfg.output.write_raw = True  # 写入原始能量数据
cfg.output.raw_path = "raw.dat"  # 原始数据文件名
cfg.output.write_hparams = True  # 写入超参数配置
cfg.output.hparams_path = "afqmc_hparams.yml"  # 超参数文件名

# 可视化配置
cfg.output.visualization.enabled = True  # 启用实时可视化
cfg.output.visualization.show = False  # 不弹窗显示（只在服务器上运行）
cfg.output.visualization.save_path = "eblock_afqmc.png"  # 保存的图片路径

# ============================================================================
# 群体控制参数 - 防止 walker 数量指数级增长/衰减
# ============================================================================
cfg.pop_control.init_noise = 0.0  # 初始 walker 加噪的幅度
# >0 可以增加 walker 多样性
# =0 表示不加噪声

cfg.pop_control.resample = True  # 是否使用重采样
# 重采样：复制高权重 walker，丢弃低权重 walker
# 维持 walker 数量稳定

cfg.pop_control.freq = 5  # 每隔多少步执行一次群体控制

cfg.pop_control.min_weight = 1.0e-3  # walker 最小权重
# 低于此值的 walker 被重新采样

cfg.pop_control.max_weight = 100.0  # walker 最大权重
# 高于此值的 walker 被分割

# ============================================================================
# 随机试验波函数参数 - 控制 VAFQMC trial wavefunction 的行为
# ============================================================================
# Trial wavefunction 路径
cfg.stochastic_trial.checkpoint = "checkpoint.pkl"  # 训练好的 trial wfn 检查点
cfg.stochastic_trial.hparams_path = "hparams.yml"  # trial wfn 超参数文件

# 采样配置
cfg.stochastic_trial.n_samples = 20  # 每次测量使用的样本数
# - 影响测量精度和内存
# - 样本越多，memory 越大

cfg.stochastic_trial.burn_in = 100  # HMC 采样的预热步数
# 预热期间 walker 趋向平衡分布

cfg.stochastic_trial.sample_update_steps = 1  # 每隔多少步更新一次样本
# =1 表示每步都更新

cfg.stochastic_trial.n_measure_samples = 5  # 每次 block 测量使用的样本数

# HMC sampler 配置
cfg.stochastic_trial.sampler.name = "hmc"  # 使用 HMC (Hybrid Monte Carlo) 采样器
# 替代方案：可能是其他 MCMC 方法

cfg.stochastic_trial.sampler.dt = 0.1  # HMC 的时间步长
# 与 propagation.dt 不同，这是 walker 探索的时间步

cfg.stochastic_trial.sampler.length = 1.0  # HMC 轨迹长度
# 影响 walker 探索效率

# ============================================================================
# 内存优化参数 - 防止 OOM 的关键参数
# ============================================================================

# local_energy_chunk_size: walker 级别的分块
# 将 walker 分成小块进行处理，降低峰值内存
# =5 表示每批处理 5 个 walker
# =0 表示禁用（所有 walker 一次性处理，高内存）
cfg.stochastic_trial.local_energy_chunk_size = 0

# sample_chunk_size: 样本级别的分块
# 决定如何处理 n_samples=20 个样本
#
# =0: 禁用分块，使用 vmap 并行处理所有样本
#     速度快，但内存需求 O(n_samples)
#     对于 n_samples=20: memory ~ 20x
#
# =1: 启用分块，使用 lax.scan 顺序处理每个样本
#     速度稍慢，但内存需求 O(1)
#     对于 n_samples=20: memory ~ 1x (每个样本单独处理)
#
# 示例（n_samples=20, n_walkers=20, 8 GPUs）:
#   sample_chunk_size=0 (vmap):
#     local_chunk=5, n_samples=20, effective_chunk=100 (per GPU)
#     n_local_walkers=20, total_configs=400 (per GPU, 8 GPUs=3200 total)
#     内存需求：O(100) configs per device
#
#   sample_chunk_size=1 (lax.scan):
#     local_chunk=5, n_samples=1, effective_chunk=5 (per GPU)
#     n_local_walkers=20, total_configs=20 (per GPU, 8 GPUs=160 total)
#     内存需求：O(5) configs per device - 显著降低！
#
# 对于大体系或 GPU 内存受限时，推荐 sample_chunk_size=1
cfg.stochastic_trial.sample_chunk_size = 0  # 1=用 lax.scan 省内存, 0=用 vmap 速度快

# ============================================================================
# Walker 初始化参数 - 控制如何生成初始 walker
# ============================================================================
# 当前配置：使用 trial wavefunction 初始化 walker
# 这意味着 walker 从 trial 参考态开始

cfg.stochastic_trial.init_walkers_burn_in = 100  # 初始化时的预热步数

cfg.stochastic_trial.init_walkers_chains_per_walker = 20  # 每个 walker 的马尔可夫链数
# 更多链 = 更高多样性 = 更好的初始采样

cfg.stochastic_trial.init_walkers_from_trial = (
    True  # 从 trial wavefunction 初始化 walker
)
# =True: walker 从 trial 参考态的分布采样
# =False: 从其他来源（如 ROHF）初始化

cfg.stochastic_trial.init_walkers_infer_steps = 10  # 初始化时的推断步数

# ============================================================================
# 运行 AFQMC
# ============================================================================
e, err = afqmc_energy_from_pickle("femoco_RHF.pkl", cfg=cfg)
print("E =", float(e), "+/-", float(err))
