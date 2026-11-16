# 实验结果说明

数据来源：cleaned.csv；SHA-256：9bc4c27d65e94d9b8e913efd978987cadf184d2922c0bb245a7f6fbc012e3daf

- timing_summary.csv：同一引擎串行时间/本模式时间；不跨引擎计算并行加速比。
- quality_summary.csv：同一固定测试集上不同外层 Bootstrap 模型的均值和标准差，不是测试总体的置信区间。
- 时间重复使用相同种子；其次数不能作为额外统计重复数。
- prevalence 为每次 Bootstrap 训练集违约率的常数概率基准；不使用测试集比例。
- AP 采用 average_precision_score；Precision/Recall/F1/混淆矩阵使用验证集 F1 最优阈值在测试集上评价。
- task_timing_summary.csv 的拟合和预测分项是逐任务中位数，核心时间另含取样与任务通信。
- 内存是 worker RSS 采样总和：POSIX 按独立进程组（含重设父进程的 R PSOCK worker），Windows 按进程树；可能重复统计共享页，不是精确物理内存；完整 worker 时间含进程启动。
- sklearn/ranger 的算法细节和内部实现不同，这组结果不能概括为 Python 比 R 快或慢。
- consistency.csv 核对同一引擎跨调度/计时重复的概率最大绝对误差，容差 1e-12。
