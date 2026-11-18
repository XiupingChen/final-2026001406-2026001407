# Windows 修复与复现

此版本针对 Windows 增补运行修复、数据审计、随机流控制和预测对照。新的实验位于 `results/windows-20261117/`。

## 变更与实验口径

- 内存监控允许采样期间正常消失的进程，仍通过真实退出码识别任务失败；新增成功退出/非零退出的回归测试。
- Bootstrap 行号使用 seed+3，模型种子使用独立的 seed+4；增加 B 保持已有抽样与模型种子不变。新种子流意味着新预测不要求逐项等于历史 macOS 预测。
- 清洗记录重复 ID、去 ID 重复行、清洗后重复行和重复特征；主实验保留原始行，并在每份 prepared/manifest.json 记录跨分区重复特征。
- 补充实验使用 3 个预先固定种子，比较普通分层划分与按完全相同特征分组的划分。分组不使用目标标签来定义组；同一组不跨越训练、验证、测试。
- 每种划分中，逻辑回归和随机森林使用相同的 6,000 条训练数据、验证与测试数据。预处理只拟合实际训练子集。逻辑回归使用标准化、C=1，森林使用 200 棵树、深度 16、叶节点下限 5。补充实验不使用外层 Bootstrap，不与阶段一的 Bootstrap 模型混为一项实验。
- 分类阈值只在验证集选取；没有根据测试结果调参。3 个划分种子的均值和标准差是描述性结果，不作为总体置信区间。
- Brier 是概率预测的均方误差，不是纯校准误差。阶段一补充 10 个等宽区间的可靠性曲线及各区间样本量，预先固定使用 bootstrap_id=0，未按效果挑选复本；稀疏区间需谨慎解释。
- 计时汇总同时保存核心加速比 `speedup` 和完整进程加速比 `process_speedup`。4 个 CPU 槽位是并发预算，不是 CPU 亲和性绑定。
- `COMPLETE.json` 在一致性核对和汇总绘图全部成功后才生成。

## 已测试环境

Windows 11，Intel Core i7-13700H（14 物理核心、20 逻辑 CPU），约 31.73 GiB RAM；Python 3.14.8；R 4.3.3；ranger 0.18.0、Rcpp 1.1.2、jsonlite 2.0.0；Rtools43 GCC 12.3.0。Python 锁定依赖见 `requirements-windows-lock.txt`；逐阶段详细记录见 `environment.json`。

测试中 Python/R/Rcpp 及新增回归用例共 21 项通过，1 项 POSIX 进程组用例在 Windows 跳过。xlrd 在 pandas 中产生未来弃用警告，目前读取原始 XLS 正常；不能将跳过写成通过。

## PowerShell 复现

以下命令在仓库根目录运行。请将 Rscript 与 Rtools 路径替换为自己的安装路径；不需要修改系统 PATH，也不需要修改 PowerShell 的永久执行策略。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows-lock.txt
$env:LC_ALL = 'C'
$env:LANG = 'C'
& 'D:\download\R\download\R-4.3.3\bin\Rscript.exe' scripts/install_r.R
.\scripts\run_windows.ps1 -Python '.\.venv\Scripts\python.exe' -Rscript 'D:\download\R\download\R-4.3.3\bin\Rscript.exe' -Rtools 'D:\download\R\rtool\download\rtools43' -Output 'results/windows-local'
```

若 R 包需要源码编译，应先在本次 PowerShell 会话中将 Rtools 的 `usr/bin` 和 `x86_64-w64-mingw32.static.posix/bin` 加入 PATH，并设置与 R 版本匹配的 Rtools 环境变量。本次使用 `RTOOLS43_HOME`。已有旧 ranger 时应更新至支持 `min.bucket` 的版本（>=0.17.0）。

运行顺序：清洗、测试、双引擎 smoke、阶段一、阶段二、阶段三、四语言分裂内核、教学森林、逻辑回归/分组划分补充实验、完整性复算。各阶段顺序执行，避免基准任务相互争抢资源。每次使用末级名称不同的新输出目录；prepared 目录自动采用该名称作为前缀，避免不同批次混用。

## 结果文件

实测汇总与校验结果见 [Windows 实跑结果](results/windows-20261117/WINDOWS_RESULTS.md)。

- `tests.xml`：机器可读测试记录。
- `cleaning_audit.json`：清洗与重复记录审计。
- `smoke/`：双引擎八项小规模端到端验证。
- `phase1_prediction/`：12 项预测实验、原始概率、指标、可靠性曲线和教学森林。
- `phase2_parallel/`：40 项调度计时与四语言分裂内核实验。
- `phase3_scaling/`：120 项单因素规模实验。
- `sensitivity/`：12 次模型拟合、逐划分索引、预测概率、重复组隔离审计和汇总。

原始结果中的绝对路径是运行时的溯源信息。重新运行应使用脚本生成新的 spec 和 manifest，不要直接执行其他电脑保存的 spec.json。数据 SHA-256、种子和相对文件内容可用于独立核对。

## 解释边界

这是一个公开数据集、单台 Windows 机器上的课程实验。模型对照与分组划分有助于检验结论，但不等于外部数据验证。不同引擎的树算法与实现仍不完全相同，不能据此给 Python 和 R 整体速度排名。RSS 为进程树采样之和，包含共享页重复计数，并非精确物理内存占用。

本次代码修改、实验运行与结果整理在 Windows 环境中完成。
