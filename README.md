# 基于随机森林的信用违约预测与并行计算

本项目使用随包提供的 UCI 信用卡客户违约数据。请按下面的顺序在**自己的电脑**运行；每一步的输出是下一步的输入。当前交付只包含数据、代码、实验配置和测试，不包含报告或幻灯片。

## 0. 环境（macOS，项目根目录）

建议 Python 3.11–3.13，安装 R 4.3+ 与 C++ 编译环境后可运行 R/ranger 和 Rcpp；仅运行 Python 主线不需要 R。代码可在 Windows/Linux 使用，对应更改虚拟环境激活命令。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

所有命令须在解压后的 `credit-default-project/` 根目录执行。文件路径中有空格时用引号。程序不从网络自动下载或用模拟数据替代真实数据。原始表在 `data/raw/default_of_credit_card_clients.xls`。

## 1. 清洗原始数据

```bash
python -m creditbench.clean \
  --input data/raw/default_of_credit_card_clients.xls \
  --output data/cleaned
python -m unittest discover -s tests -p 'test_clean_and_stages.py' -v
```

查看 `data/cleaned/audit.json`、`cleaned.csv`、`row_mapping.csv`。原表 30,000 行、23 个预测字段，无缺失、无重复 ID。清洗检查字段和合法取值，移除不用于预测的 ID（另存映射），将 EDUCATION 的 0/5/6 合并为 4、MARRIAGE 的 0 合并为 3。原数据不需要缺失填补；还款状态的 -2/-1 和账单负值保留。审计文件记录原表/清洗文件 SHA-256、前后样本数、类别计数、合并量及保留负值的计数。若原始表结构不符，程序报错，不自行猜测。

类别独热编码以及训练集拟合的中位数在第 2 步执行。这里的中位数程序只是为防御其他同格式数据的缺失；随包这份数据无缺失，不会实际填补。全体训练部分共用同一套预处理转换，以固定不同场景的特征空间；测试集不参与拟合。

## 2. 小规模端到端检查

```bash
python -m creditbench.data \
  --config configs/smoke.json --cleaned data/cleaned/cleaned.csv \
  --output data/prepared/smoke
python -m creditbench.run \
  --config configs/smoke.json --prepared data/prepared/smoke \
  --engines sklearn --output results/smoke-python
```

这一步仍使用真实数据，但仅取 400 条训练记录、4 次外层 Bootstrap、12 棵树进行功能检查；不能把它当作正式结果。检查 `results/smoke-python/COMPLETE.json` 和 `consistency.csv` 是否全部通过。每个阶段的结果目录必须是新的或空的；重跑时换目录名，不要混用旧记录。

R 可用后运行：

```bash
Rscript scripts/install_r.R
python -m pytest -q
```

`pytest` 包含 Python、R、Rcpp 的正确性核对；没有 R 时对应测试会明确跳过。请留存实际输出，不要把跳过写成通过。若 R 不可用，下面三个阶段先用 `--engines sklearn`；R 安装好后用新的输出目录完整重跑 `--engines sklearn ranger`。

## 3. 实验一：预测与类别比例

先生成输入，再运行模型：

```bash
python -m creditbench.data \
  --config configs/phase1_prediction.json --cleaned data/cleaned/cleaned.csv \
  --output data/prepared/phase1
python -m creditbench.run \
  --config configs/phase1_prediction.json --prepared data/prepared/phase1 \
  --engines sklearn ranger --output results/phase1
```

训练量 2,000/6,000/12,000；在训练量 6,000 时另设违约率 10%/30%/50%。各场景使用 200 棵树、外层 Bootstrap 20 次；只运行串行模式（本阶段研究预测，不比较调度）。固定一次 60%/20%/20% 分层划分，验证集决定 F1 最大阈值，测试集始终保持自然违约比例。`metrics.csv` 每个 Bootstrap 一行，包含 ROC-AUC、AP、Brier、固定 0.5 的 Recall，以及验证阈值下的 Precision、Recall、F1、TP/FP/FN/TN。`prevalence` 为每次 Bootstrap **训练样本**违约率的常数概率基准，不使用测试标签设定概率。阈值选择后在测试集上只做评价；不要根据测试结果反复调参。

比较类别重采样时必须同时讨论召回、误报与概率校准；这些场景改变训练先验，不能仅凭 Recall 升高就判断模型整体更好。20 次 Bootstrap 的标准差是训练扰动下的波动，并非总体泛化误差置信区间。

## 4. 实验二：同一任务的四种调度

```bash
python -m creditbench.data \
  --config configs/phase2_parallel.json --cleaned data/cleaned/cleaned.csv \
  --output data/prepared/phase2
python -m creditbench.run \
  --config configs/phase2_parallel.json --prepared data/prepared/phase2 \
  --engines sklearn ranger --output results/phase2
```

训练量 6,000、200 棵树、外层 B=20，计时重复 5 次。4 个 CPU 配额分别分为串行 1×1、外层 4×1、内层 1×4、混合 2×2。每次计时复用相同 Bootstrap 行号和模型种子，只改变调度。`consistency.csv` 检查每个引擎内部各模式预测概率一致（最大误差 <= 1e-12），不要求 sklearn 和 ranger 的概率一致。

关注 `timing_summary.csv` 的核心时间中位数、`process_seconds_median` 的完整独立运行时间、RSS 采样最大值；`task_timings.csv` 另有每个任务的模型拟合与验证/测试预测耗时，`task_timing_summary.csv` 汇总分项中位数。核心时间不含进程启动、加载、预热和输出；完整时间包含这些开销。加速比必须用同场景、同引擎的串行时间作分母。RSS 在 POSIX 系统上是独立 worker 进程组的采样总和（可捕获重设父进程的 R PSOCK worker），Windows 上是进程树采样总和；可能重复计共享内存。不要跨机器混算加速比。

## 5. 实验三：分开改变任务规模

```bash
python -m creditbench.data \
  --config configs/phase3_scaling.json --cleaned data/cleaned/cleaned.csv \
  --output data/prepared/phase3
python -m creditbench.run \
  --config configs/phase3_scaling.json --prepared data/prepared/phase3 \
  --engines sklearn ranger --output results/phase3
```

从基准 n=6,000 / T=200 / B=20 出发，分别改 n=2,000/12,000、T=400、B=40。四种模式各计时重复 3 次。这个阶段运行量较大；可先为阶段三单独运行 `--engines sklearn`，记录实际时间和内存，然后再决定是否跑 R。不要为了“更快”私自改某种模式的树数或数据。每个配置的 `config.json`、数据哈希、随机种子、逐任务时间、逐次时间及预测概率都保存于结果目录。

## 6. 自写代码与扩展验证

与合作伙伴原版本保持一致：`creditbench/kernels.py` 实现 Python/NumPy Gini 直方图分裂、穷举参考和教学森林，`R/kernels.R`、`src/histogram_split.cpp` 为 R/Rcpp 同算法分裂。下列两项是**补充实验**，建议在主实验正常后运行：

```bash
python -m creditbench.benchmark_kernel results/phase2
python -m creditbench.educational results/phase1
```

内核比较单次节点分裂，不是整片随机森林或语言本身的速度；Rcpp 没有接入 ranger。若尚未安装 R，内核可先用 `--python-only`，以后用全新主实验结果目录做跨语言比较。教学森林仅说明自写内核能组成森林，并比较分箱数对任务效果的影响。两位成员应能解释完整模型、分裂内核与两层 Bootstrap 的区别。

## 数据和复现约定

- 数据：Yeh, I. (2009), *Default of Credit Card Clients*, UCI, DOI 10.24432/C55S3H；本包附有用户提供的原始 XLS。
- `data/cleaned/`、`data/prepared/`、`results/` 均为运行生成目录。每次运行保留 `environment.json` 与原始 CSV，不把不同电脑的计时记录合并。
- 核心预测依赖 scikit-learn 或 ranger；Rcpp 是独立分裂内核。`pytest` 是扩展测试，Python-only 的内置 `unittest` 不依赖 R。
- 老师要求公开 GitHub 仓库、真实开发记录、两人分工和项目范围内的 AI 使用声明。当前是**供两人继续开发的代码版本**；最终 README 须由两位成员按真实贡献、使用情况和验证方式填写姓名学号、仓库和最终 commit。不能把原先或本次 AI 协助生成的代码表述为两位成员独立手写。
- 当前没有制作 `report.pdf`、`slides.pdf`，也没有生成或声称完成正式阶段结果。具体后续任务见 `WORK_DONE_NEXT.md`。
