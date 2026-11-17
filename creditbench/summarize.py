import argparse
from pathlib import Path

from .common import read_json, dump_json, sha256

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize(output):
    output = Path(output)
    timing = pd.read_csv(output / "timings.csv")
    metrics = pd.read_csv(output / "metrics.csv")
    task_times = pd.read_csv(output / "task_timings.csv")
    task_times.groupby(["scenario", "engine", "mode"])[
        ["task_seconds", "fit_seconds", "predict_seconds"]
    ].median().reset_index().to_csv(output / "task_timing_summary.csv", index=False)
    keys = ["scenario", "engine", "mode"]
    summary = timing.groupby(keys, sort=False).agg(
        seconds_median=("core_seconds", "median"), seconds_q25=("core_seconds", lambda x: x.quantile(.25)),
        seconds_q75=("core_seconds", lambda x: x.quantile(.75)),
        process_seconds_median=("process_wall_seconds", "median"),
        init_seconds_median=("initialization_seconds", "median"),
        rss_mib_max=("peak_process_tree_rss_mib", "max"), cpu_slots=("cpu_slots", "first"),
        n=("n", "first"), trees=("trees", "first"), bootstrap=("bootstrap", "first"), repeats=("repeat", "count")).reset_index()
    serial = summary[summary["mode"] == "serial"][["scenario", "engine", "seconds_median"]].rename(columns={"seconds_median": "serial_seconds"})
    summary = summary.merge(serial, on=["scenario", "engine"], validate="many_to_one")
    summary["speedup"] = summary.serial_seconds / summary.seconds_median
    summary["efficiency"] = summary.speedup / summary.cpu_slots
    process_serial = summary[summary["mode"] == "serial"][["scenario", "engine", "process_seconds_median"]].rename(columns={"process_seconds_median": "serial_process_seconds"})
    summary = summary.merge(process_serial, on=["scenario", "engine"], validate="many_to_one")
    summary["process_speedup"] = summary.serial_process_seconds / summary.process_seconds_median
    summary.to_csv(output / "timing_summary.csv", index=False)
    # A paired ratio per timing repeat is also retained; ratio-of-medians is the figure convention.
    paired = timing.merge(timing[timing["mode"] == "serial"][["scenario", "engine", "repeat", "core_seconds"]],
                          on=["scenario", "engine", "repeat"], suffixes=("", "_serial"), validate="many_to_one")
    paired["paired_speedup"] = paired.core_seconds_serial / paired.core_seconds
    paired.to_csv(output / "paired_speedups.csv", index=False)
    names = ["roc_auc", "pr_auc_ap", "recall_05", "recall_valid_f1", "precision_valid_f1",
             "f1_test_at_valid_threshold", "brier", "threshold", "tp", "fp", "fn", "tn"]
    quality = metrics.groupby(["scenario", "engine"])[names].agg(["mean", "std", "min", "max"])
    quality.columns = ["_".join(c) for c in quality.columns]
    quality.reset_index().to_csv(output / "quality_summary.csv", index=False)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    engine_colors = {"sklearn": "#087F8C", "ranger": "#D97331"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 160})
    scenarios = [s for s in summary.scenario.unique() if len(summary[summary.scenario == s]["mode"].unique()) > 1]
    for scenario in scenarios:
        subset = summary[summary.scenario == scenario]
        engines = [e for e in engine_colors if e in subset.engine.unique()]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), layout="constrained")
        modes = ["serial", "outer", "inner", "hybrid"]
        for i, engine in enumerate(engines):
            v = subset[subset.engine == engine].set_index("mode").loc[modes]
            x = np.arange(4)+(i-(len(engines)-1)/2)*.36
            axes[0].bar(x, v.seconds_median, width=.34, label=engine, color=engine_colors[engine],
                        yerr=np.vstack([v.seconds_median-v.seconds_q25, v.seconds_q75-v.seconds_median]), capsize=3)
            axes[1].bar(x, v.speedup, width=.34, label=engine, color=engine_colors[engine])
            axes[2].bar(x, v.rss_mib_max, width=.34, label=engine, color=engine_colors[engine])
        for ax, label in zip(axes, ["Core time (s), median/IQR", "Speedup vs same-engine serial", "Peak sum RSS (MiB)"]):
            ax.set_xticks(range(4), modes)
            ax.set_title(label)
        axes[0].legend()
        fig.suptitle(f"{scenario}: n={v.n.iloc[0]}, trees={v.trees.iloc[0]}, B={v.bootstrap.iloc[0]}")
        fig.savefig(figures/f"parallel_{scenario}.png")
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    order = [s["name"] for s in read_json(output/"config.json")["scenarios"]]
    for ax, metric in zip(axes, ["roc_auc", "pr_auc_ap", "brier"]):
        for i, engine in enumerate(e for e in engine_colors if e in metrics.engine.unique()):
            grouped = metrics[metrics.engine == engine].groupby("scenario")[metric].agg(["mean", "std"]).reindex(order)
            ax.errorbar(np.arange(len(order))+i*.12, grouped["mean"], yerr=grouped["std"].fillna(0), marker="o", linestyle="none", color=engine_colors[engine], capsize=3, label=engine)
        ax.set_title(f"{metric}: mean +/- bootstrap SD")
        ax.set_xticks(range(len(order)), order, rotation=25, ha="right")
    axes[0].legend()
    fig.savefig(figures/"prediction_quality.png")
    plt.close(fig)
    environment = read_json(output/"environment.json")
    environment["postprocessing_sha256"] = {"creditbench/summarize.py": sha256(__file__)}
    dump_json(output/"environment.json", environment)
    manifest = read_json(output / "prepared" / "manifest.json")
    (output/"RESULTS.md").write_text(
        "# 实验结果说明\n\n"
        f"数据来源：{manifest['source']['source']}；SHA-256：{manifest['source'].get('cleaned_sha256', 'synthetic')}\n\n"
        "- timing_summary.csv：同一引擎串行时间/本模式时间；不跨引擎计算并行加速比。\n"
        "- quality_summary.csv：同一固定测试集上不同外层 Bootstrap 模型的均值和标准差，不是测试总体的置信区间。\n"
        "- 时间重复使用相同种子；其次数不能作为额外统计重复数。\n"
        "- prevalence 为每次 Bootstrap 训练集违约率的常数概率基准；不使用测试集比例。\n"
        "- AP 采用 average_precision_score；Precision/Recall/F1/混淆矩阵使用验证集 F1 最优阈值在测试集上评价。\n"
        "- task_timing_summary.csv 的拟合和预测分项是逐任务中位数，核心时间另含取样与任务通信。\n"
        "- 内存是 worker RSS 采样总和：POSIX 按独立进程组（含重设父进程的 R PSOCK worker），Windows 按进程树；可能重复统计共享页，不是精确物理内存；完整 worker 时间含进程启动。\n"
        "- sklearn/ranger 的算法细节和内部实现不同，这组结果不能概括为 Python 比 R 快或慢。\n"
        "- consistency.csv 核对同一引擎跨调度/计时重复的概率最大绝对误差，容差 1e-12。\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    summarize(parser.parse_args().output)
