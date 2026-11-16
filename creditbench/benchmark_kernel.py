import argparse
import gc
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, child_env, dump_json, find_rscript, hardware, read_json
from .kernels import QuantileBinner, split_bruteforce, split_numpy, split_python


def benchmark(run, rscript=None, python_only=False):
    run = Path(run).resolve()
    config = read_json(run/"config.json")
    settings = config["kernel"]
    output = run/"kernel"
    if output.exists():
        raise FileExistsError("Kernel output exists; keep original measurements or select a new run")
    output.mkdir()
    frame = pd.read_csv(run/"prepared"/"dataset.csv")
    splits = pd.read_csv(run/"prepared"/"splits.csv")
    ids = splits.loc[splits.partition == "train", "row_id"].to_numpy(copy=True)
    np.random.default_rng(config["seed"]+7).shuffle(ids)
    x = frame.iloc[ids, :-1].to_numpy()
    y = frame.target.to_numpy(dtype=np.int32)[ids]
    z = QuantileBinner(settings["bins"]).fit(x).transform(x)
    records, cases = [], []
    for n in settings["rows"]:
        if n > len(y):
            raise ValueError(f"Kernel n={n} exceeds training partition size {len(y)}")
        xn, yn = z[:n], y[:n]
        expected = split_bruteforce(xn, yn, settings["bins"], 5)
        xp, yp = output/f"x_{n}.csv", output/f"y_{n}.csv"
        np.savetxt(xp, xn, delimiter=",", fmt="%d")
        np.savetxt(yp, yn, fmt="%d")
        cases.append({"x": str(xp), "y": str(yp), "feature": expected[0], "threshold": expected[1], "gain": expected[2]})
        for name, fn in (("Python", split_python), ("NumPy", split_numpy)):
            actual = fn(xn, yn, settings["bins"], 5)
            if actual[:2] != expected[:2] or abs(actual[2]-expected[2]) > 1e-12:
                raise AssertionError((name, actual, expected))
            batch = 1
            while True:
                start = time.perf_counter()
                for _ in range(batch):
                    fn(xn, yn, settings["bins"], 5)
                elapsed = time.perf_counter()-start
                if elapsed >= .03 or batch >= 16384:
                    break
                batch *= 2
            for repeat in range(settings["repeats"]):
                gc.collect()
                start = time.perf_counter()
                for _ in range(batch):
                    fn(xn, yn, settings["bins"], 5)
                elapsed = time.perf_counter()-start
                records.append({"engine": name, "n": n, "p": z.shape[1], "bins": settings["bins"],
                                "repeat_id": repeat, "batch": batch, "seconds": elapsed/batch,
                                "feature": actual[0], "threshold": actual[1], "gain": actual[2]})
    spec = {"cases": cases, "bins": settings["bins"], "min_leaf": 5, "repeats": settings["repeats"]}
    dump_json(output/"spec.json", spec)
    raw = pd.DataFrame(records)
    if not python_only:
        with open(output/"R_compile_and_benchmark.log", "w", encoding="utf-8") as log:
            completed = subprocess.run([find_rscript(rscript), str(ROOT/"R"/"benchmark_kernel.R"),
                                        str(output/"spec.json"), str(output/"raw_R.csv")],
                                       cwd=ROOT, env=child_env(), stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"R kernel benchmark failed: see {output/'R_compile_and_benchmark.log'}")
        raw = pd.concat([raw, pd.read_csv(output/"raw_R.csv")], ignore_index=True)
    raw.to_csv(output/"raw.csv", index=False)
    summary = raw.groupby(["n", "engine"]).seconds.agg(["median", "min", "max", "std"]).reset_index()
    baseline = summary[summary.engine == "Python"][["n", "median"]].rename(columns={"median": "python_seconds"})
    summary = summary.merge(baseline, on="n")
    summary["speedup_vs_python"] = summary.python_seconds / summary["median"]
    summary.to_csv(output/"summary.csv", index=False)
    environment = hardware()
    environment.update(algorithm="O(n*p + bins*p) histogram binary Gini split search; not a whole forest",
                       scope="Input validation, histogram construction, scan; excludes binning, compilation, file I/O; batch >=30ms target",
                       limitation="Engines are measured sequentially in fixed order; hardware drift remains possible. Kernel RSS not measured.")
    dump_json(output/"environment.json", environment)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4), layout="constrained")
    for name, frame in summary.groupby("engine"):
        ax.plot(frame.n, frame["median"]*1000, marker="o", label=name)
    ax.set(xlabel="Training rows", ylabel="Median time per split (ms, log scale)", yscale="log",
           title="Same histogram-Gini algorithm; compilation excluded")
    ax.legend()
    fig.savefig(output/"kernel_comparison.png", dpi=160)
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--rscript")
    p.add_argument("--python-only", action="store_true")
    a = p.parse_args()
    benchmark(a.run, a.rscript, a.python_only)
