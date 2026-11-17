"""CLI orchestrator. All benchmark jobs run sequentially to avoid competition."""
import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from .common import ROOT, allocations, child_env, dump_json, find_rscript, hardware, read_json, sha256
from .metrics import evaluate, validation_threshold


def monitored_run(command, log):
    """Peak sum RSS for an isolated worker group; sampled, not an OS high-water mark.

    On POSIX, R's PSOCK workers can be reparented and disappear from the
    launching R process's descendant tree.  A new session/process group keeps
    those workers identifiable without accidentally including unrelated jobs.
    Windows retains descendant-tree discovery because it has no POSIX pgid.
    """
    peak = 0
    peak_processes = 0
    sample_count = 0
    sampling_error = None
    start = time.perf_counter()
    with open(log, "w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=child_env(), stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=(os.name != "nt"))
        try:
            root = psutil.Process(process.pid)
        except psutil.NoSuchProcess:
            root = None
        while process.poll() is None:
            rss = 0
            try:
                if os.name == "nt":
                    measured = [] if root is None else [root] + root.children(recursive=True)
                else:
                    measured = []
                    for candidate in psutil.process_iter(["pid"]):
                        try:
                            if os.getpgid(candidate.pid) == process.pid:
                                measured.append(candidate)
                        except (OSError, psutil.Error):
                            pass
            except psutil.NoSuchProcess:
                # A successful worker can exit between poll() and children().
                # Reap it below and use its exit code; this is not an RSS failure.
                measured = []
            except (OSError, psutil.Error) as exc:
                sampling_error = exc
                measured = []
            live_processes = 0
            for p in measured:
                try:
                    rss += p.memory_info().rss
                    live_processes += 1
                except psutil.Error:
                    pass
            if rss > peak:
                peak = rss
                peak_processes = live_processes
            sample_count += 1
            time.sleep(0.02)
        code = process.wait()
    elapsed = time.perf_counter()-start
    if code:
        tail = Path(log).read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"Worker exit {code}; see {log}\n{tail}")
    if sampling_error is not None:
        raise RuntimeError(f"RSS process discovery failed: {sampling_error}")
    return {"process_wall_seconds": elapsed, "peak_process_tree_rss_mib": peak/2**20,
            "rss_processes_at_peak": peak_processes, "memory_samples": sample_count}


def experiment(config_path, output, prepared_source, rscript=None, engines=None):
    config = read_json(config_path)
    prepared_source = Path(prepared_source).resolve()
    if read_json(prepared_source / "preparation_config.json") != config:
        raise ValueError("Prepared inputs were generated with a different config; rerun the preparation step")
    if engines:
        config["engines"] = engines
    if any(e not in {"sklearn", "ranger"} for e in config["engines"]):
        raise ValueError("Supported forest engines: sklearn, ranger")
    schemes = allocations(config["cores"], config["hybrid_outer"])
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be new or empty (prevents accidental result mixing): {output}")
    output.mkdir(parents=True, exist_ok=True)
    prepared = output / "prepared"
    shutil.copytree(prepared_source, prepared)
    manifests = read_json(prepared / "manifest.json")
    for relative_path, expected_hash in manifests["sha256"].items():
        if sha256(prepared / relative_path) != expected_hash:
            raise ValueError(f"Prepared input changed since creation: {relative_path}")
    for scenario in manifests["scenarios"]:
        scenario["directory"] = str((prepared / scenario["name"]).resolve())
    dump_json(prepared / "manifest.json", manifests)
    dump_json(output / "config.json", config)
    environment = hardware()
    environment.update(started=dt.datetime.now(dt.timezone.utc).isoformat(),
                       memory_scope=("Whole worker lifetime; POSIX isolated process-group sum captures reparented PSOCK workers; "
                                     "Windows uses process-tree sum; 20ms sample target; shared pages may double count."),
                       timing_scope="core: after data load and 3-tree warmup, includes bootstrap materialization, fit, validation/test prediction and task IPC; excludes metrics, disk output and pool teardown.",
                       cpu_scope="CPU slot budget only; no affinity, cores can be heterogeneous/SMT.")
    r_executable = find_rscript(rscript) if "ranger" in config["engines"] else None
    if r_executable:
        rinfo = subprocess.run([r_executable, "-e", "library(ranger); library(jsonlite); print(sessionInfo())"],
                               cwd=ROOT, env=child_env(), text=True, capture_output=True)
        if rinfo.returncode:
            raise RuntimeError(rinfo.stderr + "\nRun scripts/install_r.R first.")
        environment["R_sessionInfo"] = rinfo.stdout
    environment["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for folder in ("creditbench", "R", "src")
                                    for p in (ROOT/folder).rglob("*") if p.suffix in (".py", ".R", ".cpp")}
    dump_json(output / "environment.json", environment)
    data = pd.read_csv(prepared / "dataset.csv")
    y = data.target.to_numpy(dtype=int)
    validation = np.loadtxt(prepared / "validation.csv", dtype=int, ndmin=1)
    test = np.loadtxt(prepared / "test.csv", dtype=int, ndmin=1)
    jobs = []
    for scenario in manifests["scenarios"]:
        for engine in config["engines"]:
            for repeat in range(config["timing_repeats"]):
                for mode in (schemes if scenario["parallel"] else ["serial"]):
                    jobs.append((scenario, engine, repeat, mode))
    np.random.default_rng(config["seed"]+100).shuffle(jobs)
    timings, metrics, task_timings = [], [], []
    raw = output / "raw"
    raw.mkdir()
    for index, (scenario, engine, repeat, mode) in enumerate(jobs):
        job_id = f"{scenario['name']}__{engine}__{mode}__r{repeat}"
        print(f"[{index+1}/{len(jobs)}] {job_id}", flush=True)
        outer, inner = schemes[mode]
        jobdir = raw / job_id
        jobdir.mkdir()
        spec = {"prepared": str(prepared), "scenario_dir": scenario["directory"],
                "trees": scenario["trees"], "bootstrap": scenario["bootstrap"],
                "max_depth": config["model"]["max_depth"], "min_leaf": config["model"]["min_leaf"],
                "mtry": max(1, int(np.sqrt(manifests["p"]))), "outer": outer, "inner": inner,
                "prediction_path": str(jobdir / "predictions.csv"), "result_path": str(jobdir / "timing.json")}
        dump_json(jobdir / "spec.json", spec)
        command = ([sys.executable, "-m", "creditbench.worker"] if engine == "sklearn"
                   else [r_executable, str(ROOT / "R" / "worker.R")]) + [str(jobdir / "spec.json")]
        monitoring = monitored_run(command, jobdir / "worker.log")
        measured = read_json(jobdir / "timing.json")
        for b, (task_s, fit_s, predict_s) in enumerate(zip(measured["task_seconds"],
                                                            measured["fit_seconds"], measured["predict_seconds"])):
            task_timings.append({"scenario": scenario["name"], "engine": engine,
                                 "mode": mode, "repeat": repeat, "bootstrap_id": b,
                                 "task_seconds": task_s, "fit_seconds": fit_s,
                                 "predict_seconds": predict_s})
        if measured["core_seconds"] <= 0:
            raise ValueError("Timer resolution insufficient: non-positive core time cannot define speedup")
        predicted = np.loadtxt(jobdir / "predictions.csv", delimiter=",", ndmin=2)
        if predicted.shape != (len(validation)+len(test), scenario["bootstrap"]):
            raise ValueError(f"Unexpected predictions shape {predicted.shape} in {job_id}")
        key = {"scenario": scenario["name"], "engine": engine, "mode": mode, "repeat": repeat}
        timings.append({**key, "n": scenario["n"], "train_default_rate": scenario["observed_rate"],
                        "trees": scenario["trees"], "bootstrap": scenario["bootstrap"], "outer": outer, "inner": inner,
                        "cpu_slots": outer*inner, "core_seconds": measured["core_seconds"],
                        "initialization_seconds": measured["initialization_seconds"], **monitoring})
        # Timing repeats reuse the same statistical replicates: only serial repeat 0 enters metrics.
        if mode == "serial" and repeat == 0:
            train_ids = np.loadtxt(prepared / scenario["name"] / "train.csv", dtype=int, ndmin=1)
            boot_indices = np.loadtxt(prepared / scenario["name"] / "bootstrap.csv", dtype=int,
                                      delimiter=",", ndmin=2)
            for b in range(scenario["bootstrap"]):
                threshold = validation_threshold(y[validation], predicted[:len(validation), b])
                metrics.append({"scenario": scenario["name"], "engine": engine, "bootstrap_id": b,
                                "n": scenario["n"], "train_default_rate": scenario["observed_rate"],
                                **evaluate(y[test], predicted[len(validation):, b], threshold)})
                if engine == config["engines"][0]:
                    prevalence = float(y[train_ids[boot_indices[b]]].mean())
                    constant = np.full(len(test), prevalence)
                    baseline_threshold = validation_threshold(y[validation],
                                                               np.full(len(validation), prevalence))
                    metrics.append({"scenario": scenario["name"], "engine": "prevalence", "bootstrap_id": b,
                                    "n": scenario["n"], "train_default_rate": scenario["observed_rate"],
                                    **evaluate(y[test], constant, baseline_threshold)})
        np.savez_compressed(jobdir / "predictions.npz", probabilities=predicted,
                            evaluation_row_ids=np.r_[validation, test], validation_count=len(validation))
        (jobdir / "predictions.csv").unlink()
        pd.DataFrame(timings).to_csv(output / "timings.csv", index=False)
        pd.DataFrame(task_timings).to_csv(output / "task_timings.csv", index=False)
        pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    consistency = []
    for scenario in manifests["scenarios"]:
        for engine in config["engines"]:
            baseline = np.load(raw/f"{scenario['name']}__{engine}__serial__r0"/"predictions.npz")["probabilities"]
            for s, e, repeat, mode in jobs:
                if s["name"] != scenario["name"] or e != engine:
                    continue
                current = np.load(raw/f"{s['name']}__{e}__{mode}__r{repeat}"/"predictions.npz")["probabilities"]
                error = float(np.max(np.abs(current-baseline)))
                consistency.append({"scenario": s["name"], "engine": e, "mode": mode, "repeat": repeat,
                                    "max_absolute_error": error, "passed": error <= 1e-12})
    pd.DataFrame(consistency).to_csv(output / "consistency.csv", index=False)
    if not all(c["passed"] for c in consistency):
        raise AssertionError("Scheduling changed predictions; inspect consistency.csv before interpreting speedups")
    from .summarize import summarize
    summarize(output)
    dump_json(output / "COMPLETE.json", {"completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                         "jobs": len(jobs), "max_schedule_error": max(c["max_absolute_error"] for c in consistency)})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--prepared", required=True, help="Output directory from python -m creditbench.data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rscript")
    parser.add_argument("--engines", nargs="+", choices=["sklearn", "ranger"])
    args = parser.parse_args()
    experiment(args.config, args.output, args.prepared, args.rscript, args.engines)


if __name__ == "__main__":
    main()
