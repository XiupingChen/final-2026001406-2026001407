"""One isolated scikit-learn run. Spawn-safe on Windows; no execution at import."""
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import parallel_config

from .common import dump_json, read_json

STATE = {}


def load_state(spec):
    start = time.perf_counter()
    prepared = Path(spec["prepared"])
    data = pd.read_csv(prepared / "dataset.csv").to_numpy()
    train_ids = np.loadtxt(Path(spec["scenario_dir"]) / "train.csv", dtype=int, ndmin=1)
    validation = np.loadtxt(prepared / "validation.csv", dtype=int, ndmin=1)
    test = np.loadtxt(prepared / "test.csv", dtype=int, ndmin=1)
    STATE.update(spec=spec, x=data[train_ids, :-1].astype(np.float32),
                 y=data[train_ids, -1].astype(np.int32),
                 eval_x=data[np.r_[validation, test], :-1].astype(np.float32),
                 boots=np.loadtxt(Path(spec["scenario_dir"]) / "bootstrap.csv", delimiter=",", dtype=int, ndmin=2),
                 seeds=np.loadtxt(Path(spec["scenario_dir"]) / "seeds.csv", dtype=int, ndmin=1))
    return time.perf_counter()-start


def task(b, warm=False):
    s = STATE["spec"]
    start = time.perf_counter()
    ids = STATE["boots"][b]
    x, y = STATE["x"][ids], STATE["y"][ids]
    model = RandomForestClassifier(n_estimators=min(3, s["trees"]) if warm else s["trees"],
                                   criterion="gini", max_features=s["mtry"],
                                   max_depth=s["max_depth"], min_samples_leaf=s["min_leaf"],
                                   min_samples_split=2*s["min_leaf"], bootstrap=True,
                                   n_jobs=s["inner"], random_state=int(STATE["seeds"][b]))
    # Explicit threading is essential inside a multiprocessing worker: joblib's
    # default loky context can otherwise reduce effective_n_jobs to one.
    with parallel_config(backend="threading", n_jobs=s["inner"]):
        fit_start = time.perf_counter()
        model.fit(x, y)
        fit_seconds = time.perf_counter()-fit_start
        # Robust for a rare bootstrap sample with only one class.
        predict_start = time.perf_counter()
        proba = model.predict_proba(STATE["eval_x"])
        predict_seconds = time.perf_counter()-predict_start
    positive = np.flatnonzero(model.classes_ == 1)
    prediction = proba[:, positive[0]] if len(positive) else np.zeros(len(proba))
    return prediction, time.perf_counter()-start, fit_seconds, predict_seconds


def initializer(spec, ready):
    load_state(spec)
    task(0, warm=True)
    ready.put(True)


def run(spec):
    outer = spec["outer"]
    initialization = time.perf_counter()
    pool = None
    if outer == 1:
        load_state(spec)
        task(0, warm=True)
    else:
        ctx = mp.get_context("spawn")
        ready = ctx.Queue()
        pool = ctx.Pool(outer, initializer=initializer, initargs=(spec, ready))
        for _ in range(outer):
            ready.get(timeout=180)
    init_seconds = time.perf_counter()-initialization
    try:
        start = time.perf_counter()
        values = list(map(task, range(spec["bootstrap"]))) if pool is None else pool.map(task, range(spec["bootstrap"]), chunksize=1)
        core_seconds = time.perf_counter()-start
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    predictions = np.column_stack([v[0] for v in values])
    np.savetxt(spec["prediction_path"], predictions, delimiter=",", fmt="%.17g")
    dump_json(spec["result_path"], {"core_seconds": core_seconds, "initialization_seconds": init_seconds,
                                   "task_seconds": [v[1] for v in values],
                                   "fit_seconds": [v[2] for v in values],
                                   "predict_seconds": [v[3] for v in values]})


if __name__ == "__main__":
    run(read_json(sys.argv[1]))
