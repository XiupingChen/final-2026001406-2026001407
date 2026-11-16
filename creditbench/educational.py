"""Show that our split kernel builds working trees, and quantify bin approximation.

This is a serial educational experiment, separate from scheduling benchmarks.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .common import dump_json, read_json
from .kernels import HistogramForest
from .metrics import evaluate, validation_threshold


def run(directory):
    directory = Path(directory)
    config = read_json(directory/"config.json")
    prepared = directory/"prepared"
    data = pd.read_csv(prepared/"dataset.csv").to_numpy()
    x, y = data[:, :-1], data[:, -1].astype(np.int32)
    base = config["scenarios"][0]["name"]
    train = np.loadtxt(prepared/base/"train.csv", dtype=int, ndmin=1)[:2000]
    validation = np.loadtxt(prepared/"validation.csv", dtype=int, ndmin=1)
    test = np.loadtxt(prepared/"test.csv", dtype=int, ndmin=1)
    records = []
    models = [("histogram_16", 16), ("histogram_32", 32), ("histogram_64", 64), ("sklearn_exact", None)]
    for name, bins in models:
        for repeat in range(3):
            seed = config["seed"] + repeat
            if bins is None:
                model = RandomForestClassifier(n_estimators=20, max_depth=8, min_samples_leaf=5,
                                               min_samples_split=10, n_jobs=1, random_state=seed)
            else:
                model = HistogramForest(trees=20, max_depth=8, min_leaf=5, bins=bins, seed=seed)
            start = time.perf_counter()
            model.fit(x[train], y[train])
            fit_seconds = time.perf_counter()-start
            start = time.perf_counter()
            pv, pt = model.predict_proba(x[validation]), model.predict_proba(x[test])
            predict_seconds = time.perf_counter()-start
            if bins is None:
                pv, pt = pv[:, 1], pt[:, 1]
            threshold = validation_threshold(y[validation], pv)
            records.append({"engine": name, "replicate": repeat, "n": len(train), "bins": bins,
                            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                            **evaluate(y[test], pt, threshold)})
    pd.DataFrame(records).to_csv(directory/"educational_forest.csv", index=False)
    dump_json(directory/"educational_scope.json", {
        "purpose": "Show original histogram kernel inside recursive CART and a bootstrap forest; compare bin budgets and task quality.",
        "timing": "Exploratory fit/predict observations, no separate warmup; not included in formal performance rankings.",
        "comparison": "Same training rows, tree count/depth/leaf settings; internal RNG, split candidates and implementations differ.",
        "limitation": "Quantile histogram is approximate CART. Exact sklearn baseline is task-level only; not a controlled language benchmark."
    })
    print(pd.DataFrame(records).groupby("engine")[["roc_auc", "pr_auc_ap", "brier"]].mean().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    run(parser.parse_args().run)
