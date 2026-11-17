"""Matched logistic/forest baselines and duplicate-group holdout sensitivity.

Three prespecified split seeds, no hyperparameter selection on test data.
This supplement has no outer bootstrap and is separate from timing benchmarks.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .common import ROOT, dump_json, hardware, sha256
from .data import preprocess, scenario_indices
from .metrics import evaluate, validation_threshold


def partitions(y, groups, seed, grouped):
    rows = np.arange(len(y))
    if grouped:
        outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        tv, test = next(outer.split(rows, y, groups))
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed+1)
        train, valid = next(inner.split(tv, y[tv], groups[tv]))
        return tv[train], tv[valid], test
    tv, test = train_test_split(rows, test_size=.2, stratify=y, random_state=seed)
    train, valid = train_test_split(tv, test_size=.25, stratify=y[tv], random_state=seed+1)
    return train, valid, test


def calibration_rows(y, probability, **keys):
    edges = np.linspace(0, 1, 11)
    bin_id = np.minimum(np.searchsorted(edges, probability, side="right")-1, 9)
    return [{**keys, "bin": k, "count": int(np.sum(bin_id == k)),
             "mean_probability": float(probability[bin_id == k].mean()),
             "observed_rate": float(y[bin_id == k].mean())}
            for k in range(10) if np.any(bin_id == k)]


def run(cleaned, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cleaned)
    y = df.pop("target").to_numpy(dtype=np.int32)
    groups = df.groupby(list(df.columns), dropna=False).ngroup().to_numpy()
    settings = {"seeds": [20261111, 20261112, 20261113], "n": 6000,
                "forest": {"trees": 200, "max_depth": 16, "min_leaf": 5},
                "logistic": {"C": 1.0, "max_iter": 3000},
                "group_definition": "exact cleaned predictors (labels excluded)",
                "scope": "Matched models, no outer bootstrap; 3 split seeds are descriptive, not a confidence interval.",
                "data_sha256": sha256(cleaned)}
    dump_json(output / "config.json", settings)
    records, calibration, audits = [], [], []
    for grouped in (False, True):
        method = "grouped" if grouped else "row_stratified"
        for seed in settings["seeds"]:
            train, valid, test = partitions(y, groups, seed, grouped)
            if grouped:
                assert not (set(groups[train]) & set(groups[test]) or set(groups[train]) & set(groups[valid]) or set(groups[valid]) & set(groups[test]))
            ids = scenario_indices(y, train, settings["n"], None, seed+2)
            # Fit preprocessing on exactly the rows supplied to both models.
            x, _, _ = preprocess(df, ids)
            audits.append({"split": method, "seed": seed, "train_pool": len(train),
                           "train_used": len(ids), "validation": len(valid), "test": len(test),
                           "test_rate": float(y[test].mean()),
                           "test_rows_matching_train_predictors": int(np.isin(groups[test], groups[ids]).sum())})
            split_dir = output / f"{method}_{seed}"
            split_dir.mkdir()
            partition = np.full(len(y), "unused_train", dtype=object)
            partition[ids], partition[valid], partition[test] = "train", "validation", "test"
            pd.DataFrame({"row_id": np.arange(len(y)), "group": groups, "partition": partition}).to_csv(split_dir / "splits.csv", index=False)
            models = {
                "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000)),
                "forest": RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=5,
                                                  min_samples_split=10, n_jobs=1, random_state=seed)}
            for name, model in models.items():
                print(f"Supplement: {method}, seed={seed}, model={name}", flush=True)
                with threadpool_limits(limits=1):
                    model.fit(x[ids], y[ids])
                    pv, pt = model.predict_proba(x[valid])[:, 1], model.predict_proba(x[test])[:, 1]
                keys = {"split": method, "seed": seed, "model": name}
                records.append({**keys, **evaluate(y[test], pt, validation_threshold(y[valid], pv))})
                calibration.extend(calibration_rows(y[test], pt, **keys))
                np.savez_compressed(split_dir / f"{name}_predictions.npz", validation_row_ids=valid,
                                    test_row_ids=test, validation_probabilities=pv, test_probabilities=pt)
    pd.DataFrame(records).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(calibration).to_csv(output / "calibration.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "split_audit.csv", index=False)
    metrics = pd.DataFrame(records)
    metrics.groupby(["split", "model"])[["roc_auc", "pr_auc_ap", "brier", "f1_test_at_valid_threshold"]].agg(["mean", "std"]).to_csv(output / "summary.csv")
    environment = hardware()
    environment["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / "creditbench").glob("*.py")}
    dump_json(output / "environment.json", environment)
    dump_json(output / "COMPLETE.json", {"model_fits": len(records), "grouped_overlap": int(pd.DataFrame(audits).query("split == 'grouped'").test_rows_matching_train_predictors.sum())})


def plot_phase1(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    data = pd.read_csv(directory / "prepared/dataset.csv")
    test = np.loadtxt(directory / "prepared/test.csv", dtype=int)
    valid = np.loadtxt(directory / "prepared/validation.csv", dtype=int)
    curves = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, engine in zip(axes, ("sklearn", "ranger")):
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        for scenario in ("rate_010", "base", "rate_030", "rate_050"):
            p = np.load(directory / "raw" / f"{scenario}__{engine}__serial__r0" / "predictions.npz")["probabilities"][len(valid):, 0]
            rows = calibration_rows(data.target.to_numpy()[test], p, engine=engine, scenario=scenario, bootstrap_id=0)
            curves.extend(rows)
            ax.plot([r["mean_probability"] for r in rows], [r["observed_rate"] for r in rows], marker="o", label=scenario)
        ax.set(xlabel="Mean predicted probability", ylabel="Observed default rate", title=engine, xlim=(0, 1), ylim=(0, 1))
        ax.legend()
    fig.suptitle("Reliability: prespecified bootstrap model 0; 10 equal-width bins")
    fig.savefig(directory / "figures/calibration.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(curves).to_csv(directory / "calibration.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase1", required=True)
    args = parser.parse_args()
    run(args.cleaned, args.output)
    plot_phase1(args.phase1)
