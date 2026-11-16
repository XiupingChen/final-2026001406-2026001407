import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from creditbench.common import ROOT, allocations, child_env, dump_json, find_rscript
from creditbench.data import prepare, preprocess, scenario_indices
from creditbench.kernels import HistogramForest, QuantileBinner, split_bruteforce, split_numpy, split_python
from creditbench.metrics import evaluate, validation_threshold
from creditbench.run import monitored_run


def fixtures():
    rng = np.random.default_rng(123)
    cases = []
    for n, p, bins, leaf in [(1, 1, 2, 1), (12, 3, 4, 2), (100, 9, 16, 5), (17, 2, 5, 9)]:
        cases.append((rng.integers(bins, size=(n, p), dtype=np.int32), rng.integers(2, size=n, dtype=np.int32), bins, leaf))
    cases += [(np.zeros((20, 3), dtype=np.int32), np.arange(20, dtype=np.int32)%2, 4, 2),
              (rng.integers(4, size=(20, 2), dtype=np.int32), np.ones(20, dtype=np.int32), 4, 2),
              (np.tile(np.repeat([0, 1], 10)[:, None], (1, 3)).astype(np.int32), np.repeat([0, 1], 10).astype(np.int32), 2, 1)]
    return cases


@pytest.mark.parametrize("case", fixtures())
def test_histograms_against_independent_bruteforce(case):
    expected = split_bruteforce(*case)
    for function in (split_python, split_numpy):
        actual = function(*case)
        assert actual[:2] == expected[:2]
        assert abs(actual[2]-expected[2]) <= 1e-12


def test_r_and_rcpp(tmp_path):
    try:
        rscript = find_rscript()
    except FileNotFoundError:
        pytest.skip("Rscript unavailable: cross-language validation not performed")
    cases = []
    for i, (x, y, bins, leaf) in enumerate(fixtures()):
        xp, yp = tmp_path/f"x{i}.csv", tmp_path/f"y{i}.csv"
        np.savetxt(xp, x, delimiter=",", fmt="%d")
        np.savetxt(yp, y, fmt="%d")
        cases.append({"x": str(xp), "y": str(yp), "bins": bins, "min_leaf": leaf,
                      "expected": split_bruteforce(x, y, bins, leaf)})
    path = tmp_path/"cases.json"
    dump_json(path, cases)
    result = subprocess.run([rscript, "R/test_kernels.R", str(path)], cwd=ROOT, env=child_env(), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kernel", [split_numpy, split_python])
def test_invalid_kernel_input(kernel):
    with pytest.raises(ValueError):
        kernel(np.array([[0], [3]]), np.array([0, 1]), bins=3)
    with pytest.raises(ValueError):
        kernel(np.array([[0], [1]]), np.array([0, 2]))


def test_binner_and_forest():
    x = np.arange(100.)[:, None]
    y = (x[:, 0] > 50).astype(np.int32)
    model = HistogramForest(trees=10, max_depth=4, min_leaf=2, seed=7).fit(x, y)
    p = model.predict_proba(x)
    assert np.mean((p >= .5) == y) > .95
    reference = HistogramForest(trees=10, max_depth=4, min_leaf=2, seed=7, kernel=split_python).fit(x, y)
    np.testing.assert_allclose(p, reference.predict_proba(x), atol=1e-12, rtol=0)
    # Inference does not refit bin boundaries, including out-of-range values.
    before = [e.copy() for e in model.binner.edges]
    model.predict_proba(np.array([[-1e9], [1e9]]))
    for a, b in zip(before, model.binner.edges):
        np.testing.assert_array_equal(a, b)


def test_preprocess_training_only():
    df = pd.DataFrame({"x": [1., np.nan, 3., 100000.]})
    x, _, info = preprocess(df, [0, 1, 2])
    assert x[1, 0] == 2
    assert info["numeric_medians"]["x"] == 2


def test_scenarios_and_split(tmp_path):
    config = {"data": "synthetic", "synthetic_rows": 400, "seed": 5, "test_fraction": .2,
              "validation_fraction": .2, "cores": 4, "model": {"trees": 4}, "bootstrap": 4,
              "scenarios": [{"name": "base", "n": 100, "rate": .3, "parallel": True}]}
    manifest = prepare(config, tmp_path)
    splits = pd.read_csv(tmp_path/"splits.csv")
    assert splits.row_id.nunique() == 400
    ids = np.loadtxt(tmp_path/"base"/"train.csv", dtype=int)
    assert set(splits.loc[ids, "partition"]) == {"train"}
    assert len(set(ids)) == 100
    assert manifest["scenarios"][0]["observed_rate"] == .3
    boot = np.loadtxt(tmp_path/"base"/"bootstrap.csv", delimiter=",", dtype=int)
    assert boot.shape == (4, 100) and boot.min() >= 0 and boot.max() < 100
    with pytest.raises(ValueError):
        scenario_indices(np.array([0, 0, 0, 1]), np.arange(4), 20, .5, 1)


def test_metrics_known_values_and_budget():
    y = np.array([0, 0, 1, 1])
    prob = np.array([.1, .4, .35, .8])
    result = evaluate(y, prob, .35)
    assert result["roc_auc"] == .75
    assert result["pr_auc_ap"] == pytest.approx(5/6)
    assert result["brier"] == pytest.approx(np.mean((y-prob)**2))
    assert validation_threshold(y, prob) == .35
    assert allocations(4, 2)["hybrid"] == (2, 2)
    with pytest.raises(ValueError):
        allocations(5, 2)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group regression")
def test_monitor_includes_reparented_process_group_member(tmp_path):
    # A short-lived intermediate process launches a memory-heavy child and
    # exits.  The grandchild is no longer in the top process's descendant tree,
    # but it remains in the isolated process group created by monitored_run.
    grandchild = "import time; block=bytearray(80*1024*1024); time.sleep(.5)"
    middle = ("import subprocess,sys,time; "
              f"subprocess.Popen([sys.executable,'-c',{grandchild!r}]); time.sleep(.05)")
    parent = ("import subprocess,sys,time; "
              f"subprocess.run([sys.executable,'-c',{middle!r}]); time.sleep(.4)")
    measured = monitored_run([sys.executable, "-c", parent], tmp_path / "worker.log")
    assert measured["rss_processes_at_peak"] >= 2
    assert measured["peak_process_tree_rss_mib"] >= 80
