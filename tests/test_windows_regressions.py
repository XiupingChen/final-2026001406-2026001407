"""Regression tests for monitoring races and nested experiment inputs."""
import subprocess
from unittest.mock import Mock
from types import SimpleNamespace

import numpy as np
import psutil
import pytest

from creditbench import run
from creditbench.data import prepare


@pytest.mark.parametrize("exit_code", [0, 7])
def test_disappearing_worker_uses_actual_exit_code(monkeypatch, tmp_path, exit_code):
    child = Mock(pid=123, poll=Mock(side_effect=[None, exit_code]), wait=Mock(return_value=exit_code))
    root = Mock()
    root.children.side_effect = psutil.NoSuchProcess(123)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(psutil, "Process", lambda pid: root)
    # Select Windows discovery independently of the host OS.
    monkeypatch.setattr(run, "os", SimpleNamespace(name="nt"))
    if exit_code:
        with pytest.raises(RuntimeError, match="Worker exit 7"):
            run.monitored_run(["worker"], tmp_path / "worker.log")
    else:
        assert run.monitored_run(["worker"], tmp_path / "worker.log")["memory_samples"] == 1


def test_bootstrap_and_model_seeds_keep_prefix(tmp_path):
    config = {"data": "synthetic", "synthetic_rows": 400, "seed": 5,
              "test_fraction": .2, "validation_fraction": .2, "cores": 4,
              "model": {"trees": 4}, "bootstrap": 4,
              "scenarios": [{"name": "base", "n": 100, "rate": .3, "parallel": True},
                            {"name": "more", "n": 100, "rate": .3, "parallel": True, "bootstrap": 8}]}
    prepare(config, tmp_path)
    for name in ("bootstrap.csv", "seeds.csv"):
        a = np.loadtxt(tmp_path / "base" / name, delimiter=",")
        b = np.loadtxt(tmp_path / "more" / name, delimiter=",")
        np.testing.assert_array_equal(a, b[:4])


def test_grouped_partition_and_calibration_endpoints():
    from creditbench.sensitivity import partitions, calibration_rows
    groups = np.repeat(np.arange(100), 4)
    y = (groups % 4 == 0).astype(int)
    train, valid, test = partitions(y, groups, 42, True)
    assert len(set(train) | set(valid) | set(test)) == len(y)
    assert not (set(groups[train]) & set(groups[test]))
    assert not (set(groups[train]) & set(groups[valid]))
    assert not (set(groups[valid]) & set(groups[test]))
    bins = calibration_rows(np.array([0, 1]), np.array([0., 1.]))
    assert [b['bin'] for b in bins] == [0, 9]
    assert sum(b['count'] for b in bins) == 2
