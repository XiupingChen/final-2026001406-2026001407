"""Small checks that run without network or R, against the supplied local XLS."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from creditbench.clean import clean
from creditbench.common import ROOT, read_json
from creditbench.data import prepare
from creditbench.kernels import split_bruteforce, split_numpy, split_python
from creditbench.metrics import evaluate, validation_threshold


class ProjectChecks(unittest.TestCase):
    def test_source_and_cleaning_decisions(self):
        source = ROOT / "data/raw/default_of_credit_card_clients.xls"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "clean"
            cleaned = clean(source, output)
            audit = read_json(output / "audit.json")
            data = pd.read_csv(cleaned)
            self.assertEqual((audit["rows_before"], audit["rows_after"]), (30000, 30000))
            self.assertEqual(audit["target_counts"], {"0": 23364, "1": 6636})
            self.assertEqual(audit["recoded_education_rows"], 345)
            self.assertEqual(audit["recoded_marriage_rows"], 54)
            self.assertEqual(data.shape, (30000, 24))
            self.assertTrue((data["PAY_0"] < 0).any())
            self.assertTrue((data["BILL_AMT1"] < 0).any())
            self.assertFalse(data.isna().any().any())

    def test_preparation_and_baseline(self):
        config = read_json(ROOT / "configs/smoke.json")
        with tempfile.TemporaryDirectory() as tmp:
            manifest = prepare(config, tmp, ROOT / "data/cleaned/cleaned.csv")
            self.assertEqual(manifest["split_sizes"], {"train": 18000, "validation": 6000, "test": 6000})
            split = pd.read_csv(Path(tmp) / "splits.csv")
            ids = np.loadtxt(Path(tmp) / "base/train.csv", dtype=int)
            self.assertTrue((split.loc[ids, "partition"] == "train").all())
            y = pd.read_csv(Path(tmp) / "dataset.csv").target.to_numpy()
            self.assertEqual(len(np.unique(y)), 2)
            prob = np.full(4, .25)
            result = evaluate(np.array([0, 0, 0, 1]), prob, validation_threshold(np.array([0, 0, 0, 1]), prob))
            self.assertEqual(result["roc_auc"], .5)
            self.assertEqual((result["tp"], result["fp"]), (1, 3))

    def test_histogram_matches_independent_search(self):
        x = np.array([[0, 2], [1, 2], [2, 1], [3, 0], [3, 1], [2, 0]], dtype=np.int32)
        y = np.array([0, 0, 1, 1, 1, 0], dtype=np.int32)
        ref = split_bruteforce(x, y, 4, 1)
        for kernel in (split_numpy, split_python):
            actual = kernel(x, y, 4, 1)
            self.assertEqual(actual[:2], ref[:2])
            self.assertAlmostEqual(actual[2], ref[2], places=12)


if __name__ == "__main__":
    unittest.main()
