"""One shared split, preprocessing fit on training only, explicit 0-based indices."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from .common import ROOT, dump_json, sha256

def load_raw(config, cleaned_csv=None):
    if config["data"] == "synthetic":
        x, y = make_classification(n_samples=config.get("synthetic_rows", 1000), n_features=16,
                                   n_informative=8, weights=[0.78, 0.22], random_state=config["seed"])
        return pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])]), y, {"source": "synthetic (NOT UCI)", "seed": config["seed"]}
    if config["data"] != "cleaned" or cleaned_csv is None:
        raise ValueError("Real-data experiment requires --cleaned from creditbench.clean")
    path = Path(cleaned_csv).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    from .clean import FEATURES
    if df.columns.tolist() != [*FEATURES, "target"] or len(df) != 30000:
        raise ValueError("Unexpected cleaned data schema or row count")
    y = df.pop("target").to_numpy(dtype=np.int32)
    if not set(np.unique(y)) <= {0, 1} or df.isna().any().any():
        raise ValueError("Unexpected cleaned data labels or missing values")
    if df.shape[1] != 23:
        raise ValueError("Expected 23 UCI predictors")
    return df, y, {"source": path.name, "cleaned_sha256": sha256(path),
                   "citation": "Yeh, I. (2009). Default of Credit Card Clients. UCI. DOI: 10.24432/C55S3H"}


def preprocess(df, training_rows):
    data = df.copy()
    # These undocumented category codes are pooled as 'other', not deleted.
    if "EDUCATION" in data:
        data["EDUCATION"] = data["EDUCATION"].where(data["EDUCATION"].isin([1, 2, 3, 4]), 4)
        data["MARRIAGE"] = data["MARRIAGE"].where(data["MARRIAGE"].isin([1, 2, 3]), 3)
    categorical = [c for c in ("SEX", "EDUCATION", "MARRIAGE") if c in data]
    numeric = [c for c in data if c not in categorical]
    numeric_data = data[numeric].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    medians = numeric_data.iloc[training_rows].median()
    if medians.isna().any():
        raise ValueError("All-missing training predictor")
    pieces = [numeric_data.fillna(medians).to_numpy()]
    names = numeric.copy()
    categories = {}
    for c in categorical:
        levels = sorted(data.iloc[training_rows][c].dropna().unique().tolist())
        categories[c] = levels
        for level in levels:
            pieces.append((data[c].to_numpy() == level).astype(float)[:, None])
            names.append(f"{c}_{level}")
    x = np.ascontiguousarray(np.column_stack(pieces), dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite prepared data")
    return x, names, {"numeric_medians": medians.to_dict(), "category_levels": categories,
                       "unknown_category": "all-zero encoding", "fit_partition": "training only"}


def scenario_indices(y, train, n, rate, seed):
    rng = np.random.default_rng(seed)
    rate = float(y[train].mean()) if rate is None else rate
    if n < 10 or not 0 < rate < 1:
        raise ValueError("Invalid scenario n or rate")
    n1 = int(round(n * rate))
    if not 0 < n1 < n:
        raise ValueError("Scenario must contain both classes")
    parts = []
    for label, count in ((0, n-n1), (1, n1)):
        pool = train[y[train] == label]
        if count > len(pool):
            raise ValueError(f"Need {count} distinct class-{label} rows; only {len(pool)} available")
        # The same seed across scenarios gives nested class-specific subsets.
        parts.append(rng.permutation(pool)[:count])
    return rng.permutation(np.concatenate(parts))


def prepare(config, directory, cleaned_csv=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    df, y, source = load_raw(config, cleaned_csv)
    rows = np.arange(len(y))
    trainval, test = train_test_split(rows, test_size=config["test_fraction"], stratify=y, random_state=config["seed"])
    frac = config["validation_fraction"] / (1-config["test_fraction"])
    train, validation = train_test_split(trainval, test_size=frac, stratify=y[trainval], random_state=config["seed"]+1)
    x, names, processing = preprocess(df, train)
    pd.DataFrame(x, columns=names).assign(target=y).to_csv(directory / "dataset.csv", index=False, float_format="%.17g")
    split = np.full(len(y), "train", dtype=object)
    split[validation] = "validation"
    split[test] = "test"
    pd.DataFrame({"row_id": rows, "partition": split}).to_csv(directory / "splits.csv", index=False)
    np.savetxt(directory / "validation.csv", validation, fmt="%d")
    np.savetxt(directory / "test.csv", test, fmt="%d")
    scenarios = []
    for item in config["scenarios"]:
        scenario = dict(item)
        sdir = directory / item["name"]
        sdir.mkdir(exist_ok=True)
        ids = scenario_indices(y, train, item["n"], item["rate"], config["seed"]+2)
        np.savetxt(sdir / "train.csv", ids, fmt="%d")
        b = item.get("bootstrap", config["bootstrap"])
        if b < config["cores"] and item["parallel"]:
            raise ValueError("B must be >= cores for the selected outer-parallel experiment")
        # All engines/schedules read exactly these bootstrap draws, not their own RNG draws.
        rng = np.random.default_rng(config["seed"]+3)
        boot = rng.integers(0, len(ids), size=(b, len(ids)))
        seeds = rng.integers(1, 2**30, size=b)
        np.savetxt(sdir / "bootstrap.csv", boot, fmt="%d", delimiter=",")
        np.savetxt(sdir / "seeds.csv", seeds, fmt="%d")
        scenario.update(bootstrap=b, trees=item.get("trees", config["model"]["trees"]),
                        observed_rate=float(y[ids].mean()), directory=str(sdir.resolve()))
        scenarios.append(scenario)
    metadata = {"source": source, "n": len(y), "p": x.shape[1], "features": names,
                "split_sizes": {k: int(np.sum(split == k)) for k in ("train", "validation", "test")},
                "split_default_rates": {k: float(y[split == k].mean()) for k in ("train", "validation", "test")},
                "preprocessing": processing, "scenarios": scenarios,
                "sha256": {str(p.relative_to(directory)): sha256(p) for p in directory.rglob("*.csv")}}
    dump_json(directory / "manifest.json", metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make split, preprocessing and fixed scenario inputs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cleaned", help="Required for real data; omit for synthetic smoke test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from .common import read_json
    dest = Path(args.output).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Choose a new or empty prepared-data directory: {dest}")
    config = read_json(args.config)
    manifest = prepare(config, dest, args.cleaned)
    dump_json(dest / "preparation_config.json", config)
    print(f"Prepared {manifest['n']} rows, {manifest['p']} encoded columns; split: {manifest['split_sizes']}")
