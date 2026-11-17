"""Validate and clean the supplied UCI credit-card default XLS, without downloading data."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import dump_json, sha256

FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + [
    f"PAY_{i}" for i in (0, 2, 3, 4, 5, 6)
] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
TARGET = "default payment next month"
RAW_COLUMNS = ["ID", *FEATURES, TARGET]


def clean(input_path, output_dir):
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose a new or empty output directory: {output_dir}")
    # First XLS row is the descriptive heading; the second contains column names.
    raw = pd.read_excel(input_path, sheet_name="Data", header=1, engine="xlrd")
    raw.columns = [str(c).strip() for c in raw.columns]
    if raw.columns.tolist() != RAW_COLUMNS or len(raw) != 30000:
        raise ValueError("Unexpected columns/order or row count; inspect the supplied XLS")
    if raw["ID"].isna().any() or raw["ID"].duplicated().any():
        raise ValueError("IDs must be complete and unique")
    if raw[TARGET].isna().any() or not set(raw[TARGET].unique()) <= {0, 1}:
        raise ValueError("Target must contain only binary 0/1 labels")
    if raw[FEATURES].isna().any().any():
        raise ValueError("Unexpected missing predictor in this fixed dataset")
    if not all(pd.api.types.is_numeric_dtype(raw[c]) for c in RAW_COLUMNS):
        raise ValueError("Unexpected non-numeric values")
    if not np.isfinite(raw[RAW_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values in source")
    if not set(raw.SEX.unique()) <= {1, 2}:
        raise ValueError("Unexpected SEX codes")
    if not set(raw.EDUCATION.unique()) <= set(range(7)):
        raise ValueError("Unexpected EDUCATION codes")
    if not set(raw.MARRIAGE.unique()) <= {0, 1, 2, 3}:
        raise ValueError("Unexpected MARRIAGE codes")
    for c in [f"PAY_{i}" for i in (0, 2, 3, 4, 5, 6)]:
        if not set(raw[c].unique()) <= set(range(-2, 9)):
            raise ValueError(f"Unexpected repayment status in {c}")
    if (raw.AGE < 18).any() or (raw.LIMIT_BAL <= 0).any():
        raise ValueError("Unexpected AGE or LIMIT_BAL values")
    if (raw[[f"PAY_AMT{i}" for i in range(1, 7)]] < 0).any().any():
        raise ValueError("Unexpected negative payment amount")

    clean_df = raw[FEATURES + [TARGET]].copy()
    # Pool undocumented codes into the existing 'other' category (4 or 3).
    education_before = raw.EDUCATION.value_counts().sort_index().to_dict()
    marriage_before = raw.MARRIAGE.value_counts().sort_index().to_dict()
    clean_df["EDUCATION"] = clean_df.EDUCATION.where(clean_df.EDUCATION.isin([1, 2, 3, 4]), 4)
    clean_df["MARRIAGE"] = clean_df.MARRIAGE.where(clean_df.MARRIAGE.isin([1, 2, 3]), 3)
    clean_df = clean_df.rename(columns={TARGET: "target"})
    # Never remove negative PAY_* statuses or negative BILL_AMT* values.
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = output_dir / "cleaned.csv"
    clean_df.to_csv(cleaned_path, index=False)
    pd.DataFrame({"row_id": np.arange(len(raw)), "source_id": raw.ID}).to_csv(
        output_dir / "row_mapping.csv", index=False
    )
    audit = {
        "input_name": input_path.name,
        "source_sha256": sha256(input_path), "cleaned_sha256": sha256(cleaned_path),
        "rows_before": len(raw), "rows_after": len(clean_df),
        "predictors_before": len(FEATURES), "predictors_after": len(FEATURES),
        "target_counts": {str(k): int(v) for k, v in raw[TARGET].value_counts().sort_index().items()},
        "missing_values": int(raw.isna().sum().sum()), "duplicate_ids": int(raw.ID.duplicated().sum()),
        "duplicate_raw_rows_excluding_id": int(raw.drop(columns="ID").duplicated().sum()),
        "duplicate_cleaned_rows": int(clean_df.duplicated().sum()),
        "duplicate_cleaned_predictors": int(clean_df[FEATURES].duplicated().sum()),
        "education_before": {str(k): int(v) for k, v in education_before.items()},
        "education_after": {str(k): int(v) for k, v in clean_df.EDUCATION.value_counts().sort_index().items()},
        "marriage_before": {str(k): int(v) for k, v in marriage_before.items()},
        "marriage_after": {str(k): int(v) for k, v in clean_df.MARRIAGE.value_counts().sort_index().items()},
        "recoded_education_rows": int((raw.EDUCATION != clean_df.EDUCATION).sum()),
        "recoded_marriage_rows": int((raw.MARRIAGE != clean_df.MARRIAGE).sum()),
        "negative_bill_counts": {c: int((raw[c] < 0).sum()) for c in raw if c.startswith("BILL_AMT")},
        "negative_repayment_status_counts": {c: int((raw[c] < 0).sum()) for c in raw if c.startswith("PAY_")},
        "decisions": ["Drop ID from modeling; retain row_mapping.csv for traceability.",
                      "EDUCATION 0/5/6 -> 4 (other); MARRIAGE 0 -> 3 (other).",
                      "Keep negative PAY_* statuses and negative BILL_AMT* values.",
                      "No rows removed, no missing-value imputation needed for supplied XLS.",
                      "Repeated predictor profiles may be distinct customers; retained and audited, with grouped-split sensitivity.",
                      "No encoding or fitting on validation/test; training-only encoding occurs in prepare."],
    }
    dump_json(output_dir / "audit.json", audit)
    print(f"Cleaned {len(clean_df)} rows: {cleaned_path}")
    print(f"Default rate: {clean_df.target.mean():.4%}; EDUCATION recodes: {audit['recoded_education_rows']}; "
          f"MARRIAGE recodes: {audit['recoded_marriage_rows']}")
    return cleaned_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local UCI .xls path")
    parser.add_argument("--output", required=True, help="New cleaned-data directory")
    args = parser.parse_args()
    clean(args.input, args.output)
