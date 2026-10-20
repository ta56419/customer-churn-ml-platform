"""Dataset validation and feature definitions."""

import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "support_tickets_90d"]
CATEGORICAL_FEATURES = ["contract", "payment_method"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churned"
CATEGORIES = {
    "contract": {"month_to_month", "one_year", "two_year"},
    "payment_method": {"electronic", "bank_transfer", "card"},
}


def validate_frame(frame: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    required = FEATURES + ([TARGET] if require_target else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    clean = frame[required].copy()
    if clean.isna().any().any():
        raise ValueError("Input data contains missing values")
    for column in NUMERIC_FEATURES:
        clean[column] = pd.to_numeric(clean[column], errors="raise")
        if not np.isfinite(clean[column]).all():
            raise ValueError(f"{column} must contain finite values")
    for column, allowed in CATEGORIES.items():
        unknown = set(clean[column].astype(str)) - allowed
        if unknown:
            raise ValueError(f"Unsupported {column} value(s): {', '.join(sorted(unknown))}")
    if require_target:
        values = set(clean[TARGET].unique())
        if not values <= {0, 1}:
            raise ValueError("churned must contain only 0 or 1")
        clean[TARGET] = clean[TARGET].astype(int)
    return clean
