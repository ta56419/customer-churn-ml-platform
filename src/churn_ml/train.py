"""Train, evaluate, and persist a churn classifier."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .features import FEATURES, TARGET, validate_frame
from .pipeline import build_pipeline


def train(data_path: Path, model_path: Path, threshold: float = 0.5) -> dict:
    frame = validate_frame(pd.read_csv(data_path))
    if frame[TARGET].nunique() != 2 or frame[TARGET].value_counts().min() < 2:
        raise ValueError("Training requires at least two examples of each class")
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame[TARGET], test_size=0.25, random_state=17, stratify=frame[TARGET]
    )
    model = build_pipeline().fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "rows": len(frame), "holdout_rows": len(y_test), "threshold": threshold,
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold}, model_path)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/churn.joblib"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    print(json.dumps(train(args.data, args.model, args.threshold), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
