"""FastAPI inference endpoint. Set CHURN_MODEL to the trained artifact path."""

import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .features import FEATURES, validate_frame

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")
MODEL_PATH = Path(os.getenv("CHURN_MODEL", "artifacts/churn.joblib"))


class Customer(BaseModel):
    tenure_months: float = Field(ge=0, le=600)
    monthly_charges: float = Field(ge=0)
    support_tickets_90d: float = Field(ge=0)
    contract: str
    payment_method: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.is_file()}


@app.post("/predict")
def predict(customer: Customer) -> dict:
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="Model artifact is missing; train the model first")
    try:
        row = validate_frame(pd.DataFrame([customer.model_dump()]), require_target=False)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    bundle = joblib.load(MODEL_PATH)
    probability = float(bundle["model"].predict_proba(row[FEATURES])[0, 1])
    threshold = float(bundle.get("threshold", 0.5))
    return {"churn_probability": round(probability, 4), "threshold": threshold,
            "flag_for_review": probability >= threshold}
