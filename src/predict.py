from __future__ import annotations

from dataclasses import dataclass

import joblib

import sys
from pathlib import Path

# Allow importing `src.*` when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    add_engineered_features,
    build_feature_frame,
    project_paths,
    rule_based_risk_score,
)


@dataclass(frozen=True)
class PredictionResult:
    prediction: int
    probability: float


def load_model():
    """
    Safely load the trained RandomForest pipeline bundle.
    If the model artifact is missing, automatically trains and serializes the model.
    """
    paths = project_paths()
    if not paths.model_pkl.exists():
        from src.train_model import main as train_main
        train_main()
    return joblib.load(paths.model_pkl)


def predict_transaction(model, *, type: str, amount: float, oldbalanceOrg: float, newbalanceOrig: float,
                        oldbalanceDest: float, newbalanceDest: float) -> PredictionResult:
    row = {
        "type": type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
    }
    X = add_engineered_features(build_feature_frame(row))
    model_proba = float(model.predict_proba(X)[0][1])
    rule_proba = rule_based_risk_score(row)
    # Blend learned signal with domain rules so clearly risky transactions are surfaced.
    proba = max(model_proba * 0.85 + rule_proba * 0.15, rule_proba if rule_proba >= 0.60 else 0.0, model_proba)
    pred = int(proba >= 0.4)
    return PredictionResult(prediction=pred, probability=proba)

