from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    FEATURE_COLUMNS,
    ENGINEERED_COLUMNS,
    ALLOWED_TYPES,
    coerce_schema,
    add_engineered_features,
    rule_based_risk_score,
    generate_synthetic_dataset,
    find_target_column,
    make_preprocessor,
    build_feature_frame,
)
from src.train_model import prepare_xy, train_random_forest
from src.predict import predict_transaction, PredictionResult


class TestPreprocessingAndFeatureEngineering:
    def test_coerce_schema_normalization(self):
        raw = pd.DataFrame(
            {
                "oldbalanceOrig": ["1000.50"],
                "newbalanceOrg": ["500.00"],
                "type": ["TRANSFER"],
                "amount": ["500.50"],
                "oldbalanceDest": [0.0],
                "newbalanceDest": [500.50],
            }
        )
        coerced = coerce_schema(raw)

        assert "oldbalanceOrg" in coerced.columns
        assert "newbalanceOrig" in coerced.columns
        assert coerced["oldbalanceOrg"].dtype == np.float64
        assert coerced["amount"].iloc[0] == 500.50
        assert coerced["type"].iloc[0] == "TRANSFER"

    def test_add_engineered_features(self):
        raw = pd.DataFrame(
            [
                {
                    "type": "CASH_OUT",
                    "amount": 60000.0,
                    "oldbalanceOrg": 70000.0,
                    "newbalanceOrig": 10000.0,
                    "oldbalanceDest": 5000.0,
                    "newbalanceDest": 65000.0,
                }
            ]
        )
        featured = add_engineered_features(raw)

        for col in ENGINEERED_COLUMNS:
            assert col in featured.columns

        # Verify math
        assert featured["orig_delta"].iloc[0] == pytest.approx(60000.0)
        assert featured["dest_delta"].iloc[0] == pytest.approx(60000.0)
        assert featured["orig_amount_gap"].iloc[0] == pytest.approx(0.0)
        assert featured["large_amount"].iloc[0] == 1  # >= 50,000
        assert featured["risky_type"].iloc[0] == 1   # CASH_OUT

    def test_rule_based_risk_score(self):
        # High risk: Large transfer that completely drains sender account
        fraud_txn = {
            "type": "TRANSFER",
            "amount": 150000.0,
            "oldbalanceOrg": 150000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
        }
        fraud_score = rule_based_risk_score(fraud_txn)
        assert fraud_score >= 0.70

        # Low risk: Everyday small payment with normal remaining balance
        normal_txn = {
            "type": "PAYMENT",
            "amount": 25.0,
            "oldbalanceOrg": 5000.0,
            "newbalanceOrig": 4975.0,
            "oldbalanceDest": 1000.0,
            "newbalanceDest": 1025.0,
        }
        normal_score = rule_based_risk_score(normal_txn)
        assert normal_score <= 0.20

    def test_generate_synthetic_dataset(self):
        df = generate_synthetic_dataset(n=100, seed=42)
        assert len(df) == 100
        assert "isFraud" in df.columns
        assert set(df["type"].unique()).issubset(set(ALLOWED_TYPES))
        assert find_target_column(df) == "isFraud"

    def test_make_preprocessor_pipeline(self):
        preprocessor = make_preprocessor()
        df = generate_synthetic_dataset(n=50, seed=42)
        featured = add_engineered_features(df)

        feature_cols = FEATURE_COLUMNS + ENGINEERED_COLUMNS
        X_trans = preprocessor.fit_transform(featured[feature_cols])
        assert X_trans.shape[0] == 50
        assert X_trans.shape[1] > len(feature_cols)  # One-hot expansion


class TestModelTrainingAndInference:
    def test_train_random_forest_with_smote(self):
        # Generate balanced enough synthetic data to trigger SMOTE
        df = generate_synthetic_dataset(n=120, seed=42)
        # Ensure at least 6 positive instances for SMOTE k_neighbors=5
        df.loc[:8, "isFraud"] = 1
        df.loc[9:, "isFraud"] = 0

        X, y = prepare_xy(df)
        model = train_random_forest(X, y)

        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

        preds = model.predict(X[:5])
        assert len(preds) == 5
        assert set(preds).issubset({0, 1})

    def test_predict_transaction_result(self):
        df = generate_synthetic_dataset(n=120, seed=42)
        df.loc[:8, "isFraud"] = 1
        df.loc[9:, "isFraud"] = 0
        X, y = prepare_xy(df)
        model = train_random_forest(X, y)

        # High risk scenario
        res_fraud = predict_transaction(
            model,
            type="TRANSFER",
            amount=200000.0,
            oldbalanceOrg=200000.0,
            newbalanceOrig=0.0,
            oldbalanceDest=0.0,
            newbalanceDest=0.0,
        )
        assert isinstance(res_fraud, PredictionResult)
        assert res_fraud.probability > 0.50
        assert res_fraud.prediction == 1

        # Legitimate transaction scenario
        res_legit = predict_transaction(
            model,
            type="PAYMENT",
            amount=15.0,
            oldbalanceOrg=10000.0,
            newbalanceOrig=9985.0,
            oldbalanceDest=500.0,
            newbalanceDest=515.0,
        )
        assert isinstance(res_legit, PredictionResult)
        assert 0.0 <= res_legit.probability <= 1.0
