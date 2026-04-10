from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

TARGET_COLUMN_CANDIDATES = ["isFraud", "Class", "fraud", "target"]

ALLOWED_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]

ENGINEERED_COLUMNS = [
    "orig_delta",
    "dest_delta",
    "orig_amount_gap",
    "dest_amount_gap",
    "empties_origin",
    "large_amount",
    "risky_type",
    "origin_depletion_ratio",
    "amount_to_org_balance",
]


@dataclass(frozen=True)
class DatasetPaths:
    root: Path

    @property
    def data_csv(self) -> Path:
        return self.root / "data" / "fraud.csv"

    @property
    def model_pkl(self) -> Path:
        return self.root / "models" / "fraud_model.pkl"


def project_paths() -> DatasetPaths:
    # src/ is always one level under project root
    root = Path(__file__).resolve().parents[1]
    return DatasetPaths(root=root)


def make_preprocessor() -> ColumnTransformer:
    """
    Build a preprocessing transformer that is shared across training/prediction.

    - Encodes 'type' with one-hot encoding (unknown types ignored safely)
    - Imputes missing numeric values with median
    - Imputes missing type with 'UNKNOWN'
    """
    categorical_features = ["type"]
    numeric_features = [c for c in FEATURE_COLUMNS if c != "type"] + ENGINEERED_COLUMNS

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_features),
            ("num", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe has the required feature columns with correct names/dtypes.
    Extra columns are ignored by training code; missing columns are added as NaN.
    """
    df = df.copy()

    # Normalize common column name variants
    rename_map = {
        "oldbalanceOrig": "oldbalanceOrg",
        "newbalanceOrg": "newbalanceOrig",
        "newbalanceOrg ": "newbalanceOrig",
        "oldbalanceDest ": "oldbalanceDest",
        "newbalanceDest ": "newbalanceDest",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Keep feature order stable
    df = df[FEATURE_COLUMNS + [c for c in df.columns if c not in FEATURE_COLUMNS]]

    # Types
    df["type"] = df["type"].astype("string")
    for col in FEATURE_COLUMNS:
        if col != "type":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def find_target_column(df: pd.DataFrame) -> str:
    for c in TARGET_COLUMN_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find target column. Expected one of {TARGET_COLUMN_CANDIDATES}, got {list(df.columns)}"
    )


def generate_synthetic_dataset(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a small synthetic dataset shaped like an "online payment fraud" dataset.
    This is ONLY used when data/fraud.csv is missing.
    """
    rng = np.random.default_rng(seed)

    types = rng.choice(ALLOWED_TYPES, size=n, p=[0.55, 0.18, 0.22, 0.05])
    amount = rng.lognormal(mean=7.5, sigma=1.0, size=n) / 10.0  # ~ tens to thousands

    old_org = rng.lognormal(mean=8.3, sigma=1.0, size=n) / 10.0
    old_dest = rng.lognormal(mean=8.0, sigma=1.0, size=n) / 10.0

    # A simple balance update that mostly makes sense (with some noise)
    delta_org = np.minimum(amount, old_org) * (0.9 + 0.2 * rng.random(n))
    new_org = np.maximum(0.0, old_org - delta_org)
    new_dest = old_dest + delta_org

    df = pd.DataFrame(
        {
            "type": types,
            "amount": amount,
            "oldbalanceOrg": old_org,
            "newbalanceOrig": new_org,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
        }
    )

    # Fraud pattern heuristic aligned with common online payment fraud behavior:
    # large risky transfers, near-empty sender accounts, and balance mismatches.
    emptying = (df["oldbalanceOrg"] > 1000) & (df["newbalanceOrig"] < 100)
    large = df["amount"] > df["amount"].quantile(0.85)
    very_large = df["amount"] > df["amount"].quantile(0.95)
    risky_type = df["type"].isin(["TRANSFER", "CASH_OUT"])
    orig_gap = np.abs((df["oldbalanceOrg"] - df["newbalanceOrig"]) - df["amount"])
    dest_gap = np.abs((df["newbalanceDest"] - df["oldbalanceDest"]) - df["amount"])
    mismatch = (orig_gap > 500) | (dest_gap > 500)

    p = (
        0.005
        + 0.10 * (risky_type & large).astype(float)
        + 0.18 * (risky_type & very_large).astype(float)
        + 0.20 * emptying.astype(float)
        + 0.18 * mismatch.astype(float)
    )
    p = np.clip(p, 0.005, 0.92)
    df["isFraud"] = (rng.random(n) < p).astype(int)

    # Add a tiny amount of missingness
    for col in ["oldbalanceDest", "newbalanceDest"]:
        mask = rng.random(n) < 0.01
        df.loc[mask, col] = np.nan

    return df


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    df = generate_synthetic_dataset()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def build_feature_frame(row: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame with stable feature order.
    """
    df = pd.DataFrame([row])
    df = coerce_schema(df)
    return df[FEATURE_COLUMNS]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that capture suspicious balance patterns.
    This logic is shared by both training and prediction paths.
    """
    df = coerce_schema(df).copy()

    df["orig_delta"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["dest_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["orig_amount_gap"] = (df["orig_delta"] - df["amount"]).abs()
    df["dest_amount_gap"] = (df["dest_delta"] - df["amount"]).abs()
    df["empties_origin"] = (
        (df["oldbalanceOrg"] > 1000) & (df["newbalanceOrig"] < 100)
    ).astype(int)
    df["large_amount"] = (df["amount"] >= 50000).astype(int)
    df["risky_type"] = df["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)
    safe_old = df["oldbalanceOrg"].replace(0, np.nan)
    df["origin_depletion_ratio"] = (1 - (df["newbalanceOrig"] / safe_old)).clip(lower=0).fillna(0)
    df["amount_to_org_balance"] = (df["amount"] / safe_old).replace([np.inf, -np.inf], np.nan).fillna(0)

    for col in ENGINEERED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def rule_based_risk_score(row: dict) -> float:
    """
    Compute a domain-informed fraud risk score for online payments.
    This complements the ML model when training data is synthetic or sparse.
    """
    amount = float(row.get("amount", 0) or 0)
    old_org = float(row.get("oldbalanceOrg", 0) or 0)
    new_org = float(row.get("newbalanceOrig", 0) or 0)
    old_dest = float(row.get("oldbalanceDest", 0) or 0)
    new_dest = float(row.get("newbalanceDest", 0) or 0)
    txn_type = str(row.get("type", "") or "").upper()

    risky_type = txn_type in {"TRANSFER", "CASH_OUT"}
    depletion_ratio = 1.0 - (new_org / old_org) if old_org > 0 else 0.0
    amount_to_org = amount / old_org if old_org > 0 else 0.0
    orig_gap = abs((old_org - new_org) - amount)
    dest_gap = abs((new_dest - old_dest) - amount)

    score = 0.02
    if risky_type:
        score += 0.18
    if amount >= 50000:
        score += 0.18
    if amount >= 100000:
        score += 0.10
    if depletion_ratio >= 0.90:
        score += 0.32
    elif depletion_ratio >= 0.70:
        score += 0.18
    if amount_to_org >= 0.90:
        score += 0.18
    elif amount_to_org >= 0.70:
        score += 0.10
    if orig_gap > 1000:
        score += 0.12
    if dest_gap > 1000:
        score += 0.08
    if txn_type == "PAYMENT" and amount < 10000:
        score -= 0.10

    return float(np.clip(score, 0.01, 0.98))

