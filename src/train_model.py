from __future__ import annotations

from pathlib import Path
from typing import Tuple

import sys

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Allow running as `python src/train_model.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    FEATURE_COLUMNS,
    add_engineered_features,
    coerce_schema,
    find_target_column,
    load_dataset,
    make_preprocessor,
    project_paths,
)


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = add_engineered_features(df)
    target_col = find_target_column(df)

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    X = df[FEATURE_COLUMNS + [
        "orig_delta",
        "dest_delta",
        "orig_amount_gap",
        "dest_amount_gap",
        "empties_origin",
        "large_amount",
        "risky_type",
        "origin_depletion_ratio",
        "amount_to_org_balance",
    ]]
    return X, y


def train_random_forest(X: pd.DataFrame, y: pd.Series) -> ImbPipeline:
    """
    Train a RandomForest model with SMOTE handling imbalance.

    The returned object is an imblearn Pipeline:
      preprocessor -> SMOTE -> RandomForestClassifier
    """
    pre = make_preprocessor()
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
    )

    pipe = ImbPipeline(
        steps=[
            ("preprocess", pre),
            ("smote", SMOTE(random_state=42)),
            ("model", clf),
        ]
    )

    pipe.fit(X, y)
    return pipe


def main() -> None:
    paths = project_paths()
    df = load_dataset(paths.data_csv)
    X, y = prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    model = train_random_forest(X_train, y_train)

    # quick sanity metric on hold-out (not used to pick model here)
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, zero_division=0)
    print(f"Hold-out F1-score: {score:.4f}")

    paths.model_pkl.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, paths.model_pkl)
    print(f"Saved model to: {paths.model_pkl}")


if __name__ == "__main__":
    main()

