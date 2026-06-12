from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42

TARGET_COLUMN = "target_future_revenue"

CATEGORICAL_COLUMNS = [
    "channel",
    "campaign_id",
    "campaign_name",
    "campaign_type",
    "forecast_level",
]

EXCLUDE_COLUMNS = [
    "forecast_id",
    "mode",
    "cutoff_date",
    "forecast_start_date",
    "forecast_end_date",
    "target_future_revenue",
    "target_future_roas",
    "target_future_spend",
    "target_future_clicks",
    "target_future_impressions",
    "target_future_conversions",
]


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_COLUMN])

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].clip(lower=0)

    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    available_categorical = [
        column for column in CATEGORICAL_COLUMNS if column in df.columns
    ]

    excluded = set(EXCLUDE_COLUMNS)

    candidate_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    numeric_columns = []

    for column in candidate_columns:
        if column in available_categorical:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_columns.append(column)

    feature_columns = available_categorical + numeric_columns

    if not feature_columns:
        raise ValueError("No valid feature columns found for training.")

    return feature_columns, available_categorical, numeric_columns


def build_model(
    categorical_columns: list[str],
    numeric_columns: list[str],
    quantile: float,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_columns,
            ),
        ],
        remainder="drop",
    )

    regressor = GradientBoostingRegressor(
        loss="quantile",
        alpha=quantile,
        n_estimators=350,
        learning_rate=0.035,
        max_depth=3,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    return model


def train_quantile_models(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> dict:
    quantiles = {
        "revenue_q10": 0.10,
        "revenue_q50": 0.50,
        "revenue_q90": 0.90,
    }

    models = {}

    for model_name, quantile in quantiles.items():
        print(f"Training {model_name} model...")

        model = build_model(
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            quantile=quantile,
        )

        model.fit(X_train, y_train_log)
        models[model_name] = model

    return models


def predict_revenue_from_log_model(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    log_predictions = model.predict(X)
    revenue_predictions = np.expm1(log_predictions)
    revenue_predictions = np.clip(revenue_predictions, 0, None)
    return revenue_predictions


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    predictions = predict_revenue_from_log_model(model, X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def save_training_report(
    report_path: Path,
    metrics: dict,
    training_rows: int,
    test_rows: int,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# ForecastIQ AI - Training Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Dataset",
        "",
        f"Training rows: {training_rows}",
        f"Test rows: {test_rows}",
        f"Feature count: {len(feature_columns)}",
        "",
        "## Evaluation Metrics",
        "",
        f"MAE: {metrics['mae']:.4f}",
        f"RMSE: {metrics['rmse']:.4f}",
        f"R2 Score: {metrics['r2']:.4f}",
        "",
        "## Categorical Features",
        "",
    ]

    for column in categorical_columns:
        lines.append(f"- {column}")

    lines.extend(
        [
            "",
            "## Numeric Features",
            "",
        ]
    )

    for column in numeric_columns:
        lines.append(f"- {column}")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def train_model(
    training_features_path: Path,
    model_path: Path,
    report_path: Path,
) -> None:
    df = pd.read_csv(training_features_path)
    df = clean_training_data(df)

    feature_columns, categorical_columns, numeric_columns = get_feature_columns(df)

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log, y_train, y_test = train_test_split(
        X,
        y_log,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    models = train_quantile_models(
        X_train=X_train,
        y_train_log=y_train_log,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )

    metrics = evaluate_model(
        model=models["revenue_q50"],
        X_test=X_test,
        y_test=y_test,
    )

    artifact = {
        "models": models,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "target_column": TARGET_COLUMN,
        "forecast_windows": [30, 60, 90],
        "model_type": "GradientBoostingRegressor Quantile Forecasting",
        "target_transform": "log1p",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    save_training_report(
        report_path=report_path,
        metrics=metrics,
        training_rows=len(X_train),
        test_rows=len(X_test),
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )

    print("Model training completed successfully.")
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2 Score: {metrics['r2']:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Training report saved to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train probabilistic revenue forecasting models."
    )
    parser.add_argument(
        "--training-features",
        default="output/training_features.csv",
        help="Path to generated training features CSV.",
    )
    parser.add_argument(
        "--model-path",
        default="pickle/model.pkl",
        help="Where to save trained model artifact.",
    )
    parser.add_argument(
        "--report-path",
        default="output/training_report.md",
        help="Where to save training report.",
    )

    args = parser.parse_args()

    train_model(
        training_features_path=Path(args.training_features),
        model_path=Path(args.model_path),
        report_path=Path(args.report_path),
    )


if __name__ == "__main__":
    main()