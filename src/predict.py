from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import create_prediction_dataset
from src.normalize_data import normalize_all_platforms


OUTPUT_COLUMNS = [
    "forecast_id",
    "forecast_level",
    "channel",
    "campaign_type",
    "campaign_id",
    "campaign_name",
    "forecast_window_days",
    "forecast_start_date",
    "forecast_end_date",
    "future_budget",
    "revenue_low",
    "revenue_expected",
    "revenue_high",
    "roas_low",
    "roas_expected",
    "roas_high",
    "confidence_score",
    "risk_level",
    "insight_summary",
]


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return 0.0
    return float(numerator) / float(denominator)


def prepare_features_for_model(
    prediction_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    df = prediction_df.copy()

    for column in feature_columns:
        if column not in df.columns:
            df[column] = 0

    X = df[feature_columns].copy()

    return X.replace([np.inf, -np.inf], np.nan).fillna(0)


def predict_revenue(model, X: pd.DataFrame) -> np.ndarray:
    log_predictions = model.predict(X)
    predictions = np.expm1(log_predictions)
    predictions = np.clip(predictions, 0, None)
    return predictions


def enforce_prediction_order(
    low: np.ndarray,
    expected: np.ndarray,
    high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked = np.vstack([low, expected, high]).T
    sorted_predictions = np.sort(stacked, axis=1)

    return (
        sorted_predictions[:, 0],
        sorted_predictions[:, 1],
        sorted_predictions[:, 2],
    )


def calculate_confidence_score(row: pd.Series) -> float:
    expected = float(row["revenue_expected"])

    if expected <= 0:
        return 40.0

    uncertainty_width = float(row["revenue_high"] - row["revenue_low"])
    relative_uncertainty = uncertainty_width / expected

    confidence = 100 - min(relative_uncertainty * 35, 60)

    if row["future_budget"] <= 0:
        confidence -= 15

    if row["hist_30d_rows"] < 10:
        confidence -= 10

    return float(max(30, min(95, confidence)))


def calculate_risk_level(row: pd.Series) -> str:
    roas_expected = float(row["roas_expected"])
    confidence = float(row["confidence_score"])
    spend_trend = float(row.get("spend_trend_7d_vs_30d", 1))
    revenue_trend = float(row.get("revenue_trend_7d_vs_30d", 1))

    if roas_expected < 1 or confidence < 45:
        return "High"

    if spend_trend > 1.3 and revenue_trend < 0.9:
        return "High"

    if roas_expected < 2 or confidence < 65:
        return "Medium"

    return "Low"


def generate_insight_summary(row: pd.Series) -> str:
    level = row["forecast_level"]
    channel = row["channel"]
    campaign_type = row["campaign_type"]
    campaign_name = row["campaign_name"]
    window = int(row["forecast_window_days"])
    roas = float(row["roas_expected"])
    risk = row["risk_level"]

    spend_trend = float(row.get("spend_trend_7d_vs_30d", 1))
    revenue_trend = float(row.get("revenue_trend_7d_vs_30d", 1))

    subject = channel

    if level == "campaign":
        subject = f"{channel} campaign '{campaign_name}'"
    elif level == "campaign_type":
        subject = f"{channel} {campaign_type}"
    elif level == "blended":
        subject = "overall blended marketing performance"

    trend_message = "recent spend and revenue trends are relatively stable"

    if spend_trend > 1.2 and revenue_trend < 1:
        trend_message = "recent spend is increasing faster than revenue, which may reduce efficiency"
    elif revenue_trend > 1.2 and spend_trend <= 1.2:
        trend_message = "recent revenue is improving faster than spend, which indicates positive momentum"
    elif revenue_trend < 0.8:
        trend_message = "recent revenue momentum is weak compared with the previous 30-day baseline"

    return (
        f"For the next {window} days, {subject} is forecasted with expected ROAS "
        f"around {roas:.2f}. Risk level is {risk}. The main reason is that "
        f"{trend_message}."
    )


def add_prediction_columns(
    prediction_df: pd.DataFrame,
    artifact: dict,
) -> pd.DataFrame:
    models = artifact["models"]
    feature_columns = artifact["feature_columns"]

    X = prepare_features_for_model(prediction_df, feature_columns)

    revenue_low = predict_revenue(models["revenue_q10"], X)
    revenue_expected = predict_revenue(models["revenue_q50"], X)
    revenue_high = predict_revenue(models["revenue_q90"], X)

    revenue_low, revenue_expected, revenue_high = enforce_prediction_order(
        revenue_low,
        revenue_expected,
        revenue_high,
    )

    result_df = prediction_df.copy()

    result_df["revenue_low"] = revenue_low
    result_df["revenue_expected"] = revenue_expected
    result_df["revenue_high"] = revenue_high

    result_df["roas_low"] = result_df.apply(
        lambda row: safe_divide(row["revenue_low"], row["future_budget"]),
        axis=1,
    )

    result_df["roas_expected"] = result_df.apply(
        lambda row: safe_divide(row["revenue_expected"], row["future_budget"]),
        axis=1,
    )

    result_df["roas_high"] = result_df.apply(
        lambda row: safe_divide(row["revenue_high"], row["future_budget"]),
        axis=1,
    )

    result_df["confidence_score"] = result_df.apply(calculate_confidence_score, axis=1)
    result_df["risk_level"] = result_df.apply(calculate_risk_level, axis=1)
    result_df["insight_summary"] = result_df.apply(generate_insight_summary, axis=1)

    result_df["forecast_level"] = "campaign"

    return result_df


def aggregate_forecasts(
    campaign_forecasts: pd.DataFrame,
    group_columns: list[str],
    forecast_level: str,
) -> pd.DataFrame:
    grouped = campaign_forecasts.groupby(
        group_columns + ["forecast_window_days"],
        dropna=False,
    )

    rows = []

    for keys, group_df in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {}

        for column, value in zip(group_columns + ["forecast_window_days"], keys):
            row[column] = value

        row["forecast_level"] = forecast_level
        row["future_budget"] = float(group_df["future_budget"].sum())
        row["revenue_low"] = float(group_df["revenue_low"].sum())
        row["revenue_expected"] = float(group_df["revenue_expected"].sum())
        row["revenue_high"] = float(group_df["revenue_high"].sum())

        row["roas_low"] = safe_divide(row["revenue_low"], row["future_budget"])
        row["roas_expected"] = safe_divide(row["revenue_expected"], row["future_budget"])
        row["roas_high"] = safe_divide(row["revenue_high"], row["future_budget"])

        row["confidence_score"] = float(group_df["confidence_score"].mean())

        high_risk_count = int((group_df["risk_level"] == "High").sum())
        medium_risk_count = int((group_df["risk_level"] == "Medium").sum())

        if high_risk_count > 0:
            row["risk_level"] = "High"
        elif medium_risk_count > 0:
            row["risk_level"] = "Medium"
        else:
            row["risk_level"] = "Low"

        first_row = group_df.iloc[0]
        row["forecast_start_date"] = first_row["forecast_start_date"]
        row["forecast_end_date"] = first_row["forecast_end_date"]

        row["campaign_id"] = "ALL"
        row["campaign_name"] = "ALL"

        if "channel" not in row:
            row["channel"] = "ALL"

        if "campaign_type" not in row:
            row["campaign_type"] = "ALL"

        row["forecast_id"] = (
            f"{forecast_level}_"
            f"{row['channel']}_"
            f"{row['campaign_type']}_"
            f"{row['forecast_window_days']}"
        )

        rows.append(row)

    agg_df = pd.DataFrame(rows)

    if agg_df.empty:
        return agg_df

    agg_df["insight_summary"] = agg_df.apply(generate_insight_summary, axis=1)

    return agg_df


def build_final_forecast_output(campaign_forecasts: pd.DataFrame) -> pd.DataFrame:
    campaign_level = campaign_forecasts.copy()

    channel_level = aggregate_forecasts(
        campaign_forecasts=campaign_forecasts,
        group_columns=["channel"],
        forecast_level="channel",
    )

    campaign_type_level = aggregate_forecasts(
        campaign_forecasts=campaign_forecasts,
        group_columns=["channel", "campaign_type"],
        forecast_level="campaign_type",
    )

    blended_level = aggregate_forecasts(
        campaign_forecasts=campaign_forecasts,
        group_columns=[],
        forecast_level="blended",
    )

    final_df = pd.concat(
        [
            blended_level,
            channel_level,
            campaign_type_level,
            campaign_level,
        ],
        ignore_index=True,
    )

    for column in OUTPUT_COLUMNS:
        if column not in final_df.columns:
            final_df[column] = ""

    final_df = final_df[OUTPUT_COLUMNS].copy()

    numeric_columns = [
        "future_budget",
        "revenue_low",
        "revenue_expected",
        "revenue_high",
        "roas_low",
        "roas_expected",
        "roas_high",
        "confidence_score",
    ]

    for column in numeric_columns:
        final_df[column] = pd.to_numeric(final_df[column], errors="coerce").fillna(0)
        final_df[column] = final_df[column].round(4)

    return final_df.sort_values(
        ["forecast_window_days", "forecast_level", "channel", "campaign_type", "campaign_id"]
    ).reset_index(drop=True)


def generate_predictions(
    data_dir: Path,
    model_path: Path,
    output_path: Path,
    prediction_features_output: Path | None = None,
) -> pd.DataFrame:
    artifact = joblib.load(model_path)

    unified_df = normalize_all_platforms(data_dir)
    unified_df["date"] = pd.to_datetime(unified_df["date"], errors="coerce")
    unified_df = unified_df.dropna(subset=["date"])

    prediction_df = create_prediction_dataset(unified_df)

    if prediction_features_output is not None:
        prediction_features_output.parent.mkdir(parents=True, exist_ok=True)
        prediction_df.to_csv(prediction_features_output, index=False)

    campaign_forecasts = add_prediction_columns(prediction_df, artifact)
    final_df = build_final_forecast_output(campaign_forecasts)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    return final_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate probabilistic revenue and ROAS forecasts."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Folder containing input CSV files.",
    )
    parser.add_argument(
        "--model",
        default="pickle/model.pkl",
        help="Path to trained model artifact.",
    )
    parser.add_argument(
        "--output",
        default="output/predictions.csv",
        help="Where to save prediction output CSV.",
    )
    parser.add_argument(
        "--prediction-features-output",
        default="output/prediction_features.csv",
        help="Optional path to save generated prediction features.",
    )

    args = parser.parse_args()

    final_df = generate_predictions(
        data_dir=Path(args.data_dir),
        model_path=Path(args.model),
        output_path=Path(args.output),
        prediction_features_output=Path(args.prediction_features_output)
        if args.prediction_features_output
        else None,
    )

    print("Prediction completed successfully.")
    print(f"Forecast rows: {len(final_df):,}")
    print(f"Saved predictions to: {args.output}")
    print("")
    print("Forecast levels:")
    print(final_df["forecast_level"].value_counts().to_string())


if __name__ == "__main__":
    main()