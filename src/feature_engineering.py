from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.normalize_data import normalize_all_platforms


FORECAST_WINDOWS = [30, 60, 90]
ROLLING_WINDOWS = [7, 14, 30, 60, 90]


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return 0.0
    return float(numerator) / float(denominator)


def aggregate_period(df: pd.DataFrame, prefix: str) -> dict:
    spend = float(df["spend"].sum())
    revenue = float(df["revenue"].sum())
    clicks = float(df["clicks"].sum())
    impressions = float(df["impressions"].sum())
    conversions = float(df["conversions"].sum())
    daily_budget_mean = float(df["daily_budget"].mean()) if len(df) else 0.0

    return {
        f"{prefix}_rows": int(len(df)),
        f"{prefix}_active_days": int(df["date"].nunique()) if len(df) else 0,
        f"{prefix}_spend": spend,
        f"{prefix}_revenue": revenue,
        f"{prefix}_clicks": clicks,
        f"{prefix}_impressions": impressions,
        f"{prefix}_conversions": conversions,
        f"{prefix}_avg_daily_budget": daily_budget_mean,
        f"{prefix}_ctr": safe_divide(clicks, impressions),
        f"{prefix}_cpc": safe_divide(spend, clicks),
        f"{prefix}_cpm": safe_divide(spend, impressions) * 1000,
        f"{prefix}_conversion_rate": safe_divide(conversions, clicks),
        f"{prefix}_roas": safe_divide(revenue, spend),
    }


def get_period_slice(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    return df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()


def create_base_features(
    group_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    forecast_window_days: int,
    mode: str,
) -> dict:
    group_df = group_df.sort_values("date").copy()

    latest_row = group_df[group_df["date"] <= cutoff_date].tail(1)

    if latest_row.empty:
        raise ValueError("No historical data found before cutoff date.")

    latest_row = latest_row.iloc[0]

    campaign_start_date = group_df["date"].min()
    forecast_start_date = cutoff_date + pd.Timedelta(days=1)
    forecast_end_date = cutoff_date + pd.Timedelta(days=forecast_window_days)

    features = {
        "forecast_id": (
            f"{mode}_"
            f"{latest_row['channel']}_"
            f"{latest_row['campaign_id']}_"
            f"{forecast_window_days}_"
            f"{cutoff_date.date()}"
        ),
        "mode": mode,
        "forecast_level": "campaign",
        "channel": latest_row["channel"],
        "campaign_id": str(latest_row["campaign_id"]),
        "campaign_name": latest_row["campaign_name"],
        "campaign_type": latest_row["campaign_type"],
        "cutoff_date": cutoff_date,
        "forecast_start_date": forecast_start_date,
        "forecast_end_date": forecast_end_date,
        "forecast_window_days": forecast_window_days,
        "campaign_age_days": int((cutoff_date - campaign_start_date).days + 1),
        "forecast_month": int(forecast_start_date.month),
        "forecast_quarter": int(forecast_start_date.quarter),
        "forecast_year": int(forecast_start_date.year),
    }

    for window in ROLLING_WINDOWS:
        start_date = cutoff_date - pd.Timedelta(days=window - 1)
        period_df = get_period_slice(group_df, start_date, cutoff_date)
        features.update(aggregate_period(period_df, f"hist_{window}d"))

    features["spend_trend_7d_vs_30d"] = safe_divide(
        features["hist_7d_spend"] / 7,
        features["hist_30d_spend"] / 30,
    )

    features["revenue_trend_7d_vs_30d"] = safe_divide(
        features["hist_7d_revenue"] / 7,
        features["hist_30d_revenue"] / 30,
    )

    features["roas_trend_7d_vs_30d"] = safe_divide(
        features["hist_7d_roas"],
        features["hist_30d_roas"],
    )

    return features


def estimate_future_budget(
    features: dict,
    forecast_window_days: int,
) -> float:
    recent_avg_daily_budget = features.get("hist_30d_avg_daily_budget", 0.0)
    recent_avg_daily_spend = safe_divide(
        features.get("hist_30d_spend", 0.0),
        max(features.get("hist_30d_active_days", 0), 1),
    )

    if recent_avg_daily_budget > 0:
        return float(recent_avg_daily_budget * forecast_window_days)

    return float(recent_avg_daily_spend * forecast_window_days)


def create_training_features_for_group(
    group_df: pd.DataFrame,
    min_history_days: int = 30,
    step_days: int = 7,
) -> list[dict]:
    group_df = group_df.sort_values("date").copy()

    rows: list[dict] = []

    group_start = group_df["date"].min()
    group_end = group_df["date"].max()

    first_cutoff = group_start + pd.Timedelta(days=min_history_days - 1)
    last_possible_cutoff = group_end - pd.Timedelta(days=min(FORECAST_WINDOWS))

    cutoff_date = first_cutoff

    while cutoff_date <= last_possible_cutoff:
        for forecast_window_days in FORECAST_WINDOWS:
            forecast_start = cutoff_date + pd.Timedelta(days=1)
            forecast_end = cutoff_date + pd.Timedelta(days=forecast_window_days)

            if forecast_end > group_end:
                continue

            future_df = get_period_slice(group_df, forecast_start, forecast_end)

            if future_df.empty:
                continue

            features = create_base_features(
                group_df=group_df,
                cutoff_date=cutoff_date,
                forecast_window_days=forecast_window_days,
                mode="train",
            )

            future_budget = float(future_df["daily_budget"].mean() * forecast_window_days)
            future_spend = float(future_df["spend"].sum())
            future_revenue = float(future_df["revenue"].sum())
            future_clicks = float(future_df["clicks"].sum())
            future_impressions = float(future_df["impressions"].sum())
            future_conversions = float(future_df["conversions"].sum())

            if future_budget <= 0:
                future_budget = future_spend

            features.update(
                {
                    "future_budget": future_budget,
                    "future_budget_per_day": safe_divide(
                        future_budget,
                        forecast_window_days,
                    ),
                    "target_future_spend": future_spend,
                    "target_future_revenue": future_revenue,
                    "target_future_clicks": future_clicks,
                    "target_future_impressions": future_impressions,
                    "target_future_conversions": future_conversions,
                    "target_future_roas": safe_divide(future_revenue, future_spend),
                }
            )

            rows.append(features)

        cutoff_date += pd.Timedelta(days=step_days)

    return rows


def create_prediction_features_for_group(group_df: pd.DataFrame) -> list[dict]:
    group_df = group_df.sort_values("date").copy()

    rows: list[dict] = []

    cutoff_date = group_df["date"].max()

    for forecast_window_days in FORECAST_WINDOWS:
        features = create_base_features(
            group_df=group_df,
            cutoff_date=cutoff_date,
            forecast_window_days=forecast_window_days,
            mode="predict",
        )

        future_budget = estimate_future_budget(features, forecast_window_days)

        features.update(
            {
                "future_budget": future_budget,
                "future_budget_per_day": safe_divide(
                    future_budget,
                    forecast_window_days,
                ),
            }
        )

        rows.append(features)

    return rows


def create_training_dataset(unified_df: pd.DataFrame) -> pd.DataFrame:
    all_rows: list[dict] = []

    grouped = unified_df.groupby(
        ["channel", "campaign_id", "campaign_name", "campaign_type"],
        dropna=False,
    )

    for _, group_df in grouped:
        group_rows = create_training_features_for_group(group_df)
        all_rows.extend(group_rows)

    training_df = pd.DataFrame(all_rows)

    if training_df.empty:
        raise ValueError(
            "Training feature dataset is empty. "
            "Check if campaigns have enough date history for 30/60/90 day forecasting."
        )

    return training_df.sort_values(
        ["cutoff_date", "channel", "campaign_id", "forecast_window_days"]
    ).reset_index(drop=True)


def create_prediction_dataset(unified_df: pd.DataFrame) -> pd.DataFrame:
    all_rows: list[dict] = []

    grouped = unified_df.groupby(
        ["channel", "campaign_id", "campaign_name", "campaign_type"],
        dropna=False,
    )

    for _, group_df in grouped:
        group_rows = create_prediction_features_for_group(group_df)
        all_rows.extend(group_rows)

    prediction_df = pd.DataFrame(all_rows)

    if prediction_df.empty:
        raise ValueError("Prediction feature dataset is empty.")

    return prediction_df.sort_values(
        ["channel", "campaign_id", "forecast_window_days"]
    ).reset_index(drop=True)


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ML-ready training and prediction features."
    )
    parser.add_argument("--data-dir", default="data", help="Folder containing input CSV files.")
    parser.add_argument(
        "--training-output",
        default="output/training_features.csv",
        help="Where to save training features.",
    )
    parser.add_argument(
        "--prediction-output",
        default="output/prediction_features.csv",
        help="Where to save prediction features.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    training_output = Path(args.training_output)
    prediction_output = Path(args.prediction_output)

    unified_df = normalize_all_platforms(data_dir)
    unified_df["date"] = pd.to_datetime(unified_df["date"], errors="coerce")
    unified_df = unified_df.dropna(subset=["date"])

    training_df = create_training_dataset(unified_df)
    prediction_df = create_prediction_dataset(unified_df)

    save_dataframe(training_df, training_output)
    save_dataframe(prediction_df, prediction_output)

    print("Feature engineering completed successfully.")
    print(f"Training rows: {len(training_df):,}")
    print(f"Prediction rows: {len(prediction_df):,}")
    print(f"Training features saved to: {training_output}")
    print(f"Prediction features saved to: {prediction_output}")


if __name__ == "__main__":
    main()