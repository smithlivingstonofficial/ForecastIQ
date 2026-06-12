from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.feature_engineering import create_prediction_dataset
from src.normalize_data import normalize_all_platforms
from src.predict import add_prediction_columns, build_final_forecast_output


DEFAULT_CHANNEL_MULTIPLIERS = {
    "Google Ads": 1.0,
    "Meta Ads": 1.0,
    "Microsoft Ads": 1.0,
}


COMPARISON_KEYS = [
    "forecast_level",
    "channel",
    "campaign_type",
    "campaign_id",
    "campaign_name",
    "forecast_window_days",
]


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return 0.0
    return float(numerator) / float(denominator)


def build_channel_multipliers(
    google_multiplier: float,
    meta_multiplier: float,
    microsoft_multiplier: float,
) -> dict[str, float]:
    return {
        "Google Ads": google_multiplier,
        "Meta Ads": meta_multiplier,
        "Microsoft Ads": microsoft_multiplier,
    }


def apply_budget_scenario(
    prediction_df: pd.DataFrame,
    channel_multipliers: dict[str, float],
) -> pd.DataFrame:
    scenario_df = prediction_df.copy()

    scenario_df["budget_multiplier"] = scenario_df["channel"].map(
        channel_multipliers
    ).fillna(1.0)

    scenario_df["future_budget"] = (
        scenario_df["future_budget"] * scenario_df["budget_multiplier"]
    )

    scenario_df["future_budget_per_day"] = scenario_df.apply(
        lambda row: safe_divide(
            row["future_budget"],
            row["forecast_window_days"],
        ),
        axis=1,
    )

    return scenario_df


def create_scenario_name(channel_multipliers: dict[str, float]) -> str:
    parts = []

    for channel, multiplier in channel_multipliers.items():
        percentage_change = (multiplier - 1) * 100
        sign = "+" if percentage_change >= 0 else ""
        parts.append(f"{channel}: {sign}{percentage_change:.0f}%")

    return " | ".join(parts)


def generate_forecast_from_features(
    prediction_df: pd.DataFrame,
    artifact: dict,
) -> pd.DataFrame:
    campaign_forecasts = add_prediction_columns(prediction_df, artifact)
    final_forecast = build_final_forecast_output(campaign_forecasts)

    return final_forecast


def compare_baseline_and_scenario(
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
) -> pd.DataFrame:
    baseline_columns = COMPARISON_KEYS + [
        "future_budget",
        "revenue_expected",
        "revenue_low",
        "revenue_high",
        "roas_expected",
        "confidence_score",
        "risk_level",
    ]

    scenario_columns = COMPARISON_KEYS + [
        "future_budget",
        "revenue_expected",
        "revenue_low",
        "revenue_high",
        "roas_expected",
        "confidence_score",
        "risk_level",
    ]

    baseline = baseline_df[baseline_columns].copy()
    scenario = scenario_df[scenario_columns].copy()

    baseline = baseline.rename(
        columns={
            "future_budget": "baseline_budget",
            "revenue_expected": "baseline_revenue_expected",
            "revenue_low": "baseline_revenue_low",
            "revenue_high": "baseline_revenue_high",
            "roas_expected": "baseline_roas_expected",
            "confidence_score": "baseline_confidence_score",
            "risk_level": "baseline_risk_level",
        }
    )

    scenario = scenario.rename(
        columns={
            "future_budget": "scenario_budget",
            "revenue_expected": "scenario_revenue_expected",
            "revenue_low": "scenario_revenue_low",
            "revenue_high": "scenario_revenue_high",
            "roas_expected": "scenario_roas_expected",
            "confidence_score": "scenario_confidence_score",
            "risk_level": "scenario_risk_level",
        }
    )

    comparison = baseline.merge(
        scenario,
        on=COMPARISON_KEYS,
        how="inner",
    )

    comparison["budget_change"] = (
        comparison["scenario_budget"] - comparison["baseline_budget"]
    )

    comparison["budget_change_percent"] = comparison.apply(
        lambda row: safe_divide(row["budget_change"], row["baseline_budget"]) * 100,
        axis=1,
    )

    comparison["revenue_change"] = (
        comparison["scenario_revenue_expected"]
        - comparison["baseline_revenue_expected"]
    )

    comparison["revenue_change_percent"] = comparison.apply(
        lambda row: safe_divide(
            row["revenue_change"],
            row["baseline_revenue_expected"],
        )
        * 100,
        axis=1,
    )

    comparison["roas_change"] = (
        comparison["scenario_roas_expected"]
        - comparison["baseline_roas_expected"]
    )

    comparison["scenario_recommendation"] = comparison.apply(
        generate_scenario_recommendation,
        axis=1,
    )

    numeric_columns = [
        "baseline_budget",
        "scenario_budget",
        "budget_change",
        "budget_change_percent",
        "baseline_revenue_expected",
        "scenario_revenue_expected",
        "revenue_change",
        "revenue_change_percent",
        "baseline_roas_expected",
        "scenario_roas_expected",
        "roas_change",
        "baseline_confidence_score",
        "scenario_confidence_score",
    ]

    for column in numeric_columns:
        comparison[column] = pd.to_numeric(
            comparison[column],
            errors="coerce",
        ).fillna(0)
        comparison[column] = comparison[column].round(4)

    return comparison.sort_values(
        [
            "forecast_window_days",
            "forecast_level",
            "channel",
            "campaign_type",
            "campaign_id",
        ]
    ).reset_index(drop=True)


def generate_scenario_recommendation(row: pd.Series) -> str:
    level = row["forecast_level"]
    channel = row["channel"]
    campaign_type = row["campaign_type"]
    campaign_name = row["campaign_name"]
    window = int(row["forecast_window_days"])

    revenue_change = float(row["revenue_change"])
    revenue_change_percent = float(row["revenue_change_percent"])
    roas_change = float(row["roas_change"])
    scenario_roas = float(row["scenario_roas_expected"])
    risk = row["scenario_risk_level"]

    if level == "blended":
        subject = "overall marketing performance"
    elif level == "channel":
        subject = f"{channel}"
    elif level == "campaign_type":
        subject = f"{channel} {campaign_type}"
    else:
        subject = f"{channel} campaign '{campaign_name}'"

    if revenue_change > 0 and roas_change >= 0:
        action = "This scenario looks positive because revenue is expected to improve without reducing ROAS."
    elif revenue_change > 0 and roas_change < 0:
        action = "This scenario may increase revenue, but efficiency may reduce because ROAS is expected to decline."
    elif revenue_change <= 0 and roas_change > 0:
        action = "This scenario may improve efficiency, but revenue growth may be limited."
    else:
        action = "This scenario is not attractive because both revenue and efficiency may weaken."

    return (
        f"For the next {window} days, {subject} is expected to change revenue by "
        f"{revenue_change:.2f} ({revenue_change_percent:.2f}%). "
        f"Expected scenario ROAS is {scenario_roas:.2f}. Risk level is {risk}. "
        f"{action}"
    )


def save_scenario_summary(
    comparison_df: pd.DataFrame,
    scenario_name: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blended = comparison_df[
        comparison_df["forecast_level"] == "blended"
    ].copy()

    lines = [
        "# ForecastIQ AI - Budget Scenario Summary",
        "",
        f"Scenario: {scenario_name}",
        "",
        "## Blended Forecast Impact",
        "",
    ]

    if blended.empty:
        lines.append("No blended forecast rows available.")
    else:
        for _, row in blended.iterrows():
            lines.extend(
                [
                    f"### {int(row['forecast_window_days'])}-Day Forecast",
                    "",
                    f"- Baseline budget: {row['baseline_budget']:.2f}",
                    f"- Scenario budget: {row['scenario_budget']:.2f}",
                    f"- Budget change: {row['budget_change']:.2f} ({row['budget_change_percent']:.2f}%)",
                    f"- Baseline expected revenue: {row['baseline_revenue_expected']:.2f}",
                    f"- Scenario expected revenue: {row['scenario_revenue_expected']:.2f}",
                    f"- Revenue change: {row['revenue_change']:.2f} ({row['revenue_change_percent']:.2f}%)",
                    f"- Baseline ROAS: {row['baseline_roas_expected']:.2f}",
                    f"- Scenario ROAS: {row['scenario_roas_expected']:.2f}",
                    f"- ROAS change: {row['roas_change']:.2f}",
                    f"- Scenario risk level: {row['scenario_risk_level']}",
                    "",
                    f"**Recommendation:** {row['scenario_recommendation']}",
                    "",
                ]
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_budget_simulation(
    data_dir: Path,
    model_path: Path,
    scenario_output_path: Path,
    comparison_output_path: Path,
    summary_output_path: Path,
    google_multiplier: float,
    meta_multiplier: float,
    microsoft_multiplier: float,
) -> pd.DataFrame:
    artifact = joblib.load(model_path)

    unified_df = normalize_all_platforms(data_dir)
    unified_df["date"] = pd.to_datetime(unified_df["date"], errors="coerce")
    unified_df = unified_df.dropna(subset=["date"])

    prediction_df = create_prediction_dataset(unified_df)

    channel_multipliers = build_channel_multipliers(
        google_multiplier=google_multiplier,
        meta_multiplier=meta_multiplier,
        microsoft_multiplier=microsoft_multiplier,
    )

    scenario_name = create_scenario_name(channel_multipliers)

    baseline_forecast = generate_forecast_from_features(prediction_df, artifact)

    scenario_features = apply_budget_scenario(
        prediction_df=prediction_df,
        channel_multipliers=channel_multipliers,
    )

    scenario_forecast = generate_forecast_from_features(scenario_features, artifact)

    comparison_df = compare_baseline_and_scenario(
        baseline_df=baseline_forecast,
        scenario_df=scenario_forecast,
    )

    scenario_forecast["scenario_name"] = scenario_name
    comparison_df["scenario_name"] = scenario_name

    scenario_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_output_path.parent.mkdir(parents=True, exist_ok=True)

    scenario_forecast.to_csv(scenario_output_path, index=False)
    comparison_df.to_csv(comparison_output_path, index=False)

    save_scenario_summary(
        comparison_df=comparison_df,
        scenario_name=scenario_name,
        output_path=summary_output_path,
    )

    return comparison_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run budget scenario simulation for ForecastIQ AI."
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
        "--scenario-output",
        default="output/scenario_forecasts.csv",
        help="Where to save scenario forecast CSV.",
    )

    parser.add_argument(
        "--comparison-output",
        default="output/scenario_comparison.csv",
        help="Where to save baseline vs scenario comparison CSV.",
    )

    parser.add_argument(
        "--summary-output",
        default="output/scenario_summary.md",
        help="Where to save scenario summary markdown.",
    )

    parser.add_argument(
        "--google-multiplier",
        type=float,
        default=1.0,
        help="Google Ads future budget multiplier. Example: 1.2 means +20%.",
    )

    parser.add_argument(
        "--meta-multiplier",
        type=float,
        default=1.0,
        help="Meta Ads future budget multiplier. Example: 0.9 means -10%.",
    )

    parser.add_argument(
        "--microsoft-multiplier",
        type=float,
        default=1.0,
        help="Microsoft Ads future budget multiplier. Example: 1.1 means +10%.",
    )

    args = parser.parse_args()

    comparison_df = run_budget_simulation(
        data_dir=Path(args.data_dir),
        model_path=Path(args.model),
        scenario_output_path=Path(args.scenario_output),
        comparison_output_path=Path(args.comparison_output),
        summary_output_path=Path(args.summary_output),
        google_multiplier=args.google_multiplier,
        meta_multiplier=args.meta_multiplier,
        microsoft_multiplier=args.microsoft_multiplier,
    )

    print("Budget simulation completed successfully.")
    print(f"Rows compared: {len(comparison_df):,}")
    print(f"Scenario forecast saved to: {args.scenario_output}")
    print(f"Scenario comparison saved to: {args.comparison_output}")
    print(f"Scenario summary saved to: {args.summary_output}")

    blended = comparison_df[comparison_df["forecast_level"] == "blended"]

    if not blended.empty:
        print("")
        print("Blended scenario impact:")
        display_columns = [
            "forecast_window_days",
            "baseline_budget",
            "scenario_budget",
            "budget_change_percent",
            "baseline_revenue_expected",
            "scenario_revenue_expected",
            "revenue_change_percent",
            "baseline_roas_expected",
            "scenario_roas_expected",
            "scenario_risk_level",
        ]
        print(blended[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()