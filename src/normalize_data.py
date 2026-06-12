from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


UNIFIED_COLUMNS = [
    "date",
    "channel",
    "campaign_id",
    "campaign_name",
    "campaign_type",
    "daily_budget",
    "spend",
    "clicks",
    "impressions",
    "conversions",
    "revenue",
    "ctr",
    "cpc",
    "cpm",
    "conversion_rate",
    "roas",
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def find_csv_file(data_dir: Path, keywords: list[str]) -> Path:
    csv_files = list(data_dir.glob("*.csv"))

    for file_path in csv_files:
        name = file_path.name.lower()
        if all(keyword.lower() in name for keyword in keywords):
            return file_path

    raise FileNotFoundError(
        f"Could not find CSV file with keywords {keywords} inside {data_dir}"
    )


def clean_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def get_first_existing_column(df: pd.DataFrame, possible_columns: list[str]) -> str | None:
    for column in possible_columns:
        if column in df.columns:
            return column
    return None


def derive_meta_campaign_type(campaign_name: str) -> str:
    name = str(campaign_name).lower()

    if "remarketing" in name or "retarget" in name:
        return "Remarketing"

    if "prospecting" in name or "dpa" in name:
        return "Prospecting"

    return "Generic"


def add_calculated_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ctr"] = safe_divide(df["clicks"], df["impressions"])
    df["cpc"] = safe_divide(df["spend"], df["clicks"])
    df["cpm"] = safe_divide(df["spend"], df["impressions"]) * 1000
    df["conversion_rate"] = safe_divide(df["conversions"], df["clicks"])
    df["roas"] = safe_divide(df["revenue"], df["spend"])

    return df


def normalize_bing_ads(data_dir: Path) -> pd.DataFrame:
    file_path = find_csv_file(data_dir, ["bing", "campaign"])
    raw_df = pd.read_csv(file_path)

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(raw_df["TimePeriod"], errors="coerce")
    df["channel"] = "Microsoft Ads"
    df["campaign_id"] = raw_df["CampaignId"].astype(str)
    df["campaign_name"] = raw_df["CampaignName"].astype(str)
    df["campaign_type"] = raw_df["CampaignType"].astype(str)
    df["daily_budget"] = clean_numeric(raw_df["DailyBudget"])
    df["spend"] = clean_numeric(raw_df["Spend"])
    df["clicks"] = clean_numeric(raw_df["Clicks"])
    df["impressions"] = clean_numeric(raw_df["Impressions"])
    df["conversions"] = clean_numeric(raw_df["Conversions"])
    df["revenue"] = clean_numeric(raw_df["Revenue"])

    df = add_calculated_metrics(df)

    return df[UNIFIED_COLUMNS]


def normalize_google_ads(data_dir: Path) -> pd.DataFrame:
    file_path = find_csv_file(data_dir, ["google", "ads"])
    raw_df = pd.read_csv(file_path)

    budget_column = get_first_existing_column(
        raw_df,
        [
            "campaign_budget_amount",
            "campaign_budget_amount_micros",
            "campaign_budget",
            "campaign_budget_amount_local",
        ],
    )

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(raw_df["segments_date"], errors="coerce")
    df["channel"] = "Google Ads"
    df["campaign_id"] = raw_df["campaign_id"].astype(str)
    df["campaign_name"] = raw_df["campaign_name"].astype(str)
    df["campaign_type"] = raw_df["campaign_advertising_channel_type"].astype(str)

    if budget_column is not None:
        df["daily_budget"] = clean_numeric(raw_df[budget_column])
    else:
        df["daily_budget"] = 0.0

    df["spend"] = clean_numeric(raw_df["metrics_cost_micros"]) / 1_000_000
    df["clicks"] = clean_numeric(raw_df["metrics_clicks"])
    df["impressions"] = clean_numeric(raw_df["metrics_impressions"])
    df["conversions"] = clean_numeric(raw_df["metrics_conversions"])
    df["revenue"] = clean_numeric(raw_df["metrics_conversions_value"])

    df = add_calculated_metrics(df)

    return df[UNIFIED_COLUMNS]


def normalize_meta_ads(data_dir: Path) -> pd.DataFrame:
    file_path = find_csv_file(data_dir, ["meta", "ads"])
    raw_df = pd.read_csv(file_path)

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(raw_df["date_start"], errors="coerce")
    df["channel"] = "Meta Ads"
    df["campaign_id"] = raw_df["campaign_id"].astype(str)
    df["campaign_name"] = raw_df["campaign_name"].astype(str)
    df["campaign_type"] = raw_df["campaign_name"].apply(derive_meta_campaign_type)
    df["daily_budget"] = clean_numeric(raw_df["daily_budget"])
    df["spend"] = clean_numeric(raw_df["spend"])
    df["clicks"] = clean_numeric(raw_df["clicks"])
    df["impressions"] = clean_numeric(raw_df["impressions"])

    # Dataset note:
    # Meta file has a column named "conversion".
    # Its values behave like conversion value/revenue, so we use it as revenue.
    # Conversion count is not clearly available, so we keep conversions as 0.
    df["conversions"] = 0.0
    df["revenue"] = clean_numeric(raw_df["conversion"])

    df = add_calculated_metrics(df)

    return df[UNIFIED_COLUMNS]


def normalize_all_platforms(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)

    normalized_frames = [
        normalize_google_ads(data_dir),
        normalize_bing_ads(data_dir),
        normalize_meta_ads(data_dir),
    ]

    unified_df = pd.concat(normalized_frames, ignore_index=True)

    unified_df = unified_df.sort_values(
        by=["date", "channel", "campaign_id"]
    ).reset_index(drop=True)

    return unified_df[UNIFIED_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize Google, Microsoft/Bing, and Meta Ads data."
    )
    parser.add_argument("--data-dir", default="data", help="Folder containing CSV files.")
    parser.add_argument(
        "--output-path",
        default="output/unified_marketing_data.csv",
        help="Where to save the normalized dataset.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output_path)

    unified_df = normalize_all_platforms(data_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    unified_df.to_csv(output_path, index=False)

    print("Unified dataset created successfully.")
    print(f"Rows: {len(unified_df):,}")
    print(f"Columns: {len(unified_df.columns)}")
    print(f"Date range: {unified_df['date'].min()} to {unified_df['date'].max()}")
    print(f"Channels: {', '.join(unified_df['channel'].unique())}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()