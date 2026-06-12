from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.normalize_data import normalize_all_platforms


REQUIRED_COLUMNS = [
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


def validate_required_columns(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")

    return issues


def validate_missing_values(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    important_columns = [
        "date",
        "channel",
        "campaign_id",
        "campaign_name",
        "campaign_type",
        "spend",
        "revenue",
    ]

    for column in important_columns:
        if column in df.columns:
            missing_count = int(df[column].isna().sum())
            if missing_count > 0:
                issues.append(f"{column}: {missing_count} missing values")

    return issues


def validate_numeric_values(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    numeric_columns = [
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

    for column in numeric_columns:
        if column not in df.columns:
            continue

        negative_count = int((df[column] < 0).sum())

        if negative_count > 0:
            issues.append(f"{column}: {negative_count} negative values found")

    return issues


def validate_dates(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    if "date" not in df.columns:
        return ["date column is missing"]

    invalid_dates = int(df["date"].isna().sum())

    if invalid_dates > 0:
        issues.append(f"date: {invalid_dates} invalid or missing dates")

    return issues


def validate_duplicates(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    duplicate_keys = ["date", "channel", "campaign_id"]

    if all(col in df.columns for col in duplicate_keys):
        duplicate_count = int(df.duplicated(subset=duplicate_keys).sum())

        if duplicate_count > 0:
            issues.append(
                f"Duplicate campaign-date rows found: {duplicate_count}. "
                "This may happen if platform exports contain repeated daily campaign records."
            )

    return issues


def validate_zero_activity(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    if "spend" in df.columns:
        zero_spend_count = int((df["spend"] == 0).sum())
        issues.append(f"Zero spend rows: {zero_spend_count}")

    if "revenue" in df.columns:
        zero_revenue_count = int((df["revenue"] == 0).sum())
        issues.append(f"Zero revenue rows: {zero_revenue_count}")

    if "clicks" in df.columns:
        zero_click_count = int((df["clicks"] == 0).sum())
        issues.append(f"Zero click rows: {zero_click_count}")

    if "impressions" in df.columns:
        zero_impression_count = int((df["impressions"] == 0).sum())
        issues.append(f"Zero impression rows: {zero_impression_count}")

    return issues


def validate_roas_outliers(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    if "roas" not in df.columns:
        return issues

    valid_roas = df["roas"].replace([float("inf"), -float("inf")], pd.NA).dropna()

    if valid_roas.empty:
        issues.append("ROAS could not be calculated for any row")
        return issues

    extreme_roas_count = int((valid_roas > 100).sum())

    if extreme_roas_count > 0:
        issues.append(
            f"Extreme ROAS rows found: {extreme_roas_count}. "
            "These rows may be valid, but should be handled carefully during modeling."
        )

    return issues


def build_validation_report(df: pd.DataFrame) -> dict:
    report = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "date_min": str(df["date"].min()) if "date" in df.columns else None,
        "date_max": str(df["date"].max()) if "date" in df.columns else None,
        "channels": sorted(df["channel"].dropna().unique().tolist())
        if "channel" in df.columns
        else [],
        "campaign_count": int(df["campaign_id"].nunique())
        if "campaign_id" in df.columns
        else 0,
        "issues": [],
    }

    checks = [
        validate_required_columns,
        validate_missing_values,
        validate_numeric_values,
        validate_dates,
        validate_duplicates,
        validate_zero_activity,
        validate_roas_outliers,
    ]

    for check in checks:
        report["issues"].extend(check(df))

    return report


def save_validation_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Data Validation Report",
        "",
        f"Total rows: {report['total_rows']}",
        f"Total columns: {report['total_columns']}",
        f"Date range: {report['date_min']} to {report['date_max']}",
        f"Channels: {', '.join(report['channels'])}",
        f"Campaign count: {report['campaign_count']}",
        "",
        "## Issues / Observations",
        "",
    ]

    if report["issues"]:
        for issue in report["issues"]:
            lines.append(f"- {issue}")
    else:
        lines.append("- No major issues found.")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate unified marketing dataset.")
    parser.add_argument("--data-dir", default="data", help="Folder containing input CSV files.")
    parser.add_argument(
        "--report-path",
        default="output/validation_report.md",
        help="Where to save the validation report.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    report_path = Path(args.report_path)

    df = normalize_all_platforms(data_dir)
    report = build_validation_report(df)
    save_validation_report(report, report_path)

    print("Validation completed successfully.")
    print(f"Rows checked: {report['total_rows']}")
    print(f"Campaigns checked: {report['campaign_count']}")
    print(f"Report saved to: {report_path}")

    if report["issues"]:
        print("\nIssues / observations:")
        for issue in report["issues"]:
            print(f"- {issue}")


if __name__ == "__main__":
    main()