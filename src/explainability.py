from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "forecast_level",
    "channel",
    "campaign_type",
    "campaign_id",
    "campaign_name",
    "forecast_window_days",
    "future_budget",
    "revenue_low",
    "revenue_expected",
    "revenue_high",
    "roas_low",
    "roas_expected",
    "roas_high",
    "confidence_score",
    "risk_level",
]


NUMERIC_COLUMNS = [
    "forecast_window_days",
    "future_budget",
    "revenue_low",
    "revenue_expected",
    "revenue_high",
    "roas_low",
    "roas_expected",
    "roas_high",
    "confidence_score",
]


def load_forecasts(predictions_path: Path) -> pd.DataFrame:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    df = pd.read_csv(predictions_path)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required prediction columns: {missing_columns}")

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    text_columns = [
        "forecast_level",
        "channel",
        "campaign_type",
        "campaign_id",
        "campaign_name",
        "risk_level",
    ]

    for column in text_columns:
        df[column] = df[column].fillna("Unknown").astype(str)

    return df


def load_scenario_comparison(scenario_path: Path | None) -> pd.DataFrame | None:
    if scenario_path is None:
        return None

    if not scenario_path.exists():
        return None

    df = pd.read_csv(scenario_path)

    numeric_columns = [
        "forecast_window_days",
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
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    return df


def money(value: float) -> str:
    return f"{value:,.2f}"


def pct(value: float) -> str:
    return f"{value:.2f}%"


def roas(value: float) -> str:
    return f"{value:.2f}x"


def risk_priority(risk_level: str) -> int:
    risk = str(risk_level).lower()

    if risk == "high":
        return 1

    if risk == "medium":
        return 2

    return 3


def get_window_df(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df[df["forecast_window_days"] == window].copy()


def get_blended_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["forecast_level"] == "blended"].copy()


def get_channel_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["forecast_level"] == "channel"].copy()


def get_campaign_type_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["forecast_level"] == "campaign_type"].copy()


def get_campaign_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["forecast_level"] == "campaign"].copy()


def uncertainty_ratio(row: pd.Series) -> float:
    expected = float(row["revenue_expected"])

    if expected <= 0:
        return 1.0

    return float(row["revenue_high"] - row["revenue_low"]) / expected


def explain_revenue_range(row: pd.Series) -> str:
    return (
        f"Expected revenue is {money(row['revenue_expected'])}, with a probable range "
        f"between {money(row['revenue_low'])} and {money(row['revenue_high'])}."
    )


def explain_roas_range(row: pd.Series) -> str:
    return (
        f"Expected ROAS is {roas(row['roas_expected'])}, with a range between "
        f"{roas(row['roas_low'])} and {roas(row['roas_high'])}."
    )


def explain_risk(row: pd.Series) -> str:
    risk = row["risk_level"]
    confidence = float(row["confidence_score"])
    expected_roas = float(row["roas_expected"])
    uncertainty = uncertainty_ratio(row)

    if risk == "High":
        if expected_roas < 1:
            return (
                "Risk is high because expected ROAS is below 1.00x, meaning the forecasted "
                "revenue may not justify the planned budget."
            )

        if confidence < 45:
            return (
                "Risk is high because forecast confidence is weak. The campaign may have "
                "unstable historical behavior or wide prediction uncertainty."
            )

        if uncertainty > 0.8:
            return (
                "Risk is high because the forecast range is wide, which means the outcome "
                "is less predictable."
            )

        return (
            "Risk is high because performance signals suggest the campaign may not scale "
            "efficiently under the current forecast assumptions."
        )

    if risk == "Medium":
        return (
            "Risk is medium because the forecast is usable, but efficiency or confidence "
            "should be monitored before making aggressive budget changes."
        )

    return (
        "Risk is low because forecast confidence and expected ROAS are relatively stable."
    )


def recommend_action(row: pd.Series) -> str:
    risk = row["risk_level"]
    expected_roas = float(row["roas_expected"])
    confidence = float(row["confidence_score"])
    uncertainty = uncertainty_ratio(row)

    if risk == "High":
        if expected_roas < 1:
            return (
                "Avoid scaling this segment immediately. Review campaign structure, targeting, "
                "creative quality, landing page relevance, and tracking accuracy before adding budget."
            )

        return (
            "Do not scale aggressively. Keep budget controlled, monitor daily performance, "
            "and test improvements before increasing spend."
        )

    if risk == "Medium":
        if expected_roas >= 2.5 and confidence >= 60:
            return (
                "Scale cautiously with a small budget increase and monitor ROAS movement closely."
            )

        return (
            "Maintain or slightly reduce budget until revenue efficiency becomes more consistent."
        )

    if expected_roas >= 3 and confidence >= 70 and uncertainty < 0.7:
        return (
            "This segment is a good candidate for controlled scaling because expected ROAS "
            "and confidence are healthy."
        )

    return (
        "Maintain current budget and continue monitoring. Scaling can be considered after "
        "more stable performance is observed."
    )


def create_row_insight(row: pd.Series) -> dict:
    level = row["forecast_level"]
    channel = row["channel"]
    campaign_type = row["campaign_type"]
    campaign_name = row["campaign_name"]
    window = int(row["forecast_window_days"])

    if level == "blended":
        subject = "overall marketing performance"
    elif level == "channel":
        subject = f"{channel}"
    elif level == "campaign_type":
        subject = f"{channel} - {campaign_type}"
    else:
        subject = f"{channel} campaign '{campaign_name}'"

    insight = (
        f"For the next {window} days, {subject} is forecasted to generate "
        f"{money(row['revenue_expected'])} revenue with expected ROAS of "
        f"{roas(row['roas_expected'])}. {explain_risk(row)}"
    )

    return {
        "priority": risk_priority(row["risk_level"]),
        "forecast_window_days": window,
        "forecast_level": level,
        "channel": channel,
        "campaign_type": campaign_type,
        "campaign_id": row["campaign_id"],
        "campaign_name": campaign_name,
        "risk_level": row["risk_level"],
        "confidence_score": round(float(row["confidence_score"]), 2),
        "revenue_expected": round(float(row["revenue_expected"]), 2),
        "roas_expected": round(float(row["roas_expected"]), 4),
        "insight": insight,
        "recommended_action": recommend_action(row),
    }


def build_insights_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    selected_rows = pd.concat(
        [
            get_blended_rows(df),
            get_channel_rows(df),
            get_campaign_type_rows(df),
            get_campaign_rows(df),
        ],
        ignore_index=True,
    )

    for _, row in selected_rows.iterrows():
        rows.append(create_row_insight(row))

    insights_df = pd.DataFrame(rows)

    if insights_df.empty:
        return insights_df

    return insights_df.sort_values(
        [
            "forecast_window_days",
            "priority",
            "revenue_expected",
        ],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_executive_summary(df: pd.DataFrame) -> list[str]:
    lines = [
        "## Executive Summary",
        "",
    ]

    blended = get_blended_rows(df)

    if blended.empty:
        lines.append("No blended forecast rows were found.")
        lines.append("")
        return lines

    for _, row in blended.sort_values("forecast_window_days").iterrows():
        window = int(row["forecast_window_days"])

        lines.extend(
            [
                f"### {window}-Day Business Forecast",
                "",
                f"- Planned future budget: **{money(row['future_budget'])}**",
                f"- {explain_revenue_range(row)}",
                f"- {explain_roas_range(row)}",
                f"- Forecast confidence score: **{row['confidence_score']:.2f}/100**",
                f"- Risk level: **{row['risk_level']}**",
                f"- Business interpretation: {explain_risk(row)}",
                f"- Recommended action: {recommend_action(row)}",
                "",
            ]
        )

    return lines


def build_channel_summary(df: pd.DataFrame) -> list[str]:
    lines = [
        "## Channel-Level Insights",
        "",
    ]

    channel_df = get_channel_rows(df)

    if channel_df.empty:
        lines.append("No channel-level forecast rows were found.")
        lines.append("")
        return lines

    for window in sorted(channel_df["forecast_window_days"].unique()):
        window_df = channel_df[channel_df["forecast_window_days"] == window].copy()
        window_df = window_df.sort_values("revenue_expected", ascending=False)

        lines.extend(
            [
                f"### {int(window)}-Day Channel Forecast",
                "",
                "| Channel | Budget | Expected Revenue | Expected ROAS | Risk | Confidence |",
                "|---|---:|---:|---:|---|---:|",
            ]
        )

        for _, row in window_df.iterrows():
            lines.append(
                f"| {row['channel']} | {money(row['future_budget'])} | "
                f"{money(row['revenue_expected'])} | {roas(row['roas_expected'])} | "
                f"{row['risk_level']} | {row['confidence_score']:.2f} |"
            )

        best_roas = window_df.sort_values("roas_expected", ascending=False).head(1)

        if not best_roas.empty:
            top = best_roas.iloc[0]
            lines.extend(
                [
                    "",
                    f"**Best efficiency signal:** {top['channel']} has the strongest expected ROAS "
                    f"at {roas(top['roas_expected'])}.",
                    "",
                ]
            )

    return lines


def build_campaign_type_summary(df: pd.DataFrame) -> list[str]:
    lines = [
        "## Campaign-Type Insights",
        "",
    ]

    campaign_type_df = get_campaign_type_rows(df)

    if campaign_type_df.empty:
        lines.append("No campaign-type forecast rows were found.")
        lines.append("")
        return lines

    for window in sorted(campaign_type_df["forecast_window_days"].unique()):
        window_df = campaign_type_df[
            campaign_type_df["forecast_window_days"] == window
        ].copy()

        top_rows = window_df.sort_values("revenue_expected", ascending=False).head(10)

        lines.extend(
            [
                f"### Top Campaign Types for {int(window)} Days",
                "",
                "| Channel | Campaign Type | Expected Revenue | Expected ROAS | Risk |",
                "|---|---|---:|---:|---|",
            ]
        )

        for _, row in top_rows.iterrows():
            lines.append(
                f"| {row['channel']} | {row['campaign_type']} | "
                f"{money(row['revenue_expected'])} | {roas(row['roas_expected'])} | "
                f"{row['risk_level']} |"
            )

        lines.append("")

    return lines


def build_risk_summary(df: pd.DataFrame) -> list[str]:
    lines = [
        "## Risk and Monitoring Summary",
        "",
    ]

    campaign_df = get_campaign_rows(df)

    if campaign_df.empty:
        lines.append("No campaign-level forecast rows were found.")
        lines.append("")
        return lines

    risky = campaign_df[campaign_df["risk_level"].isin(["High", "Medium"])].copy()

    if risky.empty:
        lines.append("No major campaign-level risks were detected.")
        lines.append("")
        return lines

    risky["priority"] = risky["risk_level"].apply(risk_priority)

    top_risky = risky.sort_values(
        ["priority", "revenue_expected"],
        ascending=[True, False],
    ).head(15)

    lines.extend(
        [
            "These campaigns should be monitored before scaling budget:",
            "",
            "| Risk | Channel | Campaign | Window | Expected Revenue | Expected ROAS | Confidence | Recommended Action |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )

    for _, row in top_risky.iterrows():
        lines.append(
            f"| {row['risk_level']} | {row['channel']} | {row['campaign_name']} | "
            f"{int(row['forecast_window_days'])} | {money(row['revenue_expected'])} | "
            f"{roas(row['roas_expected'])} | {row['confidence_score']:.2f} | "
            f"{recommend_action(row)} |"
        )

    lines.append("")

    return lines


def build_scaling_opportunities(df: pd.DataFrame) -> list[str]:
    lines = [
        "## Scaling Opportunities",
        "",
    ]

    campaign_df = get_campaign_rows(df)

    if campaign_df.empty:
        lines.append("No campaign-level forecasts were found.")
        lines.append("")
        return lines

    candidates = campaign_df[
        (campaign_df["risk_level"] == "Low")
        & (campaign_df["roas_expected"] >= 3)
        & (campaign_df["confidence_score"] >= 65)
    ].copy()

    if candidates.empty:
        lines.append(
            "No strong low-risk scaling candidates were found. The safer approach is to maintain current budget and monitor performance."
        )
        lines.append("")
        return lines

    candidates = candidates.sort_values(
        ["forecast_window_days", "roas_expected", "revenue_expected"],
        ascending=[True, False, False],
    ).head(15)

    lines.extend(
        [
            "These campaigns show healthier ROAS and confidence signals:",
            "",
            "| Channel | Campaign | Window | Expected Revenue | Expected ROAS | Confidence | Suggested Action |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )

    for _, row in candidates.iterrows():
        lines.append(
            f"| {row['channel']} | {row['campaign_name']} | "
            f"{int(row['forecast_window_days'])} | {money(row['revenue_expected'])} | "
            f"{roas(row['roas_expected'])} | {row['confidence_score']:.2f} | "
            f"{recommend_action(row)} |"
        )

    lines.append("")

    return lines


def build_scenario_summary(scenario_df: pd.DataFrame | None) -> list[str]:
    lines = [
        "## Budget Scenario Interpretation",
        "",
    ]

    if scenario_df is None or scenario_df.empty:
        lines.append(
            "No budget scenario comparison file was provided. Run `src.budget_simulator` to generate scenario-level insights."
        )
        lines.append("")
        return lines

    blended = scenario_df[scenario_df["forecast_level"] == "blended"].copy()

    if blended.empty:
        lines.append("No blended scenario rows were found.")
        lines.append("")
        return lines

    for _, row in blended.sort_values("forecast_window_days").iterrows():
        window = int(row["forecast_window_days"])
        revenue_change = float(row["revenue_change"])
        revenue_change_percent = float(row["revenue_change_percent"])
        roas_change = float(row["roas_change"])

        if revenue_change > 0 and roas_change >= 0:
            interpretation = (
                "This scenario is attractive because it improves expected revenue without reducing ROAS."
            )
        elif revenue_change > 0 and roas_change < 0:
            interpretation = (
                "This scenario can grow revenue, but the agency should watch efficiency because ROAS is expected to decline."
            )
        elif revenue_change <= 0 and roas_change > 0:
            interpretation = (
                "This scenario may improve efficiency, but it may limit revenue scale."
            )
        else:
            interpretation = (
                "This scenario does not look attractive because both revenue and efficiency may weaken."
            )

        lines.extend(
            [
                f"### {window}-Day Scenario",
                "",
                f"- Baseline budget: **{money(row['baseline_budget'])}**",
                f"- Scenario budget: **{money(row['scenario_budget'])}**",
                f"- Budget change: **{money(row['budget_change'])} ({pct(row['budget_change_percent'])})**",
                f"- Revenue change: **{money(revenue_change)} ({pct(revenue_change_percent)})**",
                f"- ROAS change: **{roas_change:.2f}x**",
                f"- Interpretation: {interpretation}",
                "",
            ]
        )

    return lines


def build_final_recommendations(df: pd.DataFrame) -> list[str]:
    lines = [
        "## Final Agency Recommendations",
        "",
    ]

    blended = get_blended_rows(df)

    high_risk_count = int((df["risk_level"] == "High").sum())
    medium_risk_count = int((df["risk_level"] == "Medium").sum())

    avg_confidence = float(df["confidence_score"].mean()) if not df.empty else 0

    lines.append(
        f"- Overall forecast confidence across all forecast rows is **{avg_confidence:.2f}/100**."
    )

    if high_risk_count > 0:
        lines.append(
            f"- There are **{high_risk_count} high-risk forecast rows**. These should be reviewed before budget scaling."
        )

    if medium_risk_count > 0:
        lines.append(
            f"- There are **{medium_risk_count} medium-risk forecast rows**. These can be monitored with controlled budget movement."
        )

    if not blended.empty:
        best_window = blended.sort_values("roas_expected", ascending=False).iloc[0]
        lines.append(
            f"- The strongest blended ROAS appears in the **{int(best_window['forecast_window_days'])}-day forecast** "
            f"with expected ROAS of **{roas(best_window['roas_expected'])}**."
        )

    lines.extend(
        [
            "- Use the forecast as a decision-support system, not as an exact future guarantee.",
            "- Prefer controlled budget changes over sudden aggressive scaling.",
            "- Review tracking quality, campaign naming consistency, and revenue attribution before final client decisions.",
            "- Use budget simulation outputs to compare multiple future spending plans before committing media budget.",
            "",
        ]
    )

    return lines


def generate_agency_report(
    predictions_path: Path,
    report_output_path: Path,
    insights_output_path: Path,
    scenario_comparison_path: Path | None = None,
) -> pd.DataFrame:
    forecast_df = load_forecasts(predictions_path)
    scenario_df = load_scenario_comparison(scenario_comparison_path)

    insights_df = build_insights_table(forecast_df)

    report_lines = [
        "# ForecastIQ AI - Agency Forecast Intelligence Report",
        "",
        "This report converts probabilistic revenue and ROAS forecasts into agency-ready marketing insights.",
        "",
    ]

    report_lines.extend(build_executive_summary(forecast_df))
    report_lines.extend(build_channel_summary(forecast_df))
    report_lines.extend(build_campaign_type_summary(forecast_df))
    report_lines.extend(build_risk_summary(forecast_df))
    report_lines.extend(build_scaling_opportunities(forecast_df))
    report_lines.extend(build_scenario_summary(scenario_df))
    report_lines.extend(build_final_recommendations(forecast_df))

    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    insights_output_path.parent.mkdir(parents=True, exist_ok=True)

    report_output_path.write_text("\n".join(report_lines), encoding="utf-8")
    insights_df.to_csv(insights_output_path, index=False)

    return insights_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate agency-ready forecast explanations and insights."
    )

    parser.add_argument(
        "--predictions",
        default="output/predictions.csv",
        help="Path to predictions CSV generated by src.predict.",
    )

    parser.add_argument(
        "--scenario-comparison",
        default="output/scenario_comparison.csv",
        help="Optional path to budget scenario comparison CSV.",
    )

    parser.add_argument(
        "--report-output",
        default="output/agency_forecast_report.md",
        help="Where to save the agency-ready markdown report.",
    )

    parser.add_argument(
        "--insights-output",
        default="output/agency_insights.csv",
        help="Where to save structured insights CSV.",
    )

    args = parser.parse_args()

    scenario_path = Path(args.scenario_comparison)

    if not scenario_path.exists():
        scenario_path = None

    insights_df = generate_agency_report(
        predictions_path=Path(args.predictions),
        scenario_comparison_path=scenario_path,
        report_output_path=Path(args.report_output),
        insights_output_path=Path(args.insights_output),
    )

    print("Agency explainability report generated successfully.")
    print(f"Insights created: {len(insights_df):,}")
    print(f"Report saved to: {args.report_output}")
    print(f"Structured insights saved to: {args.insights_output}")


if __name__ == "__main__":
    main()