from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_PREDICTIONS_PATH = Path("output/predictions.csv")
DEFAULT_INSIGHTS_PATH = Path("output/agency_insights.csv")
DEFAULT_REPORT_PATH = Path("output/agency_forecast_report.md")
DEFAULT_SCENARIO_PATH = Path("output/scenario_comparison.csv")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_MODEL_PATH = Path("pickle/model.pkl")


st.set_page_config(
    page_title="ForecastIQ AI",
    page_icon="📈",
    layout="wide",
)


CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.metric-card {
    padding: 1rem;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.25);
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.92));
    color: white;
    box-shadow: 0 12px 34px rgba(15, 23, 42, 0.18);
}

.metric-label {
    font-size: 0.82rem;
    opacity: 0.72;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1.2;
}

.metric-help {
    font-size: 0.78rem;
    opacity: 0.7;
    margin-top: 0.35rem;
}

.section-card {
    padding: 1rem;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.22);
    background: rgba(255, 255, 255, 0.72);
}

.risk-low {
    color: #047857;
    font-weight: 700;
}

.risk-medium {
    color: #b45309;
    font-weight: 700;
}

.risk-high {
    color: #b91c1c;
    font-weight: 700;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def money(value: float) -> str:
    return f"{value:,.2f}"


def roas(value: float) -> str:
    return f"{value:.2f}x"


def pct(value: float) -> str:
    return f"{value:.2f}%"


def safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = [
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

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    text_columns = [
        "forecast_id",
        "forecast_level",
        "channel",
        "campaign_type",
        "campaign_id",
        "campaign_name",
        "risk_level",
        "insight_summary",
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].fillna("Unknown").astype(str)

    return df


def clean_scenario(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

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


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_label(risk: str) -> str:
    risk_lower = str(risk).lower()

    if risk_lower == "low":
        return "Low"

    if risk_lower == "medium":
        return "Medium"

    if risk_lower == "high":
        return "High"

    return str(risk)


def filter_by_window(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df[df["forecast_window_days"] == window].copy()


def get_blended(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df[
        (df["forecast_level"] == "blended")
        & (df["forecast_window_days"] == window)
    ].copy()


def get_level(df: pd.DataFrame, level: str, window: int) -> pd.DataFrame:
    return df[
        (df["forecast_level"] == level)
        & (df["forecast_window_days"] == window)
    ].copy()


def show_missing_files_help(predictions_path: Path) -> None:
    st.error(f"Predictions file not found: `{predictions_path}`")

    st.markdown("Run the pipeline first from your project root:")

    st.code(
        r"""
python -m src.predict --data-dir data --model pickle/model.pkl --output output/predictions.csv
python -m src.explainability --predictions output/predictions.csv --report-output output/agency_forecast_report.md --insights-output output/agency_insights.csv
""".strip(),
        language="powershell",
    )

    st.markdown("Or run the Windows pipeline:")

    st.code(
        r"""
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run.ps1 ./data ./pickle/model.pkl ./output/predictions.csv
""".strip(),
        language="powershell",
    )


def run_budget_simulator(
    data_dir: Path,
    model_path: Path,
    google_multiplier: float,
    meta_multiplier: float,
    microsoft_multiplier: float,
) -> tuple[bool, str]:
    command = [
        sys.executable,
        "-m",
        "src.budget_simulator",
        "--data-dir",
        str(data_dir),
        "--model",
        str(model_path),
        "--google-multiplier",
        str(google_multiplier),
        "--meta-multiplier",
        str(meta_multiplier),
        "--microsoft-multiplier",
        str(microsoft_multiplier),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    output = result.stdout.strip()

    if result.stderr:
        output += "\n\n" + result.stderr.strip()

    return result.returncode == 0, output


def run_explainability() -> tuple[bool, str]:
    command = [
        sys.executable,
        "-m",
        "src.explainability",
        "--predictions",
        "output/predictions.csv",
        "--scenario-comparison",
        "output/scenario_comparison.csv",
        "--report-output",
        "output/agency_forecast_report.md",
        "--insights-output",
        "output/agency_insights.csv",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    output = result.stdout.strip()

    if result.stderr:
        output += "\n\n" + result.stderr.strip()

    return result.returncode == 0, output


def main() -> None:
    st.title("📈 ForecastIQ AI")
    st.caption(
        "AI-assisted probabilistic revenue and ROAS forecasting utility for e-commerce marketing agencies."
    )

    with st.sidebar:
        st.header("Project Files")

        predictions_path = Path(
            st.text_input("Predictions CSV", value=str(DEFAULT_PREDICTIONS_PATH))
        )

        insights_path = Path(
            st.text_input("Agency Insights CSV", value=str(DEFAULT_INSIGHTS_PATH))
        )

        report_path = Path(
            st.text_input("Agency Report Markdown", value=str(DEFAULT_REPORT_PATH))
        )

        scenario_path = Path(
            st.text_input("Scenario Comparison CSV", value=str(DEFAULT_SCENARIO_PATH))
        )

        data_dir = Path(st.text_input("Data Folder", value=str(DEFAULT_DATA_DIR)))
        model_path = Path(st.text_input("Model Path", value=str(DEFAULT_MODEL_PATH)))

        st.divider()

        if st.button("Refresh Dashboard Data"):
            st.cache_data.clear()
            st.rerun()

    if not predictions_path.exists():
        show_missing_files_help(predictions_path)
        return

    predictions_df = clean_predictions(load_csv(str(predictions_path)))

    available_windows = sorted(predictions_df["forecast_window_days"].unique().tolist())

    if not available_windows:
        st.error("No forecast windows found in predictions.csv")
        return

    selected_window = st.sidebar.selectbox(
        "Forecast Window",
        available_windows,
        index=0,
        format_func=lambda value: f"{int(value)} days",
    )

    window_df = filter_by_window(predictions_df, selected_window)
    blended_df = get_blended(predictions_df, selected_window)

    if blended_df.empty:
        st.warning("No blended forecast row found for the selected window.")
        blended_row = None
    else:
        blended_row = blended_df.iloc[0]

    st.subheader(f"{int(selected_window)}-Day Forecast Overview")

    if blended_row is not None:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            metric_card(
                "Future Budget",
                money(blended_row["future_budget"]),
                "Planned / estimated media budget",
            )

        with col2:
            metric_card(
                "Expected Revenue",
                money(blended_row["revenue_expected"]),
                f"Range: {money(blended_row['revenue_low'])} - {money(blended_row['revenue_high'])}",
            )

        with col3:
            metric_card(
                "Expected ROAS",
                roas(blended_row["roas_expected"]),
                f"Range: {roas(blended_row['roas_low'])} - {roas(blended_row['roas_high'])}",
            )

        with col4:
            metric_card(
                "Confidence",
                f"{blended_row['confidence_score']:.2f}/100",
                "Higher is better",
            )

        with col5:
            metric_card(
                "Risk Level",
                risk_label(blended_row["risk_level"]),
                "Based on ROAS, confidence, and uncertainty",
            )

        st.info(blended_row["insight_summary"])

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Channel Forecast",
            "Campaign Types",
            "Campaign Risks",
            "Budget Simulator",
            "Agency Report",
            "Raw Forecast Data",
            "Structured Insights",
        ]
    )

    with tab1:
        st.subheader("Channel-Level Forecast")

        channel_df = get_level(predictions_df, "channel", selected_window)

        if channel_df.empty:
            st.warning("No channel-level forecast found.")
        else:
            chart_df = channel_df[
                ["channel", "future_budget", "revenue_expected", "roas_expected", "risk_level"]
            ].sort_values("revenue_expected", ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Expected Revenue by Channel**")
                st.bar_chart(
                    chart_df,
                    x="channel",
                    y="revenue_expected",
                    use_container_width=True,
                )

            with col2:
                st.markdown("**Expected ROAS by Channel**")
                st.bar_chart(
                    chart_df,
                    x="channel",
                    y="roas_expected",
                    use_container_width=True,
                )

            st.dataframe(
                chart_df,
                use_container_width=True,
                hide_index=True,
            )

    with tab2:
        st.subheader("Campaign-Type Forecast")

        campaign_type_df = get_level(predictions_df, "campaign_type", selected_window)

        if campaign_type_df.empty:
            st.warning("No campaign-type forecast found.")
        else:
            top_campaign_types = campaign_type_df.sort_values(
                "revenue_expected",
                ascending=False,
            ).head(20)

            st.markdown("**Top Campaign Types by Expected Revenue**")

            st.bar_chart(
                top_campaign_types,
                x="campaign_type",
                y="revenue_expected",
                use_container_width=True,
            )

            st.dataframe(
                top_campaign_types[
                    [
                        "channel",
                        "campaign_type",
                        "future_budget",
                        "revenue_expected",
                        "revenue_low",
                        "revenue_high",
                        "roas_expected",
                        "confidence_score",
                        "risk_level",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with tab3:
        st.subheader("Campaign-Level Risks and Scaling Opportunities")

        campaign_df = get_level(predictions_df, "campaign", selected_window)

        if campaign_df.empty:
            st.warning("No campaign-level forecast found.")
        else:
            risk_filter = st.multiselect(
                "Filter by Risk Level",
                ["High", "Medium", "Low"],
                default=["High", "Medium"],
            )

            filtered_risk_df = campaign_df[
                campaign_df["risk_level"].isin(risk_filter)
            ].copy()

            st.markdown("**Campaigns to Monitor**")

            st.dataframe(
                filtered_risk_df.sort_values(
                    ["risk_level", "revenue_expected"],
                    ascending=[True, False],
                )[
                    [
                        "channel",
                        "campaign_type",
                        "campaign_name",
                        "future_budget",
                        "revenue_expected",
                        "roas_expected",
                        "confidence_score",
                        "risk_level",
                        "insight_summary",
                    ]
                ].head(50),
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            st.markdown("**Low-Risk Scaling Candidates**")

            scaling_df = campaign_df[
                (campaign_df["risk_level"] == "Low")
                & (campaign_df["roas_expected"] >= 3)
                & (campaign_df["confidence_score"] >= 65)
            ].copy()

            if scaling_df.empty:
                st.info("No strong low-risk scaling candidates found for this window.")
            else:
                st.dataframe(
                    scaling_df.sort_values(
                        ["roas_expected", "revenue_expected"],
                        ascending=[False, False],
                    )[
                        [
                            "channel",
                            "campaign_type",
                            "campaign_name",
                            "future_budget",
                            "revenue_expected",
                            "roas_expected",
                            "confidence_score",
                            "risk_level",
                        ]
                    ].head(30),
                    use_container_width=True,
                    hide_index=True,
                )

    with tab4:
        st.subheader("Budget Scenario Simulator")

        st.markdown(
            "Adjust future media budget by channel and compare the predicted impact."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            google_change = st.slider(
                "Google Ads Budget Change",
                min_value=-50,
                max_value=100,
                value=20,
                step=5,
                format="%d%%",
            )

        with col2:
            meta_change = st.slider(
                "Meta Ads Budget Change",
                min_value=-50,
                max_value=100,
                value=0,
                step=5,
                format="%d%%",
            )

        with col3:
            microsoft_change = st.slider(
                "Microsoft Ads Budget Change",
                min_value=-50,
                max_value=100,
                value=0,
                step=5,
                format="%d%%",
            )

        google_multiplier = 1 + (google_change / 100)
        meta_multiplier = 1 + (meta_change / 100)
        microsoft_multiplier = 1 + (microsoft_change / 100)

        if st.button("Run Budget Simulation"):
            with st.spinner("Running budget simulation..."):
                ok, output = run_budget_simulator(
                    data_dir=data_dir,
                    model_path=model_path,
                    google_multiplier=google_multiplier,
                    meta_multiplier=meta_multiplier,
                    microsoft_multiplier=microsoft_multiplier,
                )

                if ok:
                    ok_report, report_output = run_explainability()
                    st.cache_data.clear()
                    st.success("Budget simulation completed.")
                    st.code(output)
                    if ok_report:
                        st.info("Agency report updated with scenario interpretation.")
                    else:
                        st.warning("Scenario ran, but report refresh failed.")
                        st.code(report_output)
                else:
                    st.error("Budget simulation failed.")
                    st.code(output)

        if scenario_path.exists():
            scenario_df = clean_scenario(load_csv(str(scenario_path)))
            scenario_window_df = scenario_df[
                scenario_df["forecast_window_days"] == selected_window
            ].copy()

            blended_scenario = scenario_window_df[
                scenario_window_df["forecast_level"] == "blended"
            ]

            if not blended_scenario.empty:
                row = blended_scenario.iloc[0]

                st.markdown("### Current Scenario Impact")

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.metric(
                        "Budget Change",
                        money(row["budget_change"]),
                        pct(row["budget_change_percent"]),
                    )

                with c2:
                    st.metric(
                        "Revenue Change",
                        money(row["revenue_change"]),
                        pct(row["revenue_change_percent"]),
                    )

                with c3:
                    st.metric(
                        "ROAS Change",
                        f"{row['roas_change']:.2f}x",
                    )

                with c4:
                    st.metric(
                        "Scenario ROAS",
                        roas(row["scenario_roas_expected"]),
                    )

            st.markdown("### Scenario Comparison Table")

            st.dataframe(
                scenario_window_df[
                    [
                        "forecast_level",
                        "channel",
                        "campaign_type",
                        "campaign_name",
                        "baseline_budget",
                        "scenario_budget",
                        "budget_change_percent",
                        "baseline_revenue_expected",
                        "scenario_revenue_expected",
                        "revenue_change_percent",
                        "baseline_roas_expected",
                        "scenario_roas_expected",
                        "scenario_risk_level",
                        "scenario_recommendation",
                    ]
                ].head(100),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No scenario file found yet. Run a budget simulation first.")

    with tab5:
        st.subheader("Agency-Ready Forecast Report")

        report_text = safe_read_text(report_path)

        if not report_text:
            st.warning("Agency report file not found. Run explainability first.")
            st.code(
                "python -m src.explainability --predictions output/predictions.csv --report-output output/agency_forecast_report.md --insights-output output/agency_insights.csv",
                language="powershell",
            )
        else:
            st.markdown(report_text)

    with tab6:
        st.subheader("Raw Forecast Output")

        st.download_button(
            label="Download predictions.csv",
            data=predictions_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

        st.dataframe(
            window_df,
            use_container_width=True,
            hide_index=True,
        )

    with tab7:
        st.subheader("Structured Agency Insights")

        if insights_path.exists():
            insights_df = load_csv(str(insights_path))

            st.download_button(
                label="Download agency_insights.csv",
                data=insights_df.to_csv(index=False).encode("utf-8"),
                file_name="agency_insights.csv",
                mime="text/csv",
            )

            st.dataframe(
                insights_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("agency_insights.csv not found yet.")


if __name__ == "__main__":
    main()