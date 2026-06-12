# ForecastIQ AI

**AI-assisted probabilistic revenue and ROAS forecasting utility for e-commerce marketing agencies.**

ForecastIQ AI is built for the **NetElixir AIgnition 3.0 Hackathon**. It forecasts future e-commerce marketing performance using historical campaign data from Google Ads, Microsoft/Bing Ads, and Meta Ads.

The system predicts future revenue and ROAS as probabilistic ranges for **30, 60, and 90 day planning windows**, and generates agency-ready insights for campaign planning, risk monitoring, and budget simulation.

---

## 1. Problem Statement

Digital marketing agencies manage campaigns across multiple paid channels, but most reporting tools only explain past performance.

Agencies need forward-looking answers such as:

* What revenue can we expect in the next 30, 60, or 90 days?
* What ROAS range can we expect?
* Which channel is likely to perform better?
* Which campaign type is risky?
* What happens if we increase or decrease media budget?
* How can we explain the forecast to a client?

ForecastIQ AI solves this by combining:

* campaign data normalization,
* revenue forecasting,
* ROAS estimation,
* probabilistic prediction ranges,
* budget scenario simulation,
* risk detection,
* and agency-ready explainability.

---

## 2. Key Features

### Completed Features

* Multi-platform CSV ingestion
* Google Ads, Meta Ads, and Microsoft/Bing Ads normalization
* Unified campaign-level marketing schema
* Data validation and campaign consistency checks
* Feature engineering for 30, 60, and 90 day forecasting
* Probabilistic revenue forecasting using quantile models
* ROAS range calculation
* Channel-level forecasts
* Campaign-type-level forecasts
* Campaign-level forecasts
* Blended overall forecast
* Budget scenario simulator
* Agency-ready forecast report generator
* Structured insight generation
* Streamlit dashboard
* Linux/macOS `run.sh` pipeline
* Windows `run.ps1` pipeline

---

## 3. Tech Stack

| Layer            | Technology        |
| ---------------- | ----------------- |
| Language         | Python 3.11       |
| Data Processing  | pandas, numpy     |
| Machine Learning | scikit-learn      |
| Model Storage    | joblib / pickle   |
| Dashboard        | Streamlit         |
| Report Output    | Markdown + CSV    |
| Local Execution  | PowerShell / Bash |

---

## 4. Project Structure

```text
ForecastIQ/
│
├── data/
│   ├── bing_campaign_stats.csv
│   ├── google_ads_campaign_stats.csv
│   └── meta_ads_campaign_stats.csv
│
├── src/
│   ├── __init__.py
│   ├── load_data.py
│   ├── normalize_data.py
│   ├── validate_data.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│   ├── budget_simulator.py
│   └── explainability.py
│
├── app/
│   └── streamlit_app.py
│
├── docs/
│   └── project_development_documentation.md
│
├── pickle/
│   └── model.pkl
│
├── output/
│   ├── unified_marketing_data.csv
│   ├── validation_report.md
│   ├── training_features.csv
│   ├── prediction_features.csv
│   ├── training_report.md
│   ├── predictions.csv
│   ├── agency_forecast_report.md
│   ├── agency_insights.csv
│   ├── scenario_forecasts.csv
│   ├── scenario_comparison.csv
│   └── scenario_summary.md
│
├── run.sh
├── run.ps1
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 5. Dataset

The current project uses three CSV datasets:

```text
data/google_ads_campaign_stats.csv
data/bing_campaign_stats.csv
data/meta_ads_campaign_stats.csv
```

### Google Ads Data

Important columns:

```text
campaign_id
segments_date
metrics_clicks
metrics_conversions
metrics_cost_micros
metrics_impressions
metrics_conversions_value
campaign_advertising_channel_type
campaign_name
```

Important transformation:

```text
spend = metrics_cost_micros / 1,000,000
revenue = metrics_conversions_value
```

### Microsoft/Bing Ads Data

Important columns:

```text
CampaignId
TimePeriod
Revenue
Spend
Clicks
Impressions
Conversions
CampaignType
DailyBudget
CampaignName
```

### Meta Ads Data

Important columns:

```text
campaign_id
date_start
spend
clicks
impressions
conversion
daily_budget
campaign_name
```

Current assumption:

```text
Meta Ads conversion column is treated as revenue/conversion value.
```

This is because the Meta dataset does not contain a clearly separate revenue column.

---

## 6. Unified Data Schema

All platform data is normalized into this common structure:

```text
date
channel
campaign_id
campaign_name
campaign_type
daily_budget
spend
clicks
impressions
conversions
revenue
ctr
cpc
cpm
conversion_rate
roas
```

Calculated metrics:

```text
CTR = clicks / impressions
CPC = spend / clicks
CPM = (spend / impressions) * 1000
Conversion Rate = conversions / clicks
ROAS = revenue / spend
```

All calculations are protected against division-by-zero errors.

---

## 7. Machine Learning Methodology

ForecastIQ AI uses a probabilistic forecasting approach.

### Target

```text
target_future_revenue
```

ROAS is calculated after revenue prediction:

```text
predicted_roas = predicted_revenue / future_budget
```

### Forecast Windows

```text
30 days
60 days
90 days
```

### Model Type

The system trains three quantile models using `GradientBoostingRegressor`:

| Model         | Quantile | Purpose                   |
| ------------- | -------: | ------------------------- |
| `revenue_q10` |     0.10 | Low revenue estimate      |
| `revenue_q50` |     0.50 | Expected revenue estimate |
| `revenue_q90` |     0.90 | High revenue estimate     |

This allows the system to output:

```text
revenue_low
revenue_expected
revenue_high
roas_low
roas_expected
roas_high
```

---

## 8. Setup Instructions

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd ForecastIQ
```

### Step 2: Create a Virtual Environment

For Windows:

```powershell
py -3.11 -m venv .venv
```

For macOS/Linux:

```bash
python3.11 -m venv .venv
```

### Step 3: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\activate
```

For macOS/Linux:

```bash
source .venv/bin/activate
```

### Step 4: Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 9. Requirements

Your `requirements.txt` should contain:

```text
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
joblib==1.4.2
matplotlib==3.9.1
streamlit==1.37.1
protobuf==3.20.3
python-dotenv==1.0.1
```

Important:

Do not install:

```bash
pip install sklearn
```

Use:

```bash
pip install scikit-learn==1.5.1
```

The Python import name is `sklearn`, but the package name is `scikit-learn`.

---

## 10. Step-by-Step Pipeline Execution

### Step 1: Normalize Data

```bash
python -m src.normalize_data --data-dir data --output-path output/unified_marketing_data.csv
```

Generated file:

```text
output/unified_marketing_data.csv
```

---

### Step 2: Validate Data

```bash
python -m src.validate_data --data-dir data --report-path output/validation_report.md
```

Generated file:

```text
output/validation_report.md
```

The validation report checks:

* missing values,
* invalid dates,
* duplicate campaign-date rows,
* zero spend rows,
* zero revenue rows,
* negative values,
* extreme ROAS rows.

---

### Step 3: Generate ML Features

```bash
python -m src.feature_engineering --data-dir data --training-output output/training_features.csv --prediction-output output/prediction_features.csv
```

Generated files:

```text
output/training_features.csv
output/prediction_features.csv
```

---

### Step 4: Train Model

```bash
python -m src.train_model --training-features output/training_features.csv --model-path pickle/model.pkl --report-path output/training_report.md
```

Generated files:

```text
pickle/model.pkl
output/training_report.md
```

---

### Step 5: Generate Forecast Predictions

```bash
python -m src.predict --data-dir data --model pickle/model.pkl --output output/predictions.csv
```

Generated file:

```text
output/predictions.csv
```

---

### Step 6: Generate Agency-Ready Report

```bash
python -m src.explainability --predictions output/predictions.csv --scenario-comparison output/scenario_comparison.csv --report-output output/agency_forecast_report.md --insights-output output/agency_insights.csv
```

Generated files:

```text
output/agency_forecast_report.md
output/agency_insights.csv
```

---

## 11. Run the Full Pipeline

### Windows PowerShell

Use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run.ps1 ./data ./pickle/model.pkl ./output/predictions.csv
```

This generates:

```text
output/predictions.csv
output/prediction_features.csv
output/agency_forecast_report.md
output/agency_insights.csv
```

### macOS/Linux

Make the shell script executable:

```bash
chmod +x run.sh
```

Run:

```bash
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

This command is the expected hackathon-style execution flow.

---

## 12. Budget Scenario Simulation

The budget simulator allows testing future media budget changes.

Example:

```bash
python -m src.budget_simulator --data-dir data --model pickle/model.pkl --google-multiplier 1.2 --meta-multiplier 1.0 --microsoft-multiplier 1.0
```

This means:

```text
Google Ads budget +20%
Meta Ads budget unchanged
Microsoft Ads budget unchanged
```

Generated files:

```text
output/scenario_forecasts.csv
output/scenario_comparison.csv
output/scenario_summary.md
```

Another example:

```bash
python -m src.budget_simulator --data-dir data --model pickle/model.pkl --google-multiplier 1.1 --meta-multiplier 1.2 --microsoft-multiplier 0.9
```

This means:

```text
Google Ads budget +10%
Meta Ads budget +20%
Microsoft Ads budget -10%
```

---

## 13. Run the Streamlit Dashboard

Before running the dashboard, make sure these files exist:

```text
output/predictions.csv
output/agency_forecast_report.md
output/agency_insights.csv
```

Then run:

```bash
python -m streamlit run app/streamlit_app.py
```

The dashboard includes:

* 30/60/90 day forecast overview,
* channel-level forecast,
* campaign-type forecast,
* campaign-level risks,
* scaling opportunities,
* budget simulator,
* agency-ready markdown report,
* raw forecast output,
* structured insights.

Local dashboard URL:

```text
http://localhost:8501
```

---

## 14. Output Files

| File                                | Description                           |
| ----------------------------------- | ------------------------------------- |
| `output/unified_marketing_data.csv` | Normalized combined marketing dataset |
| `output/validation_report.md`       | Data validation report                |
| `output/training_features.csv`      | ML training features                  |
| `output/prediction_features.csv`    | ML prediction features                |
| `output/training_report.md`         | Model training report                 |
| `output/predictions.csv`            | Final forecast output                 |
| `output/agency_forecast_report.md`  | Client/agency-ready markdown report   |
| `output/agency_insights.csv`        | Structured insight table              |
| `output/scenario_forecasts.csv`     | Scenario forecast output              |
| `output/scenario_comparison.csv`    | Baseline vs scenario comparison       |
| `output/scenario_summary.md`        | Scenario explanation report           |

---

## 15. Prediction Output Format

The final `predictions.csv` contains:

```text
forecast_id
forecast_level
channel
campaign_type
campaign_id
campaign_name
forecast_window_days
forecast_start_date
forecast_end_date
future_budget
revenue_low
revenue_expected
revenue_high
roas_low
roas_expected
roas_high
confidence_score
risk_level
insight_summary
```

Forecast levels:

```text
blended
channel
campaign_type
campaign
```

---

## 16. Hackathon Submission Checklist

Before final submission, confirm:

```text
run.sh exists in the root folder
requirements.txt exists with pinned versions
data/ folder exists
pickle/model.pkl exists
src/ folder exists
README.md exists
run.sh accepts DATA_DIR, MODEL_PATH, and OUTPUT_PATH arguments
pipeline runs without manual input
pipeline does not use absolute local paths
pipeline does not require internet access during prediction
predictions are written to the provided OUTPUT_PATH
```

Expected final command:

```bash
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

On Windows, use `run.ps1` only for local testing. Keep `run.sh` for final hackathon submission.

---

## 17. Troubleshooting

### Error: `ModuleNotFoundError: No module named 'sklearn'`

Install scikit-learn:

```bash
python -m pip install scikit-learn==1.5.1
```

Do not run:

```bash
pip install sklearn
```

---

### Error: `bash is not recognized`

This happens on Windows if Bash is not installed.

Use:

```powershell
.\run.ps1 ./data ./pickle/model.pkl ./output/predictions.csv
```

or install Git Bash.

---

### Error: Streamlit protobuf issue

If Streamlit fails with protobuf descriptor errors, run:

```bash
python -m pip uninstall streamlit protobuf -y
python -m pip install streamlit==1.37.1 protobuf==3.20.3
```

Then launch Streamlit with:

```bash
python -m streamlit run app/streamlit_app.py
```

---

### Error: `model.pkl not found`

Train the model first:

```bash
python -m src.feature_engineering --data-dir data --training-output output/training_features.csv --prediction-output output/prediction_features.csv
python -m src.train_model --training-features output/training_features.csv --model-path pickle/model.pkl --report-path output/training_report.md
```

---

### Error: `predictions.csv not found`

Run the prediction pipeline:

```bash
python -m src.predict --data-dir data --model pickle/model.pkl --output output/predictions.csv
```

---

## 18. Current Limitations

* Meta Ads `conversion` column is treated as revenue due to missing separate revenue field.
* The current model forecasts revenue first and calculates ROAS from revenue and future budget.
* The system uses existing attribution data as-is.
* It does not implement full Media Mix Modeling.
* It does not connect directly to live ad platform APIs.
* LLM API integration is not required for the current offline-safe version.
* Forecasts are decision-support estimates, not guaranteed future outcomes.

---

## 19. Future Improvements

Possible improvements:

* Add live API connectors for Google Ads, Meta Ads, and Microsoft Ads.
* Add GA4 and Shopify integration.
* Improve Meta conversion/revenue mapping.
* Add advanced quantile calibration.
* Add campaign anomaly detection.
* Add LLM-based client report generation with optional API key.
* Add downloadable PDF reports.
* Add model comparison and backtesting dashboard.
* Add budget optimization engine to recommend best channel allocation.
* Add campaign naming quality checker.
* Add landing page and creative performance modules.

---

## 20. Project Status

```text
Data ingestion: Completed
Data normalization: Completed
Data validation: Completed
Feature engineering: Completed
Model training: Completed
Prediction generation: Completed
Budget simulation: Completed
Explainability report: Completed
Streamlit dashboard: Completed
Windows pipeline: Completed
Hackathon run.sh pipeline: Completed
README documentation: Completed
```

ForecastIQ AI is now a working end-to-end forecasting utility for e-commerce marketing agencies.
#   F o r e c a s t I Q  
 