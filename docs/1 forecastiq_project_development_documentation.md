# ForecastIQ AI - Project Development Documentation

## 1. Project Overview

**Project Name:** ForecastIQ AI  
**Hackathon:** NetElixir AIgnition 3.0  
**Problem Type:** Probabilistic Revenue and ROAS Forecasting for E-commerce Marketing  
**Target Users:** Digital marketing agencies, e-commerce marketers, campaign analysts, and growth teams.

ForecastIQ AI is an AI-assisted forecasting utility designed to help digital marketing agencies predict future e-commerce marketing performance using historical ad platform data.

The system currently works with three campaign datasets:

1. `google_ads_campaign_stats.csv`
2. `bing_campaign_stats.csv`
3. `meta_ads_campaign_stats.csv`

The main goal is to generate future forecasts for:

- Expected e-commerce revenue
- Expected ROAS
- Revenue low / expected / high ranges
- ROAS low / expected / high ranges
- Channel-level forecasts
- Campaign-type-level forecasts
- Campaign-level forecasts
- 30, 60, and 90 day forecasting windows

The system is designed according to the hackathon submission requirement where the evaluator can clone the repository, replace the `data/` folder with test data, run one command, and collect the predictions output.

---

## 2. What We Have Developed Till Now

Till now, we have completed the core machine learning pipeline foundation.

| File | Purpose | Status |
|---|---|---|
| `src/load_data.py` | Loads CSV files from the `data/` folder | Completed |
| `src/normalize_data.py` | Converts Google Ads, Meta Ads, and Bing Ads data into one unified schema | Completed |
| `src/validate_data.py` | Checks missing values, duplicates, zero rows, invalid values, and ROAS outliers | Completed |
| `src/feature_engineering.py` | Creates ML-ready training and prediction features for 30/60/90 day forecasting | Completed |
| `src/train_model.py` | Trains probabilistic revenue forecasting models and saves `pickle/model.pkl` | Completed |
| `src/predict.py` | Loads the trained model and generates forecast predictions | Completed |
| `run.sh` | Linux/macOS automated pipeline entry point for hackathon submission | Completed |
| `run.ps1` | Windows PowerShell local testing entry point | Completed / Recommended |
| `requirements.txt` | Pinned Python dependencies | Completed |
| `output/` | Stores generated feature files, reports, and predictions | Generated during execution |
| `pickle/model.pkl` | Trained ML model artifact | Generated after training |

---

## 3. Current Project Folder Structure

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
│   └── predict.py
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
│   └── predictions.csv
│
├── docs/
│   └── project_development_documentation.md
│
├── app/
│   └── streamlit_app.py
│
├── run.sh
├── run.ps1
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 4. Dataset Used

We are currently using only the three datasets provided by the hackathon organizers.

### 4.1 Google Ads Dataset

**File:** `google_ads_campaign_stats.csv`

Important fields:

- `campaign_id`
- `segments_date`
- `metrics_clicks`
- `metrics_conversions`
- `metrics_cost_micros`
- `metrics_impressions`
- `metrics_video_views`
- `metrics_conversions_value`
- `campaign_advertising_channel_type`
- `campaign_name`

Important transformation:

```text
spend = metrics_cost_micros / 1,000,000
revenue = metrics_conversions_value
campaign_type = campaign_advertising_channel_type
```

### 4.2 Bing / Microsoft Ads Dataset

**File:** `bing_campaign_stats.csv`

Important fields:

- `CampaignId`
- `TimePeriod`
- `Revenue`
- `Spend`
- `Clicks`
- `Impressions`
- `Conversions`
- `CampaignType`
- `DailyBudget`
- `CampaignName`

Important transformation:

```text
channel = Microsoft Ads
revenue = Revenue
spend = Spend
campaign_type = CampaignType
```

### 4.3 Meta Ads Dataset

**File:** `meta_ads_campaign_stats.csv`

Important fields:

- `campaign_id`
- `date_start`
- `cpc`
- `cpm`
- `ctr`
- `reach`
- `spend`
- `clicks`
- `impressions`
- `conversion`
- `daily_budget`
- `campaign_name`

Important assumption:

The Meta dataset has a column named `conversion`, but the values behave like revenue/conversion value. So, for the current model:

```text
revenue = conversion
conversions = 0
```

This assumption must be clearly mentioned in the final technical documentation.

---

## 5. Unified Data Schema

All platform datasets are normalized into this common schema:

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

All calculations are protected against division by zero.

---

## 6. Machine Learning Approach

The current model is a probabilistic revenue forecasting model.

### Target Variable

```text
target_future_revenue
```

ROAS is calculated after revenue prediction:

```text
predicted_roas = predicted_revenue / future_budget
```

### Forecast Windows

The model supports:

```text
30 days
60 days
90 days
```

### Model Type

We use three `GradientBoostingRegressor` models from scikit-learn with quantile loss:

| Model | Quantile | Meaning |
|---|---:|---|
| `revenue_q10` | 0.10 | Low revenue forecast |
| `revenue_q50` | 0.50 | Expected revenue forecast |
| `revenue_q90` | 0.90 | High revenue forecast |

This allows the system to output probabilistic ranges instead of a single fixed number.

---

## 7. Feature Engineering Strategy

The file `src/feature_engineering.py` creates ML-ready rows using campaign-level historical data.

### Rolling Historical Windows

The system calculates historical features for:

```text
7 days
14 days
30 days
60 days
90 days
```

Example features:

```text
hist_7d_spend
hist_7d_revenue
hist_7d_roas
hist_30d_spend
hist_30d_revenue
hist_30d_roas
hist_90d_spend
hist_90d_revenue
hist_90d_roas
spend_trend_7d_vs_30d
revenue_trend_7d_vs_30d
roas_trend_7d_vs_30d
forecast_window_days
future_budget
future_budget_per_day
campaign_age_days
forecast_month
forecast_quarter
forecast_year
```

Training logic:

```text
Past 30/60/90 day performance -> Future 30/60/90 day revenue target
```

Prediction logic:

```text
Latest available campaign history -> Next 30/60/90 day forecast
```

---

## 8. Model Output

The final prediction output is saved to:

```text
output/predictions.csv
```

Output columns:

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

This means the output includes:

1. Overall blended forecast
2. Channel-level forecast
3. Campaign-type-level forecast
4. Individual campaign-level forecast

---

## 9. Step-by-Step Setup Guide

### Step 1: Open the Project Folder

```powershell
cd C:\Users\smith\Documents\SMITH\College\MCA\Summer\ForecastIQ
```

### Step 2: Create Virtual Environment

```powershell
py -3.11 -m venv .venv
```

### Step 3: Activate Virtual Environment

```powershell
.\.venv\Scripts\activate
```

You should see:

```text
(.venv)
```

before the terminal path.

### Step 4: Upgrade pip

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### Step 5: Install Dependencies

```powershell
pip install -r requirements.txt
```

If scikit-learn is missing, install it directly:

```powershell
python -m pip install scikit-learn==1.5.1
```

Do not install `sklearn`. The correct package name is:

```text
scikit-learn
```

Verify installation:

```powershell
python -c "import sklearn; print(sklearn.__version__)"
```

Expected output:

```text
1.5.1
```

---

## 10. Step-by-Step Running Guide

### Step 1: Test Data Normalization

```powershell
python -m src.normalize_data --data-dir data --output-path output/unified_marketing_data.csv
```

Generated file:

```text
output/unified_marketing_data.csv
```

### Step 2: Run Data Validation

```powershell
python -m src.validate_data --data-dir data --report-path output/validation_report.md
```

Generated file:

```text
output/validation_report.md
```

The validation report contains:

- total rows
- date range
- channels
- campaign count
- zero spend rows
- zero revenue rows
- duplicate rows
- extreme ROAS observations

### Step 3: Generate ML Features

```powershell
python -m src.feature_engineering --data-dir data --training-output output/training_features.csv --prediction-output output/prediction_features.csv
```

Generated files:

```text
output/training_features.csv
output/prediction_features.csv
```

### Step 4: Train the Model

```powershell
python -m src.train_model --training-features output/training_features.csv --model-path pickle/model.pkl --report-path output/training_report.md
```

Generated files:

```text
pickle/model.pkl
output/training_report.md
```

### Step 5: Generate Predictions

```powershell
python -m src.predict --data-dir data --model pickle/model.pkl --output output/predictions.csv
```

Generated file:

```text
output/predictions.csv
```

---

## 11. Windows Local Pipeline Command

Because Windows PowerShell does not always support `bash`, use the PowerShell script:

```text
run.ps1
```

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run.ps1 ./data ./pickle/model.pkl ./output/predictions.csv
```

This runs the prediction pipeline locally on Windows.

---

## 12. Hackathon Submission Pipeline

The hackathon evaluator is expected to run the Linux/macOS shell script:

```text
run.sh
```

Expected command:

```bash
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

On Windows, if Bash is not installed, this command will not work:

```powershell
bash run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

That is normal. It only means Bash is not available on the local Windows system.

For local Windows testing, use:

```powershell
.\run.ps1 ./data ./pickle/model.pkl ./output/predictions.csv
```

For final hackathon submission, keep `run.sh` in the root directory.

---

## 13. Important Troubleshooting Notes

### Problem: `ModuleNotFoundError: No module named 'sklearn'`

Solution:

```powershell
python -m pip install scikit-learn==1.5.1
```

Do not use:

```powershell
pip install sklearn
```

`sklearn` is only the Python import name. The package name is `scikit-learn`.

### Problem: `bash is not recognized`

This means Bash is not installed on Windows.

Use:

```powershell
python -m src.predict --data-dir data --model pickle/model.pkl --output output/predictions.csv
```

or:

```powershell
.\run.ps1 ./data ./pickle/model.pkl ./output/predictions.csv
```

Optional fix:

```powershell
winget install --id Git.Git -e --source winget
```

Then reopen PowerShell or VS Code.

### Problem: `ImportError: cannot import name normalize_all_platforms`

This means `src/normalize_data.py` does not contain the function:

```python
normalize_all_platforms
```

Solution:

Make sure `src/normalize_data.py` includes:

```python
def normalize_all_platforms(data_dir: str | Path) -> pd.DataFrame:
```

Also make sure this file exists:

```text
src/__init__.py
```

---

## 14. Files That Must Be Included in Final Submission

Before submission, confirm these files exist:

```text
run.sh
requirements.txt
README.md
data/
pickle/model.pkl
src/
```

Also confirm:

```text
run.sh is in the root folder
model.pkl is inside pickle/
requirements.txt has pinned versions
output/predictions.csv is created successfully
no code depends on absolute local paths
no code asks for interactive input
```

---

## 15. Current Limitations

The current version has some limitations:

1. Meta Ads `conversion` column is treated as revenue because no separate revenue column is available.
2. The model predicts revenue first, then calculates ROAS from revenue and future budget.
3. Current future budget estimation uses recent daily budget or recent spend when no manual input is provided.
4. No live API integration is added yet.
5. No full attribution modeling or MMM is implemented, because the hackathon scope focuses on existing attribution data.
6. LLM-based insight generation is not added yet.
7. Streamlit dashboard is not added yet.
8. Budget simulator is not added yet.

---

## 16. Next Development Steps

### 1. `src/budget_simulator.py`

Purpose:

```text
Increase Google Ads budget by 20%
Decrease Meta Ads budget by 10%
Compare current vs scenario forecast
```

### 2. `src/explainability.py`

Purpose:

```text
Explain why revenue is expected to increase or decline
Identify risky channels
Identify top revenue drivers
Generate agency-friendly forecast reasoning
```

### 3. `app/streamlit_app.py`

Purpose:

```text
Data overview
Forecast dashboard
Budget simulator
Channel breakdown
AI insights
```

### 4. More Documentation

Recommended docs:

```text
docs/methodology.md
docs/architecture.md
docs/assumptions_limitations.md
docs/demo_workflow.md
```

---

## 17. Final Current Status

```text
Data pipeline: Completed
Data normalization: Completed
Data validation: Completed
Feature engineering: Completed
Model training: Completed
Prediction generation: Completed
Automated run.sh entry point: Completed
Windows run.ps1 testing file: Completed
Budget simulator: Pending
AI explanation layer: Pending
Streamlit dashboard: Pending
Final documentation: In progress
```

ForecastIQ AI is now at the stage where the core ML forecasting pipeline is working. The next focus should be budget simulation, explainability, and the demo dashboard.
