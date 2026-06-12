param(
    [string]$DATA_DIR = "./data",
    [string]$MODEL_PATH = "./pickle/model.pkl",
    [string]$OUTPUT_PATH = "./output/predictions.csv"
)

Write-Host "=========================================="
Write-Host "ForecastIQ AI - Windows Full Pipeline"
Write-Host "=========================================="
Write-Host "Data directory: $DATA_DIR"
Write-Host "Model path: $MODEL_PATH"
Write-Host "Predictions output: $OUTPUT_PATH"
Write-Host ""

if (!(Test-Path $DATA_DIR)) {
    Write-Host "ERROR: Data directory not found: $DATA_DIR"
    exit 1
}

if (!(Test-Path $MODEL_PATH)) {
    Write-Host "ERROR: Model file not found: $MODEL_PATH"
    Write-Host "Please train the model first and save it as pickle/model.pkl"
    exit 1
}

$OUTPUT_DIR = Split-Path -Parent $OUTPUT_PATH

if ([string]::IsNullOrWhiteSpace($OUTPUT_DIR)) {
    $OUTPUT_DIR = "."
}

if (!(Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR | Out-Null
}

$PREDICTION_FEATURES_PATH = Join-Path $OUTPUT_DIR "prediction_features.csv"
$AGENCY_REPORT_PATH = Join-Path $OUTPUT_DIR "agency_forecast_report.md"
$AGENCY_INSIGHTS_PATH = Join-Path $OUTPUT_DIR "agency_insights.csv"
$SCENARIO_COMPARISON_PATH = Join-Path $OUTPUT_DIR "scenario_comparison.csv"

Write-Host "Agency report output: $AGENCY_REPORT_PATH"
Write-Host "Agency insights output: $AGENCY_INSIGHTS_PATH"
Write-Host ""

Write-Host "Step 1: Running probabilistic forecast prediction..."

python -m src.predict `
    --data-dir $DATA_DIR `
    --model $MODEL_PATH `
    --output $OUTPUT_PATH `
    --prediction-features-output $PREDICTION_FEATURES_PATH

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Prediction pipeline failed."
    exit 1
}

Write-Host ""
Write-Host "Step 2: Generating agency-ready explainability report..."

python -m src.explainability `
    --predictions $OUTPUT_PATH `
    --scenario-comparison $SCENARIO_COMPARISON_PATH `
    --report-output $AGENCY_REPORT_PATH `
    --insights-output $AGENCY_INSIGHTS_PATH

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Explainability report generation failed."
    exit 1
}

Write-Host ""
Write-Host "Pipeline completed successfully."
Write-Host "Predictions written to: $OUTPUT_PATH"
Write-Host "Prediction features written to: $PREDICTION_FEATURES_PATH"
Write-Host "Agency report written to: $AGENCY_REPORT_PATH"
Write-Host "Agency insights written to: $AGENCY_INSIGHTS_PATH"
Write-Host "=========================================="