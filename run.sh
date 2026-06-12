#!/usr/bin/env bash
set -euo pipefail

# ForecastIQ AI - Hackathon automated entry point
# Usage:
# ./run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>

DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
PREDICTION_FEATURES_PATH="$OUTPUT_DIR/prediction_features.csv"
AGENCY_REPORT_PATH="$OUTPUT_DIR/agency_forecast_report.md"
AGENCY_INSIGHTS_PATH="$OUTPUT_DIR/agency_insights.csv"
SCENARIO_COMPARISON_PATH="$OUTPUT_DIR/scenario_comparison.csv"

echo "=========================================="
echo "ForecastIQ AI - Full Prediction Pipeline"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo "Model path: $MODEL_PATH"
echo "Predictions output: $OUTPUT_PATH"
echo "Agency report output: $AGENCY_REPORT_PATH"
echo "Agency insights output: $AGENCY_INSIGHTS_PATH"
echo ""

if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: Data directory not found: $DATA_DIR"
  exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "ERROR: Model file not found: $MODEL_PATH"
  echo "Please train the model first and save it as pickle/model.pkl"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Step 1: Running probabilistic forecast prediction..."

python -m src.predict \
  --data-dir "$DATA_DIR" \
  --model "$MODEL_PATH" \
  --output "$OUTPUT_PATH" \
  --prediction-features-output "$PREDICTION_FEATURES_PATH"

echo ""
echo "Step 2: Generating agency-ready explainability report..."

python -m src.explainability \
  --predictions "$OUTPUT_PATH" \
  --scenario-comparison "$SCENARIO_COMPARISON_PATH" \
  --report-output "$AGENCY_REPORT_PATH" \
  --insights-output "$AGENCY_INSIGHTS_PATH"

echo ""
echo "Pipeline completed successfully."
echo "Predictions written to: $OUTPUT_PATH"
echo "Prediction features written to: $PREDICTION_FEATURES_PATH"
echo "Agency report written to: $AGENCY_REPORT_PATH"
echo "Agency insights written to: $AGENCY_INSIGHTS_PATH"
echo "=========================================="