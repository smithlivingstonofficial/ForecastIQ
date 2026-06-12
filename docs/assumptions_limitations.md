# Assumptions and Limitations

## Current assumptions

1. Google Ads `metrics_cost_micros` is converted to spend by dividing by 1,000,000.
2. Google Ads `metrics_conversions_value` is treated as revenue.
3. Bing/Microsoft Ads `Revenue` is treated as revenue.
4. Meta Ads `conversion` is treated as revenue/conversion value because the observed values behave like monetary value rather than count.
5. Meta Ads conversion count is not separately available in the current dataset, so `conversions` is set to 0.0 for Meta in the unified schema.
6. Meta campaign type is derived from `campaign_name` using naming patterns such as Prospecting, Remarketing, and Generic.

## Known limitations

- This stage only performs data loading and normalization.
- Data validation, feature engineering, model training, and probabilistic forecasting will be added in later steps.
- The temporary `run.sh` is for development verification only; it will be replaced with the final prediction pipeline.
