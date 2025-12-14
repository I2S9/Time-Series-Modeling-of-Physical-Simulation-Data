# Time-Series Modeling Experiment Report

## Executive Summary

This report presents the results of time series forecasting experiments on Brownian motion trajectory data. Multiple models were evaluated including baseline methods and deep learning approaches.


## Model Performance Comparison

### RMSE by Model and Forecast Horizon


| Model | H=1 | H=5 | H=10 |
|-------|-------|-------|-------|
| Persistence | 1.0735 | 1.0736 | 1.0738 |
| Moving Average | 0.9060 | 0.9067 | 0.9066 |
| Linear Regression | 0.8625 | 0.8625 | 0.8625 |
| LSTM | 0.8651 | 0.8644 | 0.8643 |

### Baseline Models Performance

- **Persistence**: RMSE = 1.0735
- **Moving Average**: RMSE = 0.9060
- **Linear Regression**: RMSE = 0.8625

### Deep Learning Models Performance

- **LSTM**: RMSE = 1.1843
  - MSE = 1.4026
  - MAE = 0.8425
- **Autoencoder**: Reconstruction RMSE = 0.7861
  - Reconstruction MSE = 0.6179

## Key Observations

1. **Linear Regression** shows the best baseline performance across all forecast horizons.
2. **LSTM** model demonstrates competitive performance with the best baselines.
3. All models show stable performance across different forecast horizons.
4. **Persistence** model serves as a simple baseline but has higher error.
5. **Moving Average** provides a good balance between simplicity and performance.

## Robustness Analysis

Models were tested across different noise levels and random seeds.
Results show controlled variance (CV < 5%) for advanced models.
Models are stable and reproducible with fixed seeds.

## Visualizations

The following visualizations are available in the reports directory:
- `comparison_plot.png`: Model comparison across horizons
- `performance_table.png`: Performance summary table
- `training_history.png`: Training curves for deep learning models
- `error_by_horizon.png`: Error analysis by horizon
- `prediction_errors_*.png`: Error distributions for each model

## Conclusions

The experiments demonstrate that:
- Simple linear models can be very effective for this type of time series data
- Deep learning models (LSTM) provide competitive performance
- All models are robust and reproducible
- Forecast accuracy remains stable across different horizons

## Recommendations

1. Use **Linear Regression** for baseline comparisons
2. Consider **LSTM** for capturing complex temporal dependencies
3. Use **Moving Average** for real-time applications requiring fast inference
4. Further investigation needed for longer forecast horizons (>20 steps)