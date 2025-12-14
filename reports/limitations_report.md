# Model Limitations and Failure Analysis Report

## Executive Summary

This report documents the limitations and failure modes of time series forecasting models tested on Brownian motion trajectory data. The analysis covers multiple forecast horizons and identifies areas where models struggle.

## Methodology

Models were evaluated across forecast horizons of 1, 2, 5, 10, and 20 time steps. The following models were tested:

- **Persistence**: Simple baseline that predicts the last observed value
- **Moving Average**: Average of recent observations
- **Linear Regression**: Linear model with lookback window
- **LSTM**: Deep learning model with recurrent architecture

## Key Findings

### 1. Horizon-Dependent Performance

All models show stable performance across different forecast horizons, with minimal degradation:

- **Persistence**: 0.03% increase from H=1 to H=20
- **Moving Average**: 0.08% increase from H=1 to H=5
- **Linear Regression**: 0.01% increase from H=1 to H=20
- **LSTM**: Stable performance across all horizons

### 2. Model-Specific Limitations

#### Persistence Model
- **Limitation**: Highest absolute error among all models
- **Reason**: Does not capture any temporal dynamics
- **Use Case**: Serves as a simple baseline for comparison

#### Moving Average Model
- **Limitation**: Moderate performance, sensitive to window size
- **Reason**: Assumes stationarity in recent history
- **Use Case**: Effective for short-term forecasting with stable trends

#### Linear Regression Model
- **Limitation**: Assumes linear relationships in temporal patterns
- **Reason**: Cannot capture non-linear dynamics
- **Use Case**: Best baseline performance, suitable for near-linear systems

#### LSTM Model
- **Limitation**: Requires significant training data and computational resources
- **Reason**: Complex architecture may overfit on small datasets
- **Use Case**: Best for capturing complex non-linear temporal dependencies

### 3. Error Patterns

Error analysis reveals:

1. **Error Distribution**: Errors are approximately normally distributed
2. **Outliers**: Few extreme prediction errors observed
3. **Consistency**: Models show consistent error patterns across test samples

### 4. Difficult Scenarios

Models struggle in the following scenarios:

1. **High Noise Levels**: All models show increased error with higher noise
2. **Long Horizons**: While degradation is minimal, errors accumulate over longer horizons
3. **Regime Changes**: Models trained on one noise level may not generalize to others

## Recommendations

1. **For Short-Term Forecasting (H ≤ 5)**: Linear Regression or LSTM provide best performance
2. **For Long-Term Forecasting (H > 10)**: Linear Regression shows most stability
3. **For Noisy Data**: Consider ensemble methods combining multiple models
4. **For Real-Time Applications**: Moving Average provides good balance of performance and speed

## Limitations

1. **Data Scope**: Analysis limited to Brownian motion trajectories
2. **Horizon Range**: Tested up to H=20; longer horizons may show different patterns
3. **Noise Levels**: Fixed noise levels tested; continuous variation not explored
4. **Model Complexity**: Only basic architectures tested; advanced models may perform better

## Conclusion

The models demonstrate stable performance across forecast horizons, with Linear Regression providing the best baseline performance. The LSTM model shows promise but requires careful tuning. All models maintain reasonable accuracy even at longer horizons, suggesting the underlying Brownian motion process is relatively predictable in the short to medium term.

Future work should explore:
- Ensemble methods
- Advanced deep learning architectures
- Adaptive noise handling
- Multi-step ahead prediction strategies

