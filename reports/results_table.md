# Model Evaluation Results

## RMSE by Model and Forecast Horizon


| Model | H=1 | H=5 | H=10 |
|-------|-------|-------|-------|
| Persistence | 1.0735 | 1.0736 | 1.0738 |
| Moving Average | 0.9060 | 0.9067 | 0.9066 |
| Linear Regression | 0.8625 | 0.8625 | 0.8625 |
| LSTM | 0.8651 | 0.8644 | 0.8643 |

## MSE by Model and Forecast Horizon


| Model | H=1 | H=5 | H=10 |
|-------|-------|-------|-------|
| Persistence | 1.1524 | 1.1527 | 1.1530 |
| Moving Average | 0.8208 | 0.8222 | 0.8220 |
| Linear Regression | 0.7439 | 0.7439 | 0.7439 |
| LSTM | 0.7483 | 0.7472 | 0.7469 |

## Improvement


LSTM improvement vs best baseline (H=1): -0.30%


Best baseline RMSE (H=1): 0.8625

LSTM RMSE (H=1): 0.8651
