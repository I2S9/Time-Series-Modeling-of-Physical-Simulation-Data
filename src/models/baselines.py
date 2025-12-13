"""Classical baseline models for time series forecasting."""

import numpy as np
from sklearn.linear_model import LinearRegression


class PersistenceModel:
    """Persistence model that predicts the last observed value.
    
    This is the simplest baseline: it assumes the next value equals the current value.
    """
    
    def __init__(self):
        """Initialize the persistence model."""
        self.last_value = None
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "PersistenceModel":
        """Fit the model (no training needed for persistence).
        
        Args:
            X: Input features with shape (n_samples, n_features).
            y: Target values (optional, not used).
        
        Returns:
            Self for method chaining.
        """
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on empty data")
        self.last_value = X[-1].copy()
        return self
    
    def predict(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Predict future values using persistence.
        
        Args:
            X: Input features with shape (n_samples, n_features).
            horizon: Number of steps ahead to predict.
        
        Returns:
            Predictions with shape (horizon, n_features).
        """
        if self.last_value is None:
            if X.shape[0] == 0:
                raise ValueError("Cannot predict: model not fitted and no input data")
            self.last_value = X[-1].copy()
        
        predictions = np.tile(self.last_value, (horizon, 1))
        return predictions
    
    def forecast(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Forecast future values (alias for predict).
        
        Args:
            X: Input features with shape (n_samples, n_features).
            horizon: Number of steps ahead to forecast.
        
        Returns:
            Forecasts with shape (horizon, n_features).
        """
        return self.predict(X, horizon)


class MovingAverageModel:
    """Moving average model that predicts the mean of recent observations.
    
    Uses a sliding window to compute the average of the last n observations.
    """
    
    def __init__(self, window_size: int = 10):
        """Initialize the moving average model.
        
        Args:
            window_size: Number of recent observations to average.
        """
        if window_size < 1:
            raise ValueError("Window size must be at least 1")
        self.window_size = window_size
        self.history = None
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "MovingAverageModel":
        """Fit the model by storing recent history.
        
        Args:
            X: Input features with shape (n_samples, n_features).
            y: Target values (optional, not used).
        
        Returns:
            Self for method chaining.
        """
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on empty data")
        self.history = X[-self.window_size:].copy()
        return self
    
    def predict(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Predict future values using moving average.
        
        Args:
            X: Input features with shape (n_samples, n_features).
            horizon: Number of steps ahead to predict.
        
        Returns:
            Predictions with shape (horizon, n_features).
        """
        if X.shape[0] == 0:
            if self.history is None:
                raise ValueError("Cannot predict: model not fitted and no input data")
            window_data = self.history
        else:
            window_data = X[-self.window_size:]
        
        mean_value = np.mean(window_data, axis=0)
        predictions = np.tile(mean_value, (horizon, 1))
        return predictions
    
    def forecast(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Forecast future values (alias for predict).
        
        Args:
            X: Input features with shape (n_samples, n_features).
            horizon: Number of steps ahead to forecast.
        
        Returns:
            Forecasts with shape (horizon, n_features).
        """
        return self.predict(X, horizon)


class LinearRegressionModel:
    """Linear regression model for time series forecasting.
    
    Uses linear regression to predict future values based on recent history.
    """
    
    def __init__(self, lookback_window: int = 10):
        """Initialize the linear regression model.
        
        Args:
            lookback_window: Number of past time steps to use as features.
        """
        if lookback_window < 1:
            raise ValueError("Lookback window must be at least 1")
        self.lookback_window = lookback_window
        self.model = LinearRegression()
        self.is_fitted = False
    
    def _create_features(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create features and targets from time series data.
        
        Args:
            X: Input time series with shape (n_samples, n_features).
        
        Returns:
            Tuple of (features, targets) where features have shape
            (n_samples - lookback_window, lookback_window * n_features)
            and targets have shape (n_samples - lookback_window, n_features).
        """
        n_samples, n_features = X.shape
        
        if n_samples < self.lookback_window + 1:
            raise ValueError(
                f"Need at least {self.lookback_window + 1} samples, got {n_samples}"
            )
        
        n_windows = n_samples - self.lookback_window
        features = np.zeros((n_windows, self.lookback_window * n_features))
        targets = np.zeros((n_windows, n_features))
        
        for i in range(n_windows):
            features[i] = X[i : i + self.lookback_window].flatten()
            targets[i] = X[i + self.lookback_window]
        
        return features, targets
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "LinearRegressionModel":
        """Fit the linear regression model.
        
        Args:
            X: Input features with shape (n_samples, n_features).
            y: Target values (optional, not used - targets are derived from X).
        
        Returns:
            Self for method chaining.
        """
        features, targets = self._create_features(X)
        self.model.fit(features, targets)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Predict future values using linear regression.
        
        For multi-step ahead prediction, uses iterative forecasting.
        
        Args:
            X: Input features with shape (n_samples, n_features).
            horizon: Number of steps ahead to predict.
        
        Returns:
            Predictions with shape (horizon, n_features).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if X.shape[0] < self.lookback_window:
            raise ValueError(
                f"Need at least {self.lookback_window} samples for prediction, "
                f"got {X.shape[0]}"
            )
        
        predictions = np.zeros((horizon, X.shape[1]))
        current_window = X[-self.lookback_window:].copy()
        
        for h in range(horizon):
            features = current_window.flatten().reshape(1, -1)
            pred = self.model.predict(features)[0]
            predictions[h] = pred
            
            if h < horizon - 1:
                current_window = np.roll(current_window, -1, axis=0)
                current_window[-1] = pred
        
        return predictions
    
    def forecast(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Forecast future values (alias for predict).
        
        Args:
            X: Input features with shape (n_samples, n_features).
            horizon: Number of steps ahead to forecast.
        
        Returns:
            Forecasts with shape (horizon, n_features).
        """
        return self.predict(X, horizon)

