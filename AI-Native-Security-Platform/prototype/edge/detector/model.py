"""Edge Anomaly Detector using Isolation Forest."""
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Tuple, List

class EdgeAnomalyDetector:
    def __init__(self, contamination: float = 0.05, n_estimators: int = 50, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit_baseline(self, normal_telemetry_features: np.ndarray):
        """Fit model on normal campus network baseline features."""
        self.model.fit(normal_telemetry_features)
        self.is_fitted = True

    def predict_anomaly(self, feature_vector: np.ndarray) -> Tuple[bool, float]:
        """Predicts anomaly status and computes normalized anomaly score [0.0, 1.0]."""
        if not self.is_fitted:
            # Fallback bootstrap if not explicitly trained
            dummy_data = np.random.normal(loc=10.0, scale=2.0, size=(100, feature_vector.shape[0]))
            self.model.fit(dummy_data)
            self.is_fitted = True

        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        # score_samples returns negative anomaly score (lower means more anomalous)
        raw_score = self.model.score_samples(feature_vector)[0]
        # Normalize raw_score to [0, 1] range where 1.0 is highly anomalous
        normalized_score = float(np.clip(0.5 - raw_score, 0.0, 1.0))
        is_anomaly = bool(normalized_score > 0.65)

        return is_anomaly, normalized_score
