import numpy as np
from src.config import Config


class PSIDriftDetector:
    def __init__(self, n_bins: int = None, threshold: float = None, eps: float = 1e-6):
        # initializing population stability index drift monitor parameters from config
        self.n_bins = n_bins if n_bins is not None else Config.PSI_NUM_BINS
        self.threshold = threshold if threshold is not None else Config.PSI_THRESHOLD
        self.eps = eps

    def _compute_feature_psi(self, ref_col: np.ndarray, new_col: np.ndarray) -> float:
        # computing psi for a single numeric feature using reference quantile bins
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
        bin_edges = np.quantile(ref_col, quantiles)
        bin_edges = np.unique(bin_edges)

        # returning zero drift if reference feature has constant values
        if len(bin_edges) < 3:
            return 0.0

        # extending outer bin edges to capture out-of-range values in incoming traffic
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        ref_counts, _ = np.histogram(ref_col, bins=bin_edges)
        new_counts, _ = np.histogram(new_col, bins=bin_edges)

        ref_props = np.clip(ref_counts / max(len(ref_col), 1), self.eps, 1.0)
        new_props = np.clip(new_counts / max(len(new_col), 1), self.eps, 1.0)

        psi_val = np.sum((new_props - ref_props) * np.log(new_props / ref_props))
        return float(max(0.0, psi_val))

    def detect_drift(self, reference_normal: np.ndarray, incoming_normal: np.ndarray) -> dict:
        # comparing incoming normal traffic against stored normal samples across all features
        if len(reference_normal) < 10 or len(incoming_normal) < 10:
            return {
                "psi_score": 0.0,
                "max_feature_psi": 0.0,
                "drift_detected": False,
                "severity": "stable",
            }

        num_features = reference_normal.shape[1]
        feature_psis = [
            self._compute_feature_psi(reference_normal[:, j], incoming_normal[:, j])
            for j in range(num_features)
        ]

        mean_psi = float(np.mean(feature_psis))
        max_psi = float(np.max(feature_psis))
        drift_detected = bool(mean_psi >= self.threshold)

        if mean_psi < 0.10:
            severity = "stable"
        elif mean_psi < self.threshold:
            severity = "moderate_shift"
        else:
            severity = "significant_drift"

        return {
            "psi_score": mean_psi,
            "max_feature_psi": max_psi,
            "drift_detected": drift_detected,
            "severity": severity,
        }
