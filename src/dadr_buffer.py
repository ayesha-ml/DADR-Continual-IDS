import numpy as np
from src.config import Config
from src.drift_detector import PSIDriftDetector


class DADRReplayBuffer:
    def __init__(
        self,
        total_capacity: int = None,
        attack_ratio: float = None,
        normal_idx: int = 0,
        seed: int = None,
    ):
        # initializing dual-pool replay buffer capacities and drift detector from config
        self.total_capacity = (
            total_capacity if total_capacity is not None else Config.TOTAL_MEMORY_BUDGET
        )
        self.attack_ratio = (
            attack_ratio if attack_ratio is not None else Config.ATTACK_POOL_RATIO
        )
        self.normal_idx = normal_idx

        self.attack_capacity = int(round(self.total_capacity * self.attack_ratio))
        self.normal_capacity = self.total_capacity - self.attack_capacity

        self.rng = np.random.RandomState(
            seed if seed is not None else Config.SEEDS[0]
        )
        self.drift_detector = PSIDriftDetector()

        # storing attack samples per class id and normal samples in dedicated arrays
        self.attack_pools = {}
        self.normal_X = None
        self.normal_y = None
        self.normal_seen_count = 0

    def _compute_mahalanobis_distances(
        self, samples: np.ndarray, reference_dist: np.ndarray
    ) -> np.ndarray:
        # computing regularized mahalanobis distance of stored normal flows from new distribution
        mean_vec = np.mean(reference_dist, axis=0)
        cov_matrix = np.cov(reference_dist, rowvar=False)

        # handling single-feature edge case and adding ridge regularization from config
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[float(cov_matrix)]], dtype=np.float64)
        reg_cov = (
            cov_matrix
            + np.eye(cov_matrix.shape[0], dtype=np.float64) * Config.MAHALANOBIS_REG
        )

        inv_cov = np.linalg.pinv(reg_cov)
        diff = samples.astype(np.float64) - mean_vec.astype(np.float64)
        left_term = diff @ inv_cov
        sq_distances = np.sum(left_term * diff, axis=1)
        return np.sqrt(np.maximum(sq_distances, 0.0))

    def _update_attack_pool(self, X_att: np.ndarray, y_att: np.ndarray):
        # maintaining class-balanced storage across all seen attack types
        if len(X_att) == 0:
            return

        unique_attacks = np.unique(y_att)
        for cls_id in unique_attacks:
            cls_id = int(cls_id)
            cls_X = X_att[y_att == cls_id]
            if cls_id in self.attack_pools:
                self.attack_pools[cls_id] = np.vstack([self.attack_pools[cls_id], cls_X])
            else:
                self.attack_pools[cls_id] = cls_X.copy()

        # rebalancing per-class quotas equally within attack pool budget
        num_attack_classes = len(self.attack_pools)
        base_quota = self.attack_capacity // num_attack_classes
        remainder = self.attack_capacity % num_attack_classes

        for idx, cls_id in enumerate(sorted(self.attack_pools.keys())):
            quota = base_quota + (1 if idx < remainder else 0)
            pool_X = self.attack_pools[cls_id]
            if len(pool_X) > quota:
                keep_indices = self.rng.choice(len(pool_X), size=quota, replace=False)
                self.attack_pools[cls_id] = pool_X[keep_indices]

    def _reservoir_update_normal(self, X_norm: np.ndarray):
        # performing standard reservoir sampling when normal traffic remains stable
        for row in X_norm:
            self.normal_seen_count += 1
            if self.normal_X is None:
                self.normal_X = row.reshape(1, -1).copy()
                self.normal_y = np.array([self.normal_idx], dtype=np.int64)
            elif len(self.normal_X) < self.normal_capacity:
                self.normal_X = np.vstack([self.normal_X, row.reshape(1, -1)])
                self.normal_y = np.append(self.normal_y, self.normal_idx)
            else:
                replace_idx = self.rng.randint(0, self.normal_seen_count)
                if replace_idx < self.normal_capacity:
                    self.normal_X[replace_idx] = row

    def _evict_and_refresh_normal(self, X_norm: np.ndarray):
        # evicting stale normal flows via mahalanobis distance when drift is detected
        if self.normal_X is None or len(self.normal_X) == 0:
            self._reservoir_update_normal(X_norm)
            return

        # replacing up to half the normal pool (or incoming count) with fresh normal flows
        num_to_replace = min(
            len(X_norm), max(1, int(self.normal_capacity * 0.9))
        )

        # ranking stored normal samples by distance from incoming normal distribution
        distances = self._compute_mahalanobis_distances(self.normal_X, X_norm)
        evict_indices = np.argsort(distances)[-num_to_replace:]

        # selecting fresh incoming normal samples to replace evicted outliers
        if len(X_norm) > num_to_replace:
            fresh_indices = self.rng.choice(
                len(X_norm), size=num_to_replace, replace=False
            )
            fresh_samples = X_norm[fresh_indices]
        else:
            fresh_samples = X_norm.copy()

        self.normal_X[evict_indices] = fresh_samples
        self.normal_seen_count += len(X_norm)

    def update(self, X: np.ndarray, y: np.ndarray) -> dict:
        # updating attack pool and drift-aware normal pool for the current window
        normal_mask = y == self.normal_idx
        X_norm = X[normal_mask]
        X_att = X[~normal_mask]
        y_att = y[~normal_mask]

        # always updating class-balanced attack pool
        self._update_attack_pool(X_att, y_att)

        # checking for normal traffic drift before updating normal pool
        drift_info = {
            "psi_score": 0.0,
            "max_feature_psi": 0.0,
            "drift_detected": False,
            "severity": "stable",
        }

        if len(X_norm) > 0:
            if self.normal_X is None or len(self.normal_X) < self.normal_capacity:
                self._reservoir_update_normal(X_norm)
            else:
                drift_info = self.drift_detector.detect_drift(self.normal_X, X_norm)
                if drift_info["drift_detected"]:
                    self._evict_and_refresh_normal(X_norm)
                else:
                    self._reservoir_update_normal(X_norm)

        return drift_info

    def sample(self, batch_size: int):
        # sampling a mixed replay batch from both attack and normal memory pools
        X_parts, y_parts = [], []

        for cls_id, pool_X in self.attack_pools.items():
            if len(pool_X) > 0:
                X_parts.append(pool_X)
                y_parts.append(np.full(len(pool_X), cls_id, dtype=np.int64))

        if self.normal_X is not None and len(self.normal_X) > 0:
            X_parts.append(self.normal_X)
            y_parts.append(self.normal_y)

        if not X_parts:
            return None, None

        all_X = np.vstack(X_parts)
        all_y = np.concatenate(y_parts)

        n_samples = min(batch_size, len(all_X))
        indices = self.rng.choice(len(all_X), size=n_samples, replace=False)
        return all_X[indices], all_y[indices]

    def get_stats(self) -> dict:
        # returning current sample counts across normal and attack pools
        attack_counts = {int(k): len(v) for k, v in self.attack_pools.items()}
        normal_count = len(self.normal_X) if self.normal_X is not None else 0
        return {
            "normal_count": normal_count,
            "attack_total": sum(attack_counts.values()),
            "attack_per_class": attack_counts,
            "total_stored": normal_count + sum(attack_counts.values()),
        }
