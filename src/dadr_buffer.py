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
        # setting up total memory capacity and splitting ratios between attack and normal pools
        self.total_capacity = (
            total_capacity if total_capacity is not None else Config.TOTAL_MEMORY_BUDGET
        )
        self.attack_ratio = (
            attack_ratio if attack_ratio is not None else Config.ATTACK_POOL_RATIO
        )
        self.normal_idx = normal_idx

        self.attack_capacity = int(round(self.total_capacity * self.attack_ratio))
        self.normal_capacity = self.total_capacity - self.attack_capacity

        # initializing random state and median psi drift detector
        self.rng = np.random.RandomState(
            seed if seed is not None else Config.SEEDS[0]
        )
        self.drift_detector = PSIDriftDetector(threshold=0.35)

        self.attack_pools = {}
        self.normal_X = None
        self.normal_y = None
        self.normal_seen_count = 0

    def _rebalance_attack_pools(self):
        # dividing attack capacity equally across all discovered attack classes
        if not self.attack_pools:
            return
        num_attack_classes = len(self.attack_pools)
        base_quota = self.attack_capacity // num_attack_classes
        remainder = self.attack_capacity % num_attack_classes

        for idx, cls_id in enumerate(sorted(self.attack_pools.keys())):
            quota = base_quota + (1 if idx < remainder else 0)
            pool_X = self.attack_pools[cls_id]
            if len(pool_X) > quota:
                keep_indices = self.rng.choice(len(pool_X), size=quota, replace=False)
                self.attack_pools[cls_id] = pool_X[keep_indices]

    def _update_attack_pool(self, X_att: np.ndarray, y_att: np.ndarray):
        # adding incoming attack samples to their class pools and rebalancing quotas
        if len(X_att) == 0:
            return
        for cls_id in np.unique(y_att):
            cls_id = int(cls_id)
            cls_X = X_att[y_att == cls_id]
            if cls_id in self.attack_pools:
                self.attack_pools[cls_id] = np.vstack([self.attack_pools[cls_id], cls_X])
            else:
                self.attack_pools[cls_id] = cls_X.copy()
        self._rebalance_attack_pools()

    def _reservoir_update_normal(self, X_norm: np.ndarray):
        # updating the normal traffic buffer using standard reservoir sampling
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

    def _adapt_to_normal_drift(self, X_norm: np.ndarray):
        # keeping attack slots untouched while blending fresh drifted normal samples into normal pool
        n_refresh = min(len(X_norm), max(1, int(self.normal_capacity * 0.35)))
        replace_idx = self.rng.choice(len(self.normal_X), size=n_refresh, replace=False)
        fresh_idx = self.rng.choice(len(X_norm), size=n_refresh, replace=False)
        self.normal_X[replace_idx] = X_norm[fresh_idx]
        self.normal_seen_count += len(X_norm)

    def update(self, X: np.ndarray, y: np.ndarray) -> dict:
        # separating incoming batch into normal and attack traffic streams
        normal_mask = y == self.normal_idx
        X_norm = X[normal_mask]
        X_att = X[~normal_mask]
        y_att = y[~normal_mask]

        self._update_attack_pool(X_att, y_att)

        drift_info = {
            "psi_score": 0.0,
            "max_feature_psi": 0.0,
            "drift_detected": False,
            "severity": "stable",
        }

        # checking for covariate drift in normal traffic and adapting memory accordingly
        if len(X_norm) > 0:
            if self.normal_X is None or len(self.normal_X) < self.normal_capacity:
                self._reservoir_update_normal(X_norm)
            else:
                drift_info = self.drift_detector.detect_drift(self.normal_X, X_norm)
                if drift_info["drift_detected"]:
                    self._adapt_to_normal_drift(X_norm)
                else:
                    self._reservoir_update_normal(X_norm)

        return drift_info

    def sample(self, batch_size: int):
        # sampling a balanced mini batch across all stored attack and normal prototypes
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
        # returning current sample counts across normal and attack memory pools
        attack_counts = {int(k): len(v) for k, v in self.attack_pools.items()}
        normal_count = len(self.normal_X) if self.normal_X is not None else 0
        return {
            "normal_count": normal_count,
            "attack_total": sum(attack_counts.values()),
            "attack_per_class": attack_counts,
            "total_stored": normal_count + sum(attack_counts.values()),
        }
