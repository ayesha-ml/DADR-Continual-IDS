import numpy as np
from sklearn.metrics import f1_score, recall_score


class ContinualIDSMetrics:
    def __init__(self, num_classes: int, normal_idx: int = 0):
        # initializing metric tracker and task accuracy history matrix
        self.num_classes = num_classes
        self.normal_idx = normal_idx
        self.window_history = []
        self.seen_classes = set()

    def register_seen_classes(self, y: np.ndarray):
        # recording all classes encountered up to the current time window
        self.seen_classes.update(np.unique(y).tolist())

    def evaluate_window(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        # computing macro f1, attack recall, false positive rate, and false alerts per 10k flows
        active_classes = sorted(list(self.seen_classes.union(set(np.unique(y_true).tolist()))))

        macro_f1 = float(
            f1_score(y_true, y_pred, labels=active_classes, average="macro", zero_division=0)
        )

        # computing false positive rate strictly on true normal traffic
        normal_mask = y_true == self.normal_idx
        if np.any(normal_mask):
            false_positives = int(np.sum(y_pred[normal_mask] != self.normal_idx))
            total_normals = int(np.sum(normal_mask))
            fpr = float(false_positives / total_normals)
        else:
            false_positives = 0
            fpr = 0.0

        # scaling false alerts per 10,000 total network flows
        total_flows = max(len(y_true), 1)
        false_alerts_per_10k = float((false_positives / total_flows) * 10000.0)

        # computing attack-only macro recall across seen attack classes
        attack_classes = [c for c in active_classes if c != self.normal_idx]
        if attack_classes:
            per_class_recalls = recall_score(
                y_true, y_pred, labels=attack_classes, average=None, zero_division=0
            )
            present_attacks = [
                rec for cls_id, rec in zip(attack_classes, per_class_recalls)
                if np.any(y_true == cls_id)
            ]
            attack_recall = float(np.mean(present_attacks)) if present_attacks else 0.0
        else:
            attack_recall = 0.0

        # storing per-class recall snapshot for backward transfer (forgetting) calculation
        class_recall_map = {}
        all_recalls = recall_score(
            y_true, y_pred, labels=active_classes, average=None, zero_division=0
        )
        for cls_id, rec in zip(active_classes, all_recalls):
            if np.any(y_true == cls_id):
                class_recall_map[int(cls_id)] = float(rec)

        result = {
            "macro_f1": macro_f1,
            "attack_recall": attack_recall,
            "fpr": fpr,
            "false_alerts_per_10k": false_alerts_per_10k,
            "class_recalls": class_recall_map,
        }
        self.window_history.append(result)
        return result

    def compute_forgetting(self) -> float:
        # calculating average forgetting across all classes seen before the final window
        if len(self.window_history) < 2:
            return 0.0

        final_recalls = self.window_history[-1]["class_recalls"]
        forgetting_values = []

        for cls_id in self.seen_classes:
            if cls_id not in final_recalls:
                continue
            past_recalls = [
                step["class_recalls"][cls_id]
                for step in self.window_history[:-1]
                if cls_id in step["class_recalls"]
            ]
            if past_recalls:
                max_past = max(past_recalls)
                forgetting_values.append(max(0.0, max_past - final_recalls[cls_id]))

        return float(np.mean(forgetting_values)) if forgetting_values else 0.0
