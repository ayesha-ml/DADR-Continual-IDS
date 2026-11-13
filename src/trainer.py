import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.config import Config, set_seed
from src.model import ContinualMLP
from src.metrics import ContinualIDSMetrics
from src.baselines import build_strategy


class ContinualIDSTrainer:
    def __init__(
        self,
        method_name: str,
        seed: int = None,
        capacity: int = None,
        device: torch.device = None,
    ):
        # initializing streaming continual learning trainer configuration
        self.method_name = method_name
        self.seed = seed if seed is not None else Config.SEEDS[0]
        self.capacity = capacity if capacity is not None else Config.TOTAL_MEMORY_BUDGET
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _train_epoch_loop(self, model: ContinualMLP, strategy, X_train: np.ndarray, y_train: np.ndarray):
        # training model for configured epochs using strategy-specific loss function
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

        for _ in range(Config.EPOCHS_PER_WINDOW):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = strategy.compute_loss(model, bx, logits, by)
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def _predict(self, model: ContinualMLP, X_eval: np.ndarray) -> np.ndarray:
        # running batched inference on evaluation array
        model.eval()
        preds = []
        n_samples = len(X_eval)
        for start in range(0, n_samples, Config.BATCH_SIZE * 4):
            batch_x = torch.tensor(
                X_eval[start : start + Config.BATCH_SIZE * 4],
                dtype=torch.float32,
                device=self.device,
            )
            logits = model(batch_x)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        return np.concatenate(preds) if preds else np.array([], dtype=np.int64)

    def _sample_eval_subset(self, X: np.ndarray, y: np.ndarray, rng: np.random.RandomState, max_samples: int = 10000):
        # capping cumulative evaluation slice per window for fast multi-seed benchmarking
        if len(X) <= max_samples:
            return X, y
        idx = rng.choice(len(X), size=max_samples, replace=False)
        return X[idx], y[idx]

    def run(self, data_dict: dict) -> dict:
        # executing full chronological continual learning experiment on split dataset
        set_seed(self.seed)
        eval_rng = np.random.RandomState(self.seed)
        start_time = time.time()

        X_init = data_dict["X_init"]
        y_init = data_dict["y_init"]
        X_windows = data_dict["X_windows"]
        y_windows = data_dict["y_windows"]
        total_possible_classes = len(data_dict["class_names"])
        input_dim = X_init.shape[1]

        # determining initial active class dimension from T0
        init_classes = int(np.max(y_init)) + 1
        model = ContinualMLP(input_dim=input_dim, num_classes=init_classes).to(self.device)
        strategy = build_strategy(
            method_name=self.method_name,
            device=self.device,
            seed=self.seed,
            capacity=self.capacity,
        )

        preq_metrics = ContinualIDSMetrics(num_classes=total_possible_classes, normal_idx=0)
        cum_metrics = ContinualIDSMetrics(num_classes=total_possible_classes, normal_idx=0)

        # phase 1: initial training on T0
        preq_metrics.register_seen_classes(y_init)
        cum_metrics.register_seen_classes(y_init)

        strategy.before_step(model, prev_num_classes=0)
        self._train_epoch_loop(model, strategy, X_init, y_init)
        strategy.after_step(model, X_init, y_init)

        # storing bounded evaluation slice of T0 for cumulative retention tracking
        X_cum_eval, y_cum_eval = self._sample_eval_subset(X_init, y_init, eval_rng)
        cum_X_list = [X_cum_eval]
        cum_y_list = [y_cum_eval]

        window_logs = []

        # phase 2: sequential streaming across W1 ... W10
        for w_idx, (X_w, y_w) in enumerate(zip(X_windows, y_windows), start=1):
            prev_num_classes = model.num_classes
            required_classes = max(prev_num_classes, int(np.max(y_w)) + 1)

            # snapshotting teacher before class expansion (required for LwF)
            strategy.before_step(model, prev_num_classes=prev_num_classes)
            if required_classes > prev_num_classes:
                model.expand_classes(required_classes)

            # step A: prequential evaluation on incoming window BEFORE adaptation
            preq_preds = self._predict(model, X_w)
            preq_res = preq_metrics.evaluate_window(y_w, preq_preds)
            preq_metrics.register_seen_classes(y_w)

            # step B: adapting model using strategy replay/regularization
            X_train, y_train = strategy.get_training_data(X_w, y_w)
            self._train_epoch_loop(model, strategy, X_train, y_train)
            drift_info = strategy.after_step(model, X_w, y_w)

            # step C: cumulative evaluation across all seen windows AFTER adaptation
            cum_metrics.register_seen_classes(y_w)
            w_eval_X, w_eval_y = self._sample_eval_subset(X_w, y_w, eval_rng)
            cum_X_list.append(w_eval_X)
            cum_y_list.append(w_eval_y)

            X_all_seen = np.vstack(cum_X_list)
            y_all_seen = np.concatenate(cum_y_list)
            cum_preds = self._predict(model, X_all_seen)
            cum_res = cum_metrics.evaluate_window(y_all_seen, cum_preds)

            window_logs.append(
                {
                    "window": w_idx,
                    "num_classes_active": model.num_classes,
                    "preq_macro_f1": preq_res["macro_f1"],
                    "preq_attack_recall": preq_res["attack_recall"],
                    "preq_fpr": preq_res["fpr"],
                    "preq_false_alerts_10k": preq_res["false_alerts_per_10k"],
                    "cum_macro_f1": cum_res["macro_f1"],
                    "cum_attack_recall": cum_res["attack_recall"],
                    "cum_fpr": cum_res["fpr"],
                    "cum_false_alerts_10k": cum_res["false_alerts_per_10k"],
                    "psi_score": float(drift_info.get("psi_score", 0.0)),
                    "drift_detected": bool(drift_info.get("drift_detected", False)),
                }
            )

        elapsed_sec = round(time.time() - start_time, 2)
        avg_forgetting = cum_metrics.compute_forgetting()

        summary = {
            "method": strategy.name,
            "seed": self.seed,
            "capacity": self.capacity,
            "mean_preq_macro_f1": float(np.mean([w["preq_macro_f1"] for w in window_logs])),
            "mean_cum_macro_f1": float(np.mean([w["cum_macro_f1"] for w in window_logs])),
            "final_cum_macro_f1": float(window_logs[-1]["cum_macro_f1"]),
            "mean_cum_attack_recall": float(np.mean([w["cum_attack_recall"] for w in window_logs])),
            "mean_cum_fpr": float(np.mean([w["cum_fpr"] for w in window_logs])),
            "mean_cum_false_alerts_10k": float(np.mean([w["cum_false_alerts_10k"] for w in window_logs])),
            "avg_forgetting": float(avg_forgetting),
            "drift_triggers": int(sum(1 for w in window_logs if w["drift_detected"])),
            "runtime_sec": elapsed_sec,
        }

        return {
            "summary": summary,
            "window_logs": window_logs,
        }
