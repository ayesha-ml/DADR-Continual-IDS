import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.config import Config
from src.dadr_buffer import DADRReplayBuffer


class BaseCLStrategy:
    def __init__(self, name: str, device: torch.device, seed: int = None):
        # initializing base continual learning strategy attributes
        self.name = name
        self.device = device
        self.seed = seed if seed is not None else Config.SEEDS[0]
        self.rng = np.random.RandomState(self.seed)

    def before_step(self, model: nn.Module, prev_num_classes: int):
        # preparing model or teacher snapshot before training on current window
        pass

    def get_training_data(self, X_curr: np.ndarray, y_curr: np.ndarray):
        # returning features and labels used to train on the current window
        return X_curr, y_curr

    def compute_loss(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # computing standard cross-entropy classification loss
        return F.cross_entropy(logits, targets)

    def after_step(self, model: nn.Module, X_curr: np.ndarray, y_curr: np.ndarray) -> dict:
        # updating memory buffers or parameter importance after window training
        return {"drift_detected": False, "psi_score": 0.0}


class NaiveStrategy(BaseCLStrategy):
    def __init__(self, device: torch.device, seed: int = None):
        # sequential fine-tuning lower bound with zero memory or regularization
        super().__init__("Naive", device, seed)


class JointStrategy(BaseCLStrategy):
    def __init__(self, device: torch.device, seed: int = None):
        # cumulative offline retraining upper bound across all seen windows
        super().__init__("Joint", device, seed)
        self.X_history = []
        self.y_history = []

    def get_training_data(self, X_curr: np.ndarray, y_curr: np.ndarray):
        # combining all historical windows with current window data
        X_all = np.vstack(self.X_history + [X_curr]) if self.X_history else X_curr
        y_all = np.concatenate(self.y_history + [y_curr]) if self.y_history else y_curr
        return X_all, y_all

    def after_step(self, model: nn.Module, X_curr: np.ndarray, y_curr: np.ndarray) -> dict:
        # archiving current window into cumulative history
        self.X_history.append(X_curr.copy())
        self.y_history.append(y_curr.copy())
        return {"drift_detected": False, "psi_score": 0.0}


class EWCStrategy(BaseCLStrategy):
    def __init__(self, device: torch.device, seed: int = None, ewc_lambda: float = 100.0):
        # elastic weight consolidation protecting important parameters via fisher matrix
        super().__init__("EWC", device, seed)
        self.ewc_lambda = ewc_lambda
        self.fisher = {}
        self.star_params = {}

    def compute_loss(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # adding quadratic fisher penalty to cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets)
        if not self.fisher:
            return ce_loss

        ewc_loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher and param.requires_grad:
                star = self.star_params[name]
                fish = self.fisher[name]
                # handling dynamically expanded output layer dimensions cleanly
                if param.shape != star.shape:
                    old_rows = star.shape[0]
                    diff = param[:old_rows] - star
                    ewc_loss = ewc_loss + torch.sum(fish * (diff ** 2))
                else:
                    ewc_loss = ewc_loss + torch.sum(fish * ((param - star) ** 2))

        return ce_loss + 0.5 * self.ewc_lambda * ewc_loss

    def after_step(self, model: nn.Module, X_curr: np.ndarray, y_curr: np.ndarray) -> dict:
        # estimating diagonal fisher information matrix on current window samples
        model.eval()
        new_fisher = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        dataset = TensorDataset(
            torch.tensor(X_curr, dtype=torch.float32),
            torch.tensor(y_curr, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

        total_samples = 0
        for bx, by in loader:
            bx, by = bx.to(self.device), by.to(self.device)
            model.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()

            batch_len = bx.size(0)
            total_samples += batch_len
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    new_fisher[name] += (param.grad.detach() ** 2) * batch_len

        for name in new_fisher:
            new_fisher[name] /= max(total_samples, 1)
            if name in self.fisher and self.fisher[name].shape == new_fisher[name].shape:
                self.fisher[name] = 0.5 * self.fisher[name] + 0.5 * new_fisher[name]
            else:
                self.fisher[name] = new_fisher[name]

        self.star_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        return {"drift_detected": False, "psi_score": 0.0}


class LwFStrategy(BaseCLStrategy):
    def __init__(
        self,
        device: torch.device,
        seed: int = None,
        temperature: float = 2.0,
        distill_weight: float = 1.0,
    ):
        # learning without forgetting via logit knowledge distillation
        super().__init__("LwF", device, seed)
        self.temperature = temperature
        self.distill_weight = distill_weight
        self.teacher_model = None
        self.old_num_classes = 0

    def before_step(self, model: nn.Module, prev_num_classes: int):
        # freezing copy of previous window model as distillation teacher
        if prev_num_classes > 0:
            self.teacher_model = copy.deepcopy(model).to(self.device)
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False
            self.old_num_classes = prev_num_classes

    def compute_loss(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # combining cross-entropy on new labels with kl distillation on old logits
        ce_loss = F.cross_entropy(logits, targets)
        if self.teacher_model is None or self.old_num_classes == 0:
            return ce_loss

        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)[:, : self.old_num_classes]

        student_old_logits = logits[:, : self.old_num_classes]
        T = self.temperature
        distill_loss = F.kl_div(
            F.log_softmax(student_old_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction="batchmean",
        ) * (T * T)

        return ce_loss + self.distill_weight * distill_loss


class ReservoirReplayStrategy(BaseCLStrategy):
    def __init__(self, device: torch.device, seed: int = None, capacity: int = None):
        # standard single-pool reservoir replay without attack/normal separation
        super().__init__("ER-Reservoir", device, seed)
        self.capacity = capacity if capacity is not None else Config.TOTAL_MEMORY_BUDGET
        self.mem_X = None
        self.mem_y = None
        self.seen_count = 0

    def get_training_data(self, X_curr: np.ndarray, y_curr: np.ndarray):
        # mixing stored reservoir samples with current window flows
        if self.mem_X is None or len(self.mem_X) == 0:
            return X_curr, y_curr
        return np.vstack([X_curr, self.mem_X]), np.concatenate([y_curr, self.mem_y])

    def after_step(self, model: nn.Module, X_curr: np.ndarray, y_curr: np.ndarray) -> dict:
        # updating single reservoir buffer uniformly across all incoming flows
        for i in range(len(X_curr)):
            self.seen_count += 1
            if self.mem_X is None:
                self.mem_X = X_curr[i : i + 1].copy()
                self.mem_y = y_curr[i : i + 1].copy()
            elif len(self.mem_X) < self.capacity:
                self.mem_X = np.vstack([self.mem_X, X_curr[i : i + 1]])
                self.mem_y = np.append(self.mem_y, y_curr[i])
            else:
                idx = self.rng.randint(0, self.seen_count)
                if idx < self.capacity:
                    self.mem_X[idx] = X_curr[i]
                    self.mem_y[idx] = y_curr[i]
        return {"drift_detected": False, "psi_score": 0.0}


class ClassBalancedReplayStrategy(BaseCLStrategy):
    def __init__(self, device: torch.device, seed: int = None, capacity: int = None):
        # equal per-class replay budget across all classes including normal
        super().__init__("ER-ClassBalanced", device, seed)
        self.capacity = capacity if capacity is not None else Config.TOTAL_MEMORY_BUDGET
        self.class_pools = {}

    def get_training_data(self, X_curr: np.ndarray, y_curr: np.ndarray):
        # mixing class-balanced memory samples with current window flows
        if not self.class_pools:
            return X_curr, y_curr
        mem_X = np.vstack(list(self.class_pools.values()))
        mem_y = np.concatenate(
            [np.full(len(v), k, dtype=np.int64) for k, v in self.class_pools.items()]
        )
        return np.vstack([X_curr, mem_X]), np.concatenate([y_curr, mem_y])

    def after_step(self, model: nn.Module, X_curr: np.ndarray, y_curr: np.ndarray) -> dict:
        # allocating equal memory quotas across every encountered class id
        for cls_id in np.unique(y_curr):
            cls_id = int(cls_id)
            cls_X = X_curr[y_curr == cls_id]
            if cls_id in self.class_pools:
                self.class_pools[cls_id] = np.vstack([self.class_pools[cls_id], cls_X])
            else:
                self.class_pools[cls_id] = cls_X.copy()

        num_classes = len(self.class_pools)
        base_quota = self.capacity // num_classes
        remainder = self.capacity % num_classes

        for idx, cls_id in enumerate(sorted(self.class_pools.keys())):
            quota = base_quota + (1 if idx < remainder else 0)
            pool_X = self.class_pools[cls_id]
            if len(pool_X) > quota:
                keep = self.rng.choice(len(pool_X), size=quota, replace=False)
                self.class_pools[cls_id] = pool_X[keep]
        return {"drift_detected": False, "psi_score": 0.0}


class DADRVariantStrategy(BaseCLStrategy):
    def __init__(
        self,
        name: str,
        device: torch.device,
        seed: int = None,
        capacity: int = None,
        attack_ratio: float = None,
        enable_drift_refresh: bool = True,
    ):
        # unified wrapper supporting full dadr, dadr-nodrift ablation, and attack-only replay
        super().__init__(name, device, seed)
        self.enable_drift_refresh = enable_drift_refresh
        self.buffer = DADRReplayBuffer(
            total_capacity=capacity if capacity is not None else Config.TOTAL_MEMORY_BUDGET,
            attack_ratio=attack_ratio if attack_ratio is not None else Config.ATTACK_POOL_RATIO,
            normal_idx=0,
            seed=self.seed,
        )
        if not self.enable_drift_refresh:
            # disabling drift refresh by raising threshold above any possible psi score
            self.buffer.drift_detector.threshold = float("inf")

    def get_training_data(self, X_curr: np.ndarray, y_curr: np.ndarray):
        # combining dual-pool replay samples with current window flows
        stats = self.buffer.get_stats()
        if stats["total_stored"] == 0:
            return X_curr, y_curr
        mem_X, mem_y = self.buffer.sample(stats["total_stored"])
        return np.vstack([X_curr, mem_X]), np.concatenate([y_curr, mem_y])

    def after_step(self, model: nn.Module, X_curr: np.ndarray, y_curr: np.ndarray) -> dict:
        # updating dual-pool buffer and returning drift detection telemetry
        return self.buffer.update(X_curr, y_curr)


def build_strategy(method_name: str, device: torch.device, seed: int = None, capacity: int = None) -> BaseCLStrategy:
    # factory instantiating any of the 9 evaluated continual learning strategies
    key = method_name.strip().lower()
    cap = capacity if capacity is not None else Config.TOTAL_MEMORY_BUDGET

    if key == "naive":
        return NaiveStrategy(device=device, seed=seed)
    elif key == "joint":
        return JointStrategy(device=device, seed=seed)
    elif key == "ewc":
        return EWCStrategy(device=device, seed=seed)
    elif key == "lwf":
        return LwFStrategy(device=device, seed=seed)
    elif key == "er-reservoir":
        return ReservoirReplayStrategy(device=device, seed=seed, capacity=cap)
    elif key == "er-classbalanced":
        return ClassBalancedReplayStrategy(device=device, seed=seed, capacity=cap)
    elif key == "attack-only":
        return DADRVariantStrategy(
            name="Attack-Only",
            device=device,
            seed=seed,
            capacity=cap,
            attack_ratio=1.0,
            enable_drift_refresh=False,
        )
    elif key == "dadr-nodrift":
        return DADRVariantStrategy(
            name="DADR-NoDrift",
            device=device,
            seed=seed,
            capacity=cap,
            attack_ratio=Config.ATTACK_POOL_RATIO,
            enable_drift_refresh=False,
        )
    elif key == "dadr":
        return DADRVariantStrategy(
            name="DADR",
            device=device,
            seed=seed,
            capacity=cap,
            attack_ratio=Config.ATTACK_POOL_RATIO,
            enable_drift_refresh=True,
        )
    else:
        raise ValueError(f"unknown continual learning method: {method_name}")
