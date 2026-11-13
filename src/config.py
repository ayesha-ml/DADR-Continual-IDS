import os
import random
import numpy as np
import torch


class Config:
    # defining base paths for repository and google drive storage
    REPO_DIR = "/content/DADR-Continual-IDS"
    DRIVE_BASE_DIR = "/content/drive/MyDrive/DADR_Continual_IDS"
    RAW_DATA_DIR = os.path.join(DRIVE_BASE_DIR, "raw_data")
    PROCESSED_DATA_DIR = os.path.join(DRIVE_BASE_DIR, "processed_data")
    RESULTS_DIR = os.path.join(DRIVE_BASE_DIR, "results")
    FIGURES_DIR = os.path.join(DRIVE_BASE_DIR, "figures")

    # setting dataset splitting and streaming window parameters
    INIT_TRAIN_RATIO = 0.30
    NUM_STREAM_WINDOWS = 10
    DRIFT_INJECTION_WINDOW = 5

    # setting dadr memory and drift detection hyperparameters
    TOTAL_MEMORY_BUDGET = 5000
    ATTACK_POOL_RATIO = 0.70
    NORMAL_POOL_RATIO = 0.30
    PSI_THRESHOLD = 0.20
    PSI_NUM_BINS = 10
    MAHALANOBIS_REG = 1e-4

    # setting neural network and training hyperparameters
    HIDDEN_DIMS = [256, 128]
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 256
    EPOCHS_PER_WINDOW = 5

    # setting experimental evaluation seeds and ablations
    SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    BUDGET_ABLATIONS = [1000, 2000, 5000, 10000]
    RATIO_ABLATIONS = [(1.0, 0.0), (0.8, 0.2), (0.7, 0.3), (0.5, 0.5)]
    PSI_ABLATIONS = [0.10, 0.15, 0.20, 0.25, 0.30]

    # listing identifier columns to strip for preventing shortcut learning
    DROP_COLUMNS = [
        "Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP",
        "Source Port", "Src Port", "Destination Port", "Dst Port",
        "Timestamp", "Protocol", "srcip", "dstip", "sport", "dsport",
        "stime", "ltime"
    ]

    @classmethod
    def ensure_dirs(cls):
        # creating required google drive folders if missing
        for path in [
            cls.DRIVE_BASE_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.RESULTS_DIR,
            cls.FIGURES_DIR,
        ]:
            os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    # locking random seeds across python, numpy, and pytorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
