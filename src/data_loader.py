import glob
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.config import Config


class ContinualIDSDataLoader:
    def __init__(self, dataset_name="cicids2017", scenario="S1"):
        # initializing loader settings and leakage-free scaler
        self.dataset_name = dataset_name.lower()
        self.scenario = scenario.upper()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.class_names = []

    def _find_csv_files(self, data_dir):
        # finding all csv files inside the target directory
        files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        if not files:
            raise FileNotFoundError(
                f"no csv files found in {data_dir}. please place dataset csvs there."
            )
        return files

    def _clean_dataframe(self, df):
        # stripping whitespace from column headers
        df.columns = [c.strip() for c in df.columns]

        # identifying label column across dataset formats
        label_col = None
        for candidate in ["Label", "label", "Attack", "attack_cat"]:
            if candidate in df.columns:
                label_col = candidate
                break
        if label_col is None:
            raise ValueError("could not locate target label column in dataset.")

        # sorting chronologically if timestamp column exists
        for ts_col in ["Timestamp", "stime", "ltime"]:
            if ts_col in df.columns:
                df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
                df = df.sort_values(by=ts_col, kind="mergesort").reset_index(drop=True)
                break

        # normalizing benign label naming across datasets
        y_raw = df[label_col].astype(str).str.strip()
        y_raw = y_raw.replace({"BENIGN": "Normal", "normal": "Normal", "0": "Normal"})

        # dropping non-feature identifier and label columns
        cols_to_drop = [c for c in Config.DROP_COLUMNS if c in df.columns]
        cols_to_drop.extend([label_col, "id", "ID"])
        X_df = df.drop(columns=list(set(cols_to_drop)), errors="ignore")

        # keeping numeric columns only and replacing infinities
        X_df = X_df.select_dtypes(include=[np.number])
        X_df = X_df.replace([np.inf, -np.inf], np.nan).dropna()
        y_raw = y_raw.loc[X_df.index].reset_index(drop=True)
        X_df = X_df.reset_index(drop=True)

        # downcasting features to float32 for colab memory efficiency
        X_df = X_df.astype(np.float32)
        self.feature_names = list(X_df.columns)
        return X_df.values, y_raw.values

    def _inject_normal_drift(self, X_windows, y_windows, normal_idx):
        # applying covariate shift to normal flows from drift window onward for scenario s3
        drift_start = Config.DRIFT_INJECTION_WINDOW - 1
        rng = np.random.RandomState(42)
        shift_vector = rng.normal(loc=0.65, scale=0.25, size=X_windows[0].shape[1]).astype(np.float32)
        scale_vector = rng.uniform(1.15, 1.45, size=X_windows[0].shape[1]).astype(np.float32)

        for w in range(drift_start, len(X_windows)):
            normal_mask = y_windows[w] == normal_idx
            if np.any(normal_mask):
                X_windows[w][normal_mask] = (
                    X_windows[w][normal_mask] * scale_vector + shift_vector
                )
        return X_windows

    def load_and_split(self, data_dir=None, max_rows_per_file=None):
        # loading csv files from drive and constructing chronological stream
        target_dir = data_dir or os.path.join(Config.RAW_DATA_DIR, self.dataset_name)
        csv_files = self._find_csv_files(target_dir)

        dfs = []
        for f in csv_files:
            chunk = pd.read_csv(f, nrows=max_rows_per_file, low_memory=False)
            dfs.append(chunk)
        full_df = pd.concat(dfs, ignore_index=True)

        X, y_str = self._clean_dataframe(full_df)

        # encoding class labels while ensuring normal is always index 0
        unique_classes = sorted(list(set(y_str)))
        if "Normal" in unique_classes:
            unique_classes.remove("Normal")
            ordered_classes = ["Normal"] + unique_classes
        else:
            ordered_classes = unique_classes

        class_to_idx = {cls: idx for idx, cls in enumerate(ordered_classes)}
        self.class_names = ordered_classes
        y = np.array([class_to_idx[label] for label in y_str], dtype=np.int64)

        # splitting chronologically into initial training window and 10 stream windows
        n_total = len(X)
        n_init = int(n_total * Config.INIT_TRAIN_RATIO)

        X_init, y_init = X[:n_init], y[:n_init]
        X_stream, y_stream = X[n_init:], y[n_init:]

        # fitting standard scaler strictly on initial window to guarantee zero data leakage
        X_init = self.scaler.fit_transform(X_init).astype(np.float32)
        X_stream = self.scaler.transform(X_stream).astype(np.float32)

        # partitioning remaining stream into 10 equal chronological windows
        X_windows = np.array_split(X_stream, Config.NUM_STREAM_WINDOWS)
        y_windows = np.array_split(y_stream, Config.NUM_STREAM_WINDOWS)

        # applying controlled normal traffic drift if running scenario s3
        if self.scenario == "S3":
            X_windows = self._inject_normal_drift(X_windows, y_windows, normal_idx=0)

        return {
            "X_init": X_init,
            "y_init": y_init,
            "X_windows": X_windows,
            "y_windows": y_windows,
            "class_names": self.class_names,
            "feature_names": self.feature_names,
        }
