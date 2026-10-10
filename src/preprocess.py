"""
CMAPSS Data Preprocessing Module
- Loads train/test/RUL files
- Applies piecewise linear RUL (cap at 125)
- Selects informative sensors
- Min-Max normalization (fit on train only)
- Generates sliding window sequences
"""

import numpy as np
import pandas as pd
from pathlib import Path


class MinMaxScaler:
    """Lightweight MinMaxScaler (no sklearn dependency)."""
    def __init__(self):
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        denom = self.data_max_ - self.data_min_
        denom[denom == 0] = 1.0
        return (X - self.data_min_) / denom

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# Standard column names for CMAPSS
COLUMNS = (
    ["unit", "cycle"]
    + [f"op_set_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# Sensors commonly kept (low variance / constant sensors dropped)
# Literature consensus for FD001/FD003: drop 1,5,6,10,16,18,19
# For multi-condition (FD002) keep more or all useful ones
SENSORS_FD001_FD003 = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21"
]

SENSORS_FD002 = SENSORS_FD001_FD003 + ["op_set_1", "op_set_2", "op_set_3"]  # include operating settings

RUL_CAP = 125  # Standard piecewise linear cap used in most papers


def load_dataset(data_dir: Path, subset: str):
    """Load train, test and true RUL for a given subset (FD001, FD002, FD003)."""
    data_dir = Path(data_dir)
    train = pd.read_csv(
        data_dir / f"train_{subset}.txt",
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )
    test = pd.read_csv(
        data_dir / f"test_{subset}.txt",
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )
    rul = pd.read_csv(
        data_dir / f"RUL_{subset}.txt",
        sep=r"\s+",
        header=None,
        names=["RUL"],
    )
    return train, test, rul


def add_rul_labels(df: pd.DataFrame, rul_cap: int = RUL_CAP) -> pd.DataFrame:
    """Add piecewise-linear RUL column (capped)."""
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=rul_cap)
    return df


def select_features(subset: str):
    if subset in ("FD001", "FD003"):
        return SENSORS_FD001_FD003
    return SENSORS_FD002


def normalize(train_df, test_df, feature_cols):
    """Fit MinMaxScaler on training data only, transform both."""
    scaler = MinMaxScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_df, test_df, scaler


def make_sequences(df, feature_cols, sequence_length=30, target_col="RUL"):
    """
    Create sliding window sequences per unit.
    Returns X (n_samples, seq_len, n_features), y (n_samples,)
    """
    X, y, unit_ids = [], [], []
    for unit_id, group in df.groupby("unit"):
        data = group[feature_cols].values
        targets = group[target_col].values
        if len(data) < sequence_length:
            continue
        for i in range(len(data) - sequence_length + 1):
            X.append(data[i : i + sequence_length])
            y.append(targets[i + sequence_length - 1])
            unit_ids.append(unit_id)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(unit_ids)


def prepare_test_last_sequences(test_df, feature_cols, sequence_length=30):
    """
    For test set we only need the LAST sequence of each unit
    (prediction is made at the last available cycle).
    """
    X, unit_ids = [], []
    for unit_id, group in test_df.groupby("unit"):
        data = group[feature_cols].values
        if len(data) < sequence_length:
            # pad with first values if necessary (rare)
            pad = np.tile(data[0], (sequence_length - len(data), 1))
            data = np.vstack([pad, data])
        X.append(data[-sequence_length:])
        unit_ids.append(unit_id)
    return np.array(X, dtype=np.float32), np.array(unit_ids)


def full_preprocess(data_dir, subset, sequence_length=30, rul_cap=RUL_CAP):
    """End-to-end preprocessing pipeline returning train/test arrays + true RUL."""
    train, test, true_rul = load_dataset(data_dir, subset)
    train = add_rul_labels(train, rul_cap)

    feature_cols = select_features(subset)
    train, test, scaler = normalize(train, test, feature_cols)

    X_train, y_train, _ = make_sequences(train, feature_cols, sequence_length)
    X_test, test_units = prepare_test_last_sequences(test, feature_cols, sequence_length)
    y_test = true_rul["RUL"].values.astype(np.float32)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "test_units": test_units,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "sequence_length": sequence_length,
        "subset": subset,
    }
