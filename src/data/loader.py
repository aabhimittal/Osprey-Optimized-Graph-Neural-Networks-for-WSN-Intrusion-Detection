"""
Dataset loading and preprocessing.

Two entry points are supported transparently:

1. **Synthetic** – the CSV produced by :mod:`src.data.synthetic` (default).
2. **Real WSN-DS** – if the user drops the official ``WSN-DS.csv`` into
   ``data/`` we map its columns onto our feature schema.  The real file uses
   slightly different column spellings, so :data:`WSN_DS_COLUMN_MAP` normalises
   them.  Any WSN-DS columns we don't model are ignored; engineered features are
   recomputed.

The public function :func:`load_dataset` returns a fully-numeric, scaled feature
matrix plus labels, round ids and the CH-membership needed to build graphs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CLASS_NAMES, FEATURE_NAMES

# Mapping from official WSN-DS headers -> our canonical feature names.
# (The real dataset uses names such as " Is_CH?", "who CH", "Expaned Energy".)
WSN_DS_COLUMN_MAP = {
    "id": "node_id",
    "Time": "Time",
    " Is_CH ": "Is_CH",
    "Is_CH": "Is_CH",
    "who CH": "who_CH",
    "Dist_To_CH": "Dist_To_CH",
    "ADV_S": "ADV_S",
    "ADV_R": "ADV_R",
    "JOIN_S": "JOIN_S",
    "JOIN_R": "JOIN_R",
    "SCH_S": "SCH_S",
    "SCH_R": "SCH_R",
    "Rank": "Rank",
    "DATA_S": "DATA_S",
    "DATA_R": "DATA_R",
    "Data_Sent_To_BS": "Data_Sent_To_BS",
    "dist_CH_To_BS": "Dist_CH_To_BS",
    "send_code": "Send_Code",
    "Expaned Energy": "Consumed_Energy",
    "Consumed_Energy": "Consumed_Energy",
    "Attack type": "attack_type",
}

# WSN-DS attack strings -> our integer labels
WSN_DS_LABEL_MAP = {
    "Normal": 0,
    "Blackhole": 1,
    "Grayhole": 2,
    "Flooding": 3,
    "Scheduling": 4,
    "TDMA": 4,
}


@dataclass
class Dataset:
    """A preprocessed, graph-ready view of the WSN records."""
    X: np.ndarray            # (N, F) scaled features
    y: np.ndarray            # (N,)   integer labels
    round_id: np.ndarray     # (N,)   which LEACH round each node belongs to
    node_id: np.ndarray      # (N,)   node id within its round
    who_ch: np.ndarray       # (N,)   node_id of this node's cluster head
    pos: np.ndarray          # (N, 2) spatial position (for k-NN edges / plots)
    feature_names: list
    class_names: list

    @property
    def num_features(self) -> int:
        return self.X.shape[1]


def _recompute_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the two engineered features exist (real WSN-DS lacks them)."""
    if "Rank_Ratio" not in df.columns:
        df["Rank_Ratio"] = df["Rank"] / np.clip(df.get("Cluster_Size", 1), 1, None)
    if "Forward_Ratio" not in df.columns:
        df["Forward_Ratio"] = df["Data_Sent_To_BS"] / (df["DATA_R"] + 1.0)
    return df


def _load_real_wsn_ds(path: str) -> pd.DataFrame:
    """Load and normalise the official WSN-DS CSV into our schema."""
    df = pd.read_csv(path)
    df = df.rename(columns={c: WSN_DS_COLUMN_MAP.get(c.strip(), c.strip()) for c in df.columns})
    df["label"] = df["attack_type"].map(lambda s: WSN_DS_LABEL_MAP.get(str(s).strip(), 0))

    # real WSN-DS has no explicit round column; derive pseudo-rounds by binning
    # Time so we still get many small graphs instead of one giant graph.
    if "round" not in df.columns:
        df = df.sort_values("Time").reset_index(drop=True)
        df["round"] = (df["Time"] // df["Time"].max() * 200).astype(int).clip(0, 199)

    # positions are not in WSN-DS -> synthesise stable ones from node_id for k-NN
    rng = np.random.default_rng(0)
    uniq = df["node_id"].unique()
    pos_map = {nid: rng.uniform(0, 200, size=2) for nid in uniq}
    df["x"] = df["node_id"].map(lambda n: pos_map[n][0])
    df["y"] = df["node_id"].map(lambda n: pos_map[n][1])
    df = _recompute_engineered(df)
    return df


def load_dataset(csv_path: str, seed: int = 42, scaler: str = "standard") -> Dataset:
    """Load a WSN dataset (synthetic or real) into a :class:`Dataset`.

    The loader auto-detects the real WSN-DS file by the presence of an
    ``Attack type`` column.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python main.py --generate` first, "
            "or drop the official WSN-DS.csv into data/."
        )

    head = pd.read_csv(csv_path, nrows=5)
    if "Attack type" in head.columns or "attack_type" in head.columns:
        df = _load_real_wsn_ds(csv_path)
    else:
        df = pd.read_csv(csv_path)

    df = _recompute_engineered(df)

    # Missing spatial columns (should only happen for hand-made CSVs)
    if "x" not in df.columns or "y" not in df.columns:
        rng = np.random.default_rng(seed)
        df["x"] = rng.uniform(0, 200, len(df))
        df["y"] = rng.uniform(0, 200, len(df))
    if "who_CH" not in df.columns:
        df["who_CH"] = df["node_id"]

    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # --- feature scaling -------------------------------------------------- #
    if scaler == "standard":
        from sklearn.preprocessing import StandardScaler

        X = StandardScaler().fit_transform(X)
    elif scaler == "minmax":
        from sklearn.preprocessing import MinMaxScaler

        X = MinMaxScaler().fit_transform(X)

    return Dataset(
        X=X.astype(np.float32),
        y=df["label"].to_numpy(dtype=np.int64),
        round_id=df["round"].to_numpy(dtype=np.int64),
        node_id=df["node_id"].to_numpy(dtype=np.int64),
        who_ch=df["who_CH"].to_numpy(dtype=np.int64),
        pos=df[["x", "y"]].to_numpy(dtype=np.float32),
        feature_names=list(FEATURE_NAMES),
        class_names=list(CLASS_NAMES),
    )
