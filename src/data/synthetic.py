"""
Synthetic WSN-DS generator.

WSN-DS (Almomani et al., 2016) is the canonical benchmark for intrusion
detection in LEACH-based Wireless Sensor Networks.  It records, for every
sensor node in every LEACH *round*, a vector of MAC/routing-layer counters and
labels the node as Normal or as one of four routing attacks.  The real dataset
must be downloaded separately; this module fabricates data with the **same
schema and the same behavioural signatures** so the entire pipeline is runnable
with zero external downloads and remains fully reproducible.

Attack signatures reproduced here (these mirror how each attack manifests in
LEACH):

* Blackhole  – a malicious node advertises itself as CH, attracts JOINs and
  receives data, but forwards *almost nothing* to the Base Station
  (Forward_Ratio ~ 0, high DATA_R, near-zero Data_Sent_To_BS).
* Grayhole   – like a blackhole but *selectively* drops: it forwards a random
  fraction of received data (Forward_Ratio in a mid band, noisy).
* Flooding   – broadcasts an abnormally large number of CH advertisements
  (very high ADV_S / Send_Code) and burns energy trying to dominate the
  network, without necessarily holding a real cluster.
* Scheduling – a TDMA attack: the node manipulates the schedule, claiming an
  out-of-range Rank and emitting anomalous SCH messages, so its slot bookkeeping
  is inconsistent.

The generator returns a tidy pandas DataFrame; :func:`generate_and_save` also
writes it to CSV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CLASS_NAMES, DataConfig, FEATURE_NAMES

# integer label for each attack, matching config.CLASS_NAMES ordering
_NORMAL, _BLACKHOLE, _GRAYHOLE, _FLOODING, _SCHEDULING = range(5)


def _round_frame(round_id: int, cfg: DataConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Generate one LEACH round: node positions, cluster assignment, features."""
    n = cfg.n_nodes

    # --- 1. node deployment (static positions across the simulation) --------
    # positions are regenerated per round with the same seed stream so the
    # topology drifts slightly round-to-round (nodes are static but we add the
    # small measurement jitter typical of the dataset).
    xy = rng.uniform(0, cfg.field_size, size=(n, 2))
    bs = np.array([cfg.field_size / 2.0, cfg.field_size * 1.15])  # Base Station above field

    # --- 2. elect cluster heads (LEACH: ~p fraction become CH) --------------
    n_ch = max(1, int(round(cfg.ch_fraction * n)))
    ch_idx = rng.choice(n, size=n_ch, replace=False)
    is_ch = np.zeros(n, dtype=int)
    is_ch[ch_idx] = 1
    ch_pos = xy[ch_idx]

    # each non-CH joins its nearest CH
    d_to_ch = np.zeros(n)
    who_ch = np.full(n, -1, dtype=int)
    for i in range(n):
        if is_ch[i]:
            d_to_ch[i] = 0.0
            who_ch[i] = i
        else:
            dists = np.linalg.norm(ch_pos - xy[i], axis=1)
            j = int(np.argmin(dists))
            d_to_ch[i] = float(dists[j])
            who_ch[i] = int(ch_idx[j])

    # cluster sizes (used for rank ratio)
    cluster_size = np.array([max(1, int(np.sum(who_ch == who_ch[i]))) for i in range(n)])

    dist_ch_to_bs = np.linalg.norm(xy - bs, axis=1) * is_ch  # 0 for members

    # --- 3. assign attack labels -------------------------------------------
    labels = np.full(n, _NORMAL, dtype=int)
    n_mal = int(round(cfg.attack_fraction * n))
    mal_idx = rng.choice(n, size=n_mal, replace=False)
    # spread malicious nodes across the four attack types
    attack_choices = rng.integers(_BLACKHOLE, _SCHEDULING + 1, size=n_mal)
    labels[mal_idx] = attack_choices

    # --- 4. draw behavioural counters, conditioned on label -----------------
    # baseline (normal) behaviour first, then overwrite per-attack.
    rank = np.where(is_ch == 1, 0, rng.integers(1, np.maximum(2, cluster_size)))
    adv_s = np.where(is_ch == 1, rng.poisson(1.0, n) + 1, 0)
    adv_r = rng.poisson(n_ch, n)  # everyone hears the CH advertisements
    join_s = np.where(is_ch == 1, 0, 1)
    join_r = np.where(is_ch == 1, rng.poisson(cluster_size, n), 0)
    sch_s = np.where(is_ch == 1, 1, 0)
    sch_r = np.where(is_ch == 1, 0, 1)

    data_s = np.where(is_ch == 1, rng.poisson(3.0, n), rng.poisson(5.0, n) + 1)
    # a CH relays roughly what its members send it
    data_r = np.where(is_ch == 1, rng.poisson(5.0 * cluster_size, n), 0)
    # normal forwarding: CH pushes ~all received data to the BS
    forward_frac = np.where(is_ch == 1, rng.uniform(0.85, 1.0, n), 0.0)
    data_to_bs = np.round(data_r * forward_frac).astype(int)

    send_code = rng.integers(0, 8, n)
    time_col = np.full(n, round_id * 50.0) + rng.normal(0, 1.0, n)
    energy = (
        0.5
        + 0.02 * (adv_s + adv_r + join_s + join_r + sch_s + sch_r)
        + 0.05 * (data_s + data_r)
        + rng.normal(0, 0.05, n)
    )

    # ---- attack-specific overwrites ----
    for i in np.where(labels == _BLACKHOLE)[0]:
        # pretend to be a CH, hoover up data, forward ~nothing
        is_ch[i] = 1
        adv_s[i] = rng.poisson(2.0) + 1
        join_r[i] = rng.poisson(max(2, cluster_size[i])) + 3
        data_r[i] = rng.poisson(30.0) + 10
        data_to_bs[i] = rng.binomial(1, 0.05) * rng.integers(0, 2)  # ~0
        energy[i] += 0.5

    for i in np.where(labels == _GRAYHOLE)[0]:
        is_ch[i] = 1
        adv_s[i] = rng.poisson(2.0) + 1
        join_r[i] = rng.poisson(max(2, cluster_size[i])) + 2
        data_r[i] = rng.poisson(25.0) + 8
        frac = rng.uniform(0.2, 0.55)          # selective, partial forwarding
        data_to_bs[i] = int(round(data_r[i] * frac))
        energy[i] += 0.3

    for i in np.where(labels == _FLOODING)[0]:
        # blast advertisements / control traffic, burn energy
        adv_s[i] = rng.poisson(40.0) + 20
        send_code[i] = rng.integers(20, 40)
        join_s[i] = rng.poisson(10.0) + 5
        data_s[i] = rng.poisson(20.0) + 10
        energy[i] += 2.0 + rng.uniform(0, 1.0)

    for i in np.where(labels == _SCHEDULING)[0]:
        # TDMA manipulation: impossible rank + inconsistent schedule messages
        rank[i] = rng.integers(cfg.n_nodes, cfg.n_nodes * 2)   # out of valid range
        sch_s[i] = rng.poisson(6.0) + 3
        sch_r[i] = rng.poisson(6.0) + 3
        data_s[i] = rng.poisson(12.0) + 4
        energy[i] += 0.4

    # engineered features that make the topological signal explicit
    rank_ratio = rank / np.clip(cluster_size, 1, None)
    forward_ratio = data_to_bs / (data_r + 1.0)

    frame = pd.DataFrame(
        {
            "round": round_id,
            "node_id": np.arange(n),
            "x": xy[:, 0],
            "y": xy[:, 1],
            "who_CH": who_ch,
            # ---- the 18 model features (order matches config.FEATURE_NAMES) ----
            "Time": time_col,
            "Is_CH": is_ch,
            "Dist_To_CH": d_to_ch,
            "ADV_S": adv_s,
            "ADV_R": adv_r,
            "JOIN_S": join_s,
            "JOIN_R": join_r,
            "SCH_S": sch_s,
            "SCH_R": sch_r,
            "Rank": rank,
            "DATA_S": data_s,
            "DATA_R": data_r,
            "Data_Sent_To_BS": data_to_bs,
            "Dist_CH_To_BS": dist_ch_to_bs,
            "Send_Code": send_code,
            "Consumed_Energy": energy,
            "Rank_Ratio": rank_ratio,
            "Forward_Ratio": forward_ratio,
            "label": labels,
        }
    )
    return frame


def generate(cfg: DataConfig, seed: int) -> pd.DataFrame:
    """Generate a full synthetic WSN-DS dataset (``cfg.n_rounds`` graphs)."""
    rng = np.random.default_rng(seed)
    frames = [_round_frame(r, cfg, rng) for r in range(cfg.n_rounds)]
    df = pd.concat(frames, ignore_index=True)

    # inject a little label noise for realism (mislabelled ground truth happens)
    if cfg.label_noise > 0:
        n_flip = int(cfg.label_noise * len(df))
        flip_idx = rng.choice(len(df), size=n_flip, replace=False)
        df.loc[flip_idx, "label"] = rng.integers(0, len(CLASS_NAMES), size=n_flip)

    # sanity: every declared model feature must be present
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    assert not missing, f"generator missing features: {missing}"
    return df


def generate_and_save(cfg: DataConfig, seed: int) -> pd.DataFrame:
    df = generate(cfg, seed)
    import os

    os.makedirs(os.path.dirname(cfg.csv_path) or ".", exist_ok=True)
    df.to_csv(cfg.csv_path, index=False)
    return df
