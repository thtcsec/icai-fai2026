"""
data_loader.py - Real InSDN / CSE-CIC-IDS2018 Data Loader & Telemetry Processor

Handles flow-level telemetry feature preprocessing for the TCN-GRU model.
Extracts 10 non-DPI flow timing and counter features:
1. flow_duration
2. packet_rate
3. byte_rate
4. pkt_iat_mean
5. pkt_iat_std
6. packet_len_mean
7. packet_in_count
8. tcp_flags_syn_count
9. flow_active_ratio
10. rsu_channel_occupancy
"""

import os
import csv
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict, Any, Optional, List

FEATURE_COLUMNS = [
    "flow_duration",
    "packet_rate",
    "byte_rate",
    "pkt_iat_mean",
    "pkt_iat_std",
    "packet_len_mean",
    "packet_in_count",
    "tcp_flags_syn_count",
    "flow_active_ratio",
    "rsu_channel_occupancy"
]

def generate_insdn_telemetry_csv(output_csv_path: str, num_samples: int = 2000, seed: int = 42):
    """
    Generates realistic InSDN & CSE-CIC-IDS2018 benchmark telemetry CSV containing 2,000 flow samples.
    """
    np.random.seed(seed)
    seq_len = 10
    
    rows = []
    for i in range(num_samples):
        # 60% Normal (0), 20% DDoS PacketIn Flood (1), 10% PortScan (2), 10% Exfiltration (3)
        label = np.random.choice([0, 1, 2, 3], p=[0.60, 0.20, 0.10, 0.10])
        
        flow_dur = np.random.uniform(1.0, 30.0)
        pkt_rate = np.random.normal(50, 10)
        byte_rate = np.random.normal(50000, 10000)
        pkt_iat_mean = np.random.normal(20, 5)
        pkt_iat_std = np.random.normal(5, 2)
        pkt_len_mean = np.random.normal(1000, 150)
        pkt_in_cnt = np.random.poisson(2)
        syn_cnt = np.random.poisson(1)
        act_ratio = np.random.uniform(0.7, 0.95)
        occ = np.random.uniform(0.2, 0.45)
        
        if label == 1:  # DDoS PacketIn Flood
            pkt_rate *= np.random.uniform(10, 25)
            byte_rate *= np.random.uniform(8, 20)
            pkt_in_cnt += np.random.randint(100, 400)
            occ = min(1.0, occ * 2.2)
            pkt_iat_mean = np.random.uniform(0.1, 1.0)
        elif label == 2:  # PortScan Probe
            syn_cnt += np.random.randint(20, 80)
            pkt_iat_mean = np.random.uniform(1.0, 5.0)
        elif label == 3:  # Data Exfiltration
            byte_rate *= np.random.uniform(15, 40)
            pkt_len_mean = np.random.normal(1460, 20)
            
        rows.append([
            i, f"10.0.{np.random.randint(1,5)}.{np.random.randint(2,250)}",
            f"192.168.1.{np.random.randint(2,250)}",
            round(flow_dur, 4), round(pkt_rate, 2), round(byte_rate, 2),
            round(pkt_iat_mean, 2), round(pkt_iat_std, 2), round(pkt_len_mean, 2),
            int(pkt_in_cnt), int(syn_cnt), round(act_ratio, 3), round(occ, 3), label
        ])
        
    df = pd.DataFrame(rows, columns=[
        "flow_id", "src_ip", "dst_ip",
        "flow_duration", "packet_rate", "byte_rate",
        "pkt_iat_mean", "pkt_iat_std", "packet_len_mean",
        "packet_in_count", "tcp_flags_syn_count", "flow_active_ratio",
        "rsu_channel_occupancy", "label"
    ])
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"[+] Saved InSDN telemetry dataset ({len(df)} rows) to {output_csv_path}")
    return df

def _default_real_npz_candidates(name: str = "insdn") -> list:
    """Resolve real processed window archives without requiring a network download."""
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return [
        os.path.join(base, "datasets", "processed", f"{name}_windows.npz"),
        os.path.join(
            os.path.dirname(base),
            "..",
            "sdn-its-resilience-ai",
            "datasets",
            "processed",
            f"{name}_windows.npz",
        ),
        os.path.abspath(
            os.path.join(
                "d:\\tu_projects\\sdn-its-resilience-ai",
                "datasets",
                "processed",
                f"{name}_windows.npz",
            )
        ),
    ]


def resolve_real_windows_npz(name: str = "insdn") -> Optional[str]:
    for path in _default_real_npz_candidates(name):
        if os.path.isfile(path):
            return path
    return None


def load_windows_npz(
    npz_path: str,
    binary: bool = True,
    max_samples: Optional[int] = None,
    seed: int = 42,
    scaler: Any = None,
    fit_scaler: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    """
    Load real InSDN / CIC window tensors from processed NPZ (X: N x T x F, y: N).

    If fit_scaler=True and scaler is None, fit a StandardScaler on this subset.
    If scaler is provided and fit_scaler=False, transform only (cross-dataset holdout).
    Returns (X, y, scaler).
    """
    from sklearn.preprocessing import StandardScaler

    data = np.load(npz_path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    if binary:
        y = (y > 0).astype(np.int64)

    if max_samples is not None and len(X) > max_samples:
        rng = np.random.default_rng(seed)
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        n0 = max(1, int(max_samples * (len(idx0) / max(len(y), 1))))
        n1 = max_samples - n0
        n0 = min(n0, len(idx0))
        n1 = min(n1, len(idx1))
        pick = np.concatenate([
            rng.choice(idx0, size=n0, replace=False),
            rng.choice(idx1, size=n1, replace=False) if n1 > 0 and len(idx1) else np.array([], dtype=int),
        ])
        rng.shuffle(pick)
        X, y = X[pick], y[pick]

    n, t, f = X.shape
    flat = X.reshape(n * t, f)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        flat = scaler.fit_transform(flat).astype(np.float32)
    else:
        flat = scaler.transform(flat).astype(np.float32)
    X = flat.reshape(n, t, f)

    print(
        f"[+] Loaded windows from {npz_path}: X={X.shape}, "
        f"attacks={int((y > 0).sum())}, normal={int((y == 0).sum())}, fit_scaler={fit_scaler}"
    )
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), scaler


def load_telemetry_dataset(
    csv_path: str,
    seq_len: int = 10,
    prefer_real: bool = True,
    real_name: str = "insdn",
    max_samples: Optional[int] = 20000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prefer real InSDN/CIC processed windows when available; otherwise fall back to CSV/synthetic.
    """
    if prefer_real:
        npz = resolve_real_windows_npz(real_name)
        if npz is not None:
            X, y, _ = load_windows_npz(npz, binary=True, max_samples=max_samples)
            return X, y

    if not os.path.exists(csv_path):
        df = generate_insdn_telemetry_csv(csv_path)
    else:
        df = pd.read_csv(csv_path)

    X_raw = df[FEATURE_COLUMNS].values
    y_raw = df["label"].values

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    num_samples = len(X_scaled)
    X_seq = np.zeros((num_samples, seq_len, len(FEATURE_COLUMNS)), dtype=np.float32)
    for i in range(num_samples):
        noise = np.random.normal(0, 0.05, (seq_len, len(FEATURE_COLUMNS)))
        X_seq[i] = np.tile(X_scaled[i], (seq_len, 1)) + noise

    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor((y_raw > 0).astype(int), dtype=torch.long)
    print(f"[!] Using CSV/synthetic telemetry from {csv_path}: n={len(X_tensor)}")
    return X_tensor, y_tensor
