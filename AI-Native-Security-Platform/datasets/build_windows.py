"""
build_windows.py — Reconstruct T×F sliding windows from a standardized flow CSV.

The frozen archives under datasets/processed/*.npz were produced offline from
public InSDN / CIC-IDS2017 flow exports. This script is the *documented*
reconstruction recipe for the windowing step only (CSV rows → NPZ).

Required CSV columns (case-insensitive aliases accepted via FEATURE_ALIASES):
  the 10 FEATURE_COLUMNS in prototype.data.data_loader, plus a binary/multi label
  column named ``label`` (any positive label → attack).

Window label = label of the *last* flow in the window (matches the offline
prepare path that produced the frozen InSDN/CIC NPZs). This is *not*
``any-attack-in-window``.

The frozen InSDN archive was built from a label-stratified 50k subsample that
was shuffled (seed 42) *before* windowing, so row order is not capture order.
Pass rows already in the intended index order; this script does not shuffle.

Usage:
  python datasets/build_windows.py --csv path/to/flows.csv --out datasets/processed/out.npz \\
      --seq-len 10 --stride 1
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from prototype.data.data_loader import FEATURE_COLUMNS

FEATURE_ALIASES = {
    "flow_duration": ["flow_duration", "Flow Duration", "Duration"],
    "packet_rate": ["packet_rate", "Flow Packets/s", "Fwd Packets/s"],
    "byte_rate": ["byte_rate", "Flow Bytes/s"],
    "pkt_iat_mean": ["pkt_iat_mean", "Flow IAT Mean", "Fwd IAT Mean"],
    "pkt_iat_std": ["pkt_iat_std", "Flow IAT Std", "Fwd IAT Std"],
    "packet_len_mean": ["packet_len_mean", "Packet Length Mean", "Fwd Packet Length Mean"],
    "packet_in_count": ["packet_in_count", "Total Fwd Packets", "Total Packets"],
    "tcp_flags_syn_count": ["tcp_flags_syn_count", "SYN Flag Count"],
    "flow_active_ratio": ["flow_active_ratio", "Active Mean"],
    "rsu_channel_occupancy": ["rsu_channel_occupancy", "Idle Mean"],
}


def _resolve_column(df: pd.DataFrame, canonical: str) -> str:
    cols = {c.lower(): c for c in df.columns}
    for alias in FEATURE_ALIASES[canonical]:
        if alias.lower() in cols:
            return cols[alias.lower()]
    raise KeyError(f"Missing feature column for {canonical}; tried {FEATURE_ALIASES[canonical]}")


def build_windows(csv_path: str, seq_len: int = 10, stride: int = 1):
    df = pd.read_csv(csv_path)
    feats = []
    for name in FEATURE_COLUMNS:
        col = _resolve_column(df, name)
        feats.append(pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
    F = np.stack(feats, axis=1)  # N_flow x 10

    label_col = None
    for cand in ("label", "Label", "Attack", "class"):
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise KeyError("CSV needs a label column (label/Label/Attack/class)")
    y_flow = pd.to_numeric(df[label_col], errors="coerce").fillna(0).to_numpy()
    y_flow = (y_flow > 0).astype(np.int64)

    windows, labels = [], []
    for start in range(0, len(F) - seq_len + 1, stride):
        sl = slice(start, start + seq_len)
        windows.append(F[sl])
        # Window label = last flow in the window (frozen-archive convention)
        labels.append(int(y_flow[start + seq_len - 1] > 0))
    X = np.stack(windows, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    X, y = build_windows(args.csv, seq_len=args.seq_len, stride=args.stride)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y)
    digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"[+] Wrote {args.out}")
    print(f"    X={X.shape} attack={int((y > 0).sum())} normal={int((y == 0).sum())}")
    print(f"    sha256={digest}")


if __name__ == "__main__":
    main()
