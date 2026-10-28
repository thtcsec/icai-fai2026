"""
build_windows.py — Reconstruct T×F sliding windows from a standardized flow CSV.

The frozen archives under datasets/processed/*.npz were produced offline from
public InSDN / CIC-IDS2017 flow exports. Those NPZs store only X and y: they
were shuffled (seed 42) *before* windowing, and chronological order cannot be
reconstructed from them. This script is the *corrected* reconstruction recipe.

Default behaviour (corrected):
  * Does not shuffle raw flows.
  * Splits flows into train/val/test *before* windowing (contiguous blocks).
  * Windows never cross optional group/session boundaries.
  * Window label = label of the last flow in the window (not any-attack).
  * Completed training windows may be shuffled; raw observations are not.

Pass ``--legacy-flat`` to emit a single NPZ in CSV order without splitting
(the old recipe, still without shuffling). Pass ``--assume-csv-order`` when
no timestamp column exists; the script will not invent timestamps.

Usage:
  python datasets/build_windows.py --csv path/to/flows.csv --out-dir datasets/processed/corrected \\
      --seq-len 10 --stride 1 --assume-csv-order
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from prototype.data.data_loader import FEATURE_COLUMNS
from prototype.data.temporal_windows import (
    WINDOW_LABEL_SEMANTICS,
    build_partitioned_windows,
    order_flows,
    windows_from_ordered_flows,
)

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


def _feature_matrix(df: pd.DataFrame) -> np.ndarray:
    feats = []
    for name in FEATURE_COLUMNS:
        col = _resolve_column(df, name)
        feats.append(pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
    return np.stack(feats, axis=1)


def _label_vector(df: pd.DataFrame) -> np.ndarray:
    label_col = None
    for cand in ("label", "Label", "Attack", "class"):
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise KeyError("CSV needs a label column (label/Label/Attack/class)")
    y_flow = pd.to_numeric(df[label_col], errors="coerce").fillna(0).to_numpy()
    return (y_flow > 0).astype(np.int64)


def build_windows(csv_path: str, seq_len: int = 10, stride: int = 1):
    """Legacy helper: CSV-order windows, no shuffle. Kept for call-site compatibility."""
    df = pd.read_csv(csv_path)
    F = _feature_matrix(df)
    y_flow = _label_vector(df)
    batch = windows_from_ordered_flows(
        F,
        y_flow,
        seq_len=seq_len,
        stride=stride,
        source_indices=np.arange(len(F), dtype=np.int64),
        temporal_validity="csv_row_order_unverified",
        notes=("legacy build_windows(): no split, no shuffle, CSV order",),
    )
    return batch.X, batch.y


def _write_npz(path: str, batch, extra: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        X=batch.X,
        y=batch.y,
        start_indices=batch.start_indices,
        source_indices=batch.source_indices,
        group_ids=np.asarray(batch.group_ids, dtype=object),
        label_semantics=np.asarray(batch.label_semantics),
        temporal_validity=np.asarray(batch.temporal_validity),
    )
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    extra = dict(extra)
    extra.update(
        {
            "path": path,
            "sha256": digest,
            "X_shape": list(batch.X.shape),
            "attack": int((batch.y > 0).sum()),
            "normal": int((batch.y == 0).sum()),
            "label_semantics": batch.label_semantics,
            "temporal_validity": batch.temporal_validity,
            "notes": list(batch.notes),
        }
    )
    print(f"[+] Wrote {path}")
    print(
        f"    X={batch.X.shape} attack={extra['attack']} normal={extra['normal']} "
        f"validity={batch.temporal_validity} sha256={digest}"
    )
    return digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None, help="Legacy single-NPZ path (used with --legacy-flat)")
    ap.add_argument("--out-dir", default=None, help="Directory for train/val/test NPZs (corrected)")
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--timestamp-col", default=None)
    ap.add_argument("--group-col", default=None)
    ap.add_argument(
        "--assume-csv-order",
        action="store_true",
        help="Permit windowing without timestamps (temporal_validity=csv_row_order_unverified)",
    )
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--legacy-flat",
        action="store_true",
        help="Emit one NPZ over the full ordered CSV (no split). Still does not shuffle.",
    )
    ap.add_argument("--no-shuffle-train-windows", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    ordered, validity, notes = order_flows(
        df,
        timestamp_col=args.timestamp_col,
        group_col=args.group_col,
        assume_csv_order=args.assume_csv_order or args.legacy_flat,
    )
    F = _feature_matrix(ordered)
    y_flow = _label_vector(ordered)
    source_indices = ordered["_source_index"].to_numpy(dtype=np.int64)
    group_ids = None
    grp = args.group_col if args.group_col and args.group_col in ordered.columns else None
    if grp:
        group_ids = ordered[grp].to_numpy()

    if args.legacy_flat:
        if not args.out:
            raise SystemExit("--legacy-flat requires --out")
        batch = windows_from_ordered_flows(
            F,
            y_flow,
            seq_len=args.seq_len,
            stride=args.stride,
            group_ids=group_ids,
            source_indices=source_indices,
            temporal_validity=validity,
            notes=notes,
        )
        _write_npz(args.out, batch, {"mode": "legacy_flat"})
        return

    out_dir = args.out_dir or (os.path.dirname(os.path.abspath(args.out)) if args.out else None)
    if not out_dir:
        raise SystemExit("Provide --out-dir for the corrected split-then-window recipe")

    batches, split = build_partitioned_windows(
        F,
        y_flow,
        seq_len=args.seq_len,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        group_ids=group_ids,
        source_indices=source_indices,
        temporal_validity=validity,
        notes=notes,
        shuffle_train_windows=not args.no_shuffle_train_windows,
    )
    manifest = {
        "label_semantics": WINDOW_LABEL_SEMANTICS,
        "temporal_validity": validity,
        "notes": list(notes),
        "split_method": split.method,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "n_flows": int(len(F)),
        "partitions": {},
    }
    for name, batch in batches.items():
        path = os.path.join(out_dir, f"{name}_windows.npz")
        digest = _write_npz(path, batch, {"split": name})
        manifest["partitions"][name] = {
            "path": path,
            "sha256": digest,
            "n_windows": int(len(batch.X)),
            "n_source_flows": int(len({"train": split.train, "val": split.val, "test": split.test}[name])),
        }
    man_path = os.path.join(out_dir, "window_manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] Wrote {man_path}")


if __name__ == "__main__":
    main()
