"""Build an isolated chronological InSDN window corpus.

Source (NOT copied into git): D:\\huflit_logs\\public_datasets\\insdn\\Dataset.csv
~135 MB, 343,889 rows. The Thang6 RAR / campus zips are never opened.

Timestamp honesty:
  * Family B (``DD/MM/YYYY hh:mm:ss AM/PM``) is unambiguous DMY with seconds.
  * Family A (``M/D or D/M/YYYY HH:MM``) is calendar-ambiguous. Date tokens are
    kept as opaque *recording-day ids*; we never convert them to a guessed
    datetime.date. Within a recording day, clock time is well-defined.
  * Windows never cross recording days or source IPs.
  * Unparseable clocks are dropped, not invented.
  * Frozen shuffled NPZs are not inputs.

Outputs under results/corrected/chrono_insdn/ only.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from prototype.data.temporal_windows import (
    WINDOW_LABEL_SEMANTICS,
    shuffle_completed_windows,
    windows_from_ordered_flows,
)

DEFAULT_SRC = r"D:\huflit_logs\public_datasets\insdn\Dataset.csv"
OUT_DIR = os.path.join(BASE, "results", "corrected", "chrono_insdn")
SEQ_LEN = 10
STRIDE = 1
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15
SEED = 42

USECOLS = [
    "Timestamp",
    "Src IP",
    "Flow Duration",
    "Flow Pkts/s",
    "Flow Byts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Pkt Len Mean",
    "Tot Fwd Pkts",
    "SYN Flag Cnt",
    "Active Mean",
    "Idle Mean",
    "target",
]
FEATURE_FROM = [
    "Flow Duration",
    "Flow Pkts/s",
    "Flow Byts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Pkt Len Mean",
    "Tot Fwd Pkts",
    "SYN Flag Cnt",
    "Active Mean",
    "Idle Mean",
]


def _clock_seconds(ts: str) -> float | None:
    raw = str(ts).strip()
    if " " not in raw:
        return None
    time_part = raw.split(" ", 1)[1].strip()
    for fmt in ("%I:%M:%S %p", "%H:%M:%S", "%H:%M"):
        try:
            t = datetime.strptime(time_part, fmt)
            return float(t.hour * 3600 + t.minute * 60 + t.second)
        except ValueError:
            continue
    return None


def _recording_day(ts: str) -> str:
    return str(ts).strip().split(" ", 1)[0]


def _family(ts: str) -> str:
    u = str(ts).upper()
    if "AM" in u or "PM" in u:
        return "B_dmy_ampm_seconds"
    return "A_slash_date_hhmm_calendar_ambiguous"


def load_needed_columns(src: str) -> pd.DataFrame:
    lower = src.lower()
    if lower.endswith((".rar", ".zip", ".7z")) or "thang6" in lower:
        raise RuntimeError(
            f"Refusing to open campus archive {src}. Use public_datasets/insdn/Dataset.csv only."
        )
    if not os.path.isfile(src):
        raise FileNotFoundError(
            f"Public InSDN CSV not found at {src}. "
            "Place the official Dataset.csv there; do not extract the Thang6 RAR."
        )
    dtype = {c: np.float32 for c in FEATURE_FROM}
    dtype.update({"Timestamp": "string", "Src IP": "string", "target": np.float32})
    df = pd.read_csv(src, usecols=USECOLS, dtype=dtype, low_memory=True)
    df["_source_index"] = np.arange(len(df), dtype=np.int64)
    df["_recording_day"] = df["Timestamp"].map(_recording_day)
    df["_family"] = df["Timestamp"].map(_family)
    df["_clock_s"] = df["Timestamp"].map(_clock_seconds)
    n_bad = int(df["_clock_s"].isna().sum())
    if n_bad:
        df = df.loc[df["_clock_s"].notna()].copy()
    df["_label"] = (pd.to_numeric(df["target"], errors="coerce").fillna(0) > 0).astype(np.int64)
    for col in FEATURE_FROM:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df[FEATURE_FROM] = df[FEATURE_FROM].fillna(0.0)
    df["_group"] = df["_recording_day"].astype(str) + "|" + df["Src IP"].astype(str)
    return df, n_bad


def split_each_recording_day(df: pd.DataFrame) -> pd.Series:
    """Time-forward 70/15/15 *within each recording day*. No calendar mixing."""
    split = pd.Series(index=df.index, dtype=object)
    for day, idx in df.groupby("_recording_day", sort=False).groups.items():
        idx = np.asarray(idx)
        sub = df.loc[idx, ["_clock_s", "_source_index"]]
        order = np.lexsort((sub["_source_index"].to_numpy(), sub["_clock_s"].to_numpy()))
        ordered = idx[order]
        n = len(ordered)
        n_tr = int(round(n * TRAIN_RATIO))
        n_va = int(round(n * VAL_RATIO))
        split.loc[ordered[:n_tr]] = "train"
        split.loc[ordered[n_tr : n_tr + n_va]] = "val"
        split.loc[ordered[n_tr + n_va :]] = "test"
    return split


def _write(path: str, batch, extra: dict) -> dict:
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    rec = {
        **extra,
        "path": path,
        "sha256": digest,
        "X_shape": list(batch.X.shape),
        "attack": int((batch.y > 0).sum()),
        "normal": int((batch.y == 0).sum()),
        "temporal_validity": batch.temporal_validity,
        "label_semantics": batch.label_semantics,
    }
    print(
        f"[+] {os.path.basename(path)} X={batch.X.shape} "
        f"attack={rec['attack']} normal={rec['normal']}"
    )
    return rec


def main(src: str = DEFAULT_SRC, out_dir: str = OUT_DIR):
    print("[*] Chronological InSDN build (usecols only; no RAR; no frozen NPZ)")
    df, n_dropped = load_needed_columns(src)
    df["_split"] = split_each_recording_day(df)
    print(f"    rows_kept={len(df)} clock_dropped={n_dropped}")
    print("    family", df["_family"].value_counts().to_dict())
    print("    recording_days", int(df["_recording_day"].nunique()), "src_ip", int(df["Src IP"].nunique()))
    print("    split", df["_split"].value_counts().to_dict())

    train_mask = df["_split"] == "train"
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, FEATURE_FROM].to_numpy(dtype=np.float32))
    del train_mask

    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "source_csv": src,
        "source_size_mb": round(os.path.getsize(src) / 1e6, 2),
        "n_flows_kept": int(len(df)),
        "n_clock_dropped": int(n_dropped),
        "seq_len": SEQ_LEN,
        "stride": STRIDE,
        "seed": SEED,
        "split": "per_recording_day_clock_contiguous_70_15_15",
        "grouping": "recording_day + Src IP",
        "label_semantics": WINDOW_LABEL_SEMANTICS,
        "temporal_validity": "within_recording_day_clock_order_srcip_grouped",
        "format_A_note": (
            "Slash dates without AM/PM are calendar-ambiguous (MDY vs DMY). "
            "They are NOT converted to datetime.date. Ordering is clock-within-day."
        ),
        "format_B_note": "DD/MM/YYYY hh:mm:ss AM/PM parsed for clock only; day token kept opaque as recording id.",
        "scaler": "StandardScaler fit on train flows only",
        "frozen_npz_used": False,
        "thang6_archive_touched": False,
        "partitions": {},
        "recording_days": sorted(df["_recording_day"].unique().tolist()),
    }
    rng = np.random.default_rng(SEED)
    seen_source: dict[int, str] = {}
    for name in ("train", "val", "test"):
        sub = df.loc[df["_split"] == name].sort_values(
            ["_group", "_clock_s", "_source_index"], kind="mergesort"
        )
        raw = sub[FEATURE_FROM].to_numpy(dtype=np.float32)
        scaled_flows = scaler.transform(raw).astype(np.float32)
        csv_idx = sub["_source_index"].to_numpy(dtype=np.int64)
        labels = sub["_label"].to_numpy()
        groups = sub["_group"].to_numpy()
        del sub, raw
        batch = windows_from_ordered_flows(
            scaled_flows,
            labels,
            seq_len=SEQ_LEN,
            stride=STRIDE,
            group_ids=groups,
            source_indices=csv_idx,
            temporal_validity="within_recording_day_clock_order_srcip_grouped",
            notes=(
                "Scaler fit on train flows only.",
                "Format A recording days are opaque tokens; not globally calendar-sorted.",
                "Windows do not cross Src IP or recording-day boundaries.",
                "source_indices are original CSV rows; clock-adjacent flows may have non-adjacent CSV ids.",
            ),
        )
        del scaled_flows, labels, groups, csv_idx
        print(f"    {name} windows={len(batch.X)}")
        if len(batch.X):
            for idx in np.unique(batch.source_indices.ravel()):
                idx_i = int(idx)
                if idx_i in seen_source and seen_source[idx_i] != name:
                    raise AssertionError(
                        f"Source flow {idx_i} leaked into both {seen_source[idx_i]} and {name}"
                    )
                seen_source[idx_i] = name
        if name == "train" and len(batch.X):
            batch = shuffle_completed_windows(batch, rng)
        rec = _write(os.path.join(out_dir, f"{name}_windows.npz"), batch, {"split": name})
        manifest["partitions"][name] = rec
        del batch
        gc.collect()
    del df, scaler, seen_source
    gc.collect()
    man_path = os.path.join(out_dir, "manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] Wrote {man_path}")
    return manifest


if __name__ == "__main__":
    main()
