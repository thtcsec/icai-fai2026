"""Memory-light identity probe for Phase 3. usecols only. Never opens RAR."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = os.path.abspath(os.path.join(BASE, ".."))
CSV = r"D:\huflit_logs\public_datasets\insdn\Dataset.csv"
OUT = os.path.join(REPO, "docs", "audit", "phase3_identity_probe.json")
SEQ_LEN = 10

USECOLS = [
    "Timestamp",
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "target",
]


def _day(ts: str) -> str:
    return str(ts).strip().split(" ", 1)[0]


def main():
    if not os.path.isfile(CSV):
        raise FileNotFoundError(CSV)
    lower = CSV.lower()
    if lower.endswith((".rar", ".zip", ".7z")) or "thang6" in lower:
        raise RuntimeError("Refusing archive")
    print("[*] Phase 3 identity probe (8 columns)")
    df = pd.read_csv(
        CSV,
        usecols=USECOLS,
        dtype={
            "Timestamp": "string",
            "Flow ID": "string",
            "Src IP": "string",
            "Dst IP": "string",
            "Src Port": "string",
            "Dst Port": "string",
            "Protocol": "string",
            "target": np.float32,
        },
        low_memory=True,
    )
    n = len(df)
    y = (pd.to_numeric(df["target"], errors="coerce").fillna(0) > 0).astype(np.int64)
    day = df["Timestamp"].map(_day)
    five = (
        df["Src IP"].astype(str)
        + "|"
        + df["Src Port"].astype(str)
        + "|"
        + df["Dst IP"].astype(str)
        + "|"
        + df["Dst Port"].astype(str)
        + "|"
        + df["Protocol"].astype(str)
    )
    day_ip = day.astype(str) + "|" + df["Src IP"].astype(str)

    def _size_stats(series):
        vc = series.value_counts()
        n_ge = int((vc >= SEQ_LEN).sum())
        return {
            "n_unique": int(len(vc)),
            "n_size_ge_seq_len": n_ge,
            "frac_keys_windowable": float(n_ge / max(len(vc), 1)),
            "n_rows_in_windowable_keys": int(vc[vc >= SEQ_LEN].sum()),
            "median_size": float(vc.median()),
            "p99_size": float(vc.quantile(0.99)),
            "max_size": int(vc.max()),
            "n_size_1": int((vc == 1).sum()),
        }

    flow_stats = _size_stats(df["Flow ID"])
    five_stats = _size_stats(five)
    day_ip_stats = _size_stats(day_ip)
    src_ip_stats = _size_stats(df["Src IP"])

    # Label mix among windowable day|ip groups
    tmp = pd.DataFrame({"g": day_ip, "y": y})
    gsize = tmp.groupby("g").size()
    win_g = gsize[gsize >= SEQ_LEN].index
    sub = tmp[tmp["g"].isin(win_g)]
    g_attack = sub.groupby("g")["y"].mean()
    mixed = int(((g_attack > 0) & (g_attack < 1)).sum())
    pure0 = int((g_attack == 0).sum())
    pure1 = int((g_attack == 1).sum())

    dup_flow = int(df["Flow ID"].duplicated().sum())
    dup_five = int(five.duplicated().sum())
    exact_row_dup = int(df.duplicated().sum())

    out = {
        "n_rows": int(n),
        "attack_rows": int(y.sum()),
        "normal_rows": int((y == 0).sum()),
        "flow_id": flow_stats,
        "five_tuple": five_stats,
        "recording_day_src_ip": day_ip_stats,
        "src_ip_global": src_ip_stats,
        "windowable_day_ip_groups": {
            "n": int(len(win_g)),
            "pure_normal": pure0,
            "pure_attack": pure1,
            "mixed_label": mixed,
        },
        "duplicate_flow_id_extra_rows": dup_flow,
        "duplicate_five_tuple_extra_rows": dup_five,
        "duplicate_full_usecols_rows": exact_row_dup,
        "session_level_windows_viable": flow_stats["n_size_ge_seq_len"] >= 30,
        "group_disjoint_day_ip_candidate": day_ip_stats["n_size_ge_seq_len"] >= 30,
        "notes": [
            "Flow ID uniqueness near 1 implies each row is already a flow; session windows of length 10 cannot be built from Flow ID.",
            "Group-disjoint protocol will use recording_day|Src IP assigned wholly to one split.",
            "Purged chrono is a separate protocol and does not make overlapping IPs independent.",
            "Format A dates remain opaque tokens.",
        ],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in (
        "n_rows", "flow_id", "five_tuple", "recording_day_src_ip", "src_ip_global",
        "windowable_day_ip_groups", "session_level_windows_viable",
        "group_disjoint_day_ip_candidate",
    )}, indent=2))
    print(f"[+] {OUT}")
    return out


if __name__ == "__main__":
    main()
