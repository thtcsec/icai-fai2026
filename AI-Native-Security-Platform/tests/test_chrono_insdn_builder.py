"""Chrono InSDN builder: usecols CSV only; never opens archives."""

from __future__ import annotations

import os
import sys

import pytest

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(BASE, "results")
for p in (BASE, RESULTS):
    if p not in sys.path:
        sys.path.insert(0, p)

pd = pytest.importorskip("pandas")

from build_chrono_insdn import load_needed_columns, main


def _tiny_csv(path):
    rows = [
        "Timestamp,Src IP,Flow Duration,Flow Pkts/s,Flow Byts/s,Flow IAT Mean,Flow IAT Std,"
        "Pkt Len Mean,Tot Fwd Pkts,SYN Flag Cnt,Active Mean,Idle Mean,target"
    ]
    # One recording day, one IP. Need >=10 train flows after 70/15/15.
    for i in range(40):
        rows.append(
            f"10/1/2020 12:{i:02d},10.0.0.1,1,1,1,1,0,1,1,0,0,0,{int(i >= 28)}"
        )
    for i in range(40):
        rows.append(
            f"25/12/2019 01:{i:02d}:00 AM,10.0.0.2,1,1,1,1,0,1,1,0,0,0,0"
        )
    path.write_text("\n".join(rows), encoding="utf-8")


def test_refuses_rar(tmp_path):
    rar = tmp_path / "Thang6.rar"
    rar.write_bytes(b"not-an-archive")
    with pytest.raises(RuntimeError, match="Refusing to open campus archive"):
        load_needed_columns(str(rar))


def test_builder_tiny_csv(tmp_path):
    csv_path = tmp_path / "Dataset.csv"
    out = tmp_path / "chrono"
    _tiny_csv(csv_path)
    man = main(src=str(csv_path), out_dir=str(out))
    assert man["thang6_archive_touched"] is False
    assert man["frozen_npz_used"] is False
    assert os.path.isfile(out / "train_windows.npz")
    assert man["temporal_validity"] == "within_recording_day_clock_order_srcip_grouped"
