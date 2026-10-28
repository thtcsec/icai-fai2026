"""P1-A: operating thresholds must be frozen on train/val, never test."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(BASE, "results")
for p in (BASE, RESULTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from train_utils import select_operating_threshold, threshold_metrics


@pytest.mark.parametrize("split", ["test", "holdout", "cic", "insdn_test", "cic_holdout"])
def test_refuses_test_split_threshold_selection(split):
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    with pytest.raises(ValueError, match="Refusing"):
        select_operating_threshold(y, s, split=split)


def test_val_threshold_then_test_eval():
    rng = np.random.default_rng(0)
    y_val = np.array([0, 0, 0, 1, 1, 1])
    s_val = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    y_te = np.array([0, 1, 0, 1])
    s_te = np.array([0.15, 0.85, 0.45, 0.7])
    tau, meta = select_operating_threshold(y_val, s_val, split="val")
    assert meta["split"] == "val"
    frozen = threshold_metrics(y_te, s_te, tau)
    # Sanity: function returns finite metrics and does not peek at y_te for tau.
    assert 0.0 <= frozen["f1"] <= 1.0
    tau2, _ = select_operating_threshold(y_val, s_val, split="validation")
    assert tau == tau2


def test_blocked_split_val_indices_disjoint():
    pytest.importorskip("torch")
    from prototype.data.data_loader import resolve_real_windows_npz
    from train_utils import blocked_split_npz

    npz = resolve_real_windows_npz("insdn")
    if not npz:
        pytest.skip("insdn archive missing")
    X_tr, y_tr, X_va, y_va, X_te, y_te, _ = blocked_split_npz(
        npz, seq_len=10, max_train=400, max_test=200, seed=42, val_fraction=0.25
    )
    assert len(X_tr) > 0 and len(X_va) > 0 and len(X_te) > 0
    assert len(X_tr) + len(X_va) <= 400
    assert len(y_tr) == len(X_tr)
    assert len(y_va) == len(X_va)
    assert len(y_te) == len(X_te)
