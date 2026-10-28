"""P0-A: chronological ordering, window continuity, split isolation, labels."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from prototype.data.temporal_windows import (
    TEMPORAL_VALIDITY_FROZEN_NPZ,
    assert_label_alignment,
    assert_split_isolation,
    assert_windows_contiguous,
    build_partitioned_windows,
    frozen_npz_temporal_status,
    order_flows,
    shuffle_completed_windows,
    split_flows_before_windows,
    windows_from_ordered_flows,
)


def _toy_frame():
    # Two sessions, strictly increasing timestamps, known labels.
    rows = []
    t = 1000
    for session, n, label in (("s1", 12, 0), ("s2", 12, 1)):
        for i in range(n):
            rows.append(
                {
                    "timestamp": t,
                    "session_id": session,
                    "flow_duration": float(i),
                    "label": label,
                }
            )
            t += 1
    df = pd.DataFrame(rows)
    # Shuffle the *presentation* of the CSV to prove we restore order.
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)


def test_order_flows_sorts_by_timestamp_and_group():
    df = _toy_frame()
    ordered, validity, notes = order_flows(df, timestamp_col="timestamp", group_col="session_id")
    assert validity == "grouped_then_ordered"
    assert ordered["timestamp"].is_monotonic_increasing
    assert list(ordered["session_id"].unique()) == ["s1", "s2"]
    assert "invent" not in " ".join(notes).lower()


def test_refuses_to_invent_timestamps():
    df = pd.DataFrame({"label": [0, 1, 0], "x": [1, 2, 3]})
    with pytest.raises(ValueError, match="impossible without timestamps"):
        order_flows(df, assume_csv_order=False)


def test_windows_are_contiguous_and_last_flow_labelled():
    features = np.arange(20, dtype=np.float32).reshape(20, 1)
    labels = np.array([0] * 10 + [1] * 10)
    batch = windows_from_ordered_flows(features, labels, seq_len=4, stride=1)
    assert_windows_contiguous(batch)
    assert_label_alignment(batch, labels)
    # Window starting at 8 ends at flow 11, last-flow label = 1
    idx = int(np.where(batch.start_indices == 8)[0][0])
    assert batch.y[idx] == 1
    assert np.array_equal(batch.source_indices[idx], np.arange(8, 12))


def test_windows_do_not_cross_groups():
    features = np.arange(16, dtype=np.float32).reshape(16, 1)
    labels = np.zeros(16, dtype=int)
    groups = np.array([0] * 8 + [1] * 8)
    batch = windows_from_ordered_flows(
        features, labels, seq_len=5, stride=1, group_ids=groups, source_indices=np.arange(16)
    )
    assert_windows_contiguous(batch)
    # A window starting at index 6 would span groups 0 and 1 if allowed.
    assert 6 not in set(batch.start_indices.tolist())
    for src in batch.source_indices:
        g = groups[src]
        assert len(set(g.tolist())) == 1


def test_split_before_window_isolation():
    rng = np.random.default_rng(0)
    n = 100
    features = rng.normal(size=(n, 3)).astype(np.float32)
    labels = (rng.random(n) > 0.5).astype(int)
    batches, split = build_partitioned_windows(
        features,
        labels,
        seq_len=5,
        stride=1,
        train_ratio=0.6,
        val_ratio=0.2,
        seed=42,
        shuffle_train_windows=True,
    )
    assert_split_isolation(batches)
    # Flow-level partitions are contiguous and disjoint.
    assert set(split.train).isdisjoint(split.val)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.val).isdisjoint(split.test)
    for name, batch in batches.items():
        assert_windows_contiguous(batch)
        if len(batch.X):
            assert_label_alignment(batch, labels)


def test_shuffle_completed_windows_preserves_internal_order():
    features = np.arange(30, dtype=np.float32).reshape(30, 1)
    labels = np.zeros(30, dtype=int)
    batch = windows_from_ordered_flows(features, labels, seq_len=5, stride=1)
    shuffled = shuffle_completed_windows(batch, np.random.default_rng(1))
    assert_windows_contiguous(shuffled)
    # Window identity set is unchanged.
    orig = set(map(tuple, batch.source_indices.tolist()))
    got = set(map(tuple, shuffled.source_indices.tolist()))
    assert orig == got
    assert not np.array_equal(batch.start_indices, shuffled.start_indices)


def test_frozen_npz_cannot_be_unshuffled():
    status = frozen_npz_temporal_status(["X", "y"])
    assert status["chronological_reconstruction_possible"] is False
    assert status["temporal_validity"] == TEMPORAL_VALIDITY_FROZEN_NPZ
    assert status["has_timestamps"] is False


def test_real_frozen_archive_has_no_temporal_metadata():
    path = os.path.join(BASE, "datasets", "processed", "insdn_windows.npz")
    if not os.path.isfile(path):
        pytest.skip("insdn_windows.npz not present")
    data = np.load(path, allow_pickle=True)
    status = frozen_npz_temporal_status(data.files)
    assert data.files == ["X", "y"] or set(data.files) == {"X", "y"}
    assert status["chronological_reconstruction_possible"] is False


def test_split_flows_forbids_random_shuffle_method():
    y = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="forbidden"):
        split_flows_before_windows(10, y, method="random")
