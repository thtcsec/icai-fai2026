"""Group-disjoint assignment: a grouping key never crosses splits."""

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

from build_phase3_splits import assign_group_disjoint, assign_purged_chrono, group_hash


def test_group_disjoint_key_never_crosses_splits():
    keys = [f"10/1/2020|10.0.0.{i}" for i in range(40)] + [
        f"25/12/2019|10.0.1.{i}" for i in range(40)
    ]
    assigned = {k: assign_group_disjoint(k, seed=42) for k in keys}
    buckets = {"train": set(), "val": set(), "test": set()}
    for k, s in assigned.items():
        buckets[s].add(group_hash(k))
    assert not (buckets["train"] & buckets["val"])
    assert not (buckets["train"] & buckets["test"])
    assert not (buckets["val"] & buckets["test"])
    assert all(buckets[s] for s in ("train", "val", "test"))


def test_same_key_always_same_split():
    k = "27/12/2019|200.175.2.130"
    splits = {assign_group_disjoint(k, seed=42) for _ in range(20)}
    assert len(splits) == 1


def test_purged_chrono_drops_boundary_and_can_overlap_entities():
    import pandas as pd

    n = 200
    df = pd.DataFrame(
        {
            "_recording_day": ["d"] * n,
            "_clock_s": np.arange(n, dtype=float),
            "_source_index": np.arange(n),
            "_group": ["d|10.0.0.1"] * n,
        }
    )
    split = assign_purged_chrono(df, purge=9)
    assert "drop" in set(split)
    # One Src IP spans the day, so the same entity is allowed in multiple splits.
    present = {s for s in split if s != "drop"}
    assert present == {"train", "val", "test"}


def test_isolation_not_silently_relaxed():
    src = open(os.path.join(RESULTS, "build_phase3_splits.py"), encoding="utf-8").read()
    assert "Isolation was NOT relaxed" in src
    assert "GROUP-DISJOINT BLOCKED" in src
    assert "Not session-level" in src or "not_session_level" in src
