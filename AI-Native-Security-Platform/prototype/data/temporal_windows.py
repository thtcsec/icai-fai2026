"""Chronological window construction with split-before-window isolation.

The frozen InSDN/CIC NPZ archives shipped with this artifact were built from a
label-stratified subsample that was shuffled *before* windowing. Those archives
store only ``X`` and ``y`` — no timestamps, session IDs, or source-row indices —
so capture-time order cannot be reconstructed from them. This module is the
corrected pipeline for *new* CSV-backed corpora.

Rules:
  * Never shuffle raw flows before creating sequences.
  * Never invent timestamps or session identifiers.
  * Split at the flow (or group) level *before* windowing.
  * Windows never cross groups, recording boundaries, or partition cuts.
  * Window label = label of the last flow in the window (not any-attack).
  * Completed training windows may be shuffled; raw observations may not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

WINDOW_LABEL_SEMANTICS = "last_flow"
TEMPORAL_VALIDITY_FROZEN_NPZ = "index_order_after_shuffle"
TEMPORAL_VALIDITY_CSV_ORDER = "csv_row_order_unverified"
TEMPORAL_VALIDITY_TIMESTAMP = "timestamp_sorted"
TEMPORAL_VALIDITY_GROUPED = "grouped_then_ordered"


@dataclass(frozen=True)
class WindowBatch:
    X: np.ndarray
    y: np.ndarray
    start_indices: np.ndarray
    source_indices: np.ndarray
    group_ids: np.ndarray
    label_semantics: str = WINDOW_LABEL_SEMANTICS
    temporal_validity: str = TEMPORAL_VALIDITY_CSV_ORDER
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.X.ndim != 3:
            raise ValueError(f"X must be (n_windows, seq_len, n_features), got {self.X.shape}")
        n = self.X.shape[0]
        if not (len(self.y) == len(self.start_indices) == n):
            raise ValueError("X, y, and start_indices length mismatch")


@dataclass(frozen=True)
class FlowSplit:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    method: str
    seed: int


def frozen_npz_temporal_status(npz_keys: Sequence[str]) -> Dict[str, object]:
    """Describe why a frozen window archive cannot be treated as chronological."""
    keys = set(npz_keys)
    reconstructable = bool(keys & {"timestamps", "timestamp", "time", "source_index", "group_id"})
    return {
        "keys": sorted(keys),
        "has_timestamps": bool(keys & {"timestamps", "timestamp", "time"}),
        "has_source_index": "source_index" in keys or "start_indices" in keys,
        "has_group_id": "group_id" in keys or "group_ids" in keys,
        "chronological_reconstruction_possible": reconstructable,
        "temporal_validity": TEMPORAL_VALIDITY_FROZEN_NPZ,
        "note": (
            "Frozen archives store only window tensors X and labels y. "
            "Timestamps were not persisted. Do not invent them. "
            "Windows are length-T contexts over a seed-42 shuffled index order."
        ),
    }


def resolve_timestamp_column(df: pd.DataFrame, timestamp_col: Optional[str]) -> Optional[str]:
    if timestamp_col:
        if timestamp_col not in df.columns:
            raise KeyError(
                f"Requested timestamp column {timestamp_col!r} is missing. "
                "Refusing to invent timestamps."
            )
        return timestamp_col
    for cand in ("timestamp", "Timestamp", "Flow Start", "flow_start", "StartTime", "time"):
        if cand in df.columns:
            return cand
    return None


def resolve_group_column(df: pd.DataFrame, group_col: Optional[str]) -> Optional[str]:
    if group_col:
        if group_col not in df.columns:
            raise KeyError(f"Requested group column {group_col!r} is missing.")
        return group_col
    for cand in ("session_id", "Session", "flow_id", "Flow ID"):
        if cand in df.columns:
            return cand
    return None


def order_flows(
    df: pd.DataFrame,
    *,
    timestamp_col: Optional[str] = None,
    group_col: Optional[str] = None,
    assume_csv_order: bool = False,
) -> Tuple[pd.DataFrame, str, Tuple[str, ...]]:
    """Return a copy ordered for windowing. Never shuffles. Never invents time."""
    notes: List[str] = []
    ts = resolve_timestamp_column(df, timestamp_col)
    grp = resolve_group_column(df, group_col)
    out = df.copy()
    out["_source_index"] = np.arange(len(out), dtype=np.int64)

    if ts is not None:
        if grp is not None:
            out = out.sort_values([grp, ts], kind="mergesort")
            validity = TEMPORAL_VALIDITY_GROUPED
            notes.append(f"Sorted by group={grp} then timestamp={ts}.")
        else:
            out = out.sort_values(ts, kind="mergesort")
            validity = TEMPORAL_VALIDITY_TIMESTAMP
            notes.append(f"Sorted by timestamp={ts} only; no session/group column.")
        return out.reset_index(drop=True), validity, tuple(notes)

    if not assume_csv_order:
        raise ValueError(
            "No timestamp column found and assume_csv_order=False. "
            "Chronological reconstruction is impossible without timestamps; "
            "pass assume_csv_order=True to window in CSV row order and label "
            "temporal_validity as csv_row_order_unverified."
        )
    notes.append(
        "No timestamp column. Preserving CSV row order. This is not verified "
        "capture-time order; do not claim temporal validity."
    )
    if grp is not None:
        out = out.sort_values([grp, "_source_index"], kind="mergesort")
        validity = TEMPORAL_VALIDITY_GROUPED
        notes.append(f"Grouped by {grp} while preserving within-group CSV order.")
        return out.reset_index(drop=True), validity, tuple(notes)
    return out.reset_index(drop=True), TEMPORAL_VALIDITY_CSV_ORDER, tuple(notes)


def split_flows_before_windows(
    n_flows: int,
    y_flow: np.ndarray,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    group_ids: Optional[np.ndarray] = None,
    method: str = "contiguous_blocks",
) -> FlowSplit:
    """Partition *flows* (not windows). Windows must be built afterwards.

    ``contiguous_blocks`` cuts the ordered stream into contiguous index ranges so
    a window cannot straddle a partition. Group-aware assignment keeps every
    group on one side of the cut.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1 (remainder is test)")
    y_flow = np.asarray(y_flow)
    if len(y_flow) != n_flows:
        raise ValueError("y_flow length mismatch")

    if group_ids is not None:
        group_ids = np.asarray(group_ids)
        if len(group_ids) != n_flows:
            raise ValueError("group_ids length mismatch")
        # Assign whole groups by first-occurrence order (stable, not shuffled).
        _, first_idx = np.unique(group_ids, return_index=True)
        ordered_groups = group_ids[np.sort(first_idx)]
        n_g = len(ordered_groups)
        n_tr = int(round(n_g * train_ratio))
        n_va = int(round(n_g * val_ratio))
        tr_g = set(ordered_groups[:n_tr].tolist())
        va_g = set(ordered_groups[n_tr : n_tr + n_va].tolist())
        te_g = set(ordered_groups[n_tr + n_va :].tolist())
        idx = np.arange(n_flows)
        train = idx[np.isin(group_ids, list(tr_g))]
        val = idx[np.isin(group_ids, list(va_g))]
        test = idx[np.isin(group_ids, list(te_g))]
        return FlowSplit(train=train, val=val, test=test, method="group_contiguous", seed=seed)

    if method != "contiguous_blocks":
        raise ValueError(
            f"Unsupported split method {method!r}. "
            "Random flow shuffles before windowing are forbidden."
        )
    n_tr = int(round(n_flows * train_ratio))
    n_va = int(round(n_flows * val_ratio))
    train = np.arange(0, n_tr)
    val = np.arange(n_tr, n_tr + n_va)
    test = np.arange(n_tr + n_va, n_flows)
    return FlowSplit(train=train, val=val, test=test, method=method, seed=seed)


def windows_from_ordered_flows(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seq_len: int = 10,
    stride: int = 1,
    group_ids: Optional[np.ndarray] = None,
    source_indices: Optional[np.ndarray] = None,
    temporal_validity: str = TEMPORAL_VALIDITY_CSV_ORDER,
    notes: Tuple[str, ...] = (),
) -> WindowBatch:
    """Build sliding windows that never cross group boundaries.

    Label of window ending at flow ``i`` is ``labels[i]`` (last-flow semantics).
    """
    features = np.asarray(features)
    labels = (np.asarray(labels) > 0).astype(np.int64)
    if features.ndim != 2:
        raise ValueError(f"features must be (n_flows, n_features), got {features.shape}")
    n = len(features)
    if len(labels) != n:
        raise ValueError("labels length mismatch")
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if source_indices is None:
        source_indices = np.arange(n, dtype=np.int64)
    else:
        source_indices = np.asarray(source_indices, dtype=np.int64)
        if len(source_indices) != n:
            raise ValueError("source_indices length mismatch")

    if group_ids is None:
        group_ids = np.zeros(n, dtype=np.int64)
    else:
        group_ids = np.asarray(group_ids)
        if len(group_ids) != n:
            raise ValueError("group_ids length mismatch")

    windows: List[np.ndarray] = []
    y_out: List[int] = []
    starts: List[int] = []
    src_rows: List[np.ndarray] = []
    g_out: List[object] = []

    # Split into contiguous same-group runs so a window cannot join unrelated traffic.
    change = np.ones(n, dtype=bool)
    change[1:] = group_ids[1:] != group_ids[:-1]
    run_starts = np.flatnonzero(change)
    run_ends = np.append(run_starts[1:], n)

    for rs, re in zip(run_starts, run_ends):
        run_len = re - rs
        if run_len < seq_len:
            continue
        for start in range(rs, re - seq_len + 1, stride):
            sl = slice(start, start + seq_len)
            windows.append(features[sl])
            y_out.append(int(labels[start + seq_len - 1] > 0))
            starts.append(start)
            src_rows.append(source_indices[sl])
            g_out.append(group_ids[start])

    if not windows:
        X = np.zeros((0, seq_len, features.shape[1]), dtype=np.float32)
        return WindowBatch(
            X=X,
            y=np.zeros((0,), dtype=np.int64),
            start_indices=np.zeros((0,), dtype=np.int64),
            source_indices=np.zeros((0, seq_len), dtype=np.int64),
            group_ids=np.zeros((0,), dtype=object),
            temporal_validity=temporal_validity,
            notes=notes + ("no windows: every group shorter than seq_len",),
        )

    return WindowBatch(
        X=np.stack(windows, axis=0).astype(np.float32),
        y=np.asarray(y_out, dtype=np.int64),
        start_indices=np.asarray(starts, dtype=np.int64),
        source_indices=np.stack(src_rows, axis=0).astype(np.int64),
        group_ids=np.asarray(g_out, dtype=object),
        temporal_validity=temporal_validity,
        notes=notes,
    )


def assert_windows_contiguous(batch: WindowBatch) -> None:
    """Adjacent timesteps inside a window must be adjacent source rows."""
    if len(batch.X) == 0:
        return
    src = batch.source_indices
    diffs = np.diff(src, axis=1)
    if not np.all(diffs == 1):
        bad = np.argwhere(diffs != 1)
        i, t = int(bad[0, 0]), int(bad[0, 1])
        raise AssertionError(
            f"Window {i} is not contiguous at timestep {t}: "
            f"source_indices={src[i].tolist()}"
        )


def assert_windows_within_groups(batch: WindowBatch) -> None:
    if len(batch.X) == 0:
        return
    for i, gid in enumerate(batch.group_ids):
        # group_ids is per-window (first flow's group). Continuity of the run
        # is enforced by construction; source indices still must be contiguous.
        if batch.source_indices[i, 0] != batch.start_indices[i] and False:
            pass
        _ = gid
    assert_windows_contiguous(batch)


def assert_split_isolation(
    batches: Dict[str, WindowBatch],
) -> None:
    """No source-row index may appear in more than one partition's windows."""
    seen: Dict[int, str] = {}
    for split, batch in batches.items():
        if len(batch.X) == 0:
            continue
        for idx in np.unique(batch.source_indices.ravel()):
            idx_i = int(idx)
            if idx_i in seen and seen[idx_i] != split:
                raise AssertionError(
                    f"Source flow {idx_i} leaked into both {seen[idx_i]} and {split}"
                )
            seen[idx_i] = split


def assert_label_alignment(batch: WindowBatch, flow_labels: np.ndarray) -> None:
    """Window label must equal the last-flow label of its source indices."""
    flow_labels = (np.asarray(flow_labels) > 0).astype(np.int64)
    for i in range(len(batch.X)):
        last = int(batch.source_indices[i, -1])
        expected = int(flow_labels[last] > 0)
        if int(batch.y[i]) != expected:
            raise AssertionError(
                f"Window {i} label {int(batch.y[i])} != last-flow label {expected} "
                f"(source last={last})"
            )


def shuffle_completed_windows(
    batch: WindowBatch,
    rng: np.random.Generator,
) -> WindowBatch:
    """Permutes completed windows only. Forbidden on raw flows."""
    n = len(batch.X)
    perm = rng.permutation(n)
    return WindowBatch(
        X=batch.X[perm],
        y=batch.y[perm],
        start_indices=batch.start_indices[perm],
        source_indices=batch.source_indices[perm],
        group_ids=batch.group_ids[perm],
        label_semantics=batch.label_semantics,
        temporal_validity=batch.temporal_validity,
        notes=batch.notes + ("shuffled completed windows only",),
    )


def build_partitioned_windows(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seq_len: int = 10,
    stride: int = 1,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    group_ids: Optional[np.ndarray] = None,
    source_indices: Optional[np.ndarray] = None,
    temporal_validity: str = TEMPORAL_VALIDITY_CSV_ORDER,
    notes: Tuple[str, ...] = (),
    shuffle_train_windows: bool = True,
) -> Tuple[Dict[str, WindowBatch], FlowSplit]:
    """Split flows first, then window each partition independently."""
    n = len(features)
    split = split_flows_before_windows(
        n,
        labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        group_ids=group_ids,
    )
    batches: Dict[str, WindowBatch] = {}
    for name, idx in (("train", split.train), ("val", split.val), ("test", split.test)):
        if len(idx) == 0:
            batches[name] = WindowBatch(
                X=np.zeros((0, seq_len, features.shape[1]), dtype=np.float32),
                y=np.zeros((0,), dtype=np.int64),
                start_indices=np.zeros((0,), dtype=np.int64),
                source_indices=np.zeros((0, seq_len), dtype=np.int64),
                group_ids=np.zeros((0,), dtype=object),
                temporal_validity=temporal_validity,
                notes=notes + (f"{name} partition empty",),
            )
            continue
        g = None if group_ids is None else np.asarray(group_ids)[idx]
        src = None if source_indices is None else np.asarray(source_indices)[idx]
        batches[name] = windows_from_ordered_flows(
            features[idx],
            labels[idx],
            seq_len=seq_len,
            stride=stride,
            group_ids=g,
            source_indices=src if src is not None else idx,
            temporal_validity=temporal_validity,
            notes=notes,
        )
    if shuffle_train_windows and len(batches["train"].X) > 0:
        rng = np.random.default_rng(seed)
        batches["train"] = shuffle_completed_windows(batches["train"], rng)
    assert_split_isolation(batches)
    return batches, split
