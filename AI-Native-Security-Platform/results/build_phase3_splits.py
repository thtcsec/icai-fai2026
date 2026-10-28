"""Phase 3 split builders. Never writes into chrono_insdn/ or frozen NPZ.

Two *distinct* protocols:

  group_disjoint  — each recording_day|Src IP is in exactly one split.
                    This is host-day generalization, not session modeling.

  purged_chrono   — clock-forward 70/15/15 within a recording day, with a
                    seq_len-1 flow gap at each cut. Overlapping Src IPs may
                    still appear in multiple splits. Purge ≠ independence.

Format A dates stay opaque. Thang6/RAR refused via load_needed_columns.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(__file__))

from build_chrono_insdn import (
    DEFAULT_SRC,
    FEATURE_FROM,
    SEED,
    SEQ_LEN,
    STRIDE,
    TRAIN_RATIO,
    VAL_RATIO,
    _write,
    load_needed_columns,
)
from prototype.data.temporal_windows import (
    WINDOW_LABEL_SEMANTICS,
    shuffle_completed_windows,
    windows_from_ordered_flows,
)

GD_DIR = os.path.join(BASE, "results", "corrected", "phase3_group_disjoint")
PURGE_DIR = os.path.join(BASE, "results", "corrected", "phase3_purged_chrono")
MIN_WINDOWS = 100
MIN_CLASS = 20


def group_hash(gid: str) -> str:
    return hashlib.sha256(str(gid).encode("utf-8")).hexdigest()


def assign_group_disjoint(gid: str, seed: int = SEED) -> str:
    h = hashlib.sha256(f"phase3-gd|{seed}|{gid}".encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def assign_purged_chrono(df, purge: int = SEQ_LEN - 1):
    """Clock-forward 70/15/15 per recording day with a flow gap at each cut.

    ``df`` must have a contiguous RangeIndex (call reset_index first).
    """
    split = np.full(len(df), "drop", dtype=object)
    pos = np.arange(len(df))
    for _, idx in df.groupby("_recording_day", sort=False).groups.items():
        idx = np.asarray(idx, dtype=np.int64)
        clock = df["_clock_s"].to_numpy()[idx]
        src = df["_source_index"].to_numpy()[idx]
        order = np.lexsort((src, clock))
        ordered = idx[order]
        n = len(ordered)
        n_tr = int(round(n * TRAIN_RATIO))
        n_va = int(round(n * VAL_RATIO))
        tr_end = max(0, n_tr - purge)
        va_start = min(n, n_tr + purge)
        va_end = max(va_start, min(n, n_tr + n_va - purge))
        te_start = min(n, n_tr + n_va + purge)
        split[ordered[:tr_end]] = "train"
        split[ordered[va_start:va_end]] = "val"
        split[ordered[te_start:]] = "test"
    _ = pos
    return split


def _window_one(df, scaler, name, temporal_validity, notes):
    sub = df.loc[df["_split"] == name].sort_values(
        ["_group", "_clock_s", "_source_index"], kind="mergesort"
    )
    if len(sub) == 0:
        return windows_from_ordered_flows(
            np.zeros((0, len(FEATURE_FROM)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            seq_len=SEQ_LEN,
            stride=STRIDE,
            temporal_validity=temporal_validity,
            notes=notes + ("empty split",),
        )
    raw = sub[FEATURE_FROM].to_numpy(dtype=np.float32)
    scaled = scaler.transform(raw).astype(np.float32)
    batch = windows_from_ordered_flows(
        scaled,
        sub["_label"].to_numpy(),
        seq_len=SEQ_LEN,
        stride=STRIDE,
        group_ids=sub["_group"].to_numpy(),
        source_indices=sub["_source_index"].to_numpy(dtype=np.int64),
        temporal_validity=temporal_validity,
        notes=notes,
    )
    del sub, raw, scaled
    return batch


def _assert_group_disjoint(batches: dict) -> dict:
    hashes = {s: set() for s in batches}
    for split, batch in batches.items():
        if len(batch.X) == 0:
            continue
        for gid in np.unique(batch.group_ids.astype(str)):
            hashes[split].add(group_hash(gid))
    ov = {
        "train_val": int(len(hashes["train"] & hashes["val"])),
        "train_test": int(len(hashes["train"] & hashes["test"])),
        "val_test": int(len(hashes["val"] & hashes["test"])),
    }
    if any(v > 0 for v in ov.values()):
        raise AssertionError(f"Group-disjoint violated: {ov}. Isolation was NOT relaxed.")
    return {
        "n_group_hashes": {s: int(len(h)) for s, h in hashes.items()},
        "overlap": ov,
    }


def _viability(batches: dict) -> dict:
    report = {"splits": {}, "viable": True, "blockers": []}
    for name, batch in batches.items():
        n = int(len(batch.X))
        att = int((batch.y > 0).sum()) if n else 0
        nor = int((batch.y == 0).sum()) if n else 0
        rec = {"windows": n, "attack": att, "normal": nor}
        report["splits"][name] = rec
        if n < MIN_WINDOWS:
            report["viable"] = False
            report["blockers"].append(f"{name}: windows={n} < {MIN_WINDOWS}")
        if att < MIN_CLASS:
            report["viable"] = False
            report["blockers"].append(f"{name}: attack={att} < {MIN_CLASS}")
        if nor < MIN_CLASS:
            report["viable"] = False
            report["blockers"].append(f"{name}: normal={nor} < {MIN_CLASS}")
    return report


def build_group_disjoint(src: str = DEFAULT_SRC, out_dir: str = GD_DIR):
    print("[*] Phase 3 GROUP-DISJOINT (recording_day|Src IP wholly in one split)")
    df, n_dropped = load_needed_columns(src)
    df = df.reset_index(drop=True)
    df["_split"] = df["_group"].map(lambda g: assign_group_disjoint(str(g)))
    print("    flow split", df["_split"].value_counts().to_dict())

    train_df = df[df["_split"] == "train"]
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_FROM].to_numpy(dtype=np.float32))
    del train_df

    notes = (
        "Group-disjoint: recording_day|Src IP never crosses splits.",
        "Not session-level (Flow ID unused). Format A days remain opaque.",
        "Scaler fit on train flows only.",
    )
    validity = "group_disjoint_recording_day_srcip"
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ghashes = {s: set() for s in ("train", "val", "test")}
    src_rows = {s: set() for s in ("train", "val", "test")}
    fake_batches = {}
    partitions = {}
    for name in ("train", "val", "test"):
        batch = _window_one(df, scaler, name, validity, notes)
        if name == "train" and len(batch.X):
            batch = shuffle_completed_windows(batch, rng)
        n = int(len(batch.X))
        att = int((batch.y > 0).sum()) if n else 0
        print(f"    {name} windows={n} attack={att}")
        if n:
            for gid in np.unique(batch.group_ids.astype(str)):
                ghashes[name].add(group_hash(gid))
            src_rows[name].update(map(int, np.unique(batch.source_indices)))
        fake_batches[name] = batch
        rec = {
            "split": name,
            "X_shape": list(batch.X.shape),
            "attack": att,
            "normal": int((batch.y == 0).sum()) if n else 0,
        }
        partitions[name] = rec
        # keep batch only for viability + write after isolation check
    isolation_ov = {
        "train_val": int(len(ghashes["train"] & ghashes["val"])),
        "train_test": int(len(ghashes["train"] & ghashes["test"])),
        "val_test": int(len(ghashes["val"] & ghashes["test"])),
    }
    if any(v > 0 for v in isolation_ov.values()):
        raise AssertionError(
            f"Group-disjoint violated: {isolation_ov}. Isolation was NOT relaxed."
        )
    isolation = {
        "n_group_hashes": {s: int(len(h)) for s, h in ghashes.items()},
        "overlap": isolation_ov,
    }
    viability = _viability(fake_batches)
    row_ov = {
        "train_val": int(len(src_rows["train"] & src_rows["val"])),
        "train_test": int(len(src_rows["train"] & src_rows["test"])),
        "val_test": int(len(src_rows["val"] & src_rows["test"])),
    }
    manifest = {
        "protocol": "group_disjoint_recording_day_srcip",
        "not_session_level": True,
        "not_forward_in_time": True,
        "format_A_guessed": False,
        "frozen_npz_used": False,
        "thang6_archive_touched": False,
        "source_csv": src,
        "n_flows": int(len(df)),
        "n_clock_dropped": int(n_dropped),
        "seq_len": SEQ_LEN,
        "seed": SEED,
        "scaler": "StandardScaler fit on train flows only",
        "label_semantics": WINDOW_LABEL_SEMANTICS,
        "temporal_validity": validity,
        "group_isolation": isolation,
        "source_row_overlap": row_ov,
        "viability": viability,
        "partitions": {},
    }
    if not viability["viable"]:
        man_path = os.path.join(out_dir, "manifest.json")
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        raise RuntimeError(
            "GROUP-DISJOINT BLOCKED: " + "; ".join(viability["blockers"])
            + f" (wrote {man_path}; isolation was not relaxed)"
        )
    for name, batch in fake_batches.items():
        rec = _write(os.path.join(out_dir, f"{name}_windows.npz"), batch, {"split": name})
        rec["group_hashes_n"] = isolation["n_group_hashes"][name]
        manifest["partitions"][name] = rec
        fake_batches[name] = None
        gc.collect()
    man_path = os.path.join(out_dir, "manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] {man_path} viable={viability['viable']}")
    del df, scaler, fake_batches
    gc.collect()
    return manifest


def build_purged_chrono(src: str = DEFAULT_SRC, out_dir: str = PURGE_DIR):
    print("[*] Phase 3 PURGED CHRONO (forward-in-time; entities MAY overlap)")
    df, n_dropped = load_needed_columns(src)
    df = df.reset_index(drop=True)
    df["_split"] = assign_purged_chrono(df)
    print("    flow split", df["_split"].value_counts().to_dict(), "dropped", int((df["_split"]=="drop").sum()))
    train_df = df[df["_split"] == "train"]
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_FROM].to_numpy(dtype=np.float32))
    del train_df
    notes = (
        "Purged clock-forward split per recording day. seq_len-1 flow gap at cuts.",
        "Src IP / recording_day|Src IP MAY still appear in multiple splits.",
        "Purge does not imply entity independence. Not session-level.",
    )
    validity = "purged_clock_within_recording_day"
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    batches = {}
    for name in ("train", "val", "test"):
        batch = _window_one(df, scaler, name, validity, notes)
        if name == "train" and len(batch.X):
            batch = shuffle_completed_windows(batch, rng)
        batches[name] = batch
        print(f"    {name} windows={len(batch.X)}")

    hashes = {s: set(map(str, np.unique(batches[s].group_ids.astype(str)))) if len(batches[s].X) else set() for s in batches}
    entity_ov = {
        "train_val": int(len(hashes["train"] & hashes["val"])),
        "train_test": int(len(hashes["train"] & hashes["test"])),
        "val_test": int(len(hashes["val"] & hashes["test"])),
    }
    src_rows = {s: set(map(int, np.unique(batches[s].source_indices))) if len(batches[s].X) else set() for s in batches}
    row_ov = {
        "train_val": int(len(src_rows["train"] & src_rows["val"])),
        "train_test": int(len(src_rows["train"] & src_rows["test"])),
        "val_test": int(len(src_rows["val"] & src_rows["test"])),
    }
    viability = _viability(batches)
    manifest = {
        "protocol": "purged_clock_within_recording_day",
        "not_group_disjoint": True,
        "not_session_level": True,
        "entity_overlap_expected": True,
        "purge_flows": SEQ_LEN - 1,
        "format_A_guessed": False,
        "frozen_npz_used": False,
        "thang6_archive_touched": False,
        "source_csv": src,
        "n_flows_kept_in_splits": int((df["_split"] != "drop").sum()),
        "n_flows_purged_or_unassigned": int((df["_split"] == "drop").sum()),
        "n_clock_dropped": int(n_dropped),
        "temporal_validity": validity,
        "label_semantics": WINDOW_LABEL_SEMANTICS,
        "entity_group_overlap": entity_ov,
        "source_row_overlap": row_ov,
        "viability": viability,
        "partitions": {},
        "warning": (
            "A purge gap removes adjacent-window row overlap at the cut. "
            "It does NOT make overlapping Src IPs independent."
        ),
    }
    for name, batch in batches.items():
        rec = _write(os.path.join(out_dir, f"{name}_windows.npz"), batch, {"split": name})
        manifest["partitions"][name] = rec
        batches[name] = None
        gc.collect()
    man_path = os.path.join(out_dir, "manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] {man_path} entity_overlap={entity_ov} row_overlap={row_ov}")
    del df, scaler
    gc.collect()
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=("group_disjoint", "purged_chrono", "both"), default="both")
    args = ap.parse_args()
    out = {}
    if args.protocol in ("group_disjoint", "both"):
        out["group_disjoint"] = build_group_disjoint()
    if args.protocol in ("purged_chrono", "both"):
        out["purged_chrono"] = build_purged_chrono()
    return out


if __name__ == "__main__":
    main()
