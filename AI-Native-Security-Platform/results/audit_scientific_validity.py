"""Inexpensive scientific-validity checks. Does not retrain. Does not open RAR."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = os.path.abspath(os.path.join(BASE, ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(__file__))

from prototype.edge.detector.quantize import quantize_tcn_gru_model
from prototype.edge.detector.tcn_gru_model import TCNGRUResilienceModel
from train_eval_chrono import MAX_TEST, MAX_VAL, _cap
from train_utils import classifier_binary_scores, select_operating_threshold, threshold_metrics

CSV = r"D:\huflit_logs\public_datasets\insdn\Dataset.csv"
CHRONO = os.path.join(BASE, "results", "corrected", "chrono_insdn")
TCN_CHRONO = os.path.join(BASE, "results", "corrected", "checkpoints", "tcn_gru_chrono.pt")
TCN_SUBMITTED = os.path.join(BASE, "results", "checkpoints", "tcn_gru_ae_trained.pt")
DQN_CKPT = os.path.join(BASE, "results", "checkpoints", "dqn_policy_trained.pt")
TRAIN_HASH = os.path.join(BASE, "results", "corrected", "dqn_train_state_hashes.bin")
OUT = os.path.join(REPO, "docs", "audit", "icai2026-evidence-manifest.json")
REPORTED = os.path.join(BASE, "results", "corrected", "chrono_model_comparison.json")
MANIFEST = os.path.join(CHRONO, "manifest.json")


def _sha256(path: str) -> dict:
    if not os.path.isfile(path):
        return {"path": path, "exists": False}
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {
        "path": path,
        "exists": True,
        "bytes": os.path.getsize(path),
        "sha256": h.hexdigest(),
    }


def _cap_xy(split: str, n: int, seed: int):
    path = os.path.join(CHRONO, f"{split}_windows.npz")
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = (data["y"] > 0).astype(np.int64)
    validity = str(data["temporal_validity"])
    X, y = _cap(X, y, n, seed, split)
    return X, y, validity, set(data.files)


def inspect_checkpoint(path: str) -> dict:
    rec = _sha256(path)
    if not rec.get("exists"):
        rec["kind"] = "missing"
        return rec
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        rec["kind"] = "wrapped_dict_with_state_dict"
        rec["wrapper_keys"] = sorted(blob.keys())
        sd = blob["state_dict"]
        rec["extra"] = {k: blob[k] for k in blob if k != "state_dict" and not torch.is_tensor(blob[k])}
    elif isinstance(blob, dict) and all(torch.is_tensor(v) for v in blob.values()):
        rec["kind"] = "raw_state_dict"
        sd = blob
    else:
        rec["kind"] = type(blob).__name__
        rec["note"] = "unrecognized checkpoint layout"
        return rec
    rec["n_tensors"] = len(sd)
    rec["tensor_names"] = sorted(sd.keys())
    rec["has_quantized_packed_params"] = any("packed_params" in k or "qint" in str(sd[k].dtype) for k in sd)
    rec["dtypes"] = sorted({str(t.dtype) for t in sd.values()})
    return rec


def rescore_tcn_from_saved_fp32() -> dict:
    """Load saved FP32 weights, dynamically quantize, freeze tau on val, score test.

    Does not train. Reconstructs the documented subsample seeds (val=43, test=44).
    """
    Xva, yva, validity, files_va = _cap_xy("val", MAX_VAL, 43)
    Xte, yte, _, files_te = _cap_xy("test", MAX_TEST, 44)
    model = TCNGRUResilienceModel(num_features=10, num_classes=6)
    blob = torch.load(TCN_CHRONO, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        model.load_state_dict(blob["state_dict"])
        loaded_how = "wrapped_state_dict"
    else:
        model.load_state_dict(blob)
        loaded_how = "raw_state_dict"
    model.eval()
    qmodel = quantize_tcn_gru_model(model)
    s_va = classifier_binary_scores(qmodel, torch.tensor(Xva))
    tau, _ = select_operating_threshold(yva, s_va, split="val")
    s_te = classifier_binary_scores(qmodel, torch.tensor(Xte))
    m = threshold_metrics(yte, s_te, tau)
    reported = json.load(open(REPORTED, encoding="utf-8"))["models"]["tcn_gru_int8"]
    return {
        "loaded_how": loaded_how,
        "npz_keys_val": sorted(files_va),
        "temporal_validity": validity,
        "n_val": int(len(yva)),
        "n_test": int(len(yte)),
        "support_normal": int((yte == 0).sum()),
        "support_attack": int((yte == 1).sum()),
        "tau_reproduced": float(tau),
        "metrics_reproduced": m,
        "reported_f1": reported["f1"],
        "reported_tau": reported["tau"],
        "abs_f1_delta": abs(m["f1"] - reported["f1"]),
        "quantization": "torch.quantization.quantize_dynamic Linear+GRU qint8; TCN conv remains FP32",
        "saved_checkpoint_is_int8": False,
        "saved_checkpoint_is_fp32_pre_quant": True,
        "submitted_detector_ckpt_used": False,
        "frozen_npz_used": False,
        "retrained": False,
        "note": "This is checkpoint reload + dynamic quantize + val-frozen tau, not a new training run.",
    }


def isolation_and_groups() -> dict:
    srcs = {}
    groups = {}
    csv_span = {}
    clock_jump_windows = {}
    for split in ("train", "val", "test"):
        data = np.load(os.path.join(CHRONO, f"{split}_windows.npz"), allow_pickle=True)
        si = data["source_indices"]
        gid = data["group_ids"]
        srcs[split] = set(map(int, np.unique(si)))
        groups[split] = set(map(str, np.unique(gid)))
        # CSV-index adjacency inside a window (NOT required for clock order)
        diffs = np.diff(si, axis=1)
        csv_span[split] = {
            "windows": int(len(si)),
            "frac_csv_adjacent": float(np.mean(np.all(diffs == 1, axis=1))) if len(si) else None,
            "frac_any_csv_gap": float(np.mean(np.any(diffs != 1, axis=1))) if len(si) else None,
        }
        # group token format day|ip
        sample = str(gid[0]) if len(gid) else ""
        clock_jump_windows[split] = {
            "group_token_has_pipe": "|" in sample,
            "n_unique_groups": int(len(groups[split])),
        }
        del data, si, gid
    overlap_tv = srcs["train"] & srcs["val"]
    overlap_tt = srcs["train"] & srcs["test"]
    overlap_vt = srcs["val"] & srcs["test"]
    ip_only = {}
    for split, gset in groups.items():
        ip_only[split] = {g.split("|", 1)[1] if "|" in g else g for g in gset}
    return {
        "source_row_overlap": {
            "train_val": int(len(overlap_tv)),
            "train_test": int(len(overlap_tt)),
            "val_test": int(len(overlap_vt)),
        },
        "group_token_overlap": {
            "train_val": int(len(groups["train"] & groups["val"])),
            "train_test": int(len(groups["train"] & groups["test"])),
            "val_test": int(len(groups["val"] & groups["test"])),
        },
        "src_ip_overlap_across_splits": {
            "train_val": int(len(ip_only["train"] & ip_only["val"])),
            "train_test": int(len(ip_only["train"] & ip_only["test"])),
            "val_test": int(len(ip_only["val"] & ip_only["test"])),
        },
        "csv_index_contiguity_inside_windows": csv_span,
        "groups": clock_jump_windows,
        "interpretation": (
            "Zero source-row overlap means the same CSV flow is not in two splits. "
            "Nonzero Src IP overlap means the same host appears in more than one split "
            "(time-forward cut inside a recording day, not session isolation). "
            "CSV-index gaps inside windows are expected: order is clock-then-csv-index, "
            "not original file order."
        ),
    }


def csv_timestamp_stats() -> dict:
    import pandas as pd
    from build_chrono_insdn import _clock_seconds, _family, _recording_day

    if not os.path.isfile(CSV):
        return {"error": "csv_missing", "path": CSV}
    header = open(CSV, encoding="utf-8", errors="replace").readline().strip().split(",")
    df = pd.read_csv(CSV, usecols=["Timestamp", "Src IP"], dtype="string", low_memory=True)
    fam = df["Timestamp"].map(_family)
    day = df["Timestamp"].map(_recording_day)
    clock = df["Timestamp"].map(_clock_seconds)
    grp = day.astype(str) + "|" + df["Src IP"].astype(str)
    n_tie_rows = 0
    n_groups_with_ties = 0
    n_groups = int(grp.nunique())
    # Per-group clock ties (same minute/second, distinct rows)
    tmp = df.assign(_g=grp, _c=clock)
    vc = tmp.groupby(["_g", "_c"], sort=False).size()
    n_groups_with_ties = int((vc > 1).groupby(level=0).any().sum()) if len(vc) else 0
    n_tie_rows = int(vc[vc > 1].sum()) if len(vc) else 0
    dup_ts_ip = int(df.duplicated(subset=["Timestamp", "Src IP"]).sum())
    return {
        "path": CSV,
        "bytes": os.path.getsize(CSV),
        "header_columns": header,
        "has_flow_id": any(c.strip() in {"Flow ID", "FlowID", "flow_id"} for c in header),
        "has_dst_ip": any("Dst IP" in c or "dst_ip" in c.lower() for c in header),
        "n_rows": int(len(df)),
        "n_src_ip": int(df["Src IP"].nunique()),
        "n_recording_day_tokens": int(day.nunique()),
        "recording_day_tokens": sorted(day.dropna().unique().tolist()),
        "family_counts": fam.value_counts().to_dict(),
        "clock_unparseable": int(clock.isna().sum()),
        "duplicate_timestamp_srcip_rows": dup_ts_ip,
        "n_day_srcip_groups": n_groups,
        "n_groups_with_clock_ties": n_groups_with_ties,
        "n_rows_in_tied_clock_buckets": n_tie_rows,
        "grouping_used_in_builder": "recording_day|Src IP (not Flow ID / 5-tuple / session)",
        "format_A_resolution": "HH:MM (no seconds); within-minute order is CSV source_index",
        "format_B_resolution": "HH:MM:SS AM/PM",
        "thang6_opened": False,
    }


def dqn_independence() -> dict:
    from policy_sim import sample_states, states_fingerprint

    train = sample_states(4000, np.random.default_rng(42))
    ev = sample_states(5000, np.random.default_rng(2026))
    fps_tr = states_fingerprint(train)
    fps_ev = states_fingerprint(ev)
    on_disk = b""
    if os.path.isfile(TRAIN_HASH):
        on_disk = open(TRAIN_HASH, "rb").read()
    disk_set = {on_disk[i : i + 32] for i in range(0, len(on_disk), 32) if len(on_disk[i : i + 32]) == 32}
    src_train = open(os.path.join(BASE, "results", "train_dqn_checkpoint.py"), encoding="utf-8").read()
    uses_td = "gamma" in src_train and "target_net" in src_train
    return {
        "train_seed": 42,
        "eval_seed": 2026,
        "sampler": "iid Uniform[0,1]^5; no MDP transition; no trajectory",
        "exact_row_hash_overlap_regenerated": int(len(fps_tr & fps_ev)),
        "n_train_unique": int(len(fps_tr)),
        "n_eval_unique": int(len(fps_ev)),
        "on_disk_train_hashes": int(len(disk_set)),
        "on_disk_vs_regen_train_match": disk_set == fps_tr,
        "shared_scenario_template": True,
        "shared_template_detail": (
            "Train and eval use the same analytic reward_matrix and the same iid "
            "uniform sampler. Independence is exact-row disjointness, not a different environment."
        ),
        "training_objective_in_train_dqn_checkpoint": "MSE(Q(s,a), immediate_reward); 1-step regression, not Bellman TD",
        "script_uses_gamma_or_target_net": uses_td,
        "operational_enforcement": False,
        "reward_environment": "simulated_analytic_campus_contention",
    }


def metric_conventions() -> dict:
    reported = json.load(open(REPORTED, encoding="utf-8"))
    models = reported["models"]
    supports = {
        k: (m.get("support_normal"), m.get("support_attack"))
        for k, m in models.items()
        if isinstance(m, dict) and "support_attack" in m
    }
    same_pop = len(set(supports.values())) == 1
    maj = models["majority"]
    n_pos = maj["support_attack"]
    n = maj["support_normal"] + maj["support_attack"]
    closed_p = n_pos / n
    closed_f1 = 2 * closed_p * 1.0 / (closed_p + 1.0)
    return {
        "f1_function": "sklearn.metrics.f1_score default average='binary', pos_label=1 (attack class)",
        "not_macro": True,
        "not_weighted": True,
        "not_accuracy": True,
        "equivalent_heldout_population": same_pop,
        "supports_by_model": supports,
        "majority_always_attack_cm": maj["confusion_matrix"],
        "majority_f1_closed_form_from_prevalence": closed_f1,
        "majority_f1_reported": maj["f1"],
        "majority_closed_form_abs_delta": abs(closed_f1 - maj["f1"]),
        "lightgbm_representation": "flattened window (n, 100)",
        "tcn_representation": "sequence (n, 10, 10)",
        "shared_features": [
            "Flow Duration", "Flow Pkts/s", "Flow Byts/s", "Flow IAT Mean", "Flow IAT Std",
            "Pkt Len Mean", "Tot Fwd Pkts", "SYN Flag Cnt", "Active Mean", "Idle Mean",
        ],
        "threshold": "select_operating_threshold(..., split='val'); test refused",
        "subsample": {
            "max_train": 14000,
            "max_val": 3000,
            "max_test": 6000,
            "train_seed": 42,
            "val_seed": 43,
            "test_seed": 44,
            "method": "class-stratified without replacement; train shuffled; val/test index-sorted",
        },
        "preprocessing": "StandardScaler fit on train *flows* only, then applied before windowing",
        "label_semantics": "last_flow of each window",
        "num_classes_head": 6,
        "binary_scoring_rule": "1 - P(normal); logits 2-5 unsupervised",
    }


def git_safety() -> dict:
    import subprocess

    def _run(args):
        p = subprocess.run(args, cwd=REPO, capture_output=True, text=True)
        return p.returncode, p.stdout.strip(), p.stderr.strip()

    code, staged, _ = _run(["git", "diff", "--cached", "--name-only"])
    _, untracked, _ = _run(["git", "ls-files", "--others", "--exclude-standard"])
    csv_in_repo = any("huflit" in x.lower() or x.endswith("Dataset.csv") for x in (staged + "\n" + untracked).splitlines())
    heavy = [
        line
        for line in (staged + "\n" + untracked).splitlines()
        if line.endswith((".npz", ".pt", ".bin", ".rar", ".zip")) or "huflit" in line.lower()
    ]
    return {
        "git_diff_cached_ok": code == 0,
        "staged_files": [x for x in staged.splitlines() if x],
        "csv_inside_repo_or_staged": csv_in_repo,
        "heavy_untracked_or_staged": heavy,
        "gitignore_covers_chrono_npz": False,  # filled by caller after reading gitignore
    }


def main():
    print("[*] Scientific validity audit (no retrain, no RAR, no Docker)")
    evidence = {
        "audit_utc": datetime.utcnow().isoformat() + "Z",
        "verdict_placeholder": "filled after checks",
        "commands": {
            "chrono_windows": "py -3.13 results/build_chrono_insdn.py",
            "chrono_train_eval": "py -3.13 results/train_eval_chrono.py",
            "dqn_train": "py -3.13 results/train_dqn_checkpoint.py",
            "dqn_eval": "py -3.13 results/eval_dqn_checkpoint.py",
            "this_audit": "py -3.13 results/audit_scientific_validity.py",
        },
        "hashes": {
            "csv": _sha256(CSV),
            "chrono_manifest": _sha256(MANIFEST),
            "train_windows": _sha256(os.path.join(CHRONO, "train_windows.npz")),
            "val_windows": _sha256(os.path.join(CHRONO, "val_windows.npz")),
            "test_windows": _sha256(os.path.join(CHRONO, "test_windows.npz")),
            "tcn_chrono_ckpt": _sha256(TCN_CHRONO),
            "tcn_submitted_ckpt": _sha256(TCN_SUBMITTED),
            "dqn_ckpt": _sha256(DQN_CKPT),
            "frozen_insdn_npz": _sha256(os.path.join(BASE, "datasets", "processed", "insdn_windows.npz")),
            "chrono_metrics_json": _sha256(REPORTED),
        },
        "checkpoint_layouts": {
            "tcn_chrono": inspect_checkpoint(TCN_CHRONO),
            "tcn_submitted": inspect_checkpoint(TCN_SUBMITTED),
            "dqn": inspect_checkpoint(DQN_CKPT),
        },
    }
    print("    rescoring TCN INT8 from saved FP32 (no train)...")
    evidence["tcn_rescore"] = rescore_tcn_from_saved_fp32()
    print(
        f"    reproduced F1={evidence['tcn_rescore']['metrics_reproduced']['f1']:.6f} "
        f"delta={evidence['tcn_rescore']['abs_f1_delta']:.6g}"
    )
    print("    isolation / groups...")
    evidence["isolation"] = isolation_and_groups()
    print("    csv timestamp stats (Timestamp, Src IP only)...")
    evidence["csv_stats"] = csv_timestamp_stats()
    print("    dqn independence...")
    evidence["dqn"] = dqn_independence()
    evidence["metrics"] = metric_conventions()
    evidence["dataset_manifest"] = json.load(open(MANIFEST, encoding="utf-8"))
    evidence["git"] = git_safety()
    gi = open(os.path.join(BASE, ".gitignore"), encoding="utf-8").read()
    evidence["git"]["gitignore_covers_chrono_npz"] = "chrono_insdn" in gi
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, default=str)
    print(f"[+] wrote {OUT}")
    return evidence


if __name__ == "__main__":
    main()
