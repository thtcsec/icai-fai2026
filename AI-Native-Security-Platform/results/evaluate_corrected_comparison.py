"""Fair detector comparison with validation-frozen thresholds.

Uses the same blocked+purged InSDN split as the paper tables, carves validation
from train, and freezes the decision threshold before touching test labels.

This is *not* a capture-time temporal evaluation: the frozen NPZ remains
index-order-after-shuffle. LightGBM vs TCN-GRU is reported honestly.

Writes results/corrected/model_comparison.json. Does not overwrite paper CSVs.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from prototype.data.data_loader import resolve_real_windows_npz
from train_utils import (
    blocked_split_npz,
    classifier_binary_scores,
    select_operating_threshold,
    set_global_seed,
    threshold_metrics,
)

OUT_DIR = os.path.join(BASE_DIR, "results", "corrected")
OUT_JSON = os.path.join(OUT_DIR, "model_comparison.json")
DETECTOR_CKPT = os.path.join(BASE_DIR, "results", "checkpoints", "tcn_gru_ae_trained.pt")


def _pack(y_true, y_pred, extra=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if extra:
        out.update(extra)
    return out


def _eval_scores(y_val, s_val, y_te, s_te, split_name="val"):
    tau, _ = select_operating_threshold(y_val, s_val, split=split_name)
    pred = (s_te > tau).astype(int)
    return _pack(y_te, pred, extra={"tau": float(tau), "tau_provenance": "validation"})


def run(seed: int = 42):
    print("[*] Corrected comparison (val-frozen threshold, frozen shuffled NPZ)...")
    set_global_seed(seed)
    npz = resolve_real_windows_npz("insdn")
    if npz is None:
        raise FileNotFoundError("insdn_windows.npz required")
    X_tr, y_tr, X_va, y_va, X_te, y_te, _ = blocked_split_npz(
        npz, seq_len=10, seed=seed, val_fraction=0.2
    )
    y_tr_np, y_va_np, y_te_np = y_tr.numpy(), y_va.numpy(), y_te.numpy()
    X_tr_flat = X_tr.numpy().reshape(len(X_tr), -1)
    X_va_flat = X_va.numpy().reshape(len(X_va), -1)
    X_te_flat = X_te.numpy().reshape(len(X_te), -1)

    report = {
        "temporal_validity": "index_order_after_shuffle",
        "split": {
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_va)),
            "n_test": int(len(X_te)),
            "protocol": "blocked+purged; val carved from train; scaler fit on train only",
        },
        "seed": seed,
        "models": {},
        "original_paper": {
            "lightgbm_f1": 0.9886,
            "tcn_gru_int8_f1": 0.9723,
            "cross_dataset_cic_f1_065": 0.4564,
            "note": "Original F1 used test-split threshold search on shuffled-index windows.",
        },
    }

    maj = DummyClassifier(strategy="most_frequent")
    maj.fit(X_tr_flat, y_tr_np)
    report["models"]["majority"] = _pack(y_te_np, maj.predict(X_te_flat), extra={"tau": None})

    rnd = DummyClassifier(strategy="uniform", random_state=seed)
    rnd.fit(X_tr_flat, y_tr_np)
    report["models"]["uniform_random"] = _pack(y_te_np, rnd.predict(X_te_flat), extra={"tau": None})

    try:
        from lightgbm import LGBMClassifier

        lgbm = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.1,
            num_leaves=31,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm.fit(X_tr_flat, y_tr_np)
        report["models"]["lightgbm"] = _eval_scores(
            y_va_np,
            lgbm.predict_proba(X_va_flat)[:, 1],
            y_te_np,
            lgbm.predict_proba(X_te_flat)[:, 1],
        )
        print(
            f"  [+] LightGBM test F1={report['models']['lightgbm']['f1']:.4f} "
            f"tau={report['models']['lightgbm']['tau']:.4f}"
        )
    except ImportError:
        report["models"]["lightgbm"] = {"skipped": True, "reason": "lightgbm not installed"}
        print("  [!] LightGBM skipped (not installed)")

    if os.path.isfile(DETECTOR_CKPT):
        try:
            from prototype.edge.detector.inference import load_detector_checkpoint

            model, meta = load_detector_checkpoint(DETECTOR_CKPT, quantize=True)
            s_va = classifier_binary_scores(model, X_va)
            s_te = classifier_binary_scores(model, X_te)
            report["models"]["tcn_gru_int8_existing_ckpt"] = _eval_scores(
                y_va_np, s_va, y_te_np, s_te
            )
            report["models"]["tcn_gru_int8_existing_ckpt"]["checkpoint"] = meta["checkpoint_path"]
            report["models"]["tcn_gru_int8_existing_ckpt"]["note"] = (
                "Weights from original training run; threshold re-selected on validation. "
                "Not a full retrain on the carved train split."
            )
            print(
                f"  [+] TCN-GRU INT8 (existing ckpt) test F1="
                f"{report['models']['tcn_gru_int8_existing_ckpt']['f1']:.4f} "
                f"tau={report['models']['tcn_gru_int8_existing_ckpt']['tau']:.4f}"
            )
        except Exception as exc:
            report["models"]["tcn_gru_int8_existing_ckpt"] = {
                "skipped": True,
                "reason": str(exc),
            }
            print(f"  [!] TCN-GRU existing ckpt skipped: {exc}")
    else:
        report["models"]["tcn_gru_int8_existing_ckpt"] = {
            "skipped": True,
            "reason": "checkpoint missing",
        }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  [=] Wrote {OUT_JSON}")
    return report


if __name__ == "__main__":
    run()
