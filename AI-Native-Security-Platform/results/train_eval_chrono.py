"""Train/eval TCN-GRU and LightGBM on the *chronological* InSDN windows.

Never reads datasets/processed/insdn_windows.npz. Thresholds freeze on val.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(__file__))

from prototype.edge.detector.quantize import quantize_tcn_gru_model
from prototype.edge.detector.tcn_gru_model import TCNGRUResilienceModel
from train_utils import (
    classifier_binary_scores,
    select_operating_threshold,
    set_global_seed,
    threshold_metrics,
    train_autoencoder,
)

CHRONO = os.path.join(BASE, "results", "corrected", "chrono_insdn")
OUT_JSON = os.path.join(BASE, "results", "corrected", "chrono_model_comparison.json")
CKPT_OUT = os.path.join(BASE, "results", "corrected", "checkpoints", "tcn_gru_chrono.pt")
MAX_TRAIN, MAX_VAL, MAX_TEST = 14000, 3000, 6000


def _load(split: str):
    path = os.path.join(CHRONO, f"{split}_windows.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Chrono windows missing: {path}. Run build_chrono_insdn.py first.")
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = (data["y"] > 0).astype(np.int64)
    validity = str(data["temporal_validity"]) if "temporal_validity" in data.files else "unknown"
    return X, y, validity


def _cap(X, y, n, seed, split):
    if len(X) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    frac0 = len(i0) / max(len(y), 1)
    n0 = min(len(i0), max(1, int(n * frac0)))
    n1 = min(len(i1), n - n0)
    pick = np.concatenate(
        [
            rng.choice(i0, size=n0, replace=False),
            rng.choice(i1, size=n1, replace=False) if n1 else np.array([], dtype=int),
        ]
    )
    if split == "train":
        rng.shuffle(pick)
    else:
        pick = np.sort(pick)
    return X[pick], y[pick]


def _pack(y_true, y_pred, extra=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "support_normal": int((y_true == 0).sum()),
        "support_attack": int((y_true == 1).sum()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if extra:
        out.update(extra)
    return out


def run(seed: int = 42):
    print("[*] Chrono TCN-GRU / LightGBM (val-frozen tau; not frozen shuffled NPZ)")
    set_global_seed(seed)
    Xtr, ytr, validity = _load("train")
    Xva, yva, _ = _load("val")
    Xte, yte, _ = _load("test")
    Xtr, ytr = _cap(Xtr, ytr, MAX_TRAIN, seed, "train")
    Xva, yva = _cap(Xva, yva, MAX_VAL, seed + 1, "val")
    Xte, yte = _cap(Xte, yte, MAX_TEST, seed + 2, "test")
    print(f"    n train/val/test={len(Xtr)}/{len(Xva)}/{len(Xte)} validity={validity}")
    if "shuffle" in validity or "index_order" in validity:
        raise RuntimeError("Refusing to train a chrono model on shuffled-index windows")

    Xtr_t = torch.tensor(Xtr)
    ytr_t = torch.tensor(ytr)
    Xva_t = torch.tensor(Xva)
    Xte_t = torch.tensor(Xte)

    report = {
        "temporal_validity": validity,
        "frozen_npz_used": False,
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "n_test": int(len(Xte)),
        "models": {},
        "label": "chronological_recording_day_srcip_windows_val_frozen_tau",
    }

    maj = DummyClassifier(strategy="most_frequent")
    maj.fit(Xtr.reshape(len(Xtr), -1), ytr)
    report["models"]["majority"] = _pack(yte, maj.predict(Xte.reshape(len(Xte), -1)))

    try:
        from lightgbm import LGBMClassifier

        lgbm = LGBMClassifier(
            n_estimators=200, learning_rate=0.1, num_leaves=31,
            random_state=seed, n_jobs=2, verbose=-1,
        )
        lgbm.fit(Xtr.reshape(len(Xtr), -1), ytr)
        s_va = lgbm.predict_proba(Xva.reshape(len(Xva), -1))[:, 1]
        tau, _ = select_operating_threshold(yva, s_va, split="val")
        s_te = lgbm.predict_proba(Xte.reshape(len(Xte), -1))[:, 1]
        report["models"]["lightgbm"] = _pack(
            yte, (s_te > tau).astype(int), extra={"tau": float(tau), "tau_provenance": "val"}
        )
        print(f"  [+] LightGBM F1={report['models']['lightgbm']['f1']:.4f}")
    except Exception as exc:
        report["models"]["lightgbm"] = {"skipped": True, "reason": str(exc)}
        print(f"  [!] LightGBM skipped: {exc}")

    model = train_autoencoder(
        TCNGRUResilienceModel(num_features=10, num_classes=6),
        Xtr_t, ytr_t, epochs=12, batch_size=128, seed=seed,
    )
    qmodel = quantize_tcn_gru_model(model)
    os.makedirs(os.path.dirname(CKPT_OUT), exist_ok=True)
    torch.save(model.state_dict(), CKPT_OUT)
    s_va = classifier_binary_scores(qmodel, Xva_t)
    tau, _ = select_operating_threshold(yva, s_va, split="val")
    s_te = classifier_binary_scores(qmodel, Xte_t)
    m = threshold_metrics(yte, s_te, tau)
    pred = (s_te > tau).astype(int)
    packed = _pack(yte, pred, extra={"tau": float(tau), "tau_provenance": "val", "checkpoint": CKPT_OUT})
    packed.update({k: m[k] for k in ("precision", "recall", "f1")})
    report["models"]["tcn_gru_int8"] = packed
    print(f"  [+] TCN-GRU INT8 F1={packed['f1']:.4f} tau={tau:.4f}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  [=] Wrote {OUT_JSON}")
    return report


if __name__ == "__main__":
    run()
