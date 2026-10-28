"""Train LightGBM + TCN-GRU on Phase 3 *group-disjoint* windows only.

Does not read chrono_insdn/, frozen NPZ, or prior checkpoints.
Does not overwrite results/corrected/chrono_model_comparison.json
or results/corrected/checkpoints/tcn_gru_chrono.pt.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(__file__))

from prototype.edge.detector.quantize import quantize_tcn_gru_model
from prototype.edge.detector.tcn_gru_model import TCNGRUResilienceModel
from train_eval_chrono import MAX_TEST, MAX_TRAIN, MAX_VAL, _cap
from train_utils import (
    classifier_binary_scores,
    select_operating_threshold,
    set_global_seed,
    threshold_metrics,
    train_autoencoder,
)

EXP = os.path.join(BASE, "results", "corrected", "phase3_group_disjoint")
OUT_JSON = os.path.join(EXP, "model_comparison.json")
CKPT_OUT = os.path.join(EXP, "tcn_gru_fp32.pt")


def _load(split: str):
    path = os.path.join(EXP, f"{split}_windows.npz")
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = (data["y"] > 0).astype(np.int64)
    validity = str(data["temporal_validity"]) if "temporal_validity" in data.files else "unknown"
    gids = data["group_ids"] if "group_ids" in data.files else None
    return X, y, validity, gids


def _pack(y_true, y_pred, scores=None, extra=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_average": "binary",
        "pos_label": 1,
        "support_normal": int((y_true == 0).sum()),
        "support_attack": int((y_true == 1).sum()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if scores is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    if extra:
        out.update(extra)
    return out


def run(seed: int = 42):
    man_path = os.path.join(EXP, "manifest.json")
    if not os.path.isfile(man_path):
        raise FileNotFoundError("Run build_phase3_splits.py --protocol group_disjoint first")
    man = json.load(open(man_path, encoding="utf-8"))
    if not man.get("viability", {}).get("viable", False):
        raise RuntimeError(f"Refusing to train on non-viable split: {man.get('viability')}")
    if man.get("group_isolation", {}).get("overlap", {}) and any(
        man["group_isolation"]["overlap"].values()
    ):
        raise RuntimeError("Refusing to train: group overlap in manifest")

    print("[*] Phase 3 group-disjoint LightGBM / TCN (new weights; val-frozen tau)")
    set_global_seed(seed)
    Xtr, ytr, validity, gtr = _load("train")
    Xva, yva, _, gva = _load("val")
    Xte, yte, _, gte = _load("test")
    if "group_disjoint" not in validity:
        raise RuntimeError(f"Unexpected temporal_validity={validity}")
    Xtr, ytr = _cap(Xtr, ytr, MAX_TRAIN, seed, "train")
    Xva, yva = _cap(Xva, yva, MAX_VAL, seed + 1, "val")
    Xte, yte = _cap(Xte, yte, MAX_TEST, seed + 2, "test")
    print(f"    n={len(Xtr)}/{len(Xva)}/{len(Xte)} validity={validity}")

    report = {
        "protocol": "group_disjoint_recording_day_srcip",
        "not_session_level": True,
        "frozen_npz_used": False,
        "reused_prior_chrono_checkpoint": False,
        "temporal_validity": validity,
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "n_test": int(len(Xte)),
        "models": {},
        "f1_convention": "sklearn binary pos_label=1 (attack)",
    }

    maj = DummyClassifier(strategy="most_frequent")
    maj.fit(Xtr.reshape(len(Xtr), -1), ytr)
    pred = maj.predict(Xte.reshape(len(Xte), -1))
    report["models"]["majority"] = _pack(yte, pred)

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
        yte, (s_te > tau).astype(int), scores=s_te,
        extra={"tau": float(tau), "tau_provenance": "val", "input": "flattened_100d"},
    )
    print(f"  [+] LightGBM F1={report['models']['lightgbm']['f1']:.4f}")

    model = train_autoencoder(
        TCNGRUResilienceModel(num_features=10, num_classes=6),
        torch.tensor(Xtr), torch.tensor(ytr), epochs=12, batch_size=128, seed=seed,
    )
    qmodel = quantize_tcn_gru_model(model)
    os.makedirs(os.path.dirname(CKPT_OUT), exist_ok=True)
    torch.save(model.state_dict(), CKPT_OUT)
    s_va = classifier_binary_scores(qmodel, torch.tensor(Xva))
    tau, _ = select_operating_threshold(yva, s_va, split="val")
    s_te = classifier_binary_scores(qmodel, torch.tensor(Xte))
    m = threshold_metrics(yte, s_te, tau)
    packed = _pack(
        yte, (s_te > tau).astype(int), scores=s_te,
        extra={
            "tau": float(tau),
            "tau_provenance": "val",
            "checkpoint_fp32": CKPT_OUT,
            "quantization": "dynamic Linear+GRU qint8 after FP32 train",
            "saved_checkpoint_is_int8": False,
            "input": "sequence_10x10",
        },
    )
    packed.update({k: m[k] for k in ("precision", "recall", "f1")})
    report["models"]["tcn_gru_int8"] = packed
    print(f"  [+] TCN-GRU INT8 F1={packed['f1']:.4f} tau={tau:.4f}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  [=] Wrote {OUT_JSON}")
    return report


if __name__ == "__main__":
    run()
