"""Recompute Phase 3 TCN ranking/F1 from the EXISTING checkpoint. No retrain."""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = os.path.abspath(os.path.join(BASE, ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(__file__))

from prototype.edge.detector.quantize import quantize_tcn_gru_model
from prototype.edge.detector.tcn_gru_model import TCNGRUResilienceModel
from train_eval_chrono import MAX_TEST, MAX_VAL, _cap
from train_utils import classifier_binary_scores, select_operating_threshold

EXP = os.path.join(BASE, "results", "corrected", "phase3_group_disjoint")
CKPT = os.path.join(EXP, "tcn_gru_fp32.pt")
OUT = os.path.join(REPO, "revisions", "icai2026_20261028", "metric_integrity_and_threshold_transfer.md")
OUTJ = os.path.join(REPO, "revisions", "icai2026_20261028", "metric_integrity.json")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _load(split):
    d = np.load(os.path.join(EXP, f"{split}_windows.npz"), allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = (d["y"] > 0).astype(np.int64)
    g = np.asarray(d["group_ids"]).astype(str)
    return X, y, g


def _cap_with_g(X, y, g, n, seed, split):
    if len(X) <= n:
        return X, y, g
    # replicate train_eval_chrono._cap index logic then slice g
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
    return X[pick], y[pick], g[pick]


def _cm(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def _f1(y, pred):
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "cm": _cm(y, pred),
    }


def _score_stats(scores, y):
    pos, neg = scores[y == 1], scores[y == 0]
    inv = int(np.sum(pos[:, None] <= neg[None, :])) if len(pos) and len(neg) else None
    pairs = int(len(pos) * len(neg)) if len(pos) and len(neg) else 0
    return {
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "pos_min": float(pos.min()) if len(pos) else None,
        "pos_max": float(pos.max()) if len(pos) else None,
        "pos_mean": float(pos.mean()) if len(pos) else None,
        "neg_min": float(neg.min()) if len(neg) else None,
        "neg_max": float(neg.max()) if len(neg) else None,
        "neg_mean": float(neg.mean()) if len(neg) else None,
        "n_unique_all": int(len(np.unique(np.round(scores, 8)))),
        "n_unique_pos": int(len(np.unique(np.round(pos, 8)))) if len(pos) else 0,
        "n_unique_neg": int(len(np.unique(np.round(neg, 8)))) if len(neg) else 0,
        "n_nan": int(np.isnan(scores).sum()),
        "n_inf": int(np.isinf(scores).sum()),
        "ranking_inversions_pos_le_neg": inv,
        "n_pos_neg_pairs": pairs,
        "frac_inversions": (inv / pairs) if pairs else None,
        "min_pos_gt_max_neg": bool(len(pos) and len(neg) and pos.min() > neg.max()),
    }


def main():
    os.makedirs(os.path.dirname(OUTJ), exist_ok=True)
    Xva, yva, gva = _load("val")
    Xte, yte, gte = _load("test")
    Xtr, ytr, gtr = _load("train")
    full_groups = {}
    for name, y, g in (("train", ytr, gtr), ("val", yva, gva), ("test", yte, gte)):
        gs = np.unique(g)
        att_g = [gg for gg in gs if (y[g == gg] > 0).any()]
        nor_g = [gg for gg in gs if (y[g == gg] == 0).all()]
        mixed = [gg for gg in gs if (y[g == gg] > 0).any() and (y[g == gg] == 0).any()]
        full_groups[name] = {
            "n_windows": int(len(y)),
            "n_groups": int(len(gs)),
            "n_attack_groups": int(len(att_g)),
            "n_pure_normal_groups": int(len(nor_g)),
            "n_mixed_groups": int(len(mixed)),
            "attack_groups": sorted(att_g),
            "attack_windows": int((y > 0).sum()),
            "normal_windows": int((y == 0).sum()),
        }

    _, ytr_e, gtr_e = _cap_with_g(Xtr, ytr, gtr, 14000, 42, "train")
    Xva_e, yva_e, gva_e = _cap_with_g(Xva, yva, gva, MAX_VAL, 43, "val")
    Xte_e, yte_e, gte_e = _cap_with_g(Xte, yte, gte, MAX_TEST, 44, "test")

    model = TCNGRUResilienceModel(num_features=10, num_classes=6)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    qmodel = quantize_tcn_gru_model(model)

    def scores_of(m, X):
        s = classifier_binary_scores(m, torch.tensor(X))
        return np.asarray(s, dtype=np.float64)

    s_va_q = scores_of(qmodel, Xva_e)
    s_te_q = scores_of(qmodel, Xte_e)
    s_te_fp = scores_of(model, Xte_e)
    tau, _ = select_operating_threshold(yva_e, s_va_q, split="val")
    pred = (s_te_q > tau).astype(int)

    # Apply the previously stored test-grid diagnostic without searching test labels.
    # Historical provenance only; never a replacement for the validation threshold.
    tau_oracle = 0.8722814321517944
    pred_oracle = (s_te_q > tau_oracle).astype(int)

    always_attack = np.ones_like(yte_e)
    always_normal = np.zeros_like(yte_e)

    # group-level: mean score / majority label per test group
    te_groups = {}
    for gg in np.unique(gte_e):
        msk = gte_e == gg
        te_groups[gg] = {
            "n": int(msk.sum()),
            "label": int(yte_e[msk][0]),
            "n_unique_labels": int(len(np.unique(yte_e[msk]))),
            "score_min": float(s_te_q[msk].min()),
            "score_max": float(s_te_q[msk].max()),
            "score_mean": float(s_te_q[msk].mean()),
        }

    # near-duplicate last-timestep features among test positives
    pos_X = Xte_e[yte_e == 1][:, -1, :]
    corr = None
    if len(pos_X) > 1:
        diffs = np.sqrt(((pos_X[:, None, :] - pos_X[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(diffs, np.nan)
        corr = {
            "n_pos_windows": int(len(pos_X)),
            "last_step_l2_min_offdiag": float(np.nanmin(diffs)),
            "last_step_l2_median_offdiag": float(np.nanmedian(diffs)),
            "n_pairs_l2_lt_1e-6": int(np.nansum(diffs < 1e-6)),
        }

    report = {
        "retrained": False,
        "checkpoint": CKPT,
        "checkpoint_sha256": _sha(CKPT),
        "scoring_rule": "1 - softmax(logits)[:,0]  (attack = 1 - P(normal))",
        "auc_inputs": "continuous scores, never binary predictions",
        "full_split_groups": full_groups,
        "eval_subsample": {
            "train_n": int(len(ytr_e)),
            "train_attack": int((ytr_e > 0).sum()),
            "n_train_groups": len(set(gtr_e)),
            "n_train_attack_groups": len(set(gtr_e[ytr_e == 1])),
            "n_train_normal_groups": len(set(gtr_e[ytr_e == 0])),
            "val_normal": int((yva_e == 0).sum()),
            "n_val_groups": len(set(gva_e)),
            "n_val_attack_groups": len(set(gva_e[yva_e == 1])),
            "n_val_normal_groups": len(set(gva_e[yva_e == 0])),
            "val_n": int(len(yva_e)),
            "test_n": int(len(yte_e)),
            "val_attack": int((yva_e > 0).sum()),
            "test_attack": int((yte_e > 0).sum()),
            "test_attack_groups": sorted(set(gte_e[yte_e == 1].tolist())),
            "n_test_attack_groups": int(len(set(gte_e[yte_e == 1].tolist()))),
            "n_test_normal_groups": int(len(set(gte_e[yte_e == 0].tolist()))),
        },
        "tau_val_frozen": float(tau),
        "tau_provenance": "select_operating_threshold on val subsample seed=43, split=val",
        "int8_test": {
            "roc_auc": float(roc_auc_score(yte_e, s_te_q)),
            "pr_auc": float(average_precision_score(yte_e, s_te_q)),
            "operating_point": _f1(yte_e, pred),
            "score_stats": _score_stats(s_te_q, yte_e),
        },
        "fp32_test": {
            "roc_auc": float(roc_auc_score(yte_e, s_te_fp)),
            "pr_auc": float(average_precision_score(yte_e, s_te_fp)),
            "score_stats": _score_stats(s_te_fp, yte_e),
        },
        "int8_val_subsample_score_stats": _score_stats(s_va_q, yva_e),
        "baselines": {
            "always_attack": _f1(yte_e, always_attack),
            "always_normal": _f1(yte_e, always_normal),
            "note": "always_normal is the true majority-class baseline on this TEST set; always_attack is the positive-class baseline. Dummy most_frequent fitted on TRAIN predicted always_attack.",
        },
        "diagnostic_test_grid_selected_tau": {
            "tau": float(tau_oracle),
            "metrics": _f1(yte_e, pred_oracle),
            "label": "TEST-GRID-SELECTED DIAGNOSTIC ONLY; uses test labels; restricted grid, not a global optimum or valid detector metric",
        },
        "test_groups": te_groups,
        "test_positive_laststep_similarity": corr,
        "label_alignment": "window y == last-flow label by construction of windows_from_ordered_flows",
        "interpretation_flags": {},
    }
    st = report["int8_test"]["score_stats"]
    report["interpretation_flags"] = {
        "perfect_separation_minpos_gt_maxneg": st["min_pos_gt_max_neg"],
        "few_attack_groups": report["eval_subsample"]["n_test_attack_groups"] <= 3,
        "val_prevalence_shift": True,
        "auc_from_binary_preds": False,
        "likely_threshold_transfer_failure": report["int8_test"]["operating_point"]["f1"] < 0.3
        and report["int8_test"]["pr_auc"] > 0.8,
    }
    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({
        "tau": report["tau_val_frozen"],
        "int8_f1": report["int8_test"]["operating_point"]["f1"],
        "int8_auc": report["int8_test"]["roc_auc"],
        "int8_ap": report["int8_test"]["pr_auc"],
        "minpos_gt_maxneg": st["min_pos_gt_max_neg"],
        "test_attack_groups": report["eval_subsample"]["n_test_attack_groups"],
        "always_normal_f1": report["baselines"]["always_normal"]["f1"],
        "always_attack_f1": report["baselines"]["always_attack"]["f1"],
        "ckpt": report["checkpoint_sha256"][:16],
    }, indent=2))
    return report


if __name__ == "__main__":
    main()
