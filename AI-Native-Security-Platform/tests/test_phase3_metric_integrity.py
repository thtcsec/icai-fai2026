"""Metric integrity: AUC from continuous scores; majority vs always-attack."""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
J = os.path.join(REPO, "revisions", "icai2026_20261028", "metric_integrity.json")


def test_integrity_json_exists_and_no_retrain():
    assert os.path.isfile(J)
    d = json.load(open(J, encoding="utf-8"))
    assert d["retrained"] is False
    assert d["auc_inputs"].startswith("continuous")


def test_auc_not_from_binary_predictions():
    d = json.load(open(J, encoding="utf-8"))
    st = d["int8_test"]["score_stats"]
    assert st["n_unique_all"] > 2
    assert d["interpretation_flags"]["auc_from_binary_preds"] is False
    assert st["min_pos_gt_max_neg"] is True
    assert st["ranking_inversions_pos_le_neg"] == 0


def test_always_normal_is_true_majority_baseline():
    d = json.load(open(J, encoding="utf-8"))
    y = np.array([0] * 923 + [1] * 32)
    always_n = np.zeros_like(y)
    always_a = np.ones_like(y)
    assert f1_score(y, always_n, zero_division=0) == 0.0
    assert d["baselines"]["always_normal"]["f1"] == 0.0
    assert abs(d["baselines"]["always_attack"]["f1"] - f1_score(y, always_a)) < 1e-12
    assert d["eval_subsample"]["n_test_attack_groups"] == 1


def test_val_frozen_tau_distinct_from_test_grid_diagnostic():
    d = json.load(open(J, encoding="utf-8"))
    assert d["tau_val_frozen"] < 0.01
    assert "DIAGNOSTIC" in d["diagnostic_test_grid_selected_tau"]["label"]
    assert d["diagnostic_test_grid_selected_tau"]["tau"] > 0.5
