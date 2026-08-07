"""Shared training / evaluation helpers for reproducible paper tables."""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def split_dataset(
    X: torch.Tensor,
    y: torch.Tensor,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = int(len(X) * train_ratio)
    tr, te = idx[:n_train], idx[n_train:]
    return X[tr], y[tr], X[te], y[te]


def train_autoencoder(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
    cls_weight: float = 0.5,
) -> nn.Module:
    """Multi-task train: AE reconstruction on normals + supervised binary head."""
    torch.manual_seed(seed)
    model = copy.deepcopy(model)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    normal_mask = y_train == 0
    X_normal = X_train[normal_mask]
    if len(X_normal) == 0:
        X_normal = X_train

    y_cls = y_train.clone()
    y_cls[y_cls > 0] = 1
    n_pos = max(1, int((y_cls == 1).sum()))
    n_neg = max(1, int((y_cls == 0).sum()))
    # Up-weight normal when attacks dominate (real InSDN is attack-heavy)
    w = torch.ones(getattr(model, "num_classes", 6))
    w[0] = float(n_pos) / float(n_neg)
    w[1] = 1.0
    ce = nn.CrossEntropyLoss(weight=w)

    for _ in range(epochs):
        perm = torch.randperm(len(X_normal))
        for start in range(0, len(X_normal), batch_size):
            batch = X_normal[perm[start : start + batch_size]]
            opt.zero_grad()
            x_hat, _, _ = model(batch)
            loss = mse(x_hat, batch)
            loss.backward()
            opt.step()

        perm = torch.randperm(len(X_train))
        for start in range(0, len(X_train), batch_size):
            idx = perm[start : start + batch_size]
            batch = X_train[idx]
            labels = y_cls[idx]
            opt.zero_grad()
            x_hat, logits, _ = model(batch)
            loss = mse(x_hat, batch) + cls_weight * ce(logits, labels)
            loss.backward()
            opt.step()

    model.eval()
    return model


def train_reconstruction_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    forward_mode: str = "tcn_gru",
) -> nn.Module:
    torch.manual_seed(seed)
    model = copy.deepcopy(model)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    normal_mask = y_train == 0
    X_normal = X_train[normal_mask]
    if len(X_normal) == 0:
        X_normal = X_train

    for _ in range(epochs):
        perm = torch.randperm(len(X_normal))
        for start in range(0, len(X_normal), batch_size):
            batch = X_normal[perm[start : start + batch_size]]
            opt.zero_grad()
            if forward_mode == "tcn_gru":
                x_hat, _, _ = model(batch)
            else:
                x_hat, _ = model(batch)
            loss = loss_fn(x_hat, batch)
            loss.backward()
            opt.step()
    model.eval()
    return model


def reconstruction_errors(model: nn.Module, X: torch.Tensor, forward_mode: str = "tcn_gru") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        if forward_mode == "tcn_gru":
            _, _, err = model(X)
        else:
            _, err = model(X)
    return err.detach().cpu().numpy()


def classifier_binary_scores(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """Attack probability from classifier head (1 - P(normal))."""
    model.eval()
    with torch.no_grad():
        _, logits, _ = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return 1.0 - probs[:, 0]


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, tau: float) -> Dict[str, float]:
    y_pred = (scores > tau).astype(int)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
    fnr = (fn / (fn + tp) * 100.0) if (fn + tp) > 0 else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": float(fpr),
        "fnr": float(fnr),
    }


def adaptive_threshold_table(
    y_true: np.ndarray,
    test_scores: np.ndarray,
    normal_train_scores: np.ndarray,
    ks: List[float] | None = None,
) -> Tuple[List[Tuple[float, float, Dict[str, float]]], Tuple[float, float, Dict[str, float]]]:
    """Evaluate tau = mu + k*sigma using normal-train score statistics."""
    if ks is None:
        ks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    mu = float(np.mean(normal_train_scores))
    sigma = float(np.std(normal_train_scores) + 1e-12)
    rows = []
    for k in ks:
        tau = mu + k * sigma
        m = threshold_metrics(y_true, test_scores, tau)
        rows.append((float(k), float(tau), m))
    best = max(rows, key=lambda r: r[2]["f1"])
    return rows, best


def best_threshold(y_true: np.ndarray, scores: np.ndarray, candidates: List[float] | None = None) -> Tuple[float, Dict[str, float]]:
    if candidates is None:
        qs = np.unique(np.quantile(scores, np.linspace(0.05, 0.99, 60)))
        candidates = [float(x) for x in qs]
    best_tau, best = float(np.median(scores)), {"f1": -1.0}
    for tau in candidates:
        m = threshold_metrics(y_true, scores, float(tau))
        if m["f1"] > best["f1"]:
            best_tau, best = float(tau), m
    return best_tau, best


def model_param_mb(model: nn.Module) -> float:
    total = 0
    state = model.state_dict()
    for p in state.values():
        if torch.is_tensor(p):
            itemsize = 1 if p.dtype in (torch.qint8, torch.quint8, torch.int8, torch.uint8) else p.element_size()
            total += p.numel() * itemsize
    return total / (1024.0 * 1024.0)
