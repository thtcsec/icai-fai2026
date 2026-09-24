"""Shared training / evaluation helpers for reproducible paper tables."""

from __future__ import annotations

import copy
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def set_global_seed(seed: int = 42) -> None:
    """Seed every RNG that can affect a reported number.

    Model constructors draw from the global torch RNG, so seeding inside the
    training helpers is too late: weights are already initialized by then.
    Call this before building any module, otherwise results vary run to run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset(
    X: torch.Tensor,
    y: torch.Tensor,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random i.i.d. split.

    UNSAFE for the sliding-window corpora used in this project: adjacent windows
    overlap by seq_len-1 timesteps, so a random split places near-duplicate
    windows on both sides. Retained only for non-window data. Use
    `blocked_split_npz` for the InSDN/CIC window archives.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = int(len(X) * train_ratio)
    tr, te = idx[:n_train], idx[n_train:]
    return X[tr], y[tr], X[te], y[te]


def blocked_split_npz(
    npz_path: str,
    seq_len: int = 10,
    train_ratio: float = 0.7,
    n_blocks: int = 20,
    max_train: int = 14000,
    max_test: int = 6000,
    seed: int = 42,
    scaler=None,
    fit_scaler: bool = True,
):
    """Blocked+purged split controlling stride-1 *index* overlap leakage.

    Frozen InSDN/CIC windows use stride 1, so window i shares seq_len-1 feature
    timesteps with each neighbour. A random split therefore places near-
    duplicate windows on both sides of the train/test boundary. This function:

      1. cuts the *index-ordered* corpus into contiguous blocks and assigns
         whole blocks to train or test;
      2. purges seq_len-1 windows from both ends of every block so no surviving
         train window shares a timestep index with any surviving test window;
      3. stratified-subsamples inside each side independently; and
      4. fits the scaler on train only, transforming test with those statistics.

    Note: the frozen InSDN archive was shuffled before windowing, so this is
    overlap control on artificial index order, not wall-clock session isolation.
    """
    from sklearn.preprocessing import StandardScaler

    data = np.load(npz_path)
    X = data["X"].astype(np.float32)
    y = (data["y"] > 0).astype(np.int64)
    n = len(X)

    bounds = np.linspace(0, n, n_blocks + 1).astype(int)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_blocks)
    n_train_blocks = int(round(n_blocks * train_ratio))
    train_blocks = set(order[:n_train_blocks].tolist())

    purge = seq_len - 1
    tr_parts, te_parts = [], []
    for bi in range(n_blocks):
        start, end = bounds[bi] + purge, bounds[bi + 1] - purge
        if end <= start:
            continue
        idx = np.arange(start, end)
        (tr_parts if bi in train_blocks else te_parts).append(idx)

    tr_idx = np.concatenate(tr_parts)
    te_idx = np.concatenate(te_parts)

    def _subsample(idx, budget):
        yy = y[idx]
        i0, i1 = idx[yy == 0], idx[yy == 1]
        frac0 = len(i0) / max(len(idx), 1)
        n0 = min(len(i0), max(1, int(budget * frac0)))
        n1 = min(len(i1), budget - n0)
        pick = np.concatenate(
            [
                rng.choice(i0, size=n0, replace=False),
                rng.choice(i1, size=n1, replace=False) if n1 > 0 else np.array([], dtype=int),
            ]
        )
        rng.shuffle(pick)
        return np.sort(pick)

    tr_idx = _subsample(tr_idx, max_train)
    te_idx = _subsample(te_idx, max_test)

    n_feat = X.shape[2]
    tr_flat = np.nan_to_num(X[tr_idx].reshape(-1, n_feat), nan=0.0, posinf=0.0, neginf=0.0)
    te_flat = np.nan_to_num(X[te_idx].reshape(-1, n_feat), nan=0.0, posinf=0.0, neginf=0.0)

    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        scaler.fit(tr_flat)
    tr_flat = scaler.transform(tr_flat).astype(np.float32)
    te_flat = scaler.transform(te_flat).astype(np.float32)

    X_tr = torch.tensor(tr_flat.reshape(len(tr_idx), seq_len, n_feat), dtype=torch.float32)
    X_te = torch.tensor(te_flat.reshape(len(te_idx), seq_len, n_feat), dtype=torch.float32)
    y_tr = torch.tensor(y[tr_idx], dtype=torch.long)
    y_te = torch.tensor(y[te_idx], dtype=torch.long)

    print(
        f"[+] Blocked split {os.path.basename(npz_path)}: "
        f"train={len(tr_idx)} (attack {int(y[tr_idx].sum())}), "
        f"test={len(te_idx)} (attack {int(y[te_idx].sum())}), "
        f"{n_blocks} blocks, purge={purge}"
    )
    return X_tr, y_tr, X_te, y_te, scaler


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
