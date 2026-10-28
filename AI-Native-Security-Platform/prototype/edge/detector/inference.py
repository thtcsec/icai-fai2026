"""Validated detector inference contract used by policy / latency paths.

Attack probability is ``1 - P(normal)`` from the classifier head — the same
rule as ``train_utils.classifier_binary_scores``. Callers must not replace
this value with a synthetic constant in production evaluation paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from prototype.edge.detector.quantize import quantize_tcn_gru_model
from prototype.edge.detector.tcn_gru_model import TCNGRUResilienceModel

EXPECTED_SEQ_LEN = 10
EXPECTED_NUM_FEATURES = 10
EXPECTED_NUM_CLASSES = 6


@dataclass(frozen=True)
class DetectorOutput:
    p_attack: float
    reconstruction_error: float
    logits: np.ndarray
    seq_len: int
    num_features: int
    checkpoint_path: str
    quantized: bool
    scoring_rule: str = "1 - P(normal)"

    def as_policy_fields(self) -> Dict[str, float]:
        return {
            "p_attack": float(self.p_attack),
            "reconstruction_error": float(self.reconstruction_error),
        }


def load_detector_checkpoint(
    checkpoint_path: str,
    *,
    quantize: bool = True,
    map_location: str = "cpu",
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Trained detector checkpoint required but missing: {checkpoint_path}"
        )
    model = TCNGRUResilienceModel(
        num_features=EXPECTED_NUM_FEATURES,
        num_classes=EXPECTED_NUM_CLASSES,
    )
    state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state)
    except Exception as exc:
        raise RuntimeError(
            f"Detector checkpoint incompatible with TCNGRUResilienceModel "
            f"(num_features={EXPECTED_NUM_FEATURES}, num_classes={EXPECTED_NUM_CLASSES}): {exc}"
        ) from exc
    model.eval()
    if quantize:
        model = quantize_tcn_gru_model(model)
        model.eval()
    meta = {
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "quantized": bool(quantize),
        "seq_len": EXPECTED_SEQ_LEN,
        "num_features": EXPECTED_NUM_FEATURES,
        "num_classes": EXPECTED_NUM_CLASSES,
        "scoring_rule": "1 - P(normal)",
    }
    return model, meta


def _validate_window(window: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(window):
        window = torch.as_tensor(window, dtype=torch.float32)
    if window.ndim == 2:
        window = window.unsqueeze(0)
    if window.ndim != 3:
        raise ValueError(f"Expected window (B, T, F), got {tuple(window.shape)}")
    if window.shape[1] != EXPECTED_SEQ_LEN or window.shape[2] != EXPECTED_NUM_FEATURES:
        raise ValueError(
            f"Window shape {tuple(window.shape)} does not match training "
            f"(B, {EXPECTED_SEQ_LEN}, {EXPECTED_NUM_FEATURES})"
        )
    if torch.isnan(window).any() or torch.isinf(window).any():
        window = torch.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
    return window.contiguous()


def infer_window(
    model: torch.nn.Module,
    window: torch.Tensor,
    *,
    checkpoint_path: str,
    quantized: bool,
) -> DetectorOutput:
    """Run one (or batched) window and return the first sample's contract."""
    model.eval()
    window = _validate_window(window)
    with torch.no_grad():
        _x_hat, logits, rec_err = model(window)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        rec = rec_err.detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()
    p_attack = float(1.0 - probs[0, 0])
    if not np.isfinite(p_attack):
        raise RuntimeError("Detector produced a non-finite p_attack")
    return DetectorOutput(
        p_attack=p_attack,
        reconstruction_error=float(rec[0]),
        logits=logits_np[0],
        seq_len=int(window.shape[1]),
        num_features=int(window.shape[2]),
        checkpoint_path=checkpoint_path,
        quantized=quantized,
    )


def infer_p_attack_batch(model: torch.nn.Module, windows: torch.Tensor) -> np.ndarray:
    """Vectorized attack probabilities; same scoring rule as training."""
    model.eval()
    windows = _validate_window(windows)
    with torch.no_grad():
        _x_hat, logits, _rec = model(windows)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return (1.0 - probs[:, 0]).astype(np.float64)
