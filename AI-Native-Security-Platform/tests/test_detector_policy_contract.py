"""P0-B: detector outputs must feed policy; hardcoded p_attack is forbidden."""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pytest

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

torch = pytest.importorskip("torch")

from prototype.cloud.policy_engine.observation import (
    STATE_DIM,
    STATE_KEYS,
    observation_from_detector_window,
)
from prototype.edge.detector.inference import infer_window, load_detector_checkpoint


CKPT = os.path.join(BASE, "results", "checkpoints", "tcn_gru_ae_trained.pt")
TABLE2 = pathlib.Path(BASE) / "results" / "generate_table2.py"


def test_table2_source_does_not_inject_hardcoded_p_attack():
    src = TABLE2.read_text(encoding="utf-8")
    assert "_cloud_event(ctx, 5000, 0.91)" not in src
    assert "state = [0.91," not in src
    assert "infer_window(" in src
    assert "from_checkpoint" in src


def test_missing_detector_checkpoint_fails():
    with pytest.raises(FileNotFoundError, match="required but missing"):
        load_detector_checkpoint(os.path.join(BASE, "results", "checkpoints", "no_such.pt"))


def test_observation_contract_uses_detector_p_attack():
    obs = observation_from_detector_window(
        p_attack=0.37,
        reconstruction_error=0.2,
        window_last_flow=[0.0] * 10,
    )
    assert len(obs.vector) == STATE_DIM
    assert obs.vector[4] == pytest.approx(0.37)
    assert obs.provenance["anomaly_conf"] == "detector_p_attack_1_minus_p_normal"
    assert obs.provenance["switch_cpu"].startswith("placeholder")
    assert list(STATE_KEYS) == [
        "switch_cpu",
        "packetin_rate",
        "bw_util",
        "recon_error",
        "anomaly_conf",
    ]


@pytest.mark.skipif(not os.path.isfile(CKPT), reason="trained detector checkpoint missing")
def test_real_checkpoint_inference_not_constant():
    model, meta = load_detector_checkpoint(CKPT, quantize=True)
    assert meta["scoring_rule"] == "1 - P(normal)"
    npz = os.path.join(BASE, "datasets", "processed", "insdn_windows.npz")
    if not os.path.isfile(npz):
        pytest.skip("insdn archive missing")
    data = np.load(npz)
    xs = torch.tensor(data["X"][:8], dtype=torch.float32)
    xs = torch.nan_to_num(xs)
    ps = []
    for i in range(len(xs)):
        out = infer_window(model, xs[i : i + 1], checkpoint_path=CKPT, quantized=True)
        assert 0.0 <= out.p_attack <= 1.0
        assert out.scoring_rule == "1 - P(normal)"
        ps.append(out.p_attack)
    # A hardcoded override would be exactly constant; real logits should vary.
    assert len(set(np.round(ps, 5))) > 1
