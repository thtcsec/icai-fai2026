"""DQN train/eval independence: eval never trains; seeds/records do not overlap."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(BASE, "results")
for p in (BASE, RESULTS):
    if p not in sys.path:
        sys.path.insert(0, p)

torch = pytest.importorskip("torch")

from eval_dqn_checkpoint import evaluate
from policy_sim import sample_states, states_fingerprint
from prototype.cloud.policy_engine.drl_sdn_agent import DRLResilienceAgent


def test_eval_source_does_not_train_or_save():
    src = open(os.path.join(RESULTS, "eval_dqn_checkpoint.py"), encoding="utf-8").read()
    assert "train_dqn(" not in src
    assert "save_checkpoint" not in src
    assert "THIS ENTRY POINT DOES NOT TRAIN" in src


def test_eval_missing_checkpoint_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="required but missing"):
        evaluate(ckpt=str(tmp_path / "missing.pt"), eval_seed=1, n_eval=8)


def test_train_eval_hashes_disjoint():
    from policy_sim import sample_states, states_fingerprint

    train = sample_states(4000, np.random.default_rng(42))
    evals = sample_states(5000, np.random.default_rng(2026))
    assert not (states_fingerprint(train) & states_fingerprint(evals))


def test_eval_loads_checkpoint_only(tmp_path):
    agent = DRLResilienceAgent(allow_uninitialized=True)
    with torch.no_grad():
        last = agent.policy_net.net[-1]
        last.weight.zero_()
        last.bias.copy_(torch.tensor([4.0, 0.0, 0.0, 0.0]))
    ckpt = tmp_path / "dqn.pt"
    agent.save_checkpoint(str(ckpt), extra={"fixture": True})
    hash_path = tmp_path / "train_hashes.bin"
    hash_path.write_bytes(b"")
    loaded = DRLResilienceAgent.from_checkpoint(str(ckpt))
    states = sample_states(32, np.random.default_rng(2026))
    chosen = np.array([loaded.select_action(s.tolist(), eval_mode=True) for s in states])
    assert (chosen == 0).all()
    out = evaluate(
        ckpt=str(ckpt),
        eval_seed=2026,
        n_eval=32,
        train_hash_path=str(hash_path),
        write_outputs=False,
    )
    assert out["trained_this_process"] is False
    assert out["saved_checkpoint"] is False
    assert out["operational_enforcement"] is False
    dqn = [r for r in out["rows"] if "Deep Q-Network" in r["policy"]][0]
    assert dqn["qos"] == pytest.approx(0.0)


def test_baselines_share_eval_states():
    """All policies in evaluate() consume the same sampled matrix."""
    src = open(os.path.join(RESULTS, "eval_dqn_checkpoint.py"), encoding="utf-8").read()
    assert "chosen = np.array([fn(s) for s in states])" in src
    assert src.count("sample_states(n_eval") == 1
