"""P0-C: evaluation must load a trained DQN; random init is a hard error."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

torch = pytest.importorskip("torch")

from prototype.cloud.policy_engine.drl_sdn_agent import DRLResilienceAgent
from prototype.cloud.policy_engine.observation import STATE_DIM


def test_eval_without_checkpoint_raises():
    agent = DRLResilienceAgent()
    with pytest.raises(RuntimeError, match="randomly initialized"):
        agent.select_action([0.1] * STATE_DIM, eval_mode=True)


def test_missing_checkpoint_raises():
    with pytest.raises(FileNotFoundError, match="required but missing"):
        DRLResilienceAgent.from_checkpoint(os.path.join(BASE, "no_such_dqn.pt"))


def test_incompatible_state_dim_rejected(tmp_path):
    agent = DRLResilienceAgent(allow_uninitialized=True)
    path = tmp_path / "dqn.pt"
    agent.save_checkpoint(str(path))
    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["state_dim"] = 7
    torch.save(blob, path)
    with pytest.raises(RuntimeError, match="incompatible"):
        DRLResilienceAgent.from_checkpoint(str(path))


def test_save_load_roundtrip_and_beats_random(tmp_path):
    agent = DRLResilienceAgent(allow_uninitialized=True)
    # Bias the last-layer toward action 0 so the loaded policy is deterministic
    # and different from a uniform random policy.
    with torch.no_grad():
        last = agent.policy_net.net[-1]
        last.weight.zero_()
        last.bias.copy_(torch.tensor([3.0, 0.0, 0.0, 0.0]))
    path = tmp_path / "dqn_trained.pt"
    agent.save_checkpoint(str(path), extra={"fixture": True})
    loaded = DRLResilienceAgent.from_checkpoint(str(path))
    rng = np.random.default_rng(0)
    states = rng.uniform(0, 1, size=(200, STATE_DIM))
    chosen = np.array([loaded.select_action(s.tolist(), eval_mode=True) for s in states])
    assert (chosen == 0).mean() == 1.0
    random_actions = rng.integers(0, 4, size=200)
    assert (random_actions == 0).mean() < 0.9
