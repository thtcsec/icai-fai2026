"""
drl_sdn_agent.py - Deep Reinforcement Learning Agent for Closed-Loop SDN Orchestration

Ported from sdn-its-resilience-ai (models/drl_agent.py).
State Space (5 dims): [Switch_CPU_Load, PacketIn_Rate, Link_Bandwidth_Util, Reconstruction_Error, Anomaly_Confidence]
Action Space (4 actions):
  0: NORMAL_FORWARDING
  1: DYNAMIC_RATE_LIMIT (Throttle flow bandwidth to 50%)
  2: V2X_PATH_REROUTE (Redirect flow to backup AP s2)
  3: TARGETED_FLOW_ISOLATION (Apply temporary drop flow_mod)

Evaluation and latency paths MUST load a trained checkpoint via
``DRLResilienceAgent.from_checkpoint``. Constructing this class without
loading weights yields a randomly initialized network and is forbidden for
reported policy metrics.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from prototype.cloud.policy_engine.observation import STATE_DIM, STATE_KEYS

CHECKPOINT_FORMAT_VERSION = 1
REWARD_ENVIRONMENT = "simulated_analytic_campus_contention"


class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int = 5, action_dim: int = 4):
        super(DQNNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DRLResilienceAgent:
    ACTION_NAMES = [
        "NORMAL_FORWARDING",
        "DYNAMIC_RATE_LIMIT",
        "V2X_PATH_REROUTE",
        "TARGETED_FLOW_ISOLATION",
    ]

    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 4,
        lr: float = 0.001,
        gamma: float = 0.95,
        *,
        allow_uninitialized: bool = False,
    ):
        if state_dim != STATE_DIM:
            raise ValueError(f"state_dim must be {STATE_DIM} ({STATE_KEYS}), got {state_dim}")
        if action_dim != len(self.ACTION_NAMES):
            raise ValueError(f"action_dim must be {len(self.ACTION_NAMES)}, got {action_dim}")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 0.05
        self.checkpoint_path: Optional[str] = None
        self.loaded = False
        self.allow_uninitialized = allow_uninitialized

        self.policy_net = DQNNetwork(state_dim, action_dim)
        self.target_net = DQNNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def _require_loaded_for_eval(self, eval_mode: bool) -> None:
        if eval_mode and not self.loaded and not self.allow_uninitialized:
            raise RuntimeError(
                "Refusing to evaluate a randomly initialized DQN. "
                "Load trained weights with DRLResilienceAgent.from_checkpoint "
                "or save_checkpoint after training. "
                "Pass allow_uninitialized=True only for isolated unit tests."
            )

    def select_action(self, state: List[float], eval_mode: bool = True) -> int:
        self._require_loaded_for_eval(eval_mode)
        if len(state) != self.state_dim:
            raise ValueError(
                f"Observation has dim {len(state)}, expected {self.state_dim} ({STATE_KEYS})"
            )
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        self.policy_net.eval()
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def get_action_name(self, action_id: int) -> str:
        return self.ACTION_NAMES[action_id]

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload: Dict[str, Any] = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "state_dict": self.policy_net.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "state_keys": list(STATE_KEYS),
            "action_names": list(self.ACTION_NAMES),
            "reward_environment": REWARD_ENVIRONMENT,
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)
        self.checkpoint_path = os.path.abspath(path)
        self.loaded = True
        return self.checkpoint_path

    def load_checkpoint(self, path: str, map_location: str = "cpu") -> Dict[str, Any]:
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"Trained DQN checkpoint required but missing: {path}")
        blob = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(blob, dict) or "state_dict" not in blob:
            raise RuntimeError(
                f"DQN checkpoint {path} is not a dict with state_dict "
                "(incompatible or raw state_dict-only file)"
            )
        state_dim = int(blob.get("state_dim", -1))
        action_dim = int(blob.get("action_dim", -1))
        if state_dim != self.state_dim or action_dim != self.action_dim:
            raise RuntimeError(
                f"DQN checkpoint incompatible: ckpt (state={state_dim}, action={action_dim}) "
                f"vs agent (state={self.state_dim}, action={self.action_dim})"
            )
        ckpt_keys = tuple(blob.get("state_keys", STATE_KEYS))
        if ckpt_keys != STATE_KEYS:
            raise RuntimeError(
                f"DQN observation layout mismatch: ckpt {ckpt_keys} vs {STATE_KEYS}"
            )
        try:
            self.policy_net.load_state_dict(blob["state_dict"])
        except Exception as exc:
            raise RuntimeError(f"DQN state_dict load failed: {exc}") from exc
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        self.checkpoint_path = os.path.abspath(path)
        self.loaded = True
        return blob

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> "DRLResilienceAgent":
        agent = cls(allow_uninitialized=False, **kwargs)
        agent.load_checkpoint(path)
        return agent
