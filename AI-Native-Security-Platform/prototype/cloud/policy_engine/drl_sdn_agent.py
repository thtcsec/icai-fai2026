"""
drl_sdn_agent.py - Deep Reinforcement Learning Agent for Closed-Loop SDN Orchestration

Ported from sdn-its-resilience-ai (models/drl_agent.py).
State Space (5 dims): [Switch_CPU_Load, PacketIn_Rate, Link_Bandwidth_Util, Reconstruction_Error, Anomaly_Confidence]
Action Space (4 actions):
  0: NORMAL_FORWARDING
  1: DYNAMIC_RATE_LIMIT (Throttle flow bandwidth to 50%)
  2: V2X_PATH_REROUTE (Redirect flow to backup AP s2)
  3: TARGETED_FLOW_ISOLATION (Apply temporary drop flow_mod)
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Tuple, List, Dict

class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int = 5, action_dim: int = 4):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DRLResilienceAgent:
    ACTION_NAMES = [
        "NORMAL_FORWARDING",
        "DYNAMIC_RATE_LIMIT",
        "V2X_PATH_REROUTE",
        "TARGETED_FLOW_ISOLATION"
    ]

    def __init__(self, state_dim: int = 5, action_dim: int = 4, lr: float = 0.001, gamma: float = 0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 0.05
        
        self.policy_net = DQNNetwork(state_dim, action_dim)
        self.target_net = DQNNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state: List[float], eval_mode: bool = True) -> int:
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def get_action_name(self, action_id: int) -> str:
        return self.ACTION_NAMES[action_id]
