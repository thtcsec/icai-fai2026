"""
dqn-controller / dqn_agent.py - Deep Reinforcement Learning Agent for Closed-Loop SDN Resilience Control
"""

import random
import torch
import torch.nn as nn
from typing import List

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
    ACTIONS = [
        "NORMAL_FORWARDING",
        "DYNAMIC_RATE_LIMIT",
        "V2X_PATH_REROUTE",
        "TARGETED_FLOW_ISOLATION"
    ]

    def __init__(self, state_dim: int = 5, action_dim: int = 4):
        self.policy_net = DQNNetwork(state_dim, action_dim)
        self.policy_net.eval()

    def select_action(self, state: List[float]) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.policy_net(state_t)
            return int(torch.argmax(q_vals, dim=1).item())

    def get_action_description(self, action_id: int) -> str:
        return self.ACTIONS[action_id]
