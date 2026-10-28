"""Analytic campus-contention simulator shared by DQN train and eval.

This is a *simulated reward* environment. Oracle action-match is not operational
mitigation success.
"""

from __future__ import annotations

import hashlib

import numpy as np

N_ACTIONS = 4
STATE_DIM = 5
STATE_KEYS = (
    "switch_cpu",
    "packetin_rate",
    "bw_util",
    "recon_error",
    "anomaly_conf",
)
CONTAINMENT = np.array([0.0, 0.50, 0.60, 1.0])
QOS_COST = np.array([0.0, 0.25, 0.35, 0.90])
RELIEF = np.array([0.0, 0.15, 0.30, 0.20])
REWARD_ENVIRONMENT = "simulated_analytic_campus_contention"


def sample_states(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=(n, STATE_DIM))


def reward_matrix(states: np.ndarray) -> np.ndarray:
    threat = states[:, 4:5]
    bw_util = states[:, 2:3]
    switch_cpu = states[:, 0:1]
    benefit = threat * CONTAINMENT[None, :]
    collateral = (1.0 - threat) * QOS_COST[None, :]
    relief = bw_util * RELIEF[None, :]
    overload = np.zeros_like(benefit)
    overload[:, 3:4] = 0.25 * switch_cpu
    return benefit - collateral + relief - overload


def oracle_actions(states: np.ndarray) -> np.ndarray:
    return np.argmax(reward_matrix(states), axis=1)


def states_fingerprint(states: np.ndarray) -> set[bytes]:
    """Exact-row hashes so train/eval overlap is detectable."""
    out = set()
    for row in np.asarray(states, dtype=np.float64):
        out.add(hashlib.sha256(row.tobytes()).digest())
    return out


def policy_static(state: np.ndarray) -> int:
    conf = state[4]
    if conf > 0.80:
        return 3
    if conf > 0.50:
        return 1
    return 0


def policy_heuristic(state: np.ndarray) -> int:
    conf, bw = state[4], state[2]
    if conf > 0.75:
        return 2 if bw > 0.60 else 3
    if conf > 0.45:
        return 1
    return 0
