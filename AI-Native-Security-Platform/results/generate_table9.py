"""
generate_table9.py - Policy controller benchmark (DQN vs static / round-robin / heuristic)

The campus contention environment is a single-step decision problem: given a flow
state, choose one of four SDN containment actions. Because the reward function is
analytic, the optimal action a* can be obtained by brute-force enumeration, which
gives an exact oracle to score every policy against.
"""

import csv
import os
import random
import sys
import time
from collections import deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from prototype.cloud.policy_engine.drl_sdn_agent import DQNNetwork

CSV_PATH = os.path.join(BASE_DIR, "results", "table9_policy_benchmark.csv")
PNG_PATH = os.path.join(BASE_DIR, "results", "table9_policy_benchmark.png")
FIG_PATH = os.path.join(BASE_DIR, "paper", "figures", "fig8_policy_benchmark.png")

N_ACTIONS = 4
STATE_DIM = 5

# Fraction of the threat neutralised by each action.
CONTAINMENT = np.array([0.0, 0.50, 0.60, 1.0])
# Collateral damage inflicted on legitimate traffic by each action.
QOS_COST = np.array([0.0, 0.25, 0.35, 0.90])
# Congestion relief obtained per unit of link utilisation.
RELIEF = np.array([0.0, 0.15, 0.30, 0.20])


def sample_states(n: int, rng: np.random.Generator) -> np.ndarray:
    """[switch_cpu, packetin_rate, bw_util, recon_error, anomaly_conf]."""
    return rng.uniform(0.0, 1.0, size=(n, STATE_DIM))


def reward_matrix(states: np.ndarray) -> np.ndarray:
    """Exact reward of every action for every state -> shape (n, N_ACTIONS)."""
    threat = states[:, 4:5]
    bw_util = states[:, 2:3]
    switch_cpu = states[:, 0:1]

    benefit = threat * CONTAINMENT[None, :]
    collateral = (1.0 - threat) * QOS_COST[None, :]
    relief = bw_util * RELIEF[None, :]
    # Isolation is disproportionately expensive on an already saturated switch.
    overload = np.zeros_like(benefit)
    overload[:, 3:4] = 0.25 * switch_cpu
    return benefit - collateral + relief - overload


def oracle_actions(states: np.ndarray) -> np.ndarray:
    return np.argmax(reward_matrix(states), axis=1)


def policy_static(state: np.ndarray) -> int:
    """Threshold ladder on anomaly confidence alone."""
    conf = state[4]
    if conf > 0.80:
        return 3
    if conf > 0.50:
        return 1
    return 0


_rr_counter = {"i": 0}


def policy_round_robin(state: np.ndarray) -> int:
    a = _rr_counter["i"] % N_ACTIONS
    _rr_counter["i"] += 1
    return a


def policy_heuristic(state: np.ndarray) -> int:
    """Hand-tuned table that additionally consults link utilisation."""
    conf, bw = state[4], state[2]
    if conf > 0.75:
        return 2 if bw > 0.60 else 3
    if conf > 0.45:
        return 1
    return 0


def train_dqn(rng: np.random.Generator, seed: int = 42, steps: int = 4000, batch_size: int = 64):
    """Fitted Q-iteration on the single-step contention environment."""
    torch.manual_seed(seed)
    random.seed(seed)
    net = DQNNetwork(STATE_DIM, N_ACTIONS)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    replay = deque(maxlen=20000)
    epsilon = 1.0
    for step in range(steps):
        s = sample_states(1, rng)[0]
        if random.random() < epsilon:
            a = random.randrange(N_ACTIONS)
        else:
            with torch.no_grad():
                a = int(torch.argmax(net(torch.FloatTensor(s).unsqueeze(0)), dim=1).item())
        r = float(reward_matrix(s[None, :])[0, a])
        replay.append((s, a, r))
        epsilon = max(0.05, epsilon * 0.999)

        if len(replay) >= batch_size:
            batch = random.sample(replay, batch_size)
            bs = torch.FloatTensor(np.array([b[0] for b in batch]))
            ba = torch.LongTensor([b[1] for b in batch]).unsqueeze(1)
            br = torch.FloatTensor([b[2] for b in batch]).unsqueeze(1)
            q = net(bs).gather(1, ba)
            # Horizon is one step, so the Bellman target reduces to the reward.
            opt.zero_grad()
            loss_fn(q, br).backward()
            opt.step()

    net.eval()
    return net


def run_policy_benchmark(seed: int = 42, n_eval: int = 5000):
    print("[*] Policy controller benchmark (DQN vs baselines)...")
    rng = np.random.default_rng(seed)

    net = train_dqn(rng, seed=seed)
    eval_states = sample_states(n_eval, rng)
    a_star = oracle_actions(eval_states)
    rewards = reward_matrix(eval_states)

    def policy_dqn(state: np.ndarray) -> int:
        with torch.no_grad():
            return int(torch.argmax(net(torch.FloatTensor(state).unsqueeze(0)), dim=1).item())

    policies = [
        ("Static Rule Lookup", policy_static),
        ("Round-Robin Action Assignment", policy_round_robin),
        ("Heuristic Risk Table", policy_heuristic),
        ("Deep Q-Network (DQN Agent)", policy_dqn),
    ]

    rows = []
    for name, fn in policies:
        _rr_counter["i"] = 0
        for i in range(200):
            fn(eval_states[i % n_eval])

        _rr_counter["i"] = 0
        t0 = time.perf_counter()
        chosen = np.array([fn(s) for s in eval_states])
        latency_ms = ((time.perf_counter() - t0) / n_eval) * 1000.0

        accuracy = float((chosen == a_star).mean() * 100.0)
        achieved = rewards[np.arange(n_eval), chosen]
        optimal = rewards[np.arange(n_eval), a_star]
        regret = float((optimal - achieved).mean())
        # Collateral damage actually inflicted on benign traffic.
        qos = float(((1.0 - eval_states[:, 4]) * QOS_COST[chosen]).mean())

        rows.append(
            {
                "policy": name,
                "latency_ms": latency_ms,
                "qos": qos,
                "regret": regret,
                "accuracy": accuracy,
            }
        )
        print(
            f"  [+] {name}: acc={accuracy:.1f}% regret={regret:.4f} "
            f"QoS-disruption={qos:.4f} lat={latency_ms:.4f}ms"
        )

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Policy Mechanism", "Decision Latency (ms)", "QoS Disruption", "Mean Regret", "Action Accuracy (%)"]
        )
        for r in rows:
            writer.writerow(
                [
                    r["policy"],
                    f"{r['latency_ms']:.4f}",
                    f"{r['qos']:.4f}",
                    f"{r['regret']:.4f}",
                    f"{r['accuracy']:.2f}",
                ]
            )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    names = [r["policy"] for r in rows]
    colors = ["#777777", "#d9534f", "#f0ad4e", "#5cb85c"]
    y = np.arange(len(names))

    ax1.barh(y, [r["accuracy"] for r in rows], color=colors, edgecolor="black")
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("Action selection accuracy vs oracle (%)")
    ax1.set_xlim(0, 115)
    ax1.grid(axis="x", linestyle="--", alpha=0.5)
    for i, r in enumerate(rows):
        ax1.text(r["accuracy"] + 1.5, i, f"{r['accuracy']:.1f}", va="center", fontsize=8, fontweight="bold")

    ax2.barh(y, [r["qos"] for r in rows], color=colors, edgecolor="black")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("QoS disruption on benign traffic (lower is better)")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)
    for i, r in enumerate(rows):
        ax2.text(r["qos"] + 0.004, i, f"{r['qos']:.3f}", va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    return {"rows": rows, "n_eval": n_eval}


if __name__ == "__main__":
    run_policy_benchmark()
