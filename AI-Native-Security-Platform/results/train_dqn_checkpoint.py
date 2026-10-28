"""Train-only DQN entry point. Never scores the evaluation seed."""

from __future__ import annotations

import json
import os
import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from policy_sim import (
    N_ACTIONS,
    REWARD_ENVIRONMENT,
    STATE_DIM,
    reward_matrix,
    sample_states,
    states_fingerprint,
)
from prototype.cloud.policy_engine.drl_sdn_agent import DQNNetwork, DRLResilienceAgent

CKPT_PATH = os.path.join(BASE, "results", "checkpoints", "dqn_policy_trained.pt")
META_PATH = os.path.join(BASE, "results", "corrected", "dqn_train_meta.json")
TRAIN_HASH_PATH = os.path.join(BASE, "results", "corrected", "dqn_train_state_hashes.bin")
TRAIN_SEED = 42


def train_dqn(seed: int = TRAIN_SEED, steps: int = 4000, batch_size: int = 64):
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    net = DQNNetwork(STATE_DIM, N_ACTIONS)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    replay = deque(maxlen=20000)
    epsilon = 1.0
    seen = []
    for _ in range(steps):
        s = sample_states(1, rng)[0]
        seen.append(s)
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
            opt.zero_grad()
            loss_fn(q, br).backward()
            opt.step()
    net.eval()
    return net, np.asarray(seen, dtype=np.float64)


def main(seed: int = TRAIN_SEED, steps: int = 4000):
    print(f"[*] DQN train-only (simulated reward, seed={seed}, steps={steps})")
    net, train_states = train_dqn(seed=seed, steps=steps)
    agent = DRLResilienceAgent(allow_uninitialized=True)
    agent.policy_net.load_state_dict(net.state_dict())
    agent.policy_net.eval()
    agent.loaded = True
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    agent.save_checkpoint(
        CKPT_PATH,
        extra={
            "train_seed": seed,
            "eval_seed_used_here": False,
            "steps": steps,
            "reward_environment": REWARD_ENVIRONMENT,
            "train_only": True,
        },
    )
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    fps = states_fingerprint(train_states)
    with open(TRAIN_HASH_PATH, "wb") as f:
        for h in fps:
            f.write(h)
    meta = {
        "checkpoint": CKPT_PATH,
        "train_seed": seed,
        "n_train_states": int(len(train_states)),
        "n_unique_train_hashes": int(len(fps)),
        "reward_environment": REWARD_ENVIRONMENT,
        "operational_enforcement": False,
        "entry_point": "train_dqn_checkpoint.py",
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  [=] saved {CKPT_PATH}")
    print(f"  [=] train unique states {len(fps)}")
    return meta


if __name__ == "__main__":
    main()
