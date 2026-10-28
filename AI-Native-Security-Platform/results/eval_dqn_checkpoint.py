"""Evaluate a *previously saved* DQN checkpoint. Never trains or overwrites it."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(__file__))

from policy_sim import (
    N_ACTIONS,
    QOS_COST,
    REWARD_ENVIRONMENT,
    oracle_actions,
    policy_heuristic,
    policy_static,
    reward_matrix,
    sample_states,
    states_fingerprint,
)
from prototype.cloud.policy_engine.drl_sdn_agent import DRLResilienceAgent

CKPT_PATH = os.path.join(BASE, "results", "checkpoints", "dqn_policy_trained.pt")
TRAIN_HASH_PATH = os.path.join(BASE, "results", "corrected", "dqn_train_state_hashes.bin")
CSV_PATH = os.path.join(BASE, "results", "corrected", "table9_policy_benchmark.csv")
JSON_PATH = os.path.join(BASE, "results", "corrected", "table9_policy_eval.json")
EVAL_SEED = 2026  # independent of TRAIN_SEED=42
HISTORICAL_DQN_ACCURACY_PCT = 98.76


def _load_train_hashes(path: str) -> set[bytes]:
    if not os.path.isfile(path):
        return set()
    data = open(path, "rb").read()
    return {data[i : i + 32] for i in range(0, len(data), 32) if len(data[i : i + 32]) == 32}


def _rr_factory():
    box = {"i": 0}

    def fn(_state):
        a = box["i"] % N_ACTIONS
        box["i"] += 1
        return a

    fn.reset = lambda: box.__setitem__("i", 0)
    return fn


def evaluate(
    ckpt: str = CKPT_PATH,
    eval_seed: int = EVAL_SEED,
    n_eval: int = 5000,
    train_hash_path: str = TRAIN_HASH_PATH,
    write_outputs: bool = True,
):
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Trained DQN checkpoint required but missing: {ckpt}")
    print(f"[*] DQN eval-only (simulated reward, eval_seed={eval_seed})")
    print("    THIS ENTRY POINT DOES NOT TRAIN OR SAVE WEIGHTS.")
    agent = DRLResilienceAgent.from_checkpoint(ckpt)
    rng = np.random.default_rng(eval_seed)
    random.seed(eval_seed)
    states = sample_states(n_eval, rng)
    train_hashes = _load_train_hashes(train_hash_path)
    eval_hashes = states_fingerprint(states)
    overlap = train_hashes & eval_hashes
    if overlap:
        raise AssertionError(
            f"Eval states overlap training records: n={len(overlap)}. "
            "Use a distinct eval seed."
        )

    a_star = oracle_actions(states)
    rewards = reward_matrix(states)
    rr = _rr_factory()

    def policy_dqn(state: np.ndarray) -> int:
        return agent.select_action(state.tolist(), eval_mode=True)

    def policy_random(_state: np.ndarray) -> int:
        return random.randrange(N_ACTIONS)

    policies = [
        ("Uniform Random Policy", policy_random),
        ("Static Rule Lookup", policy_static),
        ("Round-Robin Action Assignment", rr),
        ("Heuristic Risk Table", policy_heuristic),
        ("Deep Q-Network (DQN Agent, simulated reward)", policy_dqn),
    ]

    rows = []
    for name, fn in policies:
        if hasattr(fn, "reset"):
            fn.reset()
        t0 = time.perf_counter()
        chosen = np.array([fn(s) for s in states])
        latency_ms = ((time.perf_counter() - t0) / n_eval) * 1000.0
        accuracy = float((chosen == a_star).mean() * 100.0)
        achieved = rewards[np.arange(n_eval), chosen]
        optimal = rewards[np.arange(n_eval), a_star]
        regret = float((optimal - achieved).mean())
        qos = float(((1.0 - states[:, 4]) * QOS_COST[chosen]).mean())
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
            f"  [+] {name}: acc={accuracy:.2f}% regret={regret:.4f} "
            f"QoS={qos:.4f} lat={latency_ms:.4f}ms"
        )

    payload = {
        "rows": rows,
        "n_eval": n_eval,
        "eval_seed": eval_seed,
        "train_seed": 42,
        "checkpoint": ckpt,
        "n_eval_train_hash_overlap": 0,
        "reward_environment": REWARD_ENVIRONMENT,
        "operational_enforcement": False,
        "historical_dqn_accuracy_pct": HISTORICAL_DQN_ACCURACY_PCT,
        "trained_this_process": False,
        "saved_checkpoint": False,
    }
    if write_outputs:
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "Policy Mechanism",
                    "Decision Latency (ms)",
                    "QoS Disruption",
                    "Mean Regret",
                    "Action Accuracy (%)",
                ]
            )
            for r in rows:
                w.writerow(
                    [
                        r["policy"],
                        f"{r['latency_ms']:.4f}",
                        f"{r['qos']:.4f}",
                        f"{r['regret']:.4f}",
                        f"{r['accuracy']:.2f}",
                    ]
                )
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT_PATH)
    ap.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    ap.add_argument("--n-eval", type=int, default=5000)
    args = ap.parse_args()
    evaluate(ckpt=args.ckpt, eval_seed=args.eval_seed, n_eval=args.n_eval)


if __name__ == "__main__":
    main()
