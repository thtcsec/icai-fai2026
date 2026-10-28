"""
generate_table9.py - Orchestrator only.

Train and eval are *separate* entry points with independent seeds:
  python results/train_dqn_checkpoint.py
  python results/eval_dqn_checkpoint.py

This file exists so older commands still work, but it never scores the training
RNG stream. Metrics are simulator-only oracle match, not SOAR enforcement.
"""

from __future__ import annotations

import argparse
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from eval_dqn_checkpoint import evaluate
from train_dqn_checkpoint import main as train_main


def run_policy_benchmark(seed: int = 42, n_eval: int = 5000, eval_only: bool = False):
    if not eval_only:
        train_main(seed=seed)
    return evaluate(eval_seed=2026, n_eval=n_eval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    run_policy_benchmark(eval_only=args.eval_only)
