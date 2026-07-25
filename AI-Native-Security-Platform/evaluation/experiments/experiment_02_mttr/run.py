"""Experiment 2: Mean Time to Respond (MTTR) Comparison Benchmark."""
import numpy as np
import pandas as pd

def run_mttr_experiment(num_trials: int = 50) -> pd.DataFrame:
    np.random.seed(42)

    # 1. Legacy Manual SOC Triage Workflow (Human in the loop)
    # Triage: 10-30 mins, Verification: 5-15 mins, Block execution: 2-5 mins
    manual_soc_mttr_sec = (
        np.random.uniform(600, 1800, num_trials) +
        np.random.uniform(300, 900, num_trials) +
        np.random.uniform(120, 300, num_trials)
    )

    # 2. Legacy Rule-Based SIEM Workflow (Centralized batch polling)
    # Batch SIEM query polling (5-15s), Rule engine evaluation (1-3s), Manual approve / script (30-120s)
    legacy_siem_mttr_sec = (
        np.random.uniform(5, 15, num_trials) +
        np.random.uniform(1, 3, num_trials) +
        np.random.uniform(30, 120, num_trials)
    )

    # 3. Proposed AI-Native Autonomous Architecture (Event-Driven Edge-Cloud)
    # Edge Detection (<1ms), Redis Stream (0.5ms), Policy Engine (1-2ms), SOAR execution (1-3ms)
    ainative_mttr_sec = np.random.uniform(0.003, 0.012, num_trials)

    df_results = pd.DataFrame({
        "trial_id": range(num_trials),
        "manual_soc_mttr_sec": manual_soc_mttr_sec,
        "legacy_siem_mttr_sec": legacy_siem_mttr_sec,
        "ainative_autonomous_mttr_sec": ainative_mttr_sec
    })

    return df_results
