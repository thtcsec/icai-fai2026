"""Synthetic Telemetry & Anomaly Stream Generator for ICAI-2026 Evaluation."""
import os
import json
import numpy as np
import pandas as pd

def generate_evaluation_dataset(num_samples: int = 5000, anomaly_ratio: float = 0.08, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    num_anomalies = int(num_samples * anomaly_ratio)
    num_normal = num_samples - num_anomalies

    # Normal campus flow telemetry distributions
    normal_duration = np.random.exponential(scale=2.5, size=num_normal) + 0.1
    normal_pkts = np.random.poisson(lam=25, size=num_normal) + 1
    normal_bytes = normal_pkts * np.random.normal(loc=600, scale=100, size=num_normal)
    normal_syn_ratio = np.random.beta(a=1, b=20, size=num_normal)
    normal_port_entropy = np.random.beta(a=2, b=10, size=num_normal)
    normal_labels = np.zeros(num_normal, dtype=int)

    # Malicious campus attack telemetry distributions (Floods, Exfiltration, Scans)
    attack_duration = np.random.exponential(scale=35.0, size=num_anomalies) + 5.0
    attack_pkts = np.random.poisson(lam=1200, size=num_anomalies) + 100
    attack_bytes = attack_pkts * np.random.normal(loc=1200, scale=300, size=num_anomalies)
    attack_syn_ratio = np.random.beta(a=15, b=2, size=num_anomalies)
    attack_port_entropy = np.random.beta(a=10, b=2, size=num_anomalies)
    attack_labels = np.ones(num_anomalies, dtype=int)

    # Combine datasets
    durations = np.concatenate([normal_duration, attack_duration])
    pkts = np.concatenate([normal_pkts, attack_pkts])
    bytes_arr = np.concatenate([normal_bytes, attack_bytes])
    syn_ratios = np.concatenate([normal_syn_ratio, attack_syn_ratio])
    port_entropies = np.concatenate([normal_port_entropy, attack_port_entropy])
    labels = np.concatenate([normal_labels, attack_labels])

    # Shuffle
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    df = pd.DataFrame({
        "flow_id": [f"flow-{i:06d}" for i in range(num_samples)],
        "src_ip": [f"10.0.{np.random.randint(1, 10)}.{np.random.randint(2, 250)}" for _ in range(num_samples)],
        "dst_ip": [f"192.168.1.{np.random.randint(2, 250)}" for _ in range(num_samples)],
        "flow_duration": np.round(durations[indices], 3),
        "packet_count": pkts[indices],
        "byte_rate": np.round(bytes_arr[indices] / np.maximum(durations[indices], 0.1), 2),
        "syn_ratio": np.round(syn_ratios[indices], 4),
        "port_entropy": np.round(port_entropies[indices], 4),
        "label": labels[indices]
    })

    return df

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "processed")
    os.makedirs(out_dir, exist_ok=True)
    df = generate_evaluation_dataset()
    df.to_csv(os.path.join(out_dir, "campus_telemetry_benchmark.csv"), index=False)
    print(f"Generated benchmark dataset with {len(df)} samples at {out_dir}/campus_telemetry_benchmark.csv")
