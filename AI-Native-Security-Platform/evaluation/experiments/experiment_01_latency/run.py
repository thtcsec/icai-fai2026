"""Experiment 1: End-to-End Latency Breakdown Benchmark."""
import time
import numpy as np
import pandas as pd
from prototype.edge.detector.model import EdgeAnomalyDetector
from prototype.edge.identity_fusion.fusion import IdentityFusionEngine
from prototype.cloud.policy_engine.policy import CloudPolicyEngine
from prototype.cloud.soar.playbooks import SOARExecutor
from prototype.shared.schemas import SecurityEvent

def run_latency_experiment(num_trials: int = 1000) -> pd.DataFrame:
    detector = EdgeAnomalyDetector()
    identity_fusion = IdentityFusionEngine()
    policy_engine = CloudPolicyEngine()
    soar = SOARExecutor()

    # Pre-fit baseline
    normal_samples = np.random.normal(loc=[1.5, 20.0, 1000.0, 0.05, 0.2], scale=[0.5, 5.0, 200.0, 0.01, 0.05], size=(100, 5))
    detector.fit_baseline(normal_samples)

    edge_detection_times = []
    identity_fusion_times = []
    policy_reasoning_times = []
    soar_execution_times = []
    total_latencies = []

    for i in range(num_trials):
        feat = np.array([25.0, 8000, 500000.0, 0.88, 0.95])

        # Step 1: Edge Detection
        t0 = time.perf_counter()
        is_anomaly, score = detector.predict_anomaly(feat)
        t1 = time.perf_counter()

        # Step 2: Identity Fusion
        context = identity_fusion.enrich("10.0.9.99")
        t2 = time.perf_counter()

        # Step 3: Cloud Policy Reasoning
        event = SecurityEvent(
            event_id=f"evt-{i}", flow_id=f"flow-{i}", src_ip="10.0.9.99",
            dst_ip="192.168.1.1", anomaly_score=score, is_anomaly=is_anomaly,
            user_role=context["role"], trust_score=context["trust_score"]
        )
        action = policy_engine.generate_mitigation(event)
        t3 = time.perf_counter()

        # Step 4: SOAR Execution
        executed_action = soar.execute_playbook(action)
        t4 = time.perf_counter()

        edge_ms = (t1 - t0) * 1000.0
        fusion_ms = (t2 - t1) * 1000.0
        policy_ms = (t3 - t2) * 1000.0
        soar_ms = (t4 - t3) * 1000.0
        total_ms = (t4 - t0) * 1000.0

        edge_detection_times.append(edge_ms)
        identity_fusion_times.append(fusion_ms)
        policy_reasoning_times.append(policy_ms)
        soar_execution_times.append(soar_ms)
        total_latencies.append(total_ms)

    df_results = pd.DataFrame({
        "trial_id": range(num_trials),
        "edge_detection_ms": edge_detection_times,
        "identity_fusion_ms": identity_fusion_times,
        "policy_reasoning_ms": policy_reasoning_times,
        "soar_execution_ms": soar_execution_times,
        "total_e2e_latency_ms": total_latencies
    })

    return df_results
