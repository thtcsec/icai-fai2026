"""Metrics Aggregator and Observability Summary Engine."""
import time
from typing import Dict, Any, List
from prototype.shared.schemas import MitigationAction

class MetricsAggregator:
    def __init__(self):
        self.processed_events_count = 0
        self.anomalies_detected_count = 0
        self.actions_executed_count = 0
        self.latencies_ms: List[float] = []

    def record_pipeline_step(self, is_anomaly: bool, total_latency_ms: float):
        self.processed_events_count += 1
        if is_anomaly:
            self.anomalies_detected_count += 1
        self.actions_executed_count += 1
        self.latencies_ms.append(total_latency_ms)

    def get_summary(self) -> Dict[str, Any]:
        avg_latency = float(sum(self.latencies_ms) / len(self.latencies_ms)) if self.latencies_ms else 0.0
        return {
            "total_events_processed": self.processed_events_count,
            "anomalies_detected": self.anomalies_detected_count,
            "actions_executed": self.actions_executed_count,
            "average_end_to_end_latency_ms": round(avg_latency, 3),
            "system_health": "OPTIMAL"
        }
