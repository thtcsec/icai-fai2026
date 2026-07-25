"""Edge Detector Main Pipeline execution module."""
import time
import uuid
import numpy as np
from prototype.edge.detector.model import EdgeAnomalyDetector
from prototype.edge.identity_fusion.fusion import IdentityFusionEngine
from prototype.edge.redis_stream.pubsub import EventBusClient
from prototype.shared.schemas import TelemetryPayload, SecurityEvent

def run_edge_pipeline_demo():
    print("=" * 60)
    print("AI-NATIVE EDGE SECURITY DETECTOR & IDENTITY FUSION RUNNER")
    print("=" * 60)

    # Initialize components
    detector = EdgeAnomalyDetector()
    identity_fusion = IdentityFusionEngine()
    event_bus = EventBusClient()

    # Train baseline on sample normal telemetry
    normal_samples = np.random.normal(loc=[1.5, 20.0, 1000.0, 0.05, 0.2], scale=[0.5, 5.0, 200.0, 0.01, 0.05], size=(200, 5))
    detector.fit_baseline(normal_samples)

    # Simulated incoming streaming telemetry
    test_telemetries = [
        # Normal Student Flow
        TelemetryPayload(
            flow_id="flow-101", src_ip="10.0.1.15", dst_ip="140.82.112.4",
            src_port=54321, dst_port=443, flow_duration=1.2, packet_count=22,
            byte_rate=1050.0, syn_ratio=0.04, port_entropy=0.22
        ),
        # Anomalous Exfiltration / Flood Stream
        TelemetryPayload(
            flow_id="flow-999", src_ip="10.0.9.99", dst_ip="198.51.100.44",
            src_port=4444, dst_port=80, flow_duration=45.0, packet_count=15000,
            byte_rate=950000.0, syn_ratio=0.92, port_entropy=0.98
        )
    ]

    for telemetry in test_telemetries:
        features = np.array([
            telemetry.flow_duration, telemetry.packet_count,
            telemetry.byte_rate, telemetry.syn_ratio, telemetry.port_entropy
        ])

        start_time = time.perf_counter()
        is_anomaly, score = detector.predict_anomaly(features)
        context = identity_fusion.enrich(telemetry.src_ip)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        event = SecurityEvent(
            event_id=f"evt-{uuid.uuid4().hex[:8]}",
            flow_id=telemetry.flow_id,
            src_ip=telemetry.src_ip,
            dst_ip=telemetry.dst_ip,
            anomaly_score=score,
            is_anomaly=is_anomaly,
            user_role=context["role"],
            device_type=context["device"],
            trust_score=context["trust_score"]
        )

        msg_id = event_bus.publish("security:telemetry:stream", event.dict())

        print(f"\n[Processed Flow {telemetry.flow_id}] ({latency_ms:.3f} ms)")
        print(f"  Src IP: {telemetry.src_ip} | Role: {event.user_role} | Device: {event.device_type}")
        print(f"  Anomaly Score: {score:.4f} | Is Anomaly: {is_anomaly}")
        print(f"  Published to Stream ID: {msg_id}")

if __name__ == "__main__":
    run_edge_pipeline_demo()
