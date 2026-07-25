"""Cloud Dynamic Policy & Threat Reasoning Engine."""
import uuid
import time
from typing import Tuple, Dict, Any
from prototype.shared.schemas import SecurityEvent, MitigationAction

class CloudPolicyEngine:
    def __init__(self, w_anomaly: float = 0.5, w_trust: float = 0.3, w_criticality: float = 0.2):
        self.w_anomaly = w_anomaly
        self.w_trust = w_trust
        self.w_criticality = w_criticality

    def evaluate_threat(self, event: SecurityEvent) -> Tuple[float, str, str]:
        """Calculates dynamic Risk Index R and decides mitigation action type."""
        # Criticality mapping based on destination IP or role
        asset_criticality = 0.9 if "10.0.0" in event.dst_ip or event.user_role == "SUSPICIOUS_UNKNOWN" else 0.4
        
        # Risk Index calculation
        risk_index = (
            self.w_anomaly * event.anomaly_score +
            self.w_trust * (1.0 - event.trust_score) +
            self.w_criticality * asset_criticality
        )

        if risk_index >= 0.70:
            severity = "CRITICAL"
            action_type = "BLOCK_IP_AND_REVOKE_TOKEN"
        elif risk_index >= 0.50:
            severity = "HIGH"
            action_type = "QUARANTINE_VLAN"
        elif risk_index >= 0.35:
            severity = "MEDIUM"
            action_type = "THROTTLE_BANDWIDTH"
        else:
            severity = "LOW"
            action_type = "LOG_PASSIVE"

        return risk_index, severity, action_type

    def generate_mitigation(self, event: SecurityEvent) -> MitigationAction:
        risk_index, severity, action_type = self.evaluate_threat(event)
        return MitigationAction(
            action_id=f"act-{uuid.uuid4().hex[:8]}",
            event_id=event.event_id,
            target_ip=event.src_ip,
            action_type=action_type,
            risk_index=risk_index,
            status="PENDING",
            details=f"Evaluated severity {severity} (R={risk_index:.3f}) for Role {event.user_role}"
        )
