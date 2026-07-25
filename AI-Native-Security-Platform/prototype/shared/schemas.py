"""Shared Pydantic data schemas for AI-Native Security Platform events."""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import time

class TelemetryPayload(BaseModel):
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str = "TCP"
    flow_duration: float
    packet_count: int
    byte_rate: float
    syn_ratio: float
    port_entropy: float
    timestamp: float = Field(default_factory=time.time)

class SecurityEvent(BaseModel):
    event_id: str
    flow_id: str
    src_ip: str
    dst_ip: str
    anomaly_score: float
    is_anomaly: bool
    user_role: str = "UNKNOWN"
    device_type: str = "BYOD"
    trust_score: float = 1.0
    detected_at: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = {}

class MitigationAction(BaseModel):
    action_id: str
    event_id: str
    target_ip: str
    action_type: str  # e.g., BLOCK_IP, QUARANTINE_VLAN, REVOKE_TOKEN, THROTTLE
    risk_index: float
    status: str = "EXECUTED"
    executed_at: float = Field(default_factory=time.time)
    details: str = ""
