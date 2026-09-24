"""Identity Context Fusion Engine for Campus Telemetry Enrichment.

The lookup table is a *simulated* campus DHCP/RADIUS lease map for the
prototype. Production deployments would query real directory services; this
artifact does not speak LDAP/RADIUS on the wire.
"""

from typing import Any, Dict

from prototype.edge.privacy.pseudonymize import principal_id


class IdentityFusionEngine:
    def __init__(self):
        # Simulated campus LDAP/RADIUS DHCP lease table (local edge only).
        self._identity_table: Dict[str, Dict[str, Any]] = {
            "10.0.1.15": {"role": "STUDENT", "device": "BYOD_LAPTOP", "trust_score": 0.85},
            "10.0.1.50": {"role": "FACULTY", "device": "RESEARCH_WORKSTATION", "trust_score": 0.95},
            "10.0.2.105": {"role": "IOT_DEVICE", "device": "SMART_HVAC_SENSOR", "trust_score": 0.50},
            "10.0.3.200": {"role": "GUEST", "device": "BYOD_SMARTPHONE", "trust_score": 0.40},
            "10.0.9.99": {"role": "SUSPICIOUS_UNKNOWN", "device": "UNREGISTERED_MAC", "trust_score": 0.10},
        }

    def enrich(self, src_ip: str) -> Dict[str, Any]:
        """Look up IP locally and return cloud-safe fields (HMAC principal_id)."""
        context = self._identity_table.get(
            src_ip,
            {"role": "GUEST_BYOD", "device": "UNMANAGED_DEVICE", "trust_score": 0.50},
        )
        return {
            "principal_id": principal_id(src_ip),
            "role": context["role"],
            "device": context["device"],
            "trust_score": float(context["trust_score"]),
        }
