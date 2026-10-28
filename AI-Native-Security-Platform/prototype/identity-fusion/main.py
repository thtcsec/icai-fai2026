"""
identity-fusion / main.py - Spatial-Temporal Identity Context Fusion Module

Enriches raw edge telemetry events with campus identity context (IP/MAC -> User/Role mapping).
This module is a *simulated* lease-table fixture, not a deployable IdP.

Unknown addresses are UNKNOWN (not a guest/student identity) with low trust.
"""

from typing import Any, Dict

SIMULATED_LEASE_TABLE = {
    "10.0.1.15": {"mac": "AA:BB:CC:DD:EE:01", "user_role": "STUDENT_BYOD", "trust_score": 0.45},
    "10.0.1.50": {"mac": "AA:BB:CC:DD:EE:02", "user_role": "FACULTY_RESEARCH", "trust_score": 0.85},
    "10.0.2.100": {"mac": "AA:BB:CC:DD:EE:03", "user_role": "CAMPUS_IOT_SENSOR", "trust_score": 0.90},
    "10.0.3.200": {"mac": "AA:BB:CC:DD:EE:04", "user_role": "ADMIN_WORKSTATION", "trust_score": 0.95},
}

UNKNOWN_IDENTITY = {
    "mac": "UNKNOWN",
    "user_role": "UNKNOWN",
    "trust_score": 0.10,
}


class IdentityContextFusion:
    def __init__(self):
        # Simulated campus RADIUS / DHCP lease table (fixture only).
        self.lease_table = dict(SIMULATED_LEASE_TABLE)
        self.table_kind = "simulated_lease_fixture"

    def enrich_event(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        src_ip = raw_event.get("src_ip", "")
        known = src_ip in self.lease_table
        identity_meta = self.lease_table.get(src_ip, UNKNOWN_IDENTITY)

        enriched_event = dict(raw_event)
        enriched_event.update(
            {
                "user_role": identity_meta["user_role"],
                "trust_score": identity_meta["trust_score"],
                "identity_source": "simulated_lease_fixture" if known else "none",
                "is_simulated": True,
                "authenticated": False,
            }
        )
        # MAC stays edge-local for the fixture; do not treat as a cloud payload.
        enriched_event["mac_address_edge_local"] = identity_meta["mac"]
        return enriched_event
