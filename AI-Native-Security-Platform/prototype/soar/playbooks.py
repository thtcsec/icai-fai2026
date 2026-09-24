"""
soar / playbooks.py - Construct SDN/SOAR action descriptors (no dataplane install).

Action names match DRLResilienceAgent.ACTION_NAMES. Outputs are structured
dictionaries describing forwarding, rate limiting, rerouting, or isolation.
Token revocation, quarantine VLANs, and BGP FlowSpec are *not* implemented here.
"""

from typing import Any, Dict


class SOARPlaybooks:
    def execute_playbook(self, action_type: str, event: Dict[str, Any]) -> Dict[str, Any]:
        principal = event.get("principal_id") or event.get("src_ip", "UNKNOWN")
        device = event.get("device", "UNKNOWN")

        if action_type in ("NORMAL_FORWARDING", "PASS_THROUGH"):
            return {
                "action": "PASS_THROUGH",
                "principal_id": principal,
                "status": "NORMAL",
            }
        if action_type == "DYNAMIC_RATE_LIMIT":
            return {
                "action": "SDN_METER_RATE_LIMIT",
                "principal_id": principal,
                "rate_limit_kbps": 1000,
                "status": "APPLIED",
            }
        if action_type == "V2X_PATH_REROUTE":
            return {
                "action": "FLOW_REROUTE",
                "principal_id": principal,
                "device": device,
                "backup_path": "agg_backup",
                "status": "EXECUTED",
            }
        if action_type == "TARGETED_FLOW_ISOLATION":
            return {
                "action": "OPENFLOW_HARD_DROP",
                "principal_id": principal,
                "cooldown_sec": 300,
                "status": "CONTAINED",
            }
        return {
            "action": "PASS_THROUGH",
            "principal_id": principal,
            "status": "NORMAL",
            "note": f"unknown action_type={action_type}",
        }
