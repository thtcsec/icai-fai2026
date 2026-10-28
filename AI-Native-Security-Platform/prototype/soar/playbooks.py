"""
soar / playbooks.py - Construct SDN/SOAR action *descriptors* (no dataplane install).

Action names match DRLResilienceAgent.ACTION_NAMES. Outputs are structured
dictionaries describing a *recommended* forwarding, rate limiting, rerouting,
or isolation action. Nothing here installs OpenFlow rules, revokes tokens,
quarantines VLANs, or talks to BGP FlowSpec.

Safe default for unknown actions: PASS_THROUGH recommended, not enforced.
"""

from typing import Any, Dict

DESCRIPTOR_KIND = "action_descriptor"
DEFAULT_STATUS = "RECOMMENDED"


class SOARPlaybooks:
    def execute_playbook(self, action_type: str, event: Dict[str, Any]) -> Dict[str, Any]:
        principal = event.get("principal_id") or "UNKNOWN"
        device = event.get("device", "UNKNOWN")
        base = {
            "kind": DESCRIPTOR_KIND,
            "status": DEFAULT_STATUS,
            "enforced": False,
            "principal_id": principal,
            "requested_action": action_type,
        }

        if action_type in ("NORMAL_FORWARDING", "PASS_THROUGH"):
            return {
                **base,
                "action": "PASS_THROUGH",
                "note": "Recommended allow; not a dataplane confirmation.",
            }
        if action_type == "DYNAMIC_RATE_LIMIT":
            return {
                **base,
                "action": "SDN_METER_RATE_LIMIT",
                "rate_limit_kbps": 1000,
                "note": "Recommended meter; not installed on a switch.",
            }
        if action_type == "V2X_PATH_REROUTE":
            return {
                **base,
                "action": "FLOW_REROUTE",
                "device": device,
                "backup_path": "agg_backup",
                "note": "Recommended reroute descriptor; no flow_mod sent.",
            }
        if action_type == "TARGETED_FLOW_ISOLATION":
            return {
                **base,
                "action": "OPENFLOW_HARD_DROP",
                "cooldown_sec": 300,
                "note": "Recommended isolation descriptor; drop rule not installed.",
            }
        return {
            **base,
            "action": "PASS_THROUGH",
            "note": f"unknown action_type={action_type}; safe default is recommended PASS_THROUGH",
        }
