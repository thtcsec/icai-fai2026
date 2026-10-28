"""SOAR playbook *descriptor* builder (no dataplane / IAM enforcement).

This class exists for the prototype experiment harness. It does not issue SDN
ACL/BGP commands, revoke tokens, or move switch ports. Returned status is
RECOMMENDED, never EXECUTED.
"""
import time
import logging
from prototype.shared.schemas import MitigationAction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SOAR-Engine")


class SOARExecutor:
    def __init__(self):
        self.action_history = []

    def execute_playbook(self, action: MitigationAction) -> MitigationAction:
        """Build a recommended descriptor. Does not enforce anything."""
        start_time = time.perf_counter()
        self._log_recommendation(action.action_type, action.target_ip)
        execution_latency = (time.perf_counter() - start_time) * 1000.0
        action.status = "RECOMMENDED"
        action.details += (
            f" | Descriptor only (enforced=false) in {execution_latency:.2f} ms"
        )
        self.action_history.append(action)
        return action

    def _log_recommendation(self, action_type: str, target_ip: str):
        logger.info(
            "[SOAR DESCRIPTOR] Would recommend %s for principal/target %s "
            "(not enforced; no SDN/IAM call)",
            action_type,
            target_ip,
        )
