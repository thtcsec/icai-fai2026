"""P2: simulated identity isolation and SOAR descriptor (not enforcement) semantics."""

from __future__ import annotations

import os
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from prototype.edge.identity_fusion.fusion import IdentityFusionEngine, UNKNOWN_IDENTITY
from prototype.soar.playbooks import SOARPlaybooks


def test_known_fixture_is_labelled_simulated():
    eng = IdentityFusionEngine()
    ctx = eng.enrich("10.0.1.15")
    assert ctx["is_simulated"] is True
    assert ctx["authenticated"] is False
    assert ctx["identity_source"] == "simulated_lease_fixture"
    assert ctx["role"] == "STUDENT"
    assert "principal_id" in ctx
    assert ctx["principal_id"] != "10.0.1.15"


def test_unknown_ip_is_not_a_guest_identity():
    eng = IdentityFusionEngine()
    ctx = eng.enrich("203.0.113.9")
    assert ctx["role"] == "UNKNOWN"
    assert ctx["device"] == "UNREGISTERED"
    assert ctx["identity_source"] == "none"
    assert ctx["trust_score"] == UNKNOWN_IDENTITY["trust_score"]
    assert ctx["role"] not in {"GUEST", "GUEST_BYOD", "STUDENT", "UNAUTHENTICATED_GUEST"}


def test_soar_descriptors_are_not_enforced():
    soar = SOARPlaybooks()
    event = {"principal_id": "abc", "device": "BYOD"}
    for action in (
        "NORMAL_FORWARDING",
        "DYNAMIC_RATE_LIMIT",
        "V2X_PATH_REROUTE",
        "TARGETED_FLOW_ISOLATION",
    ):
        out = soar.execute_playbook(action, event)
        assert out["enforced"] is False
        assert out["status"] == "RECOMMENDED"
        assert out["kind"] == "action_descriptor"
        assert out["status"] not in {"APPLIED", "EXECUTED", "CONTAINED"}


def test_unknown_action_safe_default_is_pass_through():
    soar = SOARPlaybooks()
    out = soar.execute_playbook("DELETE_ALL_FLOWS", {"principal_id": "abc"})
    assert out["action"] == "PASS_THROUGH"
    assert out["enforced"] is False
    assert out["status"] == "RECOMMENDED"


def test_cloud_soar_executor_does_not_mark_executed():
    from prototype.cloud.soar.playbooks import SOARExecutor
    from prototype.shared.schemas import MitigationAction

    exe = SOARExecutor()
    act = MitigationAction(
        action_id="a1",
        event_id="e1",
        target_ip="10.0.0.1",
        action_type="BLOCK_IP_AND_REVOKE_TOKEN",
        risk_index=0.9,
    )
    out = exe.execute_playbook(act)
    assert out.status == "RECOMMENDED"
    assert "enforced=false" in out.details.lower().replace(" ", "")
