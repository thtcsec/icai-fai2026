"""
risk_scorer.py - Dynamic Risk Scorer Engine

Ported from ct-smartcam (soar-engine/app/core/scoring.py).
Evaluates severity, threat intelligence metrics (VirusTotal, AbuseIPDB, Tor/VPN/Proxy),
and event correlation bonuses to compute composite incident risk scores.
"""

from typing import Dict, Any, Optional
from enum import Enum

class IncidentSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class DynamicRiskScorer:
    BASE_SCORES = {
        IncidentSeverity.LOW: 2.0,
        IncidentSeverity.MEDIUM: 5.0,
        IncidentSeverity.HIGH: 7.0,
        IncidentSeverity.CRITICAL: 9.0,
    }

    IP_REPUTATION_SCORES = {
        "vpn": 2.0,
        "proxy": 2.5,
        "tor": 3.0,
        "datacenter": 1.5,
        "residential": 0.0,
    }

    ALERT_TYPE_SCORES = {
        "ddos": 3.0,
        "intrusion": 2.5,
        "botnet_exfil": 3.0,
        "port_scan": 1.5,
        "loft_attack": 2.5,
        "anomaly": 1.5,
    }

    def calculate_score(
        self,
        severity: IncidentSeverity,
        alert_type: str,
        threat_intel: Optional[Dict[str, Any]] = None,
        correlated_events_count: int = 0
    ) -> float:
        base_score = self.BASE_SCORES.get(severity, 5.0)
        alert_type_score = self.ALERT_TYPE_SCORES.get(alert_type.lower(), 1.0)

        ip_reputation_score = 0.0
        if threat_intel:
            if threat_intel.get("is_tor"):
                ip_reputation_score += self.IP_REPUTATION_SCORES["tor"]
            if threat_intel.get("is_vpn"):
                ip_reputation_score += self.IP_REPUTATION_SCORES["vpn"]
            if threat_intel.get("is_proxy"):
                ip_reputation_score += self.IP_REPUTATION_SCORES["proxy"]
            if threat_intel.get("is_datacenter"):
                ip_reputation_score += self.IP_REPUTATION_SCORES["datacenter"]

            vt_score = threat_intel.get("vt_score", 0)
            if vt_score >= 8:
                ip_reputation_score += 3.0
            elif vt_score >= 5:
                ip_reputation_score += 2.0

            abuse_score = threat_intel.get("abuseipdb_score", 0)
            if abuse_score >= 70:
                ip_reputation_score += 2.0

        repeated_offense_bonus = 0.0
        if correlated_events_count >= 5:
            repeated_offense_bonus = 2.0
        elif correlated_events_count >= 2:
            repeated_offense_bonus = 1.0

        total_score = base_score + alert_type_score + ip_reputation_score + repeated_offense_bonus
        return float(min(total_score, 10.0))
