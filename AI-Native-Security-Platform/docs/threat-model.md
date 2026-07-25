# Threat Model & University Attack Vectors

## Threat Matrix (STRIDE & MITRE ATT&CK Mapping)

| Threat Category | Primary Target | Campus Attack Scenario | MITRE ATT&CK ID | AI-Native Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Spoofing / Rogue AP** | Student BYOD | Rogue Wi-Fi access point broadcasting campus SSID | `T1557` (MITM) | Identity Fusion detects MAC/IP mismatch & abnormal gateway MAC |
| **Tampering** | IoT Sensors | Compromised HVAC or camera pushing corrupted telemetry | `T1565` (Data Manipulation) | Edge Isolation Forest flags anomalous payload entropy & packet rate |
| **Information Disclosure**| Research Databases | Exfiltration of unreleased academic research via SSH | `T1048` (Exfiltration) | Policy Engine flags high byte-rate flow from non-admin role |
| **Denial of Service** | University Portal | Distributed SYN flood from infected campus dormitory endpoints | `T1498` (Network DoS) | Edge detector triggers automated BGP flowspec / rate limiting |
| **Elevation of Privilege** | Campus LDAP | Compromised student account attempting domain admin escalation | `T1078` (Valid Accounts) | Risk Resolver flags sudden role transition attempt |

## Assumptions & Trust Boundaries

1. **Trusted Control Plane**: The Redis Stream broker and Cloud Policy Engine operate within a hardened management VLAN.
2. **Untrusted Edge**: Campus edge taps operate in untrusted zones; all messages emitted to Redis Streams are digitally signed and TLS-encrypted.
3. **No Inline Bottlenecks**: Edge detection operates in passive mirror/TAP mode to avoid introducing single points of failure into physical network backbones.
