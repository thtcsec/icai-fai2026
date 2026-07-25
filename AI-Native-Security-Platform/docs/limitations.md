# System Limitations & Architectural Boundaries

## 1. Edge Hardware Constraints
While the Isolation Forest model exhibits $O(n \log n)$ time complexity during training and $O(h)$ during inference, edge deployment on resource-constrained embedded systems (e.g., dual-core ARM devices) limits maximum un-batched throughput to approximately 15,000 events/second per edge instance.

## 2. Encryption & Deep Packet Inspection (DPI)
Because edge detectors rely primarily on flow-level metadata (packet intervals, burst lengths, port ratios, byte volumes) to preserve privacy and high throughput, payload-level zero-day exploits inside encrypted TLS 1.3 tunnels cannot be inspected without inline decryption proxies.

## 3. False Positive Impact on Campus Workflows
Automated containment actions (e.g., immediate IP block) carry non-zero risk of disrupting legitimate academic activities if an anomalous research dataset upload is misclassified as data exfiltration. The platform mitigates this via role-aware policy thresholds, but conservative fallback thresholds are required for high-priority faculty accounts.
