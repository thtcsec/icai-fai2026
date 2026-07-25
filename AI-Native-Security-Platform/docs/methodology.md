# Methodology: Algorithmic & Architectural Design

## 1. Edge Telemetry Vectorization

Network packet streams at campus edge nodes are vectorized into lightweight statistical feature vectors over sliding time windows $\Delta t = 1.0\text{ s}$:

$$X = \left[ f_{\text{duration}}, f_{\text{pkt\_count}}, f_{\text{byte\_rate}}, f_{\text{port\_entropy}}, f_{\text{syn\_ratio}} \right]$$

## 2. Edge Anomaly Detection (Unsupervised Isolation Forest)

Edge nodes execute an Isolation Forest classifier trained on baseline campus telemetry. Anomaly scores $s(x, n)$ are computed as:

$$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$

where $h(x)$ is the path length of sample $x$, $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree, and $n$ is the number of external nodes. Samples with $s(x, n) > \tau_{\text{edge}}$ (where $\tau_{\text{edge}} = 0.75$) are tagged as potential threats and immediately emitted to the Redis Stream broker.

## 3. Identity Context Fusion Engine

Upon receiving an edge anomaly event, the Identity Fusion service cross-references the source IP/MAC address against campus directory services (LDAP/RADIUS/DHCP logs) to attach contextual metadata:

$$\text{Event}_{\text{enriched}} = \text{Event}_{\text{raw}} \cup \{ \text{UserRole}, \text{DeviceType}, \text{TrustScore} \}$$

* **User Roles**: `FACULTY`, `STUDENT`, `GUEST`, `SYSTEM_ADMIN`, `IOT_DEVICE`.
* **Trust Scores**: Base score derived from device compliance and historical anomaly rate.

## 4. Cloud Policy & Dynamic Threat Scoring

The Cloud Policy Engine computes a composite Risk Index $R$ combining anomaly confidence, resource criticality, and identity trust:

$$R = w_1 \cdot s(x, n) + w_2 \cdot (1 - \text{TrustScore}) + w_3 \cdot \text{AssetCriticality}$$

Dynamic mitigation thresholds determine automated actions:
* $R \ge 0.85$: **CRITICAL** $\rightarrow$ Execute immediate network containment (IP/MAC block + token revocation).
* $0.65 \le R < 0.85$: **WARNING** $\rightarrow$ Throttle bandwidth and trigger step-up multi-factor authentication (MFA).
* $R < 0.65$: **INFO** $\rightarrow$ Log to observability engine for passive monitoring.
