"""Canonical 5-D DQN observation contract.

Training (Table IX) samples:
    [switch_cpu, packetin_rate, bw_util, recon_error, anomaly_conf]
each in [0, 1] from the analytic campus-contention simulator.

The latency / detector path must encode into this same layout. Missing live
telemetry is filled with an explicit placeholder and recorded in provenance;
it is never a silent hardcoded attack probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

STATE_DIM = 5
STATE_KEYS = (
    "switch_cpu",
    "packetin_rate",
    "bw_util",
    "recon_error",
    "anomaly_conf",
)
PLACEHOLDER_NEUTRAL = 0.5
PACKET_RATE_FEATURE_INDEX = 1  # FEATURE_COLUMNS: packet_rate
BYTE_RATE_FEATURE_INDEX = 2
CHANNEL_OCCUPANCY_FEATURE_INDEX = 9


@dataclass(frozen=True)
class PolicyObservation:
    vector: List[float]
    provenance: Dict[str, str] = field(default_factory=dict)

    def as_list(self) -> List[float]:
        return list(self.vector)


def _unit(value: float, name: str) -> float:
    v = float(value)
    if not np_isfinite(v):
        raise ValueError(f"Non-finite {name}={value!r}")
    return float(min(1.0, max(0.0, v)))


def np_isfinite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


def squash_reconstruction_error(raw_mse: float) -> float:
    """Map non-negative MSE onto (0, 1) without claiming calibration."""
    x = max(0.0, float(raw_mse))
    return float(x / (1.0 + x))


def squash_zscore(z: float) -> float:
    """Map a standardized feature onto (0, 1) via a logistic on z/3."""
    import math

    return float(1.0 / (1.0 + math.exp(-float(z) / 3.0)))


def encode_observation(
    *,
    anomaly_conf: float,
    recon_error: float,
    packetin_rate: Optional[float] = None,
    bw_util: Optional[float] = None,
    switch_cpu: Optional[float] = None,
    packetin_rate_is_zscore: bool = False,
    bw_util_is_zscore: bool = False,
    recon_error_is_raw_mse: bool = True,
) -> PolicyObservation:
    provenance: Dict[str, str] = {}

    if switch_cpu is None:
        switch_cpu_v = PLACEHOLDER_NEUTRAL
        provenance["switch_cpu"] = "placeholder_neutral_0.5_not_live_telemetry"
    else:
        switch_cpu_v = _unit(switch_cpu, "switch_cpu")
        provenance["switch_cpu"] = "live"

    if packetin_rate is None:
        packetin_v = PLACEHOLDER_NEUTRAL
        provenance["packetin_rate"] = "placeholder_neutral_0.5_not_live_telemetry"
    elif packetin_rate_is_zscore:
        packetin_v = _unit(squash_zscore(packetin_rate), "packetin_rate")
        provenance["packetin_rate"] = "from_window_packet_rate_zscore_logistic"
    else:
        packetin_v = _unit(packetin_rate, "packetin_rate")
        provenance["packetin_rate"] = "live_unit_interval"

    if bw_util is None:
        bw_v = PLACEHOLDER_NEUTRAL
        provenance["bw_util"] = "placeholder_neutral_0.5_not_live_telemetry"
    elif bw_util_is_zscore:
        bw_v = _unit(squash_zscore(bw_util), "bw_util")
        provenance["bw_util"] = "from_window_channel_occupancy_zscore_logistic"
    else:
        bw_v = _unit(bw_util, "bw_util")
        provenance["bw_util"] = "live_unit_interval"

    if recon_error_is_raw_mse:
        rec_v = _unit(squash_reconstruction_error(recon_error), "recon_error")
        provenance["recon_error"] = "detector_mse_squashed"
    else:
        rec_v = _unit(recon_error, "recon_error")
        provenance["recon_error"] = "live_unit_interval"

    conf_v = _unit(anomaly_conf, "anomaly_conf")
    provenance["anomaly_conf"] = "detector_p_attack_1_minus_p_normal"

    vector = [switch_cpu_v, packetin_v, bw_v, rec_v, conf_v]
    if len(vector) != STATE_DIM:
        raise RuntimeError("observation encoding produced the wrong dimension")
    return PolicyObservation(vector=vector, provenance=provenance)


def observation_from_detector_window(
    p_attack: float,
    reconstruction_error: float,
    window_last_flow: Optional[Sequence[float]] = None,
) -> PolicyObservation:
    """Build a DQN observation from detector outputs and the last flow vector."""
    packetin = None
    bw = None
    pkt_z = False
    bw_z = False
    if window_last_flow is not None and len(window_last_flow) > max(
        PACKET_RATE_FEATURE_INDEX, CHANNEL_OCCUPANCY_FEATURE_INDEX
    ):
        packetin = float(window_last_flow[PACKET_RATE_FEATURE_INDEX])
        bw = float(window_last_flow[CHANNEL_OCCUPANCY_FEATURE_INDEX])
        pkt_z = True
        bw_z = True
    return encode_observation(
        anomaly_conf=p_attack,
        recon_error=reconstruction_error,
        packetin_rate=packetin,
        bw_util=bw,
        switch_cpu=None,
        packetin_rate_is_zscore=pkt_z,
        bw_util_is_zscore=bw_z,
        recon_error_is_raw_mse=True,
    )
