"""HMAC-SHA256 principal pseudonymization for cloud transport.

Edge lookups may use cleartext addresses against a local identity table.
Anything published onto Redis (or otherwise leaving the edge trust boundary)
must carry only ``principal_id = hex(HMAC-SHA256(secret, identifier))``.

The deployment secret is taken from ``ICAI_PSEUDONYM_SECRET`` when set; otherwise
a deterministic development secret is used so benchmarks remain reproducible.
Production deployments must rotate a per-site secret.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Optional

_ENV_KEY = "ICAI_PSEUDONYM_SECRET"
_DEV_SECRET = b"icai-fai2026-dev-pseudonym-secret-v1"


def deployment_secret() -> bytes:
    raw = os.environ.get(_ENV_KEY)
    if raw:
        return raw.encode("utf-8")
    return _DEV_SECRET


def principal_id(identifier: str, secret: Optional[bytes] = None) -> str:
    """Return a hex HMAC-SHA256 principal id for ``identifier`` (IP, MAC, …)."""
    key = secret if secret is not None else deployment_secret()
    digest = hmac.new(key, identifier.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest


def assert_no_raw_endpoint(payload: dict) -> None:
    """Raise if a cloud-bound payload still contains cleartext endpoint fields."""
    forbidden = ("src_ip", "dst_ip", "mac", "mac_address", "ip")
    present = [k for k in forbidden if k in payload and payload[k] not in (None, "", "UNKNOWN")]
    if present:
        raise ValueError(f"Cloud payload must not include cleartext endpoints: {present}")
