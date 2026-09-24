"""Redis Streams event bus with optional in-memory fallback for local unit tests.

Production-path benchmarks require use_redis=True against a live Redis 7.x
instance and exercise XADD + XREADGROUP (consumer group) rather than a Python
list append.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EventBus")

STREAM_DEFAULT = "security:telemetry:stream"
GROUP_DEFAULT = "security-cloud-cg"
CONSUMER_DEFAULT = "cloud-worker-1"


class EventBusClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        use_redis: bool = False,
        stream_name: str = STREAM_DEFAULT,
        group_name: str = GROUP_DEFAULT,
        consumer_name: str = CONSUMER_DEFAULT,
    ):
        self.use_redis = bool(use_redis)
        self.host = host
        self.port = port
        self.stream_name = stream_name
        self.group_name = group_name
        self.consumer_name = consumer_name
        self.redis_conn = None
        self._memory_streams: Dict[str, List[Dict[str, Any]]] = {}
        self._memory_offsets: Dict[str, int] = {}

        if self.use_redis:
            self._connect_redis()

    def _connect_redis(self) -> None:
        try:
            import redis

            self.redis_conn = redis.Redis(
                host=self.host, port=self.port, decode_responses=True, socket_connect_timeout=2.0
            )
            self.redis_conn.ping()
            self._ensure_consumer_group()
            logger.info("Connected to Redis at %s:%s (stream=%s)", self.host, self.port, self.stream_name)
        except Exception as e:
            raise RuntimeError(
                f"Redis required but unavailable at {self.host}:{self.port}: {e}. "
                "Start Redis 7.x (e.g. docker run -p 6379:6379 redis:7.2.4) before running "
                "latency/ablation benchmarks that claim Redis Streams."
            ) from e

    def _ensure_consumer_group(self) -> None:
        assert self.redis_conn is not None
        try:
            self.redis_conn.xgroup_create(self.stream_name, self.group_name, id="0", mkstream=True)
        except Exception as e:
            # BUSYGROUP means the group already exists — expected on repeated runs.
            if "BUSYGROUP" not in str(e).upper():
                raise

    def reset_stream(self) -> None:
        """Drop stream contents so a benchmark starts from a clean slate."""
        if self.use_redis and self.redis_conn:
            self.redis_conn.delete(self.stream_name)
            self._ensure_consumer_group()
        else:
            self._memory_streams[self.stream_name] = []
            self._memory_offsets[self.stream_name] = 0

    def publish(self, message_or_stream, message: Optional[Dict[str, Any]] = None) -> str:
        """Publish to a stream.

        Accepts either ``publish(message)`` (default stream) or the legacy
        ``publish(stream_name, message)`` used by the edge prototype.
        """
        if message is None:
            stream = self.stream_name
            payload = message_or_stream
        else:
            stream = message_or_stream
            payload = message
        payload_str = json.dumps(payload)
        if self.use_redis and self.redis_conn:
            msg_id = self.redis_conn.xadd(stream, {"payload": payload_str})
            return str(msg_id)
        if stream not in self._memory_streams:
            self._memory_streams[stream] = []
        msg_id = f"mem-{len(self._memory_streams[stream]) + 1}"
        self._memory_streams[stream].append({"id": msg_id, "payload": payload_str})
        return msg_id

    def consume_group(
        self,
        count: int = 1,
        block_ms: int = 1000,
        stream_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Read via XREADGROUP and XACK (Redis path) or in-memory cursor (fallback)."""
        stream = stream_name or self.stream_name
        results: List[Dict[str, Any]] = []
        if self.use_redis and self.redis_conn:
            entries = self.redis_conn.xreadgroup(
                groupname=self.group_name,
                consumername=self.consumer_name,
                streams={stream: ">"},
                count=count,
                block=block_ms,
            )
            if not entries:
                return results
            for _s_name, msgs in entries:
                for msg_id, data in msgs:
                    results.append({"id": msg_id, "payload": json.loads(data["payload"])})
                    self.redis_conn.xack(stream, self.group_name, msg_id)
            return results

        buf = self._memory_streams.get(stream, [])
        offset = self._memory_offsets.get(stream, 0)
        chunk = buf[offset : offset + count]
        self._memory_offsets[stream] = offset + len(chunk)
        for item in chunk:
            results.append({"id": item["id"], "payload": json.loads(item["payload"])})
        return results

    def publish_and_consume(self, message: Dict[str, Any], block_ms: int = 1000) -> Dict[str, Any]:
        """One-shot edge publish → cloud consumer-group read (includes transport)."""
        self.publish(message)
        got = self.consume_group(count=1, block_ms=block_ms)
        if not got:
            raise TimeoutError(f"No message consumed from {self.stream_name} within {block_ms} ms")
        return got[0]

    # Back-compat helpers used by older prototype code.
    def consume(self, stream_name: str, count: int = 10) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if self.use_redis and self.redis_conn:
            entries = self.redis_conn.xread({stream_name: "0-0"}, count=count)
            for _s_name, msgs in entries:
                for msg_id, data in msgs:
                    results.append({"id": msg_id, "payload": json.loads(data["payload"])})
        else:
            for item in self._memory_streams.get(stream_name, [])[:count]:
                results.append({"id": item["id"], "payload": json.loads(item["payload"])})
        return results


def require_redis(host: str = "localhost", port: int = 6379, retries: int = 20, delay_s: float = 0.5) -> EventBusClient:
    """Connect to Redis with retries; fail loudly if unavailable (no silent fallback)."""
    last_err: Optional[Exception] = None
    for _ in range(retries):
        try:
            client = EventBusClient(host=host, port=port, use_redis=True)
            client.reset_stream()
            return client
        except Exception as e:
            last_err = e
            time.sleep(delay_s)
    raise RuntimeError(f"Could not connect to Redis at {host}:{port}: {last_err}")
