"""Redis Stream / In-Memory Event Bus Interface for Event-Driven Communication."""
import json
import logging
from typing import Callable, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EventBus")

class EventBusClient:
    def __init__(self, host: str = "localhost", port: int = 6379, use_redis: bool = False):
        self.use_redis = use_redis
        self.host = host
        self.port = port
        self.redis_conn = None
        self._memory_streams: Dict[str, List[Dict[str, Any]]] = {}

        if self.use_redis:
            try:
                import redis
                self.redis_conn = redis.Redis(host=self.host, port=self.port, decode_responses=True)
                self.redis_conn.ping()
                logger.info(f"Connected to Redis at {host}:{port}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis ({e}). Falling back to In-Memory Event Bus.")
                self.use_redis = False

    def publish(self, stream_name: str, message: Dict[str, Any]) -> str:
        payload_str = json.dumps(message)
        if self.use_redis and self.redis_conn:
            msg_id = self.redis_conn.xadd(stream_name, {"payload": payload_str})
            return msg_id
        else:
            if stream_name not in self._memory_streams:
                self._memory_streams[stream_name] = []
            msg_id = f"mem-{len(self._memory_streams[stream_name]) + 1}"
            self._memory_streams[stream_name].append({"id": msg_id, "payload": payload_str})
            return msg_id

    def consume(self, stream_name: str, count: int = 10) -> List[Dict[str, Any]]:
        results = []
        if self.use_redis and self.redis_conn:
            entries = self.redis_conn.xread({stream_name: "0-0"}, count=count)
            for s_name, msgs in entries:
                for msg_id, data in msgs:
                    results.append({"id": msg_id, "payload": json.loads(data["payload"])})
        else:
            entries = self._memory_streams.get(stream_name, [])[:count]
            for item in entries:
                results.append({"id": item["id"], "payload": json.loads(item["payload"])})
        return results
