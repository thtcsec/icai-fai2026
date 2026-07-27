"""
redis-stream / redis_client.py - Async Redis Stream Pub/Sub Consumer & Producer
"""

import json
from typing import Dict, Any, List

class RedisStreamClient:
    STREAM_KEY = "security:telemetry:stream"
    CONSUMER_GROUP = "soc_analytics_group"

    def __init__(self, host: str = "localhost", port: int = 6379):
        self.host = host
        self.port = port

    def format_event_for_stream(self, event_data: Dict[str, Any]) -> Dict[str, str]:
        return {"payload": json.dumps(event_data)}

    def parse_stream_message(self, message_payload: str) -> Dict[str, Any]:
        return json.loads(message_payload)
