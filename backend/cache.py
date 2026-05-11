import json
import time
from typing import Any, Optional

try:
    import redis
    _redis_available = True
except ImportError:
    _redis_available = False

_memory_cache: dict[str, tuple[Any, float]] = {}
_TTL = 600  # 10 minutes

_redis_client: Optional[Any] = None


def init_cache(redis_url: Optional[str] = None) -> None:
    global _redis_client
    if redis_url and _redis_available:
        try:
            _redis_client = redis.from_url(redis_url, decode_responses=True)
            _redis_client.ping()
        except Exception:
            _redis_client = None


def get(key: str) -> Optional[Any]:
    if _redis_client:
        val = _redis_client.get(key)
        return json.loads(val) if val else None
    entry = _memory_cache.get(key)
    if entry and time.time() < entry[1]:
        return entry[0]
    return None


def set(key: str, value: Any, ttl: int = _TTL) -> None:
    if _redis_client:
        _redis_client.setex(key, ttl, json.dumps(value))
        return
    _memory_cache[key] = (value, time.time() + ttl)


def delete(key: str) -> None:
    if _redis_client:
        _redis_client.delete(key)
        return
    _memory_cache.pop(key, None)
