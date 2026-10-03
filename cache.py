import time
import threading
import logging
from typing import Any, Callable, Hashable, Optional

logger = logging.getLogger(__name__)


class TTLCache:
    def __init__(self, ttl_seconds: int, loader: Callable[[], Any], name: str = "cache"):
        self._ttl = ttl_seconds
        self._loader = loader
        self._name = name
        self._value: Optional[Any] = None
        self._loaded_at: float = 0.0
        self._lock = threading.RLock()

    def _is_stale(self) -> bool:
        return (time.time() - self._loaded_at) > self._ttl

    def get(self, force_refresh: bool = False) -> Any:
        if not force_refresh and self._value is not None and not self._is_stale():
            return self._value

        with self._lock:
            if not force_refresh and self._value is not None and not self._is_stale():
                return self._value

            logger.info("Refreshing cache '%s' (forced=%s)", self._name, force_refresh)
            self._value = self._loader()
            self._loaded_at = time.time()
            return self._value

    def invalidate(self):
        with self._lock:
            self._loaded_at = 0.0

    @property
    def age_seconds(self) -> float:
        return time.time() - self._loaded_at if self._loaded_at else -1

    @property
    def is_loaded(self) -> bool:
        return self._value is not None


class KeyedTTLCache:
    def __init__(self, ttl_seconds: int, max_entries: int = 256, name: str = "keyed-cache"):
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._name = name
        self._store: dict = {}
        self._lock = threading.RLock()

    def get(self, key: Hashable, loader: Callable[[], Any]) -> Any:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is not None and (now - entry[0]) <= self._ttl:
                return entry[1]

        value = loader()

        with self._lock:
            self._store[key] = (time.time(), value)
            overflow = len(self._store) - self._max_entries
            if overflow > 0:
                oldest = sorted(self._store.items(), key=lambda item: item[1][0])[:overflow]
                for stale_key, _ in oldest:
                    self._store.pop(stale_key, None)
        return value

    def clear(self):
        with self._lock:
            self._store.clear()
        logger.info("Cleared keyed cache '%s'", self._name)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)
