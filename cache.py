import time
import threading
import logging
from typing import Any, Callable, Optional

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
