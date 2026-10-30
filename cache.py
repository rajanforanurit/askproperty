import time
import threading
import logging
from typing import Any, Callable, Hashable, Optional

logger = logging.getLogger(__name__)

# After a failed background refresh, wait this long before trying again (the stale value keeps being served).
_RETRY_AFTER_FAILURE_SECONDS = 30.0


class TTLCache:
    """Single-value cache with stale-while-revalidate.

    - The first load is synchronous (nothing to serve yet).
    - Once a value exists, an expired value is still returned immediately and ONE background thread
      refreshes it, so no user request ever waits for a TTL-expiry reload.
    - force_refresh=True and invalidate() keep their original meaning: the next caller reloads synchronously.

    Two locks: _state_lock is only ever held for a few instructions (flag checks), so readers never queue
    behind a slow load; _load_lock is held for the duration of a load and serialises loaders.
    """

    def __init__(self, ttl_seconds: int, loader: Callable[[], Any], name: str = "cache"):
        self._ttl = ttl_seconds
        self._loader = loader
        self._name = name
        self._value: Optional[Any] = None
        self._loaded_at: float = 0.0
        self._hard_stale = False
        self._refreshing = False
        self._retry_at: float = 0.0
        self._state_lock = threading.Lock()
        self._load_lock = threading.Lock()

    def _is_stale(self) -> bool:
        return (time.time() - self._loaded_at) > self._ttl

    def _load(self) -> Any:
        """Caller must hold _load_lock."""
        value = self._loader()
        with self._state_lock:
            self._value = value
            self._loaded_at = time.time()
            self._hard_stale = False
        return value

    def _refresh_in_background(self) -> None:
        def run():
            try:
                with self._load_lock:
                    logger.info("Background refresh of cache '%s'", self._name)
                    self._load()
            except Exception as e:  # keep serving the stale value
                logger.warning("Background refresh of cache '%s' failed: %s", self._name, e)
                with self._state_lock:
                    self._retry_at = time.time() + _RETRY_AFTER_FAILURE_SECONDS
            finally:
                with self._state_lock:
                    self._refreshing = False

        threading.Thread(target=run, name=f"refresh-{self._name}", daemon=True).start()

    def get(self, force_refresh: bool = False) -> Any:
        if not force_refresh:
            with self._state_lock:
                value = self._value
                usable = value is not None and not self._hard_stale
                stale = usable and self._is_stale()
                start_refresh = stale and not self._refreshing and time.time() >= self._retry_at
                if start_refresh:
                    self._refreshing = True
            if usable:
                if start_refresh:
                    self._refresh_in_background()
                return value

        with self._load_lock:
            if not force_refresh:
                with self._state_lock:
                    if self._value is not None and not self._hard_stale and not self._is_stale():
                        return self._value
            logger.info("Refreshing cache '%s' (forced=%s)", self._name, force_refresh)
            return self._load()

    def invalidate(self):
        with self._state_lock:
            self._loaded_at = 0.0
            self._hard_stale = True

    @property
    def age_seconds(self) -> float:
        return time.time() - self._loaded_at if self._loaded_at else -1

    @property
    def is_loaded(self) -> bool:
        return self._value is not None


class KeyedTTLCache:
    """Keyed cache with single-flight loading: concurrent callers for the same key share one load."""

    def __init__(self, ttl_seconds: int, max_entries: int = 256, name: str = "keyed-cache"):
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._name = name
        self._store: dict = {}
        self._key_locks: dict = {}
        self._lock = threading.RLock()

    def _fresh(self, key: Hashable):
        entry = self._store.get(key)
        if entry is not None and (time.time() - entry[0]) <= self._ttl:
            return entry
        return None

    def get(self, key: Hashable, loader: Callable[[], Any]) -> Any:
        with self._lock:
            entry = self._fresh(key)
            if entry is not None:
                return entry[1]
            key_lock = self._key_locks.setdefault(key, threading.Lock())

        with key_lock:
            with self._lock:
                entry = self._fresh(key)
                if entry is not None:
                    return entry[1]
            try:
                value = loader()
            except BaseException:
                with self._lock:
                    self._key_locks.pop(key, None)
                raise

            with self._lock:
                self._store[key] = (time.time(), value)
                self._key_locks.pop(key, None)
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
