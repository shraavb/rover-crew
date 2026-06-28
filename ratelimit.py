"""Header-aware rate limiter for Cerebras calls.

Cerebras enforces ~requests/min. Hitting the cap returns HTTP 429, which makes
the control loop choppy or stalls it. This throttles two ways:

  * proactively -- a token-bucket spaced from MAX_RPM, so calls never burst
    past the sustainable rate in the first place.
  * reactively  -- on a 429, honor the server's Retry-After / reset hint and
    push the next slot out, so we recover smoothly instead of hammering.

Single process, multiple control loops could share one limiter, so it's
thread-safe.
"""
import threading
import time

import config


class RateLimiter:
    def __init__(self, rpm: float):
        self.min_interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        """Block until the next request slot is free, then claim it."""
        with self._lock:
            now = time.time()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.time()
            self._next_allowed = now + self.min_interval

    def penalize(self, retry_after: float) -> None:
        """Server said slow down: don't allow another call for retry_after sec."""
        with self._lock:
            self._next_allowed = max(self._next_allowed, time.time() + retry_after)


limiter = RateLimiter(config.MAX_RPM)
