"""Rate limiting and retry mechanisms for LLM calls."""

import time
from threading import Lock
from typing import Any, Callable

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from legalrag.core.errors import RateLimitedError


class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter enforcing client-side RPM."""

    def __init__(self, rpm: int = 10) -> None:
        self.rpm = max(1, rpm)
        self.capacity = float(self.rpm)
        self.tokens = float(self.rpm)
        self.fill_rate = float(self.rpm) / 60.0  # tokens per second
        self.last_update = time.monotonic()
        self.lock = Lock()

    def acquire(self, tokens: float = 1.0, wait: bool = True) -> bool:
        """Acquire token, waiting if necessary when wait=True."""
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.last_update = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                if not wait:
                    return False

                sleep_time = (tokens - self.tokens) / self.fill_rate

            time.sleep(min(sleep_time, 1.0))


def llm_retry_decorator(max_attempts: int = 5) -> Callable[[Any], Any]:
    """Exponential backoff retry decorator with jitter for transient errors."""
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=1.0, max=60.0),
        retry=retry_if_exception_type((RateLimitedError, TimeoutError, ConnectionError)),
    )
