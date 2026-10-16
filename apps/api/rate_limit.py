"""In-memory rate limiter for API keys."""

import time
from collections import defaultdict

from fastapi import HTTPException, status


class InMemoryRateLimiter:
    """Sliding-window in-memory rate limiter per API key."""

    def __init__(self, requests_per_minute: int = 30) -> None:
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60.0
        self.history: dict[str, list[float]] = defaultdict(list)

    def check_rate_limit(self, key: str) -> None:
        """
        Check if key has exceeded the limit in the current 60-second window.
        Raises HTTPException 429 if exceeded.
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        # Prune old timestamps
        timestamps = [t for t in self.history[key] if t > cutoff]
        self.history[key] = timestamps

        if len(timestamps) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "code": "RATE_LIMITED",
                    "message": f"Quá giới hạn tần suất yêu cầu ({self.requests_per_minute} req/min). Hãy thử lại sau.",
                },
            )

        self.history[key].append(now)


# Global default instance
_limiter: InMemoryRateLimiter | None = None


def get_rate_limiter(rpm: int = 30) -> InMemoryRateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = InMemoryRateLimiter(requests_per_minute=rpm)
    return _limiter
