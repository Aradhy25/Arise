"""Simple in-memory rate limiter for public endpoints."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException


class SimpleRateLimiter:
    def __init__(self, max_calls: int = 20, window_sec: int = 3600):
        self.max_calls = max_calls
        self.window = window_sec
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.time()
        q = self._hits[key]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.max_calls:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Sign in for higher limits.",
            )
        q.append(now)


public_limiter = SimpleRateLimiter(max_calls=30, window_sec=3600)
