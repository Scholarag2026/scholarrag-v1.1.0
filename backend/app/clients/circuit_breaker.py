"""In-job circuit breaker for external sources (decision D2).

Counts *consecutive* failures for one source inside one job. After
``settings.circuit_breaker_threshold`` consecutive failures the breaker opens and the
caller skips the remaining calls to that source, recording the degradation in the job
result instead of re-paying the full retry budget for every remaining item.

Deliberately not ``pybreaker``: state must be per job (so the next job starts clean),
and ``pybreaker`` is being removed from requirements.

Usage::

    breaker = CircuitBreaker("openalex")
    for paper in papers:
        if breaker.is_open:
            skipped += 1
            continue
        try:
            data = await client.get_work(paper.doi)
        except httpx.HTTPError:
            breaker.record_failure()
            continue
        breaker.record_success()
    result["degradation"] = breaker.status()
"""

from __future__ import annotations

from app.config import settings


class CircuitBreaker:
    """Tracks consecutive failures for one source within one job."""

    def __init__(self, source: str, threshold: int | None = None) -> None:
        self.source = source
        self.threshold = (
            threshold if threshold is not None else settings.circuit_breaker_threshold
        )
        self.consecutive_failures = 0
        self.total_failures = 0
        self.total_successes = 0

    @property
    def is_open(self) -> bool:
        """True when the breaker has tripped and remaining calls should be skipped."""
        return self.consecutive_failures >= self.threshold

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.total_successes += 1

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.total_failures += 1

    def status(self) -> dict:
        """JSON-serialisable summary for ``AnalysisJob.result``."""
        return {
            "source": self.source,
            "open": self.is_open,
            "threshold": self.threshold,
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
        }
