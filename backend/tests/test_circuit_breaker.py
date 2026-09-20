"""In-job circuit breaker (decision D2: 5 consecutive failures per source per job)."""

from app.clients.circuit_breaker import CircuitBreaker
from app.config import settings


def test_starts_closed_with_configured_threshold():
    breaker = CircuitBreaker("openalex")
    assert breaker.threshold == settings.circuit_breaker_threshold == 5
    assert breaker.is_open is False
    assert breaker.consecutive_failures == 0


def test_opens_after_threshold_consecutive_failures():
    breaker = CircuitBreaker("openalex", threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_open is False
    breaker.record_failure()
    assert breaker.is_open is True


def test_success_resets_the_consecutive_counter():
    breaker = CircuitBreaker("openalex", threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    assert breaker.consecutive_failures == 0
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_open is False


def test_status_is_json_serialisable_and_reports_totals():
    import json

    breaker = CircuitBreaker("openalex", threshold=2)
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()

    status = breaker.status()
    assert status == {
        "source": "openalex",
        "open": True,
        "threshold": 2,
        "consecutive_failures": 2,
        "total_failures": 3,
        "total_successes": 1,
    }
    assert json.loads(json.dumps(status)) == status
