"""Hard failures on unverifiable claims, exit code 2, poll retries, absolute tolerance
slack and the Crossref metadata fallback. Offline like the rest."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import run_demo
from run_demo import (
    ApiClient,
    ApiError,
    DemoError,
    build_paper_data,
    check_claim_citations,
    claim_problems,
    openalex_resolver,
    poll_task,
    prepare_claims,
)
from test_run_demo import API, FIXTURES, FakeClock, FakeServer, fake_resolver, make_runner


class FlakyTransport:
    """Fails the first ``failures`` GET /tasks/{id} polls, then defers to FakeServer."""

    def __init__(self, server: FakeServer, failures: int, *, mode: str = "connect"):
        self.server = server
        self.remaining = failures
        self.mode = mode

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        is_poll = "/tasks/" in path and not path.endswith("screening-record")
        if is_poll and request.method == "GET" and self.remaining > 0:
            self.remaining -= 1
            if self.mode == "connect":
                raise httpx.ConnectError("reset by peer", request=request)
            return httpx.Response(503, json={"detail": "upstream unavailable"})
        return self.server.handle(request)


@pytest.fixture
def protocol():
    return json.loads((FIXTURES / "protocol.json").read_text(encoding="utf-8"))


@pytest.fixture
def seeds():
    return json.loads((FIXTURES / "seed_dois.json").read_text(encoding="utf-8"))


def _poll(client: ApiClient, task_id: str, timeout_s: float = 600) -> tuple[dict, list[str]]:
    clock = FakeClock()
    logs: list[str] = []
    task = poll_task(
        client,
        task_id,
        label="t",
        timeout_s=timeout_s,
        sleep=clock.sleep,
        clock=clock,
        log=logs.append,
    )
    return task, logs


def _cli(tmp_path: Path, server: FakeServer, *extra: str) -> int:
    argv = [
        "--api-url",
        API,
        "--protocol",
        str(FIXTURES / "protocol.json"),
        "--seeds",
        str(FIXTURES / "seed_dois.json"),
        "--expected",
        str(tmp_path / "no-expected"),
        "--output",
        str(tmp_path / "run"),
        "--timeout",
        "60",
        *extra,
    ]
    return run_demo.main(
        argv,
        client_factory=lambda url: ApiClient(url, transport=httpx.MockTransport(server.handle)),
    )


# -- poll retries -------------------------------------------------------------


@pytest.mark.parametrize("mode", ["connect", "503"])
def test_poll_task_retries_transient_failures(mode):
    server = FakeServer(polls_before_done=1)
    task_id = server._new_task("writing", {"content": "x"})
    client = ApiClient(API, transport=httpx.MockTransport(FlakyTransport(server, 3, mode=mode)))
    client.token = "tok"
    task, logs = _poll(client, task_id)
    assert task["status"] == "completed"
    assert sum("poll failed" in line for line in logs) == 3


def test_poll_task_gives_up_after_max_retries():
    server = FakeServer(polls_before_done=1)
    task_id = server._new_task("writing", {"content": "x"})
    flaky = FlakyTransport(server, run_demo.POLL_MAX_RETRIES + 1)
    client = ApiClient(API, transport=httpx.MockTransport(flaky))
    client.token = "tok"
    with pytest.raises(ApiError, match="could not reach"):
        _poll(client, task_id)


def test_poll_task_does_not_retry_client_errors():
    server = FakeServer(polls_before_done=1)
    client = ApiClient(API, transport=httpx.MockTransport(server.handle))
    client.token = "tok"
    with pytest.raises(ApiError) as info:
        _poll(client, "does-not-exist")
    assert info.value.status_code == 404


# -- claim citation checks ----------------------------------------------------


def test_unparseable_citation_fails_validation_before_any_call(protocol, seeds, tmp_path):
    protocol["claims"][0]["text"] = "Direct corrections lasted (Bonilla López et al., 2018)."
    server = FakeServer()
    runner, _, _ = make_runner(server, protocol, seeds, tmp_path)
    with pytest.raises(DemoError, match="cannot parse"):
        runner.run()
    assert server.calls == []


def test_surname_mismatch_fails_before_the_llm_steps(protocol, seeds, tmp_path):
    protocol["claims"][0]["text"] = "A finding (Alpha, 2016)."  # demo-001 resolves to Beta
    server = FakeServer()
    runner, _, _ = make_runner(server, protocol, seeds, tmp_path)
    with pytest.raises(DemoError, match="citation key 'alpha_2016' differs .*'beta_2016'"):
        runner.run()
    paths = [p for _, p, _ in server.calls]
    assert not any("acquire-full-texts" in p or "/drafts" in p for p in paths)
    assert runner.state.steps[-1].name == "check claim citations"
    assert runner.state.steps[-1].status == "failed"


def test_unresolved_authors_fail_the_citation_check(protocol, seeds, tmp_path):
    server = FakeServer()
    runner, _, _ = make_runner(server, protocol, seeds, tmp_path, resolver=lambda doi: None)
    with pytest.raises(DemoError, match="no authors/year resolved"):
        runner.run()


def test_check_claim_citations_reports_every_problem():
    claims = prepare_claims(
        {
            "claims": [
                {"id": "a", "text": "X (Smith, 2020).", "source_doi": "d1",
                 "expected_status": "verified"},
                {"id": "b", "text": "Y (Jones, 2021).", "source_doi": "d2",
                 "expected_status": "verified"},
            ]
        },
        {
            "d1": {"authors": [{"name": "Ann Smith"}], "year": 2020},
            "d2": {"authors": [{"name": "Bo Brown"}], "year": 2021},
        },
    )
    assert claims[0]["citation_key"] == "smith_2020" == claims[0]["paper_key"]
    with pytest.raises(DemoError) as info:
        check_claim_citations(claims)
    assert "claim b" in str(info.value) and "claim a" not in str(info.value)


# -- exit code 2 --------------------------------------------------------------


def test_main_returns_2_when_a_claim_misses_its_expected_status(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_demo, "openalex_resolver", lambda *a, **k: fake_resolver)
    monkeypatch.setattr(run_demo.time, "sleep", lambda s: None)
    server = FakeServer()
    server.claim_statuses = ["unsupported", "unsupported"]
    assert _cli(tmp_path, server) == 2
    err = capsys.readouterr().err
    assert "DEMO CLAIM MISMATCH" in err and "claim supported-1" in err
    saved = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert saved["verification"]["claims"][0]["matches_expected"] is False


def test_main_returns_2_when_the_report_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(run_demo, "openalex_resolver", lambda *a, **k: fake_resolver)
    monkeypatch.setattr(run_demo.time, "sleep", lambda s: None)
    server = FakeServer()
    server.empty_report = True
    assert _cli(tmp_path, server) == 2


def test_claim_mismatch_takes_precedence_over_strict_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(run_demo, "openalex_resolver", lambda *a, **k: fake_resolver)
    monkeypatch.setattr(run_demo.time, "sleep", lambda s: None)
    expected_dir = tmp_path / "expected"
    expected_dir.mkdir()
    (expected_dir / "summary.json").write_text(
        json.dumps({"screening": {"flow": {"included": 100}}}), encoding="utf-8"
    )
    server = FakeServer()
    server.claim_statuses = ["verified", "verified"]
    assert _cli(tmp_path, server, "--expected", str(expected_dir), "--strict") == 2


def test_claim_problems_helper():
    assert claim_problems({"verification": None}) == [
        "the claim-verification report contains no claims at all"
    ]
    summary = {
        "verification": {
            "claims_in_report": 2,
            "claims": [
                {"id": "s", "status": "verified", "expected_status": "verified",
                 "matches_expected": True, "final_status": "verified"},
                {"id": "u", "status": "nuance", "expected_status": "unsupported",
                 "matches_expected": False},
            ],
        }
    }
    assert claim_problems(summary) == ["claim u: status 'nuance', expected 'unsupported'"]


# -- tolerances ---------------------------------------------------------------


def test_within_tolerance_has_absolute_slack_for_small_and_zero_counts():
    assert run_demo._within_tolerance(0, 0, 0.3)
    assert run_demo._within_tolerance(0, 2, 0.3)
    assert not run_demo._within_tolerance(0, 3, 0.3)
    assert run_demo._within_tolerance(3, 5, 0.3)  # 30 % of 3 is less than 2 records
    assert not run_demo._within_tolerance(3, 6, 0.3)
    assert run_demo._within_tolerance(100, 130, 0.3)
    assert not run_demo._within_tolerance(100, 131, 0.3)


# -- metadata resolver --------------------------------------------------------


def test_resolver_falls_back_to_crossref_when_openalex_is_blocked():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.host == "api.openalex.org":
            return httpx.Response(429, json={"error": "Rate limit exceeded"})
        assert request.url.host == "api.crossref.org"
        return httpx.Response(
            200,
            json={
                "message": {
                    "title": ["A title"],
                    "author": [{"given": "Zhicheng", "family": "Mao"}, {"family": "Lee"}],
                    "container-title": ["Language Teaching"],
                    "ISSN": ["0261-4448"],
                    "published-print": {"date-parts": [[2024, 1]]},
                    "is-referenced-by-count": 12,
                }
            },
        )

    sleeps: list[float] = []
    resolve = openalex_resolver(
        httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleeps.append,
        env={"OPENALEX_EMAIL": "reviewer@example.org"},
    )
    work = resolve("10.1017/s0261444823000393")
    assert work["metadata_source"] == "crossref"
    assert work["authorships"][0]["author"]["display_name"] == "Zhicheng Mao"
    assert work["publication_year"] == 2024 and work["cited_by_count"] == 12
    openalex_calls = [u for u in seen if "openalex.org" in u]
    assert len(openalex_calls) == 3 and len(sleeps) == 3  # three tries with backoff
    assert all("mailto=reviewer%40example.org" in u for u in openalex_calls)
    paper = build_paper_data({"doi": "10.1017/s0261444823000393", "year": 2024}, work)
    assert run_demo.first_author_surname(paper) == "Mao"
    assert paper["journal_issn"] == "0261-4448"


def test_resolver_returns_openalex_work_and_sends_nothing_personal_by_default():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "mailto" not in request.url.params and "api_key" not in request.url.params
        return httpx.Response(200, json=fake_resolver("10.5555/demo-001"))

    resolve = openalex_resolver(
        httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None, env={}
    )
    assert resolve("10.5555/demo-001")["authorships"][0]["author"]["display_name"] == "Ann Beta"
