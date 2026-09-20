"""Tests for demo/run_demo.py against an in-process fake of the ScholarRAG API.

No network, no database, no LLM: ``httpx.MockTransport`` routes every request to
``FakeServer``, which records the calls and replays the backend's response shapes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
import run_demo
from run_demo import (
    ApiClient,
    ApiError,
    DemoError,
    DemoRunner,
    add_seed_papers,
    append_claim_paragraphs,
    build_draft_content,
    build_paper_data,
    check_claim_text_in_draft,
    claim_problems,
    compare_with_expected,
    ensure_citation,
    fetch_draft,
    format_summary_table,
    match_claim_results,
    poll_task,
    register_or_login,
    start_generation,
    start_smart_search,
    terminate_sentence,
)

FIXTURES = Path(__file__).parent / "fixtures"
API = "http://fake.local/api/v1"

USER_ID = "11111111-1111-1111-1111-111111111111"
PROJECT_ID = "22222222-2222-2222-2222-222222222222"
DRAFT_ID = "33333333-3333-3333-3333-333333333333"
FIXTURE_PAPER_ID = "44444444-4444-4444-4444-444444444444"
#: The fixture chunk `FakeServer`'s own `/papers/{id}/fulltext-chunks/{index}` route
#: returns: its text contains `make_claim_report`'s own
#: verified-row evidence quote verbatim, so `trim_source_passage` locates it exactly.
FIXTURE_CHUNK_TEXT = (
    "The intervention produced a gain in vocabulary retention among enrolled learners."
)

SCREENING_RESULT = {
    "papers": [{"title": "Included one", "doi": "10.1/x", "screening_reason": "fits"}],
    "elapsed_minutes": 3.2,
    "stage1_applied": False,
    "criteria": {
        "query": "How is X used in applied linguistics classrooms?",
        "inclusion_criteria": ["empirical study", "language learning context"],
        "exclusion_criteria": ["non-English", "conference abstract only"],
        "wos_filter": "off",
    },
    "flow": {
        "identified": 100,
        "duplicates_removed": 5,
        "stage1_screened": 0,
        "stage1_excluded": 0,
        "stage2_screened": 95,
        "stage2_excluded": 80,
        "unscreened": 0,
        "included": 15,
        "rounds": 2,
        "stop_reason": "no_new_included",
    },
    "provenance": {
        "screener": {
            "model_configured": "deepseek-chat",
            "model_reported": ["deepseek-v4-flash"],
            "temperature": 0.0,
            "prompt_version": "sha256:abc",
            "calls": 10,
            "input_tokens": 1000,
            "output_tokens": 200,
        },
        "query_generator": {"model_configured": "deepseek-chat", "prompt_version": "sha256:q"},
        # One entry per round, read by --replay-queries to rebuild a
        # queries_override and by compare_with_expected's replay check.
        "rounds": [
            {
                "round": 1,
                "queries": [
                    {"query": "round one query a", "returned": 40, "new_unique": 38},
                    {"query": "round one query b", "returned": 30, "new_unique": 25},
                ],
                "screened": 63,
                "included_new": 8,
                "needs_review_new": 2,
                "replayed": False,
            },
            {
                "round": 2,
                "queries": [{"query": "round two query", "returned": 20, "new_unique": 15}],
                "screened": 15,
                "included_new": 7,
                "needs_review_new": 0,
                "replayed": False,
            },
        ],
    },
}

WRITING_RESULT = {
    "section_type": "literature_review",
    "content": "First paragraph (Alpha, 2021).\n\nSecond paragraph (Beta, 2022).",
    "papers_used": 12,
    "citation_audit": {
        "matched": ["Alpha, 2021", "Beta, 2022"],
        "unmatched": [],
        "needs_citation_flags": 1,
        "total": 2,
    },
    "provenance": {
        "agent": "writing",
        "model_configured": "deepseek-chat",
        "model_reported": "deepseek-v4-flash",
        "temperature": 0.7,
        "prompt_version": "sha256:w",
        "input_tokens": 3000,
        "output_tokens": 700,
    },
    # The write-result shape carries an ``uncited_
    # sentences`` field, every generated sentence classified one way or the other, so
    # the delivered-text gate's rule 1 (`_check_delivered_text_for_section`) is
    # actually exercised on the default scenario, not silently skipped for lack of the
    # field at all (the "older shape, not checked" notice path a fixture with no such
    # field takes). Both generated sentences are tagged "framing", not linked as a
    # citation: this fixture's own mocked final report (`make_claim_report`, built from
    # `FakeServer._claim_texts`) only ever verifies the FIXTURE claims
    # `_append_fixture_claims` appends, never these two -- exactly the shape a citation
    # rule 1 would otherwise report as a genuine coverage gap (a cited-looking sentence
    # never confirmed by any final-report row); a test that wants to exercise a real
    # citation link instead overrides this key (see e.g. ``test_write_section_attaches_
    # the_writer_citation_link_map_end_to_end``, below).
    "uncited_sentences": [
        {
            "paragraph_index": 0,
            "sentence": "First paragraph (Alpha, 2021).",
            "tag": "framing",
        },
        {
            "paragraph_index": 1,
            "sentence": "Second paragraph (Beta, 2022).",
            "tag": "framing",
        },
    ],
    # The gated write loop's own internal statistics and the
    # "ONE report of the final text" (every row verified) -- both claims verified on
    # the first pass, no revision needed.
    "loop_stats": {
        "length_regenerated": False,
        "loop_revised": False,
        "first_pass_verified_rate": 1.0,
        "survival_rate": 1.0,
        "sentences_removed_unverified": 0,
        "sentences_removed_uncited_finding": 0,
        "sentences_removed_no_full_text": 0,
        "citations_dropped_no_full_text": 0,
        "needs_citation_markers_removed": 0,
    },
    "claim_report": {"verifications": [], "verified_count": 2},
    "metrics": {
        "analysis": {
            "papers": 0, "papers_attempted": 0, "papers_failed": 0,
            "evidence_items": 0, "rejected_items": 0, "verbatim_rate": 1.0,
        }
    },
}


def make_claim_report(
    draft_id: str,
    claim_texts: list[str],
    statuses: list[str] | None = None,
    citation_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    statuses = statuses or ["verified", "unsupported"]
    verifications = []
    for text, status in zip(claim_texts, statuses):
        verifications.append(
            {
                "claim_text": text,
                "claim_sentence": text,
                "citation": "(Beta, 2016)",
                "paper_id": FIXTURE_PAPER_ID,
                "paper_doi": "10.5555/demo-001",
                "paper_title": "Fixture paper one",
                "status": status,
                "evidence_quote": "a gain in vocabulary retention"
                if status == "verified"
                else None,
                "evidence_quotes": ["a gain in vocabulary retention"]
                if status == "verified"
                else [],
                # `app.services.fulltext.
                # _find_evidence_location`'s own "chunk N (section)" format, so
                # `build_delivered_evidence_rows` (against `FakeServer`'s own
                # `/papers/{id}/fulltext-chunks/{index}` route below) has something real
                # to parse. Computed only for a claim that actually located a quote, the
                # same way the real service only ever sets it after relocation.
                "evidence_location": "chunk 1 (results)" if status == "verified" else None,
                "explanation": "because",
                "suggested_revision": None,
                "model_reported": "deepseek-v4-flash",
            }
        )
    return {
        "draft_id": draft_id,
        "verifications": verifications,
        "verified_count": 1,
        "unsupported_count": 1,
        "nuance_count": 0,
        "abstract_only_count": 0,
        "error_count": 0,
        "needs_rewrite": 0,
        "needs_removal": 1,
        "provenance": {
            "agent": "claim_verification",
            "model_configured": "deepseek-chat",
            "model_reported": ["deepseek-v4-flash"],
            "temperature": 0.0,
            "prompt_version": "sha256:v",
            # Recorded next to prompt_version so
            # the demo report can state which guard set and which relocation rule produced
            # this run's claim statuses.
            "guard_digest": "570a5b663e6140da",
            "quote_relocation_version": "8b5baf4f236d2854",
            # Which version of the shared
            # call-relocate-guard[-call-again] policy produced this run's claim statuses.
            "verification_policy_version": "9c6cbaf5347e3965",
            "repair_prompt_version": "sha256:fakerepairver",
            "calls": 2,
            "input_tokens": 4000,
            "output_tokens": 300,
        },
        "full_text_coverage": 1.0,
        "citation_coverage": citation_coverage,
    }


class FakeServer:
    """Stateful stand-in for the API: records calls, drives tasks to completion."""

    def __init__(
        self,
        *,
        polls_before_done: int = 2,
        register_status: int = 201,
        expire_token_after: int | None = None,
    ):
        self.calls: list[tuple[str, str, Any]] = []
        self.tasks: dict[str, dict[str, Any]] = {}
        self.polls_before_done = polls_before_done
        self.register_status = register_status
        # The number of authenticated calls the *original* access
        # token ("tok") answers before it stops validating, simulating the 15-minute
        # JWT expiry firing mid-run; None (the default) means it never expires. A
        # refreshed token ("tok2", "tok3", ...) always validates -- one expiry cycle is
        # all any test here needs to exercise.
        self.expire_token_after = expire_token_after
        self._issued_tokens = ["tok"]
        self._calls_with_original_token = 0
        self.draft_content: dict[str, Any] | None = None
        self.writing_result: dict[str, Any] = WRITING_RESULT
        self.paper_status: int = 201
        # the first call's own result carries one abstract-only paper
        # (matching the shape the promoted 2026-09-12 baseline actually saw) so the
        # retry fires by default; the retry's own result heals it, mirroring the
        # backend's real idempotency (the 11 papers the first call already acquired
        # come back as ``already_acquired`` on the second call, not ``acquired``
        # again). A test exercising "nothing abstract-only, no retry" or "still
        # abstract-only after the retry" overrides either dict before calling
        # ``runner.run()``.
        self.fulltext_result: dict[str, Any] = {
            "acquired": 11,
            "abstract_only": 1,
            "already_acquired": 0,
        }
        self.fulltext_retry_result: dict[str, Any] = {
            "acquired": 1,
            "abstract_only": 0,
            "already_acquired": 11,
        }
        self.fulltext_acquire_calls: int = 0
        self.smart_search_status: int = 202
        self.task_outcome: str = "completed"
        self.claim_statuses: list[str] | None = None  # None -> verified, unsupported
        self.empty_report: bool = False
        self.citation_coverage: dict[str, Any] | None = None
        # The standalone verify-and-heal action's own last report, so a
        # later GET .../claim-verification replays exactly what healing computed
        # instead of recomputing against the now-healed (mutated) draft content.
        self._last_report: dict[str, Any] | None = None
        self._next_task = 0

    # -- helpers ------------------------------------------------------------

    def _new_task(self, job_type: str, result: Any) -> str:
        self._next_task += 1
        task_id = f"aaaaaaaa-0000-0000-0000-{self._next_task:012d}"
        self.tasks[task_id] = {"job_type": job_type, "result": result, "polls": 0}
        return task_id

    def _task_response(self, task_id: str) -> httpx.Response:
        task = self.tasks.get(task_id)
        if task is None:
            return httpx.Response(404, json={"detail": "Task not found"})
        task["polls"] += 1
        done = task["polls"] > self.polls_before_done
        status = self.task_outcome if done else "running"
        body = {
            "id": task_id,
            "job_type": task["job_type"],
            "status": status,
            "progress": 1.0 if done else 0.5,
            "progress_message": "done" if done else f"working {task['polls']}",
            "result": task["result"] if status == "completed" else None,
            "error": "boom" if status == "failed" else None,
            "created_at": "2026-09-02T00:00:00Z",
            "updated_at": "2026-09-02T00:00:00Z",
        }
        return httpx.Response(200, json=body)

    def _build_generated_draft_content(self) -> dict[str, Any]:
        """Simulate the write job's own gated loop: the backend
        verifies, finalizes and saves the AI-written section itself, before the demo
        script ever reads the draft back. One heading (tagged by section_type) plus
        one paragraph per blank-line block of ``self.writing_result["content"]``,
        carrying ``citationLinks`` AND ``uncitedSentences``,
        each grouped by ``paragraph_index`` -- the same shape
        ``app.services.writing._build_section_tiptap_nodes`` produces, so a demo test
        exercises the write job's own B1 attrs round trip, not only its ``citation_links``
        half."""
        text = self.writing_result.get("content") or ""
        blocks = [b for b in text.split("\n\n") if b.strip()]
        links_by_paragraph: dict[int, list[dict[str, Any]]] = {}
        for link in self.writing_result.get("citation_links") or []:
            idx = link.get("paragraph_index")
            if isinstance(idx, int):
                links_by_paragraph.setdefault(idx, []).append(
                    {
                        "sentence": link.get("sentence"),
                        "keys": link.get("keys") or [],
                        "citation_text": link.get("citation_text"),
                    }
                )
        uncited_by_paragraph: dict[int, list[dict[str, Any]]] = {}
        for entry in self.writing_result.get("uncited_sentences") or []:
            idx = entry.get("paragraph_index")
            if isinstance(idx, int):
                uncited_by_paragraph.setdefault(idx, []).append(
                    {
                        "sentence": entry.get("sentence"),
                        "tag": entry.get("tag"),
                        **({"unclassified": True} if entry.get("unclassified") else {}),
                    }
                )
        nodes: list[dict[str, Any]] = [
            {
                "type": "heading",
                "attrs": {
                    "level": 2,
                    "sectionType": self.writing_result.get("section_type"),
                },
                "content": [{"type": "text", "text": "Section"}],
            }
        ]
        for i, block in enumerate(blocks):
            node: dict[str, Any] = {
                "type": "paragraph", "content": [{"type": "text", "text": block}]
            }
            attrs: dict[str, Any] = {}
            if links_by_paragraph.get(i):
                attrs["citationLinks"] = links_by_paragraph[i]
            if uncited_by_paragraph.get(i):
                attrs["uncitedSentences"] = uncited_by_paragraph[i]
            if attrs:
                node["attrs"] = attrs
            nodes.append(node)
        return {"type": "doc", "content": nodes}

    def _claim_texts(self) -> list[str]:
        texts: list[str] = []
        for node in (self.draft_content or {}).get("content", []):
            if node.get("type") == "paragraph":
                texts.append("".join(c.get("text", "") for c in node.get("content", [])))
        return texts[-2:]

    def _report(self, draft_id: str) -> dict[str, Any]:
        texts = [] if self.empty_report else self._claim_texts()
        report = make_claim_report(draft_id, texts, self.claim_statuses, self.citation_coverage)
        if self.empty_report:
            report["verified_count"] = report["unsupported_count"] = 0
        return report

    def _heal_and_report(self, draft_id: str) -> dict[str, Any]:
        """Simulate ``app.services.fulltext.verify_and_heal_claims``: the same report
        `_report` builds, plus the three additive healing keys, with the saved draft
        actually mutated -- a paragraph whose own text was reported as anything but
        "verified" is removed from ``self.draft_content``, and every surviving
        paragraph has any ``[NEEDS CITATION]`` marker stripped, exactly as
        ``app.services.fulltext.finalize_draft_document`` does on a real run."""
        report = self._report(draft_id)
        removed = 0
        if not self.empty_report and self.draft_content is not None:
            statuses = self.claim_statuses or ["verified", "unsupported"]
            bad_texts = {
                v["claim_text"]
                for v, status in zip(report["verifications"], statuses)
                if status != "verified" and v.get("claim_text")
            }
            kept_nodes = []
            for node in self.draft_content.get("content", []):
                node_text = "".join(
                    c.get("text", "")
                    for c in node.get("content") or []
                    if c.get("type") == "text"
                )
                if node.get("type") == "paragraph" and any(
                    bad and bad in node_text for bad in bad_texts
                ):
                    removed += 1
                    continue
                if node.get("type") == "paragraph" and run_demo.NEEDS_CITATION_RE.search(
                    node_text
                ):
                    node = {
                        **node,
                        "content": [
                            {"type": "text", "text": run_demo.NEEDS_CITATION_RE.sub("", node_text)}
                        ],
                    }
                kept_nodes.append(node)
            self.draft_content = {**self.draft_content, "content": kept_nodes}
        final_verifications = [v for v in report["verifications"] if v["status"] == "verified"]
        report["healed"] = True
        report["finalize_stats"] = {"sentences_removed_unverified": removed}
        report["final_report"] = {
            "verifications": final_verifications,
            "verified_count": len(final_verifications),
        }
        self._last_report = report
        return report

    # -- router --------------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.replace("/api/v1", "", 1)
        body = json.loads(request.content) if request.content else None
        self.calls.append(
            (
                request.method,
                path + (f"?{request.url.query.decode()}" if request.url.query else ""),
                body,
            )
        )
        auth = request.headers.get("Authorization", "")
        presented = auth[len("Bearer "):] if auth.startswith("Bearer ") else None
        token_ok = presented in self._issued_tokens
        if token_ok and presented == self._issued_tokens[0] and self.expire_token_after is not None:
            self._calls_with_original_token += 1
            if self._calls_with_original_token > self.expire_token_after:
                token_ok = False

        if path == "/auth/refresh" and request.method == "POST":
            if (body or {}).get("refresh_token") != "r":
                return httpx.Response(401, json={"detail": "Invalid refresh token"})
            new_token = f"tok{len(self._issued_tokens) + 1}"
            self._issued_tokens.append(new_token)
            return httpx.Response(200, json={"access_token": new_token})

        if path == "/auth/register" and request.method == "POST":
            if self.register_status != 201:
                return httpx.Response(
                    self.register_status, json={"detail": "Email already registered"}
                )
            return httpx.Response(
                201, json={"access_token": "tok", "refresh_token": "r", "user": {"id": USER_ID}}
            )
        if path == "/auth/login" and request.method == "POST":
            return httpx.Response(
                200, json={"access_token": "tok", "refresh_token": "r", "user": {"id": USER_ID}}
            )
        if not token_ok:
            return httpx.Response(401, json={"detail": "Not authenticated"})
        if path == "/projects" and request.method == "POST":
            return httpx.Response(201, json={"id": PROJECT_ID, "title": body["title"]})
        if path == f"/projects/{PROJECT_ID}/smart-search" and request.method == "POST":
            if self.smart_search_status != 202:
                return httpx.Response(
                    self.smart_search_status, json={"detail": "Smart search already in progress"}
                )
            return httpx.Response(
                202, json={"task_id": self._new_task("smart_search", SCREENING_RESULT)}
            )
        if path.startswith("/tasks/") and path.endswith("/screening-record"):
            task_id = path.split("/")[2]
            fmt = request.url.params.get("format", "json")
            if fmt == "csv":
                return httpx.Response(
                    200,
                    text="outcome,stage,reason,title,doi,year,journal,journal_issn,openalex_id\n"
                    "included,llm,fits,Included one,10.1/x,,,,\n",
                    headers={"content-type": "text/csv"},
                )
            result = self.tasks[task_id]["result"]
            return httpx.Response(
                200,
                json={
                    "criteria": result["criteria"],
                    "flow": result["flow"],
                    "provenance": result["provenance"],
                    "records": [{"outcome": "included", "title": "Included one"}],
                },
            )
        if path.startswith("/tasks/") and request.method == "GET":
            return self._task_response(path.split("/")[2])
        if path == f"/projects/{PROJECT_ID}/papers" and request.method == "POST":
            if self.paper_status != 201:
                return httpx.Response(
                    self.paper_status, json={"detail": "Paper already in project"}
                )
            return httpx.Response(
                201, json={"id": "pp", "paper_id": "p", "paper": {"doi": body["paper_data"]["doi"]}}
            )
        if path == f"/projects/{PROJECT_ID}/acquire-full-texts" and request.method == "POST":
            self.fulltext_acquire_calls += 1
            result = (
                self.fulltext_result
                if self.fulltext_acquire_calls == 1
                else self.fulltext_retry_result
            )
            return httpx.Response(202, json={"task_id": self._new_task("fulltext_acquire", result)})
        if path == f"/projects/{PROJECT_ID}/drafts" and request.method == "POST":
            return httpx.Response(201, json={"id": DRAFT_ID, "title": body["title"]})
        if path == f"/drafts/{DRAFT_ID}/generate" and request.method == "POST":
            # The write job's own gated loop verifies, finalizes and
            # saves the section itself, before the client ever polls the task done.
            self.draft_content = self._build_generated_draft_content()
            return httpx.Response(
                200, json={"task_id": self._new_task("writing", self.writing_result)}
            )
        if path == f"/drafts/{DRAFT_ID}" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "draft": {
                        "id": DRAFT_ID,
                        "project_id": PROJECT_ID,
                        "title": "Draft",
                        "paper_type": "literature_review",
                        "content": self.draft_content,
                        "current_version": 1,
                        "status": "draft",
                        "created_at": "2026-09-02T00:00:00Z",
                        "updated_at": "2026-09-02T00:00:00Z",
                    },
                    "versions": [],
                },
            )
        if path == f"/drafts/{DRAFT_ID}" and request.method == "PUT":
            self.draft_content = body["content"]
            return httpx.Response(200, json={"id": DRAFT_ID, "content": body["content"]})
        if path == f"/projects/{PROJECT_ID}/verify-and-heal" and request.method == "POST":
            report = self._heal_and_report(body["draft_id"])
            return httpx.Response(202, json={"task_id": self._new_task("claim_verify", report)})
        if path == f"/drafts/{DRAFT_ID}/claim-verification" and request.method == "GET":
            return httpx.Response(200, json=self._last_report or self._report(DRAFT_ID))
        if (
            path.startswith(f"/papers/{FIXTURE_PAPER_ID}/fulltext-chunks/")
            and request.method == "GET"
        ):
            # The delivered-evidence record's own chunk
            # lookup. One fixture chunk, regardless of index, is enough for every test
            # here (none exercises a paper with more than one chunk).
            chunk_index = int(path.rsplit("/", 1)[-1])
            return httpx.Response(
                200,
                json={
                    "paper_id": FIXTURE_PAPER_ID,
                    "chunk_index": chunk_index,
                    "chunk_count": 1,
                    "section": "results",
                    "text": FIXTURE_CHUNK_TEXT,
                },
            )
        return httpx.Response(404, json={"detail": f"unrouted {request.method} {path}"})


def fake_resolver(doi: str) -> dict[str, Any] | None:
    n = int(doi.rsplit("-", 1)[-1])
    names = ["Alpha", "Beta", "Gamma", "Delta"]
    return {
        "id": f"https://openalex.org/W{100000 + n}",
        "title": f"Fixture paper {n}",
        "publication_year": 2015 + n,
        "authorships": [{"author": {"display_name": f"Ann {names[n % 4]}"}}],
        "primary_location": {"source": {"display_name": "Fixture Journal", "issn": ["1234-5678"]}},
        "abstract_inverted_index": {"gain": [1], "A": [0]},
        "cited_by_count": 7,
    }


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def protocol() -> dict[str, Any]:
    return json.loads((FIXTURES / "protocol.json").read_text(encoding="utf-8"))


@pytest.fixture
def seeds() -> dict[str, Any]:
    return json.loads((FIXTURES / "seed_dois.json").read_text(encoding="utf-8"))


def make_client(server: FakeServer) -> ApiClient:
    return ApiClient(API, transport=httpx.MockTransport(server.handle))


def make_runner(
    server: FakeServer, protocol, seeds, tmp_path: Path, **overrides: Any
) -> tuple[DemoRunner, FakeClock, list[str]]:
    clock = FakeClock()
    logs: list[str] = []
    kwargs: dict[str, Any] = dict(
        protocol=protocol,
        seeds=seeds,
        output_dir=tmp_path / "out",
        expected_dir=None,
        timeout_s=1200,
        resolver=fake_resolver,
        sleep=clock.sleep,
        clock=clock,
        now=lambda: datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc),
        log=logs.append,
        suffix="abc123",
        password="Demo-password-1",
    )
    kwargs.update(overrides)
    return DemoRunner(make_client(server), **kwargs), clock, logs


# ---------------------------------------------------------------------------
# End-to-end against the fake server
# ---------------------------------------------------------------------------


def test_full_run_calls_every_endpoint_in_order(protocol, seeds, tmp_path):
    server = FakeServer()
    runner, clock, logs = make_runner(server, protocol, seeds, tmp_path)
    summary = runner.run()

    paths = [(m, p.split("?")[0]) for m, p, _ in server.calls]
    assert paths[0] == ("POST", "/auth/register")
    assert paths[1] == ("POST", "/projects")
    assert paths[2] == ("POST", f"/projects/{PROJECT_ID}/smart-search")
    smart_task = server.calls[2][2]
    assert smart_task == {
        "query": protocol["research_question"],
        "inclusion_criteria": protocol["inclusion_criteria"],
        "exclusion_criteria": protocol["exclusion_criteria"],
        "wos_filter": "off",
    }
    assert ("GET", "/tasks/aaaaaaaa-0000-0000-0000-000000000001") in paths
    assert ("GET", "/tasks/aaaaaaaa-0000-0000-0000-000000000001?format=json") in [
        (m, p) for m, p, _ in server.calls
    ] or ("GET", "/tasks/aaaaaaaa-0000-0000-0000-000000000001/screening-record?format=json") in [
        (m, p) for m, p, _ in server.calls
    ]
    assert ("GET", "/tasks/aaaaaaaa-0000-0000-0000-000000000001/screening-record?format=csv") in [
        (m, p) for m, p, _ in server.calls
    ]
    paper_posts = [b for m, p, b in server.calls if p == f"/projects/{PROJECT_ID}/papers"]
    assert len(paper_posts) == 12
    assert paper_posts[0]["paper_data"]["doi"] == "10.5555/demo-001"
    assert paper_posts[0]["paper_data"]["authors"] == [{"name": "Ann Beta"}]
    assert paper_posts[0]["paper_data"]["source_api"] == "openalex"
    assert ("POST", f"/projects/{PROJECT_ID}/acquire-full-texts") in paths
    assert ("POST", f"/projects/{PROJECT_ID}/drafts") in paths
    gen = [b for m, p, b in server.calls if p == f"/drafts/{DRAFT_ID}/generate"][0]
    assert gen == {
        "section_type": "literature_review",
        "context": protocol["section"]["instructions"],
        "language": "en",
        "target_words": protocol["section"]["target_words"],
    }
    # The write job's own gated loop saves the AI section
    # itself, so the demo reads it back (GET) instead of PUTting it; the demo's own
    # PUT is now only for step 9's appended claim fixtures.
    assert ("GET", f"/drafts/{DRAFT_ID}") in paths
    assert ("PUT", f"/drafts/{DRAFT_ID}") in paths
    verify = [b for m, p, b in server.calls if p == f"/projects/{PROJECT_ID}/verify-and-heal"][0]
    assert verify == {"draft_id": DRAFT_ID}
    # The standalone action's own follow-up GETs (the report, then the healed draft),
    # in order, both before the delivered-evidence step's own chunk fetch -- the
    # run's true last call, since that step runs after claim verification.
    claim_verification_idx = paths.index(("GET", f"/drafts/{DRAFT_ID}/claim-verification"))
    draft_get_idx = max(i for i, p in enumerate(paths) if p == ("GET", f"/drafts/{DRAFT_ID}"))
    assert claim_verification_idx < draft_get_idx
    chunk_paths = [
        p for p in paths if p[0] == "GET" and p[1].startswith(f"/papers/{FIXTURE_PAPER_ID}/")
    ]
    assert chunk_paths, "delivered-evidence step never fetched a chunk"
    assert paths.index(chunk_paths[0]) > draft_get_idx
    assert paths[-1] == chunk_paths[-1]

    # every authenticated call carried the Bearer token (the fake returns 401 otherwise)
    assert all(s.status == "completed" or s.status == "skipped" for s in runner.state.steps)

    # unsupported-1 was healed away; only supported-1 survives as the last paragraph.
    paragraphs = [n for n in server.draft_content["content"] if n["type"] == "paragraph"]
    assert paragraphs[-1]["content"][0]["text"] == protocol["claims"][0]["text"]
    assert not any(
        p["content"][0]["text"].endswith("(Gamma, 2017).") for p in paragraphs
    )
    assert server.draft_content["content"][0]["type"] == "heading"

    out = tmp_path / "out"
    for name in (
        "screening_record.json",
        "screening_record.csv",
        "writing_result.json",
        "draft_content.json",
        "claim_report.json",
        "delivered_evidence.json",
        "summary.json",
        "credentials.json",
    ):
        assert (out / name).exists(), name
    delivered = json.loads((out / "delivered_evidence.json").read_text(encoding="utf-8"))
    # delivered_evidence.json is the one file every section's own
    # rows are merged into, so draft_id/section_title sit on each row rather than at
    # the top level (a run with no extra sections still carries exactly the protocol
    # section's own rows, each tagged with its own draft_id/section_title).
    assert delivered["run_id"] == out.name
    assert [r["status"] for r in delivered["rows"]] == ["verified"]
    assert delivered["rows"][0]["row"] == 1
    assert delivered["rows"][0]["draft_id"] == DRAFT_ID
    assert delivered["rows"][0]["section_title"] == protocol["section"]["title"]
    assert "a gain in vocabulary retention" in delivered["rows"][0]["source_passage"]
    assert delivered["rows"][0]["source_locator"] == {"chunk_index": 0, "section": "results"}
    assert delivered["unlocated_rows"] == 0
    assert delivered["rows"][0]["sentence_in_draft"] is True
    saved = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    # The marker is emitted (build_summary reads
    # state.delivered_evidence), and the raw final-report rows -- a large duplicate of
    # claim_report.json -- are not: only "verification.claims", the two protocol-fixture
    # rows match_claim_results already produced, is on summary.json's own verification
    # block.
    assert saved["delivered_evidence"] == {"rows": 1, "unlocated_rows": 0}
    assert "final_report_verifications" not in saved["verification"]
    assert saved["screening"]["flow"]["included"] == 15
    assert saved["screening"]["provenance"]["screener"]["model_reported"] == ["deepseek-v4-flash"]
    assert saved["seeds"]["requested"] == 12 and len(saved["seeds"]["added"]) == 12
    assert "paper_data" not in saved["seeds"]
    # the one abstract-only paper from the first attempt is healed by
    # the retry (11 + 1 acquired, 0 left abstract-only), and both attempts are kept.
    assert saved["fulltext"]["acquired"] == 12
    assert saved["fulltext"]["abstract_only"] == 0
    assert saved["fulltext"]["retry_attempted"] is True
    assert [a["acquired"] for a in saved["fulltext"]["attempts"]] == [11, 1]
    assert [a["abstract_only"] for a in saved["fulltext"]["attempts"]] == [1, 0]
    assert server.fulltext_acquire_calls == 2
    assert saved["writing"]["model_reported"] == "deepseek-v4-flash"
    assert saved["writing"]["temperature"] == 0.7
    assert saved["writing"]["citation_audit"]["needs_citation_flags"] == 1
    assert saved["verification"]["verified_count"] == 1
    assert [c["status"] for c in saved["verification"]["claims"]] == ["verified", "unsupported"]
    assert all(c["matches_expected"] for c in saved["verification"]["claims"])
    assert saved["comparison"] is None
    assert summary["elapsed_seconds"] == 0.0
    assert any("ScholarRAG demo summary" in line for line in logs)
    # the screening record CSV was saved verbatim
    assert (out / "screening_record.csv").read_text(encoding="utf-8").startswith("outcome,stage")


def test_fulltext_no_retry_when_nothing_abstract_only(protocol, seeds, tmp_path):
    """The one-time retry only fires when the first attempt's own
    result carries ``abstract_only > 0``; a first attempt that already acquired
    every seed makes exactly one call and records no retry."""
    server = FakeServer()
    server.fulltext_result = {"acquired": 12, "abstract_only": 0, "already_acquired": 0}
    runner, clock, _ = make_runner(server, protocol, seeds, tmp_path)
    summary = runner.run()

    assert server.fulltext_acquire_calls == 1
    assert summary["fulltext"]["acquired"] == 12
    assert summary["fulltext"]["abstract_only"] == 0
    assert summary["fulltext"]["retry_attempted"] is False
    assert len(summary["fulltext"]["attempts"]) == 1


def test_fulltext_retry_leaves_survivors_abstract_only(protocol, seeds, tmp_path):
    """When a seed is still abstract-only after the retry, the
    top-level ``abstract_only`` reports the retry's own count (not the first
    attempt's), and both attempts survive in ``attempts`` for a reader who wants
    the raw per-attempt numbers."""
    server = FakeServer()
    server.fulltext_result = {"acquired": 10, "abstract_only": 2, "already_acquired": 0}
    server.fulltext_retry_result = {"acquired": 0, "abstract_only": 2, "already_acquired": 10}
    runner, clock, _ = make_runner(server, protocol, seeds, tmp_path)
    summary = runner.run()

    assert server.fulltext_acquire_calls == 2
    assert summary["fulltext"]["acquired"] == 10
    assert summary["fulltext"]["abstract_only"] == 2
    assert summary["fulltext"]["retry_attempted"] is True
    assert [a["abstract_only"] for a in summary["fulltext"]["attempts"]] == [2, 2]


def test_skip_search_omits_smart_search(protocol, seeds, tmp_path):
    server = FakeServer()
    runner, _, _ = make_runner(server, protocol, seeds, tmp_path, skip_search=True)
    summary = runner.run()
    assert not any("smart-search" in p for _, p, _ in server.calls)
    assert summary["screening"] is None
    assert [s["status"] for s in summary["steps"] if s["name"] == "smart search"] == ["skipped"]
    assert not (tmp_path / "out" / "screening_record.json").exists()


# ---------------------------------------------------------------------------
# --replay-queries and --search-only
# ---------------------------------------------------------------------------


def _write_expected_screening_record(expected_dir: Path, rounds: list[dict[str, Any]]) -> None:
    expected_dir.mkdir(parents=True, exist_ok=True)
    (expected_dir / "screening_record.json").write_text(
        json.dumps({"provenance": {"rounds": rounds}}), encoding="utf-8"
    )


def test_replay_queries_sends_the_expected_baselines_own_queries_as_an_override(
    protocol, seeds, tmp_path
):
    expected_dir = tmp_path / "expected"
    _write_expected_screening_record(
        expected_dir,
        [
            {"queries": [{"query": "baseline q1"}, {"query": "baseline q2"}]},
            {"queries": [{"query": "baseline q3"}]},
        ],
    )
    server = FakeServer()
    runner, _, _ = make_runner(
        server, protocol, seeds, tmp_path, expected_dir=expected_dir, replay_queries=True,
    )
    runner.run()

    body = [b for m, p, b in server.calls if p == f"/projects/{PROJECT_ID}/smart-search"][0]
    assert body["queries_override"] == [["baseline q1", "baseline q2"], ["baseline q3"]]


def test_replay_queries_without_a_screening_record_raises_a_clear_error(
    protocol, seeds, tmp_path
):
    """The baseline is read and validated before the first
    HTTP call, so a bad ``--expected`` target must fail without registering a demo user
    or creating a project on the live stack."""
    expected_dir = tmp_path / "expected-empty"
    expected_dir.mkdir()
    server = FakeServer()
    runner, _, _ = make_runner(
        server, protocol, seeds, tmp_path, expected_dir=expected_dir, replay_queries=True,
    )
    with pytest.raises(DemoError, match="screening_record.json"):
        runner.run()
    assert server.calls == []
    assert not (tmp_path / "out" / "credentials.json").exists()


def test_search_only_stops_after_the_screening_record_and_skips_the_rest(
    protocol, seeds, tmp_path
):
    server = FakeServer()
    runner, _, _ = make_runner(server, protocol, seeds, tmp_path, search_only=True)
    summary = runner.run()

    paths = [p.split("?")[0] for _, p, _ in server.calls]
    assert f"/projects/{PROJECT_ID}/smart-search" in paths
    assert f"/projects/{PROJECT_ID}/papers" not in paths
    assert f"/projects/{PROJECT_ID}/acquire-full-texts" not in paths
    assert f"/projects/{PROJECT_ID}/drafts" not in paths
    assert f"/projects/{PROJECT_ID}/verify-and-heal" not in paths

    assert summary["screening"]["flow"]["included"] == 15
    assert summary["seeds"] is None
    assert summary["writing"] is None
    assert summary["verification"] is None
    saved = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert saved["screening"]["flow"]["included"] == 15
    assert saved["writing"] is None


def test_failed_task_stops_the_run_with_a_clear_message(protocol, seeds, tmp_path):
    server = FakeServer()
    server.task_outcome = "failed"
    runner, _, _ = make_runner(server, protocol, seeds, tmp_path)
    with pytest.raises(DemoError, match="smart search: task .* failed: boom"):
        runner.run()
    assert runner.state.steps[-1].status == "failed"


def test_smart_search_409_is_reported(protocol, seeds, tmp_path):
    server = FakeServer()
    server.smart_search_status = 409
    runner, _, _ = make_runner(server, protocol, seeds, tmp_path)
    with pytest.raises(DemoError, match="Smart Search refused \\(HTTP 409"):
        runner.run()


def test_compare_with_expected_dir_is_written_into_summary(protocol, seeds, tmp_path):
    expected_dir = tmp_path / "expected"
    expected_dir.mkdir()
    expected = {
        "screening": {
            "flow": {"identified": 90, "included": 30},
            "provenance": {"screener": {"model_reported": ["deepseek-v4-flash"]}},
        },
        "writing": {"model_reported": "deepseek-v4-flash"},
        "verification": {
            "provenance": {"model_reported": ["deepseek-v4-flash"]},
            "claims": [
                {"id": "supported-1", "status": "verified"},
                {"id": "unsupported-1", "status": "unsupported"},
            ],
        },
    }
    (expected_dir / "summary.json").write_text(json.dumps(expected), encoding="utf-8")
    server = FakeServer()
    runner, _, logs = make_runner(server, protocol, seeds, tmp_path, expected_dir=expected_dir)
    summary = runner.run()
    by_check = {c["check"]: c for c in summary["comparison"]}
    assert by_check["screening.flow.identified"]["ok"] is True  # 100 vs 90 within 30 %
    assert by_check["screening.flow.included"]["ok"] is False  # 15 vs 30 outside 30 %
    assert by_check["writing.model_reported"]["ok"] is True
    assert by_check["verification.claims.supported-1.status"]["ok"] is True
    assert any("DRIFT screening.flow.included" in line for line in logs)


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------


def test_register_falls_back_to_login_on_409():
    server = FakeServer(register_status=409)
    client = make_client(server)
    user = register_or_login(client, "abc123", "Demo-password-1")
    assert user["mode"] == "logged_in"
    assert client.token == "tok"
    assert [p for _, p, _ in server.calls] == ["/auth/register", "/auth/login"]
    assert server.calls[1][2] == {
        "email": "scholarrag-demo-abc123@example.com",
        "password": "Demo-password-1",
    }


def test_register_other_errors_propagate():
    server = FakeServer(register_status=422)
    client = make_client(server)
    with pytest.raises(ApiError) as info:
        register_or_login(client, "abc123", "Demo-password-1")
    assert info.value.status_code == 422


def test_connection_failure_is_a_demo_error():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = ApiClient(API, transport=httpx.MockTransport(boom))
    with pytest.raises(ApiError, match="could not reach"):
        client.json("GET", "/projects")


def test_poll_task_times_out_with_fake_clock():
    server = FakeServer(polls_before_done=10_000)
    client = make_client(server)
    client.token = "tok"
    task_id = server._new_task("smart_search", {})
    clock = FakeClock()
    logs: list[str] = []
    with pytest.raises(DemoError, match="still 'running' after 30 s"):
        poll_task(
            client,
            task_id,
            label="smart search",
            timeout_s=30,
            sleep=clock.sleep,
            clock=clock,
            log=logs.append,
        )
    assert clock.t >= 30
    assert len(logs) >= 2  # progress lines printed as the message changes


def test_poll_task_returns_completed_task():
    server = FakeServer(polls_before_done=1)
    client = make_client(server)
    client.token = "tok"
    task_id = server._new_task("writing", {"content": "x"})
    clock = FakeClock()
    task = poll_task(
        client, task_id, label="w", timeout_s=60, sleep=clock.sleep, clock=clock, log=lambda _: None
    )
    assert task["status"] == "completed" and task["result"] == {"content": "x"}


def test_poll_task_survives_a_401_between_two_polls_by_refreshing_the_token():
    """The access token issued at registration expires after
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES (15 by default); the per-paper grounding plus
    the verify/revise/re-verify passes can make the write step alone run past that
    budget, so a real run can see the token expire between two polls of the same task.
    The backend job keeps running regardless -- a 401 here must trigger one
    ``POST /auth/refresh`` and a transparent retry, and the run must still complete,
    not die with `DEMO FAILED: ... HTTP 401`."""
    server = FakeServer(polls_before_done=2, expire_token_after=1)
    client = make_client(server)
    user = register_or_login(client, "abc123", "Demo-password-1")
    assert user["mode"] == "registered"
    assert client.token == "tok"
    assert client.refresh_token == "r"

    task_id = server._new_task("writing", {"content": "x"})
    clock = FakeClock()
    logs: list[str] = []
    task = poll_task(
        client, task_id, label="AI write", timeout_s=120,
        sleep=clock.sleep, clock=clock, log=logs.append,
    )

    assert task["status"] == "completed"
    assert task["result"] == {"content": "x"}
    # the client swapped to the refreshed token, not just the response of one call
    assert client.token == "tok2"
    refresh_calls = [p for _, p, _ in server.calls if p == "/auth/refresh"]
    assert len(refresh_calls) == 1


def test_poll_task_prints_an_elapsed_heartbeat_when_progress_never_changes():
    """The inclusion-judge stage can spend its whole budget inside one long call with no
    progress update at all; when the job's own progress and progress_message never
    change, poll_task must still print something every PROGRESS_HEARTBEAT_S seconds, so
    the wait does not look hung."""

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "t1",
                "job_type": "smart_search",
                "status": "running",
                "progress": 0.5,
                "progress_message": "judging inclusions",
                "result": None,
                "error": None,
                "created_at": "2026-09-02T00:00:00Z",
                "updated_at": "2026-09-02T00:00:00Z",
            },
        )

    client = ApiClient(API, transport=httpx.MockTransport(handle))
    client.token = "tok"
    clock = FakeClock()
    logs: list[str] = []
    with pytest.raises(DemoError, match="still 'running' after 150 s"):
        poll_task(
            client, "t1", label="smart search", timeout_s=150,
            sleep=clock.sleep, clock=clock, log=logs.append,
        )
    heartbeats = [line for line in logs if "elapsed" in line]
    assert len(heartbeats) >= 2
    # the unchanging progress_message was still printed once, up front
    assert any("judging inclusions" in line for line in logs)


def test_poll_task_prints_the_jobs_own_progress_message_as_soon_as_it_changes():
    """When the job reports a new stage of its own (e.g. moving from retrieval into the
    inclusion-judge stage), that progress line is printed on the very poll it first
    appears, not delayed until the next heartbeat."""
    messages = ["retrieving", "retrieving", "judging inclusions", "judging inclusions"]
    calls = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        message = messages[min(calls["n"], len(messages) - 1)]
        calls["n"] += 1
        return httpx.Response(
            200,
            json={
                "id": "t1",
                "job_type": "smart_search",
                "status": "running",
                "progress": 0.5,
                "progress_message": message,
                "result": None,
                "error": None,
                "created_at": "2026-09-02T00:00:00Z",
                "updated_at": "2026-09-02T00:00:00Z",
            },
        )

    client = ApiClient(API, transport=httpx.MockTransport(handle))
    client.token = "tok"
    clock = FakeClock()
    logs: list[str] = []
    with pytest.raises(DemoError):
        poll_task(
            client, "t1", label="smart search", timeout_s=20,
            sleep=clock.sleep, clock=clock, log=logs.append,
        )
    starting_idx = next(i for i, line in enumerate(logs) if "retrieving" in line)
    judging_idx = next(i for i, line in enumerate(logs) if "judging inclusions" in line)
    assert judging_idx == starting_idx + 1


def test_demo_runner_search_timeout_is_independent_of_timeout_s(protocol, seeds, tmp_path):
    """The smart search step waits up to ``search_timeout_s``, not ``timeout_s``: a tiny
    ``search_timeout_s`` still times the smart search step out even though ``timeout_s``
    (used by every other step) stays generous, and the message points at the flag that
    actually controls it."""
    server = FakeServer(polls_before_done=10_000)
    runner, clock, logs = make_runner(
        server, protocol, seeds, tmp_path, timeout_s=3900, search_timeout_s=30,
    )
    with pytest.raises(DemoError) as exc_info:
        runner.run()
    message = str(exc_info.value)
    assert "smart search" in message and "after 30 s" in message
    assert "--search-timeout" in message
    assert clock.t >= 30


def test_demo_runner_non_search_steps_keep_using_timeout_s(protocol, seeds, tmp_path):
    """A step other than Smart Search keeps waiting up to ``timeout_s`` and, on timing
    out, tells the reviewer to raise ``--timeout`` (not ``--search-timeout``), even when
    ``search_timeout_s`` is set very large."""
    server = FakeServer(polls_before_done=10_000)
    runner, clock, logs = make_runner(
        server, protocol, seeds, tmp_path,
        timeout_s=20, search_timeout_s=3900, skip_search=True,
    )
    with pytest.raises(DemoError) as exc_info:
        runner.run()
    message = str(exc_info.value)
    assert "full texts" in message and "after 20 s" in message
    assert "--timeout" in message and "--search-timeout" not in message
    assert clock.t >= 20


def test_start_smart_search_body_and_409(protocol):
    server = FakeServer()
    client = make_client(server)
    client.token = "tok"
    task_id = start_smart_search(client, PROJECT_ID, protocol)
    assert task_id.startswith("aaaaaaaa-")
    server.smart_search_status = 409
    with pytest.raises(DemoError, match="another Smart Search"):
        start_smart_search(client, PROJECT_ID, protocol)


def test_start_smart_search_sends_publication_date_max_when_the_protocol_has_it(protocol):
    """Forwarded verbatim so a replayed run sees the same date-capped
    OpenAlex corpus the baseline was captured against."""
    protocol["publication_date_max"] = "2026-09-08"
    server = FakeServer()
    client = make_client(server)
    client.token = "tok"
    start_smart_search(client, PROJECT_ID, protocol)
    body = [b for m, p, b in server.calls if p == f"/projects/{PROJECT_ID}/smart-search"][0]
    assert body["publication_date_max"] == "2026-09-08"


def test_start_smart_search_omits_publication_date_max_when_the_protocol_lacks_it(protocol):
    assert "publication_date_max" not in protocol
    server = FakeServer()
    client = make_client(server)
    client.token = "tok"
    start_smart_search(client, PROJECT_ID, protocol)
    body = [b for m, p, b in server.calls if p == f"/projects/{PROJECT_ID}/smart-search"][0]
    assert "publication_date_max" not in body


def test_start_smart_search_sends_queries_override_when_given(protocol):
    server = FakeServer()
    client = make_client(server)
    client.token = "tok"
    queries_override = [["round one query"], ["round two a", "round two b"]]
    start_smart_search(client, PROJECT_ID, protocol, queries_override=queries_override)
    body = [b for m, p, b in server.calls if p == f"/projects/{PROJECT_ID}/smart-search"][0]
    assert body["queries_override"] == queries_override


def test_start_smart_search_omits_queries_override_by_default(protocol):
    server = FakeServer()
    client = make_client(server)
    client.token = "tok"
    start_smart_search(client, PROJECT_ID, protocol)
    body = [b for m, p, b in server.calls if p == f"/projects/{PROJECT_ID}/smart-search"][0]
    assert "queries_override" not in body


def test_start_generation_sends_target_words_from_the_protocol(protocol):
    """The protocol's ``section.target_words`` is forwarded verbatim in
    the generate request body, alongside the existing section_type/context/language."""
    server = FakeServer()
    client = make_client(server)
    client.token = "tok"
    start_generation(client, DRAFT_ID, protocol)
    gen = [b for m, p, b in server.calls if p == f"/drafts/{DRAFT_ID}/generate"][0]
    assert gen["target_words"] == protocol["section"]["target_words"] == 300


def test_start_generation_omits_target_words_when_the_protocol_has_none(protocol):
    """A protocol without ``target_words`` (an older fixture, a hand-run request) still
    generates with no length control -- the key is absent, not sent as ``null``."""
    del protocol["section"]["target_words"]
    server = FakeServer()
    client = make_client(server)
    client.token = "tok"
    start_generation(client, DRAFT_ID, protocol)
    gen = [b for m, p, b in server.calls if p == f"/drafts/{DRAFT_ID}/generate"][0]
    assert "target_words" not in gen


def test_start_generation_omits_section_title_by_default(protocol):
    """The protocol section's own call site leaves
    ``section_title`` unset (see `DemoRunner._write_section`), so its long-standing
    ``section_type``-derived heading is unaffected."""
    server = FakeServer()
    client = make_client(server)
    client.token = "tok"
    start_generation(client, DRAFT_ID, protocol)
    gen = [b for m, p, b in server.calls if p == f"/drafts/{DRAFT_ID}/generate"][0]
    assert "section_title" not in gen


def test_start_generation_sends_the_requested_section_title_when_given(protocol):
    """An extra section's own requested title (from
    `demo/sections.json`) is forwarded verbatim, not left for the backend to derive
    from ``section_type`` alone -- see `DemoRunner._write_one_extra_section`."""
    server = FakeServer()
    client = make_client(server)
    client.token = "tok"
    start_generation(client, DRAFT_ID, protocol, section_title="Peer written corrective feedback")
    gen = [b for m, p, b in server.calls if p == f"/drafts/{DRAFT_ID}/generate"][0]
    assert gen["section_title"] == "Peer written corrective feedback"


def test_fetch_draft_reads_the_draft_and_its_content():
    """The write job's own gated loop already saved the AI
    section, so the demo reads it back with a plain GET instead of PUTting it."""
    server = FakeServer()
    server.draft_content = {"type": "doc", "content": [{"type": "heading"}]}
    client = make_client(server)
    client.token = "tok"
    draft = fetch_draft(client, DRAFT_ID)
    assert draft["id"] == DRAFT_ID
    assert draft["content"] == {"type": "doc", "content": [{"type": "heading"}]}


def test_add_seed_papers_treats_409_as_already_present(seeds):
    server = FakeServer()
    server.paper_status = 409
    client = make_client(server)
    client.token = "tok"
    result = add_seed_papers(client, PROJECT_ID, seeds, fake_resolver, log=lambda _: None)
    assert len(result["already_present"]) == 12 and result["added"] == []
    assert result["paper_data"]["10.5555/demo-001"]["year"] == 2016


def test_add_seed_papers_other_error_fails_the_step(seeds):
    server = FakeServer()
    server.paper_status = 500
    client = make_client(server)
    client.token = "tok"
    with pytest.raises(DemoError, match="Adding seed papers failed"):
        add_seed_papers(client, PROJECT_ID, seeds, fake_resolver, log=lambda _: None)


def test_build_paper_data_prefers_seed_fields_and_falls_back_without_openalex():
    seed = {
        "doi": "10.1/a",
        "title": "Seed title",
        "year": 2020,
        "journal": "Seed Journal",
        "openalex_id": "W1",
        "oa_pdf_url": "https://x/y.pdf",
    }
    data = build_paper_data(seed, None)
    assert data["title"] == "Seed title" and data["authors"] == [] and data["abstract"] is None
    assert data["external_id"] == "W1" and data["full_text_url"] == "https://x/y.pdf"
    data = build_paper_data(seed, fake_resolver("10.5555/demo-003"))
    assert data["authors"] == [{"name": "Ann Delta"}]
    assert data["journal_name"] == "Seed Journal" and data["journal_issn"] == "1234-5678"
    assert data["abstract"] == "A gain"


def test_ensure_citation_appends_first_author_and_year():
    paper = {"authors": [{"name": "Teun van Dijk"}], "year": 2019}
    assert ensure_citation("A finding", paper) == "A finding (Dijk, 2019)."
    assert ensure_citation("Already cited (Smith, 2020).", paper) == "Already cited (Smith, 2020)."
    assert ensure_citation("No author info", {"authors": [], "year": 2019}) == "No author info"
    # an existing but unparseable citation is left alone (no double citation)
    kept = "Finding (Bonilla López et al., 2018)."
    assert ensure_citation(kept, paper) == kept


def test_build_draft_content_shape():
    doc = build_draft_content(
        "Literature Review", "P1\n\nP2\n\n", ["C1 (A, 2020).", "C2 (B, 2021)."]
    )
    assert doc["type"] == "doc"
    kinds = [n["type"] for n in doc["content"]]
    assert kinds == ["heading", "paragraph", "paragraph", "paragraph", "paragraph"]
    assert doc["content"][0]["attrs"] == {"level": 2}
    assert doc["content"][-1]["content"][0]["text"] == "C2 (B, 2021)."


def test_terminate_sentence():
    assert terminate_sentence("Ends with a period.") == "Ends with a period."
    assert terminate_sentence("Really?  ") == "Really?"
    assert terminate_sentence("Indeed!") == "Indeed!"
    assert terminate_sentence("open [NEEDS CITATION]") == "open [NEEDS CITATION]."
    assert terminate_sentence("   ") == ""


def test_build_draft_content_attaches_citation_links_by_paragraph_index():
    """The writer's sentence-to-reference map is grouped by ``paragraph_index`` onto
    the matching generated-block paragraph node, mirroring ``drafts-tab.tsx``."""
    citation_links = [
        {
            "paragraph_index": 0,
            "sentence": "P1 cites (A, 2020).",
            "keys": ["a_2020"],
            "citation_text": "(A, 2020)",
        },
        {
            "paragraph_index": 1,
            "sentence": "P2 cites (B, 2021).",
            "keys": ["b_2021"],
            "citation_text": "(B, 2021)",
        },
    ]
    doc = build_draft_content(
        "LR", "P1 cites (A, 2020).\n\nP2 cites (B, 2021).", [], citation_links
    )
    paragraphs = [n for n in doc["content"] if n["type"] == "paragraph"]
    assert paragraphs[0]["attrs"]["citationLinks"] == [
        {"sentence": "P1 cites (A, 2020).", "keys": ["a_2020"], "citation_text": "(A, 2020)"}
    ]
    assert paragraphs[1]["attrs"]["citationLinks"] == [
        {"sentence": "P2 cites (B, 2021).", "keys": ["b_2021"], "citation_text": "(B, 2021)"}
    ]


def test_build_draft_content_paragraph_has_no_attrs_without_links():
    doc = build_draft_content("LR", "P1.\n\nP2.", [])
    paragraphs = [n for n in doc["content"] if n["type"] == "paragraph"]
    assert "attrs" not in paragraphs[0]
    assert "attrs" not in paragraphs[1]


def test_build_draft_content_ignores_a_link_for_an_appended_claim_paragraph():
    """A claim paragraph this script appends itself is never part of the writer's own
    paragraph split, so an out-of-range paragraph_index (here, one past the last
    generated block) attaches nothing -- there is no such generated paragraph node."""
    citation_links = [
        {
            "paragraph_index": 5,
            "sentence": "Not part of the generated text.",
            "keys": ["x_2020"],
            "citation_text": "(X, 2020)",
        }
    ]
    doc = build_draft_content("LR", "P1.\n\nP2.", ["Claim (X, 2020)."], citation_links)
    paragraphs = [n for n in doc["content"] if n["type"] == "paragraph"]
    assert all("attrs" not in p for p in paragraphs)


def test_build_draft_content_terminates_last_generated_block_before_claims():
    """The backend splits sentences at [.!?]+whitespace; the claims must stay separate."""
    doc = build_draft_content(
        "LR", "P1.\n\nSecond block ends open [NEEDS CITATION]", ["C1 (A, 2020)."]
    )
    texts = [n["content"][0]["text"] for n in doc["content"][1:]]
    assert texts == ["P1.", "Second block ends open [NEEDS CITATION].", "C1 (A, 2020)."]
    # already-terminated blocks and blocks without claims are left untouched
    doc = build_draft_content("LR", "P1.\n\nP2?", ["C1 (A, 2020)."])
    assert [n["content"][0]["text"] for n in doc["content"][1:]] == ["P1.", "P2?", "C1 (A, 2020)."]
    doc = build_draft_content("LR", "P1 open", [])
    assert doc["content"][-1]["content"][0]["text"] == "P1 open"


def test_write_section_attaches_the_writer_citation_link_map_end_to_end(protocol, seeds, tmp_path):
    """The writing job's ``citation_links`` flows all the way
    through ``_write_section`` into the saved draft's paragraph attributes."""
    server = FakeServer()
    server.writing_result = {
        **WRITING_RESULT,
        "citation_links": [
            {
                "paragraph_index": 0,
                "sentence": "First paragraph (Alpha, 2021).",
                "keys": ["alpha_2021"],
                "citation_text": "(Alpha, 2021)",
            },
            {
                "paragraph_index": 1,
                "sentence": "Second paragraph (Beta, 2022).",
                "keys": ["beta_2022"],
                "citation_text": "(Beta, 2022)",
            },
        ],
    }
    runner, _clock, _logs = make_runner(server, protocol, seeds, tmp_path)
    runner.run()

    paragraphs = [n for n in server.draft_content["content"] if n["type"] == "paragraph"]
    assert paragraphs[0]["attrs"]["citationLinks"] == [
        {
            "sentence": "First paragraph (Alpha, 2021).",
            "keys": ["alpha_2021"],
            "citation_text": "(Alpha, 2021)",
        }
    ]
    assert paragraphs[1]["attrs"]["citationLinks"] == [
        {
            "sentence": "Second paragraph (Beta, 2022).",
            "keys": ["beta_2022"],
            "citation_text": "(Beta, 2022)",
        }
    ]


def test_write_section_copies_the_citation_link_calls_provenance_into_the_summary(
    protocol, seeds, tmp_path
):
    """The citation-link call is a
    second, real DeepSeek request the writing job folds into its own provenance
    dict; ``_write_section`` must copy that through to ``self.state.writing`` verbatim,
    alongside (not instead of) the generation call's own token counts."""
    server = FakeServer()
    server.writing_result = {
        **WRITING_RESULT,
        "provenance": {
            **WRITING_RESULT["provenance"],
            "citation_link_call": {
                "agent": "citation_link",
                "model_reported": "deepseek-v4-flash",
                "input_tokens": 910,
                "output_tokens": 140,
            },
            "total_calls": 2,
            "total_input_tokens": 3910,
            "total_output_tokens": 840,
            "length": {
                "target_words": 300,
                "hard_maximum": 375,
                "words": 320,
                "regenerated": False,
                "over_target": True,
            },
        },
    }
    runner, _clock, _logs = make_runner(server, protocol, seeds, tmp_path)
    summary = runner.run()

    writing = runner.state.writing
    assert writing["input_tokens"] == 3000
    assert writing["output_tokens"] == 700
    assert writing["total_calls"] == 2
    assert writing["total_input_tokens"] == 3910
    assert writing["total_output_tokens"] == 840
    assert writing["citation_link_call"]["model_reported"] == "deepseek-v4-flash"
    assert writing["citation_link_call"]["input_tokens"] == 910
    # The length-control outcome the writing job itself recorded.
    # ``final_words`` is recomputed from the
    # healed draft actually saved (`_verify_claims`), not the write job's own pre-heal
    # report -- 19 is the word count of `FakeServer`'s own default draft content.
    assert writing["length"] == {
        "target_words": 300,
        "hard_maximum": 375,
        "words": 320,
        "regenerated": False,
        "over_target": True,
        "final_words": 19,
    }

    table = format_summary_table(summary, None)
    rows: dict[str, str] = {}
    for line in table.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].startswith("writing."):
            rows[parts[0]] = parts[1].strip()
    assert rows["writing.total_calls"] == "2"
    assert rows["writing.total_input_tokens"] == "3910"
    assert rows["writing.citation_link_call.input_tokens"] == "910"
    assert rows["writing.length.words"] == "320"
    assert rows["writing.length.over_target"] == "True"


def test_verify_claims_copies_citation_coverage_into_the_summary(protocol, seeds, tmp_path):
    """run_demo.py copies the claim report's citation_coverage
    object into summary.json's verification block, verbatim."""
    coverage = {
        "found": 14,
        "linked": 14,
        "sent_to_verifier": 14,
        "unresolved": 0,
        "unresolved_citations": [],
        "by_source": {"mapping": 14, "author-year": 0, "numbered": 0},
    }
    server = FakeServer()
    server.citation_coverage = coverage
    runner, _clock, _logs = make_runner(server, protocol, seeds, tmp_path)
    summary = runner.run()

    assert summary["verification"]["citation_coverage"] == coverage
    report = json.loads((tmp_path / "out" / "claim_report.json").read_text(encoding="utf-8"))
    assert report["citation_coverage"] == coverage


def test_format_summary_table_prints_guard_digest_and_quote_relocation_version(
    protocol, seeds, tmp_path
):
    """Both keys already reach summary.json
    (the whole provenance dict is copied verbatim); the human-readable report built by
    ``format_summary_table`` must also print them, so a reader of the demo run can tell
    which guard set or which relocation rule produced its claim statuses."""
    server = FakeServer()
    runner, _clock, _logs = make_runner(server, protocol, seeds, tmp_path)
    summary = runner.run()

    assert summary["verification"]["provenance"]["guard_digest"] == "570a5b663e6140da"
    assert (
        summary["verification"]["provenance"]["quote_relocation_version"]
        == "8b5baf4f236d2854"
    )

    table = format_summary_table(summary, None)
    rows: dict[str, str] = {}
    for line in table.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].startswith("verification.provenance."):
            rows[parts[0]] = parts[1].strip()
    assert rows["verification.provenance.guard_digest"] == "570a5b663e6140da"
    assert rows["verification.provenance.quote_relocation_version"] == "8b5baf4f236d2854"


def test_format_summary_table_prints_verification_policy_version_and_pass_count(
    protocol, seeds, tmp_path
):
    """verification_policy_version sits next to guard_digest/quote_relocation_version,
    and each claim's own pass count (the bounded second pass) is printed beside its
    status and quote."""
    server = FakeServer()
    runner, _clock, _logs = make_runner(server, protocol, seeds, tmp_path)
    summary = runner.run()

    assert (
        summary["verification"]["provenance"]["verification_policy_version"]
        == "9c6cbaf5347e3965"
    )

    table = format_summary_table(summary, None)
    rows: dict[str, str] = {}
    for line in table.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            rows[parts[0]] = parts[1].strip()
    assert (
        rows["verification.provenance.verification_policy_version"] == "9c6cbaf5347e3965"
    )
    # The repair-prompt version sits beside
    # verification_policy_version in this same human-readable table.
    assert rows["verification.provenance.repair_prompt_version"] == "sha256:fakerepairver"
    pass_rows = [k for k in rows if k.endswith(".passes")]
    assert pass_rows  # at least one claim's pass count line was printed
    for key in pass_rows:
        assert rows[key] == "0"  # make_claim_report's fixture predates `passes`


def test_verify_claims_tolerates_a_missing_citation_coverage(protocol, seeds, tmp_path):
    """A report from a backend build before this field existed carries no
    citation_coverage key at all; the summary field is None, not an error."""
    server = FakeServer()
    runner, _clock, _logs = make_runner(server, protocol, seeds, tmp_path)
    summary = runner.run()
    assert summary["verification"]["citation_coverage"] is None


def test_check_claim_text_in_draft_counts_verbatim_matches():
    draft_content = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "First sentence here."}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "Second sentence too."}]},
        ],
    }
    verifications = [
        {"claim_text": "First sentence here."},
        {"claim_text": "Second sentence too."},
    ]
    assert check_claim_text_in_draft(draft_content, verifications) == (2, 2)


def test_check_claim_text_in_draft_counts_a_mismatch():
    """An end-to-end check: a `claim_text` that is not the draft's own sentence (the
    model's own transcription of it) is counted against `n`."""
    draft_content = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "First sentence here."}]},
        ],
    }
    verifications = [
        {"claim_text": "First sentence here."},
        {"claim_text": "This text was never in the draft."},
    ]
    assert check_claim_text_in_draft(draft_content, verifications) == (1, 2)


def test_check_claim_text_in_draft_ignores_empty_claim_text():
    draft_content = {"type": "doc", "content": []}
    verifications = [{"claim_text": ""}, {"claim_text": None}, {}]
    assert check_claim_text_in_draft(draft_content, verifications) == (0, 0)


# -- The exit invariant, one offline test per part ------------------------


def test_assert_every_final_row_verified_passes_when_every_row_is_verified():
    run_demo.assert_every_final_row_verified(
        [{"status": "verified", "claim_text": "a"}, {"status": "verified", "claim_text": "b"}]
    )
    run_demo.assert_every_final_row_verified([])


def test_assert_every_final_row_verified_raises_on_a_non_verified_row():
    with pytest.raises(DemoError, match="non-verified row"):
        run_demo.assert_every_final_row_verified(
            [
                {"status": "verified", "claim_text": "a"},
                {"status": "needs_nuance", "claim_text": "b"},
            ]
        )


def test_assert_no_needs_citation_marker_passes_when_absent():
    draft_content = {
        "type": "doc",
        "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Clean text."}]}],
    }
    run_demo.assert_no_needs_citation_marker(draft_content)


def test_assert_no_needs_citation_marker_raises_when_present():
    draft_content = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": "Unfinished [NEEDS CITATION]."}],
            }
        ],
    }
    with pytest.raises(DemoError, match=r"\[NEEDS CITATION\] marker survived"):
        run_demo.assert_no_needs_citation_marker(draft_content)


def test_assert_no_needs_citation_marker_raises_on_the_variant_form_with_a_clause():
    """The writer sometimes writes the marker as its own editorial note with a
    colon and an explanatory clause inside the brackets, not the bare form
    (`demo/output/20260916-120417/draft_content_7.json`)."""
    draft_content = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Zhang and Hyland (2018) similarly found a very limited "
                            "range of revision operations in second drafts. "
                            "[NEEDS CITATION: evidence on peer feedback relative to "
                            "teacher feedback specifically for accuracy gains.]"
                        ),
                    }
                ],
            }
        ],
    }
    with pytest.raises(DemoError, match=r"\[NEEDS CITATION\] marker survived"):
        run_demo.assert_no_needs_citation_marker(draft_content)


def test_assert_no_needs_citation_marker_passes_an_ordinary_bracketed_citation():
    """A numbered citation or a bracketed author-year aside is never mistaken for the
    marker: neither carries "NEEDS" immediately before "CITATION"."""
    draft_content = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {
                        "type": "text",
                        "text": "A fully cited claim [12] and another [Smith, 2020].",
                    }
                ],
            }
        ],
    }
    run_demo.assert_no_needs_citation_marker(draft_content)


def test_assert_citation_coverage_fully_resolved_passes_when_none_or_zero():
    run_demo.assert_citation_coverage_fully_resolved(None)
    run_demo.assert_citation_coverage_fully_resolved({"found": 5, "unresolved": 0})


def test_assert_citation_coverage_fully_resolved_raises_when_unresolved():
    with pytest.raises(DemoError, match="never resolved to a library paper"):
        run_demo.assert_citation_coverage_fully_resolved({"found": 5, "unresolved": 2})


def test_assert_no_duplicate_leading_heading_passes_on_a_single_heading():
    draft_content = {
        "type": "doc",
        "content": [
            {
                "type": "heading",
                "attrs": {"level": 2},
                "content": [{"type": "text", "text": "Literature Review"}],
            },
            {"type": "paragraph", "content": [{"type": "text", "text": "Body text."}]},
        ],
    }
    run_demo.assert_no_duplicate_leading_heading(draft_content)


def test_assert_no_duplicate_leading_heading_passes_on_none_or_empty():
    run_demo.assert_no_duplicate_leading_heading(None)
    run_demo.assert_no_duplicate_leading_heading({"type": "doc", "content": []})


def test_assert_no_duplicate_leading_heading_passes_on_two_differently_worded_headings():
    draft_content = {
        "type": "doc",
        "content": [
            {
                "type": "heading",
                "attrs": {"level": 2},
                "content": [{"type": "text", "text": "Literature Review"}],
            },
            {
                "type": "heading",
                "attrs": {"level": 3},
                "content": [{"type": "text", "text": "Direct corrections"}],
            },
        ],
    }
    run_demo.assert_no_duplicate_leading_heading(draft_content)


def test_assert_no_duplicate_leading_heading_raises_on_two_consecutive_identical_headings():
    draft_content = {
        "type": "doc",
        "content": [
            {
                "type": "heading",
                "attrs": {"level": 2},
                "content": [{"type": "text", "text": "Literature Review"}],
            },
            {
                "type": "heading",
                "attrs": {"level": 1},
                "content": [{"type": "text", "text": "Literature Review"}],
            },
            {"type": "paragraph", "content": [{"type": "text", "text": "Body text."}]},
        ],
    }
    with pytest.raises(DemoError, match="two consecutive headings"):
        run_demo.assert_no_duplicate_leading_heading(draft_content)


def test_assert_no_duplicate_leading_heading_is_case_and_whitespace_insensitive():
    draft_content = {
        "type": "doc",
        "content": [
            {
                "type": "heading",
                "attrs": {"level": 2},
                "content": [{"type": "text", "text": "Literature  Review"}],
            },
            {
                "type": "heading",
                "attrs": {"level": 1},
                "content": [{"type": "text", "text": "literature review "}],
            },
        ],
    }
    with pytest.raises(DemoError, match="two consecutive headings"):
        run_demo.assert_no_duplicate_leading_heading(draft_content)


def test_verify_claims_raises_when_final_report_has_a_non_verified_row(protocol, seeds, tmp_path):
    """Part 1 of the exit invariant, wired into `DemoRunner._verify_claims`
    itself, not only into the pure helper: a regression that lets a non-verified row
    survive into `final_report.verifications` must stop the run here."""
    server = FakeServer()
    original_heal = server._heal_and_report

    def _bad_heal(draft_id: str) -> dict[str, Any]:
        report = original_heal(draft_id)
        report["final_report"]["verifications"][0]["status"] = "needs_nuance"
        return report

    server._heal_and_report = _bad_heal
    runner, _clock, _logs = make_runner(server, protocol, seeds, tmp_path)
    with pytest.raises(DemoError, match="non-verified row"):
        runner.run()


def test_verify_claims_raises_when_a_needs_citation_marker_survives_healing(
    protocol, seeds, tmp_path
):
    """Part 2 of the exit invariant, wired into `DemoRunner._verify_claims`:
    a `[NEEDS CITATION]` marker the healing pass failed to strip must stop the run,
    not reach a saved draft the manuscript could quote as fully cited."""
    server = FakeServer()
    original_heal = server._heal_and_report

    def _bad_heal(draft_id: str) -> dict[str, Any]:
        report = original_heal(draft_id)
        paragraphs = [n for n in server.draft_content["content"] if n["type"] == "paragraph"]
        paragraphs[0]["content"][0]["text"] += " [NEEDS CITATION]"
        return report

    server._heal_and_report = _bad_heal
    runner, _clock, _logs = make_runner(server, protocol, seeds, tmp_path)
    with pytest.raises(DemoError, match=r"\[NEEDS CITATION\] marker survived"):
        runner.run()


def test_verify_claims_raises_when_citation_coverage_has_unresolved_citations(
    protocol, seeds, tmp_path
):
    """Part 3 of the exit invariant, wired into `DemoRunner._verify_claims`:
    a citation the draft's own extractor never resolved to a library paper carries no
    row at all, verified or not -- must stop the run rather than only being compared
    one-sided against a baseline."""
    server = FakeServer()
    server.citation_coverage = {
        "found": 3,
        "linked": 2,
        "sent_to_verifier": 2,
        "unresolved": 1,
        "unresolved_citations": ["(Nobody, 2099)"],
        "by_source": {"author-year": 3},
    }
    runner, _clock, _logs = make_runner(server, protocol, seeds, tmp_path)
    with pytest.raises(DemoError, match="never resolved to a library paper"):
        runner.run()


def test_verify_claims_records_claim_text_in_draft_and_logs_no_drift_when_all_match(
    protocol, seeds, tmp_path
):
    """Checked against the final (healed) report, not the raw one --
    unsupported-1 is healed away by design, so only supported-1's own claim_text is
    ever expected to occur in the saved draft."""
    server = FakeServer()
    runner, _clock, logs = make_runner(server, protocol, seeds, tmp_path)
    summary = runner.run()
    assert summary["verification"]["claim_text_in_draft"] == "1 of 1"
    assert not any("DRIFT" in line for line in logs)


def test_verify_claims_logs_drift_when_a_claim_text_is_not_in_the_draft(
    protocol, seeds, tmp_path
):
    """A report whose own `claim_text` diverges from the saved draft must be visible
    in the run's own log, not only in a later, separate check. The corruption is a
    prefix, not a replacement: `match_claim_results`' lenient containment match must
    still recognise supported-1 as verified (the exit-invariant check would
    otherwise fail this run for an unrelated reason), while the exact, unnormalised
    substring check this test targets still sees the mismatch."""
    server = FakeServer()
    original_report = server._report

    def _bad_report(draft_id: str) -> dict[str, Any]:
        report = original_report(draft_id)
        report["verifications"][0]["claim_text"] = (
            "Restated: " + report["verifications"][0]["claim_text"]
        )
        return report

    server._report = _bad_report
    runner, _clock, logs = make_runner(server, protocol, seeds, tmp_path)
    summary = runner.run()
    assert summary["verification"]["claim_text_in_draft"] == "0 of 1"
    assert any("DRIFT" in line and "0 of 1" in line for line in logs)


def test_unpunctuated_generated_text_keeps_claims_verbatim(protocol, seeds, tmp_path):
    """WRITING_RESULT content ending without terminal punctuation: the AI's own
    trailing block is terminated with '.' before the claim fixtures are appended
    (`append_claim_paragraphs`), so the backend's sentence
    splitter never merges it with the first appended claim. unsupported-1 is healed
    away by the standalone action, leaving supported-1 as the sole surviving claim.
    The trailing block's own ``[NEEDS CITATION]`` marker is stripped by the same
    healing pass, so it survives in neither the
    saved draft nor the punctuation this test checks."""
    server = FakeServer()
    server.writing_result = {
        **WRITING_RESULT,
        "content": "First paragraph (Alpha, 2021).\n\nTrailing block, no period [NEEDS CITATION]",
    }
    runner, _clock, _logs = make_runner(server, protocol, seeds, tmp_path)
    summary = runner.run()

    paragraphs = [
        "".join(c["text"] for c in n["content"])
        for n in server.draft_content["content"]
        if n["type"] == "paragraph"
    ]
    assert paragraphs[-2] == "Trailing block, no period."
    claim_texts = [c["text"] for c in summary["writing"]["claims_appended"]]
    assert paragraphs[-1] == claim_texts[0]
    saved = json.loads((tmp_path / "out" / "draft_content.json").read_text(encoding="utf-8"))
    assert saved == server.draft_content
    report = json.loads((tmp_path / "out" / "claim_report.json").read_text(encoding="utf-8"))
    final_claim_texts = [v["claim_text"] for v in report["final_report"]["verifications"]]
    assert final_claim_texts == [claim_texts[0]]
    assert all(c["matches_expected"] for c in summary["verification"]["claims"])


def test_append_claim_paragraphs_terminates_the_last_block_before_claims():
    """The backend splits sentences at [.!?]+whitespace; an appended claim must stay
    its own sentence, not merge with an unterminated trailing block."""
    content = {
        "type": "doc",
        "content": [
            {
                "type": "heading",
                "attrs": {"level": 2},
                "content": [{"type": "text", "text": "T"}],
            },
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": "Ends open [NEEDS CITATION]"}],
            },
        ],
    }
    merged = append_claim_paragraphs(content, ["Claim (A, 2020)."])
    texts = [n["content"][0]["text"] for n in merged["content"][1:]]
    assert texts == ["Ends open [NEEDS CITATION].", "Claim (A, 2020)."]

    # An already-terminated block, or no claims at all, is left untouched.
    already_done = {
        "type": "doc",
        "content": [{"type": "paragraph", "content": [{"type": "text", "text": "P2?"}]}],
    }
    assert append_claim_paragraphs(already_done, ["C1 (A, 2020)."])["content"][0][
        "content"
    ][0]["text"] == "P2?"
    assert append_claim_paragraphs(already_done, [])["content"] == already_done["content"]


def test_match_claim_results_by_containment():
    claims = [
        {
            "id": "s",
            "text": "Alpha found X (Alpha, 2021).",
            "source_doi": "d",
            "construction": "",
            "expected_status": "verified",
        },
        {
            "id": "u",
            "text": "Missing claim (Beta, 2022).",
            "source_doi": "d",
            "construction": "",
            "expected_status": "unsupported",
        },
    ]
    report = {
        "verifications": [
            {
                "claim_text": "Alpha   found X (Alpha, 2021).",
                "status": "verified",
                "evidence_quote": "q",
            }
        ]
    }
    rows = match_claim_results(claims, report)
    assert rows[0]["status"] == "verified" and rows[0]["matches_expected"] is True
    assert rows[1]["status"] == "not_in_report" and rows[1]["matches_expected"] is False


def _guard_demoted_claim(**overrides: Any) -> dict[str, Any]:
    claim = {
        "id": "supported-1",
        "text": "Alpha found X (Alpha, 2021).",
        "source_doi": "d",
        "construction": "",
        "expected_status": "verified",
    }
    claim.update(overrides)
    return claim


def test_match_claim_results_treats_a_quote_not_verbatim_demotion_as_matching_expected():
    """The frozen ``quote_not_verbatim`` guard may demote a
    claim the verification model itself judged ``verified`` to ``needs_nuance`` because the
    model's own copy of the quote drifted from the source by a word or two. That is the
    guard doing its job, not model drift, so a fixture expecting ``verified`` still counts
    as matching when the demotion is exactly this shape."""
    claims = [_guard_demoted_claim()]
    report = {
        "verifications": [
            {
                "claim_text": "Alpha   found X (Alpha, 2021).",
                "status": "needs_nuance",
                "model_status": "verified",
                "machine_reasons": ["quote_not_verbatim"],
                "evidence_quote": "q",
            }
        ]
    }
    rows = match_claim_results(claims, report)
    assert rows[0]["status"] == "needs_nuance"
    assert rows[0]["model_status"] == "verified"
    assert rows[0]["machine_reasons"] == ["quote_not_verbatim"]
    assert rows[0]["guard_demotion"] is True
    assert rows[0]["matches_expected"] is True


def test_match_claim_results_needs_nuance_with_extra_reasons_is_not_a_guard_demotion():
    """The allowance is narrow: it fires only when ``machine_reasons`` is exactly
    ``["quote_not_verbatim"]``. A second guard reason alongside it is real drift and must
    still be reported as a mismatch."""
    claims = [_guard_demoted_claim()]
    report = {
        "verifications": [
            {
                "claim_text": "Alpha   found X (Alpha, 2021).",
                "status": "needs_nuance",
                "model_status": "verified",
                "machine_reasons": ["quote_not_verbatim", "numeric_outside_cited_passage"],
                "evidence_quote": "q",
            }
        ]
    }
    rows = match_claim_results(claims, report)
    assert rows[0]["guard_demotion"] is False
    assert rows[0]["matches_expected"] is False


def test_match_claim_results_needs_nuance_without_model_status_verified_is_not_a_guard_demotion():
    claims = [_guard_demoted_claim()]
    report = {
        "verifications": [
            {
                "claim_text": "Alpha   found X (Alpha, 2021).",
                "status": "needs_nuance",
                "model_status": "needs_nuance",
                "machine_reasons": ["quote_not_verbatim"],
                "evidence_quote": "q",
            }
        ]
    }
    rows = match_claim_results(claims, report)
    assert rows[0]["guard_demotion"] is False
    assert rows[0]["matches_expected"] is False


def test_match_claim_results_expected_verified_status_unsupported_is_never_a_match():
    """A fixture expecting ``verified`` must never pass as ``unsupported``, guard demotion
    or not: the allowance above only ever widens ``needs_nuance``."""
    claims = [_guard_demoted_claim()]
    report = {
        "verifications": [
            {
                "claim_text": "Alpha   found X (Alpha, 2021).",
                "status": "unsupported",
                "model_status": "unsupported",
                "machine_reasons": ["quote_not_verbatim"],
                "evidence_quote": "q",
            }
        ]
    }
    rows = match_claim_results(claims, report)
    assert rows[0]["guard_demotion"] is False
    assert rows[0]["matches_expected"] is False


def test_claim_problems_is_silent_on_a_guard_demoted_claim_that_still_survived():
    """The guard-demotion allowance alone (`is_guard_demotion`, folded into
    `matches_expected`) is silent when the demoted fixture nonetheless has
    `final_status: "verified"` -- isolates this allowance from the rule below, which
    is what actually decides whether the fixture survived into the final report."""
    summary = {
        "verification": {
            "claims_in_report": 1,
            "claims": [
                {
                    "id": "supported-1",
                    "status": "needs_nuance",
                    "expected_status": "verified",
                    "model_status": "verified",
                    "machine_reasons": ["quote_not_verbatim"],
                    "guard_demotion": True,
                    "matches_expected": True,
                    "final_status": "verified",
                }
            ],
        }
    }
    assert claim_problems(summary) == []


def test_claim_problems_is_silent_when_the_only_verified_fixture_is_demoted_and_healed_away():
    """`needs_nuance` is never `"verified"`, so finalize's own rule 2
    (`_finalize_paragraph_text`) removes ANY genuinely guard-demoted sentence from the
    healed draft by construction -- `final_status: "not_in_report"` is not a SEPARATE
    failure on top of the demotion, it is the demotion's own, unavoidable consequence.
    This gate accepts a recorded guard demotion here exactly as the first
    `matches_expected` allowance already does, and exactly as `check_delivered.
    check_fixture_claims`'s own rule 7 does (`_is_recorded_guard_demotion`), so this
    gate (which sets the process exit code) agrees with the other one."""
    summary = {
        "verification": {
            "claims_in_report": 1,
            "claims": [
                {
                    "id": "supported-1",
                    "status": "needs_nuance",
                    "expected_status": "verified",
                    "model_status": "verified",
                    "machine_reasons": ["quote_not_verbatim"],
                    "guard_demotion": True,
                    "matches_expected": True,
                    "final_status": "not_in_report",
                }
            ],
        }
    }
    assert claim_problems(summary) == []


def test_claim_problems_reports_when_the_verified_fixture_is_lost_for_an_unrelated_reason():
    """The gate still fires when a fixture expecting "verified" is absent from the final
    report and NOTHING recorded a guard demotion for it (``guard_demotion: False``) --
    a genuine, unexplained loss of the fixture the demotion allowance was never meant
    to cover."""
    summary = {
        "verification": {
            "claims_in_report": 1,
            "claims": [
                {
                    "id": "supported-1",
                    "status": "unsupported",
                    "expected_status": "verified",
                    "model_status": "unsupported",
                    "machine_reasons": [],
                    "guard_demotion": False,
                    "matches_expected": False,
                    "final_status": "not_in_report",
                }
            ],
        }
    }
    problems = claim_problems(summary)
    assert len(problems) == 2  # the per-claim mismatch, and the post-loop gate
    assert any(
        "no fixture expecting \"verified\" survived into the final report" in p
        and "none is a recorded guard demotion" in p
        for p in problems
    )


def test_claim_problems_is_silent_when_a_second_verified_fixture_survives():
    """The "no verified fixture survived" check is an ``any()`` over every fixture whose
    ``expected_status`` is ``"verified"``, not a per-claim check -- one demoted-and-healed
    fixture next to a second verified fixture that DID survive must not report a
    problem."""
    summary = {
        "verification": {
            "claims_in_report": 2,
            "claims": [
                {
                    "id": "supported-1",
                    "status": "needs_nuance",
                    "expected_status": "verified",
                    "model_status": "verified",
                    "machine_reasons": ["quote_not_verbatim"],
                    "guard_demotion": True,
                    "matches_expected": True,
                    "final_status": "not_in_report",
                },
                {
                    "id": "supported-2",
                    "status": "verified",
                    "expected_status": "verified",
                    "matches_expected": True,
                    "final_status": "verified",
                },
            ],
        }
    }
    assert claim_problems(summary) == []


def test_claim_problems_is_silent_on_an_unsupported_claim_absent_from_the_final_report():
    """A fixture expecting ``unsupported`` is not model
    drift when verify-and-heal removed it from the finalized report entirely -- that is
    the intended outcome for an unsupported claim, checked here from ``final_status``
    (the healed report's own status for the same claim id, recorded alongside the raw
    ``status`` used for every other comparison)."""
    summary = {
        "verification": {
            "claims_in_report": 1,
            "claims": [
                {
                    "id": "unsupported-1",
                    "status": "needs_nuance",
                    "expected_status": "unsupported",
                    "model_status": "needs_nuance",
                    "machine_reasons": [],
                    "matches_expected": False,
                    "final_status": "not_in_report",
                }
            ],
        }
    }
    assert claim_problems(summary) == []


def test_claim_problems_reports_an_unsupported_claim_that_was_raw_verified_and_absent():
    """A fixture expecting ``unsupported`` whose raw
    verification pass called it ``verified`` but that is nonetheless absent from the
    healed report (``final_status: not_in_report``) is not the intended "verify-and-heal
    correctly removed an unsupported claim" outcome the silent allowance exists for --
    the raw pass never even flagged it. Must still be reported, not silently passed."""
    summary = {
        "verification": {
            "claims_in_report": 1,
            "claims": [
                {
                    "id": "unsupported-1",
                    "status": "verified",
                    "expected_status": "unsupported",
                    "model_status": "verified",
                    "machine_reasons": [],
                    "matches_expected": False,
                    "final_status": "not_in_report",
                }
            ],
        }
    }
    problems = claim_problems(summary)
    assert len(problems) == 1
    assert "unsupported-1" in problems[0]


def test_claim_problems_reports_an_unsupported_claim_that_survives_into_the_final_report():
    summary = {
        "verification": {
            "claims_in_report": 1,
            "claims": [
                {
                    "id": "unsupported-1",
                    "status": "verified",
                    "expected_status": "unsupported",
                    "model_status": "verified",
                    "machine_reasons": [],
                    "matches_expected": False,
                    "final_status": "verified",
                }
            ],
        }
    }
    problems = claim_problems(summary)
    assert len(problems) == 1
    assert "unsupported-1" in problems[0]


def test_format_summary_table_prints_a_guard_demotion_line_not_a_mismatch():
    text = format_summary_table(
        {
            "seeds": None,
            "verification": {
                "claims": [
                    {
                        "id": "supported-1",
                        "status": "needs_nuance",
                        "expected_status": "verified",
                        "model_status": "verified",
                        "machine_reasons": ["quote_not_verbatim"],
                        "guard_demotion": True,
                        "matches_expected": True,
                        "final_status": "not_in_report",
                    }
                ]
            },
        },
        None,
    )
    line = next(line for line in text.splitlines() if "claim.supported-1.status" in line)
    assert "GUARD DEMOTION" in line
    assert "MISMATCH" not in line
    # The fixture's own `final_status` is printed on
    # the same line, so a reader sees at a glance whether the demoted claim actually made
    # it into the final, delivered report.
    assert "final_status=not_in_report" in line


def test_compare_with_expected_tolerances():
    summary = {
        "screening": {
            "flow": {"identified": 130, "included": 10},
            "provenance": {"screener": {"model_reported": ["m1"]}},
        },
        "writing": {"model_reported": "m1"},
        "verification": {
            "provenance": {"model_reported": ["m2"]},
            "claims": [{"id": "a", "status": "verified"}],
        },
    }
    expected = {
        "screening": {
            "flow": {"identified": 100, "included": 20, "stop_reason": "x"},
            "provenance": {"screener": {"model_reported": ["m1"]}},
        },
        "writing": {"model_reported": "m1"},
        "verification": {
            "provenance": {"model_reported": ["m1"]},
            "claims": [{"id": "a", "status": "unsupported"}],
        },
    }
    checks = {c["check"]: c["ok"] for c in compare_with_expected(summary, expected)}
    assert checks == {
        "screening.flow.identified": True,  # 130 is exactly +30 %
        "screening.flow.included": False,
        "screening.model_reported": True,
        "writing.model_reported": True,
        "verification.model_reported": False,
        "verification.claims.a.status": False,
    }
    # a --skip-search run never compares screening
    checks = compare_with_expected({**summary, "screening": None}, expected)
    assert not any(c["check"].startswith("screening") for c in checks)
    # no citation_coverage on either side: no coverage checks are added at all
    assert not any(c["check"].startswith("verification.citation_coverage") for c in checks)


def test_compare_with_expected_checks_citation_coverage_within_tolerance():
    expected = {
        "verification": {
            "citation_coverage": {"found": 38, "linked": 34, "unresolved": 4},
        },
    }
    summary = {
        "verification": {
            "citation_coverage": {"found": 40, "linked": 32, "unresolved": 4},
        },
    }
    checks = {c["check"]: c["ok"] for c in compare_with_expected(summary, expected)}
    assert checks == {
        "verification.citation_coverage.found": True,
        "verification.citation_coverage.linked": True,
        "verification.citation_coverage.unresolved": True,
    }


def test_compare_with_expected_flags_citation_coverage_outside_tolerance():
    expected = {
        "verification": {
            "citation_coverage": {"found": 38, "linked": 34, "unresolved": 4},
        },
    }
    summary = {
        "verification": {
            # found and linked both drift far past +-30 % (or +-2)
            "citation_coverage": {"found": 10, "linked": 5, "unresolved": 4},
        },
    }
    checks = {c["check"]: c["ok"] for c in compare_with_expected(summary, expected)}
    assert checks["verification.citation_coverage.found"] is False
    assert checks["verification.citation_coverage.linked"] is False
    assert checks["verification.citation_coverage.unresolved"] is True


def test_compare_with_expected_unresolved_never_exceeds_baseline_even_in_tolerance():
    """A rise in unresolved is a coverage regression even
    when the relative +-30 %/+-2 band would otherwise call it in tolerance. The other
    side is one-sided: a fall below the baseline is
    never drift, at any size -- a fresh run that resolves every citation the baseline
    left unresolved must not be reported as DRIFT for improving."""
    expected = {
        "verification": {
            "citation_coverage": {"found": 38, "linked": 34, "unresolved": 4},
        },
    }
    # 5 is within +-30 %/+-2 of 4 (max(1.2, 2) = 2), but it is still a regression.
    summary_regressed = {
        "verification": {
            "citation_coverage": {"found": 38, "linked": 33, "unresolved": 5},
        },
    }
    checks = {
        c["check"]: c["ok"] for c in compare_with_expected(summary_regressed, expected)
    }
    assert checks["verification.citation_coverage.unresolved"] is False
    # a small drop is fine
    summary_slightly_better = {
        "verification": {
            "citation_coverage": {"found": 38, "linked": 34, "unresolved": 3},
        },
    }
    checks_slightly_better = {
        c["check"]: c["ok"]
        for c in compare_with_expected(summary_slightly_better, expected)
    }
    assert checks_slightly_better["verification.citation_coverage.unresolved"] is True
    # a drop to zero -- far outside the relative tolerance band `found`/`linked` use --
    # is still never drift: resolving every citation the baseline left unresolved is an
    # improvement, not a regression.
    summary_far_better = {
        "verification": {
            "citation_coverage": {"found": 38, "linked": 38, "unresolved": 0},
        },
    }
    checks_far_better = {
        c["check"]: c["ok"] for c in compare_with_expected(summary_far_better, expected)
    }
    assert checks_far_better["verification.citation_coverage.unresolved"] is True


def test_compare_with_expected_citation_coverage_missing_from_run_is_drift():
    """A backend build too old to report citation_coverage, or a run where the
    linker call itself failed, must show up as DRIFT rather than being skipped."""
    expected = {
        "verification": {
            "citation_coverage": {"found": 38, "linked": 34, "unresolved": 4},
        },
    }
    summary = {"verification": {}}
    checks = {c["check"]: c["ok"] for c in compare_with_expected(summary, expected)}
    assert checks == {
        "verification.citation_coverage.found": False,
        "verification.citation_coverage.linked": False,
        "verification.citation_coverage.unresolved": False,
    }


def test_compare_with_expected_passes_when_every_claim_text_is_in_the_draft():
    expected: dict[str, Any] = {}
    summary = {"verification": {"claim_text_in_draft": "40 of 40"}}
    checks = {c["check"]: c["ok"] for c in compare_with_expected(summary, expected)}
    assert checks["verification.claim_text_in_draft"] is True


def test_compare_with_expected_fails_when_a_claim_text_is_missing_from_the_draft():
    """This is a self-consistency check on the run itself, so it applies
    (and can fail) even against an *expected* baseline that carries no such field at all,
    unlike the tolerance-band checks above."""
    expected: dict[str, Any] = {}
    summary = {"verification": {"claim_text_in_draft": "39 of 40"}}
    checks = {c["check"]: c["ok"] for c in compare_with_expected(summary, expected)}
    assert checks["verification.claim_text_in_draft"] is False


def test_compare_with_expected_claim_text_in_draft_missing_is_not_compared():
    """An older summary with no `claim_text_in_draft` field at all: skipped, not
    reported as drift."""
    expected: dict[str, Any] = {}
    summary = {"verification": {}}
    checks = {c["check"] for c in compare_with_expected(summary, expected)}
    assert "verification.claim_text_in_draft" not in checks


def test_compare_with_expected_checks_by_source_mapping_within_tolerance():
    expected = {
        "verification": {
            "citation_coverage": {
                "found": 38,
                "linked": 38,
                "unresolved": 0,
                "by_source": {"mapping": 34, "author-year": 4, "numbered": 0},
            },
        },
    }
    summary = {
        "verification": {
            "citation_coverage": {
                "found": 38,
                "linked": 38,
                "unresolved": 0,
                "by_source": {"mapping": 32, "author-year": 6, "numbered": 0},
            },
        },
    }
    checks = {c["check"]: c["ok"] for c in compare_with_expected(summary, expected)}
    assert checks["verification.citation_coverage.by_source.mapping"] is True


def test_compare_with_expected_flags_a_collapsed_citation_link_map():
    """Since the audit-key fallback
    keys every occurrence a failed linker drops as author-year, `found`, `linked` and
    `unresolved` all stay unmoved when the citation-link call fails entirely -- the exact
    scenario the coverage drift check was written to catch, and which the fallback
    reopened. `by_source.mapping` collapsing to 0 is the count that actually moves, and
    must now be reported as DRIFT."""
    expected = {
        "verification": {
            "citation_coverage": {
                "found": 38,
                "linked": 38,
                "unresolved": 0,
                "by_source": {"mapping": 34, "author-year": 4, "numbered": 0},
            },
        },
    }
    # Every citationLinks attribute stripped: a total linker failure.
    summary = {
        "verification": {
            "citation_coverage": {
                "found": 38,
                "linked": 38,
                "unresolved": 0,
                "by_source": {"mapping": 0, "author-year": 38, "numbered": 0},
            },
        },
    }
    checks = {c["check"]: c["ok"] for c in compare_with_expected(summary, expected)}
    assert checks["verification.citation_coverage.found"] is True
    assert checks["verification.citation_coverage.linked"] is True
    assert checks["verification.citation_coverage.unresolved"] is True
    assert checks["verification.citation_coverage.by_source.mapping"] is False


def test_compare_with_expected_by_source_mapping_missing_is_not_compared():
    """No `by_source` on either side (an older baseline, or a build too old to report
    it): the check is skipped rather than reported as drift, matching how the other
    coverage checks behave when `citation_coverage` itself is absent from *expected*."""
    expected = {"verification": {"citation_coverage": {"found": 38, "linked": 34}}}
    summary = {"verification": {"citation_coverage": {"found": 38, "linked": 34}}}
    checks = {c["check"] for c in compare_with_expected(summary, expected)}
    assert "verification.citation_coverage.by_source.mapping" not in checks


def test_compare_with_expected_checks_writing_total_calls_exactly():
    expected = {"writing": {"total_calls": 2}}
    summary_match = {"writing": {"total_calls": 2}}
    summary_drift = {"writing": {"total_calls": 1}}
    checks_match = {
        c["check"]: c["ok"] for c in compare_with_expected(summary_match, expected)
    }
    checks_drift = {
        c["check"]: c["ok"] for c in compare_with_expected(summary_drift, expected)
    }
    assert checks_match["writing.total_calls"] is True
    assert checks_drift["writing.total_calls"] is False


def test_compare_with_expected_writing_total_calls_missing_is_not_compared():
    expected = {"writing": {"model_reported": "m1"}}
    summary = {"writing": {"model_reported": "m1"}}
    checks = {c["check"] for c in compare_with_expected(summary, expected)}
    assert "writing.total_calls" not in checks


def _rounds(query_lists: list[list[str]]) -> list[dict[str, Any]]:
    return [{"queries": [{"query": q} for q in qs]} for qs in query_lists]


def test_compare_with_expected_replay_queries_reports_identical_lists():
    rounds = _rounds([["q1", "q2"], ["q3"]])
    expected = {"screening": {"provenance": {"rounds": rounds}}}
    summary = {"screening": {"provenance": {"rounds": rounds}}}
    checks = {c["check"]: c for c in compare_with_expected(summary, expected, replay_queries=True)}
    assert checks["screening.replay_queries"]["ok"] is True
    assert checks["screening.replay_queries"]["actual"] == "identical"


def test_compare_with_expected_replay_queries_lists_the_differences():
    expected = {"screening": {"provenance": {"rounds": _rounds([["q1", "q2"], ["q3"]])}}}
    summary = {"screening": {"provenance": {"rounds": _rounds([["q1", "different"], ["q3"]])}}}
    checks = {c["check"]: c for c in compare_with_expected(summary, expected, replay_queries=True)}
    check = checks["screening.replay_queries"]
    assert check["ok"] is False
    assert check["actual"] != "identical"
    assert "different" in check["actual"] or "round" in check["actual"].lower()


def test_compare_with_expected_replay_queries_off_by_default():
    expected = {"screening": {"provenance": {"rounds": _rounds([["q1"]])}}}
    summary = {"screening": {"provenance": {"rounds": _rounds([["q2"]])}}}
    checks = {c["check"] for c in compare_with_expected(summary, expected)}
    assert "screening.replay_queries" not in checks


def test_compare_with_expected_search_only_summary_skips_writing_and_verification():
    """A `--search-only` summary carries `writing` and
    `verification` both `None` (nothing past screening ever ran). Comparing it
    against a full baseline must not report every writing and verification check as
    drift with `actual: None`, which would make `--replay-queries --search-only
    --strict` exit 3 on a run that behaved exactly as documented."""
    expected = {
        "screening": {"flow": {"identified": 100}, "provenance": {}},
        "writing": {"model_reported": "m1", "total_calls": 2},
        "verification": {
            "provenance": {"model_reported": ["m1"]},
            "claims": [{"id": "a", "status": "verified"}],
            "citation_coverage": {"found": 5, "linked": 5},
        },
    }
    search_only_summary = {
        "screening": {"flow": {"identified": 100}, "provenance": {}},
        "seeds": None,
        "fulltext": None,
        "writing": None,
        "verification": None,
    }

    checks = compare_with_expected(search_only_summary, expected)

    assert not any(c["check"].startswith("writing") for c in checks)
    assert not any(c["check"].startswith("verification") for c in checks)
    assert any(c["check"].startswith("screening") for c in checks)


def test_format_summary_table_handles_missing_sections():
    text = format_summary_table(
        {
            "seeds": None,
            "verification": {
                "claims": [
                    {
                        "id": "x",
                        "status": "verified",
                        "expected_status": "verified",
                        "matches_expected": True,
                    }
                ]
            },
        },
        None,
    )
    assert "skipped (--skip-search)" in text
    assert "claim.x.status" in text and "no expected/summary.json" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_exit_codes(protocol, seeds, tmp_path, monkeypatch):
    monkeypatch.setattr(run_demo, "openalex_resolver", lambda *a, **k: fake_resolver)
    monkeypatch.setattr(run_demo.time, "sleep", lambda s: None)
    server = FakeServer()
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
        str(tmp_path / "run1"),
        "--timeout",
        "60",
    ]
    rc = run_demo.main(
        argv,
        client_factory=lambda url: ApiClient(url, transport=httpx.MockTransport(server.handle)),
    )
    assert rc == 0
    assert (tmp_path / "run1" / "summary.json").exists()

    failing = FakeServer()
    failing.task_outcome = "failed"
    rc = run_demo.main(
        argv,
        client_factory=lambda url: ApiClient(url, transport=httpx.MockTransport(failing.handle)),
    )
    assert rc == 1

    rc = run_demo.main(
        ["--protocol", str(tmp_path / "missing.json"), "--seeds", str(FIXTURES / "seed_dois.json")]
    )
    assert rc == 1


def test_main_reports_delivered_check_ok_on_a_clean_run(protocol, seeds, tmp_path, monkeypatch):
    """With ``uncited_sentences`` present in the write-result
    fixture (so rule 1 is actually checked, not notice-skipped for an "older shape"),
    a clean run still reports ``delivered_check.ok`` true, with no violations, and
    exits 0 -- pinning that the gate itself does not manufacture a false positive out
    of ordinary text."""
    monkeypatch.setattr(run_demo, "openalex_resolver", lambda *a, **k: fake_resolver)
    monkeypatch.setattr(run_demo.time, "sleep", lambda s: None)
    server = FakeServer()
    argv = [
        "--api-url", API,
        "--protocol", str(FIXTURES / "protocol.json"),
        "--seeds", str(FIXTURES / "seed_dois.json"),
        "--expected", str(tmp_path / "no-expected"),
        "--output", str(tmp_path / "run-delivered-ok"),
        "--timeout", "60",
    ]
    rc = run_demo.main(
        argv,
        client_factory=lambda url: ApiClient(url, transport=httpx.MockTransport(server.handle)),
    )
    assert rc == 0
    summary = json.loads(
        (tmp_path / "run-delivered-ok" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["delivered_check"]["ok"] is True
    assert summary["delivered_check"]["violations"] == []


def test_main_returns_2_on_an_injected_orphan_sentence(protocol, seeds, tmp_path, monkeypatch):
    """A sentence the write result neither links nor tags at all
    is exactly the gap the offline checker exists to catch -- ``main()`` must return
    ``EXIT_CLAIMS`` (2), with the violated rule's own name present in
    ``summary["delivered_check"]["violations"]``, not silently succeed."""
    monkeypatch.setattr(run_demo, "openalex_resolver", lambda *a, **k: fake_resolver)
    monkeypatch.setattr(run_demo.time, "sleep", lambda s: None)
    server = FakeServer()
    server.writing_result = {
        **WRITING_RESULT,
        "content": WRITING_RESULT["content"]
        + "\n\nAn orphaned sentence nothing links or tags at all.",
    }
    argv = [
        "--api-url", API,
        "--protocol", str(FIXTURES / "protocol.json"),
        "--seeds", str(FIXTURES / "seed_dois.json"),
        "--expected", str(tmp_path / "no-expected"),
        "--output", str(tmp_path / "run-delivered-violation"),
        "--timeout", "60",
    ]
    rc = run_demo.main(
        argv,
        client_factory=lambda url: ApiClient(url, transport=httpx.MockTransport(server.handle)),
    )
    assert rc == run_demo.EXIT_CLAIMS
    summary = json.loads(
        (tmp_path / "run-delivered-violation" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["delivered_check"]["ok"] is False
    assert any(
        v["rule"] == run_demo.check_delivered.RULE_UNCLASSIFIED_SENTENCE
        for v in summary["delivered_check"]["violations"]
    )


def test_main_strict_returns_3_on_drift(protocol, seeds, tmp_path, monkeypatch):
    monkeypatch.setattr(run_demo, "openalex_resolver", lambda *a, **k: fake_resolver)
    monkeypatch.setattr(run_demo.time, "sleep", lambda s: None)
    expected_dir = tmp_path / "expected"
    expected_dir.mkdir()
    (expected_dir / "summary.json").write_text(
        json.dumps({"screening": {"flow": {"included": 100}}}), encoding="utf-8"
    )
    server = FakeServer()
    argv = [
        "--api-url",
        API,
        "--protocol",
        str(FIXTURES / "protocol.json"),
        "--seeds",
        str(FIXTURES / "seed_dois.json"),
        "--expected",
        str(expected_dir),
        "--output",
        str(tmp_path / "run"),
    ]
    factory = lambda url: ApiClient(url, transport=httpx.MockTransport(server.handle))  # noqa: E731
    assert run_demo.main(argv, client_factory=factory) == 0
    assert run_demo.main(argv + ["--strict"], client_factory=factory) == 3


def test_main_search_only_exits_0_with_no_claims_to_check(protocol, seeds, tmp_path, monkeypatch):
    """--search-only never triggers the claim-mismatch exit code (2), since no
    claim verification ever runs."""
    monkeypatch.setattr(run_demo, "openalex_resolver", lambda *a, **k: fake_resolver)
    monkeypatch.setattr(run_demo.time, "sleep", lambda s: None)
    server = FakeServer()
    argv = [
        "--api-url", API,
        "--protocol", str(FIXTURES / "protocol.json"),
        "--seeds", str(FIXTURES / "seed_dois.json"),
        "--expected", str(tmp_path / "no-expected"),
        "--output", str(tmp_path / "run"),
        "--timeout", "60",
        "--search-only",
    ]
    rc = run_demo.main(
        argv,
        client_factory=lambda url: ApiClient(url, transport=httpx.MockTransport(server.handle)),
    )
    assert rc == 0
    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert summary["screening"]["flow"]["included"] == 15
    assert summary["verification"] is None


def test_main_replay_queries_forwards_the_baselines_queries(protocol, seeds, tmp_path, monkeypatch):
    monkeypatch.setattr(run_demo, "openalex_resolver", lambda *a, **k: fake_resolver)
    monkeypatch.setattr(run_demo.time, "sleep", lambda s: None)
    expected_dir = tmp_path / "expected"
    expected_dir.mkdir()
    (expected_dir / "screening_record.json").write_text(
        json.dumps(
            {"provenance": {"rounds": [{"queries": [{"query": "baseline q1"}]}]}}
        ),
        encoding="utf-8",
    )
    server = FakeServer()
    argv = [
        "--api-url", API,
        "--protocol", str(FIXTURES / "protocol.json"),
        "--seeds", str(FIXTURES / "seed_dois.json"),
        "--expected", str(expected_dir),
        "--output", str(tmp_path / "run"),
        "--timeout", "60",
        "--replay-queries",
    ]
    rc = run_demo.main(
        argv,
        client_factory=lambda url: ApiClient(url, transport=httpx.MockTransport(server.handle)),
    )
    assert rc == 0
    body = [b for m, p, b in server.calls if p == f"/projects/{PROJECT_ID}/smart-search"][0]
    assert body["queries_override"] == [["baseline q1"]]


def test_search_only_and_skip_search_together_is_a_usage_error():
    with pytest.raises(SystemExit):
        run_demo.build_parser().parse_args(["--search-only", "--skip-search"])


def test_default_search_timeout_covers_the_jobs_worst_case_with_margin():
    """The search loop's own budget is 1800 s (smart_search_max_time_minutes = 30), and
    the judge stage runs, per job, as a concurrent stage with that same 1800 s budget
    plus at most one more 420 s call: a worst case of 1800 + 1800 + 420 = 4020 s.
    DEFAULT_SEARCH_TIMEOUT_S must clear that with margin, and DEFAULT_TIMEOUT_S (every
    other step's own wait) must stay exactly as it was."""
    worst_case_job_ceiling_s = 1800 + 1800 + 420
    assert run_demo.DEFAULT_SEARCH_TIMEOUT_S > worst_case_job_ceiling_s
    assert run_demo.DEFAULT_SEARCH_TIMEOUT_S == 4500
    assert run_demo.DEFAULT_TIMEOUT_S == 40 * 60


def test_search_timeout_cli_option_defaults_to_the_named_constant():
    args = run_demo.build_parser().parse_args([])
    assert args.search_timeout == run_demo.DEFAULT_SEARCH_TIMEOUT_S


def test_search_timeout_cli_option_can_be_overridden():
    args = run_demo.build_parser().parse_args(["--search-timeout", "600"])
    assert args.search_timeout == 600.0


def test_main_forwards_search_timeout_to_demo_runner(protocol, seeds, tmp_path, monkeypatch):
    """--search-timeout and --timeout must reach DemoRunner as two distinct values, not
    collapse onto the one constructor argument every step used to share."""
    captured: dict[str, Any] = {}
    real_runner_cls = run_demo.DemoRunner

    class SpyRunner(real_runner_cls):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["timeout_s"] = kwargs.get("timeout_s")
            captured["search_timeout_s"] = kwargs.get("search_timeout_s")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(run_demo, "DemoRunner", SpyRunner)
    monkeypatch.setattr(run_demo, "openalex_resolver", lambda *a, **k: fake_resolver)
    monkeypatch.setattr(run_demo.time, "sleep", lambda s: None)
    server = FakeServer()
    argv = [
        "--api-url", API,
        "--protocol", str(FIXTURES / "protocol.json"),
        "--seeds", str(FIXTURES / "seed_dois.json"),
        "--expected", str(tmp_path / "no-expected"),
        "--output", str(tmp_path / "run"),
        "--timeout", "111",
        "--search-timeout", "222",
        "--search-only",
    ]
    rc = run_demo.main(
        argv,
        client_factory=lambda url: ApiClient(url, transport=httpx.MockTransport(server.handle)),
    )
    assert rc == 0
    assert captured["timeout_s"] == 111.0
    assert captured["search_timeout_s"] == 222.0


# --------------------------------------------------------------------------------------
# --sections, sections.json loading/validation, and the extra-sections
# step of the run.
# --------------------------------------------------------------------------------------


def test_sections_flag_defaults_to_the_repo_sections_json_when_it_exists():
    """`demo/sections.json` is a real, tracked file, so the default must resolve to it
    (not None) on an ordinary checkout."""
    args = run_demo.build_parser().parse_args([])
    assert args.sections == run_demo.DEFAULT_SECTIONS
    assert run_demo.DEFAULT_SECTIONS.exists()


def test_sections_flag_defaults_to_none_when_the_default_file_is_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(run_demo, "DEFAULT_SECTIONS", tmp_path / "no-such-sections.json")
    args = run_demo.build_parser().parse_args([])
    assert args.sections is None


def test_sections_flag_can_be_overridden_explicitly():
    args = run_demo.build_parser().parse_args(["--sections", "custom-sections.json"])
    assert args.sections == Path("custom-sections.json")


def test_load_json_array_reads_a_list(tmp_path):
    path = tmp_path / "sections.json"
    path.write_text('[{"title": "A"}, {"title": "B"}]', encoding="utf-8")
    assert run_demo._load_json_array(path, "sections.json") == [{"title": "A"}, {"title": "B"}]


def test_load_json_array_raises_when_the_file_is_missing(tmp_path):
    with pytest.raises(DemoError, match="not found"):
        run_demo._load_json_array(tmp_path / "missing.json", "sections.json")


def test_load_json_array_raises_on_invalid_json(tmp_path):
    path = tmp_path / "sections.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(DemoError, match="not valid JSON"):
        run_demo._load_json_array(path, "sections.json")


def test_load_json_array_raises_when_the_top_level_is_not_a_list(tmp_path):
    path = tmp_path / "sections.json"
    path.write_text('{"title": "A"}', encoding="utf-8")
    with pytest.raises(DemoError, match="must be a JSON array"):
        run_demo._load_json_array(path, "sections.json")


def test_validate_sections_passes_on_well_formed_entries():
    run_demo._validate_sections(
        [{"title": "A", "instructions": "Write about A.", "target_words": 400}]
    )


def test_validate_sections_raises_when_a_title_is_missing():
    with pytest.raises(DemoError, match="'title'"):
        run_demo._validate_sections([{"instructions": "Write about A."}])


def test_validate_sections_raises_when_instructions_are_missing():
    with pytest.raises(DemoError, match="'instructions'"):
        run_demo._validate_sections([{"title": "A"}])


def test_the_repo_sections_json_is_well_formed_and_has_six_entries():
    """Guards the real, shipped `demo/sections.json`: six
    sections, sized from the measured yield of about five delivered sentences per
    400-word section, so the protocol section's own six rows plus six extra
    sections' own five each land the Delivered sheet at about 35 rows.
    Each entry carries a title, instructions in protocol.json's own style, and
    target_words 400, on the same project and papers as the protocol section."""
    sections = run_demo._load_json_array(run_demo.DEFAULT_SECTIONS, "sections.json")
    run_demo._validate_sections(sections)
    assert len(sections) == 6
    titles = [s["title"] for s in sections]
    assert len(set(titles)) == 6
    for section in sections:
        assert section["target_words"] == 400
        assert "[NEEDS CITATION]" in section["instructions"]
        assert "APA 7" in section["instructions"]


# --------------------------------------------------------------------------------------
# DemoRunner._write_one_extra_section / _write_extra_sections. Each of
# the module-level step functions is monkeypatched directly rather than routed through
# FakeServer, which models exactly one draft; these two methods run the identical write
# -> verify-and-heal sequence on a SECOND (and third, and fourth) draft of the same
# project, so patching the functions is far cheaper than teaching the shared fixture a
# second draft id -- and keeps the many other tests built on it untouched.
# --------------------------------------------------------------------------------------


class _BareClient:
    """Stands in for ``self.client``: every function `_write_one_extra_section` calls
    is itself monkeypatched, so this is never actually used to make an HTTP call --
    only ``DemoRunner.__init__`` reads ``base_url`` off it, to build ``state.api_url``."""

    base_url = "http://example.invalid/api/v1"


def _make_bare_runner(tmp_path: Path) -> DemoRunner:
    runner = DemoRunner(
        client=_BareClient(),
        protocol={
            "section": {"title": "Protocol Section", "instructions": "x", "target_words": 400}
        },
        seeds={"papers": []},
        output_dir=tmp_path,
        expected_dir=None,
        timeout_s=1.0,
    )
    runner.state.project_id = PROJECT_ID
    return runner


def _patch_one_extra_section_calls(
    monkeypatch, *, draft_id: str, verified_sentence: str, healed_sentence: str | None = None
):
    """Patches every module-level function `_write_one_extra_section` calls with a
    fixed, self-consistent fake for one section carrying exactly one verified claim
    over one citation, and returns the call log (in order) for assertions.

    ``healed_sentence``, when given, makes ``fake_fetch_draft``
    return DIFFERENT content on its second call (the post-heal fetch) than its first (the
    pre-heal fetch that happens right after the write job) -- the shape a real heal
    that actually changes the text takes. Without this, ``fake_fetch_draft`` would
    return the same content on both calls, masking a defect where the pre-heal file
    is never re-written."""
    calls: list[tuple[str, Any]] = []

    def fake_create_draft(client, project_id, protocol):
        calls.append(("create_draft", protocol["section"]["title"]))
        return draft_id

    def fake_start_generation(client, this_draft_id, protocol, *, section_title=None):
        calls.append(("start_generation", (this_draft_id, section_title)))
        return f"task-write-{this_draft_id}"

    def fake_poll_task(client, task_id, *, label, timeout_s, sleep, clock, log, timeout_flag="--timeout"):
        calls.append(("poll", task_id))
        if task_id == f"task-write-{draft_id}":
            return {
                "status": "completed",
                "result": {
                    "content": verified_sentence,
                    "section_type": "literature_review",
                    "loop_stats": {"sentences_removed_unverified": 0},
                    "provenance": {"length": {"final_words": 42}},
                },
            }
        return {"status": "completed", "result": {}}

    fetch_draft_calls = {"count": 0}

    def fake_fetch_draft(client, this_draft_id):
        calls.append(("fetch_draft", this_draft_id))
        fetch_draft_calls["count"] += 1
        sentence = verified_sentence
        if healed_sentence is not None and fetch_draft_calls["count"] > 1:
            sentence = healed_sentence
        return {
            "content": {
                "type": "doc",
                "content": [
                    {
                        "type": "heading",
                        "attrs": {"level": 2},
                        "content": [{"type": "text", "text": "Extra Section"}],
                    },
                    {"type": "paragraph", "content": [{"type": "text", "text": sentence}]},
                ],
            }
        }

    def fake_start_verification(client, project_id, this_draft_id):
        calls.append(("start_verification", this_draft_id))
        return f"task-verify-{this_draft_id}"

    def fake_fetch_claim_report(client, this_draft_id):
        calls.append(("fetch_claim_report", this_draft_id))
        verification = {
            "claim_text": "Body claim.",
            "claim_sentence": healed_sentence if healed_sentence is not None else verified_sentence,
            "citation": "(Smith, 2020)",
            "paper_id": "p1",
            "status": "verified",
            "evidence_quote": "text",
            "evidence_quotes": ["text"],
            "evidence_location": "chunk 1 (results)",
            "paper_doi": "10.5555/demo-001",
            "paper_title": "Fixture paper",
        }
        return {
            "verified_count": 1,
            "unsupported_count": 0,
            "nuance_count": 0,
            "abstract_only_count": 0,
            "error_count": 0,
            "citation_coverage": {"unresolved": 0},
            "final_report": {"verifications": [verification], "verified_count": 1},
        }

    def fake_fetch_fulltext_chunk(client, paper_id, chunk_index):
        return {
            "paper_id": paper_id, "chunk_index": chunk_index, "section": "results", "text": "text",
        }

    monkeypatch.setattr(run_demo, "create_draft", fake_create_draft)
    monkeypatch.setattr(run_demo, "start_generation", fake_start_generation)
    monkeypatch.setattr(run_demo, "poll_task", fake_poll_task)
    monkeypatch.setattr(run_demo, "fetch_draft", fake_fetch_draft)
    monkeypatch.setattr(run_demo, "start_verification", fake_start_verification)
    monkeypatch.setattr(run_demo, "fetch_claim_report", fake_fetch_claim_report)
    monkeypatch.setattr(run_demo, "fetch_fulltext_chunk", fake_fetch_fulltext_chunk)
    return calls


def test_write_one_extra_section_writes_files_and_returns_a_summary(monkeypatch, tmp_path):
    runner = _make_bare_runner(tmp_path)
    section = {
        "title": "Revision behaviour after written corrective feedback",
        "instructions": "Write about revision behaviour.",
        "target_words": 400,
    }
    verified_sentence = "Learners revised more accurately after feedback (Smith, 2020)."
    calls = _patch_one_extra_section_calls(
        monkeypatch, draft_id="draft-2", verified_sentence=verified_sentence
    )

    result = runner._write_one_extra_section(2, section)

    assert [name for name, _detail in calls] == [
        "create_draft", "start_generation", "poll",
        "fetch_draft", "start_verification", "poll", "fetch_claim_report", "fetch_draft",
    ]
    # The section's own requested title is sent through to
    # the write job, not left for the backend to derive from section_type alone.
    start_generation_detail = next(detail for name, detail in calls if name == "start_generation")
    assert start_generation_detail == ("draft-2", section["title"])
    assert result["title"] == section["title"]
    assert result["draft_id"] == "draft-2"
    # Both recomputed from the healed draft actually saved
    # (the second `fetch_draft` call above), not the write job's own pre-heal
    # `loop_stats`/``provenance.length.final_words`` (0 and 42 respectively here) --
    # `verified_sentence` is 8 words, and `target_words` is 400.
    assert result["writing"]["loop_stats"] == {
        "sentences_removed_unverified": 0,
        "survival_rate": 8 / 400,
    }
    assert result["writing"]["final_words"] == 8
    assert result["verification"]["verified_count"] == 1
    assert result["verification"]["final_report_verified_count"] == 1
    assert result["delivered_evidence"] == {"rows": 1, "unlocated_rows": 0}

    assert (tmp_path / "writing_result_2.json").exists()
    assert (tmp_path / "draft_content_2.json").exists()
    assert (tmp_path / "claim_report_2.json").exists()

    assert len(runner._extra_section_delivered_records) == 1
    record = runner._extra_section_delivered_records[0]
    assert record["draft_id"] == "draft-2"
    assert record["section_title"] == section["title"]
    assert record["rows"][0]["claim_text"] == "Body claim."


def test_write_one_extra_section_saves_the_healed_draft_not_the_pre_heal_one(monkeypatch, tmp_path):
    """``draft_content_<index>.json`` must reflect the draft
    AFTER the standalone verify-and-heal action, not the one fetched right after the
    write job. ``fake_fetch_draft`` here returns different text on its second call (the
    post-heal fetch) than its first, exactly the shape a heal that rewrites a sentence
    in place takes; the file on disk must not keep the FIRST fetch's text."""
    runner = _make_bare_runner(tmp_path)
    section = {
        "title": "Revision behaviour after written corrective feedback",
        "instructions": "Write about revision behaviour.",
        "target_words": 400,
    }
    pre_heal_sentence = "Learners revised more accurately after feedback (Smith, 2020)."
    healed_sentence = "Learners revised somewhat more accurately after feedback (Smith, 2020)."
    _patch_one_extra_section_calls(
        monkeypatch,
        draft_id="draft-2",
        verified_sentence=pre_heal_sentence,
        healed_sentence=healed_sentence,
    )

    runner._write_one_extra_section(2, section)

    saved = json.loads((tmp_path / "draft_content_2.json").read_text(encoding="utf-8"))
    saved_text = " ".join(
        n.get("text", "")
        for node in saved["content"]
        for n in node.get("content") or []
        if n.get("type") == "text"
    )
    assert healed_sentence in saved_text
    assert pre_heal_sentence not in saved_text


def test_write_one_extra_section_raises_on_empty_generated_content(monkeypatch, tmp_path):
    runner = _make_bare_runner(tmp_path)
    section = {"title": "Empty Section", "instructions": "x", "target_words": 400}
    _patch_one_extra_section_calls(monkeypatch, draft_id="draft-3", verified_sentence="")

    with pytest.raises(DemoError, match="empty content"):
        runner._write_one_extra_section(3, section)


def test_write_one_extra_section_enforces_the_exit_invariant(monkeypatch, tmp_path):
    """A [NEEDS CITATION] marker left in the healed draft must still raise here, exactly
    as it does for the protocol section's own draft (`_verify_claims`)."""
    runner = _make_bare_runner(tmp_path)
    section = {"title": "Broken Section", "instructions": "x", "target_words": 400}
    calls = _patch_one_extra_section_calls(
        monkeypatch, draft_id="draft-4", verified_sentence="Unfinished [NEEDS CITATION]."
    )

    with pytest.raises(DemoError, match=r"\[NEEDS CITATION\]"):
        runner._write_one_extra_section(4, section)
    assert ("fetch_draft", "draft-4") in calls


def test_write_extra_sections_runs_every_section_starting_at_index_two(monkeypatch, tmp_path):
    runner = _make_bare_runner(tmp_path)
    sections = [
        {"title": "First Extra", "instructions": "x", "target_words": 400},
        {"title": "Second Extra", "instructions": "x", "target_words": 400},
    ]
    runner.sections = sections

    recorded: list[tuple[int, str]] = []

    def fake_write_one(index, section):
        recorded.append((index, section["title"]))
        return {"title": section["title"], "draft_id": f"draft-{index}"}

    monkeypatch.setattr(runner, "_write_one_extra_section", fake_write_one)

    runner._write_extra_sections()

    assert recorded == [(2, "First Extra"), (3, "Second Extra")]
    assert runner.state.sections == [
        {"title": "First Extra", "draft_id": "draft-2"},
        {"title": "Second Extra", "draft_id": "draft-3"},
    ]
