"""claims/verify_common: rows, resumable runs, deterministic short-circuit, meta, verifiers.

The LLM boundary is always faked: no ``app.*`` import from the real backend, no network.
One deliberate exception, `test_harness_guard_entry_point_has_parity_with_the_real_backend_
normaliser`, imports the real backend to prove guard parity; see its own docstring for why
and how it cleans up after itself.
"""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import common
import verify_common as vc
from common import read_jsonl

CHUNK = "Learners improved their scores. The effect was significant (p < .05)."

# Task authorisation 2026-09-10 (Part B, the bounded second pass): appended (string
# concatenation) to every fake ``app/services/fulltext.py`` body below, so
# ``ProductionVerifier`` -- which imports ``verify_claim_with_policy``/``ModelPassAnswer``
# from that module, never a re-implementation of its own -- has something real to call. This
# mirrors the real ``app.services.fulltext.verify_claim_with_policy`` exactly (call the
# model, relocate, guard, and -- exactly once, exactly when licensed -- call again), using
# whichever ``attribution_guard_fires_before_relocation``/``relocate_evidence_quotes``/
# ``apply_verification_guards`` that variant's own body defines.
_SHARED_POLICY_BODY = textwrap.dedent(
    """

    from types import SimpleNamespace

    VERIFICATION_POLICY_VERSION = "test-policy-version"


    class ModelPassAnswer:
        def __init__(self, **kw):
            self.__dict__.update(kw)


    class RepairRequest:
        def __init__(self, failed_segments):
            self.failed_segments = list(failed_segments)


    def _non_verbatim_segments(quotes, chunk_texts):
        failed = []
        for q in quotes:
            if q and not any(q in c for c in chunk_texts):
                failed.append(q)
        return failed


    async def verify_claim_with_policy(claim_text, chunk_texts, chunks, call_model):
        from app.agents.claim_verification_agent import QUOTE_REPAIR_PROMPT_VERSION

        passes = []
        answers = []
        last = {}

        async def _run_one_pass(repair_request):
            answer = await call_model(repair_request)
            answers.append(answer)
            evidence_quote = answer.evidence_quote
            evidence_quotes = list(answer.evidence_quotes)
            assertions = [dict(a) for a in answer.assertions]
            relocation_diagnostics = []
            if not attribution_guard_fires_before_relocation(
                claim_text, chunk_texts, evidence_quote
            ):
                evidence_quote, evidence_quotes, assertions, relocation_diagnostics = (
                    relocate_evidence_quotes(
                        evidence_quote, evidence_quotes, assertions, chunk_texts
                    )
                )
            status, machine_reasons, diagnostics = apply_verification_guards(
                answer.status,
                claim_text=claim_text,
                evidence_quote=evidence_quote,
                evidence_quotes=evidence_quotes,
                assertions=assertions,
                chunk_texts=chunk_texts,
                chunks=chunks,
            )
            if relocation_diagnostics:
                diagnostics = list(diagnostics) + ["quote_relocated"]
            passes.append({
                "model_status": answer.status,
                "quotes": evidence_quotes or ([evidence_quote] if evidence_quote else []),
                "machine_reasons": machine_reasons,
                "diagnostics": diagnostics,
                "tokens": {
                    "input_tokens": getattr(answer, "input_tokens", None),
                    "output_tokens": getattr(answer, "output_tokens", None),
                },
                "fingerprint": getattr(answer, "system_fingerprint", None),
                "repair_prompt_version": (
                    QUOTE_REPAIR_PROMPT_VERSION if repair_request is not None else None
                ),
            })
            last.update(
                status=status, machine_reasons=machine_reasons, diagnostics=diagnostics,
                evidence_quote=evidence_quote, evidence_quotes=evidence_quotes,
                assertions=assertions, quote_relocations=relocation_diagnostics,
            )

        await _run_one_pass(None)
        first_answer = answers[0]
        if (
            last["status"] == "needs_nuance"
            and list(last["machine_reasons"]) == ["quote_not_verbatim"]
            and first_answer.status == "verified"
        ):
            repaired_quotes = last["evidence_quotes"] or (
                [last["evidence_quote"]] if last["evidence_quote"] else []
            )
            failed = _non_verbatim_segments(repaired_quotes, chunk_texts)
            await _run_one_pass(RepairRequest(failed))

        final_answer = answers[-1]
        return SimpleNamespace(
            status=last["status"],
            machine_reasons=last["machine_reasons"],
            diagnostics=last["diagnostics"],
            evidence_quote=last["evidence_quote"],
            evidence_quotes=last["evidence_quotes"],
            assertions=last["assertions"],
            explanation=getattr(final_answer, "explanation", None),
            suggested_revision=getattr(final_answer, "suggested_revision", None),
            unstated_details=getattr(final_answer, "unstated_details", None) or [],
            model_status=final_answer.status,
            model_reported=getattr(final_answer, "model_reported", None),
            quote_relocations=last["quote_relocations"],
            passes=passes,
        )
    """
)


class FakeVerifier:
    model_configured = "deepseek-chat"
    prompt_version = "sha256:fakefakefake"
    temperature = 0.0

    def __init__(self, *, failures: int = 0, forbid: bool = False) -> None:
        self.failures = failures
        self.forbid = forbid
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, claim, chunks, title, authors):
        if self.forbid:
            raise AssertionError("verifier must not be called for deterministic items")
        self.calls.append({"claim": claim, "chunks": chunks, "title": title, "authors": authors})
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("boom")
        return {
            "predicted_status": "verified",
            "evidence_quote": "Learners improved their scores.",
            "explanation": "ok",
            "suggested_revision": None,
            "model_reported": "deepseek-v4-flash",
            "system_fingerprint": "fp-1",
            "provider_response_id": "resp-1",
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_tokens": 40,
            "usage_details": {"reasoning_tokens": 5},
        }


def item(i: int, chunks: list[str] | None = None) -> dict[str, Any]:
    return {
        "item_id": f"it-{i}",
        "claim": "Learners improved their scores.",
        "title": f"Paper {i}",
        "authors": ["A. Author"],
        "chunks": [CHUNK] if chunks is None else chunks,
    }


def run(items, verifier, out_path, **kw):
    kw.setdefault("concurrency", 2)
    kw.setdefault("max_attempts", 1)
    kw.setdefault("chunks_of", lambda it: list(it["chunks"]))
    return asyncio.run(vc.run_items(items, verifier, out_path=out_path, **kw))


# ---------------------------------------------------------------- make_row


def test_make_row_fidelity_flags():
    base = {"predicted_status": "verified"}
    row = vc.make_row(item(1), {**base, "evidence_quote": "improved their scores"}, CHUNK)
    assert row["quote_is_verbatim"] is True and row["quote_is_verbatim_casefold"] is True
    assert "chunks" not in row and "title" not in row and "authors" not in row

    row = vc.make_row(item(1), {**base, "evidence_quote": "improved  their\nscores"}, CHUNK)
    assert row["quote_is_verbatim"] is True

    row = vc.make_row(item(1), {**base, "evidence_quote": "learners improved"}, CHUNK)
    assert row["quote_is_verbatim"] is False and row["quote_is_verbatim_casefold"] is True

    row = vc.make_row(item(1), {**base, "evidence_quote": None}, CHUNK)
    assert row["quote_is_verbatim"] is None and row["quote_is_verbatim_casefold"] is None
    for key in vc.ROW_FIELDS:
        assert key in row


# ---------------------------------------------------------------- run_items


def test_run_items_resumes_and_records_called_at(tmp_path: Path):
    out = tmp_path / "run.jsonl"
    verifier = FakeVerifier()
    run([item(1)], verifier, out)
    assert len(verifier.calls) == 1
    progress = run([item(1), item(2), item(3)], verifier, out)
    assert progress["done"] == 2 and len(verifier.calls) == 3
    rows = read_jsonl(out)
    assert sorted(r["item_id"] for r in rows) == ["it-1", "it-2", "it-3"]
    assert len({r["item_id"] for r in rows}) == 3
    for r in rows:
        assert r["deterministic"] is False
        assert r["called_at"].endswith("+00:00") and "T" in r["called_at"]
        assert r["cache_read_tokens"] == 40 and r["usage_details"] == {"reasoning_tokens": 5}
        assert r["attempts"] == 1 and r["error"] is None and r["latency_s"] >= 0.0
    assert verifier.calls[0]["title"] == "Paper 1" and verifier.calls[0]["authors"] == ["A. Author"]


def test_run_items_short_circuits_empty_chunks_without_calling_verifier(tmp_path: Path):
    out = tmp_path / "run.jsonl"
    run([item(1, chunks=[])], FakeVerifier(forbid=True), out)
    (row,) = read_jsonl(out)
    assert row["predicted_status"] == "no_full_text"
    assert row["explanation"] == "No full-text chunks available for this paper."
    assert row["deterministic"] is True and row["latency_s"] == 0.0
    assert row["input_tokens"] is None and row["output_tokens"] is None
    assert row["cache_read_tokens"] is None and row["model_reported"] is None
    assert row["called_at"] and row["error"] is None and row["quote_is_verbatim"] is None


def test_run_items_retries_then_records_error(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(vc, "RETRY_BASE_DELAY", 0.0)
    out = tmp_path / "run.jsonl"
    verifier = FakeVerifier(failures=5)
    progress = run([item(1)], verifier, out, max_attempts=2)
    assert progress["errors"] == 1 and len(verifier.calls) == 2
    (row,) = read_jsonl(out)
    assert row["predicted_status"] == "error" and row["attempts"] == 2
    assert row["error"].startswith("RuntimeError: boom") and row["deterministic"] is False
    assert row["called_at"]

    verifier = FakeVerifier(failures=1)
    out2 = tmp_path / "run2.jsonl"
    run([item(2)], verifier, out2, max_attempts=3)
    (row,) = read_jsonl(out2)
    assert row["predicted_status"] == "verified" and row["attempts"] == 2


# ---------------------------------------------------------------- build_meta


def _row(status, called_at, inp, out, cache=0, *, deterministic=False, error=None):
    return {
        "item_id": f"{status}-{called_at}",
        "predicted_status": status,
        "called_at": called_at,
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_tokens": cache,
        "model_reported": None if deterministic or error else "deepseek-v4-flash",
        "system_fingerprint": None if deterministic or error else "fp-1",
        "latency_s": None if error else (0.0 if deterministic else 1.0),
        "deterministic": deterministic,
        "error": error,
    }


def test_build_meta_sums_cost_per_call_and_excludes_deterministic_and_error_rows(monkeypatch):
    seen: list[tuple] = []

    def fake_call_cost(model, called_at, input_tokens, output_tokens, cache_read_tokens=0):
        seen.append((model, called_at, input_tokens, output_tokens, cache_read_tokens))
        peak = called_at.startswith("2026-09-02T02")  # Wednesday 02:00 UTC = peak
        return (0.44 if peak else 0.22) * input_tokens / 1e6 + cache_read_tokens * 1e-9

    monkeypatch.setattr(common, "call_cost", fake_call_cost, raising=False)
    rows = [
        _row("verified", "2026-09-02T02:00:00+00:00", 1000, 10, cache=100),
        _row("unsupported", "2026-09-02T12:00:00+00:00", 2000, 20, cache=0),
        _row("no_full_text", "2026-09-02T12:00:01+00:00", None, None, None, deterministic=True),
        _row("error", "2026-09-02T12:00:02+00:00", None, None, None, error="RuntimeError: x"),
    ]
    verifier = FakeVerifier()
    price = common.price_record("deepseek-chat", None, None)
    meta = vc.build_meta(
        rows,
        existing=None,
        name="scifact",
        run="A",
        verifier=verifier,
        price=price,
        dry_run=False,
        session_started="2026-09-02T12:00:00+00:00",
        n_items=4,
        limit=None,
        concurrency=8,
    )
    assert len(seen) == 2 and seen[0][0] == "deepseek-v4-flash" and seen[0][4] == 100
    expected = 0.44 * 1000 / 1e6 + 100e-9 + 0.22 * 2000 / 1e6
    assert meta["total_cost"] == pytest.approx(expected)
    assert meta["cost_basis"] == "list price, tier by call time, cache-hit tokens at cache-hit rate"
    assert meta["total_input_tokens"] == 3000 and meta["total_output_tokens"] == 30
    assert meta["total_cache_read_tokens"] == 100
    assert meta["n_deterministic"] == 1 and meta["n_errors"] == 1 and meta["n_rows"] == 4
    assert meta["total_cost_flat"] is None
    assert meta["prompt_version"] == "sha256:fakefakefake" and meta["temperature"] == 0.0
    assert meta["status_counts"]["no_full_text"] == 1
    assert meta["latency_s"]["n"] == 2

    price2 = common.price_record("deepseek-chat", 1.0, 2.0)
    meta2 = vc.build_meta(
        rows, existing=meta, name="scifact", run="A", verifier=verifier, price=price2,
        dry_run=False, session_started="x", n_items=4, limit=None, concurrency=8,
    )
    assert meta2["total_cost_flat"] == pytest.approx((3000 * 1.0 + 30 * 2.0) / 1e6)
    assert len(meta2["sessions"]) == 2


# `build_meta`'s ``guard_digest``/``quote_relocation_version`` keys and `DryRunVerifier`'s
# two ``"dry-run"`` class attributes had no test, so a misspelling of either key would
# silently write ``null`` into a v4 run's meta file without any test noticing.


def test_build_meta_reads_guard_digest_and_quote_relocation_version_off_the_verifier():
    verifier = SimpleNamespace(
        model_configured="deepseek-chat",
        prompt_version="sha256:fakefakefake",
        temperature=0.0,
        guard_digest="8a6c833ffc329f89",
        quote_relocation_version="8b5baf4f236d2854",
        verification_policy_version="9c6cbaf5347e3965",
    )
    meta = vc.build_meta(
        [],
        existing=None,
        name="scifact",
        run="A",
        verifier=verifier,
        price=common.price_record("deepseek-chat", None, None),
        dry_run=False,
        session_started="2026-09-02T12:00:00+00:00",
        n_items=0,
        limit=None,
        concurrency=1,
    )
    assert meta["guard_digest"] == "8a6c833ffc329f89"
    assert meta["quote_relocation_version"] == "8b5baf4f236d2854"
    assert meta["verification_policy_version"] == "9c6cbaf5347e3965"


def test_build_meta_defaults_guard_digest_and_quote_relocation_version_to_none_when_absent():
    """A verifier stub that predates this field (`FakeVerifier` above carries neither
    attribute) must not raise, and must write `None`, not silently drop the key."""
    meta = vc.build_meta(
        [],
        existing=None,
        name="scifact",
        run="A",
        verifier=FakeVerifier(),
        price=common.price_record("deepseek-chat", None, None),
        dry_run=False,
        session_started="2026-09-02T12:00:00+00:00",
        n_items=0,
        limit=None,
        concurrency=1,
    )
    assert "guard_digest" in meta and meta["guard_digest"] is None
    assert "quote_relocation_version" in meta and meta["quote_relocation_version"] is None
    assert "verification_policy_version" in meta and meta["verification_policy_version"] is None


def test_dry_run_verifier_reports_dry_run_for_guard_digest_and_quote_relocation_version():
    assert vc.DryRunVerifier.guard_digest == "dry-run"
    assert vc.DryRunVerifier.quote_relocation_version == "dry-run"
    assert vc.DryRunVerifier.verification_policy_version == "dry-run"


# ---------------------------------------------------------------- usage_extras


class _ResolvedUsage:
    """Mimics pydantic-ai 1.107.0's ``AgentRunResult.usage``: attribute access already returns
    an object with real usage attributes of its own, and that object is also callable for
    backward compatibility. Calling it must never happen (it would emit
    ``PydanticAIDeprecationWarning`` in the real library), so ``__call__`` raises here to make
    an accidental call fail the test loudly instead of silently passing."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    def __call__(self):  # pragma: no cover - only reached if the fix regresses
        raise AssertionError(
            "usage must be read as a property, not called: result.usage already returns the "
            "resolved value on attribute access in pydantic-ai 1.107.0 (the pinned version)"
        )


def test_usage_extras_reads_an_already_resolved_usage_without_calling_it():
    class Result:
        usage = _ResolvedUsage(
            input_tokens=100, output_tokens=20, cache_read_tokens=64,
            details={"reasoning_tokens": 3},
        )

    extras = vc.usage_extras(Result())
    assert extras == {"cache_read_tokens": 64, "usage_details": {"reasoning_tokens": 3}}


def test_usage_extras_still_calls_a_bare_callable_with_no_usage_attributes():
    """Backward-compat fallback: a callable that does not already look like a resolved usage
    object (no ``input_tokens`` attribute of its own) is still called, so an older pydantic-ai
    style ``result.usage()`` method is not silently broken."""
    calls = []

    def legacy_usage():
        calls.append(1)
        return SimpleNamespace(cache_read_tokens=7, details=None)

    class Result:
        usage = staticmethod(legacy_usage)

    extras = vc.usage_extras(Result())
    assert extras == {"cache_read_tokens": 7, "usage_details": None}
    assert calls == [1]


def test_usage_extras_handles_missing_usage():
    class Result:
        pass

    assert vc.usage_extras(Result()) == {"cache_read_tokens": None, "usage_details": None}


# ---------------------------------------------------------------- verifiers


def test_dry_run_verifier_is_deterministic():
    v = vc.DryRunVerifier()
    a = asyncio.run(v("Learners improved their scores.", [CHUNK], "T", None))
    b = asyncio.run(v("Learners improved their scores.", [CHUNK], "T", None))
    assert a == b and a["predicted_status"] == "verified"
    assert a["cache_read_tokens"] == 0 and a["usage_details"] is None
    c = asyncio.run(v("Learners improved their scores.", [], "T", None))
    assert c["predicted_status"] == "no_full_text"
    d = asyncio.run(v("Unrelated claim about cats and dogs.", [CHUNK], "T", None))
    assert d["predicted_status"] == "unsupported"


FAKE_APP = {
    "app/__init__.py": "",
    "app/config.py": "class _S:\n    deepseek_model = 'deepseek-chat'\n\nsettings = _S()\n",
    "app/agents/__init__.py": "",
    "app/agents/analysis_agent.py": textwrap.dedent(
        """
        from dataclasses import dataclass
        from typing import Any

        @dataclass
        class AnalysisDependencies:
            project_id: Any
            project_description: str | None
            target_journal: str | None
            citation_style: str
        """
    ),
    "app/agents/model_config.py": "DETERMINISTIC_LONG_MODEL_SETTINGS = {'temperature': TEMP}\n",
    "app/schemas/__init__.py": "",
    "app/schemas/provenance.py": textwrap.dedent(
        """
        import hashlib
        from types import SimpleNamespace

        def prompt_version(prompt):
            return "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]

        def provenance_from_run(agent, run_result, *, model_configured, temperature, prompt):
            response = run_result.response
            usage = run_result.usage()
            return SimpleNamespace(
                agent=agent,
                model_configured=model_configured,
                model_reported=response.model_name,
                provider=response.provider_name,
                provider_response_id=response.provider_response_id,
                system_fingerprint=response.provider_details.get("system_fingerprint"),
                temperature=temperature,
                prompt_version=prompt_version(prompt),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )
        """
    ),
    "app/agents/claim_verification_agent.py": textwrap.dedent(
        """
        from types import SimpleNamespace
        from app.schemas.provenance import prompt_version

        VERIFICATION_PROMPT = "You are a meticulous academic fact-checker."
        CLAIM_VERIFICATION_PROMPT_VERSION = prompt_version(VERIFICATION_PROMPT)
        QUOTE_REPAIR_PROMPT = "Repair your quote: {segments}"
        QUOTE_REPAIR_PROMPT_VERSION = prompt_version(QUOTE_REPAIR_PROMPT)
        CALLS = []

        def format_quote_repair_prompt(failed_segments):
            numbered = "\\n".join(
                "%d. %r" % (i, seg) for i, seg in enumerate(failed_segments, 1)
            )
            return QUOTE_REPAIR_PROMPT.format(segments=numbered)

        class _Agent:
            async def run(self, prompt, deps=None, message_history=None):
                CALLS.append(
                    {"prompt": prompt, "deps": deps, "message_history": message_history}
                )
                return SimpleNamespace(
                    output=SimpleNamespace(
                        status="needs_nuance", evidence_quote="q", explanation="e",
                        suggested_revision="r",
                        assertions=[{"text": "t", "kind": "population", "verdict": "supported"}],
                    ),
                    response=SimpleNamespace(
                        model_name="deepseek-v4-flash", provider_name="deepseek",
                        provider_response_id="resp-1",
                        provider_details={"system_fingerprint": "fp-1"},
                    ),
                    usage=lambda: SimpleNamespace(
                        input_tokens=100, output_tokens=20, cache_read_tokens=64,
                        details={"reasoning_tokens": 3},
                    ),
                    all_messages=lambda: ["msg-from-first-call"],
                )

        def get_claim_verification_agent():
            return _Agent()

        def format_verification_prompt(claim_text, chunks, paper_title, paper_authors=None):
            return "|".join([paper_title, ",".join(paper_authors or []), claim_text, *chunks])
        """
    ),
    "app/services/__init__.py": "",
    # A real backend exposes the guards here; the default fake is inert (never changes
    # status/reasons) so it does not disturb tests that only exercise the plumbing.
    # `test_production_verifier_calls_the_guard_entry_point*` below install a non-inert body
    # to prove `ProductionVerifier` actually calls it. `relocate_evidence_quotes` is
    # likewise inert here (identity on the three quote fields, no diagnostics);
    # `test_production_verifier_calls_relocation_before_the_guard_entry_point` below
    # installs a non-inert body to prove `ProductionVerifier` actually calls it, in front
    # of the guard, the same way `apply_verification_guards`' own wiring is proved.
    "app/services/fulltext.py": textwrap.dedent(
        """
        GUARD_DIGEST = "test-guard-digest"
        QUOTE_RELOCATION_VERSION = "test-relocation-version"

        def normalise_verification_text(text, *, strip_wrapping_quotes=False):
            return text

        def attribution_guard_fires_before_relocation(claim_text, chunk_texts, evidence_quote):
            return False

        def relocate_evidence_quotes(evidence_quote, evidence_quotes, assertions, chunk_texts):
            return evidence_quote, evidence_quotes, list(assertions or []), []

        def apply_verification_guards(
            status, *, claim_text, evidence_quote, chunk_texts, chunks,
            evidence_quotes=None, assertions=None,
        ):
            return status, [], []
        """
    )
    + _SHARED_POLICY_BODY,
}


@pytest.fixture
def fake_backend(tmp_path: Path, monkeypatch):
    def install(
        temperature: float = 0.0,
        *,
        drop_version: bool = False,
        guard_body: str | None = None,
        agent_body: str | None = None,
    ) -> Path:
        root = tmp_path / "backend"
        for rel, body in FAKE_APP.items():
            body = body.replace("TEMP", repr(temperature))
            if drop_version and rel.endswith("claim_verification_agent.py"):
                body = body.replace("CLAIM_VERIFICATION_PROMPT_VERSION = ", "_dropped = ")
            if guard_body is not None and rel == "app/services/fulltext.py":
                body = guard_body
            if agent_body is not None and rel == "app/agents/claim_verification_agent.py":
                body = agent_body
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
        for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.syspath_prepend(str(root))
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-not-real")
        (tmp_path / ".env").write_text("DEEPSEEK_MODEL=deepseek-chat\n", encoding="utf-8")
        return root

    yield install
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        sys.modules.pop(name, None)


def test_production_verifier_against_fake_backend(fake_backend, tmp_path: Path):
    root = fake_backend(0.0)
    verifier = vc.ProductionVerifier(backend_root=root, dotenv_path=tmp_path / ".env")
    import app.agents.claim_verification_agent as fake_agent

    assert verifier.temperature == 0.0
    assert verifier.prompt_version == fake_agent.CLAIM_VERIFICATION_PROMPT_VERSION
    assert verifier.model_configured == "deepseek-chat"

    out = asyncio.run(verifier("claim X", ["c1", "c2"], "Title", ["A", "B"]))
    (call,) = fake_agent.CALLS
    expected_prompt = fake_agent.format_verification_prompt(
        "claim X", ["c1", "c2"], "Title", ["A", "B"]
    )
    assert call["prompt"] == expected_prompt
    from app.agents.analysis_agent import AnalysisDependencies

    assert isinstance(call["deps"], AnalysisDependencies)
    assert call["deps"].project_id == vc.EVAL_PROJECT_ID
    assert out["predicted_status"] == "needs_nuance" and out["evidence_quote"] == "q"
    assert out["suggested_revision"] == "r" and out["explanation"] == "e"
    assert out["model_reported"] == "deepseek-v4-flash" and out["system_fingerprint"] == "fp-1"
    assert out["provider_response_id"] == "resp-1"
    assert out["input_tokens"] == 100 and out["output_tokens"] == 20
    assert out["cache_read_tokens"] == 64 and out["usage_details"] == {"reasoning_tokens": 3}
    # The inert fake guard is a no-op, so `model_status` and `predicted_status` agree here;
    # the wiring itself (that the real guard entry point is imported and called with the
    # right arguments, and that a guard that *does* change the status is reflected in
    # `predicted_status`) is exercised by
    # `test_production_verifier_calls_the_guard_entry_point_and_records_new_fields` and
    # `test_production_verifier_applies_a_guard_that_caps_the_status` below.
    assert out["model_status"] == "needs_nuance"
    assert out["machine_reasons"] == []
    # The harness normalises every assertion's `quote`/`quotes` the same way `_verify_one`
    # does, so a fake-agent assertion with neither key still round-trips with both added
    # (both `None`/`[]`, since neither was ever set).
    assert out["assertions"] == [
        {"text": "t", "kind": "population", "verdict": "supported", "quote": None, "quotes": []}
    ]
    # No `evidence_quotes`/`unstated_details` on the fake agent's output -> both default to
    # `[]`; `diagnostics` is always present and empty for an inert guard.
    assert out["evidence_quotes"] == [] and out["unstated_details"] == []
    assert out["diagnostics"] == []


# ---------------------------------------------------------------- guard wiring
# (the harness must call the *real* guard entry point, not bypass it, so the numbers
# `summarize.py` scores reflect production behaviour)


def test_production_verifier_calls_the_guard_entry_point_and_records_new_fields(
    fake_backend, tmp_path: Path
):
    """The default fake guard is inert, so this only proves the plumbing: the entry point
    is imported from ``app.services.fulltext`` (not re-implemented) and called once per
    verification, and its three-tuple return (status, machine_reasons, diagnostics) is
    threaded into `predicted_status` / `machine_reasons` / `diagnostics` while the model's
    own answer survives separately as `model_status`."""
    root = fake_backend(0.0)
    verifier = vc.ProductionVerifier(backend_root=root, dotenv_path=tmp_path / ".env")
    out = asyncio.run(verifier("claim X", ["c1", "c2"], "Title", ["A", "B"]))
    assert out["model_status"] == "needs_nuance"
    assert out["predicted_status"] == "needs_nuance"  # inert guard: unchanged
    assert out["machine_reasons"] == []
    assert out["diagnostics"] == []
    assert out["quote_relocations"] == []


# --------------------------------------------------------------------------------------
# Quote relocation: a separate, versioned step
# ``app.services.fulltext.relocate_evidence_quotes`` runs immediately before the guard
# entry point, never inside it. The default fake above is inert (identity, no
# diagnostics), so this proves the plumbing the same way `_CAPPING_GUARD_BODY` proves the
# guard's own wiring: the entry point is imported from ``app.services.fulltext`` (not
# re-implemented), called once per verification with the model's own (normalised) quote
# and the claim's chunks, and its output -- not the model's raw answer -- is what the
# guard then sees and what the harness stores; firing it appends "quote_relocated" to
# `diagnostics` only after the guard has already returned its own list, never before.
# --------------------------------------------------------------------------------------

_RELOCATING_GUARD_BODY = textwrap.dedent(
    """
    GUARD_DIGEST = "test-guard-digest"
    QUOTE_RELOCATION_VERSION = "test-relocation-version"
    RELOCATE_CALLS = []

    def normalise_verification_text(text, *, strip_wrapping_quotes=False):
        return text

    def attribution_guard_fires_before_relocation(claim_text, chunk_texts, evidence_quote):
        return False

    def relocate_evidence_quotes(evidence_quote, evidence_quotes, assertions, chunk_texts):
        RELOCATE_CALLS.append({
            "evidence_quote": evidence_quote,
            "evidence_quotes": evidence_quotes,
            "assertions": assertions,
            "chunk_texts": list(chunk_texts),
        })
        relocated = "RELOCATED:" + evidence_quote if evidence_quote else evidence_quote
        relocated_quotes = [relocated, *evidence_quotes[1:]] if evidence_quotes else evidence_quotes
        diagnostics = [{
            "type": "quote_relocated",
            "original": evidence_quote,
            "replacement": relocated,
            "edit_distance": 1,
            "chunk_index": 0,
        }]
        return relocated, relocated_quotes, list(assertions or []), diagnostics

    def apply_verification_guards(
        status, *, claim_text, evidence_quote, chunk_texts, chunks,
        evidence_quotes=None, assertions=None,
    ):
        return status, [], []
    """
) + _SHARED_POLICY_BODY


def test_production_verifier_calls_relocation_before_the_guard_entry_point(
    fake_backend, tmp_path: Path
):
    """The stub relocation above is fed the model's own (normalised) evidence quote and
    the claim's chunks; its output, not the model's raw answer, is what is stored and
    (per the inert guard here) what the guard would have seen; the diagnostic slug is
    appended after the (inert, empty) guard diagnostics, not lost to them."""
    root = fake_backend(0.0, guard_body=_RELOCATING_GUARD_BODY)
    verifier = vc.ProductionVerifier(backend_root=root, dotenv_path=tmp_path / ".env")
    out = asyncio.run(verifier("claim X", ["c1", "c2"], "Title", ["A", "B"]))

    import app.services.fulltext as fake_fulltext

    (call,) = fake_fulltext.RELOCATE_CALLS
    assert call["evidence_quote"] == "q"  # the fake agent's own answer, normalised
    assert call["evidence_quotes"] == []
    assert call["chunk_texts"] == ["c1", "c2"]

    assert out["evidence_quote"] == "RELOCATED:q"
    assert out["evidence_quotes"] == []
    assert out["diagnostics"] == ["quote_relocated"]
    assert out["quote_relocations"] == [
        {
            "type": "quote_relocated",
            "original": "q",
            "replacement": "RELOCATED:q",
            "edit_distance": 1,
            "chunk_index": 0,
        }
    ]


# Relocation must not manufacture guard 1's own located-quote exemption, so
# `ProductionVerifier` must skip it entirely when `attribution_guard_fires_before_relocation`
# says guard 1 already fires on the model's own, pre-relocation quote, mirroring
# `app.services.fulltext._verify_claims`'s own wiring exactly.

_SKIP_ON_ATTRIBUTION_GUARD_BODY = textwrap.dedent(
    """
    GUARD_DIGEST = "test-guard-digest"
    QUOTE_RELOCATION_VERSION = "test-relocation-version"
    RELOCATE_CALLS = []

    def normalise_verification_text(text, *, strip_wrapping_quotes=False):
        return text

    def attribution_guard_fires_before_relocation(claim_text, chunk_texts, evidence_quote):
        return True

    def relocate_evidence_quotes(evidence_quote, evidence_quotes, assertions, chunk_texts):
        RELOCATE_CALLS.append(evidence_quote)
        return "RELOCATED:" + evidence_quote, evidence_quotes, list(assertions or []), [
            {"type": "quote_relocated", "original": evidence_quote,
             "replacement": "RELOCATED:" + evidence_quote, "edit_distance": 1, "chunk_index": 0}
        ]

    def apply_verification_guards(
        status, *, claim_text, evidence_quote, chunk_texts, chunks,
        evidence_quotes=None, assertions=None,
    ):
        return "unsupported", ["attribution_mismatch"], []
    """
) + _SHARED_POLICY_BODY


def test_production_verifier_skips_relocation_when_attribution_guard_already_fires(
    fake_backend, tmp_path: Path
):
    """When `attribution_guard_fires_before_relocation` reports that guard 1 already fires
    on the model's raw quote, `relocate_evidence_quotes` must never be called at all -- not
    called-and-discarded, never called -- so the stored quote is the model's own, unrelocated
    answer and `quote_relocations` is empty."""
    root = fake_backend(0.0, guard_body=_SKIP_ON_ATTRIBUTION_GUARD_BODY)
    verifier = vc.ProductionVerifier(backend_root=root, dotenv_path=tmp_path / ".env")
    out = asyncio.run(verifier("claim X", ["c1", "c2"], "Title", ["A", "B"]))

    import app.services.fulltext as fake_fulltext

    assert fake_fulltext.RELOCATE_CALLS == []
    assert out["evidence_quote"] == "q"  # the fake agent's own, unrelocated answer
    assert out["predicted_status"] == "unsupported"
    assert out["machine_reasons"] == ["attribution_mismatch"]
    assert out["quote_relocations"] == []
    assert "quote_relocated" not in out["diagnostics"]


_CAPPING_GUARD_BODY = textwrap.dedent(
    """
    GUARD_DIGEST = "test-guard-digest"
    QUOTE_RELOCATION_VERSION = "test-relocation-version"
    GUARD_CALLS = []

    def normalise_verification_text(text, *, strip_wrapping_quotes=False):
        return text

    def attribution_guard_fires_before_relocation(claim_text, chunk_texts, evidence_quote):
        return False

    def relocate_evidence_quotes(evidence_quote, evidence_quotes, assertions, chunk_texts):
        return evidence_quote, evidence_quotes, list(assertions or []), []

    def apply_verification_guards(
        status, *, claim_text, evidence_quote, chunk_texts, chunks,
        evidence_quotes=None, assertions=None,
    ):
        GUARD_CALLS.append({
            "status": status,
            "claim_text": claim_text,
            "evidence_quote": evidence_quote,
            "evidence_quotes": evidence_quotes,
            "assertions": assertions,
            "chunk_texts": list(chunk_texts),
            "chunks": list(chunks),
        })
        return "unsupported", ["always_capped_for_test"], []
    """
) + _SHARED_POLICY_BODY


def test_production_verifier_applies_a_guard_that_caps_the_status(fake_backend, tmp_path: Path):
    """A guard that changes the status must be reflected in `predicted_status` (what
    `summarize.py` scores), while `model_status` keeps the model's own, pre-guard answer,
    and the guard must see the same claim/quote/chunks the model was shown, plus the new
    `evidence_quotes`/`assertions` keywords."""
    root = fake_backend(0.0, guard_body=_CAPPING_GUARD_BODY)
    verifier = vc.ProductionVerifier(backend_root=root, dotenv_path=tmp_path / ".env")
    out = asyncio.run(verifier("claim X", ["c1", "c2"], "Title", ["A", "B"]))

    assert out["model_status"] == "needs_nuance"
    assert out["predicted_status"] == "unsupported"
    assert out["machine_reasons"] == ["always_capped_for_test"]
    assert out["diagnostics"] == []  # the capping fake's third return value

    import app.services.fulltext as fake_fulltext

    (call,) = fake_fulltext.GUARD_CALLS
    assert call["status"] == "needs_nuance"
    assert call["claim_text"] == "claim X"
    assert call["evidence_quote"] == "q"
    assert call["evidence_quotes"] == []
    assert call["assertions"] == [
        {"text": "t", "kind": "population", "verdict": "supported", "quote": None, "quotes": []}
    ]
    assert call["chunk_texts"] == ["c1", "c2"]
    assert call["chunks"] == [{"text": "c1"}, {"text": "c2"}]


# `FAKE_APP["app/services/fulltext.py"]` and `_CAPPING_GUARD_BODY` above both stub
# `normalise_verification_text` as an identity function, so no existing test proves
# `ProductionVerifier.__call__` (`verify_common.py` :351-371) actually threads a
# *non-identity* normaliser through `explanation`, `suggested_revision`, `evidence_quote`,
# every `evidence_quotes` element and every assertion's `quote`/`quotes`, nor that the
# `evidence_quote = evidence_quotes[0]` fallback discards a stale single-span quote in
# favour of the (normalised) first multi-span quote. `_MARKER_NORMALISE_GUARD_BODY`
# installs a marker transform (prefixes every non-null string with ``"N:"``) and
# `_MULTI_SPAN_AGENT_BODY` answers with both a stale `evidence_quote` and a two-element
# `evidence_quotes`, so the marker's presence (or absence) on each field is directly
# observable.

_MARKER_NORMALISE_GUARD_BODY = textwrap.dedent(
    """
    GUARD_DIGEST = "test-guard-digest"
    QUOTE_RELOCATION_VERSION = "test-relocation-version"

    def normalise_verification_text(text, *, strip_wrapping_quotes=False):
        return None if text is None else "N:" + text

    def attribution_guard_fires_before_relocation(claim_text, chunk_texts, evidence_quote):
        return False

    def relocate_evidence_quotes(evidence_quote, evidence_quotes, assertions, chunk_texts):
        return evidence_quote, evidence_quotes, list(assertions or []), []

    def apply_verification_guards(
        status, *, claim_text, evidence_quote, chunk_texts, chunks,
        evidence_quotes=None, assertions=None,
    ):
        return status, [], []
    """
) + _SHARED_POLICY_BODY

_MULTI_SPAN_AGENT_BODY = textwrap.dedent(
    """
    from types import SimpleNamespace
    from app.schemas.provenance import prompt_version

    VERIFICATION_PROMPT = "You are a meticulous academic fact-checker."
    CLAIM_VERIFICATION_PROMPT_VERSION = prompt_version(VERIFICATION_PROMPT)
    QUOTE_REPAIR_PROMPT = "Repair your quote: {segments}"
    QUOTE_REPAIR_PROMPT_VERSION = prompt_version(QUOTE_REPAIR_PROMPT)
    CALLS = []

    def format_quote_repair_prompt(failed_segments):
        numbered = "\\n".join(
            "%d. %r" % (i, seg) for i, seg in enumerate(failed_segments, 1)
        )
        return QUOTE_REPAIR_PROMPT.format(segments=numbered)

    class _Agent:
        async def run(self, prompt, deps=None, message_history=None):
            CALLS.append(
                {"prompt": prompt, "deps": deps, "message_history": message_history}
            )
            return SimpleNamespace(
                output=SimpleNamespace(
                    status="verified",
                    evidence_quote="stale single-span quote",
                    evidence_quotes=["span one", "span two"],
                    explanation="explanation text",
                    suggested_revision="revision text",
                    assertions=[
                        {
                            "text": "t", "kind": "population", "verdict": "supported",
                            "quote": "assertion quote", "quotes": ["aq1", "aq2"],
                        }
                    ],
                ),
                response=SimpleNamespace(
                    model_name="deepseek-v4-flash", provider_name="deepseek",
                    provider_response_id="resp-1",
                    provider_details={"system_fingerprint": "fp-1"},
                ),
                usage=lambda: SimpleNamespace(
                    input_tokens=100, output_tokens=20, cache_read_tokens=64,
                    details={"reasoning_tokens": 3},
                ),
                all_messages=lambda: ["msg-from-first-call"],
            )

    def get_claim_verification_agent():
        return _Agent()

    def format_verification_prompt(claim_text, chunks, paper_title, paper_authors=None):
        return "|".join([paper_title, ",".join(paper_authors or []), claim_text, *chunks])
    """
)


def test_production_verifier_normalises_every_multi_span_field_with_a_non_identity_normaliser(
    fake_backend, tmp_path: Path
):
    """The marker (``"N:"`` prefix) must reach `explanation`, `suggested_revision`, every
    `evidence_quotes` element and every assertion's `quote`/`quotes`; `evidence_quote` must
    equal the (marked) first `evidence_quotes` element, not the marked or unmarked stale
    single-span quote the fake agent also returns."""
    root = fake_backend(
        0.0, guard_body=_MARKER_NORMALISE_GUARD_BODY, agent_body=_MULTI_SPAN_AGENT_BODY
    )
    verifier = vc.ProductionVerifier(backend_root=root, dotenv_path=tmp_path / ".env")
    out = asyncio.run(verifier("claim X", ["c1", "c2"], "Title", ["A", "B"]))

    assert out["explanation"] == "N:explanation text"
    assert out["suggested_revision"] == "N:revision text"
    assert out["evidence_quotes"] == ["N:span one", "N:span two"]
    assert out["evidence_quote"] == "N:span one"
    assert out["evidence_quote"] == out["evidence_quotes"][0]
    assert out["assertions"] == [
        {
            "text": "t", "kind": "population", "verdict": "supported",
            "quote": "N:assertion quote", "quotes": ["N:aq1", "N:aq2"],
        }
    ]


# --------------------------------------------------------------------------------------
# Task authorisation 2026-09-10, Part B (the bounded second pass): the evaluation harness
# must follow the identical policy the production path does, using a stub model that
# returns two scripted outputs. `_VERBATIM_GUARD_BODY` performs a real (if simplified)
# quote-fidelity check, rather than the inert stubs above, since the trigger condition
# (`needs_nuance` with exactly `["quote_not_verbatim"]` after a model status of `verified`)
# can only fire against a guard that actually looks at the quote.
# --------------------------------------------------------------------------------------

_VERBATIM_GUARD_BODY = textwrap.dedent(
    """
    GUARD_DIGEST = "test-guard-digest"
    QUOTE_RELOCATION_VERSION = "test-relocation-version"

    def normalise_verification_text(text, *, strip_wrapping_quotes=False):
        return text

    def attribution_guard_fires_before_relocation(claim_text, chunk_texts, evidence_quote):
        return False

    def relocate_evidence_quotes(evidence_quote, evidence_quotes, assertions, chunk_texts):
        return evidence_quote, evidence_quotes, list(assertions or []), []

    def apply_verification_guards(
        status, *, claim_text, evidence_quote, chunk_texts, chunks,
        evidence_quotes=None, assertions=None,
    ):
        if not evidence_quote:
            return status, [], []
        if any(evidence_quote in c for c in chunk_texts):
            return status, [], []
        return "needs_nuance", ["quote_not_verbatim"], []
    """
) + _SHARED_POLICY_BODY

_TWO_PASS_AGENT_BODY_TEMPLATE = textwrap.dedent(
    """
    from types import SimpleNamespace
    from app.schemas.provenance import prompt_version

    VERIFICATION_PROMPT = "You are a meticulous academic fact-checker."
    CLAIM_VERIFICATION_PROMPT_VERSION = prompt_version(VERIFICATION_PROMPT)
    QUOTE_REPAIR_PROMPT = "Repair your quote: {segments}"
    QUOTE_REPAIR_PROMPT_VERSION = prompt_version(QUOTE_REPAIR_PROMPT)
    CALLS = []
    RESULTS = []
    OUTPUTS = [
        {"status": "verified", "evidence_quote": "a bad quote",
         "evidence_quotes": ["a bad quote"], "explanation": "first pass",
         "suggested_revision": None, "assertions": [], "unstated_details": []},
        SECOND_OUTPUT,
    ]

    def format_quote_repair_prompt(failed_segments):
        numbered = "\\n".join(
            "%d. %r" % (i, seg) for i, seg in enumerate(failed_segments, 1)
        )
        return QUOTE_REPAIR_PROMPT.format(segments=numbered)

    class _Agent:
        async def run(self, prompt, deps=None, message_history=None):
            index = len(CALLS)
            CALLS.append(
                {"prompt": prompt, "deps": deps, "message_history": message_history}
            )
            result = SimpleNamespace(
                output=SimpleNamespace(**OUTPUTS[index]),
                response=SimpleNamespace(
                    model_name="deepseek-v4-flash", provider_name="deepseek",
                    provider_response_id="resp-%d" % (index + 1),
                    provider_details={"system_fingerprint": "fp-%d" % (index + 1)},
                ),
                usage=lambda: SimpleNamespace(
                    input_tokens=100, output_tokens=20, cache_read_tokens=10, details=None,
                ),
            )
            result.all_messages = lambda: ["msg-%d" % (index + 1)]
            RESULTS.append(result)
            return result

    def get_claim_verification_agent():
        return _Agent()

    def format_verification_prompt(claim_text, chunks, paper_title, paper_authors=None):
        return "|".join([paper_title, ",".join(paper_authors or []), claim_text, *chunks])
    """
)

_TWO_PASS_FIXED_AGENT_BODY = _TWO_PASS_AGENT_BODY_TEMPLATE.replace(
    "SECOND_OUTPUT",
    '{"status": "verified", "evidence_quote": "c1", "evidence_quotes": ["c1"], '
    '"explanation": "second pass", "suggested_revision": None, "assertions": [], '
    '"unstated_details": []}',
)
_TWO_PASS_STILL_BAD_AGENT_BODY = _TWO_PASS_AGENT_BODY_TEMPLATE.replace(
    "SECOND_OUTPUT",
    '{"status": "verified", "evidence_quote": "still a bad quote", '
    '"evidence_quotes": ["still a bad quote"], "explanation": "second pass", '
    '"suggested_revision": None, "assertions": [], "unstated_details": []}',
)
_TWO_PASS_UNSUPPORTED_AGENT_BODY = _TWO_PASS_AGENT_BODY_TEMPLATE.replace(
    "SECOND_OUTPUT",
    '{"status": "unsupported", "evidence_quote": None, "evidence_quotes": [], '
    '"explanation": "second pass", "suggested_revision": None, "assertions": [], '
    '"unstated_details": []}',
)


def test_production_verifier_second_pass_fires_and_resolves_the_claim(
    fake_backend, tmp_path: Path
):
    """The first pass's quote is not in the chunks (`needs_nuance`, `["quote_not_verbatim"]`)
    after a model status of `verified`, so the repair turn fires; its quote is verbatim, so
    the claim resolves `verified`. Both calls are counted: two entries in `CALLS`, two
    entries in `passes`, and `input_tokens`/`output_tokens` are the sum of both calls. The
    repair call continues the SAME conversation (its own `message_history` is the first
    call's own `all_messages()`), and only the first call sends no history at all."""
    root = fake_backend(0.0, guard_body=_VERBATIM_GUARD_BODY, agent_body=_TWO_PASS_FIXED_AGENT_BODY)
    verifier = vc.ProductionVerifier(backend_root=root, dotenv_path=tmp_path / ".env")
    out = asyncio.run(verifier("claim X", ["c1", "c2"], "Title", ["A", "B"]))

    import app.agents.claim_verification_agent as fake_agent

    assert len(fake_agent.CALLS) == 2
    assert out["predicted_status"] == "verified"
    assert out["model_status"] == "verified"
    assert out["machine_reasons"] == []
    assert out["evidence_quote"] == "c1"
    assert len(out["passes"]) == 2
    assert out["passes"][0]["model_status"] == "verified"
    assert out["passes"][0]["machine_reasons"] == ["quote_not_verbatim"]
    assert out["passes"][0]["repair_prompt_version"] is None
    assert out["passes"][1]["machine_reasons"] == []
    assert out["passes"][1]["repair_prompt_version"] == fake_agent.QUOTE_REPAIR_PROMPT_VERSION
    assert out["input_tokens"] == 200 and out["output_tokens"] == 40
    assert out["cache_read_tokens"] == 20
    assert out["model_reported"] == "deepseek-v4-flash"
    assert out["system_fingerprint"] == "fp-2"  # the second (final) call's own fingerprint

    # The repair call's own prompt names the failed segment and its `message_history` is
    # exactly the first call's own `all_messages()` -- the SAME conversation, not a fresh
    # copy of the first request (task authorisation 2026-09-11).
    assert fake_agent.CALLS[0]["message_history"] is None
    assert fake_agent.CALLS[1]["message_history"] == ["msg-1"]
    assert fake_agent.CALLS[1]["prompt"] == fake_agent.format_quote_repair_prompt(["a bad quote"])
    assert fake_agent.CALLS[1]["prompt"] != fake_agent.CALLS[0]["prompt"]


def test_production_verifier_second_pass_that_again_miscopies_stays_needs_nuance(
    fake_backend, tmp_path: Path
):
    """A second pass that mis-copies the quote a second time is trusted as final anyway
    (task authorisation: "whatever it is") -- exactly two calls, never a third."""
    root = fake_backend(
        0.0, guard_body=_VERBATIM_GUARD_BODY, agent_body=_TWO_PASS_STILL_BAD_AGENT_BODY
    )
    verifier = vc.ProductionVerifier(backend_root=root, dotenv_path=tmp_path / ".env")
    out = asyncio.run(verifier("claim X", ["c1", "c2"], "Title", ["A", "B"]))

    import app.agents.claim_verification_agent as fake_agent

    assert len(fake_agent.CALLS) == 2
    assert out["predicted_status"] == "needs_nuance"
    assert out["machine_reasons"] == ["quote_not_verbatim"]
    assert len(out["passes"]) == 2


def test_production_verifier_second_pass_whose_model_says_unsupported_is_unsupported(
    fake_backend, tmp_path: Path
):
    """The second pass's own result is trusted as final whatever it is, including a status
    the first pass never reached."""
    root = fake_backend(
        0.0, guard_body=_VERBATIM_GUARD_BODY, agent_body=_TWO_PASS_UNSUPPORTED_AGENT_BODY
    )
    verifier = vc.ProductionVerifier(backend_root=root, dotenv_path=tmp_path / ".env")
    out = asyncio.run(verifier("claim X", ["c1", "c2"], "Title", ["A", "B"]))

    import app.agents.claim_verification_agent as fake_agent

    assert len(fake_agent.CALLS) == 2
    assert out["predicted_status"] == "unsupported"
    assert out["model_status"] == "unsupported"
    assert len(out["passes"]) == 2


def test_production_verifier_never_fires_a_second_pass_when_the_first_pass_is_unsupported(
    fake_backend, tmp_path: Path
):
    """The default fake agent's model status is `needs_nuance`, not `verified`, and the
    default guard body is inert -- neither of the trigger's two conditions holds, so this
    (already covered implicitly by `test_production_verifier_against_fake_backend`'s single
    call) is restated directly against the verbatim guard with a fabricated quote, to prove
    the trigger does not fire merely because the final status is `needs_nuance`."""
    root = fake_backend(0.0, guard_body=_VERBATIM_GUARD_BODY)
    verifier = vc.ProductionVerifier(backend_root=root, dotenv_path=tmp_path / ".env")
    out = asyncio.run(verifier("claim X", ["c1", "c2"], "Title", ["A", "B"]))

    import app.agents.claim_verification_agent as fake_agent

    # The default fake agent's own status is "needs_nuance", not "verified", so even though
    # its quote ("q") is not verbatim in ["c1", "c2"] (`needs_nuance`, `["quote_not_verbatim"]`
    # from the verbatim guard), the trigger's `first_answer.status == "verified"` condition
    # is false and no second pass runs.
    assert len(fake_agent.CALLS) == 1
    assert out["predicted_status"] == "needs_nuance"
    assert out["machine_reasons"] == ["quote_not_verbatim"]
    assert len(out["passes"]) == 1


def test_production_verifier_reports_verification_policy_version(fake_backend, tmp_path: Path):
    root = fake_backend(0.0)
    verifier = vc.ProductionVerifier(backend_root=root, dotenv_path=tmp_path / ".env")
    assert verifier.verification_policy_version == "test-policy-version"


def test_production_verifier_reports_repair_prompt_version(fake_backend, tmp_path: Path):
    root = fake_backend(0.0)
    verifier = vc.ProductionVerifier(backend_root=root, dotenv_path=tmp_path / ".env")
    import app.agents.claim_verification_agent as fake_agent

    assert verifier.repair_prompt_version == fake_agent.QUOTE_REPAIR_PROMPT_VERSION


def test_build_meta_records_repair_prompt_version(tmp_path: Path):
    verifier = FakeVerifier()
    verifier.repair_prompt_version = "sha256:repairrepair"
    meta = vc.build_meta(
        [], existing=None, name="t", run="A", verifier=verifier, price={},
        dry_run=False, session_started="2026-01-01T00:00:00+00:00", n_items=0,
        limit=None, concurrency=1,
    )
    assert meta["repair_prompt_version"] == "sha256:repairrepair"


def test_production_verifier_temperature_guard(fake_backend, tmp_path: Path):
    root = fake_backend(0.7)
    with pytest.raises(SystemExit) as exc:
        vc.ProductionVerifier(backend_root=root, dotenv_path=tmp_path / ".env")
    assert exc.value.code == 2
    verifier = vc.ProductionVerifier(
        backend_root=root, dotenv_path=tmp_path / ".env", allow_nonzero_temperature=True
    )
    assert verifier.temperature == 0.7


def test_production_verifier_requires_merged_backend(fake_backend, tmp_path: Path):
    root = fake_backend(0.0, drop_version=True)
    with pytest.raises(SystemExit) as exc:
        vc.ProductionVerifier(backend_root=root, dotenv_path=tmp_path / ".env")
    assert "does not expose the required claim-verification interface" in str(exc.value)


def test_add_common_args_has_dry_run_and_temperature_flag():
    import argparse

    ap = argparse.ArgumentParser()
    vc.add_common_args(ap)
    args = ap.parse_args(["--run", "A", "--dry-run"])
    assert args.dry_run is True and args.allow_nonzero_temperature is False
    assert vc.resolve_results_dir(args) == vc.DRYRUN_RESULTS_DIR


# ---------------------------------------------------------------- additional cases


def test_retry_failed_reattempts_error_rows_once(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(vc, "RETRY_BASE_DELAY", 0.0)
    out = tmp_path / "run.jsonl"
    verifier = FakeVerifier(failures=1)
    run([item(1), item(2)], verifier, out, max_attempts=1)
    rows = read_jsonl(out)
    failed = sorted(r["item_id"] for r in rows if r["predicted_status"] == "error")
    assert len(failed) == 1 and len(rows) == 2
    # without the flag the error row is treated as done
    n_calls = len(verifier.calls)
    run([item(1), item(2)], verifier, out, max_attempts=1)
    assert len(verifier.calls) == n_calls
    # with the flag it is dropped and re-attempted exactly once, without duplicates
    progress = run([item(1), item(2)], verifier, out, max_attempts=1, retry_failed=True)
    assert progress["retried_failed"] == 1 and len(verifier.calls) == n_calls + 1
    rows = read_jsonl(out)
    assert sorted(r["item_id"] for r in rows) == ["it-1", "it-2"]
    assert all(r["predicted_status"] == "verified" for r in rows)
    progress = run([item(1), item(2)], verifier, out, max_attempts=1, retry_failed=True)
    assert progress["retried_failed"] == 0 and len(verifier.calls) == n_calls + 1


def test_add_common_args_has_retry_failed_and_resolves_paths(tmp_path: Path, monkeypatch):
    import argparse

    ap = argparse.ArgumentParser()
    vc.add_common_args(ap)
    monkeypatch.chdir(tmp_path)
    args = vc.parse_common(ap, ["--run", "A", "--results-dir", "rel", "--retry-failed"])
    assert args.retry_failed is True
    assert args.results_dir == (tmp_path / "rel").resolve()
    assert vc.parse_common(ap, ["--run", "A"]).retry_failed is False


def test_production_verifier_reports_settings_failure(fake_backend, tmp_path: Path, monkeypatch):
    root = fake_backend(0.0)
    (root / "app" / "config.py").write_text(
        "raise ValueError('13 validation errors for Settings')\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        vc.ProductionVerifier(backend_root=root, dotenv_path=tmp_path / ".env")
    assert "backend settings failed to load from" in str(exc.value)
    assert "13 validation errors" in str(exc.value)


def test_execute_records_retried_failed_in_meta(tmp_path: Path, monkeypatch):
    import argparse
    import asyncio

    monkeypatch.setattr(vc, "RETRY_BASE_DELAY", 0.0)
    verifier = FakeVerifier(failures=1)
    monkeypatch.setattr(vc, "make_verifier", lambda *a, **k: verifier)
    results = tmp_path / "results"
    ns = argparse.Namespace(
        run="A", limit=None, concurrency=1, max_attempts=1, price_input=None,
        price_output=None, price_source_url=None, price_accessed=None, results_dir=results,
        dry_run=False, allow_nonzero_temperature=False, retry_failed=False, confirm_cost=True,
    )
    items = [item(1), item(2)]
    asyncio.run(vc.execute(name="t", items=items, chunks_of=lambda it: it["chunks"], args=ns))
    meta = common.read_json(results / "t_runA.meta.json")
    assert meta["n_errors"] == 1 and meta["retried_failed"] == 0
    ns.retry_failed = True
    asyncio.run(vc.execute(name="t", items=items, chunks_of=lambda it: it["chunks"], args=ns))
    meta = common.read_json(results / "t_runA.meta.json")
    assert meta["n_errors"] == 0 and meta["retried_failed"] == 1
    assert meta["sessions"][-1]["retried_failed"] == 1
    assert len({r["item_id"] for r in read_jsonl(results / "t_runA.jsonl")}) == 2


# ---------------------------------------------------------------- additional cases


class SlowVerifier(FakeVerifier):
    def __init__(self, delay: float = 0.05, **kw) -> None:
        super().__init__(**kw)
        self.delay = delay

    async def __call__(self, claim, chunks, title, authors):
        await asyncio.sleep(self.delay)
        return await super().__call__(claim, chunks, title, authors)


def test_verify_with_retries_records_real_latency():
    outcome = asyncio.run(
        vc.verify_with_retries(
            SlowVerifier(0.05), claim="c", chunks=[CHUNK], title="T", authors=None,
            max_attempts=1,
        )
    )
    assert outcome["latency_s"] >= 0.05 and outcome["error"] is None


def test_dry_run_verifier_latency_is_measured():
    fast = asyncio.run(
        vc.verify_with_retries(
            vc.DryRunVerifier(), claim="c", chunks=[CHUNK], title="T", authors=None,
            max_attempts=1,
        )
    )
    assert isinstance(fast["latency_s"], float) and fast["latency_s"] >= 0.0
    slow = asyncio.run(
        vc.verify_with_retries(
            SlowVerifier(0.03), claim="c", chunks=[CHUNK], title="T", authors=None,
            max_attempts=1,
        )
    )
    assert slow["latency_s"] > fast["latency_s"]


def _namespace(results, **over):
    import argparse

    ns = argparse.Namespace(
        run="A", limit=None, concurrency=1, max_attempts=1, price_input=None,
        price_output=None, price_source_url=None, price_accessed=None, results_dir=results,
        dry_run=False, allow_nonzero_temperature=False, retry_failed=False, confirm_cost=False,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_execute_requires_confirm_cost_before_building_the_verifier(tmp_path: Path, monkeypatch,
                                                                    capsys):
    def _boom(*a, **k):
        raise AssertionError("make_verifier must not run without --confirm-cost")

    monkeypatch.setattr(vc, "make_verifier", _boom)
    results = tmp_path / "results"
    items = [item(1), item(2), item(3, chunks=[])]
    with pytest.raises(SystemExit) as exc:
        asyncio.run(vc.execute(name="t", items=items, chunks_of=lambda it: it["chunks"],
                               args=_namespace(results)))
    assert exc.value.code == 3
    out = capsys.readouterr().out
    assert "2 model calls" in out and "USD" in out and "--confirm-cost" in out
    assert not (results / "t_runA.jsonl").exists()
    with pytest.raises(SystemExit):
        asyncio.run(vc.execute(name="t", items=items, chunks_of=lambda it: it["chunks"],
                               args=_namespace(results, limit=2)))
    # --dry-run never needs the flag; --confirm-cost lets the run proceed
    monkeypatch.setattr(vc, "make_verifier", lambda *a, **k: FakeVerifier())
    rc = asyncio.run(vc.execute(name="t", items=items, chunks_of=lambda it: it["chunks"],
                                args=_namespace(results, confirm_cost=True)))
    assert rc == 0


def test_add_common_args_has_confirm_cost_and_cost_bound():
    import argparse

    ap = argparse.ArgumentParser()
    vc.add_common_args(ap)
    assert ap.parse_args(["--run", "A"]).confirm_cost is False
    assert ap.parse_args(["--run", "A", "--confirm-cost"]).confirm_cost is True
    bound = vc.cost_upper_bound(340, "deepseek-chat", input_tokens=2600, output_tokens=1000)
    assert bound == pytest.approx(340 * (2600 * 0.44 + 1000 * 1.32) / 1e6)
    assert vc.cost_upper_bound(1, "nope", input_tokens=1, output_tokens=1) is None


# ---------------------------------------------------------------- additional cases


def test_scifact_cost_bound_matches_the_readme_figure(capsys):
    """README: 340 SciFact calls at 2,600 input + 1,000 output tokens <= 0.84 USD at peak."""
    bound = vc.cost_upper_bound(340, "deepseek-chat", input_tokens=2600, output_tokens=1000)
    assert round(bound, 2) == 0.84
    hss = vc.cost_upper_bound(25, "deepseek-chat", input_tokens=27_000, output_tokens=1000)
    assert round(hss, 2) == 0.33
    import argparse

    args = argparse.Namespace(dry_run=False, confirm_cost=False)
    with pytest.raises(SystemExit):
        vc.require_cost_confirmation(args, "scifact", 2)
    out = capsys.readouterr().out
    two = vc.cost_upper_bound(2, "deepseek-chat", input_tokens=2600, output_tokens=1000)
    assert f"<= {two:.4f} USD" in out  # four decimals: a 2-call smoke is not "0.00 USD"


def test_assumed_tokens_has_a_scifact_train_entry_matching_scifact(capsys):
    """run_scifact.py --split train writes name="scifact-train"; the cost gate must not
    silently fall back to DEFAULT_ASSUMED_TOKENS for it (same figures as the dev split,
    since the item shape, one abstract chunk per claim-document pair, is identical)."""
    assert vc.ASSUMED_TOKENS["scifact-train"] == vc.ASSUMED_TOKENS["scifact"]
    import argparse

    args = argparse.Namespace(dry_run=False, confirm_cost=False)
    with pytest.raises(SystemExit):
        vc.require_cost_confirmation(args, "scifact-train", 120)
    assert "set 'scifact-train'" in capsys.readouterr().out


def _offpeak_bound(n_calls, model, *, input_tokens, output_tokens):
    """Mirrors ``cost_upper_bound`` but at the off-peak tier, computed from
    ``common.DEEPSEEK_PRICES`` so the README off-peak figure cannot silently drift."""
    from common import DEEPSEEK_PRICES, resolve_price_model

    rates = DEEPSEEK_PRICES["models"][resolve_price_model(model)]
    per_call = (
        input_tokens * rates["input_cache_miss"]["offpeak"]
        + output_tokens * rates["output"]["offpeak"]
    ) / 1_000_000.0
    return n_calls * per_call


def test_e2_offpeak_cost_bound_matches_the_readme_figure():
    """README: two runs of E2 cost at most ~1.17 USD off-peak (2 x (0.42 + 0.165))."""
    scifact_off = _offpeak_bound(340, "deepseek-chat", input_tokens=2600, output_tokens=1000)
    hss_off = _offpeak_bound(25, "deepseek-chat", input_tokens=27_000, output_tokens=1000)
    assert round(scifact_off, 2) == 0.42
    assert round(hss_off, 2) == 0.17  # printed in the README as 0.165 (rounds to 0.17 alone)
    assert round(2 * (scifact_off + hss_off), 2) == 1.17
    # off-peak rates are exactly half the peak rates, so the bound is also exactly half
    peak = (vc.cost_upper_bound(340, "deepseek-chat", input_tokens=2600, output_tokens=1000)
            + vc.cost_upper_bound(25, "deepseek-chat", input_tokens=27_000, output_tokens=1000))
    assert round(peak / 2, 2) == round(scifact_off + hss_off, 2)


# ---------------------------------------------------------------- additional cases


def test_execute_refuses_to_mix_dry_run_and_paid_provenance_in_the_same_results_dir(
    tmp_path: Path,
):
    """A --results-dir already written by a dry run must reject a paid run (and vice versa),
    so stand-in rows are never silently reported as production output (or the reverse)."""
    import argparse

    results = tmp_path / "results"
    ns = argparse.Namespace(
        run="A", limit=None, concurrency=1, max_attempts=1, price_input=None,
        price_output=None, price_source_url=None, price_accessed=None, results_dir=results,
        dry_run=True, allow_nonzero_temperature=False, retry_failed=False, confirm_cost=False,
    )
    items = [item(1), item(2)]
    asyncio.run(vc.execute(name="t", items=items, chunks_of=lambda it: it["chunks"], args=ns))
    rows_before = read_jsonl(results / "t_runA.jsonl")
    ns.dry_run = False
    ns.confirm_cost = True
    with pytest.raises(SystemExit) as exc:
        asyncio.run(vc.execute(name="t", items=items, chunks_of=lambda it: it["chunks"], args=ns))
    assert exc.value.code == 2
    assert read_jsonl(results / "t_runA.jsonl") == rows_before
    meta = common.read_json(results / "t_runA.meta.json")
    assert meta["dry_run"] is True  # untouched


# ---------------------------------------------------------------- prompt-version
# directory guard and PROMPT.txt


def test_execute_refuses_to_mix_prompt_versions_in_the_same_results_dir(
    tmp_path: Path, monkeypatch
):
    """A --results-dir already written under prompt X must reject a run under prompt Y,
    mirroring the existing dry-run/paid mixing guard, so a v1 directory can never be
    silently overwritten by a v2 run (or vice versa)."""
    import argparse

    results = tmp_path / "results"
    v1 = FakeVerifier()
    v1.prompt_version = "sha256:v1v1v1v1v1v1"
    monkeypatch.setattr(vc, "make_verifier", lambda *a, **k: v1)
    ns = argparse.Namespace(
        run="A", limit=None, concurrency=1, max_attempts=1, price_input=None,
        price_output=None, price_source_url=None, price_accessed=None, results_dir=results,
        dry_run=False, allow_nonzero_temperature=False, retry_failed=False, confirm_cost=True,
    )
    items = [item(1), item(2)]
    asyncio.run(vc.execute(name="t", items=items, chunks_of=lambda it: it["chunks"], args=ns))
    rows_before = read_jsonl(results / "t_runA.jsonl")
    meta_before = common.read_json(results / "t_runA.meta.json")
    assert meta_before["prompt_version"] == "sha256:v1v1v1v1v1v1"

    v2 = FakeVerifier()
    v2.prompt_version = "sha256:v2v2v2v2v2v2"
    monkeypatch.setattr(vc, "make_verifier", lambda *a, **k: v2)
    with pytest.raises(SystemExit) as exc:
        asyncio.run(vc.execute(name="t", items=items, chunks_of=lambda it: it["chunks"], args=ns))
    assert exc.value.code == 2
    assert read_jsonl(results / "t_runA.jsonl") == rows_before
    meta_after = common.read_json(results / "t_runA.meta.json")
    assert meta_after["prompt_version"] == "sha256:v1v1v1v1v1v1"  # untouched


def test_execute_allows_a_fresh_directory_under_any_prompt_version(tmp_path: Path, monkeypatch):
    """The guard only fires against an *existing* meta file; a brand-new --results-dir
    is fine under any prompt version."""
    import argparse

    v2 = FakeVerifier()
    v2.prompt_version = "sha256:v2v2v2v2v2v2"
    monkeypatch.setattr(vc, "make_verifier", lambda *a, **k: v2)
    results = tmp_path / "results"
    ns = argparse.Namespace(
        run="A", limit=None, concurrency=1, max_attempts=1, price_input=None,
        price_output=None, price_source_url=None, price_accessed=None, results_dir=results,
        dry_run=False, allow_nonzero_temperature=False, retry_failed=False, confirm_cost=True,
    )
    rc = asyncio.run(
        vc.execute(name="t", items=[item(1)], chunks_of=lambda it: it["chunks"], args=ns)
    )
    assert rc == 0
    meta = common.read_json(results / "t_runA.meta.json")
    assert meta["prompt_version"] == "sha256:v2v2v2v2v2v2"


def test_execute_writes_prompt_txt_byte_exact_with_its_sha(tmp_path: Path, monkeypatch):
    class PromptedVerifier(FakeVerifier):
        system_prompt = "You are a meticulous academic fact-checker.\nRule one.\nRule two."

    verifier = PromptedVerifier()
    verifier.prompt_version = "sha256:abc123abc123"
    monkeypatch.setattr(vc, "make_verifier", lambda *a, **k: verifier)
    results = tmp_path / "results"
    import argparse

    ns = argparse.Namespace(
        run="A", limit=None, concurrency=1, max_attempts=1, price_input=None,
        price_output=None, price_source_url=None, price_accessed=None, results_dir=results,
        dry_run=False, allow_nonzero_temperature=False, retry_failed=False, confirm_cost=True,
    )
    asyncio.run(vc.execute(name="t", items=[item(1)], chunks_of=lambda it: it["chunks"], args=ns))
    prompt_txt = (results / "PROMPT.txt").read_text(encoding="utf-8")
    assert verifier.system_prompt in prompt_txt
    assert "sha256:abc123abc123" in prompt_txt


def test_execute_skips_prompt_txt_for_a_verifier_with_no_system_prompt(tmp_path: Path):
    """``DryRunVerifier`` (and any test double without ``system_prompt``) has no real
    prompt text to record; ``PROMPT.txt`` is simply not written for it."""
    import argparse

    results = tmp_path / "results"
    ns = argparse.Namespace(
        run="A", limit=None, concurrency=1, max_attempts=1, price_input=None,
        price_output=None, price_source_url=None, price_accessed=None, results_dir=results,
        dry_run=True, allow_nonzero_temperature=False, retry_failed=False, confirm_cost=False,
    )
    asyncio.run(vc.execute(name="t", items=[item(1)], chunks_of=lambda it: it["chunks"], args=ns))
    assert not (results / "PROMPT.txt").exists()


# ---------------------------------------------------------------- prompt-version mixing
# guard: the guard must scan the whole results directory, not just the one meta file this
# invocation is about to write.


def test_execute_refuses_to_mix_prompt_versions_across_different_run_letters_in_the_same_dir(
    tmp_path: Path, monkeypatch
):
    """A results directory holds *both* the A and the B test run under different file names
    (``t_runA.meta.json``, ``t_runB.meta.json``); the old single-file check only compared
    this invocation's own meta path, so a v1-run-A directory would silently accept a v2
    run B. The guard must scan every ``*_run*.meta.json`` in the directory."""
    import argparse

    results = tmp_path / "results"
    items = [item(1), item(2)]

    v1 = FakeVerifier()
    v1.prompt_version = "sha256:v1v1v1v1v1v1"
    monkeypatch.setattr(vc, "make_verifier", lambda *a, **k: v1)
    ns_a = argparse.Namespace(
        run="A", limit=None, concurrency=1, max_attempts=1, price_input=None,
        price_output=None, price_source_url=None, price_accessed=None, results_dir=results,
        dry_run=False, allow_nonzero_temperature=False, retry_failed=False, confirm_cost=True,
    )
    asyncio.run(vc.execute(name="t", items=items, chunks_of=lambda it: it["chunks"], args=ns_a))
    assert (results / "t_runA.jsonl").exists()

    v2 = FakeVerifier()
    v2.prompt_version = "sha256:v2v2v2v2v2v2"
    monkeypatch.setattr(vc, "make_verifier", lambda *a, **k: v2)
    ns_b = argparse.Namespace(
        run="B", limit=None, concurrency=1, max_attempts=1, price_input=None,
        price_output=None, price_source_url=None, price_accessed=None, results_dir=results,
        dry_run=False, allow_nonzero_temperature=False, retry_failed=False, confirm_cost=True,
    )
    with pytest.raises(SystemExit) as exc:
        asyncio.run(vc.execute(name="t", items=items, chunks_of=lambda it: it["chunks"], args=ns_b))
    assert exc.value.code == 2
    assert not (results / "t_runB.jsonl").exists()
    assert not (results / "t_runB.meta.json").exists()


def test_write_prompt_txt_refuses_to_overwrite_a_differing_prompt_record(tmp_path: Path):
    class V1:
        system_prompt = "prompt A, rule one."
        prompt_version = "sha256:aaaaaaaaaaaa"

    class V2:
        system_prompt = "prompt B, rule two."
        prompt_version = "sha256:bbbbbbbbbbbb"

    vc.write_prompt_txt(tmp_path, V1())
    written = (tmp_path / "PROMPT.txt").read_text(encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        vc.write_prompt_txt(tmp_path, V2())
    assert exc.value.code == 2
    assert (tmp_path / "PROMPT.txt").read_text(encoding="utf-8") == written  # untouched


def test_write_prompt_txt_is_idempotent_for_identical_content(tmp_path: Path):
    class V:
        system_prompt = "prompt A, rule one."
        prompt_version = "sha256:aaaaaaaaaaaa"

    vc.write_prompt_txt(tmp_path, V())
    vc.write_prompt_txt(tmp_path, V())  # no raise: identical content is a no-op
    assert (tmp_path / "PROMPT.txt").exists()


def test_find_prompt_version_conflict_none_for_a_fresh_or_matching_directory(tmp_path: Path):
    assert vc.find_prompt_version_conflict(tmp_path / "does-not-exist", "sha256:x") is None
    common.write_json(tmp_path / "t_runA.meta.json", {"prompt_version": "sha256:aaa"})
    assert vc.find_prompt_version_conflict(tmp_path, "sha256:aaa") is None
    conflict = vc.find_prompt_version_conflict(tmp_path, "sha256:bbb")
    assert conflict == (tmp_path / "t_runA.meta.json", "sha256:aaa")


# ---------------------------------------------------------------- parity with the real
# backend guard
#
# `verify_common.py` holds no copy of `app.services.fulltext._normalise_for_match` or of
# `apply_verification_guards` -- `ProductionVerifier.__init__` (`:304`) imports both
# directly from the real backend and binds them to `self._apply_guards` /
# `self._normalise_text` (see the module docstring's "Normalisation on both call paths").
# Every other test in this file fakes `app.services.fulltext` (`fake_backend`) to stay
# backend-free; this is the one deliberate exception, importing the real module to prove
# that the harness's guard call and the backend's own guard call are, literally, the same
# function, so the backend's line-break-hyphenation fold (or any future guard change) can
# never drift out of parity with what the harness scores, with no separate edit here.


def test_harness_guard_entry_point_has_parity_with_the_real_backend_normaliser():
    """Imports the real `app.services.fulltext.apply_verification_guards` (unlike every
    other test in this file) and confirms it resolves a line-break-hyphenation miss
    (`hss-verbatim-01`) exactly as `backend/tests/test_verification_guards.py` does. Since
    `ProductionVerifier` calls this exact function object, not a reimplementation, this is
    a direct parity check between the harness's guard call and the backend's own."""
    backend_root = Path(__file__).resolve().parents[2] / "backend"
    original_cwd = Path.cwd()
    path_added = str(backend_root) not in sys.path
    if path_added:
        sys.path.insert(0, str(backend_root))
    try:
        os.chdir(backend_root)
        from app.services.fulltext import apply_verification_guards
    finally:
        os.chdir(original_cwd)

    try:
        # The cached text hyphenates "post-test" at a line break; the model quoted the
        # contiguous spelling. Without folding the hyphenation, this would fire
        # `quote_not_verbatim` and demote a correct `verified` answer, exactly as
        # `ProductionVerifier` would score it, since it calls the identical function.
        chunk_text = (
            "Given the control group, there is a statistically significant difference "
            "between post-\ntest and pre-test overall scores (z=3.297, p=000<.0167)."
        )
        quote = (
            "there is a statistically significant difference between post-test and "
            "pre-test overall scores (z=3.297, p=000<.0167)."
        )
        status, machine_reasons, diagnostics = apply_verification_guards(
            "verified",
            claim_text=quote,
            evidence_quote=quote,
            chunk_texts=[chunk_text],
            chunks=[{"text": chunk_text}],
        )
        assert status == "verified"
        assert machine_reasons == []
    finally:
        if path_added:
            sys.path.remove(str(backend_root))
        for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
            sys.modules.pop(name, None)


def test_default_results_dir_is_the_result_of_record():
    """RESULTS_DIR must point at claims/results/v6 (the directory of record,
    evaluation/README.md "Results directories"), not a superseded version."""
    assert vc.RESULTS_DIR == vc.HERE / "results" / "v6"
