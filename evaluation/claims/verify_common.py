"""Shared machinery for the claim-verification evaluations (E2: SciFact and the HSS set).

Importers: ``claims/run_scifact.py``, ``claims/run_hss.py``, ``tests/test_claims_*.py``.
Interpreter: the **system** ``python`` for real runs (backend dependencies); the pure helpers
run anywhere and are unit-tested with the LLM boundary faked.

The runners call the production agent through the same shared verification policy
``app/services/fulltext.py::_verify_claims`` calls::

    prompt = format_verification_prompt(claim, chunks, title, authors)
    policy_result = await verify_claim_with_policy(claim, chunks, chunk_dicts, call_model)

where ``call_model`` builds one ``ModelPassAnswer`` per model call: runs
``get_claim_verification_agent().run(prompt, deps=AnalysisDependencies(...))``, records
``provenance_from_run(...)``, and normalises the raw output. ``verify_claim_with_policy``
relocates the quote (Part A, ``relocate_evidence_quotes``, restricted to provably neutral
edits) and guards (``apply_verification_guards``) --
the same public entry points ``_verify_claims`` itself calls, not a re-implementation, so
the numbers ``summarize.py`` scores are the numbers a user of the production service would
actually see -- and, exactly once when licensed (Part B), calls the model a second time with the
identical prompt and takes that result as final.

Recorded per row: ``result.output`` (``ClaimVerification``: status, evidence_quote,
explanation, suggested_revision, assertions), the provenance record (model_reported,
system_fingerprint, provider_response_id, tokens) for every call actually made,
``result.usage`` extras (cache_read_tokens, details), read as a property (see
``usage_extras``), never called as a method, and ``passes`` (one entry per call).
Items without chunks take production's deterministic ``no_full_text`` path and never reach
the model or a guard. ``DEEPSEEK_API_KEY`` is exported from the repo-root ``.env`` at run
time and never printed.

Row schema (one JSON line per item; ``called_at`` is ISO-8601 UTC at second resolution)::

    item_id, <item fields minus chunks/title/authors>, predicted_status (post-guard),
    model_status (pre-guard; None when no model call was made), machine_reasons (slugs for
    every guard that fired), passes (one entry per model call actually made), assertions
    (per-assertion verdicts, ``[]`` for a model or test double that does not offer them),
    evidence_quote, quote_is_verbatim (bool|null; whitespace-normalised, case-sensitive
    substring of the chunk text), quote_is_verbatim_casefold, explanation,
    suggested_revision, deterministic, called_at, latency_s, input_tokens (summed across
    every call), output_tokens (summed), cache_read_tokens (summed), usage_details (last
    call's), model_reported, system_fingerprint, provider_response_id, attempts, error

Runs are resumable (rows already present are skipped). Exceptions are retried
(``--max-attempts``) and finally recorded with ``predicted_status = "error"``; such rows are
kept on a plain re-run. ``--retry-failed`` removes them from the results file (atomic
rewrite) before the run so they are attempted again; the meta file records
``retried_failed``.

Working directory: ``app.*`` must be imported with ``backend/`` as the cwd (its ``Settings``
read ``.env`` relative to the cwd). ``parse_common`` resolves every path argument first and
``common.add_backend_to_path`` then changes directory; a failing settings import becomes a
readable ``SystemExit``.

Cost gate: a run that is not ``--dry-run`` spends money as soon as it starts, so
:func:`execute` refuses to start (exit 3) unless ``--confirm-cost`` is given; it first prints
the number of model calls it would make (items with chunks that are not yet in the results
file) and the list-price upper bound (:func:`cost_upper_bound`, peak cache-miss tariff,
``ASSUMED_TOKENS`` per set). ``--limit`` does not lift the gate.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import common  # noqa: E402
from common import (  # noqa: E402
    DEEPSEEK_PRICES,
    REPO_ROOT,
    Stopwatch,
    add_backend_to_path,
    append_jsonl,
    completed_ids,
    compute_cost,
    export_env_from_dotenv,
    import_backend_settings,
    latency_stats,
    load_dotenv_values,
    max_sentence_jaccard,
    now_iso,
    price_record,
    prune_failed_rows,
    quote_is_verbatim,
    read_json,
    read_jsonl,
    resolve_path_args,
    resolve_price_model,
    split_sentences,
    write_json,
)

DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results" / "v6"
DRYRUN_RESULTS_DIR = DATA_DIR / "_dryrun" / "results"
ENV_KEYS = ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL")
EVAL_PROJECT_ID = UUID("00000000-0000-0000-0000-00000000e7a1")
ITEM_ID = "item_id"
NO_FULL_TEXT_EXPLANATION = "No full-text chunks available for this paper."
COST_BASIS = "list price, tier by call time, cache-hit tokens at cache-hit rate"
RETRY_BASE_DELAY = 2.0
EXIT_COST_NOT_CONFIRMED = 3
# Cost-gate assumptions per set, measured on the 2026-09-03 smoke runs (README "Expected cost
# and time"): SciFact ~2,600 input tokens per call (incl. ~1,300 cached system-prompt
# tokens, billed here at the cache-miss rate = upper bound); HSS ~27,000 (whole paper);
# 1,000 output tokens per call as an upper bound including reasoning tokens.
ASSUMED_TOKENS: dict[str, tuple[int, int]] = {
    "scifact": (2_600, 1_000),
    "scifact-train": (2_600, 1_000),  # brief 5.1: run_scifact.py --split train writes this name
    "hss": (27_000, 1_000),
}
DEFAULT_ASSUMED_TOKENS: tuple[int, int] = (2_600, 1_000)
ROW_FIELDS: tuple[str, ...] = (
    "predicted_status",
    # The status/reasons the guards recorded. `model_status` is
    # `None` for the deterministic no_full_text path and for the DryRunVerifier lexical
    # stand-in, both of which never call a guard.
    "model_status",
    "machine_reasons",
    # Slugs that never change `predicted_status`, always a
    # sibling of `machine_reasons`, never one of its members (e.g. "numeric_not_in_source",
    # "centrality_unmarked"). Carries the "quote_relocated" slug too
    # whenever `quote_relocations` below is non-empty.
    "diagnostics",
    # One record per quote segment relocated onto the source's own
    # text before the guards ran (`app.services.fulltext.relocate_evidence_quotes`), each
    # ``{"type": "quote_relocated", "original", "replacement", "edit_distance",
    # "chunk_index"}``. ``[]`` for every row the DryRunVerifier or the deterministic
    # no_full_text path produced (neither calls relocation).
    "quote_relocations",
    # One record per model
    # call actually made (Part B, the bounded second pass) -- one for the common case, two
    # when the second pass fired. ``[]``
    # for the deterministic no_full_text path and the DryRunVerifier stand-in, neither of
    # which runs the shared verification policy.
    "passes",
    "assertions",
    "evidence_quote",
    # Every verbatim span used across all assertions;
    # `evidence_quote` is this list's first span.
    "evidence_quotes",
    # Peripheral assertions the v3 precedence rule marks absent, in prose.
    "unstated_details",
    "explanation",
    "suggested_revision",
    "deterministic",
    "called_at",
    "latency_s",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "usage_details",
    "model_reported",
    "system_fingerprint",
    "provider_response_id",
    "attempts",
    "error",
)
EMPTY_OUTCOME: dict[str, Any] = {
    "predicted_status": None,
    "model_status": None,
    "machine_reasons": [],
    "diagnostics": [],
    "quote_relocations": [],
    "passes": [],
    "assertions": [],
    "evidence_quote": None,
    "evidence_quotes": [],
    "unstated_details": [],
    "explanation": None,
    "suggested_revision": None,
    "deterministic": False,
    "called_at": None,
    "latency_s": None,
    "input_tokens": None,
    "output_tokens": None,
    "cache_read_tokens": None,
    "usage_details": None,
    "model_reported": None,
    "system_fingerprint": None,
    "provider_response_id": None,
    "attempts": 0,
    "error": None,
}


# --------------------------------------------------------------------------------------
# Cost gate
# --------------------------------------------------------------------------------------


def cost_upper_bound(
    n_calls: int, model: str | None, *, input_tokens: int, output_tokens: int
) -> float | None:
    """List-price upper bound (USD) for ``n_calls`` calls at the peak cache-miss tariff.

    ``None`` when ``model`` is not in :data:`common.DEEPSEEK_PRICES`.
    """
    name = resolve_price_model(model)
    if name is None:
        return None
    rates = DEEPSEEK_PRICES["models"][name]
    per_call = (
        input_tokens * rates["input_cache_miss"]["peak"] + output_tokens * rates["output"]["peak"]
    ) / 1_000_000.0
    return n_calls * per_call


def configured_model_hint() -> str:
    """``DEEPSEEK_MODEL`` from the environment or the repo-root ``.env``, else ``deepseek-chat``
    (the cost gate runs before the production verifier and the backend settings exist)."""
    value = os.environ.get("DEEPSEEK_MODEL")
    if not value:
        try:
            value = load_dotenv_values(REPO_ROOT / ".env").get("DEEPSEEK_MODEL")
        except OSError:
            value = None
    return value or "deepseek-chat"


def require_cost_confirmation(args: argparse.Namespace, name: str, n_calls: int) -> None:
    """Exit 3 with a readable estimate unless ``--dry-run`` or ``--confirm-cost`` was given."""
    if getattr(args, "dry_run", False) or getattr(args, "confirm_cost", False):
        return
    model = configured_model_hint()
    tokens_in, tokens_out = ASSUMED_TOKENS.get(name, DEFAULT_ASSUMED_TOKENS)
    bound = cost_upper_bound(n_calls, model, input_tokens=tokens_in, output_tokens=tokens_out)
    bound_txt = (
        f"<= {bound:.4f} USD at peak list price" if bound is not None
        else f"unknown ({model!r} is not in the DeepSeek price table)"
    )
    print(
        f"REFUSING TO START: this is a paid run of the production verifier: {n_calls} model "
        f"calls for set {name!r} (~{tokens_in} input + {tokens_out} output tokens each); "
        f"expected cost {bound_txt}. Re-run with --confirm-cost to proceed, or --dry-run to "
        "exercise the pipeline without an LLM."
    )
    raise SystemExit(EXIT_COST_NOT_CONFIRMED)


# --------------------------------------------------------------------------------------
# Verifiers
# --------------------------------------------------------------------------------------


def usage_extras(result: Any) -> dict[str, Any]:
    """``cache_read_tokens`` / ``usage_details`` from ``result.usage`` (not in provenance).

    ``AgentRunResult.usage`` is a plain property in pydantic-ai 2.x. In the pinned 1.107.0
    (``backend/requirements.txt``) it is presented through a callable-property shim
    (``pydantic_ai.run.AgentRunResult.usage``, ``@deprecated_callable_property``): attribute
    access already returns the resolved usage object, with real ``input_tokens`` /
    ``cache_read_tokens`` / ``details`` attributes of its own, and that same object is also
    callable for backward compatibility -- calling it with parentheses only re-emits the same
    value plus a ``PydanticAIDeprecationWarning``. It is read here as a property and never
    called for that already-resolved case, mirroring ``backend/app/schemas/provenance.py``'s
    ``model_configured``/``temperature`` handling of the same run result. A genuinely bare
    callable with no usage attributes of its own (an older pydantic-ai's plain bound method,
    or a test double) is still called, so that case is not silently broken.
    """
    usage: Any = getattr(result, "usage", None)
    if callable(usage) and not hasattr(usage, "input_tokens"):
        try:
            usage = usage()
        except Exception:  # pragma: no cover - defensive
            usage = None
    details = getattr(usage, "details", None)
    return {
        "cache_read_tokens": getattr(usage, "cache_read_tokens", None),
        "usage_details": dict(details) if isinstance(details, Mapping) and details else None,
    }


def deterministic_no_full_text() -> dict[str, Any]:
    """Production's model-free answer for an item without chunks (``_verify_claims``)."""
    return {
        **EMPTY_OUTCOME,
        "predicted_status": "no_full_text",
        "explanation": NO_FULL_TEXT_EXPLANATION,
        "deterministic": True,
        "called_at": now_iso(),
        "latency_s": 0.0,
    }


def _sum_optional(values: Any) -> int | None:
    """Sum of every non-``None`` value in *values*, or ``None`` when every one is ``None``
    (mirrors ``app.services.fulltext._aggregate_provenance``'s own ``_total`` helper): used
    to fold a claim's one or two verification-policy passes into one row-level token count."""
    present = [v for v in values if v is not None]
    return sum(present) if present else None


class ProductionVerifier:
    """The unchanged production claim-verification agent, prompt formatter and provenance.

    Aborts unless ``backend/`` exposes the required claim-verification interface
    (``CLAIM_VERIFICATION_PROMPT_VERSION`` and the rest of the imports below) and the
    deterministic settings pin ``temperature`` to 0.0 (``--allow-nonzero-temperature``
    overrides the latter only).
    """

    def __init__(
        self,
        *,
        backend_root: Path | None = None,
        dotenv_path: Path | None = None,
        allow_nonzero_temperature: bool = False,
    ) -> None:
        add_backend_to_path(backend_root)
        export_env_from_dotenv(ENV_KEYS, dotenv_path)
        try:
            from app.agents.claim_verification_agent import (
                CLAIM_VERIFICATION_PROMPT_VERSION,
                QUOTE_REPAIR_PROMPT_VERSION,
                VERIFICATION_PROMPT,
                format_quote_repair_prompt,
                format_verification_prompt,
                get_claim_verification_agent,
            )
            from app.agents.model_config import DETERMINISTIC_LONG_MODEL_SETTINGS
            from app.schemas.provenance import prompt_version, provenance_from_run
            from app.services.fulltext import (
                GUARD_DIGEST,
                QUOTE_RELOCATION_VERSION,
                VERIFICATION_POLICY_VERSION,
                ModelPassAnswer,
                normalise_verification_text,
                verify_claim_with_policy,
            )
        except ImportError as exc:
            raise SystemExit(
                f"backend/ does not expose the required claim-verification interface ({exc})"
            ) from exc
        from app.agents.analysis_agent import AnalysisDependencies

        settings = import_backend_settings()
        temperature = DETERMINISTIC_LONG_MODEL_SETTINGS.get("temperature")
        if temperature != 0.0 and not allow_nonzero_temperature:
            print(
                f"DETERMINISTIC_LONG_MODEL_SETTINGS temperature is {temperature!r}, not 0.0; "
                "pass --allow-nonzero-temperature to override",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if prompt_version(VERIFICATION_PROMPT) != CLAIM_VERIFICATION_PROMPT_VERSION:
            raise SystemExit("CLAIM_VERIFICATION_PROMPT_VERSION does not match VERIFICATION_PROMPT")

        self.agent = get_claim_verification_agent()
        self.deps = AnalysisDependencies(
            project_id=EVAL_PROJECT_ID,
            project_description=None,
            target_journal=None,
            citation_style="APA",
        )
        self.format_prompt = format_verification_prompt
        self.format_repair_prompt = format_quote_repair_prompt
        self.system_prompt = VERIFICATION_PROMPT
        self._provenance_from_run = provenance_from_run
        self._verify_claim_with_policy = verify_claim_with_policy
        self._model_pass_answer = ModelPassAnswer
        self._normalise_text = normalise_verification_text
        self.model_configured: str = str(getattr(settings, "deepseek_model", "deepseek-chat"))
        self.prompt_version: str = CLAIM_VERIFICATION_PROMPT_VERSION
        self.guard_digest: str = GUARD_DIGEST
        self.quote_relocation_version: str = QUOTE_RELOCATION_VERSION
        self.verification_policy_version: str = VERIFICATION_POLICY_VERSION
        self.repair_prompt_version: str = QUOTE_REPAIR_PROMPT_VERSION
        self.temperature: float | None = temperature

    async def __call__(
        self, claim: str, chunks: list[str], title: str, authors: list[str] | None
    ) -> dict[str, Any]:
        # The prompt is built once, outside the closure below, so the first call always
        # sends it unchanged -- mirrors ``app.services.fulltext._verify_claims``'s own
        # wiring. A licensed repair turn instead continues the SAME
        # conversation (``message_history`` from the first call's own ``all_messages()``)
        # with one further user turn, ``format_quote_repair_prompt``, naming the segments
        # that failed.
        prompt = self.format_prompt(claim, chunks, title, authors)
        provenance_records: list[Any] = []
        usage_records: list[dict[str, Any]] = []
        latest_call_result: dict[str, Any] = {}

        async def _call_model(repair_request: Any = None) -> Any:
            if repair_request is None:
                result = await self.agent.run(prompt, deps=self.deps)
            else:
                repair_prompt = self.format_repair_prompt(repair_request.failed_segments)
                result = await self.agent.run(
                    repair_prompt,
                    deps=self.deps,
                    message_history=latest_call_result["result"].all_messages(),
                )
            latest_call_result["result"] = result
            out = result.output
            # Plain dict of whatever the model returned: a real ``ClaimVerificationOutput``
            # (via ``model_dump``, so nested ``assertions`` are already plain dicts,
            # matching what ``_verify_claims`` stores) or a test double's ``SimpleNamespace``
            # (via ``vars``).
            out_dict = out.model_dump() if hasattr(out, "model_dump") else dict(vars(out))
            # This harness has its own path to the stored row (it
            # never goes through `app.services.fulltext._verify_one`), so it normalises the
            # same fields that path normalises, the same way, before the shared policy
            # relocates/guards -- which also normalises internally, but only for its own
            # logic; it returns no normalised text to the caller.
            explanation = self._normalise_text(
                out_dict.get("explanation"), strip_wrapping_quotes=True
            )
            suggested_revision = self._normalise_text(out_dict.get("suggested_revision"))
            evidence_quotes = [
                self._normalise_text(q) for q in (out_dict.get("evidence_quotes") or [])
            ]
            # `evidence_quote` keeps its name, type and position: set to the first span of
            # `evidence_quotes` when present, else the model's own (still normalised)
            # `evidence_quote` -- mirrors `_verify_one` (brief section 1.2, P3's gate).
            evidence_quote = (
                evidence_quotes[0] if evidence_quotes
                else self._normalise_text(out_dict.get("evidence_quote"))
            )
            assertions = [
                {
                    **a,
                    "quote": self._normalise_text(a.get("quote")),
                    "quotes": [self._normalise_text(q) for q in (a.get("quotes") or [])],
                }
                for a in (out_dict.get("assertions") or [])
            ]
            record = self._provenance_from_run(
                "claim_verification",
                result,
                model_configured=self.model_configured,
                temperature=self.temperature,
                prompt=self.system_prompt,
            )
            provenance_records.append(record)
            usage_records.append(usage_extras(result))
            return self._model_pass_answer(
                status=out_dict.get("status"),
                evidence_quote=evidence_quote,
                evidence_quotes=evidence_quotes,
                assertions=assertions,
                explanation=explanation,
                suggested_revision=suggested_revision,
                unstated_details=out_dict.get("unstated_details") or [],
                input_tokens=getattr(record, "input_tokens", None),
                output_tokens=getattr(record, "output_tokens", None),
                system_fingerprint=getattr(record, "system_fingerprint", None),
                model_reported=getattr(record, "model_reported", None),
            )

        # The shared verification policy: call the model, relocate the quote (Part A), guard,
        # and -- exactly once, exactly when licensed -- ask the model to repair its own quote
        # in the same conversation and take that repair turn's result as final (Part B). The
        # same function `app.services.fulltext._verify_claims` calls, not a re-implementation,
        # so this run's numbers are what a user of the production service would actually see.
        policy_result = await self._verify_claim_with_policy(
            claim, chunks, [{"text": c} for c in chunks], _call_model
        )
        final_record = provenance_records[-1]
        final_usage = usage_records[-1] if usage_records else {}
        return {
            "predicted_status": policy_result.status,
            "model_status": policy_result.model_status,
            "machine_reasons": policy_result.machine_reasons,
            "diagnostics": policy_result.diagnostics,
            "quote_relocations": policy_result.quote_relocations,
            "assertions": policy_result.assertions,
            "evidence_quote": policy_result.evidence_quote,
            "evidence_quotes": policy_result.evidence_quotes,
            "unstated_details": policy_result.unstated_details,
            "explanation": policy_result.explanation,
            "suggested_revision": policy_result.suggested_revision,
            # One
            # entry per model call actually made (Part B; one, or two when the repair turn fired),
            # each carrying its own ``repair_prompt_version`` (``None`` on the first pass),
            # so a reader can see a first-pass demotion beside the final outcome.
            "passes": policy_result.passes,
            "model_reported": getattr(final_record, "model_reported", None),
            "system_fingerprint": getattr(final_record, "system_fingerprint", None),
            "provider_response_id": getattr(final_record, "provider_response_id", None),
            # Summed across every call actually made, so a claim's total cost/token count
            # (``build_meta``'s ``total_input_tokens``/``total_cost``) reflects both calls
            # when the second pass fired, not only the first.
            "input_tokens": _sum_optional(r.input_tokens for r in provenance_records),
            "output_tokens": _sum_optional(r.output_tokens for r in provenance_records),
            "cache_read_tokens": _sum_optional(u.get("cache_read_tokens") for u in usage_records),
            "usage_details": final_usage.get("usage_details"),
        }


class DryRunVerifier:
    """Lexical stand-in used by ``--dry-run`` (no network, no LLM, deterministic)."""

    model_configured = "dry-run"
    prompt_version = "dry-run"
    guard_digest = "dry-run"
    quote_relocation_version = "dry-run"
    verification_policy_version = "dry-run"
    repair_prompt_version = "dry-run"
    temperature = 0.0

    async def __call__(
        self, claim: str, chunks: list[str], title: str, authors: list[str] | None
    ) -> dict[str, Any]:
        await asyncio.sleep(0.002)  # measurable, so the dry-run meta exercises the latency path
        if not chunks:
            status, quote, expl = "no_full_text", None, "dry-run: no chunks"
        else:
            sentences = [s for c in chunks for s in split_sentences(c)]
            best, idx = max_sentence_jaccard(claim, sentences)
            if best >= 0.5 and idx is not None:
                status, quote, expl = "verified", sentences[idx], f"dry-run: jaccard={best:.2f}"
            else:
                status, quote, expl = "unsupported", None, f"dry-run: jaccard={best:.2f}"
        return {
            "predicted_status": status,
            "evidence_quote": quote,
            "explanation": expl,
            "suggested_revision": None,
            "model_reported": "dry-run",
            "system_fingerprint": None,
            "provider_response_id": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "usage_details": None,
        }


async def verify_with_retries(
    verifier: Any,
    *,
    claim: str,
    chunks: list[str],
    title: str,
    authors: list[str] | None,
    max_attempts: int,
) -> dict[str, Any]:
    last_error = "unknown"
    called_at = now_iso()
    for attempt in range(1, max_attempts + 1):
        called_at = now_iso()
        outcome = None
        with Stopwatch() as sw:
            try:
                outcome = await verifier(claim, chunks, title, authors)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        latency = round(sw.seconds, 3)  # read *after* the block: the stopwatch is stopped here
        if outcome is not None:
            merged = {**EMPTY_OUTCOME, **outcome}
            merged["deterministic"] = False
            merged["called_at"] = called_at
            merged["latency_s"] = latency
            merged["attempts"] = attempt
            merged["error"] = None
            return merged
        if attempt < max_attempts:
            await asyncio.sleep(RETRY_BASE_DELAY**attempt if RETRY_BASE_DELAY else 0)
    return {
        **EMPTY_OUTCOME,
        "predicted_status": "error",
        "explanation": last_error[:500],
        "called_at": called_at,
        "attempts": max_attempts,
        "error": last_error,
    }


# --------------------------------------------------------------------------------------
# Rows, runs, meta
# --------------------------------------------------------------------------------------


def make_row(
    item: Mapping[str, Any], outcome: Mapping[str, Any], chunk_text: str
) -> dict[str, Any]:
    """One result row: item fields (minus chunk payload) + outcome + quote fidelity flags."""
    row = {k: v for k, v in item.items() if k not in ("chunks", "title", "authors")}
    for key in ROW_FIELDS:
        row[key] = outcome.get(key, EMPTY_OUTCOME.get(key))
    quote = outcome.get("evidence_quote")
    row["quote_is_verbatim"] = quote_is_verbatim(quote, chunk_text) if quote else None
    row["quote_is_verbatim_casefold"] = (
        quote_is_verbatim(quote, chunk_text, casefold=True) if quote else None
    )
    return row


async def run_items(
    items: Sequence[Mapping[str, Any]],
    verifier: Any,
    *,
    out_path: Path,
    concurrency: int,
    max_attempts: int,
    chunks_of: Callable[[Mapping[str, Any]], list[str]],
    row_hook: Callable[[dict[str, Any]], None] | None = None,
    retry_failed: bool = False,
) -> dict[str, int]:
    """Resumable concurrent loop; appends one row per item as soon as it finishes.

    Items whose ``chunks_of`` is empty take the deterministic ``no_full_text`` path and the
    verifier is never invoked for them. ``row_hook`` may add derived fields to a row before
    it is written (e.g. ``correct`` for the HSS set). With ``retry_failed`` rows whose
    ``predicted_status`` is ``"error"`` are removed first and attempted again.
    """
    retried = 0
    if retry_failed:
        retried = prune_failed_rows(out_path, lambda r: r.get("predicted_status") == "error")
        print(f"--retry-failed: removed {retried} error rows for re-verification")
    done = completed_ids(out_path, ITEM_ID)
    pending = [it for it in items if it[ITEM_ID] not in done]
    print(f"{len(items)} items, {len(pending)} pending (resume: {len(done)} done)")
    semaphore = asyncio.Semaphore(max(1, concurrency))
    progress = {"done": 0, "errors": 0, "deterministic": 0, "retried_failed": retried}

    async def worker(item: Mapping[str, Any]) -> None:
        chunks = chunks_of(item)
        if not chunks:
            outcome = deterministic_no_full_text()
            progress["deterministic"] += 1
        else:
            async with semaphore:
                outcome = await verify_with_retries(
                    verifier,
                    claim=item["claim"],
                    chunks=chunks,
                    title=item.get("title") or "",
                    authors=item.get("authors"),
                    max_attempts=max_attempts,
                )
        row = make_row(item, outcome, " ".join(chunks))
        if row_hook is not None:
            row_hook(row)
        append_jsonl(out_path, row)
        progress["done"] += 1
        if outcome.get("error"):
            progress["errors"] += 1
            print(f"  {item[ITEM_ID]}: ERROR {outcome['error'][:120]}")
        if progress["done"] % 25 == 0 or progress["done"] == len(pending):
            print(f"  {progress['done']}/{len(pending)} items ({progress['errors']} errors)")

    await asyncio.gather(*(worker(it) for it in pending))
    return progress


def _int(value: Any) -> int:
    return int(value or 0)


def total_call_cost(
    rows: Sequence[Mapping[str, Any]], model_configured: str | None
) -> float | None:
    """Sum of ``common.call_cost`` over model-answered rows (deterministic/error rows excluded).

    Returns ``None`` when ``common.call_cost`` (screening Task 0) is unavailable or when a
    row lacks the fields the tariff needs.
    """
    call_cost = getattr(common, "call_cost", None)
    if call_cost is None:
        return None
    total = 0.0
    for r in rows:
        if r.get("deterministic") or r.get("error"):
            continue
        model = r.get("model_reported") or model_configured
        if not model or not r.get("called_at") or r.get("input_tokens") is None:
            return None
        cost = call_cost(
            model,
            r["called_at"],
            _int(r.get("input_tokens")),
            _int(r.get("output_tokens")),
            _int(r.get("cache_read_tokens")),
        )
        if cost is None:
            return None
        total += float(cost)
    return total


def build_meta(
    rows: Sequence[Mapping[str, Any]],
    *,
    existing: Mapping[str, Any] | None,
    name: str,
    run: str,
    verifier: Any,
    price: Mapping[str, Any],
    dry_run: bool,
    session_started: str,
    n_items: int,
    limit: int | None,
    concurrency: int,
    retried_failed: int = 0,
) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]
    answered = [r for r in ok if not r.get("deterministic")]
    total_in = sum(_int(r.get("input_tokens")) for r in answered)
    total_out = sum(_int(r.get("output_tokens")) for r in answered)
    total_cache = sum(_int(r.get("cache_read_tokens")) for r in answered)
    finished = now_iso()
    sessions = list((existing or {}).get("sessions") or [])
    sessions.append(
        {"started": session_started, "finished": finished, "retried_failed": retried_failed}
    )
    model_configured = getattr(verifier, "model_configured", None)
    return {
        "name": name,
        "run": run,
        "dry_run": dry_run,
        "retried_failed": retried_failed,
        "started": (existing or {}).get("started") or session_started,
        "finished": finished,
        "sessions": sessions,
        "model_configured": model_configured,
        "model_reported": sorted({str(r["model_reported"]) for r in ok if r.get("model_reported")}),
        "system_fingerprints": sorted(
            {str(r["system_fingerprint"]) for r in ok if r.get("system_fingerprint")}
        ),
        "temperature": getattr(verifier, "temperature", None),
        "prompt_version": getattr(verifier, "prompt_version", None),
        # Which frozen guard set, and which version (if any) of the
        # quote-relocation step, this run's verifier used
        # (``app.services.fulltext.GUARD_DIGEST``/``QUOTE_RELOCATION_VERSION``).
        "guard_digest": getattr(verifier, "guard_digest", None),
        "quote_relocation_version": getattr(verifier, "quote_relocation_version", None),
        # Which version of the shared
        # call-relocate-guard[-repair-and-call-again] policy (Part B) this run's verifier used
        # (``app.services.fulltext.VERIFICATION_POLICY_VERSION``).
        "verification_policy_version": getattr(verifier, "verification_policy_version", None),
        # Which version of the repair turn's own prompt
        # template this run's verifier used
        # (``app.agents.claim_verification_agent.QUOTE_REPAIR_PROMPT_VERSION``).
        "repair_prompt_version": getattr(verifier, "repair_prompt_version", None),
        "concurrency": concurrency,
        "limit": limit,
        "n_items": n_items,
        "n_rows": len(rows),
        "n_errors": len(rows) - len(ok),
        "n_deterministic": sum(1 for r in rows if r.get("deterministic")),
        "status_counts": dict(Counter(str(r.get("predicted_status")) for r in rows)),
        "price": dict(price),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_cache_read_tokens": total_cache,
        "total_cost": total_call_cost(rows, model_configured),
        "cost_basis": COST_BASIS,
        "total_cost_flat": compute_cost(
            total_in, total_out, price.get("input_per_mtok"), price.get("output_per_mtok")
        ),
        "latency_s": latency_stats(
            [r["latency_s"] for r in answered if r.get("latency_s") is not None]
        ),
    }


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--run", required=True, choices=("A", "B"))
    ap.add_argument("--limit", type=int, default=None, help="first N claims only (smoke tests)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-attempts", type=int, default=3, help="1 call + retries per item")
    ap.add_argument("--price-input", type=float, default=None, help="USD per 1M input tokens")
    ap.add_argument("--price-output", type=float, default=None, help="USD per 1M output tokens")
    ap.add_argument("--price-source-url", default=None)
    ap.add_argument("--price-accessed", default=None, help="date the price page was read")
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", help="lexical stand-in; no LLM; data/_dryrun")
    ap.add_argument(
        "--confirm-cost",
        action="store_true",
        help="required for any run that is not --dry-run: acknowledges the paid API calls "
        "(the expected call count and list-price upper bound are printed first)",
    )
    ap.add_argument(
        "--retry-failed",
        action="store_true",
        help="drop rows with predicted_status=error from the results file and verify them again",
    )
    ap.add_argument(
        "--allow-nonzero-temperature",
        action="store_true",
        help="do not abort when the production settings pin a temperature other than 0.0",
    )


def parse_common(ap: argparse.ArgumentParser, argv: Sequence[str] | None) -> argparse.Namespace:
    """``ap.parse_args`` followed by :func:`common.resolve_path_args` (cwd changes later)."""
    return resolve_path_args(ap.parse_args(argv))


def resolve_results_dir(args: argparse.Namespace) -> Path:
    return args.results_dir or (DRYRUN_RESULTS_DIR if args.dry_run else RESULTS_DIR)


def make_verifier(dry_run: bool, *, allow_nonzero_temperature: bool = False) -> Any:
    if dry_run:
        return DryRunVerifier()
    return ProductionVerifier(allow_nonzero_temperature=allow_nonzero_temperature)


def find_prompt_version_conflict(
    results_dir: Path, prompt_version: str
) -> tuple[Path, str] | None:
    """First ``*_run*.meta.json`` in *results_dir* recorded under a different prompt version
    than *prompt_version*, or ``None`` when there is no conflict.

    Brief section 5.6: the guard must cover the whole
    results directory, not just the one ``<name>_run<X>.meta.json`` this invocation is about
    to write, because a single directory (``results/v2/``) is specified to hold *both* the A
    and the B test run under different file names. A single-file check only ever caught the
    case of re-running the exact same ``name``/``run`` pair under a new prompt; it never saw
    a same-directory, different-run-letter conflict (e.g. ``hss_runA.meta.json`` under v1
    and ``hss_runB`` about to be written under v2).
    """
    if not results_dir.exists():
        return None
    for meta_file in sorted(results_dir.glob("*_run*.meta.json")):
        other_version = read_json(meta_file, {}).get("prompt_version")
        if other_version and other_version != prompt_version:
            return meta_file, other_version
    return None


def write_prompt_txt(results_dir: Path, verifier: Any) -> None:
    """Write ``PROMPT.txt``: the byte-exact system prompt plus its sha (brief section 5.6).

    Skipped for verifiers with no real system prompt (``DryRunVerifier`` and other test
    doubles without a ``system_prompt`` attribute): there is nothing byte-exact to record.

    Refuses (``SystemExit(2)``) to overwrite an existing ``PROMPT.txt`` whose content
    differs: the file is what the manuscript cites as
    the results directory's version identifier, so a later run under a different prompt must
    never silently make it stop matching the rows an earlier run already wrote. Writing the
    identical content again (e.g. resuming, or writing run B into a directory run A already
    started under the same prompt) is a no-op, not an error.
    """
    prompt_text = getattr(verifier, "system_prompt", None)
    if prompt_text is None:
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "PROMPT.txt"
    content = f"{prompt_text}\n\n{verifier.prompt_version}\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return
        print(
            f"{path} already records a different prompt; refusing to overwrite it "
            "(use a different --results-dir)"
        )
        raise SystemExit(2)
    path.write_text(content, encoding="utf-8")


async def execute(
    *,
    name: str,
    items: Sequence[Mapping[str, Any]],
    chunks_of: Callable[[Mapping[str, Any]], list[str]],
    args: argparse.Namespace,
    row_hook: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Run all items for one evaluation set and write ``<name>_run<X>.jsonl`` + meta."""
    results_dir = resolve_results_dir(args)
    out_path = results_dir / f"{name}_run{args.run}.jsonl"
    meta_path = results_dir / f"{name}_run{args.run}.meta.json"

    # Provenance guard: a --results-dir already written by the other mode (dry-run vs paid)
    # must never be reused, or its rows would be silently accepted as this run's completed
    # work and the meta rewritten with the new mode's dry_run flag.
    existing_meta = read_json(meta_path, {})
    if existing_meta and bool(existing_meta.get("dry_run")) != bool(args.dry_run):
        existing_kind = "dry" if existing_meta.get("dry_run") else "paid"
        print(
            f"{meta_path.name} was written by a {existing_kind} run; refusing to mix run "
            "provenance (use a different --results-dir)"
        )
        raise SystemExit(2)

    already = completed_ids(out_path, ITEM_ID)
    n_calls = sum(1 for it in items if it[ITEM_ID] not in already and chunks_of(it))
    require_cost_confirmation(args, name, n_calls)  # before anything is built or written
    verifier = make_verifier(
        args.dry_run, allow_nonzero_temperature=getattr(args, "allow_nonzero_temperature", False)
    )
    # Prompt-version guard (brief section 5.6): mirrors the dry-run/paid guard above, but
    # scoped to the whole results directory, since a
    # directory holds both the A and the B test run under different file names.
    conflict = find_prompt_version_conflict(results_dir, verifier.prompt_version)
    if conflict is not None:
        conflict_path, conflict_version = conflict
        print(
            f"{conflict_path.name} was written by prompt {conflict_version}; refusing to "
            "mix prompt versions (use a different --results-dir)"
        )
        raise SystemExit(2)
    write_prompt_txt(results_dir, verifier)
    print(
        f"{name} run {args.run}: model_configured={verifier.model_configured} "
        f"temperature={verifier.temperature} prompt_version={verifier.prompt_version}"
    )
    session_started = now_iso()
    progress = await run_items(
        items,
        verifier,
        out_path=out_path,
        concurrency=args.concurrency,
        max_attempts=args.max_attempts,
        chunks_of=chunks_of,
        row_hook=row_hook,
        retry_failed=getattr(args, "retry_failed", False),
    )
    rows = read_jsonl(out_path)
    price = price_record(
        verifier.model_configured or "deepseek-chat",
        args.price_input,
        args.price_output,
        args.price_source_url,
        args.price_accessed,
    )
    meta = build_meta(
        rows,
        existing=read_json(meta_path),
        name=name,
        run=args.run,
        verifier=verifier,
        price=price,
        dry_run=args.dry_run,
        session_started=session_started,
        n_items=len(items),
        limit=args.limit,
        concurrency=args.concurrency,
        retried_failed=progress["retried_failed"],
    )
    write_json(meta_path, meta)
    print(
        f"wrote {out_path} ({meta['n_rows']} rows, {meta['n_errors']} errors, "
        f"{meta['n_deterministic']} deterministic; statuses={meta['status_counts']}) "
        f"and {meta_path}"
    )
    if meta["total_cost"] is None and not args.dry_run:
        print("  total_cost not computed (common.call_cost unavailable or rows incomplete)")
    return 0
