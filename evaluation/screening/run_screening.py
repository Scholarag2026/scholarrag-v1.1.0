"""Run the production relevance screener over a SYNERGY dataset (evaluation E1).

Interpreter: the **system** ``python`` (the backend dependencies -- pydantic-ai, pydantic-
settings -- are installed there, not in ``evaluation/.venv``).

The script imports ``app.agents.relevance_screener_agent.screen_papers`` from ``backend/``
unchanged and calls it in batches of ``--batch-size`` records (default 10) with up to
``--concurrency`` batches in flight (default 8). ``DEEPSEEK_API_KEY`` (and, if present,
``DEEPSEEK_BASE_URL`` / ``DEEPSEEK_MODEL``) are exported from the repo-root ``.env`` at run
time; the values are never printed.

Outputs (``--results-dir``, default ``evaluation/screening/results/v3/``, the results of
record)::

    <dataset>_run<A|B>.jsonl       one row per record:
        record_id, label (protocol label_field), label_included, label_abstract_screening,
        has_abstract, has_title, title (<= 300 chars), predicted (0/1/null), reason,
        padded_include (production's "no decision returned" padding, when it landed as a
        predicted=1 row), padded_decision (that same padding regardless of the status it
        landed on -- NEEDS_REVIEW today), pass_number
        (1 for an ordinary batch, 2/3 for a re-ask pass -- see below),
        batch_index, called_at (ISO-8601 UTC, set right before the call), call_id (uuid4 hex
        shared by the rows of one API call), latency_s / input_tokens / output_tokens (summed
        across every call of this call group, including the backend's own re-ask when one
        answered), calls (1 or 2, same reason),
        cache_read_tokens (null: screen_papers exposes no cache split), model_reported,
        system_fingerprint, reask_model_reported / reask_system_fingerprint (the backend
        re-ask's own identity, when one answered), provider_response_id, attempts
    <dataset>_run<A|B>.meta.json   started, finished, sessions, model_configured, distinct
        model_reported / system_fingerprints, temperature, prompt_version, prompt_version_flag
        (the --prompt-version this session was invoked with), abstract_cap (--abstract-cap,
        resolved to the backend's frozen ABSTRACT_CHAR_LIMIT when not given), n_batches,
        failures, n_missing_abstract / share_missing_abstract, n_padded_decisions (rows the
        model never returned a decision for, regardless of the status they were padded to),
        price (from common.price_record), total tokens, total_cost (common.call_cost per
        batch: list price, peak/off-peak tier by called_at, cache-miss assumed = upper
        bound), cost_basis, total_cost_flat (CLI prices), latency statistics, superseded_calls
        / superseded_input_tokens / superseded_output_tokens (calls whose every row a later
        re-ask pass, or a --retry-failed
        prune, has since removed from the results file -- already folded into n_calls/total
        tokens/total_cost/n_batches_completed above, reported separately here so the two
        contributions stay distinguishable), superseded_call_rows (the
        representative row of each of those calls, carried forward into the *next* session's
        own meta the same way reask_batches_run is, so a session that sends no calls at all
        does not silently drop an earlier session's superseded calls from its own totals)

The backend must expose ``app.agents.relevance_screener_agent.screen_papers`` returning
``ScreeningBatchResult`` with ``.include/.reasons/.provenance``; the script aborts
otherwise. It also aborts (exit 2) when the provenance reports a temperature other than
0.0 unless ``--allow-nonzero-temperature`` is given. Abstracts are passed as fetched; the
production prompt builder truncates them to ``ABSTRACT_CHAR_LIMIT`` characters (10,000
today) and shows "(no abstract)" for empty ones -- that product behaviour is what is
evaluated.

Record order: before batching, the dataset is shuffled deterministically with
``random.Random(f"{dataset}:{seed}")`` (``--record-order-seed``, default 20260902, recorded
in the meta file as ``record_order_seed``). SYNERGY exports list the human-included records
first, so file-order batches would be label-sorted -- a condition production never sees. The
shuffle is identical for runs A and B and across resumes; ``--limit`` applies after it.

Runs are resumable at record level: batches are rebuilt from the same shuffled order and
only the records of a batch that are *not* yet in the results file are sent (a batch that
was partially written -- e.g. after a ``--limit`` smoke run -- is completed with a smaller
call; a batch that is complete is skipped). No done record is ever re-sent or re-billed;
``batch_index`` stays that of the full-run batch map, so runs A and B keep identical batch
maps when both are run to completion. The meta file counts calls (``n_calls``) separately
from batches. A batch that raises ``ScreeningError`` is retried (``--max-attempts``, default 3 = two
retries); if it still fails, its records are written with ``predicted = null`` and
``reason = "UNSCREENED: <error>"`` so the failure is visible in the metrics. Those rows are
*kept* on a plain re-run; ``--retry-failed`` removes them from the results file (atomic
rewrite) before the run so they are attempted again, and the meta file records
``retried_failed``. ``--retry-failed`` also removes a row still
flagged ``padded_decision`` (a record every re-ask pass below left undecided), not only a
``predicted = null`` one -- the two ways a row can sit in the results file with no real model
decision behind it.

Production pads decisions the model did not return with the reason ``"no decision returned"``
(``NO_DECISION_REASON``); ``screen_papers`` itself now re-asks once, in a batch of just the
undecided records, before padding anything (backend ``app/agents/relevance_screener_agent.py``),
so a batch response short by only a few records is usually recovered by production without any
harness involvement. What still comes back padded after that is re-asked again here, at the
harness level: this session's own padded records are re-batched in groups of 3 (pass 2), and
whatever is still padded after that is re-asked one record at a time (pass 3, the last pass --
a record still undecided after it keeps the padded marker). ``pass_number`` records which pass
produced a row; ``padded`` remains the authoritative per-call count
(``ScreeningBatchResult.padded``, forwarded here as ``outcome["padded"]``) of *that call's*
still-undecided records. Every row is stored exactly as its own call returned it but flagged
``padded_decision = true`` regardless of the status it landed on, and counted in the meta file
(``n_padded_decisions``); ``padded_include`` / ``n_padded_include`` are kept as the narrower
legacy flag (true only when the padding happened to land as a predicted=1 row, the six
originally committed v1 runs' behaviour) so ``summarize.py`` can report metrics with and
without either kind of padding.

Working directory: importing ``app.*`` requires ``backend/`` as the cwd (its ``Settings``
read ``.env`` relative to the cwd and reject the repository-root ``.env``). ``main`` resolves
every path argument first and ``common.add_backend_to_path`` then changes directory.

``--dry-run`` exercises the whole pipeline without an LLM (a keyword heuristic stands in
for the screener) and writes to ``data/_dryrun/results/`` (git-ignored) so the real
results folder is never polluted.

Cost gate: every run that is not ``--dry-run`` spends money the moment it starts, so it
refuses to start (exit 3) unless ``--confirm-cost`` is given. Before exiting it prints the
number of batches it would send, the pass-1 estimate as the expected cost
(:func:`cost_upper_bound`: the pending records' own rendered character
count over ``CHARS_PER_INPUT_TOKEN`` plus ``PROMPT_FRAMING_TOKENS_PER_BATCH``, output tokens
per batch assumed at ``ASSUMED_OUTPUT_TOKENS_PER_BATCH``, at the peak cache-miss tariff), and
a separately labelled worst case that adds the re-ask cascade's own allowance
(:func:`reask_cost_allowance`: up to
:data:`MAX_REASK_CALLS_PER_RECORD` extra single-record calls per pending record -- the
backend's own internal re-ask runs on every call this harness makes, not only the pass-1
batch, so a record chased through the harness's own pass-2 and pass-3 re-asks can cost up to
five extra calls, not three: the pass-1 batch's internal re-ask, the pass-2 call and its own
internal re-ask, and the pass-3 call and its own internal re-ask). ``--limit`` does not lift
the gate: a smoke run is still a paid run.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import random
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import (  # noqa: E402
    DEEPSEEK_PRICES,
    REPO_ROOT,
    Protocol,
    Stopwatch,
    add_backend_to_path,
    append_jsonl,
    call_cost,
    completed_ids,
    compute_cost,
    export_env_from_dotenv,
    import_backend_settings,
    iter_jsonl,
    latency_stats,
    load_dotenv_values,
    load_protocol,
    now_iso,
    price_record,
    prune_failed_rows,
    read_json,
    read_jsonl,
    resolve_path_args,
    resolve_price_model,
    write_json,
)

DATA_DIR = HERE / "data"
# Shipped default: the results of record for the v3 screener prompt, matching the default
# of baselines.py, summarize.py and wos_gate_effect.py. Pass --results-dir to target a
# different tree, e.g. results/dev/<iteration> during prompt development.
RESULTS_DIR = HERE / "results" / "v3"
PROTOCOLS_DIR = HERE / "protocols"
DRYRUN_RESULTS_DIR = DATA_DIR / "_dryrun" / "results"
ENV_KEYS = ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL")
ROW_ID = "record_id"
DEFAULT_RECORD_ORDER_SEED = 20260902
NO_DECISION_REASON = "no decision returned"  # app.agents.relevance_screener_agent
# Cost-gate assumptions. The former fixed ASSUMED_INPUT_TOKENS_PER_BATCH
# (2,000) was not a bound: a real run sent 470,129 input tokens over 130 batches, whose
# printed peak-price cost (US$0.2860) undershot the actual (US$0.2864). CHARS_PER_INPUT_TOKEN
# (a conservative characters-per-token ratio) and PROMPT_FRAMING_TOKENS_PER_BATCH (system
# prompt, criteria block and per-record framing, independent of record count within one
# batch) are applied to the run's own rendered record text by :func:`cost_upper_bound`, so the
# estimate scales with what will actually be sent instead of a flat per-batch guess.
CHARS_PER_INPUT_TOKEN = 3.6
PROMPT_FRAMING_TOKENS_PER_BATCH = 700
ASSUMED_OUTPUT_TOKENS_PER_BATCH = 1_000
EXIT_COST_NOT_CONFIRMED = 3


# --------------------------------------------------------------------------------------
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------------------


def shuffle_records(
    records: Sequence[Mapping[str, Any]], dataset: str, seed: int
) -> list[dict[str, Any]]:
    """Deterministic per-dataset shuffle (``random.Random(f"{dataset}:{seed}")``).

    SYNERGY files list the included records first, so batching in file order would put all
    positives into the first batches. The same (dataset, seed) always yields the same order,
    which keeps runs A and B and resumed sessions on identical batches.
    """
    out = [dict(r) for r in records]
    random.Random(f"{dataset}:{seed}").shuffle(out)
    return out


def load_records(
    path: Path,
    limit: int | None = None,
    *,
    dataset: str | None = None,
    seed: int | None = DEFAULT_RECORD_ORDER_SEED,
) -> list[dict[str, Any]]:
    """Records of ``path`` in evaluation order: shuffled (when ``seed`` is given), then cut."""
    records = read_jsonl(path)
    if not records:
        raise SystemExit(f"no records in {path}; run fetch_synergy.py first")
    if seed is not None:
        records = shuffle_records(records, dataset or path.stem, seed)
    return records[:limit] if limit else records


def make_batches(
    records: Sequence[Mapping[str, Any]], batch_size: int
) -> list[list[dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return [list(records[i : i + batch_size]) for i in range(0, len(records), batch_size)]


def _approx_shown_chars(record: Mapping[str, Any], abstract_cap: int | None) -> int:
    """Approximate character count of what the model will be shown for one record: its
    title plus its abstract, the abstract capped at ``abstract_cap`` when given (the
    backend's real v2 head/marker/tail rendering can only shorten this further, never
    lengthen it, so this stays an upper bound on the record's own text)."""
    title = str(record.get("title") or "")
    abstract = str(record.get("abstract") or "")
    if abstract_cap is not None and len(abstract) > abstract_cap:
        abstract = abstract[:abstract_cap]
    return len(title) + len(abstract)


def cost_upper_bound(
    batches: Sequence[Sequence[Mapping[str, Any]]],
    model: str | None,
    *,
    abstract_cap: int | None = None,
) -> float | None:
    """List-price upper bound (USD) for the given pending ``batches`` at the peak
    cache-miss tariff, from the records' own rendered text rather
    than a flat per-batch token guess: each batch's input tokens are estimated as its
    records' combined title+abstract character count (capped at ``abstract_cap`` when
    given) divided by :data:`CHARS_PER_INPUT_TOKEN`, plus
    :data:`PROMPT_FRAMING_TOKENS_PER_BATCH` for the system prompt, criteria block and
    per-record framing; output tokens per batch are the existing
    :data:`ASSUMED_OUTPUT_TOKENS_PER_BATCH`. ``None`` when ``model`` is not in
    :data:`common.DEEPSEEK_PRICES`.
    """
    name = resolve_price_model(model)
    if name is None:
        return None
    rates = DEEPSEEK_PRICES["models"][name]
    total = 0.0
    for batch in batches:
        chars = sum(_approx_shown_chars(r, abstract_cap) for r in batch)
        input_tokens = chars / CHARS_PER_INPUT_TOKEN + PROMPT_FRAMING_TOKENS_PER_BATCH
        total += (
            input_tokens * rates["input_cache_miss"]["peak"]
            + ASSUMED_OUTPUT_TOKENS_PER_BATCH * rates["output"]["peak"]
        ) / 1_000_000.0
    return total


#: How many extra single-record calls
#: one pending record's worst-case re-ask chain can cost, at most. Five, not three:
#: ``screen_papers`` re-asks internally whenever *its own* response is short, regardless of
#: who called it, so the backend's internal re-ask fires on the harness's own pass-2 and
#: pass-3 calls too, not only on the pass-1 batch. The five are: the pass-1 batch's own
#: internal re-ask, the pass-2 batch-of-3 call, that call's own internal re-ask, the pass-3
#: single-record call, and that call's own internal re-ask. Pricing every one of the five as
#: a single-record call is deliberately pessimistic for the two batch-of-3 calls (a real
#: pass-2 call and its internal re-ask each cover up to 3 records, so their true worst case is
#: cheaper than 3 separate single-record calls -- fewer, larger calls always cost at most as
#: much as more, smaller ones under a fixed per-call framing overhead), which keeps this a
#: safe upper bound rather than a tight one.
MAX_REASK_CALLS_PER_RECORD = 5


def reask_cost_allowance(
    records: Sequence[Mapping[str, Any]],
    model: str | None,
    *,
    abstract_cap: int | None = None,
) -> float | None:
    """List-price upper bound (USD) of the extra calls the re-ask cascade
    can add on top of :func:`cost_upper_bound`'s pass-1 estimate, if every one of ``records``
    needed the full chain: the pass-1 batch estimate
    alone prices none of the backend's own internal re-ask or this harness's two further
    passes, so a pathological run (every record left undecided through every pass) can spend
    far more than a bound computed from pass-1 batches alone. Each record is priced as up to
    :data:`MAX_REASK_CALLS_PER_RECORD` extra single-record calls, at the same
    :data:`CHARS_PER_INPUT_TOKEN` / :data:`PROMPT_FRAMING_TOKENS_PER_BATCH` /
    :data:`ASSUMED_OUTPUT_TOKENS_PER_BATCH` peak-tariff assumptions as
    :func:`cost_upper_bound`. ``None`` when ``model`` is not in :data:`common.DEEPSEEK_PRICES`.
    """
    name = resolve_price_model(model)
    if name is None:
        return None
    rates = DEEPSEEK_PRICES["models"][name]
    total = 0.0
    for record in records:
        chars = _approx_shown_chars(record, abstract_cap)
        input_tokens = chars / CHARS_PER_INPUT_TOKEN + PROMPT_FRAMING_TOKENS_PER_BATCH
        per_call = (
            input_tokens * rates["input_cache_miss"]["peak"]
            + ASSUMED_OUTPUT_TOKENS_PER_BATCH * rates["output"]["peak"]
        ) / 1_000_000.0
        total += MAX_REASK_CALLS_PER_RECORD * per_call
    return total


def protocol_sha(path: Path) -> str:
    """``sha256:`` + first 12 hex digits of a protocol file's raw bytes,
    the same convention ``app.agents.model_config.prompt_version`` uses for a system prompt,
    so a run whose protocol changed after the freeze is as detectable as one whose prompt did.
    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest[:12]}"


def full_text_criteria_ids(protocol: Protocol) -> list[str]:
    """Every criterion id whose stage is "full_text", inclusion ids in
    ascending order followed by exclusion ids in ascending order, recorded in the run meta."""
    ids = [f"I{i}" for i, s in enumerate(protocol.inclusion_stages, start=1) if s == "full_text"]
    ids += [f"E{i}" for i, s in enumerate(protocol.exclusion_stages, start=1) if s == "full_text"]
    return ids


def configured_model_hint() -> str:
    """``DEEPSEEK_MODEL`` from the environment or the repo-root ``.env``, else ``deepseek-chat``.

    Used only for the cost-gate message, which must be printed *before* the production
    screener (and hence the backend settings) is constructed.
    """
    value = os.environ.get("DEEPSEEK_MODEL")
    if not value:
        try:
            value = load_dotenv_values(REPO_ROOT / ".env").get("DEEPSEEK_MODEL")
        except OSError:
            value = None
    return value or "deepseek-chat"


def require_cost_confirmation(
    args: argparse.Namespace,
    pending_batches: Sequence[Sequence[Mapping[str, Any]]],
    model: str | None,
) -> None:
    """Exit 3 with a readable estimate unless ``--dry-run`` or ``--confirm-cost`` was given.

    ``pending_batches`` are the batches (of records) that would actually be sent, each
    record already trimmed to the not-yet-done ones: the estimate is
    computed from their own rendered text, not a flat per-batch guess.

    The printed message leads with the pass-1 estimate (:func:`cost_upper_bound`) as the
    expected cost, then states a separately labelled worst case that adds a re-ask allowance
    (:func:`reask_cost_allowance`): the
    pass-1 estimate alone prices none of the backend's own internal re-ask nor this harness's
    own pass-2/pass-3 re-ask cascade, so a pathological run
    (every pending record left undecided through every pass) could spend far more than the
    pass-1 figure alone. Leading with the pass-1 figure as "expected" and the wider figure as
    a separately labelled "worst case" (rather than calling the widened total "expected", as
    an earlier version of this message did) keeps the headline number close to what an
    ordinary run actually costs, while the worst case and its own breakdown stay printed and
    checkable for the operator who wants them.
    """
    if args.dry_run or getattr(args, "confirm_cost", False):
        return
    n_pending_batches = len(pending_batches)
    abstract_cap = getattr(args, "abstract_cap", None)
    pending_records = [r for batch in pending_batches for r in batch]
    bound = cost_upper_bound(pending_batches, model, abstract_cap=abstract_cap)
    reask_bound = reask_cost_allowance(pending_records, model, abstract_cap=abstract_cap)
    if bound is not None and reask_bound is not None:
        total_bound = bound + reask_bound
        cost_txt = (
            f"<= {bound:.4f} USD at peak list price for pass 1; worst case with the full "
            f"re-ask cascade <= {total_bound:.4f} USD ({bound:.4f} pass 1 + "
            f"{reask_bound:.4f} worst-case re-ask allowance: up to "
            f"{MAX_REASK_CALLS_PER_RECORD} extra single-record calls per pending record)"
        )
    else:
        cost_txt = f"unknown ({model!r} is not in the DeepSeek price table)"
    print(
        f"REFUSING TO START: this is a paid run of the production screener over "
        f"{n_pending_batches} batches; expected cost {cost_txt}. "
        "Re-run with --confirm-cost to proceed, or --dry-run to exercise the pipeline "
        "without an LLM."
    )
    raise SystemExit(EXIT_COST_NOT_CONFIRMED)


def gold_label(record: Mapping[str, Any], label_field: str) -> int | None:
    value = record.get(label_field)
    return None if value is None else int(value)


def _sum_optional_tokens(a: int | None, b: int | None) -> int | None:
    """``a + b`` treating a missing side as 0, unless both are missing (``None``): a batch
    with no re-ask must keep reporting "unknown" as ``None``, not ``0``."""
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


def _is_retryable_row(row: Mapping[str, Any]) -> bool:
    """``--retry-failed`` selects a row with ``predicted = null`` (a genuine call failure)
    and a row still flagged ``padded_decision`` (a record every
    re-ask pass, including the single-record one, left undecided) -- the two ways a row can
    be present in the results file without a real model decision behind it."""
    return row.get("predicted") is None or bool(row.get("padded_decision"))


def _is_unbilled_failure_row(row: Mapping[str, Any]) -> bool:
    """A row from a call that raised (``predicted = null``, ``reason`` starting
    ``"UNSCREENED"``) rather than one padded by a real response:
    ``screen_batch_with_retries`` reports no usage at all for a
    raised call (see its own ``error`` branch), so such a call has no tokens to lose when
    ``--retry-failed`` prunes it, and it is already correctly reflected by ``failures`` going
    back to 0 once every one of its rows is gone and replaced by a successful retry -- unlike
    a padded row, which came from a real, billed response and must not disappear from the
    totals. The two are mutually exclusive in practice (a batch either raises for every one of
    its records or returns a real response for all of them; padded rows never carry
    ``predicted = null``), so this only ever excludes the former from the calls carried
    forward across a ``--retry-failed`` prune."""
    return row.get("predicted") is None and str(row.get("reason") or "").startswith("UNSCREENED")


def _fully_pruned_calls(
    rows: Sequence[Mapping[str, Any]], is_pruned: Callable[[Mapping[str, Any]], bool]
) -> list[dict[str, Any]]:
    """One representative row per call_id every one of whose rows in ``rows`` satisfies
    ``is_pruned``: a call some but not
    all of whose rows are about to be removed keeps a surviving row in the results file, so
    :func:`build_meta`'s own per-call_id grouping already prices it correctly from that row;
    a call about to lose *every* one of its rows leaves no trace at all once the removal
    happens, so its representative row has to be captured here, before the removal, to be
    handed to :func:`build_meta` directly or carried forward in the meta for a later session.
    Used both for ``--retry-failed``'s own prune (rows made in an earlier session, removed
    before this session sends anything) and, in principle, any other whole-row-group prune;
    the sub-batch re-ask cascade's own prune already has an equivalent, session-scoped
    mechanism (``call_records`` in ``main_async``) because it also needs to observe calls this
    same session is about to make, which have no rows to inspect yet when they are recorded.
    """
    by_call: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        call_id = row.get("call_id")
        if call_id is not None:
            by_call.setdefault(call_id, []).append(row)
    return [
        dict(group[0]) for group in by_call.values() if group and all(is_pruned(r) for r in group)
    ]


def rows_for_batch(
    batch: Sequence[Mapping[str, Any]],
    batch_index: int,
    outcome: Mapping[str, Any],
    label_field: str,
    *,
    pass_number: int = 1,
) -> list[dict[str, Any]]:
    """Turn one batch outcome (success or terminal failure) into result rows.

    ``status``/``criterion``/``quote``/``guard_applied``/``guard_reason`` are read from the
    outcome's parallel
    ``statuses``/``criteria_ids``/``quotes``/``guard_applied``/``guard_reasons`` lists when
    present (the production and dry-run screeners always supply them). A legacy outcome that
    carries only the legacy binary ``include``/``reasons`` (a test double, or an older
    harness build) gets a derived two-way ``status`` (``"INCLUDE"``/``"EXCLUDE"`` from
    ``predicted``) and empty ``criterion``/``quote``/``guard_applied``/``guard_reason``, so
    every row always carries a ``status`` column.

    ``padded_decision`` flags a row the model never
    returned a decision for, regardless of the status the padding landed on: true when the
    outcome's ``padded`` count (forwarded from ``ScreeningBatchResult.padded`` by
    :class:`ProductionScreener`) places this record's position in the trailing padded block, or
    when its ``reason`` is exactly :data:`NO_DECISION_REASON` (the two signals agree in
    production; the second alone covers test doubles that set ``reasons`` without ``padded``).

    ``to_confirm`` is the full-text inclusion criterion ids still to be
    confirmed at full text, read from the outcome's parallel ``to_confirm`` list of lists; an
    empty list when the outcome carries none (every pre-S8c test double and outcome shape).

    ``pass_number`` is the re-ask pass this batch was sent in: 1 for an
    ordinary batch (every pre-fix call site, unaffected), 2 for the batch-of-3 re-ask of
    records a pass-1 response padded, 3 for the single-record re-ask of records still padded
    after pass 2. Recorded on every row of the batch, not only a padded one, so a resolved
    record's history (it took a re-ask to decide it) stays visible too.

    ``calls`` is 2 when the backend's own internal
    re-ask (inside ``screen_papers``, distinct from this harness's own pass 2/3 above)
    answered for this batch, 1 otherwise -- defaulted to 1 when the outcome carries none
    (every pre-fix outcome shape). ``reask_model_reported``/``reask_system_fingerprint``
    carry that re-ask call's own identity when it differs from the batch's ``model_reported``/
    ``system_fingerprint``.
    """
    error = outcome.get("error")
    include = outcome.get("include") or []
    reasons = outcome.get("reasons") or []
    statuses = outcome.get("statuses") or []
    criteria_ids = outcome.get("criteria_ids") or []
    quotes = outcome.get("quotes") or []
    guard_applied = outcome.get("guard_applied") or []
    guard_reasons = outcome.get("guard_reasons") or []
    to_confirm_lists = outcome.get("to_confirm") or []
    anchor_lists = outcome.get("anchors") or []
    second_pass_answers = outcome.get("second_pass_answers") or []
    # One entry per paper, not a
    # single shared dict -- a legacy outcome shape (an older ProductionScreener,
    # or a test double) that still returns the old singular "second_pass_call" degrades to
    # every row sharing that one dict, exactly the prior behaviour.
    second_pass_calls = outcome.get("second_pass_calls")
    if second_pass_calls is None:
        legacy_call = outcome.get("second_pass_call")
        second_pass_calls = [legacy_call] * len(batch) if legacy_call else []
    n_padded = int(outcome.get("padded") or 0)
    n = len(batch)
    rows: list[dict[str, Any]] = []
    for i, rec in enumerate(batch):
        if error:
            predicted, reason, status = None, f"UNSCREENED: {error}", None
        else:
            predicted = int(bool(include[i]))
            reason = reasons[i] if i < len(reasons) else ""
            status = statuses[i] if i < len(statuses) else ("INCLUDE" if predicted else "EXCLUDE")
        criterion = criteria_ids[i] if i < len(criteria_ids) else ""
        quote = quotes[i] if i < len(quotes) else ""
        applied = bool(guard_applied[i]) if i < len(guard_applied) else False
        guard_reason = guard_reasons[i] if i < len(guard_reasons) else ""
        to_confirm = list(to_confirm_lists[i]) if i < len(to_confirm_lists) else []
        anchors = list(anchor_lists[i]) if i < len(anchor_lists) else []
        second_pass = second_pass_answers[i] if i < len(second_pass_answers) else None
        second_pass_call = second_pass_calls[i] if i < len(second_pass_calls) else None
        padded_decision = (not error) and (
            reason == NO_DECISION_REASON or (n_padded > 0 and i >= n - n_padded)
        )
        rows.append(
            {
                ROW_ID: rec[ROW_ID],
                "label": gold_label(rec, label_field),
                "label_included": rec.get("label_included"),
                "label_abstract_screening": rec.get("label_abstract_screening"),
                "has_abstract": bool((rec.get("abstract") or "").strip()),
                "has_title": bool((rec.get("title") or "").strip()),
                "title": (rec.get("title") or "")[:300],
                # OpenAlex's own work type/paratext flag, when fetch_synergy.py
                # exported them (route v2 datasets only; "" and False on a route v1 export
                # such as Nagtegaal_2019/van_de_Schoot_2017, which carries no OpenAlex id at
                # all -- see the protocol design's own section 1b measurement).
                "type": rec.get("type") or "",
                "is_paratext": bool(rec.get("is_paratext") or False),
                "predicted": predicted,
                "reason": reason,
                "status": status,
                "criterion": criterion,
                "quote": quote,
                "guard_applied": applied,
                "guard_reason": guard_reason,
                "to_confirm": to_confirm,
                # The raw anchor ledger this record's
                # decision carried, whatever the guard did with the decision itself --
                # ``[]`` for a v1/v2 run, or for any decision the model did not put a
                # ledger on. Lets an offline pass re-score any anchor set from these rows.
                "anchors": anchors,
                # The second pass's own
                # per-question answer for this record ({"population", "outcome",
                # "study_type"}), or None when it never reached the second pass. The
                # SECOND_PASS_STAGE_BATCH_SIZE-sized chunk call that produced it is
                # identified by ``second_pass_call_id``/``second_pass_model_reported``/token
                # fields below, so ``summarize.py`` can price it without double-counting a
                # chunk's shared call across every row it covers.
                "second_pass": second_pass,
                "second_pass_call_id": (
                    second_pass_call.get("call_id")
                    if second_pass is not None and second_pass_call
                    else None
                ),
                "second_pass_model_reported": (
                    second_pass_call.get("model_reported")
                    if second_pass is not None and second_pass_call
                    else None
                ),
                "second_pass_input_tokens": (
                    second_pass_call.get("input_tokens")
                    if second_pass is not None and second_pass_call
                    else None
                ),
                "second_pass_output_tokens": (
                    second_pass_call.get("output_tokens")
                    if second_pass is not None and second_pass_call
                    else None
                ),
                "padded_include": predicted == 1 and reason == NO_DECISION_REASON,
                "padded_decision": padded_decision,
                "pass_number": pass_number,
                "batch_index": batch_index,
                "called_at": outcome.get("called_at"),
                "call_id": outcome.get("call_id"),
                "latency_s": outcome.get("latency_s"),
                "input_tokens": outcome.get("input_tokens"),
                "output_tokens": outcome.get("output_tokens"),
                "cache_read_tokens": None,  # screen_papers exposes no cache split
                "model_reported": outcome.get("model_reported"),
                "system_fingerprint": outcome.get("system_fingerprint"),
                "provider_response_id": outcome.get("provider_response_id"),
                "attempts": outcome.get("attempts"),
                # 2 when the backend's own re-ask (inside
                # screen_papers) answered for this batch, else 1 -- build_meta sums this per
                # call group instead of counting one call per call_id, so n_calls reflects
                # every real API call, not one per batch. reask_model_reported/
                # reask_system_fingerprint carry the re-ask's own identity, kept distinct
                # from model_reported/system_fingerprint so a change between the batch's two
                # calls is still visible in the run meta rather than silently dropped.
                "calls": int(outcome.get("calls") or 1),
                "reask_model_reported": outcome.get("reask_model_reported"),
                "reask_system_fingerprint": outcome.get("reask_system_fingerprint"),
                # Set only when the backend's own one re-ask call
                # raised, so a still-padded record it produced is distinguishable from one
                # the model just quietly skipped again.
                "reask_error": outcome.get("reask_error"),
            }
        )
    return rows


def build_meta(
    rows: Sequence[Mapping[str, Any]],
    *,
    existing: Mapping[str, Any] | None,
    dataset: str,
    run: str,
    protocol: Protocol,
    n_records: int,
    n_batches: int,
    batch_size: int,
    concurrency: int,
    model_configured: str | None,
    temperature: float | None,
    prompt_versions: Sequence[str],
    price: Mapping[str, Any],
    dry_run: bool,
    session_started: str,
    limit: int | None,
    record_order_seed: int | None = DEFAULT_RECORD_ORDER_SEED,
    retried_failed: int = 0,
    prompt_version_flag: str | None = None,
    abstract_cap: int | None = None,
    protocol_sha_value: str | None = None,
    full_text_criteria: Sequence[str] | None = None,
    reask_batches_run: int = 0,
    superseded_call_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate per-call latency/tokens/cost from all rows (idempotent across resumes).

    Rows are grouped into *calls* by ``call_id`` (fallback for rows without one:
    ``(batch_index, called_at, provider_response_id)``): a batch that was completed over two
    sessions (record-level resume) contributes two calls, so no tokens or latency are lost.
    ``n_calls`` counts them; ``n_batches_completed`` counts distinct batches with at least one
    successful call. ``total_cost`` sums :func:`common.call_cost` per successful call (list
    price for the reported model, peak/off-peak tier by ``called_at``, cache-miss assumed =
    upper bound). ``total_cost_flat`` uses the CLI ``--price-input/--price-output`` pair when
    given.

    ``prompt_version_flag``/``abstract_cap`` are the
    CLI ``--prompt-version``/``--abstract-cap`` values this session actually ran with, distinct
    from ``prompt_version`` (the set of provenance shas reported back by the model calls
    themselves): the caller resolves a ``None`` ``--abstract-cap`` to the backend's frozen
    ``ABSTRACT_CHAR_LIMIT`` before calling this function, so ``abstract_cap`` in the written
    meta is always the number actually used, not merely "not overridden".

    ``protocol_sha_value``/``full_text_criteria`` are, respectively,
    :func:`protocol_sha` of the loaded protocol file and the full-text criterion ids
    (:func:`full_text_criteria_ids`): recorded in the meta so a post-freeze scoring pass can
    assert the protocol did not change underneath a run, the same way it already asserts the
    prompt sha.

    ``reask_batches_run`` is the cumulative count of re-ask
    sub-batches run across every session so far -- the caller seeds its own counter from the
    existing meta's own ``reask_batches_run`` (the same way it carries ``sessions`` forward)
    before adding this session's own re-ask sub-batches, so it already includes every earlier
    session's when it reaches here; written back unchanged so the next session can do the same.

    ``superseded_call_rows`` is one representative row per real call
    whose entire row group is absent from ``rows``, whichever session made that call: a
    pass-2 call every one of whose records stayed padded through pass 3 has its rows removed
    one at a time, by three separate single-record pass-3 sub-batches, so by the time the
    last one prunes, ``rows`` no longer contains any trace of that pass-2 call at all; a
    ``--retry-failed`` prune can do the same to a call made in an *earlier* session. Grouping
    ``rows`` by ``call_id`` alone would therefore silently drop such a call from ``n_calls``,
    the token/cost totals and ``n_batches_completed`` the moment its last row is gone, even
    though it was a real, paid call -- and, unlike ``reask_batches_run``, there is no running
    counter to under-count *into*, since a superseded call's cost depends on its own
    ``model_reported``/``called_at``, not just a token count. The caller (``main_async``)
    accumulates these representative rows two ways: within a session, one row per call as
    ``run_one_batch`` returns it, checked at the end against which call_ids are no longer
    present in ``rows`` at all (a later re-ask pass may have pruned it); across sessions, this
    function's own ``superseded_call_rows`` output field from the *previous* session's written
    meta, the same way ``reask_batches_run`` is seeded from it, plus whatever
    ``--retry-failed``'s own prune (of rows made in an earlier session) is about to remove
    this session, captured before the prune runs. A call_id that still has a surviving row in
    ``rows`` is never double-counted, regardless of which of those three sources supplied it.
    Folded into ``n_calls``/the token and cost totals/``n_batches_completed`` the same way a
    surviving call is; also reported as the int fields ``superseded_calls``/
    ``superseded_input_tokens``/``superseded_output_tokens`` (so the two contributions to the
    totals stay distinguishable in the written meta) and, distinctly, as the list field
    ``superseded_call_rows`` (the representative rows actually folded in this session, for the
    next session to seed its own carry-forward from -- named differently from this parameter's
    *output* counterpart deliberately: the parameter is a sequence of row mappings, the
    ``superseded_calls`` output field is an int, and conflating the two names invites feeding
    the wrong one back in).
    """
    per_call: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key: tuple[Any, ...] = (
            (row["call_id"],)
            if row.get("call_id")
            else (row["batch_index"], row.get("called_at"), row.get("provider_response_id"))
        )
        per_call.setdefault(key, row)
    existing_call_ids = {row.get("call_id") for row in rows if row.get("call_id")}
    superseded_by_id: dict[Any, Mapping[str, Any]] = {}
    for row in superseded_call_rows or []:
        call_id = row.get("call_id")
        if call_id is not None and call_id not in existing_call_ids:
            superseded_by_id.setdefault(call_id, row)
    calls = list(per_call.values()) + list(superseded_by_id.values())
    failed = [
        b
        for b in calls
        if b.get("predicted") is None and str(b.get("reason", "")).startswith("UNSCREENED")
    ]
    ok = [b for b in calls if b not in failed]
    total_in = sum(int(b["input_tokens"] or 0) for b in ok)
    total_out = sum(int(b["output_tokens"] or 0) for b in ok)
    costs = [
        call_cost(
            b.get("model_reported") or model_configured,
            b.get("called_at"),
            b.get("input_tokens"),
            b.get("output_tokens"),
            0,
        )
        for b in ok
    ]
    total_cost = sum(costs) if ok and all(c is not None for c in costs) else None
    n_missing_abstract = sum(1 for r in rows if r.get("has_abstract") is False)
    n_missing_title = sum(1 for r in rows if r.get("has_title") is False)
    # A call group's ``reask_model_reported``/
    # ``reask_system_fingerprint`` (the backend's internal re-ask, when one answered) is
    # folded into the same distinct sets as the batch's own model/fingerprint, so a change
    # between a batch's two calls is visible here rather than dropped.
    models = sorted(
        {str(b["model_reported"]) for b in ok if b.get("model_reported")}
        | {str(b["reask_model_reported"]) for b in ok if b.get("reask_model_reported")}
    )
    fingerprints = sorted(
        {str(b["system_fingerprint"]) for b in ok if b.get("system_fingerprint")}
        | {
            str(b["reask_system_fingerprint"])
            for b in ok
            if b.get("reask_system_fingerprint")
        }
    )
    # The second pass's own
    # calls, deduped by "second_pass_call_id" the same way the batch pass's own calls are
    # deduped by "call_id" -- a batch's INCLUDE survivors share one call, so summing every
    # row's copy of its tokens would overcount by the number of survivors. Unlike the batch
    # pass, a second-pass call is never later pruned by --retry-failed (that flag only
    # targets predicted=null/padded_decision rows), so there is no "superseded" case to track
    # here. ``called_at`` is not tracked per second-pass call; each row's own ``called_at``
    # (the same batch, effectively the same wall-clock window) stands in for the tariff tier.
    second_pass_by_call: dict[str, dict[str, Any]] = {}
    for row in rows:
        call_id = row.get("second_pass_call_id")
        if call_id and call_id not in second_pass_by_call:
            second_pass_by_call[call_id] = row
    second_pass_calls = list(second_pass_by_call.values())
    second_pass_total_in = sum(int(b.get("second_pass_input_tokens") or 0) for b in second_pass_calls)
    second_pass_total_out = sum(
        int(b.get("second_pass_output_tokens") or 0) for b in second_pass_calls
    )
    second_pass_costs = [
        call_cost(
            b.get("second_pass_model_reported"),
            b.get("called_at"),
            b.get("second_pass_input_tokens"),
            b.get("second_pass_output_tokens"),
            0,
        )
        for b in second_pass_calls
    ]
    second_pass_summary = {
        "calls": len(second_pass_calls),
        "model_reported": sorted(
            {str(b["second_pass_model_reported"]) for b in second_pass_calls
             if b.get("second_pass_model_reported")}
        ),
        "total_input_tokens": second_pass_total_in,
        "total_output_tokens": second_pass_total_out,
        "total_cost": (
            sum(second_pass_costs)
            if second_pass_calls and all(c is not None for c in second_pass_costs)
            else None
        ),
    }

    finished = now_iso()
    sessions = list((existing or {}).get("sessions") or [])
    sessions.append(
        {"started": session_started, "finished": finished, "retried_failed": retried_failed}
    )
    return {
        "dataset": dataset,
        "run": run,
        "dry_run": dry_run,
        "record_order_seed": record_order_seed,
        "retried_failed": retried_failed,
        "started": (existing or {}).get("started") or session_started,
        "finished": finished,
        "sessions": sessions,
        "protocol_file": str(protocol.path.name) if protocol.path else None,
        "protocol_sha": protocol_sha_value,
        "full_text_criteria": list(full_text_criteria or []),
        "label_field": protocol.label_field,
        "model_configured": model_configured,
        "model_reported": models,
        "system_fingerprints": fingerprints,
        "temperature": temperature,
        "prompt_version": sorted(set(prompt_versions)),
        "prompt_version_flag": prompt_version_flag,
        "abstract_cap": abstract_cap,
        "batch_size": batch_size,
        "concurrency": concurrency,
        "limit": limit,
        "n_records": n_records,
        "n_batches": n_batches,
        # Cumulative across every session (the caller carries this
        # forward the same way it carries ``sessions`` forward), so a resumed session's own
        # re-ask sub-batches are added on top of every earlier session's, not in place of them.
        "reask_batches_run": reask_batches_run,
        "n_batches_completed": len({b["batch_index"] for b in ok}),
        # A call group is 2 real API calls, not 1, when
        # the backend's own internal re-ask answered for it (``calls`` on the row, defaulted
        # to 1 for every earlier row) -- summed rather than counted so this reflects every
        # paid call, not one per call_id. ``ok`` already includes the superseded calls folded
        # in above, so a call a later re-ask pass fully pruned from ``rows``
        # is still counted here.
        "n_calls": sum(int(b.get("calls") or 1) for b in ok),
        # The calls/tokens folded into the
        # totals above that no longer have any row in ``rows`` at all, reported separately so
        # the contribution is visible rather than silently merged. 0 / 0 / 0 on a run whose
        # re-ask cascade never fully pruned a call (the ordinary case).
        "superseded_calls": sum(int(b.get("calls") or 1) for b in superseded_by_id.values()),
        "superseded_input_tokens": sum(
            int(b.get("input_tokens") or 0) for b in superseded_by_id.values()
        ),
        "superseded_output_tokens": sum(
            int(b.get("output_tokens") or 0) for b in superseded_by_id.values()
        ),
        # The representative rows actually
        # folded into the totals above (a strict subset of the ``superseded_call_rows``
        # argument -- one already present in ``rows`` is excluded), persisted so the *next*
        # session can seed its own carry-forward from ``existing`` the way it already seeds
        # ``reask_batches_run``, instead of recomputing this session's contribution from
        # scratch and losing every earlier session's.
        "superseded_call_rows": [dict(b) for b in superseded_by_id.values()],
        "failures": len(failed),
        "n_rows": len(rows),
        "n_unscreened_records": sum(1 for r in rows if r.get("predicted") is None),
        "n_padded_include": sum(1 for r in rows if r.get("padded_include")),
        "n_padded_decisions": sum(1 for r in rows if r.get("padded_decision")),
        "price": dict(price),
        "n_missing_abstract": n_missing_abstract,
        "share_missing_abstract": (n_missing_abstract / len(rows)) if rows else None,
        "n_missing_title": n_missing_title,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "cost_basis": "list price, tier by call time, cache-miss assumed (upper bound)",
        "total_cost": total_cost,
        "total_cost_flat": compute_cost(
            total_in, total_out, price.get("input_per_mtok"), price.get("output_per_mtok")
        ),
        "latency_s": latency_stats([b["latency_s"] for b in ok if b.get("latency_s") is not None]),
        # The inclusion-only
        # second pass's own calls/tokens/cost, separate from the totals above (a different
        # model, a different prompt) -- {"calls": 0, ...} on a run with no INCLUDE survivor.
        "second_pass": second_pass_summary,
    }


# --------------------------------------------------------------------------------------
# Screeners
# --------------------------------------------------------------------------------------


class ProductionScreener:
    """Thin wrapper around the unchanged production ``screen_papers`` function.

    ``prompt_version`` (``"v1"``/``"v2"``) and ``abstract_limit`` select
    the shipped prompt/renderer pair and override the numeric abstract cap; both are forwarded
    to ``screen_papers`` unchanged, so the backend module (not this harness) is the single
    source of truth for the byte-identical v1 rendering and the head/marker/tail v2 rendering.
    ``abstract_limit=None`` (the default) omits the keyword entirely, so ``screen_papers``'s own
    default (the frozen ``ABSTRACT_CHAR_LIMIT`` constant) applies -- ``effective_abstract_cap``
    resolves that ``None`` to the constant's actual value so callers (``main_async``) can record
    the number really used in the run meta rather than just "not overridden".
    """

    def __init__(
        self,
        backend_root: Path | None = None,
        *,
        prompt_version: str = "v3",
        abstract_limit: int | None = None,
        second_pass_model: str | None = None,
    ) -> None:
        add_backend_to_path(backend_root)
        export_env_from_dotenv(ENV_KEYS)
        try:
            from app.agents.relevance_screener_agent import (  # noqa: F401
                ScreeningBatchResult,
                ScreeningError,
                screen_papers,
            )
        except ImportError as exc:
            raise SystemExit(
                "backend/ does not expose the required screen_papers interface (it must "
                f"return ScreeningBatchResult with .include/.reasons/.provenance): {exc}"
            ) from exc
        try:
            from app.agents.relevance_screener_agent import ABSTRACT_CHAR_LIMIT
        except ImportError:
            ABSTRACT_CHAR_LIMIT = None  # noqa: N806 -- degrade to "unknown", never crash on this
        # The shipped production devices this harness applies after
        # the batch pass, exactly as they run in app.services.smart_search -- the two
        # deterministic post-model checks and the inclusion-only second pass,
        # so an evaluation run measures the same rule the product ships. Optional
        # feature detection, the same pattern ABSTRACT_CHAR_LIMIT above already uses: an older
        # backend (or a test double that only implements the core
        # interface) still runs the unchanged batch pass, just without these devices, rather
        # than aborting outright the way a missing *core* symbol does above.
        try:
            from app.agents.relevance_screener_agent import (  # noqa: F401
                apply_second_pass_guard,
                apply_table_of_contents_demotion,
                apply_type_demotion,
                build_shown_texts,
                run_second_pass_stage,
            )
        except ImportError:
            apply_second_pass_guard = None  # noqa: N806
            apply_table_of_contents_demotion = None  # noqa: N806
            apply_type_demotion = None  # noqa: N806
            build_shown_texts = None  # noqa: N806
            run_second_pass_stage = None  # noqa: N806
        settings = import_backend_settings()

        self._screen = screen_papers
        # The harness calls the same
        # concurrent stage app.services.smart_search does, not confirm_inclusions directly,
        # so an evaluation run exercises the exact code path production runs (batching,
        # concurrency bound, timeout/failure routing and budget accounting all included).
        self._run_second_pass_stage = run_second_pass_stage
        self._apply_type_demotion = apply_type_demotion
        self._apply_table_of_contents_demotion = apply_table_of_contents_demotion
        self._apply_second_pass_guard = apply_second_pass_guard
        self._build_shown_texts = build_shown_texts
        #: True when this backend ships the v2 devices at all (whether or not a
        #: second_pass_model is configured to actually run the second pass on top of them).
        self.has_v2_devices: bool = build_shown_texts is not None
        self.error_type: type[Exception] = ScreeningError
        self.model_configured: str | None = getattr(settings, "deepseek_model", None)
        self.prompt_version = prompt_version
        self.abstract_limit = abstract_limit
        self.effective_abstract_cap: int | None = (
            abstract_limit if abstract_limit is not None else ABSTRACT_CHAR_LIMIT
        )
        self.second_pass_model: str | None = second_pass_model or getattr(
            settings, "screener_second_pass_model", None
        )
        # run_second_pass_stage's own stage_deadline defaults to
        # None (unbounded) when not passed -- unlike concurrency, it has no settings
        # fallback of its own, so this harness must read
        # settings.screener_second_pass_stage_budget_seconds itself to match production.
        # None (a backend that predates this setting, or a test double) means unbounded,
        # same as passing stage_deadline=None straight through.
        self._stage_budget_seconds: float | None = getattr(
            settings, "screener_second_pass_stage_budget_seconds", None
        )

    def _shown_texts(self, papers: list[dict[str, Any]]) -> list[str]:
        """The title/abstract text exactly as ``screen_papers`` rendered it for this batch
        so the type/table-of-contents devices and the second pass's own verbatim
        check see the same text the batch pass did."""
        version = self.prompt_version if self.prompt_version in ("v2", "v3") else "v2"
        kwargs: dict[str, Any] = {"prompt_version": version}
        if self.effective_abstract_cap is not None:
            kwargs["limit"] = self.effective_abstract_cap
        return self._build_shown_texts(papers, **kwargs)

    async def __call__(self, protocol: Protocol, papers: list[dict[str, Any]]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"prompt_version": self.prompt_version}
        if self.abstract_limit is not None:
            kwargs["abstract_limit"] = self.abstract_limit
        # Forward the protocol's per-criterion stage and absence markings, so a
        # full-text criterion can never ground an EXCLUDE and an absence criterion in a cut
        # abstract is routed rather than trusted.
        if protocol.inclusion_stages:
            kwargs["inclusion_stages"] = protocol.inclusion_stages
        if protocol.exclusion_stages:
            kwargs["exclusion_stages"] = protocol.exclusion_stages
        if protocol.inclusion_absence:
            kwargs["inclusion_absence"] = protocol.inclusion_absence
        if protocol.exclusion_absence:
            kwargs["exclusion_absence"] = protocol.exclusion_absence
        result = await self._screen(
            protocol.research_question,
            papers,
            protocol.inclusion_criteria,
            protocol.exclusion_criteria,
            **kwargs,
        )
        include = list(result.include)
        if len(include) != len(papers):
            raise self.error_type(
                f"screen_papers returned {len(include)} decisions for {len(papers)} papers"
            )
        statuses = list(getattr(result, "statuses", None) or [])
        guard_applied = list(getattr(result, "guard_applied", None) or [])
        guard_reasons = list(getattr(result, "guard_reasons", None) or [])
        # Defensive padding for a legacy/test-double outcome shape whose guard_applied/
        # guard_reasons are shorter than statuses (the real backend's ScreeningBatchResult
        # validator guarantees parallel lengths and never needs this).
        if len(guard_applied) < len(statuses):
            guard_applied += [False] * (len(statuses) - len(guard_applied))
        if len(guard_reasons) < len(statuses):
            guard_reasons += [""] * (len(statuses) - len(guard_reasons))
        second_pass_answers: list[dict[str, Any] | None] = [None] * len(papers)
        # One call per
        # SECOND_PASS_STAGE_BATCH_SIZE-sized chunk of this call's own candidates, not one
        # call for the whole set -- so each record's own call is tracked by its own
        # position, not a single shared dict, whenever more than one chunk ran.
        second_pass_calls: list[dict[str, Any] | None] = [None] * len(papers)
        # The two
        # deterministic post-model devices and the inclusion-only second pass, applied in the
        # same order and on the same INCLUDE-only survivors as app.services.smart_search.
        # v1 asks for no criterion/quote/anchor at all, so none of this applies to it, the
        # same way apply_decision_guard itself is skipped for v1.
        if self.has_v2_devices and self.prompt_version != "v1" and statuses:
            shown_texts = self._shown_texts(papers)
            second_pass_candidates: list[int] = []
            for i, status in enumerate(statuses):
                if status != "INCLUDE":
                    continue
                new_status, reason = self._apply_type_demotion(
                    status, work_type=papers[i].get("type"), is_paratext=papers[i].get("is_paratext"),
                )
                if reason:
                    statuses[i] = new_status
                    guard_reasons[i] = reason
                    guard_applied[i] = True
                    continue
                new_status, reason = self._apply_table_of_contents_demotion(
                    status, shown_texts[i]
                )
                if reason:
                    statuses[i] = new_status
                    guard_reasons[i] = reason
                    guard_applied[i] = True
                    continue
                second_pass_candidates.append(i)

            if second_pass_candidates and self.second_pass_model and self._run_second_pass_stage:
                sp_papers = [papers[i] for i in second_pass_candidates]
                sp_shown = [shown_texts[i] for i in second_pass_candidates]
                # The same concurrent
                # stage app.services.smart_search calls, on this call's own candidates (this
                # harness invokes ProductionScreener once per batch of the size the caller
                # chose, so a batch's own candidates are what one stage call here covers).
                # settings.screener_second_pass_concurrency applies through
                # run_second_pass_stage's own fallback since this call passes neither
                # concurrency nor a fixed value; settings.screener_second_pass_stage_budget_
                # seconds has no such fallback inside that function, so this
                # harness computes its own deadline from it explicitly below -- None
                # (unbounded) only when the backend predates the setting.
                stage_deadline = (
                    time.monotonic() + self._stage_budget_seconds
                    if self._stage_budget_seconds is not None
                    else None
                )
                stage_result = await self._run_second_pass_stage(
                    sp_papers,
                    sp_shown,
                    research_question=protocol.research_question,
                    inclusion_criteria=protocol.inclusion_criteria,
                    exclusion_criteria=protocol.exclusion_criteria,
                    model=self.second_pass_model,
                    stage_deadline=stage_deadline,
                )
                for call in stage_result.calls:
                    call_info = {
                        "call_id": uuid.uuid4().hex,
                        "model_configured": self.second_pass_model,
                        "model_reported": getattr(call.provenance, "model_reported", None),
                        "system_fingerprint": getattr(call.provenance, "system_fingerprint", None),
                        "input_tokens": getattr(call.provenance, "input_tokens", None),
                        "output_tokens": getattr(call.provenance, "output_tokens", None),
                        "prompt_version": getattr(call.provenance, "prompt_version", None),
                    }
                    # call.indices are positions in sp_papers/sp_shown's own order, mapped
                    # back through second_pass_candidates
                    # to this record's own position in papers/statuses/second_pass_calls.
                    for local_idx in call.indices:
                        second_pass_calls[second_pass_candidates[local_idx]] = call_info
                for pos, decision in zip(second_pass_candidates, stage_result.decisions):
                    second_pass_answers[pos] = decision.second_pass
                    if decision.status != "INCLUDE":
                        statuses[pos] = decision.status
                        guard_reasons[pos] = decision.guard_reason
                        guard_applied[pos] = True
            include = [s == "INCLUDE" for s in statuses]
        prov = getattr(result, "provenance", None)
        # A batch whose response left a paper undecided
        # made a second, re-ask call inside screen_papers -- ``reask_provenance`` carries that
        # call's own tokens/model/fingerprint (backend ``ScreeningBatchResult``). Both calls'
        # tokens are summed into the single ``input_tokens``/``output_tokens`` this outcome
        # reports so the row's own total is correct without a second row or call_id; ``calls``
        # (forwarded to the row, then summed in :func:`build_meta`) records that two paid
        # calls, not one, produced this batch, and the re-ask's own model/fingerprint are kept
        # separately so a change between the two calls is still visible in the run meta.
        reask_prov = getattr(result, "reask_provenance", None)
        input_tokens = _sum_optional_tokens(
            getattr(prov, "input_tokens", None),
            getattr(reask_prov, "input_tokens", None) if reask_prov is not None else None,
        )
        output_tokens = _sum_optional_tokens(
            getattr(prov, "output_tokens", None),
            getattr(reask_prov, "output_tokens", None) if reask_prov is not None else None,
        )
        return {
            "include": include,
            "reasons": list(getattr(result, "reasons", None) or []),
            # The batch pass's own statuses/guard_applied/guard_reasons, possibly
            # further demoted above by the type/table-of-contents devices and the second
            # pass -- never the raw ``result`` fields, so a record either of them routed to
            # NEEDS_REVIEW is reported that way everywhere (rows_for_batch, the meta's
            # needs_review-style counts a scoring pass derives from these rows).
            "statuses": statuses,
            "criteria_ids": list(getattr(result, "criteria_ids", None) or []),
            "quotes": list(getattr(result, "quotes", None) or []),
            "guard_applied": guard_applied,
            "guard_reasons": guard_reasons,
            "guard_conversions": getattr(result, "guard_conversions", None),
            # Per record, the
            # second pass's own established/not_established answer ({"population",
            # "outcome", "study_type"}), or None when the second pass never reached this
            # record (not an INCLUDE, demoted by the type/TOC devices first, no
            # second_pass_model configured, or --prompt-version v1).
            "second_pass_answers": second_pass_answers,
            # One entry per paper,
            # parallel to second_pass_answers -- the specific chunk-sized
            # run_second_pass_stage call that produced that record's own answer (None when
            # the second pass never reached it), rather than a single shared dict for the
            # whole batch, since a batch's own candidates can now span more than one call.
            # ``build_meta`` dedupes on each row's own "call_id" so a chunk's tokens/cost
            # are counted once regardless of how many rows share a copy of it.
            "second_pass_calls": second_pass_calls,
            # Per record, the full-text inclusion criterion ids still to
            # be confirmed at full text; non-empty only on an INCLUDE.
            "to_confirm": list(getattr(result, "to_confirm", None) or []),
            # Per record, the raw anchor ledger
            # (``--prompt-version v3`` only; ``[]`` for v1/v2) -- carried through unchanged
            # so an offline pass can re-score any anchor set from these rows alone.
            "anchors": list(getattr(result, "anchors", None) or []),
            # How many of this batch's decisions the
            # model never returned at all (padded, currently to NEEDS_REVIEW); forwarded so
            # rows_for_batch can flag padded_decision and the meta can count n_padded_decisions.
            "padded": getattr(result, "padded", 0),
            "model_configured": getattr(prov, "model_configured", self.model_configured),
            "model_reported": getattr(prov, "model_reported", None),
            "system_fingerprint": getattr(prov, "system_fingerprint", None),
            "provider_response_id": getattr(prov, "provider_response_id", None),
            "temperature": getattr(prov, "temperature", None),
            "prompt_version": getattr(prov, "prompt_version", None),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            # 2 when a re-ask call answered, else 1.
            "calls": 1 if reask_prov is None else 2,
            "reask_model_reported": getattr(reask_prov, "model_reported", None),
            "reask_system_fingerprint": getattr(reask_prov, "system_fingerprint", None),
            # Set only when the backend's one re-ask call itself
            # raised (a transport or model failure), so a still-padded record caused by that
            # is distinguishable from one still padded because the model quietly skipped it
            # again -- the harness's own re-ask passes are useful work for the latter, not
            # the former.
            "reask_error": getattr(result, "reask_error", None),
        }


class DryRunScreener:
    """Keyword stand-in used by ``--dry-run`` (no network, no LLM)."""

    error_type: type[Exception] = RuntimeError
    model_configured = "dry-run"
    #: no real abstract truncation happens here, so there is no "number actually used" to
    #: resolve --abstract-cap's None to.
    effective_abstract_cap: int | None = None

    async def __call__(self, protocol: Protocol, papers: list[dict[str, Any]]) -> dict[str, Any]:
        keywords = {
            w.lower().strip(",.;:()") for w in protocol.research_question.split() if len(w) > 6
        }
        include, reasons = [], []
        for p in papers:
            text = f"{p.get('title') or ''} {p.get('abstract') or ''}".lower()
            hits = sorted(k for k in keywords if k in text)
            include.append(bool(hits))
            reasons.append(
                f"dry-run keyword match: {', '.join(hits[:3])}" if hits else "dry-run: no keyword"
            )
        await asyncio.sleep(0)
        return {
            "include": include,
            "reasons": reasons,
            "model_configured": "dry-run",
            "model_reported": "dry-run",
            "system_fingerprint": None,
            "provider_response_id": None,
            "temperature": 0.0,
            "prompt_version": "dry-run",
            "input_tokens": 0,
            "output_tokens": 0,
        }


async def screen_batch_with_retries(
    screener: Any, protocol: Protocol, batch: Sequence[Mapping[str, Any]], max_attempts: int
) -> dict[str, Any]:
    # "type"/"is_paratext" ride along on the paper dict the same way title/abstract
    # do -- screen_papers itself only ever reads title/abstract, but ProductionScreener's own
    # post-model type demotion (app.agents.relevance_screener_agent.apply_type_demotion) needs
    # them, and it only ever sees ``papers`` here, not the original ``batch`` records.
    papers = [
        {
            "title": r.get("title") or "",
            "abstract": r.get("abstract") or "",
            "type": r.get("type"),
            "is_paratext": r.get("is_paratext"),
        }
        for r in batch
    ]
    last_error = "unknown"
    for attempt in range(1, max_attempts + 1):
        called_at = now_iso()
        outcome = None
        with Stopwatch() as sw:
            try:
                outcome = await screener(protocol, papers)
            except screener.error_type as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # unexpected failure types are recorded, not swallowed
                last_error = f"{type(exc).__name__}: {exc}"
        latency = round(sw.seconds, 3)  # read *after* the block: the stopwatch is stopped here
        if outcome is not None:
            outcome["called_at"] = called_at
            outcome["call_id"] = uuid.uuid4().hex
            outcome["latency_s"] = latency
            outcome["attempts"] = attempt
            return outcome
        if attempt < max_attempts:
            await asyncio.sleep(2.0**attempt)
    return {
        "error": last_error, "attempts": max_attempts, "latency_s": None,
        "call_id": uuid.uuid4().hex,
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True, choices=("A", "B"))
    ap.add_argument("--limit", type=int, default=None, help="screen only the first N records")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-attempts", type=int, default=3, help="1 call + retries per batch")
    ap.add_argument(
        "--record-order-seed",
        type=int,
        default=DEFAULT_RECORD_ORDER_SEED,
        help="seed of the deterministic per-dataset shuffle applied before batching",
    )
    ap.add_argument(
        "--retry-failed",
        action="store_true",
        help="drop rows with predicted=null from the results file and screen them again",
    )
    ap.add_argument("--price-input", type=float, default=None, help="USD per 1M input tokens")
    ap.add_argument("--price-output", type=float, default=None, help="USD per 1M output tokens")
    ap.add_argument("--price-source-url", default=None)
    ap.add_argument("--price-accessed", default=None, help="date the price page was read")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=f"where run/summary files are written and read from (default: {RESULTS_DIR}, "
        "the results of record, matching baselines.py/summarize.py/wos_gate_effect.py's own "
        "default and --prompt-version's own v3 default)",
    )
    ap.add_argument("--protocols-dir", type=Path, default=PROTOCOLS_DIR)
    ap.add_argument(
        "--dry-run", action="store_true", help="no LLM; keyword stand-in; writes to data/_dryrun"
    )
    ap.add_argument(
        "--confirm-cost",
        action="store_true",
        help="required for any run that is not --dry-run: acknowledges the paid API calls "
        "(the expected batch count and list-price upper bound are printed first)",
    )
    ap.add_argument(
        "--allow-nonzero-temperature",
        action="store_true",
        help="do not abort when the provenance reports a temperature other than 0.0",
    )
    ap.add_argument(
        "--prompt-version",
        choices=("v1", "v2", "v3"),
        default="v3",
        help="which screener prompt and renderer to run; v1 is the "
        "frozen original binary prompt, v2 a protocol-eligibility prompt kept byte-exact "
        "for reproducing earlier evaluation runs, v3 the shipped prompt (anchor ledger, "
        "title-only TOPIC exclude; see relevance_screener_agent.py), the default, so an "
        "evaluation run matches production unless told otherwise; --results-dir defaults to "
        "the matching results/v3 directory",
    )
    ap.add_argument(
        "--abstract-cap",
        type=int,
        default=None,
        help="override the numeric abstract limit only, never the truncation style "
        "(default: the frozen ABSTRACT_CHAR_LIMIT constant in the backend agent module)",
    )
    ap.add_argument(
        "--prompt-version-expect",
        default=None,
        help="abort (exit 2) if any call's reported prompt version/sha does not equal this "
        "value (post-freeze re-run guard: a run whose meta disagrees with "
        "the frozen sha is discarded and re-run, not scored)",
    )
    ap.add_argument(
        "--second-pass-model",
        default=None,
        help="model for the inclusion-only second pass (the type/table-of-"
        "contents devices always run at --prompt-version v2/v3; the second pass on top of "
        "them also needs this or the backend's own screener_second_pass_model default). "
        "Ignored at --prompt-version v1.",
    )
    return resolve_path_args(ap.parse_args(argv))


async def main_async(args: argparse.Namespace, screener: Any | None = None) -> int:
    """Run one screening pass. ``screener`` overrides the dry-run/production choice (tests)."""
    resolve_path_args(args)
    protocol = load_protocol(args.protocols_dir / f"{args.dataset}.json")
    records = load_records(
        args.data_dir / f"{args.dataset}.jsonl",
        args.limit,
        dataset=args.dataset,
        seed=args.record_order_seed,
    )
    batches = make_batches(records, args.batch_size)
    results_dir = args.results_dir or (DRYRUN_RESULTS_DIR if args.dry_run else RESULTS_DIR)
    out_path = results_dir / f"{args.dataset}_run{args.run}.jsonl"
    meta_path = results_dir / f"{args.dataset}_run{args.run}.meta.json"

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

    # Cost gate first: nothing is constructed, pruned or written before it passes. With
    # --retry-failed the unscreened rows (predicted None) are pending work too, so the preview
    # counts them without pruning anything yet.
    already = completed_ids(out_path, ROW_ID)
    if getattr(args, "retry_failed", False):
        already -= {r[ROW_ID] for r in iter_jsonl(out_path) if _is_retryable_row(r)}
    pending_preview = [
        [r for r in b if r[ROW_ID] not in already]
        for b in batches
        if any(r[ROW_ID] not in already for r in b)
    ]
    model_hint = getattr(screener, "model_configured", None) or configured_model_hint()
    require_cost_confirmation(args, pending_preview, model_hint)

    if screener is None:
        screener = (
            DryRunScreener() if args.dry_run
            else ProductionScreener(
                prompt_version=args.prompt_version,
                abstract_limit=args.abstract_cap,
                second_pass_model=getattr(args, "second_pass_model", None),
            )
        )
    retried_failed = 0
    # Captured *before* the prune below, not
    # derived from it afterwards -- a row this prune removes may be the only surviving row of a
    # call made in an *earlier* session (this session has not sent anything yet), and once the
    # prune runs there is nothing left on disk to reconstruct that call's tokens/model/called_at
    # from. Empty when a call's row group only loses some of its rows here: the surviving row(s)
    # already carry that call's own tokens, so build_meta's ordinary per-call_id grouping prices
    # it correctly without help.
    superseded_by_retry: list[dict[str, Any]] = []
    if getattr(args, "retry_failed", False):
        superseded_by_retry = [
            rep
            for rep in _fully_pruned_calls(read_jsonl(out_path), _is_retryable_row)
            if not _is_unbilled_failure_row(rep)
        ]
        retried_failed = prune_failed_rows(out_path, _is_retryable_row)
        print(
            f"--retry-failed: removed {retried_failed} unscreened/still-padded rows "
            "for re-screening"
        )
    done = completed_ids(out_path, ROW_ID)
    # Record-level resume: each batch keeps its index but only its not-yet-done records are
    # sent, so a partially written batch is completed with a smaller call, never re-sent.
    pending = [
        (i, [r for r in b if r[ROW_ID] not in done])
        for i, b in enumerate(batches)
        if any(r[ROW_ID] not in done for r in b)
    ]
    n_pending_records = sum(len(b) for _, b in pending)
    print(
        f"{args.dataset} run {args.run}: {len(records)} records, {len(batches)} batches, "
        f"{len(pending)} pending batches / {n_pending_records} pending records "
        f"(resume: {len(done)} records done); model_configured={screener.model_configured}"
    )
    session_started = now_iso()
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    progress = {"done": 0, "failed": 0}
    prompt_versions: list[str] = []
    temperatures: list[float] = []
    # Seeded from the existing meta (0 on a fresh run), not always
    # 0, so a resumed session's own re-ask sub-batches are counted on top of every earlier
    # session's, the same way ``sessions`` itself is carried forward -- otherwise a second
    # session's ``n_batches`` would silently drop the first session's re-ask sub-batches.
    reask_batches_run = int(existing_meta.get("reask_batches_run") or 0)
    # One representative row per real
    # call this session makes, recorded the moment run_one_batch returns it. A later re-ask
    # pass can prune every row of an earlier call from ``out_path`` one record at a time (see
    # reask_pass below); without this record, that call would silently disappear from the
    # meta's totals the instant its last row is gone, even though it was a real, paid call.
    call_records: dict[Any, dict[str, Any]] = {}

    async def run_one_batch(
        batch_index: int, batch: list[dict[str, Any]], *, pass_number: int = 1
    ) -> list[dict[str, Any]]:
        """Screen one batch, append its rows and apply the temperature/prompt-version-expect
        guards. Shared by the initial pass and every re-ask pass, so a
        re-ask call is checked exactly like an ordinary one and reported the same way."""
        outcome = await screen_batch_with_retries(screener, protocol, batch, args.max_attempts)
        rows = rows_for_batch(
            batch, batch_index, outcome, protocol.label_field, pass_number=pass_number
        )
        if rows and rows[0].get("call_id") is not None:
            call_records[rows[0]["call_id"]] = dict(rows[0])
        append_jsonl(out_path, rows)
        if outcome.get("error"):
            progress["failed"] += 1
            attempts, error = outcome["attempts"], outcome["error"]
            print(f"  batch {batch_index}: FAILED after {attempts} attempts: {error}")
        else:
            if outcome.get("prompt_version"):
                prompt_versions.append(str(outcome["prompt_version"]))
            if outcome.get("temperature") is not None:
                temperatures.append(float(outcome["temperature"]))
                if temperatures[-1] != 0.0 and not args.allow_nonzero_temperature:
                    print(
                        f"ABORT: provenance reports temperature={temperatures[-1]} "
                        "(expected 0.0); pass --allow-nonzero-temperature to override"
                    )
                    raise SystemExit(2)
            expect = getattr(args, "prompt_version_expect", None)
            reported_prompt = outcome.get("prompt_version")
            if expect and reported_prompt is not None and reported_prompt != expect:
                print(
                    f"ABORT: provenance reports prompt_version={reported_prompt!r} "
                    f"(expected {expect!r}); this run is discarded, not scored -- re-run "
                    "after confirming the frozen prompt sha"
                )
                raise SystemExit(2)
        return rows

    session_rows: list[dict[str, Any]] = []

    async def worker(batch_index: int, batch: list[dict[str, Any]]) -> None:
        async with semaphore:
            rows = await run_one_batch(batch_index, batch)
        session_rows.extend(rows)
        progress["done"] += 1
        if progress["done"] % 10 == 0 or progress["done"] == len(pending):
            print(f"  {progress['done']}/{len(pending)} batches ({progress['failed']} failed)")

    await asyncio.gather(*(worker(i, b) for i, b in pending))

    # A record this session's own pass-1 response padded is re-asked, not
    # left with a silent "no decision returned" flag -- once in a batch of 3, then, if it is
    # still undecided, alone. Only a record still undecided after that single-record pass
    # keeps the padded marker (rows_for_batch computes padded_decision fresh from each pass's
    # own outcome, so a record a pass resolves simply stops carrying it). Only records padded
    # *this session* are chased here: a stale padded row from an earlier session needs
    # --retry-failed to be picked up again, the same rule a genuine call failure already
    # follows (see _is_retryable_row).
    #
    # Seeded from the batch index already on disk, not just this
    # session's own fresh batch map -- a resumed session whose earlier session already ran
    # re-ask sub-batches (batch_index >= len(batches)) must not hand out an index one of
    # those is already using.
    next_batch_index = max(
        [len(batches)]
        + [
            int(r["batch_index"]) + 1
            for r in iter_jsonl(out_path)
            if r.get("batch_index") is not None
        ]
    )

    async def reask_pass(
        record_ids: set[Any], pass_number: int, reask_batch_size: int
    ) -> set[Any]:
        nonlocal next_batch_index, reask_batches_run
        # Iterate ``records`` (the run's own deterministic shuffle
        # order), not ``record_ids`` (a set) -- a set's iteration order is an implementation
        # detail, unrelated to ``record_order_seed``, that only happens to be stable today
        # because every shipped dataset keys on an integer record_id.
        sub_records = [r for r in records if r[ROW_ID] in record_ids]
        if not sub_records:
            return set()
        still_padded: set[Any] = set()
        for sub_batch in make_batches(sub_records, reask_batch_size):
            # Prune only this sub-batch's own rows immediately
            # before sending it, not the whole pass's rows up front -- an abort between
            # sub-batches (the temperature/prompt-version-expect guards, a crash) must not
            # discard a still-pending sub-batch's already-written rows along with the ones
            # actually about to be re-sent.
            sub_ids = {r[ROW_ID] for r in sub_batch}
            prune_failed_rows(out_path, lambda r: r[ROW_ID] in sub_ids)
            rows = await run_one_batch(next_batch_index, sub_batch, pass_number=pass_number)
            next_batch_index += 1
            reask_batches_run += 1
            still_padded |= {r[ROW_ID] for r in rows if r.get("padded_decision")}
        return still_padded

    padded_after_pass1 = {r[ROW_ID] for r in session_rows if r.get("padded_decision")}
    if padded_after_pass1:
        print(f"  re-asking {len(padded_after_pass1)} padded record(s) in batches of 3")
        padded_after_pass2 = await reask_pass(padded_after_pass1, 2, 3)
        if padded_after_pass2:
            print(
                f"  re-asking {len(padded_after_pass2)} still-padded record(s) one at a time"
            )
            still_padded = await reask_pass(padded_after_pass2, 3, 1)
            if still_padded:
                print(
                    f"  {len(still_padded)} record(s) remain undecided after the "
                    "single-record pass"
                )

    rows = read_jsonl(out_path)
    # A call recorded above whose call_id
    # is no longer present anywhere in ``rows`` was fully pruned by a later re-ask pass -- one
    # of its sub-batches superseded every one of that call's records, one row at a time,
    # before this session ended.
    on_disk_call_ids = {r.get("call_id") for r in rows if r.get("call_id")}
    superseded_this_session = [
        rep for call_id, rep in call_records.items() if call_id not in on_disk_call_ids
    ]
    existing = read_json(meta_path)
    # build_meta is handed every call this
    # process knows was superseded and no longer has any row of its own -- this session's own
    # re-ask cascade (above), whatever --retry-failed's prune just removed (captured before it
    # ran), and every earlier session's own contribution, carried forward from the previous
    # meta's own superseded_call_rows the same way reask_batches_run is seeded from it a few
    # lines up. Without this seeding, a session that sends no calls at all (everything already
    # resolved) would recompute the meta from what is on disk *right now* and silently drop
    # every call an earlier session's cascade had already superseded.
    superseded_call_rows = superseded_by_retry + superseded_this_session
    seen_superseded_ids = {r.get("call_id") for r in superseded_call_rows if r.get("call_id")}
    for row in (existing or {}).get("superseded_call_rows") or []:
        call_id = row.get("call_id")
        if call_id is not None and call_id not in seen_superseded_ids:
            superseded_call_rows.append(row)
            seen_superseded_ids.add(call_id)
    if existing:
        prompt_versions.extend(existing.get("prompt_version") or [])
        if existing.get("temperature") is not None and not temperatures:
            temperatures.append(float(existing["temperature"]))
    price = price_record(
        screener.model_configured or "deepseek-chat",
        args.price_input,
        args.price_output,
        args.price_source_url,
        args.price_accessed,
    )
    meta = build_meta(
        rows,
        existing=existing,
        dataset=args.dataset,
        run=args.run,
        protocol=protocol,
        n_records=len(records),
        n_batches=len(batches) + reask_batches_run,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        model_configured=screener.model_configured,
        temperature=temperatures[0] if temperatures else None,
        prompt_versions=prompt_versions,
        price=price,
        dry_run=args.dry_run,
        session_started=session_started,
        limit=args.limit,
        record_order_seed=args.record_order_seed,
        retried_failed=retried_failed,
        prompt_version_flag=args.prompt_version,
        abstract_cap=getattr(screener, "effective_abstract_cap", args.abstract_cap),
        protocol_sha_value=protocol_sha(protocol.path) if protocol.path else None,
        full_text_criteria=full_text_criteria_ids(protocol),
        reask_batches_run=reask_batches_run,
        superseded_call_rows=superseded_call_rows,
    )
    write_json(meta_path, meta)
    print(
        f"wrote {out_path} ({meta['n_rows']} rows, {meta['failures']} failed batches, "
        f"{meta['n_unscreened_records']} unscreened) and {meta_path}"
    )
    if meta["total_cost"] is None:
        print(
            "  cost not computed: model not in the DeepSeek price table "
            "(pass --price-input/--price-output for a flat estimate in total_cost_flat)"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
