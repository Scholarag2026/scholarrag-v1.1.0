"""Policy replay of every cached v3 and v4 verification answer through the shipped policy at
HEAD: names every row whose guarded first pass
licenses the repair turn (``_SECOND_PASS_TRIGGER_REASONS``), then re-runs exactly those rows'
repair turn live so it actually executes, and records what it returned.

Phase 1, detection (free, no network). Every model-answered row (not ``deterministic``, not
``error``) in ``results/v3/{hss,real}_run{A,B}.jsonl`` and ``results/v4/...`` is replayed
through ``verify_claim_with_policy``, reusing its own recorded ``model_status``,
``evidence_quote``/``evidence_quotes``, ``assertions``, ``explanation``, ``suggested_revision``
and ``unstated_details`` as the first (and only free) ``ModelPassAnswer`` -- the same,
network-free technique ``replay_real_test_18.py`` used for one row, generalised to every row.
If the policy's own trigger check licenses a second pass, the second ``call_model`` call
raises a sentinel (``_WouldTriggerError``); ``verify_claim_with_policy`` catches it unchanged (its
own, pre-existing "second call failed" branch), keeps the first pass's result and appends a
``second_pass_error`` entry to ``passes`` -- so ``len(policy_result.passes) > 1`` after the
call means the trigger fired, without any network call and without this script deciding the
trigger condition itself.

Phase 2, live repair replay (gated behind ``--confirm-cost``). The shipped repair turn
continues the *same conversation* (``message_history`` from the first call's own
``AgentRunResult.all_messages()``); a cached row has no such object, so reproducing the exact
historical trigger under a real, live repair call needs one first: for each row phase 1
names, this script makes one live "skeleton" call for that item's own claim/chunks (the same
prompt production would send), takes the real ``all_messages()`` it gets back, and replaces
only the ``ToolCallPart.args`` JSON of the assistant's turn with the cached historical
answer's own status/explanation/quotes/assertions. This is verified live, not guessed:
overriding this field on a genuine skeleton and asking the model to "restate your prior
status" echoes the overridden value back, confirming the model reads the edited history as
its own answer (checked once, interactively, before this script was written; not re-checked
on every run). Every other field of every message -- ids, tool name, prompt text -- is
untouched, so the conversation's shape is exactly what a real first pass produces; only its
semantic content is pinned to the historical trigger case. The repair prompt
(``format_quote_repair_prompt``, the same helper ``ProductionVerifier`` calls) is then sent as
a genuine live call continuing that conversation. This costs one extra live call per
triggered row (the skeleton) beyond the repair call itself.

Also reports every row where the ordinary, already-executed v5 live run's own fresh model
call independently triggered the repair turn (no replay, no skeleton needed), cross-referenced
from the already-written ``results/v5/*.jsonl``.

Phase 2 builds the repair prompt's failed segments the same way ``verify_claim_with_policy``
does: it relocates the cached, un-relocated quotes (``attribution_guard_fires_before_relocation``
then ``relocate_evidence_quotes``, Part A) before computing ``_non_verbatim_quote_segments``
(Part B), not the other way round. This is a no-op for
``real-test-18``, the one row this script has actually replayed live: relocating its cached
quotes returns an empty relocation list, the same unchanged quote list, and the same single
failing segment before and after.

Usage (system python, from ``evaluation/claims/``)::

    python replay_repair_turn_v5.py --confirm-cost

Writes ``results/v5/repair_turn_policy_replay.json``: the phase-1 census (how many candidate
rows were checked, how many license the repair turn, from which source files), each triggered
row's cached first pass, the free replay's own guarded result, the phase-2 live repair
replay's own result (skeleton call, repair call, final policy result), what the ordinary v5
live run separately recorded for that same ``item_id`` in both run A and run B, and every row
where the v5 live run itself, independent of this replay, recorded more than one pass.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from build_hss_set import load_cache  # noqa: E402
from common import (  # noqa: E402
    add_backend_to_path,
    call_cost,
    export_env_from_dotenv,
    now_iso,
    price_record,
    read_jsonl,
    write_json,
)
from run_hss import ITEM_KEYS  # noqa: E402
from verify_common import ENV_KEYS, EVAL_PROJECT_ID, usage_extras  # noqa: E402

OUT_PATH = HERE / "results" / "v5" / "repair_turn_policy_replay.json"

SOURCES: list[dict[str, Any]] = [
    {
        "claims_path": HERE / "hss_test_v3_claims.jsonl",
        "cache_dir": HERE / "data" / "hss_test_v3_fulltext",
        "runs": ["hss_runA", "hss_runB"],
    },
    {
        "claims_path": HERE / "real_claims_test.jsonl",
        "cache_dir": HERE / "data" / "real_claims_fulltext",
        "runs": ["real_runA", "real_runB"],
    },
]
SOURCE_VERSIONS = ["v3", "v4"]


class _WouldTriggerError(Exception):
    """Detection-phase sentinel: raised instead of a real second call so
    ``verify_claim_with_policy``'s own (unchanged) exception handling around the optional
    second call tells us, for free, whether its trigger check licensed one."""


def load_items() -> dict[str, dict[str, Any]]:
    """Every hss and real item, keyed by ``item_id``, with its chunks/title/authors attached
    (mirrors ``run_hss.load_items``, called directly so this script needs no subprocess)."""
    items: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        claims = read_jsonl(source["claims_path"])
        cache = load_cache(source["cache_dir"])
        for row in claims:
            chunk_doi = row.get("chunk_doi")
            if chunk_doi:
                paper = cache.get(chunk_doi)
                if paper is None:
                    raise SystemExit(f"{row['item_id']}: {chunk_doi} missing from cache")
                chunks = [str(c["text"]) for c in paper["chunks"]]
            else:
                chunks = []
            item = {k: row.get(k) for k in ITEM_KEYS}
            item["title"] = row.get("title") or ""
            item["authors"] = row.get("authors")
            item["chunks"] = chunks
            items[item["item_id"]] = item
    return items


def cached_first_pass_kwargs(cached: dict[str, Any], model_pass_answer: Any) -> Any:
    """Build a ``ModelPassAnswer`` from a cached result row (same fields
    ``replay_real_test_18.cached_first_pass_kwargs`` uses, generalised to any row)."""
    return model_pass_answer(
        status=cached["model_status"],
        evidence_quote=cached["evidence_quote"],
        evidence_quotes=list(cached.get("evidence_quotes") or []),
        assertions=[dict(a) for a in (cached.get("assertions") or [])],
        explanation=cached.get("explanation"),
        suggested_revision=cached.get("suggested_revision"),
        unstated_details=list(cached.get("unstated_details") or []),
        input_tokens=cached.get("input_tokens"),
        output_tokens=cached.get("output_tokens"),
        system_fingerprint=cached.get("system_fingerprint"),
        model_reported=cached.get("model_reported"),
    )


def candidate_rows() -> list[dict[str, Any]]:
    """Every model-answered row in every ``results/{v3,v4}/*.jsonl`` file, tagged with its
    source version and run name."""
    out: list[dict[str, Any]] = []
    for version in SOURCE_VERSIONS:
        for source in SOURCES:
            for run_name in source["runs"]:
                path = HERE / "results" / version / f"{run_name}.jsonl"
                for row in read_jsonl(path):
                    if row.get("deterministic") or row.get("error"):
                        continue
                    out.append({"source_version": version, "run_name": run_name, "row": row})
    return out


def v5_rows_for(item_id: str) -> dict[str, dict[str, Any] | None]:
    """The v5 run's own recorded row for *item_id* in each of the four v5 result files."""
    out: dict[str, dict[str, Any] | None] = {}
    for source in SOURCES:
        for run_name in source["runs"]:
            path = HERE / "results" / "v5" / f"{run_name}.jsonl"
            row = next((r for r in read_jsonl(path) if r.get("item_id") == item_id), None)
            out[run_name] = row
    return out


def find_v5_multi_pass() -> list[dict[str, Any]]:
    """Every row in the v5 run's own four files whose ``passes`` has more than one entry --
    a genuinely occurring, live repair-turn firing during the v5 run itself, independent of
    this script's own cached-answer replay."""
    found: list[dict[str, Any]] = []
    for source in SOURCES:
        for run_name in source["runs"]:
            path = HERE / "results" / "v5" / f"{run_name}.jsonl"
            for row in read_jsonl(path):
                if len(row.get("passes") or []) > 1:
                    found.append({"run_name": run_name, "row": row})
    return found


async def _run(confirm_cost: bool) -> int:
    add_backend_to_path()
    export_env_from_dotenv(ENV_KEYS)
    from app.agents.analysis_agent import AnalysisDependencies
    from app.agents.claim_verification_agent import (
        CLAIM_VERIFICATION_PROMPT_VERSION,
        QUOTE_REPAIR_PROMPT_VERSION,
        VERIFICATION_PROMPT,
        format_quote_repair_prompt,
        format_verification_prompt,
        get_claim_verification_agent,
    )
    from app.agents.model_config import DETERMINISTIC_LONG_MODEL_SETTINGS
    from app.config import settings
    from app.schemas.provenance import prompt_version, provenance_from_run
    from app.services.fulltext import (
        GUARD_DIGEST,
        QUOTE_RELOCATION_VERSION,
        VERIFICATION_POLICY_VERSION,
        ModelPassAnswer,
        normalise_verification_text,
        verify_claim_with_policy,
    )

    temperature = DETERMINISTIC_LONG_MODEL_SETTINGS.get("temperature")
    if temperature != 0.0:
        raise SystemExit(
            f"DETERMINISTIC_LONG_MODEL_SETTINGS temperature is {temperature!r}, not 0.0"
        )
    if prompt_version(VERIFICATION_PROMPT) != CLAIM_VERIFICATION_PROMPT_VERSION:
        raise SystemExit("CLAIM_VERIFICATION_PROMPT_VERSION does not match VERIFICATION_PROMPT")

    model_configured = str(getattr(settings, "deepseek_model", "deepseek-chat"))
    items = load_items()
    agent = get_claim_verification_agent()
    deps = AnalysisDependencies(
        project_id=EVAL_PROJECT_ID, project_description=None, target_journal=None,
        citation_style="APA",
    )

    # ---- Phase 1: detection, no network ----
    candidates = candidate_rows()
    triggered: list[dict[str, Any]] = []
    for cand in candidates:
        row = cand["row"]
        item = items[row["item_id"]]

        async def call_model(repair_request: Any = None, row=row) -> Any:
            if repair_request is None:
                return cached_first_pass_kwargs(row, ModelPassAnswer)
            raise _WouldTriggerError()

        policy_result = await verify_claim_with_policy(
            item["claim"], item["chunks"], [{"text": c} for c in item["chunks"]], call_model
        )
        if len(policy_result.passes) > 1:
            triggered.append(
                {
                    **cand,
                    "replayed_first_pass_guarded_status": policy_result.status,
                    "replayed_first_pass_machine_reasons": policy_result.machine_reasons,
                }
            )
    print(
        f"phase 1 (free replay): {len(candidates)} cached model-answered rows checked, "
        f"{len(triggered)} license the repair turn under the current policy: "
        + ", ".join(
            f"{c['source_version']}/{c['run_name']}/{c['row']['item_id']}" for c in triggered
        )
    )
    if triggered and not confirm_cost:
        raise SystemExit(
            f"{len(triggered)} row(s) license the repair turn; re-run with --confirm-cost to "
            "spend the live API calls phase 2 needs (one skeleton call plus one repair call "
            "per row), or omit it to see phase 1's free census only"
        )

    # ---- Phase 2: live repair replay of exactly the triggered rows ----
    entries: list[dict[str, Any]] = []
    for cand in triggered:
        row = cand["row"]
        item_id = row["item_id"]
        item = items[item_id]
        prompt = format_verification_prompt(
            item["claim"], item["chunks"], item["title"], item["authors"]
        )
        live_calls: list[dict[str, Any]] = []

        async def _live_call(result: Any) -> dict[str, Any]:
            out = result.output
            out_dict = out.model_dump() if hasattr(out, "model_dump") else dict(vars(out))
            record = provenance_from_run(
                "claim_verification", result, model_configured=model_configured,
                temperature=temperature, prompt=VERIFICATION_PROMPT,
            )
            usage = usage_extras(result)
            call_record = {
                "called_at": now_iso(),
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "cache_read_tokens": usage.get("cache_read_tokens"),
                "model_reported": record.model_reported,
                "system_fingerprint": record.system_fingerprint,
                "provider_response_id": record.provider_response_id,
            }
            live_calls.append(call_record)
            return out_dict

        # Skeleton call: a genuine first pass for this exact item, made only to obtain a
        # real, correctly shaped message history to continue from.
        skeleton_result = await agent.run(prompt, deps=deps)
        await _live_call(skeleton_result)
        msgs = list(skeleton_result.all_messages())
        resp_idx = next(i for i, m in enumerate(msgs) if type(m).__name__ == "ModelResponse")
        tool_call_part = msgs[resp_idx].parts[0]
        overridden_args = dict(tool_call_part.args_as_dict())
        overridden_args.update(
            status=row.get("model_status"),
            explanation=row.get("explanation"),
            suggested_revision=row.get("suggested_revision"),
            unstated_details=list(row.get("unstated_details") or []),
            evidence_quote=row.get("evidence_quote"),
            evidence_quotes=list(row.get("evidence_quotes") or []),
            assertions=[dict(a) for a in (row.get("assertions") or [])],
        )
        new_part = dataclasses.replace(tool_call_part, args=json.dumps(overridden_args))
        new_response = dataclasses.replace(msgs[resp_idx], parts=[new_part])
        history = list(msgs)
        history[resp_idx] = new_response

        # Mirror `verify_claim_with_policy`'s own order exactly: it relocates the quotes
        # (Part A) before computing the segments named to the repair turn (Part B), not the
        # other way round. Relocating the cached, un-relocated quotes here first, then
        # computing the failed segments from the relocated result, keeps this replay faithful
        # to production on any row where relocation moves a quote (a no-op for the one row
        # this script has actually replayed, `real-test-18`, whose relocation list is empty).
        from app.services.fulltext import (
            _non_verbatim_quote_segments,
            attribution_guard_fires_before_relocation,
            relocate_evidence_quotes,
        )

        pre_repair_quote = overridden_args.get("evidence_quote")
        pre_repair_quotes = list(overridden_args.get("evidence_quotes") or [])
        pre_repair_assertions = [dict(a) for a in (overridden_args.get("assertions") or [])]
        if not attribution_guard_fires_before_relocation(
            item["claim"], item["chunks"], pre_repair_quote
        ):
            pre_repair_quote, pre_repair_quotes, pre_repair_assertions, _pre_reloc = (
                relocate_evidence_quotes(
                    pre_repair_quote, pre_repair_quotes, pre_repair_assertions, item["chunks"]
                )
            )
        failed_segments_source = pre_repair_quotes or (
            [pre_repair_quote] if pre_repair_quote else []
        )
        failed_segments = _non_verbatim_quote_segments(failed_segments_source, item["chunks"])
        repair_prompt_text = format_quote_repair_prompt(failed_segments)

        repair_result = await agent.run(repair_prompt_text, deps=deps, message_history=history)
        repair_out = await _live_call(repair_result)
        repair_explanation = normalise_verification_text(
            repair_out.get("explanation"), strip_wrapping_quotes=True
        )
        repair_quotes = [
            normalise_verification_text(q) for q in (repair_out.get("evidence_quotes") or [])
        ]
        repair_quote = (
            repair_quotes[0] if repair_quotes
            else normalise_verification_text(repair_out.get("evidence_quote"))
        )
        # Guard the repaired answer exactly as production would (relocate, then the frozen
        # guards) -- this script reports the guarded final status, not the model's raw one.
        from app.services.fulltext import apply_verification_guards

        repair_assertions = [
            {
                **a,
                "quote": normalise_verification_text(a.get("quote")),
                "quotes": [normalise_verification_text(q) for q in (a.get("quotes") or [])],
            }
            for a in (repair_out.get("assertions") or [])
        ]
        no_attribution_guard = attribution_guard_fires_before_relocation(
            item["claim"], item["chunks"], repair_quote
        )
        if not no_attribution_guard:
            repair_quote, repair_quotes, repair_assertions, _reloc = relocate_evidence_quotes(
                repair_quote, repair_quotes, repair_assertions, item["chunks"]
            )
        final_status, final_reasons, final_diag = apply_verification_guards(
            repair_out.get("status"), claim_text=item["claim"], evidence_quote=repair_quote,
            evidence_quotes=repair_quotes, assertions=repair_assertions,
            chunk_texts=item["chunks"], chunks=[{"text": c} for c in item["chunks"]],
        )

        v5_rows = v5_rows_for(item_id)
        live_outcomes = {
            run_name: {
                "predicted_status": v5_row.get("predicted_status"),
                "model_status": v5_row.get("model_status"),
                "machine_reasons": v5_row.get("machine_reasons"),
                "n_passes": len(v5_row.get("passes") or []),
                "repair_fired_live_in_ordinary_v5_run": len(v5_row.get("passes") or []) > 1,
            }
            for run_name, v5_row in v5_rows.items() if v5_row is not None
        }

        entries.append(
            {
                "source_version": cand["source_version"],
                "run_name": cand["run_name"],
                "item_id": item_id,
                "claim": item["claim"],
                "before": {
                    "predicted_status": row.get("predicted_status"),
                    "model_status": row.get("model_status"),
                    "machine_reasons": row.get("machine_reasons"),
                },
                "free_replay_under_current_policy": {
                    "status": cand["replayed_first_pass_guarded_status"],
                    "machine_reasons": cand["replayed_first_pass_machine_reasons"],
                },
                "live_repair_replay": {
                    "failed_segments_named_to_model": failed_segments,
                    "repair_prompt_version": QUOTE_REPAIR_PROMPT_VERSION,
                    "skeleton_call": live_calls[0],
                    "repair_call": live_calls[1],
                    "repair_model_status": repair_out.get("status"),
                    "repair_explanation": repair_explanation,
                    "repair_evidence_quotes": repair_quotes,
                    "after": {
                        "status": final_status,
                        "machine_reasons": final_reasons,
                        "diagnostics": final_diag,
                    },
                },
                "after_live_in_ordinary_v5_run": live_outcomes,
            }
        )

    total_cost = None
    if any(entries):
        price = price_record(model_configured, None, None)
        all_calls = [
            c
            for e in entries
            for c in (
                e["live_repair_replay"]["skeleton_call"],
                e["live_repair_replay"]["repair_call"],
            )
        ]
        total_cost = sum(
            call_cost(
                c["model_reported"] or model_configured, c["called_at"],
                c["input_tokens"] or 0, c["output_tokens"] or 0, c["cache_read_tokens"] or 0,
            ) or 0.0
            for c in all_calls
        )
    else:
        price = price_record(model_configured, None, None)

    v5_multi_pass = find_v5_multi_pass()
    print(
        f"the ordinary v5 live run independently recorded {len(v5_multi_pass)} row(s) where "
        "the repair turn fired for real: "
        + ", ".join(f"{m['run_name']}/{m['row']['item_id']}" for m in v5_multi_pass)
    )

    output = {
        "purpose": (
            "v5 task authorisation: policy replay of every cached v3 and v4 verification "
            "answer through the shipped policy at HEAD, naming the rows that license the "
            "repair turn, re-running each one's repair turn live (skeleton call plus repair "
            "call), and cross-referencing the ordinary, already-executed v5 live run's own "
            "outcome for the same item_id. Also names every row where the ordinary v5 run's "
            "own fresh model call independently triggered the repair turn."
        ),
        "phase1_candidates_checked": len(candidates),
        "phase1_triggered": [
            f"{c['source_version']}/{c['run_name']}/{c['row']['item_id']}" for c in triggered
        ],
        "entries": entries,
        "v5_live_repair_firings": [
            {
                "run_name": m["run_name"],
                "item_id": m["row"]["item_id"],
                "n_passes": len(m["row"].get("passes") or []),
                "first_pass": m["row"]["passes"][0],
                "second_pass": m["row"]["passes"][1],
                "final_predicted_status": m["row"].get("predicted_status"),
                "final_machine_reasons": m["row"].get("machine_reasons"),
            }
            for m in v5_multi_pass
        ],
        "meta": {
            "guard_digest": GUARD_DIGEST,
            "quote_relocation_version": QUOTE_RELOCATION_VERSION,
            "verification_policy_version": VERIFICATION_POLICY_VERSION,
            "repair_prompt_version": QUOTE_REPAIR_PROMPT_VERSION,
            "prompt_version": CLAIM_VERIFICATION_PROMPT_VERSION,
            "model_configured": model_configured,
            "temperature": temperature,
            "price": price,
            "total_cost": total_cost,
        },
    }
    write_json(OUT_PATH, output)
    print(
        f"wrote {OUT_PATH}: {len(entries)} triggered row(s) replayed live, "
        f"total_cost={total_cost!r}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--confirm-cost", action="store_true",
        help="required if phase 1 finds any row that licenses a repair turn: acknowledges "
        "the live API calls (one skeleton call plus one repair call per row) phase 2 makes",
    )
    args = ap.parse_args(argv)
    return asyncio.run(_run(args.confirm_cost))


if __name__ == "__main__":
    raise SystemExit(main())
