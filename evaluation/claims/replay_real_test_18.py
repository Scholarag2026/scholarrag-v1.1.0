"""Live replay of ``real-test-18`` (real run A) through the shipped verification policy.

The deterministic replay that ``FREEZE_DIGESTS_v4.md``'s property 2 rested on fed the 156 cached
v3 answers through ``relocate_evidence_quotes`` and ``apply_verification_guards`` directly,
never through ``verify_claim_with_policy`` (Part A + the frozen guards + Part B, the bounded
second pass). That replay found that ``real-test-18``'s cached first pass -- model status
``verified``, capped by the guards to ``needs_nuance`` with
``machine_reasons == ["quote_not_verbatim"]`` -- is exactly ``_SECOND_PASS_TRIGGER_REASONS``,
so it is the one row in the whole corpus that licenses Part B. What the all-cached replay
could not show is what a live second call actually returns, because its ``call_model``
closure could only ever hand back the one recorded answer.

This script closes that gap for the one item, at the smallest possible cost: it reuses
``results/v3/real_runA.jsonl``'s own recorded first-pass answer for this item (no network
call -- the same ``ModelPassAnswer`` the guard-only replay already used) and runs it through
the real, unchanged ``app.services.fulltext.verify_claim_with_policy``. Every one of that
function's own internal calls (``attribution_guard_fires_before_relocation``,
``relocate_evidence_quotes``, ``apply_verification_guards``) is the live HEAD code, and the
frozen policy's own trigger check decides, exactly as it would in production or in
``ProductionVerifier``, whether a second pass is licensed. Because the first pass reproduces
the known trigger shape, the policy fires its second pass, and this script's ``call_model``
answers that one second call with a real, paid request to the frozen prompt (temperature
0.0, same claim, same chunks, same title/authors) -- the one live model call this task's
authorisation budgeted for. If the policy did not license a second pass (it does, for this
item, deterministically, since the guards are unchanged from the check's own reproduction),
no live call would be made at all.

Usage (system python, from ``evaluation/claims/``, spends real API budget)::

    python replay_real_test_18.py --confirm-cost

Writes ``results/v4/real-test-18_policy_replay.json``: the cached first pass, the live
second pass (tokens, fingerprint, model-reported string, quotes, guarded status), the final
policy result (status, machine_reasons, diagnostics, both ``passes`` entries), and a small
meta block (guard/prompt/policy digests, price, cost of the one live call). Interpreter: the
system python (imports ``app.*`` from ``backend/``), same requirement as ``run_hss.py``.
"""

from __future__ import annotations

import argparse
import asyncio
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
from verify_common import (  # noqa: E402
    ENV_KEYS,
    EVAL_PROJECT_ID,
    usage_extras,
)

ITEM_ID = "real-test-18"
CLAIMS_PATH = HERE / "real_claims_test.jsonl"
CACHE_DIR = HERE / "data" / "real_claims_fulltext"
V3_RUN_PATH = HERE / "results" / "v3" / "real_runA.jsonl"
OUT_PATH = HERE / "results" / "v4" / f"{ITEM_ID}_policy_replay.json"


def _find_row(rows: list[dict[str, Any]], item_id: str, source: Path) -> dict[str, Any]:
    for row in rows:
        if row.get("item_id") == item_id:
            return row
    raise SystemExit(f"{item_id}: not found in {source}")


def load_item_and_cached_first_pass() -> tuple[dict[str, Any], dict[str, Any]]:
    """The item's claim/title/authors/chunks and its v3 cached first-pass answer."""
    item = _find_row(read_jsonl(CLAIMS_PATH), ITEM_ID, CLAIMS_PATH)
    cached = _find_row(read_jsonl(V3_RUN_PATH), ITEM_ID, V3_RUN_PATH)
    cache = load_cache(CACHE_DIR, [item["chunk_doi"]])
    paper = cache.get(item["chunk_doi"])
    if paper is None:
        raise SystemExit(f"{item['chunk_doi']}: missing from cache {CACHE_DIR}")
    item = dict(item)
    item["chunks"] = [str(c["text"]) for c in paper["chunks"]]
    return item, cached


def cached_first_pass_kwargs(cached: dict[str, Any]) -> dict[str, Any]:
    """Build the ``ModelPassAnswer`` constructor kwargs from a cached v3 result row.

    Pure and network-free: this is the same recorded answer the eval-check's own
    guard-only replay used for this row, reused verbatim as the shared policy's first
    ``call_model()`` return value, so the first pass costs nothing and reproduces the known
    trigger shape (``model_status`` ``verified``, guarded to ``needs_nuance`` with
    ``machine_reasons == ["quote_not_verbatim"]``) rather than a fresh, unpredictable draw.
    """
    return {
        "status": cached["model_status"],
        "evidence_quote": cached["evidence_quote"],
        "evidence_quotes": list(cached.get("evidence_quotes") or []),
        "assertions": [dict(a) for a in (cached.get("assertions") or [])],
        "explanation": cached.get("explanation"),
        "suggested_revision": cached.get("suggested_revision"),
        "unstated_details": list(cached.get("unstated_details") or []),
        "input_tokens": cached.get("input_tokens"),
        "output_tokens": cached.get("output_tokens"),
        "system_fingerprint": cached.get("system_fingerprint"),
        "model_reported": cached.get("model_reported"),
    }


async def _run(confirm_cost: bool) -> int:
    add_backend_to_path()
    export_env_from_dotenv(ENV_KEYS)
    from app.agents.analysis_agent import AnalysisDependencies
    from app.agents.claim_verification_agent import (
        CLAIM_VERIFICATION_PROMPT_VERSION,
        VERIFICATION_PROMPT,
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

    item, cached = load_item_and_cached_first_pass()
    model_configured = str(getattr(settings, "deepseek_model", "deepseek-chat"))
    prompt = format_verification_prompt(
        item["claim"], item["chunks"], item["title"], item["authors"]
    )
    agent = get_claim_verification_agent()
    deps = AnalysisDependencies(
        project_id=EVAL_PROJECT_ID,
        project_description=None,
        target_journal=None,
        citation_style="APA",
    )

    call_count = 0
    live_calls: list[dict[str, Any]] = []

    async def call_model(repair_request: Any = None) -> ModelPassAnswer:
        # Task authorisation 2026-09-11: `verify_claim_with_policy` now calls
        # `call_model(repair_request)` on every pass (`None` on the first), and would
        # build a repair turn from *repair_request* on the second -- this script's own
        # frozen output (`results/v4/real-test-18_policy_replay.json`) already records
        # what its identical-prompt second pass returned under the 2026-09-10 policy and
        # is never regenerated (FROZEN); *repair_request* is accepted, not used, purely
        # so this script's own `call_model` still matches the shared policy's calling
        # convention rather than raising `TypeError` if it is ever imported or re-run.
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Part A's own first pass: the cached v3 answer, no network call.
            return ModelPassAnswer(**cached_first_pass_kwargs(cached))
        # Part B's bounded second pass, licensed by the policy itself (this script never
        # decides this): a real, paid call under the identical prompt.
        if not confirm_cost:
            raise SystemExit(
                "second pass licensed but --confirm-cost not given; refusing to spend API "
                "budget without explicit confirmation"
            )
        result = await agent.run(prompt, deps=deps)
        out = result.output
        out_dict = out.model_dump() if hasattr(out, "model_dump") else dict(vars(out))
        explanation = normalise_verification_text(
            out_dict.get("explanation"), strip_wrapping_quotes=True
        )
        suggested_revision = normalise_verification_text(out_dict.get("suggested_revision"))
        evidence_quotes = [
            normalise_verification_text(q) for q in (out_dict.get("evidence_quotes") or [])
        ]
        evidence_quote = (
            evidence_quotes[0] if evidence_quotes
            else normalise_verification_text(out_dict.get("evidence_quote"))
        )
        assertions = [
            {
                **a,
                "quote": normalise_verification_text(a.get("quote")),
                "quotes": [normalise_verification_text(q) for q in (a.get("quotes") or [])],
            }
            for a in (out_dict.get("assertions") or [])
        ]
        record = provenance_from_run(
            "claim_verification",
            result,
            model_configured=model_configured,
            temperature=temperature,
            prompt=VERIFICATION_PROMPT,
        )
        usage = usage_extras(result)
        live_calls.append(
            {
                "called_at": now_iso(),
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "cache_read_tokens": usage.get("cache_read_tokens"),
                "model_reported": record.model_reported,
                "system_fingerprint": record.system_fingerprint,
                "provider_response_id": record.provider_response_id,
            }
        )
        return ModelPassAnswer(
            status=out_dict.get("status"),
            evidence_quote=evidence_quote,
            evidence_quotes=evidence_quotes,
            assertions=assertions,
            explanation=explanation,
            suggested_revision=suggested_revision,
            unstated_details=out_dict.get("unstated_details") or [],
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            system_fingerprint=record.system_fingerprint,
            model_reported=record.model_reported,
        )

    started = now_iso()
    policy_result = await verify_claim_with_policy(
        item["claim"], item["chunks"], [{"text": c} for c in item["chunks"]], call_model
    )
    finished = now_iso()

    second_pass_fired = len(policy_result.passes) > 1
    total_cost = None
    if live_calls:
        price = price_record(model_configured, None, None)
        total_cost = sum(
            call_cost(
                c["model_reported"] or model_configured,
                c["called_at"],
                c["input_tokens"] or 0,
                c["output_tokens"] or 0,
                c["cache_read_tokens"] or 0,
            )
            or 0.0
            for c in live_calls
        )
    else:
        price = price_record(model_configured, None, None)

    output = {
        "item_id": ITEM_ID,
        "purpose": (
            "Live replay of the one cached v3 "
            "answer (real_runA) whose guarded first pass licenses verify_claim_with_policy's "
            "bounded second pass, to record what a live second model call actually returns "
            "for it."
        ),
        "source_run": "results/v3/real_runA.jsonl",
        "claim": item["claim"],
        "chunk_doi": item["chunk_doi"],
        "cached_first_pass": {
            "model_status": cached["model_status"],
            "predicted_status_after_guard": cached["predicted_status"],
            "machine_reasons_after_guard": cached["machine_reasons"],
            "input_tokens": cached.get("input_tokens"),
            "output_tokens": cached.get("output_tokens"),
            "system_fingerprint": cached.get("system_fingerprint"),
            "model_reported": cached.get("model_reported"),
            "called_at": cached.get("called_at"),
            "note": "reused verbatim, no network call for this pass",
        },
        "second_pass_licensed": second_pass_fired,
        "second_pass_live_call": live_calls[0] if live_calls else None,
        "final": {
            "status": policy_result.status,
            "machine_reasons": policy_result.machine_reasons,
            "diagnostics": policy_result.diagnostics,
            "evidence_quote": policy_result.evidence_quote,
            "evidence_quotes": policy_result.evidence_quotes,
            "assertions": policy_result.assertions,
            "explanation": policy_result.explanation,
            "suggested_revision": policy_result.suggested_revision,
            "model_status": policy_result.model_status,
            "model_reported": policy_result.model_reported,
            "quote_relocations": policy_result.quote_relocations,
            "passes": policy_result.passes,
        },
        "meta": {
            "guard_digest": GUARD_DIGEST,
            "quote_relocation_version": QUOTE_RELOCATION_VERSION,
            "verification_policy_version": VERIFICATION_POLICY_VERSION,
            "prompt_version": CLAIM_VERIFICATION_PROMPT_VERSION,
            "model_configured": model_configured,
            "temperature": temperature,
            "n_live_calls": len(live_calls),
            "price": price,
            "total_cost": total_cost,
            "started": started,
            "finished": finished,
        },
    }
    write_json(OUT_PATH, output)
    print(
        f"wrote {OUT_PATH}: second_pass_licensed={second_pass_fired} "
        f"final_status={policy_result.status!r} n_live_calls={len(live_calls)} "
        f"total_cost={total_cost!r}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--confirm-cost",
        action="store_true",
        help="required if the frozen policy licenses a second pass for this item (it does): "
        "acknowledges the one paid API call this script may make",
    )
    args = ap.parse_args(argv)
    return asyncio.run(_run(args.confirm_cost))


if __name__ == "__main__":
    raise SystemExit(main())
