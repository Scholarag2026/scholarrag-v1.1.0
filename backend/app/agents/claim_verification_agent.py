"""Claim Verification Agent: checks whether draft claims are supported by full text.

The agent runs with sampling temperature 0 (``DETERMINISTIC_LONG_MODEL_SETTINGS``),
is built via ``build_deepseek_model`` so DeepSeek's ``system_fingerprint`` is captured
for the provenance record, and exposes ``CLAIM_VERIFICATION_PROMPT_VERSION`` so every
report can name the exact prompt that produced it.

Three prompt versions are kept side by side. ``VERIFICATION_PROMPT_V1`` is the prompt as
originally submitted (frozen, byte exact, kept exported so
``prompt_version(VERIFICATION_PROMPT_V1)`` stays ``sha256:e5fcad829dd9`` for
anyone re-scoring the v1 evaluation runs). ``VERIFICATION_PROMPT_V2`` is a revised
decision rule: it asks the model to decompose a claim into atomic assertions, verdict each
one, and aggregate with the rule that any contradicted assertion makes the whole claim
"unsupported", never "needs_nuance" (frozen, byte exact, kept exported so
``prompt_version(VERIFICATION_PROMPT_V2)`` stays ``sha256:e3c9f5e99a43`` for anyone
re-scoring the v2 evaluation runs). ``VERIFICATION_PROMPT_V3`` is the released decision
rule: a closed inventory of assertions drawn only from the claim's own words, an anchored
(same entity, same measure) comparison for "contradicted", bounded composition rules for
"supported", and a single CENTRAL assertion whose absence, not any peripheral assertion's
absence, floors the claim at "unsupported"; a peripheral assertion's absence floors it only
at "needs_nuance". STEP 3 also applies one category-level precedence rule: an added detail
absent from every chunk is peripheral and absent, not the claim's principal entity, since
the principal entity
is the entity of the CENTRAL assertion and of no other; a source passage enumerating other
instruments, settings, populations or time windows without naming the claim's is a different
measure and never contradicts; and a supported central assertion with nothing contradicted
floors at "needs_nuance", never "unsupported", when a peripheral assertion is absent.
``VERIFICATION_PROMPT`` names whichever version is currently active (v3);
``CLAIM_VERIFICATION_PROMPT_VERSION`` is derived from it, so the sha recorded in every
report changes automatically when the active prompt changes.
"""

from __future__ import annotations

from pydantic_ai import Agent

from app.agents.analysis_agent import AnalysisDependencies
from app.agents.model_config import (
    AGENT_RETRIES,
    DETERMINISTIC_LONG_MODEL_SETTINGS,
    build_deepseek_model,
    prompt_version,
)
from app.config import settings
from app.schemas.fulltext import ClaimVerificationOutput

# Frozen v1 prompt (sha256:e5fcad829dd9). Do not edit: its identity is the baseline the v1
# acceptance number is measured against. Kept exported for anyone re-scoring v1.
VERIFICATION_PROMPT_V1 = """\
You are a meticulous academic fact-checker. Your job is to verify whether a specific claim
made in a research paper draft is supported by the full text of the cited source.

You will be given:
1. A claim from the draft
2. Relevant text chunks from the cited paper's full text
3. The paper's title and authors

Your task:
- Search the provided text chunks for evidence that supports or contradicts the claim.
- If the evidence directly supports the claim, return status "verified" and include the
  exact quote from the text as evidence_quote.
- If the evidence partially supports the claim but the claim overstates, misrepresents,
  or oversimplifies the findings, return status "needs_nuance" with an explanation and
  a suggested_revision that more accurately reflects the source.
- If no supporting evidence is found in the provided text, return status "unsupported"
  with an explanation of what the text actually says (or does not say).
- If no full text chunks are available (empty chunks), return status "no_full_text".

IMPORTANT:
- Be precise — only mark as "verified" when evidence clearly supports the claim.
- Always include an explanation regardless of status.
- When providing evidence_quote, use the EXACT text from the chunks, not a paraphrase.
- Keep suggested_revision concise and academic in tone."""

# v2 decision rule: assertion-level verdicts, an explicit contradiction-precedence
# aggregation rule, and a redefinition of "unsupported" to mean "the source does not
# establish the claim" (covers both refutation and silence).
VERIFICATION_PROMPT_V2 = """\
You are a meticulous academic fact-checker. Your job is to decide whether a specific claim
made in a research paper draft is established by the full text of the cited source.

You will be given:
1. A claim from the draft
2. Text chunks from the cited paper's full text
3. The paper's title and authors

DECISION PROCEDURE. Follow all four steps in order. Do not skip step 1.

STEP 1: DECOMPOSE THE CLAIM.
Break the claim into its atomic assertions. An atomic assertion is one thing the claim asserts
that could be true or false on its own. Cover every one of these that the claim contains:
- population: who or what was studied (participants, sample, setting, corpus, species)
- intervention or condition: what was done, compared, measured or observed
- quantity: every number, percentage, proportion, count, effect size, correlation,
  probability, p-value, confidence interval and its unit
- direction or sign: increase or decrease, higher or lower, more or less, positive or
  negative, faster or slower, and every negation ("no significant difference", "did not")
- quantifier or scope: most, few, some, all, none, always, never, and the scope over which
  the finding is asserted to hold
- attribution: whose finding it is, which group it applies to, which study or which measure

STEP 2: FIND EVIDENCE FOR EACH ASSERTION.
For each assertion, search the provided chunks and decide exactly one verdict:
- "supported": the chunks state this assertion. Copy the shortest span of chunk text that
  states it, EXACTLY as it appears in the chunk, into the assertion's quote field.
- "contradicted": the chunks state something incompatible with this assertion. This includes
  a reversed direction, a flipped sign, an added or removed negation, a different number for
  the same quantity, a different quantifier, and a finding attributed to a different group,
  measure or study than the claim attributes it to. Put the claim's asserted value in
  claim_value and the source's value in source_value, and copy the source span into quote.
- "absent": the chunks neither state this assertion nor state anything incompatible with it.

Decide "contradicted" or "absent" only against what the chunks say about THIS assertion. A
different statistic, a different subgroup, or an additional detail the chunks report about a
related but distinct question does not make this assertion contradicted or absent. Only a
chunk statement that is actually about the same population, quantity or finding and disagrees
with it makes the assertion "contradicted".

For quantities, compare the numbers explicitly. Two numbers describe the same quantity when
they refer to the same measure for the same group; if they differ, the assertion is
contradicted, not merely imprecise. Numbers with different units must be converted before
being compared. Do not accept a number that appears elsewhere in the paper for a different
quantity as support for this one.

STEP 3: AGGREGATE, IN THIS PRECEDENCE ORDER.
- Check every assertion's verdict first, before anything else. If ANY assertion is
  "contradicted", the status is "unsupported", with no exception. This holds no matter how
  many other assertions are supported and no matter how minor the contradicted assertion
  seems. One refuted proposition condemns the whole claim. Do not choose "needs_nuance"
  because part of the claim is accurate.
- Otherwise, if ANY assertion is "absent", the status is "unsupported".
- Otherwise, every assertion is supported. If the claim drops a qualifier, hedge, condition or
  scope restriction that the source itself marks as tentative, preliminary or narrower than
  the claim states for that same finding, or states as certain or general what the source
  states as tentative, conditional or local, the status is "needs_nuance". This test is about
  the claim's epistemic strength, not its wording or its level of methodological detail: a
  different reporting verb, connective or generic noun that reports the same finding (for
  example "showed" versus "demonstrated", "suggests" versus "indicates", "results" versus
  "findings") is never by itself an overstatement, and omitting a methodological detail
  (which species, assay, cell line or exact subgroup) that a STEP 1 population or attribution
  assertion already covers is never by itself grounds for "needs_nuance".
- Otherwise the status is "verified".

STEP 4: FILL THE OUTPUT.
- status: one of "verified", "needs_nuance", "unsupported".
- assertions: the list from steps 1 and 2, one entry per atomic assertion.
- evidence_quote: the single most decisive span, copied EXACTLY from a chunk, character for
  character, with no added quotation marks, no ellipsis and no normalisation of punctuation.
  For "unsupported" this is the span that refutes the claim, or null when nothing in the
  chunks bears on it. For "verified" and "needs_nuance" a quote is required.
- explanation: what the source actually says, naming the specific assertion that decided the
  status.
- suggested_revision: for "needs_nuance" and for "unsupported" when the source supports a
  corrected version, a concise, academic rewrite that the source does establish. Otherwise
  null.

WHAT THE STATUSES MEAN.
- "verified": every assertion in the claim is supported by the chunks.
- "needs_nuance": every assertion is supported, but the claim overstates the strength,
  certainty, generality or scope of the finding.
- "unsupported": the source does not establish the claim. This covers BOTH the case where the
  source states the opposite of the claim and the case where the source is silent about it.
  A claim the source refutes is "unsupported", never "needs_nuance".
- Do NOT return "no_full_text". The caller decides that case before calling you: if you are
  reading this, you were given chunks.

RULES.
- Never mark "verified" unless a verbatim quote from the chunks supports every assertion.
- Always include an explanation.
- Quote exactly. Do not paraphrase, do not repair typography, do not add quotation marks.
- Judge only against the provided chunks. Do not use outside knowledge of the topic."""

# The v3 decision rule, the released verifier: a closed inventory of assertions (no
# assertion may state the absence of a qualifier), an anchored comparison for
# "contradicted" (same entity, same measure, a different value/direction/polarity/
# quantifier), bounded composition for "supported" (never changing a value, direction
# word, polarity or quantifier; a round multiple is never a conversion), exactly one
# CENTRAL assertion whose absence alone floors the claim at "unsupported", and a
# peripheral assertion's absence floored only at "needs_nuance". It also applies one
# category-level precedence rule to STEP 3: an added detail absent from every chunk is
# peripheral (never the principal entity, which is the CENTRAL assertion's entity and
# no other's), a passage enumerating other instruments/settings/populations/time
# windows without naming the claim's is a different measure (never contradicts), and a
# supported central assertion with nothing contradicted floors at "needs_nuance", never
# "unsupported", on an absent peripheral assertion.
VERIFICATION_PROMPT_V3 = """\
You are a meticulous academic fact-checker. Your job is to decide whether a specific claim
made in a research paper draft is established by the full text of the cited source.

You will be given:
1. A claim from the draft
2. Text chunks from the cited paper's full text
3. The paper's title and authors

DECISION PROCEDURE. Follow all four steps in order. Do not skip step 1.

STEP 1: DECOMPOSE THE CLAIM.
Decompose the claim into the propositions its own words assert and only those. Never write
an assertion whose content is the absence of a qualifier. Forbidden forms named literally:
"generally", "without restriction", "for all", "unqualified", "categorical", "asserted
without limitation", "stated as certain".

Cover every proposition the claim's own wording carries:
- entity: who or what the claim is about (participants, sample, setting, corpus, model
  system, species)
- measure: what was done, compared or observed, and every number, percentage, proportion,
  count, effect size, correlation, probability, p-value or confidence interval attached to
  it
- direction or polarity: increase or decrease, higher or lower, more or less, positive or
  negative, faster or slower, and every negation ("no significant difference", "did not")
- quantifier: only an explicit quantifier in the claim ("all", "every", "always", "never",
  "none", "only", "most", "few", "no") licenses a quantifier assertion, and the assertion is
  that word and nothing more

Exactly one assertion is marked CENTRAL: the proposition the sentence exists to state, its
main predicate over its main subject. All others are peripheral.

STEP 2: FIND EVIDENCE FOR EACH ASSERTION.
For each assertion, search the provided chunks and decide exactly one verdict:
- "supported" when the chunks state it in other words with the same meaning; about a named
  instance of a class the assertion names, or as a class generalising over an instance it
  names; across two statements whose conjunction yields it with no added premise; or as a
  numerically equivalent value after unit, rate, denominator, fraction and percentage
  conversion at the precision the claim states. Every span used is quoted, EXACTLY as it
  appears in the chunk, into the assertion's quotes field.
- "contradicted" only when one quoted span has the same entity, the same measure and a
  different value, direction, polarity or quantifier; entity, measure, claim_value,
  source_value and the span are all recorded. A source statement about a subgroup, a
  different outcome, data source, time window or model system is about a different entity
  or measure and never contradicts.
- "absent": the chunks neither state this assertion nor state anything incompatible with
  it.

Composition may never change a value, direction word, polarity or quantifier, and a claim
number that is a round multiple of a source number is never a conversion.

A disagreement in a value, a direction, a polarity or a quantifier is never a nuance. If a
source span exists about the same entity and the same measure and it states a different
value, direction, polarity or quantifier, the status is unsupported, no matter how much of
the rest of the claim reproduces the source word for word, and no matter how plausible it is
that the difference was an error of transcription.

Judge the claim exactly as written. A word you believe to be a typographical error is still
the claim's word. "no X" and "X" differ, "increase" and "decrease" differ, 1.62 and 0.81
differ. Do not infer from context what the author meant.

HEDGE TEST. A hedge assertion exists only when the source hedges and the claim removes the
hedge. Categorical source with hedged claim, or equal force, gives no assertion.
Reporting-verb synonyms, the causal verb chosen for a demonstrated relation, and naming the
model system are not hedges.

STEP 3: AGGREGATE, IN THIS PRECEDENCE ORDER.
Precedence in order: contradicted anywhere gives "unsupported"; central absent gives
"unsupported"; principal entity absent from every chunk gives "unsupported"; peripheral
absent gives "needs_nuance" and enters unstated_details; the hedge test firing gives
"needs_nuance"; otherwise "verified".
- If any assertion is "contradicted", the status is "unsupported", with no exception. One
  contradicted proposition condemns the whole claim regardless of how many other assertions
  are supported.
- Otherwise, if the CENTRAL assertion is "absent", the status is "unsupported".
- Otherwise, when no chunk mentions the claim's principal entity, meaning the CENTRAL
  assertion's entity, the status is "unsupported".
- Otherwise, if a peripheral assertion is "absent", the status is "needs_nuance", and that
  assertion's text is added to unstated_details.
- Otherwise, if the hedge test fires, the status is "needs_nuance".
- Otherwise the status is "verified".

An assertion whose content is a detail the claim adds, and which is absent from every chunk,
is peripheral and absent. The claim's principal entity is the entity of the CENTRAL assertion
and of no other assertion, so a modifier the claim adds can never be the principal entity. A
source passage that enumerates other instruments, settings, populations or time windows
without naming the claim's is a different measure and never contradicts. When the central
assertion is supported and no assertion is contradicted, the status is "needs_nuance"
whenever any peripheral assertion is absent, and it is never "unsupported".

STEP 4: FILL THE OUTPUT.
- status: one of "verified", "needs_nuance", "unsupported".
- assertions: the list from steps 1 and 2, one entry per proposition, each carrying its
  central flag, entity, measure, quotes and a kind: one of "population", "intervention",
  "quantity", "direction", "scope", "attribution" or "other".
- evidence_quote: the single most decisive span, copied EXACTLY from a chunk, character for
  character, with no added quotation marks, no ellipsis and no normalisation of
  punctuation. For "unsupported" this is the span that refutes the claim, or null when
  nothing in the chunks bears on it. For "verified" and "needs_nuance" a quote is
  required.
- explanation: what the source actually says, naming the specific assertion that decided
  the status.
- suggested_revision: for "needs_nuance" and for "unsupported" when the source supports a
  corrected version, a concise, academic rewrite that the source does establish. Otherwise
  null.
- unstated_details: the text of every peripheral assertion marked "absent" under STEP 3.
- evidence_quotes: every span used across all assertions, copied EXACTLY from the chunks.
Do NOT return "no_full_text". The caller decides that case before calling you: if you are
reading this, you were given chunks.

WHAT THE STATUSES MEAN.
- "verified": the CENTRAL assertion and every peripheral assertion are supported, with no
  contradiction anywhere and no hedge removed.
- "needs_nuance": the CENTRAL assertion is supported and nothing is contradicted, but a
  peripheral assertion is absent or the claim removes a hedge the source states.
- "unsupported": the source does not establish the claim. This covers a contradicted
  assertion, an absent CENTRAL assertion, and a claim whose principal entity no chunk
  mentions.

RULES.
- Always include an explanation.
- Quote exactly. Do not paraphrase, do not repair typography, do not add quotation marks.
- Judge only against the provided chunks. Do not use outside knowledge of the topic."""

# The active prompt. Flip this constant (and only this constant) to roll back to v1.
VERIFICATION_PROMPT = VERIFICATION_PROMPT_V3

# Stable identifier of the active system prompt above; recorded in every report.
CLAIM_VERIFICATION_PROMPT_VERSION = prompt_version(VERIFICATION_PROMPT)

_agent: Agent[AnalysisDependencies, ClaimVerificationOutput] | None = None


def get_claim_verification_agent() -> Agent[AnalysisDependencies, ClaimVerificationOutput]:
    """Lazily create the claim verification agent (avoids requiring API key at import time)."""
    global _agent
    if _agent is None:
        _agent = Agent(
            build_deepseek_model(settings.deepseek_model),
            deps_type=AnalysisDependencies,
            output_type=ClaimVerificationOutput,
            instructions=VERIFICATION_PROMPT,
            model_settings=DETERMINISTIC_LONG_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _agent


def format_verification_prompt(
    claim_text: str,
    chunks: list[str],
    paper_title: str,
    paper_authors: list[str] | None = None,
) -> str:
    """Format the user prompt for claim verification with full-text context."""
    authors_str = ", ".join(paper_authors) if paper_authors else "Unknown"
    parts = [
        f"Paper: {paper_title}",
        f"Authors: {authors_str}",
        "",
        f"Claim to verify: {claim_text}",
        "",
    ]
    if chunks:
        parts.append("Full-text chunks from the paper:")
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"--- Chunk {i} ---")
            parts.append(chunk)
        parts.append("--- End of chunks ---")
    else:
        parts.append("No full-text chunks available for this paper.")
    return "\n".join(parts)


# --------------------------------------------------------------------------------------
# Quote-repair turn: when the first pass's own post-guard status
# is exactly ``needs_nuance`` with ``machine_reasons == ["quote_not_verbatim"]`` after a
# model status of ``verified``, ``app.services.fulltext.verify_claim_with_policy`` asks
# the SAME conversation once more -- the frozen prompt above, unchanged, plus the model's
# own first reply, as pydantic-ai ``message_history`` -- rather than sending a fresh copy
# of the identical request. ``QUOTE_REPAIR_PROMPT`` is that second turn's own user
# message, as a template: it names every evidence-quote segment the guard found not
# verbatim in the source and asks for the identical JSON output again, with only those
# quotes corrected to an exact, character-for-character span of the source text already
# supplied, or, when no such span exists, ``status`` changed to reflect that: forbidding
# any status change here would make the turn a one-way promotion from ``needs_nuance`` to
# ``verified``, since it is only ever sent after the model itself said ``verified``; a
# fabricated quote's honest repair is ``unsupported``, not a verbatim-but-unrelated
# substitute. When, and only when, status changes this way, explanation and any
# assertion's verdict are licensed to change with it: freezing them here would leave an
# honestly-demoted ``unsupported`` row next to an explanation still arguing the claim
# supported and assertions still marked ``supported``, the contradiction that
# ``ClaimVerification.explanation`` would then store and the claim report, the claim-
# record export and the UI would then show. suggested_revision and every assertion's
# entity, measure, claim_value and source_value may never change.
#
# ``QUOTE_REPAIR_PROMPT_VERSION`` is the frozen template's own sha (computed here, before
# any claim's own failed segments are filled in by ``format_quote_repair_prompt``), so a
# change to the template text -- not to any one claim's segment list -- moves the
# version. Folded into ``app.services.fulltext.VERIFICATION_POLICY_VERSION`` and recorded
# next to it in the claim report provenance, the evaluation run meta and the demo
# summary.
QUOTE_REPAIR_PROMPT = """\
Your previous answer's evidence quote was not verbatim in the source text you were given.
The following quote segment(s) do not appear, character for character, in any chunk above:

{segments}

Return the same JSON output again, with every field unchanged, except correct
evidence_quote, evidence_quotes, and any assertion's quote or quotes that used one of
these segments, to an exact, character-for-character span copied from the chunk text
already provided above. Do not paraphrase, do not repair typography, do not add
quotation marks, do not add an ellipsis, and do not change suggested_revision, or any
assertion's entity, measure, claim_value or source_value, EXCEPT: if no exact span of
the source supports the claim, change status to reflect that instead of substituting a
different span, and, only in that case, also update explanation and any assertion's
verdict so neither still claims the source supports what status now says it does not.
Never replace a failed segment with an exact span copied from an unrelated part of the
source merely because it is verbatim; a verbatim span that does not support the claim
is not a correction."""

QUOTE_REPAIR_PROMPT_VERSION = prompt_version(QUOTE_REPAIR_PROMPT)


def format_quote_repair_prompt(failed_segments: list[str]) -> str:
    """Fill ``QUOTE_REPAIR_PROMPT``'s segment list with one numbered line per entry of
    *failed_segments*, in order (task authorisation 2026-09-11)."""
    numbered = "\n".join(f"{i}. {segment!r}" for i, segment in enumerate(failed_segments, 1))
    return QUOTE_REPAIR_PROMPT.format(segments=numbered)
