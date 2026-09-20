#!/usr/bin/env python
"""Deterministic, offline checks over a demo run's delivered text.

The question behind this module: why do defects in the text ScholarRAG actually
delivers keep being found only by a full, paid demo run and a manual read-through,
instead of being caught first? The answer is this module: a set of invariants over the
artefacts any demo run already writes (``draft_content.json``, ``writing_result.json``,
``claim_report.json``, ``delivered_evidence.json``), checked as code, with no network
call, no database and no LLM call, so they run before a paid run (against a fixture, or
a previous run) and again inside every run, right after each section is written and
again at the end.

Seventeen invariants:

1. Every body sentence of every saved draft is either a verified cited sentence of the
   final report (its ``claim_text``/``claim_sentence`` found verbatim, or containing
   it) or a sentence the writer's own citation-link call tagged "framing". Nothing
   else survives: no sentence the call leaves unclassified, and no sentence it tagged
   "finding" (an uncited empirical claim) either.
2. No literal "[NEEDS CITATION]" marker survives in the draft, and no citation the
   draft's own extractor recognised is left with no verified row behind it
   (``citation_coverage.unresolved == 0``).
3. No section opens with two consecutive headings carrying the same text.
4. Every row of the final (healed) report is "verified", and its ``claim_text`` occurs
   verbatim in the section's own saved draft.
5. Every evidence quote of every ``delivered_evidence.json`` row is a verbatim (under
   the verifier guard's own letters-and-digits fold) substring of that row's own
   ``source_passage``, and every row's ``passage_located`` is true.
6. No sentence is truncated at an abbreviation: neither a draft sentence nor a final
   report row's ``claim_text``/``claim_sentence`` ends with a period directly after a
   token this module knows is not sentence-final (the "vs." case), and none begins
   with a lowercase-letter fragment left over from a false split.
7. Every protocol fixture claim expecting "verified" is present in the final report,
   and every fixture expecting "unsupported" is absent from the draft.
8. The deterministic coherence pass `app.services.fulltext` runs after finalize,
   checked independently against the delivered draft rather than inferred from the
   write job's own statistics: (a) no body paragraph other than the document's own
   opening one carries zero citations; (b) no enumeration opener ("Three gaps
   emerge.") has fewer sentences than it announces following it in the same
   paragraph, unless the very next sentence resolves the same count as a colon-free
   list of its own; (c) no UNCITED sentence opens with a discourse connective/ordinal
   or an unresolved deictic ("However", "the same authors", "these gaps") once an
   earlier enumeration shortfall (b) has fired in the same paragraph. Rule 8c is
   deliberately narrower than a naive version of this check would be -- it never
   fires on a cited sentence, and never on a paragraph's own first sentence merely
   because a heading precedes it, because neither shape is one the app-side fix can
   safely remove or rewrite without deleting cited content or adding an LLM turn
   (see `check_dangling_framing`'s own docstring for the residue this rule accepts
   rather than gates the exit code on).
9. A heading with no body -- because it is the last node of the section's own draft, or
   because the very next node is itself another heading -- never survives finalize
   (`app.services.fulltext._drop_headings_with_no_body`), checked here independently
   against the delivered draft's own node sequence.
10. Every delivered, cited body sentence asserts nothing beyond what its own verified
    claims and citations cover -- a content-bearing residue outside a closed set of
    three discourse frames (F1-F3) is a violation. Separately, every writer
    proposition of the WHOLE SECTION -- not only the ones whose own sentence still
    survives in the delivered draft -- is matched to a verdict individually; one with
    no matching row at all is the dedup-defect shape (design section 1), reported
    under this same rule name, with the section's own match rate reported as a
    notice regardless of outcome.
11. No delivered sentence, and no verified claim behind one, carries a comparative
    meta-evaluation of the literature ("the strongest evidence",
    "treated most directly", or a comparative such as "provide stronger evidence" or
    "the most controlled comparison").
12. No delivered sentence opens with a citation-shaped subject immediately followed
    by a present-participle/gerund reporting-verb form and no finite verb before it
    -- a subject with no finite verb of its own at all ("Teng and Ma (2024)
    cautioning that ...").
13. No delivered sentence opens with a past-tense or third-person reporting verb
    followed by "that", with no subject before it at all ("Showed that one student
    resubmitted her essay 13 times ...").
14. A section whose own final report has no verified row at all while its own
    citation audit found at least one citation, or whose own write-stage
    ``loop_stats`` records that the empty-section regeneration backstop also
    finalized to zero cited sentences on its second attempt, is a violation: a
    heading delivered over a hollow paragraph.
15. No delivered body paragraph echoes the section's own title or a heading still
    standing in its own draft -- the section repeating its own title, or a model's
    own sub-heading demoted to plain text once its real body was removed, delivered
    as if it were a finding or framing sentence of its own.
16. No delivered body paragraph reads as a heading rather than a sentence: no
    sentence-final punctuation, at most ten words, no citation, and no finite verb
    -- whatever position it is in, and whatever left it without a heading marker of
    its own.
17. Inside a generated section, the section's own title is the only heading a
    delivered draft ever carries. No heading node survives anywhere but the
    section's own first node, whether or not it has a body of its own -- every
    other heading the writer wrote is dropped whole at finalize now, whatever it
    says.

Self-contained by design: this module has no import of ``run_demo`` (which imports
this module, not the reverse) and no import of the backend (``demo/`` scripts run
under the system Python, which does not carry the backend's dependencies -- see
``demo/run_demo.py``'s own module docstring). Where the logic mirrors backend or
``run_demo.py`` code -- the "[NEEDS CITATION]" marker, the "vs." abbreviation, the
verifier guard's own letters-and-digits fold, the draft's own paragraph/heading
extraction -- the docstring says so; the constant or function is a fresh, independent
copy, not an import, exactly as ``run_demo.py`` already duplicates the same handful of
backend constants for the same reason.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

CHECK_DIR = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = CHECK_DIR / "protocol.json"

# ---------------------------------------------------------------------------
# Rule names -- also what a reader searches this module's own report for.
# ---------------------------------------------------------------------------

RULE_UNCLASSIFIED_SENTENCE = "unclassified_sentence"
RULE_UNCITED_FINDING_SURVIVED = "uncited_finding_survived"
RULE_NEEDS_CITATION_MARKER = "needs_citation_marker"
RULE_UNRESOLVED_CITATION = "unresolved_citation"
RULE_DUPLICATE_HEADING = "duplicate_heading"
RULE_ROW_NOT_VERIFIED = "final_report_row_not_verified"
RULE_CLAIM_TEXT_NOT_IN_DRAFT = "claim_text_not_in_draft"
RULE_EVIDENCE_NOT_VERBATIM = "evidence_quote_not_verbatim"
RULE_PASSAGE_NOT_LOCATED = "passage_not_located"
RULE_TRUNCATED_SENTENCE = "truncated_sentence"
RULE_FIXTURE_VERIFIED_MISSING = "fixture_verified_missing"
RULE_FIXTURE_UNSUPPORTED_PRESENT = "fixture_unsupported_present"
#: Mirrors the deterministic coherence pass `_drop_dangling_framing_sentences` and its
#: two callers, which run after finalize (`app.services.fulltext`): rule 8, a body
#: paragraph with no citation at all, other than the document's own opening paragraph
#: (8a); an enumeration opener with fewer sentences than announced following it in the
#: same paragraph, and no colon-free list resolving the same count one sentence later
#: (8b); and an UNCITED framing sentence opening with a discourse connective/ordinal
#: or an unresolved deictic once an earlier enumeration shortfall has fired in the
#: same paragraph (8c). Named separately from the app-side fix's own rule constants
#: only because this module never imports the backend (see the module docstring).
#: Rule 8c is narrowed to exactly the in-paragraph-enumeration-cascade shape the
#: app-side fix's own rule (c) can actually guarantee: a cited sentence, and a
#: paragraph's own first sentence merely because a heading precedes it, are both
#: outside what a deterministic pass can safely remove or rewrite (removing a cited
#: sentence would delete supported content; rewriting either needs an LLM turn), so
#: both are residue, tracked separately rather than gated on this rule's own exit
#: code -- see `check_dangling_framing`'s own docstring below.
RULE_PARAGRAPH_HAS_NO_CITATION = "paragraph_has_no_citation"
RULE_ENUMERATION_OPENER_SHORT_OF_COUNT = "enumeration_opener_short_of_count"
RULE_DANGLING_FRAMING_SENTENCE = "dangling_framing_sentence"
#: Rule 9, a heading with no body at all (mirrors
#: `app.services.fulltext._drop_headings_with_no_body`).
RULE_HEADING_WITH_NO_BODY = "heading_with_no_body"
#: Rule 10, a delivered cited sentence that asserts material no verified claim
#: covers -- a residue span content-bearing enough that it is not one of the three
#: permitted discourse frames (F1-F3), or a section's own citation-link proposition
#: with no verification-list row behind it at all (the dedup-defect shape), checked
#: over the whole section rather than only the sentences still delivered. Two
#: violation shapes under one rule name, exactly as the design specifies.
RULE_SENTENCE_EXCEEDS_VERIFIED_CLAIM = "sentence_exceeds_verified_claim"
#: Rule 11, a delivered sentence carrying a comparative meta-evaluation of the
#: literature ("the strongest evidence", "treated most directly") -- the writer's own
#: comparative judgement about sources, never a claim any one citation supports.
RULE_COMPARATIVE_META_EVALUATION = "comparative_meta_evaluation"
#: Rule 12: a delivered sentence opening with a citation-shaped subject immediately
#: followed by a present-participle/gerund reporting-verb form, with no finite verb
#: anywhere before it -- a subject with no finite verb of its own at all ("Teng and
#: Ma (2024) cautioning that ...").
RULE_PARTICIPLE_AFTER_ATTRIBUTION = "participle_after_attribution"
#: Rule 13: a delivered sentence whose own first two tokens are a past-tense or
#: third-person reporting verb followed by "that", with no subject before it at
#: all ("Showed that one student resubmitted her essay 13 times while another
#: made one resubmission (Zhang & Hyland, 2018)."). A residue whose own content is
#: entirely zero (a bare "showed that"/"reports that") passes the app's own
#: coverage check as a permitted frame with no regard for whether a subject
#: actually precedes the verb; this shape results when the sentence's own
#: citation trails at the end instead.
RULE_VERB_INITIAL_SENTENCE = "verb_initial_sentence"
#: New rule: a section whose own ``final_report.verifications`` is empty while its
#: own ``writing_result.citation_audit.total`` is greater than zero, or whose own
#: ``writing_result.loop_stats.empty_section_after_regeneration`` is true -- a heading
#: delivered over a hollow paragraph, the shape none of rules 1-13 can see because
#: every in-run assertion and every existing rule here passes trivially on an empty
#: section.
RULE_EMPTY_CITED_SECTION = "empty_cited_section"
#: New rule: a delivered body paragraph whose own text echoes the section's own title
#: or a heading still standing in its own draft -- the section repeating its own
#: title, or a model's own sub-heading demoted to plain text once its real body was
#: removed, delivered as if it were a finding or framing sentence of its own.
RULE_TITLE_ECHO_IN_BODY = "title_echo_in_body"
#: Rule 16: a delivered body paragraph that reads as a heading rather than a
#: sentence (`is_heading_shaped_paragraph`) -- no sentence-final punctuation, at
#: most ten words, no citation, no finite verb -- whatever position it is in.
#: Independent of rule 15: a heading-shaped fragment that shares no word with any
#: title or heading (the shape rule 15 was written for but cannot, by design, reach
#: on its own) is still caught here, by its own shape rather than by echoing
#: something else.
RULE_HEADING_SHAPED_PARAGRAPH = "heading_shaped_paragraph"
#: New rule, companion to rule 9: inside a generated section, the section's own
#: title (its very first node) is the only heading it ever carries -- the app side
#: (`app.services.fulltext.finalize_generated_section`, `_strip_duplicate_leading_
#: heading`, `_rebuild_finalized_node_list`) now drops every OTHER heading the
#: writer emits, whole, at any position, whatever it says. A heading node anywhere
#: but node 0 of a delivered section's draft is therefore always a violation now,
#: independent of whether it has a body of its own (rule 9's own question).
RULE_NON_TITLE_HEADING_IN_SECTION = "non_title_heading_in_section"


@dataclass(frozen=True)
class Violation:
    """One broken invariant. *rule* is one of the ``RULE_*`` constants above; *section*
    names the section it was found in (e.g. "protocol section", "extra section 2");
    *detail* is a plain-English sentence naming what broke; *sentence* is the specific
    text the violation is about, when there is one."""

    rule: str
    section: str
    detail: str
    sentence: str | None = None

    def __str__(self) -> str:
        tail = f" -- {self.sentence!r}" if self.sentence else ""
        return f"[{self.rule}] {self.section}: {self.detail}{tail}"


class DeliveredTextViolationError(Exception):
    """Raised by ``run_demo.py``'s own wiring the moment ``check_run_directory`` (or a
    single section's own check) finds at least one `Violation`; carries the list so the
    caller can print each one with its own rule name."""

    def __init__(self, violations: Sequence[Violation]):
        self.violations = list(violations)
        super().__init__("; ".join(str(v) for v in self.violations))


# ---------------------------------------------------------------------------
# Sentence-level helpers (self-contained; see the module docstring)
# ---------------------------------------------------------------------------

#: Mirrors ``app.services.fulltext._NEEDS_CITATION_RE`` (duplicated, not imported --
#: see the module docstring): tolerant of extra internal whitespace and case, and of
#: the leading space the marker itself leaves behind once removed. Widened
#: from the exact-form ``\[\s*NEEDS\s+CITATION\s*\]`` to also catch whatever the
#: writer put between "CITATION" and the closing bracket -- a colon and an
#: explanatory clause, an em dash, or nothing -- since the model sometimes writes the
#: marker as its own editorial note rather than the bare form the writing rules ask
#: for; a plain numbered citation ("[12]") or bracketed author-year aside
#: ("[Smith, 2020]") is never matched, since neither carries "NEEDS" immediately
#: before "CITATION". A correction of this rule's own pattern, not a new rule: rule
#: 2, part 1 (``check_needs_citation_marker``) already existed to catch a surviving
#: marker; it simply could not see this shape of it before.
NEEDS_CITATION_RE = re.compile(r"\s*\[\s*NEEDS\s+CITATION\b[^\]]*\]", re.IGNORECASE)


def normalize_sentence(text: str | None) -> str:
    """*text* with any "[NEEDS CITATION]" marker removed and whitespace collapsed --
    the one normalised form every sentence-identity comparison in this module uses, so
    a marker or a whitespace difference alone never causes a false mismatch (mirrors
    ``app.services.fulltext._finalize_sentence_norm``)."""
    stripped = NEEDS_CITATION_RE.sub(" ", text or "")
    return re.sub(r"\s+", " ", stripped).strip()


#: Mirrors ``app.services.fulltext._ABBREVIATIONS_NOT_SENTENCE_FINAL``: a lower-case
#: token whose own trailing period is not a real sentence end, so a naive splitter cuts
#: a cited sentence in two (the "vs." case: "Luo et al. (2025) found ChatGPT's
#: precision exceeded Grammarly's (94-98% vs. 85%) ..." would otherwise be extracted
#: as two claims, the first ending mid-clause at "vs.").
ABBREVIATIONS_NOT_SENTENCE_FINAL = frozenset({"vs"})

#: Mirrors ``app.services.fulltext._CLAIM_SENTENCE_SPLIT_RE``: split on a sentence-
#: ending punctuation mark followed by whitespace, unless what follows is a
#: parenthesised year or year-placeholder -- the one shape a narrative citation's own
#: "et al. (2020)" ever takes, which must not be split apart from its year.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?!\(\s*(?:\d{4}|n\.d\.|in\s+press))")

#: Mirrors ``app.services.fulltext._FINALIZE_BOLD_HEADING_BLOCK_RE`` /
#: ``app.services.writing._MARKDOWN_BOLD_HEADING_RE``: a whole sentence that is
#: nothing but one bold run, start to end. The writer's other way of marking a
#: sub-heading, one `_build_section_tiptap_nodes` promotes to a real heading node
#: whenever it is the WHOLE of its own block AND it still reads as a sub-heading
#: (`_is_bold_heading_sentence`) rather than a claim sentence; a
#: citation-link call given no structural context about the text can still tag that
#: narrower shape "finding" or leave it unclassified, so rule 1 must never report it as
#: a coverage defect, exactly as the backend's own finalize rules now never touch it.
_BOLD_HEADING_SENTENCE_RE = re.compile(r"^\*\*([^*]+)\*\*$")

#: Mirrors ``app.services.fulltext._CITATION_RE`` (duplicated, not imported -- see the
#: module docstring): an author-year citation in brackets or parentheses, used only by
#: `_is_bold_heading_sentence` below to tell a bolded claim from a genuine sub-heading.
_CITATION_RE = re.compile(
    r"[\[\(]"
    r"([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-z]+))?)"
    r",?\s*"
    r"(\d{4})"
    r"[\]\)]"
)

#: Mirrors ``app.services.fulltext._SENTENCE_TERMINAL_PUNCTUATION``.
_SENTENCE_TERMINAL_PUNCTUATION = (".", "!", "?")


def _is_bold_heading_sentence(sentence: str) -> bool:
    """True when *sentence* is a whole-line bold run that still reads as a sub-heading
    rather than a claim sentence the writer happened to bold (mirrors
    ``app.services.fulltext._is_bold_heading_candidate``): it must match
    `_BOLD_HEADING_SENTENCE_RE`, and its own inner wording must not end in
    `_SENTENCE_TERMINAL_PUNCTUATION` or carry a citation (`_CITATION_RE`). Without this
    narrowing, a bolded claim -- now correctly rendered as body prose by the backend's
    own fix -- would still be silently exempted from rule 1 by this checker alone."""
    match = _BOLD_HEADING_SENTENCE_RE.match(sentence)
    if not match:
        return False
    inner = match.group(1).strip()
    if not inner or inner.endswith(_SENTENCE_TERMINAL_PUNCTUATION):
        return False
    return not _CITATION_RE.search(inner)


def _blocked_by_abbreviation(text: str, split_start: int) -> bool:
    """True when the word right before *split_start* is one of
    `ABBREVIATIONS_NOT_SENTENCE_FINAL` plus its own period, so the split point
    `_SENTENCE_SPLIT_RE` found there is a false sentence boundary, not a real one."""
    word = re.search(r"([A-Za-z]+)\.$", text[:split_start])
    return word is not None and word.group(1).lower() in ABBREVIATIONS_NOT_SENTENCE_FINAL


def split_sentences(text: str) -> list[str]:
    """*text* cut into sentences on the same boundaries the backend's own claim
    extractor uses (see the two constants above): a real sentence boundary is
    ``[.!?]`` followed by whitespace, except right after one of
    `ABBREVIATIONS_NOT_SENTENCE_FINAL` or right before a narrative citation's own
    parenthesised year. Empty fragments (leading/trailing whitespace, or two
    boundaries back to back) are dropped."""
    sentences: list[str] = []
    cursor = 0
    for match in _SENTENCE_SPLIT_RE.finditer(text):
        if _blocked_by_abbreviation(text, match.start()):
            continue
        piece = text[cursor : match.start()].strip()
        if piece:
            sentences.append(piece)
        cursor = match.end()
    tail = text[cursor:].strip()
    if tail:
        sentences.append(tail)
    return sentences


#: A single letter directly followed by a period at the very end of the text checked:
#: the shape "U.S.", "a.m." or "e.g." take right where a real sentence boundary would
#: also be legal, which `ABBREVIATIONS_NOT_SENTENCE_FINAL`'s own
#: fixed word list cannot enumerate exhaustively. Used only to suppress rule 6's own
#: lower-case-start signal on the FRAGMENT THAT FOLLOWS such a split -- not to suppress
#: the split itself, since nothing here can safely rejoin the two fragments back
#: together -- because a fragment opening lower-case right after this shape is exactly
#: what a genuine abbreviation boundary looks like, not a sign of a wrongly split
#: sentence.
_SINGLE_LETTER_ABBREVIATION_RE = re.compile(r"\b[A-Za-z]\.$")


def _split_point_is_abbreviation_shaped(text: str, split_start: int) -> bool:
    """True when the text immediately before *split_start* -- one of
    `_SENTENCE_SPLIT_RE`'s own split points -- ends in a single letter and a period."""
    return bool(_SINGLE_LETTER_ABBREVIATION_RE.search(text[:split_start]))


def _split_sentences_with_abbreviation_flags(text: str) -> list[tuple[str, bool]]:
    """Like `split_sentences`, but pairs each fragment with whether the split point
    immediately BEFORE it (if any) was abbreviation-shaped
    (`_split_point_is_abbreviation_shaped`) -- the first fragment of *text* is never
    flagged, since nothing precedes it. Kept separate from
    `split_sentences` itself so every other caller of that function (rules 1 and 4) is
    unaffected; only `check_truncated_sentences` (rule 6) needs this extra flag."""
    results: list[tuple[str, bool]] = []
    cursor = 0
    preceded_by_abbreviation = False
    for match in _SENTENCE_SPLIT_RE.finditer(text):
        if _blocked_by_abbreviation(text, match.start()):
            continue
        piece = text[cursor : match.start()].strip()
        if piece:
            results.append((piece, preceded_by_abbreviation))
        cursor = match.end()
        preceded_by_abbreviation = _split_point_is_abbreviation_shaped(text, match.start())
    tail = text[cursor:].strip()
    if tail:
        results.append((tail, preceded_by_abbreviation))
    return results


def _ends_at_abbreviation_reason(sentence: str) -> str | None:
    """Half of rule 6: *sentence* ends with a period directly after a token
    `ABBREVIATIONS_NOT_SENTENCE_FINAL` names, so the real sentence continues past it
    (the "vs." case). Safe to apply to any text a reader expects to end at a real
    sentence boundary, whether or not it is a full sentence in its own right, since
    the pattern is narrow and specific."""
    tail = re.search(r"([A-Za-z]+)\.\s*$", sentence)
    if tail is not None and tail.group(1).lower() in ABBREVIATIONS_NOT_SENTENCE_FINAL:
        return f"ends with {tail.group(0).strip()!r}, an abbreviation, not a real sentence end"
    return None


#: Every character `_prefix_is_legitimate_numeral_opening` accepts in a prefix that
#: is nothing but a bare number or percentage before the sentence's own first word
#: ("88% ", "3.5%, ", "139 "): digits, a decimal point, a thousands comma, a percent
#: sign, and whitespace. A digit is not a lower-case letter, and neither is a
#: percent sign -- a sentence that opens with one is not a fragment merely because
#: the first real WORD after it happens to be a lower-case preposition ("88% of 33
#: students ...").
_NUMERAL_PREFIX_CHARS = frozenset("0123456789.,% \t")

#: Quotation marks (straight and the curly opening pair) a genuine quoted opening can
#: start with, checked as the first non-space character of the prefix.
_OPENING_QUOTE_CHARS = "\"'‘“"

#: Opening bracket characters a genuine parenthesised-citation opening can start
#: with, checked the same way.
_OPENING_BRACKET_CHARS = "([‘“"


def _prefix_has_unmatched_closer(prefix: str) -> bool:
    """True when *prefix* -- everything before a sentence's own first alphabetic
    character -- carries a ``)`` or ``]`` with no ``(`` or ``[`` earlier in the same
    prefix to balance it: the tell that this text was cut out of the MIDDLE of a
    parenthetical rather than opening one of its own. "85%), but recall rose ..."
    carries exactly this shape -- a lone ``)`` with nothing before it to close --
    which is why it still reads as a fragment even though it also opens with a
    digit; "88% of 33 students ..." carries no bracket at all and is not caught
    here."""
    paren_depth = 0
    bracket_depth = 0
    for char in prefix:
        if char == "(":
            paren_depth += 1
        elif char == ")":
            if paren_depth > 0:
                paren_depth -= 1
            else:
                return True
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            if bracket_depth > 0:
                bracket_depth -= 1
            else:
                return True
    return False


def _prefix_is_legitimate_opening(prefix: str) -> bool:
    """True when *prefix* -- non-empty, and already cleared of a stray unmatched
    closing bracket by `_prefix_has_unmatched_closer` -- is one of the three
    legitimate ways a sentence can open before its own first word: a bare number or
    percentage (`_NUMERAL_PREFIX_CHARS`), an opening parenthesis or bracket (a
    parenthesised citation leading the sentence), or a quotation mark."""
    if all(char in _NUMERAL_PREFIX_CHARS for char in prefix):
        return True
    first_visible = prefix.lstrip()[:1]
    return first_visible in _OPENING_BRACKET_CHARS or first_visible in _OPENING_QUOTE_CHARS


def _begins_with_lowercase_fragment_reason(sentence: str) -> str | None:
    """The other half of rule 6: *sentence* opens with a lower-case letter rather than
    a fresh clause, as the second half of a wrongly split sentence would (for
    example: "85%), but recall rose only from 10% to 55% ..."). Only meaningful for
    text that is meant to stand as a full sentence on its
    own -- a draft body sentence, or a final report row's own ``claim_sentence`` -- not
    for a row's ``claim_text``, which is routinely a narrower proposition extracted
    from the middle of a sentence and legitimately opens lower-case (e.g. "found that
    ...", "showed that ..."; see `check_truncated_sentences`).

    Judged on the sentence's own first LETTER, not its first character: a fragment
    left over from a false split, such as "85%), but recall rose ...", opens with
    digits and punctuation carried over from the clause it was cut out of, and the
    tell is the case of the first real word after them, not of the digits (which have
    no case of their own) -- UNLESS what precedes that first letter is itself a
    legitimate opening a real sentence can have and nothing else: a bare digit or
    percentage ("88% of 33 students submitted ...", a grammatical, complete
    sentence), a parenthesised citation, or a quotation mark
    (`_prefix_is_legitimate_opening`). A prefix carrying a stray, unmatched closing
    bracket is never exempted this way even when it also looks like a number or a
    percentage -- ``")"`` with nothing before it to close is exactly the tell of a
    fragment cut out of the middle of a parenthetical
    (`_prefix_has_unmatched_closer`), which "88% of ..." does not carry and "85%),
    but recall rose ..." still does."""
    first_letter = re.search(r"[A-Za-z]", sentence)
    if first_letter is None or not first_letter.group(0).islower():
        return None
    prefix = sentence[: first_letter.start()]
    if prefix and not _prefix_has_unmatched_closer(prefix) and _prefix_is_legitimate_opening(prefix):
        return None
    return "begins with a lower-case letter, as a fragment cut from a previous sentence would"


def truncation_reasons(sentence: str, *, check_lowercase_start: bool = True) -> list[str]:
    """Rule 6: why *sentence* reads like a fragment cut from a longer sentence, or an
    empty list when it does not. Two independent, cheap signals -- this is not a
    grammar checker. ``check_lowercase_start=False`` (`check_truncated_sentences`'s own
    ``claim_text`` case) skips `_begins_with_lowercase_fragment_reason`."""
    reasons: list[str] = []
    abbreviation_reason = _ends_at_abbreviation_reason(sentence)
    if abbreviation_reason:
        reasons.append(abbreviation_reason)
    if check_lowercase_start:
        lowercase_reason = _begins_with_lowercase_fragment_reason(sentence)
        if lowercase_reason:
            reasons.append(lowercase_reason)
    return reasons


# ---------------------------------------------------------------------------
# Draft (Tiptap document) helpers
# ---------------------------------------------------------------------------


def _node_text(node: Mapping[str, Any]) -> str:
    return "".join(
        child.get("text") or ""
        for child in node.get("content") or []
        if isinstance(child, dict) and child.get("type") == "text"
    )


def draft_paragraph_texts(content: Mapping[str, Any] | None) -> list[str]:
    """Every paragraph node's own concatenated text, in document order (mirrors
    ``run_demo._draft_paragraph_texts``)."""
    return [
        _node_text(node)
        for node in (content or {}).get("content") or []
        if isinstance(node, dict) and node.get("type") == "paragraph"
    ]


def draft_heading_texts(content: Mapping[str, Any] | None) -> list[str]:
    """Every heading node's own concatenated text, in document order (mirrors
    ``run_demo._draft_heading_texts``)."""
    return [
        _node_text(node)
        for node in (content or {}).get("content") or []
        if isinstance(node, dict) and node.get("type") == "heading"
    ]


#: A bracket or parenthesis group containing a plausible year (1900-2099, an optional
#: trailing letter for "2020a"), with up to 120 characters of author names, "et al.",
#: "&"/"and", semicolons and page locators before it and up to 20 after (`n.d.`/`in
#: press` are not matched -- neither ever appears in this corpus). Deliberately not a
#: reimplementation of `_CITATION_RE`'s own author-name shape (`[A-Z][a-z]+`, ASCII
#: only): this corpus cites authors named "López" and "Hébert", and a
#: bracket/paren group that carries a year at all is, in academic prose, a citation
#: regardless of what the author name looks like -- narrower author-shape matching is
#: for `_CITATION_RE`'s own, different job (telling a bolded claim from a bolded
#: sub-heading), not for this module's rule 8, which only ever needs "does this
#: sentence cite something at all". Also matches the bare-year narrative style ("Zhang
#: and Hyland (2021) found...", where the parenthesis holds only the year) since the
#: author name sits outside the bracket entirely on that shape.
_BRACKETED_YEAR_RE = re.compile(
    r"[\[\(][^\[\]\(\)]{0,120}\b(?:19|20)\d{2}[a-z]?\b[^\[\]\(\)]{0,20}[\]\)]"
)

#: A numbered-citation group such as "[3]" or "[3, 7]" -- the other citation shape a
#: delivered paragraph renders, with no year inside the brackets at all.
_NUMBERED_CITATION_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")

#: Mirrors ``app.services.fulltext._FINALIZE_HEADING_BLOCK_RE``: a whole paragraph
#: node whose own text is nothing but a markdown heading line ("## Direct Corrections
#: ..."), which an older saved run can carry as a plain "paragraph" Tiptap node
#: rather than a real "heading" node. `_paragraph_descriptors` treats such a node
#: exactly like a real heading node -- excluded from rule 8a's own paragraph count --
#: so a historical run's own restated markdown title is never itself reported as "a
#: body paragraph with no citation".
_MARKDOWN_HEADING_LINE_RE = re.compile(r"^#{1,6}[ \t]+")


def _paragraph_is_heading_shaped(text: str) -> bool:
    """True when *text* is really a heading, not a body paragraph, in either of the
    two forms a saved draft has used: a literal markdown "#" line
    (`_MARKDOWN_HEADING_LINE_RE`), or a whole-line bold run that still reads as a
    sub-heading rather than a claim sentence (`_is_bold_heading_sentence`, already
    used the same way by rule 1).

    Does not reuse `is_heading_shaped_paragraph` (rule 16's own predicate): the two
    answer different questions. This one asks whether *text* still carries markdown
    or bold heading SYNTAX -- a shape only an older saved run's own stored node can
    have, from before this product promoted such a block to a real heading node --
    and is used to exempt that node from rules 1 and 8a, which must not mistake it
    for an ordinary body paragraph. `is_heading_shaped_paragraph` instead asks
    whether *text* reads as a heading by its own CONTENT alone, with no marker at
    all -- the shape a leading-heading fragment takes once its marker really is
    gone (the fixed leading-heading strip in `app.services.fulltext` and
    `app.services.writing` now keeps a genuine markdown heading marked instead of
    ever demoting it, so this content-shape check is what still catches the marker-
    less remainder). A node this function accepts already reads as a heading to a
    reader on its own markup; rule 16 does not need to, and should not, exempt it
    the way rules 1/8a/15 do, since a heading-shaped paragraph is exactly what rule
    16 exists to report, whatever left it that way."""
    stripped = text.strip()
    return bool(_MARKDOWN_HEADING_LINE_RE.match(stripped)) or _is_bold_heading_sentence(stripped)


def _paragraph_has_rendered_citation(text: str) -> bool:
    """Used by `_paragraph_descriptors` (rule 8a) and `check_dangling_framing` (rules
    8b/8c, to tell a cited sentence from an uncited framing one) alike: whether *text*
    carries a citation at all, independently of whether a historical run's own
    ``attrs.citationLinks`` is present or trustworthy."""
    return bool(_BRACKETED_YEAR_RE.search(text) or _NUMBERED_CITATION_RE.search(text))


@dataclass(frozen=True)
class ParagraphDescriptor:
    """One body paragraph of a saved draft, as `check_dangling_framing` and
    `check_paragraph_has_citation` need it: ``index`` is this paragraph's own
    position among paragraphs only (the same numbering
    `check_body_sentence_classification` already reports), ``text`` is its rendered
    text, ``has_citation`` is whether it carries a citation at all (its own
    ``attrs.citationLinks``, when present and non-empty, or a rendered citation pattern
    found directly in its text -- either is enough, since a historical run's own
    ``citationLinks`` attr is not always present), and ``has_unclassified`` is whether
    this paragraph's own ``attrs.uncitedSentences`` carries an entry marked
    ``"unclassified": True`` (a citation-link retry that failed, or never mentioned the
    sentence): the guarantee that such a sentence is never silently dropped outranks
    rule 8a here too (`check_paragraph_has_citation`), mirroring
    `app.services.fulltext.finalize_generated_section`'s own exemption."""

    index: int
    text: str
    has_citation: bool
    has_unclassified: bool


def _paragraph_descriptors(content: Mapping[str, Any] | None) -> list[ParagraphDescriptor]:
    descriptors: list[ParagraphDescriptor] = []
    index = 0
    for node in (content or {}).get("content") or []:
        if not isinstance(node, dict):
            continue
        node_type = node.get("type")
        if node_type != "paragraph":
            continue
        text = _node_text(node)
        if _paragraph_is_heading_shaped(text):
            continue
        attrs = node.get("attrs") or {}
        has_citation = bool(attrs.get("citationLinks")) or _paragraph_has_rendered_citation(text)
        has_unclassified = any(
            entry.get("unclassified") for entry in attrs.get("uncitedSentences") or []
        )
        descriptors.append(
            ParagraphDescriptor(
                index=index,
                text=text,
                has_citation=has_citation,
                has_unclassified=has_unclassified,
            )
        )
        index += 1
    return descriptors


# ---------------------------------------------------------------------------
# Rule 5: the verifier guard's own letters-and-digits fold
# ---------------------------------------------------------------------------


def _keep_letters_and_digits(text: str) -> str:
    """*text* folded the way the verification guard folds it for its own verbatim
    check (mirrors ``run_demo._keep_letters_and_digits`` /
    ``app.services.fulltext._normalise_for_match``): NFKC, then letters and digits
    only, case-folded. Every dash, hyphen, quote mark and stray control byte vanishes,
    so a quote that differs from its source only in typography still matches."""
    kept: list[str] = []
    for character in text:
        for folded_char in unicodedata.normalize("NFKC", character):
            if folded_char.isalpha() or folded_char.isdigit():
                kept.append(folded_char.casefold())
    return "".join(kept)


def is_verbatim_under_guard_fold(quote: str, passage: str) -> bool:
    """True when *quote* is a verbatim substring of *passage*, either literally or
    under the guard's own letters-and-digits fold (see `_keep_letters_and_digits`)."""
    if not quote or not passage:
        return False
    if quote in passage:
        return True
    return _keep_letters_and_digits(quote) in _keep_letters_and_digits(passage)


# ---------------------------------------------------------------------------
# Per-section rule checks (rules 1-4, 6-9)
# ---------------------------------------------------------------------------


def check_duplicate_leading_heading(
    draft_content: Mapping[str, Any] | None, section: str
) -> list[Violation]:
    """Rule 3, two parts.

    Part 1 compares only ADJACENT heading nodes among the heading-only list (mirrors
    ``run_demo.assert_no_duplicate_leading_heading``), never every pair, since a
    legitimate sub-heading later in the body may happen to repeat the section title in
    wording without being a duplicate of the opening heading. Case-insensitive, like
    ``run_demo._norm``, the comparison it mirrors.

    Part 2 catches the shape part 1 misses: the section's own raw first two content
    nodes both being heading nodes, WHATEVER their wording. Every section's own draft
    is written to its own document (`_write_one_extra_section` creates one draft per
    section), so node 0 is always that section's own injected H2 heading
    (`_build_section_tiptap_nodes`); a second heading node immediately after it is
    always the model's own restated (possibly differently worded) title that
    `writing._strip_duplicate_leading_heading` failed to recognise as a duplicate --
    never a legitimate first body block, which is always a paragraph.
    Skipped when part 1 already reported the same pair (identical text), so a genuine
    duplicate is never counted twice."""
    headings = draft_heading_texts(draft_content)
    violations = []
    for first, second in zip(headings, headings[1:]):
        first_norm = normalize_sentence(first).lower()
        if first_norm and first_norm == normalize_sentence(second).lower():
            violations.append(
                Violation(
                    RULE_DUPLICATE_HEADING,
                    section,
                    f"two consecutive headings carry the same text: {first!r}",
                )
            )

    nodes = (draft_content or {}).get("content") or []
    if (
        len(nodes) >= 2
        and isinstance(nodes[0], dict) and nodes[0].get("type") == "heading"
        and isinstance(nodes[1], dict) and nodes[1].get("type") == "heading"
    ):
        first_text = _node_text(nodes[0])
        second_text = _node_text(nodes[1])
        already_reported = normalize_sentence(first_text).lower() == normalize_sentence(
            second_text
        ).lower()
        if not already_reported:
            violations.append(
                Violation(
                    RULE_DUPLICATE_HEADING,
                    section,
                    "the section opens with two consecutive heading nodes: "
                    f"{first_text!r} then {second_text!r}",
                )
            )
    return violations


#: Mirrors ``app.services.fulltext._TITLE_ECHO_STOPWORDS`` (independent copy -- this
#: module never imports the backend, see the module docstring): kept identical by the
#: shared fixture both sides' own tests read, ``demo/fixtures/title_echo_cases.json``.
_TITLE_ECHO_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "with", "for", "and", "or", "to", "at",
    "by", "from", "into", "onto", "as", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "their", "its", "it",
})

_TITLE_ECHO_WORD_RE = re.compile(r"[0-9A-Za-zÀ-ɏ]+")

#: Mirrors ``app.services.fulltext._TITLE_ECHO_SUFFIXES``.
_TITLE_ECHO_SUFFIXES = ("ing", "ed", "es", "s")


def _title_echo_stem(word: str) -> str:
    """Mirrors ``app.services.fulltext._title_echo_stem``."""
    for suffix in _TITLE_ECHO_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _title_echo_words(text: str) -> set[str]:
    """Mirrors ``app.services.fulltext._title_echo_words``."""
    return {
        _title_echo_stem(word)
        for word in _TITLE_ECHO_WORD_RE.findall((text or "").lower())
        if word not in _TITLE_ECHO_STOPWORDS
    }


def is_title_echo(paragraph_text: str, reference_texts: Sequence[str]) -> bool:
    """Mirrors ``app.services.fulltext._is_title_echo`` -- see that function's own
    docstring for the containment-coefficient rule, the word-count guard, and the
    two-shared-word floor. Public (no leading underscore) here, unlike its app-side
    twin, since this module's own tests import it directly to check it against the
    shared fixture."""
    paragraph_words = _title_echo_words(paragraph_text)
    if not paragraph_words or len(paragraph_words) > 12:
        return False
    for reference in reference_texts:
        reference_words = _title_echo_words(reference)
        if not reference_words:
            continue
        overlap = paragraph_words & reference_words
        smaller = min(len(paragraph_words), len(reference_words))
        if smaller and len(overlap) >= 2 and len(overlap) / smaller >= 0.5:
            return True
    return False


def check_title_echo_in_body(
    draft_content: Mapping[str, Any] | None, section: str
) -> list[Violation]:
    """New rule: a delivered body paragraph that echoes the section's own title (this
    section's own first heading node, always its injected H2 -- `check_duplicate_
    leading_heading`'s own part 2 docstring) or another heading still standing in its
    own draft (`is_title_echo`) is a violation: the section repeating its own title,
    or a model's own sub-heading demoted to plain text by finalize once its real body
    was removed, delivered as if it were a finding or framing sentence of its own.

    A paragraph node that is itself heading-shaped (`_paragraph_is_heading_shaped`,
    an older saved run's own restated markdown title, stored as a plain "paragraph"
    node from before this product promoted that shape to a real heading node) is
    never checked here either: it already reads as a heading to a reader, and rule
    8a's own `_paragraph_descriptors` already excludes it from "a body paragraph
    with no citation" for the identical reason."""
    headings = draft_heading_texts(draft_content)
    if not headings:
        return []
    violations = []
    for paragraph in draft_paragraph_texts(draft_content):
        if _paragraph_is_heading_shaped(paragraph):
            continue
        if is_title_echo(paragraph, headings):
            violations.append(
                Violation(
                    RULE_TITLE_ECHO_IN_BODY,
                    section,
                    "a delivered body paragraph echoes the section's own title or "
                    "one of its headings rather than stating a finding or framing "
                    "sentence of its own",
                    paragraph,
                )
            )
    return violations


def check_heading_shaped_paragraph(
    draft_content: Mapping[str, Any] | None, section: str
) -> list[Violation]:
    """Rule 16: a delivered body paragraph that is heading-shaped
    (`is_heading_shaped_paragraph`) is a violation, whatever its position -- unlike
    rule 15, this does not require sharing a word with any title or heading: a bare
    noun phrase such as "Comparative Scope and Accuracy" is caught by its own shape
    alone, the exact case rule 15 cannot reach because it shares no stem with either
    the section's title or its one standing heading."""
    violations = []
    for paragraph in draft_paragraph_texts(draft_content):
        if is_heading_shaped_paragraph(paragraph):
            violations.append(
                Violation(
                    RULE_HEADING_SHAPED_PARAGRAPH,
                    section,
                    "a delivered body paragraph reads as a heading rather than a "
                    "sentence: no sentence-final punctuation, at most ten words, "
                    "no citation, no finite verb",
                    paragraph,
                )
            )
    return violations


def check_needs_citation_marker(
    draft_content: Mapping[str, Any] | None, section: str
) -> list[Violation]:
    """Rule 2, part 1."""
    violations = []
    for text in draft_paragraph_texts(draft_content):
        if NEEDS_CITATION_RE.search(text):
            violations.append(
                Violation(
                    RULE_NEEDS_CITATION_MARKER,
                    section,
                    "a [NEEDS CITATION] marker survived in the saved draft",
                    text,
                )
            )
    return violations


def check_citation_coverage_resolved(
    claim_report: Mapping[str, Any] | None, section: str
) -> list[Violation]:
    """Rule 2, part 2. ``citation_coverage`` is ``None`` on a report from a build
    before this field existed, or on a plain-string report; there is nothing to check
    then (mirrors ``run_demo.assert_citation_coverage_fully_resolved``)."""
    coverage = (claim_report or {}).get("citation_coverage")
    if coverage is None:
        return []
    unresolved = coverage.get("unresolved") or 0
    if unresolved:
        return [
            Violation(
                RULE_UNRESOLVED_CITATION,
                section,
                f"{unresolved} citation(s) never resolved to a library paper "
                "(citation_coverage.unresolved != 0)",
            )
        ]
    return []


def _final_report_verifications(claim_report: Mapping[str, Any] | None) -> list[dict] | None:
    """``claim_report["final_report"]["verifications"]``, or ``None`` when
    *claim_report* carries no ``final_report`` at all (a build from before the
    standalone verify-and-heal action existed)."""
    final_report = (claim_report or {}).get("final_report")
    if final_report is None:
        return None
    return list(final_report.get("verifications") or [])


def check_empty_cited_section(
    writing_result: Mapping[str, Any] | None,
    claim_report: Mapping[str, Any] | None,
    section: str,
) -> list[Violation]:
    """New rule. A section this hollow passes every one of rules 1-13 in silence: an
    empty ``verifications`` list means rule 4 (`check_final_report_rows`) has no rows
    to check at all, and every other rule here only ever looks at text that survived
    -- none of them can see that nothing did. ``verifications`` being ``None`` (no
    ``final_report`` on this build at all) is not itself a violation of this rule;
    that gap is already reported by rule 4's own caller as a missing-file notice."""
    verifications = _final_report_verifications(claim_report)
    citation_audit = (writing_result or {}).get("citation_audit") or {}
    loop_stats = (writing_result or {}).get("loop_stats") or {}
    after_regeneration = bool(loop_stats.get("empty_section_after_regeneration"))
    empty_but_cited = (
        verifications is not None
        and not verifications
        and bool(citation_audit.get("total"))
    )
    if not empty_but_cited and not after_regeneration:
        return []
    detail = (
        "final_report.verifications is empty while citation_audit.total > 0"
        if empty_but_cited
        else "loop_stats.empty_section_after_regeneration is true"
    )
    return [
        Violation(
            RULE_EMPTY_CITED_SECTION,
            section,
            f"the section delivers no cited sentence at all ({detail})",
        )
    ]


def check_final_report_rows(
    draft_content: Mapping[str, Any] | None,
    claim_report: Mapping[str, Any] | None,
    section: str,
) -> list[Violation]:
    """Rule 4."""
    verifications = _final_report_verifications(claim_report)
    if verifications is None:
        return []
    paragraphs = draft_paragraph_texts(draft_content)
    violations = []
    for row in verifications:
        if row.get("status") != "verified":
            violations.append(
                Violation(
                    RULE_ROW_NOT_VERIFIED,
                    section,
                    f"final report row has status {row.get('status')!r}, not 'verified'",
                    row.get("claim_text"),
                )
            )
            continue
        claim_text = row.get("claim_text") or ""
        if claim_text and not any(claim_text in paragraph for paragraph in paragraphs):
            violations.append(
                Violation(
                    RULE_CLAIM_TEXT_NOT_IN_DRAFT,
                    section,
                    "claim_text does not occur verbatim in any paragraph of the saved draft",
                    claim_text,
                )
            )
    return violations


def check_truncated_sentences(
    draft_content: Mapping[str, Any] | None,
    claim_report: Mapping[str, Any] | None,
    section: str,
) -> list[Violation]:
    """Rule 6, over every draft body sentence and every final-report row's own
    ``claim_text``/``claim_sentence`` (the truncation the "vs." case actually shows up
    in: the draft's own paragraph text is untruncated prose, but the extractor's own
    recorded ``claim_text`` was cut at the false split point).

    A whole, correct sentence containing an internal abbreviation this module's
    splitter still cuts on (e.g. "U.S.", not in the fixed
    `ABBREVIATIONS_NOT_SENTENCE_FINAL` list) is split into two fragments; the SECOND
    fragment legitimately opens lower-case ("intensive English program, ...") and must
    not be reported as truncated on that account alone. The lower-case-start signal is
    therefore skipped for a fragment whose own preceding split point was abbreviation-
    shaped (`_split_sentences_with_abbreviation_flags`); the abbreviation-ending signal
    still applies to every fragment regardless."""
    violations = []
    for paragraph_index, paragraph in enumerate(draft_paragraph_texts(draft_content)):
        for sentence, preceded_by_abbreviation in _split_sentences_with_abbreviation_flags(
            paragraph
        ):
            for reason in truncation_reasons(
                sentence, check_lowercase_start=not preceded_by_abbreviation
            ):
                violations.append(
                    Violation(
                        RULE_TRUNCATED_SENTENCE,
                        section,
                        f"paragraph {paragraph_index}: {reason}",
                        sentence,
                    )
                )
    seen: set[str] = set()
    for row in _final_report_verifications(claim_report) or []:
        # ``claim_sentence`` (checked first, so a value shared with ``claim_text`` is
        # only ever checked under its stronger rule) is meant to be the full sentence,
        # so both signals apply. ``claim_text`` is routinely a narrower proposition
        # extracted from the middle of ``claim_sentence`` (e.g. "found that 60% of ...
        # preferred teacher feedback"), so it legitimately opens lower-case; only the
        # abbreviation-ending signal applies to it.
        for field_name, check_lowercase in (("claim_sentence", True), ("claim_text", False)):
            value = row.get(field_name)
            if not value or value in seen:
                continue
            seen.add(value)
            for reason in truncation_reasons(value, check_lowercase_start=check_lowercase):
                violations.append(
                    Violation(
                        RULE_TRUNCATED_SENTENCE,
                        section,
                        f"final report {field_name}: {reason}",
                        value,
                    )
                )
    return violations


def _draft_uncited_entries(draft_content: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Every uncited-sentence entry recorded on any paragraph node's own
    ``attrs.uncitedSentences``. ``writing_result.json`` is a
    one-time snapshot from generation; the saved draft is not -- the write job's own
    revision pass and the standalone verify-and-heal action both persist onto this same
    attribute on every later heal (`app.services.writing._build_section_tiptap_nodes`,
    `app.services.fulltext._close_citation_link_coverage_gap`). Reading only
    ``writing_result.json`` would report a sentence a later heal classified correctly as
    if it were still exactly what generation first left it as."""
    entries: list[dict[str, Any]] = []
    for node in (draft_content or {}).get("content") or []:
        if not isinstance(node, dict) or node.get("type") != "paragraph":
            continue
        attrs = node.get("attrs") or {}
        entries.extend(attrs.get("uncitedSentences") or [])
    return entries


def check_body_sentence_classification(
    draft_content: Mapping[str, Any] | None,
    writing_result: Mapping[str, Any] | None,
    claim_report: Mapping[str, Any] | None,
    section: str,
) -> tuple[list[Violation], list[str]]:
    """Rule 1. Returns ``(violations, notices)``: a notice, not a violation, when
    *writing_result* carries no ``uncited_sentences`` field at all (an older shape,
    from before that field existed) -- with no ground truth for "framing" prose at
    all, treating every uncited sentence as a violation would flood the report with
    false positives instead of naming a real defect, so this check is skipped and
    says so instead.

    ``writing_result``'s own ``uncited_sentences`` is unioned with every paragraph's own
    ``attrs.uncitedSentences`` on the saved draft (`_draft_uncited_entries`), so a
    sentence a later heal classifies -- after ``writing_result.json`` was already
    written -- is read from its fresher, healed classification, not reported as a
    stale defect.

    A sentence that is itself a whole-line bold heading (`_is_bold_heading_sentence`)
    is never reported: it is the writer's other way of marking a sub-heading,
    promoted by `_build_section_tiptap_nodes` to a real heading node whenever it is
    the whole of its own block AND it does not carry a citation or end in sentence
    punctuation, and a citation-link call given no structural context can still tag
    that narrower shape "finding" or leave it unclassified.

    A ``"finding"``-tagged entry still carrying ``"unclassified": True`` (a
    citation-link retry that failed or never mentioned the sentence, kept by finalize
    rather than removed) is reported under a distinct message from an ordinary,
    positively classified finding finalize should have removed: the two point a
    reader at different subsystems to fix."""
    notices: list[str] = []
    if writing_result is None:
        notices.append(f"{section}: writing_result.json not available; rule 1 not checked")
        return [], notices
    if "uncited_sentences" not in writing_result:
        notices.append(
            f"{section}: writing_result.json has no 'uncited_sentences' field "
            "(older shape, from before this field existed); rule 1 not checked"
        )
        return [], notices

    uncited = [
        *(writing_result.get("uncited_sentences") or []),
        *_draft_uncited_entries(draft_content),
    ]
    framing = {normalize_sentence(u.get("sentence")) for u in uncited if u.get("tag") == "framing"}
    finding = {
        normalize_sentence(u.get("sentence"))
        for u in uncited
        if u.get("tag") == "finding" and not u.get("unclassified")
    }
    finding_unclassified = {
        normalize_sentence(u.get("sentence"))
        for u in uncited
        if u.get("tag") == "finding" and u.get("unclassified")
    }
    final_sentences = [
        normalize_sentence(row.get("claim_sentence") or row.get("claim_text") or "")
        for row in (_final_report_verifications(claim_report) or [])
    ]
    final_sentences = [s for s in final_sentences if s]

    violations: list[Violation] = []
    for paragraph_index, paragraph in enumerate(draft_paragraph_texts(draft_content)):
        for sentence in split_sentences(paragraph):
            norm = normalize_sentence(sentence)
            if not norm:
                continue
            if _is_bold_heading_sentence(norm):
                continue
            if norm in framing:
                continue
            if any(norm in final or final in norm for final in final_sentences):
                continue
            if norm in finding:
                violations.append(
                    Violation(
                        RULE_UNCITED_FINDING_SURVIVED,
                        section,
                        f"paragraph {paragraph_index}: an uncited 'finding' sentence "
                        "survived into the delivered draft (finalize should have "
                        "removed it)",
                        sentence,
                    )
                )
                continue
            if norm in finding_unclassified:
                violations.append(
                    Violation(
                        RULE_UNCITED_FINDING_SURVIVED,
                        section,
                        f"paragraph {paragraph_index}: an uncited sentence survived into "
                        "the delivered draft because a citation-link retry failed or "
                        "never classified it (not a case finalize should have removed)",
                        sentence,
                    )
                )
                continue
            violations.append(
                Violation(
                    RULE_UNCLASSIFIED_SENTENCE,
                    section,
                    f"paragraph {paragraph_index}: sentence is neither a verified cited "
                    "sentence of the final report nor tagged 'framing' by the citation-"
                    "link call -- it was never classified at all",
                    sentence,
                )
            )
    return violations, notices


# ---------------------------------------------------------------------------
# Rules 10 and 11: a delivered cited sentence asserts nothing beyond what its own
# verified claims cover, and carries no comparative meta-evaluation of the
# literature.
#
# This is an independent, from-scratch copy of the residue/frame classifier in
# `app.services.sentence_coverage` -- this module never imports the backend (module
# docstring). The two independent copies are kept in step by the shared fixture
# `demo/fixtures/frame_spans.json`: both sides' own tests read it and must classify
# every span the same way.
# ---------------------------------------------------------------------------

_RESIDUE_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ɏ]+")


def _residue_fold(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return folded.casefold()


def _residue_tokens_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [(_residue_fold(m.group(0)), m.start(), m.end()) for m in _RESIDUE_TOKEN_RE.finditer(text)]


def _residue_token_list(text: str) -> list[str]:
    return [t for t, _s, _e in _residue_tokens_with_spans(text)]


def _residue_singular(tok: str) -> str:
    for suffix in ("ies", "es", "s"):
        if len(tok) > 4 and tok.endswith(suffix):
            return tok[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return tok


_RESIDUE_STOP = frozenset(
    """
a an the this that these those such its their his her our your my it they them we
and or but nor so as if then than thus also both either neither
of in on at to for with within without from by about into onto over under across
among between through during after before while whereas though although yet however
is are was were be been being has have had do does did not no nor only just
which who whom whose what when where why how there here s t d ll re ve
per via each any all more less most least own same other another
""".split()
)

#: Kept identical, word for word, to `app.services.sentence_coverage.ATTRIB`
#: -- see that lexicon's own docstring for why `caution` (and its four inflections)
#: and the seven other reporting verbs below were added, and why `suggest`,
#: `indicate`, `highlight`, `confirm` and `reveal` deliberately were not (each is
#: already in `_RESIDUE_NEVER_FRAME` below, by design). `concede`'s own four
#: inflections are added the same way, identically.
_RESIDUE_ATTRIB = frozenset(
    """
study studies research researcher researchers paper papers article articles work works
review reviews synthesis author authors et al
add adds added note notes noted report reports reported find finds found
show shows showed shown document documents documented examine examines examined
demonstrate demonstrates demonstrated observe observes observed argue argues argued
conclude concludes concluded state states stated describe describes described
write writes wrote reporting according similar similarly likewise same
caution cautions cautioned cautioning warn warns warned warning
propose proposes proposed proposing contend contends contended contending
maintain maintains maintained maintaining emphasise emphasises emphasised emphasising
emphasize emphasizes emphasized emphasizing stress stresses stressed stressing
acknowledge acknowledges acknowledged acknowledging
concede concedes conceded conceding
""".split()
)

_RESIDUE_NEVER_FRAME = frozenset(
    """
suggest suggests suggesting suggested imply implies implying implied
indicate indicates indicating indicated underscore underscores underscoring
highlight highlights highlighting confirm confirms confirming
prove proves proving proven reveal reveals revealing revealed establish establishes
established establishing adjudicate adjudicates
strongest clearest best greatest weakest definitive conclusive foremost primary
""".split()
)

#: The one shared predicate with `app.services.sentence_coverage`'s own copy
#: (`COORDINATING_CONJUNCTIONS`, same name and value there): a residue whose own
#: last token is one of these is never the bare grammatical subject the F1
#: branch below permits.
_RESIDUE_COORDINATING_CONJUNCTIONS = frozenset({"and", "or", "but", "yet", "so", "nor"})


def _ends_in_coordinating_conjunction(tokens: list[str]) -> bool:
    """True when *tokens* (a residue's own token list) ends in a coordinating
    conjunction. The checker's own copy of the identical predicate
    `app.services.sentence_coverage.ends_in_coordinating_conjunction` applies in the
    app's own gate."""
    return bool(tokens) and tokens[-1] in _RESIDUE_COORDINATING_CONJUNCTIONS


#: The one shared predicate with `app.services.sentence_coverage`'s own
#: copy (`SUBORDINATORS`, same name and value there): a residue whose own last token
#: is one of these is a dependent clause the writer is attaching to what follows, never
#: the bare grammatical subject the F1 branch below permits.
_RESIDUE_SUBORDINATORS = frozenset(
    """
because although while whereas since if unless until when as that which who whose
where
""".split()
)


def _ends_in_subordinator(tokens: list[str]) -> bool:
    """True when *tokens* (a residue's own token list) ends in a subordinating
    conjunction or subordinator. The checker's own copy of the identical predicate
    `app.services.sentence_coverage.ends_in_subordinator` applies in the app's own
    gate."""
    return bool(tokens) and tokens[-1] in _RESIDUE_SUBORDINATORS


_RESIDUE_NUMBER_WORDS = frozenset(
    """
one two three four five six seven eight nine ten eleven twelve thirteen fourteen
fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy
eighty ninety hundred thousand half dozen
""".split()
)

#: Mirrors ``app.services.sentence_coverage.NUMBER_WORD_VALUES`` -- see that
#: mapping's own docstring for why "half" is excluded.
_RESIDUE_NUMBER_WORD_VALUES: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "thousand": 1000, "dozen": 12,
}

_RESIDUE_DIGIT_RUN_RE = re.compile(r"\d+(?:\.\d+)?")


def _residue_numeral_key(value: float) -> str:
    """Mirrors ``app.services.sentence_coverage._numeral_key``."""
    return str(int(value)) if value == int(value) else str(value)


def numeral_keys_in_text(text: str) -> set[str]:
    """Mirrors ``app.services.sentence_coverage.numeral_keys_in_text``."""
    keys: set[str] = set()
    for match in _RESIDUE_DIGIT_RUN_RE.finditer(text or ""):
        try:
            keys.add(_residue_numeral_key(float(match.group(0))))
        except ValueError:
            continue
    for word in _residue_token_list(text or ""):
        value = _RESIDUE_NUMBER_WORD_VALUES.get(word)
        if value is not None:
            keys.add(_residue_numeral_key(value))
    return keys


def covered_numeral_keys(evidence_quotes: Sequence[str] | None) -> frozenset[str]:
    """Mirrors ``app.services.sentence_coverage.covered_numeral_keys``."""
    keys: set[str] = set()
    for quote in evidence_quotes or []:
        keys.update(numeral_keys_in_text(quote))
    return frozenset(keys)

_RESIDUE_QUANTIFIERS = frozenset(
    """
most least few fewer many much several majority minority all none some only
significantly substantially dramatically considerably
""".split()
)

#: Kept identical, word for word, to `app.services.sentence_coverage.PREP_OPENERS`
#: -- see that lexicon's own docstring for why "regarding" and "concerning" were
#: added, and why "with respect to" was not.
_RESIDUE_PREP_OPENERS = frozenset(
    """
on in by for with within across beyond alongside among at from after before
despite during through under over unlike like besides against beside
regarding concerning
""".split()
)

#: Kept identical, word for word, to `app.services.sentence_coverage.
#: DISCOURSE_OPENERS` -- see that lexicon's own docstring for why "where" and
#: "notably" were added.
_RESIDUE_DISCOURSE_OPENERS = frozenset(
    """
however moreover furthermore similarly likewise conversely nevertheless nonetheless
meanwhile overall together taken additionally also first second third fourth finally
thus therefore hence consequently accordingly instead again indeed still where notably
""".split()
)

_RESIDUE_FRAME_BUDGET = 3

_RESIDUE_CITATION_RE = re.compile(
    r"[\[\(][^\[\]\(\)]{0,120}\b(?:19|20)\d{2}[a-z]?\b[^\[\]\(\)]{0,20}[\]\)]"
)
_RESIDUE_NARRATIVE_CITATION_RE = re.compile(
    r"(?:[A-ZÀ-ɏ][\w'’\-]*\.?[\s,]*(?:and|&|et\s+al\.)?[\s,]*){1,6}"
    r"\(\s*(?:19|20)\d{2}[a-z]?\s*\)"
)

#: Kept identical, word for word, to `app.services.sentence_coverage.
#: _ING_REPORTING_VERBS` -- the present-participle/gerund ("-ing") subset of
#: `ATTRIB`/`REPORTING_VERBS` there (and, identically, of `_RESIDUE_ATTRIB` above).
_RESIDUE_ING_REPORTING_VERBS = frozenset(
    """
cautioning warning proposing contending maintaining emphasising emphasizing
stressing acknowledging conceding
""".split()
)


def _opens_with_participle_after_attribution(sentence: str) -> bool:
    """Rule 12's own predicate. The checker's independent copy of
    `app.services.sentence_coverage.opens_with_participle_after_attribution` -- see
    that function's own docstring; kept identical here in shape and value: True when
    *sentence* opens with a citation-shaped subject (a narrative "Author (Year)"
    attribution, or a bracketed "(Author, Year)" one) immediately followed by a
    present-participle/gerund ("-ing") reporting-verb form, with no finite verb
    anywhere before it."""
    stripped = (sentence or "").lstrip()
    match = _RESIDUE_NARRATIVE_CITATION_RE.match(stripped) or _RESIDUE_CITATION_RE.match(
        stripped
    )
    if not match:
        return False
    next_word = re.match(r"\s+([A-Za-zÀ-ɏ]+)", stripped[match.end() :])
    if not next_word:
        return False
    return _residue_fold(next_word.group(1)) in _RESIDUE_ING_REPORTING_VERBS


#: Kept identical, word for word, to `app.services.sentence_coverage.
#: REPORTING_VERBS` -- the verb-shaped subset of `ATTRIB`/`_RESIDUE_ATTRIB` above
#: (a reporting verb, as opposed to a literature noun or an adverb/conjunction
#: also carried by that lexicon).
_RESIDUE_REPORTING_VERBS = frozenset(
    """
add adds added note notes noted report reports reported find finds found
show shows showed shown document documents documented examine examines examined
demonstrate demonstrates demonstrated observe observes observed argue argues argued
conclude concludes concluded state states stated describe describes described
write writes wrote
caution cautions cautioned cautioning warn warns warned warning
propose proposes proposed proposing contend contends contended contending
maintain maintains maintained maintaining emphasise emphasises emphasised emphasising
emphasize emphasizes emphasized emphasizing stress stresses stressed stressing
acknowledge acknowledges acknowledged acknowledging
concede concedes conceded conceding
""".split()
)

#: Kept identical, word for word, to `app.services.sentence_coverage.
#: _REPORTING_VERB_BASE_FORMS` -- the base (dictionary) form of every verb
#: `_RESIDUE_REPORTING_VERBS` carries, used only to derive
#: `_RESIDUE_PAST_OR_THIRD_PERSON_REPORTING_VERBS`, below, by subtraction.
_RESIDUE_REPORTING_VERB_BASE_FORMS = frozenset(
    """
add note report find show document examine demonstrate observe argue conclude
state describe write caution warn propose contend maintain emphasise emphasize
stress acknowledge concede
""".split()
)

#: Kept identical, word for word, to `app.services.sentence_coverage.
#: _PAST_OR_THIRD_PERSON_REPORTING_VERBS` -- every inflection of
#: `_RESIDUE_REPORTING_VERBS` that is neither a base form nor a
#: present-participle/gerund, the shape a lead-in whose own subject has been cut
#: away leaves behind ("showed", "shows", "found", "reports", "noted", ...).
_RESIDUE_PAST_OR_THIRD_PERSON_REPORTING_VERBS = frozenset(
    v
    for v in _RESIDUE_REPORTING_VERBS
    if v not in _RESIDUE_REPORTING_VERB_BASE_FORMS and v not in _RESIDUE_ING_REPORTING_VERBS
)


def _opens_with_reporting_verb(sentence: str) -> bool:
    """Rule 13's own predicate. The checker's independent copy of
    `app.services.sentence_coverage.opens_with_reporting_verb` -- see that
    function's own docstring; kept identical here in shape and value: True when
    *sentence* itself opens with a past-tense or third-person-singular reporting
    verb (any non-base, non-gerund inflection in `_RESIDUE_REPORTING_VERBS`,
    case-insensitive) immediately followed by "that", with nothing before either
    word -- not merely any sentence that happens to open with a reporting verb at
    all ("Studies show that ...", "Reports from teachers suggest ..." are both
    ordinary sentences, not this shape)."""
    stripped = (sentence or "").lstrip()
    match = re.match(r"([A-Za-zÀ-ɏ]+)\s+([A-Za-zÀ-ɏ]+)", stripped)
    if not match:
        return False
    first_word = _residue_fold(match.group(1))
    second_word = _residue_fold(match.group(2))
    return (
        first_word in _RESIDUE_PAST_OR_THIRD_PERSON_REPORTING_VERBS
        and second_word == "that"
    )


#: Kept identical, word for word, to `app.services.sentence_coverage.
#: _HEADING_SHAPE_FINITE_VERB_FORMS` -- the "be"/"have"/"do" auxiliary and copula
#: forms `_RESIDUE_STOP` already carries, plus every inflection of
#: `_RESIDUE_REPORTING_VERBS`.
_HEADING_SHAPE_FINITE_VERB_FORMS = (
    frozenset("is are was were be been being has have had do does did".split())
    | _RESIDUE_REPORTING_VERBS
)


def _is_title_case_phrase(text: str) -> bool:
    """Kept identical, in shape and value, to
    `app.services.sentence_coverage._is_title_case_phrase`: True when every word of
    *text* that is not a `_RESIDUE_STOP` stopword starts with a capital letter."""
    words = _RESIDUE_TOKEN_RE.findall(text)
    content_words = [w for w in words if w.casefold() not in _RESIDUE_STOP]
    if not content_words:
        return False
    return all(w[:1].isupper() for w in content_words)


def is_heading_shaped_paragraph(text: str) -> bool:
    """The checker's own copy of `app.services.sentence_coverage.
    is_heading_shaped_paragraph` -- see that function's own docstring; kept
    identical here in shape and value. Rule 16's own predicate: True when *text* --
    a whole delivered body paragraph -- reads as a heading rather than a sentence:
    no sentence-final punctuation, at most ten words, no citation, and no finite
    verb either by `_HEADING_SHAPE_FINITE_VERB_FORMS` or, when a word from that
    lexicon still appears, by `_is_title_case_phrase` overriding it.

    Distinct from `_paragraph_is_heading_shaped` (rule 9's own predicate, used by
    rule 8a too): that check answers a different question -- does this text carry
    markdown/bold heading SYNTAX at all, the shape an older saved run can carry as a
    plain "paragraph" node -- and stays False on exactly the text this rule exists
    for, since a fixed leading-heading strip (`app.services.fulltext`,
    `app.services.writing`) now keeps a genuine markdown heading marked as a heading
    node instead of ever demoting it to plain text with its marker gone. This rule
    is the content-shape guard for what is left once that marker really is gone:
    written with no markdown marker at all, whatever removed it."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped[-1] in _SENTENCE_TERMINAL_PUNCTUATION:
        return False
    if _RESIDUE_CITATION_RE.search(stripped) or _RESIDUE_NARRATIVE_CITATION_RE.search(stripped):
        return False
    words = _RESIDUE_TOKEN_RE.findall(stripped)
    if not words or len(words) > 10:
        return False
    folded_tokens = _residue_token_list(stripped)
    has_lexicon_verb = any(tok in _HEADING_SHAPE_FINITE_VERB_FORMS for tok in folded_tokens)
    if has_lexicon_verb and not _is_title_case_phrase(stripped):
        return False
    return True


@dataclass
class _Residue:
    text: str
    tokens: list[str]
    before_first_claim: bool
    trailing_gap: str
    leading_gap: str
    # The residue's own character offset in its sentence, so the bare-leading-subject
    # frame test below can tell "this residue opens the sentence" apart from "this
    # residue merely follows a single space", which `leading_gap` alone cannot (a
    # non-initial residue set off by one space also has an empty, once-stripped,
    # leading gap). Defaults to -1 (never sentence-initial) for the
    # position-independent fixture test, which never sets it.
    start: int = -1


def _residue_covered_indices(sentence: str, spans: list[str]) -> set[int]:
    sent_tokens = [t for t, _s, _e in _residue_tokens_with_spans(sentence)]
    covered: set[int] = set()
    for span in spans:
        needle = _residue_token_list(span or "")
        if not needle:
            continue
        n = len(needle)
        for i in range(len(sent_tokens) - n + 1):
            if sent_tokens[i : i + n] == needle:
                covered.update(range(i, i + n))
    return covered


def _residues(sentence: str, claims: list[str], citations: list[str] | None = None) -> list[_Residue]:
    spans = list(claims)
    spans += list(citations or [])
    spans += [m.group(0) for m in _RESIDUE_CITATION_RE.finditer(sentence)]
    spans += [m.group(0) for m in _RESIDUE_NARRATIVE_CITATION_RE.finditer(sentence)]
    toks = _residue_tokens_with_spans(sentence)
    covered = _residue_covered_indices(sentence, spans)
    claim_covered = _residue_covered_indices(sentence, list(claims))
    first_claim = min(claim_covered) if claim_covered else len(toks)
    out: list[_Residue] = []
    run: list[int] = []
    for i in range(len(toks)):
        if i in covered:
            if run:
                out.append(_make_residue_span(sentence, toks, run, first_claim))
                run = []
        else:
            run.append(i)
    if run:
        out.append(_make_residue_span(sentence, toks, run, first_claim))
    return out


def _make_residue_span(sentence, toks, run, first_claim) -> _Residue:
    start = toks[run[0]][1]
    end = toks[run[-1]][2]
    prev_end = toks[run[0] - 1][2] if run[0] > 0 else 0
    next_start = toks[run[-1] + 1][1] if run[-1] + 1 < len(toks) else len(sentence)
    return _Residue(
        text=sentence[start:end],
        tokens=[toks[i][0] for i in run],
        before_first_claim=run[-1] < first_claim,
        trailing_gap=sentence[end:next_start],
        leading_gap=sentence[prev_end:start],
        start=start,
    )


def _residue_content_tokens(residue: _Residue, claim_tokens: set[str]) -> list[str]:
    claim_stems = {_residue_singular(t) for t in claim_tokens}
    out = []
    for tok in residue.tokens:
        if tok in _RESIDUE_STOP or tok in _RESIDUE_ATTRIB:
            continue
        if _residue_singular(tok) in claim_stems:
            continue
        out.append(tok)
    return out


def _is_frame(
    residue: _Residue,
    claim_tokens: set[str],
    covered_numerals: frozenset[str] = frozenset(),
) -> str | None:
    """The frame class ("f1"/"f2"/"f3") *residue* is permitted as, or ``None`` --
    design section 3.3. Mirrors ``app.services.sentence_coverage.is_frame``: a
    residue is ordinarily disqualified outright by carrying any numeral, but not
    when every numeral it carries is also in *covered_numerals* -- the sentence's
    own verified evidence quotes' numerals (`covered_numeral_keys`) -- in which
    case it still has to clear every other rule below on its own merits."""
    left = _residue_content_tokens(residue, claim_tokens)
    residue_numerals = numeral_keys_in_text(residue.text)
    numeric_raw = any(t.isdigit() or t in _RESIDUE_NUMBER_WORDS for t in residue.tokens) or bool(
        re.search(r"[0-9%]", residue.text)
    )
    numeric = numeric_raw and not (residue_numerals and residue_numerals <= covered_numerals)
    if not left and not numeric and not (set(residue.tokens) & _RESIDUE_NEVER_FRAME):
        return "f1"
    if set(residue.tokens) & _RESIDUE_NEVER_FRAME:
        return None
    if numeric:
        return None
    if set(residue.tokens) & _RESIDUE_QUANTIFIERS:
        return None
    # A residue that opens the sentence and abuts the covered proposition's own
    # leading verb with no clause-boundary punctuation at all between them is that
    # verb's own grammatical subject, not a separable embellishment -- see the
    # identical comment and rationale in `app.services.sentence_coverage.is_frame`,
    # this module's own independent copy of which this is. Unconditional on the
    # three-token budget below, and gated on the residue's OWN text carrying no
    # boundary punctuation either (a residue that is already a complete clause, set
    # off from what follows by its own internal comma, is never a bare subject). Also
    # gated on the residue's own last token not being a coordinating conjunction
    # ("and", "or", "but", "yet", "so", "nor"), and not being a subordinator either;
    # the branch is capped by the same three-content-token budget every other frame
    # shape already answers to. See the identical comment and rationale in
    # `app.services.sentence_coverage.is_frame`.
    if (
        residue.before_first_claim
        and residue.start == 0
        and not residue.trailing_gap.strip()
        and not any(ch in residue.text for ch in (",", ";", ":", "—", "–"))
        and not _ends_in_coordinating_conjunction(residue.tokens)
        and not _ends_in_subordinator(residue.tokens)
        and len(left) <= _RESIDUE_FRAME_BUDGET
    ):
        return "f1"
    if len(left) > _RESIDUE_FRAME_BUDGET:
        return None
    if residue.before_first_claim and ":" in residue.trailing_gap:
        return "f2"
    if (
        residue.before_first_claim
        and "," in residue.trailing_gap
        and residue.tokens
        and residue.tokens[0] in (_RESIDUE_PREP_OPENERS | _RESIDUE_DISCOURSE_OPENERS)
    ):
        return "f3"
    return None


def _draft_citation_link_entries(draft_content: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Every citation-link entry recorded on any paragraph node's own
    ``attrs.citationLinks`` -- the same fresher-copy read `_draft_uncited_entries`
    already does for rule 1 (design section 3.2)."""
    entries: list[dict[str, Any]] = []
    for node in (draft_content or {}).get("content") or []:
        if not isinstance(node, dict) or node.get("type") != "paragraph":
            continue
        attrs = node.get("attrs") or {}
        entries.extend(attrs.get("citationLinks") or [])
    return entries


#: The adjective list shared by the "most"/"least" superlative-degree forms below
#: and the "more"/"less" comparative-degree forms alongside them -- "controlled" is
#: the one addition, needed for "Experimental work offers the most controlled
#: comparison." Kept identical to `app.services.sentence_coverage._DEGREE_
#: ADJECTIVES`.
_META_DEGREE_ADJECTIVES = r"(?:direct|directly|convincing|compelling|robust|striking|controlled)"
_META_SUPERLATIVE = (
    rf"(?:strongest|clearest|best|(?:most|least)\s+{_META_DEGREE_ADJECTIVES}|weakest|"
    r"definitive|conclusive|foremost|leading|prime|chief)"
)
#: Comparative evaluations of the literature -- "Li and Hébert (2023) provide
#: stronger evidence: ..." -- a bare comparative ("stronger"/"clearer"/"weaker") or
#: "more"/"less" plus the same bounded adjective list `_META_SUPERLATIVE` already
#: answers to, never a bare "more"/"less" alone: an ordinary comparative finding
#: about the studied phenomenon ("teacher feedback addressed more error types than
#: AWE", "recall rose from 10% to 55%") carries neither this adjective list nor,
#: immediately after it, one of `_META_LIT_NOUN`'s own literature nouns, so it is
#: never reached by either alternative below. Kept identical to `app.services.
#: sentence_coverage._COMPARATIVE`.
_META_COMPARATIVE = rf"(?:stronger|clearer|weaker|(?:more|less)\s+{_META_DEGREE_ADJECTIVES})"
_META_SUPERLATIVE_OR_COMPARATIVE = rf"(?:{_META_SUPERLATIVE}|{_META_COMPARATIVE})"
#: "comparison", "design", "test" and "tests" are added for "Experimental work
#: offers the most controlled comparison.": evidence, support, comparison, design,
#: study, work and test are the nouns a comparative evaluation of the literature can
#: land on. Kept identical to `app.services.sentence_coverage._LIT_NOUN`.
_META_LIT_NOUN = (
    r"(?:evidence|support|study|studies|research|work|finding|findings|demonstration|"
    r"account|case|treatment|example|illustration|comparison|design|test|tests)"
)
_META_EVALUATION_RE = re.compile(
    rf"\b{_META_SUPERLATIVE_OR_COMPARATIVE}\b(?:\s+\w+){{0,2}}\s+\b{_META_LIT_NOUN}\b"
    rf"|\b{_META_LIT_NOUN}\b\s+is\s+(?:\w+\s+){{0,2}}\b{_META_SUPERLATIVE_OR_COMPARATIVE}\b"
    rf"|\bis\s+(?:treated|discussed|addressed|examined|shown|handled)\s+"
    rf"{_META_SUPERLATIVE_OR_COMPARATIVE}\b",
    re.IGNORECASE,
)


def check_sentence_covered_by_verified_claims(
    draft_content: Mapping[str, Any] | None,
    writing_result: Mapping[str, Any] | None,
    claim_report: Mapping[str, Any] | None,
    section: str,
) -> tuple[list[Violation], list[str]]:
    """Rule 10 (design section 3): every delivered, cited body sentence, checked
    against the union of its own verified final-report claims and its own citation
    links.

    For every draft body sentence that has at least one matching final-report row
    (rule 1 already decides whether an uncited sentence may stand -- this rule is not
    its business), the sentence is covered by every ``claim_text`` of its own rows plus
    every rendered citation (their own ``citation`` field and every citation-link
    ``citation_text`` sharing the sentence) plus any bracketed-year or narrative
    citation this module's own regexes recognise inside it. A maximal uncovered token
    run is a residue; a residue that is not one of the three permitted frames (F1-F3,
    design section 3.3) is a violation.

    Separately, every writer proposition of THIS SECTION -- every distinct, non-empty
    ``proposition`` recorded on any of its own citation-link entries, whether its own
    sentence still survives in the delivered draft or not -- is matched to a verdict
    individually against the FULL (pre-finalize) verification list's own ``claim_text``
    values, never by subtracting section totals (``len(links) - len(claims verified)``,
    which only ever says how many are missing, never which). The match is against any
    status, not only ``verified``: a proposition the verifier assessed and correctly
    demoted (``needs_nuance``, ``unsupported``) was individually matched to a verdict
    and is not this rule's business, exactly as a sentence rule 1 lets stand is not
    this rule's business either; only a proposition the verifier never saw AT ALL,
    under its own exact text, is the dedup-defect shape (design section 1), reported as
    its own violation under this same rule name. Scoped to the whole section rather
    than to only the sentences the per-sentence loop above visits, because a sentence a
    coverage gate deletes takes its own citation-link propositions out of
    ``draft_paragraph_texts`` along with it -- scoping this half to delivered sentences
    only, as the residue half above must (a residue is only ever a property of
    delivered text), would make exactly this shape invisible to this checker: a
    fully verified proposition whose sentence a gate
    defect silently deleted. The match rate is reported as a notice regardless of
    outcome, so it is visible in the run's own output rather than only in a violation
    count."""
    verifications = _final_report_verifications(claim_report)
    if verifications is None or writing_result is None:
        return [], []

    links = list(writing_result.get("citation_links") or []) + _draft_citation_link_entries(
        draft_content
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in verifications:
        key = normalize_sentence(row.get("claim_sentence") or row.get("claim_text") or "")
        if key:
            groups.setdefault(key, []).append(row)

    violations: list[Violation] = []
    for paragraph in draft_paragraph_texts(draft_content):
        for raw_sentence in split_sentences(paragraph):
            sentence = normalize_sentence(raw_sentence)
            if not sentence:
                continue
            group = [
                row
                for norm, rows in groups.items()
                if norm == sentence or (norm and (norm in sentence or sentence in norm))
                for row in rows
            ]
            if not group:
                continue  # rule 1's own business, not rule 10's.
            claims = [row.get("claim_text") or "" for row in group]
            citations = [row.get("citation") or "" for row in group]
            sentence_links = [
                link
                for link in links
                if normalize_sentence(link.get("sentence")) == sentence
            ]
            citations += [link.get("citation_text") or "" for link in sentence_links]
            claim_tokens: set[str] = set()
            for claim_text in claims:
                claim_tokens.update(_residue_token_list(claim_text))
            evidence_quotes: list[str] = []
            for row in group:
                evidence_quotes.extend(row.get("evidence_quotes") or [])
                if row.get("evidence_quote"):
                    evidence_quotes.append(row["evidence_quote"])
            covered_numerals = covered_numeral_keys(evidence_quotes)
            residue_spans = _residues(sentence, claims, citations)
            bad_residues = [
                r for r in residue_spans if _is_frame(r, claim_tokens, covered_numerals) is None
            ]
            for residue in bad_residues:
                violations.append(
                    Violation(
                        RULE_SENTENCE_EXCEEDS_VERIFIED_CLAIM,
                        section,
                        f"sentence asserts material no verified claim covers: "
                        f"{residue.text!r}",
                        raw_sentence,
                    )
                )

    # Matched against the FULL (pre-finalize) verification list when a run carries one,
    # not only the delivered `final_report` rows `verifications` above holds: a
    # proposition the verifier assessed and correctly demoted (``needs_nuance``,
    # ``unsupported``, and so on) was individually matched to a verdict and is not this
    # rule's business -- only a proposition the verifier never saw AT ALL, under its
    # own exact text, is the dedup-defect shape. A run whose `claim_report` predates
    # the raw list, or a synthetic fixture that only ever sets `final_report` (as this
    # module's own tests do), falls back to the delivered-only list, exactly the
    # comparison already in effect before this section-wide check existed.
    full_verifications = list((claim_report or {}).get("verifications") or verifications)
    claim_norms = {
        normalize_sentence(row.get("claim_text") or "") for row in full_verifications
    }
    seen_props: set[str] = set()
    proposition_total = 0
    proposition_matched = 0
    for link in links:
        proposition_raw = (link.get("proposition") or "").strip()
        if not proposition_raw:
            continue
        proposition = normalize_sentence(proposition_raw)
        if not proposition or proposition in seen_props:
            continue
        seen_props.add(proposition)
        proposition_total += 1
        if proposition in claim_norms:
            proposition_matched += 1
            continue
        violations.append(
            Violation(
                RULE_SENTENCE_EXCEEDS_VERIFIED_CLAIM,
                section,
                f"citation-link proposition on this sentence has no verified row: "
                f"{proposition_raw!r}",
                link.get("sentence") or "",
            )
        )
    notices = (
        [
            f"{section}: {proposition_matched} of {proposition_total} writer "
            "propositions matched a verdict"
        ]
        if proposition_total
        else []
    )
    return violations, notices


def check_comparative_meta_evaluation(
    draft_content: Mapping[str, Any] | None,
    claim_report: Mapping[str, Any] | None,
    section: str,
) -> list[Violation]:
    """Rule 11 (design edit 8): a delivered sentence, or the verified claim behind it,
    carrying a comparative meta-evaluation of the literature -- checked over both the
    final-report rows' own ``claim_sentence``/``claim_text`` and every delivered draft
    sentence, so a phrase surviving in either place is caught. One violation per
    underlying row/sentence: the first matching field of a row wins."""
    violations: list[Violation] = []
    seen_norms: set[str] = set()
    for row in _final_report_verifications(claim_report) or []:
        for field_name in ("claim_sentence", "claim_text"):
            value = row.get(field_name)
            if not value:
                continue
            norm = normalize_sentence(value)
            if norm in seen_norms:
                continue
            match = _META_EVALUATION_RE.search(value)
            if match:
                seen_norms.add(norm)
                violations.append(
                    Violation(
                        RULE_COMPARATIVE_META_EVALUATION,
                        section,
                        f"sentence carries a comparative meta-evaluation of the "
                        f"literature: {match.group(0)!r}",
                        value,
                    )
                )
                break  # one violation per row: the first matching field wins.
    for paragraph in draft_paragraph_texts(draft_content):
        for raw_sentence in split_sentences(paragraph):
            norm = normalize_sentence(raw_sentence)
            if norm in seen_norms:
                continue
            match = _META_EVALUATION_RE.search(raw_sentence)
            if match:
                seen_norms.add(norm)
                violations.append(
                    Violation(
                        RULE_COMPARATIVE_META_EVALUATION,
                        section,
                        f"sentence carries a comparative meta-evaluation of the "
                        f"literature: {match.group(0)!r}",
                        raw_sentence,
                    )
                )
    return violations


def check_participle_after_attribution(
    draft_content: Mapping[str, Any] | None, section: str
) -> list[Violation]:
    """Rule 12: every delivered body sentence that opens with a citation-shaped
    subject immediately followed by a present-participle/gerund reporting-verb
    form, with no finite verb before it -- a subject with no finite verb of its own
    at all ("Teng and Ma (2024) cautioning that ...")."""
    violations: list[Violation] = []
    for paragraph in draft_paragraph_texts(draft_content):
        for raw_sentence in split_sentences(paragraph):
            if _opens_with_participle_after_attribution(raw_sentence):
                violations.append(
                    Violation(
                        RULE_PARTICIPLE_AFTER_ATTRIBUTION,
                        section,
                        "sentence opens with a citation subject immediately "
                        "followed by a participle/gerund reporting verb, with no "
                        "finite verb before it",
                        raw_sentence,
                    )
                )
    return violations


def check_verb_initial_sentence(
    draft_content: Mapping[str, Any] | None, section: str
) -> list[Violation]:
    """Rule 13: every delivered body sentence whose own first word is a reporting
    verb (any inflection in `_RESIDUE_REPORTING_VERBS`, case-insensitive) -- a
    finite verb standing in as the sentence's own predicate with no subject in
    front of it at all ("Showed that one student resubmitted her essay 13 times
    while another made one resubmission (Zhang & Hyland, 2018).")."""
    violations: list[Violation] = []
    for paragraph in draft_paragraph_texts(draft_content):
        for raw_sentence in split_sentences(paragraph):
            if _opens_with_reporting_verb(raw_sentence):
                violations.append(
                    Violation(
                        RULE_VERB_INITIAL_SENTENCE,
                        section,
                        "sentence opens with a bare reporting verb, with no "
                        "subject before it",
                        raw_sentence,
                    )
                )
    return violations


def _loose_norm(text: str | None) -> str:
    return normalize_sentence(text).lower()


#: Mirrors ``demo.run_demo._GUARD_DEMOTION_REASONS`` / ``is_guard_demotion`` (duplicated,
#: not imported -- see the module docstring): the one guard-demotion shape that does not
#: count as model drift for a fixture expecting "verified" -- the verification model
#: itself judged the claim ``verified`` (``model_status``) and the frozen
#: ``quote_not_verbatim`` guard alone (no other reason alongside it) demoted the reported
#: status to ``needs_nuance`` because its own copy of the quote drifted from the source by
#: a word or two. `run_demo.py`'s own printed table calls this a "GUARD DEMOTION" and
#: does not count it as a failure; this rule must not disagree with that reading of the
#: exact same run.
_GUARD_DEMOTION_REASONS = ["quote_not_verbatim"]


def _is_recorded_guard_demotion(raw_row: Mapping[str, Any] | None) -> bool:
    if not raw_row or raw_row.get("status") != "needs_nuance":
        return False
    return (
        raw_row.get("model_status") == "verified"
        and list(raw_row.get("machine_reasons") or []) == _GUARD_DEMOTION_REASONS
    )


def check_fixture_claims(
    protocol_claims: Sequence[Mapping[str, Any]] | None,
    draft_content: Mapping[str, Any] | None,
    claim_report: Mapping[str, Any] | None,
    section: str,
) -> tuple[list[Violation], list[str]]:
    """Rule 7. Only meaningful for the section the demo runner appends the protocol's
    own fixture claims to (the protocol section); called with *protocol_claims* only
    for that section. Returns ``(violations, notices)``: a notice, not violations, when
    *claim_report* carries no ``final_report`` at all (a build from before the
    standalone verify-and-heal action existed) -- there is no healed answer to check a
    fixture against on that shape, so guessing from the raw, pre-heal draft would
    report a healed-away claim as if it were still a live defect.

    Reconciling this rule with `run_demo.py`'s own tolerant reading of the same run:
    a fixture expecting "verified" also passes when it is not in the
    final report but its own raw (pre-heal) verification row records a guard demotion
    (`_is_recorded_guard_demotion`) -- the frozen `quote_not_verbatim` guard, and only
    that guard, demoted a model verdict of ``verified`` to ``needs_nuance``, which
    finalize then removes exactly like a genuine failure. Absent that recorded
    demotion, a missing "verified" fixture is still reported: this rule does not
    excuse a real loss, only the one guard shape the frozen verifier prompt and guard
    set cannot avoid and `run_demo.py` already prints as "GUARD DEMOTION" rather than
    a failure."""
    if not protocol_claims:
        return [], []
    verifications = _final_report_verifications(claim_report)
    if verifications is None:
        return [], [
            f"{section}: claim_report.json has no 'final_report' (older shape); rule 7 not checked"
        ]
    final_texts = [
        _loose_norm(row.get("claim_sentence") or row.get("claim_text")) for row in verifications
    ]
    final_texts = [t for t in final_texts if t]
    paragraphs = [_loose_norm(p) for p in draft_paragraph_texts(draft_content)]
    raw_verifications = list((claim_report or {}).get("verifications") or [])

    violations = []
    for claim in protocol_claims:
        text = _loose_norm(claim.get("text"))
        if not text:
            continue
        expected = claim.get("expected_status")
        if expected == "verified":
            present = any(text in final or final in text for final in final_texts)
            if not present:
                raw_hit = next(
                    (
                        row
                        for row in raw_verifications
                        if (norm := _loose_norm(row.get("claim_text")))
                        and (text in norm or norm in text)
                    ),
                    None,
                )
                if _is_recorded_guard_demotion(raw_hit):
                    present = True
            if not present:
                violations.append(
                    Violation(
                        RULE_FIXTURE_VERIFIED_MISSING,
                        section,
                        f"fixture {claim.get('id')!r} expects 'verified' but is not "
                        "present in the final report, and no recorded guard demotion "
                        "(quote_not_verbatim alone, model_status verified) accounts "
                        "for its absence",
                        claim.get("text"),
                    )
                )
        elif expected == "unsupported":
            if any(text in paragraph for paragraph in paragraphs):
                violations.append(
                    Violation(
                        RULE_FIXTURE_UNSUPPORTED_PRESENT,
                        section,
                        f"fixture {claim.get('id')!r} expects 'unsupported' but its "
                        "text is still present in the saved draft",
                        claim.get("text"),
                    )
                )
    return violations, []


# ---------------------------------------------------------------------------
# Rule 8: the deterministic coherence pass, checked independently
# ---------------------------------------------------------------------------

#: Mirrors ``app.services.fulltext._ENUMERATION_NUMBER_WORDS`` / `_ENUMERATION_OPENER_RE`.
_ENUMERATION_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
#: Narrowed to the three verbs that only ever introduce a list ("emerge", "follow",
#: "stand out"). "are"/"were"/"exist(s/ed)"/"remain(s/ed)" are deliberately excluded:
#: a plain factual sentence uses them just as often as a real announcement ("Three
#: studies were conducted in naturalistic classrooms." is not announcing anything to
#: be counted) -- see `app.services.fulltext._ENUMERATION_
#: OPENER_RE`'s own docstring note for the four false-positive examples this closes.
_ENUMERATION_OPENER_RE = re.compile(
    r"^[\s*_]*(?P<num>[A-Za-z]+|\d+)\s+[a-z]+s\s+"
    r"(?:emerge[sd]?|stands?\s+out|follow(?:s|ed)?)\b",
    re.IGNORECASE,
)
#: The ordinal words `_next_sentence_resolves_enumeration_as_a_
#: list` looks for, mirroring ``app.services.fulltext._ORDINAL_SEQUENCE_WORDS``.
_ORDINAL_SEQUENCE_WORDS = ["first", "second", "third", "fourth", "fifth", "sixth"]

#: Mirrors ``app.services.fulltext._DANGLING_CONNECTIVE_RE``: sentence-initial,
#: including "first" (mirrors the app-side fix's own list).
_DANGLING_CONNECTIVE_RE = re.compile(
    r"^[\s*_]*(however|moreover|similarly|crucially|notably|complementing|furthermore|"
    r"additionally|conversely|nevertheless|nonetheless|consequently|meanwhile|"
    r"likewise|correspondingly|first|second|third|fourth|fifth|sixth|finally)\b",
    re.IGNORECASE,
)

#: The three literal deictic phrases this rule targets, sentence-initial
#: (mirrors ``app.services.fulltext._DANGLING_DEICTIC_RE``'s own literal half): "the
#: study('s)", "the studies", "the authors", "the same authors".
_DANGLING_LITERAL_DEICTIC_RE = re.compile(
    r"^[\s*_]*(the\s+stud(?:y|ies)'?s?\b|the\s+same\s+authors?\b|the\s+authors?\b)",
    re.IGNORECASE,
)

#: An unresolved deictic built from a demonstrative and one of a curated, narrow set of
#: abstract referent nouns -- two representative examples ("this finding", "these
#: gaps") plus the closest synonyms the report's own dangling rows need ("these
#: dimensions", "these themes", "this distinction", ...) -- searched ANYWHERE in the
#: sentence, not only at its start, since an in-paragraph cascade sentence rule 8c
#: still checks can carry the unresolved reference at its own end rather than its
#: start ("... to examine these dimensions."). Deliberately NOT a generic "this/these/
#: such plus any noun" template (unlike the app-side fix's own `_opens_with_dangling_
#: marker`): "this review", "this study" and "this paper" are common, entirely benign
#: self-references a generic template would misfire on, and a curated noun list avoids
#: them.
_DANGLING_TEMPLATED_DEICTIC_RE = re.compile(
    r"(?:^|[^A-Za-z])(?:this|these|such)\s+"
    r"(finding|findings|gap|gaps|dimension|dimensions|theme|themes|distinction|"
    r"distinctions|concern|concerns|issue|issues)\b",
    re.IGNORECASE,
)


def _enumeration_announced_count(sentence: str) -> int | None:
    """Mirrors ``app.services.fulltext._enumeration_announced_count``, including its
    own colon exception: "Three gaps remain: A; B; and C." resolves its own count in
    the same sentence and is never a shortfall, whatever follows it in the paragraph."""
    if ":" in sentence:
        return None
    match = _ENUMERATION_OPENER_RE.match(sentence)
    if not match:
        return None
    raw = match.group("num").lower()
    if raw.isdigit():
        return int(raw)
    return _ENUMERATION_NUMBER_WORDS.get(raw)


def _next_sentence_resolves_enumeration_as_a_list(sentence: str, announced: int) -> bool:
    """Mirrors ``app.services.fulltext._next_fragment_resolves_enumeration_as_a_list``:
    True when *sentence* -- the very next sentence after an enumeration opener with no
    colon of its own -- itself lists at least *announced* items via sentence-internal
    ordinal markers, in increasing order ("First, designs are short; second, corpora
    are small; third, engagement is untracked.")."""
    if announced > len(_ORDINAL_SEQUENCE_WORDS):
        return False
    lowered = sentence.lower()
    positions: list[int] = []
    for word in _ORDINAL_SEQUENCE_WORDS[:announced]:
        match = re.search(rf"\b{word}\b", lowered)
        if match is None:
            return False
        positions.append(match.start())
    return positions == sorted(positions)


def _opens_with_dangling_connective_or_ordinal(sentence: str) -> bool:
    return bool(_DANGLING_CONNECTIVE_RE.match(sentence))


def _has_unresolved_deictic(sentence: str) -> bool:
    return bool(
        _DANGLING_LITERAL_DEICTIC_RE.match(sentence)
        or _DANGLING_TEMPLATED_DEICTIC_RE.search(sentence)
    )


def check_paragraph_has_citation(
    draft_content: Mapping[str, Any] | None, section: str
) -> list[Violation]:
    """Rule 8a: a body paragraph with no citation at all, other than the document's own
    first paragraph (mirrors ``app.services.fulltext.finalize_generated_section``'s and
    ``_rebuild_finalized_node_list``'s own rule (a), applied here to the delivered
    draft directly rather than inferred from the write job's own removal path) or one
    carrying an "unclassified" sentence (`ParagraphDescriptor.has_unclassified`),
    which the same two functions also exempt."""
    violations = []
    for descriptor in _paragraph_descriptors(draft_content):
        if descriptor.index == 0 or descriptor.has_citation or descriptor.has_unclassified:
            continue
        violations.append(
            Violation(
                RULE_PARAGRAPH_HAS_NO_CITATION,
                section,
                f"paragraph {descriptor.index}: carries no citation at all and is not "
                "the document's own opening paragraph",
                descriptor.text,
            )
        )
    return violations


def check_dangling_framing(
    draft_content: Mapping[str, Any] | None, section: str
) -> list[Violation]:
    """Rules 8b and 8c, over every paragraph's own sentences.

    8b: an uncited enumeration opener ("Three gaps emerge.") is a violation the instant
    fewer sentences than announced follow it in the same paragraph AND the very next
    sentence does not itself resolve the same count as a colon-free list of its own
    (`_next_sentence_resolves_enumeration_as_a_list`) -- purely a property of the
    delivered text, so, unlike 8c, it needs no ``trouble_before`` signal to fire.

    8c: an UNCITED sentence that opens with a discourse connective/ordinal or an
    unresolved deictic is a violation once ``trouble_before`` is true for it.
    ``trouble_before`` is whether an EARLIER sentence of the same paragraph fired rule
    8b -- never whether an earlier sentence fired 8c itself, and it starts ``False`` at
    every paragraph's own first sentence, whatever precedes the paragraph. Only 8b's
    own shortfall represents a genuine absence -- the announced items are not there to
    be counted -- which is why only 8b propagates: an earlier sentence that survives is
    still visibly present in the delivered text, whatever it itself opens with, and is
    a real antecedent for a "however"/"this"/"such" that follows it in the very next
    sentence.

    Rule 8c never fires on a cited sentence, nor on a paragraph's own first sentence
    merely because a heading preceded it: both are shapes the app-side fix (scoped to
    an UNCITED fragment, and to "something earlier in THIS paragraph was removed") can
    never itself satisfy -- a cited sentence cannot be dropped without deleting
    supported content, and a paragraph's own first sentence has nothing earlier in the
    SAME paragraph for the app-side pass to react to. Gating the exit code on a shape
    the app can never produce clean text for would mean every future paid run carrying
    one exits 2 for a defect nothing in the product could close. Narrowed to exactly
    what `_drop_dangling_framing_sentences`'s own rule (c) guarantees -- an uncited
    fragment, trouble only from an in-paragraph 8b cascade -- so the two are one rule,
    checked twice: the residue this narrowing accepts (a cited dangling sentence, or an
    uncited paragraph-opening one with nothing earlier to react to) is tracked
    separately rather than gated on this rule's own exit code."""
    violations: list[Violation] = []
    for descriptor in _paragraph_descriptors(draft_content):
        sentences = split_sentences(descriptor.text)
        trouble_before = False
        for sentence_index, sentence in enumerate(sentences):
            is_cited = _paragraph_has_rendered_citation(sentence)
            following_sentences = sentences[sentence_index + 1 :]

            fired_b = False
            if not is_cited:
                announced = _enumeration_announced_count(sentence)
                if announced is not None and len(following_sentences) < announced:
                    resolved_as_list = following_sentences and (
                        _next_sentence_resolves_enumeration_as_a_list(
                            following_sentences[0], announced
                        )
                    )
                    if not resolved_as_list:
                        violations.append(
                            Violation(
                                RULE_ENUMERATION_OPENER_SHORT_OF_COUNT,
                                section,
                                f"paragraph {descriptor.index} sentence {sentence_index}: "
                                f"announces {announced} but only "
                                f"{len(following_sentences)} sentence(s) follow it in "
                                "the paragraph",
                                sentence,
                            )
                        )
                        fired_b = True
            if not fired_b and not is_cited and trouble_before and (
                _opens_with_dangling_connective_or_ordinal(sentence)
                or _has_unresolved_deictic(sentence)
            ):
                violations.append(
                    Violation(
                        RULE_DANGLING_FRAMING_SENTENCE,
                        section,
                        f"paragraph {descriptor.index} sentence {sentence_index}: "
                        "opens with a discourse connective/ordinal or an "
                        "unresolved deictic, with no surviving antecedent before it",
                        sentence,
                    )
                )
            if fired_b:
                trouble_before = True
    return violations


# ---------------------------------------------------------------------------
# Rule 9: a heading with no body
# ---------------------------------------------------------------------------


def _is_heading_node(node: Mapping[str, Any]) -> bool:
    """True for a real ``"heading"`` node, or a ``"paragraph"`` node
    `_paragraph_is_heading_shaped` recognises as one -- the same test
    `_paragraph_descriptors` applies, so a historical run's own restated markdown
    title counts as a heading for this rule too, not as a bodiless paragraph under it."""
    node_type = node.get("type")
    if node_type == "heading":
        return True
    return node_type == "paragraph" and _paragraph_is_heading_shaped(_node_text(node))


def check_heading_has_no_body(
    draft_content: Mapping[str, Any] | None, section: str
) -> list[Violation]:
    """Rule 9: a heading immediately followed by another heading, or by nothing at
    all (the end of the section's own draft), has no body of its own. Same underlying
    defect as ``app.services.fulltext._heading_no_body_keep_mask``: rule (a) (dropping
    a paragraph with no surviving citation) can leave the heading that introduced it
    with nothing underneath, and nothing looked for that shape before this rule --
    `check_duplicate_leading_heading` only ever compares two consecutive headings'
    TEXT, never whether a heading has a body at all. The document's own very first
    node is never flagged even when it is a bodiless heading, matching the app-side
    fix's own exemption: a section reduced to nothing still keeps its own title.

    This rule is deliberately BROADER than either app-side caller on one shape --
    `_is_heading_node` also counts a ``"paragraph"`` node whose
    own text is a markdown heading line or a bold sub-heading (`_paragraph_is_heading_
    shaped`), which neither `finalize_generated_section` (`_heading_no_body_keep_mask`
    over `_is_heading_block`, text blocks) nor `finalize_draft_document`
    (`_heading_no_body_keep_mask` over ``node.get("type") == "heading"``, the saved
    node list) ever treats as a heading. This only matters for a HISTORICAL draft
    written before `_build_section_tiptap_nodes` started promoting such a line to a
    real heading node: on a new run this shape cannot occur at all, so the two
    predicates only disagree on old fixtures, never on what the app itself can leave
    behind today."""
    nodes = [node for node in (draft_content or {}).get("content") or [] if isinstance(node, dict)]
    violations = []
    for index, node in enumerate(nodes):
        if index == 0 or not _is_heading_node(node):
            continue
        if index + 1 >= len(nodes) or _is_heading_node(nodes[index + 1]):
            reason = (
                "it is the section's own last node"
                if index + 1 >= len(nodes)
                else "it is immediately followed by another heading"
            )
            violations.append(
                Violation(
                    RULE_HEADING_WITH_NO_BODY,
                    section,
                    f"node {index} ({_node_text(node)!r}): has no body -- {reason}",
                    _node_text(node),
                )
            )
    return violations


def check_non_title_heading(
    draft_content: Mapping[str, Any] | None, section: str
) -> list[Violation]:
    """Companion to rule 9: a generated section's own title -- node 0 -- is the
    only heading it ever carries now. A real ``"heading"`` node anywhere else in
    the delivered node list is always a violation, whether or not it has a body of
    its own (a bodiless one is ALSO caught by rule 9; this rule fires independently
    of that, since a heading with a perfectly good body underneath it is still a
    heading this product's own writer is never supposed to deliver any more).

    Only a real ``"heading"`` node is checked here -- not the broader
    `_is_heading_node`/`_paragraph_is_heading_shaped` sense rule 9 and rule 8a use,
    which also catches a historical run's own markdown-syntax "paragraph" node: a
    fresh run can no longer produce that shape at all (`_strip_inline_emphasis`'s
    own heading exemption still only ever protects a block already promoted to a
    real heading node), so a plain paragraph that merely reads heading-shaped is
    rule 16's own question, not this one."""
    nodes = [node for node in (draft_content or {}).get("content") or [] if isinstance(node, dict)]
    violations = []
    for index, node in enumerate(nodes):
        if index == 0 or node.get("type") != "heading":
            continue
        violations.append(
            Violation(
                RULE_NON_TITLE_HEADING_IN_SECTION,
                section,
                f"node {index} ({_node_text(node)!r}): a generated section carries "
                "only its own title heading; every other heading the writer wrote "
                "is dropped whole at finalize, so this one should never have "
                "survived",
                _node_text(node),
            )
        )
    return violations


# ---------------------------------------------------------------------------
# Rule 5: delivered_evidence.json
# ---------------------------------------------------------------------------


def check_delivered_evidence_rows(rows: Sequence[Mapping[str, Any]]) -> list[Violation]:
    """Rule 5, over the rows of an already-loaded ``delivered_evidence.json``."""
    violations = []
    for row in rows:
        section = row.get("section_title") or "protocol section"
        label = f"delivered_evidence row {row.get('row')}"
        if not row.get("passage_located"):
            violations.append(
                Violation(
                    RULE_PASSAGE_NOT_LOCATED,
                    section,
                    f"{label}: passage_located is not true",
                    row.get("sentence"),
                )
            )
            continue
        passage = row.get("source_passage") or ""
        for quote in row.get("evidence_quotes") or []:
            if not is_verbatim_under_guard_fold(quote, passage):
                violations.append(
                    Violation(
                        RULE_EVIDENCE_NOT_VERBATIM,
                        section,
                        f"{label}: evidence quote is not verbatim in source_passage "
                        "under the verifier guard's own fold",
                        quote,
                    )
                )
    return violations


# ---------------------------------------------------------------------------
# File loading and orchestration
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class SectionFiles:
    """The three per-section artefacts a demo run writes, plus the names of any of
    them that were not found on disk (say so rather than skipping silently)."""

    name: str
    index: int | None
    draft_content: Any | None = None
    writing_result: Any | None = None
    claim_report: Any | None = None
    missing: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return (
            self.draft_content is None
            and self.writing_result is None
            and self.claim_report is None
        )


def _section_filenames(index: int | None) -> dict[str, str]:
    suffix = "" if index is None else f"_{index}"
    return {
        "draft_content": f"draft_content{suffix}.json",
        "writing_result": f"writing_result{suffix}.json",
        "claim_report": f"claim_report{suffix}.json",
    }


def load_section_files(
    run_dir: Path, *, index: int | None, name: str | None = None
) -> SectionFiles:
    """Read one section's own three files from *run_dir* (the protocol section's own,
    unindexed names when *index* is ``None``; extra section *index*'s own
    ``_<index>``-suffixed names otherwise -- the exact scheme
    ``run_demo.DemoRunner._write_one_extra_section`` writes). A file that does not
    exist is recorded in ``.missing`` rather than raising, so a historical run from
    before a file existed, or one this checker is asked about mid-run, is still
    checked as far as it can be."""
    filenames = _section_filenames(index)
    section_name = name or ("protocol section" if index is None else f"extra section {index}")
    values: dict[str, Any | None] = {}
    missing: list[str] = []
    for key, filename in filenames.items():
        path = run_dir / filename
        if path.exists():
            values[key] = _read_json(path)
        else:
            values[key] = None
            missing.append(filename)
    return SectionFiles(
        name=section_name,
        index=index,
        missing=missing,
        draft_content=values["draft_content"],
        writing_result=values["writing_result"],
        claim_report=values["claim_report"],
    )


def discover_extra_section_indices(run_dir: Path) -> list[int]:
    """Every ``N`` such that ``draft_content_N.json`` exists under *run_dir*, sorted
    (extra sections are indexed 2, 3, 4, ...)."""
    indices: set[int] = set()
    for path in run_dir.glob("draft_content_*.json"):
        suffix = path.stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            indices.add(int(suffix))
    return sorted(indices)


def check_section(
    files: SectionFiles, *, protocol_claims: Sequence[Mapping[str, Any]] | None = None
) -> tuple[list[Violation], list[str]]:
    """Rules 1-4 and 6, rules 8 and 9, rules 10, 11, 12 and 13, the four new rules
    (``empty_cited_section``, ``title_echo_in_body``, ``heading_shaped_paragraph``,
    ``non_title_heading_in_section``), and (when *protocol_claims* is given) 7,
    over one section's own files. Rule 5 (``delivered_evidence.json``) is a
    whole-run artefact and is checked separately by
    `check_delivered_evidence_file`."""
    violations: list[Violation] = []
    notices: list[str] = [
        f"{files.name}: {filename} not found; some rules could not be checked"
        for filename in files.missing
    ]
    if files.draft_content is None:
        return violations, notices

    violations += check_duplicate_leading_heading(files.draft_content, files.name)
    violations += check_title_echo_in_body(files.draft_content, files.name)
    violations += check_heading_shaped_paragraph(files.draft_content, files.name)
    violations += check_needs_citation_marker(files.draft_content, files.name)
    violations += check_citation_coverage_resolved(files.claim_report, files.name)
    violations += check_empty_cited_section(files.writing_result, files.claim_report, files.name)
    violations += check_final_report_rows(files.draft_content, files.claim_report, files.name)
    violations += check_truncated_sentences(files.draft_content, files.claim_report, files.name)
    violations += check_paragraph_has_citation(files.draft_content, files.name)
    violations += check_dangling_framing(files.draft_content, files.name)
    violations += check_heading_has_no_body(files.draft_content, files.name)
    violations += check_non_title_heading(files.draft_content, files.name)
    classification_violations, classification_notices = check_body_sentence_classification(
        files.draft_content, files.writing_result, files.claim_report, files.name
    )
    violations += classification_violations
    notices += classification_notices
    coverage_violations, coverage_notices = check_sentence_covered_by_verified_claims(
        files.draft_content, files.writing_result, files.claim_report, files.name
    )
    violations += coverage_violations
    notices += coverage_notices
    violations += check_comparative_meta_evaluation(
        files.draft_content, files.claim_report, files.name
    )
    violations += check_participle_after_attribution(files.draft_content, files.name)
    violations += check_verb_initial_sentence(files.draft_content, files.name)
    if protocol_claims:
        fixture_violations, fixture_notices = check_fixture_claims(
            protocol_claims, files.draft_content, files.claim_report, files.name
        )
        violations += fixture_violations
        notices += fixture_notices
    return violations, notices


def check_delivered_evidence_file(run_dir: Path) -> tuple[list[Violation], list[str]]:
    """Rule 5, reading ``delivered_evidence.json`` from *run_dir*. Its absence is
    reported as a notice, not a violation: a run from before the delivered-evidence
    record existed, or one this checker is asked about before its last step has run,
    has nothing here to check yet, which is not itself a defect."""
    path = run_dir / "delivered_evidence.json"
    if not path.exists():
        return [], [
            f"{run_dir.name}: delivered_evidence.json not found; rule 5 (evidence "
            "verbatim / passage_located) not checked"
        ]
    record = _read_json(path)
    return check_delivered_evidence_rows(record.get("rows") or []), []


@dataclass
class CheckResult:
    violations: list[Violation]
    notices: list[str]
    sections_checked: list[str]

    @property
    def ok(self) -> bool:
        return not self.violations


def check_run_directory(
    run_dir: Path | str, *, protocol: Mapping[str, Any] | None = None
) -> CheckResult:
    """The comprehensive, whole-run check: every section present under *run_dir* (the
    protocol section's own unindexed files, plus every extra section
    `discover_extra_section_indices` finds), rules 1-4 and 6-9 on each, and rule 5 once
    over ``delivered_evidence.json``. *protocol* (``protocol.json``, parsed), when
    given, supplies the fixture claims rule 7 checks against the protocol section
    only -- the section the demo runner appends them to."""
    run_dir = Path(run_dir)
    violations: list[Violation] = []
    notices: list[str] = []
    sections_checked: list[str] = []

    protocol_claims = (protocol or {}).get("claims") if protocol else None
    protocol_files = load_section_files(run_dir, index=None)
    if not protocol_files.is_empty:
        sections_checked.append(protocol_files.name)
        section_violations, section_notices = check_section(
            protocol_files, protocol_claims=protocol_claims
        )
        violations += section_violations
        notices += section_notices
    else:
        # `check_section`'s own missing-file notices are only ever built inside that
        # function, which is never called for an entirely empty section -- so a run
        # directory with none of the protocol section's three files (an empty or
        # failed run) would otherwise produce no notice at all, contradicting this
        # module's own stated rule that a missing file is named rather than silently
        # skipped. Every missing file is named here, exactly as `check_section` would
        # name it, plus one summary line saying no rules could be checked for this
        # section at all.
        notices += [
            f"{protocol_files.name}: {filename} not found; some rules could not be checked"
            for filename in protocol_files.missing
        ]
        notices.append(
            f"{protocol_files.name}: no files found at all; no rules could be checked "
            "for this section"
        )

    for index in discover_extra_section_indices(run_dir):
        files = load_section_files(run_dir, index=index)
        sections_checked.append(files.name)
        section_violations, section_notices = check_section(files)
        violations += section_violations
        notices += section_notices

    evidence_violations, evidence_notices = check_delivered_evidence_file(run_dir)
    violations += evidence_violations
    notices += evidence_notices

    return CheckResult(violations=violations, notices=notices, sections_checked=sections_checked)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="a demo run directory (e.g. demo/output/<timestamp>, or demo/expected)",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=None,
        help=f"protocol.json, for rule 7 (fixture claims); default {DEFAULT_PROTOCOL}",
    )
    args = parser.parse_args(argv)

    protocol_path = args.protocol or DEFAULT_PROTOCOL
    protocol = _read_json(protocol_path) if protocol_path.exists() else None

    result = check_run_directory(args.run_dir, protocol=protocol)
    for notice in result.notices:
        print(f"NOTICE: {notice}", file=sys.stderr)
    if result.violations:
        print(
            f"\n{len(result.violations)} delivered-text violation(s) in {args.run_dir}:",
            file=sys.stderr,
        )
        for violation in result.violations:
            print(f"  {violation}", file=sys.stderr)
        return 2
    print(
        f"OK: no delivered-text violations in {args.run_dir} "
        f"({len(result.sections_checked)} section(s) checked)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
