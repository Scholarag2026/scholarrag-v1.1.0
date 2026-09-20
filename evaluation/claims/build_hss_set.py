"""Build the 30-claim HSS (applied-linguistics) claim-verification set by protocol (E2).

Interpreter: the **system** ``python`` (backend dependencies for the acquisition step).
Importers: ``run_hss.py`` (``doi_slug``, ``load_cache``), ``tests/test_claims_hss.py``.
No LLM is involved anywhere in this script.

Step 1 (network; skipped with ``--skip-fetch``): candidate open-access articles are listed
from OpenAlex for fully-OA applied-linguistics journals (``JOURNALS``), then acquired through
the *production* pipeline -- ``UnpaywallClient().lookup(doi)`` -> ``fetch_pdf_from_url`` ->
``extract_text_from_pdf`` -> ``chunk_text`` -- keeping a DOI only when the text has at least
``--min-chars`` characters and ``--min-chunks`` chunks. Selected sources are written to
``claims/hss_sources.json`` (committed) and the full texts to
``claims/data/hss_fulltext/<doi-slug>.json`` (gitignored)::

    {doi, title, authors: [str], source_url, fetched_at, n_chars, chunks: [{section, text}]}

Step 2 (deterministic; ``--seed`` only rotates the round-robin start): candidate sentences
are ranked with ``common.select_candidate_sentences`` (results/findings/discussion/
conclusion chunks first) and assigned to five construction rules, one claim per sentence,
no sentence reused, papers balanced round-robin:

    verbatim (5)      claim == sentence                     expected: verified
    paraphrase (5)    fixed ordered substitution table      expected: verified
                      (reporting verbs / connectives / generic nouns only; technical
                      terms are never substituted)
    altered (10)      direction flip / number x2 (% capped  expected: unsupported | needs_nuance
                      at 99) / quantifier flip
    wrong_paper (5)   sentence from paper X, chunks of Y    expected: unsupported
    no_full_text (5)  sentence, chunks=[]                   expected: no_full_text (no model call)

Output: ``claims/hss_claims.jsonl`` (committed; not under ``data/`` because that path is
gitignored) with rows ``{item_id: "hss-<rule>-<nn>", rule, expected, claim, original_sentence,
alteration|null, source_doi, chunk_doi (Y for wrong_paper; null for no_full_text),
chunk_index (chunk of the source paper the sentence came from), title, authors (of the paper
whose chunks are shown; the source paper for no_full_text), built_at}`` plus
``claims/hss_claims.build.json`` (counts per rule, sources, seed, tables used).

A fetched source is dropped as
``"wrong_work"`` when the extracted text's own title/author does not match the OpenAlex record
(:func:`check_fetched_text_identity`); reference-list and back-matter chunks are dropped before
candidate mining regardless of their section label (:func:`is_reference_or_frontmatter_chunk`,
:func:`_reference_or_backmatter_indices`); ``is_clean_sentence`` additionally rejects soft
hyphens, a repeated numbered heading label, a bracketed-translation bare title and a
three-or-more-name attribution; ``alter()`` additionally never touches a cross-reference label,
a word framing a "(or X)" parenthetical, an already-doubly-negated "significant", "all but one"
or a quantifier/direction inside a negated intensional frame; veto matching is by
:func:`normalise_for_veto` (NFKC, diacritics stripped, whitespace collapsed, case-folded), with
``--veto-strict`` (default on) raising loudly on a near-miss (:func:`find_veto_near_misses`);
and every HTTP call carries explicit connect/read timeouts, with a per-candidate wall-clock
budget in ``acquire_sources`` (``--candidate-timeout-s``) so a build cannot hang indefinitely.

The wrong-work identity check can now also be applied to an already
cached payload, not just a fresh fetch (:func:`check_cache_identity`, ``build_items_from_cache``'s
``identity_check`` / ``--identity-check-cache``, off by default so the frozen 30-item test set's
one disclosed mis-fetched source stays byte identical); the identity check's title/surname
comparison and ``normalise_for_veto`` both fold diacritics the same way (:func:`_fold_diacritics`)
so a name such as OpenAlex's "Majid ELANİ SHİRVAN" (Turkish dotted capital I) is not rejected;
``HEADING_PREFIX_RE`` allows a small set of lowercase function words inside a Title-Case run;
a quantifier "all" in object position after a benefactive/purposive preposition ("for all
learners") is never flipped; a direction flip is vetoed for the whole sentence, not just the
framing word, when the sentence carries a "(or X)" parenthetical; and ``--candidate-timeout-s``
now also bounds ``--probe-journals``'s own ``acquire_one`` call.

The cache-side identity check now defaults **on** (``--no-identity-check``
is the opt-out), with the frozen 30-item test set's one disclosed mis-fetch
(``10.55593/ej.26103a4``) kept unconditionally via the module-level
:data:`KNOWN_IDENTITY_EXCEPTIONS` rather than an opt-in flag no recipe ever passed; the
object-position quantifier guard is keyed on the quantifier phrase's alias ("all of" and "every"
now inherit "all"'s guard) rather than the literal flip-table token, closing the hand-off to a
second, unguarded quantifier later in the same sentence; a floating quantifier immediately before
its own finite verb ("who all identified") is never flipped; a quantifier or direction flip whose
destination is itself a negation is never applied when the sentence already carries that same
negation elsewhere; ``ATTRIBUTION_RE`` accepts "(in press)"/"(n.d.)"/"(forthcoming)"/"(in
preparation)" wherever it accepted a four-digit year, rejects a bare pronoun subject ("They
report ...") with a reporting verb, and rejects "These/Those/Such authors ..."; a sentence whose
subject is elided into the *previous* sentence's own attribution ("Participants were ...", right
after a sentence naming another study) is rejected via a one-sentence look-back
(:func:`previous_sentence_in_chunk`); ``CROSS_REFERENCE_LABEL_RE`` covers "session"/"phase"/
"round"/"stage"/"wave"/"cycle"/"week"/"day"/"item"/"task"/"condition"/"group"; a numeric
operator never touches a page number or a sample-size marker inside a citation
(``PAGE_NUMBER_BEFORE_RE`` / ``SAMPLE_SIZE_BEFORE_RE``, alongside the pre-existing year and
cross-reference-label guards -- a CEFR-style code such as "B2" was already out of ``NUMBER_RE``'s
reach); a numbered/dashed
list label glued to its description (``LIST_LABEL_DASH_RE``) and a footnote digit glued directly
to a word with no period (``FOOTNOTE_DIGIT_GLUED_RE``) are rejected; a block quotation introduced
by a colon but never closed by a citation or quotation mark is detected within a bounded lookback
window (``QUOTE_COLON_INTRO_RE``); the non-round numeric operator requires a minimum relative
change (``MIN_RELATIVE_CHANGE``) so the 99% cap can never be the entire visible alteration; and
``over_specified`` is only ever applied to a sentence that itself reports an empirical finding
(:func:`_reports_empirical_finding`), never with a modifier duplicating a detail the sentence
already states.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
import re
import sys
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import (  # noqa: E402
    add_backend_to_path,
    append_jsonl,
    export_env_from_dotenv,
    import_backend_settings,
    load_dotenv_values,
    normalise_whitespace,
    now_iso,
    read_json,
    read_jsonl,
    resolve_path_args,
    select_candidate_sentences,
    split_sentences,
    tokenize,
    write_json,
)

REPO_ROOT = HERE.parent.parent
DEFAULT_SOURCES = HERE / "hss_sources.json"
DEFAULT_CACHE_DIR = HERE / "data" / "hss_fulltext"
DEFAULT_OUT = HERE / "hss_claims.jsonl"
DEFAULT_VETO = HERE / "hss_veto.json"
DEFAULT_PROBE_OUT = HERE / "hss_test_v3_probe.json"
DEFAULT_SEED = 20260902
DEFAULT_MIN_CHARS = 15_000
DEFAULT_MIN_CHUNKS = 3
DEFAULT_N_SOURCES = 10
OPENALEX_WORKS = "https://api.openalex.org/works"
CROSSREF_WORKS = "https://api.crossref.org/works"
# Explicit connect/read timeouts on every HTTP call: a
# bare float timeout still leaves some httpx phases unbounded on Windows, which is how an
# earlier build hung indefinitely on one candidate. ``DEFAULT_CANDIDATE_TIMEOUT_S`` is the
# wall-clock budget for one whole ``acquire_one`` call (lookup + fetch + extract + chunk
# together), not just one HTTP request.
CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 30.0
WRITE_TIMEOUT_S = 30.0
POOL_TIMEOUT_S = 10.0
DEFAULT_CANDIDATE_TIMEOUT_S = 120.0

# Fully open-access applied-linguistics journals (ISSN-L looked up on
# https://api.openalex.org/sources?search=<name>, 2026-09-02).
JOURNALS: tuple[dict[str, str], ...] = (
    {"name": "Studies in Second Language Learning and Teaching", "issn": "2083-5205",
     "openalex": "S2764526439"},
    {"name": "Language Learning & Technology", "issn": "1094-3501", "openalex": "S86672399"},
    {"name": "Language Testing in Asia", "issn": "2229-0443", "openalex": "S2736419388"},
    {"name": "Asian-Pacific Journal of Second and Foreign Language Education",
     "issn": "2363-5169", "openalex": "S3034690390"},
    {"name": "TESL-EJ", "issn": "1072-4303", "openalex": ""},
    {"name": "Journal of Language and Education", "issn": "2411-7390", "openalex": ""},
    {"name": "Eurasian Journal of Applied Linguistics", "issn": "2149-1135", "openalex": ""},
    {"name": "JALT CALL Journal", "issn": "1832-4215", "openalex": ""},
    # Ten entries verified against
    # api.openalex.org/sources/issn:<issn> under the production filter
    # (primary_location.source.issn:<issn>,open_access.is_oa:true,type:article,
    # publication_year:2019-2024). The original eight above are never reordered or edited.
    {"name": "Reading in a Foreign Language", "issn": "1539-0578", "openalex": "S5407051349"},
    {"name": "Australian Journal of Applied Linguistics", "issn": "2209-0959",
     "openalex": "S4210220284"},
    {"name": "Language Education and Assessment", "issn": "2209-3591",
     "openalex": "S4210222903"},
    {"name": "Technology in Language Teaching and Learning", "issn": "2652-1687",
     "openalex": "S4210196789"},
    {"name": "Apples: Journal of Applied Language Studies", "issn": "1457-9863",
     "openalex": "S2764441637"},
    {"name": "Colombian Applied Linguistics Journal", "issn": "0123-4641",
     "openalex": "S2764502050"},
    {"name": "PROFILE Issues in Teachers' Professional Development", "issn": "1657-0790",
     "openalex": "S2739108755"},
    {"name": "GEMA Online Journal of Language Studies", "issn": "1675-8021",
     "openalex": "S2738451673"},
    {"name": "Indonesian Journal of Applied Linguistics", "issn": "2301-9468",
     "openalex": "S2764809198"},
    {"name": "TESL Canada Journal", "issn": "0826-435X", "openalex": "S2764904417"},
)
# Observed 2026-09-02 through the production fetcher (recorded in hss_claims.build.json):
# LLT (scholarspace.manoa.hawaii.edu) answers 403; Language Testing in Asia and APJSFLE
# (springeropen.com track/pdf) return an HTML page instead of the PDF; MDPI "Languages" 403;
# Australian J. of Applied Linguistics resolves to HTML (confirmed again 2026-09-02 for the
# v3 addition above); IJLTR disconnects. TESL-EJ, jle.hse.ru, dergipark.org.tr and
# castledown.online serve PDFs directly. Three of the ten v3 additions (Apples, PROFILE,
# GEMA) are on castledown-publishers.com, the same family that serves PDFs directly above;
# Reading in a Foreign Language shares a host family with the 403 recorded for LLT. The
# probe (--probe-journals) decides the real outcome per journal before the budget is spent.

# Two candidates were researched and rejected before being added to JOURNALS (brief section
# 2.3): CALL-EJ, whose OpenAlex sources/issn: and api.crossref.org/journals/: paths both
# answer HTTP 404, so list_candidates has no queue on either path; and International Journal
# of Language Testing, whose OpenAlex record carries works_count 0 under the production
# filter. Recorded (never network-derived at run time) in every build json's
# ``journals_rejected``.
REJECTED_JOURNAL_CANDIDATES: tuple[dict[str, str], ...] = (
    {"journal": "CALL-EJ", "issn": "1442-438X",
     "reason": "sources/issn:1442-438X and api.crossref.org/journals/1442-438X both HTTP 404"},
    {"journal": "International Journal of Language Testing", "issn": "2476-5880",
     "openalex": "S7407062991", "reason": "works_count 0 under the production OA filter"},
)

# The sixth rule `over_specified`, appended after `altered` (never inserted between
# existing rules, so `hss-<rule>-<nn>` ids for the five original rules are unaffected). Its
# QUOTAS entry defaults to 0, so a caller that never passes --quotas builds the same 30 items
# in the same order as before -- byte identical.
RULE_ORDER: tuple[str, ...] = (
    "verbatim", "paraphrase", "altered", "over_specified", "wrong_paper", "no_full_text",
)
QUOTAS: dict[str, int] = {
    "verbatim": 5, "paraphrase": 5, "altered": 10, "over_specified": 0, "wrong_paper": 5,
    "no_full_text": 5,
}
FILL_ORDER: tuple[str, ...] = (
    "altered", "over_specified", "paraphrase", "verbatim", "wrong_paper", "no_full_text",
)
EXPECTED: dict[str, list[str]] = {
    "verbatim": ["verified"],
    "paraphrase": ["verified"],
    "altered": ["unsupported", "needs_nuance"],
    "over_specified": ["needs_nuance"],
    "wrong_paper": ["unsupported"],
    "no_full_text": ["no_full_text"],
}
PREFERRED_SECTION_RE = re.compile(r"results|findings|discussion|conclusion", re.IGNORECASE)

# A claim built by appending a stated setting, population
# qualifier, time window or instrument that the cited paper's own text never mentions -- the
# `needs_nuance` construction (the main finding still holds; the added detail does not).
# Frozen against development data only (its own sha256 is recorded in
# STIMULUS_CONSTANTS_SHA256 below); never changed once the held-out set is built.
OVER_SPECIFIED_MODIFIERS: tuple[str, ...] = (
    "among adult learners",
    "in an online setting",
    "during the winter semester",
    "using a mobile application",
    "in a bilingual classroom",
    "over a two-year period",
    "among graduate students",
    "in a rural school",
    "using eye-tracking software",
    "during a single class period",
    "among heritage speakers",
    "in a study-abroad context",
)
# Function words stripped before checking a modifier's content tokens against the cited
# paper's text: a modifier is rejected only when one of its *substantive* words (not "in",
# "a", "the", ...) already appears in the paper.
OVER_SPECIFIED_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "in", "on", "at", "of", "for", "with", "and", "or", "to", "from",
    "this", "that", "these", "those", "using", "via", "by", "as", "into", "over", "under",
    "among", "during",
})

# Ordered, whole-word, case-preserving substitutions for the ``paraphrase`` rule. Only
# reporting verbs, discourse connectives and generic nouns are substituted; technical terms
# (effect, increase, significant, participants, ...) are never touched, so a paraphrase is
# faithful by construction and its expected status is ``verified``.
SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("showed", "demonstrated"),
    ("shows", "demonstrates"),
    ("revealed", "demonstrated"),
    ("found", "observed"),
    ("students", "learners"),
    ("results", "findings"),
    ("suggest", "indicate"),
    ("suggests", "indicates"),
    ("teachers", "instructors"),
    ("study", "investigation"),
    ("in addition", "moreover"),
    ("however", "nevertheless"),
    ("furthermore", "moreover"),
    ("therefore", "consequently"),
)
# A substitution must not split a fixed term: "case study" / "pilot study" / "cohort
# study" keep their noun (SUBSTITUTIONS are otherwise whole-word and case-preserving).
SUBSTITUTION_GUARDS: dict[str, str] = {
    "study": r"(?<!\bcase )(?<!\bpilot )(?<!\bcohort )(?<!\blongitudinal )(?<!\bfield )",
}
# "results" -> "findings" is meant for the noun sense ("the results showed ..."); the same
# whole-word substitution also matches the verb sense ("which results from/in X"), where
# "findings" is not a verb and the substitution produces an ungrammatical, meaning-altering
# sentence (observed on dev-set item hss-dev-30 "paraphrase-01"). Guarded with a
# lookahead, mirroring DIRECTION_GUARDS's use of post_guard below.
SUBSTITUTION_POST_GUARDS: dict[str, str] = {
    "results": r"(?!\s+(?:from|in)\b)",
}
# Ordered direction flips for the ``altered`` rule (first applicable pair, first occurrence).
DIRECTION_FLIPS: tuple[tuple[str, str], ...] = (
    ("increased", "decreased"), ("decreased", "increased"),
    ("increases", "decreases"), ("decreases", "increases"),
    ("increase", "decrease"), ("decrease", "increase"),
    ("increasing", "decreasing"), ("decreasing", "increasing"),
    ("higher", "lower"), ("lower", "higher"),
    ("positively", "negatively"), ("negatively", "positively"),
    ("positive", "negative"), ("negative", "positive"),
    ("improved", "worsened"), ("worsened", "improved"),
    ("improve", "worsen"), ("worsen", "improve"),
    ("improvement", "decline"), ("outperformed", "underperformed"),
    ("better", "worse"), ("worse", "better"),
    ("greater", "smaller"), ("stronger", "weaker"), ("weaker", "stronger"),
    ("faster", "slower"), ("slower", "faster"),
    ("more", "less"), ("fewer", "more"), ("less", "more"),
    ("statistically significant", "not statistically significant"),
    ("significantly", "not significantly"),
    ("is significant", "is not significant"), ("are significant", "are not significant"),
    ("was significant", "was not significant"), ("were significant", "were not significant"),
    ("significant", "no significant"),
)
QUANTIFIER_FLIPS: tuple[tuple[str, str], ...] = (
    ("all of", "none of"), ("all", "no"), ("most", "few"), ("majority", "minority"),
    ("always", "never"), ("every", "no"), ("often", "rarely"), ("frequently", "rarely"),
)
NEGATION_GUARD = r"(?<!\bno )(?<!\bnot )(?<!\bnon-)"
# A quantifier flip must stay a quantifier: "the most important" is a superlative and
# "at all" an idiom, neither says how many. "all" in object position right after a benefactive
# or purposive preposition ("for all learners", "developed to all students", "shared among
# all colleagues", "compatible with all versions", "consistent across all sites") reads as a
# scope, not a count claimed by the sentence: flipping it produces a claim close to a
# contradiction in terms rather than a checkable negation (e.g. "... illustrate the potential
# of developing inclusive practices for all
# learners" -> "... for no learners"). Subject-position "All learners preferred ..." is
# unaffected: nothing precedes "All" there.
QUANTIFIER_GUARDS: dict[str, str] = {
    "most": r"(?<!\bthe )",
    "all": r"(?<!\bat )(?<!\bfor )(?<!\bto )(?<!\bamong )(?<!\bwith )(?<!\bacross )",
}
# The object-position guard above
# was keyed on the literal source token "all", so "all of" and "every" -- the identical
# construction in the identical syntactic position ("cater to every single variety", "for all
# of the students") -- had no guard at all, because ``QUANTIFIER_GUARDS.get(src, "")`` never
# matched their spelling. Every quantifier phrase in :data:`QUANTIFIER_FLIPS` that denotes the
# same universal quantifier is aliased to "all" here, so the lookup below always finds the
# benefactive/purposive lookbehind regardless of which spelling the flip pair uses -- and,
# because the guard lives in the compiled pattern itself (not in a first-match-only check), a
# guarded occurrence is never matched at all, so the fill loop cannot hand the flip to a
# second, later quantifier phrase in the same sentence either (direction flips close the
# same escape with a sentence-level veto; here the per-phrase guard already closes it).
# "each" and "none of"/"no" are included for the same phrase family even though no
# :data:`QUANTIFIER_FLIPS` entry currently sources from them, so any future flip added for
# them inherits the guard automatically rather than needing a fresh fix.
QUANTIFIER_GUARD_ALIASES: dict[str, str] = {
    "all of": "all", "every": "all", "each": "all", "none of": "all", "no": "all",
}
# "all but one" is a partitive, not the quantifier "all"; negating it produces "no but one",
# which is not a sentence of English (e.g. "All but one of the motivational drivers" -> "No
# but one of the motivational drivers"). Checked as a post-guard (right after the matched
# word), so "All learners" (no
# "but" following) still flips normally.
QUANTIFIER_POST_GUARDS: dict[str, str] = {"all": r"(?!\s+but\b)"}
# A quantifier flip whose *destination* is itself a negation ("no", "never", "none of") must
# not fire when the sentence already carries that same negation elsewhere: the result asserts
# the same negation twice, over two different clauses, reading as self-contradictory rather
# than as a single checkable claim -- the same defect class :data:`NEGATED_SIGNIFICANT_RE`
# guards for "significant", generalised (e.g. "unequal power relations always exist ... has
# never been resolved" -> flipping "always" to "never" produces two "never"s governing two
# different clauses).
QUANTIFIER_NEGATION_DESTINATIONS: frozenset[str] = frozenset({"no", "never", "none of"})


def _quantifier_already_negated(text: str, dst: str) -> bool:
    """True when *text* already contains *dst* (one of
    :data:`QUANTIFIER_NEGATION_DESTINATIONS`) as a whole word/phrase; always ``False`` for any
    other *dst*, so a non-negating flip (e.g. "most"->"few") is never blocked by this check."""
    if dst not in QUANTIFIER_NEGATION_DESTINATIONS:
        return False
    return re.search(r"\b" + re.escape(dst) + r"\b", text, re.IGNORECASE) is not None
# "significantly" either postmodifies a verb ("differed significantly": negating it to
# "differed not significantly" is ungrammatical -- the idiomatic negation is "did not differ
# significantly", which this rule cannot produce) or premodifies an adjective/participle
# ("significantly higher/different"), which negates cleanly ("not significantly
# higher/different"). Applied as a *post*-guard (checked right after the matched word), so the
# flip only fires in the second case; DIRECTION_FLIPS entries earlier in the list already
# cover the direction words that would otherwise double as the following adjective.
DIRECTION_GUARDS: dict[str, str] = {
    "significantly": r"(?=\s+(?:higher|lower|greater|smaller|more|less|better|worse|"
                      r"different|correlated|associated|improved|increased|decreased|"
                      r"related|predicted|affected)\b)",
}
# "significant" -> "no significant" is attributive ("a significant difference"); negating it
# while leaving a leading "a"/"an"/"the" produces a double determiner ("a no significant
# difference"). The article is consumed into the match so it is replaced along with the
# word ("a significant difference" -> "no significant difference"); no article -> unaffected.
DIRECTION_ARTICLE_STRIP: frozenset[str] = frozenset({"significant"})
# A word inside (or immediately framing) a parenthetical alternative, "rated higher (or
# lower) in a factor analysis", must never be flipped: flipping either side produces a
# duplicated pair, "lower (or lower)" or "higher (or higher)" (e.g. "rated significantly lower
# (or lower)"). The pre-guard
# protects the word *inside* the parenthetical ("lower" in "(or lower)"); the post-guard
# protects the word the parenthetical follows ("higher" before "(or lower)").
DIRECTION_PAREN_OR_GUARD = r"(?<!\(or )"
DIRECTION_PAREN_OR_POST_GUARD = r"(?!\s*\(or\s+\w+\))"
# The per-word guards above stop a flip landing *on* the word framing or inside "(or X)", but
# the loop then simply advances to the next applicable :data:`DIRECTION_FLIPS` pair, which can
# still flip an unrelated word in the same sentence and leave the degenerate parenthetical
# untouched (e.g. "rated significantly higher (or lower)" survived
# "higher"->"lower" being guarded only to have "significantly"->"not significantly" fire next,
# producing "rated not significantly higher (or lower)" -- the base sentence, a description of
# an analysis procedure rather than a finding, is unfit for this construction regardless of
# which word is flipped). Checked once for the whole sentence: any "(or <word>)" anywhere vetoes
# every direction flip for that sentence, not just the word it frames.
DIRECTION_PAREN_OR_SENTENCE_RE = re.compile(r"\(or\s+\w+\)", re.IGNORECASE)
# A sentence that already carries "no significant"/"not significant" must never also flip a
# second, unrelated "significant"/"significantly" into a negated form: the result reads as if
# the sentence negates the same finding twice (e.g. "'no
# significant difference' flipped twice"). This guard extends from
# "significant" to "significantly", the same double-negation risk one suffix letter away.
# Checked once for the whole sentence, not per-match, since the concern is sentence-level
# double negation, not a specific word's local context.
NEGATED_SIGNIFICANT_RE = re.compile(r"\bno\s+significant\b|\bnot\s+significant\b", re.IGNORECASE)
# A cross-reference label ("Excerpt 2", "Table 3", "Figure 4", "Section 5", "Study 2") is an
# identifier, not a substantive quantity; flipping its number changes what the sentence points
# at rather than what it claims (e.g. "Excerpt
# 2" -> "Excerpt 3"). Also covers "session"/"phase"/"round"/"stage"/
# "wave"/"cycle"/"week"/"day"/"item"/"task"/"condition"/"group" (e.g.
# "session 4" -> "session 5", a real four-session study; the label always
# *precedes* the number, so a duration phrased the other way round -- "6 weeks", "three
# sessions" -- is unaffected, since the guard only matches text ending in the label word).
CROSS_REFERENCE_LABEL_RE = re.compile(
    r"\b(?:excerpt|table|fig(?:ure)?|section|sec|study|appendix|chapter|example|equation|eq|"
    r"session|phase|round|stage|wave|cycle|week|day|item|task|condition|group)"
    r"\.?\s*$",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(\s?(?:%|percent))?")
PVALUE_BEFORE_RE = re.compile(r"[pP]\s*[<=>]\s*$")
# A page number ("p. 45", "pp. 12-14") inside a citation identifies where to find something,
# not a substantive quantity the source sentence asserts. The optional trailing "<number>-"
# clause guards the *second* number of a
# range too ("pp. 12-14"'s "14" is checked against "...pp. 12-", not just "...pp. ").
PAGE_NUMBER_BEFORE_RE = re.compile(
    r"\bpp?\.\s*(?:\d+(?:\.\d+)?\s*[-–]\s*)?$", re.IGNORECASE,
)
# A sample-size marker inside a citation ("N = 45", "n = 6") names how many participants a
# *cited* study had, not a quantity the source sentence itself asserts.
SAMPLE_SIZE_BEFORE_RE = re.compile(r"\(\s*[Nn]\s*=\s*$")
HYPHEN_BREAK_RE = re.compile(r"[A-Za-z]- [A-Za-z]")  # "ac- cused": PDF line-break hyphenation
# A soft hyphen (U+00AD, invisible unless the PDF actually broke the line there) surviving
# extraction, with or without the accompanying space: "anal­ yses", "presenta­ tion"
# Its mere presence -- not just the
# broken-word shape -- is rejected: a soft hyphen never belongs in clean running prose.
SOFT_HYPHEN_RE = re.compile("\u00ad")
NUMERIC_TOKEN = r"\.?\d+(?:\.\d+)?"
NUMERIC_TOKEN_RE = re.compile(r"(?<![\w.])" + NUMERIC_TOKEN + r"(?![\w])")
ADJACENT_NUMBERS_RE = re.compile(NUMERIC_TOKEN + r"\s+" + NUMERIC_TOKEN)  # "post .17 45 .183"
FOOTNOTE_MARK_RE = re.compile(r"[A-Za-z]\.\d")  # "the UK.2 Most participants"
# A superscript footnote-reference digit flattened directly onto a word by PDF extraction,
# with no intervening period at all ("classrooms1", "the UK1"): distinct from
# ``FOOTNOTE_MARK_RE``'s "the UK.2" shape (a period already present) and from a genuine short
# alphanumeric token such as "L2"/"T2" (a single capital letter plus one digit; the 4-letter
# floor here excludes it). A single trailing digit only (``(?![0-9])``) so an ordinary word
# followed by a real multi-digit number stays out of scope (e.g. "different SFI classrooms1,
# we found that ...").
FOOTNOTE_DIGIT_GLUED_RE = re.compile(r"[a-z]{4,}[1-9](?![0-9])\b")
KEYWORDS_RE = re.compile(r"\bkeywords\s*:", re.IGNORECASE)
ELLIPSIS_RE = re.compile(r"(?:\.\s?\.\s?\.|\u2026)\s*$")  # ". . ." / "..." / "..." (U+2026)
# Reference-list fragments: "Author, A. B., & Other, C. (2004). Title." / "Journal, 74(1),
# 59-109." / "Fall 2015 survey results."
REFERENCE_RE = re.compile(
    r"(?:\b[A-Z][A-Za-z'\-]+,\s(?:[A-Z]\.\s?)+.*\(\d{4}[a-z]?\)\.)"
    r"|(?:\(\d{4}[a-z]?\)\.\s+[A-Z])"
    r"|(?:\b\d{4}\s+survey\s+results\.)"
    r"|(?:\b\d{1,3}\(\d{1,2}\),\s\d+[-\u2013]\d+\.)"
)
# "; " followed by a Title-Case run of >= 4 capitalised words (reference / running header)
TITLE_CASE_AFTER_SEMICOLON_RE = re.compile(r";\s+(?:[A-Z][\w'\-]*\s+){3,}[A-Z][\w'\-]*")
# A section heading glued to the following sentence: "... Learning Experience The L2 ..."
HEADING_MERGE_RE = re.compile(r"[a-z]\s+(?:The|A|An|This|These|In|On|Of)\s+[A-Z0-9]")
# ... or glued to its front: "Conclusion There is no doubt that ..."
# ("Conclusion There is ...", "Theoretical Frameworks A variety of ...": a known heading word,
# or a Title-Case run, followed by a capitalised function word that starts the sentence).
# The Title-Case run may interleave a small closed set of lowercase function words
# (e.g. "Impact of the Classroom Language
# Assessment Course on Pre-Service Teachers As the data below indicate, ..." was not rejected
# because the run "Impact ... Teachers" mixes capitalised words with "of"/"the"/"on", which the
# original ``(?:[A-Z][a-z\-]+\s+){1,4}`` (capitalised words only, at most four) never matches.
# The sentence-opener alternation is unchanged and still exact-case ("The", "In", ...), so a
# genuine capitalised subject phrase such as "The Classroom Language Assessment Course was
# piloted ..." stays eligible: its lowercase predicate verb never matches that alternation.
HEADING_PREFIX_RE = re.compile(
    r"^(?:Conclusions?|Discussion|Results?|Findings|Introduction|Background|Method(?:s|ology)?|"
    r"Abstract|Summary|Implications|Limitations|Analysis|Participants|Procedure|Instruments?|"
    r"Materials|Measures)\s+[A-Z]"
    r"|^[A-Z][\w\-]*\s+(?:(?:[A-Z][\w\-]*|(?:of|the|on|in|for|and|to|a|an))\s+){0,12}"
    r"(?:A|An|The|This|These|It|We|There|Our|Some|Many|Most|"
    r"Several|Although|While|As|For|In|On|Of)\s+[a-z]"
)
# A numbered heading glued to the front of a sentence that repeats the same label a second
# time: "Factor 3: Activity-induced boredom Factor 3 explained 12% of the variance ..."
# (HEADING_PREFIX_RE only covers a known heading word or a
# Title-Case run, neither of which matches "Factor 3:"). Requiring the exact "Word N" pair to
# repeat keeps this narrow: a genuine mention of "Group 2 participants ... Group 1" never
# repeats the identical label twice in one sentence.
HEADING_NUMBERED_REPEAT_RE = re.compile(r"^([A-Z][A-Za-z]+)\s+(\d+):\s.*?\b\1\s+\2\b")
# A reference-list / book title that survives sentence splitting with its bracketed
# translation still attached: "Klassrumsinteraktion och flerspraakighet: Ett kritiskt
# perspektiv [Classroom interaction and multilingualism: A critical perspective]."
# (a bare title, not a claim). The
# chunk-level reference filter (``is_reference_or_frontmatter_chunk``) is the primary defence;
# this is a sentence-level backstop for a mixed prose/reference chunk.
BARE_TITLE_BRACKETED_RE = re.compile(r":\s.*\[[^\[\]]+\]\.\s*$")
# A numbered or dashed list item's category label glued to its description once the sentence
# splitter has stripped the list's own numbering: "Sentence and Text Editing - ChatGPT helped
# ..." was list item 3 of an enumerated inventory. A short Title-Case run (the label, which may
# interleave
# a small set of lowercase function words the way HEADING_PREFIX_RE's own Title-Case branch
# does) immediately followed by an en dash, em dash or hyphen-as-dash *with whitespace on both
# sides* and a capitalised word is not the sentence's own grammatical subject. The dash must be
# surrounded by spaces on both sides: a genuine hyphenated compound whose second element happens
# to be capitalised ("Ten Anglo-Canadian students ...", "Spanish-English bilinguals ...",
# "Non-Native speakers ...", "L2-English writers ...") glues the dash directly onto both
# neighbouring words with no surrounding whitespace, so it never matches this pattern regardless
# of the case of the character after the dash (the previous
# comment's claim that the character after the dash is always lowercase or a digit for a genuine
# compound was false whenever the compound's second element is itself capitalised, which cost 70
# of 9,582 candidate sentences in the committed caches, mostly bilingual/national-population
# descriptions, a spurious rejection).
LIST_LABEL_DASH_RE = re.compile(
    r"^[A-Z][\w'/]*(?:\s+(?:[A-Z][\w'/]*|and|of|the|in|on|for|to|a|an)){0,6}\s+[–—-]\s+[A-Z]"
)
# The whitespace tightening above (``\s*`` -> ``\s+`` around
# the dash) was necessary to stop rejecting a genuine hyphenated compound ("Anglo-Canadian"),
# but it was also, undocumented, the *only* thing that had been rejecting a PDF running header
# such as "TESL-EJ 26.3, November 2022 \nFang & Xu \n \n12 \nTo Cite this Article ..." --
# after whitespace normalisation, "TESL-EJ 26.3, November 2022 Fang & Xu 12 To Cite this
# Article Fang, F. & Xu, Y." The loose ``\s*`` dash spacing let the zero-repetition case of the
# Title-Case-run group match the journal abbreviation's own internal hyphen ("TESL-EJ" alone
# satisfies "Title-Case word, dash, capital word"), which is unrelated to this pattern's actual
# purpose (a list label glued to its description) but coincidentally screened out every running
# header of this shape too. Tightening the spacing correctly stopped that accidental catch,
# releasing 13 running-header sentences across the three committed caches into the candidate
# pool undetected by any other guard (e.g. hss-altered-01 and hss-altered-02
# of a rebuilt hss-dev-band2 moved from real claim sentences to a mutated journal-volume number
# in "TESL-EJ 26.3, November 2022 Fang & Xu 12 To Cite this Article Fang, F. & Xu, Y." and
# "TESL-EJ 28.1, May 2024 Van Horn 14 Several groups of students ..."). A running header has its
# own distinct, narrow shape -- a hyphenated journal abbreviation immediately followed by a
# "volume.issue, Month year" citation line -- that a genuine hyphenated compound never has
# (nothing in "Anglo-Canadian students ..." or "COVID-19 disrupted ..." looks like a volume and
# issue number), so this is a dedicated pattern rather than a further change to
# ``LIST_LABEL_DASH_RE`` itself.
RUNNING_HEADER_VOLUME_RE = re.compile(
    r"^[A-Z][A-Za-z]*-[A-Z]{2,}\s+\d+\.\d+,\s+\w+\s+\d{4}\b"
)
# A sentence opening with a bare description of a study's sample ("Participants were ...")
# has no subject of its own; when the immediately preceding sentence in the same chunk names
# another study (:data:`ATTRIBUTION_RE`), the elided subject belongs to that other study, not
# to the source paper (e.g.
# "Participants were 44 adult learners of FL Arabic ..." describes Al Khalil (2016), named
# only in "Al Khalil (2016) used a motivational thermometer ...", the sentence before it).
ELIDED_SUBJECT_OPENER_RE = re.compile(
    r"^(?:Participants?|The\s+participants?|The\s+sample)\s+(?:were|included|consisted\s+of|"
    r"comprised)\b",
    re.IGNORECASE,
)
# Anaphoric noun phrases whose referent lies outside the sentence ("this number", "these
# findings"): eligible only when the sentence itself carries an antecedent -- a numeral, or a
# preceding clause (>= 4 words before a comma) the phrase can point back to.
ANAPHORA_RE = re.compile(
    r"\b(?:this|these|that|those)\s+(?:numbers?|findings?|results?|figures?|values?|outcomes?|"
    r"percentages?|proportions?|rates?)\b",
    re.IGNORECASE,
)
# Author-stance openers are opinions about an unstated referent, not checkable claims.
STANCE_OPENER_RE = re.compile(r"^(?:We|I)\s+(?:believe|think|feel|suspect|assume)\b")
# Attributed findings: a literature-review sentence reports *another* study's result or
# procedure ("Smith (2019) found ...", "Smith (2019) recruited ...", "The authors found
# ...", "his contribution proved ...", "Some excellent examples (...) show ...", "In
# Smith's (2019) study ...", "the study by Smith (2019)").
# Cited to the HSS paper it attributes the finding to the wrong primary source, so it is
# not a usable 'verified' stimulus. A parenthetical citation after the claim ("..., in line
# with earlier work (Smith et al., 2019)") and "the authors of this study" stay eligible.
_REPORTING_VERBS = (
    r"(?:found|finds?|indicated?s?|show(?:ed|s|n)?|argued?s?|reported?s?|demonstrated?s?|"
    r"suggested?s?|claimed?s?|noted?s?|observed?s?|concluded?s?|revealed?s?|proposed?s?|"
    r"stated?s?|pointed|points|maintained?s?|confirmed?s?|examined?s?|investigated?s?|"
    r"explored?s?|highlighted?s?|emphasi[sz]ed?s?|discovered?s?|identified|identif(?:y|ies)|"
    r"proved?s?|established?s?|documented?s?|contended?s?|asserted?s?|posited?s?)"
)
# Several ``_REPORTING_VERBS`` entries above are built as
# "stem+d?+s?", which only covers the bare present-tense form when the stem itself already
# ends in "e" ("argue"/"argued"/"argues"); for a stem such as "report" or "suggest" this
# leaves the plain present tense a plural subject takes ("they report", "the authors
# suggest") and the correctly-inflected "-s" form unmatched (matching only "reporte(d?)(s?)"
# literally). ``_REPORTING_VERBS`` itself is left as-is here, since five other ATTRIBUTION_RE
# branches already depend on its exact matching behaviour and this session's byte-identical
# guarantee is for the frozen 30-item set, not for widening every branch's recall; this
# correctly-formed list is used only by the new pronoun-subject alternative below (e.g.
# "they report on the benefits ...").
_PRONOUN_REPORTING_VERBS = (
    r"(?:found|finds?|show(?:ed|s|n)?|argue[ds]?|report(?:ed|s)?|indicate[ds]?|"
    r"demonstrate[ds]?|suggest(?:ed|s)?|claim(?:ed|s)?|note[ds]?|observe[ds]?|conclude[ds]?|"
    r"reveal(?:ed|s)?|propose[ds]?|state[ds]?|maintain(?:ed|s)?|confirm(?:ed|s)?|"
    r"examine[ds]?|investigate[ds]?|explore[ds]?|highlight(?:ed|s)?|emphasi[sz]e[ds]?|"
    r"discover(?:ed|s)?|prove[ds]?|establish(?:ed|es)?|document(?:ed|s)?|contend(?:ed|s)?|"
    r"assert(?:ed|s)?|posit(?:ed|s)?|identif(?:y|ies|ied))"
)
# A cited study's *procedure* ("Smith (2019) recruited ...", "Newcomer and Collier (2015),
# however, recruited ...") is another study's method, not the paper's own claim.
_PROCEDURE_VERBS = (
    r"(?:recruited|sampled|surveyed|interviewed|administered|collected|conducted|carried\s+out|"
    r"employed|adopted|used|utili[sz]ed|analy[sz]ed|measured|tested|compared|randomi[sz]ed|"
    r"assigned|enrolled|selected|implemented|designed|developed|applied|trained|taught|"
    r"observed|followed|assessed|evaluated|included|excluded|divided|split|recorded)"
)
_CITED_VERBS = r"(?:" + _REPORTING_VERBS + r"|" + _PROCEDURE_VERBS + r")"
# A floating quantifier ("who all identified", "the students all reported") sits between a
# subject and its own finite verb rather than before a noun phrase: "all"/"every" there is an
# adverbial, not a determiner, and flipping it to "no" produces "who no identified", not a
# sentence of English regardless of which verb follows. The determiner reading ("all the
# students identified as
# ...") is unaffected because a determiner/article intervenes between the quantifier and the
# verb there, so this post-guard (checked immediately after the matched word, no intervening
# token allowed) never fires on it. The verb list reuses ATTRIBUTION_RE's own reporting/
# procedure verbs plus a short list of common finite verbs and auxiliaries.
_FLOATING_QUANTIFIER_VERB_RE = (
    r"(?:" + _CITED_VERBS + r"|is|are|was|were|have|has|had|do|does|did|can|could|will|would|"
    r"shall|should|must|might|may|agreed?s?|responded?s?|participated?s?|felt|believe[ds]?|"
    r"said|preferred?s?|scored?s?|completed?s?|attended?s?|enrolled?s?|wanted?s?|needed?s?|"
    r"liked?s?|disliked?s?|struggled?s?|managed?s?)\b"
)
QUANTIFIER_FLOATING_POST_GUARDS: dict[str, str] = {
    "all": r"(?!\s+" + _FLOATING_QUANTIFIER_VERB_RE + r")",
    "every": r"(?!\s+" + _FLOATING_QUANTIFIER_VERB_RE + r")",
}
_ADVERBS = (
    r"(?:has|have|had|also|further|later|recently|similarly|likewise|however|then|thus|"
    r"therefore|nevertheless|instead|in\s+contrast|by\s+contrast|on\s+the\s+other\s+hand)"
)
_NAME = r"[A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+){0,2}"
# A comma-separated author list of three or more ("Arias, Maturana and Restrepo"), as opposed
# to a single "and"/"&"-joined pair: "the study of NAME (YEAR)"
# matched only a two-name pair because _NAME allows only one internal "and"/"&".
_NAME_LIST = _NAME + r"(?:,\s*" + _NAME + r")*(?:,?\s*(?:and|&)\s*" + _NAME + r")?"
# Possessive or "the study by" citations: "In Smith's (2019) study ...", "the work of
# Smith et al. (2019)" frame the sentence as a description of another study.
_STUDY_NOUNS = (
    r"(?:stud(?:y|ies)|work|research|investigations?|papers?|articles?|analys[ie]s|"
    r"experiments?|surveys?|reviews?|projects?|reports?|findings?|results?|data|sample|"
    r"participants|model|framework|taxonomy)"
)
# A parenthetical citation year, generalised to the forthcoming-publication forms a four-digit
# year alone never matches: "(in press)", "(n.d.)", "(forthcoming)", "(in preparation)" (e.g.
# "Pontier & Tian (in press) found
# that ..." slipped through every ATTRIBUTION_RE branch below because each one required
# ``\(\d{4}[a-z]?\)`` literally).
_CITATION_YEAR = r"(?:\d{4}[a-z]?|n\.d\.|in\s+press|forthcoming|in\s+preparation)"
ATTRIBUTION_RE = re.compile(
    r"\b" + _NAME + r"(?:\s+(?:and|&)\s+" + _NAME + r"|\s+et\s+al\.?)?\s*\(" + _CITATION_YEAR
    + r"\)\s*,?\s*(?:" + _ADVERBS + r"\s*,?\s*)*" + _CITED_VERBS + r"\b"
    r"|\b" + _NAME + r"(?:\s+(?:and|&)\s+" + _NAME + r"|\s+et\s+al\.?)?\s*['\u2018\u2019\u201f]s?"
    r"\s*\(" + _CITATION_YEAR + r"\)\s*" + _STUDY_NOUNS + r"\b"
    r"|\b(?:in|by|from|see|following|according\s+to)\s+" + _NAME
    + r"(?:\s+(?:and|&)\s+" + _NAME + r"|\s+et\s+al\.?)?\s*\(" + _CITATION_YEAR + r"\)"
    r"|\b(?:[Tt]wo|[Tt]hree|[Ff]our|[Ff]ive|[Ss]everal|[Bb]oth|[Tt]hese|[Tt]hose|[Ss]ome|"
    r"[Mm]any|[Oo]ther|[Pp]revious|[Rr]ecent|[Ee]arlier|[Ss]imilar)\s+(?:studies|papers|"
    r"articles|reports|investigations|reviews|authors|researchers|scholars)\s*\([^()]*"
    r"\b(?:19|20)\d{2}[a-z]?\b"
    r"|\b" + _STUDY_NOUNS + r"\s+(?:by|of|from)\s+(?:" + _NAME_LIST + r"|" + _NAME
    + r"\s+et\s+al\.?)\s*\(" + _CITATION_YEAR + r"\)"
    r"|(?:^|[,;:]\s+)[Tt]he\s+(?:authors?|researchers?|scholars?|investigators?)\b"
    r"(?!\s+of\s+(?:this|the\s+present|the\s+current)\b)"
    # "These/Those/Such authors found ..." (deliverable 3): the demonstrative substitutes for
    # "The" and names no antecedent of its own within the sentence.
    r"|(?:^|[,;:]\s+)(?:[Tt]hese|[Tt]hose|[Ss]uch)\s+"
    r"(?:authors?|researchers?|scholars?|investigators?)\b"
    r"|\b(?:[Rr]ecent|[Pp]revious|[Pp]rior|[Ee]arlier|[Pp]ast|[Ee]xisting|[Oo]ther|[Mm]uch|"
    r"[Ss]ome|[Ss]everal|[Mm]any|[Nn]umerous)\s+(?:[\w\-]+\s+)?(?:research|stud(?:y|ies)|work|"
    r"literature|scholarship|researchers|scholars|authors|examples|papers|articles|reports|"
    r"reviews|investigations)\s+(?:\([^()]*\)\s+)?(?:(?:has|have|had|also|further|"
    r"consistently|repeatedly)\s+)*" + _REPORTING_VERBS + r"\b"
    r"|^(?:Research|Studies|Scholars|Researchers|The\s+literature)\s+"
    r"(?:(?:has|have|also|consistently|repeatedly)\s+)*" + _REPORTING_VERBS + r"\b"
    r"|\b(?:[Hh]is|[Hh]er|[Tt]heir)\s+(?:contributions?|stud(?:y|ies)|work|papers?|research|"
    r"findings?|results?|analys[ie]s|investigations?|experiments?|survey)\b"
    # A bare pronoun subject ("They report ...", "He found ...") with a reporting verb: the
    # antecedent is unresolvable from the sentence alone, the same defect class "The authors
    # found ..." already covers (e.g.
    # "Drawing on a corpus of 30 studies published in English, they report on the benefits and
    # challenges ..."). A non-reporting verb ("They were asked to ...") is unaffected.
    r"|(?:^|[,;:]\s+|\.\s+)(?:[Tt]hey|[Hh]e|[Ss]he)\s+"
    r"(?:(?:also|further|then|subsequently)\s+)*" + _PRONOUN_REPORTING_VERBS + r"\b"
)
# Quoted passages: a block quotation is closed by a parenthetical citation with a page
# reference ("(Dewaele & MacIntyre, 2016. p. 262)", "(Smith, 2019, pp. 12-14)"); a sentence
# followed by one, or preceded in its chunk by an unclosed quotation mark, is another
# author's text and never a stimulus (candidates_for_paper).
QUOTE_CITATION_RE = re.compile(
    r"^\s*\(\s*[^()]*\b(?:19|20)\d{2}[a-z]?\s*[.,]?\s*pp?\.\s*\d+"
)
QUOTE_CITATION_WINDOW = 60
# A block quotation introduced by a colon ("... in the interview:", "... pointed out the
# following in the interview:") but never closed by a page-referenced citation or a quotation
# mark: this is a study participant's own words, not the source paper's own claim (e.g.
# "The teacher pointed out the following
# in the interview: I used ChatGPT outputs ... While I always ensured to review and validate
# the suggestions, ChatGPT's assistance saved me time ..."). Bounded to a fixed lookback
# window (:data:`QUOTE_COLON_INTRO_WINDOW`) rather than scanned back to the chunk's start,
# since nothing in plain extracted text reliably marks where such an unmarked quotation ends.
QUOTE_COLON_INTRO_RE = re.compile(
    r"\b(?:in\s+the\s+interview|said\s+the\s+following|stated\s+the\s+following|"
    r"wrote\s+the\s+following|(?:pointed\s+out|noted|remarked|commented|explained|"
    r"responded|replied|reported)\s+the\s+following)\s*:",
    re.IGNORECASE,
)
QUOTE_COLON_INTRO_WINDOW = 300
# Direction and quantifier flips are not applied inside a negated frame: "it is not known if
# X are higher" stays compatible with the source whichever way X is flipped, and "our aim is
# not to create the illusion that all Y" is compatible with the source whatever the quantifier
# (e.g. "our aim is not to create the illusion
# that all authors share a unified view" -> a quantifier flip inside that clause is the same
# defect class as the "not known if" case already guarded here).
HEDGE_RE = re.compile(
    r"\b(?:not|never)\s+(?:known|clear|certain|established)\s+(?:if|whether)\b"
    r"|\b(?:unclear|unknown|uncertain)\s+(?:if|whether)\b"
    r"|\b(?:is|was|are|were)\s+not\s+to\s+\w+\s+the\s+illusion\s+that\b"
    r"|\b(?:aim|purpose|intention)s?\s+(?:is|was|are|were)\s+not\s+to\b[^.?!]*?\bthat\b",
    re.IGNORECASE,
)


def _hedge_split(text: str) -> tuple[str, str]:
    """(head, tail) split at the first :data:`HEDGE_RE` match; ``(text, "")`` when none. Only
    ``head`` is eligible for a direction or quantifier flip."""
    hedge = HEDGE_RE.search(text)
    return (text[: hedge.start()], text[hedge.start():]) if hedge else (text, "")


TITLE_PREFIX_CHARS = 40
MIN_ALPHA_RATIO = 0.72
MAX_NUMERIC_TOKENS = 3


# --------------------------------------------------------------------------------------
# Pure construction helpers
# --------------------------------------------------------------------------------------


def doi_slug(doi: str) -> str:
    """``10.14746/ssllt.2020.10.1.2`` -> ``10-14746_ssllt-2020-10-1-2`` (cache file stem)."""
    out = doi.strip().lower().replace("/", "_")
    return re.sub(r"[^a-z0-9_]", "-", out)


def _match_case(src_word: str, dst: str) -> str:
    return dst[0].upper() + dst[1:] if src_word[:1].isupper() else dst


def _sub_word(
    text: str, src: str, dst: str, *, guard: str = "", post_guard: str = "", count: int = 0,
    strip_article: bool = False,
) -> tuple[str, int]:
    prefix = guard + (r"(?:\b(?:a|an|the)\s+)?" if strip_article else "")
    pattern = re.compile(prefix + r"\b" + re.escape(src) + r"\b" + post_guard, re.IGNORECASE)
    n = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal n
        n += 1
        return _match_case(m.group(0), dst)

    return pattern.sub(repl, text, count=count), n


def paraphrase(sentence: str) -> tuple[str, list[tuple[str, str]]]:
    """Apply every entry of ``SUBSTITUTIONS`` in order (``SUBSTITUTION_GUARDS`` protect fixed
    terms such as "case study"; ``SUBSTITUTION_POST_GUARDS`` protects a different part of
    speech such as "results from/in"); returns (text, substitutions made)."""
    text = normalise_whitespace(sentence)
    made: list[tuple[str, str]] = []
    for src, dst in SUBSTITUTIONS:
        if dst.casefold() in text.casefold():
            continue  # never double a qualifier ("statistically statistically significant")
        text, n = _sub_word(
            text, src, dst, guard=SUBSTITUTION_GUARDS.get(src, ""),
            post_guard=SUBSTITUTION_POST_GUARDS.get(src, ""),
        )
        if n:
            made.append((src, dst))
    return text, made


def _alter_direction(text: str) -> tuple[str, str] | None:
    """First applicable flip of ``DIRECTION_FLIPS``; never inside a negated frame (:func:`
    _hedge_split`: only the text *before* "it is not known if/whether ..." or "our aim is not
    to create the illusion that ..." is eligible), never negates "significantly" unless it
    premodifies an adjective/participle (``DIRECTION_GUARDS``: "differed significantly" has no
    grammatical negation here), never double-negates a sentence that already contains "no
    significant"/"not significant" (``NEGATED_SIGNIFICANT_RE``, checked for both "significant"
    and "significantly"), never touches any word at all in a sentence carrying a parenthetical
    alternative such as "(or lower)" (``DIRECTION_PAREN_OR_SENTENCE_RE``: a sentence-level
    veto, not just a guard on the word framing it: guarding only the
    framing word let the loop flip a different, unrelated word in the same sentence and leave
    the degenerate parenthetical in place) and drops a leading article when negating
    "significant" (``DIRECTION_ARTICLE_STRIP``: "a significant difference" -> "no significant
    difference", not "a no significant difference")."""
    if DIRECTION_PAREN_OR_SENTENCE_RE.search(text):
        return None
    head, tail = _hedge_split(text)
    for src, dst in DIRECTION_FLIPS:
        if src in ("significant", "significantly") and NEGATED_SIGNIFICANT_RE.search(head):
            continue  # already negated elsewhere in the sentence: avoid a double negation
        new, n = _sub_word(
            head, src, dst, guard=NEGATION_GUARD + DIRECTION_PAREN_OR_GUARD,
            post_guard=DIRECTION_GUARDS.get(src, "") + DIRECTION_PAREN_OR_POST_GUARD,
            strip_article=src in DIRECTION_ARTICLE_STRIP, count=1,
        )
        if n:
            return new + tail, f"direction_flip:{src}->{dst}"
    return None


# Factors chosen on development data (build seed and
# sentence index), never a round multiple, 0.5, 0.25 or a unit-conversion ratio of the
# source number, so the resulting perturbation cannot be mistaken for a unit change.
NUMERIC_FACTORS_NONROUND: tuple[float, ...] = (1.37, 0.63, 1.19, 0.81)
# 10/100/1000/60 and their reciprocals read as a unit conversion (cm<->m, %<->fraction,
# seconds<->minutes); an integer, 0.5 or 0.25 reads as a tidy multiple/half/quarter -- neither
# is an adversarial perturbation of the source number.
_UNIT_CONVERSION_RATIOS: tuple[float, ...] = (10.0, 100.0, 1000.0, 60.0, 0.1, 0.01, 0.001, 1 / 60)
_ROUND_RATIO_TOLERANCE = 1e-9


def _is_round_or_unit_ratio(ratio: float) -> bool:
    """True when *ratio* is (within float tolerance) an integer, 0.5, 0.25, or one of
    ``_UNIT_CONVERSION_RATIOS``: none of these read as an adversarial numeric perturbation."""
    candidates = (round(ratio), 0.5, 0.25, *_UNIT_CONVERSION_RATIOS)
    return any(abs(ratio - c) < _ROUND_RATIO_TOLERANCE for c in candidates)


def _alter_number(
    text: str, *, factors: Sequence[float] = (2.0,), reject_round: bool = False,
    min_relative_change: float = 0.0,
) -> tuple[str, str] | None:
    """Multiply the first eligible number by each of *factors* in turn (first applicable
    number, then first applicable factor). Defaults (``factors=(2.0,)``,
    ``reject_round=False``, ``min_relative_change=0.0``) reproduce the exact legacy doubling
    behaviour, including the year skip, the p-value skip and the percent cap
    (``min(value, 99.0)``), so every frozen build stays byte identical. ``reject_round=True``
    additionally rejects a factor that is itself a round multiple, half, quarter or
    unit-conversion ratio (:func:`_is_round_or_unit_ratio`), trying the next factor, then the
    next number, before returning ``None``. ``min_relative_change`` additionally rejects a
    factor whose *rendered*
    change is smaller than that fraction of the source value -- most damagingly when the
    percent cap is the entire reason for the difference ("100%" -> "99%" is a 1% relative
    change, an artefact of the cap rather than of the chosen factor) -- trying the next factor
    before the next number. A source value of ``0`` never divides by itself here (any factor
    leaves it at ``0``, already rejected below by ``new == raw``), so the check is skipped for
    it rather than raising.
    """
    for m in NUMBER_RE.finditer(text):
        raw, pct = m.group(1), m.group(2)
        if "." not in raw and len(raw) == 4 and 1900 <= int(raw) <= 2099:
            continue  # a year
        if PVALUE_BEFORE_RE.search(text[: m.start()]):
            continue  # a p-value
        if CROSS_REFERENCE_LABEL_RE.search(text[: m.start()]):
            continue  # a cross-reference label ("Excerpt 2", "Table 3", ...), not a quantity
        if PAGE_NUMBER_BEFORE_RE.search(text[: m.start()]):
            continue  # a page number ("p. 45", "pp. 12-14"), an identifier, not a quantity
        if SAMPLE_SIZE_BEFORE_RE.search(text[: m.start()]):
            continue  # a cited study's own sample size ("N = 45"), not this sentence's claim
        source_value = float(raw)
        for factor in factors:
            if reject_round and _is_round_or_unit_ratio(factor):
                continue
            value = source_value * factor
            if pct:
                value = min(value, 99.0)
            if (
                min_relative_change and source_value != 0
                and abs(value - source_value) / abs(source_value) < min_relative_change
            ):
                continue
            if "." in raw:
                new = f"{value:.{len(raw.split('.')[1])}f}"
            else:
                new = str(int(round(value)))
            if new == raw:
                continue
            return text[: m.start(1)] + new + text[m.end(1):], f"numeric:{raw}->{new}"
    return None


# A numeric alteration whose rendered value differs from the source by less than this fraction
# reads as rounding noise, not a deliberate, checkable distortion. Applied only to the
# non-round operator below; the legacy
# doubling operator (used by every frozen build) keeps ``min_relative_change=0.0`` and is
# untouched.
MIN_RELATIVE_CHANGE = 0.05


def _alter_number_nonround(text: str) -> tuple[str, str] | None:
    """The non-round numeric operator (brief section 2.2 item 3): ``_alter_number`` with
    ``NUMERIC_FACTORS_NONROUND``, ``reject_round=True`` and a minimum relative-change floor
    (:data:`MIN_RELATIVE_CHANGE`). Emits the same ``numeric:`` prefix as the legacy doubling
    operator, since both describe the same construction category."""
    return _alter_number(
        text, factors=NUMERIC_FACTORS_NONROUND, reject_round=True,
        min_relative_change=MIN_RELATIVE_CHANGE,
    )


def _alter_quantifier(text: str) -> tuple[str, str] | None:
    """First applicable flip of ``QUANTIFIER_FLIPS``; never inside a negated frame
    (:func:`_hedge_split`, shared with ``_alter_direction``: "our aim is not to create the
    illusion that all Y" stays compatible with the source whatever the quantifier), never "all
    but one" -> "no but one" (``QUANTIFIER_POST_GUARDS``), never a floating quantifier
    immediately before its own finite verb (``QUANTIFIER_FLOATING_POST_GUARDS``: "who all
    identified" -> "who no identified"), never the object-position benefactive/purposive
    reading under any of its spellings (``QUANTIFIER_GUARDS`` looked up via
    ``QUANTIFIER_GUARD_ALIASES``, so "all of" and "every" inherit "all"'s guard) and never a
    second, self-contradictory negation in the same sentence
    (:func:`_quantifier_already_negated`)."""
    head, tail = _hedge_split(text)
    for src, dst in QUANTIFIER_FLIPS:
        if _quantifier_already_negated(head, dst):
            continue
        guard_key = QUANTIFIER_GUARD_ALIASES.get(src, src)
        post_guard = (
            QUANTIFIER_POST_GUARDS.get(src, "") + QUANTIFIER_FLOATING_POST_GUARDS.get(src, "")
        )
        new, n = _sub_word(
            head, src, dst, guard=QUANTIFIER_GUARDS.get(guard_key, ""),
            post_guard=post_guard, count=1,
        )
        if n:
            return new + tail, f"quantifier_flip:{src}->{dst}"
    return None


# Keyed by the three operator names the builder already
# emits as the `alteration` label's prefix (`"<operator>:<from>-><to>"`). `numeric` maps to the
# *non-round* variant here: the legacy round-doubling `_alter_number` stays the unrestricted
# default in `alter()` below (so the frozen builds stay byte identical), and is only reached
# through this table when a caller explicitly restricts `alter(..., allowed=["numeric"])`.
ALTER_FNS: dict[str, Callable[[str], tuple[str, str] | None]] = {
    "direction_flip": _alter_direction,
    "numeric": _alter_number_nonround,
    "quantifier_flip": _alter_quantifier,
}


def alter(sentence: str, *, allowed: Sequence[str] | None = None) -> tuple[str, str] | None:
    """First applicable rule, or ``None``.

    ``allowed=None`` (default) tries direction flip -> numeric x2 (round doubling) ->
    quantifier flip, exactly today's order and functions, so every frozen build stays byte
    identical. With ``allowed`` given, only those operators (:data:`ALTER_FNS` keys) are
    tried, in the table's order -- e.g. ``allowed=["numeric"]`` tries only the non-round
    numeric operator even when a direction flip is also possible.
    """
    text = normalise_whitespace(sentence)
    fns: Sequence[Callable[[str], tuple[str, str] | None]] = (
        (_alter_direction, _alter_number, _alter_quantifier)
        if allowed is None else tuple(ALTER_FNS[op] for op in allowed)
    )
    for fn in fns:
        out = fn(text)
        if out is not None and out[0] != text:
            return out
    return None


def _sha256_of_value(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


# The three stimulus constants that may change only against
# development data and never after the held-out set is built. Recorded verbatim in every
# build json so a later edit is a detectable, disclosed event rather than a silent drift.
STIMULUS_CONSTANTS_SHA256: dict[str, str] = {
    "OVER_SPECIFIED_MODIFIERS": _sha256_of_value(list(OVER_SPECIFIED_MODIFIERS)),
    "OVER_SPECIFIED_STOPWORDS": _sha256_of_value(sorted(OVER_SPECIFIED_STOPWORDS)),
    "NUMERIC_FACTORS_NONROUND": _sha256_of_value(list(NUMERIC_FACTORS_NONROUND)),
}


def parse_quotas(
    spec: str | None, *, rule_order: Sequence[str] = RULE_ORDER,
    defaults: Mapping[str, int] = QUOTAS,
) -> dict[str, int]:
    """``"rule=n,rule=n"`` -> a full quota dict merged over ``{rule: 0 for rule in rule_order}``
    so an unnamed rule is 0 (without this, ``assign_rules``'s ``quota = quotas[rule]`` would
    raise ``KeyError``). ``None`` returns *defaults* unchanged. Raises ``SystemExit`` naming
    any rule not in *rule_order*.
    """
    if spec is None:
        return dict(defaults)
    out = {rule: 0 for rule in rule_order}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        rule, _, n = part.partition("=")
        rule = rule.strip()
        if rule not in rule_order:
            raise SystemExit(f"--quotas: unknown rule {rule!r}; must be one of {rule_order}")
        out[rule] = int(n)
    return out


def parse_alteration_quotas(
    spec: str | None, *, altered_quota: int, alter_fns: Mapping[str, Any] = ALTER_FNS,
) -> dict[str, int] | None:
    """``"operator=n,..."`` validated to name only :data:`ALTER_FNS` keys and to sum to
    *altered_quota* (the ``altered`` rule's own quota). ``None`` when *spec* is not given: the
    fill loop then tries every operator, unrestricted, exactly as it does today. Raises
    ``SystemExit`` on an unknown operator or a sum that does not match *altered_quota* --
    there is no silent redistribution, only an explicit new vector.
    """
    if spec is None:
        return None
    out: dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        op, _, n = part.partition("=")
        op = op.strip()
        if op not in alter_fns:
            raise SystemExit(
                f"--alteration-quotas: unknown operator {op!r}; must be one of "
                f"{sorted(alter_fns)}"
            )
        out[op] = int(n)
    total = sum(out.values())
    if total != altered_quota:
        raise SystemExit(
            f"--alteration-quotas sums to {total}, must equal the altered quota "
            f"({altered_quota})"
        )
    return out


def is_clean_sentence(
    sentence: str, title: str | None = None, previous_sentence: str | None = None,
) -> bool:
    """Reject PDF-extraction artefacts and non-claims: hyphenation breaks, table fragments,
    number soup, trailing ellipses, reference-list lines, questions, section headings glued
    to the next sentence, running headers (the paper's own title, case-folded, first
    ``TITLE_PREFIX_CHARS`` characters, when ``title`` is given), author-stance openers
    ("We believe ..."), anaphoric fragments ("this number", "these findings") without an
    antecedent in the sentence (a numeral or a preceding clause) and attributed findings of
    other studies (``ATTRIBUTION_RE``: "Smith (2019) found ...", "The authors found ...",
    "his contribution proved ...", "Last, the researchers showed ...", "Smith (2019)
    recruited ...", "They report ..."), section headings glued to the front of a sentence
    ("Conclusion There is ..."), a numbered/dashed list label glued to its description
    (``LIST_LABEL_DASH_RE``), a PDF running header naming the journal, volume/issue and date
    (``RUNNING_HEADER_VOLUME_RE``: "TESL-EJ 26.3, November 2022 Fang & Xu 12 ..."), a footnote
    digit glued directly to a word
    (``FOOTNOTE_DIGIT_GLUED_RE``), and a bare sample-description opener ("Participants
    were ...") whose subject is elided into ``previous_sentence``'s own attribution
    (``ELIDED_SUBJECT_OPENER_RE``, only checked when ``previous_sentence`` is given); an
    "L2"/"T2"-style token is not a numeral antecedent."""
    if sentence.rstrip().endswith("?"):
        return False  # a question is not a claim
    stripped = sentence.strip()
    if STANCE_OPENER_RE.match(stripped) or ATTRIBUTION_RE.search(stripped):
        return False
    if (
        previous_sentence and ELIDED_SUBJECT_OPENER_RE.match(stripped)
        and ATTRIBUTION_RE.search(previous_sentence)
    ):
        return False
    anaphor = ANAPHORA_RE.search(stripped)
    if anaphor is not None:
        before = stripped[: anaphor.start()]
        has_numeral = re.search(r"(?<![A-Za-z])\d", stripped) is not None  # not "L2"
        has_clause = "," in before and len(before.split()) >= 4
        if not (has_numeral or has_clause):
            return False
    for pattern in (
        HYPHEN_BREAK_RE, ADJACENT_NUMBERS_RE, FOOTNOTE_MARK_RE, FOOTNOTE_DIGIT_GLUED_RE,
        KEYWORDS_RE, ELLIPSIS_RE, REFERENCE_RE, TITLE_CASE_AFTER_SEMICOLON_RE, HEADING_MERGE_RE,
        HEADING_PREFIX_RE, SOFT_HYPHEN_RE, HEADING_NUMBERED_REPEAT_RE, BARE_TITLE_BRACKETED_RE,
        LIST_LABEL_DASH_RE, RUNNING_HEADER_VOLUME_RE,
    ):
        if pattern.search(sentence):
            return False
    if title:
        prefix = normalise_whitespace(title).casefold()[:TITLE_PREFIX_CHARS]
        if len(prefix) >= 15 and prefix in normalise_whitespace(sentence).casefold():
            return False
    if len(NUMERIC_TOKEN_RE.findall(sentence)) > MAX_NUMERIC_TOKENS:
        return False
    letters = sum(1 for ch in sentence if ch.isalpha() or ch == " ")
    return letters / max(len(sentence), 1) >= MIN_ALPHA_RATIO


def load_excluded_dois(paths: Sequence[Path]) -> set[str]:
    """Normalised (lowercase, ``https://doi.org/`` prefix stripped) DOIs from one or more
    sources JSON files (``--exclude-sources``, repeatable). Used to
    keep ``hss-dev-30`` disjoint from the papers a ``--sources`` file such as
    ``hss_sources.json`` already selected for the frozen test set. Missing files are read as
    ``[]`` (``common.read_json``'s default), not an error.
    """
    out: set[str] = set()
    for path in paths:
        for entry in read_json(path, []) or []:
            doi = _bare_doi(entry.get("doi")) if isinstance(entry, Mapping) else None
            if doi:
                out.add(doi)
    return out


def _fold_diacritics(text: str) -> str:
    """NFKC-normalise (folds compatibility characters such as ligatures), then NFKD-decompose
    and drop every combining mark. Shared by :func:`normalise_for_veto` and the wrong-work
    identity check (:func:`_identity_tokens`, the surname comparison in
    :func:`check_fetched_text_identity`), which had the same defect class: Python's own
    ``str.casefold()`` maps the Turkish dotted capital I (U+0130) to
    ``"i"`` plus a *combining* dot above (U+0307), so a name such as ``"Majid ELANİ SHİRVAN"``
    casefolds to ``"...shi̇rvan"`` -- not a substring of the plain-ASCII "shirvan" a PDF's
    own text carries -- unless combining marks are stripped. Decomposing (and stripping) before
    ``casefold()`` handles both this case and an ordinary pre-existing accent (e.g. "Wedín"
    matching "Wedin"), since NFKD decomposes an accented letter into base + combining mark
    whether or not ``casefold()`` was involved in producing it.
    """
    folded = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", text))
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def normalise_for_veto(text: str) -> str:
    """Canonical form veto membership is tested on: :func:`_fold_diacritics` (folds
    compatibility characters such as ligatures, strips diacritics), then
    ``normalise_whitespace`` and case-fold. A single wrong accent in ``hss_veto.json``
    (e.g. "Wedin" transcribed as "Wedín") used to disable that
    entry outright under plain ``casefold()`` matching; this form treats the two as identical.
    """
    return normalise_whitespace(_fold_diacritics(text)).casefold()


def load_veto(path: Path) -> set[str]:
    """Sentences vetoed by Claude Code agent review sessions, not the authors
    (``hss_veto.json``: ``{"sentences": [...]}``).

    A safety net for extraction artefacts the heuristics miss; the file is committed and
    its contents are recorded in ``hss_claims.build.json``. Missing file -> empty set.
    """
    data = read_json(path, {}) or {}
    return {normalise_whitespace(str(x)) for x in data.get("sentences") or [] if str(x).strip()}


def find_veto_near_misses(
    veto_raw: Sequence[str], candidates_by_doi: Mapping[str, Sequence[Mapping[str, Any]]],
    *, threshold: float = 0.90,
) -> list[dict[str, Any]]:
    """Veto entries that do not exact-match (:func:`normalise_for_veto`) any surviving
    candidate sentence in *candidates_by_doi*, but whose normalised text is at least
    *threshold* similar (``difflib.SequenceMatcher`` ratio) to one that did survive.

    Operates on the *post*-veto-filter candidate pool (``candidates_for_paper`` already
    dropped every exact match), so a hit here always means a veto entry that should have
    excluded a sentence did not -- the diacritics-transcription defect: a single wrong
    character silently disabled an entry, and the same
    exact-string check the rebuild used to verify its own vetoes could not see the gap.
    Returns ``[{"veto": str, "closest": str, "doi": str, "ratio": float}, ...]``, one entry
    per matched veto sentence (a veto entry with no near match anywhere -- most of them, on
    any build that draws from a different source pool than the one a veto entry was written
    against -- contributes nothing).
    """
    out: list[dict[str, Any]] = []
    for v in veto_raw:
        v_norm = normalise_for_veto(v)
        if not v_norm:
            continue
        best: tuple[float, str, str] | None = None
        for doi, cands in candidates_by_doi.items():
            for cand in cands:
                sentence = str(cand.get("sentence", ""))
                s_norm = normalise_for_veto(sentence)
                if s_norm == v_norm:
                    continue  # exact normalised match: already excluded upstream
                ratio = difflib.SequenceMatcher(None, v_norm, s_norm).ratio()
                if ratio >= threshold and (best is None or ratio > best[0]):
                    best = (ratio, sentence, doi)
        if best is not None:
            out.append({"veto": v, "closest": best[1], "doi": best[2], "ratio": best[0]})
    return out


def raise_on_veto_near_misses(misses: Sequence[Mapping[str, Any]]) -> None:
    """``SystemExit`` naming every :func:`find_veto_near_misses` hit (``--veto-strict``,
    default on); a no-op when *misses* is empty."""
    if not misses:
        return
    lines = [
        f"  ratio={m['ratio']:.3f} veto={m['veto']!r} closest={m['closest']!r} (doi={m['doi']})"
        for m in misses
    ]
    raise SystemExit(
        "--veto-strict: veto entr" + ("y" if len(misses) == 1 else "ies")
        + " near-matched a surviving candidate sentence without excluding it (likely a "
        "transcription error in hss_veto.json):\n" + "\n".join(lines)
    )


def compute_original_sentence_overlap(
    items: Sequence[Mapping[str, Any]], paths: Sequence[Path]
) -> list[dict[str, Any]]:
    """One entry per *paths* file: how many ``original_sentence`` values *items* (this build)
    shares with that file's own ``original_sentence`` values (brief section 4's band-2
    independence measurement -- band 2 is drawn from the same ten dev papers and the same
    candidate ranking as ``hss_dev_claims.jsonl``, so sentence reuse is measured, not assumed).

    Each file is read as jsonl (the shape both ``hss_claims.jsonl`` and
    ``hss_dev_claims.jsonl`` use); a file that does not exist contributes zero shared
    sentences rather than raising. Comparison is on ``normalise_whitespace``d text, matching
    how the same sentence can be re-emitted with different incidental whitespace. Returns
    ``[{"file": name, "n_shared": k, "shared": [...]}]`` in *paths* order; ``shared`` is
    sorted for determinism.
    """
    own = {
        normalise_whitespace(str(it.get("original_sentence") or "")) for it in items
    }
    own.discard("")
    out: list[dict[str, Any]] = []
    for path in paths:
        other: set[str] = set()
        if Path(path).exists():
            for row in read_jsonl(path):
                text = normalise_whitespace(str(row.get("original_sentence") or ""))
                if text:
                    other.add(text)
        shared = sorted(own & other)
        out.append({"file": Path(path).name, "n_shared": len(shared), "shared": shared})
    return out


def is_quoted_in_chunk(sentence: str, chunk_text: str) -> bool:
    """True when ``sentence`` sits inside a quotation in ``chunk_text``: the text right after
    it (``QUOTE_CITATION_WINDOW`` characters) starts with a page-referenced parenthetical
    citation (``QUOTE_CITATION_RE``), or the chunk before it has an unclosed quotation mark
    (an odd number of ``"`` or more U+201C than U+201D). Not found in the chunk -> False."""
    text = normalise_whitespace(chunk_text)
    start = text.find(normalise_whitespace(sentence))
    if start < 0:
        return False
    end = start + len(normalise_whitespace(sentence))
    if QUOTE_CITATION_RE.match(text[end:end + QUOTE_CITATION_WINDOW]):
        return True
    before = text[:start]
    if before.count('"') % 2 == 1 or before.count("\u201c") > before.count("\u201d"):
        return True
    window = before[-QUOTE_COLON_INTRO_WINDOW:]
    return QUOTE_COLON_INTRO_RE.search(window) is not None


def previous_sentence_in_chunk(chunk_text: str, sentence: str) -> str | None:
    """The sentence immediately before *sentence* in *chunk_text* (:func:`common.
    split_sentences` over the raw chunk, matched by normalised, case-folded text), or ``None``
    when *sentence* is the chunk's first sentence or is not found verbatim. Used to catch a
    subject elided into the previous sentence's own citation (see
    :data:`ELIDED_SUBJECT_OPENER_RE`)."""
    target = normalise_whitespace(sentence).casefold()
    sentences = split_sentences(chunk_text)
    for i, s in enumerate(sentences):
        if normalise_whitespace(s).casefold() == target:
            return sentences[i - 1] if i > 0 else None
    return None


# Reference-list / bibliography signal patterns:
# "Retrieved from/on", a bare DOI, a "pp. <n>" page range, a "(YYYY)" year in parentheses and
# an initial-letter author name ("Smith, J. A."). None of these is rare in prose on its own
# (a single parenthetical citation year is normal); it is their *density* over a chunk that
# distinguishes a bibliography from running text (e.g.
# 10.17011/apples/urn.201903251959 chunk 2 is 2,999 characters of pure reference list carrying
# the section label "discussion").
REFERENCE_CHUNK_RETRIEVED_RE = re.compile(r"\bretrieved\s+(?:from|on)\b", re.IGNORECASE)
REFERENCE_CHUNK_DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+")
REFERENCE_CHUNK_PP_RE = re.compile(r"\bpp?\.\s?\d")
REFERENCE_CHUNK_YEAR_PAREN_RE = re.compile(r"\(\d{4}[a-z]?\)")
REFERENCE_CHUNK_INITIAL_AUTHOR_RE = re.compile(r"\b[A-Z][A-Za-z'\-]+,\s(?:[A-Z]\.\s?){1,3}")
REFERENCE_CHUNK_DENSITY_THRESHOLD = 6.0  # pattern hits per 1,000 characters
REFERENCE_CHUNK_MIN_LENGTH = 150  # below this, one citation can look arbitrarily "dense"
# A chunk that opens with a bare heading and nothing else on the same line: "References",
# "Bibliography", "Works Cited" (with or without a trailing colon).
REFERENCES_HEADING_START_RE = re.compile(
    r"^\s*(?:references|bibliography|works\s+cited)\s*(?:[:\n]|$)", re.IGNORECASE,
)


def reference_chunk_pattern_count(text: str) -> int:
    """Sum of hits for the five reference-list signal patterns in *text*."""
    return sum(
        len(p.findall(text)) for p in (
            REFERENCE_CHUNK_RETRIEVED_RE, REFERENCE_CHUNK_DOI_RE, REFERENCE_CHUNK_PP_RE,
            REFERENCE_CHUNK_YEAR_PAREN_RE, REFERENCE_CHUNK_INITIAL_AUTHOR_RE,
        )
    )


def is_reference_or_frontmatter_chunk(
    text: str, *, density_threshold: float = REFERENCE_CHUNK_DENSITY_THRESHOLD,
    min_length: int = REFERENCE_CHUNK_MIN_LENGTH,
) -> bool:
    """True when *text* reads as a reference list rather than prose: the density (per 1,000
    characters) of the five signal patterns (:func:`reference_chunk_pattern_count`) exceeds
    *density_threshold*. Chunks shorter than *min_length* are never flagged (too little text
    for a density estimate to mean anything)."""
    stripped = str(text or "").strip()
    if len(stripped) < min_length:
        return False
    density = reference_chunk_pattern_count(stripped) / (len(stripped) / 1000.0)
    return density > density_threshold


def _reference_or_backmatter_indices(chunks: Sequence[Mapping[str, Any]]) -> set[int]:
    """Original-list indices to exclude from candidate mining: any chunk whose reference
    pattern density exceeds the threshold regardless of its own section label, plus every
    chunk from the first bare References/Bibliography/Works Cited heading onward (that
    heading's own chunk is dropped too, even if short)."""
    out: set[int] = set()
    heading_seen = False
    for i, c in enumerate(chunks):
        text = str(c.get("text", ""))
        if not heading_seen and REFERENCES_HEADING_START_RE.match(text.strip()):
            heading_seen = True
        if heading_seen or is_reference_or_frontmatter_chunk(text):
            out.add(i)
    return out


def candidates_for_paper(
    chunks: Sequence[Mapping[str, Any]],
    *,
    title: str | None = None,
    veto: set[str] | None = None,
    dropped_chunks_out: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Ranked eligible sentences; results/findings/discussion/conclusion chunks first.

    ``chunk_index`` refers to the position in the *full* chunk list of the paper (dropped
    chunks are simply never scanned; surviving indices are not renumbered). Sentences
    containing the paper's ``title`` (running headers), reference-list lines, quoted
    passages (``is_quoted_in_chunk``) and the ``veto`` set (matched by
    :func:`normalise_for_veto`, not plain case-folding) are skipped. Any chunk flagged by
    :func:`is_reference_or_frontmatter_chunk`, and every chunk from the first bare
    References/Bibliography/Works Cited heading onward, is dropped before mining; its original
    index is appended to ``dropped_chunks_out`` when given.
    """
    vetoed = {normalise_for_veto(v) for v in (veto or set())}
    excluded_chunks = _reference_or_backmatter_indices(chunks)
    if dropped_chunks_out is not None:
        dropped_chunks_out.extend(sorted(excluded_chunks))
    eligible = [i for i in range(len(chunks)) if i not in excluded_chunks]
    preferred = [
        i for i in eligible if PREFERRED_SECTION_RE.search(str(chunks[i].get("section", "")))
    ]
    others = [i for i in eligible if i not in preferred]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in (preferred, others):
        texts = [str(chunks[i].get("text", "")) for i in group]
        for cand in select_candidate_sentences(texts):
            key = cand.sentence.casefold()
            chunk_text = texts[cand.chunk_index]
            previous = previous_sentence_in_chunk(chunk_text, cand.sentence)
            if (
                key in seen or normalise_for_veto(cand.sentence) in vetoed
                or not is_clean_sentence(cand.sentence, title, previous_sentence=previous)
            ):
                continue
            idx = group[cand.chunk_index]
            if is_quoted_in_chunk(cand.sentence, chunk_text):
                continue
            seen.add(key)
            out.append({
                "sentence": cand.sentence,
                "chunk_index": idx,
                "section": str(chunks[idx].get("section", "")),
                "score": cand.score,
            })
    return out


def load_cache(cache_dir: Path, dois: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    """``{doi: payload}`` for every ``<doi-slug>.json`` in the cache (optionally restricted)."""
    wanted = {doi_slug(d) for d in dois} if dois else None
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(cache_dir).glob("*.json")):
        if wanted is not None and path.stem not in wanted:
            continue
        payload = read_json(path)
        if isinstance(payload, dict) and payload.get("doi") and payload.get("chunks"):
            out[str(payload["doi"])] = payload
    return out


def _content_tokens(text: str, stopwords: frozenset[str]) -> set[str]:
    return {w for w in tokenize(text) if w not in stopwords}


# A finding verb signals the sentence reports an empirical result, as opposed to a definition,
# an aim/purpose statement, an acknowledgement, or a description of another study or of a
# procedure (e.g. "we checked ... if the statement had been rated significantly higher
# (or lower) ..." describes an analysis procedure, not a finding, and carries no finding verb
# and no statistic). Not exhaustive of every possible finding verb, but wide enough to cover
# every verb this module's own DIRECTION_FLIPS/SUBSTITUTIONS vocabularies already use for a
# finding ("showed"/"demonstrated"/"improved"/"increased"/...).
FINDING_VERB_RE = re.compile(
    r"\b(?:found|finds?|show(?:ed|s|n)?|demonstrat(?:ed|es)|report(?:ed|s)?|indicat(?:ed|es)|"
    r"reveal(?:ed|s)?|observ(?:ed|es)?|increas(?:ed|es)?|decreas(?:ed|es)?|correlat(?:ed|es)?|"
    r"predict(?:ed|s)?|improv(?:ed|es)?|worsen(?:ed|s)?|outperform(?:ed|s)?|"
    r"underperform(?:ed|s)?|achiev(?:ed|es)?|scor(?:ed|es)?|suggest(?:ed|s)?)\b",
    re.IGNORECASE,
)


def _reports_empirical_finding(sentence: str) -> bool:
    """True when *sentence* reports an empirical finding: a finding verb
    (:data:`FINDING_VERB_RE`) or a statistic (a percent sign or a numeral). Used to gate the
    ``over_specified`` construction (:func:`_over_specify`) so a qualifier is never appended to
    a definition, an aim, an acknowledgement or a claim about another study."""
    return (
        FINDING_VERB_RE.search(sentence) is not None or "%" in sentence
        or NUMERIC_TOKEN_RE.search(sentence) is not None
    )


def _over_specify(
    sentence: str,
    *,
    paper_text: str,
    modifiers: Sequence[str] = OVER_SPECIFIED_MODIFIERS,
    stopwords: frozenset[str] = OVER_SPECIFIED_STOPWORDS,
    index: int = 0,
    rejections_out: list[tuple[str, str, str]] | None = None,
) -> tuple[str, str] | None:
    """Append exactly one modifier from *modifiers* whose content tokens are all absent both
    from the cited paper's own (normalised, case-folded) full text and from *sentence* itself,
    appended to *sentence* as a trailing clause: the paper supports the sentence's main finding
    but not the added setting/population/time-window/instrument detail (``needs_nuance`` by
    construction). Never applied when *sentence* does not itself report an empirical finding
    (:func:`_reports_empirical_finding`), and never
    with a modifier that would duplicate a detail *sentence* already states.

    ``index`` rotates the starting modifier (round robin across items sharing a paper) so
    repeated calls do not always try the same one first. Returns ``(claim, modifier)``, or
    ``None`` when the sentence is not a finding or every modifier is rejected (the fill loop
    then advances to the next candidate sentence). Every rejected ``(sentence, modifier,
    matched_token)`` triple is appended to *rejections_out* when given, whether or not a
    modifier is eventually accepted. Raises ``ValueError`` without ``paper_text``: there is
    nothing to check absence against.
    """
    if not paper_text:
        raise ValueError("_over_specify requires paper_text to check modifier absence")
    if not _reports_empirical_finding(sentence):
        return None
    haystack = _content_tokens(normalise_whitespace(paper_text).casefold(), stopwords)
    text = normalise_whitespace(sentence)
    sentence_tokens = _content_tokens(text.casefold(), stopwords)
    n = len(modifiers)
    for i in range(n):
        modifier = modifiers[(index + i) % n]
        modifier_tokens = _content_tokens(modifier.casefold(), stopwords)
        hit = next(
            (t for t in modifier_tokens if t in haystack or t in sentence_tokens), None,
        )
        if hit is not None:
            if rejections_out is not None:
                rejections_out.append((text, modifier, hit))
            continue
        return f"{text.rstrip('.')}, {modifier}.", modifier
    return None


def _rule_claim(
    rule: str, sentence: str, *, paper_text: str | None = None,
    rejections_out: list[tuple[str, str, str]] | None = None,
) -> tuple[str, str | None] | None:
    """Claim text and alteration label for ``sentence`` under ``rule`` (None = not applicable).

    ``paper_text`` and ``rejections_out`` are only consulted for ``rule == "over_specified"``;
    both ``None`` reproduces every other rule's behaviour exactly.
    """
    if rule == "altered":
        out = alter(sentence)
        return None if out is None else out
    if rule == "paraphrase":
        text, subs = paraphrase(sentence)
        if not subs:
            return None
        return text, "; ".join(f"{a}->{b}" for a, b in subs)
    if rule == "over_specified":
        if paper_text is None:
            raise ValueError("_rule_claim('over_specified', ...) requires paper_text")
        out = _over_specify(sentence, paper_text=paper_text, rejections_out=rejections_out)
        return None if out is None else (out[0], f"over_specified:{out[1]}")
    return sentence, None


def _alteration_operator(alteration: str | None) -> str | None:
    """First segment of an ``operator:from->to`` alteration label, or ``None`` for a missing
    or unrecognised value (mirrors ``summarize._operator_of``, kept local to avoid a
    cross-module import for one three-line helper)."""
    if not alteration or ":" not in str(alteration):
        return None
    return str(alteration).split(":", 1)[0]


def assign_rules(
    candidates_by_doi: Mapping[str, Sequence[Mapping[str, Any]]],
    seed: int,
    *,
    meta_by_doi: Mapping[str, Mapping[str, Any]] | None = None,
    quotas: Mapping[str, int] = QUOTAS,
    expected: Mapping[str, list[str]] = EXPECTED,
    texts_by_doi: Mapping[str, str] | None = None,
    rejections_out: list[dict[str, Any]] | None = None,
    alteration_quotas: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Deterministic round-robin assignment of candidate sentences to construction rules.

    Papers are visited in sorted-DOI order rotated by ``seed % n_papers``; the cursor keeps
    advancing across rules so every paper contributes about equally. A sentence is used at
    most once. Raises ``ValueError`` when a quota cannot be filled.

    ``texts_by_doi`` (doi -> full cached text) is required only when ``quotas`` gives the
    ``over_specified`` rule a non-zero share; its rejected ``(sentence, modifier,
    matched_token)`` triples are appended (as dicts) to ``rejections_out`` when given.

    ``alteration_quotas`` (operator -> target count, summing to ``quotas["altered"]``)
    restricts the ``altered`` rule's fill loop to :data:`ALTER_FNS` operators that have not
    yet met their own sub-quota (:func:`alter`'s ``allowed``); an operator that cannot be
    filled after every paper is exhausted raises ``SystemExit`` naming it, the requested
    count and the count achieved -- the only permitted response is an explicit new vector.
    ``None`` (default) tries every operator, unrestricted, exactly as today.
    """
    dois = sorted(candidates_by_doi)
    if not dois:
        raise ValueError("no papers with candidate sentences")
    n = len(dois)
    start = seed % n
    order = dois[start:] + dois[:start]
    used: set[tuple[str, str]] = set()
    picked: dict[str, list[dict[str, Any]]] = {rule: [] for rule in RULE_ORDER}
    operator_counts: dict[str, int] = {op: 0 for op in (alteration_quotas or {})}
    cursor = 0
    for rule in FILL_ORDER:
        quota = quotas[rule]
        misses = 0
        while len(picked[rule]) < quota:
            if misses >= n:
                if rule == "altered" and alteration_quotas:
                    short = [
                        (op, target, operator_counts[op])
                        for op, target in alteration_quotas.items()
                        if operator_counts[op] < target
                    ]
                    if short:
                        op, target, got = short[0]
                        raise SystemExit(
                            f"cannot fill alteration operator {op!r}: {got}/{target} after "
                            f"exhausting {n} papers"
                        )
                raise ValueError(
                    f"cannot fill rule {rule!r}: {len(picked[rule])}/{quota} after "
                    f"exhausting {n} papers"
                )
            doi = order[cursor % n]
            cursor += 1
            choice = None
            for cand in candidates_by_doi[doi]:
                key = (doi, normalise_whitespace(cand["sentence"]).casefold())
                if key in used:
                    continue
                if rule == "altered" and alteration_quotas:
                    remaining_ops = [
                        op for op, target in alteration_quotas.items()
                        if operator_counts[op] < target
                    ]
                    built = (
                        alter(cand["sentence"], allowed=remaining_ops) if remaining_ops
                        else None
                    )
                elif rule == "over_specified":
                    paper_text = texts_by_doi.get(doi) if texts_by_doi else None
                    rejected: list[tuple[str, str, str]] = []
                    if paper_text is None:
                        built = _rule_claim(rule, cand["sentence"], paper_text=None)
                    else:
                        out = _over_specify(
                            cand["sentence"], paper_text=paper_text,
                            index=len(picked[rule]), rejections_out=rejected,
                        )
                        built = None if out is None else (out[0], f"over_specified:{out[1]}")
                    if rejections_out is not None:
                        rejections_out.extend(
                            {"source_doi": doi, "sentence": s, "modifier": m,
                             "matched_token": t}
                            for s, m, t in rejected
                        )
                else:
                    built = _rule_claim(rule, cand["sentence"])
                if built is None:
                    continue
                choice = (key, cand, built)
                break
            if choice is None:
                misses += 1
                continue
            misses = 0
            key, cand, (claim, alteration) = choice
            used.add(key)
            if rule == "altered" and alteration_quotas:
                op = _alteration_operator(alteration)
                if op in operator_counts:
                    operator_counts[op] += 1
            if rule == "wrong_paper":
                chunk_doi: str | None = order[(order.index(doi) + 1) % n]
            elif rule == "no_full_text":
                chunk_doi = None
            else:
                chunk_doi = doi
            shown = meta_by_doi.get(chunk_doi or doi, {}) if meta_by_doi else {}
            picked[rule].append({
                "rule": rule,
                "expected": list(expected[rule]),
                "claim": claim,
                "original_sentence": cand["sentence"],
                "alteration": alteration,
                "source_doi": doi,
                "chunk_doi": chunk_doi,
                "chunk_index": int(cand["chunk_index"]),
                "title": shown.get("title"),
                "authors": shown.get("authors"),
            })
    items: list[dict[str, Any]] = []
    for rule in RULE_ORDER:
        for i, it in enumerate(picked[rule], 1):
            items.append({"item_id": f"hss-{rule}-{i:02d}", **it})
    return items


def build_items_from_cache(
    cache_dir: Path,
    seed: int = DEFAULT_SEED,
    *,
    dois: Sequence[str] | None = None,
    min_chunks: int = 1,
    veto: set[str] | None = None,
    veto_file: Path | None = None,
    expected: Mapping[str, list[str]] = EXPECTED,
    altered_expected_mode: str = "legacy",
    quotas: Mapping[str, int] = QUOTAS,
    alteration_quotas: Mapping[str, int] | None = None,
    veto_strict: bool = True,
    identity_check: bool = True,
    identity_check_exceptions: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the full-text cache, rank candidates, assign rules; returns (items, build info).

    ``expected`` overrides ``EXPECTED`` per rule (``--altered-expected``: pass
    ``{**EXPECTED, "altered": ["unsupported"]}`` for the strict
    dev-set convention without mutating the module-level default, which the frozen test set
    keeps). ``altered_expected_mode`` ("strict" | "legacy") is recorded verbatim in the build
    info so a reader can tell which convention produced a given ``hss_claims.build.json``
    without diffing ``expected``. ``quotas`` and ``alteration_quotas`` default to the module
    constants (over_specified 0, no operator restriction), so an un-updated caller reproduces
    the frozen 30-item builds byte for byte. ``veto_strict`` (default on) raises ``SystemExit``
    when a veto entry near-matches, without exactly
    matching, a surviving candidate sentence (:func:`find_veto_near_misses`) -- a likely
    transcription error in ``hss_veto.json`` that would otherwise silently fail to exclude a
    known-defective stimulus. ``identity_check`` (default **on**; was off by default earlier,
    when the guard shipped fail-open since nothing in the repository's own rebuild recipes ever
    passed the opt-in flag) applies :func:`check_cache_identity` to the cache before candidate
    mining,
    dropping any source whose own joined chunk text does not match its own title/authors and
    recording it in ``info["wrong_work_sources"]``. The frozen 30-item test set's cache carries
    one disclosed mis-fetch (``10.55593/ej.26103a4``); rather than requiring a caller to
    remember to exempt it, it is kept unconditionally via the module-level
    :data:`KNOWN_IDENTITY_EXCEPTIONS`, so the default-on check still reproduces that set's
    byte-identical ``--skip-fetch`` rebuild. Pass ``identity_check=False`` to disable the check
    entirely (the CLI's ``--no-identity-check``). ``identity_check_exceptions`` (DOIs, only
    consulted when ``identity_check`` is true) names *additional* sources to keep regardless,
    on top of :data:`KNOWN_IDENTITY_EXCEPTIONS` -- a disclosed override, not a way to hide a
    fresh mis-fetch. Both the exception DOIs actually applied and their reasons (where known)
    are recorded in ``info["identity_check_exceptions"]`` / ``info["identity_check_exception_
    reasons"]`` so a reader can tell an unexempted clean build from one that silently kept a
    known mis-fetch.
    """
    cache = load_cache(cache_dir, dois)
    cache = {d: p for d, p in cache.items() if len(p.get("chunks") or []) >= min_chunks}
    wrong_work_sources: list[dict[str, Any]] = []
    identity_check_exceptions_applied: list[str] = []
    if identity_check:
        exceptions = {_bare_doi(x) for x in KNOWN_IDENTITY_EXCEPTIONS} | {
            _bare_doi(x) for x in (identity_check_exceptions or []) if x
        }
        exceptions.discard(None)
        identity_check_exceptions_applied = sorted(exceptions)
        cache, wrong_work_sources = check_cache_identity(cache, exceptions=exceptions)
    dropped_reference_chunks: dict[str, int] = {}
    candidates = {}
    for d, p in cache.items():
        dropped_here: list[int] = []
        candidates[d] = candidates_for_paper(
            p["chunks"], title=p.get("title"), veto=veto, dropped_chunks_out=dropped_here,
        )
        dropped_reference_chunks[d] = len(dropped_here)
    if veto_strict and veto:
        raise_on_veto_near_misses(find_veto_near_misses(veto, candidates))
    candidates = {d: c for d, c in candidates.items() if c}
    meta = {d: {"title": p.get("title"), "authors": p.get("authors")} for d, p in cache.items()}
    texts_by_doi = {
        d: " ".join(str(c.get("text", "")) for c in p.get("chunks") or [])
        for d, p in cache.items()
    }
    built_at = now_iso()
    over_specified_rejections: list[dict[str, Any]] = []
    items = assign_rules(
        candidates, seed, meta_by_doi=meta, quotas=quotas, expected=expected,
        texts_by_doi=texts_by_doi, rejections_out=over_specified_rejections,
        alteration_quotas=alteration_quotas,
    )
    for it in items:
        it["built_at"] = built_at
    info = {
        "built_at": built_at,
        "seed": seed,
        "n_items": len(items),
        "counts": {rule: sum(1 for it in items if it["rule"] == rule) for rule in RULE_ORDER},
        "expected": dict(expected),
        "altered_expected_mode": altered_expected_mode,
        "sources": sorted(candidates),
        "n_sources": len(candidates),
        "candidates_per_source": {d: len(c) for d, c in candidates.items()},
        "dropped_reference_chunks": dropped_reference_chunks,
        "identity_check": identity_check,
        "wrong_work_sources": wrong_work_sources,
        "n_wrong_work_sources": len(wrong_work_sources),
        "identity_check_exceptions": identity_check_exceptions_applied,
        "identity_check_exception_reasons": {
            d: KNOWN_IDENTITY_EXCEPTIONS[d]
            for d in identity_check_exceptions_applied if d in KNOWN_IDENTITY_EXCEPTIONS
        },
        "veto_strict": veto_strict,
        "substitutions": [list(x) for x in SUBSTITUTIONS],
        "over_specified_rejections": over_specified_rejections,
        "n_over_specified_rejections": len(over_specified_rejections),
        "alterations": {
            "order": ["direction_flip", "numeric_x2_percent_capped_99", "quantifier_flip"],
            "direction_flips": [list(x) for x in DIRECTION_FLIPS],
            "quantifier_flips": [list(x) for x in QUANTIFIER_FLIPS],
            "numeric_mode": "nonround" if alteration_quotas else "legacy",
            "numeric_factors_nonround": list(NUMERIC_FACTORS_NONROUND),
        },
        "candidate_ranking": "common.select_candidate_sentences; results/findings/discussion/"
        "conclusion chunks first; is_clean_sentence(title) rejects extraction artefacts, "
        "reference lines, running headers, author-stance openers, anaphoric fragments and "
        "attributed findings or procedures of other studies (ATTRIBUTION_RE: citation "
        "subjects followed by a reporting or procedural verb, 'the authors/researchers ...' "
        "in any clause, 'his/her/their study ...', 'Some/several ... studies/examples (...) "
        "show ...'); headings glued to the front of a sentence (HEADING_PREFIX_RE); an "
        "'L2'-style token is not a numeral antecedent for an anaphoric opener; paraphrase "
        "never splits 'case/pilot/cohort study' (SUBSTITUTION_GUARDS); quoted passages are "
        "skipped (is_quoted_in_chunk: a page-referenced parenthetical citation right after "
        "the sentence, or an unclosed quotation mark before it in the chunk); quantifier "
        "flips never touch 'the most' or 'at all' (QUANTIFIER_GUARDS); 'significantly' is "
        "only negated when it premodifies an adjective/participle, never when it postmodifies "
        "a verb (DIRECTION_GUARDS: 'differed significantly' has no grammatical negation here); "
        "hss_veto.json sentences are skipped",
        "veto": sorted(veto or []),
        "veto_file": veto_file.name if veto_file else None,
        "stimulus_constants_sha256": dict(STIMULUS_CONSTANTS_SHA256),
    }
    return items, info


# --------------------------------------------------------------------------------------
# Acquisition through the production pipeline (network; no LLM)
# --------------------------------------------------------------------------------------


def openalex_query_url(issn: str, mailto: str, per_page: int = 50) -> str:
    filt = (
        f"primary_location.source.issn:{issn},open_access.is_oa:true,type:article,"
        "publication_year:2019-2024"
    )
    select = (
        "id,doi,title,publication_year,open_access,best_oa_location,primary_location,authorships"
    )
    return (
        f"{OPENALEX_WORKS}?filter={filt}&select={select}&sort=cited_by_count:desc"
        f"&per-page={per_page}&mailto={mailto}"
    )


def _authors(work: Mapping[str, Any]) -> list[str]:
    names = []
    for a in work.get("authorships") or []:
        name = (a.get("author") or {}).get("display_name")
        if name:
            names.append(str(name))
    return names


_EMAIL_PARAMS = ("mailto", "email")


def public_url(url: str) -> str:
    """``url`` without ``mailto=`` / ``email=`` query parameters (kept only on the wire)."""
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if k not in _EMAIL_PARAMS]
    return urlunsplit(parts._replace(query=urlencode(query, safe=":,|/")))


def _bare_doi(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", value.strip(), flags=re.IGNORECASE).lower()


def crossref_query_url(issn: str, mailto: str, rows: int = 50) -> str:
    filt = f"issn:{issn},from-pub-date:2019,until-pub-date:2024,type:journal-article"
    return (
        f"{CROSSREF_WORKS}?filter={filt}&select=DOI,title,issued,author&sort=is-referenced-by-count"
        f"&order=desc&rows={rows}&mailto={mailto}"
    )


def crossref_to_works(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Crossref ``message.items`` -> OpenAlex-shaped work dicts (doi, title, year, authorships)."""
    out = []
    for it in (payload.get("message") or {}).get("items") or []:
        doi = _bare_doi(it.get("DOI"))
        if not doi:
            continue
        parts = ((it.get("issued") or {}).get("date-parts") or [[None]])[0]
        authors = [
            {"author": {"display_name": " ".join(
                x for x in (a.get("given"), a.get("family")) if x
            )}}
            for a in it.get("author") or []
        ]
        out.append({
            "id": None,
            "doi": doi,
            "title": (it.get("title") or [None])[0],
            "publication_year": parts[0] if parts else None,
            "authorships": authors,
        })
    return out


async def list_candidates(
    journal: Mapping[str, str], mailto: str, *, attempts: int = 3, backoff_s: float = 10.0
) -> list[dict[str, Any]]:
    """Article candidates for one journal: OpenAlex (OA filter) or, when OpenAlex answers
    HTTP 429 (daily budget of a shared IP), Crossref by ISSN. ``selected_from`` records the
    query actually used; open access is established by Unpaywall in ``acquire_one`` anyway."""
    import httpx

    url = openalex_query_url(journal["issn"], mailto)
    works: list[dict[str, Any]] = []
    timeout = httpx.Timeout(
        connect=CONNECT_TIMEOUT_S, read=READ_TIMEOUT_S, write=WRITE_TIMEOUT_S,
        pool=POOL_TIMEOUT_S,
    )
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for attempt in range(1, attempts + 1):
            resp = await client.get(url)
            if resp.status_code == 429 and attempt < attempts:
                await asyncio.sleep(backoff_s * attempt)
                continue
            break
        if resp.status_code == 200:
            works = [{**w, "doi": _bare_doi(w.get("doi"))} for w in resp.json().get("results", [])]
        else:
            print(f"  OpenAlex HTTP {resp.status_code} for {journal['name']}; using Crossref")
            url = crossref_query_url(journal["issn"], mailto)
            headers = {"User-Agent": f"ScholarRAG-eval (mailto:{mailto})"}
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            works = crossref_to_works(resp.json())
    return [
        {**w, "journal": journal["name"], "issn": journal["issn"],
         "selected_from": public_url(url)}
        for w in works if w.get("doi")
    ]


async def unpaywall_record(doi: str, email: str) -> dict[str, Any]:
    """Raw Unpaywall record (licence string); the URL itself comes from ``UnpaywallClient``."""
    import httpx

    timeout = httpx.Timeout(
        connect=CONNECT_TIMEOUT_S, read=READ_TIMEOUT_S, write=WRITE_TIMEOUT_S,
        pool=POOL_TIMEOUT_S,
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(f"https://api.unpaywall.org/v2/{doi}", params={"email": email})
        return resp.json() if resp.status_code == 200 else {}


def load_pipeline() -> dict[str, Any]:
    """Import the production acquisition functions (backend on sys.path, .env exported)."""
    add_backend_to_path()
    export_env_from_dotenv(("UNPAYWALL_EMAIL", "OPENALEX_EMAIL", "DATABASE_URL"))
    import_backend_settings()  # readable SystemExit when Settings cannot load from the cwd
    from app.clients.unpaywall import UnpaywallClient
    from app.services.fulltext import chunk_text, extract_text_from_pdf, fetch_pdf_from_url

    return {
        "unpaywall": UnpaywallClient(),
        "fetch_pdf": fetch_pdf_from_url,
        "extract": extract_text_from_pdf,
        "chunk": chunk_text,
    }


_TITLE_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Function words stripped before comparing title tokens: present in nearly every title/prefix
# regardless of subject matter, so they inflate overlap without discriminating anything.
TITLE_OVERLAP_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "in",
    "into", "is", "its", "not", "of", "on", "or", "our", "that", "the", "their", "this", "to",
    "was", "were", "with",
})
IDENTITY_TITLE_PREFIX_CHARS = 3000
IDENTITY_MIN_TOKEN_OVERLAP = 0.3
# The one disclosed, named
# exception to the cache identity check, now that the check defaults on -- the frozen 30-item
# test set's cache carries one known mis-fetch (10.55593/ej.26103a4) that must be kept so
# ``hss_claims.jsonl``'s ``--skip-fetch`` rebuild stays byte identical. Applied unconditionally
# whenever ``identity_check`` is true, never something a caller
# must remember to pass, and its reason string is recorded alongside the DOI in every build
# json (``info["identity_check_exceptions"]`` / ``info["identity_check_exception_reasons"]``)
# so a reader can tell an unexempted clean build from one that silently kept a known mis-fetch.
KNOWN_IDENTITY_EXCEPTIONS: Mapping[str, str] = {
    "10.55593/ej.26103a4": "frozen 30-item test set: disclosed mis-fetch, kept so the "
                           "committed hss_claims.jsonl stays byte identical",
}


def _identity_tokens(text: str) -> set[str]:
    folded = _fold_diacritics(text).casefold()
    return {w for w in _TITLE_TOKEN_RE.findall(folded) if w not in TITLE_OVERLAP_STOPWORDS}


def check_fetched_text_identity(
    *, expected_title: str, extracted_text: str, expected_authors: Sequence[str] | None = None,
    prefix_chars: int = IDENTITY_TITLE_PREFIX_CHARS,
    min_token_overlap: float = IDENTITY_MIN_TOKEN_OVERLAP,
) -> tuple[bool, str | None]:
    """(ok, detected_title_fragment): does the freshly extracted PDF text actually describe
    ``expected_title`` (the OpenAlex/Crossref record for the DOI just fetched)?

    A candidate's OpenAlex ``oa_url`` can serve a wholly different article (e.g.
    ``10.55593/ej.27108a7``'s ``oa_url`` cached Ambele (2022) instead of the
    named Thongwichit and Ulla TESL-EJ paper). Checked by normalised token overlap between
    ``expected_title`` and the first ``prefix_chars`` characters of ``extracted_text`` (a
    title page rarely repeats every title word verbatim -- headers wrap, hyphenate, or
    truncate -- so this is a threshold, not an exact match), plus, when ``expected_authors``
    is given, a literal case-insensitive check that the first author's surname appears
    somewhere in that same prefix. ``(True, None)`` on a match or when ``expected_title`` is
    empty (nothing to check against); ``(False, fragment)`` on a mismatch, where ``fragment``
    is the whitespace-normalised first 200 characters of the extracted text, for the caller
    to log what was actually fetched.

    The surname comparison is diacritic-folded on both sides (:func:`_fold_diacritics`),
    not just case-folded: a plain ``.casefold()`` maps a Turkish dotted
    capital I (as in OpenAlex's "Majid ELANİ SHİRVAN") to "i" plus a combining dot, never a
    substring of a PDF's plain-ASCII "Shirvan", so an unfolded comparison would reject a
    correctly fetched source.
    """
    prefix = extracted_text[:prefix_chars]
    title_tokens = _identity_tokens(expected_title)
    if not title_tokens:
        return True, None
    prefix_tokens = _identity_tokens(prefix)
    overlap = len(title_tokens & prefix_tokens) / len(title_tokens)
    author_ok = True
    if expected_authors:
        first = next((a for a in expected_authors if a and a.strip()), None)
        surname = first.strip().split()[-1] if first else None
        if surname and len(surname) > 2:
            folded_prefix = normalise_whitespace(_fold_diacritics(prefix)).casefold()
            author_ok = _fold_diacritics(surname).casefold() in folded_prefix
    if overlap >= min_token_overlap and author_ok:
        return True, None
    fragment = normalise_whitespace(prefix)[:200]
    return False, fragment or None


def check_cache_identity(
    cache: Mapping[str, Mapping[str, Any]], *, exceptions: set[str] | None = None,
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    """(kept_cache, wrong_work_sources): apply :func:`check_fetched_text_identity` to every
    cached payload's own joined chunk text against its own recorded title/authors.

    Previously, the identity check was wired only into :func:`acquire_one`
    (the network fetch path), so it could never fire on a ``--skip-fetch`` rebuild over an
    already-mis-fetched cache -- exactly how two committed payloads survived every prior
    rebuild: ``10.55593/ej.27108a7`` (Ambele 2022, cached under the Thongwichit & Ulla TESL-EJ
    DOI) and ``10.55593/ej.26103a4`` (Weng, Zhu & Kim's 2019 TESOL International Journal
    article, cached under Kim & Weng's 2022 TESL-EJ systematic-review DOI). ``exceptions`` (a
    :func:`_bare_doi`-normalised set) is kept unconditionally -- a named, disclosed override
    for a source already accepted as frozen, never a way to silence a fresh discovery.

    Called from :func:`build_items_from_cache` whenever its own ``identity_check`` argument is
    true (default true): the frozen 30-item test set's cache still
    carries the ``10.55593/ej.26103a4`` mis-fetch as a disclosed, separately tracked
    provenance defect in a committed artifact, not one this builder rewrites, so
    :func:`build_items_from_cache` always includes it in ``exceptions`` via the module-level
    :data:`KNOWN_IDENTITY_EXCEPTIONS`, keeping that set's byte-identical ``--skip-fetch``
    rebuild intact under the new default.
    """
    exceptions = exceptions or set()
    kept: dict[str, Mapping[str, Any]] = {}
    dropped: list[dict[str, Any]] = []
    for d in sorted(cache):
        p = cache[d]
        if _bare_doi(d) in exceptions:
            kept[d] = p
            continue
        text = " ".join(str(c.get("text", "")) for c in p.get("chunks") or [])
        ok, fragment = check_fetched_text_identity(
            expected_title=str(p.get("title") or ""), extracted_text=text,
            expected_authors=p.get("authors"),
        )
        if ok:
            kept[d] = p
        else:
            dropped.append({"doi": d, "title": p.get("title"), "detected": fragment})
    return kept, dropped


def _exception_reason(exc: Exception) -> str:
    """``"http_403"`` for an HTTP 403 raised anywhere in the fetch/extraction pipeline
    (``httpx.HTTPStatusError`` carries the response on ``.response``); otherwise the
    exception's class name and message (brief section 2.2 item 7's reason enum)."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 403:
        return "http_403"
    return f"{type(exc).__name__}: {exc}"


async def acquire_one(
    work: Mapping[str, Any], pipeline: Mapping[str, Any], *, email: str, min_chars: int,
    min_chunks: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """(source record, cache payload, reason) for one DOI.

    ``source``/``payload`` are ``None`` and ``reason`` names the rejection when acquisition
    fails at any stage: ``"not_oa"`` (Unpaywall has no OA location), ``"html_not_pdf"`` (the
    fetched bytes are not a PDF), ``"wrong_work: <fragment>"`` (the extracted text does not
    describe ``work``'s title/author -- :func:`check_fetched_text_identity`), ``"min_chars"``
    / ``"min_chunks"`` (the extracted text is too short or too thin), or the caught
    exception's class name and message. ``reason`` is ``None`` only when a source is kept.
    """
    doi = work["doi"]
    pdf_url = await pipeline["unpaywall"].lookup(doi)
    if not pdf_url:
        return None, None, "not_oa"
    pdf = await pipeline["fetch_pdf"](pdf_url)
    if not pdf or not pdf[:5].startswith(b"%PDF"):
        return None, None, "html_not_pdf"
    try:
        text = await asyncio.to_thread(pipeline["extract"], pdf)
    except Exception as exc:  # noqa: BLE001 - any extraction failure disqualifies the DOI
        print(f"  {doi}: extraction failed ({type(exc).__name__})")
        return None, None, _exception_reason(exc)
    identity_ok, detected = check_fetched_text_identity(
        expected_title=str(work.get("title") or ""), expected_authors=_authors(work),
        extracted_text=text,
    )
    if not identity_ok:
        print(f"  {doi}: wrong_work (fetched text does not match the expected title/author; "
              f"detected: {detected!r})")
        return None, None, f"wrong_work: {detected}"
    if len(text) < min_chars:
        return None, None, "min_chars"
    chunks = await asyncio.to_thread(pipeline["chunk"], text)
    if len(chunks) < min_chunks:
        return None, None, "min_chunks"
    record = await unpaywall_record(doi, email)
    best = record.get("best_oa_location") or {}
    source = {
        "doi": doi,
        "title": work.get("title"),
        "year": work.get("publication_year"),
        "journal": work["journal"],
        "issn": work["issn"],
        "openalex_id": work.get("id"),
        "selected_from": work["selected_from"],
        "oa_url": pdf_url,
        "license": best.get("license") or record.get("best_oa_location", {}).get("license"),
        "unpaywall_accessed": now_iso(),
        "n_chars": len(text),
        "n_chunks": len(chunks),
    }
    payload = {
        "doi": doi,
        "title": work.get("title"),
        "authors": _authors(work),
        "source_url": pdf_url,
        "fetched_at": now_iso(),
        "n_chars": len(text),
        "chunks": chunks,
    }
    return source, payload, None


async def acquire_sources(
    *, sources_path: Path, cache_dir: Path, n_sources: int, min_chars: int, min_chunks: int,
    per_journal: int, mailto: str, max_failures: int = 5,
    notes: dict[str, Any] | None = None,
    excluded: set[str] | None = None,
    attempts_out: list[dict[str, Any]] | None = None,
    candidate_timeout_s: float = DEFAULT_CANDIDATE_TIMEOUT_S,
) -> list[dict[str, Any]]:
    """Round-robin over ``JOURNALS`` until ``n_sources`` articles pass the full pipeline.

    A journal is dropped after ``max_failures`` consecutive failures (host blocks the
    production fetcher); per-journal outcomes are written into ``notes`` when given.
    ``excluded`` (``--exclude-sources``, already normalised by
    :func:`load_excluded_dois`) is removed from every candidate queue before the round robin
    starts, so a DOI already selected for another set (e.g. the frozen HSS test set) can never
    be re-selected here. ``attempts_out``, when given,
    receives one ``{doi, journal, outcome, reason}`` per DOI actually tried, in try order,
    ``outcome`` in ``{"kept", "dropped"}`` and ``reason`` ``None`` when kept.
    ``candidate_timeout_s`` is a wall-clock budget for one
    whole ``acquire_one`` call: exceeding it drops the candidate with reason
    ``"candidate_wall_clock_budget_exceeded"`` instead of hanging the build indefinitely (a
    real Windows/httpx failure mode observed in the P7 build).
    """
    pipeline = load_pipeline()
    cache_dir.mkdir(parents=True, exist_ok=True)
    excluded = excluded or set()
    existing = {s["doi"]: s for s in (read_json(sources_path, []) or [])}
    selected: list[dict[str, Any]] = []
    for doi, src in existing.items():
        if (cache_dir / f"{doi_slug(doi)}.json").exists():
            selected.append(src)
    print(f"{len(selected)} sources already cached")
    queues: list[list[dict[str, Any]]] = []
    for journal in JOURNALS:
        try:
            cands = await list_candidates(journal, mailto)
        except Exception as exc:  # noqa: BLE001
            print(f"  OpenAlex failed for {journal['name']}: {exc}")
            cands = []
        print(f"  {journal['name']}: {len(cands)} OA candidates")
        queues.append([c for c in cands if c["doi"] not in existing and c["doi"] not in excluded])
    taken = {j["name"]: sum(1 for s in selected if s["journal"] == j["name"]) for j in JOURNALS}
    failures = {j["name"]: 0 for j in JOURNALS}
    tried = {j["name"]: 0 for j in JOURNALS}
    dropped: dict[str, str] = {}
    while len(selected) < n_sources and any(queues):
        for jq, journal in zip(queues, JOURNALS, strict=True):
            name = journal["name"]
            if len(selected) >= n_sources or not jq or taken[name] >= per_journal:
                continue
            if failures[name] >= max_failures:
                if name not in dropped:
                    dropped[name] = f"dropped after {max_failures} consecutive failures"
                    print(f"  {name}: {dropped[name]}")
                jq.clear()
                continue
            work = jq.pop(0)
            tried[name] += 1
            print(f"  trying {work['doi']} ({journal['name']})")
            try:
                source, payload, reason = await asyncio.wait_for(
                    acquire_one(
                        work, pipeline, email=mailto, min_chars=min_chars, min_chunks=min_chunks
                    ),
                    timeout=candidate_timeout_s,
                )
            except (TimeoutError, asyncio.TimeoutError):
                print(f"    failed: candidate_wall_clock_budget_exceeded ({candidate_timeout_s}s)")
                source, payload, reason = None, None, "candidate_wall_clock_budget_exceeded"
            except Exception as exc:  # noqa: BLE001
                print(f"    failed: {type(exc).__name__}: {exc}")
                source, payload, reason = None, None, _exception_reason(exc)
            if attempts_out is not None:
                attempts_out.append({
                    "doi": work["doi"], "journal": name,
                    "outcome": "kept" if source is not None else "dropped", "reason": reason,
                })
            if source is None:
                failures[name] += 1
                continue
            failures[name] = 0
            write_json(cache_dir / f"{doi_slug(source['doi'])}.json", payload)
            selected.append(source)
            taken[name] += 1
            print(f"    kept ({source['n_chars']} chars, {source['n_chunks']} chunks, "
                  f"license={source['license']})")
    if notes is not None:
        for journal in JOURNALS:
            name = journal["name"]
            notes[name] = {
                "issn": journal["issn"], "tried": tried[name], "kept": taken[name],
                "dropped": dropped.get(name),
            }
    write_json(sources_path, selected)
    print(f"wrote {sources_path} ({len(selected)} sources)")
    return selected


async def probe_journals(
    mailto: str, *, candidate_timeout_s: float = DEFAULT_CANDIDATE_TIMEOUT_S,
) -> list[dict[str, Any]]:
    """For each :data:`JOURNALS` entry, take the first OA candidate from
    :func:`list_candidates` and run :func:`acquire_one` with :func:`load_pipeline`'s pipeline
    (brief section 2.2 item 8): one row per journal ``{journal, issn, openalex, doi, outcome,
    reason}``, ``outcome`` in ``{"probe_ok", "probe_failed"}``. Selects nothing, writes no
    sources file and never touches ``n_attempts``.

    ``candidate_timeout_s`` wraps this ``acquire_one`` call in the
    same wall-clock budget :func:`acquire_sources` already enforces -- taking each journal's
    first untested OA candidate is exactly the population that hung the P7 build, and until
    now the ``--candidate-timeout-s`` CLI flag reached only the ``acquire_sources`` path, never
    ``--probe-journals``. Exceeding the budget records ``outcome="probe_failed"``,
    ``reason="candidate_wall_clock_budget_exceeded"``, the same reason string
    :func:`acquire_sources` uses for the same condition.
    """
    pipeline = load_pipeline()
    rows: list[dict[str, Any]] = []
    for journal in JOURNALS:
        row: dict[str, Any] = {
            "journal": journal["name"], "issn": journal["issn"],
            "openalex": journal["openalex"], "doi": None,
        }
        try:
            cands = await list_candidates(journal, mailto)
        except Exception as exc:  # noqa: BLE001
            rows.append({**row, "outcome": "probe_failed", "reason": _exception_reason(exc)})
            continue
        if not cands:
            rows.append({**row, "outcome": "probe_failed", "reason": "no OA candidates"})
            continue
        work = cands[0]
        row["doi"] = work["doi"]
        try:
            source, _payload, reason = await asyncio.wait_for(
                acquire_one(
                    work, pipeline, email=mailto, min_chars=DEFAULT_MIN_CHARS,
                    min_chunks=DEFAULT_MIN_CHUNKS,
                ),
                timeout=candidate_timeout_s,
            )
        except (TimeoutError, asyncio.TimeoutError):
            source, reason = None, "candidate_wall_clock_budget_exceeded"
        except Exception as exc:  # noqa: BLE001
            source, reason = None, _exception_reason(exc)
        rows.append({
            **row, "outcome": "probe_ok" if source is not None else "probe_failed",
            "reason": reason,
        })
    return rows


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    ap.add_argument("--min-chunks", type=int, default=DEFAULT_MIN_CHUNKS)
    ap.add_argument("--n-sources", type=int, default=DEFAULT_N_SOURCES)
    ap.add_argument("--per-journal", type=int, default=3, help="max articles per journal")
    ap.add_argument("--max-failures", type=int, default=5,
                    help="drop a journal after this many consecutive acquisition failures")
    ap.add_argument("--skip-fetch", action="store_true", help="build from the cache only")
    ap.add_argument("--veto", type=Path, default=DEFAULT_VETO,
                    help="JSON {sentences: [...]} of original sentences never to use")
    ap.add_argument(
        "--exclude-sources", type=Path, action="append", default=[],
        help="sources JSON (e.g. hss_sources.json) whose DOIs must never be selected here; "
        "repeatable. Used to keep hss-dev-30 disjoint from the frozen test set's papers.",
    )
    ap.add_argument(
        "--altered-expected", choices=("strict", "legacy"), default="legacy",
        help="'strict' sets EXPECTED['altered']=['unsupported'] (brief 5.2, hss-dev-30); "
        "'legacy' (default) keeps the disjunctive ['unsupported','needs_nuance'] the frozen "
        "hss_claims.jsonl test set was built with and must keep byte-identical",
    )
    ap.add_argument(
        "--quotas", default=None,
        help="'rule=n,rule=n,...' overrides QUOTAS; an unnamed rule defaults to 0. "
        "Default (omitted): today's five-rule QUOTAS, byte "
        "identical, over_specified=0.",
    )
    ap.add_argument(
        "--alteration-quotas", default=None,
        help="'direction_flip=n,numeric=n,quantifier_flip=n', must sum to the altered quota; "
        "an unfillable sub-quota raises SystemExit "
        "naming the operator and the count achieved. Default (omitted): every operator "
        "unrestricted, exactly as today.",
    )
    ap.add_argument(
        "--probe-journals", action="store_true",
        help="probe each JOURNALS entry's first OA candidate through the production "
        "pipeline, write --probe-out, and exit; selects nothing and does not increment "
        "n_attempts",
    )
    ap.add_argument("--probe-out", type=Path, default=DEFAULT_PROBE_OUT)
    ap.add_argument(
        "--overlap-with", type=Path, action="append", default=[],
        help="claims jsonl file (e.g. hss_dev_claims.jsonl) to compare this build's "
        "original_sentence values against; repeatable. Records "
        "info['original_sentence_overlap'] per file (brief section 4's band-2 independence "
        "measurement). Default (omitted): no comparison, so an unrelated build stays byte "
        "identical.",
    )
    ap.add_argument(
        "--veto-strict", dest="veto_strict", action="store_true", default=True,
        help="(default) raise SystemExit when a hss_veto.json entry near-matches, without "
        "exactly matching, a surviving candidate sentence -- a likely transcription error.",
    )
    ap.add_argument(
        "--no-veto-strict", dest="veto_strict", action="store_false",
        help="disable the veto near-miss check.",
    )
    ap.add_argument(
        "--candidate-timeout-s", type=float, default=DEFAULT_CANDIDATE_TIMEOUT_S,
        help="wall-clock budget (seconds) for one whole acquire_one call; exceeding it drops "
        "the candidate with reason 'candidate_wall_clock_budget_exceeded' instead of hanging "
        "the build indefinitely. Applies to both "
        "acquire_sources and --probe-journals.",
    )
    ap.add_argument(
        "--identity-check-cache", dest="identity_check_cache", action="store_true",
        default=True,
        help="(default: on) apply check_fetched_text_identity to every "
        "cached payload's own joined chunk text before candidate mining, dropping (and "
        "recording in info['wrong_work_sources']) any source whose text does not match its "
        "own title/author. The frozen 30-item test set's one disclosed mis-fetch "
        "(10.55593/ej.26103a4) is kept unconditionally via the module-level "
        "KNOWN_IDENTITY_EXCEPTIONS, so this default-on check does not break the "
        "byte-identical --skip-fetch rebuild rule; no recipe needs this flag passed "
        "explicitly any more. Use --no-identity-check to disable the check entirely.",
    )
    ap.add_argument(
        "--no-identity-check", dest="identity_check_cache", action="store_false",
        help="disable the cache-side wrong-work identity check entirely (the opt-out; "
        "the check used to ship fail-open as an opt-in flag no repository recipe ever "
        "passed).",
    )
    ap.add_argument(
        "--identity-check-exempt", dest="identity_check_exempt", action="append", default=[],
        help="DOI (repeatable) to keep even when the identity check is on, in addition to "
        "the module-level KNOWN_IDENTITY_EXCEPTIONS (always applied): a named, disclosed "
        "exception, not a way to silence a fresh mis-fetch discovery.",
    )
    return resolve_path_args(ap.parse_args(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.probe_journals:
        mailto = load_dotenv_values(REPO_ROOT / ".env").get("OPENALEX_EMAIL") or ""
        if not mailto:
            raise SystemExit("OPENALEX_EMAIL missing from .env (needed for OpenAlex/Unpaywall)")
        rows = asyncio.run(probe_journals(mailto, candidate_timeout_s=args.candidate_timeout_s))
        write_json(args.probe_out, rows)
        print(f"wrote {args.probe_out} ({len(rows)} rows); nothing selected, n_attempts unchanged")
        return 0

    build_path = args.out.parent / f"{args.out.stem}.build.json"
    previous = read_json(build_path, {}) or {}
    excluded = load_excluded_dois(args.exclude_sources)
    quotas = parse_quotas(args.quotas)
    alteration_quotas = parse_alteration_quotas(
        args.alteration_quotas, altered_quota=quotas["altered"]
    )
    n_items_requested = sum(quotas.values())
    print(f"quotas: {quotas} (n_items={n_items_requested})")

    journal_notes: dict[str, Any] = {}
    attempts: list[dict[str, Any]] = []
    if not args.skip_fetch:
        mailto = load_dotenv_values(REPO_ROOT / ".env").get("OPENALEX_EMAIL") or ""
        if not mailto:
            raise SystemExit("OPENALEX_EMAIL missing from .env (needed for OpenAlex/Unpaywall)")
        asyncio.run(
            acquire_sources(
                sources_path=args.sources,
                cache_dir=args.cache_dir,
                n_sources=args.n_sources,
                min_chars=args.min_chars,
                min_chunks=args.min_chunks,
                per_journal=args.per_journal,
                mailto=mailto,
                max_failures=args.max_failures,
                notes=journal_notes,
                excluded=excluded,
                attempts_out=attempts,
                candidate_timeout_s=args.candidate_timeout_s,
            )
        )
        n_attempts = int(previous.get("n_attempts") or 0) + 1
    else:
        # No fetching invocation happened, so the
        # acquisition record's own attempt count is unchanged, not incremented.
        n_attempts = previous.get("n_attempts")
    sources = read_json(args.sources, []) or []
    dois = [s["doi"] for s in sources] or None
    veto = load_veto(args.veto)
    expected = dict(EXPECTED)
    if args.altered_expected == "strict":
        expected["altered"] = ["unsupported"]
    items, info = build_items_from_cache(
        args.cache_dir, args.seed, dois=dois, min_chunks=args.min_chunks, veto=veto,
        veto_file=args.veto if args.veto.exists() else None, expected=expected,
        altered_expected_mode=args.altered_expected, quotas=quotas,
        alteration_quotas=alteration_quotas, veto_strict=args.veto_strict,
        identity_check=args.identity_check_cache,
        identity_check_exceptions=args.identity_check_exempt,
    )
    if args.out.exists():
        args.out.unlink()
    append_jsonl(args.out, items)
    info["sources_file"] = args.sources.name if args.sources.exists() else None

    def _merged(current: Any, key: str) -> Any:
        """On --skip-fetch, keep the historical record: nothing was fetched this run, so an
        empty/default current value must never silently erase what a real acquisition wrote
        at the same build-json path."""
        if args.skip_fetch and key in previous:
            return previous[key]
        return current

    info["journals"] = _merged(journal_notes, "journals")
    info["attempts"] = _merged(attempts, "attempts")
    info["journals_rejected"] = _merged(
        [dict(x) for x in REJECTED_JOURNAL_CANDIDATES], "journals_rejected"
    )
    info["n_attempts"] = n_attempts
    info["min_chars"] = _merged(args.min_chars, "min_chars")
    info["min_chunks"] = _merged(args.min_chunks, "min_chunks")
    info["excluded_dois"] = _merged(sorted(excluded), "excluded_dois")
    info["n_excluded"] = _merged(len(excluded), "n_excluded")
    info["quotas"] = dict(quotas)
    info["n_items_requested"] = n_items_requested
    info["alteration_quotas"] = dict(alteration_quotas) if alteration_quotas else None
    if args.overlap_with:
        info["original_sentence_overlap"] = compute_original_sentence_overlap(
            items, args.overlap_with
        )
    write_json(build_path, info)
    print(f"wrote {args.out} ({len(items)} items, {json.dumps(info['counts'])}) and {build_path}")
    # The selected DOI set and the excluded DOI set must be disjoint (a
    # --skip-fetch run reuses whatever --sources already lists, so this must be checked
    # unconditionally, not only on the fetch path above).
    selected_dois = {s["doi"] for s in sources if isinstance(s, Mapping) and s.get("doi")}
    overlap = selected_dois & excluded
    if overlap:
        raise SystemExit(
            f"{args.sources} selects {sorted(overlap)}, which --exclude-sources also lists; "
            "the dev and test paper sets must be disjoint"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
