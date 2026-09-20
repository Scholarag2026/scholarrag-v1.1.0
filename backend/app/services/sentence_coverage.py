"""Deterministic sentence-residue and frame classification.

The invariant this module exists to check: a delivered, cited sentence asserts nothing
beyond what its own verified propositions (plus its own rendered citations, plus a
closed, grammatical class of discourse frames) cover. Shared by
``app.agents.citation_link_agent`` (the write-time coverage check that sets
``coverage_incomplete``/``residue_spans``/``meta_evaluation`` on a validated
``CitationLink``) and ``app.services.fulltext`` (the finalize gate's own,
independent residue check and deterministic trailing excision).

``demo/check_delivered.py`` deliberately does not import this module (its own module
docstring: demo never imports the backend) -- it carries its own, independently written
copy of the same logic (rules 10 and 11), kept in step with this one by the shared
fixture ``demo/fixtures/frame_spans.json`` (design section 3.4): both sides' tests read
the same fixture and must agree on every span's frame class.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ɏ]+")


def _fold(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return folded.casefold()


def tokens_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Every maximal run of letters/digits in *text*, folded (accents dropped, case
    folded), with its own character offsets."""
    return [(_fold(m.group(0)), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def token_list(text: str) -> list[str]:
    return [t for t, _s, _e in tokens_with_spans(text)]


def _singular(tok: str) -> str:
    for suffix in ("ies", "es", "s"):
        if len(tok) > 4 and tok.endswith(suffix):
            return tok[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return tok


# --- committed lexicons (identical to the prototype's own) -----------------------

STOP = frozenset(
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

#: `caution`'s own four inflections, and seven other common reporting verbs of the same
#: plain-attribution class (`warn`, `propose`, `contend`, `maintain`,
#: `emphasise`/`emphasize`, `stress`, `acknowledge`), are included here and,
#: identically, in `REPORTING_VERBS` below and in `demo.check_delivered._RESIDUE_ATTRIB`,
#: so that a bare attribution clause like "cautioned that" -- exactly like the
#: already-accepted "reports that"/"noted that" -- is never treated as residue on an
#: otherwise fully verified sentence such as "Hyland (2025) cautioned that automated
#: programmes failed to identify many important errors, potentially misleading students
#: about their grammatical accuracy." Deliberately NOT added:
#: `suggest`, `indicate`, `highlight`, `confirm`, `reveal` -- each already lives in
#: `NEVER_FRAME` below, by design (an inference or ranking marker, not a bare
#: report of what a source said), and adding it here would contradict that.
#:
#: `concede`'s own four inflections are added the same way, for the same reason
#: ("Yet the authors conceded that their non-randomised, descriptive design is
#: necessarily descriptive (Liu and Wu, 2019)." lost its own residue check over
#: "conceded that", the same lexicon-gap shape as `caution` above, on a verb the
#: earlier fix did not add) -- identically to `REPORTING_VERBS` below and to
#: `demo.check_delivered._RESIDUE_ATTRIB`.
ATTRIB = frozenset(
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

#: A token that can never appear inside a permitted frame span, regardless of its own
#: budget or position: an inference marker or a ranking term (shared with the
#: comparative meta-evaluation guard below).
NEVER_FRAME = frozenset(
    """
suggest suggests suggesting suggested imply implies implying implied
indicate indicates indicating indicated underscore underscores underscoring
highlight highlights highlighting confirm confirms confirming
prove proves proving proven reveal reveals revealing revealed establish establishes
established establishing adjudicate adjudicates
strongest clearest best greatest weakest definitive conclusive foremost primary
""".split()
)

NUMBER_WORDS = frozenset(
    """
one two three four five six seven eight nine ten eleven twelve thirteen fourteen
fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy
eighty ninety hundred thousand half dozen
""".split()
)

#: The integer value of every `NUMBER_WORDS` entry that has one -- used only to
#: normalise a residue's own number word onto the same key a digit form of the same
#: count normalises to (`_numeral_key`), so "seventy" and "70" compare equal.
#: "half" is deliberately excluded: it never names a count on its own (it always
#: modifies another number, "one and a half"), so it has no single value to map.
NUMBER_WORD_VALUES: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "thousand": 1000, "dozen": 12,
}

_DIGIT_RUN_RE = re.compile(r"\d+(?:\.\d+)?")


def _numeral_key(value: float) -> str:
    """*value* as the one canonical string key every numeral -- digit form or
    number word -- normalises to for comparison: a whole number renders with no
    decimal point (``70``, not ``70.0``), so "seventy" (`NUMBER_WORD_VALUES`) and
    "70" (a digit run) compare equal."""
    return str(int(value)) if value == int(value) else str(value)


def numeral_keys_in_text(text: str) -> set[str]:
    """Every numeral *text* itself asserts -- a digit run (`_DIGIT_RUN_RE`) or a
    `NUMBER_WORD_VALUES` word -- each normalised to `_numeral_key`. Shared by a
    residue's own numeral set and by the numerals a verified evidence quote
    carries, so the two are compared on the same footing regardless of which form
    (digit or word) either one happens to use."""
    keys: set[str] = set()
    for match in _DIGIT_RUN_RE.finditer(text or ""):
        try:
            keys.add(_numeral_key(float(match.group(0))))
        except ValueError:
            continue
    for word in token_list(text or ""):
        value = NUMBER_WORD_VALUES.get(word)
        if value is not None:
            keys.add(_numeral_key(value))
    return keys


def covered_numeral_keys(evidence_quotes: Sequence[str] | None) -> frozenset[str]:
    """The union of `numeral_keys_in_text` over every quote of *evidence_quotes* --
    the verified evidence quotes of a sentence's own kept claims -- for `is_frame`
    to check a residue's own numeral against."""
    keys: set[str] = set()
    for quote in evidence_quotes or []:
        keys.update(numeral_keys_in_text(quote))
    return frozenset(keys)

QUANTIFIERS = frozenset(
    """
most least few fewer many much several majority minority all none some only
significantly substantially dramatically considerably
""".split()
)

#: "regarding" and "concerning" were missing here, so a comma-set opener such as
#: "Regarding error detection and accuracy, Luo et al. (2025) report that ..." could
#: never reach the F3 branch below even when its own leftover content tokens sat
#: within budget (blocked twice over: once by this gap and once by the token
#: budget). "with respect to" is deliberately NOT added: every entry here is matched as a
#: single token (`residue.tokens[0] in PREP_OPENERS`, `is_frame` below), and this
#: set carries no multi-word entry to extend that check for -- adding the phrase here
#: would only add its own first word "with", already present, and silently match
#: nothing of the phrase itself.
PREP_OPENERS = frozenset(
    """
on in by for with within across beyond alongside among at from after before
despite during through under over unlike like besides against beside
regarding concerning
""".split()
)

#: "where" was missing here, so a comma-set opener naming its own topic instead of
#: a preposition or a connective adverb -- "Where peer comments focus is concerned,
#: feedback in one peer-mediated study covered global aspects such as content,
#: organization, and logic (Li & Hebert, 2023)." -- could never reach the F3 branch
#: below even though its own leftover content tokens sat within budget: the first
#: token check (`residue.tokens[0] in (PREP_OPENERS | DISCOURSE_OPENERS)`) found
#: neither list. "where" is already in `SUBORDINATORS` below, for the unrelated
#: purpose of blocking F1 when a residue ENDS in it (a dependent clause the writer
#: is attaching to what follows); membership in `SUBORDINATORS` never confers
#: opener status, which only ever depends on `PREP_OPENERS`/`DISCOURSE_OPENERS`
#: matching the residue's own FIRST token, so both memberships are correct at once.
#: "notably" was missing here too: a bare one-token emphasis-marker residue,
#: comma-set, asserting nothing of its own -- one observed instance was the single
#: word "Notably" (`writing_result_3.json`'s `coverage_incomplete_reasons`), which
#: blocked the sentence it opened from ever reaching the F3 branch below even though
#: it carried no content tokens of its own to budget against. The same class of gap
#: recurs across several discourse-opener words ("regarding", "where", "extending"), each
#: fixed by adding the missing word here; "notably" itself needed the same fix.
DISCOURSE_OPENERS = frozenset(
    """
however moreover furthermore similarly likewise conversely nevertheless nonetheless
meanwhile overall together taken additionally also first second third fourth finally
thus therefore hence consequently accordingly instead again indeed still where notably
""".split()
)

#: A residue whose own last token is one of these never qualifies as the bare
#: grammatical subject the F1 branch below permits: it is an independent
#: clause the writer joined to the next one with a coordinating conjunction, not a
#: subject abutting its own verb. Shared, by name and by value, with
#: ``demo.check_delivered``'s own independent copy of this classifier, so the two
#: never drift on the bound this finding requires.
COORDINATING_CONJUNCTIONS = frozenset({"and", "or", "but", "yet", "so", "nor"})


def ends_in_coordinating_conjunction(tokens: list[str]) -> bool:
    """True when *tokens* (a residue's own token list) ends in a coordinating
    conjunction -- the one shared predicate the frame gate and
    ``demo.check_delivered``'s own copy both call, so a leading residue such as
    "Teacher feedback remains the gold standard for second language writing
    instruction and" is never mistaken for a bare subject."""
    return bool(tokens) and tokens[-1] in COORDINATING_CONJUNCTIONS

#: A residue whose own last token is a subordinating conjunction or
#: subordinator is a dependent clause the writer is attaching to what follows, not the
#: bare grammatical subject the F1 branch below permits: "Automated feedback
#: cannot replace a teacher because" asserts a whole unverified clause of its own
#: merely for opening the sentence and carrying no internal punctuation. Shared, by
#: name and by value, with ``demo.check_delivered``'s own independent copy, so the two
#: never drift on the bound this finding requires.
SUBORDINATORS = frozenset(
    """
because although while whereas since if unless until when as that which who whose
where
""".split()
)


def ends_in_subordinator(tokens: list[str]) -> bool:
    """True when *tokens* (a residue's own token list) ends in a subordinating
    conjunction or subordinator -- the shared predicate ``demo.check_delivered``'s own
    copy also calls, so a leading residue such as "Automated feedback cannot replace a
    teacher because" is never mistaken for a bare subject."""
    return bool(tokens) and tokens[-1] in SUBORDINATORS

#: F2/F3's own cap on how much a permitted lead-in may assert (design section 3.3).
FRAME_BUDGET = 3

_CITATION_RE = re.compile(
    r"[\[\(][^\[\]\(\)]{0,120}\b(?:19|20)\d{2}[a-z]?\b[^\[\]\(\)]{0,20}[\]\)]"
)
_NARRATIVE_CITATION_RE = re.compile(
    r"(?:[A-ZÀ-ɏ][\w'’\-]*\.?[\s,]*(?:and|&|et\s+al\.)?[\s,]*){1,6}"
    r"\(\s*(?:19|20)\d{2}[a-z]?\s*\)"
)


@dataclass
class Residue:
    """One maximal run of sentence tokens that no covering span (a proposition, a
    rendered citation) reaches -- the unit both the coverage check (edit 3) and the
    finalize gate's own trailing excision (edit 6) reason about."""

    text: str
    tokens: list[str]
    start: int
    end: int
    before_first_claim: bool
    is_trailing: bool
    trailing_gap: str
    leading_gap: str


def _covered_indices(sentence: str, spans: list[str]) -> set[int]:
    """Indices of *sentence*'s own tokens covered by any of *spans*, matched as a
    contiguous token subsequence (every occurrence, not just the first). Used for
    residue DETECTION: whether a citation mark is verified or not, it is never itself
    residue, so every
    occurrence of a citation-shaped span is covered regardless of how many times it is
    listed.

    `trailing_excision_cut` below needs a stricter, non-blanket notion of "covered" (a
    citation mention belonging to an UNVERIFIED, about-to-be-excised proposition must
    not mask the residue in front of it just because it looks identical to a KEPT
    citation elsewhere in the same sentence) and uses its own `_covered_indices_strict`
    instead.
    """
    sent_tokens = [t for t, _s, _e in tokens_with_spans(sentence)]
    covered: set[int] = set()
    for span in spans:
        needle = token_list(span or "")
        if not needle:
            continue
        n = len(needle)
        for i in range(len(sent_tokens) - n + 1):
            if sent_tokens[i : i + n] == needle:
                covered.update(range(i, i + n))
    return covered


def _covered_indices_strict(sentence: str, spans: list[str]) -> set[int]:
    """Like `_covered_indices`, but consumes exactly one occurrence per entry of
    *spans* (leftmost still-uncovered occurrence first), and never auto-detects a
    citation-shaped span the caller did not list explicitly. Used only by
    `trailing_excision_cut`: a sentence rendering the same citation string twice, once
    for a kept proposition and once for one being excised, must not have the excised
    proposition's own trailing mention silently covered just because the string is
    identical to the kept one's."""
    sent_tokens = [t for t, _s, _e in tokens_with_spans(sentence)]
    covered: set[int] = set()
    for span in spans:
        needle = token_list(span or "")
        if not needle:
            continue
        n = len(needle)
        for i in range(len(sent_tokens) - n + 1):
            if any(j in covered for j in range(i, i + n)):
                continue
            if sent_tokens[i : i + n] == needle:
                covered.update(range(i, i + n))
                break
    return covered


def residues(
    sentence: str, claims: list[str], citations: list[str] | None = None
) -> list[Residue]:
    """Every maximal uncovered token run of *sentence*, once it is covered by *claims*
    (verified propositions or claim texts) and *citations* (rendered citation strings),
    plus every bracketed-year or narrative author-year citation this module's own
    regexes recognise inside the sentence."""
    spans = list(claims)
    spans += list(citations or [])
    spans += [m.group(0) for m in _CITATION_RE.finditer(sentence)]
    spans += [m.group(0) for m in _NARRATIVE_CITATION_RE.finditer(sentence)]
    toks = tokens_with_spans(sentence)
    covered = _covered_indices(sentence, spans)
    claim_covered = _covered_indices(sentence, list(claims))
    first_claim = min(claim_covered) if claim_covered else len(toks)
    out: list[Residue] = []
    run: list[int] = []
    for i in range(len(toks)):
        if i in covered:
            if run:
                out.append(_make_residue(sentence, toks, run, first_claim, len(toks)))
                run = []
        else:
            run.append(i)
    if run:
        out.append(_make_residue(sentence, toks, run, first_claim, len(toks)))
    return out


def _make_residue(sentence, toks, run, first_claim, total_tokens) -> Residue:
    start = toks[run[0]][1]
    end = toks[run[-1]][2]
    prev_end = toks[run[0] - 1][2] if run[0] > 0 else 0
    next_start = toks[run[-1] + 1][1] if run[-1] + 1 < len(toks) else len(sentence)
    return Residue(
        text=sentence[start:end],
        tokens=[toks[i][0] for i in run],
        start=start,
        end=end,
        before_first_claim=run[-1] < first_claim,
        is_trailing=run[-1] == total_tokens - 1,
        trailing_gap=sentence[end:next_start],
        leading_gap=sentence[prev_end:start],
    )


def content_tokens(residue: Residue, claim_tokens: set[str]) -> list[str]:
    """*residue*'s own tokens once function words, attribution/reporting verbs, and
    anything whose crude singular form already occurs in a covering claim are removed
    -- what is left is what the residue actually asserts on its own."""
    claim_stems = {_singular(t) for t in claim_tokens}
    out = []
    for tok in residue.tokens:
        if tok in STOP or tok in ATTRIB:
            continue
        if _singular(tok) in claim_stems:
            continue
        out.append(tok)
    return out


def is_frame(
    residue: Residue,
    claim_tokens: set[str],
    covered_numerals: frozenset[str] = frozenset(),
) -> str | None:
    """The frame class ("f1", "f2" or "f3") *residue* is permitted as, or ``None`` when
    it is not a permitted frame at all (design section 3.3).

    A residue that carries a numeral (a digit run or a `NUMBER_WORDS` word) is
    ordinarily never a permitted frame at all, whatever else it says or how short
    it is: a bare "assigned to four feedback conditions" lead-in asserts a fact of
    its own no citation on the sentence necessarily covers. *covered_numerals* --
    the union of `numeral_keys_in_text` over the sentence's own verified evidence
    quotes (`covered_numeral_keys`) -- narrows that: when EVERY numeral the residue
    itself carries (`numeral_keys_in_text(residue.text)`) is also in
    *covered_numerals*, the numeral no longer disqualifies the residue on its own;
    it still has to clear every other rule below (the three-content-token budget,
    the never-frame/quantifier lexicons, the colon/comma-opener shape) exactly as
    a non-numeric residue does. A residue carrying more than one numeral, only
    some of which are covered, is not exempted at all -- every one of its own
    numerals must be covered, not just one."""
    left = content_tokens(residue, claim_tokens)
    residue_numerals = numeral_keys_in_text(residue.text)
    numeric_raw = any(t.isdigit() or t in NUMBER_WORDS for t in residue.tokens) or bool(
        re.search(r"[0-9%]", residue.text)
    )
    numeric = numeric_raw and not (residue_numerals and residue_numerals <= covered_numerals)
    if not left and not numeric and not (set(residue.tokens) & NEVER_FRAME):
        return "f1"
    if set(residue.tokens) & NEVER_FRAME:
        return None
    if numeric:
        return None
    if set(residue.tokens) & QUANTIFIERS:
        return None
    # A residue that opens the
    # sentence and abuts the covered proposition's own leading verb with no
    # clause-boundary punctuation at all between them is that verb's own grammatical
    # subject, not a separable embellishment -- there is no comma or colon anywhere
    # for a deterministic cut to use, so the two cannot be split into "frame" and
    # "content" at all. Unconditional on the three-token budget below: the budget
    # exists to cap what an OPTIONAL lead-in may assert before being discarded, and
    # this lead-in is not optional, it is the clause's own subject.
    #
    # Gated on the residue's OWN text carrying no boundary punctuation either (design
    # row 4's own shape: "Affect
    # also shaped outcomes, though" is sentence-initial and abuts its own trailing
    # content with no punctuation between them too, but the residue itself is already a
    # complete clause -- subject "Affect", verb "shaped", object "outcomes" -- set off
    # from what follows by its OWN internal comma before the subordinator "though". A
    # bare subject never contains a clause boundary of its own; a residue that does is
    # asserting something complete on its own account and must stay flagged.)
    #
    # Also gated on the residue's own last token not being a
    # coordinating conjunction ("and", "or", "but", "yet", "so", "nor"). Without this
    # bound the branch is unconditional on length, so a whole independent clause of
    # unverified, evaluative material ("Teacher feedback remains the gold standard for
    # second language writing instruction and") was accepted as a bare subject merely
    # for opening the sentence and carrying no internal punctuation of its own -- a
    # coordinating conjunction at the residue's own end is the writer joining two
    # clauses, never a subject abutting its own verb.
    #
    # Two further bounds, for the same reason. First, the
    # residue's own last token must not be a subordinator either ("because", "although",
    # "while", ...) -- "Automated feedback cannot replace a teacher because" is a
    # dependent clause the writer is attaching to what follows, not a subject. Second,
    # the branch is now capped by the SAME three-content-token budget every other frame
    # shape (F2, F3) already answers to (`left`, computed above): without it, "While
    # teachers remain indispensable to writing instruction" and "Research that has
    # repeatedly demonstrated the limits of automated feedback over the past decade
    # shows" were accepted at any length merely for not ending in one of the six
    # coordinating conjunctions, each asserting an independent clause of unverified
    # material that rule 10 exists to catch.
    if (
        residue.before_first_claim
        and residue.start == 0
        and not residue.trailing_gap.strip()
        and not any(ch in residue.text for ch in BOUNDARY_CHARS)
        and not ends_in_coordinating_conjunction(residue.tokens)
        and not ends_in_subordinator(residue.tokens)
        and len(left) <= FRAME_BUDGET
    ):
        return "f1"
    if len(left) > FRAME_BUDGET:
        return None
    if residue.before_first_claim and ":" in residue.trailing_gap:
        return "f2"
    if (
        residue.before_first_claim
        and "," in residue.trailing_gap
        and residue.tokens
        and residue.tokens[0] in (PREP_OPENERS | DISCOURSE_OPENERS)
    ):
        return "f3"
    return None


def sentence_residue_violations(
    sentence: str,
    propositions: list[str],
    citations: list[str] | None = None,
    evidence_quotes: Sequence[str] | None = None,
) -> tuple[list[Residue], list[Residue]]:
    """``(non_frame_residues, frame_residues)`` for *sentence*, given the propositions
    and citations that are meant to cover it. *evidence_quotes* -- the verified
    evidence quotes of the sentence's own kept claims -- narrows `is_frame`'s own
    numeral disqualification (`covered_numeral_keys`); omitted or empty, every
    numeral disqualifies exactly as before this parameter existed."""
    claim_tokens: set[str] = set()
    for p in propositions:
        claim_tokens.update(token_list(p))
    covered_numerals = covered_numeral_keys(evidence_quotes)
    res = residues(sentence, list(propositions), list(citations or []))
    non_frame = [r for r in res if is_frame(r, claim_tokens, covered_numerals) is None]
    frame = [r for r in res if is_frame(r, claim_tokens, covered_numerals) is not None]
    return non_frame, frame


def coverage_incomplete(
    sentence: str,
    propositions: list[str],
    citations: list[str] | None = None,
    evidence_quotes: Sequence[str] | None = None,
) -> tuple[bool, list[str]]:
    """(is_incomplete, residue_spans): whether *sentence* has at least one non-frame
    residue once covered by *propositions* and *citations*, and every such residue's own
    verbatim text (edit 3's own ``coverage_incomplete``/``residue_spans``)."""
    non_frame, _frame = sentence_residue_violations(
        sentence, propositions, citations, evidence_quotes
    )
    return bool(non_frame), [r.text for r in non_frame]


# --- edit 8: the comparative meta-evaluation guard --------------------------------

#: The adjective list shared by the "most"/"least" superlative-degree forms below
#: and the "more"/"less" comparative-degree forms alongside them -- "controlled" is
#: the one addition, needed for "Experimental work offers the most controlled
#: comparison."
_DEGREE_ADJECTIVES = r"(?:direct|directly|convincing|compelling|robust|striking|controlled)"
_SUPERLATIVE = (
    rf"(?:strongest|clearest|best|(?:most|least)\s+{_DEGREE_ADJECTIVES}|weakest|"
    r"definitive|conclusive|foremost|leading|prime|chief)"
)
#: Comparative evaluations of the literature -- "Li and Hébert (2023) provide
#: stronger evidence: ..." -- a bare comparative ("stronger"/"clearer"/"weaker") or
#: "more"/"less" plus the same bounded adjective list `_SUPERLATIVE` already answers
#: to, never a bare "more"/"less" alone: an ordinary comparative finding about the
#: studied phenomenon ("teacher feedback addressed more error types than AWE",
#: "recall rose from 10% to 55%") carries neither this adjective list nor,
#: immediately after it, one of `_LIT_NOUN`'s own literature nouns, so it is never
#: reached by either alternative below.
_COMPARATIVE = rf"(?:stronger|clearer|weaker|(?:more|less)\s+{_DEGREE_ADJECTIVES})"
_SUPERLATIVE_OR_COMPARATIVE = rf"(?:{_SUPERLATIVE}|{_COMPARATIVE})"
#: "comparison", "design", "test" and "tests" are added for "Experimental work
#: offers the most controlled comparison.": evidence, support, comparison, design,
#: study, work and test are the nouns a comparative evaluation of the literature can
#: land on.
_LIT_NOUN = (
    r"(?:evidence|support|study|studies|research|work|finding|findings|demonstration|"
    r"account|case|treatment|example|illustration|comparison|design|test|tests)"
)
META_EVALUATION_RE = re.compile(
    rf"\b{_SUPERLATIVE_OR_COMPARATIVE}\b(?:\s+\w+){{0,2}}\s+\b{_LIT_NOUN}\b"
    rf"|\b{_LIT_NOUN}\b\s+is\s+(?:\w+\s+){{0,2}}\b{_SUPERLATIVE_OR_COMPARATIVE}\b"
    rf"|\bis\s+(?:treated|discussed|addressed|examined|shown|handled)\s+"
    rf"{_SUPERLATIVE_OR_COMPARATIVE}\b",
    re.IGNORECASE,
)


def has_meta_evaluation(text: str) -> bool:
    """True when *text* carries a comparative meta-evaluation of the literature (a
    superlative or ranking term within two words of a literature noun, or the passive
    "is treated/discussed/... most directly" frame) -- edit 8."""
    return bool(META_EVALUATION_RE.search(text or ""))


def meta_evaluation_match(text: str) -> str | None:
    match = META_EVALUATION_RE.search(text or "")
    return match.group(0) if match else None


# --- edit 6: deterministic trailing excision --------------------------------------

#: A clause boundary that licenses a deterministic trailing cut (design edit 6,
#: outcome 2): a comma, a semicolon, a colon, or an em/en dash.
BOUNDARY_CHARS = (",", ";", ":", "—", "–")


def _contains_subsequence(haystack: list[str], needle: list[str]) -> bool:
    if not needle:
        return False
    n = len(needle)
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def _strict_residues(sentence: str, claims: list[str], citations: list[str] | None) -> list[Residue]:
    """`residues`, but covered by `_covered_indices_strict` (consume-per-entry, no
    citation auto-detection) instead -- see that function's own docstring for why
    `trailing_excision_cut` needs this stricter notion of "covered"."""
    spans = list(claims) + list(citations or [])
    toks = tokens_with_spans(sentence)
    covered = _covered_indices_strict(sentence, spans)
    claim_covered = _covered_indices_strict(sentence, list(claims))
    first_claim = min(claim_covered) if claim_covered else len(toks)
    out: list[Residue] = []
    run: list[int] = []
    for i in range(len(toks)):
        if i in covered:
            if run:
                out.append(_make_residue(sentence, toks, run, first_claim, len(toks)))
                run = []
        else:
            run.append(i)
    if run:
        out.append(_make_residue(sentence, toks, run, first_claim, len(toks)))
    return out


def trailing_excision_cut(
    sentence: str, verified_propositions: list[str], citations: list[str] | None = None
) -> int | None:
    """The character offset to cut *sentence* at for a safe, deterministic trailing
    excision (design edit 6, outcome 2), or ``None`` when no such cut exists.

    A cut exists only when: exactly one non-frame residue remains once *sentence* is
    covered by *verified_propositions* and *citations*; that residue is the sentence's
    own trailing run (nothing covered follows it); its own leading gap (the text between
    the previous covered token and it) carries a boundary character
    (`BOUNDARY_CHARS`); and the prefix up to that boundary contains at least one whole
    verified proposition, verbatim. The returned offset is the position of the boundary
    character itself, so slicing ``sentence[:cut]`` removes the boundary mark along with
    the residue that follows it.

    Callers pass only KEPT (verified) propositions and the citations of KEPT links --
    never a dropped or unverified link's own citation text, even when it renders
    identically to a kept one elsewhere in the same sentence (`_covered_indices_strict`).
    """
    claim_tokens: set[str] = set()
    for p in verified_propositions:
        claim_tokens.update(token_list(p))
    non_frame = [
        r
        for r in _strict_residues(sentence, verified_propositions, citations)
        if is_frame(r, claim_tokens) is None
    ]
    if len(non_frame) != 1:
        return None
    only = non_frame[0]
    if not only.is_trailing:
        return None
    boundary_pos: int | None = None
    for i, ch in enumerate(only.leading_gap):
        if ch in BOUNDARY_CHARS:
            boundary_pos = i
    if boundary_pos is None:
        return None
    prev_end = only.start - len(only.leading_gap)
    cut_at = prev_end + boundary_pos
    if cut_at <= 0:
        return None
    prefix_tokens = token_list(sentence[:cut_at])
    if not any(
        _contains_subsequence(prefix_tokens, token_list(p))
        for p in verified_propositions
        if (p or "").strip()
    ):
        return None
    return cut_at


def apply_trailing_excision(sentence: str, cut_at: int) -> str:
    """*sentence* truncated at *cut_at* (a `trailing_excision_cut` result), with its own
    terminal punctuation re-applied so the excised sentence still reads as a complete
    one."""
    prefix = sentence[:cut_at].rstrip().rstrip(",;:—–").rstrip()
    stripped_original = sentence.rstrip()
    terminal = stripped_original[-1] if stripped_original and stripped_original[-1] in ".?!" else "."
    return f"{prefix}{terminal}"


# --- lead-in residues -------------------------------------------------------------
#
# A coverage-incomplete residue whose own leading attribution carries a numeral
# ("a 16-week case study", "seventy ESL students") or a noun outside `ATTRIB` ("a
# case") correctly blocks F1 (F1's own budget and `numeric` checks, above), even
# when the sentence's own proposition -- everything the residue is not -- has
# already verified in full: this class of sentence is removed whole by the
# ordinary coverage-incomplete path below, deliberately, rather than rewritten. A
# deterministic rewrite that manufactures a subject-less "VERB that ..." skeleton,
# or one that keeps a subject the lead-in itself did not carry, is not part of the
# guarantee this module exists to check (deliver only what is verified, or omit
# it); a lead-in this narrow cannot be told apart, in general, from a genuine
# editorial generalisation or inference asserted on its own account, so no
# lead-in is ever rewritten here, regardless of shape.

#: The verb-shaped subset of `ATTRIB` -- a reporting verb, as opposed to a literature
#: noun ("study", "research", "paper", ...) or an adverb/conjunction ("similarly",
#: "according", ...) also carried by that lexicon. Used by `opens_with_reporting_
#: verb`, below, and by the participle-after-attribution check further down.
REPORTING_VERBS = frozenset(
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

#: The base (dictionary) form of every verb `REPORTING_VERBS` carries, one entry
#: per verb -- used only to derive `_PAST_OR_THIRD_PERSON_REPORTING_VERBS`, below,
#: by subtraction.
_REPORTING_VERB_BASE_FORMS = frozenset(
    """
add note report find show document examine demonstrate observe argue conclude
state describe write caution warn propose contend maintain emphasise emphasize
stress acknowledge concede
""".split()
)

#: The subset of `REPORTING_VERBS` that is a present-participle/gerund
#: ("-ing") form -- computed, not hand-listed, so a future addition to
#: `REPORTING_VERBS` is covered automatically. Every current member is one of the ten
#: verbs added alongside `caution`/`concede` (`cautioning`, `warning`, `proposing`,
#: `contending`, `maintaining`, `emphasising`, `emphasizing`, `stressing`,
#: `acknowledging`, `conceding`); none of the fifteen pre-existing reporting verbs
#: (`add`, `note`, `report`, `find`, `show`, `document`, `examine`, `demonstrate`,
#: `observe`, `argue`, `conclude`, `state`, `describe`, `write`) has an `-ing` member
#: in this lexicon at all.
_ING_REPORTING_VERBS = frozenset(v for v in REPORTING_VERBS if v.endswith("ing"))

#: Every inflection of `REPORTING_VERBS` that is neither a base form nor a
#: present-participle/gerund -- a past-tense or third-person-singular-present form
#: ("showed", "shows", "found", "reports", "noted", ...), the shape a lead-in
#: whose own subject has been cut away leaves behind. Deliberately excludes the
#: base form ("show", "report", ...): a bare base form immediately followed by
#: "that" is not a shape any part of this pipeline produces.
_PAST_OR_THIRD_PERSON_REPORTING_VERBS = frozenset(
    v
    for v in REPORTING_VERBS
    if v not in _REPORTING_VERB_BASE_FORMS and v not in _ING_REPORTING_VERBS
)


# --- a narrow, deterministic check for a sentence whose own grammatical subject
# is a citation, with a bare participle/gerund immediately after it and no finite
# verb anywhere in between. Shared with `demo.check_delivered`'s own independent
# copy (`_opens_with_participle_after_attribution`, rule 12), the same way
# `ends_in_coordinating_conjunction`/`ends_in_subordinator`, above, are: named and
# valued identically, so the two never drift.
# ---------------------------------------------------------------------------


def opens_with_participle_after_attribution(sentence: str) -> bool:
    """True when *sentence* opens with a citation-shaped subject -- a narrative
    "Author (Year)" attribution, or a bracketed "(Author, Year)" one -- immediately
    followed (nothing but whitespace between them) by a present-participle/gerund
    ("-ing") reporting-verb form, with no finite verb anywhere before it: a subject
    with no finite verb of its own at all, e.g. "Teng and Ma (2024) cautioning that
    their scale may not fully reflect learners' feedback literacy in academic
    writing."

    Deliberately narrow: the participle must directly abut the citation. "Teng and
    Ma (2024) reported cautioning that ..." -- a finite verb, "reported", governing
    the participle as its own object -- never matches, because "cautioning" is not
    the word immediately after the citation."""
    stripped = (sentence or "").lstrip()
    match = _NARRATIVE_CITATION_RE.match(stripped) or _CITATION_RE.match(stripped)
    if not match:
        return False
    next_word = re.match(r"\s+([A-Za-zÀ-ɏ]+)", stripped[match.end() :])
    if not next_word:
        return False
    return _fold(next_word.group(1)) in _ING_REPORTING_VERBS


# --- a narrow, deterministic check for a sentence whose own first two tokens are
# a past-tense or third-person reporting verb followed by "that" -- the shape a
# lead-in whose own subject has been cut away leaves behind, e.g. "Showed that
# one student resubmitted her essay 13 times while another made one resubmission
# (Zhang & Hyland, 2018)." An ordinary sentence that merely happens to open with a
# reporting verb in some other shape ("Studies show that ...", "Reports from
# teachers suggest ...") is not this shape and does not match. Shared with
# `demo.check_delivered`'s own independent copy (`_opens_with_reporting_verb`),
# the same way `opens_with_participle_after_attribution` above is.
# ---------------------------------------------------------------------------


def opens_with_reporting_verb(sentence: str) -> bool:
    """True when *sentence* itself opens with a past-tense or third-person-
    singular reporting verb (any non-base, non-gerund inflection in
    `REPORTING_VERBS`, case-insensitive) immediately followed by "that", with
    nothing before either word -- the shape a lead-in whose own subject was cut
    away leaves behind ("Showed that ...", "Found that ...", "Reports that ...",
    "Noted that ..."), not merely any sentence that happens to open with a
    reporting verb at all."""
    stripped = (sentence or "").lstrip()
    match = re.match(r"([A-Za-zÀ-ɏ]+)\s+([A-Za-zÀ-ɏ]+)", stripped)
    if not match:
        return False
    first_word = _fold(match.group(1))
    second_word = _fold(match.group(2))
    return first_word in _PAST_OR_THIRD_PERSON_REPORTING_VERBS and second_word == "that"


# --- a narrow, deterministic check for a body paragraph that reads as a heading
# rather than a sentence -- the shape a markdown sub-heading takes once its own
# markup is gone, whatever removed the markup. Shared with `demo.check_delivered`'s
# own independent copy (`is_heading_shaped_paragraph`), the same way `opens_with_
# reporting_verb` above is.
# ---------------------------------------------------------------------------

#: The "be"/"have"/"do" auxiliary and copula forms `STOP` already carries, plus every
#: inflection of `REPORTING_VERBS` -- together, the tokens whose presence this check
#: treats as evidence that a candidate heading fragment is actually a finite clause,
#: not a bare noun phrase. Not a general finite-verb detector (this module has none),
#: only a narrow lexicon lookup against the verb forms already committed elsewhere.
_HEADING_SHAPE_FINITE_VERB_FORMS = (
    frozenset("is are was were be been being has have had do does did".split())
    | REPORTING_VERBS
)

#: A whole word this module treats as a heading's own connector when deciding whether
#: a phrase is title case ("Comparative Scope and Accuracy" is still title case even
#: though "and" is lower-case) -- reuses `STOP`, the same stopword list the residue
#: classifier already answers to, rather than a second, separately maintained list.
_TITLE_CASE_CONNECTOR_WORDS = STOP


def _is_title_case_phrase(text: str) -> bool:
    """True when every word of *text* that is not one of `_TITLE_CASE_CONNECTOR_WORDS`
    starts with a capital letter -- the shape a heading takes even when it carries no
    verb this module's own lexicons would otherwise have to name ("Comparative Scope
    and Accuracy"). A phrase with no such word at all (all-connector, or no alphabetic
    word) is never title case -- there is nothing capitalised to judge it by."""
    words = _TOKEN_RE.findall(text)
    content_words = [w for w in words if w.casefold() not in _TITLE_CASE_CONNECTOR_WORDS]
    if not content_words:
        return False
    return all(w[:1].isupper() for w in content_words)


def is_heading_shaped_paragraph(text: str) -> bool:
    """True when *text* -- a whole delivered body paragraph -- reads as a heading
    rather than a sentence: no sentence-final punctuation, at most ten words, no
    citation, and no finite verb either by `_HEADING_SHAPE_FINITE_VERB_FORMS` or, when
    a word from that lexicon still appears, by `_is_title_case_phrase` overriding it
    (a title-cased phrase is heading-shaped regardless of a stray lexicon match, since
    capitalisation is the stronger signal there).

    The shape this check exists for: a markdown sub-heading -- "## Comparative Scope
    and Accuracy" -- whose own body paragraph was removed by an earlier rule, leaving
    the bare heading text as what would otherwise be delivered as the section's own
    body prose. `app.services.fulltext`'s own leading-heading handling keeps a
    genuine markdown heading marked as a heading node instead of ever demoting it to
    plain text, so this check is the second, independent guard for every other path
    that could still leave a heading-shaped fragment sitting in a paragraph node: one
    the model wrote with no markdown marker at all, or one some future change
    reintroduces.

    Deliberately narrow, the same way the frame classifier above is: an ordinary
    short sentence such as "Uptake was uneven." is excluded by its own trailing
    period alone, and a sentence carrying a citation is excluded regardless of its
    own length or verb."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped[-1] in ".!?":
        return False
    if _CITATION_RE.search(stripped) or _NARRATIVE_CITATION_RE.search(stripped):
        return False
    words = _TOKEN_RE.findall(stripped)
    if not words or len(words) > 10:
        return False
    folded_tokens = token_list(stripped)
    has_lexicon_verb = any(tok in _HEADING_SHAPE_FINITE_VERB_FORMS for tok in folded_tokens)
    if has_lexicon_verb and not _is_title_case_phrase(stripped):
        return False
    return True
