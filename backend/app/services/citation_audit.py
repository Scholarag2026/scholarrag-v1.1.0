"""Code-level audit of APA in-text citations in generated text.

The writing prompt instructs the model to cite only library papers and to flag
unsupported claims with ``[NEEDS CITATION]``. Those are *instructions*; this module is
the *check*: it extracts every ``(Author, Year)`` / ``Author (Year)`` citation from the
generated text, compares surname and year with the first authors of the project
library, and counts how often the flag was actually emitted.

Recognised forms (APA 7th):

* ``(Smith, 2020)``, ``(Smith & Jones, 2020)``, ``(Smith and Jones, 2020)``,
  ``(Smith, Jones, & Brown, 2020)``, ``(Smith et al., 2020)``, with optional ``a``/``b``
  year suffixes, page locators, ``see`` / ``e.g.,`` prefixes and ``;``-separated lists;
* ``Smith (2020)``, ``Smith and Jones (2020)``, ``Smith et al. (2020)`` and the
  possessive forms ``Smith's (2020)``, ``Smith et al.'s (2020)``;
* ``(Foo, 1999, as cited in Smith, 2020)`` — only the cited source (Smith) is checked.

Matching is on the *last token* of the first author's surname (so ``van Dijk`` and
``Dijk`` agree) after Unicode-diacritic folding, plus the year. A library entry without
a year matches on surname alone. Group authors (``World Health Organization, 2020``) and
capitalised prose words directly followed by a parenthesised year are reported as
unmatched; the audit is a signal for a human reader, not a verdict.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from pydantic import BaseModel, Field

#: Widened from the exact literal ``"[NEEDS CITATION]"`` to the same pattern
#: `app.services.fulltext._NEEDS_CITATION_RE` uses, so the audit counts a marker the
#: writer typed with a colon and an explanatory clause inside the brackets (e.g.
#: "[NEEDS CITATION: evidence on peer feedback relative to teacher feedback ...]") the
#: same as the bare form, and never counts an ordinary numbered or author-year
#: citation in brackets ("[12]", "[Smith, 2020]"), since neither contains "NEEDS"
#: immediately before "CITATION".
NEEDS_CITATION_FLAG = re.compile(r"\[\s*NEEDS\s+CITATION\b[^\]]*\]", re.IGNORECASE)

# Lower-case surname particles kept with the surname ("van Dijk", "de la Cruz"). Public:
# `app.services.citation_render` reuses this exact set to split a reference entry's
# author name into family and given parts, so "the surname" means the same thing in a
# citation match and in a rendered reference list.
NAME_PARTICLES = frozenset(
    "van von de del della der den du da das dos di la le el al ter ten op bin ibn".split()
)
_PARTICLE_RE = r"(?i:" + "|".join(sorted(NAME_PARTICLES, key=len, reverse=True)) + r")"
# One capitalised name token: "Smith", "O'Brien", "García-López". An apostrophe may
# only appear inside the token, so a possessive "'s" is not swallowed.
_TOKEN = r"[A-Z\u00C0-\u00D6\u00D8-\u00DE\u0100-\u017F](?:[\w\-]|['’](?!s\b))*"
# The surname (with optional particles) must start at a word boundary so that a prose
# word ending in a particle ("laTER Smith", "unDER Smith") is not glued onto it.
_SURNAME = rf"(?<![\w'’\-])(?:{_PARTICLE_RE}\s+)*{_TOKEN}"
_POSSESSIVE = r"(?:['’]s)?"
# A year, or an APA placeholder for undated / forthcoming work ("n.d.", "in press").
_YEAR = r"(?P<year>\d{4}[a-z]?|n\.d\.|in\s+press)"
# Co-authors after the first surname: " et al." | ", Jones, & Brown" | " and Jones".
_AUTHOR_TAIL = rf"(?:\s+et\s+al\.?|(?:,\s+{_SURNAME})*,?\s+(?:&|and)\s+{_SURNAME})?"

_PAREN_GROUP = re.compile(r"\(([^()]*)\)")
_PAREN_CITE = re.compile(rf"(?P<surname>{_SURNAME}){_AUTHOR_TAIL},\s*{_YEAR}")
_NARRATIVE_CITE = re.compile(
    rf"(?P<surname>{_SURNAME}){_AUTHOR_TAIL}{_POSSESSIVE}\s*\(\s*{_YEAR}[^()]*\)"
)
_AS_CITED_IN = re.compile(r"as cited in", re.IGNORECASE)
_INITIALS = re.compile(r"(?:[A-Z]\.?){1,3}")
#: One capitalised name-shaped token inside an already-matched `_NARRATIVE_CITE` span,
#: for `_disambiguate_lead_in` below. Compiled from `_SURNAME` itself, not a second,
#: independently written pattern, so "what counts as a name" cannot drift between the
#: trigger and the match it is applied to.
_NAME_TOKEN_RE = re.compile(_SURNAME)


class CitationAudit(BaseModel):
    """Outcome of checking a text's in-text citations against the library."""

    matched: list[str] = Field(default_factory=list)  # "Smith, 2020" found in library
    unmatched: list[str] = Field(default_factory=list)  # cited but not in library
    needs_citation_flags: int = 0  # occurrences of the "[NEEDS CITATION ...]" marker
    total: int = 0  # distinct (surname, year) citations checked


def _normalize(token: str) -> str:
    """Case- and diacritic-insensitive comparison key for one name token."""
    decomposed = unicodedata.normalize("NFKD", token)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.replace("’", "'").casefold()


def _field(paper: Any, name: str) -> Any:
    if isinstance(paper, dict):
        return paper.get(name)
    return getattr(paper, name, None)


def _first_author_name(authors: Any) -> str:
    if not authors:
        return ""
    if isinstance(authors, str):
        return authors
    if isinstance(authors, dict):
        return str(authors.get("name") or authors.get("family") or "")
    if not isinstance(authors, (list, tuple)):
        return ""
    first = authors[0]
    if isinstance(first, dict):
        return str(first.get("name") or first.get("family") or "")
    return str(first or "")


def _surname_key(name: str) -> str | None:
    """Comparison key for an author name in any common format.

    Handles ``"Smith, J."``, ``"J. Smith"``, ``"Smith J"``, ``"Jane Smith"`` and
    ``"Teun van Dijk"`` (returns the key of the last surname token, ``"dijk"``).
    """
    name = name.strip()
    if not name:
        return None
    if "," in name:
        surname = name.split(",", 1)[0].strip()
        tokens = surname.split()
    else:
        tokens = name.split()
        while len(tokens) > 1 and _INITIALS.fullmatch(tokens[-1]):
            tokens.pop()
    if not tokens:
        return None
    return _normalize(tokens[-1])


def _library_year(raw: Any) -> str | None:
    """``2020`` / ``2020.0`` / ``"2020-01-01"`` -> ``"2020"``; anything else -> ``None``."""
    if raw is None:
        return None
    match = re.match(r"\d{4}", str(raw))
    return match.group(0) if match else None


def _library_index(papers: Iterable[Any]) -> tuple[set[tuple[str, str]], set[str]]:
    """Return ``({(surname_key, year)}, {surname_key with unknown year})``."""
    dated: set[tuple[str, str]] = set()
    undated: set[str] = set()
    for paper in papers:
        key = _surname_key(_first_author_name(_field(paper, "authors")))
        if key is None:
            continue
        year = _library_year(_field(paper, "year"))
        if year is None:
            undated.add(key)
        else:
            dated.add((key, year))
    return dated, undated


def _year_key(raw: str) -> str | None:
    """``"2020a"`` -> ``"2020"``; placeholders (``"n.d."``, ``"in press"``) -> ``None``."""
    return raw[:4] if raw[:4].isdigit() else None


def _year_label(raw: str) -> str:
    """Display form: ``"2020a"`` -> ``"2020"``; placeholders keep their (collapsed) text."""
    return _year_key(raw) or " ".join(raw.split())


def _find_citations(text: str) -> list[tuple[int, str, str | None, str]]:
    """All ``(position, surname_as_written, year_key, year_label)`` citations in the text."""
    found: list[tuple[int, str, str | None, str]] = []

    for group in _PAREN_GROUP.finditer(text):
        inner = group.group(1)
        offset = group.start(1)
        # "(Foo, 1999, as cited in Smith, 2020)": Smith is the source in the library.
        parts = _AS_CITED_IN.split(inner)
        if len(parts) > 1:
            offset += len(inner) - len(parts[-1])
            inner = parts[-1]
        for match in _PAREN_CITE.finditer(inner):
            raw_year = match.group("year")
            found.append(
                (
                    offset + match.start(),
                    match.group("surname"),
                    _year_key(raw_year),
                    _year_label(raw_year),
                )
            )

    for match in _NARRATIVE_CITE.finditer(text):
        raw_year = match.group("year")
        found.append(
            (match.start(), match.group("surname"), _year_key(raw_year), _year_label(raw_year))
        )

    return sorted(found, key=lambda item: item[0])


def citation_spans(text: str) -> list[tuple[int, str, str | None, str]]:
    """Every citation span in *text*, public wrapper over `_find_citations`.

    Returns ``(position, surname_as_written, year_key, year_label)`` tuples, sorted by
    position, in the identical shape and with the identical recognition `_find_citations`
    always used internally: no citation shape is added, dropped or reordered by exposing
    this publicly, and `audit_citations` itself keeps calling `_find_citations` directly,
    so its own output is unchanged by this function's existence.

    This is the authoritative count of "how many citations exist" in a piece of text for
    the writer's citation-link map: a link that names a sentence and a
    citation this function does not find in that sentence is invented and is dropped, and
    a span this function finds with no surviving link falls through to the regex-based
    extractor, so the mapping can never lose a citation the audit itself would see.
    """
    return _find_citations(text)


def _disambiguate_lead_in(
    match: re.Match[str], surname: str, year: str | None, dated: set, undated: set
) -> str:
    """The real first-author surname for one `_NARRATIVE_CITE` match, when the token the
    regex itself captured is a capitalised lead-in word rather than a surname (the
    audit's own version of the same ambiguity already fixed for extraction in
    `app.services.fulltext._resolve_narrative_citation_key`).

    A capitalised word immediately followed by a comma inside the match is a lead-in,
    not a surname: a genuine narrative citation's own author is followed by "et al." or
    "and", never directly by a comma -- the comma only appears when `_AUTHOR_TAIL`'s
    co-author list swallows a lead-in word such as "Interestingly," or "However," ahead
    of the real authors ("Interestingly, Liu and Wu (2019)"). Triggered, every
    capitalised name-shaped token in the whole match is tried against the library in
    order and the first one present wins, so a real first author who also happens to be
    followed by a comma (an Oxford-comma three-author list, "Zhang, Li and Wang (2021)")
    is still picked correctly: it is the first token tried, and it is in the library it
    was cited from. Untriggered, or when no candidate is in the library at all (no
    citation in the whole match names a paper this library holds), *surname* is
    returned unchanged, exactly the pre-fix behaviour."""
    end = match.end("surname")
    if match.string[end : end + 1] != ",":
        return surname
    for candidate in _NAME_TOKEN_RE.finditer(match.group(0)):
        token = candidate.group(0)
        key = _normalize(token.split()[-1])
        if key in undated or (year is not None and (key, year) in dated):
            return token
    return surname


def audit_citations(text: str, papers: list) -> CitationAudit:
    """Check every in-text citation in ``text`` against the first authors of ``papers``.

    ``papers`` may hold ORM rows, dicts or any objects exposing ``authors`` (``list[str]``
    or ``list[dict]`` with ``name``) and ``year``. Pure function; never raises on odd
    input shapes.
    """
    text = text or ""
    dated, undated = _library_index(papers or [])
    narrative_by_position = {match.start(): match for match in _NARRATIVE_CITE.finditer(text)}
    seen: set[tuple[str, str | None]] = set()
    matched: list[str] = []
    unmatched: list[str] = []

    for position, surname, year, year_label in _find_citations(text):
        narrative_match = narrative_by_position.get(position)
        if narrative_match is not None and narrative_match.group("surname") == surname:
            surname = _disambiguate_lead_in(narrative_match, surname, year, dated, undated)
        key = (_normalize(surname.split()[-1]), year)
        if key in seen:
            continue
        seen.add(key)
        label = f"{surname}, {year_label}"
        in_library = key[0] in undated or (year is not None and (key[0], year) in dated)
        (matched if in_library else unmatched).append(label)

    return CitationAudit(
        matched=matched,
        unmatched=unmatched,
        needs_citation_flags=len(NEEDS_CITATION_FLAG.findall(text)),
        total=len(seen),
    )
