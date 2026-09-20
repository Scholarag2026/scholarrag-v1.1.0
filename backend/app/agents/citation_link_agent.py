"""Citation Link Agent.

A second, small structured call run after a section's single markdown completion has
already succeeded (`app.services.writing.generate_section`). It returns a
citation-to-reference map (`CitationLinkMap`), one entry per citation (several entries
may share a `sentence` when that sentence carries more than one citation), beside the
already-generated prose: which citation cites which paper, in the writer's own
author-year keys, the citation exactly as it was rendered, and a `proposition` -- the specific
claim this one citation supports, narrowed to a verbatim span of the sentence rather
than the whole sentence. This is what makes claim verification independent of how a
citation is rendered and, via `proposition`, narrower than "the whole sentence
supports every citation in it" -- `app.services.fulltext.extract_claims_from_document`
consumes this map directly and falls back to the pre-existing regexes for any citation
span the map does not cover, whether that is a whole unlinked sentence or one citation
inside an otherwise-linked one.

Why a second call, not a change to the generation schema: the section's own generation
already runs close to its output-token cap, so wrapping the markdown in JSON risks
truncating paid prose. This call instead runs over text already stored, in the same try/except
pattern the citation audit already uses (`app.services.writing.generate_section`), so its
own failure -- a provider outage, a malformed structured response -- costs only the
mapping: the extractor then behaves exactly as it did before this feature existed.

Validation (V1-V5, `validate_citation_links` below) runs here, not inside the writing
prompt, because none of it needs the model at all: whether a citation link agrees with
the text it claims to describe is a pure function of two strings and the citation audit's
own recognition of what a citation looks like (`app.services.citation_audit`).
"""

from __future__ import annotations

from typing import NamedTuple

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.agents.model_config import AGENT_RETRIES, FAST_MODEL_SETTINGS
from app.config import settings
from app.schemas.provenance import LLMCallProvenance, provenance_from_run
from app.services.citation_audit import citation_spans

CITATION_LINK_PROMPT = """\
You are given an academic paper section that has already been written, split into
numbered paragraphs, and the list of author-year keys ("surname_year", lower case) of
every paper available to cite in this project. You may also be given an EVIDENCE
CATALOG: a list of short ids, each with the exact quote it stands for and the
author-year key of the paper it was extracted from.

Find every in-text citation in the text -- parenthetical ("(Author, Year)"), narrative
("Author (Year)"), or any variant with multiple authors, "et al.", or an "and"/"&"
co-author list. Report one entry PER CITATION, not per sentence: if a single sentence
carries two or more separately rendered citations (e.g. "Smith (2020) and Jones (2021)
both found this." or "This was shown before (Smith, 2020) and confirmed later (Jones,
2021)."), report one entry for each of them, all sharing that same sentence. Only group
several papers into the same entry's keys when they are cited together inside one
citation (e.g. "(Smith, 2020; Jones, 2021)" is a single citation_text naming two keys).

For each citation, report:
- paragraph_index: the number of the paragraph the sentence is in, exactly as given
- sentence: the sentence the citation appears in, copied verbatim, exactly as it
  appears in the text, including its own final punctuation
- citation_text: this citation itself, copied verbatim, exactly as it appears inside
  the sentence -- if the sentence has two citations, each entry's citation_text is only
  its own citation, not the other one
- keys: the author-year key(s) that this one citation cites, chosen from the key list
  given (more than one only when several papers are cited together inside this same
  citation_text)
- proposition: the specific claim this one citation supports, copied verbatim as a
  single contiguous span of the sentence -- not the whole sentence unless the whole
  sentence itself is one claim. When the sentence states more than one claim (e.g. "X
  improves Y (Smith, 2020), while Z remains contested (Jones, 2021)."), proposition is
  only the clause this citation's own claim belongs to, not the other clause and not the
  citation_text itself. The propositions you report for one sentence, taken together
  with the citations rendered in it, must account for every clause of that sentence
  that asserts something: when a clause asserts something and no citation you have
  already reported covers it, report that clause too, as its own proposition, under the
  key of whichever citation in the sentence is nearest to it -- let verification decide
  whether that citation actually supports it. Never invent a claim about the field as a
  whole or about other papers and attach it to one citation to make it "fit": a sentence
  that ranks or compares sources ("the strongest evidence", "treated most directly",
  "the clearest demonstration") is not itself a proposition any citation supports.
- evidence_ids: when an EVIDENCE CATALOG is given, the id(s) of every catalog item,
  belonging to one of this citation's own keys, whose quote this citation's proposition
  restates. Empty when the catalog has nothing relevant, when no catalog is given, or
  when this citation's key is not the source of any catalog item.

Rules:
- Do not invent a sentence, a citation, or a paragraph number that is not actually in the
  text given. Do not paraphrase or summarise a sentence or a proposition; copy them
  exactly as they appear in the text.
- A sentence with no citation at all is not reported as a citation.
- Report every citation you find, even one whose paper is not in the key list given -- in
  that case, give your best-guess key built from the citation's own surname and year
  ("surname_year", lower case).

Separately, also find every sentence in the text that carries NO citation at all and
report it once as an uncited sentence, with:
- paragraph_index and sentence, exactly as above
- tag: "framing" for a transitional, purpose-stating, or meta-discourse sentence (for
  example "This section reviews the literature on X." or "In summary, the studies above
  show mixed results."); "finding" for a sentence that states a substantive claim,
  result, or fact about the topic with no citation attached.
Do not report a sentence as uncited if it carries a citation reported above.
"""

class CitationLink(BaseModel):
    """One sentence-to-reference entry the linker returns."""

    paragraph_index: int = Field(
        description="0-based paragraph number: an index into the section's paragraphs, "
        "split on blank lines exactly as split_paragraphs(content) does."
    )
    sentence: str = Field(
        description="The exact sentence, verbatim, that carries this citation."
    )
    citation_text: str = Field(
        description="The citation exactly as rendered in the sentence, e.g. "
        "'(Smith, 2020)' or 'Smith and Jones (2020)'."
    )
    keys: list[str] = Field(
        default_factory=list,
        description="Author-year keys ('surname_year', lower case) of every paper this "
        "one citation cites.",
    )
    proposition: str = Field(
        default="",
        description="A verbatim, contiguous span of `sentence` naming the specific "
        "claim this citation supports -- the whole sentence only when the whole "
        "sentence itself is the claim.",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Ids, from the evidence catalog given (if any), whose quote this "
        "citation's proposition restates.",
    )


class UncitedSentence(BaseModel):
    """One sentence of the section that carries no citation at all."""

    paragraph_index: int = Field(
        description="0-based paragraph number, exactly as `CitationLink.paragraph_index`."
    )
    sentence: str = Field(description="The exact sentence, verbatim, carrying no citation.")
    tag: str = Field(
        description="'framing' for a transitional or meta-discourse sentence, "
        "'finding' for a substantive, uncited claim."
    )


class CitationLinkMap(BaseModel):
    """What the linker returns for one section (the agent's ``output_type``)."""

    links: list[CitationLink] = Field(default_factory=list)
    uncited_sentences: list[UncitedSentence] = Field(default_factory=list)


_agent: Agent[None, CitationLinkMap] | None = None


def get_citation_link_agent() -> Agent[None, CitationLinkMap]:
    """Lazily create the citation link agent (avoids requiring an API key at import time)."""
    global _agent
    if _agent is None:
        _agent = Agent(
            f"deepseek:{settings.deepseek_model}",
            deps_type=None,
            output_type=CitationLinkMap,
            instructions=CITATION_LINK_PROMPT,
            model_settings=FAST_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _agent


def split_paragraphs(content: str) -> list[str]:
    """The one paragraph split every call site for this feature shares: blank-line blocks, blank
    ones dropped, kept otherwise unstripped.

    ``app.services.fulltext.extract_claims_from_document`` (consumption), this module's
    own prompt (production) and the frontend/demo attachment sites
    (`frontend/src/components/drafts-tab.tsx`, `demo/run_demo.py`) all index paragraphs
    this exact way, so a `CitationLink.paragraph_index` means the same paragraph on every
    side without either side re-deriving the formula independently.
    """
    return [block for block in content.split("\n\n") if block.strip()]


def _build_evidence_catalog_block(evidence_catalog: list[dict] | None) -> str:
    """"EVIDENCE CATALOG" lines the linker prompt shows above the section text,
    one per catalog item (``{"id", "quote", "key"}``, as
    ``app.services.writing._build_evidence_context`` returns them)."""
    if not evidence_catalog:
        return "EVIDENCE CATALOG:\n(none given)"
    lines = ["EVIDENCE CATALOG:"]
    for item in evidence_catalog:
        lines.append(f"[{item['id']}] ({item['key']}): \"{item['quote']}\"")
    return "\n".join(lines)


def build_link_prompt(
    content: str, paper_keys: list[str], evidence_catalog: list[dict] | None = None
) -> str:
    """User-turn prompt: the section's paragraphs, numbered, plus the available keys and
    the evidence catalog, when one is given."""
    paragraphs = split_paragraphs(content)
    numbered = "\n\n".join(f"[Paragraph {i}]\n{p}" for i, p in enumerate(paragraphs))
    keys_line = ", ".join(sorted(paper_keys)) if paper_keys else "(no papers in the library)"
    evidence_block = _build_evidence_catalog_block(evidence_catalog)
    return (
        f"AUTHOR-YEAR KEYS AVAILABLE:\n{keys_line}\n\n"
        f"{evidence_block}\n\n"
        f"SECTION TEXT (0-indexed paragraphs):\n\n{numbered}\n"
    )


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").split())


def _key_surname_year(key: str) -> tuple[str, str] | None:
    """('surname', 'year') from a 'surname_year' key, or None when it is not shaped like
    one -- a key this malformed cannot be checked against anything, so V5 drops it."""
    if "_" not in key:
        return None
    surname, _, year = key.rpartition("_")
    if not surname or not year.isdigit():
        return None
    return surname.lower(), year


def _citation_text_surname_years(citation_text: str) -> list[tuple[str, str]]:
    """(surname, year) pairs `citation_audit` itself recognises inside one citation_text
    snippet, independent of any project library -- what V5 checks a link's keys against."""
    return [
        (surname.split()[-1].lower(), year)
        for _pos, surname, year, _label in citation_spans(citation_text)
        if year is not None
    ]


def _snap_sentence_to_fragment(sentence: str, citation_text: str, content: str) -> str:
    """The sentence fragment `content`'s own splitter (``app.services.fulltext.
    _sentence_fragments``) produces -- the exact discipline `_finalize_paragraph_text`
    later looks a link up by -- for the fragment that contains *citation_text* and
    whose own text agrees with *sentence*, or *sentence* itself when no fragment
    agrees.

    A citation-link call reports a sentence in its own words, copied from the text it
    was shown; finalize's own splitter (`_sentence_fragments`) can cut that same text
    on a different boundary -- an abbreviation outside its allow-list ("e.g.", "i.e.",
    "approx."), or two real sentences the model reported together as one ``sentence``.
    When that happens, the validated link's own ``sentence`` never equals any fragment
    `_finalize_paragraph_text` builds its per-sentence link map from
    (``links_by_sentence``), so a validated, verified citation becomes invisible to it:
    the sentence is processed as though it carried no citation at all, and the whole
    paragraph can be dropped as dangling. Snapping every
    validated link's ``sentence`` to finalize's own fragment here, at validation time,
    is the same discipline the standalone heal path's `_claims_for_paragraph_text`
    already applies (rebuilding a claim's sentence fresh rather than trusting a stored,
    possibly stale one) -- applied on the write path too, where the defect actually
    originates.

    Matched with the same containment rule `unclassified_body_sentences` and
    `_paragraph_index_for_sentence` already use (a fragment/sentence pair counts as a
    match when either contains the other, normalised), then confirmed by
    *citation_text* actually occurring in the candidate fragment, so a same-length but
    unrelated fragment sharing no citation is never picked. Lazy import:
    ``app.services.fulltext`` already imports this module lazily (`split_paragraphs`),
    so a module-level import here would be circular.
    """
    from app.services.fulltext import _finalize_sentence_norm, _sentence_fragments

    citation_norm = _finalize_sentence_norm(citation_text)
    if not citation_norm:
        return sentence
    model_norm = _finalize_sentence_norm(sentence)
    for block in split_paragraphs(content):
        for start, end in _sentence_fragments(block):
            fragment = block[start:end].strip()
            frag_norm = _finalize_sentence_norm(fragment)
            if not frag_norm or citation_norm not in frag_norm:
                continue
            if frag_norm == model_norm or frag_norm in model_norm or model_norm in frag_norm:
                return fragment
    return sentence


def validate_citation_link(
    link: CitationLink,
    content_norm: str,
    evidence_by_id: dict[str, str] | None = None,
    content: str | None = None,
) -> dict | None:
    """Apply V1-V6 to one model-returned
    link.

    Returns ``None`` when V1, V2 or V3 rejects the link outright (a sentence or citation
    the model invented, or a sentence the citation audit itself finds no real citation
    in); otherwise the validated ``{paragraph_index, sentence, keys, citation_text,
    proposition, evidence_ids}`` dict, with V5 having dropped any key that disagrees
    with what the citation audit itself reads out of ``citation_text`` (V4: a key merely
    absent from the project library, as opposed to disagreeing with the citation's own
    text, is kept as-is -- validated here without ever consulting the library).

    When *content* (the raw, un-normalised generated text) is given, ``sentence`` is
    snapped, after V1-V3 pass, to the fragment `_snap_sentence_to_fragment` finds --
    finalize's own splitter's fragment for this same citation, rather than the model's
    own copy of it (see that function's docstring for why). ``content`` defaults to
    ``None`` for a caller that has no raw text to snap against (every pure V1-V6 test
    of this function, none of which exercises the fragment-boundary defect), in which
    case ``sentence`` is the model's own string, unchanged, exactly as before this fix.

    ``proposition`` is
    kept exactly as the model wrote it only when it is a real, contiguous span of the
    (possibly snapped) ``sentence`` (checked whitespace-insensitively, same as V1 checks
    the original ``sentence`` against the generated text); a proposition that is empty,
    whitespace-only, or not actually a substring of it -- a hallucination or a
    paraphrase -- is exactly as unusable as no proposition at all, so ``sentence``
    itself is used instead. This guarantees every validated link carries a non-empty
    proposition, so a later reader
    (`app.services.fulltext.extract_claims_from_document`) never has to re-derive the
    fallback on its own for a link this function has already validated.

    V6: ``evidence_ids`` is kept only for ids present in
    ``evidence_by_id`` (the ``{id: key}`` map built from the evidence catalog shown to
    the model) whose paper key is one of this link's own, already-validated ``keys`` --
    an id from another paper's evidence, or one the model invented outright, is dropped
    silently, exactly like V5 drops a disagreeing key rather than rejecting the whole
    link. ``evidence_by_id`` defaults to ``None`` (no catalog was shown), in which case
    every link's ``evidence_ids`` is ``[]``.
    """
    sentence_norm = _normalize_ws(link.sentence)
    if not sentence_norm or sentence_norm not in content_norm:
        return None  # V1: the sentence must occur in the generated text.
    citation_text_norm = _normalize_ws(link.citation_text)
    if not citation_text_norm or citation_text_norm not in sentence_norm:
        return None  # V2: the citation_text must occur inside that sentence.
    if not citation_spans(link.sentence):
        # V3: citation_audit.citation_spans is authoritative about how many citations
        # exist; a sentence it finds no real citation in never had one to link.
        return None

    sentence = (
        _snap_sentence_to_fragment(link.sentence, link.citation_text, content)
        if content
        else link.sentence
    )
    sentence_norm = _normalize_ws(sentence)

    text_pairs = _citation_text_surname_years(link.citation_text)
    kept_keys: list[str] = []
    for key in link.keys:
        parsed = _key_surname_year(key)
        if parsed is None:
            continue
        if any(parsed == pair for pair in text_pairs):
            kept_keys.append(f"{parsed[0]}_{parsed[1]}")  # V4: kept even if not in the library.
        # else: V5, surname/year disagree with the citation's own text -- key dropped.

    proposition_stripped = (link.proposition or "").strip()
    proposition = (
        proposition_stripped
        if proposition_stripped and _normalize_ws(proposition_stripped) in sentence_norm
        else sentence
    )

    evidence_by_id = evidence_by_id or {}
    evidence_ids = [
        evidence_id
        for evidence_id in link.evidence_ids
        if evidence_by_id.get(evidence_id) in kept_keys
    ]

    return {
        "paragraph_index": link.paragraph_index,
        "sentence": sentence,
        "keys": kept_keys,
        "citation_text": link.citation_text,
        "proposition": proposition,
        "evidence_ids": evidence_ids,
    }


def _v3_rejected_a_real_sentence(link: CitationLink, content_norm: str) -> bool:
    """True when *link* is a link V3 alone rejects: V1 and V2 both pass (a real
    sentence of the text, really carrying this citation_text), but ``citation_spans``
    itself finds no real citation in it. The one shape `validate_citation_links` can
    record as an uncited finding instead of discarding: the model named a genuine
    sentence and the audit agrees no citation is really in it, which is the definition
    of an uncited finding, not a fabrication (V1) or a misattributed citation (V2)."""
    sentence_norm = _normalize_ws(link.sentence)
    if not sentence_norm or sentence_norm not in content_norm:
        return False
    citation_text_norm = _normalize_ws(link.citation_text)
    if not citation_text_norm or citation_text_norm not in sentence_norm:
        return False
    return not citation_spans(link.sentence)


def validate_citation_links(
    link_map: CitationLinkMap,
    content: str,
    evidence_by_id: dict[str, str] | None = None,
    uncited_from_rejected: list[dict] | None = None,
) -> list[dict]:
    """Validate every link in *link_map* against *content* (V1-V6); invented links
    dropped.

    ``uncited_from_rejected``, when given, is appended to (mutated in place, never
    replaced) with one ``{"paragraph_index", "sentence", "tag": "finding"}`` entry for
    every link V3 alone rejects (see `_v3_rejected_a_real_sentence`) -- the one place
    this module knows the model named a genuine sentence and the citation audit itself
    found no citation in it, information a caller would otherwise discard and later pay
    a coverage-gap retry to rediscover (`app.services.fulltext.
    resolve_unclassified_sentences`). Defaults to ``None``: a caller that does not pass
    a list gets today's behaviour, a V3 rejection dropped silently.
    """
    content_norm = _normalize_ws(content)
    validated: list[dict] = []
    for link in link_map.links:
        result = validate_citation_link(link, content_norm, evidence_by_id, content)
        if result is not None:
            validated.append(result)
        elif uncited_from_rejected is not None and _v3_rejected_a_real_sentence(
            link, content_norm
        ):
            uncited_from_rejected.append({
                "paragraph_index": link.paragraph_index,
                "sentence": link.sentence,
                "tag": "finding",
            })
    _flag_sentence_coverage(validated)
    return validated


def _flag_sentence_coverage(validated_links: list[dict]) -> None:
    """A deterministic, no-model-call check over the
    validated links of one sentence, mutating each link dict in place with three new
    keys the model itself never sees or produces (kept off the ``CitationLink`` pydantic
    schema on purpose, so the structured-output contract the model answers against is
    unchanged):

    - ``coverage_incomplete``: ``True`` when this sentence's own propositions and
      rendered citations, taken together, leave at least one non-frame residue (a
      content-bearing span the sentence asserts that no citation supports) -- see
      ``app.services.sentence_coverage``.
    - ``residue_spans``: every such residue's own verbatim text, for the coverage-gap
      closer's retry prompt (``app.services.writing._close_citation_link_coverage_gap``,
      via ``app.services.fulltext.unclassified_body_sentences``) and the revision
      block's own "not covered" list (``app.services.writing.generate_section``) to
      quote.
    - ``meta_evaluation``: ``True`` when this link's own proposition carries a
      comparative meta-evaluation of the literature (edit 8).

    Grouped by sentence (a sentence with two citations is validated against the union of
    both propositions, not each in isolation, since neither citation alone is expected to
    cover the other's own clause) using the sentence's own raw text, whitespace and all
    -- the residue computation folds whitespace internally, so this need not match
    finalize's own normalisation.
    """
    from app.services import sentence_coverage as sc

    by_sentence: dict[str, list[dict]] = {}
    for link in validated_links:
        by_sentence.setdefault(link.get("sentence") or "", []).append(link)
    for sentence, links in by_sentence.items():
        propositions = [link.get("proposition") or sentence for link in links]
        citations = [link.get("citation_text") or "" for link in links]
        incomplete, residue_spans = sc.coverage_incomplete(sentence, propositions, citations)
        for link in links:
            link["coverage_incomplete"] = incomplete
            link["residue_spans"] = list(residue_spans)
            link["meta_evaluation"] = sc.has_meta_evaluation(link.get("proposition") or "")


def validate_uncited_sentence(sentence: UncitedSentence, content_norm: str) -> dict | None:
    """One ``UncitedSentence`` survives only when its own text is really present in the
    generated text (mirrors V1 above) and the citation audit really finds no citation in
    it -- a defence against the model mislabelling a cited sentence as uncited, not just
    trusting its own claim."""
    sentence_norm = _normalize_ws(sentence.sentence)
    if not sentence_norm or sentence_norm not in content_norm:
        return None
    if citation_spans(sentence.sentence):
        return None
    tag = sentence.tag if sentence.tag in ("framing", "finding") else "finding"
    return {
        "paragraph_index": sentence.paragraph_index,
        "sentence": sentence.sentence,
        "tag": tag,
    }


def validate_uncited_sentences(link_map: CitationLinkMap, content: str) -> list[dict]:
    """Validate every ``uncited_sentences`` entry in *link_map* against *content*."""
    content_norm = _normalize_ws(content)
    validated: list[dict] = []
    for sentence in link_map.uncited_sentences:
        result = validate_uncited_sentence(sentence, content_norm)
        if result is not None:
            validated.append(result)
    return validated


class CitationLinkResult(NamedTuple):
    """What one linker call produces: the validated map, plus the call's own provenance
    (the linker is a second, real DeepSeek call, and its tokens are recorded here), plus
    every uncited sentence of the section, tagged."""

    links: list[dict]
    provenance: LLMCallProvenance
    uncited_sentences: list[dict] = []


async def link_citations(
    content: str,
    paper_keys: list[str],
    evidence_catalog: list[dict] | None = None,
) -> CitationLinkResult:
    """Run the linker over already-generated text and return the validated map.

    ``paper_keys`` is the same ``surname_year`` key space
    ``app.services.fulltext._build_paper_lookup`` indexes papers under; passed only to
    tell the model which keys are meaningful, never consulted by validation itself (V4).
    ``evidence_catalog``, when given, is the same
    ``{"id", "quote", "key"}`` list ``app.services.writing._build_evidence_context``
    returned when building the writer's context; shown to the model so it can report
    each citation's ``evidence_ids``, and reduced here to an ``{id: key}`` map for V6.
    """
    agent = get_citation_link_agent()
    prompt = build_link_prompt(content, paper_keys, evidence_catalog)
    result = await agent.run(prompt)
    evidence_by_id = {item["id"]: item["key"] for item in (evidence_catalog or [])}
    uncited_from_rejected: list[dict] = []
    validated = validate_citation_links(
        result.output, content, evidence_by_id, uncited_from_rejected
    )
    uncited = validate_uncited_sentences(result.output, content) + uncited_from_rejected
    provenance = provenance_from_run(
        "citation_link",
        result,
        model_configured=settings.deepseek_model,
        temperature=FAST_MODEL_SETTINGS.get("temperature"),
        prompt=CITATION_LINK_PROMPT,
    )
    return CitationLinkResult(links=validated, provenance=provenance, uncited_sentences=uncited)
