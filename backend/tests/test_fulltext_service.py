"""Tests for full-text PDF extraction, section detection, and chunking."""

from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURE_DIR / "sample_paper.pdf"


def _read_sample_pdf_bytes() -> bytes:
    """Read the sample PDF fixture as raw bytes."""
    assert SAMPLE_PDF.exists(), f"Fixture not found: {SAMPLE_PDF}"
    return SAMPLE_PDF.read_bytes()


# ---------- extract_text_from_pdf tests ----------


def test_extract_text_from_pdf_returns_string():
    """extract_text_from_pdf should return a non-empty string."""
    from app.services.fulltext import extract_text_from_pdf

    pdf_bytes = _read_sample_pdf_bytes()
    text = extract_text_from_pdf(pdf_bytes)
    assert isinstance(text, str)
    assert len(text) > 100  # Sanity check: our sample has ~1000 chars


def test_extract_text_from_pdf_contains_expected_content():
    """Extracted text should contain key phrases from the sample paper."""
    from app.services.fulltext import extract_text_from_pdf

    pdf_bytes = _read_sample_pdf_bytes()
    text = extract_text_from_pdf(pdf_bytes)
    assert "AI Tutoring" in text or "AI tutoring" in text.lower()
    assert "Introduction" in text
    assert "Methods" in text
    assert "Results" in text
    assert "Discussion" in text


def test_extract_text_from_pdf_truncates_oversized(monkeypatch):
    """If text exceeds MAX_TEXT_SIZE, it should be truncated."""
    from app.services import fulltext

    # Temporarily set a very small max size
    monkeypatch.setattr(fulltext, "MAX_TEXT_SIZE", 200)

    pdf_bytes = _read_sample_pdf_bytes()
    text = fulltext.extract_text_from_pdf(pdf_bytes)
    # The text should be truncated to approximately MAX_TEXT_SIZE // 2 = 100 chars
    assert len(text) <= 200


# ---------- detect_sections tests ----------


def test_detect_sections_finds_all_sections():
    """detect_sections should identify sections from the sample paper text."""
    from app.services.fulltext import detect_sections, extract_text_from_pdf

    pdf_bytes = _read_sample_pdf_bytes()
    text = extract_text_from_pdf(pdf_bytes)
    sections = detect_sections(text)

    section_names = [s["section"] for s in sections]
    # Our sample PDF has: preamble, Abstract, Introduction, Methods, Results, Discussion, References
    # The exact names depend on pymupdf text extraction; check for at least a few
    assert len(sections) >= 3, f"Expected at least 3 sections, got {len(sections)}: {section_names}"

    # Check that we have some known sections (case-insensitive, they're lowered)
    lower_names = [n.lower() for n in section_names]
    found_intro = any("introduction" in n for n in lower_names)
    found_methods = any("methods" in n for n in lower_names)
    assert found_intro, f"Expected 'introduction' section; found: {section_names}"
    assert found_methods, f"Expected 'methods' section; found: {section_names}"


def test_detect_sections_no_headers():
    """Text without recognizable headers should return a single preamble section."""
    from app.services.fulltext import detect_sections

    text = "This is just a plain paragraph.\nWith some lines.\nNo headers here."
    sections = detect_sections(text)
    assert len(sections) == 1
    assert sections[0]["section"] == "preamble"
    assert "plain paragraph" in sections[0]["text"]


# ---------- chunk_text tests ----------


def test_chunk_text_uses_sections_when_available():
    """chunk_text should use section-based chunking when sections are detected."""
    from app.services.fulltext import chunk_text, extract_text_from_pdf

    pdf_bytes = _read_sample_pdf_bytes()
    text = extract_text_from_pdf(pdf_bytes)
    chunks = chunk_text(text)

    assert len(chunks) >= 3
    # Each chunk must have both 'section' and 'text' keys
    for chunk in chunks:
        assert "section" in chunk
        assert "text" in chunk
        assert len(chunk["text"]) > 0


def test_chunk_text_falls_back_to_paragraph_splitting():
    """When only one section is detected, chunk_text should fall back to paragraph splitting."""
    from app.services.fulltext import chunk_text

    # Create text that is a single section but has multiple paragraphs
    paragraphs = [
        f"Paragraph number {i}. " + "Lorem ipsum dolor sit amet. " * 20 for i in range(10)
    ]
    text = "\n\n".join(paragraphs)

    chunks = chunk_text(text, max_chunk_size=500)
    assert len(chunks) > 1
    # All chunks should be named chunk_0, chunk_1, etc.
    for i, chunk in enumerate(chunks):
        assert chunk["section"] == f"chunk_{i}"


def test_chunk_text_respects_max_chunks():
    """chunk_text should return at most MAX_CHUNKS chunks."""
    from app.services import fulltext
    from app.services.fulltext import chunk_text

    # Create text with many paragraphs to exceed MAX_CHUNKS
    paragraphs = [f"Paragraph {i}. Some content here." for i in range(100)]
    text = "\n\n".join(paragraphs)

    chunks = chunk_text(text, max_chunk_size=50)
    assert len(chunks) <= fulltext.MAX_CHUNKS


# `acquire_full_texts` is only one of four
# paths that write `metadata["fulltext_chunks"]` -- PDF upload (`api/fulltext.py`), paste
# (`api/fulltext.py`) and bulk-upload confirm (`api/papers.py` via
# `services/paper_upload.py`) each call `chunk_text` and would otherwise store its raw
# output with no reference-list drop of their own. `chunk_text` applies
# `drop_reference_and_backmatter_chunks` to its own output by default, so all four
# callers inherit the guard with no edit needed to those other three files;
# `acquire_full_texts` passes `drop_reference_chunks=False` and keeps its own explicit
# drop so it can still count and report the number dropped.
_CHUNK_TEXT_GOOD_TEXT = (
    "The results indicate that students who received corrective feedback showed higher "
    "post-test scores than those in the control group, suggesting a durable effect of "
    "feedback interventions over time."
)
_CHUNK_TEXT_REFERENCE_LIST_TEXT = (
    "Rosen, J., & Wedin, A. (2015). Klassrumsinteraktion och flerspraakighet. "
    "Apples, 9(1), 1-18. Retrieved from https://apples.jyu.fi\n"
    "Sinclair, J. M., & Coulthard, R. M. (1975). Towards an analysis of discourse. "
    "Oxford, UK: Oxford University Press. pp. 12-34.\n"
    "Snell, J., Shaw, S., & Copland, F. (2015). Linguistic ethnography: Interdisciplinary "
    "explorations. London, UK: Palgrave Macmillan. 10.1057/9781137035035\n"
    "Swain, M. (1985). Communicative competence: Some roles of comprehensible input and "
    "comprehensible output in its development. Rowley, MA: Newbury House. pp. 235-253.\n"
    "Swain, M. (1995). Three functions of output in second language learning. Oxford, "
    "UK: Oxford University Press. pp. 125-144.\n"
)


def test_chunk_text_drops_reference_chunks_by_default():
    """The upload and paste API routes, and the bulk-upload confirm path, all call
    `chunk_text(text)` with no keyword argument, so the default must filter out a trailing
    references section before it is ever stored."""
    from app.services.fulltext import chunk_text

    text = (
        "Introduction\n" + _CHUNK_TEXT_GOOD_TEXT + "\n\n"
        "References\n" + _CHUNK_TEXT_REFERENCE_LIST_TEXT
    )

    chunks = chunk_text(text)

    sections = [c["section"] for c in chunks]
    assert "references" not in sections
    assert "introduction" in sections


def test_chunk_text_keeps_raw_output_when_drop_reference_chunks_is_false():
    """`acquire_full_texts` passes `drop_reference_chunks=False` and performs its own
    explicit drop afterward (so it can count and report the number dropped); the keyword
    argument must actually suppress the guard when set."""
    from app.services.fulltext import chunk_text

    text = (
        "Introduction\n" + _CHUNK_TEXT_GOOD_TEXT + "\n\n"
        "References\n" + _CHUNK_TEXT_REFERENCE_LIST_TEXT
    )

    chunks = chunk_text(text, drop_reference_chunks=False)

    sections = [c["section"] for c in chunks]
    assert "references" in sections
    assert "introduction" in sections


# ---------- _find_evidence_location tests ----------
#
# Regression coverage for a real DeepSeek response recorded against the authors'
# committed demo run (paper a85547fd-..., fulltext_source=unpaywall): the model wraps
# its quote in literal double quotes and joins non-contiguous excerpts with ' ... ',
# while the pymupdf-extracted chunk carries a hard line break mid-sentence and a page
# number / running head ("460\nZhicheng Mao et al.") injected between two of the words.


def test_find_evidence_location_matches_a_realistic_multi_fragment_quote():
    """A ' ... '-joined quote still resolves even when its longest fragment straddles
    a page break, because a shorter fragment from the same quote is searched next."""
    from app.services.fulltext import _find_evidence_location

    evidence_quote = (
        '"Rather than seeking to identify the most effective WCF type via a comparison '
        "experimental design, naturalistic research in this area suggests that WCF "
        "options should be deployed flexibly, depending on students' learning needs and "
        'specific contexts." ... "Focused WCF entailed in feedback innovative practices '
        'was found to bring benefits to both teachers and students."'
    )
    chunk_text = (
        "Rather than seeking to identify the most effective WCF type via a comparison\n"
        "experimental design, naturalistic research in this area suggests that WCF "
        "options should be deployed\n460\nZhicheng Mao et al.\n\nflexibly, depending on "
        "students' learning needs and specific contexts. Focused WCF entailed in "
        "feedback innovative practices was found to bring benefits to both teachers and "
        "students."
    )
    chunks = [{"section": "results", "text": chunk_text}]

    assert _find_evidence_location(evidence_quote, chunks) == "chunk 1 (results)"


def test_find_evidence_location_normalizes_a_single_line_wrapped_quote():
    """A single quoted fragment split by an ordinary mid-sentence line wrap still matches."""
    from app.services.fulltext import _find_evidence_location

    evidence_quote = '"scores rose by 12 percent among the treatment group"'
    chunks = [
        {"section": "abstract", "text": "irrelevant text here"},
        {
            "section": "results",
            "text": "Test\nscores rose by 12 percent\namong the treatment group.",
        },
    ]

    assert _find_evidence_location(evidence_quote, chunks) == "chunk 2 (results)"


def test_find_evidence_location_returns_chunk_n_without_section_suffix():
    """A chunk without a 'section' key returns the plain 'chunk N' form."""
    from app.services.fulltext import _find_evidence_location

    chunks = [{"text": "the effect was statistically significant"}]
    assert _find_evidence_location("the effect was statistically significant", chunks) == "chunk 1"


def test_find_evidence_location_returns_none_when_the_quote_is_missing():
    from app.services.fulltext import _find_evidence_location

    assert _find_evidence_location(None, [{"section": "results", "text": "anything"}]) is None
    assert _find_evidence_location("", [{"section": "results", "text": "anything"}]) is None


def test_find_evidence_location_returns_none_when_no_fragment_is_found():
    from app.services.fulltext import _find_evidence_location

    chunks = [{"section": "results", "text": "completely unrelated content"}]
    quote = '"a claim the paper never actually makes" ... "nor this one either"'
    assert _find_evidence_location(quote, chunks) is None
