"""Tests for export service — Tiptap to HTML, LaTeX, DOCX, PDF converters."""



SAMPLE_TIPTAP = {
    "type": "doc",
    "content": [
        {
            "type": "heading",
            "attrs": {"level": 1},
            "content": [{"type": "text", "text": "Introduction"}],
        },
        {
            "type": "paragraph",
            "content": [
                {"type": "text", "text": "This is "},
                {"type": "text", "marks": [{"type": "bold"}], "text": "bold"},
                {"type": "text", "text": " and "},
                {"type": "text", "marks": [{"type": "italic"}], "text": "italic"},
                {"type": "text", "text": " text."},
            ],
        },
        {
            "type": "bulletList",
            "content": [
                {
                    "type": "listItem",
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": "Item one"}],
                        }
                    ],
                },
                {
                    "type": "listItem",
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": "Item two"}],
                        }
                    ],
                },
            ],
        },
        {
            "type": "blockquote",
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": "A quoted passage."}],
                }
            ],
        },
        {
            "type": "heading",
            "attrs": {"level": 2},
            "content": [{"type": "text", "text": "Methods"}],
        },
        {
            "type": "orderedList",
            "content": [
                {
                    "type": "listItem",
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": "First step"}],
                        }
                    ],
                },
            ],
        },
    ],
}


# --- tiptap_to_html tests ---


def test_tiptap_to_html_headings():
    from app.services.export import tiptap_to_html

    html = tiptap_to_html(SAMPLE_TIPTAP)
    assert "<h1>Introduction</h1>" in html
    assert "<h2>Methods</h2>" in html


def test_tiptap_to_html_paragraph_with_marks():
    from app.services.export import tiptap_to_html

    html = tiptap_to_html(SAMPLE_TIPTAP)
    assert "<strong>bold</strong>" in html
    assert "<em>italic</em>" in html


def test_tiptap_to_html_bullet_list():
    from app.services.export import tiptap_to_html

    html = tiptap_to_html(SAMPLE_TIPTAP)
    assert "<ul>" in html
    assert "<li>" in html
    assert "Item one" in html


def test_tiptap_to_html_ordered_list():
    from app.services.export import tiptap_to_html

    html = tiptap_to_html(SAMPLE_TIPTAP)
    assert "<ol>" in html
    assert "First step" in html


def test_tiptap_to_html_blockquote():
    from app.services.export import tiptap_to_html

    html = tiptap_to_html(SAMPLE_TIPTAP)
    assert "<blockquote>" in html
    assert "A quoted passage." in html


def test_tiptap_to_html_empty():
    from app.services.export import tiptap_to_html

    html = tiptap_to_html(None)
    assert "<body>" in html  # still returns valid HTML document


def test_tiptap_to_html_empty_doc():
    from app.services.export import tiptap_to_html

    html = tiptap_to_html({"type": "doc", "content": []})
    assert "<body>" in html


# --- tiptap_to_latex tests ---


def test_tiptap_to_latex_sections():
    from app.services.export import tiptap_to_latex

    latex = tiptap_to_latex(SAMPLE_TIPTAP)
    assert "\\section{Introduction}" in latex
    assert "\\subsection{Methods}" in latex


def test_tiptap_to_latex_marks():
    from app.services.export import tiptap_to_latex

    latex = tiptap_to_latex(SAMPLE_TIPTAP)
    assert "\\textbf{bold}" in latex
    assert "\\textit{italic}" in latex


def test_tiptap_to_latex_bullet_list():
    from app.services.export import tiptap_to_latex

    latex = tiptap_to_latex(SAMPLE_TIPTAP)
    assert "\\begin{itemize}" in latex
    assert "\\item" in latex
    assert "Item one" in latex


def test_tiptap_to_latex_ordered_list():
    from app.services.export import tiptap_to_latex

    latex = tiptap_to_latex(SAMPLE_TIPTAP)
    assert "\\begin{enumerate}" in latex
    assert "First step" in latex


def test_tiptap_to_latex_blockquote():
    from app.services.export import tiptap_to_latex

    latex = tiptap_to_latex(SAMPLE_TIPTAP)
    assert "\\begin{quote}" in latex
    assert "A quoted passage." in latex


def test_tiptap_to_latex_special_chars():
    from app.services.export import tiptap_to_latex

    content = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": "Cost is $100 & 50% off #1"}],
            }
        ],
    }
    latex = tiptap_to_latex(content)
    assert "\\$100" in latex
    assert "\\&" in latex
    assert "\\%" in latex
    assert "\\#" in latex


def test_tiptap_to_latex_empty():
    from app.services.export import tiptap_to_latex

    latex = tiptap_to_latex(None)
    assert latex == ""


# --- export function tests ---


def test_export_docx_produces_valid_file():
    from app.services.export import export_docx

    result = export_docx("Test Title", SAMPLE_TIPTAP)
    data = result.read()
    assert len(data) > 100
    assert data[:2] == b"PK"  # DOCX is a zip file


def test_export_pdf_produces_valid_file():
    from app.services.export import export_pdf

    result = export_pdf("Test Title", SAMPLE_TIPTAP)
    data = result.read()
    assert len(data) > 100
    assert data[:5] == b"%PDF-"


def test_export_latex_produces_valid_file():
    from app.services.export import export_latex

    result = export_latex("Test Title", SAMPLE_TIPTAP)
    data = result.read().decode("utf-8")
    assert "\\documentclass" in data
    assert "\\begin{document}" in data
    assert "\\section{Introduction}" in data
    assert "\\maketitle" in data
    assert "Test Title" in data


def test_export_pdf_empty_content():
    from app.services.export import export_pdf

    result = export_pdf("Empty Paper", None)
    data = result.read()
    assert data[:5] == b"%PDF-"


def test_export_latex_empty_content():
    from app.services.export import export_latex

    result = export_latex("Empty Paper", None)
    data = result.read().decode("utf-8")
    assert "\\documentclass" in data
    assert "Empty Paper" in data


def test_export_latex_no_longer_accepts_citation_style():
    """citation_style is applied
    once, before export, by `app.services.citation_render.render_document`; export_latex's
    own (unused) parameter is removed."""
    import inspect

    from app.services.export import export_latex

    assert "citation_style" not in inspect.signature(export_latex).parameters
