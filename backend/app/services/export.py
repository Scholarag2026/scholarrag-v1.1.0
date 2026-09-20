"""Export service — converts Tiptap JSON to DOCX, PDF, and LaTeX formats."""

from __future__ import annotations

import html as html_module
import io
import re
from pathlib import Path

import jinja2
import weasyprint
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


# --- Special character escaping ---


_LATEX_SPECIAL = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_LATEX_ESCAPE_RE = re.compile("|".join(re.escape(k) for k in _LATEX_SPECIAL))


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    # Handle backslash first (before other escapes add backslashes)
    text = text.replace("\\", r"\textbackslash{}")
    return _LATEX_ESCAPE_RE.sub(lambda m: _LATEX_SPECIAL[m.group()], text)


# --- Tiptap JSON → HTML ---


def _render_marks_html(node: dict) -> str:
    """Render a text node with its marks as HTML."""
    text = html_module.escape(node.get("text", ""))
    for mark in node.get("marks", []):
        mark_type = mark.get("type", "")
        if mark_type == "bold":
            text = f"<strong>{text}</strong>"
        elif mark_type == "italic":
            text = f"<em>{text}</em>"
        elif mark_type == "underline":
            text = f"<u>{text}</u>"
        elif mark_type == "strike":
            text = f"<s>{text}</s>"
    return text


def _node_to_html(node: dict) -> str:
    """Convert a single Tiptap node to HTML."""
    node_type = node.get("type", "")
    content = node.get("content", [])

    if node_type == "text":
        return _render_marks_html(node)

    if node_type == "heading":
        level = node.get("attrs", {}).get("level", 1)
        level = min(level, 4)
        inner = "".join(_node_to_html(c) for c in content)
        return f"<h{level}>{inner}</h{level}>\n"

    if node_type == "paragraph":
        inner = "".join(_node_to_html(c) for c in content)
        return f"<p>{inner}</p>\n"

    if node_type == "bulletList":
        items = "".join(_node_to_html(c) for c in content)
        return f"<ul>\n{items}</ul>\n"

    if node_type == "orderedList":
        items = "".join(_node_to_html(c) for c in content)
        return f"<ol>\n{items}</ol>\n"

    if node_type == "listItem":
        inner = "".join(_node_to_html(c) for c in content)
        # Strip wrapping <p> tags inside list items for cleaner output
        inner = inner.replace("<p>", "").replace("</p>", "").strip()
        return f"<li>{inner}</li>\n"

    if node_type == "blockquote":
        inner = "".join(_node_to_html(c) for c in content)
        return f"<blockquote>{inner}</blockquote>\n"

    if node_type == "hardBreak":
        return "<br>\n"

    if node_type == "horizontalRule":
        return "<hr>\n"

    if node_type == "doc":
        return "".join(_node_to_html(c) for c in content)

    # Unknown node type — render children
    return "".join(_node_to_html(c) for c in content)


_PDF_CSS = """\
@page {
    size: A4;
    margin: 2.54cm;
}
body {
    font-family: "Times New Roman", Times, serif;
    font-size: 12pt;
    line-height: 2.0;
    color: #000;
}
h1 {
    font-size: 14pt;
    font-weight: bold;
    text-align: center;
    margin-top: 24pt;
    margin-bottom: 12pt;
}
h2 {
    font-size: 13pt;
    font-weight: bold;
    margin-top: 18pt;
    margin-bottom: 6pt;
}
h3 {
    font-size: 12pt;
    font-weight: bold;
    font-style: italic;
    margin-top: 12pt;
    margin-bottom: 6pt;
}
h4 {
    font-size: 12pt;
    font-weight: bold;
    margin-top: 12pt;
    margin-bottom: 6pt;
}
blockquote {
    margin-left: 40px;
    font-style: italic;
    border-left: 3px solid #ccc;
    padding-left: 12px;
}
.title {
    font-size: 16pt;
    font-weight: bold;
    text-align: center;
    margin-bottom: 24pt;
}
"""


def tiptap_to_html(content: dict | None) -> str:
    """Convert Tiptap JSON to a full HTML document with academic styling."""
    body = ""
    if content and content.get("content"):
        body = "".join(_node_to_html(c) for c in content["content"])

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{_PDF_CSS}</style></head>
<body>
{body}
</body>
</html>"""


# --- Tiptap JSON → LaTeX ---


def _render_marks_latex(node: dict) -> str:
    """Render a text node with its marks as LaTeX."""
    text = _escape_latex(node.get("text", ""))
    for mark in node.get("marks", []):
        mark_type = mark.get("type", "")
        if mark_type == "bold":
            text = f"\\textbf{{{text}}}"
        elif mark_type == "italic":
            text = f"\\textit{{{text}}}"
        elif mark_type == "underline":
            text = f"\\underline{{{text}}}"
    return text


def _node_to_latex(node: dict) -> str:
    """Convert a single Tiptap node to LaTeX."""
    node_type = node.get("type", "")
    content = node.get("content", [])

    if node_type == "text":
        return _render_marks_latex(node)

    if node_type == "heading":
        level = node.get("attrs", {}).get("level", 1)
        inner = "".join(_node_to_latex(c) for c in content)
        cmds = {1: "section", 2: "subsection", 3: "subsubsection", 4: "paragraph"}
        cmd = cmds.get(level, "paragraph")
        return f"\\{cmd}{{{inner}}}\n\n"

    if node_type == "paragraph":
        inner = "".join(_node_to_latex(c) for c in content)
        return f"{inner}\n\n"

    if node_type == "bulletList":
        items = "".join(_node_to_latex(c) for c in content)
        return f"\\begin{{itemize}}\n{items}\\end{{itemize}}\n\n"

    if node_type == "orderedList":
        items = "".join(_node_to_latex(c) for c in content)
        return f"\\begin{{enumerate}}\n{items}\\end{{enumerate}}\n\n"

    if node_type == "listItem":
        inner = "".join(_node_to_latex(c) for c in content).strip()
        return f"\\item {inner}\n"

    if node_type == "blockquote":
        inner = "".join(_node_to_latex(c) for c in content)
        return f"\\begin{{quote}}\n{inner}\\end{{quote}}\n\n"

    if node_type == "hardBreak":
        return "\\\\\n"

    if node_type == "horizontalRule":
        return "\\noindent\\rule{\\textwidth}{0.4pt}\n\n"

    if node_type == "doc":
        return "".join(_node_to_latex(c) for c in content)

    # Unknown node — render children
    return "".join(_node_to_latex(c) for c in content)


def tiptap_to_latex(content: dict | None) -> str:
    """Convert Tiptap JSON to LaTeX body content (no preamble)."""
    if not content or not content.get("content"):
        return ""
    return "".join(_node_to_latex(c) for c in content["content"])


# --- Export functions ---


def export_docx(title: str, content: dict | None) -> io.BytesIO:
    """Export draft as DOCX. Moved from drafts.py inline code."""
    doc = Document()

    # Style
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Times New Roman"
    font.size = Pt(12)
    paragraph_format = style.paragraph_format
    paragraph_format.line_spacing = 2.0

    # Title
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title_para.add_run(title)
    run.bold = True
    run.font.size = Pt(14)
    run.font.name = "Times New Roman"

    if content and content.get("content"):
        _add_docx_nodes(doc, content["content"])

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


def _add_docx_nodes(doc: Document, nodes: list[dict]) -> None:
    """Recursively add Tiptap nodes to a DOCX document."""
    for node in nodes:
        node_type = node.get("type", "")
        child_content = node.get("content", [])

        if node_type == "heading":
            level = node.get("attrs", {}).get("level", 1)
            level = min(level, 4)
            text = _extract_text(child_content)
            doc.add_heading(text, level=level)

        elif node_type == "paragraph":
            para = doc.add_paragraph()
            _add_inline_content(para, child_content)

        elif node_type in ("bulletList", "orderedList"):
            for li in child_content:
                if li.get("type") == "listItem":
                    for li_child in li.get("content", []):
                        if li_child.get("type") == "paragraph":
                            style = (
                                "List Bullet"
                                if node_type == "bulletList"
                                else "List Number"
                            )
                            para = doc.add_paragraph(style=style)
                            _add_inline_content(
                                para, li_child.get("content", [])
                            )

        elif node_type == "blockquote":
            for bq_child in child_content:
                if bq_child.get("type") == "paragraph":
                    para = doc.add_paragraph()
                    para.paragraph_format.left_indent = Pt(36)
                    para.style.font.italic = True
                    _add_inline_content(para, bq_child.get("content", []))


def _extract_text(nodes: list[dict]) -> str:
    """Extract plain text from Tiptap inline nodes."""
    parts = []
    for node in nodes:
        if node.get("type") == "text":
            parts.append(node.get("text", ""))
        elif node.get("content"):
            parts.append(_extract_text(node["content"]))
    return "".join(parts)


def _add_inline_content(para, nodes: list[dict]) -> None:
    """Add inline nodes (text with marks) to a DOCX paragraph."""
    for node in nodes:
        if node.get("type") == "text":
            run = para.add_run(node.get("text", ""))
            for mark in node.get("marks", []):
                mark_type = mark.get("type", "")
                if mark_type == "bold":
                    run.bold = True
                elif mark_type == "italic":
                    run.italic = True
                elif mark_type == "underline":
                    run.underline = True
        elif node.get("type") == "hardBreak":
            para.add_run().add_break()


def export_pdf(title: str, content: dict | None) -> io.BytesIO:
    """Export draft as PDF via weasyprint."""
    # Build HTML with title
    body_html = ""
    if content and content.get("content"):
        body_html = "".join(_node_to_html(c) for c in content["content"])

    full_html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{_PDF_CSS}</style></head>
<body>
<div class="title">{html_module.escape(title)}</div>
{body_html}
</body>
</html>"""

    pdf_bytes = weasyprint.HTML(string=full_html).write_pdf()
    buf = io.BytesIO(pdf_bytes)
    buf.seek(0)
    return buf


def export_latex(
    title: str,
    content: dict | None,
) -> io.BytesIO:
    """Export draft as LaTeX via Jinja2 template."""
    body = tiptap_to_latex(content)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=False,  # LaTeX, not HTML
    )
    template = env.get_template("paper.tex.j2")

    tex_content = template.render(
        title=_escape_latex(title),
        body=body,
    )

    buf = io.BytesIO(tex_content.encode("utf-8"))
    buf.seek(0)
    return buf
