"""Background task for processing uploaded article files."""
from __future__ import annotations

import asyncio
import logging
import re
from uuid import UUID

from app.agents.paper_metadata_agent import extract_metadata_from_text
from app.models.analysis_job import JobStatus
from app.services import task as task_service
from app.services.fulltext import chunk_text, extract_text_from_pdf

logger = logging.getLogger(__name__)

_PDF_EXTS = {".pdf"}
_DOCX_EXTS = {".docx", ".doc"}
_BIB_EXTS = {".bib"}


def parse_bibtex(content: str) -> list[dict]:
    """Simple BibTeX parser. Extracts title, author, year, journal, doi from entries."""
    entries = []
    pattern = r'@\w+\{[^@]*\}'
    matches = re.findall(pattern, content, re.DOTALL)
    for match in matches:
        entry = {}
        for field in ["title", "author", "year", "journal", "doi", "abstract"]:
            field_pattern = rf'{field}\s*=\s*\{{([^}}]*)\}}'
            field_match = re.search(field_pattern, match, re.IGNORECASE)
            if field_match:
                entry[field] = field_match.group(1).strip()
        if entry.get("title"):
            entries.append(entry)
    return entries


def _extract_text_from_docx(docx_bytes: bytes) -> str:
    """Extract text from DOCX bytes using python-docx."""
    import io
    from docx import Document
    doc = Document(io.BytesIO(docx_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


async def process_uploaded_files(
    project_id: UUID,
    job_id: UUID,
    session_factory,
    files_data: list[dict],
) -> None:
    """Background task: process uploaded files and extract metadata."""
    async with session_factory() as db:
        await task_service.update_job_status(
            db, job_id,
            status=JobStatus.running,
            progress=0.0,
            progress_message="Processing uploaded files...",
        )

    try:
        total = len(files_data)
        extracted_papers = []
        fulltext_data = {}

        for idx, file_info in enumerate(files_data):
            name = file_info["name"]
            content = file_info["content"]
            ext = file_info["extension"].lower()

            async with session_factory() as db:
                await task_service.update_job_status(
                    db, job_id,
                    progress=(idx / total) * 0.9,
                    progress_message=f"Processing {name} ({idx + 1}/{total})...",
                    status=JobStatus.running,
                )

            if ext in _PDF_EXTS:
                # pymupdf extraction and chunking are CPU-bound blocking calls; run them
                # off the event loop (D20 / issue #27).
                text = await asyncio.to_thread(extract_text_from_pdf, content)
                metadata = await extract_metadata_from_text(text)
                chunks = await asyncio.to_thread(chunk_text, text)
                extracted_papers.append({
                    "index": idx,
                    "file_name": name,
                    "title": metadata.title,
                    "authors": metadata.authors,
                    "year": metadata.year,
                    "journal_name": metadata.journal_name,
                    "doi": metadata.doi,
                    "abstract": metadata.abstract,
                    "source_api": "upload",
                    "has_full_text": True,
                })
                fulltext_data[str(idx)] = {
                    "chunks": chunks,
                    "char_count": len(text),
                }

            elif ext in _DOCX_EXTS:
                # python-docx parsing and chunking are CPU-bound blocking calls; run
                # them off the event loop (D20 / issue #27).
                text = await asyncio.to_thread(_extract_text_from_docx, content)
                metadata = await extract_metadata_from_text(text)
                chunks = await asyncio.to_thread(chunk_text, text)
                extracted_papers.append({
                    "index": idx,
                    "file_name": name,
                    "title": metadata.title,
                    "authors": metadata.authors,
                    "year": metadata.year,
                    "journal_name": metadata.journal_name,
                    "doi": metadata.doi,
                    "abstract": metadata.abstract,
                    "source_api": "upload",
                    "has_full_text": True,
                })
                fulltext_data[str(idx)] = {
                    "chunks": chunks,
                    "char_count": len(text),
                }

            elif ext in _BIB_EXTS:
                text = content.decode("utf-8", errors="replace")
                entries = parse_bibtex(text)
                for bib_idx, entry in enumerate(entries):
                    authors_raw = entry.get("author", "")
                    authors = [{"name": a.strip()} for a in authors_raw.split(" and ") if a.strip()]
                    year_str = entry.get("year", "")
                    year = int(year_str) if year_str.isdigit() else None
                    paper_index = len(extracted_papers)
                    extracted_papers.append({
                        "index": paper_index,
                        "file_name": f"{name}[{bib_idx}]",
                        "title": entry.get("title", "Untitled"),
                        "authors": authors,
                        "year": year,
                        "journal_name": entry.get("journal"),
                        "doi": entry.get("doi"),
                        "abstract": entry.get("abstract"),
                        "source_api": "upload",
                        "has_full_text": False,
                    })
            else:
                logger.warning("Unsupported file extension: %s", ext)

        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id,
                status=JobStatus.completed,
                progress=1.0,
                progress_message="Extraction complete",
                result={
                    "papers": extracted_papers,
                    "_fulltext_data": fulltext_data,
                },
            )

    except Exception as e:
        logger.exception("Paper upload processing failed")
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id,
                status=JobStatus.failed,
                error=str(e),
            )
