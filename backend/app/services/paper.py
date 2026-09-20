from collections.abc import Iterator
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.models.paper import Paper, SourceApi
from app.models.project_paper import ProjectPaper
from app.schemas.paper import PaperData
from app.services.wos_import import batch_enrich_wos, is_wos_indexed

# Postgres caps bind parameters at 32767 per statement. Keep each IN(...) chunk well
# under that so get_or_create_papers_bulk never blows the limit on a large library.
_BULK_LOOKUP_CHUNK_SIZE = 5000


def _chunked(values: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


async def get_or_create_paper(db: AsyncSession, paper_data: PaperData) -> Paper:
    """Find a paper by DOI or create it. Returns the Paper record."""
    if paper_data.doi:
        result = await db.execute(
            select(Paper).where(Paper.doi == paper_data.doi)
        )
        existing = result.scalar_one_or_none()
        if existing:
            # Backfill WoS data if missing
            if existing.wos_collection is None and (paper_data.journal_issn or existing.journal_issn):
                issn = paper_data.journal_issn or existing.journal_issn
                indexed, collection, categories = await is_wos_indexed(db, issn)
                existing.is_wos_indexed = indexed
                existing.wos_collection = collection
                existing.wos_categories = categories
                await db.flush()
            return existing

    # Fallback: check by external_id (e.g., S2 paperId)
    if paper_data.external_id:
        result = await db.execute(
            select(Paper).where(Paper.external_id == paper_data.external_id)
        )
        existing = result.scalar_one_or_none()
        if existing:
            # Backfill WoS data if missing
            if existing.wos_collection is None and (paper_data.journal_issn or existing.journal_issn):
                issn = paper_data.journal_issn or existing.journal_issn
                indexed, collection, categories = await is_wos_indexed(db, issn)
                existing.is_wos_indexed = indexed
                existing.wos_collection = collection
                existing.wos_categories = categories
                await db.flush()
            return existing

    # WoS lookup for new paper
    wos_indexed = None
    wos_collection = None
    wos_categories = None
    if paper_data.journal_issn:
        wos_indexed, wos_collection, wos_categories = await is_wos_indexed(
            db, paper_data.journal_issn
        )
    # Fall back to pre-enriched WoS data from search results
    if not wos_collection and paper_data.wos_collection:
        wos_collection = paper_data.wos_collection
        wos_categories = paper_data.wos_categories
        wos_indexed = True

    paper = Paper(
        doi=paper_data.doi,
        title=paper_data.title,
        authors=paper_data.authors,
        year=paper_data.year,
        journal_name=paper_data.journal_name,
        journal_issn=paper_data.journal_issn,
        is_wos_indexed=wos_indexed,
        wos_collection=wos_collection,
        wos_categories=wos_categories,
        citation_count=paper_data.citation_count,
        abstract=paper_data.abstract,
        source_api=SourceApi(paper_data.source_api) if paper_data.source_api else SourceApi.manual,
        external_id=paper_data.external_id,
        full_text_url=paper_data.full_text_url,
    )
    db.add(paper)
    await db.flush()
    await db.refresh(paper)
    return paper


def _new_paper_from_data(paper_data: PaperData) -> Paper:
    """Build an unsaved Paper from already WoS-enriched PaperData."""
    if paper_data.wos_collection:
        wos_indexed: bool | None = True
    elif paper_data.journal_issn:
        wos_indexed = False
    else:
        wos_indexed = None

    return Paper(
        doi=paper_data.doi,
        title=paper_data.title,
        authors=paper_data.authors,
        year=paper_data.year,
        journal_name=paper_data.journal_name,
        journal_issn=paper_data.journal_issn,
        is_wos_indexed=wos_indexed,
        wos_collection=paper_data.wos_collection,
        wos_categories=paper_data.wos_categories,
        citation_count=paper_data.citation_count,
        abstract=paper_data.abstract,
        source_api=(
            SourceApi(paper_data.source_api) if paper_data.source_api else SourceApi.manual
        ),
        external_id=paper_data.external_id,
        full_text_url=paper_data.full_text_url,
    )


async def get_or_create_papers_bulk(
    db: AsyncSession, papers: list[PaperData]
) -> list[Paper]:
    """Resolve a whole list of PaperData to Paper rows with a bounded query count.

    Existing rows are matched by exact DOI first, then by external_id; duplicates inside
    the batch collapse onto a single row. New rows are added and flushed once (so they
    have ids) — **the caller owns the transaction and must commit**.

    Cost is 1 WoS batch query + a handful of chunked lookups + 1 flush regardless of
    list length. The per-item ``get_or_create_paper`` path cost up to three round
    trips per reference, which was the next graph bottleneck once the network fixes
    landed (issue PAPER-LOOKUP-N-PLUS-1). The lookup is chunked (see
    ``_BULK_LOOKUP_CHUNK_SIZE``) rather than a single ``IN(...)`` because Postgres
    caps bind parameters at 32767 per statement, which a full library's worth of
    referenced works can exceed.

    Returns:
        One ``Paper`` per input element, in input order.
    """
    if not papers:
        return []

    # Chunked IN(...) queries for the whole list instead of one lookup per paper.
    await batch_enrich_wos(db, papers)

    dois = list({p.doi for p in papers if p.doi})
    external_ids = list({p.external_id for p in papers if p.external_id})

    by_doi: dict[str, Paper] = {}
    by_external_id: dict[str, Paper] = {}
    for chunk in _chunked(dois, _BULK_LOOKUP_CHUNK_SIZE):
        result = await db.execute(select(Paper).where(Paper.doi.in_(chunk)))
        for existing in result.scalars().all():
            if existing.doi:
                by_doi.setdefault(existing.doi, existing)
            if existing.external_id:
                by_external_id.setdefault(existing.external_id, existing)
    for chunk in _chunked(external_ids, _BULK_LOOKUP_CHUNK_SIZE):
        result = await db.execute(select(Paper).where(Paper.external_id.in_(chunk)))
        for existing in result.scalars().all():
            if existing.doi:
                by_doi.setdefault(existing.doi, existing)
            if existing.external_id:
                by_external_id.setdefault(existing.external_id, existing)

    resolved: list[Paper] = []
    created = 0
    for data in papers:
        found = by_doi.get(data.doi) if data.doi else None
        if found is None and data.external_id:
            found = by_external_id.get(data.external_id)

        if found is not None:
            if found.wos_collection is None and data.wos_collection:
                found.is_wos_indexed = True
                found.wos_collection = data.wos_collection
                found.wos_categories = data.wos_categories
            resolved.append(found)
            continue

        paper = _new_paper_from_data(data)
        db.add(paper)
        created += 1
        resolved.append(paper)
        if paper.doi:
            by_doi.setdefault(paper.doi, paper)
        if paper.external_id:
            by_external_id.setdefault(paper.external_id, paper)

    if created:
        await db.flush()
    return resolved


async def add_paper_to_project(
    db: AsyncSession,
    project_id: UUID,
    paper_id: UUID,
    relevance_score: float | None = None,
    user_notes: str | None = None,
    tags: list[str] | None = None,
) -> ProjectPaper:
    """Link a paper to a project. Raises 409 if already linked."""
    existing = await db.execute(
        select(ProjectPaper).where(
            ProjectPaper.project_id == project_id,
            ProjectPaper.paper_id == paper_id,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Paper already in project")

    pp = ProjectPaper(
        project_id=project_id,
        paper_id=paper_id,
        relevance_score=relevance_score,
        user_notes=user_notes,
        tags=tags,
    )
    db.add(pp)
    await db.flush()

    # Reload with paper relationship
    result = await db.execute(
        select(ProjectPaper)
        .options(joinedload(ProjectPaper.paper))
        .where(ProjectPaper.id == pp.id)
    )
    return result.scalar_one()


async def list_project_papers(
    db: AsyncSession, project_id: UUID, page: int = 1, limit: int = 50
) -> tuple[list[ProjectPaper], int]:
    """List papers in a project with pagination."""
    count_result = await db.execute(
        select(func.count()).select_from(ProjectPaper).where(ProjectPaper.project_id == project_id)
    )
    total = count_result.scalar() or 0

    offset = (page - 1) * limit
    result = await db.execute(
        select(ProjectPaper)
        .options(joinedload(ProjectPaper.paper))
        .where(ProjectPaper.project_id == project_id)
        .order_by(ProjectPaper.added_at.desc())
        .offset(offset)
        .limit(limit)
    )
    papers = list(result.scalars().unique().all())
    return papers, total


async def remove_paper_from_project(
    db: AsyncSession, project_id: UUID, paper_id: UUID
) -> None:
    """Remove a paper from a project."""
    result = await db.execute(
        select(ProjectPaper).where(
            ProjectPaper.project_id == project_id,
            ProjectPaper.paper_id == paper_id,
        )
    )
    pp = result.scalar_one_or_none()
    if not pp:
        raise HTTPException(status_code=404, detail="Paper not in project")
    await db.delete(pp)
