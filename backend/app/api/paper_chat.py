"""API endpoint for chatting with library papers."""

from __future__ import annotations

from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.paper_selector_agent import select_papers_for_section
from app.config import settings
from app.database import get_db
from app.dependencies import get_current_user
from app.models.chat_message import ChatMessage
from app.models.paper import Paper
from app.models.project_paper import ProjectPaper
from app.models.user import User
from app.services import project as project_service
from app.services.citation_render import render_answer

router = APIRouter(tags=["paper-chat"])

# Newest N project papers considered for relevance narrowing on each chat turn.
CHAT_CANDIDATE_LIMIT = 200
# Papers whose full context is loaded and sent to the model.
CHAT_CONTEXT_LIMIT = 20


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class ChatReference(BaseModel):
    title: str
    authors: str
    year: int | None
    journal_name: str | None = None
    doi: str | None


class ChatResponse(BaseModel):
    answer: str
    references: list[ChatReference]
    unresolved_citations: list[str] = []


CHAT_SYSTEM_PROMPT = """\
You are a research assistant who has read and analyzed a collection of academic papers.
Answer the user's question based ONLY on the papers provided below.

RULES:
1. Every claim or statement must be supported by an in-text citation: (Author, Year)
2. If you cannot answer from the provided papers, say so honestly
3. Synthesize across multiple papers — don't just list individual paper summaries
4. Compare and contrast findings when relevant
5. Do not add a "References" section yourself; one is generated automatically, in the
   project's chosen style, from the paper metadata after your answer
6. Use academic but accessible language
7. Be thorough but concise — answer the question directly
8. When quoting, use exact quotes with page numbers if available
"""


@router.post("/projects/{project_id}/chat", response_model=ChatResponse)
async def chat_with_papers(
    project_id: UUID,
    req: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Chat with library papers — AI answers questions with citations."""
    project = await project_service.get_project(db, project_id, user.id)
    # citation_style is a str-enum: .value is "IEEE", str() is "CitationStyle.ieee".
    raw_style = project.citation_style
    citation_style = getattr(raw_style, "value", raw_style) or "APA"

    # Candidate window: newest CHAT_CANDIDATE_LIMIT papers, metadata columns only.
    # `Paper.metadata_["deep_analysis"]` keeps the huge `fulltext_chunks` blob out of the query.
    candidate_result = await db.execute(
        select(
            Paper.id.label("id"),
            Paper.title.label("title"),
            Paper.year.label("year"),
            Paper.metadata_["deep_analysis"].label("deep_analysis"),
        )
        .join(ProjectPaper, ProjectPaper.paper_id == Paper.id)
        .where(ProjectPaper.project_id == project_id)
        .order_by(ProjectPaper.added_at.desc())
        .limit(CHAT_CANDIDATE_LIMIT)
    )
    candidates = candidate_result.all()

    if not candidates:
        return ChatResponse(
            answer="No papers in your library yet. Add papers first to chat with them.",
            references=[],
        )

    paper_summaries = []
    for row in candidates:
        analysis = row.deep_analysis if isinstance(row.deep_analysis, dict) else {}
        themes = analysis.get("themes", [])
        paper_summaries.append({
            "id": str(row.id),
            "title": row.title,
            "year": row.year,
            "themes": ", ".join(themes) if themes else "N/A",
        })

    try:
        selected_ids = await select_papers_for_section(
            "chat", paper_summaries, question=req.question
        )
    except Exception:
        selected_ids = []

    selected_set = {str(pid) for pid in selected_ids}
    selected_rows = [c for c in candidates if str(c.id) in selected_set]
    if not selected_rows:
        selected_rows = list(candidates)
    selected_rows = selected_rows[:CHAT_CONTEXT_LIMIT]

    # Context load: only the selected papers, only the columns the prompt needs.
    context_result = await db.execute(
        select(
            Paper.id.label("id"),
            Paper.title.label("title"),
            Paper.year.label("year"),
            Paper.doi.label("doi"),
            Paper.journal_name.label("journal_name"),
            Paper.authors.label("authors"),
            Paper.abstract.label("abstract"),
            Paper.metadata_["deep_analysis"].label("deep_analysis"),
        ).where(Paper.id.in_([c.id for c in selected_rows]))
    )
    by_id = {row.id: row for row in context_result.all()}
    selected_papers = [by_id[c.id] for c in selected_rows if c.id in by_id]

    # Build context from deep analyses
    context_parts = []
    references_list = []

    for p in selected_papers:
        authors_str = ", ".join(
            a.get("name", "?") if isinstance(a, dict) else str(a)
            for a in (p.authors or [])
        )
        analysis = p.deep_analysis if isinstance(p.deep_analysis, dict) else {}

        context_parts.append(f"\n### {p.title}")
        context_parts.append(f"Authors: {authors_str}")
        context_parts.append(f"Year: {p.year or 'n.d.'}")
        # The model can only cite what it sees: without these lines the APA
        # references it writes are missing the journal and DOI.
        if p.journal_name:
            context_parts.append(f"Journal: {p.journal_name}")
        if p.doi:
            context_parts.append(f"DOI: https://doi.org/{p.doi}")

        if analysis.get("key_findings"):
            context_parts.append(f"Key Findings: {analysis['key_findings']}")
            if analysis.get("methodology"):
                context_parts.append(f"Methodology: {analysis['methodology']}")
            if analysis.get("theoretical_framework"):
                context_parts.append(
                    f"Theoretical Framework: {analysis['theoretical_framework']}"
                )
            if analysis.get("key_quotes_with_citations"):
                context_parts.append("Key Quotes:")
                for q in analysis["key_quotes_with_citations"]:
                    context_parts.append(f"  - {q}")
            if analysis.get("limitations"):
                context_parts.append(f"Limitations: {analysis['limitations']}")
            if analysis.get("future_directions"):
                context_parts.append(f"Future Directions: {analysis['future_directions']}")
        elif p.abstract:
            context_parts.append(f"Abstract: {p.abstract}")

        references_list.append(ChatReference(
            title=p.title,
            authors=authors_str,
            year=p.year,
            journal_name=p.journal_name,
            doi=p.doi,
        ))

    papers_context = "\n".join(context_parts)

    # Call DeepSeek
    user_prompt = f"""Based on the following papers, answer this question:

Question: {req.question}

Papers:
{papers_context}

Answer thoroughly with in-text citations (Author, Year). Do not add your own References \
section; one is generated automatically."""

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.deepseek_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.deepseek_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.deepseek_model,
                "messages": [
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.5,
                "max_tokens": 4096,
            },
        )
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"]

    # Render the project's chosen citation style from the model's own author-year text
    # before the answer is shown or persisted, so chat history matches what the user saw.
    rendered = render_answer(answer, selected_papers, citation_style)
    answer = rendered.text

    # Persist chat messages
    user_msg = ChatMessage(project_id=project_id, role="user", content=req.question)
    db.add(user_msg)
    assistant_msg = ChatMessage(project_id=project_id, role="assistant", content=answer)
    db.add(assistant_msg)
    await db.commit()

    return ChatResponse(
        answer=answer, references=references_list, unresolved_citations=rendered.unresolved
    )


@router.get("/projects/{project_id}/chat/history")
async def get_chat_history(
    project_id: UUID,
    page: int = Query(1, ge=1),
    limit: int = Query(200, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return one page of chat messages, newest page first, ordered oldest -> newest."""
    await project_service.get_project(db, project_id, user.id)
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.project_id == project_id)
        .order_by(ChatMessage.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
    )
    messages = list(result.scalars().all())
    messages.reverse()
    return [
        {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
        for m in messages
    ]


@router.delete("/projects/{project_id}/chat/history")
async def clear_chat_history(
    project_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete all chat messages for a project."""
    await project_service.get_project(db, project_id, user.id)
    await db.execute(
        delete(ChatMessage).where(ChatMessage.project_id == project_id)
    )
    await db.commit()
    return {"success": True}
