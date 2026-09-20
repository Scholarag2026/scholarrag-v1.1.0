"""Tests for field foundations schemas."""

import pytest
from pydantic import ValidationError


def test_foundational_work_minimal():
    from app.schemas.field_foundations import FoundationalWork

    work = FoundationalWork(
        suggested_title="Situated Cognition and the Culture of Learning",
        suggested_authors=["Brown, J. S.", "Collins, A.", "Duguid, P."],
        suggested_year=1989,
        why_essential="Introduced situated learning theory",
    )
    assert work.suggested_title == "Situated Cognition and the Culture of Learning"
    assert len(work.suggested_authors) == 3
    assert work.suggested_year == 1989
    assert work.verified is False
    assert work.matched_paper is None


def test_foundational_work_with_matched_paper():
    from app.schemas.field_foundations import FoundationalWork
    from app.schemas.paper import PaperData

    paper = PaperData(
        title="Situated Cognition and the Culture of Learning",
        doi="10.3102/0013189X018001032",
        source_api="openalex",
    )
    work = FoundationalWork(
        suggested_title="Situated Cognition and the Culture of Learning",
        suggested_authors=["Brown, J. S."],
        suggested_year=1989,
        why_essential="Introduced situated learning theory",
        verified=True,
        matched_paper=paper,
    )
    assert work.verified is True
    assert work.matched_paper is not None
    assert work.matched_paper.doi == "10.3102/0013189X018001032"


def test_foundational_works_list():
    from app.schemas.field_foundations import FoundationalWork, FoundationalWorksList

    works = FoundationalWorksList(
        works=[
            FoundationalWork(
                suggested_title="Paper A",
                suggested_authors=["Author A"],
                suggested_year=2000,
                why_essential="Reason A",
            ),
            FoundationalWork(
                suggested_title="Paper B",
                suggested_authors=["Author B"],
                suggested_year=2005,
                why_essential="Reason B",
            ),
        ]
    )
    assert len(works.works) == 2


def test_foundational_works_list_empty():
    from app.schemas.field_foundations import FoundationalWorksList

    works = FoundationalWorksList(works=[])
    assert works.works == []


def test_field_foundations_request_minimal():
    from app.schemas.field_foundations import FieldFoundationsRequest

    req = FieldFoundationsRequest(topic="Educational Technology")
    assert req.topic == "Educational Technology"
    assert req.research_questions == []


def test_field_foundations_request_with_questions():
    from app.schemas.field_foundations import FieldFoundationsRequest

    req = FieldFoundationsRequest(
        topic="Educational Technology",
        research_questions=[
            "How does gamification affect learning outcomes?",
            "What role does AI play in personalized learning?",
        ],
    )
    assert len(req.research_questions) == 2


def test_field_foundations_result():
    from app.schemas.field_foundations import FieldFoundationsResult, FoundationalWork

    result = FieldFoundationsResult(
        field="Educational Technology",
        works=[
            FoundationalWork(
                suggested_title="Paper A",
                suggested_authors=["Author A"],
                suggested_year=2000,
                why_essential="Reason A",
                verified=True,
            ),
            FoundationalWork(
                suggested_title="Paper B",
                suggested_authors=["Author B"],
                suggested_year=2005,
                why_essential="Reason B",
                verified=False,
            ),
        ],
        verified_count=1,
        total_count=2,
    )
    assert result.field == "Educational Technology"
    assert result.verified_count == 1
    assert result.total_count == 2
    assert len(result.works) == 2
