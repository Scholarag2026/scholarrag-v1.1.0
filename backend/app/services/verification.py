# backend/app/services/verification.py
"""Paper-record verification against CrossRef.

Two kinds of output are kept apart:

* **Record checks** (``doi_exists``, ``metadata_match``) establish whether the
  bibliographic record is real and internally consistent. They alone determine
  ``overall_status``.
* **Indicators** (``wos_indexed``, ``recency``, ``citation_count``) describe the venue
  and reception of the work. They are reported with neutral statuses and never change
  ``overall_status``: an old, rarely cited paper with a valid DOI is a verified record.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from app.clients.crossref import CrossRefClient
from app.schemas.verification import PaperVerificationResult, VerificationCheck

RECORD_CHECK_STATUSES = frozenset({"pass", "fail", "warning", "skipped"})
INDICATOR_STATUSES = frozenset({"info", "note", "skipped"})

RECENCY_WINDOW_YEARS = 10
CITATION_COUNT_THRESHOLD = 10


def _check_recency(year: int | None) -> VerificationCheck:
    """Indicator: was the paper published within the last ten years?"""
    if year is None:
        return VerificationCheck(
            check_type="recency",
            status="skipped",
            message="No publication year available",
        )
    current_year = datetime.now(timezone.utc).year
    age = current_year - year
    details = {"year": year, "age_years": age, "window_years": RECENCY_WINDOW_YEARS}
    if age <= RECENCY_WINDOW_YEARS:
        return VerificationCheck(
            check_type="recency",
            status="info",
            message=f"Published in {year} ({age} years ago)",
            details=details,
        )
    return VerificationCheck(
        check_type="recency",
        status="note",
        message=(
            f"Published in {year} — older than {RECENCY_WINDOW_YEARS} years ({age} years ago)"
        ),
        details=details,
    )


def _make_wos_check(is_indexed: bool | None, collection: str | None = None) -> VerificationCheck:
    """Indicator: is the journal in the Web of Science Core Collection list?

    ``details`` carries the structured values (``indexed``, ``collection``) the
    frontend reads; the message is for humans only.
    """
    if is_indexed is None:
        return VerificationCheck(
            check_type="wos_indexed",
            status="skipped",
            message="No ISSN available or WoS journal list not loaded",
        )
    details = {"indexed": is_indexed, "collection": collection}
    if is_indexed:
        return VerificationCheck(
            check_type="wos_indexed",
            status="info",
            message="Journal is indexed in Web of Science Core Collection",
            details=details,
        )
    return VerificationCheck(
        check_type="wos_indexed",
        status="note",
        message="Journal not found in Web of Science Core Collection",
        details=details,
    )


def _check_citation_count(count: int | None) -> VerificationCheck:
    """Indicator: how often has the paper been cited?"""
    if count is None:
        return VerificationCheck(
            check_type="citation_count",
            status="skipped",
            message="No citation count available",
        )
    details = {"count": count, "threshold": CITATION_COUNT_THRESHOLD}
    if count >= CITATION_COUNT_THRESHOLD:
        return VerificationCheck(
            check_type="citation_count",
            status="info",
            message=f"{count} citations",
            details=details,
        )
    return VerificationCheck(
        check_type="citation_count",
        status="note",
        message=f"{count} citations (fewer than {CITATION_COUNT_THRESHOLD})",
        details=details,
    )


def _check_metadata_match(
    local_title: str, local_year: int | None, crossref_data: dict
) -> VerificationCheck:
    """Record check: does the local title/year agree with the CrossRef record?"""
    issues = []

    cr_title = crossref_data.get("title") or ""
    if cr_title and local_title:
        # Fuzzy title match — check if significant words overlap
        local_words = set(local_title.lower().split())
        cr_words = set(cr_title.lower().split())
        if local_words and cr_words:
            overlap = len(local_words & cr_words) / max(len(local_words), len(cr_words))
            if overlap <= 0.5:
                issues.append(
                    f"Title mismatch: local='{local_title[:60]}'"
                    f" vs CrossRef='{cr_title[:60]}'"
                )

    cr_year = crossref_data.get("year")
    if local_year and cr_year and local_year != cr_year:
        issues.append(f"Year mismatch: local={local_year} vs CrossRef={cr_year}")

    if issues:
        return VerificationCheck(
            check_type="metadata_match",
            status="warning",
            message="; ".join(issues),
            details={"crossref_title": cr_title, "crossref_year": cr_year},
        )

    return VerificationCheck(
        check_type="metadata_match",
        status="pass",
        message="Metadata matches CrossRef record",
    )


def _indicators(
    is_wos_indexed: bool | None,
    year: int | None,
    citation_count: int | None,
    wos_collection: str | None = None,
) -> list[VerificationCheck]:
    """The three indicators, always in the same order."""
    return [
        _make_wos_check(is_wos_indexed, wos_collection),
        _check_recency(year),
        _check_citation_count(citation_count),
    ]


async def verify_paper(
    paper_id: UUID,
    doi: str | None,
    title: str,
    year: int | None,
    citation_count: int | None,
    crossref_client: CrossRefClient,
    is_wos_indexed: bool | None = None,
    wos_collection: str | None = None,
) -> PaperVerificationResult:
    """Run the record checks and compute the indicators for a single paper."""
    if not doi:
        return PaperVerificationResult(
            paper_id=paper_id,
            doi=doi,
            title=title,
            overall_status="no_doi",
            checks=[
                VerificationCheck(
                    check_type="doi_exists",
                    status="skipped",
                    message="No DOI available — cannot verify against CrossRef",
                )
            ],
            indicators=_indicators(is_wos_indexed, year, citation_count, wos_collection),
        )

    crossref_data = await crossref_client.verify_doi(doi)

    if crossref_data is None:
        return PaperVerificationResult(
            paper_id=paper_id,
            doi=doi,
            title=title,
            overall_status="fail",
            checks=[
                VerificationCheck(
                    check_type="doi_exists",
                    status="fail",
                    message=f"DOI {doi} not found in CrossRef",
                )
            ],
            indicators=_indicators(is_wos_indexed, year, citation_count, wos_collection),
        )

    checks = [
        VerificationCheck(
            check_type="doi_exists",
            status="pass",
            message=f"DOI {doi} verified in CrossRef",
        ),
        _check_metadata_match(title, year, crossref_data),
    ]

    # Indicators prefer the CrossRef values when present.
    effective_year = crossref_data.get("year") or year
    effective_count = crossref_data.get("citation_count") or citation_count
    indicators = _indicators(is_wos_indexed, effective_year, effective_count, wos_collection)

    overall = "warning" if any(c.status == "warning" for c in checks) else "pass"

    return PaperVerificationResult(
        paper_id=paper_id,
        doi=doi,
        title=title,
        overall_status=overall,
        checks=checks,
        indicators=indicators,
    )
