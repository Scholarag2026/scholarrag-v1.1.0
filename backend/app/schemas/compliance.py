from __future__ import annotations

from pydantic import BaseModel


class ComplianceCheck(BaseModel):
    check_type: str  # "word_count", "sections_present", "abstract_length", "reference_count"
    status: str  # "pass", "fail", "warning"
    message: str
    details: dict | None = None


class ComplianceReport(BaseModel):
    overall_status: str  # "pass", "fail", "warning"
    checks: list[ComplianceCheck]
    total_word_count: int
    sections_found: list[str]
