"""Schemas for qualitative coding analysis."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class Theme(BaseModel):
    id: str
    name: str
    description: str
    parent_id: str | None = None


class Code(BaseModel):
    id: str
    name: str
    description: str
    theme_id: str
    color: str


class CodebookSchema(BaseModel):
    codes: list[Code]
    themes: list[Theme]


class CodingSegment(BaseModel):
    segment_id: str
    text: str
    codes: list[str]
    confidence: Literal["high", "medium", "low"]
    reasoning: str


class Uncertainty(BaseModel):
    segment_id: str
    issue: str
    suggested_codes: list[str]


class CodingResult(BaseModel):
    codebook: CodebookSchema
    coded_segments: list[CodingSegment]
    uncertainties: list[Uncertainty]
    annotation_notes: list[str]


class CodingSessionUpdate(BaseModel):
    coded_segments: list[CodingSegment]


class KappaScore(BaseModel):
    code_name: str
    kappa: float
    agreement_pct: float


class InterCoderReport(BaseModel):
    overall_kappa: float
    per_code_kappa: list[KappaScore]
    agreement_pct: float
    disagreement_segments: list[dict[str, Any]]
