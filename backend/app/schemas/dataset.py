from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

ColumnDtype = Literal["numeric", "categorical", "text"]
ColumnRole = Literal[
    "independent", "dependent", "control", "participant_id", "text_data"
]


class ColumnInfo(BaseModel):
    name: str
    dtype: ColumnDtype
    missing_count: int
    unique_count: int


class ColumnUpdate(BaseModel):
    """Delta form: identifies the target column by ``original_name``."""

    original_name: str
    name: str | None = None  # new name (rename)
    dtype: ColumnDtype | None = None
    role: ColumnRole | None = None
    deleted: bool | None = None


class ColumnStateUpdate(BaseModel):
    """Full-state form: the column object the frontend renders and PUTs back.

    Extra keys the frontend carries (``missing_count``, ``missing_pct``,
    ``unique_count``, ``sample_values``) are accepted and ignored.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    new_name: str | None = None
    dtype: ColumnDtype | None = None
    role: ColumnRole | None = None
    deleted: bool | None = None


class NormalizedColumnUpdate(BaseModel):
    """Canonical internal form both request shapes are normalized into."""

    original_name: str
    new_name: str | None = None
    dtype: ColumnDtype | None = None
    role: ColumnRole | None = None
    deleted: bool = False


class DatasetUploadResponse(BaseModel):
    id: str
    filename: str
    row_count: int
    columns: list[dict[str, Any]]
    status: str
    available_sheets: list[str]
    sheet_name: str | None = None


class DatasetDetail(BaseModel):
    id: str
    filename: str
    columns: list[dict[str, Any]]
    row_count: int
    preview_rows: list[dict[str, Any]]
    data_type: str


class DatasetListItem(BaseModel):
    id: str
    filename: str
    file_type: str
    row_count: int
    data_type: str
    status: str
    created_at: str


class DatasetUpdateRequest(BaseModel):
    """Accepts BOTH the delta shape (``column_updates``) and the full-state shape
    (``columns``) the frontend has always sent."""

    model_config = ConfigDict(extra="ignore")

    column_updates: list[ColumnUpdate] = []
    columns: list[ColumnStateUpdate] | None = None
    data_edits: list[dict[str, Any]] = []  # [{row_index, column_name, new_value}]

    def normalized_column_updates(self) -> list[NormalizedColumnUpdate]:
        """Flatten both request shapes into one ordered list of canonical updates."""
        updates: list[NormalizedColumnUpdate] = []
        for u in self.column_updates:
            updates.append(
                NormalizedColumnUpdate(
                    original_name=u.original_name,
                    new_name=u.name,
                    dtype=u.dtype,
                    role=u.role,
                    deleted=bool(u.deleted),
                )
            )
        for c in self.columns or []:
            updates.append(
                NormalizedColumnUpdate(
                    original_name=c.name,
                    new_name=c.new_name,
                    dtype=c.dtype,
                    role=c.role,
                    deleted=bool(c.deleted),
                )
            )
        return updates
