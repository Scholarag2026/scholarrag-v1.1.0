import pytest
from pydantic import ValidationError

from app.schemas.dataset import (
    ColumnInfo,
    ColumnUpdate,
    DatasetDetail,
    DatasetListItem,
    DatasetUploadResponse,
)


def test_column_info_valid():
    col = ColumnInfo(name="age", dtype="numeric", missing_count=2, unique_count=45)
    assert col.name == "age"
    assert col.dtype == "numeric"


def test_column_info_invalid_dtype():
    with pytest.raises(ValidationError):
        ColumnInfo(name="x", dtype="invalid", missing_count=0, unique_count=1)


def test_column_update_rename():
    update = ColumnUpdate(original_name="age", name="age_years")
    assert update.original_name == "age"
    assert update.name == "age_years"
    assert update.dtype is None
    assert update.role is None
    assert update.deleted is None


def test_column_update_with_role():
    update = ColumnUpdate(original_name="score", role="independent")
    assert update.role == "independent"


def test_dataset_upload_response():
    resp = DatasetUploadResponse(
        id="abc-123",
        filename="data.csv",
        row_count=100,
        columns=[{"name": "age", "dtype": "numeric", "missing_count": 0, "unique_count": 50}],
        status="validated",
        available_sheets=[],
        sheet_name=None,
    )
    assert resp.row_count == 100
    assert len(resp.columns) == 1


def test_dataset_upload_response_with_sheets():
    resp = DatasetUploadResponse(
        id="abc-123",
        filename="data.xlsx",
        row_count=0,
        columns=[],
        status="uploaded",
        available_sheets=["Sheet1", "Sheet2", "Data"],
        sheet_name="Sheet1",
    )
    assert len(resp.available_sheets) == 3


def test_dataset_detail_with_preview():
    detail = DatasetDetail(
        id="abc-123",
        filename="data.csv",
        columns=[{"name": "age", "dtype": "numeric", "missing_count": 0, "unique_count": 50}],
        row_count=100,
        preview_rows=[{"age": 25}, {"age": 30}],
        data_type="quantitative",
    )
    assert len(detail.preview_rows) == 2


def test_dataset_list_item():
    item = DatasetListItem(
        id="abc-123",
        filename="survey.xlsx",
        file_type="xlsx",
        row_count=500,
        data_type="mixed",
        status="validated",
        created_at="2026-03-21T10:00:00Z",
    )
    assert item.file_type == "xlsx"
