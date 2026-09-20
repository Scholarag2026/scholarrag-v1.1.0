"""Dataset service — upload, parse, preview, and manage datasets.

Provides functions for parsing CSV/TSV/XLSX files, detecting column types,
generating previews, and CRUD operations on Dataset records with file storage.
"""

from __future__ import annotations

import asyncio
import io
import os
import uuid
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetStatus, DataType, FileType
from app.schemas.dataset import NormalizedColumnUpdate
from app.services.storage import storage_service

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS: set[str] = {".csv", ".xlsx", ".xls", ".tsv"}
TEXT_LENGTH_THRESHOLD: int = 50
CATEGORICAL_UNIQUE_RATIO: float = 0.3

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_csv(path_or_bytes: str | bytes) -> pd.DataFrame:
    """Parse a CSV file from a file path or raw bytes into a DataFrame."""
    if isinstance(path_or_bytes, bytes):
        return pd.read_csv(io.BytesIO(path_or_bytes))
    return pd.read_csv(path_or_bytes)


def parse_tsv(path_or_bytes: str | bytes) -> pd.DataFrame:
    """Parse a TSV file from a file path or raw bytes into a DataFrame."""
    if isinstance(path_or_bytes, bytes):
        return pd.read_csv(io.BytesIO(path_or_bytes), sep="\t")
    return pd.read_csv(path_or_bytes, sep="\t")


def parse_xlsx(
    path_or_bytes: str | bytes, sheet_name: str | None = None
) -> pd.DataFrame:
    """Parse an XLSX file from a file path or raw bytes into a DataFrame.

    If *sheet_name* is ``None`` the first sheet is used.
    """
    source = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    return pd.read_excel(source, sheet_name=sheet_name or 0, engine="openpyxl")


def get_xlsx_sheet_names(content: bytes) -> list[str]:
    """Return the list of sheet names from XLSX bytes."""
    import openpyxl

    wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    names = wb.sheetnames
    wb.close()
    return names


# ---------------------------------------------------------------------------
# Column / data-type detection
# ---------------------------------------------------------------------------


def detect_column_types(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect the semantic type of each column in *df*.

    Returns a list of dicts with keys: ``name``, ``dtype``, ``missing_count``,
    ``unique_count``.

    Classification rules (applied in order):
    1. Numeric dtype -> ``"numeric"``
    2. Average string length > TEXT_LENGTH_THRESHOLD -> ``"text"``
    3. unique/total ratio < CATEGORICAL_UNIQUE_RATIO -> ``"categorical"``
    4. Otherwise -> ``"categorical"``
    """
    result: list[dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique())

        if pd.api.types.is_numeric_dtype(series):
            dtype = "numeric"
        else:
            # Compute average string length on non-null values
            non_null = series.dropna().astype(str)
            avg_len = non_null.str.len().mean() if len(non_null) > 0 else 0.0
            if avg_len > TEXT_LENGTH_THRESHOLD:
                dtype = "text"
            elif len(df) > 0 and (unique_count / len(df)) < CATEGORICAL_UNIQUE_RATIO:
                dtype = "categorical"
            else:
                dtype = "categorical"

        result.append(
            {
                "name": col,
                "dtype": dtype,
                "missing_count": missing_count,
                "unique_count": unique_count,
            }
        )
    return result


def detect_data_type(columns: list[dict[str, Any]]) -> str:
    """Return ``"quantitative"``, ``"qualitative"``, or ``"mixed"`` based on column types."""
    has_numeric = any(c["dtype"] == "numeric" for c in columns)
    has_text = any(c["dtype"] == "text" for c in columns)
    has_categorical = any(c["dtype"] == "categorical" for c in columns)

    if has_text:
        if has_numeric:
            return "mixed"
        return "qualitative"
    if has_numeric and has_categorical:
        return "mixed"
    if has_numeric:
        return "quantitative"
    return "qualitative"


# ---------------------------------------------------------------------------
# Preview / conversion helpers
# ---------------------------------------------------------------------------


def get_preview_rows(df: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    """Return the first *limit* rows as a list of dicts, converting NaN to None."""
    preview_df = df.head(limit)
    records = preview_df.to_dict(orient="records")
    # Replace NaN with None for JSON compatibility
    for row in records:
        for key, value in row.items():
            if pd.isna(value):
                row[key] = None
    return records


def df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to Parquet bytes using PyArrow."""
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def parquet_bytes_to_df(content: bytes) -> pd.DataFrame:
    """Deserialize Parquet bytes back into a DataFrame."""
    buf = io.BytesIO(content)
    table = pq.read_table(buf)
    return table.to_pandas()


# ---------------------------------------------------------------------------
# Async CRUD / business-logic functions
# ---------------------------------------------------------------------------


def _validate_upload(filename: str, content: bytes) -> FileType:
    """Validate file extension and size; return the corresponding FileType."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    if len(content) > MAX_FILE_SIZE:
        raise ValueError(
            f"File size {len(content)} exceeds maximum allowed size of {MAX_FILE_SIZE} bytes"
        )
    mapping = {
        ".csv": FileType.csv,
        ".xlsx": FileType.xlsx,
        ".xls": FileType.xlsx,
        ".tsv": FileType.tsv,
    }
    return mapping[ext]


def _parse_content(file_type: FileType, content: bytes, sheet_name: str | None) -> pd.DataFrame:
    """Parse raw bytes into a DataFrame according to file type."""
    if file_type == FileType.csv:
        return parse_csv(content)
    elif file_type == FileType.tsv:
        return parse_tsv(content)
    elif file_type == FileType.xlsx:
        return parse_xlsx(content, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def _process_upload_sync(
    file_type: FileType,
    content: bytes,
    sheet_name: str | None,
    raw_key: str,
    parquet_key: str,
) -> dict[str, Any]:
    """All CPU/disk work of an upload, in one hop. Safe to run in a worker thread (D20).

    pandas/openpyxl parsing, per-column type detection, pyarrow serialization and both
    storage writes are blocking; with ``--workers 2`` running them on the event loop takes
    out half the app's request capacity for the duration.
    """
    df = _parse_content(file_type, content, sheet_name)
    columns = detect_column_types(df)
    data_type_str = detect_data_type(columns)
    storage_service.save(raw_key, content)
    storage_service.save(parquet_key, df_to_parquet_bytes(df))
    return {
        "columns": columns,
        "data_type": data_type_str,
        "row_count": len(df),
        "preview": get_preview_rows(df),
    }


async def upload_dataset(
    db: AsyncSession,
    project_id: uuid.UUID,
    user_id: uuid.UUID,
    filename: str,
    content: bytes,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    """Full upload flow: validate, parse, store raw + parquet, create DB record.

    Returns a metadata dict with dataset info.
    """
    file_type = _validate_upload(filename, content)

    dataset_id = uuid.uuid4()
    raw_key = f"datasets/{project_id}/{dataset_id}/raw/{filename}"
    parquet_key = f"datasets/{project_id}/{dataset_id}/data.parquet"

    parsed = await asyncio.to_thread(
        _process_upload_sync, file_type, content, sheet_name, raw_key, parquet_key
    )
    data_type = DataType(parsed["data_type"])

    dataset = Dataset(
        id=dataset_id,
        project_id=project_id,
        user_id=user_id,
        filename=filename,
        raw_storage_path=raw_key,
        parquet_path=parquet_key,
        sheet_name=sheet_name,
        file_type=file_type,
        file_size=len(content),
        row_count=parsed["row_count"],
        columns=parsed["columns"],
        data_type=data_type,
        status=DatasetStatus.validated,
    )
    db.add(dataset)
    await db.flush()

    return {
        "id": str(dataset.id),
        "filename": filename,
        "file_type": file_type.value,
        "file_size": len(content),
        "row_count": parsed["row_count"],
        "columns": parsed["columns"],
        "data_type": data_type.value,
        "status": DatasetStatus.validated.value,
        "preview": parsed["preview"],
    }


async def get_dataset(
    db: AsyncSession,
    dataset_id: uuid.UUID,
    user_id: uuid.UUID,
) -> Dataset:
    """Get a single dataset with authorization check."""
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.user_id == user_id)
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise ValueError("Dataset not found or access denied")
    return dataset


async def get_dataset_preview(
    db: AsyncSession,
    dataset_id: uuid.UUID,
    user_id: uuid.UUID,
) -> dict[str, Any]:
    """Get dataset metadata together with preview rows."""
    dataset = await get_dataset(db, dataset_id, user_id)
    parquet_key = dataset.cleaned_parquet_path or dataset.parquet_path
    if not parquet_key:
        raise ValueError("No parquet data available for this dataset")
    preview = await asyncio.to_thread(_preview_rows_sync, parquet_key)
    columns = dataset.columns or []
    return {
        "id": str(dataset.id),
        "project_id": str(dataset.project_id),
        "filename": dataset.filename,
        "file_type": dataset.file_type.value,
        "file_size": dataset.file_size,
        "row_count": dataset.row_count,
        "column_count": len(columns),
        "columns": columns,
        "data_type": dataset.data_type.value,
        "status": dataset.status.value,
        "preview": preview,
        # Alias consumed by the frontend preview table (data-preview-table.tsx).
        "preview_rows": preview,
    }


async def list_datasets(
    db: AsyncSession,
    project_id: uuid.UUID,
    user_id: uuid.UUID,
    page: int = 1,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List datasets for a project owned by the user, newest first, with pagination."""
    result = await db.execute(
        select(Dataset)
        .where(Dataset.project_id == project_id, Dataset.user_id == user_id)
        .order_by(Dataset.created_at.desc(), Dataset.id)
        .offset((page - 1) * limit)
        .limit(limit)
    )
    datasets = result.scalars().all()
    return [
        {
            "id": str(ds.id),
            "filename": ds.filename,
            "file_type": ds.file_type.value,
            "file_size": ds.file_size,
            "row_count": ds.row_count,
            "columns": ds.columns,
            "data_type": ds.data_type.value,
            "status": ds.status.value,
            "created_at": ds.created_at.isoformat() if ds.created_at else None,
        }
        for ds in datasets
    ]


async def delete_dataset(
    db: AsyncSession,
    dataset_id: uuid.UUID,
    user_id: uuid.UUID,
) -> None:
    """Delete a dataset record and its associated storage files."""
    dataset = await get_dataset(db, dataset_id, user_id)

    # Delete storage files
    storage_service.delete(dataset.raw_storage_path)
    if dataset.parquet_path:
        storage_service.delete(dataset.parquet_path)
    if dataset.cleaned_parquet_path:
        storage_service.delete(dataset.cleaned_parquet_path)

    await db.delete(dataset)
    await db.flush()


def _read_parquet_key(parquet_key: str) -> pd.DataFrame:
    """Read + deserialize one parquet key. Blocking; safe to run in a worker thread."""
    return parquet_bytes_to_df(storage_service.read(parquet_key))


def _preview_rows_sync(parquet_key: str) -> list[dict[str, Any]]:
    """Read parquet and build preview rows in one blocking hop."""
    return get_preview_rows(_read_parquet_key(parquet_key))


def load_dataset_df(dataset: Dataset) -> pd.DataFrame:
    """Load the current DataFrame for a dataset (cleaned if available, else original parquet).

    Kept synchronous and blocking on purpose: it is only meant to be called from inside
    code that already runs off the event loop, e.g. the ``_clean`` helper in
    :func:`apply_dataset_updates`, which is itself wrapped in ``asyncio.to_thread``.
    Callers running directly on the event loop (job bodies, request handlers) must use
    :func:`load_dataset_df_async` instead.
    """
    parquet_key = dataset.cleaned_parquet_path or dataset.parquet_path
    if not parquet_key:
        raise ValueError("No parquet data available for this dataset")
    return _read_parquet_key(parquet_key)


async def load_dataset_df_async(dataset: Dataset) -> pd.DataFrame:
    """Load a dataset's DataFrame without blocking the event loop (D20)."""
    parquet_key = dataset.cleaned_parquet_path or dataset.parquet_path
    if not parquet_key:
        raise ValueError("No parquet data available for this dataset")
    return await asyncio.to_thread(_read_parquet_key, parquet_key)


# ---------------------------------------------------------------------------
# Column-update planning and application
# ---------------------------------------------------------------------------


@dataclass
class ColumnUpdatePlan:
    """The resolved effect of a batch of column updates."""

    columns: list[dict[str, Any]]
    dropped: list[str]
    renames: dict[str, str]
    dtype_changes: dict[str, str]
    applied: int


def plan_column_updates(
    columns: list[dict[str, Any]],
    updates: list[NormalizedColumnUpdate],
) -> ColumnUpdatePlan:
    """Resolve *updates* against the stored *columns* metadata.

    Pure and synchronous. Updates naming a column that does not exist are ignored
    (they are no-ops from an already-applied edit). Renaming onto an existing column
    name raises ``ValueError``.
    """
    by_name: dict[str, dict[str, Any]] = {c["name"]: dict(c) for c in columns}
    order: list[str] = [c["name"] for c in columns]
    dropped: list[str] = []
    renames: dict[str, str] = {}
    dtype_changes: dict[str, str] = {}
    applied = 0

    for u in updates:
        col = by_name.get(u.original_name)
        if col is None:
            continue
        if u.deleted:
            if u.original_name not in dropped:
                dropped.append(u.original_name)
                applied += 1
            continue

        changed = False
        if u.dtype and u.dtype != col.get("dtype"):
            col["dtype"] = u.dtype
            dtype_changes[u.original_name] = u.dtype
            changed = True
        if u.role and u.role != col.get("role"):
            col["role"] = u.role
            changed = True
        if u.new_name and u.new_name != u.original_name:
            if u.new_name in by_name:
                raise ValueError(
                    f"Cannot rename '{u.original_name}' to existing column "
                    f"'{u.new_name}'"
                )
            renames[u.original_name] = u.new_name
            col["name"] = u.new_name
            changed = True
        if changed:
            applied += 1

    new_columns = [by_name[n] for n in order if n not in dropped]
    return ColumnUpdatePlan(
        columns=new_columns,
        dropped=dropped,
        renames=renames,
        dtype_changes=dtype_changes,
        applied=applied,
    )


def cleaned_parquet_key(project_id: uuid.UUID, dataset_id: uuid.UUID) -> str:
    """Storage key for a dataset's cleaned parquet (overwritten in place)."""
    return f"datasets/{project_id}/{dataset_id}/cleaned.parquet"


def apply_data_edits(
    df: pd.DataFrame, edits: list[dict[str, Any]]
) -> pd.DataFrame:
    """Apply per-cell edits, addressed by positional row index and current column name.

    Synchronous by design — track T7 wraps the whole cleaning block in ``asyncio.to_thread``.
    """
    if not edits:
        return df
    out = df.copy()
    for edit in edits:
        column = edit.get("column_name")
        row = edit.get("row_index")
        if column not in out.columns:
            raise ValueError(f"Unknown column in data edit: {column}")
        if not isinstance(row, int) or isinstance(row, bool) or row < 0 or row >= len(out):
            raise ValueError(f"Row index out of range in data edit: {row}")
        out.iat[row, out.columns.get_loc(column)] = edit.get("new_value")
    return out


def apply_plan_to_df(df: pd.DataFrame, plan: ColumnUpdatePlan) -> pd.DataFrame:
    """Apply deletes, renames and dtype coercions from *plan* to *df*.

    Synchronous by design (see ``apply_data_edits``).
    """
    out = df
    to_drop = [c for c in plan.dropped if c in out.columns]
    if to_drop:
        out = out.drop(columns=to_drop)

    renames = {old: new for old, new in plan.renames.items() if old in out.columns}
    if renames:
        out = out.rename(columns=renames)

    for original, dtype in plan.dtype_changes.items():
        target = plan.renames.get(original, original)
        if target not in out.columns:
            continue
        if dtype == "numeric":
            out[target] = pd.to_numeric(out[target], errors="coerce")
        else:
            out[target] = out[target].map(lambda v: v if pd.isna(v) else str(v))
    return out


def refresh_column_stats(
    columns: list[dict[str, Any]], df: pd.DataFrame
) -> list[dict[str, Any]]:
    """Recompute missing/unique counts for surviving columns, preserving name/dtype/role."""
    refreshed: list[dict[str, Any]] = []
    for col in columns:
        entry = dict(col)
        name = entry.get("name")
        if name in df.columns:
            series = df[name]
            entry["missing_count"] = int(series.isna().sum())
            entry["unique_count"] = int(series.nunique())
        refreshed.append(entry)
    return refreshed


async def update_dataset(
    db: AsyncSession,
    dataset_id: uuid.UUID,
    user_id: uuid.UUID,
    column_updates: list[NormalizedColumnUpdate],
    data_edits: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply column updates and cell edits, rewriting the cleaned parquet when the
    dataframe actually changes.

    Raises ``LookupError`` when the dataset does not exist or is not owned by
    *user_id*, and ``ValueError`` for an invalid update or edit.
    """
    try:
        dataset = await get_dataset(db, dataset_id, user_id)
    except ValueError as exc:
        raise LookupError("Dataset not found") from exc

    plan = plan_column_updates(list(dataset.columns or []), column_updates)
    edits = list(data_edits or [])

    frame_changed = bool(plan.dropped or plan.renames or plan.dtype_changes or edits)
    if frame_changed:
        # Blocking pandas/pyarrow work (including the per-column missing/unique stats
        # refresh) runs off the event loop (D20 / issue #27).
        def _clean() -> tuple[pd.DataFrame, str, list[dict[str, Any]]]:
            df = load_dataset_df(dataset)
            df = apply_data_edits(df, edits)
            df = apply_plan_to_df(df, plan)
            key = cleaned_parquet_key(dataset.project_id, dataset.id)
            storage_service.save(key, df_to_parquet_bytes(df))
            columns = refresh_column_stats(plan.columns, df)
            return df, key, columns

        df, key, columns = await asyncio.to_thread(_clean)
        dataset.cleaned_parquet_path = key
        dataset.row_count = len(df)
        plan.columns = columns

    if plan.applied or frame_changed:
        dataset.columns = plan.columns
        dataset.data_type = DataType(detect_data_type(plan.columns))

    await db.flush()

    return {
        "id": str(dataset.id),
        "project_id": str(dataset.project_id),
        "filename": dataset.filename,
        "file_type": dataset.file_type.value,
        "file_size": dataset.file_size,
        "row_count": dataset.row_count,
        "columns": dataset.columns,
        "data_type": dataset.data_type.value,
        "status": "updated" if (plan.applied or frame_changed) else "unchanged",
        "dataset_status": dataset.status.value,
        "cleaned_parquet_path": dataset.cleaned_parquet_path,
        "applied_column_updates": plan.applied,
        "applied_data_edits": len(edits),
    }
