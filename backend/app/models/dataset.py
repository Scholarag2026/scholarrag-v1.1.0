import enum
import uuid

from sqlalchemy import (
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class FileType(str, enum.Enum):
    csv = "csv"
    xlsx = "xlsx"
    tsv = "tsv"


class DataType(str, enum.Enum):
    quantitative = "quantitative"
    qualitative = "qualitative"
    mixed = "mixed"


class DatasetStatus(str, enum.Enum):
    uploaded = "uploaded"
    validated = "validated"
    error = "error"


class CoderType(str, enum.Enum):
    ai = "ai"
    human = "human"


class CodingStatus(str, enum.Enum):
    in_progress = "in_progress"
    completed = "completed"


class Dataset(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "datasets"

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    raw_storage_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    parquet_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    cleaned_parquet_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    sheet_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    file_type: Mapped[FileType] = mapped_column(SAEnum(FileType), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    columns: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    data_type: Mapped[DataType] = mapped_column(SAEnum(DataType), default=DataType.quantitative)
    status: Mapped[DatasetStatus] = mapped_column(SAEnum(DatasetStatus), default=DatasetStatus.uploaded)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class Codebook(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "codebooks"

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    codes: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    themes: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class CodingSession(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "coding_sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    codebook_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("codebooks.id", ondelete="CASCADE"), nullable=False
    )
    coder_type: Mapped[CoderType] = mapped_column(SAEnum(CoderType), nullable=False)
    coded_segments: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[CodingStatus] = mapped_column(SAEnum(CodingStatus), default=CodingStatus.in_progress)
