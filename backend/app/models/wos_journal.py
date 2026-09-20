"""WosJournal - stores ISSNs from the Web of Science Master Journal List."""

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


class WosJournal(Base, UUIDMixin):
    __tablename__ = "wos_journals"

    issn: Mapped[str | None] = mapped_column(String(20), index=True)
    eissn: Mapped[str | None] = mapped_column(String(20), index=True)
    journal_title: Mapped[str] = mapped_column(String(500), nullable=False)
    collection: Mapped[str | None] = mapped_column(String(10), nullable=True)
    categories: Mapped[str | None] = mapped_column(String(1000), nullable=True)
