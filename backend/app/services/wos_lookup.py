"""WoS Master Journal List ISSN lookup service.

This module is kept for backwards compatibility.  New code should import
directly from :mod:`app.services.wos_import`.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.wos_import import (  # noqa: F401  (re-exported)
    get_wos_journal_count,
    import_wos_csv,
    is_wos_indexed as _is_wos_indexed,
)


async def is_wos_indexed(
    db: AsyncSession, issn: str | None, eissn: str | None = None
) -> bool | None:
    """Check if a journal ISSN is in the WoS Master Journal List.

    Returns True if found, False if not found, None if no ISSNs to check
    or the journal list is empty.

    .. deprecated::
        Use :func:`app.services.wos_import.is_wos_indexed` for the full
        (indexed, collection, categories) tuple.
    """
    if not issn and not eissn:
        return None

    total = await get_wos_journal_count(db)
    if total == 0:
        return None  # No WoS data loaded yet

    indexed, _collection, _categories = await _is_wos_indexed(db, issn, eissn)
    return indexed
