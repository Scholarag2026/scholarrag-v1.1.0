# backend/tests/test_wos_lookup.py
import pytest

from app.services.verification import _make_wos_check
from app.services.wos_lookup import import_wos_csv, is_wos_indexed


@pytest.mark.asyncio
async def test_is_wos_indexed_no_issn(db_session):
    """Returns None when no ISSN provided."""
    result = await is_wos_indexed(db_session, issn=None, eissn=None)
    assert result is None


@pytest.mark.asyncio
async def test_is_wos_indexed_empty_table(db_session):
    """Returns None when no WoS data loaded."""
    result = await is_wos_indexed(db_session, issn="0028-0836")
    assert result is None


@pytest.mark.asyncio
async def test_import_wos_csv(db_session):
    """Imports CSV and can look up journals."""
    csv_content = (
        "Journal Title,ISSN,eISSN\n"
        "Nature,0028-0836,1476-4687\n"
        "Science,0036-8075,1095-9203\n"
        "Cell,0092-8674,1097-4172\n"
    )
    count = await import_wos_csv(db_session, csv_content, "SCIE")
    assert count == 3

    # Should find Nature by ISSN
    result = await is_wos_indexed(db_session, issn="0028-0836")
    assert result is True

    # Should find Nature by eISSN
    result = await is_wos_indexed(db_session, issn=None, eissn="1476-4687")
    assert result is True

    # Should find Science by ISSN (lowercase normalized)
    result = await is_wos_indexed(db_session, issn="0036-8075")
    assert result is True

    # Should NOT find a random ISSN
    result = await is_wos_indexed(db_session, issn="9999-9999")
    assert result is False

    # Should find when ISSN matches eISSN column
    result = await is_wos_indexed(db_session, issn="1476-4687")
    assert result is True


@pytest.mark.asyncio
async def test_import_wos_csv_replaces_existing(db_session):
    """Reimporting CSV replaces all existing data."""
    csv1 = "Journal Title,ISSN,eISSN\nNature,0028-0836,1476-4687\n"
    csv2 = "Journal Title,ISSN,eISSN\nScience,0036-8075,1095-9203\n"

    count1 = await import_wos_csv(db_session, csv1, "SCIE")
    assert count1 == 1
    assert await is_wos_indexed(db_session, issn="0028-0836") is True

    count2 = await import_wos_csv(db_session, csv2, "SCIE")
    assert count2 == 1
    # Nature should be gone
    assert await is_wos_indexed(db_session, issn="0028-0836") is False
    # Science should be present
    assert await is_wos_indexed(db_session, issn="0036-8075") is True


def test_wos_check_indexed_is_info():
    """WoS indexing is an indicator: ``info`` when the journal is listed."""
    check = _make_wos_check(True)
    assert check.check_type == "wos_indexed"
    assert check.status == "info"  # indicator, never affects overall_status
    assert "indexed" in check.message.lower()


def test_wos_check_not_indexed_is_note():
    """WoS indexing is an indicator: ``note`` when the journal is not listed."""
    check = _make_wos_check(False)
    assert check.check_type == "wos_indexed"
    assert check.status == "note"  # indicator, never affects overall_status
    assert "not found" in check.message.lower()


def test_wos_check_skipped():
    """Verification check returns skipped when None."""
    check = _make_wos_check(None)
    assert check.check_type == "wos_indexed"
    assert check.status == "skipped"


def test_detect_collection_all_canonical_filenames():
    """The parenthesized tag wins — 'SCIE' inside 'SCIENCES' must not shadow SSCI."""
    from app.services.wos_import import _detect_collection

    assert _detect_collection("Science Citation Index Expanded (SCIE).csv") == "SCIE"
    assert _detect_collection("Social Sciences Citation Index (SSCI).csv") == "SSCI"
    assert _detect_collection("Arts & Humanities Citation Index (AHCI).csv") == "AHCI"
    assert _detect_collection("Emerging Sources Citation Index (ESCI).csv") == "ESCI"
    assert _detect_collection("some other file.csv") == "WOS"
