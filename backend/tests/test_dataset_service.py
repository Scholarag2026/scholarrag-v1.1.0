import os
import pytest
import pandas as pd

from app.services.dataset import parse_csv, parse_xlsx, detect_column_types, get_preview_rows


@pytest.fixture
def sample_csv_path():
    return os.path.join(os.path.dirname(__file__), "fixtures", "sample_data.csv")


@pytest.fixture
def sample_text_path():
    return os.path.join(os.path.dirname(__file__), "fixtures", "sample_text.csv")


def test_parse_csv(sample_csv_path):
    df = parse_csv(sample_csv_path)
    assert len(df) == 10
    assert "age" in df.columns
    assert "gender" in df.columns


def test_detect_column_types(sample_csv_path):
    df = parse_csv(sample_csv_path)
    col_info = detect_column_types(df)
    names = {c["name"]: c["dtype"] for c in col_info}
    assert names["age"] == "numeric"
    assert names["gender"] == "categorical"
    assert names["score"] == "numeric"
    assert names["satisfaction"] == "categorical"


def test_detect_text_columns(sample_text_path):
    df = parse_csv(sample_text_path)
    col_info = detect_column_types(df)
    names = {c["name"]: c["dtype"] for c in col_info}
    assert names["response"] == "text"


def test_get_preview_rows(sample_csv_path):
    df = parse_csv(sample_csv_path)
    preview = get_preview_rows(df, limit=5)
    assert len(preview) == 5
    assert "age" in preview[0]


def test_detect_missing_counts(sample_csv_path):
    df = parse_csv(sample_csv_path)
    col_info = detect_column_types(df)
    for col in col_info:
        assert "missing_count" in col
        assert "unique_count" in col
