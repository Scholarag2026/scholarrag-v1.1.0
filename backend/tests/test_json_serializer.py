"""The engine-level JSON serializer must survive numpy scalars and non-finite floats."""

import json

import numpy as np
import pytest


def test_to_jsonable_converts_numpy_bool():
    from app.core.json_utils import to_jsonable

    out = to_jsonable({"is_normal": np.bool_(True)})
    assert out == {"is_normal": True}
    assert isinstance(out["is_normal"], bool)


def test_to_jsonable_converts_numpy_int_and_float():
    from app.core.json_utils import to_jsonable

    out = to_jsonable({"i": np.int64(3), "f": np.float64(1.5)})
    assert out == {"i": 3, "f": 1.5}
    assert isinstance(out["i"], int)
    assert isinstance(out["f"], float)


def test_to_jsonable_maps_non_finite_to_none():
    from app.core.json_utils import to_jsonable

    assert to_jsonable(float("nan")) is None
    assert to_jsonable(float("inf")) is None
    assert to_jsonable(float("-inf")) is None
    assert to_jsonable(np.float64("nan")) is None
    assert to_jsonable({"k": [float("nan"), 1.0]}) == {"k": [None, 1.0]}


def test_to_jsonable_converts_ndarray():
    from app.core.json_utils import to_jsonable

    assert to_jsonable(np.array([1, 2, 3])) == [1, 2, 3]


def test_to_jsonable_leaves_plain_values_alone():
    from app.core.json_utils import to_jsonable

    payload = {"a": [1, "x", None, True], "b": {"c": 2.5}}
    assert to_jsonable(payload) == payload


def test_json_serializer_emits_rfc8259_json():
    from app.core.json_utils import json_serializer

    payload = {"is_normal": np.bool_(False), "kappa": float("nan"), "n": np.int64(7)}
    text = json_serializer(payload)
    assert "NaN" not in text
    assert json.loads(text) == {"is_normal": False, "kappa": None, "n": 7}


def test_engine_uses_numpy_aware_json_serializer():
    from app.core.json_utils import json_serializer
    from app.database import engine

    assert engine.dialect._json_serializer is json_serializer


@pytest.mark.asyncio
async def test_postgres_accepts_serializer_output(db_session):
    """Postgres rejects bare NaN in jsonb — the serializer output must be accepted."""
    from sqlalchemy import text

    from app.core.json_utils import json_serializer

    payload = json_serializer({"is_normal": np.bool_(True), "kappa": float("nan")})
    await db_session.execute(text("CREATE TEMP TABLE t5_json_probe (payload jsonb)"))
    await db_session.execute(
        text("INSERT INTO t5_json_probe (payload) VALUES (CAST(:p AS jsonb))"),
        {"p": payload},
    )
    row = (await db_session.execute(text("SELECT payload FROM t5_json_probe"))).scalar_one()
    data = json.loads(row) if isinstance(row, str) else row
    assert data == {"is_normal": True, "kappa": None}
