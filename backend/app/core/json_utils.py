"""JSON helpers that survive numpy scalars and non-finite floats.

Analysis results (descriptive statistics, kappa reports, agent output) are written to
JSONB columns. numpy scalars are not JSON-serialisable at all, and bare ``NaN`` is
accepted by ``json.dumps`` but rejected by Postgres' strict RFC-8259 parser. Both
failure modes are killed here, at the engine level.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars and non-finite floats into JSON-safe values.

    - ``np.integer`` -> ``int``
    - ``np.floating`` -> ``float`` (NaN/Inf -> ``None``)
    - ``np.bool_`` -> ``bool``
    - ``np.ndarray`` -> ``list``
    - ``float`` NaN/Inf -> ``None``
    - dict/list/tuple/set -> converted element-wise

    Anything else is returned unchanged.
    """
    if value is None or isinstance(value, (str, bytes, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value


def json_serializer(obj: Any) -> str:
    """SQLAlchemy ``json_serializer``: numpy-aware and never emits ``NaN``/``Infinity``."""
    return json.dumps(to_jsonable(obj), allow_nan=False, ensure_ascii=False)
