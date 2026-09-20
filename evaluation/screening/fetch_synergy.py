"""Fetch one SYNERGY screening dataset and write ``evaluation/screening/data/<id>.jsonl``.

Interpreter: ``evaluation/.venv/Scripts/python`` or the system ``python`` (needs ``httpx``).

Routes
------
``v1``  Download the labelled CSV listed in ``data/synergy_index.csv`` (asreview
        systematic-review-datasets, tag ``metadata-v1-final``; columns ``record_id, title,
        abstract, label_included, label_abstract_screening, duplicate_record_id``).
``v2``  Download ``<name>_ids.csv`` from asreview/synergy-dataset (columns ``doi,
        openalex_id, label_included[, pmid, method]``) and hydrate title / abstract / ISSN /
        year from OpenAlex in batches of 50 (<= 4 requests per second, retries with back-off).
        Some datasets (e.g. Nagtegaal_2019) have no identifiers in v2 -- use ``v1`` for them.
``plus``  SYNERGY+ v3.0 release datasets not (yet) in the ``v1``/``v2`` indices. Reads the
        local read-only cache at ``--cache-dir/<dataset>/``
        first (``~/.synergy_dataset_source/synergy-dataset-plus/<id>/`` by default):
        ``metadata.json`` for ``publication.doi`` / ``publication.eligibility_criteria`` /
        ``data.n_records`` / ``data.n_records_included`` (catalogue figures only, never the
        counted n); ``labels.csv`` (columns ``openalex_id, doi, pmid, lens_id,
        label_included, label_abstract_included``) if cached, else downloaded from the
        pinned URL below; a ``works_*.zip`` (each entry a JSON list of OpenAlex work
        objects) if cached, else title/abstract/issn/year are hydrated by ``openalex_id``
        with the existing ``hydrate_by_ids``. ``label_abstract_included`` is renamed to the
        existing schema key ``label_abstract_screening``; ``label_included`` is kept.
        ``record_id`` is the 1-based ``labels.csv`` row number. The pinned raw URL, verified
        2026-09-06 by content-diffing the downloaded ``Fong_2021/labels.csv`` byte for byte
        (modulo line endings) against the read-only local cache copy, and by row/positive
        counts (``Anmarkrud_2021``: 2,284 rows, 64 ``label_included`` positives, matching
        ``metadata.json``'s catalogue figures exactly): the ``releases/synergy_plus_v3.0``
        branch of the same ``asreview/synergy-dataset`` GitHub repository the ``v1``/``v2``
        routes already use (not a separate "synergy-dataset-plus" repository, which does not
        exist; the ``data.doi`` OSF deposit in ``metadata.json`` hosts the collection's raw
        per-review exports, not this processed SYNERGY row format, so it is recorded for
        provenance but not used as a download source).

``--hydrate included|all`` (route ``v1`` only) looks records up so that ``openalex_id`` /
``doi`` / ``issn`` / ``year`` become available for the WoS gate-effect analysis.
``included`` targets every record with *any* positive label (``label_included == 1`` or
``label_abstract_screening == 1``), i.e. the human-included set of both protocol label
choices. Lookups go first by DOI for records whose CSV carries one, then by exact
normalised-title match (top 5 hits) for the rest, which must also lie within +-1 year of
the export's year when the export carries one; the method is stored per record in
``match_method`` (``"doi"`` / ``"title_exact_year"``, or ``"title_exact"`` when the export
has no year to check, e.g. Nagtegaal_2019) and the service in ``lookup_source``.
``--lookup openalex`` (default; ``works?filter=doi:...`` in batches of 50, ``title.search``)
or ``--lookup crossref`` (``/works/<doi>``, ``query.bibliographic``; used when the shared-IP
OpenAlex daily budget is spent -- Crossref carries ISSNs but no OpenAlex id). Hydration never
changes title or abstract text: the screener sees exactly the SYNERGY export, as the human
screeners did.

Besides ``<id>.jsonl`` the script writes ``<id>.fetch.json`` with the fetch provenance and
counts (``n, n_included, n_abstract_screening_included, n_missing_abstract,
share_missing_abstract, n_missing_title, n_with_openalex_id, n_with_issn, n_with_doi,
hydrate, hydrate_lookup, n_hydrate_targets, n_targets_with_issn``); ``summarize.py`` reads
it for the missing-abstract share and the hydration mode.

Output record schema (one JSON object per line)::

    record_id (int), title (str), abstract (str), label_included (0/1),
    label_abstract_screening (0/1 or null), openalex_id (str|null), doi (str|null,
    lower-case, no prefix), issn (list[str]), year (int|null), source_route ("v1"|"v2"),
    match_method ("openalex_id"|"doi"|"title_exact"|"title_exact_year"|null), lookup_source
    ("openalex"|"crossref"|null)

The dataverse.nl download used by the ``synergy-dataset`` CLI is not used (too slow from
the authors' network); everything comes from GitHub raw files and the OpenAlex API.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import REPO_ROOT, append_jsonl, load_dotenv_values, now_iso, write_json  # noqa: E402

DATA_DIR = HERE / "data"
INDEX_URL = (
    "https://raw.githubusercontent.com/asreview/systematic-review-datasets/"
    "metadata-v1-final/index.csv"
)
V2_IDS_URL = (
    "https://raw.githubusercontent.com/asreview/synergy-dataset/master/datasets/"
    "{name}/{name}_ids.csv"
)
#: SYNERGY+ v3.0 release branch of the same repository (verified 2026-09-06; see the module
#: docstring's ``plus`` route section for how this was confirmed).
PLUS_BRANCH = "releases/synergy_plus_v3.0"
PLUS_LABELS_URL = (
    f"https://raw.githubusercontent.com/asreview/synergy-dataset/{PLUS_BRANCH}/"
    "{name}/labels.csv"
)
DEFAULT_PLUS_CACHE_DIR = Path.home() / ".synergy_dataset_source" / "synergy-dataset-plus"
PLUS_LABEL_COLUMN_MAPPING: dict[str, str] = {
    "label_abstract_included": "label_abstract_screening"
}
OPENALEX_WORKS = "https://api.openalex.org/works"
CROSSREF_WORKS = "https://api.crossref.org/works"
LOOKUP_SOURCES: tuple[str, ...] = ("openalex", "crossref")
OPENALEX_API_KEY: str | None = None  # set by main() from --api-key / OPENALEX_API_KEY in .env
# is_paratext is fetched
# beside the type, free (OpenAlex bills nothing for extra select
# fields), so a re-fetched route v2 export carries both and
# app.agents.relevance_screener_agent.apply_type_demotion can be applied downstream without
# a second OpenAlex request.
OPENALEX_SELECT = (
    "id,doi,title,publication_year,type,is_paratext,abstract_inverted_index,primary_location"
)
MAX_REQUESTS_PER_SECOND = 4.0
MAX_RETRY_DELAY_S = 120.0  # OpenAlex may send Retry-After of ~1 day when the daily quota is spent
OPENALEX_BATCH = 50
OPENALEX_DOI_BATCH = 50  # probed 2026-09-02 (see report); fall back to 25 if rejected


# --------------------------------------------------------------------------------------
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------------------


def to_int_label(value: Any) -> int | None:
    """``'1'``/``1``/``True`` -> 1, ``'0'``/``0`` -> 0, blank -> None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    s = str(value).strip()
    if s == "":
        return None
    try:
        return 1 if int(float(s)) == 1 else 0
    except ValueError:
        return None


def to_year(value: Any) -> int | None:
    """``'2010'`` / ``'2010.0'`` -> 2010; blank or non-numeric -> None."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        year = int(float(s))
    except ValueError:
        return None
    return year if 1000 <= year <= 2100 else None


def parse_v1_csv(text: str) -> list[dict[str, Any]]:
    """Rows of a SYNERGY v1 CSV as output records.

    ``doi`` and ``year`` are kept when the CSV carries those columns (some v1 exports, e.g.
    van_de_Schoot_2017, do; Nagtegaal_2019 does not); ``openalex_id`` / ``issn`` stay empty
    until ``--hydrate``.
    """
    records: list[dict[str, Any]] = []
    for i, row in enumerate(csv.DictReader(io.StringIO(text)), start=1):
        rid = row.get("record_id")
        try:
            record_id = int(rid) if rid not in (None, "") else i
        except ValueError:
            record_id = i
        records.append(
            {
                "record_id": record_id,
                "title": (row.get("title") or "").strip(),
                "abstract": (row.get("abstract") or "").strip(),
                "label_included": to_int_label(row.get("label_included")) or 0,
                "label_abstract_screening": to_int_label(row.get("label_abstract_screening")),
                "openalex_id": None,
                "doi": normalise_doi(row.get("doi")),
                "issn": [],
                "year": to_year(row.get("year")),
                "source_route": "v1",
                "match_method": None,
            }
        )
    return records


def parse_ids_csv(text: str) -> list[dict[str, Any]]:
    """Rows of a SYNERGY v2 ``<name>_ids.csv`` (record_id = 1-based row number)."""
    records: list[dict[str, Any]] = []
    for i, row in enumerate(csv.DictReader(io.StringIO(text)), start=1):
        records.append(
            {
                "record_id": i,
                "title": "",
                "abstract": "",
                "label_included": to_int_label(row.get("label_included")) or 0,
                "label_abstract_screening": to_int_label(row.get("label_abstract_screening")),
                "openalex_id": short_openalex_id(row.get("openalex_id")),
                "doi": normalise_doi(row.get("doi")),
                "issn": [],
                "year": None,
                "source_route": "v2",
                "match_method": None,
                "pmid": (row.get("pmid") or "").strip() or None,
                "method": (row.get("method") or "").strip() or None,
            }
        )
    return records


def dedupe_v2_records(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """One record per distinct ``openalex_id`` (SYNERGY's own unit of analysis).

    The v2 ``<name>_ids.csv`` files can list the same work several times (found by DOI and
    by title search) and rows without any identifier; those rows cannot be hydrated and are
    dropped. Duplicates keep the first row's ``record_id`` and the maximum label.
    """
    kept: dict[str, dict[str, Any]] = {}
    dropped = duplicates = 0
    for rec in records:
        oid = rec.get("openalex_id")
        if not oid:
            dropped += 1
            continue
        if oid in kept:
            duplicates += 1
            first = kept[oid]
            first["label_included"] = max(first["label_included"], rec["label_included"])
            if rec.get("label_abstract_screening") is not None:
                first["label_abstract_screening"] = max(
                    first.get("label_abstract_screening") or 0, rec["label_abstract_screening"]
                )
            for key in ("doi", "pmid"):
                if not first.get(key) and rec.get(key):
                    first[key] = rec[key]
            continue
        kept[oid] = rec
    info = {
        "n_rows_in_ids_csv": len(records),
        "n_dropped_no_openalex_id": dropped,
        "n_duplicate_ids": duplicates,
    }
    return list(kept.values()), info


def parse_plus_labels(text: str) -> list[dict[str, Any]]:
    """Rows of a SYNERGY+ ``labels.csv`` (``record_id`` = 1-based row number).

    Columns: ``openalex_id, doi, pmid, lens_id, label_included, label_abstract_included``
    (verified on the cached ``Fong_2021`` and ``Taschner_2024``
    files). ``label_abstract_included`` is renamed to the existing schema key
    ``label_abstract_screening`` here, at parse time, so every downstream consumer of a
    "plus" record sees the same key every other route uses; ``label_included`` keeps its
    name. There is no title/abstract text in this file (rule 3: those come from a cached
    ``works_*.zip`` or OpenAlex hydration by id).
    """
    records: list[dict[str, Any]] = []
    for i, row in enumerate(csv.DictReader(io.StringIO(text)), start=1):
        records.append(
            {
                "record_id": i,
                "title": "",
                "abstract": "",
                "label_included": to_int_label(row.get("label_included")) or 0,
                "label_abstract_screening": to_int_label(row.get("label_abstract_included")),
                "openalex_id": short_openalex_id(row.get("openalex_id")),
                "doi": normalise_doi(row.get("doi")),
                "issn": [],
                "year": None,
                "source_route": "plus",
                "match_method": None,
                "pmid": (row.get("pmid") or "").strip() or None,
                "lens_id": (row.get("lens_id") or "").strip() or None,
            }
        )
    return records


def load_plus_works(zip_path: str | Path) -> dict[str, dict[str, Any]]:
    """``{short_openalex_id: work_to_fields(work)}`` merged from every JSON entry of a
    SYNERGY+ ``works_*.zip`` (each entry a JSON list of OpenAlex work objects, e.g.
    ``works_0_20.json``; verified on the cached ``Taschner_2024/works_1.zip``, whose 192
    entries cover its 3,828 ``labels.csv`` rows one to one)."""
    out: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            with zf.open(name) as fh:
                works = json.loads(fh.read().decode("utf-8"))
            for work in works:
                fields = work_to_fields(work)
                if fields["openalex_id"]:
                    out[fields["openalex_id"]] = fields
    return out


def apply_plus_works(
    records: list[dict[str, Any]], works: Mapping[str, Mapping[str, Any]]
) -> int:
    """Fill title/abstract/issn/year (and doi when the record has none) from a works lookup
    built by :func:`load_plus_works`. Returns the number of records matched."""
    hits = 0
    for rec in records:
        work = works.get(rec.get("openalex_id") or "")
        if work is None:
            continue
        hits += 1
        rec.update(
            title=work["title"], abstract=work["abstract"], issn=work["issn"],
            year=work["year"], match_method="openalex_id",
        )
        if not rec.get("doi"):
            rec["doi"] = work["doi"]
    return hits


def plus_metadata(cache_dir: Path, dataset_id: str) -> dict[str, Any]:
    """``publication``/``data`` fields of ``<cache_dir>/<dataset_id>/metadata.json`` this
    route needs, read from the local read-only cache (never downloaded: the brief assumes
    it is always present for a "plus" dataset)."""
    path = cache_dir / dataset_id / "metadata.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    publication = raw.get("publication") or {}
    data = raw.get("data") or {}
    return {
        "publication_doi": normalise_doi(publication.get("doi")),
        "eligibility_criteria": publication.get("eligibility_criteria"),
        "catalogue_n_records": data.get("n_records"),
        "catalogue_n_records_included": data.get("n_records_included"),
    }


def short_openalex_id(value: str | None) -> str | None:
    """``https://openalex.org/W123`` -> ``W123``."""
    if not value:
        return None
    s = value.strip()
    if not s:
        return None
    return s.rsplit("/", 1)[-1].upper()


DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


def is_valid_doi(value: str | None) -> bool:
    """``10.<registrant>/<suffix>`` with no whitespace; export cells such as
    ``10.1037.a0037593`` or ``10.1159/x; 10.1159/x`` (two DOIs in one cell) are not."""
    if not value:
        return False
    s = normalise_doi(value) or ""
    return DOI_RE.match(s) is not None


def normalise_doi(value: str | None) -> str | None:
    if not value:
        return None
    s = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    return s or None


def reconstruct_abstract(inverted_index: Mapping[str, Sequence[int]] | None) -> str:
    """Rebuild the abstract text from OpenAlex's ``abstract_inverted_index``."""
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions.append((idx, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def work_to_fields(work: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the fields we keep from one OpenAlex work object."""
    source = ((work.get("primary_location") or {}).get("source") or {}) or {}
    issns: list[str] = []
    for candidate in [source.get("issn_l"), *(source.get("issn") or [])]:
        if candidate and candidate not in issns:
            issns.append(candidate)
    return {
        "openalex_id": short_openalex_id(work.get("id")),
        "doi": normalise_doi(work.get("doi")),
        "title": (work.get("title") or "").strip(),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "issn": issns,
        "year": work.get("publication_year"),
        "type": work.get("type"),
        "is_paratext": bool(work.get("is_paratext") or False),
        "source_display_name": source.get("display_name"),
    }


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise_title(title: str | None) -> str:
    return _NON_ALNUM.sub(" ", (title or "").lower()).strip()


def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def hydration_targets(records: Sequence[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """Records to look up for ``--hydrate``: ``all``, or ``included`` = any positive label."""
    if mode == "all":
        return list(records)
    if mode == "included":
        return [
            r for r in records
            if r.get("label_included") == 1 or r.get("label_abstract_screening") == 1
        ]
    return []


def dataset_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "n": len(records),
        "n_included": sum(1 for r in records if r.get("label_included") == 1),
        "n_abstract_screening_included": sum(
            1 for r in records if r.get("label_abstract_screening") == 1
        ),
        "n_missing_abstract": sum(1 for r in records if not (r.get("abstract") or "").strip()),
        "n_missing_title": sum(1 for r in records if not (r.get("title") or "").strip()),
        "n_with_openalex_id": sum(1 for r in records if r.get("openalex_id")),
        "n_with_issn": sum(1 for r in records if r.get("issn")),
        "n_with_doi": sum(1 for r in records if r.get("doi")),
    }


# --------------------------------------------------------------------------------------
# Network helpers
# --------------------------------------------------------------------------------------


class RateLimiter:
    """Simple minimum-interval limiter (<= ``per_second`` request starts per second)."""

    def __init__(self, per_second: float) -> None:
        self.interval = 1.0 / per_second
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delay = self._last + self.interval - now
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()


def fetch_with_retries(
    client: httpx.Client,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    limiter: RateLimiter | None = None,
    attempts: int = 4,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(attempts):
        if limiter:
            limiter.wait()
        try:
            resp = client.get(url, params=params)
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 2.0**attempt
                if delay > MAX_RETRY_DELAY_S:
                    print(
                        f"  HTTP {resp.status_code}; server asks to wait {delay:.0f}s "
                        f"(capped at {MAX_RETRY_DELAY_S:.0f}s; daily quota spent?)",
                        file=sys.stderr,
                    )
                    delay = MAX_RETRY_DELAY_S
                else:
                    print(f"  HTTP {resp.status_code}; retrying in {delay:.0f}s", file=sys.stderr)
                time.sleep(delay)
                last_error = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
                continue
            resp.raise_for_status()
            return resp
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = exc
            time.sleep(2.0**attempt)
    raise RuntimeError(f"giving up on {url}: {last_error}")


def _safe_print(text: str) -> None:
    """``print`` that never raises on a console codec that cannot encode a character (e.g.
    the Windows ``cp936`` console meeting a non-breaking space in a ``eligibility_criteria``
    string copied verbatim from a source paper's metadata); such characters are replaced,
    not the whole run aborted after the output files were already written."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def ensure_index(path: Path, client: httpx.Client) -> Path:
    if not path.exists():
        print(f"index not found at {path}; downloading {INDEX_URL}")
        resp = fetch_with_retries(client, INDEX_URL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(resp.text, encoding="utf-8")
    return path


def index_row(index_path: Path, dataset_id: str) -> dict[str, str]:
    with index_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("dataset_id") == dataset_id:
                return row
    raise SystemExit(f"dataset {dataset_id!r} not found in {index_path}")


def openalex_params(email: str | None, api_key: str | None = None, **extra: Any) -> dict[str, Any]:
    params: dict[str, Any] = {"per-page": OPENALEX_BATCH, "select": OPENALEX_SELECT}
    if email:
        params["mailto"] = email
    if api_key:
        params["api_key"] = api_key
    params.update(extra)
    return params


def hydrate_by_ids(
    records: list[dict[str, Any]], client: httpx.Client, email: str | None, limiter: RateLimiter
) -> int:
    """Fill title/abstract/issn/year for records that carry an ``openalex_id``. Returns hits."""
    with_ids = [r for r in records if r.get("openalex_id")]
    hits = 0
    for n_done, batch in enumerate(chunked(with_ids, OPENALEX_BATCH), start=1):
        ids = "|".join(r["openalex_id"] for r in batch)
        resp = fetch_with_retries(
            client,
            OPENALEX_WORKS,
            params=openalex_params(email, OPENALEX_API_KEY, filter=f"openalex_id:{ids}"),
            limiter=limiter,
        )
        found = {w["openalex_id"]: w for w in (work_to_fields(x) for x in resp.json()["results"])}
        for rec in batch:
            work = found.get(rec["openalex_id"])
            if work is None:
                continue
            hits += 1
            rec.update(
                title=work["title"],
                abstract=work["abstract"],
                issn=work["issn"],
                year=work["year"],
                match_method="openalex_id",
            )
            if not rec.get("doi"):
                rec["doi"] = work["doi"]
        n_seen = min(n_done * OPENALEX_BATCH, len(with_ids))
        print(f"  hydrated batch {n_done} ({n_seen}/{len(with_ids)})")
    return hits


def hydrate_by_doi(
    records: list[dict[str, Any]], client: httpx.Client, email: str | None, limiter: RateLimiter
) -> int:
    """Attach ``openalex_id`` / ``issn`` / ``year`` to records that carry a DOI but no
    ``openalex_id`` (``works?filter=doi:<a>|<b>...`` in batches of ``OPENALEX_DOI_BATCH``).

    Matching is on the normalised DOI; unmatched records are left untouched. Title and
    abstract text are never modified.
    """
    targets = [r for r in records if r.get("doi") and not r.get("openalex_id")]
    hits = 0
    for n_done, batch in enumerate(chunked(targets, OPENALEX_DOI_BATCH), start=1):
        dois = "|".join(r["doi"] for r in batch)
        resp = fetch_with_retries(
            client,
            OPENALEX_WORKS,
            params=openalex_params(email, OPENALEX_API_KEY, filter=f"doi:{dois}"),
            limiter=limiter,
        )
        found: dict[str, dict[str, Any]] = {}
        for work in (work_to_fields(x) for x in resp.json().get("results", [])):
            if work["doi"] and work["doi"] not in found:
                found[work["doi"]] = work
        for rec in batch:
            work = found.get(rec["doi"])
            if work is None:
                continue
            hits += 1
            rec.update(
                openalex_id=work["openalex_id"],
                issn=work["issn"],
                year=rec.get("year") or work["year"],
                match_method="doi",
                lookup_source="openalex",
            )
        n_seen = min(n_done * OPENALEX_DOI_BATCH, len(targets))
        print(f"  doi lookups {n_seen}/{len(targets)} (matched {hits})")
    return hits


_TITLE_QUERY_STRIP = re.compile(r"[^\w\s]+", re.UNICODE)


def title_query(title: str) -> str:
    """Title as an OpenAlex ``title.search`` value: punctuation removed, <= 300 chars."""
    return " ".join(_TITLE_QUERY_STRIP.sub(" ", title).split())[:300]


def hydrate_by_title(
    records: list[dict[str, Any]], client: httpx.Client, email: str | None, limiter: RateLimiter
) -> int:
    """Attach identifiers to v1 records via exact normalised-title match in OpenAlex."""
    hits = 0
    for i, rec in enumerate(records, start=1):
        title = rec.get("title") or ""
        if rec.get("openalex_id") or len(normalise_title(title)) < 15:
            continue
        safe = title_query(title)
        try:
            resp = fetch_with_retries(
                client,
                OPENALEX_WORKS,
                params=openalex_params(
                    email, OPENALEX_API_KEY, filter=f"title.search:{safe}", **{"per-page": 5}
                ),
                limiter=limiter,
            )
        except (RuntimeError, httpx.HTTPError) as exc:
            rid = rec.get("record_id")
            print(f"  title lookup failed for record {rid}: {exc}", file=sys.stderr)
            continue
        target = normalise_title(title)
        for work in (work_to_fields(x) for x in resp.json().get("results", [])):
            if normalise_title(work["title"]) == target:
                hits += 1
                rec.update(
                    openalex_id=work["openalex_id"],
                    doi=work["doi"],
                    issn=work["issn"],
                    year=work["year"],
                    match_method="title_exact",
                    lookup_source="openalex",
                )
                break
        if i % 50 == 0:
            print(f"  title lookups {i}/{len(records)} (matched {hits})")
    return hits


def crossref_work_to_fields(message: Mapping[str, Any]) -> dict[str, Any]:
    """Crossref ``message`` -> ``{doi, title, issn, year, container_title}`` (no OpenAlex id,
    no abstract)."""
    titles = message.get("title") or []
    containers = message.get("container-title") or []
    parts = ((message.get("issued") or {}).get("date-parts") or [[None]])[0]
    return {
        "doi": normalise_doi(message.get("DOI")),
        "title": str(titles[0]) if titles else "",
        "issn": [n for n in (str(x) for x in message.get("ISSN") or []) if n],
        "year": to_year(parts[0] if parts else None),
        "container_title": str(containers[0]) if containers else "",
    }


# Abstracting series that re-publish articles under their original titles (Elsevier
# "Yearbook of ..." volumes, e.g. Yearbook of Psychiatry and Applied Mental Health, ISSN
# 0084-3970). The 2026-09-03 spot check of 40 title-only matches found 3 such hits; a match
# on them would attribute the yearbook's ISSN instead of the journal the screeners saw.
REPRINT_VENUE_RE = re.compile(r"^\s*year\s*book\b", re.IGNORECASE)


def is_reprint_venue(container_title: str | None) -> bool:
    return bool(container_title) and REPRINT_VENUE_RE.match(container_title) is not None


def crossref_params(email: str | None, **extra: Any) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if email:
        params["mailto"] = email
    params.update(extra)
    return params


def hydrate_by_doi_crossref(
    records: list[dict[str, Any]], client: httpx.Client, email: str | None, limiter: RateLimiter
) -> int:
    """Attach ``issn`` / ``year`` from Crossref ``/works/<doi>`` to records with a DOI but no
    ISSN. Unmatched (HTTP 404 / errors) records are left untouched. Returns hits."""
    targets = [r for r in records if r.get("doi") and not r.get("issn")]
    hits = 0
    for i, rec in enumerate(targets, start=1):
        try:
            resp = fetch_with_retries(
                client, f"{CROSSREF_WORKS}/{rec['doi']}", params=crossref_params(email),
                limiter=limiter,
            )
        except (RuntimeError, httpx.HTTPError) as exc:
            print(f"  crossref doi lookup failed for record {rec.get('record_id')}: {exc}",
                  file=sys.stderr)
            continue
        work = crossref_work_to_fields(resp.json().get("message") or {})
        if not work["issn"]:
            continue
        hits += 1
        rec.update(
            issn=work["issn"],
            year=rec.get("year") or work["year"],
            match_method="doi",
            lookup_source="crossref",
        )
        if i % 50 == 0:
            print(f"  crossref doi lookups {i}/{len(targets)} (matched {hits})")
    return hits


def hydrate_by_title_crossref(
    records: list[dict[str, Any]], client: httpx.Client, email: str | None, limiter: RateLimiter
) -> int:
    """Exact normalised-title match among the top 5 ``query.bibliographic`` hits.

    A title-only match carries a same-title risk (comments, reprints, conference abstracts
    that repeat a title). When the record carries a ``year`` the hit must be within +-1 year
    of it (``match_method = "title_exact_year"``); records without a year cannot be
    year-checked and are matched on the title alone (``match_method = "title_exact"``).
    Hits in re-publishing series (:func:`is_reprint_venue`) are skipped. A malformed export
    DOI (:func:`is_valid_doi` false) is replaced by the matched record's DOI and kept as
    ``doi_export``.
    """
    hits = 0
    for i, rec in enumerate(records, start=1):
        title = rec.get("title") or ""
        if rec.get("issn") or len(normalise_title(title)) < 15:
            continue
        rec_year = to_year(rec.get("year"))
        try:
            resp = fetch_with_retries(
                client,
                CROSSREF_WORKS,
                params=crossref_params(
                    email, **{"query.bibliographic": title_query(title), "rows": 5,
                              "select": "DOI,title,ISSN,issued,container-title"}
                ),
                limiter=limiter,
            )
        except (RuntimeError, httpx.HTTPError) as exc:
            print(f"  crossref title lookup failed for record {rec.get('record_id')}: {exc}",
                  file=sys.stderr)
            continue
        target = normalise_title(title)
        items = (resp.json().get("message") or {}).get("items") or []
        for work in (crossref_work_to_fields(x) for x in items):
            if normalise_title(work["title"]) != target or not work["issn"]:
                continue
            if is_reprint_venue(work["container_title"]):
                continue  # yearbook reprint of the article: not the screeners' venue
            year_checked = rec_year is not None and work["year"] is not None
            if year_checked and abs(rec_year - work["year"]) > 1:
                continue  # same title, different year: do not attribute this venue
            hits += 1
            export_doi = rec.get("doi")
            if export_doi and not is_valid_doi(export_doi) and work["doi"]:
                # a malformed export DOI (e.g. "10.1037.a0037593") could not be looked up;
                # keep it for provenance and carry the DOI of the title-matched record
                rec["doi_export"] = export_doi
                export_doi = None
            rec.update(
                doi=export_doi or work["doi"],
                issn=work["issn"],
                year=rec.get("year") or work["year"],
                match_method="title_exact_year" if year_checked else "title_exact",
                lookup_source="crossref",
            )
            break
        if i % 50 == 0:
            print(f"  crossref title lookups {i}/{len(records)} (matched {hits})")
    return hits


def fetch_info(
    dataset_id: str,
    route: str,
    source_url: str,
    hydrate: str,
    counts: Mapping[str, int],
    limit: int | None = None,
    *,
    hydrate_lookup: str | None = None,
    n_hydrate_targets: int | None = None,
    n_targets_with_issn: int | None = None,
    label_column_mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Contents of ``<dataset>.fetch.json`` (provenance + counts; read by summarize.py).

    ``label_column_mapping`` is set only by ``--route plus``:
    the source column name renamed to the existing schema key it is written under.
    """
    n = counts["n"]
    info: dict[str, Any] = {
        "dataset_id": dataset_id,
        "route": route,
        "source_url": source_url,
        "fetched_at": now_iso(),
        "limit": limit,
        "n": n,
        "n_included": counts["n_included"],
        "n_abstract_screening_included": counts["n_abstract_screening_included"],
        "n_missing_abstract": counts["n_missing_abstract"],
        "share_missing_abstract": (counts["n_missing_abstract"] / n) if n else None,
        "n_missing_title": counts["n_missing_title"],
        "n_with_openalex_id": counts["n_with_openalex_id"],
        "n_with_issn": counts["n_with_issn"],
        "n_with_doi": counts["n_with_doi"],
        "hydrate": hydrate,
        "hydrate_lookup": hydrate_lookup,
        "n_hydrate_targets": n_hydrate_targets,
        "n_targets_with_issn": n_targets_with_issn,
    }
    if label_column_mapping is not None:
        info["label_column_mapping"] = dict(label_column_mapping)
    return info


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True, help="SYNERGY dataset id, e.g. Nagtegaal_2019")
    ap.add_argument("--route", choices=("v1", "v2", "plus"), default="v1")
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_PLUS_CACHE_DIR,
        help="route plus only: local read-only SYNERGY+ release cache "
        "(default ~/.synergy_dataset_source/synergy-dataset-plus)",
    )
    ap.add_argument(
        "--hydrate",
        choices=("none", "included", "all"),
        default="none",
        help="route v1 only: look up records (any positive label, or all) to attach ISSN/DOI",
    )
    ap.add_argument(
        "--lookup",
        choices=LOOKUP_SOURCES,
        default="openalex",
        help="service used for --hydrate lookups (crossref when the OpenAlex budget is spent)",
    )
    ap.add_argument("--index", type=Path, default=DATA_DIR / "synergy_index.csv")
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    ap.add_argument(
        "--email", default=None, help="OpenAlex mailto (default: OPENALEX_EMAIL in .env)"
    )
    ap.add_argument(
        "--api-key", default=None, help="OpenAlex API key (default: OPENALEX_API_KEY in .env)"
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="keep only the first N rows (smoke tests)"
    )
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument(
        "--rps", type=float, default=MAX_REQUESTS_PER_SECOND, help="max lookup requests/second"
    )
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dotenv = load_dotenv_values(REPO_ROOT / ".env")
    email = args.email or dotenv.get("OPENALEX_EMAIL") or None
    global OPENALEX_API_KEY
    OPENALEX_API_KEY = args.api_key or dotenv.get("OPENALEX_API_KEY") or None
    limiter = RateLimiter(args.rps)
    headers = {
        "User-Agent": "ScholarRAG-evaluation/1.1 (+https://github.com/asreview/synergy-dataset)"
    }
    out_path = args.out_dir / f"{args.dataset}.jsonl"
    hydrate_label = "none"
    hydrate_lookup: str | None = None
    targets: list[dict[str, Any]] = []
    v2_info: dict[str, int] = {}
    plus_info: dict[str, Any] = {}
    label_column_mapping: dict[str, str] | None = None

    with httpx.Client(timeout=args.timeout, follow_redirects=True, headers=headers) as client:
        if args.route == "plus":
            label_column_mapping = dict(PLUS_LABEL_COLUMN_MAPPING)
            meta = plus_metadata(args.cache_dir, args.dataset)
            source_url = PLUS_LABELS_URL.format(name=args.dataset)
            labels_path = args.cache_dir / args.dataset / "labels.csv"
            if labels_path.exists():
                labels_text = labels_path.read_text(encoding="utf-8")
                labels_source = "cache"
            else:
                print(f"labels.csv not cached for {args.dataset}; downloading {source_url}")
                labels_text = fetch_with_retries(client, source_url).text
                labels_source = "download"
            records = parse_plus_labels(labels_text)
            if args.limit:
                records = records[: args.limit]
            works_paths = sorted((args.cache_dir / args.dataset).glob("works_*.zip"))
            if works_paths:
                works: dict[str, dict[str, Any]] = {}
                for wp in works_paths:
                    works.update(load_plus_works(wp))
                hydrate_label = "openalex_id"
                works_source = "cache"
                hits = apply_plus_works(records, works)
                print(
                    f"  filled {hits}/{len(records)} from {len(works_paths)} cached works file(s)"
                )
            else:
                hydrate_label = "openalex_id"
                works_source = "openalex_id"
                print(f"no works archive cached for {args.dataset}; hydrating by OpenAlex id")
                hits = hydrate_by_ids(records, client, email, limiter)
                print(f"  hydrated {hits}/{len(records)}")
            plus_info = {
                "labels_source": labels_source,
                "works_source": works_source,
                "publication_doi": meta["publication_doi"],
                "eligibility_criteria": meta["eligibility_criteria"],
                "catalogue_n_records": meta["catalogue_n_records"],
                "catalogue_n_records_included": meta["catalogue_n_records_included"],
                "catalogue_agrees_n": meta["catalogue_n_records"] == len(records)
                if args.limit is None else None,
                "catalogue_agrees_n_included": (
                    meta["catalogue_n_records_included"]
                    == sum(1 for r in records if r.get("label_included") == 1)
                    if args.limit is None else None
                ),
            }
        elif args.route == "v1":
            row = index_row(ensure_index(args.index, client), args.dataset)
            source_url = row["url"]
            print(f"downloading v1 CSV: {source_url}")
            records = parse_v1_csv(fetch_with_retries(client, source_url).text)
            if args.limit:
                records = records[: args.limit]
            if args.hydrate != "none":
                hydrate_label = args.hydrate
                hydrate_lookup = args.lookup
                targets = hydration_targets(records, args.hydrate)
                by_doi = hydrate_by_doi if args.lookup == "openalex" else hydrate_by_doi_crossref
                by_title = (
                    hydrate_by_title if args.lookup == "openalex" else hydrate_by_title_crossref
                )
                n_doi = sum(1 for r in targets if r.get("doi"))
                if n_doi:
                    print(f"hydrating {n_doi} of {len(targets)} targets by DOI via {args.lookup}")
                    hits_doi = by_doi(targets, client, email, limiter)
                    print(f"  matched {hits_doi}/{n_doi} by DOI")
                rest = [r for r in targets if not r.get("issn")]
                print(f"hydrating {len(rest)} records by title via {args.lookup} ...")
                hits = by_title(rest, client, email, limiter)
                print(f"  matched {hits}/{len(rest)} by exact title")
        else:
            source_url = V2_IDS_URL.format(name=args.dataset)
            print(f"downloading v2 ids CSV: {source_url}")
            raw_records = parse_ids_csv(fetch_with_retries(client, source_url).text)
            n_ids = sum(1 for r in raw_records if r["openalex_id"])
            if n_ids:
                records, v2_info = dedupe_v2_records(raw_records)
                print(
                    f"  {v2_info['n_rows_in_ids_csv']} rows -> {len(records)} distinct works "
                    f"({v2_info['n_dropped_no_openalex_id']} without identifier dropped, "
                    f"{v2_info['n_duplicate_ids']} duplicate ids merged)"
                )
            else:
                records = raw_records
            if args.limit:
                records = records[: args.limit]
            n_ids = sum(1 for r in records if r["openalex_id"])
            if n_ids == 0:
                print(
                    f"WARNING: {args.dataset} has no OpenAlex identifiers in the v2 ids file; "
                    "use --route v1 instead.",
                    file=sys.stderr,
                )
            else:
                hydrate_label = "openalex_id"
                print(f"hydrating {n_ids} records from OpenAlex in batches of {OPENALEX_BATCH} ...")
                hits = hydrate_by_ids(records, client, email, limiter)
                print(f"  hydrated {hits}/{n_ids}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".jsonl.tmp")
    if tmp.exists():
        tmp.unlink()
    append_jsonl(tmp, records)
    tmp.replace(out_path)

    counts = dataset_counts(records)
    info = fetch_info(
        args.dataset,
        args.route,
        source_url,
        hydrate_label,
        counts,
        args.limit,
        hydrate_lookup=hydrate_lookup,
        n_hydrate_targets=len(targets) if hydrate_lookup else None,
        n_targets_with_issn=(
            sum(1 for r in targets if r.get("issn")) if hydrate_lookup else None
        ),
        label_column_mapping=label_column_mapping,
    )
    info.update(v2_info)
    info.update(plus_info)
    info_path = out_path.with_suffix(".fetch.json")
    write_json(info_path, info)
    print(f"wrote {out_path} and {info_path.name}")
    for key, value in info.items():
        _safe_print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
