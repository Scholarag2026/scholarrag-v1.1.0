"""Download and verify the SciFact release used by evaluation E2.

Interpreter: ``evaluation/.venv/Scripts/python`` or the system ``python`` (stdlib + httpx).
Importers: ``tests/test_claims_scifact.py``; ``run_scifact.py`` reads the extracted files.

Source: Wadden et al. (2020), "Fact or Fiction: Verifying Scientific Claims", EMNLP,
DOI 10.18653/v1/2020.emnlp-main.609. Licence: claims CC BY 4.0; abstracts (S2ORC) ODC-By 1.0
(https://github.com/allenai/scifact/blob/master/LICENSE.md).

Steps: download ``data.tar.gz`` to ``claims/data/scifact_data.tar.gz`` (skipped when the file
is present with the expected size unless ``--force``), extract the ``data/`` folder into
``claims/data/`` with a safe member filter (no absolute paths, no ``..``), verify the line
counts (claims_dev 300, claims_train 809, claims_test 300, corpus 5183) and the dev gold
pair distribution, and write ``claims/data/scifact.fetch.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import now_iso, read_jsonl, scifact_gold_for_doc, write_json  # noqa: E402

DEFAULT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DEFAULT_OUT_DIR = HERE / "data"
TARBALL_NAME = "scifact_data.tar.gz"
FETCH_INFO_NAME = "scifact.fetch.json"
EXPECTED_BYTES = 3_115_079
EXPECTED_COUNTS: dict[str, int] = {
    "claims_dev": 300,
    "claims_train": 809,
    "claims_test": 300,
    "corpus": 5183,
}
EXPECTED_GOLD_PAIRS: dict[str, int] = {"SUPPORT": 138, "CONTRADICT": 71, "NOT_ENOUGH_INFO": 131}
LICENSE = (
    "claims CC BY 4.0; abstracts (S2ORC) ODC-By 1.0 "
    "(https://github.com/allenai/scifact/blob/master/LICENSE.md)"
)
CITATION_DOI = "10.18653/v1/2020.emnlp-main.609"


def download(url: str, dest: Path, timeout: float = 120.0) -> int:
    """Stream ``url`` to ``dest``; returns the byte count. Uses httpx (Git-Bash curl fails TLS)."""
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_bytes():
                fh.write(chunk)
                total += len(chunk)
    return total


def safe_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """Regular files/dirs under ``data/`` whose path is relative and contains no ``..``."""
    out: list[tarfile.TarInfo] = []
    for member in tar.getmembers():
        posix = PurePosixPath(member.name)
        if posix.is_absolute() or member.name.startswith(("/", "\\")):
            continue
        if ".." in posix.parts or not (member.isfile() or member.isdir()):
            continue
        if not posix.parts or posix.parts[0] != "data":
            continue
        out.append(member)
    return out


def extract(tar_path: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        members = safe_members(tar)
        tar.extractall(out_dir, members=members, filter="data")
    return [m.name for m in members]


def count_lines(path: Path) -> int:
    with Path(path).open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def verify_layout(data_dir: Path) -> dict[str, int]:
    """Line counts of the four SciFact files under ``data_dir`` (SystemExit if one is absent)."""
    counts: dict[str, int] = {}
    for key in EXPECTED_COUNTS:
        path = Path(data_dir) / f"{key}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        counts[key] = count_lines(path)
    return counts


def check_counts(counts: dict[str, int]) -> None:
    bad = {k: (v, EXPECTED_COUNTS[k]) for k, v in counts.items() if v != EXPECTED_COUNTS.get(k)}
    if bad:
        raise SystemExit(f"unexpected line counts (got, expected): {bad}")


def dev_gold_pairs(data_dir: Path) -> dict[str, int]:
    claims = read_jsonl(Path(data_dir) / "claims_dev.jsonl")
    corpus = {int(d["doc_id"]) for d in read_jsonl(Path(data_dir) / "corpus.jsonl")}
    counter: Counter[str] = Counter()
    for claim in claims:
        for doc_id in claim.get("cited_doc_ids") or []:
            if int(doc_id) in corpus:
                counter[scifact_gold_for_doc(claim, doc_id)] += 1
    return {k: counter.get(k, 0) for k in EXPECTED_GOLD_PAIRS}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--url", default=DEFAULT_URL)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    tar_path = out_dir / TARBALL_NAME
    if args.force or not tar_path.exists() or tar_path.stat().st_size != EXPECTED_BYTES:
        print(f"downloading {args.url} -> {tar_path}")
        n = download(args.url, tar_path)
        print(f"  {n} bytes")
    else:
        print(f"{tar_path} present ({tar_path.stat().st_size} bytes); skipping download")
    extract(tar_path, out_dir)
    data_dir = out_dir / "data"
    counts = verify_layout(data_dir)
    check_counts(counts)
    gold = dev_gold_pairs(data_dir)
    if gold != EXPECTED_GOLD_PAIRS:
        raise SystemExit(f"dev gold pairs {gold} != expected {EXPECTED_GOLD_PAIRS}")
    info: dict[str, Any] = {
        "url": args.url,
        "fetched_at": now_iso(),
        "bytes": tar_path.stat().st_size,
        "sha256": sha256_of(tar_path),
        "counts": counts,
        "dev_gold_pairs": gold,
        "license": LICENSE,
        "citation_doi": CITATION_DOI,
    }
    write_json(out_dir / FETCH_INFO_NAME, info)
    print(f"verified {data_dir}: {counts}; dev gold pairs {gold}; wrote {FETCH_INFO_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
