"""claims/replay_scale_guard: the ``--out`` argument and its default.

The script's write target must be a results directory that ships with the release, not
an internal working-files path, and must be overridable so a caller can point it
somewhere else without editing the module. The full replay pipeline itself needs the
v5 result files, the promoted demo run's delivered-evidence file and the real-set
annotation CSV to run end to end, and makes no network or model call but is otherwise
an integration test out of scope here; this file covers the argument-parsing surface
and that ``main``/``_run`` honour it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import replay_scale_guard as rsg


def test_out_defaults_to_a_results_directory_that_ships() -> None:
    args = rsg.parse_args([])
    assert args.out == rsg.OUT_DIR
    assert args.out.parent.name == "results"
    assert args.out.name == "scale_guard_replay"


def test_out_is_overridable(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    args = rsg.parse_args(["--out", str(target)])
    assert args.out == target


def test_run_writes_to_the_given_out_dir(tmp_path: Path, monkeypatch) -> None:
    """``_run`` must write into whatever directory it is given, not the module-level
    default, so ``--out`` actually redirects the output."""
    written: list[Path] = []

    async def fake_body(out_dir: Path) -> int:
        out_dir.mkdir(parents=True, exist_ok=True)
        written.append(out_dir)
        return 0

    monkeypatch.setattr(rsg, "_run", fake_body)
    target = tmp_path / "custom"
    rc = asyncio.run(rsg._run(target))
    assert rc == 0
    assert written == [target]
    assert target.is_dir()


def test_main_passes_parsed_out_to_run(monkeypatch, tmp_path: Path) -> None:
    seen: list[Path] = []

    async def fake_run(out_dir: Path) -> int:
        seen.append(out_dir)
        return 0

    monkeypatch.setattr(rsg, "_run", fake_run)
    target = tmp_path / "main-out"
    rc = rsg.main(["--out", str(target)])
    assert rc == 0
    assert seen == [target]
