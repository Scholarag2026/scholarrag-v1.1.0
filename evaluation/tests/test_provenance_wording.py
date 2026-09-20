"""Guard against re-introducing false non-automated/human-labour provenance claims.

Patching only the exact phrases a one-off grep had named lets a new phrase slip through
every time. This test instead re-applies the *same* regex an exhaustive sweep used, over
the *same* file set, and fails on any match that is not on the explicit allow-list below
(``file`` + exact matched ``phrase`` + the count and required context found at fix time). A
brand-new match, or more instances of an
already-allowed phrase than recorded, means someone re-introduced (or added) a false or
ambiguous claim of non-automated/human labour and must either name the true actor (a Claude
Code agent, "not the authors") or add a justified allow-list entry.

Each allow-list entry also pins one or more required context snippets (e.g. ``"not"``,
``"re-read"``, ``"must be"``, an unrelated-meaning false positive such as an unrelated
journal or document title, or an in-file marker such as ``"test fixture"`` / ``"pattern
source"`` / ``"false positive"``) that the matched line, or its immediate one-line
neighbourhood (to absorb prose word-wrap), must still contain. A phrase-count-only
allow-list cannot catch someone keeping the exact allow-listed phrase and count while
turning a not-yet-done disclosure into a claim that the work is already finished;
requiring the safe-context words to still be present makes that flip fail here too.

This file's own docstring, comments and the regex pattern's own source text are scanned
like any other file: only the ``ALLOWED_HITS`` dict literal below (a data literal that
necessarily quotes every banned phrase as a key) is exempted, via the
``PROVENANCE-SCAN-EXEMPT-START``/``-END`` sentinel comments. Every remaining hit in this
file is allow-listed below under its own path key, each with a genuine context pin (the
regex pattern source lines are marked inline with a ``# pattern source, not a claim``
comment; the allow-list's own explanatory comments are pinned to the "false positive" /
"test fixture" wording they already use, or to a natural "not").

Interpreter: ``evaluation/.venv/Scripts/python`` or plain ``python`` (stdlib only: os, re).
No LLM, no network.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from common import EVAL_ROOT, REPO_ROOT

# Identical to the regex used by the 2026-09-05 exhaustive provenance sweep.
PROVENANCE_PATTERN = re.compile(
    r"(?i)\b(by hand|hand[- ]?(check|review|spot|assigned|filled|coded|categoris)|"
    r"manual(ly)?|human (read|check|review|coder|coding|annotat)|"  # pattern source
    r"expert (human )?annotat|"
    r"(read|checked|judged|confirmed|overr\w+) by (a |the )?(human|author|person))"
)

SCANNED_EXTENSIONS = {".py", ".md", ".json", ".csv", ".txt"}

# Directories/files the sweep is not allowed to touch or need not scan: gitignored /
# generated / third-party dirs.
EXCLUDED_DIR_NAMES = {".venv", "__pycache__", "data"}
EXCLUDED_PATH_PREFIXES: tuple[str, ...] = ()

# The digest lives outside evaluation/ and is gitignored (not present in every checkout),
# so it is scanned only when present.
DIGEST_PATH = REPO_ROOT / "softwarex publication" / "human-review" / "provenance-digest.md"

# Sentinel comments marking the *only* exempted range of this file: the ALLOWED_HITS dict
# literal below necessarily quotes every banned phrase as a dict key, which would otherwise
# be a tautological self-match. Everything else in this file (this docstring, the regex
# pattern's own source text, every other comment) is scanned like any other file and must
# carry its own allow-list entry with a genuine context pin, under the
# "evaluation/tests/test_provenance_wording.py" key below.
# ---------------------------------------------------------------------------------------
# Explicit allow-list: (repo-relative posix path, exact regex match text) -> (count found at
# fix time, required context snippet(s); a hit's matched line or its immediate one-line
# neighbourhood must contain at least one). Each entry was individually reviewed and is one
# of:
#   * a real, already-happened human labour claim (e.g. the SYNERGY/SciFact human screeners,
#     or an as-yet-unverified claim the sweep found no evidence to contradict);
#   * a negation ("not expert human annotations", "not yet re-read by the authors") or a
#     statement that the authors have NOT YET re-read/countersigned agent work;
#   * the rubric text in INSTRUCTIONS_v2.md, which says it was drafted for a human
#     annotator but was in fact executed by Claude Opus sessions (also disclosed there);
#   * data (protocol/spot-check JSON) this sweep is barred from modifying, where the match
#     is either already correctly worded (names the harness fix agent, or says "not yet
#     re-read") or a false positive on an unrelated meaning of "manual" ("software manual",
#     "Manual Therapy" journal title, "Diagnostic and Statistical Manual");
#   * a test docstring/assertion in this file's own siblings that quotes or forbids a
#     banned phrase, or a test fixture literal unrelated to provenance ("reason": "manual",
#     "hand-filled" as arbitrary stub content), each marked inline as such;
#   * this file's own regex pattern source, or this allow-list's explanatory comments,
#     marked inline "pattern source" / "false positive" / "test fixture", or naturally
#     containing "not".
# A count higher than listed here means a *new* occurrence was added; a hit whose line (or
# one-line neighbourhood) is missing every required context snippet means an existing
# occurrence's wording changed -- e.g. a negation flipped into a claim -- and both are a
# failure.
# PROVENANCE-SCAN-EXEMPT-START
ALLOWED_HITS: dict[tuple[str, str], tuple[int, tuple[str, ...]]] = {
    ("softwarex publication/human-review/provenance-digest.md", "checked by the author"): (2, ("must be",)),
    ("softwarex publication/human-review/provenance-digest.md", "read by the author"): (2, ("re-read",)),
    ("evaluation/README.md", "checked by the author"): (1, ("must be",)),
    ("evaluation/README.md", "expert annotat"): (1, ("not",)),
    ("evaluation/README.md", "expert human annotat"): (1, ("not",)),
    ("evaluation/README.md", "human review"): (1, ("reviewers",)),
    ("evaluation/README.md", "read by the author"): (1, ("re-read",)),
    ("evaluation/claims/annotation/INSTRUCTIONS_v2.md", "human annotat"): (1, ("not",)),
    ("evaluation/claims/annotation/INSTRUCTIONS_v3.md", "human annotat"): (1, ("not",)),
    # 2 hits: the v2-pass section 5 disclosure ("not human expert annotations") plus the
    # v3-pass section 9.9 restatement of the same disclosure ("not human or expert annotations").
    ("evaluation/claims/annotation/PROVENANCE.md", "expert annotat"): (2, ("not",)),
    # PROVENANCE.md section 8.10 reproduces attestation3_B.md's own text verbatim ("model
    # session, not a human annotator"); the same disclosure, quoted rather than authored here.
    ("evaluation/claims/annotation/PROVENANCE.md", "human annotat"): (1, ("not",)),
    # PROVENANCE.md section 8.7: "a future human reviewer confirms or overrides in the
    # workbook" is a forward-looking disclosure of pending human review, not a claim of
    # human labour done.
    ("evaluation/claims/annotation/PROVENANCE.md", "human review"): (1, ("future",)),
    # The real-claims attestation: "model session, not a human annotator", the same
    # disclosure pattern as the INSTRUCTIONS_*.md entries above, not a claim of human labour.
    ("evaluation/claims/annotation/attestation3_B.md", "human annotat"): (1, ("not",)),
    # The real-claims annotation files: all four "manual" hits are the substring inside
    # "manualized", quoted verbatim from the Vally et al. (2015) abstract ("eight weeks of
    # manualized training in dialogic book-sharing"), not a provenance claim.
    ("evaluation/claims/annotation/real_adjudication_decisions_v3.json", "manual"): (
        1, ("manualized",),
    ),
    ("evaluation/claims/annotation/real_annotation_adjudicated_v3.csv", "manual"): (
        1, ("manualized",),
    ),
    ("evaluation/claims/annotation/real_annotation_completed_v3.csv", "manual"): (
        1, ("manualized",),
    ),
    # Two matches here (not one): real_annotator3_C.csv duplicates the same quoted passage in
    # both its label/justification and annotator_label/annotator_note columns on the same row.
    ("evaluation/claims/annotation/real_annotator3_C.csv", "manual"): (2, ("manualized",)),
    # v6: the guard-7 replay's summary.md, scored against the same annotation csvs; the
    # shared _md_hss_vs_annotation renderer (name="hss" and name="real") writes the same
    # "not expert human annotations" sentence once per Table (E2-d and E2-g).
    ("evaluation/claims/results/v6/summary.md", "expert human annotat"): (2, ("not",)),
    ("evaluation/claims/summarize.py", "expert human annotat"): (1, ("not",)),
    # Tool-comparison evidence: describes the compared tool's (Ai2 ScholarQA) own user
    # workflow -- its users verify citations by hand via excerpt popups -- not a claim about
    # this project's provenance.
    ("evaluation/comparison/tool-comparison-evidence.json", "manually"): (1, ("users verify",)),
    ("evaluation/screening/protocols/Nagtegaal_2019.json", "read by the author"): (
        2, ("re-read",),
    ),
    ("evaluation/screening/protocols/README.md", "read by the author"): (1, ("re-read",)),
    ("evaluation/screening/protocols/Smid_2020.json", "manual"): (2, ("software manual",)),
    (
        "evaluation/screening/protocols/spot_checks/van_de_Schoot_2017.spot_check.json",
        "Manual",
    ): (2, ("Manual Therapy", "Diagnostic and Statistical Manual")),
    ("evaluation/screening/protocols/van_de_Schoot_2017.json", "read by the author"): (
        2, ("re-read",),
    ),
    ("evaluation/screening/results/v3/fn_categorisation/fn_taxonomy.md", "checked by the author"): (
        1, ("must be",),
    ),
    # The disclosure paragraph and the two dataset paper titles/screener-reason quotes,
    # produced under results/v3/.
    ("evaluation/screening/results/v3/summary.json", "manual"): (2, ("reminders",)),
    ("evaluation/screening/results/v3/summary.json", "by hand"): (1, ("not by",)),
    ("evaluation/screening/results/v3/summary.md", "checked by the author"): (1, ("must be",)),
    ("evaluation/screening/results/v3/summary.md", "read by the author"): (1, ("re-read",)),
    ("evaluation/screening/results/v3/Nagtegaal_2019_false_negatives.md", "manual"): (
        2, ("reminders",),
    ),
    ("evaluation/screening/results/v3/van_de_Schoot_2017_false_negatives.md", "by hand"): (
        1, ("not by",),
    ),
    ("evaluation/screening/summarize.py", "checked by the author"): (1, ("must be",)),
    ("evaluation/screening/summarize.py", "read by the author"): (1, ("re-read",)),
    ("evaluation/tests/test_claims_hss.py", "manual"): (1, ("test fixture",)),
    ("evaluation/tests/test_io.py", "by hand"): (3, ("not",)),
    ("evaluation/tests/test_io.py", "expert annotat"): (1, ("not",)),
    ("evaluation/tests/test_io.py", "expert human annotat"): (1, ("not",)),
    ("evaluation/tests/test_io.py", "hand-filled"): (2, ("not",)),
    ("evaluation/tests/test_io.py", "manual"): (1, ("not",)),
    ("evaluation/tests/test_provenance_wording.py", "by hand"): (1, ("pattern source",)),
    ("evaluation/tests/test_provenance_wording.py", "manual"): (
        5, ("pattern source", "false positive", "test fixture", "not"),
    ),
    ("evaluation/tests/test_provenance_wording.py", "Manual"): (2, ("false positive",)),
    ("evaluation/tests/test_provenance_wording.py", "expert human annotat"): (1, ("not",)),
    ("evaluation/tests/test_provenance_wording.py", "read by the author"): (1, ("not",)),
    ("evaluation/tests/test_provenance_wording.py", "hand-filled"): (1, ("test fixture",)),
}
# PROVENANCE-SCAN-EXEMPT-END

EXEMPT_START_MARKER = "PROVENANCE-SCAN-EXEMPT-START"
EXEMPT_END_MARKER = "PROVENANCE-SCAN-EXEMPT-END"


def _is_excluded_dir(dirpath: str) -> bool:
    parts = Path(dirpath).parts
    if any(name in EXCLUDED_DIR_NAMES for name in parts):
        return True
    norm = os.path.normpath(dirpath)
    return any(norm.startswith(prefix) for prefix in EXCLUDED_PATH_PREFIXES)


def _scan_lines(lines: list[str], rel_key: str, hits: list[tuple[str, str, str]]) -> None:
    """Append one ``(rel_key, phrase, context_window)`` tuple per regex match found outside
    the ``PROVENANCE-SCAN-EXEMPT`` sentinel range. ``context_window`` is the matched line
    plus its immediate neighbours (joined), so prose that wraps a safe word ("not",
    "re-read", ...) onto an adjacent line still satisfies the context pin."""
    exempt = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Exact-match only (not a substring check): prose that merely *names* the sentinel
        # (e.g. this file's own docstring, above) must not itself be treated as the boundary.
        if stripped == f"# {EXEMPT_START_MARKER}":
            exempt = True
            continue
        if stripped == f"# {EXEMPT_END_MARKER}":
            exempt = False
            continue
        if exempt:
            continue
        for m in PROVENANCE_PATTERN.finditer(line):
            window = "\n".join(lines[max(0, i - 1):i + 2])
            hits.append((rel_key, m.group(0), window))


def _scan_provenance_hits() -> list[tuple[str, str, str]]:
    """Re-run the exhaustive sweep's scan: same regex, same file set, same exclusions, plus
    this file's own docstring/comments/pattern source, everywhere outside the ALLOWED_HITS
    dict literal itself (see the PROVENANCE-SCAN-EXEMPT sentinels above)."""
    hits: list[tuple[str, str, str]] = []
    self_path = Path(__file__).resolve()
    for dirpath, dirnames, filenames in os.walk(EVAL_ROOT):
        rel_dir = os.path.relpath(dirpath, EVAL_ROOT)
        dirnames[:] = [
            d for d in dirnames
            if not _is_excluded_dir(os.path.join(rel_dir, d) if rel_dir != "." else d)
        ]
        for fn in filenames:
            if Path(fn).suffix not in SCANNED_EXTENSIONS:
                continue
            full = Path(dirpath) / fn
            if full.resolve() == self_path:
                rel_key = "evaluation/tests/test_provenance_wording.py"
            else:
                rel_key = "evaluation/" + os.path.relpath(full, EVAL_ROOT).replace(os.sep, "/")
            with full.open(encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
            _scan_lines(lines, rel_key, hits)

    if DIGEST_PATH.exists():
        with DIGEST_PATH.open(encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
        _scan_lines(lines, "softwarex publication/human-review/provenance-digest.md", hits)

    return hits


def test_no_unexpected_manual_or_human_provenance_claims():
    """Every hit of the provenance regex must be on the reviewed allow-list, at no more than
    the count recorded there, and each hit's one-line neighbourhood must still contain one of
    that entry's required context snippets; anything else is an unreviewed new claim, or an
    existing occurrence whose safe wording (e.g. a negation) was lost."""
    hits = _scan_provenance_hits()
    windows_by_key: dict[tuple[str, str], list[str]] = {}
    for rel_key, phrase, window in hits:
        windows_by_key.setdefault((rel_key, phrase), []).append(window)

    unexpected: list[str] = []
    for key, windows in sorted(windows_by_key.items()):
        file_, phrase = key
        count = len(windows)
        allowed_count, required_contexts = ALLOWED_HITS.get(key, (0, ()))
        if count > allowed_count:
            unexpected.append(f"{file_}: {count - allowed_count} new match(es) of {phrase!r}")
            continue
        for window in windows:
            lowered = window.lower()
            if required_contexts and not any(ctx.lower() in lowered for ctx in required_contexts):
                unexpected.append(
                    f"{file_}: a match of {phrase!r} lost its required context "
                    f"{required_contexts!r}; surrounding text:\n{window}"
                )
    assert not unexpected, (
        "Unreviewed manual/human provenance wording found (not on the allow-list in "
        "test_provenance_wording.py, or missing its required safe-context wording):\n"
        + "\n".join(unexpected)
    )


def test_digest_file_is_reachable_when_present():
    """Sanity check: if the (gitignored) digest exists, this test file's path to it is
    correct, so the guard above is not silently scanning nothing."""
    if not DIGEST_PATH.exists():
        return
    assert DIGEST_PATH.is_file()
    assert "provenance-digest" in DIGEST_PATH.name
