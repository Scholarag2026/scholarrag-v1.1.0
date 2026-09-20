"""Pure helpers shared by the ScholarRAG evaluation scripts.

Importers: ``evaluation/screening/*.py``, ``evaluation/claims/*.py`` and the unit tests in
``evaluation/tests``. Nothing here touches the network, an LLM or the backend; every
function is deterministic and unit-tested so that the numbers reported in the manuscript
can be re-derived from the committed result files.

Conventions
-----------
* Binary labels are ``0``/``1`` integers; a prediction of ``None`` means "not screened"
  (for example a batch that failed after retries) and is counted separately.
* Metrics return ``None`` when their denominator is zero instead of raising.
* WSS@R follows Cohen et al. (2006): ``WSS@R = (TN + FN) / N - (1 - R)``.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent
BACKEND_ROOT = REPO_ROOT / "backend"

# --------------------------------------------------------------------------------------
# Binary classification metrics
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BinaryCounts:
    """Confusion counts for one binary run. ``skipped`` = predictions that were ``None``."""

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    skipped: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def n_positive(self) -> int:
        return self.tp + self.fn

    @property
    def n_predicted_positive(self) -> int:
        return self.tp + self.fp


def _safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def confusion_counts(
    labels: Sequence[int],
    preds: Sequence[int | None],
    *,
    none_as: int | None = None,
) -> BinaryCounts:
    """Count TP/FP/TN/FN. ``None`` predictions are skipped unless ``none_as`` maps them."""
    if len(labels) != len(preds):
        raise ValueError(f"labels ({len(labels)}) and preds ({len(preds)}) differ in length")
    tp = fp = tn = fn = skipped = 0
    for label, pred in zip(labels, preds, strict=True):
        if pred is None:
            if none_as is None:
                skipped += 1
                continue
            pred = none_as
        if label not in (0, 1) or pred not in (0, 1):
            raise ValueError(f"labels/preds must be 0 or 1, got label={label!r} pred={pred!r}")
        if label == 1 and pred == 1:
            tp += 1
        elif label == 1:
            fn += 1
        elif pred == 1:
            fp += 1
        else:
            tn += 1
    return BinaryCounts(tp=tp, fp=fp, tn=tn, fn=fn, skipped=skipped)


def recall(c: BinaryCounts) -> float | None:
    return _safe_div(c.tp, c.tp + c.fn)


def precision(c: BinaryCounts) -> float | None:
    return _safe_div(c.tp, c.tp + c.fp)


def specificity(c: BinaryCounts) -> float | None:
    return _safe_div(c.tn, c.tn + c.fp)


def f1(c: BinaryCounts) -> float | None:
    p, r = precision(c), recall(c)
    if p is None or r is None:
        return None
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def inclusion_rate(c: BinaryCounts) -> float | None:
    return _safe_div(c.tp + c.fp, c.n)


def prevalence(c: BinaryCounts) -> float | None:
    return _safe_div(c.tp + c.fn, c.n)


def wss(c: BinaryCounts, recall_level: float) -> float | None:
    """Work saved over sampling at recall ``R`` (Cohen et al. 2006): (TN+FN)/N - (1-R)."""
    if c.n == 0:
        return None
    return (c.tn + c.fn) / c.n - (1.0 - recall_level)


def wss_at_achieved_recall(c: BinaryCounts) -> float | None:
    """WSS evaluated at the recall the classifier actually achieved."""
    r = recall(c)
    return None if r is None else wss(c, r)


def wss_at_95(c: BinaryCounts) -> float | None:
    """WSS@95 for a fixed binary decision; only defined when achieved recall >= 0.95."""
    r = recall(c)
    if r is None or r < 0.95:
        return None
    return wss(c, 0.95)


def binary_metrics(c: BinaryCounts) -> dict[str, Any]:
    """All headline screening metrics for one run, as a JSON-ready dict."""
    return {
        "n": c.n,
        "n_skipped": c.skipped,
        "tp": c.tp,
        "fp": c.fp,
        "tn": c.tn,
        "fn": c.fn,
        "prevalence": prevalence(c),
        "recall": recall(c),
        "precision": precision(c),
        "f1": f1(c),
        "specificity": specificity(c),
        "inclusion_rate": inclusion_rate(c),
        "wss_at_achieved_recall": wss_at_achieved_recall(c),
        "wss_at_95": wss_at_95(c),
    }


def wss_from_ranking(
    scores: Sequence[float], labels: Sequence[int], recall_level: float = 0.95
) -> float | None:
    """WSS@R for a *ranked* list: screen in descending score order until R of the positives
    have been found; the unscreened remainder is the work saved (Cohen et al. 2006)."""
    n = len(scores)
    if n == 0 or len(labels) != n:
        return None
    n_pos = sum(1 for x in labels if x == 1)
    if n_pos == 0:
        return None
    target = math.ceil(recall_level * n_pos - 1e-9)
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    found = 0
    screened = 0
    for i in order:
        screened += 1
        if labels[i] == 1:
            found += 1
            if found >= target:
                break
    return (n - screened) / n - (1.0 - recall_level)


def predictions_matching_count(scores: Sequence[float], k: int) -> tuple[list[int], float | None]:
    """Mark exactly the ``k`` highest-scoring items as 1 (ties broken by position).

    Returns ``(predictions, threshold)`` where ``threshold`` is the lowest included score.
    Used to give a similarity baseline the same inclusion count as the LLM run.
    """
    n = len(scores)
    k = max(0, min(k, n))
    preds = [0] * n
    if k == 0:
        return preds, None
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    for i in order[:k]:
        preds[i] = 1
    return preds, float(scores[order[k - 1]])


# --------------------------------------------------------------------------------------
# Agreement and multi-class metrics
# --------------------------------------------------------------------------------------


def _paired(a: Sequence[Any], b: Sequence[Any]) -> list[tuple[Any, Any]]:
    if len(a) != len(b):
        raise ValueError(f"sequences differ in length: {len(a)} vs {len(b)}")
    return [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]


def percent_agreement(a: Sequence[Any], b: Sequence[Any]) -> float | None:
    pairs = _paired(a, b)
    if not pairs:
        return None
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def cohens_kappa(a: Sequence[Any], b: Sequence[Any]) -> float | None:
    """Cohen's kappa for two categorical runs (binary or multi-class). ``None`` pairs skipped.

    When expected agreement is 1 (both runs constant and identical) kappa is undefined and
    ``None`` is returned; :func:`kappa_undefined` reports that case so tables can print
    ``-`` with a footnote instead of a number. Two constant runs that *disagree* have
    ``pe = 0.5`` and give kappa 0.
    """
    pairs = _paired(a, b)
    if not pairs:
        return None
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    ca = Counter(x for x, _ in pairs)
    cb = Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    if math.isclose(pe, 1.0):
        return None
    return (po - pe) / (1.0 - pe)


def kappa_undefined(a: Sequence[Any], b: Sequence[Any]) -> bool:
    """True when both runs are constant and identical on the compared pairs (pe = 1)."""
    pairs = _paired(a, b)
    if not pairs:
        return False
    return len({x for x, _ in pairs}) == 1 and len({y for _, y in pairs}) == 1 and all(
        x == y for x, y in pairs
    )


def confusion_matrix(
    gold: Sequence[str],
    pred: Sequence[str],
    gold_classes: Sequence[str] | None = None,
    pred_classes: Sequence[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Nested counts ``matrix[gold][pred]`` with stable class ordering."""
    if len(gold) != len(pred):
        raise ValueError("gold and pred differ in length")
    g_classes = list(gold_classes) if gold_classes else sorted(set(gold))
    p_classes = list(pred_classes) if pred_classes else sorted(set(pred))
    for g in gold:
        if g not in g_classes:
            g_classes.append(g)
    for p in pred:
        if p not in p_classes:
            p_classes.append(p)
    matrix = {g: {p: 0 for p in p_classes} for g in g_classes}
    for g, p in zip(gold, pred, strict=True):
        matrix[g][p] += 1
    return matrix


def per_class_prf(
    gold: Sequence[str], pred: Sequence[str], classes: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Per-class precision / recall / F1 with support (gold count) and predicted count.

    Predictions outside ``classes`` count as misses for the gold class but are never a
    false positive for any listed class (this is how ``needs_nuance`` stays visible
    without being merged into another class). A class with gold support that is never
    predicted has precision ``None`` (undefined) and F1 ``0.0`` by convention (sklearn
    ``zero_division=0``); a class without gold support and without predictions has F1
    ``None``.
    """
    if len(gold) != len(pred):
        raise ValueError("gold and pred differ in length")
    out: dict[str, dict[str, Any]] = {}
    for cls in classes:
        tp = sum(1 for g, p in zip(gold, pred, strict=True) if g == cls and p == cls)
        support = sum(1 for g in gold if g == cls)
        predicted = sum(1 for p in pred if p == cls)
        p_ = _safe_div(tp, predicted)
        r_ = _safe_div(tp, support)
        if r_ is None:
            f_ = None  # no gold support: nothing to recall
        elif p_ is None:
            f_ = 0.0  # support > 0 but never predicted: recall 0, F1 0 by convention
        elif p_ + r_ == 0:
            f_ = 0.0
        else:
            f_ = 2 * p_ * r_ / (p_ + r_)
        out[cls] = {
            "precision": p_,
            "recall": r_,
            "f1": f_,
            "support": support,
            "predicted": predicted,
            "tp": tp,
        }
    return out


def macro_f1(prf: Mapping[str, Mapping[str, Any]]) -> float | None:
    """Unweighted mean of F1 over every class with gold support (a never-predicted class
    contributes 0.0, never a dropped denominator); ``None`` when no class has support."""
    values = [
        0.0 if v.get("f1") is None else float(v["f1"])
        for v in prf.values()
        if v.get("support", 0) > 0 or v.get("f1") is not None
    ]
    return None if not values else sum(values) / len(values)


def accuracy(gold: Sequence[Any], pred: Sequence[Any]) -> float | None:
    return percent_agreement(gold, pred)


def wilson_interval(
    count: int, n: int, *, alpha: float = 0.05
) -> tuple[float, float] | tuple[None, None]:
    """Wilson score interval for a binomial proportion (``count`` successes out of ``n``).

    Returns ``(lo, hi)`` clipped to ``[0, 1]``, or ``(None, None)`` when ``n`` is 0.
    ``alpha`` is the two-sided significance level (``0.05`` -> a 95 % interval); the
    critical value is the exact standard-normal quantile (``statistics.NormalDist``), not a
    hardcoded z, so any ``alpha`` is exact, not just 0.05.
    """
    if n <= 0:
        return None, None
    z = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z2 = z * z
    p_hat = count / n
    denom = 1.0 + z2 / n
    center = p_hat + z2 / (2 * n)
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return max(0.0, lo), min(1.0, hi)


# --------------------------------------------------------------------------------------
# Text helpers (quote fidelity, lexical baseline, sentence selection)
# --------------------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

# NFKC alone folds none of these -- the dash and quote variants are
# canonical (not compatibility) code points -- which is why the map is explicit. Mirrored
# byte-for-byte in ``backend/app/agents/relevance_screener_agent.py``'s private
# ``_UNICODE_FOLD_MAP`` because the backend cannot import ``evaluation/``.
_UNICODE_FOLD_MAP: dict[str, str] = {
    # hyphen, non-breaking hyphen, figure dash, en dash, em dash, horizontal bar, minus sign
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "−": "-",
    # left/right single quotation mark, single low-9 quotation mark, modifier letter apostrophe
    "‘": "'", "’": "'", "‛": "'", "ʼ": "'",
    # left/right double quotation mark, double high-reversed-9 quotation mark
    "“": '"', "”": '"', "‟": '"',
    # no-break space, thin space, narrow no-break space
    " ": " ", " ": " ", " ": " ",
    # horizontal ellipsis
    "…": "...",
}
_UNICODE_FOLD_RE = re.compile("|".join(re.escape(k) for k in _UNICODE_FOLD_MAP))


def normalise_unicode_punctuation(text: str) -> str:
    """NFKC-normalise, then fold Unicode dashes/quotes/spaces/ellipsis to ASCII.
    Applied before whitespace normalisation and casefolding in
    :func:`quote_is_verbatim`, so a verbatim check does not flip on a curly quote or an en
    dash the model copied faithfully but the shown text renders with a different code point
    (record 1065's U+2010 case)."""
    text = unicodedata.normalize("NFKC", text or "")
    return _UNICODE_FOLD_RE.sub(lambda m: _UNICODE_FOLD_MAP[m.group(0)], text)


def normalise_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text or "").strip()


def quote_is_verbatim(quote: str | None, chunk: str | None, *, casefold: bool = False) -> bool:
    """True when ``quote`` is a substring of ``chunk`` after Unicode punctuation folding and
    whitespace normalisation."""
    if not quote or not chunk:
        return False
    q = normalise_whitespace(normalise_unicode_punctuation(quote))
    c = normalise_whitespace(normalise_unicode_punctuation(chunk))
    if not q:
        return False
    if casefold:
        q, c = q.casefold(), c.casefold()
    return q in c


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def token_jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def max_sentence_jaccard(claim: str, sentences: Sequence[str]) -> tuple[float, int | None]:
    """Highest token-Jaccard between ``claim`` and any sentence, with that sentence's index."""
    best, best_i = 0.0, None
    for i, s in enumerate(sentences):
        j = token_jaccard(claim, s)
        if j > best:
            best, best_i = j, i
    return best, best_i


def split_sentences(text: str) -> list[str]:
    """Light sentence splitter (period/!/? followed by whitespace and a capital/digit)."""
    text = normalise_whitespace(text)
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


FINDING_MARKERS: tuple[str, ...] = (
    "found",
    "show",
    "reveal",
    "indicate",
    "suggest",
    "significant",
    "result",
    "increase",
    "decrease",
    "higher",
    "lower",
    "associated",
    "effect",
    "correlat",
    "participants",
    "outperform",
    "improv",
    "%",
)

_FIGURE_REF_RE = re.compile(r"\b(?:fig\.?|figure|table)\s*\d", re.IGNORECASE)


def score_candidate_sentence(
    sentence: str,
    *,
    min_words: int = 12,
    max_words: int = 40,
    keywords: Sequence[str] = FINDING_MARKERS,
) -> float:
    """Heuristic eligibility score for a sentence to become an HSS test claim.

    0 means ineligible (too short/long, figure/table reference, unbalanced parentheses,
    does not start with a capital letter or end with terminal punctuation). Eligible
    sentences score ``1 + number of finding markers`` so result-like statements rank first.
    """
    s = normalise_whitespace(sentence)
    words = s.split()
    if len(words) < min_words or len(words) > max_words:
        return 0.0
    if not re.match(r"^[A-Z]", s) or not s.endswith((".", "!", "?")):
        return 0.0
    if s.count("(") != s.count(")"):
        return 0.0
    if _FIGURE_REF_RE.search(s):
        return 0.0
    low = s.lower()
    hits = sum(1 for k in keywords if k in low)
    return 1.0 + hits


@dataclass(frozen=True)
class CandidateSentence:
    chunk_index: int
    sentence: str
    score: float


def select_candidate_sentences(
    chunks: Sequence[str], limit: int | None = None, **score_kwargs: Any
) -> list[CandidateSentence]:
    """Rank eligible sentences from ``chunks`` (score desc, then chunk order, then position).

    Deterministic and duplicate-free (by normalised text).
    """
    seen: set[str] = set()
    ranked: list[tuple[float, int, int, str]] = []
    for ci, chunk in enumerate(chunks):
        for si, sentence in enumerate(split_sentences(chunk)):
            key = normalise_whitespace(sentence).casefold()
            if key in seen:
                continue
            score = score_candidate_sentence(sentence, **score_kwargs)
            if score <= 0:
                continue
            seen.add(key)
            ranked.append((-score, ci, si, normalise_whitespace(sentence)))
    ranked.sort()
    out = [CandidateSentence(chunk_index=ci, sentence=s, score=-neg) for neg, ci, _, s in ranked]
    return out[:limit] if limit is not None else out


# --------------------------------------------------------------------------------------
# SciFact label mapping
# --------------------------------------------------------------------------------------

SCIFACT_LABELS: tuple[str, ...] = ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")
SCIFACT_TO_STATUS: dict[str, str] = {
    "SUPPORT": "verified",
    "CONTRADICT": "unsupported",
    "NOT_ENOUGH_INFO": "unsupported",
}
VERIFICATION_STATUSES: tuple[str, ...] = (
    "verified",
    "needs_nuance",
    "unsupported",
    "no_full_text",
    "error",
)


def scifact_gold_for_doc(claim: Mapping[str, Any], doc_id: int | str) -> str:
    """Gold label for one (claim, cited document) pair.

    SUPPORT / CONTRADICT when the claim's evidence for that document carries that label,
    otherwise NOT_ENOUGH_INFO (SciFact: a cited document without evidence is NEI).
    """
    evidence = claim.get("evidence") or {}
    entries = evidence.get(str(doc_id)) or evidence.get(doc_id) or []
    labels = {e.get("label") for e in entries if isinstance(e, Mapping)}
    if "SUPPORT" in labels and "CONTRADICT" in labels:
        # Never occurs in SciFact dev but guard anyway: mixed evidence is not "verified".
        return "CONTRADICT"
    if "SUPPORT" in labels:
        return "SUPPORT"
    if "CONTRADICT" in labels:
        return "CONTRADICT"
    return "NOT_ENOUGH_INFO"


# --------------------------------------------------------------------------------------
# Resumable JSONL runs
# --------------------------------------------------------------------------------------


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def append_jsonl(path: str | Path, rows: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> int:
    """Append one row or many rows in a single write; returns the number written."""
    if isinstance(rows, Mapping):
        rows = [rows]
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if not payload:
        return 0
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(payload)
    return payload.count("\n")


def completed_ids(path: str | Path, id_field: str) -> set[Any]:
    """Ids already present in a results file (resume support)."""
    return {row[id_field] for row in iter_jsonl(path) if id_field in row}


def pending_items(
    items: Iterable[Mapping[str, Any]], id_field: str, done: set[Any]
) -> list[Mapping[str, Any]]:
    return [it for it in items if it[id_field] not in done]


def prune_failed_rows(path: str | Path, is_failed: Callable[[Mapping[str, Any]], bool]) -> int:
    """Drop rows for which ``is_failed`` is true and rewrite the file atomically (tmp + replace).

    Used by ``--retry-failed``: the removed rows are re-attempted by the following run, which
    appends fresh rows for them. Returns the number of rows removed (0 when the file is absent
    or nothing matched, in which case the file is left untouched).
    """
    p = Path(path)
    if not p.exists():
        return 0
    rows = read_jsonl(p)
    kept = [r for r in rows if not is_failed(r)]
    removed = len(rows) - len(kept)
    if removed == 0:
        return 0
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    return removed


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# Latency and cost
# --------------------------------------------------------------------------------------


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile (p in [0, 100])."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    rank = max(1, math.ceil(p / 100.0 * len(vals)))
    return float(vals[min(rank, len(vals)) - 1])


def latency_stats(values: Sequence[float]) -> dict[str, Any]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0, "median": None, "p90": None, "mean": None, "total": 0.0}
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "p90": percentile(vals, 90),
        "mean": statistics.fmean(vals),
        "total": sum(vals),
    }


# DeepSeek list prices, fetched 2026-09-02 (23:51 UTC) from the pricing page below. The
# page lists only the concrete V4 models; the alias ``deepseek-chat`` used by
# ``settings.deepseek_model`` is mapped to ``deepseek-v4-flash`` (the model the API reports
# for that alias -- runs must confirm this from ``model_reported``). Prices are USD per 1M
# tokens; off-peak rates are half of the peak rates.
#
# From 2026-09-11 the API started self-reporting this same
# deployment's calls as ``"deepseek-flash"`` instead of ``"deepseek-v4-flash"`` (confirmed
# unchanged price row: ``model_configured`` is still ``"deepseek-chat"`` and its resolved
# price-table entry, tiers included, is identical before and after, by the fingerprint/token
# evidence that the underlying deployment, not just this string, changed on DeepSeek's own
# side). Mapped to the same ``"deepseek-v4-flash"`` price row so ``total_call_cost`` recovers
# a real total instead of degrading to ``None`` on every row reporting the new string.
DEEPSEEK_PRICE_PAGE = "https://api-docs.deepseek.com/quick_start/pricing"
DEEPSEEK_PRICES: dict[str, Any] = {
    "source_url": DEEPSEEK_PRICE_PAGE,
    "accessed": "2026-09-02",
    "currency": "USD",
    "unit": "per 1M tokens",
    "peak_hours_utc": "01:00-04:00 and 06:00-10:00, Monday-Friday; all other hours off-peak",
    "alias": {"deepseek-chat": "deepseek-v4-flash", "deepseek-flash": "deepseek-v4-flash"},
    "models": {
        "deepseek-v4-flash": {
            "input_cache_miss": {"peak": 0.44, "offpeak": 0.22},
            "input_cache_hit": {"peak": 0.014, "offpeak": 0.007},
            "output": {"peak": 1.32, "offpeak": 0.66},
        },
        "deepseek-v4-pro": {
            "input_cache_miss": {"peak": 1.32, "offpeak": 0.66},
            "input_cache_hit": {"peak": 0.044, "offpeak": 0.022},
            "output": {"peak": 3.96, "offpeak": 1.98},
        },
    },
}
PEAK_HOURS_UTC: frozenset[int] = frozenset({1, 2, 3, 6, 7, 8, 9})


def resolve_price_model(model: str | None) -> str | None:
    """Alias (``deepseek-chat``) -> concrete price-table name; ``None`` when unknown."""
    if not model:
        return None
    name = DEEPSEEK_PRICES["alias"].get(model, model)
    return name if name in DEEPSEEK_PRICES["models"] else None


def is_peak_utc(called_at: str | datetime) -> bool:
    """True when ``called_at`` (ISO-8601 with offset) falls in DeepSeek's UTC peak windows.

    Peak = Monday-Friday, 01:00-04:00 and 06:00-10:00 UTC (end exclusive). Naive datetimes
    are taken as UTC.
    """
    dt = datetime.fromisoformat(called_at) if isinstance(called_at, str) else called_at
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.weekday() < 5 and dt.hour in PEAK_HOURS_UTC


def call_cost(
    model: str | None,
    called_at: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    cache_read_tokens: int | None = 0,
) -> float | None:
    """USD cost of one call at list price, tiered by call time.

    ``cache_read_tokens`` are billed at the cache-hit rate, the remaining input tokens at
    the cache-miss rate, output tokens at the output rate. ``None`` when the model is not
    in the table, the timestamp is missing or a token count is unknown.
    """
    name = resolve_price_model(model)
    if name is None or called_at is None or input_tokens is None or output_tokens is None:
        return None
    tier = "peak" if is_peak_utc(called_at) else "offpeak"
    rates = DEEPSEEK_PRICES["models"][name]
    cached = max(0, min(int(cache_read_tokens or 0), int(input_tokens)))
    missed = int(input_tokens) - cached
    return (
        missed * rates["input_cache_miss"][tier]
        + cached * rates["input_cache_hit"][tier]
        + int(output_tokens) * rates["output"][tier]
    ) / 1_000_000.0


def price_record(
    model: str,
    price_input: float | None,
    price_output: float | None,
    source_url: str | None = None,
    accessed: str | None = None,
) -> dict[str, Any]:
    """Price entry stored in a run's meta file.

    Defaults come from ``DEEPSEEK_PRICES`` (tiered). Explicit CLI prices are recorded as a
    flat ``input_per_mtok`` / ``output_per_mtok`` pair (both tiers) and override the source
    URL / access date when given.
    """
    name = resolve_price_model(model)
    return {
        "model": model,
        "price_model": name,
        "currency": DEEPSEEK_PRICES["currency"],
        "unit": DEEPSEEK_PRICES["unit"],
        "tiers": dict(DEEPSEEK_PRICES["models"][name]) if name else None,
        "peak_hours_utc": DEEPSEEK_PRICES["peak_hours_utc"],
        "input_per_mtok": price_input,
        "output_per_mtok": price_output,
        "source_url": source_url or DEEPSEEK_PRICES["source_url"],
        "accessed": accessed or DEEPSEEK_PRICES["accessed"],
    }


def compute_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    price_input: float | None,
    price_output: float | None,
) -> float | None:
    """Cost in the price currency; prices are per 1M tokens. ``None`` if anything is unknown."""
    if None in (input_tokens, output_tokens, price_input, price_output):
        return None
    return (input_tokens * price_input + output_tokens * price_output) / 1_000_000.0


# --------------------------------------------------------------------------------------
# Screening protocols
# --------------------------------------------------------------------------------------

LABEL_FIELDS: tuple[str, ...] = ("label_abstract_screening", "label_included")
PROTOCOL_REQUIRED: tuple[str, ...] = (
    "dataset_id",
    "source_review_doi",
    "research_question",
    "inclusion_criteria",
    "exclusion_criteria",
    "label_field",
)


@dataclass
class Protocol:
    dataset_id: str
    source_review_doi: str
    research_question: str
    inclusion_criteria: list[str]
    exclusion_criteria: list[str]
    label_field: str
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None
    #: The stage at which each
    #: criterion can be decided, "abstract" or "full_text", parallel to
    #: ``inclusion_criteria``/``exclusion_criteria``. A criterion whose element in the JSON
    #: is a plain string is implicitly "abstract".
    inclusion_stages: list[str] = field(default_factory=list)
    exclusion_stages: list[str] = field(default_factory=list)
    #: One-line justification, required by :func:`load_protocol` whenever the parallel stage
    #: is "full_text"; "" for an "abstract"-stage criterion that carries none.
    inclusion_stage_rationales: list[str] = field(default_factory=list)
    exclusion_stage_rationales: list[str] = field(default_factory=list)
    #: Whether the criterion's test is that something is *not* present (as against a
    #: positively established fact such as "exact duplicate" or "book or book chapter").
    #: Defaults to False for an inclusion criterion, True for an exclusion criterion.
    inclusion_absence: list[bool] = field(default_factory=list)
    exclusion_absence: list[bool] = field(default_factory=list)


class ProtocolError(ValueError):
    """Raised when a protocol JSON file is malformed."""


#: The stage at which a criterion can be decided. A full-text criterion
#: can never ground an EXCLUDE at title-and-abstract stage (see the relevance screener guard).
CRITERION_STAGES: tuple[str, ...] = ("abstract", "full_text")

_NEGATION_OF_I_RE = re.compile(r"^negation of I(\d+)\b")


def _str_list(value: Any, name: str) -> list[str]:
    """Plain texts of a criteria list: accepts either the legacy list-of-strings shape or
    the widened shape (see :func:`_criteria_stage_fields`); either way, every
    element yields a non-empty ``text``."""
    if not isinstance(value, list) or not value:
        raise ProtocolError(f"{name} must be a list of non-empty strings")
    texts: list[str] = []
    for item in value:
        text = item.get("text") if isinstance(item, dict) else item
        if not isinstance(text, str) or not text.strip():
            raise ProtocolError(f"{name} must be a list of non-empty strings")
        texts.append(text.strip())
    return texts


def _criteria_stage_fields(
    value: Any, name: str, *, default_absence: bool
) -> tuple[list[str], list[str], list[bool]]:
    """Stage, stage_rationale and absence for each element of a criteria list.
    A plain string implies ``{"stage": "abstract"}``. ``stage`` defaults to
    ``"abstract"``; ``stage_rationale`` is required (non-empty) when ``stage`` is
    ``"full_text"``; ``absence`` defaults to ``default_absence`` (False for
    ``inclusion_criteria``, True for ``exclusion_criteria``) and marks a criterion whose test
    is that something is *not* present. Called on the same ``value`` as :func:`_str_list`, to
    which the returned lists stay index-parallel.
    """
    if not isinstance(value, list) or not value:
        raise ProtocolError(f"{name} must be a list of non-empty strings")
    stages: list[str] = []
    rationales: list[str] = []
    absences: list[bool] = []
    for item in value:
        if isinstance(item, str):
            stages.append("abstract")
            rationales.append("")
            absences.append(default_absence)
            continue
        if not isinstance(item, dict):
            raise ProtocolError(f"{name} must be a list of strings or criterion objects")
        stage = item.get("stage", "abstract")
        if stage not in CRITERION_STAGES:
            raise ProtocolError(f"{name}: stage must be one of {CRITERION_STAGES}, got {stage!r}")
        rationale = item.get("stage_rationale") or ""
        if not isinstance(rationale, str):
            raise ProtocolError(f"{name}: stage_rationale must be a string")
        if stage == "full_text" and not rationale.strip():
            raise ProtocolError(
                f"{name}: a full_text criterion requires a non-empty stage_rationale"
            )
        absence = item.get("absence", default_absence)
        if not isinstance(absence, bool):
            raise ProtocolError(f"{name}: absence must be a boolean")
        stages.append(stage)
        rationales.append(rationale)
        absences.append(absence)
    return stages, rationales, absences


def _check_negation_stage_invariant(
    exclusion_provenance: Any,
    inclusion_stages: list[str],
    exclusion_stages: list[str],
    path: Path,
) -> None:
    """Enforce that a derived exclusion criterion shares its source inclusion criterion's
    stage: where ``criteria_provenance.exclusion[i].location`` begins
    "negation of I{n}", ``exclusion_stages[i]`` must equal ``inclusion_stages[n-1]``. Without
    this a model could dodge the full-text flag by citing "I1" instead of "E1" for the same
    underlying fact (record 450 already did once)."""
    if not isinstance(exclusion_provenance, list):
        return
    for i, entry in enumerate(exclusion_provenance):
        if i >= len(exclusion_stages) or not isinstance(entry, dict):
            continue
        location = entry.get("location")
        if not isinstance(location, str):
            continue
        m = _NEGATION_OF_I_RE.match(location)
        if not m:
            continue
        n = int(m.group(1))
        if not (1 <= n <= len(inclusion_stages)):
            continue
        if exclusion_stages[i] != inclusion_stages[n - 1]:
            raise ProtocolError(
                f"protocol {path}: exclusion_criteria[{i}] (E{i + 1}) stage "
                f"{exclusion_stages[i]!r} must equal inclusion_criteria[{n - 1}] (I{n}) stage "
                f"{inclusion_stages[n - 1]!r} because its provenance is {location!r}"
            )


def load_protocol(path: str | Path) -> Protocol:
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProtocolError(f"protocol not found: {p}") from exc
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"protocol {p} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProtocolError(f"protocol {p} must be a JSON object")
    missing = [k for k in PROTOCOL_REQUIRED if k not in raw]
    if missing:
        raise ProtocolError(f"protocol {p} missing fields: {', '.join(missing)}")
    if raw["label_field"] not in LABEL_FIELDS:
        raise ProtocolError(
            f"label_field must be one of {LABEL_FIELDS}, got {raw['label_field']!r}"
        )
    rq = raw["research_question"]
    if not isinstance(rq, str) or not rq.strip():
        raise ProtocolError("research_question must be a non-empty string")
    known = set(PROTOCOL_REQUIRED) | {"notes"}
    inclusion_stages, inclusion_rationales, inclusion_absence = _criteria_stage_fields(
        raw["inclusion_criteria"], "inclusion_criteria", default_absence=False
    )
    exclusion_stages, exclusion_rationales, exclusion_absence = _criteria_stage_fields(
        raw["exclusion_criteria"], "exclusion_criteria", default_absence=True
    )
    provenance = raw.get("criteria_provenance")
    if isinstance(provenance, dict):
        _check_negation_stage_invariant(
            provenance.get("exclusion"), inclusion_stages, exclusion_stages, p
        )
    return Protocol(
        dataset_id=str(raw["dataset_id"]),
        source_review_doi=str(raw["source_review_doi"]),
        research_question=rq.strip(),
        inclusion_criteria=_str_list(raw["inclusion_criteria"], "inclusion_criteria"),
        exclusion_criteria=_str_list(raw["exclusion_criteria"], "exclusion_criteria"),
        label_field=raw["label_field"],
        notes=str(raw.get("notes") or ""),
        extra={k: v for k, v in raw.items() if k not in known},
        path=p,
        inclusion_stages=inclusion_stages,
        exclusion_stages=exclusion_stages,
        inclusion_stage_rationales=inclusion_rationales,
        exclusion_stage_rationales=exclusion_rationales,
        inclusion_absence=inclusion_absence,
        exclusion_absence=exclusion_absence,
    )


QUERY_MODES: tuple[str, ...] = ("rq", "rq+criteria")


def protocol_query_text(p: Protocol, mode: str = "rq") -> str:
    """Query document for the TF-IDF baseline.

    ``"rq"`` (default, as in the revision design: similarity between record and research
    question) uses the research question alone; ``"rq+criteria"`` appends the inclusion
    criteria. The chosen mode is recorded in ``<dataset>_baselines.json``.
    """
    if mode not in QUERY_MODES:
        raise ValueError(f"query mode must be one of {QUERY_MODES}, got {mode!r}")
    if mode == "rq":
        return p.research_question
    return " ".join([p.research_question, *p.inclusion_criteria])


# --------------------------------------------------------------------------------------
# Environment / backend bootstrap (no secrets are ever printed)
# --------------------------------------------------------------------------------------


def load_dotenv_values(path: str | Path) -> dict[str, str]:
    """Minimal ``KEY=VALUE`` parser (comments and blank lines ignored, quotes stripped)."""
    values: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return values
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def export_env_from_dotenv(
    keys: Sequence[str], path: str | Path | None = None, *, override: bool = False
) -> list[str]:
    """Copy selected keys from the repo-root ``.env`` into ``os.environ``. Returns key names."""
    values = load_dotenv_values(path or (REPO_ROOT / ".env"))
    exported: list[str] = []
    for key in keys:
        if key in values and values[key] and (override or not os.environ.get(key)):
            os.environ[key] = values[key]
            exported.append(key)
    return exported


def add_backend_to_path(backend_root: Path | None = None) -> Path:
    """Put ``backend/`` on ``sys.path`` **and make it the working directory**.

    ``app.config.Settings`` reads ``.env`` relative to the current directory
    (``SettingsConfigDict(env_file=".env")``) and forbids extra keys, so importing ``app.*``
    from the repository root (whose ``.env`` carries deployment-only keys) fails with a
    pydantic ``ValidationError``. Callers must therefore resolve every relative CLI path
    (:func:`resolve_path_args`) *before* calling this.
    """
    root = (backend_root or BACKEND_ROOT).resolve()
    if not (root / "app").is_dir():
        raise FileNotFoundError(f"backend package not found at {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    return root


def resolve_path_args(args: Any) -> Any:
    """Make every ``pathlib.Path`` attribute of an argparse namespace absolute (in place).

    Called at the top of every CLI ``main`` so that ``--results-dir out`` given from the
    repository root still lands in the original cwd after :func:`add_backend_to_path`. A
    ``list[Path]`` attribute (a repeatable ``action="append"`` flag such as
    ``build_hss_set.py --exclude-sources``) is resolved element-wise; any other list is
    left untouched.
    """
    for key, value in list(vars(args).items()):
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
        elif isinstance(value, list) and value and all(isinstance(v, Path) for v in value):
            setattr(args, key, [v.resolve() for v in value])
    return args


def import_backend_settings() -> Any:
    """``app.config.settings`` with any failure turned into a readable ``SystemExit``."""
    try:
        from app.config import settings
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError, ImportError, ...
        raise SystemExit(
            f"backend settings failed to load from {Path.cwd()}: {type(exc).__name__}: {exc}"
        ) from exc
    return settings


# --------------------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Stopwatch:
    """``with Stopwatch() as sw: ...; sw.seconds``.

    ``seconds`` is *live*: inside the block it is the time elapsed so far, after the block
    it is frozen at the block's duration. ``seconds`` must be updated on every read, not only
    in ``__exit__``, or a read from inside the block would return the constructor default 0.0
    and every recorded latency would be 0.0.
    """

    def __init__(self) -> None:
        self._t0: float | None = None
        self._elapsed: float | None = None

    def __enter__(self) -> Stopwatch:
        self._t0 = time.perf_counter()
        self._elapsed = None
        return self

    def __exit__(self, *exc: object) -> None:
        self._elapsed = self.seconds

    @property
    def seconds(self) -> float:
        if self._elapsed is not None:
            return self._elapsed
        if self._t0 is None:
            return 0.0
        return time.perf_counter() - self._t0


def fmt(value: Any, digits: int = 3) -> str:
    """Format a metric for markdown tables (``-`` for None)."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)
