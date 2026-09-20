"""Binary metrics, WSS, kappa and multi-class helpers in ``common``."""

from __future__ import annotations

import math

import pytest

from common import (
    BinaryCounts,
    binary_metrics,
    cohens_kappa,
    confusion_counts,
    confusion_matrix,
    f1,
    inclusion_rate,
    macro_f1,
    per_class_prf,
    percent_agreement,
    precision,
    predictions_matching_count,
    recall,
    specificity,
    wilson_interval,
    wss,
    wss_at_95,
    wss_at_achieved_recall,
    wss_from_ranking,
)

LABELS = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
PREDS = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]  # tp=3 fn=1 fp=1 tn=5


def test_confusion_counts_basic():
    c = confusion_counts(LABELS, PREDS)
    assert (c.tp, c.fp, c.tn, c.fn, c.skipped) == (3, 1, 5, 1, 0)
    assert c.n == 10


def test_confusion_counts_skips_none_unless_mapped():
    preds = PREDS[:-1] + [None]
    c = confusion_counts(LABELS, preds)
    assert c.skipped == 1 and c.n == 9
    c2 = confusion_counts(LABELS, preds, none_as=1)
    assert c2.skipped == 0 and c2.fp == 2


def test_confusion_counts_rejects_bad_values():
    with pytest.raises(ValueError):
        confusion_counts([1, 2], [1, 1])
    with pytest.raises(ValueError):
        confusion_counts([1], [1, 0])


def test_binary_metric_values():
    c = confusion_counts(LABELS, PREDS)
    assert recall(c) == 0.75
    assert precision(c) == 0.75
    assert f1(c) == pytest.approx(0.75)
    assert specificity(c) == pytest.approx(5 / 6)
    assert inclusion_rate(c) == 0.4


def test_metrics_return_none_on_zero_denominator():
    c = BinaryCounts()
    assert recall(c) is None and precision(c) is None and f1(c) is None
    assert inclusion_rate(c) is None and wss(c, 0.95) is None
    m = binary_metrics(c)
    assert m["n"] == 0 and m["recall"] is None


def test_wss_cohen_definition():
    # (TN + FN)/N - (1 - R): 6/10 - 0.25 = 0.35 at achieved recall 0.75
    c = confusion_counts(LABELS, PREDS)
    assert wss_at_achieved_recall(c) == pytest.approx(0.35)
    assert wss_at_95(c) is None  # achieved recall below 0.95
    perfect = confusion_counts([1, 1, 0, 0, 0], [1, 1, 1, 0, 0])
    assert wss_at_95(perfect) == pytest.approx(2 / 5 - 0.05)
    assert wss(perfect, 1.0) == pytest.approx(0.4)


def test_wss_from_ranking():
    scores = [0.9, 0.8, 0.1, 0.7, 0.05]
    labels = [1, 0, 0, 1, 0]
    # ranked order: 0, 1, 3, 2, 4 -> both positives found after screening 3 of 5
    assert wss_from_ranking(scores, labels, 1.0) == pytest.approx(2 / 5)
    assert wss_from_ranking(scores, labels, 0.5) == pytest.approx(4 / 5 - 0.5)
    assert wss_from_ranking([], [], 0.95) is None
    assert wss_from_ranking([0.1], [0], 0.95) is None


def test_predictions_matching_count_exact_k_and_ties():
    scores = [0.2, 0.9, 0.9, 0.5]
    preds, thr = predictions_matching_count(scores, 2)
    assert preds == [0, 1, 1, 0] and thr == 0.9
    preds, thr = predictions_matching_count(scores, 3)
    assert sum(preds) == 3 and thr == 0.5
    assert predictions_matching_count(scores, 0) == ([0, 0, 0, 0], None)
    preds, _ = predictions_matching_count(scores, 99)
    assert sum(preds) == 4


def test_percent_agreement_and_kappa_known_values():
    a = [1, 1, 0, 0, 1]
    b = [1, 0, 0, 0, 1]
    assert percent_agreement(a, b) == 0.8
    # po = 0.8; pe = 0.6*0.4 + 0.4*0.6 = 0.48; kappa = 0.32/0.52
    assert cohens_kappa(a, b) == pytest.approx(0.32 / 0.52)
    assert cohens_kappa([1, 0, 1], [1, 0, 1]) == 1.0
    assert cohens_kappa([1, 0], [0, 1]) == pytest.approx(-1.0)


def test_kappa_edge_cases():
    assert cohens_kappa([], []) is None
    assert cohens_kappa([None, None], [1, 0]) is None
    assert cohens_kappa([1, 1, 1], [1, 1, 1]) is None  # pe == 1: undefined
    assert cohens_kappa(["a", "b", None], ["a", "b", "c"]) == 1.0
    assert math.isclose(cohens_kappa(["x", "y", "z"], ["x", "y", "z"]), 1.0)


def test_confusion_matrix_and_per_class_prf():
    gold = ["SUPPORT", "SUPPORT", "CONTRADICT", "NEI"]
    pred = ["verified", "needs_nuance", "unsupported", "unsupported"]
    m = confusion_matrix(gold, pred, ["SUPPORT", "CONTRADICT", "NEI"], ["verified", "unsupported"])
    assert m["SUPPORT"]["verified"] == 1
    assert m["SUPPORT"]["needs_nuance"] == 1  # unknown predicted class appended, never dropped
    assert m["NEI"]["unsupported"] == 1

    mapped_gold = ["verified", "verified", "unsupported", "unsupported"]
    prf = per_class_prf(mapped_gold, pred, ["verified", "unsupported"])
    assert prf["verified"]["recall"] == 0.5  # needs_nuance counts as a miss
    assert prf["verified"]["precision"] == 1.0
    assert prf["unsupported"]["recall"] == 1.0 and prf["unsupported"]["precision"] == 1.0
    assert macro_f1(prf) == pytest.approx((2 / 3 + 1.0) / 2)


def test_kappa_undefined_for_constant_identical_runs():
    from common import cohens_kappa, kappa_undefined

    assert cohens_kappa([1, 1, 1], [1, 1, 1]) is None
    assert kappa_undefined([1, 1, 1], [1, 1, 1]) is True
    assert kappa_undefined([1, 0, 1], [1, 0, 1]) is False
    assert cohens_kappa([1, 1, 1], [0, 0, 0]) == 0.0
    assert cohens_kappa([1, 0, 1], [1, 0, 1]) == 1.0


def test_protocol_query_text_modes():
    from common import Protocol, protocol_query_text

    p = Protocol("X", "d", "What works?", ["crit A", "crit B"], ["e"], "label_included")
    assert protocol_query_text(p) == "What works?"
    assert protocol_query_text(p, "rq") == "What works?"
    assert protocol_query_text(p, "rq+criteria") == "What works? crit A crit B"
    with pytest.raises(ValueError):
        protocol_query_text(p, "other")


def test_never_predicted_class_with_support_scores_f1_zero_not_dropped():
    """A class the system never predicts has precision undefined but F1 = 0 by convention
    (sklearn zero_division=0); macro-F1 averages over every class with gold support, so a
    degenerate single-class system cannot score higher than one predicting both classes."""
    gold = ["verified", "unsupported", "unsupported"]
    pred = ["unsupported", "unsupported", "unsupported"]
    prf = per_class_prf(gold, pred, ["verified", "unsupported"])
    assert prf["verified"]["precision"] is None and prf["verified"]["recall"] == 0.0
    assert prf["verified"]["f1"] == 0.0
    assert prf["unsupported"]["f1"] == pytest.approx(0.8)
    assert macro_f1(prf) == pytest.approx(0.8 / 2)
    # a class with no gold support and no prediction is neither penalised nor counted
    prf = per_class_prf(["a", "a"], ["a", "a"], ["a", "b"])
    assert prf["b"]["f1"] is None and prf["b"]["support"] == 0
    assert macro_f1(prf) == pytest.approx(1.0)
    assert macro_f1({}) is None


# ---------------------------------------------------------------- Wilson interval


def test_wilson_interval_known_values():
    """n=10, count=5 (p=0.5) is a commonly cited worked example of the Wilson score interval
    at the 95 % level; n=10, count=10 (p=1.0) must not overshoot 1.0 at either bound."""
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.23659309, abs=1e-6)
    assert hi == pytest.approx(0.76340691, abs=1e-6)
    lo2, hi2 = wilson_interval(54, 60)
    assert lo2 == pytest.approx(0.79850535, abs=1e-6)
    assert hi2 == pytest.approx(0.95335717, abs=1e-6)
    lo3, hi3 = wilson_interval(10, 10)
    assert hi3 == 1.0 and 0.0 < lo3 < 1.0
    lo4, hi4 = wilson_interval(0, 10)
    assert lo4 == pytest.approx(0.0, abs=1e-9) and 0.0 < hi4 < 1.0


def test_wilson_interval_symmetric_at_half_and_widens_as_alpha_shrinks():
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(1.0 - hi)
    lo90, hi90 = wilson_interval(5, 10, alpha=0.10)
    assert lo90 > lo and hi90 < hi  # a smaller alpha (wider confidence) widens the interval


def test_wilson_interval_zero_n_returns_none():
    assert wilson_interval(0, 0) == (None, None)


# ---------------------------------------------------------------- Unicode quote check


def test_quote_is_verbatim_folds_unicode_punctuation_to_ascii():
    """NFKC alone folds none of U+2010, the en/em dashes or the curly quotes to ASCII,
    which is why common.py's fold map is explicit. Record 1065's case: the shown text
    carries a U+2010 hyphen where the model's quote used a plain ASCII hyphen (or vice
    versa); the guard's outcome must not flip on that one code point."""
    from common import quote_is_verbatim

    shown = "A double‐blind randomized trial of the intervention."
    assert quote_is_verbatim("double-blind randomized", shown, casefold=True) is True
    assert quote_is_verbatim("double‐blind randomized", shown, casefold=True) is True

    curly = "The authors’ conclusion was “not significant” overall."
    assert quote_is_verbatim("authors' conclusion", curly, casefold=True) is True
    assert quote_is_verbatim('"not significant"', curly, casefold=True) is True

    dash_variants = "a‑b‒c–d—e―f−g"
    assert quote_is_verbatim("a-b-c-d-e-f-g", dash_variants, casefold=True) is True

    spaced = "figures shown here exactly…"
    assert quote_is_verbatim("figures shown here exactly...", spaced, casefold=True) is True


def test_quote_is_verbatim_unicode_folding_agrees_with_the_backend_copy(monkeypatch):
    """The backend keeps a private copy of this normalisation
    (``app.agents.relevance_screener_agent._quote_is_verbatim``) because it cannot import
    ``evaluation/``; this fixture (including record 1065's U+2010 case) must agree between
    the two on every case, mirroring the pre-existing agreement test in
    ``backend/tests/test_relevance_screener_agent.py``.

    The "app" package is registered explicitly by file location, with
    ``submodule_search_locations`` pinned at the real ``backend/app`` directory, rather than
    relying on ``sys.path`` order: other tests in this suite install their own fake ``app``
    packages via ``sys.path``/``sys.modules`` tricks (claim-verification fixtures
    elsewhere in this suite), and
    a plain ``import app.agents...`` here could otherwise resolve "app" to a stale fake left
    over from one of them.
    """
    import importlib.util
    import os
    import sys

    from common import REPO_ROOT, quote_is_verbatim

    os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")
    backend_root = REPO_ROOT / "backend"
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(
        "app", backend_root / "app" / "__init__.py",
        submodule_search_locations=[str(backend_root / "app")],
    )
    app_module = importlib.util.module_from_spec(spec)
    sys.modules["app"] = app_module
    spec.loader.exec_module(app_module)
    cwd_before = os.getcwd()
    os.chdir(backend_root)  # app.config's Settings reads .env relative to the cwd
    try:
        from app.agents.relevance_screener_agent import (
            _quote_is_verbatim as backend_quote_is_verbatim,
        )
    finally:
        os.chdir(cwd_before)
        for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
            monkeypatch.delitem(sys.modules, name, raising=False)

    shown = "Title: T\nAbstract: A double‐blind trial reported “significant” gains."
    fixtures = [
        ("double-blind trial", True),
        ("double‐blind trial", True),
        ('"significant" gains', True),
        ("a phrase never shown", False),
        ("", False),
    ]
    for quote, expected in fixtures:
        assert quote_is_verbatim(quote, shown, casefold=True) == expected
        assert backend_quote_is_verbatim(quote, shown) == expected
