"""Pure helpers of the screening scripts (run_screening, baselines, wos_gate_effect)."""

from __future__ import annotations

import pytest

import run_screening as rs
import wos_gate_effect as wg
from common import Protocol

RECORDS = [
    {
        "record_id": i,
        "title": f"Paper {i}",
        "abstract": "text",
        "label_included": i % 3 == 0,
        "label_abstract_screening": i % 2,
    }
    for i in range(1, 8)
]
PROTOCOL = Protocol(
    dataset_id="X",
    source_review_doi="10.1/x",
    research_question="Do reminders improve guideline adherence among clinicians?",
    inclusion_criteria=["reminders"],
    exclusion_criteria=["patients only"],
    label_field="label_abstract_screening",
)


def test_make_batches_and_gold_label():
    batches = rs.make_batches(RECORDS, 3)
    assert [len(b) for b in batches] == [3, 3, 1]
    assert batches[2][0]["record_id"] == 7
    with pytest.raises(ValueError):
        rs.make_batches(RECORDS, 0)
    assert rs.gold_label(RECORDS[0], "label_abstract_screening") == 1
    assert rs.gold_label({"record_id": 1}, "label_included") is None


def test_rows_for_batch_success_and_failure():
    batch = RECORDS[:2]
    ok = {
        "include": [True, False],
        "reasons": ["matches", "off-topic"],
        "model_reported": "deepseek-v4",
        "system_fingerprint": "fp",
        "provider_response_id": "id1",
        "latency_s": 1.5,
        "input_tokens": 100,
        "output_tokens": 20,
        "attempts": 1,
    }
    rows = rs.rows_for_batch(batch, 4, ok, "label_abstract_screening")
    assert [r["predicted"] for r in rows] == [1, 0]
    assert rows[0]["reason"] == "matches" and rows[0]["batch_index"] == 4
    assert rows[0]["label"] == 1 and rows[1]["label"] == 0
    assert rows[0]["model_reported"] == "deepseek-v4" and rows[1]["latency_s"] == 1.5

    failed = rs.rows_for_batch(
        batch, 5, {"error": "ScreeningError: boom", "attempts": 3}, "label_included"
    )
    assert all(r["predicted"] is None for r in failed)
    assert failed[0]["reason"] == "UNSCREENED: ScreeningError: boom" and failed[0]["attempts"] == 3


def test_build_meta_aggregates_per_batch():
    ok_rows = rs.rows_for_batch(
        RECORDS[:2],
        0,
        {
            "include": [True, True],
            "reasons": [],
            "model_reported": "m1",
            "latency_s": 2.0,
            "input_tokens": 10,
            "output_tokens": 5,
            "attempts": 1,
        },
        "label_included",
    )
    bad_rows = rs.rows_for_batch(RECORDS[2:4], 1, {"error": "x", "attempts": 3}, "label_included")
    price = {"input_per_mtok": 1.0, "output_per_mtok": 2.0}
    meta = rs.build_meta(
        ok_rows + bad_rows,
        existing={"started": "t0", "sessions": [{"started": "t0", "finished": "t1"}]},
        dataset="X",
        run="A",
        protocol=PROTOCOL,
        n_records=4,
        n_batches=2,
        batch_size=2,
        concurrency=1,
        model_configured="deepseek-chat",
        temperature=0.0,
        prompt_versions=["sha256:abc", "sha256:abc"],
        price=price,
        dry_run=False,
        session_started="t2",
        limit=None,
    )
    assert meta["failures"] == 1 and meta["n_batches_completed"] == 1
    assert meta["n_unscreened_records"] == 2 and meta["n_rows"] == 4
    assert meta["total_input_tokens"] == 10 and meta["total_output_tokens"] == 5
    assert meta["total_cost_flat"] == pytest.approx((10 * 1.0 + 5 * 2.0) / 1e6)
    assert meta["model_reported"] == ["m1"] and meta["prompt_version"] == ["sha256:abc"]
    assert meta["started"] == "t0" and len(meta["sessions"]) == 2
    assert meta["latency_s"]["median"] == 2.0


def test_dry_run_screener_is_deterministic():
    import asyncio

    screener = rs.DryRunScreener()
    papers = [
        {"title": "Reminders for clinicians", "abstract": ""},
        {"title": "Unrelated", "abstract": ""},
    ]
    out = asyncio.run(screener(PROTOCOL, papers))
    assert out["include"] == [True, False] and out["model_reported"] == "dry-run"


def test_baselines_metrics_match_llm_count():
    pytest.importorskip("sklearn")
    import baselines as bl

    records = [
        {
            "record_id": 1,
            "title": "Reminders improve guideline adherence",
            "abstract": "clinicians",
        },
        {"record_id": 2, "title": "Deep learning for images", "abstract": "convolution"},
        {"record_id": 3, "title": "Nudging clinicians with reminders", "abstract": "adherence"},
        {"record_id": 4, "title": "Soil chemistry", "abstract": "nitrogen"},
    ]
    labels = [1, 0, 1, 0]
    out = bl.baseline_metrics(records, labels, "reminders guideline adherence clinicians", k=2)
    assert out["include_all"]["recall"] == 1.0 and out["include_all"]["precision"] == 0.5
    assert out["tfidf"]["metrics"]["inclusion_rate"] == 0.5  # exactly k included
    assert out["tfidf"]["metrics"]["recall"] == 1.0
    assert out["tfidf"]["wss_at_95_ranking"] == pytest.approx(2 / 4 - 0.05)
    run_rows = [{"record_id": 1, "predicted": 1}, {"record_id": 2, "predicted": 0}]
    assert bl.llm_inclusion_count(run_rows) == 1
    chosen, lab = bl.select_records(
        [dict(r, label_included=int(x)) for r, x in zip(records, labels, strict=True)],
        run_rows,
        Protocol("X", "d", "q", ["i"], ["e"], "label_included"),
    )
    assert [r["record_id"] for r in chosen] == [1, 2] and lab == [1, 0]


# ---------------------------------------------------------------- baselines --label


def test_baselines_defaults_to_primary_label_and_rq_plus_criteria(tmp_path):
    pytest.importorskip("sklearn")
    import baselines as bl

    args = bl.parse_args(["--dataset", "X"])
    assert args.label == "primary" and args.query == "rq+criteria"
    assert bl.parse_args(["--dataset", "X", "--label", "sensitivity"]).label == "sensitivity"


def test_select_records_label_field_override_and_default(tmp_path):
    pytest.importorskip("sklearn")
    import baselines as bl

    records = [
        {"record_id": 1, "label_included": 1, "label_abstract_screening": 0},
        {"record_id": 2, "label_included": 0, "label_abstract_screening": 1},
    ]
    protocol = Protocol("X", "d", "q", ["i"], ["e"], "label_abstract_screening")
    chosen, labels = bl.select_records(records, None, protocol)
    assert labels == [0, 1]  # default: protocol.label_field
    chosen, labels = bl.select_records(records, None, protocol, label_field="label_included")
    assert labels == [1, 0]  # explicit override wins


def test_baselines_main_scores_primary_label_by_default(tmp_path):
    pytest.importorskip("sklearn")
    import json

    import baselines as bl

    data = tmp_path / "data"
    protos = tmp_path / "protocols"
    results = tmp_path / "results"
    for d in (data, protos, results):
        d.mkdir()
    records = [
        {"record_id": 1, "title": "Reminders improve guideline adherence", "abstract": "x",
         "label_included": 1, "label_abstract_screening": 0},
        {"record_id": 2, "title": "Deep learning for images", "abstract": "y",
         "label_included": 0, "label_abstract_screening": 1},
    ]
    with (data / "X.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    proto = {
        "dataset_id": "X", "source_review_doi": "10.1/x", "research_question": "q",
        "inclusion_criteria": ["reminders"], "exclusion_criteria": ["e"],
        "label_field": "label_abstract_screening", "primary_label": "label_included",
    }
    (protos / "X.json").write_text(json.dumps(proto), encoding="utf-8")
    rc = bl.main(["--dataset", "X", "--k", "1", "--data-dir", str(data),
                  "--protocols-dir", str(protos), "--results-dir", str(results)])
    assert rc == 0
    out = json.loads((results / "X_baselines.json").read_text(encoding="utf-8"))
    assert out["label_field"] == "label_included" and out["label_choice"] == "primary"
    assert out["query_mode"] == "rq+criteria" and "reminders" in out["query_text"]
    rc = bl.main(["--dataset", "X", "--k", "1", "--label", "sensitivity",
                  "--data-dir", str(data), "--protocols-dir", str(protos),
                  "--results-dir", str(results)])
    out = json.loads((results / "X_baselines.json").read_text(encoding="utf-8"))
    assert out["label_field"] == "label_abstract_screening" and out["label_choice"] == "sensitivity"


WOS_CSV = (
    '"Journal title","ISSN","eISSN","Publisher name","Publisher address","Languages",'
    '"Web of Science Categories"\n'
    '"J ONE","0001-3072","1467-6281","P","A","English","Business"\n'
    '"J TWO","","2056-5127","P","A","English","Management"\n'
)


def test_normalise_issn_and_parse_wos():
    assert wg.normalise_issn(" 1234567x ") == "1234-567X"
    assert wg.normalise_issn("1234-5678") == "1234-5678"
    assert wg.normalise_issn("12345") is None and wg.normalise_issn("") is None
    assert wg.normalise_issn("ABCD-EFGH") is None
    assert wg.parse_wos_issns(WOS_CSV) == {"0001-3072", "1467-6281", "2056-5127"}


def test_gate_effect_counts():
    collections = {"SSCI": {"0001-3072"}, "AHCI": {"2056-5127"}}
    records = [
        {"record_id": 1, "issn": ["0001-3072"], "label_included": 1},
        {"record_id": 2, "issn": ["9999-9999"], "label_included": 1},
        {"record_id": 3, "issn": [], "label_included": 1},
        {"record_id": 4, "issn": ["2056-5127", "0001-3072"], "label_included": 0},
    ]
    out = wg.gate_effect(records, collections, "label_included")
    inc = out["included_records"]
    assert inc["n"] == 3 and inc["n_with_issn"] == 2 and inc["n_unresolved"] == 1
    assert inc["n_in_wos_any"] == 1 and inc["share_in_wos_any"] == pytest.approx(1 / 3)
    assert inc["share_in_wos_any_of_resolved"] == 0.5
    assert inc["per_collection"]["SSCI"]["n"] == 1 and inc["per_collection"]["AHCI"]["n"] == 0
    allr = out["all_records"]
    assert allr["n_in_wos_any"] == 2 and allr["per_collection"]["AHCI"]["n"] == 1


def _ok_outcome(**over):
    base = {
        "include": [True, False],
        "reasons": ["yes", "no"],
        "model_reported": "deepseek-v4-flash",
        "system_fingerprint": "fp",
        "provider_response_id": "id",
        "latency_s": 1.0,
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "attempts": 1,
        "called_at": "2026-09-02T12:00:00+00:00",
    }
    base.update(over)
    return base


def test_rows_carry_called_at_has_abstract_and_cache_field():
    batch = [dict(RECORDS[0], abstract="some text"), dict(RECORDS[1], abstract="")]
    rows = rs.rows_for_batch(batch, 0, _ok_outcome(), "label_abstract_screening")
    assert rows[0]["called_at"] == "2026-09-02T12:00:00+00:00"
    assert rows[0]["has_abstract"] is True and rows[1]["has_abstract"] is False
    assert rows[0]["cache_read_tokens"] is None
    assert list(rows[0].keys()) == [
        "record_id", "label", "label_included", "label_abstract_screening", "has_abstract",
        "has_title", "title", "type", "is_paratext", "predicted", "reason", "status",
        "criterion", "quote", "guard_applied", "guard_reason", "to_confirm", "anchors",
        "second_pass", "second_pass_call_id", "second_pass_model_reported",
        "second_pass_input_tokens", "second_pass_output_tokens", "padded_include",
        "padded_decision", "pass_number", "batch_index", "called_at", "call_id", "latency_s",
        "input_tokens", "output_tokens", "cache_read_tokens", "model_reported",
        "system_fingerprint", "provider_response_id", "attempts", "calls",
        "reask_model_reported", "reask_system_fingerprint", "reask_error",
    ]
    assert rows[0]["pass_number"] == 1  # default: every pre-fix call site is pass 1


def test_rows_for_batch_carries_the_anchor_ledger_and_defaults_it_to_empty():
    """The ``anchors`` outcome key (v3 only) lands on the matching row unchanged; an outcome
    that never mentions it (every v1/v2 call site) still gets ``[]`` per row, the same
    convention ``to_confirm`` already uses."""
    batch = [dict(RECORDS[0], abstract="some text"), dict(RECORDS[1], abstract="text")]
    ledger = [
        [{"slot": "population", "quote": "some text"}],
        [],
    ]
    rows = rs.rows_for_batch(
        batch, 0, _ok_outcome(anchors=ledger), "label_abstract_screening",
    )
    assert rows[0]["anchors"] == ledger[0]
    assert rows[1]["anchors"] == []

    rows_no_ledger = rs.rows_for_batch(batch, 0, _ok_outcome(), "label_abstract_screening")
    assert rows_no_ledger[0]["anchors"] == []
    assert rows_no_ledger[1]["anchors"] == []


def test_rows_for_batch_records_an_explicit_pass_number():
    """A re-ask pass records which pass produced the row (2: the batch-of-3 re-ask, 3: the
    single-record re-ask), so a padded row's history is visible."""
    batch = [dict(RECORDS[0], abstract="some text")]
    rows = rs.rows_for_batch(batch, 0, _ok_outcome(include=[True]), "label_included",
                              pass_number=3)
    assert rows[0]["pass_number"] == 3


def test_rows_for_batch_forwards_calls_and_reask_identity():
    """``calls``, ``reask_model_reported``, ``reask_system_fingerprint`` and ``reask_error``
    all reach the row unchanged."""
    batch = [dict(RECORDS[0], abstract="some text")]
    outcome = _ok_outcome(
        include=[True],
        calls=2,
        reask_model_reported="deepseek-v4-flash-2",
        reask_system_fingerprint="fp2",
        reask_error=None,
    )
    rows = rs.rows_for_batch(batch, 0, outcome, "label_included")
    assert rows[0]["calls"] == 2
    assert rows[0]["reask_model_reported"] == "deepseek-v4-flash-2"
    assert rows[0]["reask_system_fingerprint"] == "fp2"
    assert rows[0]["reask_error"] is None

    failed_reask = rs.rows_for_batch(
        batch, 0, _ok_outcome(include=[True], reask_error="TimeoutError: timed out"),
        "label_included",
    )
    assert failed_reask[0]["reask_error"] == "TimeoutError: timed out"


def test_rows_for_batch_defaults_calls_to_one_and_reask_fields_to_none():
    """An outcome that never mentions ``calls``/``reask_*`` (every pre-fix outcome shape,
    including a legacy test double) still constructs a row cleanly."""
    batch = [dict(RECORDS[0], abstract="some text")]
    rows = rs.rows_for_batch(batch, 0, _ok_outcome(include=[True]), "label_included")
    assert rows[0]["calls"] == 1
    assert rows[0]["reask_model_reported"] is None
    assert rows[0]["reask_system_fingerprint"] is None
    assert rows[0]["reask_error"] is None


def test_build_meta_uses_tiered_call_cost_and_missing_abstract_share():
    from common import price_record

    off_rows = rs.rows_for_batch(
        [dict(RECORDS[0], abstract="a"), dict(RECORDS[1], abstract="")], 0,
        _ok_outcome(called_at="2026-09-02T12:00:00+00:00"), "label_included",
    )
    peak_rows = rs.rows_for_batch(
        [dict(RECORDS[2], abstract="b"), dict(RECORDS[3], abstract="c")], 1,
        _ok_outcome(called_at="2026-09-02T02:00:00+00:00"), "label_included",
    )
    kwargs = dict(
        existing=None, dataset="X", run="A", protocol=PROTOCOL, n_records=4, n_batches=2,
        batch_size=2, concurrency=1, model_configured="deepseek-chat", temperature=0.0,
        prompt_versions=["v"], dry_run=False, session_started="t", limit=None,
    )
    meta = rs.build_meta(off_rows + peak_rows, price=price_record("deepseek-chat", None, None),
                         **kwargs)
    assert meta["total_cost"] == pytest.approx((0.22 + 0.66) + (0.44 + 1.32))
    assert meta["cost_basis"].startswith("list price, tier by call time, cache-miss assumed")
    assert meta["total_cost_flat"] is None
    assert meta["n_missing_abstract"] == 1 and meta["share_missing_abstract"] == 0.25
    assert meta["price"]["price_model"] == "deepseek-v4-flash"
    flat = rs.build_meta(off_rows + peak_rows, price=price_record("deepseek-chat", 1.0, 2.0),
                         **kwargs)
    assert flat["total_cost_flat"] == pytest.approx(2 * (1.0 + 2.0))
    assert flat["total_cost"] == meta["total_cost"]
    # model_reported wins over model_configured; unknown model -> None cost
    odd = rs.rows_for_batch(RECORDS[4:6], 2, _ok_outcome(model_reported="other"), "label_included")
    assert rs.build_meta(odd, price=price_record("deepseek-chat", None, None), **kwargs)[
        "total_cost"
    ] is None


class _FakeScreener:
    error_type = RuntimeError
    model_configured = "fake"

    def __init__(self, temperature=0.0):
        self.temperature = temperature
        self.calls = 0

    async def __call__(self, protocol, papers):
        self.calls += 1
        return {
            "include": [True] * len(papers),
            "reasons": ["r"] * len(papers),
            "model_reported": "deepseek-v4-flash",
            "system_fingerprint": None,
            "provider_response_id": None,
            "temperature": self.temperature,
            "prompt_version": "v1",
            "input_tokens": 10,
            "output_tokens": 5,
        }


def _setup_dataset(tmp_path):
    import json

    data = tmp_path / "data"
    data.mkdir()
    with (data / "X.jsonl").open("w", encoding="utf-8") as fh:
        for r in RECORDS:
            fh.write(json.dumps(dict(r, label_included=int(r["label_included"]))) + "\n")
    protos = tmp_path / "protocols"
    protos.mkdir()
    (protos / "X.json").write_text(
        json.dumps(
            {
                "dataset_id": "X",
                "source_review_doi": "10.1/x",
                "research_question": PROTOCOL.research_question,
                "inclusion_criteria": PROTOCOL.inclusion_criteria,
                "exclusion_criteria": PROTOCOL.exclusion_criteria,
                "label_field": "label_abstract_screening",
            }
        ),
        encoding="utf-8",
    )
    return data, protos, tmp_path / "results"


def test_temperature_guard_exits_unless_allowed(tmp_path):
    import asyncio

    data, protos, results = _setup_dataset(tmp_path)
    argv = ["--dataset", "X", "--run", "A", "--batch-size", "3", "--confirm-cost",
            "--data-dir", str(data), "--protocols-dir", str(protos), "--results-dir",
            str(results)]
    with pytest.raises(SystemExit) as exc:
        asyncio.run(rs.main_async(rs.parse_args(argv), screener=_FakeScreener(0.7)))
    assert exc.value.code == 2
    rc = asyncio.run(
        rs.main_async(rs.parse_args(argv + ["--allow-nonzero-temperature"]),
                      screener=_FakeScreener(0.7))
    )
    assert rc == 0


def test_backend_guard_rejects_unmerged_backend(tmp_path, monkeypatch):
    import sys

    fake = tmp_path / "backend"
    (fake / "app" / "agents").mkdir(parents=True)
    (fake / "app" / "__init__.py").write_text("", encoding="utf-8")
    (fake / "app" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    (fake / "app" / "agents" / "relevance_screener_agent.py").write_text(
        "class ScreeningError(Exception):\n    pass\n\n"
        "async def screen_papers(q, papers, inc, exc):\n    return [True] * len(papers)\n",
        encoding="utf-8",
    )
    (fake / "app" / "config.py").write_text("settings = object()\n", encoding="utf-8")
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(fake))
    with pytest.raises(SystemExit) as exc:
        rs.ProductionScreener(backend_root=fake)
    assert "does not expose the required screen_papers interface" in str(exc.value)
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_resume_skips_done_records_and_appends_session(tmp_path):
    import asyncio
    import json

    from common import read_json, read_jsonl

    data, protos, results = _setup_dataset(tmp_path)
    argv = ["--dataset", "X", "--run", "A", "--batch-size", "3", "--dry-run",
            "--data-dir", str(data), "--protocols-dir", str(protos), "--results-dir",
            str(results)]
    assert asyncio.run(rs.main_async(rs.parse_args(argv))) == 0
    out = results / "X_runA.jsonl"
    rows = read_jsonl(out)
    assert len(rows) == 7 and rows[0]["has_abstract"] is True and rows[0]["called_at"]
    # drop the last batch (record 7) and resume
    partial = [r for r in rows if r["batch_index"] < 2]
    out.write_text("".join(json.dumps(r) + "\n" for r in partial), encoding="utf-8")
    assert asyncio.run(rs.main_async(rs.parse_args(argv))) == 0
    rows = read_jsonl(out)
    ids = [r["record_id"] for r in rows]
    assert sorted(ids) == list(range(1, 8)) and len(ids) == len(set(ids))
    meta = read_json(results / "X_runA.meta.json")
    assert len(meta["sessions"]) == 2 and meta["n_rows"] == 7
    assert meta["temperature"] == 0.0 and meta["share_missing_abstract"] == 0.0
    assert meta["cost_basis"] and meta["total_cost"] is None  # dry-run model has no price


# ---------------------------------------------------------------- additional cases


def test_shuffle_records_is_seeded_and_breaks_file_order():
    # 30 records, positives at file indices 0-9 (the Smid_2020 situation)
    recs = [{"record_id": i, "label_included": int(i < 10)} for i in range(30)]
    shuffled = rs.shuffle_records(recs, "Smid_2020", 20260902)
    assert sorted(r["record_id"] for r in shuffled) == list(range(30))
    assert [r["record_id"] for r in shuffled] != list(range(30))
    pos_batches = {i // 10 for i, r in enumerate(shuffled) if r["label_included"] == 1}
    assert len(pos_batches) > 1  # positives no longer confined to batch 0
    assert rs.shuffle_records(recs, "Smid_2020", 20260902) == shuffled  # reproducible
    assert rs.shuffle_records(recs, "Smid_2020", 7) != shuffled  # seed matters
    assert rs.shuffle_records(recs, "Other", 20260902) != shuffled  # dataset in the key
    assert recs[0]["record_id"] == 0  # input untouched


def test_load_records_shuffles_before_limit(tmp_path):
    import json

    data = tmp_path / "X.jsonl"
    data.write_text(
        "".join(json.dumps({"record_id": i, "label_included": int(i < 3)}) + "\n"
                for i in range(30)),
        encoding="utf-8",
    )
    first = rs.load_records(data, 10, dataset="X", seed=20260902)
    assert len(first) == 10 and [r["record_id"] for r in first] != list(range(10))
    again = rs.load_records(data, None, dataset="X", seed=20260902)
    assert [r["record_id"] for r in again[:10]] == [r["record_id"] for r in first]


def _meta_kwargs(**over):
    base = dict(
        existing=None, dataset="X", run="A", protocol=PROTOCOL, n_records=2, n_batches=1,
        batch_size=2, concurrency=1, model_configured="deepseek-chat", temperature=0.0,
        prompt_versions=["v"], price={}, dry_run=False, session_started="t", limit=None,
        record_order_seed=20260902, retried_failed=0,
    )
    base.update(over)
    return base


def test_rows_carry_has_title_and_meta_counts_missing_titles():
    batch = [dict(RECORDS[0], title=""), dict(RECORDS[1], title="T")]
    rows = rs.rows_for_batch(batch, 0, _ok_outcome(), "label_abstract_screening")
    assert rows[0]["has_title"] is False and rows[1]["has_title"] is True
    assert rows[0]["padded_include"] is False
    meta = rs.build_meta(rows, **_meta_kwargs())
    assert meta["n_missing_title"] == 1 and meta["record_order_seed"] == 20260902
    assert meta["retried_failed"] == 0 and meta["n_padded_include"] == 0


def test_rows_flag_padded_no_decision_includes():
    outcome = _ok_outcome(include=[True, True], reasons=["yes", rs.NO_DECISION_REASON])
    rows = rs.rows_for_batch(RECORDS[:2], 0, outcome, "label_included")
    assert rows[0]["padded_include"] is False and rows[1]["padded_include"] is True
    assert rows[1]["predicted"] == 1  # production behaviour kept as-is in the row
    assert rs.build_meta(rows, **_meta_kwargs())["n_padded_include"] == 1


class _FlakyScreener(_FakeScreener):
    """Fails every call whose batch contains one of ``fail_titles`` until healed."""

    error_type = RuntimeError

    def __init__(self, fail_titles):
        super().__init__()
        self.fail_titles = set(fail_titles)
        self.healed = False

    async def __call__(self, protocol, papers):
        if not self.healed and any(p["title"] in self.fail_titles for p in papers):
            self.calls += 1
            raise RuntimeError("outage")
        return await super().__call__(protocol, papers)


async def _no_sleep(_seconds):
    return None


def test_retry_failed_reattempts_error_rows_once_without_duplicates(tmp_path, monkeypatch):
    import asyncio

    from common import read_json, read_jsonl

    monkeypatch.setattr(rs.asyncio, "sleep", _no_sleep)
    data, protos, results = _setup_dataset(tmp_path)
    base = ["--dataset", "X", "--run", "A", "--batch-size", "3", "--max-attempts", "2",
            "--record-order-seed", "0", "--confirm-cost", "--data-dir", str(data),
            "--protocols-dir", str(protos), "--results-dir", str(results)]
    screener = _FlakyScreener({"Paper 7"})
    asyncio.run(rs.main_async(rs.parse_args(base), screener=screener))
    rows = read_jsonl(results / "X_runA.jsonl")
    failed_ids = sorted(r["record_id"] for r in rows if r["predicted"] is None)
    assert failed_ids and len(rows) == 7
    screener.healed = True
    # without the flag: nothing is re-attempted
    calls_before = screener.calls
    asyncio.run(rs.main_async(rs.parse_args(base), screener=screener))
    assert screener.calls == calls_before
    assert sorted(r["record_id"] for r in read_jsonl(results / "X_runA.jsonl")
                  if r["predicted"] is None) == failed_ids
    # with the flag: exactly the failed records are re-run, once, no duplicates
    asyncio.run(rs.main_async(rs.parse_args(base + ["--retry-failed"]), screener=screener))
    rows = read_jsonl(results / "X_runA.jsonl")
    ids = [r["record_id"] for r in rows]
    assert len(ids) == 7 and len(set(ids)) == 7
    assert all(r["predicted"] is not None for r in rows)
    meta = read_json(results / "X_runA.meta.json")
    assert meta["retried_failed"] == len(failed_ids) and meta["failures"] == 0
    assert meta["sessions"][-1]["retried_failed"] == len(failed_ids)
    # a further --retry-failed run finds nothing to retry and makes no calls
    calls_before = screener.calls
    asyncio.run(rs.main_async(rs.parse_args(base + ["--retry-failed"]), screener=screener))
    assert screener.calls == calls_before
    assert read_json(results / "X_runA.meta.json")["retried_failed"] == 0


def test_record_order_is_identical_across_runs_and_resumes(tmp_path):
    import asyncio

    from common import read_jsonl

    data, protos, results = _setup_dataset(tmp_path)
    argv = ["--dataset", "X", "--run", "A", "--batch-size", "2", "--dry-run", "--data-dir",
            str(data), "--protocols-dir", str(protos), "--results-dir", str(results)]
    asyncio.run(rs.main_async(rs.parse_args(argv)))
    asyncio.run(rs.main_async(rs.parse_args([*argv[:3], "B", *argv[4:]])))
    by_batch_a = {r["record_id"]: r["batch_index"] for r in read_jsonl(results / "X_runA.jsonl")}
    by_batch_b = {r["record_id"]: r["batch_index"] for r in read_jsonl(results / "X_runB.jsonl")}
    assert by_batch_a == by_batch_b
    assert [by_batch_a[i] for i in range(1, 8)] != [(i - 1) // 2 for i in range(1, 8)]


def test_production_screener_reports_settings_failure_as_systemexit(tmp_path, monkeypatch):
    import sys

    fake = tmp_path / "backend"
    (fake / "app" / "agents").mkdir(parents=True)
    (fake / "app" / "__init__.py").write_text("", encoding="utf-8")
    (fake / "app" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    (fake / "app" / "agents" / "relevance_screener_agent.py").write_text(
        "class ScreeningError(Exception):\n    pass\n\nclass ScreeningBatchResult:\n    pass\n\n"
        "async def screen_papers(q, papers, inc, exc):\n    return None\n",
        encoding="utf-8",
    )
    (fake / "app" / "config.py").write_text(
        "raise ValueError('13 validation errors for Settings')\n", encoding="utf-8"
    )
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        rs.ProductionScreener(backend_root=fake)
    assert "backend settings failed to load from" in str(exc.value)
    assert "13 validation errors" in str(exc.value)
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(fake.resolve())])


def test_relative_results_dir_survives_backend_chdir(tmp_path, monkeypatch):
    """--results-dir given relative to the launch cwd must not move with os.chdir(backend)."""
    monkeypatch.chdir(tmp_path)
    args = rs.parse_args(["--dataset", "X", "--run", "A", "--results-dir", "out",
                          "--data-dir", "d", "--protocols-dir", "p"])
    assert args.results_dir == (tmp_path / "out").resolve()
    assert args.data_dir.is_absolute() and args.protocols_dir.is_absolute()
    monkeypatch.chdir(tmp_path.parent)
    assert args.results_dir == (tmp_path / "out").resolve()


def test_baselines_exclude_unscreened_rows_and_record_query_mode(tmp_path):
    pytest.importorskip("sklearn")
    import baselines as bl

    records = [dict(r, label_included=int(r["label_included"])) for r in RECORDS]
    run_rows = [{"record_id": r["record_id"], "predicted": 1} for r in records]
    run_rows[2]["predicted"] = None  # record 3 unscreened by the LLM
    chosen, labels = bl.select_records(records, run_rows, Protocol("X", "d", "q", ["i"], ["e"],
                                                                   "label_included"))
    assert [r["record_id"] for r in chosen] == [1, 2, 4, 5, 6, 7] and len(labels) == 6
    assert bl.llm_inclusion_count(run_rows) == 6
    args = bl.parse_args(["--dataset", "X"])
    assert args.query == "rq+criteria"  # new default, an approximation of the v2 turn
    assert bl.parse_args(["--dataset", "X", "--query", "rq"]).query == "rq"


# ---------------------------------------------------------------- baselines --k-basis


def test_llm_screened_in_count_and_k_basis_flag_default():
    """A full-text-inclusion-criterion protocol routes every confirmed match to
    NEEDS_REVIEW, never INCLUDE, so ``llm_inclusion_count`` (predicted == 1) is
    structurally zero; ``llm_screened_in_count`` counts the retained set instead (status
    INCLUDE or NEEDS_REVIEW), falling back to ``predicted`` for a v1-style row with no
    ``status`` column."""
    pytest.importorskip("sklearn")
    import baselines as bl

    rows = [
        {"record_id": 1, "status": "INCLUDE", "predicted": 0},
        {"record_id": 2, "status": "NEEDS_REVIEW", "predicted": 0},
        {"record_id": 3, "status": "EXCLUDE", "predicted": 0},
    ]
    assert bl.llm_screened_in_count(rows) == 2
    assert bl.llm_inclusion_count(rows) == 0  # predicted is 0 on every row above

    v1_rows = [{"record_id": 1, "predicted": 1}, {"record_id": 2, "predicted": 0}]
    assert bl.llm_screened_in_count(v1_rows) == 1  # no status column: falls back to predicted

    args = bl.parse_args(["--dataset", "X"])
    assert args.k_basis == "predicted"
    assert bl.parse_args(["--dataset", "X", "--k-basis", "screened_in"]).k_basis == "screened_in"


def test_baselines_main_k_basis_screened_in_matches_retained_count(tmp_path):
    """End-to-end: ``--k-basis screened_in`` matches the TF-IDF baseline's k to the
    screener's retained count, not its (structurally zero) INCLUDE count, and records
    ``k_basis`` in the output."""
    pytest.importorskip("sklearn")
    import json

    import baselines as bl

    data = tmp_path / "data"
    protos = tmp_path / "protocols"
    results = tmp_path / "results"
    for d in (data, protos, results):
        d.mkdir()
    records = [
        {"record_id": 1, "title": "Reminders improve guideline adherence", "abstract": "x",
         "label_included": 1},
        {"record_id": 2, "title": "Deep learning for images", "abstract": "y",
         "label_included": 0},
        {"record_id": 3, "title": "Nudging clinicians with reminders", "abstract": "z",
         "label_included": 1},
    ]
    with (data / "X.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    proto = {
        "dataset_id": "X", "source_review_doi": "10.1/x", "research_question": "q",
        "inclusion_criteria": ["reminders"], "exclusion_criteria": ["e"],
        "label_field": "label_included",
    }
    (protos / "X.json").write_text(json.dumps(proto), encoding="utf-8")
    # Every record has predicted == 0 (no INCLUDE decision at all, the structural-zero
    # case), but records 1 and 3 are NEEDS_REVIEW (retained); record 2 is EXCLUDE.
    run_rows = [
        {"record_id": 1, "predicted": 0, "status": "NEEDS_REVIEW"},
        {"record_id": 2, "predicted": 0, "status": "EXCLUDE"},
        {"record_id": 3, "predicted": 0, "status": "NEEDS_REVIEW"},
    ]
    (results / "X_runA.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in run_rows), encoding="utf-8"
    )
    rc = bl.main(["--dataset", "X", "--k-basis", "screened_in", "--data-dir", str(data),
                  "--protocols-dir", str(protos), "--results-dir", str(results)])
    assert rc == 0
    out = json.loads((results / "X_baselines.json").read_text(encoding="utf-8"))
    assert out["k_basis"] == "screened_in"
    assert out["k"] == 2  # records 1 and 3 are screened in; record 2 is excluded

    # an explicit --k still overrides k-basis entirely, and is recorded as such
    rc = bl.main(["--dataset", "X", "--k-basis", "screened_in", "--k", "1",
                  "--data-dir", str(data), "--protocols-dir", str(protos),
                  "--results-dir", str(results)])
    assert rc == 0
    out = json.loads((results / "X_baselines.json").read_text(encoding="utf-8"))
    assert out["k_basis"] == "explicit" and out["k"] == 1


# ---------------------------------------------------------------- additional cases


class _SlowScreener(rs.DryRunScreener):
    """Dry-run screener that takes a measurable amount of wall-clock time per batch."""

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.calls: list[list[str]] = []

    async def __call__(self, protocol, papers):
        import asyncio

        self.calls.append([p["title"] for p in papers])
        await asyncio.sleep(self.delay)
        return await super().__call__(protocol, papers)


def test_screen_batch_with_retries_records_real_latency():
    import asyncio

    screener = _SlowScreener(0.05)
    outcome = asyncio.run(rs.screen_batch_with_retries(screener, PROTOCOL, RECORDS[:3], 1))
    assert outcome["latency_s"] >= 0.05
    rows = rs.rows_for_batch(RECORDS[:3], 0, outcome, "label_abstract_screening")
    assert all(r["latency_s"] >= 0.05 for r in rows)


def test_dry_run_screener_latency_is_measured_not_defaulted():
    import asyncio

    outcome = asyncio.run(rs.screen_batch_with_retries(rs.DryRunScreener(), PROTOCOL, RECORDS, 1))
    assert isinstance(outcome["latency_s"], float) and outcome["latency_s"] >= 0.0
    slow = asyncio.run(rs.screen_batch_with_retries(_SlowScreener(0.03), PROTOCOL, RECORDS, 1))
    assert slow["latency_s"] > outcome["latency_s"]


def test_production_run_requires_confirm_cost(tmp_path, monkeypatch, capsys):
    import asyncio

    data, protos, results = _setup_dataset(tmp_path)

    def _boom(*a, **k):
        raise AssertionError("ProductionScreener must not be constructed without --confirm-cost")

    monkeypatch.setattr(rs, "ProductionScreener", _boom)
    argv = ["--dataset", "X", "--run", "A", "--batch-size", "3", "--data-dir", str(data),
            "--protocols-dir", str(protos), "--results-dir", str(results)]
    with pytest.raises(SystemExit) as exc:
        asyncio.run(rs.main_async(rs.parse_args(argv)))
    assert exc.value.code == 3
    out = capsys.readouterr().out
    assert "3 batches" in out and "USD" in out and "--confirm-cost" in out
    assert not (results / "X_runA.jsonl").exists()
    # --limit alone does not lift the requirement (a smoke run also spends money)
    with pytest.raises(SystemExit):
        asyncio.run(rs.main_async(rs.parse_args(argv + ["--limit", "2"])))
    # an injected screener (tests) is also gated unless the flag or --dry-run is given
    with pytest.raises(SystemExit):
        asyncio.run(rs.main_async(rs.parse_args(argv), screener=_SlowScreener(0.0)))
    assert asyncio.run(
        rs.main_async(rs.parse_args(argv + ["--confirm-cost"]), screener=_SlowScreener(0.0))
    ) == 0


# ---------------------------------------------------------------- cost bound rework


def test_cost_bound_uses_the_batches_own_rendered_text_length():
    batches = [
        [{"title": "T1", "abstract": "a" * 100}, {"title": "T2", "abstract": "b" * 200}],
        [{"title": "T3", "abstract": "c" * 50}],
    ]
    bound = rs.cost_upper_bound(batches, "deepseek-chat")
    rates = rs.DEEPSEEK_PRICES["models"]["deepseek-v4-flash"]
    expected = 0.0
    for batch in batches:
        chars = sum(len(r["title"]) + len(r["abstract"]) for r in batch)
        input_tokens = chars / rs.CHARS_PER_INPUT_TOKEN + rs.PROMPT_FRAMING_TOKENS_PER_BATCH
        expected += (
            input_tokens * rates["input_cache_miss"]["peak"]
            + rs.ASSUMED_OUTPUT_TOKENS_PER_BATCH * rates["output"]["peak"]
        ) / 1_000_000.0
    assert bound == pytest.approx(expected)
    assert rs.cost_upper_bound(batches, "unknown-model") is None
    assert rs.cost_upper_bound([], "deepseek-chat") == 0.0


def test_cost_bound_respects_the_abstract_cap_and_exceeds_a_known_actual():
    """A record whose abstract exceeds the cap must be estimated at the capped length (a
    true bound is on what will actually be *sent*, not on the raw record); the bound must
    still safely exceed a small known actual cost computed from real token counts."""
    long_abstract = "x" * 20000
    batches = [[{"title": "T", "abstract": long_abstract}]]
    bound_capped = rs.cost_upper_bound(batches, "deepseek-chat", abstract_cap=10000)
    bound_uncapped = rs.cost_upper_bound(batches, "deepseek-chat", abstract_cap=None)
    assert bound_capped < bound_uncapped

    from common import call_cost

    actual = call_cost("deepseek-v4-flash", "2026-09-02T12:00:00+00:00", 2000, 500)
    assert bound_capped > actual


# ---------------------------------------------------------------- cost gate


def test_reask_cost_allowance_prices_max_reask_calls_per_record():
    """Each pending record is priced as up to ``MAX_REASK_CALLS_PER_RECORD`` extra
    single-record calls, at the same per-call assumptions :func:`cost_upper_bound` uses."""
    records = [{"title": "T1", "abstract": "a" * 100}, {"title": "T2", "abstract": "b" * 200}]
    allowance = rs.reask_cost_allowance(records, "deepseek-chat")
    rates = rs.DEEPSEEK_PRICES["models"]["deepseek-v4-flash"]
    expected = 0.0
    for r in records:
        chars = len(r["title"]) + len(r["abstract"])
        input_tokens = chars / rs.CHARS_PER_INPUT_TOKEN + rs.PROMPT_FRAMING_TOKENS_PER_BATCH
        per_call = (
            input_tokens * rates["input_cache_miss"]["peak"]
            + rs.ASSUMED_OUTPUT_TOKENS_PER_BATCH * rates["output"]["peak"]
        ) / 1_000_000.0
        expected += rs.MAX_REASK_CALLS_PER_RECORD * per_call
    assert allowance == pytest.approx(expected)
    assert rs.reask_cost_allowance(records, "unknown-model") is None
    assert rs.reask_cost_allowance([], "deepseek-chat") == 0.0


def test_reask_cost_allowance_respects_the_abstract_cap():
    long_abstract = "x" * 20000
    records = [{"title": "T", "abstract": long_abstract}]
    capped = rs.reask_cost_allowance(records, "deepseek-chat", abstract_cap=10000)
    uncapped = rs.reask_cost_allowance(records, "deepseek-chat", abstract_cap=None)
    assert capped < uncapped


def test_resume_sends_only_pending_records_of_a_partial_batch(tmp_path):
    import asyncio

    from common import read_json, read_jsonl

    data, protos, results = _setup_dataset(tmp_path)
    base = ["--dataset", "X", "--run", "A", "--batch-size", "3", "--confirm-cost",
            "--data-dir", str(data), "--protocols-dir", str(protos), "--results-dir",
            str(results)]
    screener = _SlowScreener(0.0)
    asyncio.run(rs.main_async(rs.parse_args(base + ["--limit", "2"]), screener=screener))
    assert [len(c) for c in screener.calls] == [2]
    done_titles = {t for c in screener.calls for t in c}
    asyncio.run(rs.main_async(rs.parse_args(base), screener=screener))
    later = screener.calls[1:]
    assert [len(c) for c in later] == [1, 3, 1]  # rest of batch 0, batch 1, batch 2
    assert not any(t in done_titles for c in later for t in c), "done records were re-sent"
    rows = read_jsonl(results / "X_runA.jsonl")
    ids = [r["record_id"] for r in rows]
    assert sorted(ids) == list(range(1, 8)) and len(ids) == len(set(ids))
    assert sorted({r["batch_index"] for r in rows}) == [0, 1, 2]
    meta = read_json(results / "X_runA.meta.json")
    assert meta["n_calls"] == 4 and meta["n_batches"] == 3 and meta["latency_s"]["n"] == 4


def test_build_meta_counts_every_call_of_a_split_batch():
    rows = []
    for rid, call in [(1, "c1"), (2, "c1"), (3, "c2")]:
        rows.append({
            "record_id": rid, "label": 1, "label_included": 1, "has_abstract": True,
            "has_title": True, "predicted": 1, "reason": "ok", "padded_include": False,
            "batch_index": 0, "called_at": "2026-09-02T12:00:00+00:00", "call_id": call,
            "latency_s": 1.0, "input_tokens": 100, "output_tokens": 10,
            "model_reported": "deepseek-v4-flash", "system_fingerprint": "fp",
            "provider_response_id": None, "attempts": 1,
        })
    meta = rs.build_meta(
        rows, existing=None, dataset="X", run="A", protocol=PROTOCOL, n_records=3, n_batches=1,
        batch_size=3, concurrency=1, model_configured="deepseek-chat", temperature=0.0,
        prompt_versions=["v"], price={"input_per_mtok": None, "output_per_mtok": None},
        dry_run=False, session_started="2026-09-02T12:00:00+00:00", limit=None,
    )
    assert meta["n_calls"] == 2 and meta["total_input_tokens"] == 200
    assert meta["latency_s"]["n"] == 2 and meta["n_batches_completed"] == 1


def test_build_meta_counts_a_batchs_reask_as_a_second_call():
    """A call group whose ``calls`` field is 2 (the backend's own internal re-ask answered
    for that batch) must count as two real API calls in ``n_calls``, not one, and a
    model/fingerprint change on the re-ask must still land in the distinct
    ``model_reported``/``system_fingerprints`` sets."""
    rows = [
        {
            "record_id": 1, "label": 1, "label_included": 1, "has_abstract": True,
            "has_title": True, "predicted": 1, "reason": "ok", "padded_include": False,
            "batch_index": 0, "called_at": "2026-09-02T12:00:00+00:00", "call_id": "c1",
            "latency_s": 1.0, "input_tokens": 170, "output_tokens": 17,
            "model_reported": "deepseek-v4-flash", "system_fingerprint": "fp",
            "reask_model_reported": "deepseek-v4-flash-2", "reask_system_fingerprint": "fp2",
            "provider_response_id": None, "attempts": 1, "calls": 2,
        },
        {
            "record_id": 2, "label": 1, "label_included": 1, "has_abstract": True,
            "has_title": True, "predicted": 1, "reason": "ok", "padded_include": False,
            "batch_index": 1, "called_at": "2026-09-02T12:00:00+00:00", "call_id": "c2",
            "latency_s": 1.0, "input_tokens": 10, "output_tokens": 1,
            "model_reported": "deepseek-v4-flash", "system_fingerprint": "fp",
            "reask_model_reported": None, "reask_system_fingerprint": None,
            "provider_response_id": None, "attempts": 1, "calls": 1,
        },
    ]
    meta = rs.build_meta(
        rows, existing=None, dataset="X", run="A", protocol=PROTOCOL, n_records=2, n_batches=2,
        batch_size=1, concurrency=1, model_configured="deepseek-chat", temperature=0.0,
        prompt_versions=["v"], price={"input_per_mtok": None, "output_per_mtok": None},
        dry_run=False, session_started="2026-09-02T12:00:00+00:00", limit=None,
    )
    assert meta["n_calls"] == 3  # 2 (call group c1) + 1 (call group c2)
    assert meta["total_input_tokens"] == 180 and meta["total_output_tokens"] == 18
    assert meta["model_reported"] == ["deepseek-v4-flash", "deepseek-v4-flash-2"]
    assert meta["system_fingerprints"] == ["fp", "fp2"]


def test_build_meta_defaults_calls_to_one_for_a_pre_fix_row():
    rows = [{
        "record_id": 1, "label": 1, "label_included": 1, "has_abstract": True,
        "has_title": True, "predicted": 1, "reason": "ok", "padded_include": False,
        "batch_index": 0, "called_at": "2026-09-02T12:00:00+00:00", "call_id": "c1",
        "latency_s": 1.0, "input_tokens": 10, "output_tokens": 1,
        "model_reported": "deepseek-v4-flash", "system_fingerprint": "fp",
        "provider_response_id": None, "attempts": 1,
    }]
    meta = rs.build_meta(
        rows, existing=None, dataset="X", run="A", protocol=PROTOCOL, n_records=1, n_batches=1,
        batch_size=1, concurrency=1, model_configured="deepseek-chat", temperature=0.0,
        prompt_versions=["v"], price={"input_per_mtok": None, "output_per_mtok": None},
        dry_run=False, session_started="2026-09-02T12:00:00+00:00", limit=None,
    )
    assert meta["n_calls"] == 1


# ------------------------------------------------ Superseded call accounting


def test_build_meta_folds_superseded_calls_into_totals_and_batch_count():
    """A call whose every row has since been pruned from ``rows`` by a later re-ask pass
    (here, call ``c1``, absent from ``rows``) must still be counted via the
    ``superseded_calls`` representative row the caller collects before pruning: folded into
    n_calls/the token totals/n_batches_completed the same way a surviving call is, and also
    reported on its own."""
    surviving = rs.rows_for_batch(
        RECORDS[:2], 1, _ok_outcome(input_tokens=10, output_tokens=1, call_id="c2"),
        "label_included",
    )
    superseded = rs.rows_for_batch(
        RECORDS[2:5],
        0,
        _ok_outcome(
            include=[True, True, True], reasons=["ok"] * 3,
            input_tokens=170, output_tokens=17, call_id="c1",
        ),
        "label_included",
    )
    meta = rs.build_meta(
        surviving, existing=None, dataset="X", run="A", protocol=PROTOCOL, n_records=5,
        n_batches=2, batch_size=3, concurrency=1, model_configured="deepseek-chat",
        temperature=0.0, prompt_versions=["v"],
        price={"input_per_mtok": None, "output_per_mtok": None}, dry_run=False,
        session_started="2026-09-02T12:00:00+00:00", limit=None,
        superseded_call_rows=superseded,
    )
    assert meta["n_calls"] == 2  # c1 (superseded) + c2 (surviving)
    assert meta["total_input_tokens"] == 180 and meta["total_output_tokens"] == 18
    assert meta["n_batches_completed"] == 2  # batch_index 0 (superseded) and 1 (surviving)
    assert meta["superseded_calls"] == 1
    assert meta["superseded_input_tokens"] == 170
    assert meta["superseded_output_tokens"] == 17
    assert meta["total_cost"] is not None  # c1's cost is priced even though it has no row
    assert [r["call_id"] for r in meta["superseded_call_rows"]] == ["c1"]


def test_build_meta_does_not_double_count_a_superseded_call_still_on_disk():
    """A representative row passed via ``superseded_calls`` whose call_id still has a
    surviving row in ``rows`` must not be counted a second time."""
    rows = rs.rows_for_batch(
        RECORDS[:1], 0, _ok_outcome(include=[True], reasons=["ok"], input_tokens=10,
                                     output_tokens=1, call_id="c1"),
        "label_included",
    )
    meta = rs.build_meta(
        rows, existing=None, dataset="X", run="A", protocol=PROTOCOL, n_records=1, n_batches=1,
        batch_size=1, concurrency=1, model_configured="deepseek-chat", temperature=0.0,
        prompt_versions=["v"], price={"input_per_mtok": None, "output_per_mtok": None},
        dry_run=False, session_started="2026-09-02T12:00:00+00:00", limit=None,
        superseded_call_rows=rows,  # same call_id, already present in rows
    )
    assert meta["n_calls"] == 1 and meta["total_input_tokens"] == 10
    assert meta["superseded_calls"] == 0
    assert meta["superseded_call_rows"] == []


# ------------------------------------------------ Fully pruned calls


def test_fully_pruned_calls_ignores_a_call_with_a_surviving_row():
    """A call some but not all of whose rows satisfy ``is_pruned`` keeps a surviving row on
    disk, which already carries that call's own tokens -- it must not be reported as fully
    pruned."""
    rows = rs.rows_for_batch(
        RECORDS[:2], 0, _ok_outcome(include=[True, True], reasons=["ok", "ok"], call_id="c1"),
        "label_included",
    )
    rows[0]["predicted"] = None  # one row of c1 becomes retryable, the other does not
    assert rs._fully_pruned_calls(rows, lambda r: r["predicted"] is None) == []


def test_fully_pruned_calls_reports_a_call_every_one_of_whose_rows_is_pruned():
    """A call every one of whose rows satisfies ``is_pruned`` has nothing left to reconstruct
    it from once the prune runs, so its representative row (the first one seen) is returned."""
    c1 = rs.rows_for_batch(
        RECORDS[:2], 0, _ok_outcome(include=[True, True], reasons=["ok", "ok"], call_id="c1"),
        "label_included",
    )
    c2 = rs.rows_for_batch(
        RECORDS[2:3], 1, _ok_outcome(include=[True], reasons=["ok"], call_id="c2"),
        "label_included",
    )
    for row in c1:
        row["predicted"] = None
    out = rs._fully_pruned_calls(c1 + c2, lambda r: r["predicted"] is None)
    assert [r["call_id"] for r in out] == ["c1"]


class _ThreeOfSevenPaddingScreener(_FakeScreener):
    """Pads exactly the last 3 records of a 7-record pass-1 batch, stays fully padded on the
    pass-2 batch-of-3 re-ask, and heals each pass-3 single-record re-ask. The pass-2 call's
    entire row group is pruned, one row at a time, by the three pass-3 single-record
    sub-batches, so by the time the last one prunes, the pass-2 call has no row left in the
    results file at all."""

    def __init__(self):
        super().__init__()
        self.real_calls = 0
        self.real_input_tokens = 0
        self.real_output_tokens = 0

    async def __call__(self, protocol, papers):
        self.calls += 1
        n = len(papers)
        input_tokens, output_tokens = n * 100, n * 20
        self.real_calls += 1
        self.real_input_tokens += input_tokens
        self.real_output_tokens += output_tokens
        # pass 1 (n=7) heals 4 of 7; pass 2's batch-of-3 (n=3) heals none; pass 3's
        # single-record sub-batches (n=1) each heal their one record.
        n_ok = 4 if n == 7 else (1 if n == 1 else 0)
        statuses = ["INCLUDE"] * n_ok + ["NEEDS_REVIEW"] * (n - n_ok)
        reasons = ["ok"] * n_ok + [rs.NO_DECISION_REASON] * (n - n_ok)
        return {
            "include": [s == "INCLUDE" for s in statuses],
            "reasons": reasons,
            "statuses": statuses,
            "criteria_ids": [""] * n,
            "quotes": [""] * n,
            "guard_applied": [False] * n,
            "guard_reasons": [""] * n,
            "to_confirm": [[] for _ in range(n)],
            "padded": n - n_ok,
            "model_reported": "deepseek-v4-flash",
            "system_fingerprint": "fp",
            "provider_response_id": None,
            "temperature": 0.0,
            "prompt_version": "v2",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


def test_run_screening_meta_counts_a_pass2_call_every_one_of_whose_records_reaches_pass3(
    tmp_path,
):
    """End to end: pass 1 pads 3 of 7 records; the pass-2 batch-of-3 re-ask leaves all 3
    still padded; pass 3 heals them one record at a time, pruning the pass-2 call's three
    rows one by one until none is left. The written meta's
    n_calls/total_input_tokens/total_output_tokens must equal the screener's own real
    totals, not silently drop the pass-2 call."""
    import asyncio

    from common import read_json

    data, protos, results = _setup_dataset(tmp_path)
    base = ["--dataset", "X", "--run", "A", "--batch-size", "7", "--confirm-cost",
            "--data-dir", str(data), "--protocols-dir", str(protos), "--results-dir",
            str(results)]
    screener = _ThreeOfSevenPaddingScreener()
    asyncio.run(rs.main_async(rs.parse_args(base), screener=screener))

    assert screener.real_calls == 5  # 1 (pass 1) + 1 (pass 2) + 3 (pass 3)
    meta = read_json(results / "X_runA.meta.json")
    assert meta["n_calls"] == screener.real_calls
    assert meta["total_input_tokens"] == screener.real_input_tokens
    assert meta["total_output_tokens"] == screener.real_output_tokens
    assert meta["n_batches"] == meta["n_batches_completed"] == 5
    assert meta["superseded_calls"] == 1  # the pass-2 call, pruned row by row by pass 3


# ---------------------------------------- Superseded totals across sessions


def test_run_screening_meta_keeps_superseded_totals_across_a_session_that_sends_nothing(
    tmp_path,
):
    """A second session run with the identical command, after everything is already
    resolved, sends no calls at all. build_meta is handed only what its own (empty)
    call_records and the current on-disk rows say, so without the carry-forward it
    recomputes the meta down to just what is on disk right now and silently drops the first
    session's own superseded pass-2 call. The second session's meta must equal the first's
    own real totals."""
    import asyncio

    from common import read_json

    data, protos, results = _setup_dataset(tmp_path)
    base = ["--dataset", "X", "--run", "A", "--batch-size", "7", "--confirm-cost",
            "--data-dir", str(data), "--protocols-dir", str(protos), "--results-dir",
            str(results)]
    screener = _ThreeOfSevenPaddingScreener()
    asyncio.run(rs.main_async(rs.parse_args(base), screener=screener))
    session1_calls = screener.real_calls
    session1_in = screener.real_input_tokens
    session1_out = screener.real_output_tokens
    meta1 = read_json(results / "X_runA.meta.json")
    assert meta1["n_calls"] == session1_calls == 5
    assert meta1["superseded_call_rows"] and len(meta1["superseded_call_rows"]) == 1

    # identical command, second session: every record is already decided, so nothing is
    # pending and the screener is never called again.
    asyncio.run(rs.main_async(rs.parse_args(base), screener=screener))
    assert screener.real_calls == session1_calls  # session 2 sent nothing at all

    meta2 = read_json(results / "X_runA.meta.json")
    assert meta2["n_calls"] == session1_calls
    assert meta2["total_input_tokens"] == session1_in
    assert meta2["total_output_tokens"] == session1_out
    assert meta2["n_batches_completed"] == meta1["n_batches_completed"] == 5
    assert meta2["superseded_call_rows"] == meta1["superseded_call_rows"]


class _StubbornThenHealsScreener(_FakeScreener):
    """Pads whichever record's title is in ``stubborn_titles`` on every call and every pass,
    for as long as ``heal`` stays ``False``; every other record heals on its first call.
    Flipping ``heal`` to ``True`` between two ``main_async`` invocations models a record a
    whole first session's re-ask cascade never resolves, left for --retry-failed to pick up
    in a later session."""

    def __init__(self, stubborn_titles):
        super().__init__()
        self.stubborn_titles = set(stubborn_titles)
        self.heal = False
        self.real_calls = 0
        self.real_input_tokens = 0
        self.real_output_tokens = 0

    async def __call__(self, protocol, papers):
        self.calls += 1
        self.real_calls += 1
        n = len(papers)
        input_tokens, output_tokens = n * 100, n * 20
        self.real_input_tokens += input_tokens
        self.real_output_tokens += output_tokens
        stuck = [(not self.heal) and p.get("title") in self.stubborn_titles for p in papers]
        statuses = ["NEEDS_REVIEW" if s else "INCLUDE" for s in stuck]
        reasons = [rs.NO_DECISION_REASON if s else "ok" for s in stuck]
        return {
            "include": [st == "INCLUDE" for st in statuses],
            "reasons": reasons,
            "statuses": statuses,
            "criteria_ids": [""] * n,
            "quotes": [""] * n,
            "guard_applied": [False] * n,
            "guard_reasons": [""] * n,
            "to_confirm": [[] for _ in range(n)],
            "padded": sum(stuck),
            "model_reported": "deepseek-v4-flash",
            "system_fingerprint": "fp",
            "provider_response_id": None,
            "temperature": 0.0,
            "prompt_version": "v2",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


def test_retry_failed_carries_forward_an_earlier_sessions_superseded_call(tmp_path):
    """One record's row survives an entire first session only as a lone-call, still-padded
    row (its earlier pass-2 call was already superseded, and correctly counted, within that
    same session). --retry-failed in a second session prunes that lone row, the only row of
    its call, before re-screening the record. The second session's meta must count every
    call from both sessions: the first session's own re-ask cascade, the call
    --retry-failed's own prune just erased, and the brand-new call that finally heals the
    record."""
    import asyncio

    from common import read_json, read_jsonl

    data, protos, results = _setup_dataset(tmp_path)
    base = ["--dataset", "X", "--run", "A", "--batch-size", "7", "--confirm-cost",
            "--data-dir", str(data), "--protocols-dir", str(protos), "--results-dir",
            str(results)]
    screener = _StubbornThenHealsScreener({"Paper 7"})
    asyncio.run(rs.main_async(rs.parse_args(base), screener=screener))
    session1_calls = screener.real_calls
    assert session1_calls == 3  # pass 1 (7 records), pass 2 (1), pass 3 (1) -- still padded
    stuck_rows = [r for r in read_jsonl(results / "X_runA.jsonl") if r["record_id"] == 7]
    assert len(stuck_rows) == 1 and stuck_rows[0]["padded_decision"] is True

    screener.heal = True
    asyncio.run(
        rs.main_async(rs.parse_args(base + ["--retry-failed"]), screener=screener)
    )

    meta = read_json(results / "X_runA.meta.json")
    assert screener.real_calls == session1_calls + 1  # the healing call, this session
    assert meta["n_calls"] == screener.real_calls
    assert meta["total_input_tokens"] == screener.real_input_tokens
    assert meta["total_output_tokens"] == screener.real_output_tokens
    assert meta["retried_failed"] == 1


# ---------------------------------------------------------------- additional cases


def test_cost_gate_preview_counts_failed_batches_under_retry_failed_and_prints_4_decimals(
    tmp_path, monkeypatch, capsys
):
    import asyncio

    from common import read_jsonl

    monkeypatch.setattr(rs.asyncio, "sleep", _no_sleep)
    data, protos, results = _setup_dataset(tmp_path)
    base = ["--dataset", "X", "--run", "A", "--batch-size", "3", "--max-attempts", "2",
            "--record-order-seed", "0", "--data-dir", str(data),
            "--protocols-dir", str(protos), "--results-dir", str(results)]
    screener = _FlakyScreener({"Paper 7"})
    screener.model_configured = "deepseek-chat"  # priced model for the gate message
    asyncio.run(rs.main_async(rs.parse_args(base + ["--confirm-cost"]), screener=screener))
    rows = read_jsonl(results / "X_runA.jsonl")
    assert len(rows) == 7 and any(r["predicted"] is None for r in rows)
    failed_records = [{"title": r["title"], "abstract": "text"} for r in rows
                       if r["predicted"] is None]
    capsys.readouterr()
    # every record has a row, so without --retry-failed nothing is pending ...
    with pytest.raises(SystemExit):
        asyncio.run(rs.main_async(rs.parse_args(base), screener=screener))
    assert "over 0 batches" in capsys.readouterr().out
    # ... but with --retry-failed the failed batch is paid work and the preview must say so
    with pytest.raises(SystemExit) as exc:
        asyncio.run(rs.main_async(rs.parse_args(base + ["--retry-failed"]), screener=screener))
    assert exc.value.code == 3
    out = capsys.readouterr().out
    assert "over 1 batches" in out
    # The printed bound is pass 1 plus the worst-case re-ask allowance, not pass 1 alone.
    pass1_bound = rs.cost_upper_bound([failed_records], "deepseek-chat")
    reask_bound = rs.reask_cost_allowance(failed_records, "deepseek-chat")
    total_bound = pass1_bound + reask_bound
    assert f"<= {total_bound:.4f} USD" in out
    assert f"{pass1_bound:.4f} pass 1" in out
    assert f"{reask_bound:.4f} worst-case re-ask allowance" in out
    assert len(read_jsonl(results / "X_runA.jsonl")) == 7  # the gate never prunes


# ---------------------------------------------------------------- additional cases



# ---------------------------------------------------------------- status/criterion/quote


def test_rows_default_status_from_predicted_when_outcome_carries_no_statuses():
    """Legacy outcomes (no ``statuses``/``criteria_ids``/``quotes``/``guard_applied`` keys,
    e.g. a fake test screener) still get a sensible three-way ``status`` column derived from
    the existing binary ``predicted``, and empty ``criterion``/``quote``/``guard_applied``."""
    batch = RECORDS[:2]
    ok = {"include": [True, False], "reasons": ["matches", "off-topic"]}
    rows = rs.rows_for_batch(batch, 0, ok, "label_abstract_screening")
    assert rows[0]["status"] == "INCLUDE" and rows[1]["status"] == "EXCLUDE"
    assert rows[0]["criterion"] == "" and rows[0]["quote"] == ""
    assert rows[0]["guard_applied"] is False and rows[1]["guard_applied"] is False
    failed = rs.rows_for_batch(batch, 1, {"error": "boom", "attempts": 1}, "label_included")
    assert failed[0]["status"] is None and failed[0]["criterion"] == ""


def test_rows_carry_real_status_criterion_quote_and_guard_applied():
    outcome = _ok_outcome(
        include=[False, True],
        statuses=["NEEDS_REVIEW", "INCLUDE"],
        criteria_ids=["E1", ""],
        quotes=["some quote", ""],
        guard_applied=[True, False],
    )
    rows = rs.rows_for_batch(RECORDS[:2], 0, outcome, "label_included")
    assert rows[0]["status"] == "NEEDS_REVIEW" and rows[0]["criterion"] == "E1"
    assert rows[0]["quote"] == "some quote" and rows[0]["guard_applied"] is True
    assert rows[1]["status"] == "INCLUDE" and rows[1]["guard_applied"] is False


def test_rows_carry_to_confirm():
    """The full-text inclusion ids still to be confirmed at full text reach the row,
    parallel to ``statuses``; an outcome that carries none (an older-style test double)
    defaults every row to an empty list."""
    outcome = _ok_outcome(
        include=[True, True],
        statuses=["INCLUDE", "INCLUDE"],
        criteria_ids=["", ""],
        quotes=["", ""],
        to_confirm=[["I1", "I3"], []],
    )
    rows = rs.rows_for_batch(RECORDS[:2], 0, outcome, "label_included")
    assert rows[0]["to_confirm"] == ["I1", "I3"]
    assert rows[1]["to_confirm"] == []

    legacy_rows = rs.rows_for_batch(RECORDS[:2], 0, _ok_outcome(), "label_included")
    assert legacy_rows[0]["to_confirm"] == []
    assert legacy_rows[1]["to_confirm"] == []


# ---------------------------------------------------------------- prompt-version/cap


def _fake_backend_module(tmp_path):
    """A fake ``app.agents.relevance_screener_agent`` recording the ``prompt_version`` and
    ``abstract_limit`` a call received, and rendering a shown-text stand-in shaped like the
    real one (v1: bare ``"..."`` truncation, no tail; v2: a head, the cut marker, and a
    tail), so a caller can assert on the actual text a
    ``--prompt-version``/``--abstract-cap`` pair produces without a real model call."""
    fake = tmp_path / "backend"
    (fake / "app" / "agents").mkdir(parents=True)
    (fake / "app" / "__init__.py").write_text("", encoding="utf-8")
    (fake / "app" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    (fake / "app" / "agents" / "relevance_screener_agent.py").write_text(
        '''
CALLS = []
ABSTRACT_CHAR_LIMIT = 3000


class ScreeningError(Exception):
    pass


class _Prov:
    def __init__(self, prompt_version):
        self.model_configured = "fake"
        self.model_reported = "fake"
        self.system_fingerprint = None
        self.provider_response_id = None
        self.temperature = 0.0
        self.prompt_version = prompt_version
        self.input_tokens = 1
        self.output_tokens = 1


class ScreeningBatchResult:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _render(abstract, prompt_version, limit):
    if prompt_version == "v1":
        return abstract[:limit] + "..." if len(abstract) > limit else abstract
    if len(abstract) <= limit:
        return abstract
    head = round(limit * 7 / 10)
    tail = limit - head
    return abstract[:head] + " [...] " + (abstract[-tail:] if tail else "")


async def screen_papers(q, papers, inc, exc, *, prompt_version="v2", abstract_limit=3000):
    CALLS.append({"prompt_version": prompt_version, "abstract_limit": abstract_limit})
    n = len(papers)
    shown = [_render(p.get("abstract") or "", prompt_version, abstract_limit) for p in papers]
    return ScreeningBatchResult(
        include=[True] * n,
        statuses=["INCLUDE"] * n,
        criteria_ids=[""] * n,
        quotes=[""] * n,
        reasons=shown,
        guard_applied=[False] * n,
        guard_conversions=0,
        provenance=_Prov(f"sha256:{prompt_version}"),
    )
''',
        encoding="utf-8",
    )
    (fake / "app" / "config.py").write_text(
        "settings = type('S', (), {'deepseek_model': 'fake'})()\n", encoding="utf-8"
    )
    return fake


def _fake_backend_module_v2(
    tmp_path,
    *,
    include_types: list[str | None] | None = None,
    include_paratext: list[bool] | None = None,
    include_abstracts: list[str] | None = None,
    second_pass_established: list[tuple[str, str, str]] | None = None,
    second_pass_raises: bool = False,
    second_pass_model: str | None = "fake-second-pass",
    stage_budget_seconds: float | None = 1800.0,
):
    """A fake backend with real-equivalent (not stubbed)
    ``apply_type_demotion``/``apply_table_of_contents_demotion``/``build_shown_texts``
    logic, so a test here exercises this harness's own wiring (which survivors reach the
    second pass, which rows carry which guard reason) rather than the correctness of the
    backend functions themselves (already covered in ``backend/tests``).

    ``screen_papers`` always returns one INCLUDE per paper (no criterion needed). Each
    paper's OpenAlex type/paratext flag comes from ``include_types``/``include_paratext``
    (by position, default ``None``/``False``); its shown abstract, for the table-of-contents
    check, from ``include_abstracts`` (default the paper's own abstract).
    ``second_pass_established`` is one ``(population, outcome, study_type)`` tuple per
    record that reaches the second pass, in the order ``confirm_inclusions`` receives them;
    ``second_pass_raises`` makes ``confirm_inclusions`` raise ``ScreeningError`` instead.
    """
    fake = tmp_path / "backend"
    (fake / "app" / "agents").mkdir(parents=True)
    (fake / "app" / "__init__.py").write_text("", encoding="utf-8")
    (fake / "app" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    (fake / "app" / "agents" / "relevance_screener_agent.py").write_text(
        f'''
import re

ABSTRACT_CHAR_LIMIT = 3000
NONARTICLE_TYPES = frozenset({{"paratext", "book", "book-review", "reference-entry"}})
NOT_ESTABLISHED = "not_established"

_TOC_CHAPTER_RE = re.compile(r"\\d{{1,2}}\\.\\s+[A-Z]")


class ScreeningError(Exception):
    pass


class _Prov:
    def __init__(self, **kw):
        self.model_configured = "fake"
        self.model_reported = "fake"
        self.system_fingerprint = None
        self.provider_response_id = None
        self.temperature = 0.0
        self.prompt_version = "sha256:fake"
        self.input_tokens = 100
        self.output_tokens = 10
        self.__dict__.update(kw)


class ScreeningBatchResult:
    def __init__(self, **kw):
        self.__dict__.update(kw)


async def screen_papers(q, papers, inc, exc, **kwargs):
    n = len(papers)
    return ScreeningBatchResult(
        include=[True] * n,
        statuses=["INCLUDE"] * n,
        criteria_ids=[""] * n,
        quotes=[""] * n,
        reasons=["on topic"] * n,
        guard_applied=[False] * n,
        guard_reasons=[""] * n,
        to_confirm=[[] for _ in range(n)],
        guard_conversions=0,
        provenance=_Prov(model_reported="deepseek-v4-flash", system_fingerprint="fp_a"),
    )


def build_shown_texts(papers, *, prompt_version="v2", limit=ABSTRACT_CHAR_LIMIT):
    return [f"Title: {{p.get('title') or ''}}\\nAbstract: {{p.get('abstract') or ''}}" for p in papers]


def apply_type_demotion(status, *, work_type, is_paratext):
    if status != "INCLUDE":
        return status, ""
    if bool(is_paratext) or (work_type in NONARTICLE_TYPES):
        return "NEEDS_REVIEW", "nonarticle_type"
    return status, ""


def apply_table_of_contents_demotion(status, shown_text):
    if status != "INCLUDE":
        return status, ""
    if len(_TOC_CHAPTER_RE.findall(shown_text or "")) >= 4:
        return "NEEDS_REVIEW", "table_of_contents"
    return status, ""


class _SlotAnswer:
    def __init__(self, established, quote=""):
        self.established = established
        self.quote = quote


class _AnswerV2:
    def __init__(self, population, outcome, study_type):
        self.population = _SlotAnswer(population, "verbatim")
        self.outcome = _SlotAnswer(outcome, "verbatim")
        self.study_type = study_type


_SECOND_PASS_RAISES = {second_pass_raises!r}
_SECOND_PASS_ESTABLISHED = {second_pass_established or []!r}


async def confirm_inclusions(papers, *, research_question="", inclusion_criteria=None,
                              exclusion_criteria=None, model=None):
    if _SECOND_PASS_RAISES:
        raise ScreeningError("TimeoutError: model timed out")
    answers = [_AnswerV2(*_SECOND_PASS_ESTABLISHED[i]) for i in range(len(papers))]
    return ScreeningBatchResult(
        answers=answers,
        provenance=_Prov(
            model_reported="fake-second-pass-served", system_fingerprint="fp_second",
            input_tokens=55, output_tokens=15,
        ),
    )


def apply_second_pass_guard(status, verdict, *, answers=None, shown_text=None):
    if status != "INCLUDE":
        return status, ""
    if answers.population.established != "established" or answers.outcome.established != "established":
        return "NEEDS_REVIEW", "not_established"
    if answers.study_type not in ("study", "synthesis"):
        return "NEEDS_REVIEW", "not_established"
    return status, ""


class _StageDecision:
    def __init__(self, status, guard_reason, second_pass=None):
        self.status = status
        self.guard_reason = guard_reason
        self.second_pass = second_pass


class _StageCall:
    def __init__(self, provenance, indices):
        self.provenance = provenance
        self.latency_s = 0.0
        self.n_records = len(indices)
        self.indices = indices


class _StageResult:
    def __init__(self, decisions, calls, wall_time_s=0.0, n_skipped_budget=0):
        self.decisions = decisions
        self.calls = calls
        self.wall_time_s = wall_time_s
        self.n_skipped_budget = n_skipped_budget


#: The last stage_deadline this fake's own
#: run_second_pass_stage received, so a test of the harness's own deadline-computing logic
#: (ProductionScreener.__call__) can read it back without this fake needing its own
#: assertions or a mock framework across the tmp_path/importlib boundary.
LAST_STAGE_DEADLINE = "unset"


async def run_second_pass_stage(candidates, shown_texts, *, research_question="",
                                 inclusion_criteria=None, exclusion_criteria=None,
                                 model=None, concurrency=None, batch_size=5,
                                 stage_deadline=None, now_fn=None, judge=None):
    """Wiring-only stand-in: a single call over every
    candidate this fake was given (this harness's own tests never exercise more than a
    handful of records at once, so batching/concurrency/budget accounting are not this
    fake's own concern -- those are covered directly in backend/tests). Routes through the
    same fake ``confirm_inclusions``/``apply_second_pass_guard`` above, so
    ``second_pass_established``/``second_pass_raises`` still control the outcome exactly as
    they did when this harness called ``confirm_inclusions`` directly."""
    global LAST_STAGE_DEADLINE
    LAST_STAGE_DEADLINE = stage_deadline
    if not candidates:
        return _StageResult(decisions=[], calls=[])
    try:
        result = await confirm_inclusions(
            candidates, research_question=research_question,
            inclusion_criteria=inclusion_criteria, exclusion_criteria=exclusion_criteria,
            model=model,
        )
    except ScreeningError:
        decisions = [
            _StageDecision("NEEDS_REVIEW", "second_pass_unavailable") for _ in candidates
        ]
        return _StageResult(decisions=decisions, calls=[])
    decisions = []
    for shown_text, answer in zip(shown_texts, result.answers):
        status, reason = apply_second_pass_guard(
            "INCLUDE", "", answers=answer, shown_text=shown_text,
        )
        second_pass = {{
            "population": answer.population.established,
            "outcome": answer.outcome.established,
            "study_type": answer.study_type,
        }}
        decisions.append(_StageDecision(status, reason, second_pass))
    call = _StageCall(result.provenance, list(range(len(candidates))))
    return _StageResult(decisions=decisions, calls=[call])
''',
        encoding="utf-8",
    )
    (fake / "app" / "config.py").write_text(
        "settings = type('S', (), {'deepseek_model': 'fake', "
        f"'screener_second_pass_model': {second_pass_model!r}, "
        f"'screener_second_pass_stage_budget_seconds': {stage_budget_seconds!r}}})()\n",
        encoding="utf-8",
    )
    return fake


def _clear_app_modules(monkeypatch):
    import sys

    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)


# ---------------------------------------------------------------- v2 wiring


def test_production_screener_reports_no_v2_devices_against_a_pre_v2_backend(
    tmp_path, monkeypatch,
):
    """A backend that only implements the core interface (every test above this section)
    must not be treated as broken: ``has_v2_devices`` is False and the batch pass's own
    INCLUDE decisions pass through untouched."""
    import asyncio

    fake = _fake_backend_module(tmp_path)
    _clear_app_modules(monkeypatch)
    screener = rs.ProductionScreener(backend_root=fake)
    assert screener.has_v2_devices is False
    out = asyncio.run(screener(PROTOCOL, [{"title": "t", "abstract": "some text"}]))
    assert out["statuses"] == ["INCLUDE"]
    assert out["second_pass_answers"] == [None]
    assert out["second_pass_calls"] == [None]


def test_type_demotion_routes_a_nonarticle_include_to_needs_review(tmp_path, monkeypatch):
    import asyncio

    fake = _fake_backend_module_v2(tmp_path, second_pass_established=[])
    _clear_app_modules(monkeypatch)
    screener = rs.ProductionScreener(backend_root=fake, prompt_version="v2")
    assert screener.has_v2_devices is True
    papers = [{"title": "t", "abstract": "some text", "type": "book", "is_paratext": False}]
    out = asyncio.run(screener(PROTOCOL, papers))
    assert out["statuses"] == ["NEEDS_REVIEW"]
    assert out["guard_reasons"] == ["nonarticle_type"]
    assert out["guard_applied"] == [True]
    assert out["second_pass_answers"] == [None]  # demoted before it could reach the second pass


def test_ordinary_type_reaches_the_second_pass_instead_of_being_demoted(tmp_path, monkeypatch):
    import asyncio

    fake = _fake_backend_module_v2(
        tmp_path, second_pass_established=[("established", "established", "study")],
    )
    _clear_app_modules(monkeypatch)
    screener = rs.ProductionScreener(backend_root=fake, prompt_version="v2")
    papers = [{"title": "t", "abstract": "some text", "type": "article", "is_paratext": False}]
    out = asyncio.run(screener(PROTOCOL, papers))
    assert out["statuses"] == ["INCLUDE"]
    assert out["second_pass_answers"] == [
        {"population": "established", "outcome": "established", "study_type": "study"}
    ]
    assert out["second_pass_calls"][0]["model_reported"] == "fake-second-pass-served"


def test_stage_deadline_falls_back_to_the_configured_stage_budget(tmp_path, monkeypatch):
    """run_second_pass_stage's own stage_deadline defaults to None (unbounded) when not
    passed, and has no settings fallback of its own. Unlike concurrency, this harness must
    compute one itself from settings.screener_second_pass_stage_budget_seconds, or an
    evaluation run has no stage budget at all."""
    import asyncio
    import time

    fake = _fake_backend_module_v2(
        tmp_path,
        second_pass_established=[("established", "established", "study")],
        stage_budget_seconds=1800.0,
    )
    _clear_app_modules(monkeypatch)
    screener = rs.ProductionScreener(backend_root=fake, prompt_version="v2")
    papers = [{"title": "t", "abstract": "some text", "type": "article", "is_paratext": False}]
    before = time.monotonic()
    asyncio.run(screener(PROTOCOL, papers))
    after = time.monotonic()

    import app.agents.relevance_screener_agent as fake_rs  # the fake now cached in sys.modules

    deadline = fake_rs.LAST_STAGE_DEADLINE
    assert deadline is not None
    # Computed from settings.screener_second_pass_stage_budget_seconds (1800.0), around
    # "now" at call time -- a generous window covers the test's own real wall-clock cost.
    assert before + 1700.0 < deadline < after + 1900.0


def test_stage_deadline_is_none_when_the_backend_predates_the_stage_budget_setting(
    tmp_path, monkeypatch,
):
    """A backend whose settings module carries no screener_second_pass_stage_budget_seconds
    at all degrades to an unbounded stage_deadline, the same value passing neither would
    mean -- never a crash, and never a silently-wrong finite deadline."""
    import asyncio

    fake = _fake_backend_module_v2(
        tmp_path,
        second_pass_established=[("established", "established", "study")],
        stage_budget_seconds=None,
    )
    _clear_app_modules(monkeypatch)
    screener = rs.ProductionScreener(backend_root=fake, prompt_version="v2")
    papers = [{"title": "t", "abstract": "some text", "type": "article", "is_paratext": False}]
    asyncio.run(screener(PROTOCOL, papers))

    import app.agents.relevance_screener_agent as fake_rs

    assert fake_rs.LAST_STAGE_DEADLINE is None


def test_table_of_contents_abstract_routes_an_include_to_needs_review(tmp_path, monkeypatch):
    import asyncio

    fake = _fake_backend_module_v2(tmp_path, second_pass_established=[])
    _clear_app_modules(monkeypatch)
    screener = rs.ProductionScreener(backend_root=fake, prompt_version="v2")
    toc = "1. Introduction 2. Method 3. Results 4. Discussion 5. Conclusion 6. References"
    papers = [{"title": "t", "abstract": toc, "type": "article", "is_paratext": False}]
    out = asyncio.run(screener(PROTOCOL, papers))
    assert out["statuses"] == ["NEEDS_REVIEW"]
    assert out["guard_reasons"] == ["table_of_contents"]


def test_second_pass_demotes_on_a_not_established_answer(tmp_path, monkeypatch):
    import asyncio

    fake = _fake_backend_module_v2(
        tmp_path, second_pass_established=[("established", "not_established", "study")],
    )
    _clear_app_modules(monkeypatch)
    screener = rs.ProductionScreener(backend_root=fake, prompt_version="v2")
    papers = [{"title": "t", "abstract": "some text", "type": None, "is_paratext": False}]
    out = asyncio.run(screener(PROTOCOL, papers))
    assert out["statuses"] == ["NEEDS_REVIEW"]
    assert out["guard_reasons"] == ["not_established"]
    assert out["second_pass_answers"] == [
        {"population": "established", "outcome": "not_established", "study_type": "study"}
    ]


def test_second_pass_failure_routes_the_batch_passs_own_include_to_needs_review(
    tmp_path, monkeypatch,
):
    """A second-pass call that raises must not crash the batch, and must not leave its
    candidates' INCLUDE standing unconfirmed either: a second pass that cannot run leaves
    every one of its own candidates for a human to read, exactly the same "never silently
    ship an unconfirmed INCLUDE" rule app.services.smart_search follows on this same
    failure. The harness's own second_pass_unavailable guard reason matches the backend's,
    so a run that hit this path is visible in the same place either codebase reports it."""
    import asyncio

    fake = _fake_backend_module_v2(tmp_path, second_pass_raises=True)
    _clear_app_modules(monkeypatch)
    screener = rs.ProductionScreener(backend_root=fake, prompt_version="v2")
    papers = [{"title": "t", "abstract": "some text", "type": "article", "is_paratext": False}]
    out = asyncio.run(screener(PROTOCOL, papers))
    assert out["statuses"] == ["NEEDS_REVIEW"]
    assert out["guard_reasons"] == ["second_pass_unavailable"]
    assert out["second_pass_answers"] == [None]
    # A chunk that raises never records a call at all (no tokens were billed for it), so
    # this candidate's own entry stays None rather than carrying an "error" string; the
    # guard_reason above is what a reader now checks.
    assert out["second_pass_calls"] == [None]


def test_v1_prompt_never_reaches_the_v2_devices(tmp_path, monkeypatch):
    """v1 asks for no criterion, quote or anchor at all; the v2 devices must stay off
    for it, the same way apply_decision_guard itself is skipped for v1 in the backend."""
    import asyncio

    fake = _fake_backend_module_v2(tmp_path, second_pass_established=[])
    _clear_app_modules(monkeypatch)
    screener = rs.ProductionScreener(backend_root=fake, prompt_version="v1")
    papers = [{"title": "t", "abstract": "some text", "type": "book", "is_paratext": False}]
    out = asyncio.run(screener(PROTOCOL, papers))
    assert out["statuses"] == ["INCLUDE"]  # the type demotion never ran


def test_rows_for_batch_carries_type_is_paratext_and_second_pass():
    batch = [dict(RECORDS[0], type="article", is_paratext=False)]
    outcome = _ok_outcome(
        include=[True],
        reasons=["ok"],
        second_pass_answers=[
            {"population": "established", "outcome": "established", "study_type": "study"}
        ],
        second_pass_call={
            "call_id": "sp1", "model_reported": "fake-second-pass-served",
            "input_tokens": 55, "output_tokens": 15,
        },
    )
    rows = rs.rows_for_batch(batch, 0, outcome, "label_abstract_screening")
    assert rows[0]["type"] == "article"
    assert rows[0]["is_paratext"] is False
    assert rows[0]["second_pass"] == {
        "population": "established", "outcome": "established", "study_type": "study",
    }
    assert rows[0]["second_pass_call_id"] == "sp1"
    assert rows[0]["second_pass_model_reported"] == "fake-second-pass-served"
    assert rows[0]["second_pass_input_tokens"] == 55
    assert rows[0]["second_pass_output_tokens"] == 15


def test_rows_for_batch_second_pass_fields_are_none_when_it_never_ran():
    batch = [dict(RECORDS[0])]
    rows = rs.rows_for_batch(batch, 0, _ok_outcome(), "label_abstract_screening")
    assert rows[0]["type"] == ""
    assert rows[0]["is_paratext"] is False
    assert rows[0]["second_pass"] is None
    assert rows[0]["second_pass_call_id"] is None
    assert rows[0]["second_pass_model_reported"] is None


def test_build_meta_prices_the_second_pass_once_per_call_not_once_per_row():
    """Two rows sharing one second-pass call (both INCLUDE survivors of the same batch) must
    not double the second pass's own tokens/cost."""
    rows = [
        {
            **_row_defaults(i),
            "predicted": 1, "reason": "ok", "padded_decision": False, "pass_number": 1,
            "batch_index": 0, "call_id": "c0",
            "second_pass_call_id": "sp1",
            "second_pass_model_reported": "deepseek-flash",
            "second_pass_input_tokens": 100,
            "second_pass_output_tokens": 40,
        }
        for i in range(2)
    ]
    meta = rs.build_meta(
        rows, existing=None, dataset="d", run="A", protocol=PROTOCOL, n_records=2, n_batches=1,
        batch_size=10, concurrency=1, model_configured="deepseek-chat", temperature=0.0,
        prompt_versions=["sha256:x"], price={}, dry_run=False, session_started="t0",
        limit=None,
    )
    assert meta["second_pass"]["calls"] == 1
    assert meta["second_pass"]["total_input_tokens"] == 100
    assert meta["second_pass"]["total_output_tokens"] == 40
    assert meta["second_pass"]["model_reported"] == ["deepseek-flash"]


def test_build_meta_second_pass_is_empty_when_nothing_reached_it():
    rows = [
        {
            **_row_defaults(0),
            "predicted": 1, "reason": "ok", "padded_decision": False, "pass_number": 1,
            "batch_index": 0, "call_id": "c0",
        }
    ]
    meta = rs.build_meta(
        rows, existing=None, dataset="d", run="A", protocol=PROTOCOL, n_records=1, n_batches=1,
        batch_size=10, concurrency=1, model_configured="deepseek-chat", temperature=0.0,
        prompt_versions=["sha256:x"], price={}, dry_run=False, session_started="t0",
        limit=None,
    )
    assert meta["second_pass"] == {
        "calls": 0, "model_reported": [], "total_input_tokens": 0, "total_output_tokens": 0,
        "total_cost": None,
    }


def test_production_screener_forwards_prompt_version_and_abstract_cap(tmp_path, monkeypatch):
    import asyncio
    import importlib

    fake = _fake_backend_module(tmp_path)
    _clear_app_modules(monkeypatch)
    screener = rs.ProductionScreener(backend_root=fake, prompt_version="v1", abstract_limit=500)
    out = asyncio.run(screener(PROTOCOL, [{"title": "t", "abstract": "z" * 600}]))
    mod = importlib.import_module("app.agents.relevance_screener_agent")
    assert mod.CALLS[-1] == {"prompt_version": "v1", "abstract_limit": 500}
    # v1 at cap 500: bare "..." truncation, no cut marker, no tail
    assert out["reasons"][0] == "z" * 500 + "..."
    assert " [...] " not in out["reasons"][0]
    assert out["statuses"] == ["INCLUDE"] and out["guard_applied"] == [False]
    _clear_app_modules(monkeypatch)


def test_production_screener_v2_cap_500_differs_from_v1_at_the_same_cap(tmp_path, monkeypatch):
    import asyncio
    import importlib

    fake = _fake_backend_module(tmp_path)
    _clear_app_modules(monkeypatch)
    screener = rs.ProductionScreener(backend_root=fake, prompt_version="v2", abstract_limit=500)
    out = asyncio.run(screener(PROTOCOL, [{"title": "t", "abstract": "z" * 600}]))
    mod = importlib.import_module("app.agents.relevance_screener_agent")
    assert mod.CALLS[-1] == {"prompt_version": "v2", "abstract_limit": 500}
    assert " [...] " in out["reasons"][0]
    assert out["reasons"][0] != "z" * 500 + "..."
    _clear_app_modules(monkeypatch)


def test_production_screener_defaults_to_v3_and_no_cap_override(tmp_path, monkeypatch):
    """v3 is the shipped prompt, so an evaluation run started with no --prompt-version flag
    matches production by default."""
    import asyncio
    import importlib

    fake = _fake_backend_module(tmp_path)
    _clear_app_modules(monkeypatch)
    screener = rs.ProductionScreener(backend_root=fake)
    asyncio.run(screener(PROTOCOL, [{"title": "t", "abstract": "short"}]))
    mod = importlib.import_module("app.agents.relevance_screener_agent")
    assert mod.CALLS[-1] == {"prompt_version": "v3", "abstract_limit": 3000}
    _clear_app_modules(monkeypatch)


def _fake_backend_module_with_reask(tmp_path):
    """Like :func:`_fake_backend_module`, but ``screen_papers`` also returns a
    ``reask_provenance`` and a ``reask_error``, so :meth:`ProductionScreener.__call__`'s
    forwarding of both can be exercised without a real backend."""
    fake = tmp_path / "backend"
    (fake / "app" / "agents").mkdir(parents=True)
    (fake / "app" / "__init__.py").write_text("", encoding="utf-8")
    (fake / "app" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    (fake / "app" / "agents" / "relevance_screener_agent.py").write_text(
        '''
ABSTRACT_CHAR_LIMIT = 3000


class ScreeningError(Exception):
    pass


class _Prov:
    def __init__(self, **kw):
        self.model_configured = "fake"
        self.model_reported = "fake"
        self.system_fingerprint = None
        self.provider_response_id = None
        self.temperature = 0.0
        self.prompt_version = "sha256:fake"
        self.input_tokens = 100
        self.output_tokens = 10
        self.__dict__.update(kw)


class ScreeningBatchResult:
    def __init__(self, **kw):
        self.__dict__.update(kw)


async def screen_papers(q, papers, inc, exc, **kwargs):
    n = len(papers)
    return ScreeningBatchResult(
        include=[True] * n,
        statuses=["INCLUDE"] * n,
        criteria_ids=[""] * n,
        quotes=[""] * n,
        reasons=["ok"] * n,
        guard_applied=[False] * n,
        guard_conversions=0,
        provenance=_Prov(model_reported="deepseek-v4-flash", system_fingerprint="fp_a"),
        reask_provenance=_Prov(
            model_reported="deepseek-v4-flash-2", system_fingerprint="fp_reask",
            input_tokens=70, output_tokens=7,
        ),
        reask_error=None,
    )
''',
        encoding="utf-8",
    )
    (fake / "app" / "config.py").write_text(
        "settings = type('S', (), {'deepseek_model': 'fake'})()\n", encoding="utf-8"
    )
    return fake


def test_production_screener_sums_tokens_and_counts_two_calls_when_a_reask_answered(
    tmp_path, monkeypatch
):
    import asyncio

    fake = _fake_backend_module_with_reask(tmp_path)
    _clear_app_modules(monkeypatch)
    screener = rs.ProductionScreener(backend_root=fake)
    out = asyncio.run(screener(PROTOCOL, [{"title": "t", "abstract": "short"}]))
    _clear_app_modules(monkeypatch)

    assert out["calls"] == 2
    assert out["input_tokens"] == 170 and out["output_tokens"] == 17
    assert out["model_reported"] == "deepseek-v4-flash"
    assert out["reask_model_reported"] == "deepseek-v4-flash-2"
    assert out["system_fingerprint"] == "fp_a"
    assert out["reask_system_fingerprint"] == "fp_reask"
    assert out["reask_error"] is None


def test_production_screener_reports_one_call_and_no_reask_fields_without_a_reask(
    tmp_path, monkeypatch
):
    import asyncio

    fake = _fake_backend_module(tmp_path)
    _clear_app_modules(monkeypatch)
    screener = rs.ProductionScreener(backend_root=fake)
    out = asyncio.run(screener(PROTOCOL, [{"title": "t", "abstract": "short"}]))
    _clear_app_modules(monkeypatch)

    assert out["calls"] == 1
    assert out["reask_model_reported"] is None
    assert out["reask_system_fingerprint"] is None
    assert out["reask_error"] is None


def test_prompt_version_and_abstract_cap_flags_are_parsed():
    args = rs.parse_args(
        ["--dataset", "X", "--run", "A", "--prompt-version", "v1", "--abstract-cap", "500"]
    )
    assert args.prompt_version == "v1" and args.abstract_cap == 500
    default = rs.parse_args(["--dataset", "X", "--run", "A"])
    # v3 is the shipped prompt and the CLI default, so an evaluation run started with no
    # --prompt-version flag matches production.
    assert default.prompt_version == "v3" and default.abstract_cap is None
    v2 = rs.parse_args(["--dataset", "X", "--run", "A", "--prompt-version", "v2"])
    assert v2.prompt_version == "v2"
    with pytest.raises(SystemExit):
        rs.parse_args(["--dataset", "X", "--run", "A", "--prompt-version", "v4"])


def test_prompt_version_and_abstract_cap_flags_reach_the_meta_block(tmp_path, monkeypatch):
    """--prompt-version/--abstract-cap must land in the written .meta.json, not just in
    parse_args's namespace. Runs main_async for real (through the fake backend, no network)
    and reads the .meta.json back from disk."""
    import asyncio

    from common import read_json

    fake = _fake_backend_module(tmp_path)
    _clear_app_modules(monkeypatch)
    data, protos, results = _setup_dataset(tmp_path)
    argv = [
        "--dataset", "X", "--run", "A", "--batch-size", "3", "--confirm-cost",
        "--prompt-version", "v1", "--abstract-cap", "500",
        "--data-dir", str(data), "--protocols-dir", str(protos), "--results-dir", str(results),
    ]
    args = rs.parse_args(argv)
    screener = rs.ProductionScreener(
        backend_root=fake, prompt_version=args.prompt_version, abstract_limit=args.abstract_cap
    )
    assert asyncio.run(rs.main_async(args, screener=screener)) == 0
    meta = read_json(results / "X_runA.meta.json")
    assert meta["prompt_version_flag"] == "v1"
    assert meta["abstract_cap"] == 500
    _clear_app_modules(monkeypatch)


def test_abstract_cap_flag_resolves_to_backend_constant_when_not_given(tmp_path, monkeypatch):
    """No --abstract-cap given: the meta must still record the number actually used (the
    backend's frozen ABSTRACT_CHAR_LIMIT), not None ("not overridden" is not "unknown")."""
    import asyncio

    from common import read_json

    fake = _fake_backend_module(tmp_path)
    _clear_app_modules(monkeypatch)
    data, protos, results = _setup_dataset(tmp_path)
    argv = [
        "--dataset", "X", "--run", "A", "--batch-size", "3", "--confirm-cost",
        "--data-dir", str(data), "--protocols-dir", str(protos), "--results-dir", str(results),
    ]
    args = rs.parse_args(argv)
    screener = rs.ProductionScreener(backend_root=fake)
    assert asyncio.run(rs.main_async(args, screener=screener)) == 0
    meta = read_json(results / "X_runA.meta.json")
    assert meta["prompt_version_flag"] == "v3"
    assert meta["abstract_cap"] == 3000  # fake backend's ABSTRACT_CHAR_LIMIT
    _clear_app_modules(monkeypatch)


# ---------------------------------- padded decisions


def _fake_backend_module_with_padding(tmp_path, padded):
    """Minimal fake backend module whose ``screen_papers`` always reports ``padded`` decisions
    the model did not return (padded to the tail of ``statuses``/``reasons`` as NEEDS_REVIEW /
    ``NO_DECISION_REASON``, matching the shipped backend), so :meth:`ProductionScreener.__call__`
    forwarding and the downstream ``padded_decision`` row flag can be tested without a real
    model call."""
    fake = tmp_path / f"backend_padded_{padded}"
    (fake / "app" / "agents").mkdir(parents=True)
    (fake / "app" / "__init__.py").write_text("", encoding="utf-8")
    (fake / "app" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    (fake / "app" / "agents" / "relevance_screener_agent.py").write_text(
        f'''
class ScreeningError(Exception):
    pass


class _Prov:
    model_configured = "fake"
    model_reported = "fake"
    system_fingerprint = None
    provider_response_id = None
    temperature = 0.0
    prompt_version = "sha256:fake"
    input_tokens = 1
    output_tokens = 1


class ScreeningBatchResult:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    @property
    def include(self):
        return [s == "INCLUDE" for s in self.statuses]


async def screen_papers(q, papers, inc, exc, *, prompt_version="v2", abstract_limit=3000):
    n = len(papers)
    n_ok = n - {padded}
    return ScreeningBatchResult(
        statuses=["INCLUDE"] * n_ok + ["NEEDS_REVIEW"] * {padded},
        criteria_ids=[""] * n,
        quotes=[""] * n,
        reasons=["ok"] * n_ok + ["no decision returned"] * {padded},
        guard_applied=[False] * n,
        guard_conversions=0,
        padded={padded},
        provenance=_Prov(),
    )
''',
        encoding="utf-8",
    )
    (fake / "app" / "config.py").write_text(
        "settings = type('S', (), {'deepseek_model': 'fake'})()\n", encoding="utf-8"
    )
    return fake


def test_production_screener_forwards_padded_count(tmp_path, monkeypatch):
    import asyncio

    fake = _fake_backend_module_with_padding(tmp_path, 1)
    _clear_app_modules(monkeypatch)
    screener = rs.ProductionScreener(backend_root=fake)
    out = asyncio.run(
        screener(PROTOCOL, [{"title": "t1", "abstract": "a"}, {"title": "t2", "abstract": "b"}])
    )
    assert out["padded"] == 1
    _clear_app_modules(monkeypatch)


def test_rows_and_meta_flag_padded_decision_from_forwarded_count(tmp_path, monkeypatch):
    import asyncio

    fake = _fake_backend_module_with_padding(tmp_path, 1)
    _clear_app_modules(monkeypatch)
    screener = rs.ProductionScreener(backend_root=fake)
    outcome = asyncio.run(
        screener(PROTOCOL, [{"title": "t1", "abstract": "a"}, {"title": "t2", "abstract": "b"}])
    )
    rows = rs.rows_for_batch(RECORDS[:2], 0, outcome, "label_included")
    # record 2 got the padded NEEDS_REVIEW; predicted stays 0 (not "INCLUDE"), so the narrower
    # legacy padded_include is False even though this row *is* a padded non-decision.
    assert rows[0]["padded_decision"] is False
    assert rows[1]["status"] == "NEEDS_REVIEW" and rows[1]["padded_include"] is False
    assert rows[1]["padded_decision"] is True
    assert rs.build_meta(rows, **_meta_kwargs())["n_padded_decisions"] == 1
    _clear_app_modules(monkeypatch)


# ---------------------------------- re-ask instead of padding


class _PaddingScreener(_FakeScreener):
    """Pads every decision a call is short of, healing (deciding every record) only once a
    call's batch is at most ``heal_at`` records; ``heal_at=0`` never heals at all. Stands in
    for a model that skips records more often the more of them it is shown at once, so the
    batch-of-3-then-1 re-ask cascade can be exercised without a real LLM."""

    def __init__(self, heal_at: int):
        super().__init__()
        self.heal_at = heal_at
        self.batch_sizes_seen: list[int] = []

    async def __call__(self, protocol, papers):
        self.calls += 1
        n = len(papers)
        self.batch_sizes_seen.append(n)
        n_ok = n if (self.heal_at and n <= self.heal_at) else 0
        statuses = ["INCLUDE"] * n_ok + ["NEEDS_REVIEW"] * (n - n_ok)
        reasons = ["ok"] * n_ok + [rs.NO_DECISION_REASON] * (n - n_ok)
        return {
            "include": [s == "INCLUDE" for s in statuses],
            "reasons": reasons,
            "statuses": statuses,
            "criteria_ids": [""] * n,
            "quotes": [""] * n,
            "guard_applied": [False] * n,
            "guard_reasons": [""] * n,
            "to_confirm": [[] for _ in range(n)],
            "padded": n - n_ok,
            "model_reported": "deepseek-v4-flash",
            "system_fingerprint": None,
            "provider_response_id": None,
            "temperature": 0.0,
            "prompt_version": "v2",
            "input_tokens": 10,
            "output_tokens": 5,
        }


def test_run_screening_reasks_padded_records_in_batches_of_three_before_giving_up(tmp_path):
    import asyncio

    from common import read_json, read_jsonl

    data, protos, results = _setup_dataset(tmp_path)
    base = ["--dataset", "X", "--run", "A", "--batch-size", "7", "--confirm-cost",
            "--data-dir", str(data), "--protocols-dir", str(protos), "--results-dir",
            str(results)]
    screener = _PaddingScreener(heal_at=3)
    asyncio.run(rs.main_async(rs.parse_args(base), screener=screener))

    rows = read_jsonl(results / "X_runA.jsonl")
    assert len(rows) == 7 and len({r["record_id"] for r in rows}) == 7
    assert not any(r["padded_decision"] for r in rows)  # every record eventually decided
    assert {r["pass_number"] for r in rows} == {2}  # pass 1 padded all 7; pass 2 (<=3) healed
    meta = read_json(results / "X_runA.meta.json")
    assert meta["n_padded_decisions"] == 0
    assert meta["n_batches"] == 1 + 3  # the original batch plus the three pass-2 sub-batches


def test_run_screening_only_heals_stubborn_records_at_the_single_record_pass(tmp_path):
    import asyncio

    from common import read_json, read_jsonl

    data, protos, results = _setup_dataset(tmp_path)
    base = ["--dataset", "X", "--run", "A", "--batch-size", "7", "--confirm-cost",
            "--data-dir", str(data), "--protocols-dir", str(protos), "--results-dir",
            str(results)]
    screener = _PaddingScreener(heal_at=1)
    asyncio.run(rs.main_async(rs.parse_args(base), screener=screener))

    rows = read_jsonl(results / "X_runA.jsonl")
    assert len(rows) == 7
    assert not any(r["padded_decision"] for r in rows)
    pass_numbers = {r["pass_number"] for r in rows}
    assert 3 in pass_numbers  # some records needed the single-record pass to be decided
    assert 1 not in pass_numbers  # every record was padded at pass 1 and got reprocessed
    meta = read_json(results / "X_runA.meta.json")
    assert meta["n_padded_decisions"] == 0


def test_run_screening_keeps_the_padded_marker_only_after_the_single_record_pass_fails_too(
    tmp_path,
):
    import asyncio

    from common import read_json, read_jsonl

    data, protos, results = _setup_dataset(tmp_path)
    base = ["--dataset", "X", "--run", "A", "--batch-size", "7", "--confirm-cost",
            "--data-dir", str(data), "--protocols-dir", str(protos), "--results-dir",
            str(results)]
    screener = _PaddingScreener(heal_at=0)  # never decides anything
    asyncio.run(rs.main_async(rs.parse_args(base), screener=screener))

    rows = read_jsonl(results / "X_runA.jsonl")
    assert len(rows) == 7
    assert all(r["padded_decision"] for r in rows)
    assert {r["pass_number"] for r in rows} == {3}  # stuck through the last (single) pass
    assert 1 in screener.batch_sizes_seen  # the single-record pass really was attempted
    meta = read_json(results / "X_runA.meta.json")
    assert meta["n_padded_decisions"] == 7


def test_reask_pass_batches_padded_records_in_the_runs_own_shuffle_order(tmp_path):
    """A re-ask sub-batch's record order follows the run's own deterministic shuffle
    (record_order_seed), not a set's arbitrary iteration order."""
    import asyncio

    class _OrderRecordingPaddingScreener(_PaddingScreener):
        def __init__(self, heal_at):
            super().__init__(heal_at)
            self.titles_seen: list[list[str]] = []

        async def __call__(self, protocol, papers):
            self.titles_seen.append([p["title"] for p in papers])
            return await super().__call__(protocol, papers)

    data, protos, results = _setup_dataset(tmp_path)
    base = ["--dataset", "X", "--run", "A", "--batch-size", "7", "--confirm-cost",
            "--data-dir", str(data), "--protocols-dir", str(protos), "--results-dir",
            str(results)]
    screener = _OrderRecordingPaddingScreener(heal_at=3)
    asyncio.run(rs.main_async(rs.parse_args(base), screener=screener))

    expected_order = [
        r["record_id"] for r in rs.shuffle_records(RECORDS, "X", rs.DEFAULT_RECORD_ORDER_SEED)
    ]
    # call 0 is pass 1 (all 7 records); call 1 is pass 2's first sub-batch of 3, which must
    # be the first 3 padded records in the run's own shuffle order.
    pass2_first_batch_ids = [int(t.split()[-1]) for t in screener.titles_seen[1]]
    assert pass2_first_batch_ids == expected_order[:3]


def test_reask_pass_only_prunes_the_subbatch_about_to_run(tmp_path):
    """A sub-batch's own rows are pruned immediately before its own call, not for the whole
    pass up front. An abort between sub-batches (here, a prompt-version mismatch on the
    second pass-2 sub-batch) must not discard a still-pending sub-batch's already-written
    (pass-1) rows."""
    import asyncio

    from common import read_jsonl

    class _AbortOnThirdCallScreener(_PaddingScreener):
        async def __call__(self, protocol, papers):
            outcome = await super().__call__(protocol, papers)
            if self.calls == 3:  # the second pass-2 sub-batch
                outcome["prompt_version"] = "MISMATCH"
            return outcome

    data, protos, results = _setup_dataset(tmp_path)
    base = ["--dataset", "X", "--run", "A", "--batch-size", "7", "--confirm-cost",
            "--prompt-version-expect", "v2",
            "--data-dir", str(data), "--protocols-dir", str(protos), "--results-dir",
            str(results)]
    screener = _AbortOnThirdCallScreener(heal_at=3)
    with pytest.raises(SystemExit) as exc:
        asyncio.run(rs.main_async(rs.parse_args(base), screener=screener))
    assert exc.value.code == 2

    rows = read_jsonl(results / "X_runA.jsonl")
    ids = {r["record_id"] for r in rows}
    assert ids == set(range(1, 8))  # every record still has a row; none silently discarded


def test_next_batch_index_and_reask_batches_run_are_seeded_from_disk(tmp_path):
    """A resumed session seeds its re-ask batch index from the highest ``batch_index``
    already on disk (not just this session's own fresh batch map) and carries
    ``reask_batches_run`` forward from the existing meta (the same way ``sessions`` is
    carried), so neither collides with nor drops an earlier session's own re-ask
    sub-batches."""
    import asyncio
    import json

    from common import read_json, read_jsonl, write_json

    data, protos, results = _setup_dataset(tmp_path)
    results.mkdir(parents=True, exist_ok=True)
    out_path = results / "X_runA.jsonl"
    meta_path = results / "X_runA.meta.json"

    # Simulate a completed prior session: record 1 was resolved by an earlier re-ask
    # sub-batch and its row still sits at batch_index=5 (>= len(batches)=1); records 2-7 are
    # still stuck from pass 1 (batch_index=0, padded_decision=True) and are retryable.
    rows = []
    rows.append({
        **{k: v for k, v in _row_defaults(1).items()},
        "predicted": 1, "reason": "ok", "padded_decision": False, "pass_number": 2,
        "batch_index": 5, "call_id": "prior-c1",
    })
    for i in range(2, 8):
        rows.append({
            **{k: v for k, v in _row_defaults(i).items()},
            "predicted": 0, "reason": rs.NO_DECISION_REASON, "padded_decision": True,
            "pass_number": 1, "batch_index": 0, "call_id": "prior-c0",
        })
    out_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    write_json(meta_path, {
        "dry_run": False, "reask_batches_run": 7, "sessions": [],
        "started": "2026-09-02T12:00:00+00:00", "prompt_version": ["v2"], "temperature": 0.0,
    })

    base = ["--dataset", "X", "--run", "A", "--batch-size", "7", "--confirm-cost",
            "--retry-failed",
            "--data-dir", str(data), "--protocols-dir", str(protos), "--results-dir",
            str(results)]
    screener = _PaddingScreener(heal_at=3)  # pads the 6 retried records at pass 1, heals at pass 2
    asyncio.run(rs.main_async(rs.parse_args(base), screener=screener))

    all_rows = read_jsonl(out_path)
    assert {r["record_id"] for r in all_rows} == set(range(1, 8))
    record_1_row = next(r for r in all_rows if r["record_id"] == 1)
    assert record_1_row["batch_index"] == 5  # untouched: it was never selected for retry
    pass2_rows = [r for r in all_rows if r["pass_number"] == 2 and r["record_id"] != 1]
    assert pass2_rows  # this session's own re-ask really ran
    # seeded from disk (max existing batch_index 5, +1 = 6), not from len(batches) (1) alone
    assert min(r["batch_index"] for r in pass2_rows) == 6

    meta = read_json(meta_path)
    # carried forward (7) plus this session's own new re-ask sub-batches (2: two groups of 3)
    assert meta["reask_batches_run"] == 9
    assert meta["n_batches"] == 1 + 9


def _row_defaults(record_id: int) -> dict:
    return {
        "record_id": record_id, "label": record_id % 3 == 0, "label_included": record_id % 3 == 0,
        "label_abstract_screening": record_id % 2, "has_abstract": True, "has_title": True,
        "title": f"Paper {record_id}", "status": "NEEDS_REVIEW", "criterion": "", "quote": "",
        "guard_applied": False, "guard_reason": "", "to_confirm": [], "padded_include": False,
        "called_at": "2026-09-02T12:00:00+00:00", "latency_s": 1.0,
        "input_tokens": 10, "output_tokens": 5, "cache_read_tokens": None,
        "model_reported": "deepseek-v4-flash", "system_fingerprint": "fp",
        "provider_response_id": None, "attempts": 1, "calls": 1,
        "reask_model_reported": None, "reask_system_fingerprint": None, "reask_error": None,
    }


def test_retry_failed_also_reattempts_records_still_padded_after_all_passes(tmp_path):
    import asyncio

    from common import read_json, read_jsonl

    data, protos, results = _setup_dataset(tmp_path)
    base = ["--dataset", "X", "--run", "A", "--batch-size", "7", "--confirm-cost",
            "--data-dir", str(data), "--protocols-dir", str(protos), "--results-dir",
            str(results)]
    stubborn = _PaddingScreener(heal_at=0)
    asyncio.run(rs.main_async(rs.parse_args(base), screener=stubborn))
    stuck = read_jsonl(results / "X_runA.jsonl")
    assert len(stuck) == 7 and all(r["padded_decision"] for r in stuck)

    healer = _FakeScreener()
    # without --retry-failed: a stuck-padded row is not re-attempted, same as a genuine
    # UNSCREENED failure without the flag
    asyncio.run(rs.main_async(rs.parse_args(base), screener=healer))
    assert healer.calls == 0
    assert all(r["padded_decision"] for r in read_jsonl(results / "X_runA.jsonl"))

    # with --retry-failed: every still-padded record is dropped and re-screened, once, with
    # no duplicate rows
    asyncio.run(rs.main_async(rs.parse_args(base + ["--retry-failed"]), screener=healer))
    assert healer.calls > 0
    rows = read_jsonl(results / "X_runA.jsonl")
    ids = [r["record_id"] for r in rows]
    assert len(ids) == 7 and len(set(ids)) == 7
    assert not any(r["padded_decision"] for r in rows)
    meta = read_json(results / "X_runA.meta.json")
    assert meta["retried_failed"] == 7


class _RealisticTokenPaddingScreener(_PaddingScreener):
    """Like :class:`_PaddingScreener`, but reports input/output tokens computed by the same
    assumed-cost formula :func:`run_screening.cost_upper_bound`/``reask_cost_allowance`` use,
    and always reports two calls' worth of usage (``"calls": 2``, both calls' tokens
    summed): the shape ``ProductionScreener`` reports whenever a harness batch's
    backend-internal re-ask answers (every one of this harness's own calls, including its
    pass-2/pass-3 re-asks, can trigger that internal re-ask, not only the pass-1 batch).
    ``real_cost_usd`` is this screener's own running tally of every real call it reported, at
    the same peak-tariff assumptions the gate uses, so a test can compare the pre-run cost
    gate's printed worst-case bound against a ground truth independent of the harness's own
    meta."""

    def __init__(self, heal_at: int):
        super().__init__(heal_at)
        self.real_cost_usd = 0.0

    async def __call__(self, protocol, papers):
        outcome = await super().__call__(protocol, papers)
        chars = sum(rs._approx_shown_chars(p, None) for p in papers)
        per_call_input = chars / rs.CHARS_PER_INPUT_TOKEN + rs.PROMPT_FRAMING_TOKENS_PER_BATCH
        per_call_output = rs.ASSUMED_OUTPUT_TOKENS_PER_BATCH
        rates = rs.DEEPSEEK_PRICES["models"]["deepseek-v4-flash"]
        per_call_cost = (
            per_call_input * rates["input_cache_miss"]["peak"]
            + per_call_output * rates["output"]["peak"]
        ) / 1_000_000.0
        # Two real calls for this one harness-level outcome: the harness's own call, plus the
        # backend's own internal re-ask, which answers here every time (the worst case the
        # gate must price for).
        outcome["input_tokens"] = per_call_input * 2
        outcome["output_tokens"] = per_call_output * 2
        outcome["calls"] = 2
        self.real_cost_usd += per_call_cost * 2
        return outcome


def test_cost_gate_bound_is_never_below_the_runs_actual_worst_case_cost(tmp_path, capsys):
    """require_cost_confirmation must price the re-ask cascade (the backend's own internal
    re-ask, which fires on every call this harness makes, its pass-2/pass-3 re-asks
    included, not only the pass-1 batch), not the pass-1 batches alone, or a pathological
    run (every record padded through every pass) can spend more than the confirmed
    worst-case bound promised. Checked against the screener's own tally of every real call
    it reported, not the harness's own meta."""
    import asyncio
    import re

    data, protos, results = _setup_dataset(tmp_path)
    base = ["--dataset", "X", "--run", "A", "--batch-size", "7",
            "--data-dir", str(data), "--protocols-dir", str(protos), "--results-dir",
            str(results)]

    gate_screener = _RealisticTokenPaddingScreener(heal_at=0)
    gate_screener.model_configured = "deepseek-chat"  # priced model for the gate message
    with pytest.raises(SystemExit) as exc:
        asyncio.run(rs.main_async(rs.parse_args(base), screener=gate_screener))
    assert exc.value.code == rs.EXIT_COST_NOT_CONFIRMED
    printed = capsys.readouterr().out
    match = re.search(r"worst case with the full re-ask cascade <= ([\d.]+) USD", printed)
    assert match, printed
    printed_worst_case = float(match.group(1))

    run_screener = _RealisticTokenPaddingScreener(heal_at=0)
    run_screener.model_configured = "deepseek-chat"
    asyncio.run(rs.main_async(rs.parse_args(base + ["--confirm-cost"]), screener=run_screener))
    assert run_screener.calls == 11  # 1 (pass 1) + 3 (pass 2) + 7 (pass 3), heal_at=0
    assert run_screener.real_cost_usd > 0
    assert printed_worst_case >= run_screener.real_cost_usd


def test_prompt_version_expect_aborts_on_mismatch(tmp_path):
    import asyncio

    data, protos, results = _setup_dataset(tmp_path)
    argv = ["--dataset", "X", "--run", "A", "--batch-size", "3", "--confirm-cost",
            "--prompt-version-expect", "sha256:frozen", "--data-dir", str(data),
            "--protocols-dir", str(protos), "--results-dir", str(results)]
    screener = _FakeScreener(0.0)
    screener.calls = 0
    with pytest.raises(SystemExit) as exc:
        asyncio.run(rs.main_async(rs.parse_args(argv), screener=screener))
    assert exc.value.code == 2

    other_results = tmp_path / "results2"
    good_argv = ["--dataset", "X", "--run", "A", "--batch-size", "3", "--confirm-cost",
                 "--prompt-version-expect", "v1", "--data-dir", str(data),
                 "--protocols-dir", str(protos), "--results-dir", str(other_results)]
    assert asyncio.run(rs.main_async(rs.parse_args(good_argv), screener=_FakeScreener(0.0))) == 0


# ---------------------------------------------------------------- abstract cap shares


def test_abstract_length_shares_on_a_synthetic_corpus(tmp_path):
    import abstract_length_shares as als

    records = [
        {"record_id": 1, "abstract": "x" * 1000},
        {"record_id": 2, "abstract": "x" * 3000},
        {"record_id": 3, "abstract": "x" * 3500},
        {"record_id": 4, "abstract": "x" * 5000},
        {"record_id": 5, "abstract": ""},
        {"record_id": 6, "abstract": "   "},
    ]
    shares = als.length_shares(records)
    assert shares["n_records"] == 6 and shares["n_nonempty_abstracts"] == 4
    assert shares["mean"] == pytest.approx((1000 + 3000 + 3500 + 5000) / 4)
    assert shares["median"] == pytest.approx((3000 + 3500) / 2)
    assert shares["share_at_2000"] == pytest.approx(1 / 4)
    assert shares["share_at_3000"] == pytest.approx(2 / 4)
    assert shares["share_at_4000"] == pytest.approx(3 / 4)


def test_abstract_length_shares_empty_corpus_returns_none_aggregates():
    import abstract_length_shares as als

    shares = als.length_shares([{"record_id": 1, "abstract": ""}])
    assert shares["n_nonempty_abstracts"] == 0
    assert shares["mean"] is None and shares["median"] is None
    assert shares["share_at_2000"] is None


def test_abstract_length_shares_main_writes_json_and_prints_only_the_path(tmp_path, capsys):
    import json

    import abstract_length_shares as als

    data = tmp_path / "data"
    data.mkdir()
    with (data / "X.jsonl").open("w", encoding="utf-8") as fh:
        for i, n in enumerate([500, 2500, 4500], start=1):
            fh.write(json.dumps({"record_id": i, "abstract": "y" * n}) + "\n")
    rc = als.main(["--dataset", "X", "--data-dir", str(data)])
    assert rc == 0
    out_path = data / "X.abstract_lengths.json"
    printed = capsys.readouterr().out.strip()
    assert printed == str(out_path)
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert result["dataset"] == "X" and result["n_nonempty_abstracts"] == 3
    assert result["share_at_2000"] == pytest.approx(1 / 3)
    assert result["share_at_3000"] == pytest.approx(2 / 3)
    assert result["share_at_4000"] == pytest.approx(2 / 3)


def test_abstract_length_shares_main_exits_when_no_records(tmp_path):
    import abstract_length_shares as als

    with pytest.raises(SystemExit):
        als.main(["--dataset", "missing", "--data-dir", str(tmp_path)])


def test_refuses_to_mix_dry_run_and_paid_provenance_in_the_same_results_dir(tmp_path):
    """A --results-dir already written by a dry run must reject a paid run (and vice versa),
    so stand-in rows are never silently reported as production output (or the reverse)."""
    import asyncio

    from common import read_json, read_jsonl

    data, protos, results = _setup_dataset(tmp_path)
    dry_argv = ["--dataset", "X", "--run", "A", "--batch-size", "3", "--dry-run",
                "--data-dir", str(data), "--protocols-dir", str(protos), "--results-dir",
                str(results)]
    assert asyncio.run(rs.main_async(rs.parse_args(dry_argv))) == 0
    rows_before = read_jsonl(results / "X_runA.jsonl")
    paid_argv = ["--dataset", "X", "--run", "A", "--batch-size", "3", "--confirm-cost",
                 "--data-dir", str(data), "--protocols-dir", str(protos), "--results-dir",
                 str(results)]
    with pytest.raises(SystemExit) as exc:
        asyncio.run(rs.main_async(rs.parse_args(paid_argv), screener=_FakeScreener(0.0)))
    assert exc.value.code == 2
    assert read_jsonl(results / "X_runA.jsonl") == rows_before
    meta = read_json(results / "X_runA.meta.json")
    assert meta["dry_run"] is True  # untouched
    # the reverse direction is also refused: a paid meta, then a --dry-run pass
    other = tmp_path / "results2"
    assert asyncio.run(rs.main_async(
        rs.parse_args([*paid_argv[:-1], str(other)]), screener=_FakeScreener(0.0)
    )) == 0
    rows_before2 = read_jsonl(other / "X_runA.jsonl")
    with pytest.raises(SystemExit) as exc:
        asyncio.run(rs.main_async(rs.parse_args([*dry_argv[:-1], str(other)])))
    assert exc.value.code == 2
    assert read_jsonl(other / "X_runA.jsonl") == rows_before2


def test_default_results_dirs_are_the_result_of_record():
    """RESULTS_DIR in every screening script that writes or reads run output must point
    at results/v3 (the directory of record, evaluation/README.md "Results directories"),
    not a superseded version, so a bare invocation with no --results-dir uses the right
    data."""
    import baselines as bl

    assert rs.RESULTS_DIR == rs.HERE / "results" / "v3"
    assert wg.RESULTS_DIR == wg.HERE / "results" / "v3"
    assert bl.RESULTS_DIR == bl.HERE / "results" / "v3"
