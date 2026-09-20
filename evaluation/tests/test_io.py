"""Resumable JSONL, cost/latency, dotenv parsing and protocol loading."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from conftest import load_script_module

from common import (
    EVAL_ROOT,
    ProtocolError,
    append_jsonl,
    completed_ids,
    compute_cost,
    latency_stats,
    load_dotenv_values,
    load_protocol,
    markdown_table,
    pending_items,
    percentile,
    price_record,
    protocol_query_text,
    read_jsonl,
)


def test_jsonl_append_read_and_resume(tmp_path: Path):
    out = tmp_path / "nested" / "run.jsonl"
    assert append_jsonl(out, {"id": 1, "v": "a"}) == 1
    assert append_jsonl(out, [{"id": 2, "v": "b"}, {"id": 3, "v": "c"}]) == 2
    assert append_jsonl(out, []) == 0
    rows = read_jsonl(out)
    assert [r["id"] for r in rows] == [1, 2, 3]
    done = completed_ids(out, "id")
    assert done == {1, 2, 3}
    items = [{"id": i} for i in range(1, 6)]
    assert [it["id"] for it in pending_items(items, "id", done)] == [4, 5]
    assert read_jsonl(tmp_path / "missing.jsonl") == []
    assert completed_ids(tmp_path / "missing.jsonl", "id") == set()


def test_cost_and_latency():
    assert compute_cost(1_000_000, 500_000, 0.28, 0.42) == pytest.approx(0.28 + 0.21)
    assert compute_cost(10, 10, None, 0.42) is None
    rec = price_record("deepseek-chat", 0.28, 0.42, "https://example.org/pricing", "2026-09-02")
    assert rec["model"] == "deepseek-chat" and rec["input_per_mtok"] == 0.28
    assert rec["source_url"] == "https://example.org/pricing" and rec["accessed"] == "2026-09-02"
    empty = price_record("deepseek-chat", None, None)
    assert empty["input_per_mtok"] is None and empty["source_url"].startswith("https://")

    stats = latency_stats([1.0, 2.0, 3.0, 4.0, 10.0])
    assert stats["n"] == 5 and stats["median"] == 3.0 and stats["p90"] == 10.0
    assert stats["total"] == 20.0
    assert percentile([5, 1, 3], 50) == 3.0 and percentile([], 90) is None
    assert latency_stats([])["median"] is None


def test_load_dotenv_values(tmp_path: Path):
    env = tmp_path / ".env"
    lines = ["# comment", "A=1", 'B="two words"', "export C='x'", "BAD LINE", ""]
    env.write_text("\n".join(lines), encoding="utf-8")
    values = load_dotenv_values(env)
    assert values == {"A": "1", "B": "two words", "C": "x"}
    assert load_dotenv_values(tmp_path / "none") == {}


def _protocol_dict(**over):
    base = {
        "dataset_id": "X_2020",
        "source_review_doi": "10.1/abc",
        "research_question": "Does X affect Y?",
        "inclusion_criteria": ["studies about X"],
        "exclusion_criteria": ["not about X"],
        "label_field": "label_included",
        "notes": "n",
    }
    base.update(over)
    return base


def test_load_protocol_valid_and_invalid(tmp_path: Path):
    p = tmp_path / "X_2020.json"
    p.write_text(json.dumps(_protocol_dict(extra_field=1)), encoding="utf-8")
    proto = load_protocol(p)
    assert proto.dataset_id == "X_2020" and proto.label_field == "label_included"
    assert proto.extra == {"extra_field": 1}
    assert protocol_query_text(proto) == "Does X affect Y?"
    assert protocol_query_text(proto, "rq+criteria") == "Does X affect Y? studies about X"

    for bad in (
        _protocol_dict(label_field="nope"),
        _protocol_dict(inclusion_criteria="not a list"),
        _protocol_dict(inclusion_criteria=[""]),
        _protocol_dict(research_question=""),
        {k: v for k, v in _protocol_dict().items() if k != "exclusion_criteria"},
    ):
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ProtocolError):
            load_protocol(p)
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProtocolError):
        load_protocol(p)
    with pytest.raises(ProtocolError):
        load_protocol(tmp_path / "missing.json")


# ---------------------------------------------------------------- stage-aware criteria


def test_both_criterion_forms_load_and_stage_defaults_to_abstract(tmp_path: Path):
    """A plain string and the widened {text, stage, stage_rationale, absence} object may be
    mixed in the same list; a plain string (or an object with no ``stage``) defaults to
    "abstract", and ``Protocol.inclusion_criteria``/``.exclusion_criteria`` stay ``list[str]``
    of the texts either way."""
    p = tmp_path / "X_2020.json"
    p.write_text(
        json.dumps(
            _protocol_dict(
                inclusion_criteria=[
                    "studies about X",
                    {"text": "a full-text fact", "stage": "full_text", "stage_rationale": "why"},
                ],
                exclusion_criteria=[
                    {"text": "not about X"},
                    {"text": "exact duplicate", "absence": False},
                ],
            )
        ),
        encoding="utf-8",
    )
    proto = load_protocol(p)
    assert proto.inclusion_criteria == ["studies about X", "a full-text fact"]
    assert proto.exclusion_criteria == ["not about X", "exact duplicate"]
    assert isinstance(proto.inclusion_criteria, list)
    assert all(isinstance(x, str) for x in proto.inclusion_criteria)
    assert proto.inclusion_stages == ["abstract", "full_text"]
    assert proto.inclusion_stage_rationales == ["", "why"]
    assert proto.exclusion_stages == ["abstract", "abstract"]
    # defaults: False for inclusion, True for exclusion, unless overridden
    assert proto.inclusion_absence == [False, False]
    assert proto.exclusion_absence == [True, False]


def test_full_text_criterion_without_a_rationale_raises(tmp_path: Path):
    p = tmp_path / "X_2020.json"
    p.write_text(
        json.dumps(
            _protocol_dict(
                inclusion_criteria=[{"text": "a full-text fact", "stage": "full_text"}],
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError):
        load_protocol(p)
    # a blank stage_rationale is treated the same as a missing one
    p.write_text(
        json.dumps(
            _protocol_dict(
                inclusion_criteria=[
                    {"text": "a full-text fact", "stage": "full_text", "stage_rationale": "  "}
                ],
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError):
        load_protocol(p)


def test_unknown_stage_and_non_boolean_absence_raise(tmp_path: Path):
    p = tmp_path / "X_2020.json"
    p.write_text(
        json.dumps(_protocol_dict(inclusion_criteria=[{"text": "t", "stage": "full-text"}])),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError):
        load_protocol(p)
    p.write_text(
        json.dumps(_protocol_dict(exclusion_criteria=[{"text": "t", "absence": "yes"}])),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError):
        load_protocol(p)


def test_negation_stage_invariant_raises_when_violated(tmp_path: Path):
    """Where ``criteria_provenance.exclusion[i].location`` begins "negation of I{n}",
    ``exclusion_stages[i]`` must equal ``inclusion_stages[n-1]``, or the model could dodge
    the full-text flag by citing "I1" instead of "E1" for the same fact."""
    p = tmp_path / "X_2020.json"
    base = _protocol_dict(
        inclusion_criteria=[{"text": "uses X", "stage": "full_text", "stage_rationale": "why"}],
        exclusion_criteria=[{"text": "does not use X", "stage": "abstract"}],
    )
    base["criteria_provenance"] = {
        "inclusion": [{"id": "I1", "source": "paper", "location": "p.1", "derived": False}],
        "exclusion": [
            {
                "id": "E1",
                "source": "derived",
                "location": "negation of I1; the source states no separate exclusion list",
                "derived": True,
            }
        ],
    }
    p.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ProtocolError, match="negation of I1"):
        load_protocol(p)

    # matching stages: no error
    base["exclusion_criteria"] = [
        {"text": "does not use X", "stage": "full_text", "stage_rationale": "why"}
    ]
    p.write_text(json.dumps(base), encoding="utf-8")
    proto = load_protocol(p)
    assert proto.exclusion_stages == ["full_text"]

    # a location that does not match "negation of I{n}" is never checked
    base["criteria_provenance"]["exclusion"][0]["location"] = "the source states its own list"
    base["exclusion_criteria"] = ["does not use X"]
    p.write_text(json.dumps(base), encoding="utf-8")
    proto = load_protocol(p)
    assert proto.exclusion_stages == ["abstract"]


def test_committed_protocols_load():
    files = sorted((EVAL_ROOT / "screening" / "protocols").glob("*.json"))
    assert files, "at least one protocol JSON must be committed"
    ids = set()
    for f in files:
        proto = load_protocol(f)
        assert proto.dataset_id == f.stem
        assert proto.inclusion_criteria and proto.exclusion_criteria
        assert re.match(r"^10\.\d{4,}/", proto.source_review_doi), proto.source_review_doi
        assert "2026-" in proto.notes, f"{f.name}: notes must record an access date"
        assert proto.extra["dataset_source"]["n_records"] > 0
        ids.add(proto.dataset_id)
    assert {"Nagtegaal_2019", "van_de_Schoot_2017", "Smid_2020"} <= ids


# ---------------------------------------------------------------- primary_label


def test_new_and_completed_protocols_carry_primary_label_included():
    """The three new SYNERGY+ protocols, the completed Nagtegaal_2019, and the held-out
    `van_Dis_2020` (chosen by an abstract-coverage audit) all score against `primary_label
    = "label_included"` and carry a Crossref-shaped `source_review_doi`. This is scoped to
    the five protocols that carry the key, not every file in the directory:
    `Smid_2020.json` and `van_de_Schoot_2017.json` carry no `primary_label`, so a
    directory-wide assertion would fail on files that do not carry the key."""
    protocols_dir = EVAL_ROOT / "screening" / "protocols"
    for name in ("Fong_2021", "Taschner_2024", "Anmarkrud_2021", "Nagtegaal_2019", "van_Dis_2020"):
        proto = load_protocol(protocols_dir / f"{name}.json")
        assert proto.extra.get("primary_label") == "label_included", name
        assert re.match(r"^10\.\d{4,}/", proto.source_review_doi), (name, proto.source_review_doi)


def test_new_protocols_flag_derived_criteria():
    """Every inclusion/exclusion criterion not quoted verbatim from the source (metadata string
    or paper) must carry `derived: true` in `criteria_provenance`, and every criterion must be
    accounted for there (one provenance entry per criterion, in order)."""
    protocols_dir = EVAL_ROOT / "screening" / "protocols"
    for name in ("Fong_2021", "Taschner_2024", "Anmarkrud_2021"):
        proto = load_protocol(protocols_dir / f"{name}.json")
        prov = proto.extra["criteria_provenance"]
        assert len(prov["inclusion"]) == len(proto.inclusion_criteria), name
        assert len(prov["exclusion"]) == len(proto.exclusion_criteria), name
        for entry in prov["inclusion"] + prov["exclusion"]:
            assert isinstance(entry["derived"], bool), (name, entry)
            assert entry["source"] in ("metadata", "paper", "derived"), (name, entry)
        assert any(e["derived"] for e in prov["exclusion"]) or all(
            e["source"] == "paper" for e in prov["exclusion"]
        ), f"{name}: exclusion_criteria must either be flagged derived or sourced from the paper"


def test_anmarkrud_2021_exclusion_criteria_omit_realized_screening_counts():
    """`_build_screening_prompt`
    (backend/app/agents/relevance_screener_agent.py) renders every `exclusion_criteria` string
    verbatim to the screener, as `E{i}: {criterion}`, before it reads a single record. The five
    Anmarkrud_2021 exclusion criteria must therefore state the review's reason categories only,
    never the review's own realized abstract-screening counts (e.g. "1,672 of 2,618 records") or
    a rank claim like "the largest reason": either would disclose the review's outcome
    distribution to the model pre-screening. Those counts remain audit evidence in
    `criteria_provenance.exclusion[i].location`, which is never passed to the screener."""
    protocols_dir = EVAL_ROOT / "screening" / "protocols"
    proto = load_protocol(protocols_dir / "Anmarkrud_2021.json")
    assert len(proto.exclusion_criteria) == 5
    for criterion in proto.exclusion_criteria:
        assert not re.search(r"\d", criterion), criterion
        assert "largest reason" not in criterion, criterion
    prov = proto.extra["criteria_provenance"]["exclusion"]
    assert len(prov) == len(proto.exclusion_criteria)
    for entry in prov:
        assert re.search(r"\d", entry["location"]), entry


def test_nagtegaal_2019_i1_completion_carries_derived_from():
    """The Nagtegaal_2019 I1 completion (the Munscher/Vetter/Scheuerle 2016 taxonomy
    categories) must be marked `derived_from` with the taxonomy's own Crossref-shaped DOI, and
    the completion must not have touched any other criterion."""
    protocols_dir = EVAL_ROOT / "screening" / "protocols"
    proto = load_protocol(protocols_dir / "Nagtegaal_2019.json")
    derived_from = proto.extra.get("derived_from")
    assert derived_from is not None, "Nagtegaal_2019.json must carry a derived_from field"
    assert derived_from["criterion"] == "I1"
    assert re.match(r"^10\.\d{4,}/", derived_from["doi"]), derived_from["doi"]
    assert derived_from["doi"] == "10.1002/bdm.1897"
    for term in ("decision information", "decision structure", "decision assistance"):
        assert term in proto.inclusion_criteria[0], term
    # Unchanged text and counts for every other criterion (only I1 was completed).
    assert len(proto.inclusion_criteria) == 4
    assert len(proto.exclusion_criteria) == 6
    assert proto.inclusion_criteria[1].startswith("The nudge aims to promote evidence-based")
    assert proto.inclusion_criteria[2].startswith("The report is an experiment")
    expected_i4 = "Written in English; no constraint on year of publication."
    assert proto.inclusion_criteria[3] == expected_i4


def test_readme_names_the_protocols_with_and_without_primary_label():
    """`Smid_2020.json` and `van_de_Schoot_2017.json` carry no `primary_label` key, so
    README.md must not claim `primary_label` is present on every protocol. It must instead
    name the files that do carry the key (including the held-out `van_Dis_2020`), name the
    two that do not, and spell out the documented default
    (`protocol.extra.get("primary_label", "label_included")`), so the invariant the README
    states matches what `load_protocol` actually returns for all committed protocols."""
    protocols_dir = EVAL_ROOT / "screening" / "protocols"
    with_key, without_key = [], []
    for f in sorted(protocols_dir.glob("*.json")):
        proto = load_protocol(f)
        (with_key if "primary_label" in proto.extra else without_key).append(f.stem)
    assert set(with_key) == {
        "Fong_2021",
        "Taschner_2024",
        "Anmarkrud_2021",
        "Nagtegaal_2019",
        "van_Dis_2020",
    }
    assert set(without_key) == {"Smid_2020", "van_de_Schoot_2017"}

    readme = (protocols_dir / "README.md").read_text(encoding="utf-8")
    assert "every protocol" not in readme
    for name in with_key:
        assert f"`{name}.json`" in readme
    for name in without_key:
        assert f"`{name}.json`" in readme
    assert 'protocol.extra.get("primary_label", "label_included")' in readme

    # The two sentences that enumerate the with-primary_label files must list every file in
    # with_key, not merely mention each name somewhere in the document, which the sentences
    # could satisfy by accident if a new file were added to the directory but not to these
    # two sentences.
    flat = re.sub(r"\s+", " ", readme)
    name_re = re.compile(r"`([A-Za-z_0-9]+)\.json`")

    schema_row = re.search(r"Present as an explicit key only in(.*?);", flat)
    assert schema_row, "README schema-table wording for `primary_label` changed unexpectedly"
    assert set(name_re.findall(schema_row.group(1))) == set(with_key)

    section_sentence = re.search(r"Only(.*?carry an explicit `primary_label` key\.)", flat)
    assert section_sentence, "README 'Choosing the primary label' wording changed unexpectedly"
    assert set(name_re.findall(section_sentence.group(1))) == set(with_key)


def test_protocol_readme_table_counts_match_the_protocols():
    """The 'Committed so far' table of protocols/README.md repeats each protocol's ISSN
    resolution count; it must equal dataset_source.hydration.n_targets_with_issn."""
    readme = (EVAL_ROOT / "screening" / "protocols" / "README.md").read_text(encoding="utf-8")
    checked = 0
    for f in sorted((EVAL_ROOT / "screening" / "protocols").glob("*.json")):
        hyd = load_protocol(f).extra["dataset_source"].get("hydration") or {}
        if "n_targets_with_issn" not in hyd:
            continue
        row = next(line for line in readme.splitlines() if line.startswith(f"| `{f.name}`"))
        assert f"ISSN resolved for {hyd['n_targets_with_issn']}/{hyd['n_targets']}" in row, row
        if hyd.get("by_doi"):
            assert f"{hyd['by_doi']} by DOI" in row, row
            assert f"{hyd['by_title_exact_year'] + hyd['by_title_exact']} by" in row, row
        checked += 1
    assert checked >= 2


def test_protocol_spot_check_method_string_matches_its_evidence_file():
    """The spot-check provenance string must name the record count of the file it points at,
    or a reviewer who opens the evidence file finds a different number of records than the
    protocol promises."""
    checked = 0
    for f in sorted((EVAL_ROOT / "screening" / "protocols").glob("*.json")):
        hyd = load_protocol(f).extra["dataset_source"].get("hydration") or {}
        spot_check = hyd.get("spot_check")
        if not spot_check:
            continue
        assert f"{spot_check['n']}-record" in spot_check["method"], spot_check["method"]
        evidence = json.loads(
            (EVAL_ROOT / "screening" / spot_check["evidence_file"]).read_text(encoding="utf-8")
        )
        assert len(evidence["records"]) == spot_check["n"]
        checked += 1
    assert checked >= 1


def test_protocol_notes_do_not_use_the_retired_ceiling_convention():
    """The retired E1-d ceiling convention (upper bound = share of *resolved*
    records) must not survive in any protocol's ``notes`` string: a reviewer who reads the
    protocol JSON directly, rather than summarize.py's output, would take it as the
    definition. Any ``lower-upper`` interval quoted in a protocol's notes must match the
    lower/upper bound ``gate_table_row`` computes from the committed gate file, not the
    resolved-only point estimate."""
    from conftest import load_script_module

    sm = load_script_module("screening", "summarize")
    interval_re = re.compile(r"(0\.\d{3})-(0\.\d{3})")
    # The protocol notes' embedded intervals were computed on the retired title/abstract
    # WoS-gate basis, which no shipped results directory carries any more (results/v3/ only
    # computes the final-inclusion basis); a redacted fixture keeps this consistency check
    # working without a whole superseded results directory.
    results_dir = EVAL_ROOT / "tests" / "fixtures" / "screening_v1_wos_gate"
    checked = 0
    for f in sorted((EVAL_ROOT / "screening" / "protocols").glob("*.json")):
        notes = load_protocol(f).notes
        assert "upper bound = in WoS /" not in notes, f"{f.name}: {notes}"
        match = interval_re.search(notes)
        gate_file = results_dir / f"{f.stem}_wos_gate.json"
        if not match or not gate_file.exists():
            continue
        gate = sm.gate_table_row(json.loads(gate_file.read_text(encoding="utf-8")), None)
        lower, upper = round(float(match.group(1)), 3), round(float(match.group(2)), 3)
        assert lower == round(gate["share_lower_bound"], 3), f.name
        assert upper == round(gate["share_upper_bound"], 3), f.name
        checked += 1
    assert checked >= 2


def test_readme_states_wilson_interval_for_spot_check():
    """The 20/20 hydration spot-check is a sample, not a census; README.md must carry the
    Wilson-interval caveat next to the figure so a reviewer does not read '20/20 correct' as
    proof the hydration is clean."""
    readme = (EVAL_ROOT / "README.md").read_text(encoding="utf-8")
    lines = readme.splitlines()
    spot_check_line = next(i for i, line in enumerate(lines) if "20/20 and 20/20 correct" in line)
    window = "\n".join(lines[spot_check_line : spot_check_line + 4])
    assert "Wilson" in window and "0.839" in window


def test_readme_label_noise_paragraph_qualifies_recall_within_a_dataset():
    """README.md's over-inclusive-criterion sentence must carry the same 'only within a
    dataset ... not as a ranking of the three datasets' qualification as
    screening.summarize.LABEL_NOISE_NOTE, so a reader of README.md alone (not just the
    module constant) sees that van_de_Schoot_2017's recall is not a cross-dataset ranking."""
    readme = (EVAL_ROOT / "README.md").read_text(encoding="utf-8")
    paras = readme.split("\n\n")
    para = next(p for p in paras if "over-inclusive" in p)
    assert "within a dataset" in para
    assert "not as a ranking of the three datasets" in para


def test_readme_does_not_describe_fn_categories_as_hand_filled():
    """The false-negative categorisation is 903 rule-assigned rows plus 20 overrides made by
    a Claude Code agent, every low-confidence match read and confirmed or overridden by that
    agent, a seed-20260905 10 % audit and an independent blind check -- not a blanket
    'hand-filled'/'manual coding' label that hides the rule engine and implies every row was
    independently transcribed by a person."""
    readme = (EVAL_ROOT / "README.md").read_text(encoding="utf-8")
    assert "hand-filled" not in readme
    assert "categorise_fn.py" in readme and "fn_overrides.json" in readme
    assert "kappa 1.0" in readme


def test_readme_discloses_fn_coding_agent_provenance():
    """The 20 overrides, the 372 low-confidence-or-unmatched rows, the 55-row seed-20260905
    audit and the 73-row second read were all done by an automated language-model pass, not
    the authors, and the README must say so plainly (not the old impersonal 'by hand'
    wording, nor the old 'hand overrides' phrasing) and must state that the authors did not
    check these category assignments."""
    readme = (EVAL_ROOT / "README.md").read_text(encoding="utf-8")
    flat = re.sub(r"\s+", " ", readme)
    assert "by hand" not in readme
    assert "hand overrides" not in readme
    assert "automated language-model pass" in flat
    assert "372" in readme
    assert "authors did not check" in flat


def test_fn_taxonomy_discloses_fn_coding_agent_provenance():
    """fn_taxonomy.md's own indicative-distribution note must carry the same disclosure as
    README.md, not the old 'hand overrides' / bare 'independent blind check' wording that
    named no agent."""
    taxonomy = (
        EVAL_ROOT / "screening" / "results" / "v3" / "fn_categorisation" / "fn_taxonomy.md"
    ).read_text(encoding="utf-8")
    assert "hand overrides" not in taxonomy
    assert "Claude Code" in taxonomy
    assert "372" in taxonomy
    assert "re-read" in taxonomy


def test_screening_summary_discloses_fn_coding_agent_provenance():
    """The false-negative-categories caption rendered above the per-dataset category tables
    must carry the same Claude Code agent-provenance disclosure as README.md, not a bare
    'kappa 1.0' that hides who read the rows, and not the old 'hand overrides' phrasing."""
    sm = load_script_module("screening", "summarize")
    caption = sm._fn_categories_caption(
        EVAL_ROOT / "screening" / "results" / "v3"
    )
    assert "by hand" not in caption
    assert "hand overrides" not in caption
    assert "Claude Code" in caption
    assert "372" in caption
    assert "re-read by the" in caption and "authors" in caption


def test_fn_blind_check_recomputes_perfect_agreement_from_the_deposit():
    """<results_dir>/fn_categorisation/fn_blind_check/fn_blind_check.py must recompute the
    blind check's agreement and Cohen's kappa from its own shipped sample/labels and the
    committed *_fn_categories.csv files, not print a number that nothing on disk backs up."""
    sm = load_script_module("screening", "summarize")
    fbc = sm._fn_blind_check_module(EVAL_ROOT / "screening" / "results" / "v3")
    result = fbc.compute()
    assert result.n == 73
    assert result.n_agree == 73
    assert result.kappa == 1.0
    assert result.mismatches == ()


def test_annotation_readme_discloses_model_provenance():
    """The annotation package's own README must not upgrade the committed labels to human or
    expert annotations: whenever the completed sheet exists it must name Claude and point to
    PROVENANCE.md, and must not use the human/human-annotated vocabulary PROVENANCE.md bans."""
    annotation_dir = EVAL_ROOT / "claims" / "annotation"
    if not (annotation_dir / "hss_annotation_completed_v2.csv").exists():
        pytest.skip("round-2 annotation package not present")
    readme = (annotation_dir / "README.md").read_text(encoding="utf-8")
    assert "Claude" in readme and "PROVENANCE.md" in readme
    assert "human labels" not in readme and "human-annotated" not in readme


def test_readme_references_the_claims_annotation_package():
    """evaluation/README.md must point a reviewer at claims/annotation/ and must not describe
    its labels as human or expert, matching PROVENANCE.md's explicit ban on that vocabulary."""
    annotation_dir = EVAL_ROOT / "claims" / "annotation"
    if not (annotation_dir / "hss_annotation_adjudicated_v2.csv").exists():
        pytest.skip("round-2 annotation package not present")
    readme = (EVAL_ROOT / "README.md").read_text(encoding="utf-8")
    assert "claims/annotation" in readme
    assert "not expert human annotations" in re.sub(r"\s+", " ", readme)
    for para in readme.split("\n\n"):
        if "Fleiss" in para:
            assert "model" in para or "not expert human" in para, para


def test_markdown_table_formatting():
    md = markdown_table(["a", "b"], [[1, 0.12345], [None, True]])
    lines = md.splitlines()
    assert lines[0] == "| a | b |" and lines[1] == "|---|---|"
    assert lines[2] == "| 1 | 0.123 |" and lines[3] == "| - | yes |"


def test_deepseek_price_table_and_tiering():
    from common import DEEPSEEK_PRICES, call_cost, is_peak_utc, resolve_price_model

    assert DEEPSEEK_PRICES["source_url"] == "https://api-docs.deepseek.com/quick_start/pricing"
    assert DEEPSEEK_PRICES["accessed"] == "2026-09-02"
    assert resolve_price_model("deepseek-chat") == "deepseek-v4-flash"
    assert resolve_price_model("deepseek-v4-pro") == "deepseek-v4-pro"
    # The API self-reports the deepseek-chat alias's deployment as "deepseek-flash", still
    # the same price row as "deepseek-v4-flash", not a new tier.
    assert resolve_price_model("deepseek-flash") == "deepseek-v4-flash"
    assert resolve_price_model("gpt-x") is None and resolve_price_model(None) is None

    assert is_peak_utc("2026-09-02T02:30:00+00:00") is True  # Wednesday, 01-04 window
    assert is_peak_utc("2026-09-02T05:00:00+00:00") is False
    assert is_peak_utc("2026-09-05T07:00:00+00:00") is False  # Saturday
    assert is_peak_utc("2026-09-02T09:59:59+00:00") is True
    assert is_peak_utc("2026-09-02T10:00:00+00:00") is False
    assert is_peak_utc("2026-09-02T04:30:00+02:00") is True  # 02:30 UTC

    off = call_cost("deepseek-chat", "2026-09-02T12:00:00+00:00", 1_000_000, 1_000_000)
    assert off == pytest.approx(0.22 + 0.66)
    cached = call_cost(
        "deepseek-chat",
        "2026-09-02T12:00:00+00:00",
        1_000_000,
        1_000_000,
        cache_read_tokens=500_000,
    )
    assert cached == pytest.approx(0.5 * 0.22 + 0.5 * 0.007 + 0.66)
    peak = call_cost("deepseek-chat", "2026-09-02T02:00:00+00:00", 1_000_000, 1_000_000)
    assert peak == pytest.approx(0.44 + 1.32)
    assert call_cost("unknown-model", "2026-09-02T12:00:00+00:00", 10, 10) is None
    assert call_cost("deepseek-chat", "2026-09-02T12:00:00+00:00", None, 10) is None
    assert call_cost("deepseek-chat", None, 10, 10) is None


def test_price_record_defaults_from_table():
    rec = price_record("deepseek-chat", None, None)
    assert rec["model"] == "deepseek-chat" and rec["price_model"] == "deepseek-v4-flash"
    assert rec["tiers"]["output"]["peak"] == 1.32
    assert rec["tiers"]["input_cache_hit"]["offpeak"] == 0.007
    assert rec["source_url"] == "https://api-docs.deepseek.com/quick_start/pricing"
    assert rec["accessed"] == "2026-09-02" and rec["currency"] == "USD"
    assert rec["input_per_mtok"] is None and rec["output_per_mtok"] is None
    flat = price_record("deepseek-chat", 0.5, 1.0, "https://example.org/p", "2026-01-01")
    assert flat["input_per_mtok"] == 0.5 and flat["source_url"] == "https://example.org/p"
    assert flat["accessed"] == "2026-01-01"
    unknown = price_record("other-model", None, None)
    assert unknown["price_model"] is None and unknown["tiers"] is None


# ---------------------------------------------------------------- additional cases


def test_prune_failed_rows_rewrites_atomically(tmp_path: Path):
    from common import prune_failed_rows

    out = tmp_path / "run.jsonl"
    append_jsonl(out, [{"id": 1, "predicted": 1}, {"id": 2, "predicted": None}, {"id": 3}])
    n = prune_failed_rows(out, lambda r: r.get("predicted") is None)
    assert n == 2
    assert [r["id"] for r in read_jsonl(out)] == [1]
    assert not out.with_suffix(".jsonl.tmp").exists()
    assert prune_failed_rows(tmp_path / "missing.jsonl", lambda r: True) == 0
    assert prune_failed_rows(out, lambda r: False) == 0


def test_resolve_path_args_makes_every_path_absolute(tmp_path: Path, monkeypatch):
    import argparse

    from common import resolve_path_args

    monkeypatch.chdir(tmp_path)
    ns = argparse.Namespace(
        results_dir=Path("out"), data_dir=None, limit=3, name="x",
        exclude_sources=[Path("a.json"), Path("b.json")], tags=["a", "b"],
    )
    resolve_path_args(ns)
    assert ns.results_dir.is_absolute() and ns.results_dir == (tmp_path / "out").resolve()
    assert ns.data_dir is None and ns.limit == 3 and ns.name == "x"
    assert all(p.is_absolute() for p in ns.exclude_sources)
    assert ns.exclude_sources == [(tmp_path / "a.json").resolve(), (tmp_path / "b.json").resolve()]
    assert ns.tags == ["a", "b"]  # a plain str list is left untouched


def test_add_backend_to_path_changes_cwd_to_backend_root(tmp_path: Path, monkeypatch):
    """The backend Settings read ``.env`` relative to the cwd, so imports must run from there."""
    from common import add_backend_to_path

    root = tmp_path / "backend"
    (root / "app").mkdir(parents=True)
    (root / "app" / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "config.py").write_text(
        "from pathlib import Path\n"
        f"assert Path.cwd().resolve() == Path({str(root.resolve())!r}).resolve(), Path.cwd()\n"
        "settings = object()\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    got = add_backend_to_path(root)
    assert got == root.resolve() and Path.cwd().resolve() == root.resolve()
    import importlib

    importlib.import_module("app.config")  # would raise AssertionError before the fix
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(root.resolve())])


# ---------------------------------------------------------------- additional cases


def test_readme_discloses_when_no_paid_screening_or_claims_run_exists():
    """README.md must not let a dry-run/keyword-stand-in number pass as a measured screener
    or verifier performance figure while no paid E1/E2 run is committed: whenever none of
    ``screening/results/``'s versioned run directories (``v1/``, ``v2/``, ...; moved aside
    from the bare ``results/`` root when screener v2 shipped) holds a ``*_run[AB].jsonl``
    the disclaimer sentence must be present, and any lone precision-looking figure
    (e.g. ``0.068``) may only appear on a line that also says ``dry-run``."""
    readme = (EVAL_ROOT / "README.md").read_text(encoding="utf-8")
    screening_results = EVAL_ROOT / "screening" / "results"
    has_real_run = (
        any(screening_results.glob("v*/*_run[AB].jsonl")) if screening_results.exists() else False
    )
    if not has_real_run:
        assert "No paid run has been made in this tree yet." in readme
    for line in readme.splitlines():
        if "0.068" in line:
            assert "dry-run" in line, line


def test_requirements_txt_pins_the_tools_the_readme_runs_without_an_interpreter_choice():
    """README's ``tests/`` row says ``either`` interpreter may run pytest, and the lint
    command (``ruff check``) is documented without naming an interpreter either; a reviewer
    who only builds ``evaluation/.venv`` per the Setup section must still be able to run both,
    so both must be pinned in ``evaluation/requirements.txt``."""
    req = (EVAL_ROOT / "requirements.txt").read_text(encoding="utf-8")
    readme = (EVAL_ROOT / "README.md").read_text(encoding="utf-8")
    tests_row = next(line for line in readme.splitlines() if line.startswith("| `tests/` |"))
    assert "either" in tests_row
    assert "ruff check evaluation" in readme
    assert re.search(r"^pytest==", req, re.MULTILINE), "pytest must be pinned in requirements.txt"
    assert re.search(r"^ruff==", req, re.MULTILINE), "ruff must be pinned in requirements.txt"


def test_stopwatch_reports_elapsed_time_inside_and_after_the_block():
    import time

    from common import Stopwatch

    with Stopwatch() as sw:
        time.sleep(0.05)
        inside = sw.seconds
    assert inside >= 0.05, "seconds must be live while the block is running"
    after = sw.seconds
    assert after >= inside
    time.sleep(0.02)
    assert sw.seconds == after, "seconds must freeze once the block has exited"
