"""claims/build_hss_set (pure construction rules) and claims/run_hss (item loading).

No network, no LLM, no backend import: the acquisition step is never exercised here.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from conftest import skip_unless_cache_dir

import build_hss_set as bh
import run_hss as rh
from common import append_jsonl, write_json

DOIS = ["10.1000/paper-a", "10.1000/paper-b", "10.1000/paper-c"]


def sentence(p: int, i: int) -> str:
    return (
        f"The results showed that learners in group {p}{i} achieved significantly higher "
        f"scores than the control group after {i + 3} weeks of instruction."
    )


def make_cache(cache_dir: Path, n_sentences: int = 14) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for p, doi in enumerate(DOIS):
        body = " ".join(sentence(p, i) for i in range(n_sentences))
        payload = {
            "doi": doi,
            "title": f"Paper {p}",
            "authors": [f"Author {p}"],
            "source_url": f"https://example.org/{p}.pdf",
            "fetched_at": "2026-09-02T00:00:00+00:00",
            "n_chars": len(body) * 3,
            "chunks": [
                # "Paper N. Author N." echoes the title/authors above so the default-on cache
                # identity check keeps this synthetic source; each fragment is short enough
                # that it is never itself mined as a candidate sentence.
                {"section": "introduction", "text": f"Paper {p}. Author {p}. Short intro. "
                 "Nothing of note here today."},
                {"section": "results", "text": body},
                {"section": "conclusion", "text": "We conclude that this happened in the end."},
            ],
        }
        write_json(cache_dir / f"{bh.doi_slug(doi)}.json", payload)


# ---------------------------------------------------------------- pure helpers


def test_doi_slug():
    assert bh.doi_slug("10.14746/ssllt.2020.10.1.2") == "10-14746_ssllt-2020-10-1-2"
    assert bh.doi_slug("10.1186/s40468-021-00123-x") == "10-1186_s40468-021-00123-x"


def test_paraphrase_substitutes_at_least_one_token():
    text, subs = bh.paraphrase("The results showed that participants found the task easy.")
    assert text != "The results showed that participants found the task easy."
    assert len(subs) >= 1 and all(len(s) == 2 for s in subs)
    assert "findings" in text and "demonstrated" in text and "participants" in text
    assert text[0].isupper()
    assert bh.paraphrase("Nothing here matches any rule at all.") == (
        "Nothing here matches any rule at all.", []
    )
    # a sentence made only of technical terms is left alone (no paraphrase possible)
    assert bh.paraphrase("The gain was statistically significant.")[1] == []


def test_paraphrase_does_not_substitute_the_verb_sense_of_results(tmp_path: Path):
    """'results' -> 'findings' is meant for the noun sense ("the results showed ...") but a
    whole-word substitution also matches the verb sense ("which results from the ICC level"
    -> "which findings from the ICC level"), producing an ungrammatical, meaning-altering
    paraphrase that a real reader (and the verifier) correctly flags as not a faithful
    restatement. This is exactly the defect the 'verified'-by-construction paraphrase rule
    promises never to introduce."""
    text, subs = bh.paraphrase("Which results from the ICC level, they communicate better.")
    assert "findings" not in text and ("results", "findings") not in subs
    text2, subs2 = bh.paraphrase("This results in higher scores overall for every learner.")
    assert "findings" not in text2 and ("results", "findings") not in subs2
    # the noun sense is unaffected
    text3, subs3 = bh.paraphrase("The results showed significant gains for the treatment group.")
    assert "findings" in text3 and ("results", "findings") in subs3


def test_alter_direction_number_quantifier():
    out = bh.alter("Scores were significantly higher in the treatment group.")
    assert out is not None
    text, alteration = out
    assert "lower" in text and alteration.startswith("direction_flip")
    text, alteration = bh.alter("The study enrolled 24 students over 6 weeks.")
    assert text.startswith("The study enrolled 48 students") and alteration.startswith("numeric")
    text, alteration = bh.alter("About 60% of the learners passed the test.")
    assert text.startswith("About 99%") and alteration == "numeric:60->99"
    text, alteration = bh.alter("Most learners preferred the online format.")
    assert text.startswith("Few learners") and alteration.startswith("quantifier_flip")
    assert bh.alter("This sentence offers no handle for any rule.") is None
    # years and p-values are not numeric handles
    assert bh.alter("Data were collected in 2019 (p < 0.05) for the cohort.") is None
    # percentage cap
    text, _ = bh.alter("Exactly 75% of the learners passed the final examination.")
    assert "99%" in text


def test_alter_direction_never_negates_significantly_postmodifying_a_verb():
    # "differed significantly" -> "differed not significantly" is ungrammatical; the
    # idiomatic negation is "did not differ significantly", which this sentence offers no
    # handle for, so no direction flip should fire (this is the committed hss-altered-10
    # defect: DIRECTION_FLIPS applied "significantly"->"not significantly" unconditionally).
    sentence = (
        "However, according to the results of the ANCOVAs, it was only in the area of "
        "intrinsic motivation that the improvements of the two groups differed significantly."
    )
    assert bh.alter(sentence) is None
    # premodifying an adjective/participle is still a valid, checkable direction claim
    text, alteration = bh.alter("The gains were significantly different across conditions.")
    assert "not significantly different" in text and alteration.startswith("direction_flip")


def test_alter_direction_drops_the_article_when_negating_significant():
    # "showed a significant difference" -> "showed a no significant difference" is
    # ungrammatical (double determiner); the article must be dropped: "showed no significant
    # difference" (this is the hss-altered-10 replacement candidate surfaced by fixing the
    # "significantly" postmodifier defect above).
    text, alteration = bh.alter(
        "The performance of the group showed a significant difference between conditions."
    )
    assert "showed no significant difference" in text
    assert "a no significant" not in text
    assert alteration == "direction_flip:significant->no significant"
    # no article to drop: unaffected
    text, _ = bh.alter("NESTs need to expend significant cognitive labor on their tasks.")
    assert "expend no significant cognitive labor" in text


def test_assign_rules_quotas_balance_and_determinism(tmp_path: Path):
    make_cache(tmp_path / "cache")
    cands = {doi: bh.candidates_for_paper(bh.load_cache(tmp_path / "cache")[doi]["chunks"])
             for doi in DOIS}
    items = bh.assign_rules(cands, 20260902)
    assert len(items) == 30
    counts = {r: sum(1 for it in items if it["rule"] == r) for r in bh.RULE_ORDER}
    assert counts == {"verbatim": 5, "paraphrase": 5, "altered": 10, "over_specified": 0,
                      "wrong_paper": 5, "no_full_text": 5}
    originals = [it["original_sentence"] for it in items]
    assert len(set(originals)) == 30
    for it in items:
        assert it["item_id"].startswith(f"hss-{it['rule']}-")
        assert it["expected"] == bh.EXPECTED[it["rule"]]
        if it["rule"] == "wrong_paper":
            assert it["chunk_doi"] != it["source_doi"] and it["chunk_doi"] in DOIS
        elif it["rule"] == "no_full_text":
            assert it["chunk_doi"] is None
        else:
            assert it["chunk_doi"] == it["source_doi"]
        if it["rule"] in ("verbatim", "wrong_paper", "no_full_text"):
            assert it["claim"] == it["original_sentence"] and it["alteration"] is None
        if it["rule"] == "altered":
            assert it["claim"] != it["original_sentence"] and it["alteration"]
        if it["rule"] == "paraphrase":
            assert it["claim"] != it["original_sentence"] and it["alteration"]
    per_paper = {doi: sum(1 for it in items if it["source_doi"] == doi) for doi in DOIS}
    assert max(per_paper.values()) - min(per_paper.values()) <= 1
    ids = [it["item_id"] for it in items]
    assert ids == sorted(ids, key=lambda s: (bh.RULE_ORDER.index(s.split("-")[1]), s))
    assert bh.assign_rules(cands, 20260902) == items
    assert bh.assign_rules(cands, 7)[0]["source_doi"] != items[0]["source_doi"]


def test_assign_rules_fails_loudly_when_short(tmp_path: Path):
    make_cache(tmp_path / "cache", n_sentences=3)
    cands = {doi: bh.candidates_for_paper(bh.load_cache(tmp_path / "cache")[doi]["chunks"])
             for doi in DOIS}
    with pytest.raises(ValueError):
        bh.assign_rules(cands, 1)


def test_candidates_prefer_result_sections_and_keep_chunk_index(tmp_path: Path):
    make_cache(tmp_path / "cache")
    chunks = bh.load_cache(tmp_path / "cache")[DOIS[0]]["chunks"]
    cands = bh.candidates_for_paper(chunks)
    assert cands and all(c["chunk_index"] == 1 for c in cands[:14])
    assert all(set(c) >= {"sentence", "chunk_index", "section"} for c in cands)
    plain = bh.candidates_for_paper([{"section": "chunk_0", "text": chunks[1]["text"]}])
    assert len(plain) == 14


def test_build_items_from_cache_attaches_meta(tmp_path: Path):
    make_cache(tmp_path / "cache")
    items, info = bh.build_items_from_cache(tmp_path / "cache", seed=20260902)
    assert len(items) == 30 and info["seed"] == 20260902
    assert info["counts"] == {"verbatim": 5, "paraphrase": 5, "altered": 10,
                              "over_specified": 0, "wrong_paper": 5, "no_full_text": 5}
    assert sorted(info["sources"]) == sorted(DOIS)
    assert info["substitutions"] and info["alterations"]
    for it in items:
        assert it["built_at"] and it["title"].startswith("Paper ") and it["authors"]
        if it["rule"] == "wrong_paper":
            p = DOIS.index(it["chunk_doi"])
            assert it["title"] == f"Paper {p}"


def test_load_excluded_dois_normalises_and_unions_multiple_files(tmp_path: Path):
    p1 = tmp_path / "src1.json"
    p2 = tmp_path / "src2.json"
    write_json(p1, [{"doi": "10.1000/PAPER-A"}, {"doi": "https://doi.org/10.1000/paper-b"}])
    write_json(p2, [{"doi": "10.1000/paper-c"}])
    assert bh.load_excluded_dois([p1, p2]) == {
        "10.1000/paper-a", "10.1000/paper-b", "10.1000/paper-c",
    }
    assert bh.load_excluded_dois([]) == set()
    assert bh.load_excluded_dois([tmp_path / "missing.json"]) == set()


def test_build_hss_set_cli_altered_expected_strict_vs_legacy_default(tmp_path: Path):
    make_cache(tmp_path / "cache")
    out_strict = tmp_path / "hss_dev_claims.jsonl"
    rc = bh.main([
        "--skip-fetch", "--cache-dir", str(tmp_path / "cache"), "--out", str(out_strict),
        "--seed", "20260902", "--sources", str(tmp_path / "hss_dev_sources.json"),
        "--altered-expected", "strict",
    ])
    assert rc == 0
    rows = [json.loads(line) for line in out_strict.read_text(encoding="utf-8").splitlines()]
    altered = [r for r in rows if r["rule"] == "altered"]
    assert len(altered) == 10 and all(r["expected"] == ["unsupported"] for r in altered)
    build = json.loads(
        (tmp_path / "hss_dev_claims.build.json").read_text(encoding="utf-8")
    )
    assert build["expected"]["altered"] == ["unsupported"]
    assert build["altered_expected_mode"] == "strict"
    assert build["n_sources"] == len(DOIS)

    # default ("legacy") keeps the disjunctive expected the frozen test set was built with
    out_legacy = tmp_path / "hss_claims_legacy.jsonl"
    rc2 = bh.main([
        "--skip-fetch", "--cache-dir", str(tmp_path / "cache"), "--out", str(out_legacy),
        "--seed", "20260902", "--sources", str(tmp_path / "hss_sources_legacy.json"),
    ])
    assert rc2 == 0
    rows2 = [json.loads(line) for line in out_legacy.read_text(encoding="utf-8").splitlines()]
    altered2 = [r for r in rows2 if r["rule"] == "altered"]
    assert all(r["expected"] == ["unsupported", "needs_nuance"] for r in altered2)
    build2 = json.loads(
        (tmp_path / "hss_claims_legacy.build.json").read_text(encoding="utf-8")
    )
    assert build2["altered_expected_mode"] == "legacy"
    # EXPECTED itself (the frozen test set's module-level default) is never mutated
    assert bh.EXPECTED["altered"] == ["unsupported", "needs_nuance"]


def test_build_hss_set_cli_exclude_sources_disjointness_assertion(tmp_path: Path):
    make_cache(tmp_path / "cache")
    sources_path = tmp_path / "hss_dev_sources.json"
    write_json(sources_path, [{"doi": d} for d in DOIS])
    exclude_path = tmp_path / "hss_sources.json"
    write_json(exclude_path, [{"doi": DOIS[0]}])
    out = tmp_path / "hss_dev_claims.jsonl"
    with pytest.raises(SystemExit):
        bh.main([
            "--skip-fetch", "--cache-dir", str(tmp_path / "cache"), "--out", str(out),
            "--seed", "20260902", "--sources", str(sources_path),
            "--exclude-sources", str(exclude_path),
        ])


def test_build_hss_set_cli_exclude_sources_flag_is_repeatable(tmp_path: Path):
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    write_json(p1, [])
    write_json(p2, [])
    args = bh.parse_args(["--exclude-sources", str(p1), "--exclude-sources", str(p2)])
    assert args.exclude_sources == [p1.resolve(), p2.resolve()]
    assert bh.parse_args([]).exclude_sources == []


def test_compute_original_sentence_overlap_two_file_fixture(tmp_path: Path):
    items = [
        {"original_sentence": "Shared sentence here."},
        {"original_sentence": "Only in this build."},
    ]
    f1 = tmp_path / "f1.jsonl"
    append_jsonl(f1, [{"original_sentence": "Shared sentence here."}])
    f2 = tmp_path / "f2.jsonl"
    append_jsonl(f2, [{"original_sentence": "A different sentence entirely."}])
    result = bh.compute_original_sentence_overlap(items, [f1, f2])
    assert result == [
        {"file": "f1.jsonl", "n_shared": 1, "shared": ["Shared sentence here."]},
        {"file": "f2.jsonl", "n_shared": 0, "shared": []},
    ]


def test_compute_original_sentence_overlap_missing_file_is_zero_not_an_error(tmp_path: Path):
    items = [{"original_sentence": "Anything."}]
    result = bh.compute_original_sentence_overlap(items, [tmp_path / "missing.jsonl"])
    assert result == [{"file": "missing.jsonl", "n_shared": 0, "shared": []}]


def test_build_hss_set_cli_overlap_with_flag_is_repeatable(tmp_path: Path):
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text("", encoding="utf-8")
    p2.write_text("", encoding="utf-8")
    args = bh.parse_args(["--overlap-with", str(p1), "--overlap-with", str(p2)])
    assert args.overlap_with == [p1.resolve(), p2.resolve()]
    assert bh.parse_args([]).overlap_with == []


def test_build_hss_set_cli_overlap_with_records_shared_original_sentences(tmp_path: Path):
    make_cache(tmp_path / "cache")
    out = tmp_path / "hss_claims.jsonl"
    rc = bh.main([
        "--skip-fetch", "--cache-dir", str(tmp_path / "cache"), "--out", str(out),
        "--seed", "20260902", "--sources", str(tmp_path / "hss_sources.json"),
    ])
    assert rc == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    shared_sentence = rows[0]["original_sentence"]

    compare_path = tmp_path / "compare.jsonl"
    append_jsonl(compare_path, [
        {"original_sentence": shared_sentence},
        {"original_sentence": "unrelated sentence never in this build"},
    ])

    out2 = tmp_path / "hss_claims2.jsonl"
    rc2 = bh.main([
        "--skip-fetch", "--cache-dir", str(tmp_path / "cache"), "--out", str(out2),
        "--seed", "20260902", "--sources", str(tmp_path / "hss_sources.json"),
        "--overlap-with", str(compare_path),
    ])
    assert rc2 == 0
    build2 = json.loads((tmp_path / "hss_claims2.build.json").read_text(encoding="utf-8"))
    overlap = build2["original_sentence_overlap"]
    assert len(overlap) == 1
    assert overlap[0]["file"] == "compare.jsonl"
    assert overlap[0]["n_shared"] == 1
    assert overlap[0]["shared"] == [shared_sentence]


def test_build_hss_set_cli_no_overlap_with_flag_stays_byte_identical(tmp_path: Path):
    """Default (no --overlap-with) never adds original_sentence_overlap, so every existing
    build (frozen hss_claims.build.json / hss_dev_claims.build.json) stays byte identical."""
    make_cache(tmp_path / "cache")
    out = tmp_path / "hss_claims.jsonl"
    rc = bh.main([
        "--skip-fetch", "--cache-dir", str(tmp_path / "cache"), "--out", str(out),
        "--seed", "20260902", "--sources", str(tmp_path / "hss_sources.json"),
    ])
    assert rc == 0
    build = json.loads((tmp_path / "hss_claims.build.json").read_text(encoding="utf-8"))
    assert "original_sentence_overlap" not in build


def test_build_hss_set_cli_skip_fetch(tmp_path: Path):
    with pytest.raises(SystemExit) as exc:
        bh.main(["--help"])
    assert exc.value.code == 0
    make_cache(tmp_path / "cache")
    out = tmp_path / "hss_claims.jsonl"
    rc = bh.main(["--skip-fetch", "--cache-dir", str(tmp_path / "cache"), "--out", str(out),
                  "--seed", "20260902", "--sources", str(tmp_path / "hss_sources.json")])
    assert rc == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 30
    build = json.loads((tmp_path / "hss_claims.build.json").read_text(encoding="utf-8"))
    assert build["counts"]["altered"] == 10


# ---------------------------------------------------------------- run_hss


def test_run_hss_items_and_correct(tmp_path: Path):
    make_cache(tmp_path / "cache")
    items, _ = bh.build_items_from_cache(tmp_path / "cache", seed=20260902)
    claims_path = tmp_path / "hss_claims.jsonl"
    append_jsonl(claims_path, items)
    loaded = rh.load_items(claims_path, tmp_path / "cache")
    assert len(loaded) == 30
    cache = bh.load_cache(tmp_path / "cache")
    for it in loaded:
        assert set(it) >= {"item_id", "rule", "expected", "claim", "title", "authors", "chunks"}
        if it["rule"] == "no_full_text":
            assert it["chunks"] == []
        else:
            assert it["chunks"] == [c["text"] for c in cache[it["chunk_doi"]]["chunks"]]
            assert len(it["chunks"]) == 3
    row = {"expected": ["unsupported", "needs_nuance"], "predicted_status": "needs_nuance"}
    rh.mark_correct(row)
    assert row["correct"] is True
    row = {"expected": ["verified"], "predicted_status": "unsupported"}
    rh.mark_correct(row)
    assert row["correct"] is False


def test_run_hss_cli_help_and_dry_run(tmp_path: Path):
    with pytest.raises(SystemExit) as exc:
        rh.main(["--help"])
    assert exc.value.code == 0
    make_cache(tmp_path / "cache")
    items, _ = bh.build_items_from_cache(tmp_path / "cache", seed=20260902)
    claims_path = tmp_path / "hss_claims.jsonl"
    append_jsonl(claims_path, items)
    results = tmp_path / "results"
    rc = rh.main(["--run", "B", "--dry-run", "--claims", str(claims_path), "--cache-dir",
                  str(tmp_path / "cache"), "--results-dir", str(results)])
    assert rc == 0
    rows = [json.loads(x) for x in (results / "hss_runB.jsonl").read_text("utf-8").splitlines()]
    assert len(rows) == 30 and all("correct" in r and "rule" in r and "expected" in r for r in rows)
    nft = [r for r in rows if r["rule"] == "no_full_text"]
    assert all(r["predicted_status"] == "no_full_text" and r["deterministic"] for r in nft)
    assert all(r["correct"] for r in nft)
    assert all(r["correct"] for r in rows if r["rule"] == "verbatim")
    meta = json.loads((results / "hss_runB.meta.json").read_text("utf-8"))
    assert meta["n_deterministic"] == 5 and meta["name"] == "hss"


def test_run_hss_item_keys_carry_alteration(tmp_path: Path):
    """run_hss.ITEM_KEYS gains ``alteration``, which it drops today, so the field
    round-trips into the run row and summarize._operator_of has something to read."""
    assert "alteration" in rh.ITEM_KEYS
    make_cache(tmp_path / "cache")
    items, _ = bh.build_items_from_cache(tmp_path / "cache", seed=20260902)
    claims_path = tmp_path / "hss_claims.jsonl"
    append_jsonl(claims_path, items)
    loaded = rh.load_items(claims_path, tmp_path / "cache")
    altered = [it for it in loaded if it["rule"] == "altered"]
    assert altered and all(it.get("alteration") for it in altered)
    verbatim = [it for it in loaded if it["rule"] == "verbatim"]
    assert verbatim and all(it.get("alteration") is None for it in verbatim)


def test_run_hss_cli_name_flag_writes_prefixed_result_files(tmp_path: Path):
    """brief section 4 (band 2): run_hss.py hardcodes name="hss" today, so two dev sets in one
    directory would both resolve to hss_run<X>.jsonl; --name hss-band2 must write
    hss-band2_run<X>.jsonl instead."""
    make_cache(tmp_path / "cache")
    items, _ = bh.build_items_from_cache(tmp_path / "cache", seed=20260902)
    claims_path = tmp_path / "hss_claims.jsonl"
    append_jsonl(claims_path, items)
    results = tmp_path / "results"
    rc = rh.main(["--run", "A", "--dry-run", "--claims", str(claims_path), "--cache-dir",
                  str(tmp_path / "cache"), "--results-dir", str(results), "--name", "hss-band2"])
    assert rc == 0
    assert (results / "hss-band2_runA.jsonl").exists()
    assert not (results / "hss_runA.jsonl").exists()
    meta = json.loads((results / "hss-band2_runA.meta.json").read_text("utf-8"))
    assert meta["name"] == "hss-band2"
    args = rh.parse_args(["--run", "A"])
    assert args.name == "hss"  # default unchanged


def test_crossref_to_works_shape():
    payload = {"message": {"items": [
        {"DOI": "10.17323/JLE.2020.10316", "title": ["Modeling"],
         "issued": {"date-parts": [[2020, 6]]},
         "author": [{"given": "Rod", "family": "Roscoe"}, {"family": "Solo"}]},
        {"title": ["no doi"]},
    ]}}
    works = bh.crossref_to_works(payload)
    assert len(works) == 1
    w = works[0]
    assert w["doi"] == "10.17323/jle.2020.10316" and w["publication_year"] == 2020
    assert bh._authors(w) == ["Rod Roscoe", "Solo"] and w["title"] == "Modeling"
    assert "api.crossref.org" in bh.crossref_query_url("1072-4303", "x@y.z")


def test_is_clean_sentence_rejects_pdf_artefacts():
    assert bh.is_clean_sentence("The learners in the treatment group improved their scores.")
    assert bh.is_clean_sentence("Only 11.8% of the 45 participants reported it (p < .05).")
    assert not bh.is_clean_sentence("Only 11.8% of participants reported being ac- cused of it.")
    assert not bh.is_clean_sentence(
        "Motivation Questionnaire post .17 45 .183 Results for Research Question One As above."
    )
    assert not bh.is_clean_sentence("Table 3 (n = 45) M 3.21 SD 0.87 t 2.10 df 44 p .041 d 0.6.")
    assert not bh.is_clean_sentence("Both schools are top performing in the UK.2 Most agreed.")
    assert not bh.is_clean_sentence("Keywords: motivation; engagement; motivated behavior 20 1.")
    assert bh.alter("The findings are significant as they show a clear pattern.")[0].startswith(
        "The findings are not significant"
    )
    chunks = [{"section": "results", "text": "The mean was 3.2 (SD 1.1) and ac- cused words appear "
               "in this long sentence of the results section here. The learners in the treatment "
               "group improved their writing scores substantially over the semester."}]
    cands = bh.candidates_for_paper(chunks)
    assert [c["sentence"][:12] for c in cands] == ["The learners"]


# ---------------------------------------------------------------- additional cases


TITLE = ("Towards a better understanding of the L2 Learning Experience, the Cinderella of the "
         "L2 Motivational Self System")
ARTEFACT_REFERENCE = ("Gallup Student Poll; Engaged today - ready for tomorrow: Fall 2015 "
                      "survey results.")
ARTEFACT_HEADER = ("For example, in a large-scale survey in China we found that: Towards a "
                   "better understanding of the L2 Learning Experience, the Cinderella of the "
                   "L2 Motivational. . .")


def test_is_clean_sentence_rejects_reference_lines_ellipses_and_running_headers():
    assert not bh.is_clean_sentence(ARTEFACT_REFERENCE)
    assert not bh.is_clean_sentence(ARTEFACT_HEADER)
    assert not bh.is_clean_sentence(ARTEFACT_HEADER, title=TITLE)
    assert not bh.is_clean_sentence("The learners improved their scores over the year\u2026")
    assert not bh.is_clean_sentence("The learners improved their scores over the year. . .")
    # the title alone (running header) is rejected when the title is known
    assert not bh.is_clean_sentence(
        "Moreover, our data suggest that towards a better understanding of the L2 learning "
        "experience, the Cinderella of the L2 Motivational Self System, is overdue.",
        title=TITLE,
    )
    assert bh.is_clean_sentence(
        "The learners in the treatment group improved their scores significantly.", title=TITLE
    )
    assert not bh.is_clean_sentence(
        "Fredricks, J. A., Blumenfeld, P. C., & Paris, A. H. (2004). School engagement: "
        "Potential of the concept, state of the evidence."
    )
    assert not bh.is_clean_sentence("Review of Educational Research, 74(1), 59-109.")
    assert not bh.is_clean_sentence(
        "Dewaele and MacIntyre (2014) reported this; Foreign Language Enjoyment And Anxiety "
        "In The Classroom."
    )
    # a normal sentence with a semicolon and a citation is fine
    assert bh.is_clean_sentence(
        "Enjoyment rose over the semester; anxiety, in contrast, remained stable (Dewaele, "
        "2014) for most learners."
    )


def test_candidates_honour_title_and_veto_list(tmp_path: Path):
    chunks = [{"section": "results", "text": (
        f"{ARTEFACT_HEADER} The learners in the treatment group improved their writing scores "
        "substantially over the semester. Scores were significantly higher for the experimental "
        f"group than for the control group after ten weeks. {ARTEFACT_REFERENCE}")}]
    cands = bh.candidates_for_paper(chunks, title=TITLE)
    texts = [c["sentence"] for c in cands]
    assert len(texts) == 2 and all("Cinderella" not in t and "Gallup" not in t for t in texts)
    veto = {"The learners in the treatment group improved their writing scores substantially "
            "over the semester."}
    cands = bh.candidates_for_paper(chunks, title=TITLE, veto=veto)
    assert [c["sentence"][:6] for c in cands] == ["Scores"]
    veto_path = tmp_path / "hss_veto.json"
    write_json(veto_path, {"reason": "manual", "sentences": sorted(veto)})  # test fixture
    assert bh.load_veto(veto_path) == veto
    assert bh.load_veto(tmp_path / "absent.json") == set()


def test_committed_veto_file_and_build_info(tmp_path: Path):
    veto = bh.load_veto(bh.DEFAULT_VETO)
    assert isinstance(veto, set)
    make_cache(tmp_path / "cache")
    items, info = bh.build_items_from_cache(tmp_path / "cache", seed=20260902, veto={"x y z"})
    assert info["veto"] == ["x y z"] and info["veto_file"] is None
    assert info["expected"]["paraphrase"] == ["verified"]
    assert len(items) == 30


def test_paraphrase_never_touches_technical_terms():
    subs = dict(bh.SUBSTITUTIONS)
    for term in ("effect", "increase", "significant", "participants"):
        assert term not in subs
    # "case study" / "pilot study" are terms, not the generic noun
    text, made = bh.paraphrase(
        "The limitations relate to generalizability, because the studies involve a small "
        "number of participants or are conducted as a case study."
    )
    assert "case study" in text and made == []
    text, made = bh.paraphrase("A pilot study preceded the main study of the learners.")
    assert text == "A pilot study preceded the main investigation of the learners."
    assert made == [("study", "investigation")]
    text, made = bh.paraphrase(
        "The results showed a significant effect of feedback on participants' scores."
    )
    assert "significant effect" in text and "participants" in text
    assert ("results", "findings") in made and ("showed", "demonstrated") in made
    assert bh.EXPECTED["paraphrase"] == ["verified"]


def test_selected_from_never_carries_an_email():
    assert "@" not in bh.public_url(bh.crossref_query_url("1072-4303", "x@y.z"))
    assert "@" not in bh.public_url(bh.openalex_query_url("1072-4303", "x@y.z"))
    assert bh.public_url("https://api.example.org/works?a=1&mailto=x@y.z&b=2") == (
        "https://api.example.org/works?a=1&b=2"
    )
    for src in json.loads(bh.DEFAULT_SOURCES.read_text(encoding="utf-8")):
        assert "@" not in src["selected_from"]


def test_committed_hss_claims_are_clean():
    rows = [json.loads(x) for x in bh.DEFAULT_OUT.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 30
    titles = {s["doi"]: s["title"] for s in json.loads(bh.DEFAULT_SOURCES.read_text("utf-8"))}
    veto = bh.load_veto(bh.DEFAULT_VETO)
    for r in rows:
        assert bh.is_clean_sentence(r["original_sentence"], title=titles.get(r["source_doi"]))
        assert r["original_sentence"] not in veto
        assert "Gallup" not in r["claim"] and "Cinderella" not in r["claim"]
        if r["rule"] == "paraphrase":
            assert r["expected"] == ["verified"]


def test_is_clean_sentence_rejects_leading_heading_words():
    assert not bh.is_clean_sentence(
        "Conclusion There is no doubt that the refinement of the component is a timely task."
    )
    assert not bh.is_clean_sentence(
        "Discussion The learners in the treatment group improved their writing scores."
    )
    assert bh.is_clean_sentence(
        "Conclusions drawn from the data were cautious about the effect of feedback on learners."
    )
    # a Title-Case heading run glued to a sentence that starts with a function word
    assert not bh.is_clean_sentence(
        "Theoretical Frameworks A variety of theoretical frameworks and concepts have been "
        "applied to the study of teacher agency."
    )
    assert not bh.is_clean_sentence(
        "Foreign Language Enjoyment This construct was measured with the ten-item scale."
    )
    assert bh.is_clean_sentence(
        "Japanese Foreign Language Enjoyment scores were higher for the older learners overall."
    )
    assert bh.is_clean_sentence("In New Zealand the learners reported higher enjoyment overall.")


def test_is_clean_sentence_rejects_a_heading_run_with_interleaved_function_words():
    """The committed hss-no_full_text-02 item's claim is exactly this literal sentence.
    HEADING_PREFIX_RE's Title-Case branch must recognise a heading that mixes capitalised
    words with lowercase function words ("Impact of the Classroom Language Assessment Course
    on Pre-Service Teachers") as glued to the sentence that follows it, not just a run of 1-4
    consecutive capitalised words."""
    assert not bh.is_clean_sentence(
        "Impact of the Classroom Language Assessment Course on Pre-Service Teachers As the "
        "data below indicate, it became apparent that the pre-service teachers underwent a "
        "radical change towards their conception of what language assessment implied."
    )
    # a normal Title-Case subject phrase followed by an ordinary predicate is untouched: its
    # lowercase verb never matches the (exact-case) sentence-opener alternation
    assert bh.is_clean_sentence(
        "The Classroom Language Assessment Course was piloted successfully across every "
        "school this term."
    )


def test_is_clean_sentence_rejects_questions_and_merged_headings():
    assert not bh.is_clean_sentence(
        "So, where does this leave the third core component of the L2 Motivational Self "
        "System, the L2 Learning Experience?"
    )
    assert not bh.is_clean_sentence(
        "Why has it not featured more prominently in either the theoretical or the research "
        "developments of the past decade?"
    )
    assert not bh.is_clean_sentence(
        "The origins of the L2 Learning Experience The L2 Motivational Self System was partly "
        "the outcome of empirical research conducted in Hungary and partly of theoretical "
        "advances in the fields of applied linguistics and psychology."
    )
    assert bh.is_clean_sentence(
        "The L2 Motivational Self System was partly the outcome of empirical research "
        "conducted in Hungary and partly of theoretical advances in applied linguistics."
    )


# ---------------------------------------------------------------- additional cases

WRONG_PAPER_02 = (
    "We believe this number would have been significantly lower just a month or two later."
)
ALTERED_03_ORIGINAL = (
    "Moreover, a post-test was not administered in the study and as a result, it is not known "
    "if L2 students can become more effective English writers through the use of intelligent "
    "writing assistants."
)


def test_is_clean_sentence_rejects_anaphoric_openers_and_fragments():
    assert not bh.is_clean_sentence(WRONG_PAPER_02)
    assert not bh.is_clean_sentence("These findings were echoed by the second cohort.")
    assert not bh.is_clean_sentence("This result held across all three tasks.")
    assert not bh.is_clean_sentence("We believe that the effect would replicate elsewhere.")
    # a numeral or noun antecedent in the sentence itself keeps it eligible
    assert bh.is_clean_sentence(
        "Twelve learners reported anxiety, and this number rose to 20 after the second task."
    )
    assert bh.is_clean_sentence(
        "The learners improved their scores, and these findings held in the delayed test."
    )


def test_alter_direction_skips_flips_inside_not_known_whether_clauses():
    out = bh.alter(ALTERED_03_ORIGINAL)
    assert out is None or not out[1].startswith("direction_flip")
    assert bh._alter_direction("It is not known whether scores were higher.") is None
    assert bh._alter_direction("Scores were higher after training.") == (
        "Scores were lower after training.", "direction_flip:higher->lower"
    )


def test_committed_veto_lists_the_two_weak_stimuli_and_build_records_them():
    veto = bh.load_veto(bh.DEFAULT_VETO)
    assert WRONG_PAPER_02 in veto and ALTERED_03_ORIGINAL in veto
    build = json.loads(bh.DEFAULT_OUT.with_name("hss_claims.build.json").read_text("utf-8"))
    assert WRONG_PAPER_02 in build["veto"] and ALTERED_03_ORIGINAL in build["veto"]
    rows = [json.loads(x) for x in bh.DEFAULT_OUT.read_text(encoding="utf-8").splitlines()]
    originals = {r["original_sentence"] for r in rows}
    assert WRONG_PAPER_02 not in originals and ALTERED_03_ORIGINAL not in originals


# ---------------------------------------------------------------- additional cases

PARAPHRASE_02_ORIGINAL = (
    "The authors found that witnessing other students cheat increases cheating behavior in "
    "the observers."
)
PARAPHRASE_04_ORIGINAL = (
    "The results of his contribution proved that there is a significant correlation between "
    "the ideal L2-self and WTC."
)
PARAPHRASE_05_ORIGINAL = (
    "Furthermore, Elahi Shirvan and Talebzadeh (2018a) indicated that learners\u2019 feeling of "
    "enjoyment fluctuates over time under the influence of topics presented to them."
)
ALTERED_09_ORIGINAL = (
    "In other words, the application of the verb \u201cto engage\u201d offers a natural way of "
    "mapping the most important facets of the learning experience, which in turn allows us to "
    "capture the key aspects in measurable terms."
)
ROUND3_VETO = (PARAPHRASE_02_ORIGINAL, PARAPHRASE_04_ORIGINAL, PARAPHRASE_05_ORIGINAL,
               ALTERED_09_ORIGINAL)


def test_is_clean_sentence_rejects_attributed_findings_of_other_studies():
    """A literature-review sentence reports another study's finding; cited to the HSS paper it
    attributes the finding to the wrong primary source, so it is not a 'verified' stimulus."""
    for text in (PARAPHRASE_02_ORIGINAL, PARAPHRASE_04_ORIGINAL, PARAPHRASE_05_ORIGINAL):
        assert not bh.is_clean_sentence(text), text
    assert not bh.is_clean_sentence("Smith (2019) found that feedback improved accuracy.")
    assert not bh.is_clean_sentence("Smith and Jones (2019) argued that feedback helps.")
    assert not bh.is_clean_sentence("Smith et al. (2019) reported a large effect.")
    assert not bh.is_clean_sentence(
        "As Smith et al. (2019) showed, feedback improved accuracy in the treatment group."
    )
    assert not bh.is_clean_sentence("The researchers found that feedback improved accuracy.")
    assert not bh.is_clean_sentence("Their study showed that feedback improved accuracy.")
    assert not bh.is_clean_sentence("In her work, feedback was found to improve accuracy.")
    assert not bh.is_clean_sentence(
        "Recent research has found no fixed nonverbal cue indicating enjoyment (Elahi Shirvan "
        "& Talebzadeh, 2018b) and language anxiety (Gregersen, MacIntyre, & Olson, 2017)."
    )
    assert not bh.is_clean_sentence("Previous studies have shown that feedback helps.")
    assert not bh.is_clean_sentence("Studies have consistently reported a positive effect.")
    # the paper's own findings stay eligible, including a parenthetical citation at the end
    assert bh.is_clean_sentence("The learners in the treatment group improved their scores.")
    assert bh.is_clean_sentence("Our results showed that feedback improved accuracy.")
    assert bh.is_clean_sentence(
        "Feedback improved accuracy, in line with earlier work (Smith et al., 2019)."
    )
    assert bh.is_clean_sentence("The authors of this study observed a large effect.")
    # The attribution opener is position-independent and an "L2"-style token is
    # not a numeral antecedent for an anaphoric opener
    assert not bh.is_clean_sentence(PARAPHRASE_05_R3_ORIGINAL)
    assert not bh.is_clean_sentence(VERBATIM_03_R3_ORIGINAL)
    assert not bh.is_clean_sentence(
        "In their discussion, the authors reported that feedback improved accuracy."
    )
    assert not bh.is_clean_sentence("These findings held for the L2 learners in the cohort.")
    assert bh.is_clean_sentence("Here, the authors of this study observed a large effect.")
    # A cited study's *procedure* and literature-citing subjects with an
    # adjective or a parenthetical citation are attributions too
    assert not bh.is_clean_sentence(
        "Among these studies, Newcomer and Collier (2015), however, recruited participants "
        "both from second to eighth grade and universities."
    )
    assert not bh.is_clean_sentence(
        "Some excellent examples (e.g., Babino & Stewart, 2018; Ilieva & Ravindran, 2018) "
        "show how this can be achieved."
    )
    assert not bh.is_clean_sentence("Several recent studies have shown that feedback helps.")
    assert not bh.is_clean_sentence("Smith (2019) used a pre-test and a post-test design.")
    # possessive citations and "the study by ..." frames describe another study
    assert not bh.is_clean_sentence(
        "In Liyanage et al.‟s (2015) study, eight teacher participants were interviewed "
        "individually for around 30 to 45 minutes."
    )
    assert not bh.is_clean_sentence(
        "In Smith’s (2019) work, learners were interviewed twice during the semester."
    )
    assert not bh.is_clean_sentence(
        "In the study by Smith and Jones (2019), learners were interviewed twice a term."
    )
    assert not bh.is_clean_sentence(
        "Activity theory is also applied in studies by Yang (2018) as well as Yang and Clark "
        "(2018) to analyze teachers‟ action within the context of an activity system."
    )
    assert not bh.is_clean_sentence(
        "For instance, in White (2018), even though it is unclear in which school context "
        "this study is situated, it identifies that the student group is immigrant learners."
    )
    assert not bh.is_clean_sentence(
        "Two studies (Kang, 2017; Tutunis & Hacifazlioglu, 2018) are qualitative in nature "
        "but also utilize a quantitative approach in data analysis."
    )
    assert bh.is_clean_sentence(
        "Second most common strategy is classroom observation (n=17), followed by focus "
        "group discussion (n=6) among the studies reviewed."
    )
    assert bh.is_clean_sentence(
        "We recruited participants both from second to eighth grade and universities."
    )
    assert bh.is_clean_sentence("Of the 40 L2 learners, these findings held for the majority.")


def test_alter_quantifier_never_flips_superlative_most_or_at_all():
    assert bh._alter_quantifier("mapping the most important facets of the experience") is None
    assert bh._alter_quantifier("They were not confident at all about the results.") is None
    assert bh._alter_quantifier("Most learners preferred written feedback.") == (
        "Few learners preferred written feedback.", "quantifier_flip:most->few"
    )
    assert bh._alter_quantifier("All learners preferred written feedback.") == (
        "No learners preferred written feedback.", "quantifier_flip:all->no"
    )
    out = bh.alter(ALTERED_09_ORIGINAL)
    assert out is None or "the few important" not in out[0]


def test_committed_veto_lists_the_round3_stimuli_and_the_set_excludes_them():
    veto = bh.load_veto(bh.DEFAULT_VETO)
    assert len(veto) >= 6 and all(s in veto for s in ROUND3_VETO)
    build = json.loads(bh.DEFAULT_OUT.with_name("hss_claims.build.json").read_text("utf-8"))
    assert all(s in build["veto"] for s in ROUND3_VETO)
    rows = [json.loads(x) for x in bh.DEFAULT_OUT.read_text(encoding="utf-8").splitlines()]
    originals = {r["original_sentence"] for r in rows}
    assert not originals & set(ROUND3_VETO)
    assert not any("the few important" in r["claim"] for r in rows)
    counts = json.loads(json.dumps(build["counts"]))
    # this is the frozen, committed hss_claims.build.json (predates over_specified): only
    # the five original rules, never regenerated here.
    assert counts == {"verbatim": 5, "paraphrase": 5, "altered": 10, "wrong_paper": 5,
                      "no_full_text": 5}
    assert "attribution" in json.dumps(build["candidate_ranking"]).lower()


# ---------------------------------------------------------------- additional cases

VERBATIM_03_R3_ORIGINAL = (
    "These outcomes offered justification for developing a new theoretical construct that was "
    "centered around the Ideal L2 Self, with an obvious second component to be included in "
    "this construct being the Ought-to L2 Self."
)
VERBATIM_04_R3_ORIGINAL = (
    "In addition, those studying more FLs also scored significantly higher on FLE, where FLCA "
    "was not associated with studying more FLs."
)
PARAPHRASE_05_R3_ORIGINAL = (
    "Last, the researchers showed the video to participants from various external sites "
    "through focus group interviews."
)
ROUND4_VETO = (VERBATIM_03_R3_ORIGINAL, VERBATIM_04_R3_ORIGINAL, PARAPHRASE_05_R3_ORIGINAL)
QUOTED_SENTENCE = (
    "Those studying more languages also scored significantly higher on enjoyment, where "
    "anxiety was not associated with studying more languages."
)
OWN_SENTENCE = (
    "Our own learners in the treatment group improved their writing scores substantially "
    "over the semester."
)
INNER_SENTENCE = (
    "The learners in the treatment group improved their writing scores substantially over "
    "the semester."
)


def test_candidates_for_paper_rejects_quoted_passages():
    """A block quotation closed by a page-referenced citation, or any sentence inside an open
    quotation mark, is another author's text and never a stimulus."""
    chunks = [{"section": "results", "text": (
        f"{OWN_SENTENCE} Prior work summarised the pattern as follows. {QUOTED_SENTENCE} "
        "(Dewaele &\nMacIntyre, 2016. p. 262)\nAn analysis of the qualitative data followed."
    )}]
    texts = [c["sentence"] for c in bh.candidates_for_paper(chunks)]
    assert QUOTED_SENTENCE not in texts and OWN_SENTENCE in texts
    chunks = [{"section": "results", "text": (
        "One reviewer wrote: “Scores rose in every cohort we followed for a full "
        f"academic year. {INNER_SENTENCE} This continued into the next year for all of the "
        "participants involved”. Scores were significantly higher for the experimental "
        "group than for the control group after ten weeks."
    )}]
    texts = [c["sentence"] for c in bh.candidates_for_paper(chunks)]
    assert INNER_SENTENCE not in texts
    assert any(t.startswith("Scores were significantly higher") for t in texts)
    chunks = [{"section": "results", "text": (
        'She said: "Scores rose in every cohort we followed for a full academic year. '
        f'{INNER_SENTENCE} This continued into the next year for all of the participants '
        f'involved". {OWN_SENTENCE}'
    )}]
    texts = [c["sentence"] for c in bh.candidates_for_paper(chunks)]
    assert INNER_SENTENCE not in texts and OWN_SENTENCE in texts
    assert bh.QUOTE_CITATION_RE.match(" (Dewaele & MacIntyre, 2016. p. 262)")
    assert bh.QUOTE_CITATION_RE.match("(Smith, 2019, pp. 12-14)")
    assert not bh.QUOTE_CITATION_RE.match(" (Smith, 2019)")


def test_committed_veto_lists_the_round4_stimuli_and_the_set_excludes_them():
    veto = bh.load_veto(bh.DEFAULT_VETO)
    assert len(veto) >= 9 and all(s in veto for s in ROUND3_VETO + ROUND4_VETO)
    build = json.loads(bh.DEFAULT_OUT.with_name("hss_claims.build.json").read_text("utf-8"))
    assert all(s in build["veto"] for s in ROUND4_VETO)
    assert "quot" in json.dumps(build["candidate_ranking"]).lower()
    rows = [json.loads(x) for x in bh.DEFAULT_OUT.read_text(encoding="utf-8").splitlines()]
    originals = {r["original_sentence"] for r in rows}
    assert not originals & set(ROUND4_VETO)
    # frozen, committed build.json: five original rules only.
    assert build["counts"] == {"verbatim": 5, "paraphrase": 5, "altered": 10,
                               "wrong_paper": 5, "no_full_text": 5}


# ---------------------------------------------------------------- additional cases

ALTERED_10_R4_ORIGINAL = (
    "However, according to the results of the ANCOVAs, it was only in the area of intrinsic "
    "motivation that the improvements of the two groups differed significantly."
)
ROUND5_VETO = (ALTERED_10_R4_ORIGINAL,)


def test_committed_veto_lists_the_round5_stimulus_and_the_set_excludes_it():
    # hss-altered-10 flipped "significantly" -> "not significantly" postmodifying the verb
    # "differed" (ungrammatical); DIRECTION_GUARDS now blocks that flip and the sentence is
    # additionally vetoed so no other alteration rule re-selects it.
    # Assert the property (this stimulus is listed and the built set excludes it), not
    # the veto list's total length, which later builds also grow.
    veto = bh.load_veto(bh.DEFAULT_VETO)
    assert all(s in veto for s in ROUND3_VETO + ROUND4_VETO + ROUND5_VETO)
    build = json.loads(bh.DEFAULT_OUT.with_name("hss_claims.build.json").read_text("utf-8"))
    assert all(s in build["veto"] for s in ROUND5_VETO)
    assert "DIRECTION_GUARDS" in build["candidate_ranking"]
    rows = [json.loads(x) for x in bh.DEFAULT_OUT.read_text(encoding="utf-8").splitlines()]
    originals = {r["original_sentence"] for r in rows}
    assert not originals & set(ROUND5_VETO)
    assert not any("a no significant" in r["claim"] for r in rows)
    assert not any("differed not significantly" in r["claim"] for r in rows)
    # frozen, committed build.json: five original rules only.
    assert build["counts"] == {"verbatim": 5, "paraphrase": 5, "altered": 10,
                               "wrong_paper": 5, "no_full_text": 5}


# ---------------------------------------------------------------- over_specified


def test_over_specify_appends_a_modifier_absent_from_the_paper_and_raises_without_paper_text():
    src = "Scores improved significantly for the treatment group."
    paper_text = "The treatment group scored higher than the control group after instruction."
    out = bh._over_specify(src, paper_text=paper_text)
    assert out is not None
    claim, modifier = out
    assert claim != src and modifier in bh.OVER_SPECIFIED_MODIFIERS
    assert claim.startswith("Scores improved significantly for the treatment group,")
    with pytest.raises(ValueError):
        bh._over_specify(src, paper_text="")
    with pytest.raises(ValueError):
        bh._rule_claim("over_specified", src)


def test_over_specify_rejects_a_modifier_whose_content_tokens_appear_in_the_paper():
    src = "Scores improved significantly for the treatment group."
    first_two = bh.OVER_SPECIFIED_MODIFIERS[:2]
    paper_text = " ".join(first_two) + " Nothing else relevant appears in this paper at all."
    out = bh._over_specify(src, paper_text=paper_text)
    assert out is not None
    _claim, modifier = out
    assert modifier not in first_two


def test_over_specify_returns_none_when_every_modifier_is_rejected():
    src = "Scores improved significantly for the treatment group."
    paper_text = " ".join(bh.OVER_SPECIFIED_MODIFIERS)  # every modifier's own tokens present
    assert bh._over_specify(src, paper_text=paper_text) is None


def test_over_specify_index_rotates_the_starting_modifier():
    src = "Scores improved significantly for the treatment group."
    paper_text = "Nothing relevant appears in this paper about the topic at hand."
    out0 = bh._over_specify(src, paper_text=paper_text, index=0)
    out3 = bh._over_specify(src, paper_text=paper_text, index=3)
    assert out0[1] == bh.OVER_SPECIFIED_MODIFIERS[0]
    assert out3[1] == bh.OVER_SPECIFIED_MODIFIERS[3]


def test_rule_claim_over_specified_prefixes_the_alteration_label():
    src = "Scores improved significantly for the treatment group."
    paper_text = "The treatment group scored higher than the control group after instruction."
    out = bh._rule_claim("over_specified", src, paper_text=paper_text)
    assert out is not None
    claim, label = out
    assert claim != src and label.startswith("over_specified:")


def test_rule_order_fill_order_expected_quotas_carry_over_specified():
    assert bh.RULE_ORDER == (
        "verbatim", "paraphrase", "altered", "over_specified", "wrong_paper", "no_full_text",
    )
    assert bh.FILL_ORDER[1] == "over_specified"
    assert bh.EXPECTED["over_specified"] == ["needs_nuance"]
    assert bh.QUOTAS["over_specified"] == 0  # default build stays byte identical


def test_over_specified_items_never_equal_their_original_sentence(tmp_path: Path):
    make_cache(tmp_path / "cache", n_sentences=20)
    out = tmp_path / "claims.jsonl"
    rc = bh.main([
        "--skip-fetch", "--cache-dir", str(tmp_path / "cache"), "--out", str(out),
        "--seed", "20260902", "--sources", str(tmp_path / "sources.json"),
        "--quotas", "over_specified=5,altered=0",
    ])
    assert rc == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    over = [r for r in rows if r["rule"] == "over_specified"]
    assert len(over) == 5
    assert all(r["claim"] != r["original_sentence"] for r in over)
    assert all(r["alteration"] and r["alteration"].startswith("over_specified:") for r in over)
    assert all(r["expected"] == ["needs_nuance"] for r in over)


def test_build_json_records_over_specified_rejections_and_stimulus_digests(tmp_path: Path):
    make_cache(tmp_path / "cache", n_sentences=20)
    out = tmp_path / "claims.jsonl"
    bh.main([
        "--skip-fetch", "--cache-dir", str(tmp_path / "cache"), "--out", str(out),
        "--seed", "20260902", "--sources", str(tmp_path / "sources.json"),
        "--quotas", "over_specified=2,altered=0",
    ])
    build = json.loads((tmp_path / "claims.build.json").read_text(encoding="utf-8"))
    assert "over_specified_rejections" in build
    assert build["n_over_specified_rejections"] == len(build["over_specified_rejections"])
    assert set(build["stimulus_constants_sha256"]) >= {
        "OVER_SPECIFIED_MODIFIERS", "OVER_SPECIFIED_STOPWORDS", "NUMERIC_FACTORS_NONROUND",
    }


# ---------------------------------------------------------------- ALTER_FNS


def test_alter_fns_dispatch_table_matches_the_three_operator_names():
    assert set(bh.ALTER_FNS) == {"direction_flip", "numeric", "quantifier_flip"}
    assert bh.ALTER_FNS["direction_flip"] is bh._alter_direction
    assert bh.ALTER_FNS["numeric"] is bh._alter_number_nonround
    assert bh.ALTER_FNS["quantifier_flip"] is bh._alter_quantifier


def test_alter_allowed_restricts_to_the_named_operators_in_table_order():
    src = "The study enrolled 24 students over 6 weeks and scores were significantly higher."
    out = bh.alter(src)  # default: direction flip wins first, byte identical to today
    assert out is not None and out[1].startswith("direction_flip")
    out2 = bh.alter(src, allowed=["numeric"])
    assert out2 is not None and out2[1].startswith("numeric")
    assert bh.alter("The scores improved over the semester.", allowed=["quantifier_flip"]) is None


# ---------------------------------------------------------------- non-round numeric


def test_alter_number_default_factor_is_unchanged_from_the_frozen_builds():
    assert bh._alter_number("The study enrolled 24 students over 6 weeks.") == (
        "The study enrolled 48 students over 6 weeks.", "numeric:24->48"
    )
    assert bh.NUMERIC_FACTORS_NONROUND == (1.37, 0.63, 1.19, 0.81)


def test_is_round_or_unit_ratio_known_values():
    assert bh._is_round_or_unit_ratio(2.0) is True
    assert bh._is_round_or_unit_ratio(0.5) is True
    assert bh._is_round_or_unit_ratio(0.25) is True
    assert bh._is_round_or_unit_ratio(10.0) is True
    assert bh._is_round_or_unit_ratio(60.0) is True
    assert bh._is_round_or_unit_ratio(1 / 60) is True
    assert bh._is_round_or_unit_ratio(1.37) is False
    assert bh._is_round_or_unit_ratio(0.63) is False


def test_alter_number_nonround_uses_the_fixed_factor_set_and_never_a_round_ratio():
    text, label = bh._alter_number_nonround("The study enrolled 100 students over 6 weeks.")
    assert label.startswith("numeric:100->")
    new_value = float(label.split("->")[1])
    assert not bh._is_round_or_unit_ratio(new_value / 100.0)
    assert text.startswith("The study enrolled ")


def test_alter_number_nonround_keeps_the_percent_cap():
    text, label = bh._alter_number_nonround("Exactly 75% of the learners passed the test.")
    assert "99%" in text
    assert float(label.split("->")[1]) <= 99.0


# ---------------------------------------------------------------- --quotas


def test_parse_quotas_merges_over_zero_default_and_validates_rule_names():
    q = bh.parse_quotas("verbatim=8,altered=18,wrong_paper=8,no_full_text=8")
    assert q == {"verbatim": 8, "paraphrase": 0, "altered": 18, "over_specified": 0,
                "wrong_paper": 8, "no_full_text": 8}
    assert bh.parse_quotas(None) == dict(bh.QUOTAS)
    with pytest.raises(SystemExit):
        bh.parse_quotas("not_a_rule=1")


def test_build_hss_set_cli_quotas_flag_overrides_defaults(tmp_path: Path):
    make_cache(tmp_path / "cache", n_sentences=20)
    out = tmp_path / "claims.jsonl"
    rc = bh.main([
        "--skip-fetch", "--cache-dir", str(tmp_path / "cache"), "--out", str(out),
        "--seed", "20260902", "--sources", str(tmp_path / "sources.json"),
        "--quotas", "verbatim=2,altered=2,wrong_paper=2,no_full_text=2",
    ])
    assert rc == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8
    build = json.loads((tmp_path / "claims.build.json").read_text(encoding="utf-8"))
    assert build["quotas"] == {"verbatim": 2, "paraphrase": 0, "altered": 2, "over_specified": 0,
                              "wrong_paper": 2, "no_full_text": 2}
    assert build["n_items_requested"] == 8


# ---------------------------------------------------------------- --alteration-quotas


def test_parse_alteration_quotas_validates_operators_and_sum():
    q = bh.parse_alteration_quotas("direction_flip=2,numeric=1,quantifier_flip=1", altered_quota=4)
    assert q == {"direction_flip": 2, "numeric": 1, "quantifier_flip": 1}
    assert bh.parse_alteration_quotas(None, altered_quota=4) is None
    with pytest.raises(SystemExit):
        bh.parse_alteration_quotas("bogus_op=4", altered_quota=4)
    with pytest.raises(SystemExit):
        bh.parse_alteration_quotas("direction_flip=1,numeric=1", altered_quota=4)


def _make_cache_with_quantifier_sentence(cache_dir: Path, n_sentences: int = 20) -> None:
    """Like ``make_cache``, plus one quantifier-flip-eligible sentence per paper (``sentence``
    alone offers direction and numeric handles only, never a quantifier word)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for p, doi in enumerate(DOIS):
        body = " ".join(sentence(p, i) for i in range(n_sentences))
        body += f" Most learners in cohort {p} preferred the online format over the printed one."
        payload = {
            "doi": doi,
            "title": f"Paper {p}",
            "authors": [f"Author {p}"],
            "source_url": f"https://example.org/{p}.pdf",
            "fetched_at": "2026-09-02T00:00:00+00:00",
            "n_chars": len(body) * 3,
            "chunks": [
                # see make_cache's identical comment: keeps the default-on identity check happy.
                {"section": "introduction", "text": f"Paper {p}. Author {p}. Short intro. "
                 "Nothing of note here today."},
                {"section": "results", "text": body},
                {"section": "conclusion", "text": "We conclude that this happened in the end."},
            ],
        }
        write_json(cache_dir / f"{bh.doi_slug(doi)}.json", payload)


def test_assign_rules_alteration_quotas_fills_the_named_sub_quotas(tmp_path: Path):
    _make_cache_with_quantifier_sentence(tmp_path / "cache", n_sentences=20)
    cands = {doi: bh.candidates_for_paper(bh.load_cache(tmp_path / "cache")[doi]["chunks"])
             for doi in DOIS}
    items = bh.assign_rules(
        cands, 20260902,
        quotas={**{r: 0 for r in bh.RULE_ORDER}, "altered": 6},
        alteration_quotas={"direction_flip": 3, "numeric": 2, "quantifier_flip": 1},
    )
    altered = [it for it in items if it["rule"] == "altered"]
    assert len(altered) == 6
    ops = [it["alteration"].split(":", 1)[0] for it in altered]
    assert ops.count("direction_flip") == 3
    assert ops.count("numeric") == 2
    assert ops.count("quantifier_flip") == 1


def test_assign_rules_alteration_quotas_raises_systemexit_naming_operator_and_counts(
    tmp_path: Path,
):
    # this fixture's sentences offer no quantifier-flip handle at all
    make_cache(tmp_path / "cache", n_sentences=10)
    cands = {doi: bh.candidates_for_paper(bh.load_cache(tmp_path / "cache")[doi]["chunks"])
             for doi in DOIS}
    with pytest.raises(SystemExit) as exc:
        bh.assign_rules(
            cands, 20260902,
            quotas={**{r: 0 for r in bh.RULE_ORDER}, "altered": 3},
            alteration_quotas={"direction_flip": 0, "numeric": 0, "quantifier_flip": 3},
        )
    msg = str(exc.value)
    assert "quantifier_flip" in msg and "3" in msg and "0" in msg


def test_build_hss_set_cli_alteration_quotas_flag(tmp_path: Path):
    _make_cache_with_quantifier_sentence(tmp_path / "cache", n_sentences=20)
    out = tmp_path / "claims.jsonl"
    rc = bh.main([
        "--skip-fetch", "--cache-dir", str(tmp_path / "cache"), "--out", str(out),
        "--seed", "20260902", "--sources", str(tmp_path / "sources.json"),
        "--quotas", "altered=6",
        "--alteration-quotas", "direction_flip=3,numeric=2,quantifier_flip=1",
    ])
    assert rc == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    ops = [r["alteration"].split(":", 1)[0] for r in rows if r["rule"] == "altered"]
    assert ops.count("direction_flip") == 3
    assert ops.count("numeric") == 2
    assert ops.count("quantifier_flip") == 1
    build = json.loads((tmp_path / "claims.build.json").read_text(encoding="utf-8"))
    assert build["alteration_quotas"] == {
        "direction_flip": 3, "numeric": 2, "quantifier_flip": 1,
    }
    assert build["alterations"]["numeric_mode"] == "nonround"


# ---------------------------------------------------------------- JOURNALS widened


def test_journals_widened_to_eighteen_with_unique_issns_and_source_ids():
    assert len(bh.JOURNALS) == 18
    issns = [j["issn"] for j in bh.JOURNALS]
    assert len(set(issns)) == 18
    new_ids = {
        "S5407051349", "S4210220284", "S4210222903", "S4210196789", "S2764441637",
        "S2764502050", "S2739108755", "S2738451673", "S2764809198", "S2764904417",
    }
    openalex_ids = {j["openalex"] for j in bh.JOURNALS if j["openalex"]}
    assert new_ids <= openalex_ids
    # the original eight are not reordered or edited
    assert bh.JOURNALS[0] == {
        "name": "Studies in Second Language Learning and Teaching", "issn": "2083-5205",
        "openalex": "S2764526439",
    }
    assert bh.JOURNALS[7]["name"] == "JALT CALL Journal"


def test_rejected_journal_candidates_records_both_with_reasons():
    names = {j["journal"] for j in bh.REJECTED_JOURNAL_CANDIDATES}
    assert names == {"CALL-EJ", "International Journal of Language Testing"}
    assert all(j.get("reason") for j in bh.REJECTED_JOURNAL_CANDIDATES)


# ---------------------------------------------------------------- --probe-journals


def test_probe_journals_writes_one_row_per_journal_and_selects_nothing(
    tmp_path: Path, monkeypatch
):
    async def fake_list_candidates(journal, mailto, **kw):
        return [{"doi": f"10.9999/{journal['issn']}", "journal": journal["name"],
                 "issn": journal["issn"], "selected_from": "https://x"}]

    async def fake_acquire_one(work, pipeline, *, email, min_chars, min_chunks):
        return {"doi": work["doi"]}, {"doi": work["doi"], "chunks": []}, None

    monkeypatch.setattr(bh, "list_candidates", fake_list_candidates)
    monkeypatch.setattr(bh, "acquire_one", fake_acquire_one)
    monkeypatch.setattr(bh, "load_pipeline", lambda: {})
    monkeypatch.setattr(bh, "load_dotenv_values", lambda path: {"OPENALEX_EMAIL": "x@y.z"})
    probe_out = tmp_path / "probe.json"
    sources_out = tmp_path / "sources.json"
    write_json(sources_out, [])
    claims_out = tmp_path / "claims.jsonl"
    rc = bh.main([
        "--probe-journals", "--probe-out", str(probe_out), "--sources", str(sources_out),
        "--out", str(claims_out),
    ])
    assert rc == 0
    rows = json.loads(probe_out.read_text(encoding="utf-8"))
    assert len(rows) == len(bh.JOURNALS)
    assert all(r["outcome"] == "probe_ok" for r in rows)
    assert json.loads(sources_out.read_text(encoding="utf-8")) == []  # nothing selected
    assert not claims_out.exists()
    assert not (tmp_path / "claims.build.json").exists()  # n_attempts never touched


def test_probe_journals_records_probe_failed_when_no_candidates(tmp_path: Path, monkeypatch):
    async def empty_candidates(journal, mailto, **kw):
        return []

    monkeypatch.setattr(bh, "list_candidates", empty_candidates)
    monkeypatch.setattr(bh, "load_pipeline", lambda: {})
    monkeypatch.setattr(bh, "load_dotenv_values", lambda path: {"OPENALEX_EMAIL": "x@y.z"})
    probe_out = tmp_path / "probe.json"
    rc = bh.main(["--probe-journals", "--probe-out", str(probe_out)])
    assert rc == 0
    rows = json.loads(probe_out.read_text(encoding="utf-8"))
    assert all(r["outcome"] == "probe_failed" for r in rows)


# ---------------------------------------------------------------- acquisition record


def test_acquire_sources_records_per_doi_attempts_and_reasons(tmp_path: Path, monkeypatch):
    async def fake_list_candidates(journal, mailto, **kw):
        return [{"doi": f"10.9999/{journal['issn']}-1", "journal": journal["name"],
                 "issn": journal["issn"], "selected_from": "https://x", "title": "T",
                 "publication_year": 2021, "id": None, "authorships": []}]

    async def fake_acquire_one(work, pipeline, *, email, min_chars, min_chunks):
        if work["journal"] == "Language Learning & Technology":
            return None, None, "not_oa"
        return (
            {"doi": work["doi"], "title": "T", "year": 2021, "journal": work["journal"],
             "issn": work["issn"], "openalex_id": None, "selected_from": work["selected_from"],
             "oa_url": "https://x/pdf", "license": None, "unpaywall_accessed": "now",
             "n_chars": 20000, "n_chunks": 5},
            {"doi": work["doi"], "title": "T", "authors": [], "source_url": "https://x/pdf",
             "fetched_at": "now", "n_chars": 20000,
             "chunks": [{"section": "results", "text": sentence(0, i)} for i in range(5)]},
            None,
        )

    monkeypatch.setattr(bh, "list_candidates", fake_list_candidates)
    monkeypatch.setattr(bh, "acquire_one", fake_acquire_one)
    monkeypatch.setattr(bh, "load_pipeline", lambda: {})
    monkeypatch.setattr(bh, "load_dotenv_values", lambda path: {"OPENALEX_EMAIL": "x@y.z"})

    sources = tmp_path / "sources.json"
    cache_dir = tmp_path / "cache"
    out = tmp_path / "claims.jsonl"
    rc = bh.main([
        "--sources", str(sources), "--cache-dir", str(cache_dir), "--out", str(out),
        "--seed", "1", "--n-sources", "3", "--per-journal", "1",
        "--quotas", "verbatim=1", "--min-chars", "1", "--min-chunks", "1",
        # this fake payload's title ("T") never appears in its own body; unrelated to what
        # this test checks (the attempts/reasons record), so the cache-side identity check
        # (default on) is disabled here rather than reworking the fixture.
        "--no-identity-check",
    ])
    assert rc == 0
    build = json.loads((tmp_path / "claims.build.json").read_text(encoding="utf-8"))
    assert build["min_chars"] == 1 and build["min_chunks"] == 1
    assert build["n_attempts"] == 1
    assert isinstance(build["attempts"], list) and build["attempts"]
    assert {a["outcome"] for a in build["attempts"]} <= {"kept", "dropped"}
    assert any(
        a["outcome"] == "dropped" and a["reason"] == "not_oa" for a in build["attempts"]
    )
    assert len(build["journals_rejected"]) == 2


def test_build_hss_set_skip_fetch_merges_acquisition_record_instead_of_overwriting(
    tmp_path: Path,
):
    make_cache(tmp_path / "cache", n_sentences=20)
    out = tmp_path / "claims.jsonl"
    build_path = tmp_path / "claims.build.json"
    write_json(build_path, {
        "journals": {"X": {"issn": "1", "tried": 2, "kept": 1, "dropped": None}},
        "attempts": [{"doi": "10.1/a", "journal": "X", "outcome": "kept", "reason": None}],
        "n_attempts": 2, "min_chars": 15000, "min_chunks": 3,
        "journals_rejected": [{"journal": "Y", "issn": "2", "reason": "r"}],
        "excluded_dois": ["10.9/z"], "n_excluded": 1,
    })
    rc = bh.main([
        "--skip-fetch", "--cache-dir", str(tmp_path / "cache"), "--out", str(out),
        "--seed", "20260902", "--sources", str(tmp_path / "sources_new.json"),
        "--min-chars", "999", "--min-chunks", "1",
    ])
    assert rc == 0
    build = json.loads(build_path.read_text(encoding="utf-8"))
    assert build["journals"] == {"X": {"issn": "1", "tried": 2, "kept": 1, "dropped": None}}
    assert build["attempts"] == [
        {"doi": "10.1/a", "journal": "X", "outcome": "kept", "reason": None}
    ]
    assert build["n_attempts"] == 2  # unchanged: no fetching invocation happened this run
    assert build["min_chars"] == 15000 and build["min_chunks"] == 3  # historical, not this run
    assert build["journals_rejected"] == [{"journal": "Y", "issn": "2", "reason": "r"}]
    assert build["excluded_dois"] == ["10.9/z"] and build["n_excluded"] == 1


# ---------------------------------------------------------------- per-paper cap at scale


def test_build_items_from_cache_caps_at_five_items_per_paper_at_v3_scale(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    dois = [f"10.1000/paper-{i}" for i in range(20)]
    for i, doi in enumerate(dois):
        body = " ".join(sentence(i, j) for j in range(25))
        write_json(cache_dir / f"{bh.doi_slug(doi)}.json", {
            "doi": doi, "title": f"Paper {i}", "authors": [f"Author {i}"],
            "source_url": "https://example.org", "fetched_at": "2026-09-02T00:00:00+00:00",
            "n_chars": len(body) * 3,
            "chunks": [{"section": "results", "text": body}],
        })
    items, info = bh.build_items_from_cache(
        cache_dir, seed=1,
        quotas={"verbatim": 8, "paraphrase": 8, "altered": 18, "over_specified": 10,
                "wrong_paper": 8, "no_full_text": 8},
        identity_check=False,  # unrelated to this test; these titles never appear in their body
    )
    assert len(items) == 60
    per_paper: dict[str, int] = {}
    for it in items:
        per_paper[it["source_doi"]] = per_paper.get(it["source_doi"], 0) + 1
    assert max(per_paper.values()) <= 5
    assert info["n_sources"] == 20


# ---------------------------------------------------------------- CLI flags exist


def test_parse_args_exposes_new_v3_flags():
    args = bh.parse_args([
        "--quotas", "verbatim=1", "--alteration-quotas", "numeric=1", "--probe-journals",
    ])
    assert args.quotas == "verbatim=1"
    assert args.alteration_quotas == "numeric=1"
    assert args.probe_journals is True
    defaults = bh.parse_args([])
    assert defaults.quotas is None
    assert defaults.alteration_quotas is None
    assert defaults.probe_journals is False
    assert defaults.probe_out == bh.DEFAULT_PROBE_OUT


# ================================================================================
# Source-level hygiene
# ================================================================================


# ---------------------------------------------------------------- (1) wrong-work identity


def test_check_fetched_text_identity_flags_a_mismatched_fetch():
    ok, detected = bh.check_fetched_text_identity(
        expected_title=(
            "Translanguaging Pedagogy in Thailand's English Medium of Instruction Classrooms: "
            "Teachers' Perspectives and Practices"
        ),
        expected_authors=["Napapat Thongwichit", "Mark B. Ulla"],
        extracted_text=(
            "Issues in Educational Research, 32(3), 2022\n871\nSupporting English teaching in "
            "Thailand by accepting translanguaging: Views from Thai university teachers\n"
            "Eric A. Ambele\nMahasarakham University, Thailand\n\nThis article reports on a "
            "study of teacher views on translanguaging practices in university classrooms."
        ),
    )
    assert ok is False
    assert detected and ("Ambele" in detected or "Supporting English teaching" in detected)


def test_check_fetched_text_identity_accepts_a_matching_fetch():
    title = "Willingness to Communicate and Enjoyment in the L2 Classroom"
    ok, detected = bh.check_fetched_text_identity(
        expected_title=title,
        expected_authors=["Jane Q. Smith"],
        extracted_text=(
            f"{title}\nJane Q. Smith\nUniversity of Somewhere\n\nThis study examines "
            "willingness to communicate and enjoyment among learners in the second language "
            "classroom over one semester."
        ),
    )
    assert ok is True and detected is None


def test_check_fetched_text_identity_folds_diacritics_in_the_author_surname():
    """A plain ``str.casefold()`` maps the Turkish dotted capital I (U+0130, as OpenAlex
    records "Majid ELANİ SHİRVAN") to "i" plus a *combining* dot above (U+0307), which is
    never a substring of a PDF's plain-ASCII "Shirvan". The real committed cache entry for
    10.32601/ejal.710194 has a title/text token overlap of 1.000 and is unambiguously the
    right article, yet an unfolded comparison would incorrectly reject it."""
    title = (
        "Tracing the signature dynamics of foreign language classroom anxiety and foreign "
        "language enjoyment: A retrodictive qualitative modeling"
    )
    ok, detected = bh.check_fetched_text_identity(
        expected_title=title,
        expected_authors=["Majid ELANİ SHİRVAN", "Nahid TALEBZADEH"],
        extracted_text=(
            f"{title}\nMajid Elahi Shirvana*, Nahid Talebzadeha\nUniversity of Bojnord, "
            "Bojnord, Iran\n\nThis study traces classroom anxiety and enjoyment over time."
        ),
    )
    assert ok is True and detected is None
    # a genuinely wrong author is still caught (the fold must not silently disable the check)
    ok2, detected2 = bh.check_fetched_text_identity(
        expected_title=title, expected_authors=["Someone ELSE"],
        extracted_text=f"{title}\nA Completely Different Author\n\nUnrelated body text here.",
    )
    assert ok2 is False and detected2


def test_fold_diacritics_strips_combining_marks_including_casefolds_own():
    assert bh._fold_diacritics("SHİRVAN").casefold() == "shirvan"
    assert bh._fold_diacritics("Wedín") == bh._fold_diacritics("Wedin")
    assert bh._fold_diacritics("flerspråkighet") == bh._fold_diacritics("flersprakighet")


def test_check_fetched_text_identity_skips_when_no_expected_title():
    assert bh.check_fetched_text_identity(
        expected_title="", expected_authors=None, extracted_text="Anything at all here.",
    ) == (True, None)


async def _empty_unpaywall_record(doi, email):
    return {}


def test_acquire_one_returns_wrong_work_when_identity_check_fails(monkeypatch):
    """No real network and no backend import: the pipeline dict is fully faked, so
    ``load_pipeline`` (which imports the backend) is never called."""

    class FakeUnpaywall:
        async def lookup(self, doi):
            return "https://example.org/paper.pdf"

    async def fake_fetch_pdf(url):
        return b"%PDF-1.4 fake bytes"

    def fake_extract(pdf_bytes):
        return (
            "Issues in Educational Research, 32(3), 2022\n871\nSupporting English teaching in "
            "Thailand by accepting translanguaging: Views from Thai university teachers\n"
            "Eric A. Ambele\nMahasarakham University, Thailand\n\n" + ("x" * 20000)
        )

    def fake_chunk(text):
        return [{"section": "results", "text": text}]

    monkeypatch.setattr(bh, "unpaywall_record", _empty_unpaywall_record)
    pipeline = {
        "unpaywall": FakeUnpaywall(), "fetch_pdf": fake_fetch_pdf, "extract": fake_extract,
        "chunk": fake_chunk,
    }
    work = {
        "doi": "10.55593/ej.27108a7",
        "title": (
            "Translanguaging Pedagogy in Thailand's English Medium of Instruction Classrooms: "
            "Teachers' Perspectives and Practices"
        ),
        "authorships": [{"author": {"display_name": "Napapat Thongwichit"}}],
        "journal": "TESL-EJ", "issn": "1072-4303", "selected_from": "https://x",
        "publication_year": 2024, "id": None,
    }
    source, payload, reason = asyncio.run(bh.acquire_one(
        work, pipeline, email="x@y.z", min_chars=1, min_chunks=1,
    ))
    assert source is None and payload is None
    assert reason is not None and reason.startswith("wrong_work")


def test_acquire_one_keeps_a_matching_fetch(monkeypatch):
    class FakeUnpaywall:
        async def lookup(self, doi):
            return "https://example.org/paper.pdf"

    async def fake_fetch_pdf(url):
        return b"%PDF-1.4 fake bytes"

    title = "Willingness to Communicate and Enjoyment in the L2 Classroom"

    def fake_extract(pdf_bytes):
        return (
            f"{title}\nJane Q. Smith\n\n"
            + "Learners reported higher willingness to communicate over the semester. " * 400
        )

    def fake_chunk(text):
        return [{"section": "results", "text": text}]

    monkeypatch.setattr(bh, "unpaywall_record", _empty_unpaywall_record)
    pipeline = {
        "unpaywall": FakeUnpaywall(), "fetch_pdf": fake_fetch_pdf, "extract": fake_extract,
        "chunk": fake_chunk,
    }
    work = {
        "doi": "10.1000/x", "title": title,
        "authorships": [{"author": {"display_name": "Jane Q. Smith"}}],
        "journal": "TESL-EJ", "issn": "1072-4303", "selected_from": "https://x",
        "publication_year": 2024, "id": None,
    }
    source, payload, reason = asyncio.run(bh.acquire_one(
        work, pipeline, email="x@y.z", min_chars=1, min_chunks=1,
    ))
    assert reason is None and source is not None and payload is not None


# ------------------------------------------------- (1b) wrong-work identity over cached payloads


def _wrong_work_cache() -> dict[str, dict]:
    """One correctly matched payload and one mis-fetched payload (the same shape the P7 build's
    committed cache actually carries): the DOI's own title/authors describe a different article
    than the joined chunk text. Each payload's real content sentence is deliberately >= 12
    words (``common.score_candidate_sentence``'s own eligibility floor) and set off with a
    period from the heading/byline fragment before it, so it is minable as a candidate in its
    own right."""
    return {
        "10.1/good": {
            "doi": "10.1/good", "title": "Willingness to Communicate in the L2 Classroom",
            "authors": ["Jane Q. Smith"],
            "chunks": [{"section": "results", "text": (
                "Willingness to Communicate in the L2 Classroom. Jane Q. Smith. University of "
                "Somewhere.\n\nLearners in the treatment group reported higher willingness to "
                "communicate with peers throughout the semester overall."
            )}],
        },
        "10.1/bad": {
            "doi": "10.1/bad", "title": "The Named TESL-EJ Article", "authors": ["Real Author"],
            "chunks": [{"section": "results", "text": (
                "Issues in Educational Research, 32(3), 2022. Supporting English teaching in "
                "Thailand by accepting translanguaging. Eric A. Ambele.\n\nThis report "
                "describes a wholly unrelated study of teacher views conducted at a "
                "different institution entirely."
            )}],
        },
    }


def test_check_cache_identity_drops_a_mismatched_payload():
    kept, dropped = bh.check_cache_identity(_wrong_work_cache())
    assert list(kept) == ["10.1/good"]
    assert len(dropped) == 1
    assert dropped[0]["doi"] == "10.1/bad"
    assert dropped[0]["detected"] and "Ambele" in dropped[0]["detected"]


def test_check_cache_identity_keeps_a_named_exception():
    kept, dropped = bh.check_cache_identity(_wrong_work_cache(), exceptions={"10.1/bad"})
    assert set(kept) == {"10.1/good", "10.1/bad"}
    assert dropped == []


def test_check_cache_identity_flags_the_known_committed_mis_fetched_sources():
    """Regression/escalation guard: these two committed cache payloads are a disclosed
    provenance defect in a committed artifact, not something this builder rewrites, but the
    identity check must be able to *detect* both of them once ``--identity-check-cache`` is
    on, wherever the cache lives."""
    skip_unless_cache_dir(bh.HERE / "data" / "hss_test_v3_fulltext")
    skip_unless_cache_dir(bh.HERE / "data" / "hss_fulltext")
    ej_27108a7 = json.loads(
        (bh.HERE / "data" / "hss_test_v3_fulltext" / "10-55593_ej-27108a7.json")
        .read_text(encoding="utf-8")
    )
    ej_26103a4 = json.loads(
        (bh.HERE / "data" / "hss_fulltext" / "10-55593_ej-26103a4.json").read_text(
            encoding="utf-8"
        )
    )
    cache = {ej_27108a7["doi"]: ej_27108a7, ej_26103a4["doi"]: ej_26103a4}
    kept, dropped = bh.check_cache_identity(cache)
    assert kept == {}
    assert {d["doi"] for d in dropped} == {"10.55593/ej.27108a7", "10.55593/ej.26103a4"}


def test_build_items_from_cache_identity_check_on_by_default_drops_a_mismatched_source(
    tmp_path: Path,
):
    """The identity check defaults on. ``make_cache``'s synthetic papers echo their own
    title/authors in the intro chunk (see its own comment) so they are correctly kept; the
    two ``_wrong_work_cache`` payloads are not, and "10.1/bad" is correctly dropped without
    anyone having to pass a flag."""
    make_cache(tmp_path)
    cache = _wrong_work_cache()
    for doi, payload in cache.items():
        write_json(tmp_path / f"{bh.doi_slug(doi)}.json", {**payload, "chunks": payload["chunks"]})
    items, info = bh.build_items_from_cache(tmp_path, seed=1, min_chunks=1)
    assert info["identity_check"] is True
    assert [d["doi"] for d in info["wrong_work_sources"]] == ["10.1/bad"]
    assert "10.1/bad" not in info["sources"]
    assert "10.1/good" in info["sources"] and set(DOIS) <= set(info["sources"])
    assert all(it["source_doi"] != "10.1/bad" and it.get("chunk_doi") != "10.1/bad" for it in items)


def test_build_items_from_cache_identity_check_can_be_disabled_to_keep_a_mismatched_source(
    tmp_path: Path,
):
    """The opt-out (``identity_check=False``, the CLI's ``--no-identity-check``): the
    mismatched source is kept and nothing is recorded as wrong-work."""
    make_cache(tmp_path)
    cache = _wrong_work_cache()
    for doi, payload in cache.items():
        write_json(tmp_path / f"{bh.doi_slug(doi)}.json", {**payload, "chunks": payload["chunks"]})
    items, info = bh.build_items_from_cache(tmp_path, seed=1, min_chunks=1, identity_check=False)
    assert info["identity_check"] is False
    assert info["wrong_work_sources"] == []
    assert "10.1/bad" in info["sources"]


def test_build_items_from_cache_identity_check_on_drops_and_records_a_mismatched_source(
    tmp_path: Path,
):
    """Only the two ``_wrong_work_cache`` payloads: ``make_cache``'s synthetic papers echo
    their own title/authors (see its own comment), so an identity check no longer flags them
    too. A one-item ``verbatim`` quota keeps this test to only the two payloads under test."""
    cache = _wrong_work_cache()
    for doi, payload in cache.items():
        write_json(tmp_path / f"{bh.doi_slug(doi)}.json", {**payload, "chunks": payload["chunks"]})
    quotas = bh.parse_quotas("verbatim=1")
    items, info = bh.build_items_from_cache(
        tmp_path, seed=0, min_chunks=1, identity_check=True, quotas=quotas,
    )
    assert info["identity_check"] is True
    assert info["sources"] == ["10.1/good"]
    assert [d["doi"] for d in info["wrong_work_sources"]] == ["10.1/bad"]
    assert all(it["source_doi"] != "10.1/bad" and it.get("chunk_doi") != "10.1/bad" for it in items)


def test_build_items_from_cache_default_identity_check_keeps_the_known_disclosed_exception(
    tmp_path: Path,
):
    """The frozen 30-item test set's one disclosed mis-fetch (``10.55593/ej.26103a4``) is kept
    unconditionally by the *default* identity check, via the module-level
    ``KNOWN_IDENTITY_EXCEPTIONS`` (no flag needed), while an unnamed mismatched source in the
    same cache is still dropped."""
    skip_unless_cache_dir(bh.HERE / "data" / "hss_fulltext")
    ej_26103a4 = json.loads(
        (bh.HERE / "data" / "hss_fulltext" / "10-55593_ej-26103a4.json")
        .read_text(encoding="utf-8")
    )
    write_json(tmp_path / "10-55593_ej-26103a4.json", ej_26103a4)
    cache = _wrong_work_cache()
    for doi, payload in cache.items():
        write_json(tmp_path / f"{bh.doi_slug(doi)}.json", {**payload, "chunks": payload["chunks"]})
    quotas = bh.parse_quotas("verbatim=1")
    items, info = bh.build_items_from_cache(tmp_path, seed=0, min_chunks=1, quotas=quotas)
    assert info["identity_check"] is True
    assert "10.55593/ej.26103a4" in info["sources"]
    assert [d["doi"] for d in info["wrong_work_sources"]] == ["10.1/bad"]
    assert info["identity_check_exceptions"] == ["10.55593/ej.26103a4"]
    assert (
        "byte identical"
        in info["identity_check_exception_reasons"]["10.55593/ej.26103a4"]
    )


def test_build_items_from_cache_identity_check_on_with_named_exception_keeps_the_source(
    tmp_path: Path,
):
    cache = _wrong_work_cache()
    for doi, payload in cache.items():
        write_json(tmp_path / f"{bh.doi_slug(doi)}.json", {**payload, "chunks": payload["chunks"]})
    quotas = bh.parse_quotas("verbatim=2")
    items, info = bh.build_items_from_cache(
        tmp_path, seed=0, min_chunks=1, identity_check=True, quotas=quotas,
        identity_check_exceptions=["10.1/bad"],
    )
    assert info["wrong_work_sources"] == []
    assert info["sources"] == ["10.1/bad", "10.1/good"]
    # the additional, caller-supplied exception is unioned with KNOWN_IDENTITY_EXCEPTIONS
    # (always applied), not a replacement for it.
    assert info["identity_check_exceptions"] == ["10.1/bad", "10.55593/ej.26103a4"]


def test_parse_args_identity_check_cache_defaults_on_and_can_be_disabled():
    args = bh.parse_args([])
    assert args.identity_check_cache is True
    assert args.identity_check_exempt == []
    args2 = bh.parse_args([
        "--no-identity-check", "--identity-check-exempt", "10.1/x",
        "--identity-check-exempt", "10.1/y",
    ])
    assert args2.identity_check_cache is False
    assert args2.identity_check_exempt == ["10.1/x", "10.1/y"]
    args3 = bh.parse_args(["--identity-check-cache"])
    assert args3.identity_check_cache is True  # explicit-on is still accepted, just redundant


# ---------------------------------------------------------------- (2) reference/front-matter chunks


def test_is_reference_or_frontmatter_chunk_flags_dense_bibliography_text():
    reference_chunk = (
        "Fredricks, J. A., Blumenfeld, P. C., & Paris, A. H. (2004). School engagement: "
        "Potential of the concept, state of the evidence. Review of Educational Research, "
        "74(1), 59-109. Rosen, J., & Wedin, A. (2015). Klassrumsinteraktion och "
        "flerspråkighet: Ett kritiskt perspektiv. Retrieved from https://example.org/x.pdf "
        "pp. 12-30."
    )
    assert bh.is_reference_or_frontmatter_chunk(reference_chunk)
    prose_chunk = (
        "The learners in the treatment group improved their writing scores substantially over "
        "the semester, and the qualitative interviews echoed this pattern across every cohort "
        "we followed. Scores were significantly higher for the experimental group than for "
        "the control group after ten weeks of instruction, and the improvement held in the "
        "delayed post-test administered one month later for every participant who completed "
        "both sessions of the study (Smith, 2019). Teachers reported that the intervention "
        "was easy to integrate into their existing lesson plans without extra preparation."
    )
    assert not bh.is_reference_or_frontmatter_chunk(prose_chunk)
    assert not bh.is_reference_or_frontmatter_chunk("Too short to judge.")


def test_candidates_for_paper_drops_dense_reference_chunks_and_records_the_count():
    reference_chunk_text = (
        "Fredricks, J. A., Blumenfeld, P. C., & Paris, A. H. (2004). School engagement: "
        "Potential of the concept, state of the evidence. Review of Educational Research, "
        "74(1), 59-109. Rosen, J., & Wedin, A. (2015). Klassrumsinteraktion och "
        "flerspråkighet: Ett kritiskt perspektiv. Retrieved from https://example.org/x.pdf "
        "pp. 12-30."
    )
    chunks = [
        {"section": "introduction", "text": "Short intro. Nothing of note here today."},
        {"section": "discussion", "text": reference_chunk_text},  # mislabelled reference list
        {"section": "results", "text": (
            "The learners in the treatment group improved their writing scores substantially "
            "over the semester. Scores were significantly higher for the experimental group "
            "than for the control group after ten weeks of instruction."
        )},
    ]
    dropped: list[int] = []
    cands = bh.candidates_for_paper(chunks, dropped_chunks_out=dropped)
    assert dropped == [1]
    assert cands and all(c["chunk_index"] != 1 for c in cands)
    assert any(c["chunk_index"] == 2 for c in cands)


def test_candidates_for_paper_drops_chunks_from_a_references_heading_onward():
    chunks = [
        {"section": "results", "text": (
            "The learners in the treatment group improved their writing scores substantially "
            "over the semester. Scores were significantly higher for the experimental group "
            "than for the control group after ten weeks of instruction."
        )},
        {"section": "references", "text": "References"},
        {"section": "references", "text": (
            "Smith, J. (2019). A study of learners. Journal of Applied Linguistics, 10(2), "
            "1-20."
        )},
    ]
    dropped: list[int] = []
    cands = bh.candidates_for_paper(chunks, dropped_chunks_out=dropped)
    assert dropped == [1, 2]
    assert cands and all(c["chunk_index"] == 0 for c in cands)


def test_build_items_from_cache_records_dropped_reference_chunk_counts(tmp_path: Path):
    make_cache(tmp_path / "cache")
    _items, info = bh.build_items_from_cache(tmp_path / "cache", seed=20260902)
    assert info["dropped_reference_chunks"] == {doi: 0 for doi in DOIS}


# ---------------------------------------------------------------- (3) candidate-sentence hygiene


def test_is_clean_sentence_rejects_soft_hyphen_artefacts():
    assert not bh.is_clean_sentence(
        "However, co-occurrence anal­ yses revealed that even unnecessary changes led "
        "to improvement of the text overall."
    )
    assert not bh.is_clean_sentence(
        "The presenta­ tion-assimilation-discussion model was used for instruction."
    )
    assert bh.is_clean_sentence(
        "The presentation-assimilation-discussion model was used for instruction throughout."
    )


def test_is_clean_sentence_rejects_bracketed_translation_titles():
    assert not bh.is_clean_sentence(
        "Klassrumsinteraktion och flerspråkighet: Ett kritiskt perspektiv "
        "[Classroom interaction and multilingualism: A critical perspective]."
    )
    assert bh.is_clean_sentence(
        "The learners reported higher enjoyment overall in the treatment group this term."
    )


def test_is_clean_sentence_rejects_repeated_numbered_heading_labels():
    assert not bh.is_clean_sentence(
        "Factor 3: Activity-induced boredom Factor 3 explained 12% of the variance and nine "
        "participants were significantly associated with it."
    )
    assert bh.is_clean_sentence(
        "The third factor, activity-induced boredom, explained 12% of the variance overall."
    )
    # a genuine mention of a numbered group is not rejected
    assert bh.is_clean_sentence(
        "Group 2 participants reported higher enjoyment than Group 1 across every session."
    )


def test_is_clean_sentence_rejects_attribution_with_three_or_more_named_authors():
    assert not bh.is_clean_sentence(
        "This is also expressed in the study of Arias, Maturana and Restrepo (2012), whose "
        "studies show conceptual coherence in the academic discourse of participants."
    )
    assert not bh.is_clean_sentence(
        "This point is echoed in the work of Smith, Jones, Patel and Diaz (2020), which "
        "reframes the debate around classroom translanguaging."
    )
    assert bh.is_clean_sentence(
        "The study of teacher agency conducted here shows a clear pattern across cohorts."
    )


# ---------------------------------------------------------------- (4) alteration grammar guards


def test_alter_number_never_touches_a_cross_reference_label():
    assert bh._alter_number("Excerpt 2 below shows the prompt used in the study.") is None
    text, label = bh._alter_number("Excerpt 2 shows that scores rose by 15% overall.")
    assert text.startswith("Excerpt 2 shows that scores rose by") and "30%" in text
    assert label == "numeric:15->30"
    assert bh._alter_number("As shown in Table 3, scores improved across cohorts.") is None
    assert bh._alter_number("See Figure 4 for the full distribution of scores.") is None
    assert bh._alter_number("Study 2 replicated the effect with a new sample of teachers.") is None


def test_alter_number_never_touches_a_page_number_a_sample_size_marker_or_a_cefr_code():
    """Deliverable 6: a page number and a sample-size marker inside a citation are
    identifiers, not quantities the source sentence itself asserts; a CEFR-style code
    ("B2"/"C1"/"L2") is already outside NUMBER_RE's own reach, since its digit is glued
    directly to a preceding letter."""
    assert bh._alter_number(
        "Feedback was discussed in detail (Dewaele, 2014, p. 45)."
    ) is None
    assert bh._alter_number(
        "This finding echoes earlier work (Smith, 2019, pp. 12-14)."
    ) is None
    assert bh._alter_number(
        "The effect was robust across the sample (N = 45)."
    ) is None
    assert bh._alter_number(
        "Only a few participants noticed the change (n = 6)."
    ) is None
    assert not bh.NUMBER_RE.search("Learners at B2 level improved their scores.")
    assert not bh.NUMBER_RE.search("The L2 learners outperformed the L1 comparison group.")
    # an ordinary quantity in the same sentence, outside the citation, is unaffected
    text, label = bh._alter_number(
        "24 students improved their scores (Smith, 2019, p. 45)."
    )
    assert text.startswith("48 students") and label == "numeric:24->48"


def test_alter_direction_never_creates_a_duplicated_or_alternative_pair():
    assert bh._alter_direction(
        "Participants rated the item higher (or lower) during the second session."
    ) is None
    out = bh.alter(
        "If the statement had been rated significantly higher (or lower) in a factor "
        "analysis, the loading would have changed."
    )
    assert out is None or "lower (or lower)" not in out[0]
    assert out is None or "higher (or higher)" not in out[0]


def test_alter_direction_vetoes_the_whole_sentence_when_it_contains_a_paren_or_alternative():
    """The per-word guards above stop a flip landing on the word framing "(or lower)", but the
    loop then simply moves on to the next applicable DIRECTION_FLIPS pair and flips a
    *different*, unrelated word instead. This is the literal committed hss-altered-09 (v3
    test set) sentence, which survived as "rated significantly lower (or lower)" and, after
    the per-word guard alone, would still become "rated not significantly higher (or lower)"
    via the significantly->not significantly fallback. Only a sentence-level veto (any
    direction flip in a sentence matching ``\\(or\\s+\\w+\\)`` is rejected outright) closes
    this."""
    source_sentence = (
        "In choosing the statements representative of each factor, we checked, firstly, if "
        "the statement had been rated significantly higher (or lower) in a factor relative "
        "to other factors."
    )
    assert bh.alter(source_sentence) is None
    assert bh._alter_direction(source_sentence) is None


def test_alter_direction_never_double_negates_an_already_negated_significant():
    text = (
        "There is no significant gap in scores overall, but a significant pattern emerged "
        "in the qualitative data."
    )
    assert bh._alter_direction(text) is None


def test_alter_quantifier_never_flips_all_but_one_into_no_but_one():
    assert bh._alter_quantifier(
        "All but one of the motivational drivers are positive reinforcers based on strategy."
    ) is None
    # the ordinary quantifier flip is unaffected
    assert bh._alter_quantifier("All learners preferred written feedback this term.") == (
        "No learners preferred written feedback this term.", "quantifier_flip:all->no"
    )


def test_alter_quantifier_skips_flips_inside_a_negated_intensional_frame():
    text = (
        "As editors of this issue, our aim is not to create the illusion that all "
        "contributors share one unified theoretical position on the topic."
    )
    out = bh.alter(text)
    assert out is None or not out[1].startswith("quantifier_flip")
    assert bh._alter_quantifier(
        "Our aim is not to create the illusion that all contributors agree on everything."
    ) is None
    # unhedged, the same quantifier still flips normally
    assert bh._alter_quantifier("All contributors agree on the theoretical framework.") == (
        "No contributors agree on the theoretical framework.", "quantifier_flip:all->no"
    )


def test_alter_quantifier_never_flips_all_in_object_position_after_a_benefactive_preposition():
    """"for all learners" -> "for no learners" reads close to a contradiction in terms
    (developing a practice for no one is not a coherent claim to negate into), not a
    checkable negation the way subject-position "All learners preferred ..." -> "No learners
    preferred ..." is. This is the literal committed hss-altered-14 (v3 test set) original
    sentence."""
    source_sentence = (
        "These critical reflections on the positionality of languages and power differential "
        "serve to challenge hegemonic discourse and practices that discredit individuals of "
        "minority languages and illustrate the potential of developing inclusive practices "
        "for all learners."
    )
    assert bh._alter_quantifier(source_sentence) is None
    assert bh.alter(source_sentence) is None
    # every named benefactive/purposive preposition is guarded
    for prep in ("for", "to", "among", "with", "across"):
        assert bh._alter_quantifier(f"This applies {prep} all students in the cohort.") is None
    # subject position (nothing precedes "All") is unaffected
    assert bh._alter_quantifier("All students in the cohort benefited from the change.") == (
        "No students in the cohort benefited from the change.", "quantifier_flip:all->no"
    )


# ---------------------------------------------------------------- (5) normalised veto matching


def test_normalise_for_veto_strips_diacritics_and_folds_case_and_whitespace():
    assert bh.normalise_for_veto("Rosén, J., & Wedín, Å.") == (
        bh.normalise_for_veto("rosen, j.,   &  wedin, a.")
    )
    assert bh.normalise_for_veto("flerspråkighet") == bh.normalise_for_veto("flersprakighet")
    assert bh.normalise_for_veto("A  Study") != bh.normalise_for_veto("A Different Study")


def test_candidates_for_paper_veto_matches_despite_diacritic_differences():
    chunks = [{"section": "results", "text": (
        "The learners in the treatment group improved their scores significantly this term. "
        "Wedín and Rosén reported a similar pattern in an earlier survey of "
        "teachers."
    )}]
    veto = {"Wedin and Rosen reported a similar pattern in an earlier survey of teachers."}
    cands = bh.candidates_for_paper(chunks, veto=veto)
    texts = [c["sentence"] for c in cands]
    assert texts and not any("report" in t.lower() for t in texts)


def test_find_veto_near_misses_flags_a_near_duplicate_that_did_not_exact_match():
    veto_raw = {"Wedin and Rosen reported a similar pattern in an earlier study of teachers."}
    candidates_by_doi = {
        "10.1/x": [{
            "sentence": (
                "Wedin and Rosen reported a similar pattern in an earlier study of the "
                "teachers."
            ),
            "chunk_index": 0, "section": "results", "score": 1.0,
        }],
    }
    misses = bh.find_veto_near_misses(veto_raw, candidates_by_doi)
    assert len(misses) == 1 and misses[0]["veto"] == next(iter(veto_raw))
    with pytest.raises(SystemExit):
        bh.raise_on_veto_near_misses(misses)


def test_find_veto_near_misses_is_silent_when_a_veto_entry_does_not_apply_to_this_cache():
    veto_raw = {"An entirely unrelated sentence about a different paper's own cohort design."}
    candidates_by_doi = {
        "10.1/x": [{
            "sentence": "The learners improved their scores significantly this term overall.",
            "chunk_index": 0, "section": "results", "score": 1.0,
        }],
    }
    assert bh.find_veto_near_misses(veto_raw, candidates_by_doi) == []
    bh.raise_on_veto_near_misses([])  # no-op, does not raise


def test_parse_args_veto_strict_defaults_on_and_can_be_disabled():
    assert bh.parse_args([]).veto_strict is True
    assert bh.parse_args(["--no-veto-strict"]).veto_strict is False


# ---------------------------------------------------------------- (6) HTTP timeouts / budget


def test_list_candidates_uses_explicit_connect_and_read_timeouts(monkeypatch):
    import httpx

    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"results": []}

    class FakeClient:
        def __init__(self, *a, **kw):
            captured["timeout"] = kw.get("timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **kw):
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    asyncio.run(bh.list_candidates({"name": "X", "issn": "1"}, "x@y.z"))
    timeout = captured["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == bh.CONNECT_TIMEOUT_S
    assert timeout.read == bh.READ_TIMEOUT_S


def test_acquire_sources_skips_a_candidate_that_exceeds_its_wall_clock_budget(
    tmp_path: Path, monkeypatch,
):
    """No real network and no backend import: ``load_pipeline`` is monkeypatched away, so this
    only exercises ``acquire_sources``'s own timeout wrapping."""

    async def fake_list_candidates(journal, mailto, **kw):
        return [{"doi": f"10.9999/{journal['issn']}-slow", "journal": journal["name"],
                 "issn": journal["issn"], "selected_from": "https://x", "title": "T",
                 "publication_year": 2021, "id": None, "authorships": []}]

    async def slow_acquire_one(work, pipeline, *, email, min_chars, min_chunks):
        await asyncio.sleep(1.0)
        return {"doi": work["doi"]}, {"doi": work["doi"], "chunks": []}, None

    monkeypatch.setattr(bh, "list_candidates", fake_list_candidates)
    monkeypatch.setattr(bh, "acquire_one", slow_acquire_one)
    monkeypatch.setattr(bh, "load_pipeline", lambda: {})

    attempts: list[dict] = []
    sources = asyncio.run(bh.acquire_sources(
        sources_path=tmp_path / "sources.json", cache_dir=tmp_path / "cache",
        n_sources=1, min_chars=1, min_chunks=1, per_journal=1, mailto="x@y.z",
        max_failures=1, attempts_out=attempts, candidate_timeout_s=0.05,
    ))
    assert sources == []
    assert attempts
    assert any(
        a["outcome"] == "dropped" and a["reason"] == "candidate_wall_clock_budget_exceeded"
        for a in attempts
    )


def test_probe_journals_skips_a_candidate_that_exceeds_its_wall_clock_budget(monkeypatch):
    """``--probe-journals`` takes each journal's first untested OA candidate through the
    production pipeline. Like ``acquire_sources``, it wraps its own ``acquire_one`` call in a
    wall-clock budget, so the CLI's ``--candidate-timeout-s`` flag reaches this path too."""

    async def fake_list_candidates(journal, mailto, **kw):
        return [{"doi": f"10.9999/{journal['issn']}-slow", "journal": journal["name"],
                 "issn": journal["issn"], "selected_from": "https://x", "title": "T",
                 "publication_year": 2021, "id": None, "authorships": []}]

    async def slow_acquire_one(work, pipeline, *, email, min_chars, min_chunks):
        await asyncio.sleep(1.0)
        return {"doi": work["doi"]}, {"doi": work["doi"], "chunks": []}, None

    monkeypatch.setattr(bh, "list_candidates", fake_list_candidates)
    monkeypatch.setattr(bh, "acquire_one", slow_acquire_one)
    monkeypatch.setattr(bh, "load_pipeline", lambda: {})

    rows = asyncio.run(bh.probe_journals("x@y.z", candidate_timeout_s=0.05))
    assert rows and len(rows) == len(bh.JOURNALS)
    assert all(
        r["outcome"] == "probe_failed" and r["reason"] == "candidate_wall_clock_budget_exceeded"
        for r in rows
    )


def test_probe_journals_cli_passes_candidate_timeout_s_through(tmp_path: Path, monkeypatch):
    captured: dict[str, object] = {}

    async def fake_probe_journals(mailto, **kw):
        captured.update(kw)
        return []

    monkeypatch.setattr(bh, "probe_journals", fake_probe_journals)
    monkeypatch.setattr(bh, "load_dotenv_values", lambda path: {"OPENALEX_EMAIL": "x@y.z"})
    probe_out = tmp_path / "probe.json"
    rc = bh.main([
        "--probe-journals", "--probe-out", str(probe_out), "--candidate-timeout-s", "7",
    ])
    assert rc == 0
    assert captured.get("candidate_timeout_s") == 7.0


# ---------------------------------------------------------------- byte-identical frozen rebuilds


def _rows_without_built_at(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r.pop("built_at", None)
    return rows


def test_frozen_hss_claims_rebuild_byte_identically_with_skip_fetch(tmp_path: Path):
    """Rebuilding this already-frozen 30-item test set from its committed cache with
    --skip-fetch must reproduce the committed jsonl file item for item (every field but the
    per-run ``built_at`` timestamp, which is always "now").

    ``hss_dev_claims.jsonl`` is *not* checked here even though its own cache
    (``evaluation/claims/data/hss_dev_fulltext``, gitignored) is complete for all ten of its
    source DOIs: a fresh rebuild differs from the committed file in 6 of its 30 rows, because
    ``hss_veto.json`` has grown from 10 entries (recorded in ``hss_dev_claims.build.json``'s
    own ``built_at``-time state) to its current 33 since that file was built, not because of
    anything the builder itself changes. See
    ``test_dev_claims_rebuild_is_stable_against_a_recorded_digest`` below for the dev set's own
    regression guard, pinned against the current veto file rather than the (now stale)
    committed jsonl.
    """
    skip_unless_cache_dir(bh.DEFAULT_CACHE_DIR)
    out = tmp_path / "hss_claims.jsonl"
    rc = bh.main([
        "--skip-fetch", "--cache-dir", str(bh.DEFAULT_CACHE_DIR), "--out", str(out),
        "--seed", "20260902", "--sources", str(bh.HERE / "hss_sources.json"),
        "--veto", str(bh.DEFAULT_VETO),
    ])
    assert rc == 0
    assert _rows_without_built_at(out) == _rows_without_built_at(bh.DEFAULT_OUT)


def test_dev_claims_rebuild_is_stable_against_a_recorded_digest(tmp_path: Path):
    """Regression guard for the dev set: the committed ``hss_dev_claims.jsonl`` predates most
    of the entries now in ``hss_veto.json`` (it was built when the veto file carried only 10
    entries, and the file has grown since), so it is not byte-identical to a fresh rebuild
    through no fault of the builder's guards; comparing against it the way the frozen test
    set is compared above would conflate that historical drift with an actual regression.
    Instead this pins a sha256 digest of a rebuild against the *current* ``hss_veto.json``
    and the current guards, so a future change to the builder or the veto list that silently
    shifts the dev set is caught here without needing the committed jsonl to be re-frozen
    every time the veto list legitimately grows.
    """
    skip_unless_cache_dir(bh.HERE / "data" / "hss_dev_fulltext")
    out = tmp_path / "hss_dev_claims.jsonl"
    rc = bh.main([
        "--skip-fetch", "--cache-dir", str(bh.HERE / "data" / "hss_dev_fulltext"),
        "--out", str(out), "--seed", "20260906",
        "--sources", str(bh.HERE / "hss_dev_sources.json"),
        "--altered-expected", "strict",
        "--exclude-sources", str(bh.HERE / "hss_sources.json"),
        "--veto", str(bh.DEFAULT_VETO),
    ])
    assert rc == 0
    rows = _rows_without_built_at(out)
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()
    assert digest == "f2a79164ea45fa50079786dcafc3bdc5dff1ca9cb2e959d2d50ff214b0976285"


# --------------------------------------------------------------------------------------
# Every literal sentence below was read directly out of the committed hss_test_v3_claims.jsonl
# / hss_dev_band2_claims.jsonl for the item id named in each test's docstring.

ALTERED_18_R4_ORIGINAL = (
    "Pontier & Tian (in press) found that a different set of graduate students in the same "
    "TESOL program who all identified as bi/multilingual had difficulty understanding "
    "translanguaging as a form of social justice despite listing several advantages of "
    "bilingualism."
)
ALTERED_13_R4_ORIGINAL = (
    "The lower ratings given in session 4 tie in with a notable increase of weak points "
    "brought forward."
)
ALTERED_12_R4_ORIGINAL = (
    "While I always ensured to review and validate the suggestions, ChatGPT’s assistance "
    "saved me time and allowed me to focus more on providing personalised feedback to each "
    "student."
)
# The chunk this sentence sits in (10.29140/tltl.v6n1.1142), trimmed to the relevant span: a
# colon-introduced interview quotation with no quotation marks at all.
ALTERED_12_R4_CHUNK = (
    "The sections presented, the incorporation of emoticons, and the use of clear and "
    "appropriate language for the age and level of the students should be highlighted. "
    "The teacher pointed out the following in the interview: I used ChatGPT outputs as a "
    "reference to provide targeted feedback to my students. " + ALTERED_12_R4_ORIGINAL
    + " Reliability of Outputs The quality of the information offered by ChatGPT should also "
    "be considered."
)
PARAPHRASE_05_R4_ORIGINAL = (
    "Sentence and Text Editing – ChatGPT helped students by suggesting improvements to "
    "their sentences and compositions."
)
WRONG_PAPER_04_R4_ORIGINAL = (
    "In our own earlier studies in different SFI classrooms1, we found that classroom time "
    "was dominated by whole class interaction, sandwiched between students’ individual "
    "work with exercises (Norlund Shaswar, 2014; Wedin, Rosén, & Hennius, 2018)."
)
ALTERED_01_BAND2_R4_ORIGINAL = (
    "Whether in a colonial or neo-colonial context where unequal power relations always "
    "exist, racial inequality in English language education has never been resolved."
)
ALTERED_04_BAND2_R4_ORIGINAL = (
    "In the probability response format, each respondent provides an estimate of the "
    "percentage of time (from 0% to 100%) in which he/she would be willing to talk in each "
    "situation named in each item."
)
ALTERED_05_BAND2_R4_PREVIOUS = (
    "Al Khalil (2016) used a motivational thermometer to measure learners’ task "
    "motivation."
)
ALTERED_05_BAND2_R4_ORIGINAL = (
    "Participants were 44 adult learners of FL Arabic in the United States, and completed "
    "six oral interactive tasks with a native speaker of Arabic."
)


def test_alter_quantifier_covers_all_of_and_every_by_alias_not_just_the_literal_all():
    """Both literal corpus sentences require the guard to cover "all of" and "every" as well
    as the literal token "all": all three share the identical benefactive/purposive
    object-position construction."""
    assert bh._alter_quantifier(
        "rather than trying to cater to every single variety in the class"
    ) is None
    assert bh._alter_quantifier(
        "I recommend this book to all hematologists and historians and to every beginner of "
        "science research."
    ) is None
    assert bh._alter_quantifier(
        "for all of the students, this was their first experience with a portfolio."
    ) is None
    assert bh._alter_quantifier(
        "the questionnaire was sent to all of the teachers in the district."
    ) is None
    # subject-position "every"/"all of" (no preposition before it) is unaffected
    assert bh._alter_quantifier("Every teacher agreed with the plan.") == (
        "No teacher agreed with the plan.", "quantifier_flip:every->no"
    )
    text, alteration = bh._alter_quantifier(
        "All of the teachers agreed with the plan."
    )
    assert text.startswith("None of the teachers")
    assert alteration == "quantifier_flip:all of->none of"


def test_alter_quantifier_never_flips_a_floating_quantifier_before_its_own_finite_verb():
    """hss-altered-18, second defect: "who all identified" -> "who no identified" is not a
    sentence of English, regardless of which verb follows."""
    assert bh._alter_quantifier(
        "a different set of graduate students in the same TESOL program who all identified "
        "as bi/multilingual had difficulty understanding translanguaging"
    ) is None
    assert bh.alter(ALTERED_18_R4_ORIGINAL) is None
    # the determiner reading (a noun, not a verb, immediately follows the quantifier) is
    # unaffected
    assert bh._alter_quantifier("All teachers reported higher engagement this term.") == (
        "No teachers reported higher engagement this term.", "quantifier_flip:all->no"
    )


def test_alter_quantifier_never_double_negates_a_sentence_that_already_carries_the_destination():
    """hss-altered-01: flipping "always" to "never" produces two "never"s governing two
    different clauses in the same sentence, a self-contradiction. A control with no
    pre-existing negation still flips."""
    assert bh._alter_quantifier(ALTERED_01_BAND2_R4_ORIGINAL) is None
    assert bh.alter(ALTERED_01_BAND2_R4_ORIGINAL) is None
    assert bh._alter_quantifier(
        "Learners always preferred written feedback over oral feedback."
    ) == ("Learners never preferred written feedback over oral feedback.",
          "quantifier_flip:always->never")
    # a non-destination flip ("most"->"few") is never blocked by this check even when the
    # sentence happens to contain "no"/"never" elsewhere
    text, alteration = bh._alter_quantifier(
        "Most students never missed a single class session this term."
    )
    assert text.startswith("Few students") and alteration == "quantifier_flip:most->few"


def test_attribution_re_accepts_forthcoming_publication_forms_in_place_of_a_year():
    """hss-altered-18: "Pontier & Tian (in press) found ..." must match ATTRIBUTION_RE, which
    needs a branch for forms like "(in press)" that carry no four-digit year."""
    assert bh.is_clean_sentence(ALTERED_18_R4_ORIGINAL) is False
    assert not bh.is_clean_sentence("Smith and Jones (in press) found that feedback helps.")
    assert not bh.is_clean_sentence("Smith (n.d.) reported a large effect on motivation.")
    assert not bh.is_clean_sentence("Smith (forthcoming) argued that the effect is robust.")
    assert not bh.is_clean_sentence(
        "Smith (in preparation) showed that the intervention worked."
    )


def test_attribution_re_rejects_a_bare_pronoun_subject_with_a_reporting_verb():
    """"Drawing on a corpus of 30 studies published in English, they report on the benefits
    and challenges of plurilingual approaches in different educational settings and contexts"
    has an unresolvable pronoun subject. A pronoun subject with a non-reporting verb stays
    eligible, and "these/those/such authors" is covered the same way "the authors" already
    is."""
    assert not bh.is_clean_sentence(
        "Drawing on a corpus of 30 studies published in English, they report on the benefits "
        "and challenges of plurilingual approaches in different educational settings and "
        "contexts."
    )
    assert not bh.is_clean_sentence("He found that feedback improved accuracy over time.")
    assert not bh.is_clean_sentence("She argued that the intervention was effective overall.")
    assert bh.is_clean_sentence("They were asked to complete the questionnaire twice.")
    assert not bh.is_clean_sentence(
        "These authors found that motivation predicted persistence over time."
    )
    assert not bh.is_clean_sentence(
        "Those authors reported a large effect of feedback on accuracy."
    )


def test_alter_number_cross_reference_labels_cover_session_and_the_other_new_words():
    """hss-altered-13: Strobl et al. ran exactly four sessions; "session 4" -> "session 5"
    points at nothing. A duration phrased the other way round ("6 weeks") is unaffected
    because the label word only ever *follows* the guarded number there."""
    assert bh.alter(ALTERED_13_R4_ORIGINAL, allowed=["numeric"]) is None
    for label in ("session", "phase", "round", "stage", "wave", "cycle", "week", "day", "item",
                  "task", "condition", "group"):
        # CROSS_REFERENCE_LABEL_RE is matched against the text *before* the number, the way
        # _alter_number itself uses it -- not against the whole "label N" string.
        assert bh.CROSS_REFERENCE_LABEL_RE.search(f"described in {label} ")
    assert bh._alter_number("The study enrolled 24 students over 6 weeks.") == (
        "The study enrolled 48 students over 6 weeks.", "numeric:24->48"
    )


def test_is_clean_sentence_rejects_a_list_label_glued_to_its_description_by_a_dash():
    """hss-paraphrase-05: "Sentence and Text Editing - ChatGPT helped ..." was list item 3 of
    an enumerated inventory; its numbering was stripped, welding the category label to its
    description by an en dash."""
    assert not bh.is_clean_sentence(PARAPHRASE_05_R4_ORIGINAL)
    assert bh.LIST_LABEL_DASH_RE.search(PARAPHRASE_05_R4_ORIGINAL)
    # an ordinary Title-Case-opening sentence, and a genuine hyphenated compound, are unaffected
    assert bh.is_clean_sentence(
        "New Zealand teachers reported higher enjoyment overall this term."
    )
    assert not bh.LIST_LABEL_DASH_RE.search("State-of-the-art methods were used in this study.")
    assert not bh.LIST_LABEL_DASH_RE.search("COVID-19 disrupted classroom instruction broadly.")
    # A genuine hyphenated compound whose second element is itself capitalised (a
    # bilingual/national-population description) glues the dash directly onto both
    # neighbouring words with no surrounding whitespace, unlike the spaced dash of a list
    # label glued to its description, and must stay eligible.
    anglo_canadian = (
        "Ten Anglo-Canadian students completed oral tasks in their French L2 immersion "
        "classroom this term."
    )
    assert not bh.LIST_LABEL_DASH_RE.search(anglo_canadian)
    assert bh.is_clean_sentence(anglo_canadian)


def test_is_clean_sentence_rejects_a_footnote_digit_glued_directly_to_a_word():
    """hss-wrong_paper-04: "different SFI classrooms1, we found ..." has a superscript
    footnote digit flattened onto the word with no intervening period, unlike
    FOOTNOTE_MARK_RE's "the UK.2" shape. An "L2"/"T2"-style token is unaffected."""
    assert not bh.is_clean_sentence(WRONG_PAPER_04_R4_ORIGINAL)
    assert bh.FOOTNOTE_DIGIT_GLUED_RE.search(WRONG_PAPER_04_R4_ORIGINAL)
    assert bh.is_clean_sentence("The L2 learners in this cohort improved their scores overall.")
    assert bh.is_clean_sentence("Both schools are top performing in the UK this term overall.")


def test_is_quoted_in_chunk_detects_a_colon_introduced_interview_quotation():
    """hss-altered-12: a block quotation introduced by "... in the interview:" with no
    quotation marks at all is a study participant's own words, not the source paper's own
    claim. An ordinary sentence well outside the lookback window, or one with no
    reported-speech colon at all, stays eligible."""
    assert bh.is_quoted_in_chunk(ALTERED_12_R4_ORIGINAL, ALTERED_12_R4_CHUNK)
    far_chunk = (
        "in the interview: I used ChatGPT outputs as a reference. " + ("Filler text here. " * 60)
        + ALTERED_12_R4_ORIGINAL
    )
    assert not bh.is_quoted_in_chunk(ALTERED_12_R4_ORIGINAL, far_chunk)
    assert not bh.is_quoted_in_chunk(
        ALTERED_12_R4_ORIGINAL,
        "The teacher summarised the approach as follows. " + ALTERED_12_R4_ORIGINAL,
    )


def test_is_clean_sentence_rejects_a_sample_description_elided_into_the_previous_sentence():
    """hss-altered-05: "Participants were 44 adult learners of FL Arabic ..." describes Al
    Khalil (2016), named only in the sentence before it. Without a previous sentence, or with
    a previous sentence that names no other study, the opener stays eligible."""
    assert not bh.is_clean_sentence(
        ALTERED_05_BAND2_R4_ORIGINAL, previous_sentence=ALTERED_05_BAND2_R4_PREVIOUS,
    )
    assert bh.is_clean_sentence(ALTERED_05_BAND2_R4_ORIGINAL)
    assert bh.is_clean_sentence(
        ALTERED_05_BAND2_R4_ORIGINAL,
        previous_sentence="The tasks were administered over two separate sessions.",
    )


def test_previous_sentence_in_chunk_finds_the_immediately_preceding_sentence():
    chunk = (
        f"{ALTERED_05_BAND2_R4_PREVIOUS} {ALTERED_05_BAND2_R4_ORIGINAL} Their task motivation "
        "was measured after each task."
    )
    assert bh.previous_sentence_in_chunk(chunk, ALTERED_05_BAND2_R4_ORIGINAL) == (
        ALTERED_05_BAND2_R4_PREVIOUS
    )
    assert bh.previous_sentence_in_chunk(chunk, ALTERED_05_BAND2_R4_PREVIOUS) is None
    assert bh.previous_sentence_in_chunk(chunk, "Not present at all in this chunk.") is None


def test_alter_number_nonround_floors_a_change_the_percent_cap_would_otherwise_swallow():
    """hss-altered-04: "100%" -> "99%" is a 1% relative change, entirely an artefact of the
    percent cap rather than of the chosen factor. The next non-round factor that clears the
    floor is used instead. A legitimately large relative change (e.g. the existing 75% -> 99%
    case) is unaffected."""
    text, label = bh._alter_number_nonround(ALTERED_04_BAND2_R4_ORIGINAL)
    assert label != "numeric:100->99"
    assert label.startswith("numeric:100->")
    new_value = float(label.split("->")[1])
    assert abs(new_value - 100.0) / 100.0 >= bh.MIN_RELATIVE_CHANGE
    text2, label2 = bh._alter_number_nonround("Exactly 75% of the learners passed the test.")
    assert "99%" in text2 and label2 == "numeric:75->99"


def test_over_specify_never_applies_to_a_sentence_that_reports_no_finding():
    """hss-over_specified-07: "we checked ... if the statement had been rated significantly
    higher (or lower) ..." describes an analysis procedure, not a finding, and carries no
    finding verb and no statistic."""
    procedure_sentence = (
        "In choosing the statements representative of each factor, we checked, firstly, if "
        "the statement had been rated significantly higher in a factor relative to other "
        "factors."
    )
    assert bh._reports_empirical_finding(procedure_sentence) is False
    assert bh._over_specify(
        procedure_sentence, paper_text="Nothing relevant appears in this paper at all.",
    ) is None
    # a definition, an aim and a claim about another study are not findings either
    assert bh._reports_empirical_finding(
        "Willingness to communicate is defined as the intention to initiate discourse."
    ) is False
    assert bh._reports_empirical_finding(
        "This study aims to examine the role of enjoyment in second language acquisition."
    ) is False
    # a genuine finding, including one that reports a bare statistic with no listed verb, is
    # still eligible
    assert bh._reports_empirical_finding("Scores improved significantly for the group.") is True
    assert bh._reports_empirical_finding("Only 18% of participants used the tool daily.") is True


def test_over_specify_never_appends_a_modifier_duplicating_a_detail_already_in_the_sentence():
    """The qualifier must not duplicate a detail the sentence already states (deliverable 4):
    a modifier whose content words already appear in the sentence itself is rejected the same
    way one whose words appear in the cited paper is."""
    src = "Scores improved significantly among heritage speakers in the treatment group."
    paper_text = "Nothing relevant appears in this paper about the topic at hand."
    # index=10 starts the round-robin at OVER_SPECIFIED_MODIFIERS[10] == "among heritage
    # speakers", the exact modifier the sentence already states: this is the only index that
    # actually exercises the sentence-token half of the duplicate check (with the default
    # index=0 the assertion holds trivially on both the parent and an earlier builder, since
    # the first modifier tried, "among adult learners", is accepted either way).
    out = bh._over_specify(src, paper_text=paper_text, index=10)
    assert out is not None
    _claim, modifier = out
    assert modifier != "among heritage speakers"


# ---------------------------------------------------------------- Running headers

# Both sentences below were read directly out of the committed caches
# (data/hss_dev_fulltext/10-55593_ej-26103a9.json and .../10-55593_ej-28109a8.json) by running
# select_candidate_sentences over their non-reference chunks. LIST_LABEL_DASH_RE's dash
# spacing (``\s*`` vs ``\s+``) closes the "Anglo-Canadian" false positive but does not catch a
# PDF running header, because the loose spacing only ever screened these out as an accidental
# side effect of matching the journal abbreviation's own internal hyphen ("TESL-EJ" alone
# satisfies "Title-Case word, dash, capital word"), not because it targets running headers at
# all.
RUNNING_HEADER_R5_ORIGINAL_1 = (
    "TESL-EJ 26.3, November 2022 Fang & Xu 12 To Cite this Article Fang, F. & Xu, Y."
)
RUNNING_HEADER_R5_ORIGINAL_2 = (
    "TESL-EJ 28.1, May 2024 Van Horn 14 Several groups of students, with a focus on improving "
    "pronunciation, employed ChatGPT for voice-to-text conversations."
)


def test_is_clean_sentence_rejects_a_journal_volume_running_header():
    """These two sentences are exactly what a rebuild of the hss-dev-band2 recipe (seed
    20260908, hss_dev_sources.json cache) surfaces at the hss-altered-01 and hss-altered-02
    fill positions once LIST_LABEL_DASH_RE no longer incidentally rejects them: a page
    running header, not a claim, its only numeral a volume/issue pair or a page number the
    sentence itself asserts nothing about."""
    assert not bh.is_clean_sentence(RUNNING_HEADER_R5_ORIGINAL_1)
    assert not bh.is_clean_sentence(RUNNING_HEADER_R5_ORIGINAL_2)
    assert bh.RUNNING_HEADER_VOLUME_RE.search(RUNNING_HEADER_R5_ORIGINAL_1)
    assert bh.RUNNING_HEADER_VOLUME_RE.search(RUNNING_HEADER_R5_ORIGINAL_2)
    # LIST_LABEL_DASH_RE itself does not fire on either (confirming the two guards are
    # independent, not a re-widening of the dash pattern's own spacing)
    assert not bh.LIST_LABEL_DASH_RE.search(RUNNING_HEADER_R5_ORIGINAL_1)
    assert not bh.LIST_LABEL_DASH_RE.search(RUNNING_HEADER_R5_ORIGINAL_2)
    # a genuine hyphenated compound (the fix that keeps "Anglo-Canadian" eligible, which this
    # guard must not undo) and an ordinary sentence naming the journal without a
    # volume/issue/date stay eligible
    anglo_canadian = (
        "Ten Anglo-Canadian students completed oral tasks in their French L2 immersion "
        "classroom this term."
    )
    assert bh.is_clean_sentence(anglo_canadian)
    assert not bh.RUNNING_HEADER_VOLUME_RE.search(anglo_canadian)
    assert bh.is_clean_sentence(
        "TESL-EJ has published numerous articles about English as a lingua franca overall."
    )


def test_committed_builds_do_not_carry_the_round5_running_headers():
    """Neither committed build (hss_test_v3_claims.jsonl, hss_dev_band2_claims.jsonl) contains
    either running header as an original_sentence: the committed files are frozen
    (built before the LIST_LABEL_DASH_RE whitespace tightening that let these two sentences
    through), so this documents that the frozen data stays valid: the new guard changes what a
    *future* rebuild selects, not any file already committed. Neither sentence is in
    hss_veto.json either (confirming the frozen files avoided them for an unrelated reason:
    they were simply never reached by the earlier build's own round-robin fill order, not
    because an earlier pass had already vetoed them)."""
    veto = bh.load_veto(bh.DEFAULT_VETO)
    test_v3_rows = [
        json.loads(x)
        for x in (bh.HERE / "hss_test_v3_claims.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    band2_rows = [
        json.loads(x)
        for x in (bh.HERE / "hss_dev_band2_claims.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    originals = {r["original_sentence"] for r in test_v3_rows + band2_rows}
    assert RUNNING_HEADER_R5_ORIGINAL_1 not in originals
    assert RUNNING_HEADER_R5_ORIGINAL_2 not in originals
    assert RUNNING_HEADER_R5_ORIGINAL_1 not in veto
    assert RUNNING_HEADER_R5_ORIGINAL_2 not in veto


def test_dev_band2_rebuild_never_selects_a_running_header_as_a_claim(tmp_path: Path):
    """Regression guard: rebuilding the exact hss-dev-band2 recipe recorded in the committed
    hss_dev_band2_claims.build.json (seed 20260908, quotas altered=10/over_specified=5,
    alteration-quotas numeric=5/quantifier_flip=5, altered-expected strict,
    --identity-check-cache, --overlap-with hss_dev_claims.jsonl) against the real committed
    hss_dev_fulltext cache and the current hss_veto.json never selects either running
    header, confirmed by a direct rerun rather than by reading the already-frozen committed
    jsonl checked above.

    Diffed against a rerun without the running-header guard under identical arguments, four
    rows change, not two. hss-altered-01 and hss-altered-02 move from
    RUNNING_HEADER_R5_ORIGINAL_1/2 to a real claim sentence each, which frees two
    quantifier_flip fill slots; those two freed slots are refilled by the two sentences that
    had occupied hss-altered-07 and hss-altered-08 without the guard, so those two rows also
    change, each to a different sentence under a different alteration operator. This four-row
    reshuffle, not an unrelated seed or veto-file difference, is what confirms the guard is
    what fixes the two running headers."""
    skip_unless_cache_dir(bh.HERE / "data" / "hss_dev_fulltext")
    out = tmp_path / "hss_dev_band2_claims.jsonl"
    rc = bh.main([
        "--skip-fetch", "--cache-dir", str(bh.HERE / "data" / "hss_dev_fulltext"),
        "--out", str(out), "--seed", "20260908",
        "--sources", str(bh.HERE / "hss_dev_sources.json"),
        "--altered-expected", "strict",
        "--quotas", "altered=10,over_specified=5",
        "--alteration-quotas", "numeric=5,quantifier_flip=5",
        "--identity-check-cache",
        "--overlap-with", str(bh.HERE / "hss_dev_claims.jsonl"),
        "--veto", str(bh.DEFAULT_VETO),
    ])
    assert rc == 0
    rows = _rows_without_built_at(out)
    originals = [r["original_sentence"] for r in rows]
    assert RUNNING_HEADER_R5_ORIGINAL_1 not in originals
    assert RUNNING_HEADER_R5_ORIGINAL_2 not in originals
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()
    assert digest == "2df67670e748127b7c8e6f1aad0e05476c6fe36984e5bd7edfa7f165bba436ac"
