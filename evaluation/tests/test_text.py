"""Quote fidelity, lexical overlap, sentence heuristics and SciFact gold mapping."""

from __future__ import annotations

from common import (
    max_sentence_jaccard,
    quote_is_verbatim,
    scifact_gold_for_doc,
    score_candidate_sentence,
    select_candidate_sentences,
    split_sentences,
    token_jaccard,
)

CHUNK = "We found that  reminders\nincreased hand hygiene compliance by 12%. Effects faded later."


def test_quote_is_verbatim_normalises_whitespace_only():
    assert quote_is_verbatim("reminders increased hand hygiene compliance", CHUNK)
    assert quote_is_verbatim("We found that reminders\n  increased", CHUNK)
    assert not quote_is_verbatim("Reminders increased hand hygiene", CHUNK)  # case differs
    assert quote_is_verbatim("REMINDERS increased", CHUNK, casefold=True)
    assert not quote_is_verbatim("reminders decreased", CHUNK)
    assert not quote_is_verbatim("", CHUNK) and not quote_is_verbatim(None, CHUNK)
    assert not quote_is_verbatim("x", "")


def test_token_jaccard():
    assert token_jaccard("a b c", "a b c") == 1.0
    assert token_jaccard("A, b!", "a b d") == 2 / 3
    assert token_jaccard("", "a") == 0.0
    sentences = ["unrelated", "hand hygiene compliance"]
    best, idx = max_sentence_jaccard("hand hygiene compliance", sentences)
    assert best == 1.0 and idx == 1
    assert max_sentence_jaccard("x", []) == (0.0, None)


def test_split_sentences():
    parts = split_sentences(CHUNK)
    assert len(parts) == 2 and parts[1] == "Effects faded later."
    assert split_sentences("") == []
    assert split_sentences("No split at 3.5 percent. Yes split here.") == [
        "No split at 3.5 percent.",
        "Yes split here.",
    ]


def test_score_candidate_sentence_rules():
    good = (
        "We found that participants in the reminder condition showed significantly "
        "higher compliance."
    )
    assert score_candidate_sentence(good) > 1.0
    assert score_candidate_sentence("Too short sentence here.") == 0.0
    assert score_candidate_sentence("As shown in Figure 2, " + good) == 0.0
    assert score_candidate_sentence("lowercase start " + good) == 0.0
    assert score_candidate_sentence(good[:-1] + " (see below.") == 0.0
    long = " ".join(["word"] * 50) + "."
    assert score_candidate_sentence("Word " + long) == 0.0


def test_select_candidate_sentences_is_deterministic_and_deduplicated():
    s1 = (
        "We found that participants in the reminder condition showed significantly "
        "higher compliance."
    )
    s2 = (
        "The study was conducted in three hospitals across the northern region during "
        "winter months."
    )
    chunks = [f"{s1} {s2}", f"{s2} {s1}"]
    out = select_candidate_sentences(chunks)
    # ranked by score then order; duplicates dropped
    assert [c.sentence for c in out] == [s1, s2]
    assert out[0].chunk_index == 0 and out[0].score > out[1].score
    assert select_candidate_sentences(chunks, limit=1)[0].sentence == s1


def test_scifact_gold_for_doc():
    claim = {
        "id": 3,
        "evidence": {"14717500": [{"sentences": [2], "label": "SUPPORT"}]},
        "cited_doc_ids": [14717500, 999],
    }
    assert scifact_gold_for_doc(claim, 14717500) == "SUPPORT"
    assert scifact_gold_for_doc(claim, "14717500") == "SUPPORT"
    assert scifact_gold_for_doc(claim, 999) == "NOT_ENOUGH_INFO"
    contra = {"evidence": {"1": [{"sentences": [0], "label": "CONTRADICT"}]}}
    assert scifact_gold_for_doc(contra, 1) == "CONTRADICT"
    assert scifact_gold_for_doc({"evidence": {}}, 1) == "NOT_ENOUGH_INFO"
