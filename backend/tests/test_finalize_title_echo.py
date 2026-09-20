"""Two companion fixes to `finalize_generated_section`:

* A body paragraph that echoes the section's own title is dropped whole and
  counted under ``sentences_removed_title_echo`` (`_is_title_echo`). Every heading
  the writer wrote inside the body -- whatever its own wording, wherever it sits --
  is dropped whole too, but under a separate counter (``headings_removed``): the
  section title is the only heading a saved section ever carries now, so there is
  no "heading still standing in the text" case left for the title-echo rule to
  compare a paragraph against.
* Fix B's own companion guard: a link whose normalised sentence equals no fragment
  norm of the paragraph it was bucketed into -- including one bucketed onto an
  out-of-range index -- is counted under ``links_unmatched_to_sentence``, so a link
  the sentence snap (`app.agents.citation_link_agent.validate_citation_link`) does not
  reach can never again be lost in silence, mislabelled as ordinary dangling residue.

No network, no database: pure-function tests over `finalize_generated_section`.
"""
from app.services.fulltext import finalize_generated_section


def _link(sentence, keys, citation_text="(X, 2020)", paragraph_index=0, **extra):
    return {
        "paragraph_index": paragraph_index,
        "sentence": sentence,
        "keys": keys,
        "citation_text": citation_text,
        **extra,
    }


# --------------------------------------------------------------------------------------
# sentences_removed_title_echo
# --------------------------------------------------------------------------------------


def test_a_heading_matching_the_section_title_is_dropped_whole():
    """Real shape: the model's own true opening
    paragraph fails verification and is removed, leaving a heading-shaped line
    restating the section's own external title as finalize's own first surviving
    block -- dropped whole, whatever its own wording, under ``headings_removed``,
    not the title-echo counter (which only ever applies to a body paragraph now)."""
    removed_opening = "The real opening paragraph turns out unsupported (Lee, 2021)."
    heading_line = "# Metalinguistic Coded Feedback in L2 Writing"
    text = f"{removed_opening}\n\n{heading_line}"
    links = [_link(removed_opening, ["lee_2021"], citation_text="(Lee, 2021)")]
    status = {(removed_opening, removed_opening, "lee_2021"): "unsupported"}

    result = finalize_generated_section(
        text, links, [], status,
        section_title="Metalinguistic codes as written corrective feedback",
    )
    assert result.text == ""
    assert result.stats["headings_removed"] == 1
    assert result.stats["sentences_removed_title_echo"] == 0


def test_a_sub_heading_is_dropped_whole_once_its_real_body_is_gone():
    """Observed model shape: the model's own sub-heading restates the
    section's title with a dropped qualifier and an added suffix; the paragraph that
    was its real opening body is removed by an earlier rule (here, simulated by an
    unverified citation), leaving the sub-heading as finalize's own first surviving
    block -- dropped whole, under ``headings_removed``, whatever its wording; the
    body paragraph that follows it is unaffected and still delivered."""
    removed_opening = "The real opening paragraph turns out unsupported (Lee, 2021)."
    heading_line = "## Engagement with Written Corrective Feedback in Second-Language Writing"
    body = (
        "Research on this topic has moved from whether feedback works to how "
        "learners engage with it."
    )
    text = f"{removed_opening}\n\n{heading_line}\n\n{body}"
    links = [_link(removed_opening, ["lee_2021"], citation_text="(Lee, 2021)")]
    status = {(removed_opening, removed_opening, "lee_2021"): "unsupported"}

    result = finalize_generated_section(
        text, links, [], status,
        section_title="Learner engagement with written corrective feedback",
    )

    assert heading_line.lstrip("# ") not in result.text
    assert body in result.text
    assert result.text == body
    assert result.stats["headings_removed"] == 1
    assert result.stats["sentences_removed_title_echo"] == 0


def test_no_section_title_and_no_heading_in_text_never_drops_anything():
    """``section_title`` defaults to ``None`` -- every existing caller of this
    function, and a caller with no title to check against -- in which case only a
    heading still standing IN the text itself would ever be checked, and this text
    has none at all. Written as a real, terminally-punctuated sentence, not a bare
    noun phrase -- the shape the heading-shape guard (a companion, independent fix)
    exists to catch is deliberately absent here."""
    text = "Metalinguistic coded feedback in L2 writing remains understudied."
    result = finalize_generated_section(text, [], [], {})
    assert result.text == text
    assert result.stats["sentences_removed_title_echo"] == 0


def test_a_genuine_short_framing_sentence_under_the_same_heading_survives():
    """A short, uncited framing sentence that merely shares the heading's own topic
    vocabulary -- not a restatement of it -- is not a title echo and is kept, exactly
    like any other framing sentence with no citation of its own. The heading itself
    is dropped whole regardless (``headings_removed``), never delivered."""
    heading_line = "## Learner engagement with written corrective feedback"
    framing = "Attention to feedback is unevenly distributed across learners."
    text = f"{heading_line}\n\n{framing}"
    uncited = [{"paragraph_index": 1, "sentence": framing, "tag": "framing"}]

    result = finalize_generated_section(text, [], uncited, {})

    assert framing in result.text
    assert result.text == framing
    assert result.stats["headings_removed"] == 1
    assert result.stats["sentences_removed_title_echo"] == 0


def test_a_heading_the_paragraph_echoes_is_still_reported_after_removal():
    """The surviving link/uncited paragraph_index remap runs correctly around a
    title-echo removal: a real, cited sentence in the paragraph AFTER the echoed one
    keeps its own link, renumbered onto the filtered block list. The echoing
    paragraph is written as a real, terminally-punctuated sentence, not a bare noun
    phrase, so this test exercises the title-echo removal specifically rather than
    the heading-shape guard (a companion, independent fix) that would otherwise
    also catch this same text."""
    heading_line = "## Learner engagement with written corrective feedback"
    echo = "Engagement with written corrective feedback shapes second-language writing."
    cited = "Zhang and Hyland (2018) found substantial variation in revision behaviour."
    text = f"{heading_line}\n\n{echo}\n\n{cited}"
    links = [_link(cited, ["zhang_2018"], citation_text="(2018)")]
    status = {(cited, cited, "zhang_2018"): "verified"}

    result = finalize_generated_section(
        text, links, [], status,
        section_title="Learner engagement with written corrective feedback",
    )

    assert echo not in result.text
    assert cited in result.text
    assert result.citation_links[0]["sentence"] == cited
    assert result.stats["sentences_removed_title_echo"] == 1


def test_orphaned_sub_headings_unrelated_to_the_title_are_dropped_whole():
    """The shape an earlier fix restored as a heading node, now dropped whole
    instead (scope broadened after that fix): the section's genuine opening
    paragraph (its one finding sentence, unsupported) is entirely removed, leaving
    the writer's own markdown sub-heading -- worded nothing like the section title,
    sharing no stem with it at all -- as `finalize_generated_section`'s own first
    surviving block, immediately followed by a cited paragraph, a second heading,
    and one more cited paragraph. Both sub-headings are dropped whole
    (``headings_removed == 2``), whatever they say; both cited paragraphs are kept,
    untouched, exactly as if the headings between them had never been there."""
    removed_finding = (
        "Automated written corrective feedback has become a significant "
        "component of second-language writing instruction (Lee, 2021)."
    )
    orphaned_heading = "## Comparative Scope and Accuracy"
    first_cited = (
        "Liu and Wu (2019) surveyed seventy ESL students about their "
        "preferences among correction types."
    )
    second_heading = "## Revision Behaviour and Engagement"
    second_cited = "Koltovskaia (2022) reported that teachers kept using Grammarly regardless."
    text = (
        f"{removed_finding}\n\n{orphaned_heading}\n\n{first_cited}\n\n"
        f"{second_heading}\n\n{second_cited}"
    )
    links = [
        _link(removed_finding, ["lee_2021"], citation_text="(Lee, 2021)", paragraph_index=0),
        _link(first_cited, ["liu_2019"], citation_text="(2019)", paragraph_index=1),
        _link(second_cited, ["koltovskaia_2022"], citation_text="(2022)", paragraph_index=2),
    ]
    status = {
        (removed_finding, removed_finding, "lee_2021"): "unsupported",
        (first_cited, first_cited, "liu_2019"): "verified",
        (second_cited, second_cited, "koltovskaia_2022"): "verified",
    }

    result = finalize_generated_section(
        text, links, [], status,
        section_title="Automated written corrective feedback",
    )

    assert "Comparative Scope and Accuracy" not in result.text
    assert "Revision Behaviour and Engagement" not in result.text
    assert first_cited in result.text
    assert second_cited in result.text
    assert result.text == f"{first_cited}\n\n{second_cited}"
    assert result.stats["headings_removed"] == 2
    assert result.stats["sentences_removed_title_echo"] == 0


# --------------------------------------------------------------------------------------
# links_unmatched_to_sentence (Fix B's own companion guard)
# --------------------------------------------------------------------------------------


def test_links_unmatched_to_sentence_is_zero_when_every_link_matches_a_real_fragment():
    text = "Direct correction outperformed coded feedback on accuracy (Liu, 2019)."
    links = [_link(text, ["liu_2019"], citation_text="(Liu, 2019)")]
    status = {(text, text, "liu_2019"): "verified"}

    result = finalize_generated_section(text, links, [], status)

    assert result.stats["links_unmatched_to_sentence"] == 0


def test_links_unmatched_to_sentence_counts_a_linker_reported_merged_sentence():
    """The linker reports two of finalize's own
    sentence fragments as one ``sentence``. Invisible to `_finalize_paragraph_text`'s
    own exact-match lookup, so the paragraph looks link-free and is dropped as
    dangling -- but it is also counted here, by name, instead of only ever showing up
    as an unexplained `sentences_removed_dangling` count."""
    s1 = "Direct correction outperformed coded feedback on accuracy (Liu, 2019)."
    s2 = "Learner preferences point in a compatible direction."
    text = "Opening framing paragraph with no citation of its own.\n\n%s %s" % (s1, s2)
    merged = "%s %s" % (s1, s2)
    links = [_link(merged, ["liu_2019"], citation_text="(Liu, 2019)", paragraph_index=1)]
    proposition = "Direct correction outperformed coded feedback on accuracy"
    status = {(merged, proposition, "liu_2019"): "verified"}

    result = finalize_generated_section(text, links, [], status)

    assert result.citation_links == []
    assert result.stats["links_unmatched_to_sentence"] == 1
    assert result.stats.get("sentences_removed_dangling", 0) == 2


def test_links_unmatched_to_sentence_counts_an_out_of_range_paragraph_index():
    """Out-of-range fallback: the link's own sentence is absent from the text entirely, and its
    reported ``paragraph_index`` (a fallback `_paragraph_index_for_sentence` trusts
    only when no block contains the sentence at all) is out of range -- bucketed onto
    an index the per-paragraph loop never visits, so it would otherwise be lost with
    no record at all."""
    text = "Opening framing paragraph with no citation of its own."
    links = [_link(
        "A sentence that is not in this text at all.", ["liu_2019"],
        citation_text="(Liu, 2019)", paragraph_index=97,
    )]

    result = finalize_generated_section(text, links, [], {})

    assert result.citation_links == []
    assert result.stats["links_unmatched_to_sentence"] == 1
