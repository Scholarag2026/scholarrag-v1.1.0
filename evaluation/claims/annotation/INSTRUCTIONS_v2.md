# Annotation instructions (round 2)

This rubric was drafted for a single annotator working alone. The three sessions that executed it
on 2026-09-05 were Claude Opus sessions run through Claude Code, not human annotators; no author of
this software has read the 30 items against the cited papers. See `PROVENANCE.md` section 5 for
what the resulting labels do and do not establish.

## Purpose

This is an independent check of the labels used in a claim-verification evaluation. Each row in
hss_annotation_sheet_v2.csv (or your copy annotator2_A.csv / annotator2_B.csv /
annotator2_C.csv) shows a claim sentence, the paper it is cited to, and a passage that was
retrieved from that paper's text. Your job is to read the claim and the passage and assign one
label per row. You are one of three independent annotation sessions checking the same 30 items;
do not open the other two sessions' files, and do not open any results, summary, key or
provenance file in this folder.

## Labels

Use exactly one of these four labels for every row, written in the annotator_label column:

- verified: the cited paper's text supports the claim as stated.
- needs_nuance: the paper supports only a weaker, narrower or conditional version; the claim
  overstates, generalises or drops a qualifier.
- unsupported: the cited paper contradicts the claim, or does not address it, so the citation
  does not support it.
- no_full_text: the item shows the marker "Evidence withheld" and no passage or paper text is
  available for it; apply this label in that case regardless of how plausible the claim sounds.

## Rule

Judge only from the cited_paper_passage column and, when a path is listed for that row in
source_texts_index_v2.json, the cited paper's own cached text at that path. If the passage says
"No matching passage found", open the cited paper's text and search for the claim's key terms
before deciding. Do not use prior knowledge about the claim's subject matter. Do not open any
file that is not listed for the row you are judging.

## What to fill in

For each row, fill in:

- annotator_label: one of the four labels above.
- annotator_confidence: high, medium, or low.
- annotator_note: a short note (a sentence or two) explaining the decision, ideally quoting the
  part of the passage or paper text that drove it.
- annotator_name: your session identifier.
- annotated_on: the date you completed the row.

Leave every other column exactly as given; do not edit ann_id, claim_text,
cited_paper_title, cited_paper_doi, cited_paper_first_author_year or
cited_paper_passage.

## Time estimate

Working through all 30 rows carefully, including opening the cited paper's text where the rule
above requires it, should take about 1 to 2 hours.
