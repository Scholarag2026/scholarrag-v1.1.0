# Annotation instructions (round 3)

This rubric was drafted for a single annotator working alone. The three sessions that executed it
against the 60-item HSS set on 2026-09-07 (session identifiers annotator3_A, annotator3_B,
annotator3_C) and the three further sessions that executed it against the 26-item real-claims set,
also on 2026-09-07 (the same three session identifiers, three separate sessions run one after the
other, not the same sessions reused), were all Claude Opus sessions run through Claude Code, not
human annotators; no author of this software has read the 60 HSS items or the 26 real-claims items
against the cited papers and countersigned the labels. See `PROVENANCE.md` section 9 for what the
resulting labels do and do not establish.

Note on scope: this file's `## Purpose` section below names only the HSS set's file names
(`hss_annotation_sheet_v3.csv`, `annotator3_A.csv` / `_B` / `_C`, `source_texts_index_v3.json`, and
"the same 60 items") and governed the HSS sessions unchanged. It also governed the real-claims
sessions: each of the three real-claims sessions substituted `real_annotation_sheet_real_v3.csv`
for the sheet, `real_annotator3_A.csv` / `_B` / `_C` for its own output, and
`source_texts_index_real_v3.json` for the index, judging 26 items (`annr-01` to `annr-26`) rather
than 60, exactly as each session's own attestation records. The four label definitions and the
judging rule below are identical for both sets; no rubric wording changed between them. See
`PROVENANCE.md` section 9.2 for the full account.

## Purpose

This is an independent check of the labels used in a claim-verification evaluation. Each row in
hss_annotation_sheet_v3.csv (or your copy annotator3_A.csv / annotator3_B.csv /
annotator3_C.csv) shows a claim sentence, the paper it is cited to, and a passage that was
retrieved from that paper's text. Your job is to read the claim and the passage and assign one
label per row. You are one of three independent annotation sessions checking the same 60 items;
do not open the other two sessions' files, and do not open any results, summary, key or
provenance file in this folder.

## Labels

Use exactly one of these four labels for every row, written in the annotator_label column:

- verified: the cited paper's text supports the claim as stated.
- needs_nuance: the paper supports only a weaker, narrower or conditional version; the claim
  overstates, generalises or drops a qualifier. This also covers the case where the paper supports
  the claim's main finding but the claim also states a detail, such as a setting, a population, a
  time window or an instrument, that the cited paper does not state.
- unsupported: the cited paper contradicts the claim, or does not address it, so the citation
  does not support it.
- no_full_text: the item shows the marker "Evidence withheld" and no passage or paper text is
  available for it; apply this label in that case regardless of how plausible the claim sounds.

## Rule

Judge only from the cited_paper_passage column and, when a path is listed for that row in
source_texts_index_v3.json, the cited paper's own cached text at that path. If the passage says
"No matching passage found", open the cited paper's text and search for the claim's key terms
before deciding. A claim that adds a detail the shown passage does not contain must be checked
against the cited paper's own text before needs_nuance is chosen; the shown passage is a window,
not the whole paper. Do not use prior knowledge about the claim's subject matter. Do not open any
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

Working through all 60 rows carefully, including opening the cited paper's text where the rule
above requires it, should take about 2 to 4 hours.
