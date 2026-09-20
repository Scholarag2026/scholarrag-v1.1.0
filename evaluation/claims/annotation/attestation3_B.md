# Attestation, annotator B

Session identifier: annotator3_B. Labelling completed 2026-09-07. This session is a Claude Code
model session, not a human annotator.

## What I was asked to do

Label all 60 items of `evaluation/claims/annotation/hss_annotation_sheet_v3.csv` blind, using the
rubric `evaluation/claims/annotation/INSTRUCTIONS_v3.md`, with one label and one justification per
item, working in two halves with an interim write, and without opening any key, any other
annotator's file, or any results, summary, provenance, build, veto, report or internal development-tooling file.

## Every file I opened

Read (in this order):

1. `evaluation/claims/annotation/INSTRUCTIONS_v3.md`
2. `evaluation/claims/annotation/hss_annotation_sheet_v3.csv`
3. `evaluation/claims/annotation/source_texts_index_v3.json`
4. `evaluation/claims/data/hss_test_v3_fulltext/10-14746_ssllt-2022-12-1-3.json` (ann-01, ann-54)
5. `evaluation/claims/data/hss_test_v3_fulltext/10-17323_jle-2021-10771.json` (ann-14, ann-20)
6. `evaluation/claims/data/hss_test_v3_fulltext/10-18806_tesl-v36i1-1301.json` (ann-18)
7. `evaluation/claims/data/hss_test_v3_fulltext/10-14483_22487085-17087.json` (ann-21)
8. `evaluation/claims/data/hss_test_v3_fulltext/10-14483_22487085-12966.json` (ann-25)
9. `evaluation/claims/data/hss_test_v3_fulltext/10-14483_22487085-14086.json` (ann-30)
10. `evaluation/claims/data/hss_test_v3_fulltext/10-14483_22487085-14514.json` (ann-31)
11. `evaluation/claims/data/hss_test_v3_fulltext/10-14746_ssllt-2021-11-1-2.json` (ann-37)
12. `evaluation/claims/data/hss_test_v3_fulltext/10-17011_apples_urn-201903251959.json` (ann-41, ann-45)
13. `evaluation/claims/data/hss_test_v3_fulltext/10-14746_ssllt-2022-12-1-4.json` (ann-42, ann-46)
14. `evaluation/claims/data/hss_test_v3_fulltext/10-17323_jle-2022-15962.json` (ann-43)
15. `evaluation/claims/data/hss_test_v3_fulltext/10-29140_jaltcall-v18n2-632.json` (ann-47)
16. `evaluation/claims/data/hss_test_v3_fulltext/10-55593_ej-26103a1.json` (ann-55)
17. `evaluation/claims/data/hss_test_v3_fulltext/10-17323_jle-2021-11494.json` (ann-56)

Each cached paper was opened only for the items listed beside it, all of which list that path in
`source_texts_index_v3.json`. I opened them through a small search script that prints the paper's
title and the windows of text around the search terms, so I saw the cited paper's own text and
nothing else. A directory listing of `evaluation/claims/data/hss_test_v3_fulltext/` was produced by
`ls`; it shows four filenames that belong to no item in this sheet, and I did not open those four.

Written:

- `evaluation/claims/annotation/annotator3_B.csv` (interim write with ann-01 to ann-30 labelled,
  then the final write with all 60 rows)
- `evaluation/claims/annotation/attestation3_B.md` (this file)

Scratch scripts were written outside the repository, under the Windows temp directory.

## What I did not open

No key file, no other annotator's file, no results, summary, provenance, build, veto, report or
internal development-tooling file, no `evaluation/claims/results/` content, and no `real_*` sheet. I ran no
`git log` and no repository-wide search. I used no prior knowledge of these papers beyond the
passage shown for each item and the cached text of that item's own cited paper.

## Method

I worked in two halves. For the first half, ann-01 to ann-30, I judged each row from the passage,
opened the cited paper's cached text for every row whose passage said "No matching passage found"
and for every row whose claim added a detail the passage did not contain, then wrote the interim
sheet with those 30 rows labelled. I then repeated the procedure for ann-31 to ann-60 and rewrote
the sheet with all 60 rows.

Where a claim added a setting, a population, a time window or an instrument, I searched the cited
paper's own cached text for that detail before choosing `needs_nuance`, as the rubric requires. In
every such case the search returned zero hits, for example zero hits for "heritage" in Wedin and
Shaswar and in Aubrey, zero hits for "mobile" in the three Colombian papers, zero hits for
eye-tracking in either Kruk paper or in Becirovic et al., zero hits for "bilingual" in Parra, and
zero hits for "winter" or "semester" in Savski and Prabjandee.

## Result

60 of 60 items labelled: 26 unsupported, 16 verified, 10 needs_nuance, 8 no_full_text. Confidence
is high on 59 items and medium on one, ann-57, where the paper contradicts the claim's
normalization base of 1370 words with its own "normalized per 1000 words" while the two reported
figures and the direction of the comparison still match.
