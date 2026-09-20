# Attestation: blind annotator C, real claims sheet

Session identifier: annotator3_C
Date of labelling: 2026-09-07
Items labelled: 26 of 26 (annr-01 to annr-26), one label and one justification each.

## Procedure

I read the rubric first, then the sheet, then labelled the items in two halves. After the
first half (annr-01 to annr-13) I wrote the partially completed sheet to
`evaluation/claims/annotation/real_annotator3_C.csv`, then labelled annr-14 to annr-26 and
rewrote the same file with all 26 rows. Every row was judged from the passage in the sheet
plus the cited paper's own cached text at the path listed for that row in
`source_texts_index_real_v3.json`, as the rubric's rule requires. I used no prior knowledge
of the subject matter and no external sources.

The output file keeps the six sheet columns exactly as given and fills in annotator_label,
annotator_confidence, annotator_note, annotator_name and annotated_on. Two extra columns,
`label` and `justification`, repeat the label and the note, as the task asked for.

## Every file I opened

Rubric and sheet:

1. evaluation/claims/annotation/INSTRUCTIONS_v3.md
2. evaluation/claims/annotation/real_annotation_sheet_real_v3.csv
3. evaluation/claims/annotation/source_texts_index_real_v3.json

Cited papers' cached text, all under evaluation/claims/data/real_claims_fulltext/, one per item:

4.  10-3389_feduc-2020-00059.json (annr-01)
5.  10-1111_jcpp-12352.json (annr-02)
6.  10-14742_ajet-1026.json (annr-03)
7.  10-14483_calj-v18n2-10610.json (annr-04)
8.  10-3389_fpubh-2022-1036071.json (annr-05)
9.  10-1007_s10833-018-9320-9.json (annr-06)
10. 10-1080_09243453-2020-1746363.json (annr-07)
11. 10-3389_fpsyg-2022-1057730.json (annr-08)
12. 10-21601_ijese_10966.json (annr-09)
13. 10-1375_ajgc-17-2-160.json (annr-10)
14. 10-14742_ajet-4310.json (annr-11)
15. 10-3402_rlt-v20i0-14430.json (annr-12)
16. 10-3991_ijet-v15i23-18799.json (annr-13)
17. 10-3991_ijet-v9i8-3805.json (annr-14)
18. 10-1007_s10993-023-09649-4.json (annr-15)
19. 10-1007_s11251-008-9065-6.json (annr-16)
20. 10-1007_s11858-017-0878-0.json (annr-17)
21. 10-1111_jcpp-12605.json (annr-18)
22. 10-3384_rela-2000-7426-rela9063.json (annr-19)
23. 10-4018_978-1-4666-3914-0-ch013.json (annr-20)
24. 10-14483_22487085-13373.json (annr-21)
25. 10-1007_s10734-010-9356-0.json (annr-22)
26. 10-1007_bf02213420.json (annr-23)
27. 10-1007_s10643-019-00971-3.json (annr-24)
28. 10-23917_humaniora-v19i2-6809.json (annr-25)
29. 10-30935_scimath_9511.json (annr-26)

Files I wrote:

- evaluation/claims/annotation/real_annotator3_C.csv (interim write after annr-13, final write after annr-26)
- evaluation/claims/annotation/real_attestation3_C.md (this file)

Helper scripts used for searching the cached paper texts were written outside the repository,
in the session temporary directory, and no other file in the repository was created or changed.

## Files I did not open

I opened no key file, no other annotator's file, and no results, summary, provenance, build,
veto, report or internal development-tooling file. I did not run git log and did not search the repository.
I listed the file names in evaluation/claims/data/real_claims_fulltext/ once to confirm the
directory layout before reading the paths given in the index; I opened only the 26 files
listed above from that directory.

## Label counts

- verified: 16
- needs_nuance: 8
- unsupported: 2
- no_full_text: 0 (no item carried the "Evidence withheld" marker)

Nothing was committed.
