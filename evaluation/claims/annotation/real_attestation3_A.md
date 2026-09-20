# Attestation, blind annotator A, real claims

Session identifier: annotator3_A
Date of labelling: 2026-09-07
Sheet labelled: evaluation/claims/annotation/real_annotation_sheet_real_v3.csv (26 items, annr-01 to annr-26)
Output: evaluation/claims/annotation/real_annotator3_A.csv

## Procedure

I read the rubric, then the sheet, then worked through the 26 items in two halves. After the first
13 items I wrote an interim version of the output file with those rows labelled and the remaining
rows blank, then completed items 14 to 26 and rewrote the file. Every item was judged from the
cited_paper_passage column plus the cited paper's own cached text at the path listed for that row
in source_texts_index_real_v3.json. No item carried the "Evidence withheld" marker, so the
no_full_text label was not used. One item (annr-03) showed "No matching passage found in the cited
paper", and per the rubric I searched the cited paper's cached text for the claim's key terms
before deciding.

I did not open any key file, any other annotator's file, or any results, summary, provenance,
build, veto, report or internal development-tooling file, and I did not run git log or search the repository
outside the files listed below.

## Files opened

Rubric and sheet:

1. evaluation/claims/annotation/INSTRUCTIONS_v3.md
2. evaluation/claims/annotation/real_annotation_sheet_real_v3.csv
3. evaluation/claims/annotation/source_texts_index_real_v3.json

Cited papers' cached text, one per item, all under evaluation/claims/data/real_claims_fulltext/:

4. 10-3389_feduc-2020-00059.json (annr-01)
5. 10-1111_jcpp-12352.json (annr-02)
6. 10-14742_ajet-1026.json (annr-03)
7. 10-14483_calj-v18n2-10610.json (annr-04)
8. 10-3389_fpubh-2022-1036071.json (annr-05)
9. 10-1007_s10833-018-9320-9.json (annr-06)
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

A directory listing of evaluation/claims/data/real_claims_fulltext/ was produced while checking the
structure of the cached files; only the 26 files above were read.

## Files written

- evaluation/claims/annotation/real_annotator3_A.csv (interim after item 13, final after item 26)
- evaluation/claims/annotation/real_attestation3_A.md (this file)

Nothing was committed. To confirm that, I ran `git status --porcelain` restricted to
evaluation/claims/annotation/. That listing incidentally showed the file names
real_annotator3_B.csv, real_annotator3_C.csv and real_attestation3_B.md as untracked. I did not
open those files or read any of their contents, and no label in this sheet was influenced by them.
I ran no other git command.

## Label counts

- verified: 15
- needs_nuance: 10
- unsupported: 1
- no_full_text: 0

Confidence: 10 high, 16 medium, 0 low.
