# Attestation, annotator C

Session identifier: annotator3_C
Date of labelling: 2026-09-07
Items labelled: 60 of 60 (ann-01 through ann-60)
Output sheet: `evaluation/claims/annotation/annotator3_C.csv`

## Declaration

I labelled all 60 rows of `hss_annotation_sheet_v3.csv` from the claim text, the
`cited_paper_passage` column, and, where the rubric's rule required it, the cited paper's own
cached full text at the path listed for that row in `source_texts_index_v3.json`. I did not open
the key, any other annotator's sheet, any results, summary, provenance, build, veto or report
file, and I did not run `git log` or search the repository for the expected labels. Nothing was
committed.

I opened the cached paper text in two situations only, both required by the rubric: when the
passage read "No matching passage found in the cited paper", and when the claim stated a detail
(a setting, population, time window or instrument) that the shown passage did not contain. In the
second case I searched the whole cached text for that detail before choosing `needs_nuance`.

## Files opened

Rubric and sheet:

1. `evaluation/claims/annotation/INSTRUCTIONS_v3.md`
2. `evaluation/claims/annotation/hss_annotation_sheet_v3.csv`
3. `evaluation/claims/annotation/source_texts_index_v3.json`

Cached paper texts, all under `evaluation/claims/data/hss_test_v3_fulltext/`, each listed in
`source_texts_index_v3.json` for the rows named:

4. `10-14746_ssllt-2022-12-1-3.json` (rows ann-01, ann-54)
5. `10-17323_jle-2021-10771.json` (rows ann-14, ann-20)
6. `10-18806_tesl-v36i1-1301.json` (row ann-18)
7. `10-14483_22487085-17087.json` (row ann-21)
8. `10-14483_22487085-12966.json` (row ann-25)
9. `10-14483_22487085-14086.json` (row ann-30)
10. `10-14483_22487085-14514.json` (row ann-31)
11. `10-14746_ssllt-2021-11-1-2.json` (row ann-37)
12. `10-17011_apples_urn-201903251959.json` (rows ann-41, ann-45)
13. `10-14746_ssllt-2022-12-1-4.json` (rows ann-42, ann-46)
14. `10-17323_jle-2022-15962.json` (row ann-43)
15. `10-29140_jaltcall-v18n2-632.json` (row ann-47)
16. `10-55593_ej-26103a1.json` (row ann-55)
17. `10-17323_jle-2021-11494.json` (row ann-56)

A directory listing of `evaluation/claims/data/hss_test_v3_fulltext/` was produced by `ls`; the
listing showed file names and sizes only, and no cached text outside the numbered list above was
read. The remaining rows were decided from the `cited_paper_passage` column alone, which either
reproduced the paper's sentence or carried the "Evidence withheld" marker.

Working scripts were written to a temporary directory outside the repository
(`%TEMP%\psearch.py`, `%TEMP%\annot_c.py`, `%TEMP%\annot_c_part2.py`) and no repository file other
than `annotator3_C.csv` and this attestation was created or modified.

## Label counts

| label | count |
| --- | --- |
| unsupported | 26 |
| verified | 16 |
| needs_nuance | 10 |
| no_full_text | 8 |
| total | 60 |

## Notes on judgement

- `no_full_text` was applied to exactly the eight rows carrying the "Evidence withheld" marker
  with an empty `paths` list in the index (ann-06, ann-08, ann-15, ann-23, ann-33, ann-51, ann-53,
  ann-60), regardless of how plausible the claim read.
- `unsupported` covers two situations: the passage stating the opposite of the claim, usually a
  reversed direction word or a changed number, and rows where the passage read "No matching
  passage found" and a search of the cited paper's cached text for the claim's key terms returned
  nothing, so the citation does not address the claim.
- `needs_nuance` was used only where the paper supported the claim's main sentence but the claim
  appended a detail, such as "using eye-tracking software", "using a mobile application", "among
  heritage speakers", "in a bilingual classroom" or "during the winter semester", that a search of
  the cited paper's full cached text did not find.
- `verified` covers rows where the passage repeated the claim verbatim or differed only in
  neutral wording, such as "instructors" for "teachers", "learners" for "students", "observed" for
  "found" or "nevertheless" for "however".
- Confidence was recorded as high for all 60 rows, because every decision rested either on an
  explicit contradiction in the passage, an exact match, an explicit withheld marker, or an
  exhaustive search of the cited paper's cached text.
