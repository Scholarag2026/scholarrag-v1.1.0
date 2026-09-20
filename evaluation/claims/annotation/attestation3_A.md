# Attestation, annotator A

Session identifier: annotator3_A
Date of labelling: 2026-09-07
Items labelled: 60 of 60 (ann-01 through ann-60)
Output sheet: `evaluation/claims/annotation/annotator3_A.csv`

## Procedure

I read the rubric and the sheet, labelled ann-01 through ann-30, wrote that partial sheet to
`annotator3_A.csv` as an interim save, then labelled ann-31 through ann-60 and rewrote the file
with all 60 rows. Every row carries exactly one of the four rubric labels, a confidence value, and
a one-sentence justification quoting the decisive words. Where the shown passage said "No matching
passage found" or where the claim added a detail the passage did not contain, I searched the cited
paper's cached text at the path listed for that row in `source_texts_index_v3.json` before
deciding, as the rubric requires. I judged only from the passage and the cited paper's own text,
not from prior knowledge of the subject matter.

## Files opened

Rubric and sheet:

- `evaluation/claims/annotation/INSTRUCTIONS_v3.md`
- `evaluation/claims/annotation/hss_annotation_sheet_v3.csv`
- `evaluation/claims/annotation/source_texts_index_v3.json`

Cited paper cached texts, all under `evaluation/claims/data/hss_test_v3_fulltext/`:

- `10-14483_22487085-12966.json`
- `10-14483_22487085-14086.json`
- `10-14483_22487085-14514.json`
- `10-14483_22487085-17087.json`
- `10-14746_ssllt-2021-11-1-2.json`
- `10-14746_ssllt-2022-12-1-3.json`
- `10-14746_ssllt-2022-12-1-4.json`
- `10-17011_apples_urn-201903251959.json`
- `10-17323_jle-2021-10771.json`
- `10-17323_jle-2021-11494.json`
- `10-17323_jle-2022-15962.json`
- `10-18806_tesl-v36i1-1301.json`
- `10-29140_jaltcall-v18n2-632.json`
- `10-29140_tltl-v6n1-1142.json`
- `10-29140_tltl-v6n1-1168.json`
- `10-32601_ejal-710204.json`
- `10-32601_ejal-776002.json`
- `10-32601_ejal-911245.json`
- `10-55593_ej-25100a3.json`
- `10-55593_ej-26103a1.json`

That is 20 of the 26 cached texts in that folder. I also listed the contents of
`evaluation/claims/data/hss_test_v3_fulltext/` to resolve the paths given in the index. The rows
whose cached text I did not need to open were decided from the shown passage alone, either because
the passage reproduced the claim word for word, because it stated the opposite of the claim, or
because the item was marked "Evidence withheld".

## Files I did not open

I did not open any key file, any other annotator's sheet or attestation, any results, summary,
provenance, build, veto, report or internal development-tooling file, and I did not run `git log` or search the
repository outside the files listed above.

## Helper scripts

Labelling used two throwaway scripts that I created in this folder and deleted afterwards,
`_tmp_labels_A.py` (the label and justification table) and `_tmp_write_A.py` (which merged that
table into the sheet columns and wrote `annotator3_A.csv`), plus a small local search script under
the system temporary directory used to grep the cached paper texts. No repository file other than
`annotator3_A.csv` and this attestation was created or modified, and nothing was committed.
