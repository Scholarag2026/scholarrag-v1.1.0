# Attestation: independent annotator A

Annotator name: Claude Opus annotator A
Date: 2026-09-05
Labelled file: evaluation/claims/annotation/annotator2_A.csv (30 rows)

## Every file I opened

1. evaluation/claims/annotation/INSTRUCTIONS_v2.md
2. evaluation/claims/annotation/source_texts_index_v2.json
3. evaluation/claims/annotation/annotator2_A.csv
4. evaluation/claims/data/hss_fulltext/10-32601_ejal-710194.json (listed for ann-04)
5. evaluation/claims/data/hss_fulltext/10-55593_ej-26103a4.json (listed for ann-08)
6. evaluation/claims/data/hss_fulltext/10-32601_ejal-651346.json (listed for ann-13)
7. evaluation/claims/data/hss_fulltext/10-29140_jaltcall-v17n2-336.json (listed for ann-25)
8. evaluation/claims/data/hss_fulltext/10-55593_ej-27105a9.json (listed for ann-30)

Files 4 to 8 are exactly the cached paper texts that source_texts_index_v2.json lists for the five
rows whose cited_paper_passage reads "No matching passage found in the cited paper (best overlap
below threshold)" (ann-04, ann-08, ann-13, ann-25, ann-30). I opened them only to search for the
key terms of those five claims. All other rows were judged from the cited_paper_passage column
alone.

## Attestation

I opened no file other than the eight listed above: no key, no jsonl, no build json, no other
annotator's file, no other file under evaluation/, and no internal design note; I ran no git
command, made no LLM or paid API call, and used no prior knowledge of the cited papers beyond
the passages and cached texts named here.

## Label distribution

- verified: 10 (ann-03, ann-05, ann-09, ann-10, ann-12, ann-15, ann-17, ann-19, ann-20, ann-29)
- needs_nuance: 0
- unsupported: 15 (ann-04, ann-06, ann-07, ann-08, ann-11, ann-13, ann-14, ann-21, ann-22,
  ann-23, ann-24, ann-25, ann-27, ann-28, ann-30)
- no_full_text: 5 (ann-01, ann-02, ann-16, ann-18, ann-26)
