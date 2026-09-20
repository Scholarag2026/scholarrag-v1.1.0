Redacted slice of the pre-fix, superseded claim-verification run that documented four
line-break-hyphenation quote-fidelity misses (`hss-verbatim-01`, `hss-paraphrase-04`,
`real-test-12`, `real-test-20`), kept only so
`test_claims_quote_fidelity_guard.py::test_quote_fidelity_guard_no_longer_fires_on_the_v3_line_break_hyphenation_misses`
can regression-test that the fixed guard no longer fires on them. Each file keeps only the
rows for those four item ids, and each row keeps only `item_id`, `model_status`,
`machine_reasons`, `evidence_quote` and `evidence_quotes`; every other field (token counts,
timings, provenance, and the like) is dropped. Not an evaluation result: the run this slice
was drawn from was superseded by the current `results/v3/`/`results/v6/` runs, whose rows no
longer reproduce the miss.
