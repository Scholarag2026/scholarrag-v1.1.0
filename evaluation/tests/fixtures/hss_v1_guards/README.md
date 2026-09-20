Redacted slice of the retired 30-item HSS claim set's v1 verifier run
(`claims/results/v1/hss_run{A,B}.jsonl`, superseded by the 60-item set `results/v6/` scores),
kept only so `test_claims_numeric_guard.py` and `test_claims_quote_fidelity_guard.py` can
regression-test the numeric and quote-fidelity guards against real claim text and real
guard-input fields. Each row keeps only `item_id`, `rule`, `claim`, `chunk_doi`,
`predicted_status`, `evidence_quote` and `quote_is_verbatim`; every other field from the
original run (token counts, timings, provenance, explanations, and the like) is dropped.
Not an evaluation result: no accuracy, precision or recall figure is computed from this
directory.
