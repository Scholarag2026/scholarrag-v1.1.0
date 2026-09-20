# E2 claim verification -- summary

Primary label of record: Table E2-h (the three reviewers' majority verdict).

Generated 2026-09-16T03:25:02+00:00.

## Table E2-a SciFact (dev set, one item per cited claim-document row)

n = 340 rows over 339 distinct claim-document pairs: claim 1245 cites document 7662395 twice in the SciFact dev set's own `cited_doc_ids`, so `run_scifact.build_items` emits that pair twice; both copies are sent to the model, billed and counted in every metric below (both copies score the same predicted status in both runs, so this is a rounding-level effect, not a scoring error). Runs A and B build items from the same dev set, so the duplicate is identical in both.

| run | n | verified P | verified R | verified F1 | unsupported P | unsupported R | unsupported F1 | macro-F1 | accuracy | lenient P | lenient R | lenient F1 | quote verbatim | quote verbatim (casefold) | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLM run A | 340 | 0.983 | 0.413 | 0.582 | 0.868 | 0.980 | 0.921 | 0.751 | 0.750 | 0.983 | 0.413 | 0.582 | 1.000 | 1.000 | 0 |
| LLM run B | 340 | 0.983 | 0.413 | 0.582 | 0.876 | 0.980 | 0.925 | 0.753 | 0.750 | 0.983 | 0.413 | 0.582 | 1.000 | 1.000 | 0 |
| lexical baseline | 340 | 0.500 | 0.007 | 0.014 | 0.595 | 0.995 | 0.744 | 0.379 | 0.594 | 0.500 | 0.007 | 0.014 | - | - | 0 |

Strict: gold SUPPORT -> verified, CONTRADICT/NOT_ENOUGH_INFO -> unsupported; needs_nuance / no_full_text / error predictions count as misses and never as hits. Lenient: verified vs not-verified. A class that is never predicted has precision undefined (-) and F1 = 0 by convention; macro-F1 averages over both classes.

Confusion (run A; rows = gold, columns = predicted):

| gold | verified | needs_nuance | unsupported | no_full_text | error |
|---|---|---|---|---|---|
| SUPPORT | 57 | 51 | 30 | 0 | 0 |
| CONTRADICT | 0 | 1 | 70 | 0 | 0 |
| NOT_ENOUGH_INFO | 1 | 2 | 128 | 0 | 0 |

needs_nuance by gold class (run A): SUPPORT=51, CONTRADICT=1, NOT_ENOUGH_INFO=2

Confusion (run B; rows = gold, columns = predicted):

| gold | verified | needs_nuance | unsupported | no_full_text | error |
|---|---|---|---|---|---|
| SUPPORT | 57 | 53 | 28 | 0 | 0 |
| CONTRADICT | 0 | 1 | 70 | 0 | 0 |
| NOT_ENOUGH_INFO | 1 | 2 | 128 | 0 | 0 |

needs_nuance by gold class (run B): SUPPORT=53, CONTRADICT=1, NOT_ENOUGH_INFO=2

A/B agreement (n=340, compared 340; excluded 0 deterministic, 0 error): status 0.979 (kappa 0.959); binary 0.994 (kappa 0.979).

## Table E2-f Label-mapping sensitivity (SciFact)

SciFact's three gold labels (SUPPORT, CONTRADICT, NOT_ENOUGH_INFO) are scored against the verifier's four statuses under three mappings: strict (Table E2-a's own convention: SUPPORT -> verified, CONTRADICT/NOT_ENOUGH_INFO -> unsupported, a predicted needs_nuance / no_full_text / error is a miss for its gold class and never a hit); lenient (identical, except a predicted needs_nuance counts as verified); and three-way, available only for rows that carry the per-assertion `verdict` field (prompt v2), which splits a predicted unsupported row by whether any assertion was `contradicted`, scoring CONTRADICT against that split bucket instead of the class it otherwise shares with NOT_ENOUGH_INFO.

| run | mapping | verified P | verified R | verified F1 | unsupported P | unsupported R | unsupported F1 | macro-F1 | CONTRADICT recall | gold-CONTRADICT -> verified |
|---|---|---|---|---|---|---|---|---|---|---|
| run A | strict (needs_nuance is a miss) | 0.983 | 0.413 | 0.582 | 0.868 | 0.980 | 0.921 | 0.751 | 0.986 | 0 |
| run A | lenient (needs_nuance -> verified) | 0.964 | 0.783 | 0.864 | 0.868 | 0.980 | 0.921 | 0.892 | 0.986 | 1 |
| run B | strict (needs_nuance is a miss) | 0.983 | 0.413 | 0.582 | 0.876 | 0.980 | 0.925 | 0.753 | 0.986 | 0 |
| run B | lenient (needs_nuance -> verified) | 0.965 | 0.797 | 0.873 | 0.876 | 0.980 | 0.925 | 0.899 | 0.986 | 1 |

Three-way mapping (predicted unsupported split by the per-assertion contradiction signal; the contradicted bucket is where a gold CONTRADICT row should land):

| run | verified P | verified R | verified F1 | contradicted-bucket P | contradicted-bucket R | contradicted-bucket F1 | other-bucket P | other-bucket R | other-bucket F1 | macro-F1 | CONTRADICT recall | gold-CONTRADICT -> verified |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run A | 0.983 | 0.413 | 0.582 | 0.775 | 0.873 | 0.821 | 0.811 | 0.916 | 0.860 | 0.754 | 0.873 | 0 |
| run B | 0.983 | 0.413 | 0.582 | 0.805 | 0.873 | 0.838 | 0.812 | 0.924 | 0.864 | 0.761 | 0.873 | 0 |

Sentence the manuscript may state: the mapping choice moves the headline numbers only at the margin the model's own needs_nuance and per-assertion signals create; CONTRADICT recall and the count of gold-CONTRADICT rows predicted verified are the figures a reviewer should read as the safety check, not macro-F1 alone.

## Table E2-b HSS set (applied-linguistics full texts; labels are by construction)

Run A: overall accuracy 0.917 (55/60); needs_nuance answers: 9.

| rule | n | expected | correct | accuracy | needs_nuance | predicted statuses | quote verbatim (verified rows) |
|---|---|---|---|---|---|---|---|
| verbatim | 8 | verified | 7 | 0.875 | 1 | needs_nuance=1, verified=7 | 1.000 |
| paraphrase | 8 | verified | 7 | 0.875 | 1 | needs_nuance=1, verified=7 | 1.000 |
| altered | 18 | unsupported | 18 | 1.000 | 0 | unsupported=18 | - |
| over_specified | 10 | needs_nuance | 7 | 0.700 | 7 | needs_nuance=7, unsupported=3 | - |
| wrong_paper | 8 | unsupported | 8 | 1.000 | 0 | unsupported=8 | - |
| no_full_text | 8 | no_full_text | 8 | 1.000 | 0 | no_full_text=8 | - |

Run B: overall accuracy 0.900 (54/60); needs_nuance answers: 10.

| rule | n | expected | correct | accuracy | needs_nuance | predicted statuses | quote verbatim (verified rows) |
|---|---|---|---|---|---|---|---|
| verbatim | 8 | verified | 7 | 0.875 | 1 | needs_nuance=1, verified=7 | 1.000 |
| paraphrase | 8 | verified | 6 | 0.750 | 2 | needs_nuance=2, verified=6 | 1.000 |
| altered | 18 | unsupported | 18 | 1.000 | 0 | unsupported=18 | - |
| over_specified | 10 | needs_nuance | 7 | 0.700 | 7 | needs_nuance=7, unsupported=3 | - |
| wrong_paper | 8 | unsupported | 8 | 1.000 | 0 | unsupported=8 | - |
| no_full_text | 8 | no_full_text | 8 | 1.000 | 0 | no_full_text=8 | - |

| system | n | binary n | correct | accuracy | correct per rule | verified P | verified R | verified F1 |
|---|---|---|---|---|---|---|---|---|
| LLM run A | 60 | 52 | 55 | 0.917 | verbatim=7/8, paraphrase=7/8, altered=18/18, over_specified=7/10, wrong_paper=8/8, no_full_text=8/8 | 1.000 | 0.875 | 0.933 |
| LLM run B | 60 | 52 | 54 | 0.900 | verbatim=7/8, paraphrase=6/8, altered=18/18, over_specified=7/10, wrong_paper=8/8, no_full_text=8/8 | 1.000 | 0.812 | 0.897 |
| lexical baseline | 60 | 52 | 32 | 0.533 | verbatim=8/8, paraphrase=8/8, altered=0/18, over_specified=0/10, wrong_paper=8/8, no_full_text=8/8 | 0.364 | 1.000 | 0.533 |

Lexical baseline: max token-Jaccard between the claim and any chunk sentence >= 0.5 -> verified, else unsupported; items without chunks -> no_full_text. It cannot answer needs_nuance. Its binary verified view (binary n, P/R/F1) excludes the 8 no-chunk item(s), which are rejected by construction, on the same denominator as the LLM rows above.

A/B agreement (n=60, compared 52; excluded 8 deterministic, 0 error): status 0.981 (kappa 0.967).

A/B agreement pairs only items with a model decision in both runs: deterministic no_full_text items (never sent to the model) and error rows are excluded and counted. The binary verified metrics exclude error rows (reported as errors), so a failed call is never scored as a correct rejection, and the deterministic no_full_text items, so a by-construction rejection is never a true negative.

needs_nuance counts answers of that status per rule; for paraphrase items they are reported here and are not scored as correct (expected: verified).

All figures above are point estimates on the stated n with no confidence interval; the 95 % Wilson interval of an accuracy of 0.917 on n=60 is about +-0.07 (Table E2-h), so a few-point difference between the LLM run and the lexical baseline, or between runs A and B, is not interpreted as a difference (see README.md, 'No confidence intervals').

## Table E2-c provenance/cost/latency

| run | model configured | model reported | fingerprints | temperature | prompt version | concurrency | errors | deterministic | input tok | output tok | cache-read tok | cost USD | latency median s | latency p90 s | date |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scifact run A | deepseek-chat | deepseek-flash | 1 | 0.000 | sha256:49fcfbfaf2f6 | 8 | 0 | 0 | 1111374 | 281021 | 897280 | 0.239 | 2.980 | 4.428 | 2026-09-14 |
| scifact run B | deepseek-chat | deepseek-flash | 1 | 0.000 | sha256:49fcfbfaf2f6 | 8 | 0 | 0 | 1085899 | 274247 | 995932 | 0.208 | 2.885 | 4.156 | 2026-09-14 |
| hss run A | deepseek-chat | deepseek-flash | 1 | 0.000 | sha256:49fcfbfaf2f6 | 8 | 0 | 8 | 990859 | 46070 | 980218 | 0.040 | 3.422 | 4.725 | 2026-09-11 |
| hss run B | deepseek-chat | deepseek-flash | 1 | 0.000 | sha256:49fcfbfaf2f6 | 8 | 0 | 8 | 997953 | 47265 | 986362 | 0.041 | 3.376 | 4.693 | 2026-09-11 |

Cost basis: list price, tier by call time, cache-hit tokens at cache-hit rate.

Latency is wall-clock per model call measured while up to 'concurrency' calls were in flight (harness default 8; production uses analysis_concurrency = 4), so medians and p90 include provider-side queueing.

## Guard reasons (machine_reasons)

| run | rows with a reason | status changed (guarded_count) | n | slug: count |
|---|---|---|---|---|
| scifact run A | 281 | 7 | 340 | assertion_status_inconsistent: 279; attribution_mismatch: 34; numeric_value_absent_from_source: 25 |
| scifact run B | 280 | 4 | 340 | assertion_status_inconsistent: 278; attribution_mismatch: 34; numeric_value_absent_from_source: 25 |
| hss run A | 38 | 0 | 60 | assertion_status_inconsistent: 38; attribution_mismatch: 4; numeric_outside_cited_passage: 1; numeric_value_absent_from_source: 2 |
| hss run B | 39 | 0 | 60 | assertion_status_inconsistent: 39; attribution_mismatch: 4; numeric_outside_cited_passage: 1; numeric_value_absent_from_source: 2 |

A row's own status is never changed twice in a way this table would double count: each guard that fires appends one slug to the row's own machine_reasons list, so a row with two reasons is counted once per slug and once in 'rows with a reason'. That column does not imply the guard changed anything: a report-only guard, or a status-changing guard that caps a status onto the value the model already gave, still appends a slug. 'status changed (guarded_count)' is the stricter figure: it counts only rows where predicted_status differs from the model's own model_status, i.e. a guard actually made the status stricter. Rows from a prompt version that predates this field report zero of both counts, not an error.

## Table E2-h reviewer majority, primary label of record

The constructed and real claim sets are scored here against the three reviewers' majority verdict (`results/v6/human_labels.json`), the label of record.

| set | run | n | accuracy | wilson 95% | kappa | verified precision | verified recall | falsely verified |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constructed | run_A | 60 | 0.9167 | 0.819-0.964 | 0.8784 | 1.0000 | 0.8750 | 0 |
| constructed | run_B | 60 | 0.9000 | 0.799-0.953 | 0.8545 | 1.0000 | 0.8125 | 0 |
| real | run_A | 26 | 0.8846 | 0.710-0.960 | 0.8069 | 1.0000 | 1.0000 | 0 |
| real | run_B | 26 | 0.8846 | 0.710-0.960 | 0.8069 | 1.0000 | 1.0000 | 0 |
