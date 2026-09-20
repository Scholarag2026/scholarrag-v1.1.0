# ScholarRAG evaluation harness (SoftwareX revision, v1.1.0)

This folder holds the reproducible evaluation added for the journal revision. It answers
the reviewers' three questions about the LLM components with numbers derived from public
datasets and committed result files:

* **E1 - screening performance vs human labels**: the unchanged production title/abstract
  screener (`app.agents.relevance_screener_agent.screen_papers`) is run over three SYNERGY
  systematic-review datasets and scored against the human screening decisions (recall,
  precision, F1, specificity, inclusion rate, WSS@R), next to include-all and TF-IDF baselines.
* **E1 - reproducibility**: every dataset is screened twice (runs A and B, temperature 0);
  percent agreement and Cohen's kappa between the two runs are reported, and every call's
  model, system fingerprint, response id, prompt version and token counts are recorded.
* **E1 - cost and latency**: per-run token totals, list-price cost (tiered by call time) and
  batch latency statistics.
* **E2 - claim verification**: the unchanged production claim-verification agent on the
  SciFact dev set and on a 60-item humanities/social-science full-text set, next to a lexical
  baseline (section "Claim verification (E2)" below).
* **E2 - claim-verification labels**: on the constructed and real-claims sets, the label
  every accuracy figure is scored against is now the majority verdict of three reviewers who
  each judged every item independently against the stored source text, not a model-authored
  annotation (section "Claim-verification labels (v6)" below); SciFact's own published labels
  are unaffected.

Nothing in `evaluation/` is imported by the product. The scripts import production code by
adding `backend/` to `sys.path` and calling the public functions unchanged.

## Layout

```
evaluation/
  README.md                this file
  common.py                pure helpers: metrics, kappa, WSS, JSONL resume, price table, protocols
  ruff.toml                lint config for the whole tree
  .venv/                   evaluation-only interpreter (gitignored)
  screening/
    fetch_synergy.py       SYNERGY dataset -> data/<id>.jsonl + data/<id>.fetch.json (network, no LLM)
    run_screening.py       runs A/B of the production screener (LLM; system python)
    baselines.py           include-all and TF-IDF (k matched to the LLM inclusion count)
    wos_gate_effect.py     recall ceiling of the optional WoS venue gate (licensed CSVs, not shipped)
    spot_check_hydration.py  seeded sample of title-only hydration matches + Crossref record (spot-check judged by an automated pass, not the authors)
    summarize.py           tables E1-a..d, false-negative lists, figure4_screening.json
    protocols/<id>.json    research question + criteria transcribed from each source review
    protocols/README.md    protocol schema, provenance rules, label_field policy, record order
    data/                  fetched datasets and dry-run output (gitignored)
    results/               committed run files, meta, baselines, gate effect, summaries;
                           see "Results directories" below
  claims/
    fetch_scifact.py       SciFact release tarball -> data/ (network, no LLM)
    run_scifact.py         runs A/B of the production verifier on the 340 SciFact dev pairs
    build_hss_set.py       acquires 21 OA applied-linguistics papers, builds the 60-claim HSS set
    run_hss.py             runs A/B of the production verifier on the HSS set
    baseline_lexical.py    lexical-overlap baseline (--set scifact | hss)
    verify_common.py       shared runner: production/dry-run verifier, resume, retry, meta
    summarize.py           tables E2-a..d, figure4_claims.json
    freeze_v6.py            replays every cached model answer under results/v6/inputs/
                           through the guard set at HEAD (no model call); writes results/v6/
                           and FREEZE_DIGESTS_v6.md
    human_review_labels.py  the three reviewers' majority label per claim item, inter-rater
                           agreement, and v6 scored against that label; writes
                           results/v6/human_labels.json, human_review_scores.json
    delivered_human_scores.py  majority label of record over the reviewers' filled Delivered
                           workbooks for one promoted demo run; writes
                           results/v6/delivered_human_scores_run_<id>.json
    FREEZE_DIGESTS_v6.md    digests for the guard-7 replay; see "Claim-verification
                           labels (v6)" below
    hss_sources.json       a smaller, 10-paper source list, not the set results/v6/ scores
    hss_claims.jsonl       a smaller, 30-item claim set, not the set results/v6/ scores
    hss_test_v3_sources.json   the 21 papers results/v6/ scores against (DOI, journal, ISSN,
                           OpenAlex id, licence where recorded, access date)
    hss_test_v3_claims.jsonl   the 60 committed items results/v6/ scores; hss_test_v3_claims.build.json = build provenance
    hss_veto.json          vetoed-sentence list assembled by an automated review pass, not the authors; honoured by build_hss_set.py
    annotation/            blind triple model annotation (Claude Opus) of the 60 HSS items and
                           the real-claims set, used as a secondary check alongside the human
                           majority label (results/v6/human_labels.json); see
                           annotation/README.md, annotation/PROVENANCE.md
    data/                  SciFact files, full-text cache, dry-run output (gitignored)
    results/               committed run files, meta, baselines, summaries;
                           versioned (v1/, v2/, v3/, v4/, v5/, v6/), see "Results directories" below
  tests/                   pytest suite; no network, no LLM
```

## Results directories

`screening/results/v3/` and `claims/results/v6/` are each the one result of record for their
evaluation, and every script's own `RESULTS_DIR` default already points at it, so the
commands below need no explicit `--results-dir` unless a different directory is wanted.

`screening/results/v3/` (screener prompt `sha256:9bc741046cd6`; the anchor ledger, the
title-only TOPIC exclude, the full-text-to-confirm queue routing, the type/table-of-contents
demotions and the inclusion-only second pass; see `evaluation/screening/FREEZE_v3.md`) is the
screening result of record, over the three datasets in "Datasets and licences" above.
`run_screening.py`, `baselines.py`, `wos_gate_effect.py` and `summarize.py` all default
`--results-dir` to it.

`claims/results/v6/` (verifier prompt `sha256:49fcfbfaf2f6`, `GUARD_DIGEST 8a6c833ffc329f89`,
`QUOTE_RELOCATION_VERSION 0269a13351c0da7d`, `VERIFICATION_POLICY_VERSION 667bcc9209126ddc`;
see `evaluation/claims/FREEZE_DIGESTS_v6.md`) is the claim-verification result of record.
`results/v6/` is a deterministic replay of previously stored model answers through the guard
set shipped at HEAD; no new model call was made to produce it. `verify_common.py`,
`baseline_lexical.py` and `summarize.py` all default `--results-dir` to it.

## Setup

Two interpreters are used (see "Interpreters and working directory" below); both are
Python 3.13 in this repository, though any recent Python 3 should work.

1. Evaluation-only interpreter (`evaluation/.venv`, gitignored, stdlib + `httpx` +
   `scikit-learn` + `pytest` + `ruff`):

   ```
   python -m venv evaluation/.venv
   evaluation/.venv/Scripts/python -m pip install -r evaluation/requirements.txt
   ```

   (POSIX: `evaluation/.venv/bin/python`.) `evaluation/requirements.txt` pins the versions
   this harness was built and tested against, including `pytest` and `ruff` so the test suite
   and lint gate below run from this interpreter without touching the system one;
   `pip install httpx scikit-learn pytest ruff` also works if pinning is not needed.
2. System interpreter, for the scripts that call the production backend
   (`run_screening.py`, `claims/run_scifact.py`, `claims/run_hss.py`,
   `claims/build_hss_set.py`): install the backend's own dependencies with
   `pip install -r backend/requirements.txt` (repository root). These scripts `chdir` into
   `backend/` at import time (see "Working-directory rule" below) and need
   `DEEPSEEK_API_KEY` in the repository-root `.env` for anything beyond `--dry-run`.

Every command below is written for Windows (`evaluation/.venv/Scripts/python`); on
POSIX (Linux/macOS) replace every `\.venv/Scripts/python` with `.venv/bin/python` and every
other path separator is already POSIX-compatible.

## Interpreters and working directory

| Script | Interpreter | Why |
|---|---|---|
| `screening/fetch_synergy.py`, `baselines.py`, `wos_gate_effect.py`, `spot_check_hydration.py`, `summarize.py`, `claims/fetch_scifact.py`, `baseline_lexical.py`, `claims/summarize.py` | `evaluation/.venv/Scripts/python` (or system `python`) | httpx / scikit-learn / stdlib only |
| `screening/run_screening.py`, `claims/run_scifact.py`, `claims/run_hss.py`, `claims/build_hss_set.py` | system `python` (3.13) | need the backend's pydantic-ai / pydantic-settings |
| `tests/` | either | `cd evaluation && python -m pytest tests -q -p no:cacheprovider` |

Lint from the repository root: `python -m ruff check evaluation --config evaluation/ruff.toml`.

**Working-directory rule.** `backend/app/config.py` builds its `Settings` with
`SettingsConfigDict(env_file=".env")`, i.e. relative to the *current* directory, and rejects
unknown keys; the repository-root `.env` carries deployment-only keys, so `import app.config`
from the repository root fails with a pydantic `ValidationError`. Every production-backed
script therefore (1) resolves all path arguments to absolute paths at parse time
(`common.resolve_path_args`), (2) calls `common.add_backend_to_path`, which puts `backend/` on
`sys.path` **and `os.chdir`s into it**, and (3) imports the settings through
`common.import_backend_settings`, which turns any failure into
`SystemExit("backend settings failed to load from <cwd>: ...")`. The commands below can be
launched from the repository root as written; relative `--results-dir` etc. are interpreted
relative to the directory you launch from.

**Real, paid runs are committed.** `screening/results/v3/` and `claims/results/v6/` hold the
real, paid runs of the production screener and verifier; every performance figure in
`results/v3/summary.md` and `results/v6/summary.md` comes from those runs, not from the
`--dry-run` stand-in (Tables E1-c and E2-c give their cost and latency).

## Paid runs: the cost gate

**Every command that is not `--dry-run` spends money the moment it starts.** The merged
production interface is on this branch and `DEEPSEEK_API_KEY` is read from the repo-root
`.env`, so `run_screening.py`, `run_scifact.py` and `run_hss.py` are live. They therefore
refuse to start (exit status 3) unless `--confirm-cost` is given; before exiting they print
the number of batches / model calls they would make and the list-price upper bound
(`run_screening.cost_upper_bound`: 2,000 input + 1,000 output tokens per batch;
`verify_common.cost_upper_bound`: 2,600 / 27,000 input + 1,000 output tokens per SciFact /
HSS call, all at the peak cache-miss tariff). `--limit` does **not** lift the gate: a
`--limit 30` smoke run is a paid run and needs the flag too. Recommended order for every
runner: (1) `--dry-run` (no LLM, writes under `data/_dryrun/`), (2) `--limit 30
--confirm-cost` and check `model_reported`, `temperature` and `latency_s` in the meta file,
(3) the full run with `--confirm-cost`.

## Testing without an LLM

No test and no `--dry-run` invocation contacts an LLM or the network. `run_screening.py
--dry-run` swaps the production screener for a deterministic keyword stand-in and writes to
`screening/data/_dryrun/results/` so the committed `results/` folder is never touched; the
unit tests patch the screener boundary (`main_async(args, screener=...)`) and the HTTP
boundary (`fetch_with_retries`). `run_screening.py` refuses to start against a backend whose
`screen_papers` does not return `ScreeningBatchResult` (the pre-merge interface), and exits
with status 2 if the provenance reports a temperature other than 0.0 (override with
`--allow-nonzero-temperature`). `DEEPSEEK_API_KEY` is exported from the repo-root `.env` at
run time and never printed or written. The claims runners have the same `--dry-run`
(lexical stand-in) and the same guards. Latency is measured by `common.Stopwatch`, whose
`seconds` is a live property (an earlier implementation froze it at 0.0 when read inside the
`with` block); the tests drive both runners with a screener/verifier that sleeps and assert
the recorded `latency_s`.

## Datasets and licences

| Dataset | Domain | Records / included | Gold label | Licence |
|---|---|---|---|---|
| `Nagtegaal_2019` (SYNERGY v1) | public administration / behavioural science | 2,019 / 101 final, 392 title-abstract | `label_abstract_screening` | CC0 |
| `van_de_Schoot_2017` (SYNERGY v1) | psychology (PTSD trajectories) | 6,189 / 43 final, 388 title-abstract | `label_abstract_screening` | CC-BY 4.0 |
| `Smid_2020` (SYNERGY v2) | social-science methodology (SEM simulation) | 2,627 / 27 final | `label_included` | CC0 (SYNERGY collection) |

SYNERGY: De Bruin et al. 2023, DOI 10.34894/HE6NAQ (collection licence CC0; per-dataset
licences from the v1 index). Files are downloaded from the asreview GitHub raw endpoints;
nothing is redistributed here. Route v2 datasets are de-duplicated by OpenAlex id and rows
without any identifier are dropped (`<id>.fetch.json` records the counts). Route v1 datasets
are *hydrated* for the WoS gate analysis: every record with any positive label
(`--hydrate included`) is looked up by DOI, then by exact normalised title, in OpenAlex or -
when the shared-IP OpenAlex daily budget is spent - in Crossref (`--lookup crossref`); the
service, date and resolution counts are recorded in `<id>.fetch.json` and in the protocol's
`dataset_source.hydration`. Hydration never changes title or abstract text. **Title-only
matches carry a same-title risk** (a comment, reprint, erratum or conference abstract that
repeats the title would attribute a wrong ISSN): a title match must therefore be within
+-1 year of the export's year when the export has one (`match_method = title_exact_year`;
van_de_Schoot_2017 has a year for 6,173 records, Nagtegaal_2019 has none), and hits in
re-publishing series (`fetch_synergy.is_reprint_venue`: "Yearbook of ..." container titles)
are skipped. A malformed export DOI (`fetch_synergy.is_valid_doi` false, e.g.
`10.1037.a0037593`) is replaced by the Crossref DOI of the title-matched work and kept as
`doi_export`. The committed counts come from one hydration run per dataset under exactly
these rules (van_de_Schoot_2017 re-run in full on 2026-09-03: 112 by DOI + 219 by title and
year = 331/388 included records resolved, 42/43 final inclusions; Nagtegaal_2019 297/392,
title only). An earlier title-only pass had resolved two more van_de_Schoot records (349,
899) whose Crossref year lies 2 and 5 years from the export year; the year rule leaves them
unresolved, so the E1-d lower bound moved from 0.822 to 0.817 while the upper bound stayed
at 0.964 and the resolved-only point estimate stayed at 0.958. A
seeded random sample of 20 title matches per dataset was read record by record against the
matched Crossref record (`spot_check_hydration.py --judgements ...`; per-record
`judgement` / `judged_by` / `judged_on` in the committed
`screening/protocols/spot_checks/<id>.spot_check.json`, summary in each protocol's
`dataset_source.hydration.spot_check`): 20/20 and 20/20 correct after the reprint-venue
rule (a sample of 40 title-only matches had found 3 *Yearbook of Psychiatry and Applied Mental
Health* reprints, which motivated the rule). 20/20 is a sample, not a census: its
95 % Wilson interval is [0.839, 1.000], so up to about one in six of the 297
Nagtegaal_2019 title-only matches (and of the 219 van_de_Schoot_2017 title-and-year
matches) could still be a same-title mismatch, an error the E1-d intervals do not carry.
The 2026-09-03 judgements were produced by an automated pass and have not been verified by
the authors of this software.

**Label policy.** The evaluated task is title-and-abstract screening, so the gold label is
`label_abstract_screening` when the export has it (the human screeners' title/abstract
decision); recall against the final inclusions (`label_included`) is reported as a secondary
number. `Smid_2020` carries only `label_included`, so its precision is a lower bound and
`summarize.py` says so. Records without an abstract are screened on the title alone (the
product shows "(no abstract)"); their share is reported per dataset and among the false
negatives, and records with neither title nor abstract are counted separately (`has_title`,
"Missing title (n)" in Table E1-b; `Smid_2020` has 33 such works, none of them included). The
production prompt builder truncates every abstract to 10,000 characters -- this is the
product's behaviour and is evaluated as is.

The gold labels are the source reviews' own screening decisions as distributed by
SYNERGY, not a re-annotation for this evaluation; SYNERGY publishes no per-record
inter-screener agreement, so records counted here as LLM false positives or false
negatives include human label noise and review-specific screening thresholds, and this
note is printed under Table E1-a (`summarize.label_field_note`). `van_de_Schoot_2017`'s
protocol (`inclusion_criteria[4]`) records a deliberately over-inclusive title/abstract
stage ("if there is any doubt the study is included"), so precision is not compared
across datasets here; note that this criterion is also passed to the screener in the
prompt (`protocols/README.md`: the criteria lists go unchanged to `screen_papers`), so
it raises this dataset's recall as well, and recall is read as a capability number only
within a dataset -- across runs A and B and against the baselines on the same records --
not as a ranking of the three datasets.

**Record order and resume.** `run_screening.py` shuffles every dataset deterministically
before batching (`random.Random(f"{dataset}:{seed}")`, `--record-order-seed` default
20260902, stored as `record_order_seed` in the meta file). SYNERGY exports list the
inclusions first, so file-order batches of 10 would be label-sorted, which production never
sees. The order is identical for runs A and B and across resumes; `--limit` applies after the
shuffle. Resume is at **record** level: batches are rebuilt from the same order and only the
records of a batch that are not yet in the results file are sent, so a batch left half-done
by a `--limit` smoke run is completed with a smaller call and no record is ever re-sent or
re-billed. Rows carry a `call_id` shared by the rows of one API call; the meta file counts
`n_calls` next to `n_batches`, and tokens, cost and latency are aggregated per call.

## Metrics

With TP/FP/TN/FN counted on the screened records (records the screener could not process are
excluded and reported as "unscreened"), N = TP+FP+TN+FN:

* recall = TP / (TP+FN); precision = TP / (TP+FP); F1 = 2PR / (P+R); specificity = TN / (TN+FP);
  inclusion rate = (TP+FP) / N; prevalence = (TP+FN) / N.
* WSS@R = (TN+FN)/N - (1-R), work saved over sampling at recall R (Cohen AM et al. 2006,
  J Am Med Inform Assoc 13:206-219, DOI 10.1197/jamia.M1929). For a fixed binary decision it
  is reported at the achieved recall and, when recall >= 0.95, at R = 0.95; for the TF-IDF
  ranking WSS@95 is computed by screening in descending similarity order.
* Cohen's kappa = (po - pe) / (1 - pe) between runs A and B on records screened in both
  (Cohen J 1960, Educ Psychol Meas 20:37-46, DOI 10.1177/001316446002000104). When pe = 1
  (both runs constant and identical) kappa is undefined: the harness stores `null`, sets
  `agreement_AB.kappa_undefined`, and the tables print `-` with a footnote.
* **Padded decisions.** Production pads any decision the model failed to return with
  INCLUDE and the visible reason `"no decision returned"`. The harness stores those rows as
  production returned them, flags them (`padded_include`), counts them (`n_padded_include`)
  and reports the metrics twice: as returned (Table E1-a) and with padded rows treated as
  unscreened (Table E1-b, `metrics_padded_as_unscreened`). `figure4_screening.json` carries
  the padded-as-unscreened variant as a "LLM run X (padded as unscreened)" row whenever such
  rows exist, for programmatic use; Fig. 4 itself does not plot a fifth bar series for it and
  instead states the padded-as-unscreened recall/precision in its footnote, sourced from that
  same row.
* The TF-IDF baseline uses cosine similarity between each record (title + abstract) and the
  **research question** (`--query rq`, the default, as defined in the revision design); the
  variant with the inclusion criteria appended (`--query rq+criteria`) is available and the
  choice is recorded as `query_mode` in `<id>_baselines.json` and in the summary label. It
  includes exactly as many records as the LLM run (k matched) and is evaluated on the records
  the LLM actually screened (`predicted` not null), so precision/recall are directly
  comparable; include-all has recall 1 by construction. `--k-basis screened_in` matches k to
  the run's retained (INCLUDE or NEEDS_REVIEW) count instead of its INCLUDE count, the basis
  `results/v3/<id>_baselines.json` uses because all three of its protocols name a full-text
  inclusion criterion, which leaves the INCLUDE count at zero.
* **No confidence intervals.** All figures in Tables E1-a/b/d, and every column of Table
  E1-e except retained recall's own 95 % Wilson interval, are point estimates on the stated
  n with no confidence interval reported; on the 60-item HSS set (E2-b) the 95 % Wilson
  interval of an accuracy of 0.917 is about +-0.07, and Smid_2020 (27 included records) is
  also a small dataset, so a difference of a few points between systems, datasets or runs A
  and B is not interpreted as a difference.

### The shipped screening rule

The screener defaults to INCLUDE: a record is excluded only when the shown title and
abstract themselves fail a numbered criterion, and an EXCLUDE must name that criterion and
quote a phrase copied character for character from the shown text; a record the shown text
cannot decide, and a criterion that can only be judged from the full text, is NEEDS_REVIEW
rather than a soft EXCLUDE. Two further checks run before a record is treated as a genuine
INCLUDE. First, a deterministic check: a record whose source marks it a non-article type
(a book, a reference entry, a table of contents, and similar) is routed to NEEDS_REVIEW
without another model call. Second, only the records that survive the first check -- never
the whole batch -- reach the same model the batch pass uses, but with reasoning left on (at
effort high, the provider's own documented default since the request leaves the effort
unset) instead of disabled, and asked directly whether the shown text establishes
the population, the outcome and the study type
the review's own criteria name; any answer of "not established" also routes the record to
NEEDS_REVIEW. A record standing as INCLUDE during the search is provisional until this
check runs: the second pass itself runs only once, for the whole job, after every search
round has finished (not once per round, and not once per original batch of ten), over every
provisional INCLUDE the whole job produced, so a live search's own many rounds are never
slowed by the judge and a round's own stopping decision sees a provisional INCLUDE exactly
as it always did. Judged together as one concurrent stage, five records per call, up to
`screener_second_pass_concurrency` (default 8) calls in flight at once; a call that fails or
times out routes only its own small chunk to NEEDS_REVIEW with `second_pass_unavailable`,
and the stage stops launching new calls once it has run for
`screener_second_pass_stage_budget_seconds` (default 1800s, entirely separate from the
search's own time budget) rather than let the judge overrun -- so the stage's own wall-clock
ceiling is fixed at 1800s plus at most one further already-launched 420s call (about 37
minutes), never open-ended: at the shipped defaults and this freeze's own measured per-call
latencies (7.4 to 55.2s), about 250 candidates (50 calls, 7 concurrent waves) is expected to
take on the order of 6 minutes typically, comfortably inside that ceiling, and in an
unlucky run whose calls run long, the stage still stops at the same ~37-minute ceiling,
having launched only as many of the 50 calls as fit; its own last few chunks (on the order
of the final 10, about 50 candidates) are never sent at all and are routed to
NEEDS_REVIEW with `second_pass_unavailable` for a human to read instead. Every call's own
provenance record carries the model DeepSeek actually served and its system fingerprint,
whichever model was configured. Every one of these checks
can only move a record from INCLUDE to NEEDS_REVIEW -- none of them can produce an EXCLUDE,
and none of them can promote a record into INCLUDE. One structural consequence follows from
this rule on its own: when a review's own protocol defers even one inclusion criterion to
full-text reading, the criterion cannot be confirmed from a title and abstract, so no record
can ever clear it at this stage -- the screener queues such a record for a human to read
rather than guessing. This is by design: the screener prioritises routing a genuinely
undecidable record to a reviewer over shipping a confident-looking guess.

The model also returns a population/subject/outcome anchor ledger (a verbatim quote or an
explicit absence for each slot) on every INCLUDE, but the guard's own check against that
ledger ships off: no production or harness caller passes `anchored_slots` to
`apply_decision_guard`, so it defaults to empty and never demotes an INCLUDE on that basis.
The inclusion-only second pass above is what polices an INCLUDE's grounding today; the
ledger is still recorded on every decision so a future run can turn the check on without a
further model call.

### Final-inclusion report (Table E1-e)

Table E1-a is scored on each dataset's own primary label. Table E1-e reports five figures
against the review's final inclusions specifically, on every dataset, regardless of which
label a dataset's other tables use as primary, because a screening pipeline's practical value
is what it does to the review's own inclusion set: **retained recall**, the share of final
inclusions the screener kept as INCLUDE or NEEDS_REVIEW (i.e. did not exclude outright), with
a 95 % Wilson interval; **screened-in share**, the fraction of the corpus routed to INCLUDE or
NEEDS_REVIEW; **reading saved**, work saved over sampling (Cohen et al. 2006) evaluated at
that retained recall rather than at the INCLUDE-only recall Table E1-a's own WSS columns use,
i.e. the share of the corpus a reviewer never has to open because the screener excluded it
outright; **auto-inclusion precision**, the share of INCLUDE decisions that are a genuine
final inclusion, with its own n, printed as "none, n = 0" when no record was auto-included at
all (the direct consequence of the routing rule above, on any protocol naming a full-text
inclusion criterion); and **queue precision**, the share of NEEDS_REVIEW rows that are a
genuine final inclusion, i.e. how often the records a reviewer actually reads contain one.
The table also reports the run-to-run agreement between A and B on the three-way INCLUDE/
EXCLUDE/NEEDS_REVIEW status (Cohen's kappa), once per dataset.

## Prices, cost and tokens

DeepSeek list prices (USD per 1M tokens) fetched on 2026-09-02 from
<https://api-docs.deepseek.com/quick_start/pricing> and stored in `common.DEEPSEEK_PRICES`:

| model | input, cache hit (off-peak / peak) | input, cache miss (off-peak / peak) | output (off-peak / peak) |
|---|---|---|---|
| deepseek-v4-flash | 0.007 / 0.014 | 0.22 / 0.44 | 0.66 / 1.32 |
| deepseek-v4-pro | 0.022 / 0.044 | 0.66 / 1.32 | 1.98 / 3.96 |

These are the prices used to compute every cost figure in this document; the provider's list
price read on 2026-09-16 is lower (see `README.md`'s requirements table), but the runs
recorded here were made and priced against the 2026-09-02 figures above.

Peak hours are 01:00-04:00 and 06:00-10:00 UTC, Monday to Friday; all other hours are off-peak
(half price). The name `deepseek-flash` configured in the product (the provider's current
name for the deployment previously aliased `deepseek-chat`) is priced as `deepseek-v4-flash`;
runs must confirm the served model from the `model_reported` column. Every call is
timestamped (`called_at`, ISO-8601 UTC) right before the request and priced with
`common.call_cost` in its own tier. The screening interface exposes only input/output token
totals (no cache-hit split), so all input tokens are billed at the cache-miss rate and the
screening cost is an **upper bound at list price** (`cost_basis` in the meta file); the claims
interface exposes `cache_read_tokens`, which are billed at the cache-hit rate. The served
model runs in thinking mode by default, so output tokens as reported by the API may include
reasoning tokens; they are recorded exactly as reported. `--price-input/--price-output` add a
flat estimate in `total_cost_flat` without replacing the tiered figure.

**Latency.** Latency medians and p90 are wall-clock per call (per batch for E1, per item for
E2) measured while up to `--concurrency` calls (default 8; production uses
`analysis_concurrency = 4`) were in flight, so they include provider-side queueing. The
concurrency is stored in every meta file and printed in Tables E1-c / E2-c.

**Cost and time.** The committed runs' actual cost and latency are the figures reported in
Table E1-c (`screening/results/v3/summary.md`) and Table E2-c (`claims/results/v6/summary.md`):
US$3.57 total for the six screening pass-1 runs (`evaluation/screening/FREEZE_v3.md` section
3) and US$0.58 total for the four SciFact/HSS runs (`evaluation/claims/FREEZE_DIGESTS_v6.md`).

## Command sequence (E1)

```
# data (network, no LLM). --lookup crossref when OpenAlex answers 429 (daily budget)
evaluation/.venv/Scripts/python evaluation/screening/fetch_synergy.py --dataset Nagtegaal_2019 --route v1 --hydrate included --lookup crossref
evaluation/.venv/Scripts/python evaluation/screening/fetch_synergy.py --dataset van_de_Schoot_2017 --route v1 --hydrate included --lookup crossref
evaluation/.venv/Scripts/python evaluation/screening/fetch_synergy.py --dataset Smid_2020 --route v2
# LLM runs: PAID (system python; DEEPSEEK_API_KEY from .env, never printed). Without
# --confirm-cost the runner prints the batch count and cost bound and exits 3.
# --prompt-version defaults to v3, the shipped prompt; --results-dir defaults to
# results/v3 (the directory of record), matching baselines.py/summarize.py/
# wos_gate_effect.py's own default, named explicitly below for clarity.
python evaluation/screening/run_screening.py --dataset <id> --run A --dry-run --results-dir results/v3           # no LLM
python evaluation/screening/run_screening.py --dataset <id> --run A --limit 30 --confirm-cost --results-dir results/v3  # paid smoke
python evaluation/screening/run_screening.py --dataset <id> --run A --confirm-cost --results-dir results/v3
python evaluation/screening/run_screening.py --dataset <id> --run B --confirm-cost --results-dir results/v3
# --second-pass-model overrides the backend's own screener_second_pass_model default
# (deepseek-flash) for this run only; every run's own meta file records the model
# actually configured next to the model_reported/system_fingerprint DeepSeek served.
# repair batches that failed after --max-attempts (rows with predicted = null are re-screened)
python evaluation/screening/run_screening.py --dataset <id> --run A --retry-failed --confirm-cost --results-dir results/v3
# baselines, gate, summary
evaluation/.venv/Scripts/python evaluation/screening/baselines.py --dataset <id> --run A --results-dir results/v3
evaluation/.venv/Scripts/python evaluation/screening/wos_gate_effect.py --dataset <id> --results-dir results/v3
evaluation/.venv/Scripts/python evaluation/screening/summarize.py --results-dir results/v3
```

Expected sizes per run (batches of 10): Nagtegaal_2019 202, van_de_Schoot_2017 619,
Smid_2020 263; two runs each. Runs are resumable at record level (re-run the same command:
only records missing from the results file are sent, a half-done batch is completed with a
smaller call, and the meta file counts `n_calls` next to `n_batches`); a batch that still fails after `--max-attempts` (default 3) is written with `predicted = null`
and counted as unscreened. Such rows are kept by a plain re-run; `--retry-failed` removes
them from the results file (atomic rewrite) and screens them again, and the meta file records
`retried_failed` (also per session). Smoke test without an LLM:
`python evaluation/screening/run_screening.py --dataset Nagtegaal_2019 --run A --dry-run --limit 30 --results-dir results/v3`.

Outputs in `screening/results/v3/` (each script's own `--results-dir` already defaults there): `<id>_run<A|B>.jsonl` + `.meta.json`, `<id>_baselines.json`,
`<id>_wos_gate.json`, `<id>_false_negatives.md`, `summary.json`, `summary.md` (tables E1-a
performance, E1-b secondary, E1-c provenance/cost/latency, E1-d WoS gate),
`figure4_screening.json`. No categorisation file is written for a fresh run. `summarize.py`
reads a `<id>_fn_categories.csv` from the results directory when one is committed there and
then reports its category counts in `summary.md`; `results/v3/` ships none, so its false
negatives are published with the screener's own stated reason and no coded category.

**False-negative categorisation.** This package lives at `results/v3/fn_categorisation/` and
codes the 923 false negatives of an earlier screener run, whose own results directory is no
longer shipped. It is kept beside `results/v3/` so that the coding, its taxonomy and its
checks stay published, and `summarize.py` resolves it relative to `--results-dir` (see
`_fn_categories_caption`/`_fn_blind_check_module`). None of its codes is carried over to the
false negatives of the shipped v3 runs, which are a different and larger set of 1,614 rows:
the cue-word rules below leave 648 of those rows unmatched and flag 971 low-confidence, and
the disclosure below is written for the 923 rows it was produced on.
`results/v3/fn_categorisation/categorise_fn.py` assigns each of the 923
false-negative reasons one of five taxonomy codes (`FN-ABS` missing or truncated abstract,
`FN-POP` population or clinical construct judged out of scope, `FN-DESIGN` study design or
publication type judged ineligible, `FN-DEF` intervention or method not matching the
protocol's technical definition, `FN-OTHER` off-topic dismissal or no reason recorded; full
definitions and cue words in `results/v3/fn_categorisation/fn_taxonomy.md`) by matching the
taxonomy's own cue-word list against the screener's stated reason, in the taxonomy's
precedence order. The rule matched 903 of the 923 rows. The category column was assigned by
an automated language-model pass for the remaining 20 rows (recorded in
`results/v3/fn_categorisation/fn_overrides.json`,
`"<dataset>|<run>|<record_id>": "<code>"`, which always wins over the rule), for every one
of the 372 rows the rule flagged low-confidence or could not match at all, and for a
seed-`20260905` random 10 % audit sample of the high-confidence rule matches (55 rows, all
55 agreeing with the rule). A second, independent automated pass then read a further 73
rows with the assigned category withheld (drawn stratified by dataset: 30 of 446
Nagtegaal_2019 rows, 30 of 464 van_de_Schoot_2017 rows and all 13 Smid_2020 rows, a census;
29 of the 73 overlapped the first pass's 372 rows; none were among the 20 overrides) and
agreed with the committed category on every row (kappa 1.0, recomputed from the shipped
sample and labels by
`results/v3/fn_categorisation/fn_blind_check/fn_blind_check.py`). The original draw's
random seed was not recorded, so the committed row list in
`results/v3/fn_categorisation/fn_blind_check/sample_blind.tsv` is shipped as the frozen
sample rather than as a seed that could regenerate it. The authors did not check these
category assignments.

The WoS gate analysis reads the four licensed Web of Science collection CSVs from the
repository root when present and writes aggregate counts only; it exits 0 when they are absent.
Table E1-d reports the gate's recall ceiling over the human-included records as an interval
`lower-upper` next to the resolution fraction (`ISSN/n`): a record with no ISSN has an
unknown venue, so the lower bound counts every unresolved record as outside WoS (in WoS /
all included) and the upper bound counts every unresolved record as inside WoS
((in WoS + unresolved) / all included); a fully resolved set prints a single value. The
share of *resolved* records only (in WoS / resolved) is reported alongside in parentheses as
a point estimate, not a bound, since it assumes the unresolved records are in WoS at the
same rate as the resolved ones. All three are stored in `summary.json`
(`wos_gate_table.share_lower_bound` / `share_resolved_only` / `share_upper_bound`) and as a
`"WoS gate"` row in `figure4_screening.json`. With the 2026-09-03 hydration the intervals
are Nagtegaal_2019 0.640-0.883 (resolved-only 0.845; 251 of 392 included, 297 resolved),
van_de_Schoot_2017 0.817-0.964 (resolved-only 0.958; 317/388, 331 resolved) and Smid_2020
0.963-1.000 (resolved-only 1.000; 26/27, 26 resolved). The table note states the same-title
risk of title-only matches.

## Claim verification (E2)

**Purpose.** Measure the production claim-verification agent
(`app/agents/claim_verification_agent.py`, prompt `CLAIM_VERIFICATION_PROMPT_VERSION`,
temperature 0.0) on (a) a public scientific benchmark and (b) a small humanities/social-science
full-text set built for this revision, and compare it with a lexical baseline. The harness
calls the unchanged production code path (`format_verification_prompt` ->
`agent.run(prompt, deps=AnalysisDependencies(...))` -> `provenance_from_run`) and records per
call the model the provider reported, its system fingerprint, response id, token counts
(input, output, cache-hit) and call time.

**SciFact (E2-a).** Wadden et al. (2020), *Fact or Fiction: Verifying Scientific Claims*,
EMNLP, DOI [10.18653/v1/2020.emnlp-main.609](https://doi.org/10.18653/v1/2020.emnlp-main.609).
Licence: claims CC BY 4.0; abstracts (S2ORC) ODC-By 1.0
(`https://github.com/allenai/scifact/blob/master/LICENSE.md`). `fetch_scifact.py` downloads
the release tarball (3,115,079 bytes), extracts it with a path-safe filter, verifies the line
counts (dev 300 / train 809 / test 300 / corpus 5,183) and writes `data/scifact.fetch.json`.
The dev set's 300 claims cite 340 claim-document rows (138 SUPPORT, 71 CONTRADICT,
131 NOT_ENOUGH_INFO); each row is one item whose single "full-text chunk" is the abstract's
sentences joined by one space. The 340 rows cover 339 distinct claim-document pairs: dev claim
1245 lists document 7662395 twice in its own `cited_doc_ids`, so `build_items` emits the item
twice (`item_id` "1245:7662395" appears twice in `scifact_run<A|B>.jsonl`) and both copies are
sent to the model, billed and counted in every metric above (both copies score `unsupported`
in both runs, so the numeric effect is a rounding-level shift in run-to-run agreement).

*Scoring map.* SUPPORT -> `verified`; CONTRADICT and NOT_ENOUGH_INFO -> `unsupported`. The
agent may also answer `needs_nuance`, `no_full_text` or fail (`error`); these are never a gold
class. In the *strict* table they count as a miss for the gold class and never as a hit
(`common.per_class_prf`); `needs_nuance` answers are additionally reported by gold class. The
*lenient* view is binary: `verified` vs everything else, computed over model-answered rows
only (error rows are excluded from the confusion counts and reported as `n_errors`, so a
failed call is never a "correct rejection"; the HSS `verified_binary` view does the same).
Both runs (A, B) use identical inputs; A/B agreement pairs only items that carry a model
decision in both runs (deterministic `no_full_text` items and error rows are excluded and
counted as `n_excluded_deterministic` / `n_excluded_error`) and is reported as percent
agreement and Cohen's kappa (status level and binary); when both runs are constant and
identical kappa is undefined and printed as `-` with the same footnote as in E1.

*Label-mapping sensitivity (Table E2-f).* The scoring map above is a choice: collapsing
CONTRADICT and NOT_ENOUGH_INFO onto one `unsupported` class discards the distinction between
a refuted claim and an unevidenced one, so `claims/summarize.py` rescores every SciFact run
under three mappings and reports per-class P/R/F1, macro-F1, CONTRADICT recall and the count
of gold-CONTRADICT rows predicted `verified` for each: *strict* (the mapping above),
*lenient* (a predicted `needs_nuance` counts as `verified`, everything else unchanged), and
*three-way* (available only when a row carries the per-assertion `verdict` field prompt v2
adds, absent from every v1 row), which splits a predicted `unsupported` row by whether any
assertion was itself verdict `contradicted` and scores CONTRADICT against that split bucket
instead of the class it otherwise shares with NOT_ENOUGH_INFO. The sentence the manuscript
may state: "Mapping choice moves the headline numbers only at the margin the model's own
needs-nuance and per-assertion signals create; CONTRADICT recall and the count of
gold-CONTRADICT rows predicted verified are the figures a reviewer should read as the safety
check, not macro-F1 alone, and both are reported for every mapping in Table E2-f."

*Quote fidelity.* Share of `verified` answers whose `evidence_quote` is a verbatim substring
of the chunk text after whitespace normalisation (case-sensitive); a case-insensitive share
and the share of null quotes are reported alongside.

**HSS set (E2-b).** 60 claims over 21 open-access applied-linguistics articles
(`hss_test_v3_sources.json`: DOI, journal, ISSN, OpenAlex id, licence where recorded, access
date) drawn from Eurasian Journal of Applied Linguistics (4), Colombian Applied Linguistics
Journal (4), Studies in Second Language Learning and Teaching (3), Journal of Language and
Education (3), Technology in Language Teaching and Learning (2), TESL-EJ (2), TESL Canada
Journal (1), JALT CALL Journal (1) and Apples: Journal of Applied Language Studies (1). Papers
were acquired with the production pipeline (`UnpaywallClient.lookup` -> `fetch_pdf_from_url`
-> `extract_text_from_pdf` -> `chunk_text`; kept only with >= 15,000 characters and >= 3
chunks). Candidate sentences are ranked deterministically
(`common.select_candidate_sentences`, results/discussion/conclusion chunks first) and assigned
round-robin over papers to six construction rules, one claim per sentence, no sentence reused
(`build_hss_set.py`):

| rule | n | construction | expected status |
|---|---|---|---|
| verbatim | 8 | claim = sentence | verified |
| paraphrase | 8 | fixed ordered substitution table (>= 1 substitution; reporting verbs, connectives and generic nouns only; technical terms are never substituted) | verified |
| altered | 18 | direction flip, first number x2 (percentages capped at 99), or quantifier flip | unsupported |
| over_specified | 10 | an additional, unsupported modifier appended to a sentence that already reports an empirical finding | needs_nuance |
| wrong_paper | 8 | sentence from paper X, chunks of paper Y | unsupported |
| no_full_text | 8 | sentence with no chunks | no_full_text (deterministic; no model call) |

`needs_nuance` answers on paraphrase items are reported in their own column of Table E2-b and
are not scored as correct. Sentences that are not claims (PDF artefacts, reference-list
entries, section headings, questions, attributed findings from another study, and quoted
passages) are rejected by rule before assignment; the committed set (`hss_test_v3_claims.jsonl`,
`hss_test_v3_claims.build.json`) is the output of these rules. Labels are by construction, not
by expert annotation.

As an independent check on those labels, the 60 items were annotated independently three
times, each annotation blind to the others, and adjudicated in a fourth pass
(`claims/annotation/`; method, blinding checks and agreement statistics are in
`claims/annotation/PROVENANCE.md`). These are large-language-model annotations (Claude Opus,
checking a DeepSeek verifier), not expert human annotations. The label every reported accuracy
figure is scored against is now the three human reviewers' majority verdict
(`results/v6/human_labels.json`; see "Claim-verification labels (v6)" below), not this
construction-rule or model-annotation label; score the verifier against the model annotation
instead with `claims/summarize.py --labels
claims/annotation/hss_annotation_adjudicated_v3.csv` (adds Table E2-d; no LLM or network call,
see `claims/annotation/README.md`).

*Curation.* The 60 items were read by an automated review pass, not by the authors of this
software; the authors have not verified this set independently of the human-review workbook
above. `hss_veto.json` is a safety net assembled by that same automated review, honoured by
the builder and recorded (with reasons) in `hss_test_v3_claims.build.json`. Every item except the
eight `no_full_text` ones is verified with *all* chunks of its paper, in production order.
`hss_test_v3_claims.jsonl`, `hss_test_v3_claims.build.json`, `hss_test_v3_sources.json` and
`hss_veto.json` are committed; the full-text cache (`data/hss_test_v3_fulltext/`) is not.

**Lexical baseline.** `baseline_lexical.py`: maximum token-Jaccard between the claim and any
chunk sentence, `>= 0.5` -> `verified`, otherwise `unsupported`; items without chunks ->
`no_full_text`. `--set scifact` (default; `--match-run` restricts it to the items of an LLM
run file) writes `results/v3/scifact_baseline.json` (the default `--results-dir`); `--set hss`
scores the HSS items exactly as `run_hss.py` loads them and writes
`results/v3/hss_baseline.json` with per-rule accuracy and the
binary verified metrics (Table E2-b, `figure4_claims.json`); the binary view is computed over
the items with chunks only, the no-chunk items being counted as `n_excluded_deterministic`,
so that it shares the denominator of the LLM rows (n = 25). On the HSS set the baseline marks
single-word alterations `verified` (Jaccard close to 1), which is the point of including it.

**No confidence intervals.** All figures in Tables E2-a/b/c/d are point estimates on the stated
n with no confidence intervals; E2-b's accuracy is on n = 60 (binary view n = 52, 16
positives) and the 95 % Wilson interval of an accuracy of 0.917 on n = 60 is about +-0.07
(Table E2-h), so a difference of a few points between the LLM run and a lexical baseline, or
between runs A and B, is not interpreted as a difference.

**Cost and provenance.** Meta files record `model_configured` (`deepseek-flash`), the distinct
`model_reported` values the API returned (expected `deepseek-flash`, priced as
`deepseek-v4-flash`), system fingerprints,
`temperature`, `prompt_version`, `concurrency`, token totals including cache-hit tokens,
`retried_failed`, and `total_cost` = sum over model-answered rows of
`common.call_cost(model_reported, called_at, input, output, cache_read)`. Deterministic
(`no_full_text`) and error rows carry no tokens and no cost. Error rows
(`predicted_status = "error"` after `--max-attempts`) are kept by a plain re-run and
re-attempted with `--retry-failed`.

**Commands**

```
evaluation/.venv/Scripts/python evaluation/claims/fetch_scifact.py
python evaluation/claims/build_hss_set.py --skip-fetch          # rebuild from the cache (no network)
python evaluation/claims/build_hss_set.py                       # full acquisition: Unpaywall/Crossref/PDF, no LLM
python evaluation/claims/run_scifact.py --run A --dry-run                    # no LLM
python evaluation/claims/run_scifact.py --run A --limit 30 --confirm-cost    # PAID smoke
python evaluation/claims/run_scifact.py --run A --confirm-cost ; ... --run B --confirm-cost   # PAID
python evaluation/claims/run_hss.py --run A --confirm-cost ; ... --run B --confirm-cost       # PAID
python evaluation/claims/run_scifact.py --run A --retry-failed --confirm-cost  # repair error rows
evaluation/.venv/Scripts/python evaluation/claims/baseline_lexical.py --match-run evaluation/claims/results/v6/scifact_runA.jsonl
evaluation/.venv/Scripts/python evaluation/claims/baseline_lexical.py --set hss
evaluation/.venv/Scripts/python evaluation/claims/summarize.py
# optional: score the HSS runs against the blind model annotation (Table E2-d; no LLM call)
evaluation/.venv/Scripts/python evaluation/claims/summarize.py --labels evaluation/claims/annotation/hss_annotation_adjudicated_v3.csv
```

Every runner accepts `--dry-run` (lexical stand-in, results under `claims/data/_dryrun/results/`,
no LLM) and is resumable; without `--dry-run` a runner refuses to start unless `--confirm-cost`
is given (see "Paid runs: the cost gate"); `summarize.py --results-dir <dir>` writes `summary.json`,
`summary.md` (Tables E2-a/b/c) and `figure4_claims.json`; `summarize.py --labels <csv>` additionally
scores the committed HSS run(s) against a blind-annotation CSV (Table E2-d; no LLM or network
call) without altering any other block, see `claims/annotation/README.md`.

### Prompt versions and the dev / test split

The claim-verification prompt used for every reported result is
`CLAIM_VERIFICATION_PROMPT_VERSION sha256:49fcfbfaf2f6`, together with `GUARD_DIGEST
8a6c833ffc329f89`, `QUOTE_RELOCATION_VERSION 0269a13351c0da7d` and
`VERIFICATION_POLICY_VERSION 667bcc9209126ddc` (see `evaluation/claims/FREEZE_DIGESTS_v6.md`
for the full digest table). The screening prompt used for every reported result is
`SCREENER_PROMPT_VERSION sha256:9bc741046cd6` (see `evaluation/screening/FREEZE_v3.md`).

Both prompts were fixed before the runs reported in `results/v6/summary.md` and
`results/v3/summary.md`. The claim-verification prompt was developed against the SciFact
**train** split (120 claim-document pairs, stratified 40 SUPPORT / 40 CONTRADICT / 40
NOT_ENOUGH_INFO, `claims/scifact_train_dev120.json`) and a constructed set of applied-
linguistics claims built from papers disjoint from the HSS test set
(`claims/hss_dev_claims.jsonl`). The screening prompt was developed against two SYNERGY
reviews disjoint from the three reported reviews, `Fong_2021` and `van_Dis_2020` (see
`evaluation/screening/protocols/README.md`).

`results/v6/summary.md` and `results/v3/summary.md` are the results of record; every reported
figure is in one of those two files.

## Human-review workbook

`evaluation/claims/export_review_sheet.py` builds an Excel workbook (`Guide`, `Review`,
`Model_labels`, `Sources`, `Provenance`, plus `Screening` when a screening record is
given and `Delivered` when a `delivered_evidence.json` is given) so a person can
countersign or override the frozen claim-verification and screening runs and judge the
final, delivered text a user actually receives, and `evaluation/claims/score_review_sheet.py`
reads a filled copy back and computes agreement. Interpreter for both: either the
evaluation venv or plain system python (`openpyxl`; system python is needed for a
lone-carriage-return cell to round-trip correctly, since the evaluation venv has no
`lxml`); no backend import, no network, no LLM call in either script. See "Layout" and
"Results directories" above for the versioned results directories a workbook is exported
from.

**Frozen versions a workbook reports against.** Verifier prompt `sha256:49fcfbfaf2f6`
(the cached model answers `results/v6/` replays, now under `results/v6/inputs/`; see
"Results directories" above); screener prompt `sha256:9bc741046cd6`, the prompt
`demo/expected/screening_record.json` was generated against, the current production
screening prompt, and the value this export checks with
`--screening-prompt-version-expect` (see "Screening" above and
`evaluation/screening/FREEZE_v3.md`); the seven code-level guards described in the root
`CHANGELOG.md` (`GUARD_DIGEST 8a6c833ffc329f89`), the quote-relocation step
(`QUOTE_RELOCATION_VERSION 0269a13351c0da7d`) and the quote-repair turn
(`QUOTE_REPAIR_PROMPT_VERSION sha256:7ec7cffe852d`) run under `VERIFICATION_POLICY_VERSION
667bcc9209126ddc`, at the commit recorded on the workbook's own Provenance sheet
(`guard_commit`). The exporter reads the verifier prompt sha from the results directory's
own meta files rather than taking the caller's word for it, and refuses
(`--screening-prompt-version-expect`) a screening record whose own provenance carries a
different screener prompt.

**Export command**, a runnable template that builds a reviewer workbook for the claim
sets above and, when `--delivered-evidence` is given, adds a Delivered sheet for the
promoted run's own delivered text (the key csv is never sent to the reviewer):

```
cd evaluation
python claims/export_review_sheet.py \
  --items claims/hss_test_v3_claims.jsonl claims/real_claims_test.jsonl \
  --cache-dir claims/data/hss_test_v3_fulltext claims/data/real_claims_fulltext \
  --results-dir claims/results/v6 --runs A,B \
  --annotation claims/annotation/hss_annotation_adjudicated_v3.csv \
               claims/annotation/real_annotation_adjudicated_v3.csv \
  --out claims/results/v6/review-workbook.xlsx \
  --key-out claims/results/v6/review-workbook-key.csv \
  --screening-record ../demo/expected/screening_record.json \
  --screening-prompt-version-expect sha256:9bc741046cd6 \
  --demo-run-id 20260916-154140 \
  --delivered-evidence ../demo/expected/delivered_evidence.json \
  --paper-authors ../demo/tools/selected.json
```

`--results-dir claims/results/v6` is every script's own default results directory (see
"Results directories" above). `--cache-dir` reads the gitignored full-text fetch caches
(`claims/data/hss_test_v3_fulltext/`, `claims/data/real_claims_fulltext/`); a fresh clone
does not carry them and must build them first with `claims/build_hss_set.py` and
`claims/harvest_real_claims.py` (both shipped, no LLM call), or pass
`--allow-missing-chunks` to export the workbook without the cached source text. `--out`
and `--key-out` above write inside the clone; the
key csv is a reviewer answer key and must never be sent to the reviewer along with the
workbook. `--guard-commit` is left unset above so it auto-detects the last commit that
touched `backend/app/services/fulltext.py` (see `resolve_guard_commit`); pass it
explicitly only to pin a specific historical commit. `--demo-run-id 20260916-154140` and
`--delivered-evidence ../demo/expected/delivered_evidence.json` name the promoted
baseline run this repository ships; substitute a different run id and its own
`delivered_evidence.json` (for example under `../demo/output/<run id>/`) to review a
different run instead.

`--paper-authors` is required whenever `--delivered-evidence` is given, on this full-export
path and on the `--delivered-only` path described below: it feeds the Delivered sheet's
`paper_authors` column, which the Guide text and the misattribution rule both depend on.

A separate `--delivered-only` mode writes just three sheets, `Guide`, `Provenance` and
`Delivered`, with no item list at all, for a workbook whose only purpose is the delivered
product's own fidelity review:

```
cd evaluation
python claims/export_review_sheet.py --delivered-only \
  --demo-run-id 20260916-154140 \
  --delivered-evidence ../demo/expected/delivered_evidence.json \
  --paper-authors ../demo/tools/selected.json \
  --out claims/results/v6/delivered-workbook.xlsx \
  --key-out claims/results/v6/delivered-workbook-key.csv
```

In this mode `--items`, `--cache-dir`, `--results-dir` and `--key-out` are optional, and
`--annotation`/`--screening-record`/`--screening-record-b` are refused. The workbook id is
computed from the run id and the built rows themselves (`compute_delivered_workbook_id`),
not from an item list and shuffle seed, so re-exporting the same run gives the same id.

`claims/results/v6/inputs/` carries a byte-for-byte copy of the same run/meta files, kept
as the frozen input set the cached verifier replay reads from; either directory produces
the same export.

`--demo-run-id` (else the screening record's own filename, else `"n/a"`) and `export_time`
(computed once per run, from `common.now_iso()`, and never overridable from the command
line) are stamped together on the workbook's own Provenance and Guide sheets, so the two
sheets can never disagree about which run or which export they describe; a re-export
against the same `--out` path overwrites the file and gets a new `export_time` but the
same `demo_run_id` and the same opaque item order (and therefore the same workbook id
and the same key csv), since neither the item order nor the export seed changes between
re-exports of the same inputs.

**Score command**, reading back a filled copy of the workbook the export command above
produced:

```
cd evaluation
python claims/score_review_sheet.py \
  --workbook claims/results/v6/review-workbook.xlsx \
  --key claims/results/v6/review-workbook-key.csv \
  --out claims/results/v6/review-workbook-scores.json \
  --demo-run-id 20260916-154140 --export-time <export time>
```

`--demo-run-id` and `--export-time` are both optional but, given together, make the
scorer refuse (exit 2, no output written) a workbook whose own `Provenance!demo_run_id`
or `Provenance!export_time` does not match exactly. This is deliberately two checks, not
one: `demo_run_id` and the workbook id both depend only on the kept item order and the
export seed, so two exports of the same demo run share both and `--demo-run-id` alone
cannot tell them apart; `export_time` is the one Provenance value that does, and it is
what catches a superseded copy of an earlier export of the same demo run being sent back
instead of the current one. Both values are printed together on the Guide sheet's own
workbook-id line, for a reviewer to copy without opening the Provenance sheet.

**Denominators.** The scorer reports counts, not only rates, next to every accuracy
figure, because more than one subset of the same census is compared:

- `agreement_with_verifier` (Review sheet), overall and `per_set` (`constructed` /
  `real`), each carry `n_items` (every row of that set), `n_judged` (rows whose verifier
  status is not `no_full_text`, i.e. rows a human could actually compare against source
  text) and `n_withheld` (the `no_full_text` rows, evidence withheld by construction and
  excluded from the accuracy figure, reported on their own). `agreement_with_model_labels`
  carries the same three fields, over whichever items an `--annotation` file covers.
- `screening.overall` and `screening.per_status` exclude `unscreened` rows (there is no
  human-comparable tool decision for one) and, separately, any row whose
  `needs_review_reason` is `unanchored_exclude` (a guard-demoted exclusion the Guide sheet
  itself dictates the human's own answer for) from both the headline accuracy/kappa and
  the affected `per_status` block, reporting that split on its own, under an
  `unanchored_exclude` key, in the same `{n, n_correct, accuracy}` shape, so neither
  exclusion silently deflates the number a reviewer would otherwise read as the tool's
  general screening accuracy.
- `quote_is_verbatim_rate` excludes the same `unanchored_exclude` rows from its own
  `{n, n_yes, rate}` denominator and reports them separately, in the same shape, since a
  guard-demoted row's quote was never anchored in the shown text by construction.
  `criterion_is_right_rate` is reported the same `{n, n_yes, rate}` way, with no such
  exclusion, over every sampled exclusion row that carries a criterion.
- The **Delivered headline** (`delivered` in the scored json, printed first on the
  markdown report) is the product-level number: over every row of the Delivered sheet,
  how often the delivered sentence states what its source states
  (`sentence_correct_rate`) and how often the quoted evidence actually supports it
  (`quote_supports_rate`), each with a count and a 95% Wilson interval. `n_judged` is
  `n_rows` minus `n_fidelity_excluded` (a row where the cited source or the quoted
  passage could not be located, or the sentence text itself was not found in the draft);
  an excluded row is never counted toward either rate. The same shape is repeated
  `per_section`, one row per generated section, over that section's own Delivered rows
  only.
- `screen_in_share` (the numbered-criterion and off-topic exclusion strata) is reported
  per stratum as `{n, n_screened_in, rate}` over that stratum's own sampled exclusions
  only, not over the whole Screening sheet.

## Claim-verification labels (v6)

The single workbook the previous section describes carries one person's own countersign or
override of the frozen runs. A separate, later step goes further: three completed copies of
that same exported workbook (one per reviewer, distinguished only by file name, no identity
recorded anywhere in them) are joined and reduced to one majority verdict per item, and that
majority verdict, not a model-authored annotation, is now the label every constructed-set and
real-claims accuracy figure in `results/v6/summary.json`/`summary.md` is scored against.
SciFact keeps its own published gold label throughout; only the constructed (HSS) and real
sets have ever carried a project-authored label to replace.

`evaluation/claims/human_review_labels.py` (system or evaluation-venv python; `openpyxl`; no
backend import, no network, no LLM call) takes the three reviewers' filled Review-and-
Screening workbooks and the workbook key as command-line arguments (`--workbook`, repeated
exactly three times in reviewer order, and `--key`; no default location, and a run with
neither refuses with a plain message), and does this in one pass:

1. Reads the three workbooks and the key (both outside this repository and never committed)
   and builds one majority label per `Review`-sheet item: the status at least two of the
   three reviewers returned, or `no_majority` on a three-way split (none on the current
   data). Inter-rater agreement is reported two ways, the mean of the three pairwise
   percent-agreement rates and Fleiss' kappa, both over the items the reviewers actually judged
   freely (excluding the eight constructed items whose evidence is withheld by construction and
   whose verdict the protocol fixes rather than leaving to judgement).
2. Scores `results/v6/{hss,real}_run{A,B}.jsonl` against that majority label: accuracy with a
   95% Wilson interval, Cohen's kappa, `verified` precision/recall and the falsely-`verified`
   count, per set and per run. Writes `results/v6/human_labels.json` (the majority label and
   each reviewer's own verdict, keyed by the evaluation's own item id; no reviewer identity)
   and `results/v6/human_review_scores.json` (the scoring above).
3. Folds the result into `results/v6/summary.json` (a new `human_review` block, alongside
   the existing `hss_vs_annotation`/`real_vs_annotation` model-annotation blocks) and
   `results/v6/summary.md` (Table E2-h, the label of record); `summary.json` keeps the
   model-annotation blocks for other consumers even though `summary.md` no longer prints
   them as tables.

The three reviewers' own judgement of one promoted demonstration run's delivered sentences
(the `Delivered` sheet: whether a sentence is a faithful reading of its cited source, and
whether the stored quotation supports it) is scored separately, by
`evaluation/claims/delivered_human_scores.py`, the tool of record for that scoring; it writes
`results/v6/delivered_human_scores_run_<id>.json`, a record of that one run, kept for
reference and not part of the claim-set scoring above.

On this data the majority label matches the model-authored label (`annotation/`) on every one
of the 60 constructed items, so the constructed-set figures are unchanged in value and simply
stop being model-authored; the real-claims set is the one where the two label sources part
ways; see `results/v6/human_review_scores.json` and Table E2-h for the current numbers.
