# Screening protocols

One JSON file per SYNERGY dataset, named `<dataset_id>.json` (the same id as in
`data/synergy_index.csv`, e.g. `Nagtegaal_2019.json`). The protocol is what the production
screener receives: `research_question` becomes the `user_query`, and the two criteria lists are
passed unchanged as `inclusion_criteria` / `exclusion_criteria` to
`app.agents.relevance_screener_agent.screen_papers`.

## Schema

| Field | Type | Meaning |
|---|---|---|
| `dataset_id` | string | SYNERGY dataset id; must equal the file stem. |
| `source_review_doi` | string | DOI of the systematic review the labels come from. |
| `research_question` | string | The review question, compiled from the source article (Introduction / Objectives). |
| `inclusion_criteria` | list of strings, or of stage-aware objects (see below) | Eligibility criteria as stated in the source review (Methods). |
| `exclusion_criteria` | list of strings, or of stage-aware objects (see below) | Exclusion reasons as stated in the source review (Methods or PRISMA flow diagram). |
| `primary_label` | `"label_included"` | The primary gold label the acceptance gate is scored on. Present as an explicit key only in `Fong_2021.json`, `Taschner_2024.json`, `Anmarkrud_2021.json`, `Nagtegaal_2019.json` and `van_Dis_2020.json`; `Smid_2020.json` and `van_de_Schoot_2017.json` carry no `primary_label` key and are scored under a documented default (see "Choosing the primary label and `label_field`" below). |
| `label_field` | `"label_abstract_screening"` or `"label_included"` | The label the source review's own title-and-abstract stage produced; reported as a sensitivity number, not the primary metric. |
| `notes` | string | Where each item was taken from, how the question was compiled, and anything derived rather than quoted. |

Additional keys (e.g. `source_review`, `dataset_source`, `criteria_provenance`, `derived_from`)
are allowed and are kept in `Protocol.extra`; `common.load_protocol()` validates the required
fields and rejects unknown `label_field` values. `python -m pytest evaluation/tests -q` loads
every committed protocol.

### Stage-aware criteria

Some eligibility criteria cannot be decided from a title and abstract: that an instrument was
administered, that an effect size can be computed, that a diagnostic interview was given.
Showing those criteria to the screener unmarked lets an abstract's silence read as failure. Each
element of `inclusion_criteria`/`exclusion_criteria` may therefore stay a plain string (implicitly
`{"stage": "abstract"}`) or become an object:

| Key | Type | Meaning |
|---|---|---|
| `text` | string | The criterion text itself (what a plain string element already carries). |
| `stage` | `"abstract"` or `"full_text"` | The stage at which the criterion can be decided. Defaults to `"abstract"`. At title-and-abstract stage a `"full_text"` criterion can never ground an EXCLUDE: the screener guard routes such a decision to NEEDS_REVIEW instead (`backend/app/agents/relevance_screener_agent.py`). |
| `stage_rationale` | string | Required, non-empty, when `stage` is `"full_text"`; states the reason as a property of the criterion text, naming no record, result or run. Optional otherwise (used, for example, to record that a criterion is explicitly the review's own abstract-stage instruction). |
| `absence` | boolean | Marks a criterion whose test is that something is *not* present, as against a positively established fact such as "exact duplicate" or "book or book chapter". Defaults to `false` for an inclusion criterion, `true` for an exclusion criterion. Feeds the guard's cut-abstract check: an EXCLUDE anchored to an absence criterion in a cut abstract is routed to NEEDS_REVIEW, because the missing tail, not the record, may be why nothing was found. |

`Protocol.inclusion_criteria`/`.exclusion_criteria` stay `list[str]` of the texts either way;
`load_protocol()` also exposes six parallel list fields (`inclusion_stages`, `exclusion_stages`,
`inclusion_stage_rationales`, `exclusion_stage_rationales`, `inclusion_absence`,
`exclusion_absence`). **Invariant:** where `criteria_provenance.exclusion[i].location` begins
"negation of I{n}", `exclusion_stages[i]` must equal `inclusion_stages[n-1]`, or `load_protocol()`
raises `ProtocolError` -- otherwise a model could dodge a full-text flag by citing the inclusion
id instead of its negated exclusion id for the same underlying fact.

## Choosing the primary label and `label_field`

The primary gold label for screening was changed from the title-and-abstract label
(`label_abstract_screening`) to the review's final inclusion label (`label_included`) after the
v1 runs had been scored. The reason is that `label_included` is the label the SYNERGY and
ASReview benchmark convention scores against, and that the title-and-abstract label marks
records held for full-text reading, which a reader of a title and abstract cannot reproduce.
`primary_label` is the label `run_screening.py`'s acceptance gate and `summarize.py`'s
`metrics_primary` block score against, and its value is `"label_included"` throughout. Only
`Fong_2021.json`, `Taschner_2024.json`, `Anmarkrud_2021.json`, `Nagtegaal_2019.json` and
`van_Dis_2020.json` carry an explicit `primary_label` key. `Smid_2020.json` and `van_de_Schoot_2017.json` predate this
convention and carry no `primary_label` key at all (no phase of this project adds one to either file);
for those two, the acceptance gate and `metrics_primary` must read
`protocol.extra.get("primary_label", "label_included")`, i.e. treat a missing key the same as an
explicit `"label_included"`, not as `None` or a validation error. This default is a
specification for whichever phase implements the acceptance gate and `metrics_primary`,
not yet an implemented behaviour: as of this writing neither
`summarize.py` nor `run_screening.py` reads `primary_label` at all. `label_field` keeps its
existing meaning, the label the source review's own title-and-abstract stage produced, and is
now reported as a sensitivity number (`metrics_sensitivity`) rather than as the primary metric.
Datasets that only provide `label_included` (no separate title/abstract label) set `label_field`
to `"label_included"` too, so `primary_label` (explicit or defaulted) and `label_field` are equal
for those protocols (`Smid_2020`, whose v2 export carries no title/abstract label). Either way the gold
label is the source review's own screening decision as distributed by SYNERGY, not a
re-annotation for this evaluation, and SYNERGY publishes no per-record inter-screener
agreement, so records counted as LLM false positives or false negatives include human label
noise and review-specific screening thresholds (`van_de_Schoot_2017`'s protocol records a
deliberately over-inclusive title/abstract stage, so precision is not compared across
datasets); `summarize.label_field_note()` prints this under Table E1-a.

## Record order and batching

`run_screening.py` does **not** screen records in file order. SYNERGY exports list the
human-included records first, so file-order batches of 10 would be label-sorted (for
`Smid_2020` all 27 inclusions would sit in batches 0-2), a condition the production screener
never meets. Before batching, the record list is shuffled with
`random.Random(f"{dataset_id}:{seed}")` (`--record-order-seed`, default 20260902); the seed is
written to the run meta file as `record_order_seed`, the order is identical for runs A and B
and across resumed sessions, and `--limit` applies after the shuffle. The protocol itself is
unaffected: batch composition is a property of the run, not of the criteria.

## Hydration (identifiers for the WoS gate analysis)

`--hydrate included` targets every record with any positive label (`label_included == 1` or
`label_abstract_screening == 1`), so the denominator of the gate-effect analysis is the
human-included set of the protocol's `label_field`. Lookups never change title or abstract
text. The date, service (`--lookup openalex|crossref`) and resolution counts are recorded in
`dataset_source.hydration` of each protocol and in `data/<id>.fetch.json`; unresolved
records are reported as such and never enter a share.

`dataset_source.hydration` sub-fields (all counts refer to one hydration run, whose date is
`date`; the jsonl written by that run is the only source of the counts):

| Key | Meaning |
|---|---|
| `mode`, `lookup`, `date` | `--hydrate` mode, service (`openalex` / `crossref`), ISO date of the run. |
| `n_targets`, `n_targets_with_issn`, `n_final_inclusions_with_issn` | records looked up, records that resolved to an ISSN, and the subset of those with `label_included == 1`. |
| `by_doi`, `by_title_exact`, `by_title_exact_year` | resolved records per `match_method` of the jsonl: `doi` (the export's DOI), `title_exact` (exact normalised title, no year available to check) and `title_exact_year` (exact title *and* year within +-1 of the export's year). The three add up to `n_targets_with_issn`. |
| `year_check` | whether the +-1-year rule could be applied (the export carries a year) and which records it left unresolved. |
| `reprint_venues_excluded` | `{date, rule, records_rehydrated, resolved_again, note}`: hits in re-publishing series (`fetch_synergy.is_reprint_venue`, "Yearbook of ..." container titles) are skipped by the title lookup; `records_rehydrated` lists the records whose earlier yearbook match was discarded and looked up again, `resolved_again` how many of them resolved elsewhere, and the note explains the case. |
| `doi_export_corrected` | records whose export DOI was malformed (`fetch_synergy.is_valid_doi` false) and now carry the Crossref DOI of the title-matched work; the export value is kept in the jsonl as `doi_export`. |
| `spot_check` | `{date, judged_by, method, n, correct, incorrect, unsure, incorrect_record_ids, year_within_1, year_identical, evidence_file, note}`: a seeded random sample of title matches (`spot_check_hydration.py --n 20 --seed 20260903`) whose matched Crossref record (title, container title, year, type) was read and judged record by record, not by an automated matcher (see `judged_by` below for who). The judged file is committed as `spot_checks/<id>.spot_check.json` (per record: `judgement` = `correct` / `incorrect` / `unsure`, `judgement_note`, `judged_by`, `judged_on`; the judgements themselves are in `spot_checks/<id>.judgements.json` and are merged with `--judgements`). `judged_by` names who judged; the 2026-09-03 files were judged by the harness fix agent and are to be re-read by the authors before the manuscript cites them. |

## `Nagtegaal_2019` criterion completion (2026-09-06)

Inclusion criterion I1 names the Munscher, Vetter & Scheuerle (2016) nudge taxonomy without
reproducing it. The taxonomy's three categories, and one example of each, were added before any
v2 run and are marked `derived_from` in the protocol JSON, citing Munscher R, Vetter M, Scheuerle
T. A review and taxonomy of choice architecture techniques. Journal of Behavioral Decision Making
2016;29(5):511-524, DOI 10.1002/bdm.1897. No other criterion in `Nagtegaal_2019.json` was changed.

Before:

> The study deals with nudges (softly steering interventions, classified with the Munscher,
> Vetter & Scheuerle 2016 nudge taxonomy) applied to healthcare professionals at the individual
> level.

After:

> The study deals with nudges (softly steering interventions, classified with the Munscher,
> Vetter & Scheuerle 2016 nudge taxonomy) applied to healthcare professionals at the individual
> level. The taxonomy's three main categories are (a) decision information -- changing how
> information is presented without changing the options themselves (e.g. presenting guidelines
> in plain English, providing a social reference point); (b) decision structure -- altering the
> arrangement of options and the decision-making format (e.g. reducing the number of easily
> selectable options, changing a default); and (c) decision assistance -- closing the
> intention-behaviour gap with tools that help people follow through on an intention (e.g.
> reminders, asking people to specify when and where they will act).

The three categories and their one example each were read from Nagtegaal, Tummers, Noordegraaf &
Bekkers (2019) p.4 ("Theory"), which explains the taxonomy this review adopted and is itself
quoting Munscher et al. (2016) for the category names and examples; the primary taxonomy paper
is closed access (OpenAlex: `open_access.is_oa=false`, no repository copy, checked 2026-09-06),
so its own page-level table of sub-techniques was not read directly. The authors should verify
the category descriptions against the primary source before the manuscript relies on them.

## Provenance rules

- Quote or closely transcribe the review's own wording; do not invent criteria the review did
  not state. When the review has no itemised list, derive items from the Methods text and the
  PRISMA exclusion reasons and say so in `notes` (see `Nagtegaal_2019.json`).
- Record the source (article page or PDF) and the access date in `notes`.
- Do not tune the wording after seeing screening results; the protocol is fixed before run A.
- A **fidelity correction** (a criterion's wording is checked against the source review's own
  text and found narrower, broader or otherwise different from what the source states) is not
  tuning and is allowed at any time, provided `notes` quotes the source sentence, states the date,
  and lists every earlier run that used the old wording. If a run has already been made on that
  dataset, the correction is also a diagnosis of that run's own rows and `notes` must additionally
  record that the dataset's held-out status ends as of the correction date, that any number
  re-measured on the dataset after that date is a development-set number rather than a held-out
  one, and which diagnosis (report and section) the correction came from; a correction that quotes
  a source sentence but omits this is incomplete, because a diagnosis of a run's misses can nearly
  always find some source sentence to quote for a change in the direction the misses favour.
  `van_Dis_2020.json`'s I5/E1 correction (2026-09-07) is the record of this: its `notes` state that
  it was made by reading van_Dis_2020's own run rows and that its held-out status ends as of that
  date. A rewrite justified only by which dev-set
  records it recovers, with no source sentence behind it, is tuning and is not a fidelity
  correction, however it is labelled. Before recommending or making a fidelity correction, read
  the target protocol file's own current `notes` and its git history (`git log -p -- <file>`), not
  only an earlier diagnosis report: a diagnosis written from run rows and an older report can be
  wrong about whether a fix was ever committed. `Fong_2021.json`'s E1/E2/E3 case (2026-09-07) is
  the record of exactly this: a diagnosis recommended reapplying a set of E1/E2/E3 fixes on the
  premise that they were "never applied"; the fixes had in fact been committed once, run for one
  dev iteration, and then reverted for two of the three criteria by name because that wording was
  sized to the dev-set records it recovered, a fact recorded in the file's own `notes` at the time
  the diagnosis was written. The reapplication was not made a second time; see `Fong_2021.json`'s
  `notes` for the full account.

## Adding a dataset

1. Pick the dataset (HSS-leaning, 1k-6k records, ideally with `title_abstract_inclusions = True`
   in `data/synergy_index.csv`; v2-only datasets such as `Smid_2020` are fetched by their
   `<name>_ids.csv`).
2. Fetch it: `evaluation/.venv/Scripts/python evaluation/screening/fetch_synergy.py --dataset <id>
   --route v1 --hydrate included` (or `--route v2` when the dataset has OpenAlex ids). This also
   writes `data/<id>.fetch.json`; paste its counts into `dataset_source`.
3. Open the source review (the `reference` column of the index is a hint only -- verify it
   against the paper that actually reports the screening, see the caveat below), extract the
   question and the criteria, and create `<id>.json` following the schema above.
4. Validate: `python -m pytest evaluation/tests/test_io.py -q -k protocols`.
5. Run A and B with `run_screening.py`, then `baselines.py`, `wos_gate_effect.py`, `summarize.py`.

## Index-reference caveat (van_de_Schoot_2017)

The v1 index lists `https://doi.org/10.1080/10705511.2016.1247646` (the GRoLTS checklist paper,
Structural Equation Modeling 2016) as the reference for `van_de_Schoot_2017`. That paper is not
the review whose screening produced the labels; the criteria come from Appendix A of van de
Schoot et al., *Bayesian PTSD-Trajectory Analysis with Informed Priors ...*, Multivariate
Behavioral Research 2018;53(2):267-291, DOI 10.1080/00273171.2017.1412293 (Crossref-verified
2026-09-02). The protocol records both DOIs.

## Committed so far

| File | Source review | Route / gold label | Records (included / TA-included / no abstract) |
|---|---|---|---|
| `Nagtegaal_2019.json` | Nagtegaal R, Tummers L, Noordegraaf M, Bekkers V. Nudging healthcare professionals towards evidence-based medicine: a systematic scoping review. J Behav Public Adm 2019;2(2). DOI 10.30636/jbpa.22.71. Dataset licence CC0. | v1 / `label_abstract_screening` | 2,019 (101 / 392 / 169); ISSN resolved for 297/392 TA-included (Crossref, exact title; no export year to check) |
| `van_de_Schoot_2017.json` | van de Schoot R, Sijbrandij M, Depaoli S, Winter SD, Olff M, van Loey NE. Bayesian PTSD-Trajectory Analysis with Informed Priors Based on a Systematic Literature Search and Expert Elicitation. Multivariate Behav Res 2018;53(2):267-291. DOI 10.1080/00273171.2017.1412293. Dataset licence CC-BY 4.0. | v1 / `label_abstract_screening` | 6,189 (43 / 388 / 764); ISSN resolved for 331/388 TA-included (Crossref: 112 by DOI, 219 by exact title within +-1 year) |
| `Smid_2020.json` | Smid SC, McNeish D, Miocevic M, van de Schoot R. Bayesian Versus Frequentist Estimation for Structural Equation Models in Small Sample Contexts: A Systematic Review. Struct Equ Modeling 2020;27(1):131-161. DOI 10.1080/10705511.2019.1577140. SYNERGY collection licence CC0. | v2 / `label_included` (no title/abstract label in the v2 export) | 2,627 (27 / - / 640); 33 works without title or abstract, none of them included; ISSN for 2,403/2,627 |
| `van_Dis_2020.json` | van Dis EAM, van Veen SC, Hagenaars MA, Batelaan NM, Bockting CLH, van den Heuvel RM, Cuijpers P, Engelhard IM. Long-term Outcomes of Cognitive Behavioral Therapy for Anxiety-Related Disorders: A Systematic Review and Meta-analysis. JAMA Psychiatry 2020;77(3):265-273. DOI 10.1001/jamapsychiatry.2019.3986. Dataset licence CC-BY Attribution 4.0 International. Held-out (chosen by an abstract-coverage audit as the sole set clearing the 0.80-coverage/40-positives floor; exempted from the 6,500-record cap as the named fallback). | v1 / `label_abstract_screening` | 10,953 (73 / 806 / 438); not yet hydrated (`n_with_issn` = 0 in the export; 7,100/10,953 records already carry a DOI for a future hydration run) |
| `Taschner_2024.json` (withdrawn, unused) | Taschner J, Dicke T, Reinhold S, Holzberger D. "Yes, I Can!" A Systematic Review and Meta-Analysis of Intervention Studies Promoting Teacher Self-Efficacy. Rev Educ Res 2024;95(1):3-52. DOI 10.3102/00346543231221499. SYNERGY+ deposit licence CC BY 4.0 (OSF DOI 10.17605/OSF.IO/UYB7X). | plus / `label_abstract_screening` | 3,828 (109 / 775 / 2,694 missing abstract, 70%); withdrawn as a held-out set because 70% of records carry no abstract; the protocol file is kept in the repository, unused, per instruction ("Withdrawn sets keep their protocol files, unused, with a note") |

Counts are from `fetch_synergy.py` on 2026-09-03 and equal each protocol's `dataset_source.hydration` block (`tests/test_io.py` checks the resolution counts of this table against the protocols; hydration via Crossref on 2026-09-03 because
the shared-IP OpenAlex daily budget was exhausted; see each protocol's
`dataset_source.hydration`). The missing-abstract count for `van_de_Schoot_2017` equals the
SYNERGY index value (764) now that hydration never back-fills abstracts. For `Smid_2020` the
v2 ids file lists 3,980 rows; rows without any identifier (1,067) are dropped and duplicate
OpenAlex ids (286) merged, which yields SYNERGY's published 2,627 works. Missing abstracts on
route v2 are works for which OpenAlex has no `abstract_inverted_index`.
