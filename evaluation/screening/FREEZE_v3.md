# Screener v3 freeze

Records the production cutover to the v3 screener prompt (the anchor ledger, the
title-only TOPIC exclude, the full-text-to-confirm queue routing, the type/table-of-contents
demotions and the inclusion-only second pass, all wired into
`app.services.smart_search.run_smart_search` and into `evaluation/screening/run_screening.py`)
and the full re-run this freeze reports. `FREEZE_v2.md` is unedited and stays the record of
the v2 prompt (the v2 runs it once described under `results/v2/` have since been removed as
superseded); this file does not supersede it, since v2 stays available as an explicit
`--prompt-version v2` selection and `SCREENER_PROMPT_V2`'s own digest test still pins its
bytes.

## 1. What changed in production

`SCREENER_PROMPT` (`backend/app/agents/relevance_screener_agent.py`) now points at
`SCREENER_PROMPT_V3` instead of `SCREENER_PROMPT_V2`; `_ACTIVE_PROMPT_VERSION` is `"v3"`.
Every production Smart Search job now runs the v3 prompt, the deterministic
`full_text_to_confirm` guard (an INCLUDE naming an unconfirmed full-text inclusion criterion
is routed to NEEDS_REVIEW, never shipped as INCLUDE), the two post-model devices
(`apply_type_demotion`, `apply_table_of_contents_demotion`) and, on whatever survives both,
the inclusion-only second pass (`run_second_pass_stage`, built on `confirm_inclusions` /
`apply_second_pass_guard`; see below) at `settings.screener_second_pass_model`.

`deepseek_model` and `deepseek_reasoner_model` moved from the retired names `deepseek-chat`/
`deepseek-reasoner` to `deepseek-flash`, the provider's current name for the same served
deployment. This rename created a defect, not surfaced a pre-existing one: `deepseek-chat`
left thinking mode off by default, so DeepSeek's rejection of `tool_choice` outright while
thinking is on ("Thinking mode does not support this tool_choice") never applied to it;
`deepseek-flash` defaults thinking on, so every structured-output call broke the moment the
name changed. Fixed in `backend/app/agents/model_config.py` by disabling thinking
(`extra_body={"thinking": {"type": "disabled"}}`) on every shared tier used for
tool-calling structured output, found by a smoke test before any paid call of the re-run
below. That same blanket fix, applied without exception, also reached the screener's own
inclusion-only second pass, which needs thinking left on to reason about population,
outcome and study type rather than pattern-match them; `SECOND_PASS_V2_MODEL_SETTINGS`
(`backend/app/agents/model_config.py`) now gives that one call shape thinking left on, its
own wider timeout, and no thinking-disable, while every other tier keeps thinking disabled
for `deepseek-flash` tool-calling compatibility (section 4 reports what this changed on the
demo rescore).

`screener_second_pass_model` defaults to `deepseek-flash`, the same model `deepseek_model`
uses, with reasoning left on and `screener_second_pass_output_mode` `"prompted"` (section 4)
-- this does not change the three benchmark runs in section 3 at all, since the second pass
never fires on any of them (see section 3's own reading of why).

The second pass is now a concurrent stage (`app.agents.relevance_screener_agent
.run_second_pass_stage`), and it runs once for the whole job, after every search round has
finished, not once per round and not once per original batch of ten -- a live search job
runs many short rounds, each with only a handful of INCLUDE candidates, and a reasoning-on
judge call takes 7.4 to 55.2s of its own, so paying that latency once per round inside a
30-minute search budget would leave most of a job's own rounds never reached. A record
standing as INCLUDE during the search loop is a provisional include (a round's own stopping
decision and a later round's own query generation see it exactly as before this stage
existed), and once the loop ends every provisional include the whole job produced is judged
together, five records per call, up to `screener_second_pass_concurrency` (default 8) calls
in flight at once, with a per-call 420s timeout unchanged from before and a
`screener_second_pass_stage_budget_seconds` (default 1800s) ceiling on the stage itself,
entirely separate from the search's own time budget, so the search's own stopping decisions
are never shortened by how long the judge takes; a call that fails, times out or is never
launched for lack of remaining stage budget routes only its own small chunk to NEEDS_REVIEW
with `second_pass_unavailable`, never the whole job. The stage's own wall-clock ceiling is
therefore fixed at 1800s plus at most one further already-launched 420s call (about 37
minutes), never open-ended. On the 140-record demo set (37 second-pass candidates, 8 chunks,
all in one job-level call) this measured a 55.4s stage wall time, 7.4 to 55.2s per call,
still well inside the 420s timeout -- concurrency, not a faster judge, is what shortens the
stage; at these same per-call latencies, about 250 candidates (50 calls, 7 concurrent waves)
is expected to take on the order of 6 minutes typically, comfortably inside the ceiling,
and in an unlucky run whose calls run long, the stage still stops at the same
~37-minute ceiling, having launched only as many of the 50 calls as fit; its own last few
chunks (on the order of the final 10, about 50 candidates) are never sent at all and are
routed to NEEDS_REVIEW with `second_pass_unavailable` for a
human to read instead. The harness (`evaluation/screening/run_screening.py`) calls the same
function, once per harness batch. No new paid measurement was made for this restructuring:
`run_second_pass_stage` itself, and the calls it makes, are unchanged; only when
`run_smart_search` calls it moved. Re-running the same 140 records through this concurrent
stage reproduced the sequential rescore below on 133 of 140 records; the 7 that moved split
into 3 whose batch pass itself returned a different status this time (`S-009`, `S-068`,
`S-113`) and 4 whose second-pass judge answered "established" on one measurement and "not
established" on the other (`S-015`, `S-064`, `S-078`, `S-124`), the same call-to-call
variation this freeze's own runs A and B already document at temperature 0 (section 3,
43/169/84 disagreements) -- not a defect the concurrency restructuring introduced, since
each chunk's own call is unaffected by what else runs alongside it.

The anchor ledger's own inclusion-side check (`apply_decision_guard`'s `anchored_slots`
parameter, which would demote an INCLUDE to NEEDS_REVIEW when the model's own
population/subject/outcome ledger leaves a chosen slot with no verbatim quote) ships off:
no caller in `app.services.smart_search` or `evaluation/screening/run_screening.py` passes
`anchored_slots`, so it defaults to empty and never demotes anything in production or in the
harness. The second pass's own population/outcome/study-type establishment check is what
polices an INCLUDE's grounding today; the anchor ledger itself is still recorded on every
decision (`ScreeningBatchResult.anchors`) so a future run can turn the check on without a
further model call, but this freeze does not turn it on.

## 2. Frozen digests

| Item | Value |
|---|---|
| `prompt_version(SCREENER_PROMPT_V3)` (= `SCREENER_PROMPT_VERSION`, the active prompt) | `sha256:9bc741046cd6` |
| `prompt_version(SCREENER_PROMPT_V2)` (unchanged, no longer active) | `sha256:fb9de89e0534` |
| `prompt_version(SCREENER_PROMPT_V1)` (unchanged) | `sha256:3c422c45cd8b` |
| `Nagtegaal_2019.json` protocol sha | `sha256:1ae48c142315` (full-text: I1, E1, E5) |
| `Smid_2020.json` protocol sha | `sha256:5ff574cb49e8` (full-text: I2, I4, I6, E4, E7, E8, E9, E10, E11) |
| `van_de_Schoot_2017.json` protocol sha | `sha256:e271b929c830` (full-text: I2, E4) |

The two protocol digests `FREEZE_v2.md` section 4 already carries for `Nagtegaal_2019.json`
and `Smid_2020.json` are unchanged by this task; `van_de_Schoot_2017.json`'s digest is
recorded here for the first time (it was previously reported as already meeting its recall
targets and was not re-frozen before now).

## 3. The re-run

Dates and cost, run in order, all six calls on 2026-09-15 between 05:16 and 05:29 UTC
(Monday-Friday off-peak: 04:00-06:00 UTC is outside DeepSeek's 01:00-04:00/06:00-10:00 peak
windows), `--prompt-version v3`, default batch size 10 / concurrency 8, no
`--second-pass-model` override. `screener_second_pass_model` defaulted to `deepseek-flash`
throughout, the same model a later measurement (section 4) settled on as the shipped
default, with reasoning on and `screener_second_pass_output_mode` `"prompted"`. This does
not matter for the table below: the second pass never fires on any of these three
datasets (see the reading immediately after the table), so no call in this section used the
second-pass model at all:

| dataset | run | records | started (UTC) | finished (UTC) | model_reported | system_fingerprint | cost (USD) | in / out tokens | latency median / p90 (s) |
|---|---|---|---|---|---|---|---|---|---|
| Nagtegaal_2019 | A | 2,019 | 05:16:27 | 05:17:44 | deepseek-flash | aeb56401ca74e127821c4f9126dcb669 | 0.3898 | 1,343,874 / 142,684 | 2.841 / 3.510 |
| Nagtegaal_2019 | B | 2,019 | 05:17:46 | 05:19:02 | deepseek-flash | aeb56401ca74e127821c4f9126dcb669 | 0.3808 | 1,312,401 / 139,461 | 2.807 / 3.470 |
| Smid_2020 | A | 2,627 | 05:19:04 | 05:20:29 | deepseek-flash | aeb56401ca74e127821c4f9126dcb669 | 0.3735 | 1,258,386 / 146,441 | 2.477 / 2.933 |
| Smid_2020 | B | 2,627 | 05:20:31 | 05:21:58 | deepseek-flash | aeb56401ca74e127821c4f9126dcb669 | 0.3808 | 1,281,668 / 149,807 | 2.507 / 2.938 |
| van_de_Schoot_2017 | A | 6,189 | 05:22:01 | 05:25:31 | deepseek-flash | aeb56401ca74e127821c4f9126dcb669 | 1.0366 | 3,604,488 / 369,045 | 2.572 / 3.159 |
| van_de_Schoot_2017 | B | 6,189 | 05:25:33 | 05:29:02 | deepseek-flash | aeb56401ca74e127821c4f9126dcb669 | 1.0114 | 3,516,589 / 360,243 | 2.625 / 3.059 |

Every one of the six pass-1 runs: 0 failed batches, 0 unscreened records, one served model
and one system fingerprint throughout. Second-pass calls: 0 on all six -- every one of the
three protocols names at least one full-text inclusion criterion, so `apply_decision_guard`'s
`full_text_to_confirm` check demotes every genuine INCLUDE to NEEDS_REVIEW before it can ever
reach the second pass, on both runs of all three datasets (see `results/v3/summary.md` Table
E1-e, "auto-inclusion precision" `none, n = 0` on every row). The screener under this rule
never auto-includes a record whose protocol defers a criterion to full text; it prioritises
queuing that record for further reading rather than guessing at it.

Pass-1 total: **US$3.5729** (the second pass never fires on these three datasets, so it is
unaffected by which model `screener_second_pass_model` names). Demo rescore (below, at
`deepseek-flash`, reasoning on, output mode `"prompted"`): **US$0.0623** pass 1 and second
pass combined (the demo protocol carries no full-text criteria, so its second pass does run
-- see section 4). Concurrent-stage timing re-run of the same 140 records (above, one
sentence in "What changed in production"): **US$0.3913**. Grand total for this freeze's
paid work: **US$4.0265**.

Results are under `evaluation/screening/results/v3/` (new; `results/v2/` was untouched by
this freeze and has since been removed as superseded):
`<dataset>_run<A|B>.jsonl` + `.meta.json`, `summary.json`, `summary.md`,
`figure4_screening.json`, `<dataset>_false_negatives.md`, and `demo_human_rescore.json`
(section 4). The false negatives of this freeze carry the screener's own stated reason and
no coded category, so no `<dataset>_fn_categories.csv` is part of it.

Also under `results/v3/`: `<dataset>_baselines.json` (an include-all baseline and a TF-IDF
cosine-similarity ranker, `baselines.py --k-basis screened_in`, matched to each dataset's own
run-A screened-in count and scored on the same `label_included` basis as Table E1-e), added to
`summary.json`, `summary.md` (Table E1-e) and `figure4_screening.json` beside the model rows.

## 4. Demo human rescore

The 140 stratified-sample records from the three reviewer workbooks under
`softwarex publication/human-review/` (read via openpyxl in read-only mode; no reviewer
identity is recorded anywhere in this freeze or in `demo_human_rescore.json`, only the
aggregated majority verdict and an any-reviewer-include flag per record) were re-screened
with the shipped production functions directly (`screen_papers` at the now-active v3 prompt,
`apply_type_demotion`, `apply_table_of_contents_demotion`, `confirm_inclusions`,
`apply_second_pass_guard`, in the same order `run_smart_search` calls them), using
`demo/protocol.json`'s own research question and criteria and each record's `title`/
`abstract_shown` already present in the workbooks, plus each record's OpenAlex `type`/
`is_paratext` fetched fresh and free via `OpenAlexClient.get_works_batch`. No demo stack was
started; this calls the shipped `app.agents.relevance_screener_agent` functions the product
itself calls, not a separate script.

| quantity | value |
|---|---|
| Final statuses (of 140) | 24 INCLUDE, 67 EXCLUDE, 49 NEEDS_REVIEW |
| Majority verdict (of 140) | 23 include, 93 exclude, 24 needs_review |
| Inclusions upheld (final INCLUDE with majority include / final INCLUDE) | 0.917 (22/24) |
| Inclusions upheld, per reviewer (sorted, no reviewer identity) | 0.833, 0.917, 0.917 |
| Exclusions upheld, lenient (final EXCLUDE with majority not include / final EXCLUDE) | 1.000 (67/67) |
| Exclusions upheld, strict (final EXCLUDE with majority exclude / final EXCLUDE) | 0.985 (66/67) |
| Queue agreement (final NEEDS_REVIEW with majority needs_review / final NEEDS_REVIEW) | 0.449 (22/49) |
| Hard gate: majority-include records excluded | 0 |
| Hard gate: any-single-reviewer-include records excluded | 0 (passes) |
| Pass-1 provenance | 14 calls, 70,592 in / 10,249 out tokens, deepseek-flash, fingerprint aeb56401ca74e127821c4f9126dcb669, 45.3 s total latency, ~US$0.0223 |
| Second-pass provenance | 37 of 140 records reached it (the batch pass's own INCLUDE survivors of the type/table-of-contents demotions); 8 calls, 19,827 in / 54,018 out tokens (about 1,459.9 output tokens per record), deepseek-flash, reasoning on (effort high, the provider's own documented default -- the request itself leaves the effort unset), output mode "prompted", fingerprint aeb56401ca74e127821c4f9126dcb669, 55.4 s stage wall time, 251.7 s total latency across calls (7.4 to 55.2 s per call, about 31.5 s mean), 0 malformed or failed replies, ~US$0.0400 |

Reading: the hard gate holds (no record any reviewer marked include is excluded outright).
The batch pass runs at temperature 0 and is unchanged by this task, but the provider is not
bit-deterministic at that setting: this measurement's own candidate set is 37 records, and
section 3's own runs A and B of the same three datasets disagree on 43, 169 and 84 records
respectively despite the identical temperature ("What changed in production" above records
the same kind of call-to-call variation in the second-pass judge itself). The second-pass
judge itself is not run at a temperature at all: `SECOND_PASS_V2_MODEL_SETTINGS` keeps
reasoning on (the provider does not apply a temperature while reasoning), at effort high --
the provider's own documented default, since the request itself leaves the effort unset --
instead of disabling it, which is why its output here (about 1,459.9 tokens per record)
reflects genuine reasoning rather than the roughly 70 tokens per record a disabled-thinking
call produces. This measurement is what set the shipped defaults
(`screener_second_pass_model` `deepseek-flash`, `screener_second_pass_output_mode`
`"prompted"`): the same reasoning-on behaviour, at a fraction of the cost and latency of a
larger model, with `"prompted"` output mode sending the answer schema in the prompt instead
of a forced tool_choice, which this served model rejects outright while reasoning is on. The
full per-record breakdown (final status, guard reason, every field of the second pass's own
population/outcome/study-type answer, majority verdict, any-reviewer-include flag; no
reviewer identity) is in `results/v3/demo_human_rescore.json`, replayable without a further
model call.

## 5. What this freeze does not change

`SCREENER_PROMPT_V1`, `SCREENER_PROMPT_V2` and their digests; `evaluation/screening/results/v2/**`
and the frozen v1 run under `results/v1/` (both untouched by this freeze, both since removed
as superseded); `demo/protocol.json`; `demo/expected/**`; the
manuscript and its figures; any workbook or credentials file. `evaluation/screening/protocols/
Nagtegaal_2019.json` and `Smid_2020.json` are untouched by this task (their digests above are
unchanged from `FREEZE_v2.md`'s own 2026-09-14 staleness note); `van_de_Schoot_2017.json` was
also untouched -- its digest above is recorded for the first time, not changed.

`demo/run_demo.py --strict`'s own comparison against `demo/expected/` was not run: it drives
a locally running instance (`docker compose up -d`, `http://localhost:8000`) through the
full ten-step live workflow (Smart Search, full-text acquisition, AI Write, claim
verification) end to end before it ever reaches the `--strict` comparison, so there is no
offline entry point that reaches it without a live stack and further paid calls across three
subsystems, both outside this task. `demo/expected/` itself is untouched (unchanged from the
v2 prompt it was frozen against, `SCREENER_PROMPT_VERSION sha256:fb9de89e0534` in
`demo/tests/test_expected_baseline.py`), so a live `--strict` run today would compare the
now-active v3 prompt's decisions against a v2-frozen baseline and its own drift would not be
a defect in this freeze; re-freezing `demo/expected/` against v3 is a separate task.
