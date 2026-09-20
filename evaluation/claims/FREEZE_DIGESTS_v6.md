# Freeze digests, v6 (guard-7 replay of the cached model answers)

Branch `softwarex-revision`. **v6 is a deterministic re-scoring of the cached model answers
under `evaluation/claims/results/v6/inputs/` through the shipped guard set; no model was
called to produce it.** Every row of `results/v6/` is one of those cached model answers,
replayed through `app.services.fulltext.verify_claim_with_policy`/
`apply_verification_guards` at HEAD (guard 7, the scale-word-fidelity guard, committed at
`26dadc4`).

## Digests: before and after, confirmed live against this checkout

```
GUARD_DIGEST                        570a5b663e6140da  ->  8a6c833ffc329f89   (moved: guard 7)
VERIFICATION_POLICY_VERSION         667bcc9209126ddc  ->  667bcc9209126ddc   (unchanged)
QUOTE_RELOCATION_VERSION            0269a13351c0da7d  ->  0269a13351c0da7d   (unchanged)
CLAIM_VERIFICATION_PROMPT_VERSION   sha256:49fcfbfaf2f6  ->  sha256:49fcfbfaf2f6   (unchanged)
QUOTE_REPAIR_PROMPT_VERSION         sha256:7ec7cffe852d  ->  sha256:7ec7cffe852d   (unchanged)
temperature                         0.0  ->  0.0   (unchanged; no model call was made)
```

`GUARD_DIGEST` is the only digest that moves, because guard 7 is the only production change
this replay applies. It is gated on `status == "verified"` and can only ever cap a row to
`needs_nuance` (never promotes, never reaches `unsupported`), so it cannot alter which prompt
was sent, how many model calls were made, or what a model returned: every cached answer
remains a valid observation of the frozen prompt at temperature 0, and replaying it through the
guard set at HEAD is a legitimate freeze rather than a fresh draw.

Served model and fingerprint, copied from every `results/v6/inputs/*.meta.json` (identical
across all six files, confirmed above) and carried unchanged into every
`results/v6/*.meta.json`, since no new model call was made:

```
model_configured   deepseek-chat
model_reported     ["deepseek-flash"]
system_fingerprint  aeb56401ca74e127821c4f9126dcb669
```

## Method

`evaluation/claims/freeze_v6.py`, run once (system python, from `evaluation/claims/`, no
`--confirm-cost` flag exists because no model call is made):

```
python freeze_v6.py
```

For every model-answered row of the six `results/v6/inputs/{scifact,hss,real}_run{A,B}.jsonl`
files, a `ModelPassAnswer` is rebuilt from that row's own stored `model_status`,
`evidence_quote(s)`, `assertions`, `explanation`, `suggested_revision` and
`unstated_details` (the technique `replay_repair_turn_v5.py` and `replay_scale_guard.py` both
already use), and replayed through the real `verify_claim_with_policy` **twice**: once with
guard 7 (`_guard_scale_fidelity`) monkeypatched to a no-op (the pre-guard-7 baseline) and once
with the real, shipped guard. Rows are replayed strictly one at a time (never concurrently):
the monkeypatch is shared, mutable module state, and interleaving it across rows would be a
race.

The delta between the two passes isolates guard 7's own effect from the effect of one
substitution both passes share: neither the cached rows nor `delivered_evidence.json` store the
source paper's full text, so (as `replay_scale_guard.py` already established, and as this
task's own replay confirms again below) the row's own evidence quotes stand in for the source
chunks. That substitution can
shift an *earlier* guard's own verdict on a handful of rows, but because both the before- and
after-guard-7 passes use the identical substituted chunks, any such shift is identical in both
and cancels out of the delta. The delta is applied to a row's own predicted status **only**
when that row's cached `predicted_status` was itself `"verified"` -- the exact
population guard 7's own gate targets in real production, where the guard chain ran against the
real source text. A row whose cached status already left `verified` before guard 7 could run is
written to `results/v6/` byte-for-byte as cached, regardless of what the substitution-based
replay's own before/after shows for it, because production's real guard chain is authoritative
for that row.

Baselines (`hss_baseline.json`, `scifact_baseline.json`) are copied byte-for-byte: the lexical
stand-in never calls a guard, so it cannot see the guard-7 change. `PROMPT.txt` is copied
byte-for-byte (the verifier prompt is unchanged).

## Per-row provenance

Every row of every `results/v6/*.jsonl` file carries an added `v6_replay` object (the cached
answer's own fields are otherwise unchanged): `source_row_id` (the row's own `item_id`),
`source_file`, `guard_digest_before` (`570a5b663e6140da`), `guard_digest_after`
(`8a6c833ffc329f89`), `status_before`/`status_after` (the cached status / v6's written status),
`status_changed`, `guard7_examined` (whether the row reached guard 7 with status `verified` in
the paired replay), `guard7_findings` (the new machine-reason slugs guard 7 contributed, empty
unless it actually changed the row), and `repair_turn_would_fire` (an anomaly flag; see
"Repair-turn sentinel" below). Deterministic `no_full_text` rows and `error` rows carry the same
object with `guard7_examined: false` and a `note` explaining why the guard chain never ran for
them.

## The replay, every number

**Zero rows changed status anywhere.** `results/v6/*.meta.json`'s own `status_counts` are
byte-identical to `results/v6/inputs/*.meta.json`'s, file for file:

| file | n rows | n candidates | n examined by guard 7 (status was `verified`) | n status changed | repair-turn anomalies |
| --- | --- | --- | --- | --- | --- |
| `scifact_runA` | 340 | 340 | 58 | 0 | 0 |
| `scifact_runB` | 340 | 340 | 58 | 0 | 0 |
| `hss_runA` | 60 | 52 | 14 | 0 | 0 |
| `hss_runB` | 60 | 52 | 13 | 0 | 0 |
| `real_runA` | 26 | 26 | 6 | 0 | 0 |
| `real_runB` | 26 | 26 | 6 | 0 | 0 |
| **total** | 852 | 836 | **155** | **0** | **0** |

Not one of these 155 cached-verified rows is demoted in any file. Every headline metric
`summarize.py` computes over `results/v6/` is therefore identical to the cached answers' own
scoring, to the last decimal:

| set | run | metric | cached | v6 | delta |
| --- | --- | --- | --- | --- | --- |
| scifact | A | accuracy | 0.7500 | 0.7500 | 0.000 |
| scifact | A | macro-F1 | 0.7513 | 0.7513 | 0.000 |
| scifact | A | verified precision | 0.9828 | 0.9828 | 0.000 |
| scifact | A | verified recall | 0.4130 | 0.4130 | 0.000 |
| scifact | A | unsupported precision | 0.8684 | 0.8684 | 0.000 |
| scifact | A | unsupported recall | 0.9802 | 0.9802 | 0.000 |
| scifact | B | accuracy | 0.7500 | 0.7500 | 0.000 |
| scifact | B | macro-F1 | 0.7534 | 0.7534 | 0.000 |
| scifact | B | verified precision | 0.9828 | 0.9828 | 0.000 |
| scifact | B | verified recall | 0.4130 | 0.4130 | 0.000 |
| scifact | B | unsupported precision | 0.8761 | 0.8761 | 0.000 |
| scifact | B | unsupported recall | 0.9802 | 0.9802 | 0.000 |
| hss | A | accuracy | 0.9167 | 0.9167 | 0.000 |
| hss | B | accuracy | 0.9000 | 0.9000 | 0.000 |
| hss vs annotation | A | accuracy | 0.9167 | 0.9167 | 0.000 |
| hss vs annotation | B | accuracy | 0.9000 | 0.9000 | 0.000 |
| real vs annotation | A | accuracy | 0.4231 | 0.4231 | 0.000 |
| real vs annotation | B | accuracy | 0.4231 | 0.4231 | 0.000 |
| quote verbatim (verified rows) | A/B | scifact | 1.000 | 1.000 | 0.000 |
| quote verbatim (verified rows) | A/B | hss | 1.000 | 1.000 | 0.000 |
| quote verbatim (verified rows) | A/B | real | 0.833 | 0.833 | 0.000 |

No headline figure changes.

## Guard-7 detector reach, ungated (diagnostic only, never applied to `predicted_status`)

`results/v6/guard7_reach_diagnostic.json`: `scale_alignment_findings`/`_scale_negation_scope_shift`
called directly on every candidate row's own claim and quotes, bypassing guard 7's own
`status == "verified"` gate entirely, so a reader can see the guard's reach on this data even
though the gate keeps every one of these firings off the scored population.

**Eleven firings across all 852 cached rows, every one already predicted `unsupported`:**

| file | item(s) |
| --- | --- |
| `scifact_runA` | `814:33387953` |
| `scifact_runB` | (none) |
| `hss_runA` | `hss-altered-01`, `hss-altered-12`, `hss-altered-14`, `hss-altered-17` |
| `hss_runB` | `hss-altered-01`, `hss-altered-12`, `hss-altered-14`, `hss-altered-17` |
| `real_runA` | `real-test-09` |
| `real_runB` | `real-test-09` |

One SciFact row in run A only, the four HSS `quantifier_flip` items in both runs, one real row
in both runs: the guard agrees with the frozen verifier wherever both speak on this data. It
never contradicts an existing `unsupported` status, and it never had a chance to act on any of
these eleven regardless, since none of them ever reached `verified`.

## Repair-turn sentinel

Zero anomalies (`repair_turn_would_fire` false on every one of the 836 model-answered rows,
both passes): guard 7's gate on `verified` means it can never redirect a single model call,
which holds here by direct measurement as well as by construction, exactly as
`replay_scale_guard.py` already found.

## Output files written (`evaluation/claims/results/v6/`)

| file | note |
| --- | --- |
| `PROMPT.txt` | copied byte-for-byte from `results/v6/inputs/` |
| `hss_baseline.json`, `scifact_baseline.json` | copied byte-for-byte from `results/v6/inputs/` (lexical stand-in, no guard) |
| `{scifact,hss,real}_run{A,B}.jsonl` | the cached rows, each with an added `v6_replay` object |
| `{scifact,hss,real}_run{A,B}.meta.json` | the cached session/model/cost fields, `guard_digest` updated, plus a `replay` block |
| `guard7_reach_diagnostic.json` | the ungated detector-reach table above |
| `summary.json`, `summary.md`, `figure4_claims.json` | `summarize.py`'s own recipe against `results/v6/`, plus the human-review block (Table E2-h; see below) |
| `human_labels.json`, `human_review_scores.json` | the three reviewers' majority label and v6's own scoring against it |
| `delivered_human_scores_run_20260916-154140.json` | the three reviewers' majority label on the promoted run's 68 verified claim rows, produced by `evaluation/claims/delivered_human_scores.py`; sha256 `6a56ef431ba79b9ab4a9737050c6b23c366e726f092cc44d939fe78b033b46de` |

## Determinism

`freeze_v6.py` was run twice into the same output directory; every `*.jsonl` row file (the
actual replay content) came back byte-for-byte identical between the two runs. `*.meta.json`
and `summary.json`/`summary.md` carry their own `generated`/`replay.generated_at` wall-clock
timestamps and are not expected to be byte-identical run to run; `evaluation/tests/
test_freeze_v6.py` pins the row-level determinism claim, excluding those timestamp fields.

## Ownership

This task touches: `evaluation/claims/freeze_v6.py`, `evaluation/claims/human_review_labels.py`,
`evaluation/claims/results/v6/**` (including `results/v6/inputs/`, the cached model answers the
replay scores), this ledger, `evaluation/README.md`, and the test files under
`evaluation/tests/` that exercise this replay. No file under `backend/`, `demo/`, `docs/`,
`frontend/`, `"softwarex publication/"`, or any internal working file, `.mission/`, `README.md`, or
`CHANGELOG.md` is touched by this replay.
