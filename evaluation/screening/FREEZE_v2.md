# Screener v2 freeze

Date: 2026-09-07. This freezes the shipped state of the screener: the iteration-11 prompt
(`SCREENER_PROMPT_VERSION` `sha256:fb9de89e0534`), the guard it ships with, the protocol files as
committed, and the current status of every dataset that has a run against this prompt. It
supersedes the freeze this file previously recorded (iteration 9, `sha256:1ffc09342901`), which
was rejected on the held-out set and was then revised by a diagnosis of five failure patterns and
follow-up prompt work. Repository `HEAD` at this freeze:
`a29225f0d649012834a2a628d56b02b1521b80d0`.

Every `results/dev/v2-iter*/` and `results/v2-superseded-2026-09-07/` directory this file cites
has since been removed from the repository as superseded by `screening/results/v3/`, the
screening result of record; the historical record below is otherwise unchanged.

## 1. Frozen `SCREENER_PROMPT_VERSION`

Derived directly from the shipped module (`TEST_DATABASE_URL=... DEEPSEEK_API_KEY=test-key-not-real
python -c "from app.agents.relevance_screener_agent import SCREENER_PROMPT_VERSION,
SCREENER_PROMPT_V1_VERSION; print(SCREENER_PROMPT_VERSION, SCREENER_PROMPT_V1_VERSION)"`, cwd
`backend/`):

- `SCREENER_PROMPT_VERSION` (active, v2, protocol-eligibility prompt) = **`sha256:fb9de89e0534`**.
  This is the iteration-11 prompt: three category-level fixes and a no-abstract guard rule,
  committed at `f6eedf8`, `d16aecb` and `69a4ed1`.
- `SCREENER_PROMPT_V1_VERSION` (frozen original binary PRISMA prompt) = `sha256:3c422c45cd8b`,
  unmoved since it was first committed; recorded on the three committed v1 runs (`Nagtegaal_2019`,
  `Smid_2020`, `van_de_Schoot_2017`; section 8).

### Iterations 10 and 12: trialed, found worse, reverted; neither is part of the frozen prompt

Two clauses were added to `SCREENER_PROMPT_V2` after iteration 9 and each was later removed
byte-for-byte, restoring the prior sha exactly.

- **Iteration 10** (`sha256:3ca518e3b5c7`) added a full-text-exclusion carve-out; `apply_decision_guard`
  was not edited to match, so the clause could never produce a standing EXCLUDE, and iteration 10
  was found ineligible under gate D5 (one padded decision, run A) and reverted to `6309d98`'s exact
  byte content before iteration 11 was built. See the excerpt of that history preserved in section 3.
- **Iteration 12** (`sha256:c2fa4cd7ce6d`, commit `2acfa4c`, "defined terms govern everywhere they
  appear in the protocol") appended one further sentence to the kinds-not-words bullet. Recomputed:
  `van_Dis_2020` run A moved from H1 `38/66 = 0.5758` /
  H2 `48/73 = 0.6575` (iteration 11) to H1 `35/66 = 0.5303` / H2 `44/73 = 0.6027` (iteration 12), a
  regression on both counts, and `Fong_2021` run A's H3 moved from `171/822 = 0.2080` to
  `193/822 = 0.2348`, closer to the 0.25 bar with less headroom. Commit `82347c5` removed exactly the
  appended sentence, restoring `SCREENER_PROMPT_V2` to iteration 11's byte content
  (`SCREENER_PROMPT_VERSION` back to `sha256:fb9de89e0534`) and inverted the iteration-12 test to
  assert the sentence's absence. The run data at `results/dev/v2-iter12/` stays committed as the
  record of what was measured; the clause itself is not part of the frozen prompt.

## 2. Abstract cap and split

Unchanged since the iteration-9 freeze. `ABSTRACT_CHAR_LIMIT = 10000` (one constant; everything
else derives from it). v2's `_head_tail_split` gives a 7:3 split of that budget:
`ABSTRACT_HEAD_CHARS = 7000`, `ABSTRACT_TAIL_CHARS = 3000`. Below the cap a v2 abstract is shown
whole; above it, a 7,000-character head, the cut marker `ABSTRACT_CUT_MARKER = " [...] "`, and a
3,000-character tail. v1's truncation style (`abstract[:limit] + "..."`) is unchanged and bound to
`prompt_version="v1"`, not to this cap.

## 3. Guard constants, five guard reasons, and stage rules

### Guard constants (`backend/app/agents/relevance_screener_agent.py`)

| constant | value |
|---|---|
| `MAX_REASON_LENGTH` | 300 |
| `MAX_CRITERION_LENGTH` | 8 |
| `MAX_QUOTE_LENGTH` | 300 |
| `NO_DECISION_REASON` | `"no decision returned"` |
| `GUARD_DEMOTION_NOTE` | `"guard: EXCLUDE not anchored to a shown criterion id and verbatim quote"` |
| `GUARD_REASON_NO_ABSTRACT` | `"no_abstract"` |
| `GUARD_REASON_FULL_TEXT_CRITERION` | `"full_text_criterion"` |
| `GUARD_REASON_CUT_ABSTRACT` | `"cut_abstract"` |
| `GUARD_REASON_UNANCHORED_EXCLUDE` | `"unanchored_exclude"` |
| `GUARD_REASON_UNQUOTED_CRITERION` | `"unquoted_criterion"` |
| `TOPIC_CRITERION_ID` | `"TOPIC"` |
| `ABSTRACT_CUT_MARKER` | `" [...] "` |
| `NO_ABSTRACT_PLACEHOLDER` | `"(no abstract)"` |

`apply_decision_guard` returns one of exactly five guard reasons (or `""` for no demotion):

1. **`no_abstract`** (added for iteration 11). Checked first, ahead of every other test: an EXCLUDE on a
   record whose rendered abstract is the `(no abstract)` placeholder is always demoted to
   NEEDS_REVIEW, whatever criterion it names, because there is nothing to test any criterion
   against. The label `no_abstract` is given when the EXCLUDE's quote is still a verbatim substring
   of the shown text -- which covers a quote copied only from the title, the diagnosis's own P5
   failure mode -- and `unanchored_exclude` when the quote is invented outright.
2. **`full_text_criterion`**. An EXCLUDE naming a full-text exclusion id is demoted here when its
   quote is verbatim; a model-native NEEDS_REVIEW naming a full-text exclusion id is left
   attributed here, unchanged, only when its quote is non-empty and verbatim explicit contrary
   evidence.
3. **`cut_abstract`**. An absence criterion (`absence_ids`) grounding an EXCLUDE on a record whose
   shown text still carries `ABSTRACT_CUT_MARKER` is demoted here -- the missing tail, not the
   record, may be why nothing was found.
4. **`unanchored_exclude`**. The catch-all: an EXCLUDE naming an unknown id, naming a full-text
   inclusion id (never testable at abstract stage at all), failing the no-abstract or full-text
   checks above without a verbatim quote, or simply lacking a known id and verbatim quote.
5. **`unquoted_criterion`**. Any other criterion-naming NEEDS_REVIEW (an ordinary abstract-stage id,
   or a full-text inclusion id) with an empty or non-verbatim quote, so it is never folded into the
   unattributed/undecidable bucket alongside a record the model legitimately could not decide.

### Stage rules -- S8c and S8e stand unchanged

- A full-text inclusion criterion is never tested at abstract stage; an EXCLUDE naming one is
  always `unanchored_exclude`. On an INCLUDE, it is listed in `to_confirm` when not already
  confirmed; silence about it never grounds an EXCLUDE or NEEDS_REVIEW.
- A full-text exclusion criterion can never ground a standing EXCLUDE (see reason 2 above).
- Abstract-stage criteria and the research question (topic) are judged first; a record that fails
  one, or is off topic, is EXCLUDE on that ground and a full-text criterion is never reached. An
  off-topic EXCLUDE with no numbered criterion covering the mismatch names the reserved
  `TOPIC_CRITERION_ID = "TOPIC"`.

### Iteration-11 additions, from the van_Dis diagnosis's five patterns

- **Edit 1 (patterns P1 and P3, kinds not words).** A criterion that gives a category by listing
  kinds or examples is satisfied by a member of a listed kind, not only by a listed word or brand
  name; EXCLUDE only when the shown text names something plainly outside the category, NEEDS_REVIEW
  otherwise.
- **Edit 2 (pattern P2, absence is not failed by silence).** A criterion phrased as an absence is
  failed only when the shown text shows the thing missing from the study as a whole; naming it
  anywhere, in any arm, satisfies the criterion regardless of what else the study contains, and
  silence about it is NEEDS_REVIEW, not EXCLUDE.
- **Edit 3 and the guard rule (pattern P5, no abstract).** The prompt states outright that a record
  with no abstract can never ground an EXCLUDE on any criterion and can never be judged off topic;
  the guard enforces this unconditionally via the `no_abstract` reason above, closing the gap the
  diagnosis found (a quote copied from the title alone had been passing the ordinary anchor test).

The iteration-10 clause (full-text-exclusion carve-out) and the iteration-12 clause (defined terms
govern everywhere) are not part of this list; both were trialed and are recorded in section 1 as
rejected, not frozen, changes.

## 4. Protocol JSON in use: sha256 and a line-ending fix

`protocol_sha()` is `sha256:` plus the first 12 hex digits of the raw file bytes, read directly off
disk (`Path.read_bytes()`), recorded in every run's meta file.

### The fix

Six of the seven protocol files in `evaluation/screening/protocols/` were checked out on this
machine with CRLF line endings, while the git blob committed for every one of them was already LF
-- confirmed by comparing `git show HEAD:<path>` against a byte-for-byte CRLF-to-LF conversion of
the working-tree copy: the two matched exactly for all six files, with no other byte differing.
`Path.read_bytes()` reads whatever is actually on disk, so `protocol_sha()` on an unpatched
Windows checkout of this repository returns a value that differs from the sha of the file git
itself has always stored. This freeze adds `.gitattributes` with
`evaluation/screening/protocols/*.json text eol=lf` so every future checkout gets LF regardless of
the checking-out machine's `core.autocrlf` setting, and rewrites the six affected files' on-disk
bytes to LF, with no other byte changed. `git diff --stat` against `HEAD` on all six files is empty
after the rewrite: the working tree now matches the blob that was already committed.

The six files, old (CRLF-on-disk) sha next to the new (LF, matching the committed blob) sha:

| file | old sha (CRLF, this checkout) | new sha (LF, matches `git show HEAD`) |
|---|---|---|
| `Anmarkrud_2021.json` | `sha256:8d226773e606` | `sha256:f2084fcad5fd` |
| `Fong_2021.json` | `sha256:a580233b9749` | `sha256:bd28c18aa6a9` |
| `Nagtegaal_2019.json` | `sha256:60766fc0f43f` | `sha256:331f71b49dc3` |
| `Smid_2020.json` | `sha256:4311a6aa5e2c` | `sha256:c4eb8a2c80c1` |
| `Taschner_2024.json` | `sha256:6affc3a61987` | `sha256:7b5ca48b41e1` |
| `van_de_Schoot_2017.json` | `sha256:fd8b5fc2df11` | `sha256:e271b929c830` |

`van_Dis_2020.json` was already LF on disk; its sha is unaffected by this fix (its own history is
section 5).

**What this means for already-committed run metadata.** Every run of `Nagtegaal_2019`, `Smid_2020`
or `van_de_Schoot_2017` under iteration 11 -- `results/v2/*_runA.meta.json` and the superseded
iteration-9 files in `results/v2-superseded-2026-09-07/` -- recorded `protocol_sha` as computed
from the CRLF-on-disk bytes at the time each run executed, i.e. the "old" column above. Read against
the working tree as it stands after this freeze's rewrite, those recorded values now differ from
what `protocol_sha()` returns on the same file. Nothing about the protocol's actual content changed
by even one visible character -- the criteria text, wording and every prior fidelity note are
identical before and after; only the line-ending bytes on this one checkout changed, and only to
match what the repository's own git history already stored. The runs remain valid measurements of
the same criteria text; the sha difference is a checkout artifact this freeze fixes going forward,
not a protocol edit, and it is recorded here so it is never mistaken for one.

### Current table

| dataset | sha256 (full) | `protocol_sha()` | `primary_label` | `label_field` | full-text criterion ids |
|---|---|---|---|---|---|
| `van_Dis_2020` | `5d6b603ae5790207fde48bd37da076259ae13f7eb99136473365446133ac74c7` | `sha256:5d6b603ae579` | `label_included` | `label_abstract_screening` | I4, I6, I7, E2 |
| `Fong_2021` | `bd28c18aa6a9ad0bb01e9eb7aa47280accba7f8476a17a002dbb5dbb13df5a17` | `sha256:bd28c18aa6a9` | `label_included` | `label_abstract_screening` | I1, I3, E1, E3 |
| `Nagtegaal_2019` | `331f71b49dc36ab3c65741fc5d7a3f048b3f4eed8ac3da22cb2fc4799dd2573c` | `sha256:331f71b49dc3` | `label_included` | `label_abstract_screening` | E5 |
| `Smid_2020` | `c4eb8a2c80c1ac380ad117a9e3e543c1438fb44c9cbd90ef923d475bb6b0925d` | `sha256:c4eb8a2c80c1` | (no key; defaults to `label_included`) | `label_included` | I2, I6, E4, E7, E8, E10 |
| `van_de_Schoot_2017` | `e271b929c8306cde702a081caba6aa647612b0e979e0aa2df713acdf38dbdd53` | `sha256:e271b929c830` | (no key; defaults to `label_included`) | `label_abstract_screening` | I2, E4 |
| `Anmarkrud_2021` | `f2084fcad5fdd4e4f3f14bdf29a5a237eb3e6da107282e24a71f59e1c6155c3f` | `sha256:f2084fcad5fd` | -- withdrawn candidate, no committed run -- | | |
| `Taschner_2024` | `7b5ca48b41e1ef97ab2e053eef70c2f5ff2132c084056a3100f38292f804f0f2` | `sha256:7b5ca48b41e1` | -- withdrawn candidate, no committed run -- | | |

### 2026-09-14 staleness note

The "Current table" above is frozen as of this file's own iteration-11 freeze and is **not**
updated in place by this note, per section 10's "no further edit" rule for the frozen record
itself. Two exploratory protocol edits, both outside this freeze's scope, have since changed the
on-disk bytes of `Nagtegaal_2019.json` and `Smid_2020.json`; this note records what changed, why,
and the new digests, so a committed v2 run meta read against the current tree is never mistaken
for a reproduction of the frozen protocol.

**The first edit** (commit `e6b373f`) marked
`Nagtegaal_2019.json` I1 and its negation E1 `stage: "full_text"` (the Munscher, Vetter and
Scheuerle nudge-taxonomy classification is not decidable from a title and abstract) and
`Smid_2020.json` E9 `stage: "full_text"` (whether a named model falls under the review's own SEM
definition is the same kind of external-classification judgement). Digests after the first edit:

| file | frozen `protocol_sha()` (table above) | after first edit `protocol_sha()` | full-text criterion ids after first edit |
|---|---|---|---|
| `Nagtegaal_2019.json` | `sha256:331f71b49dc3` | `sha256:1ae48c142315` | I1, E1, E5 |
| `Smid_2020.json` | `sha256:c4eb8a2c80c1` | `sha256:c70f0897c136` | I2, I6, E4, E7, E8, E9, E10 |

**The second edit** additionally marked `Smid_2020.json` I4 ("Bayesian estimation was compared to frequentist
estimation methods...") and its negation E11 ("No comparison of estimation methods.")
`stage: "full_text"`, for the same external-classification reason as E9: whether the performance
of Bayesian and frequentist estimation was investigated for the exact same model, and whether a
named estimator is frequentist under the review's own definition, are Methods-section facts, not
properties an abstract states about itself. `Nagtegaal_2019.json` is unchanged by the second edit.
Digest after the second edit:

| file | after first edit `protocol_sha()` | after second edit `protocol_sha()` | full-text criterion ids after second edit |
|---|---|---|---|
| `Smid_2020.json` | `sha256:c70f0897c136` | `sha256:5ff574cb49e8` | I2, I4, I6, E4, E7, E8, E9, E10, E11 |

Neither edit touches `SCREENER_PROMPT_V2`, `SCREENER_PROMPT_V1`, the guard
constants, the abstract cap, or any metric definition recorded in sections 1 to 9; the frozen
prompt digest (`sha256:fb9de89e0534`) and the production `SCREENER_PROMPT` pointer are unchanged
by either edit. A `FREEZE_v3.md` cut after the full re-freeze (not part of either edit) is the
place these two files' rows are updated in a frozen table of their own; until then, any run whose
meta reports `protocol_sha` other than one of the values in this note (for one of these edits) or
the "Current table" above (for anything else) is measuring one of these two intermediate states,
not a fresh, uncommitted edit.

## 5. `van_Dis_2020` I5 and E1: fidelity corrections, both dated 2026-09-07

Both corrections are quoted verbatim, with their source sentence and page, in the protocol file's
own `notes` field; this section summarises them and their dates.

**First correction.** The committed I5 read "CBT is compared with one
of the following control conditions: ...", a closed six-item list. The source review's own sentence
(JAMA Psychiatry 2020;77(3):265-273, Methods, "Inclusion Criteria", p. 266) reads "Comparison groups
**included** care as usual ... relaxation, psychoeducation, pill placebo, supportive therapy, or
waiting list" -- an open enumeration, not a closed list. I5 was corrected to restore that wording.
E1 read "The study did not use CBT (e.g., applied relaxation, ...)"; read together with I5's own
listing of relaxation as an eligible comparator, the only internally consistent reading is that the
parenthetical names the therapy under evaluation, not a comparator, so E1 was reworded to "The
therapy the study evaluates is not CBT (...)" with an added sentence: "A trial that evaluates CBT
alongside one of these therapies has used CBT and does not fail this criterion." Both corrections
are marked `source: "paper"`, `page: "266"` in `criteria_provenance`.

**Second correction (commit `af58b8a`).** The first correction had also
added a first sentence to I5 ("The comparison group is a control condition rather than a second
active psychotherapy under test."), sourced from the review's Study Selection line rather than from
the p. 266 sentence I5 otherwise quotes. Review found this narrowed the source's own open
enumeration in a direction that would exclude two records the review itself included (record 35, no
care-as-usual/waiting-list/placebo arm at all; record 24, CBT evaluated alongside EMDR and
relaxation training). The added sentence was removed; I5 now reads exactly the source's own
sentence. `criteria_provenance.exclusion[E1]`'s `derived` flag was corrected from `false` to `true`,
since E1's added second sentence is an inference from internal consistency with I5, not a further
quotation from p. 266.

Both corrections landed before the iteration-11 run that measured them: `protocols/van_Dis_2020.json`
hashes to `sha256:5d6b603ae579` both as committed now and as recorded in
`results/dev/v2-iter11/van_Dis_2020_runA.meta.json`. The original, uncorrected wording
(`sha256:125deb0a7008`) is what produced the pre-diagnosis held-out runs, `results/v2/van_Dis_2020_run{A,B}`
at commit `3d20f19` (section 6).

## 6. `van_Dis_2020`: from held-out set to development set

**Original choice (2026-09-06).** Per an abstract-coverage
audit, `van_Dis_2020` was the sole candidate in the audited pool clearing both floors (>= 0.80
abstract coverage, >= 40 abstract-bearing positives): 0.960 coverage, 66 abstract-bearing positives,
73 final inclusions, exempted from the amendment's 6,500-record cap as its named fallback (10,953
records). `Fong_2021` stayed the dev-tuning set throughout.

**The pre-diagnosis miss, commit `3d20f19`.** The frozen held-out runs
A and B (prompt `sha256:1ffc09342901`, protocol `sha256:125deb0a7008`, uncorrected I5/E1 wording)
were scored and returned **REJECT**: H1 (INCLUDE-only recall, abstract-bearing positives) 38/66 = 0.576 (run A)
and 39/66 = 0.591 (run B) against a 0.70 bar; H2 (screened-in recall, all positives)
47/73 = 0.644 and 49/73 = 0.671 against a 0.85 bar; A/B kappa on the three-way status 0.8164
against a 0.85 bar. A later check found this scoring had used bars the acceptance
amendment does not state -- the binding bar is H1 >= 0.75 (binding 50, short 12 and
11) and H2 >= 0.90 (binding 66, short 19 and 17) -- which changes the arithmetic but not the
outcome: REJECT either way.

**The diagnosis retires the set, 2026-09-07.** A diagnosis pass read `van_Dis_2020`'s own
run rows -- title, abstract, criterion, quote and the model's own reason on every one of the 28
positives excluded on either run -- to find what was wrong and to write the section 5 protocol
corrections. The protocol's own `notes` field records the consequence: any number re-measured on
`van_Dis_2020` after 2026-09-07 is a development-set number, not a held-out one. `van_Dis_2020` is
therefore no longer usable as a held-out acceptance set for any later claim, and
the same coverage-audit rule found no other candidate in the audited pool clearing both
floors, so no replacement held-out set exists in this repository as of this freeze.

**Development numbers, iteration 11 (`results/dev/v2-iter11/`, not owned by this freeze but recomputed
independently and cited here for the record).**

`van_Dis_2020` run A: n = 10,953, positives = 73, abstract-bearing positives = 66, prevalence =
0.006665. H1 38/66 = 0.5758 (fail against 0.75, binding 50, short 12; fail against the 0.80 stretch,
binding 53, short 15). H2 48/73 = 0.6575 (fail against 0.90, binding 66, short 18). H3 (needs-review
share, abstract-bearing) 1,277/10,515 = 0.1214 (pass against 0.25 and against the 0.15 stretch). H4
(INCLUDE rate, all records) 766/10,953 = 0.0699 (pass against `max(0.25, 5 x prevalence) = 0.25`).
H6 (zero padded decisions, zero unanchored EXCLUDE surviving the guard): pass. No run B exists at
this iteration, so H5 (A/B kappa) is not computable for `van_Dis_2020`. Net effect of the three
prompt edits and two protocol corrections on H1: zero -- nine positive decisions changed and they
cancel (four gained, four lost, one guard-demoted). H2 gained exactly one record, the no-abstract
positive the new guard rule now catches.

`Fong_2021` run A: n = 1,299, positives = 122, abstract-bearing positives = 91, prevalence =
0.093918. H1 74/91 = 0.8132 (pass against 0.75, margin 5; pass against the 0.80 stretch, margin 1).
H2 114/122 = 0.9344 (pass against 0.90, margin 4). H3 171/822 = 0.2080 (pass against 0.25,
headroom 35; **fails** the 0.15 stretch, over by 47). H4 140/1,299 = 0.1078 (pass against
`max(0.25, 5 x prevalence) = 0.4696`). H6: pass. `Fong_2021` run B (`results/dev/v2-iter11/`):
INCLUDE-only recall 75/122 = 0.615, screened-in recall 115/122 = 0.943; A/B Cohen's kappa on
the three-way status = 0.9266, on include versus not = 0.9326 (H5 pass on either, >= 0.85).
**Correction:** an earlier version of this paragraph cited
0.9326 as the three-way status kappa. 0.9326 is `summarize.py`'s `agreement_AB.kappa`, the kappa on
the binary include/not-include column; the three-way status kappa is `agreement_AB_status.kappa` =
0.9266. Both pass H5; only the label was wrong.

**Classification of the residual misses (iteration 11, recomputed
independently from the run rows and quotes, not from `summary.json`).** `van_Dis_2020` run A: 25
positives still hard EXCLUDE, all abstract-bearing -- 20 model error, 4 criteria effect, 1 label
noise. `Fong_2021` run A: 8 positives still hard EXCLUDE -- 6 model error, 1 criteria effect, 1
label noise. Combined: **26 model error, 5 criteria effect, 2 label noise, 33 records**. Model
error is the shown text supporting inclusion or NEEDS_REVIEW under the criteria as written;
criteria effect is the protocol as written excluding a record the source review nonetheless
included; label noise is the record failing the review's own criteria on the shown text, so the
positive label itself is not supportable.

**Escalation to a stronger model was evaluated and rejected.** A stronger model, tried on the
residual misses above, resolved most of them, but at cost and latency that would exceed a single
job's 30-minute time budget on a dataset with a larger excluded share than `van_Dis_2020`'s own.
The screening path does not call a second, stronger model by default; the residual misses above
are reported as measured.

## 7. Superseded runs directory

`evaluation/screening/results/v2-superseded-2026-09-07/` holds the iteration-9 legacy runs for
`Nagtegaal_2019`, `Smid_2020` and `van_de_Schoot_2017` (prompt `sha256:1ffc09342901`, 7 and 14
padded decisions on `Nagtegaal_2019` and `van_de_Schoot_2017` respectively), moved out of
`results/v2/` by `git mv` (commit `49a5332`) ahead of re-running the same three datasets under the
shipped iteration-11 prompt into `results/v2/` itself (section 8). A `README.md` in that directory
records the same reason. The files are otherwise untouched: same rows, same decisions, same
provenance, only relocated.

## 8. Legacy re-runs under the shipped iteration-11 prompt

`Nagtegaal_2019`, `Smid_2020` and `van_de_Schoot_2017`, run A, prompt `sha256:fb9de89e0534`, model
`deepseek-v4-flash`, temperature 0, record-order seed 20260902 (unchanged from the superseded
runs), committed at `a29225f`.

| dataset | n | padded / failed / unscreened | needs-review-by-reason (no_abstract, unanchored_exclude, undecidable, full_text_criterion) | INCLUDE-only recall | screened-in recall | needs-review share (abstract-bearing) | INCLUDE rate | prevalence (primary / protocol label) | cost (USD) |
|---|---|---|---|---|---|---|---|---|---|
| `Nagtegaal_2019` A | 2,019 | 0 / 0 / 0 | 167, 63, 5, 1 | 0.881 (89/101) | 0.891 (90/101) | 0.0362 | 0.1387 | 0.0500 (101/2019) / 0.1942 | 0.325636 |
| `Smid_2020` A | 2,627 | 0 / 0 / 0 | 640, 35, 7, 4 | 0.704 (19/27) | 0.852 (23/27) | 0.0232 | 0.0171 | 0.0103 (27/2627) / 0.0103 | 0.333001 |
| `van_de_Schoot_2017` A | 6,189 | 0 / 0 / 0 (1 batch failed and was cleared by one `--retry-failed`) | 760, 161, 6, 0 | 0.930 (40/43) | 0.930 (40/43) | 0.0304 | 0.0204 | 0.00695 (43/6189) / 0.0627 | 0.925597 |

"Needs-review-by-reason" here is `summarize.py`'s `needs_review_by_reason` categorisation of all
NEEDS_REVIEW rows, not the row-level `guard_reason` field (the two differ). No dataset needed a
second `--retry-failed`.
Total measured spend for these runs (these three plus `Fong_2021` run B) was $1.756011,
within the $2.08 remaining budget at the time.

### Two corrections

Neither correction changes any number's sign or the run itself; both are read from the same
committed `.jsonl` rows.

**H1 as section 9 defines it is scored on abstract-bearing positives only.** The "INCLUDE-only
recall" column in the table above is `recall_include_only` from `summary.json`, divided by every
`label_included` positive whether or not it has an abstract. H1 divides by the abstract-bearing
positives only. The two differ on all three sets:

| dataset | run | table figure above (all positives) | H1 (abstract-bearing positives only) |
|---|---|---|---|
| `Nagtegaal_2019` | A | 0.881 (89/101) | 0.8900 (89/100) |
| `Smid_2020` | A | 0.704 (19/27) | 0.8261 (19/23) |
| `van_de_Schoot_2017` | A | 0.930 (40/43) | 0.9286 (39/42) |

**A no-abstract positive can reach INCLUDE, not only NEEDS_REVIEW.** The guard (section 3) demotes
an EXCLUDE on a no-abstract record; it does not touch an INCLUDE on one, and the model itself does
return INCLUDE for a no-abstract positive on these corpora. Three examples, read directly from the
row data: `van_de_Schoot_2017` record 898 on run A (status INCLUDE, `has_abstract` false; the same
record fell back to NEEDS_REVIEW on run B), and `Fong_2021` records 899 and 1144 (status INCLUDE,
`has_abstract` false, on both run A and run B). This is why `van_de_Schoot_2017` run A's H1
numerator (39) is one below its all-positives INCLUDE-only numerator (40): record 898 counts
towards the all-positives figure but falls outside the abstract-bearing denominator entirely.

### Run B added for the three legacy sets

`Nagtegaal_2019`, `Smid_2020` and `van_de_Schoot_2017`, run B, same prompt, model, temperature,
abstract cap and record-order seed as run A above, into `results/v2/`. All three completed with 0 failures, 0 unscreened
records and 0 padded decisions; none needed `--retry-failed`.

| dataset | run | H1 (abstract-bearing) | H2 = screened-in recall (all positives) | H3 (NEEDS_REVIEW share, abstract-bearing) | H4 (INCLUDE rate) | cost (USD) |
|---|---|---|---|---|---|---|
| `Nagtegaal_2019` | B | 0.9000 (90/100) | 0.9109 (92/101) | 0.0373 | 0.1382 | 0.325723 |
| `Smid_2020` | B | 0.7391 (17/23) | 0.8148 (22/27) | 0.0237 | 0.0152 | 0.332954 |
| `van_de_Schoot_2017` | B | 0.9286 (39/42) | 0.9535 (41/43) | 0.0324 | 0.0218 | 0.924774 |

A/B agreement, both readings `summarize.py` computes (`agreement_AB.kappa` is the binary
include/not-include column; `agreement_AB_status.kappa` is the three-way status):

| dataset | binary include kappa A/B | three-status kappa A/B |
|---|---|---|
| `Nagtegaal_2019` | 0.8900 | 0.8622 |
| `Smid_2020` | 0.8685 | 0.9563 |
| `van_de_Schoot_2017` | 0.8943 | 0.9049 |

All three clear H5 (>= 0.85) on both readings. With run B added, `Nagtegaal_2019` and
`van_de_Schoot_2017` each pass H2 on at least one run (`Nagtegaal_2019` run B 92/101 = 0.9109;
`van_de_Schoot_2017` run A and B both pass, 40/43 = 0.9302 and 41/43 = 0.9535); `Smid_2020` misses
H2 on both runs (23/27 = 0.8519 and 22/27 = 0.8148). This does not reopen or reverse the acceptance
decision, which was reached on run A alone; it is recorded here as the
additional reproducibility evidence the acceptance report's H5 row asked for.

## 9. The internal acceptance bar

Six criteria, evaluated per dataset on run A (and run B when it exists). This numbering
folds an original held-out H1/H2/H3(two-part)/H5/H6/H7 scheme into six items for a
per-iteration, per-run-pair report; the six criteria measured are identical, only the numbering is
consolidated.

| id | measure | bar | stretch |
|---|---|---|---|
| H1 | INCLUDE-only recall, abstract-bearing positives | >= 0.75 | 0.80 |
| H2 | screened-in recall (INCLUDE or NEEDS_REVIEW), all positives | >= 0.90 | 0.90 (no further stretch) |
| H3 | NEEDS_REVIEW share of abstract-bearing records | <= 0.25 | 0.15 |
| H4 | INCLUDE rate, all records | <= `max(0.25, 5 x prevalence)` | dataset-dependent; no fixed stretch |
| H5 | A/B Cohen's kappa on the three-way status, when both runs exist | >= 0.85 | -- |
| H6 | zero padded decisions and zero unanchored EXCLUDE surviving the guard | 0 and 0 | -- |

A miss is reported as a miss (section 6); nothing in this freeze routes around H1 or H2 by loosening
a bar, and no code path was added to reach for a stronger model by default (section 6, the
escalation finding).

## 10. No further edit

No prompt, guard, protocol or metric edit follows this freeze beyond what sections 1 to 9 already
record. Any run scored against this record whose reported prompt sha, protocol sha, cap, or
gate/metric definition disagrees with sections 1 to 9 above is discarded and re-run, not scored, per
the same rule `run_screening.py --prompt-version-expect` already enforces for the prompt sha alone.
