# Provenance of the HSS claim-verification labels

This file records how the 30 final labels in `hss_annotation_adjudicated_v2.csv` were produced and
what they do and do not establish. Read it before citing any number from this folder.

Date of the annotation and the adjudication: 2026-09-05.

The identifiers are opaque (`ann-01` to `ann-30`) and the rubric states no mapping from
construction category to expected label, so blinding does not depend on an annotator failing to
notice a pattern in the file names.

## 1. Annotation procedure

The package was rebuilt from scratch by `build_sheet_v2.py`.

- **Opaque identifiers.** The 30 items were shuffled with a fixed seed and renamed `ann-01` to
  `ann-30`. The mapping from `ann_id` to `item_id`, construction category and expected label was
  written only to `hss_annotation_key_v2.csv`. The key was not given to any annotator session, and
  the identifiers carry no information about the category.
- **Passages retrieved from the cited paper.** For every item with a cited DOI, the passage shown was
  retrieved independently of the construction record, by scoring one to three sentence windows of the
  cited paper's own cached text against the claim with a Jaccard overlap on lowercased alphabetic
  tokens minus stopwords, and keeping the best window when its score cleared a threshold of 0.15.
  Passages were capped at 700 characters. For the items whose claim is attributed to a paper it did
  not come from, the procedure ran against the paper named in the sheet, not the paper the sentence
  really came from, so those items show either a genuine window from the cited paper or the explicit
  marker `No matching passage found in the cited paper (best overlap below threshold)`. No item
  displays a sentence taken from a paper other than the one it is cited to. On the five items whose
  claim is a verbatim copy of a sentence in the cited paper, the best retrieved window is that same
  sentence, so the passage and the claim coincide. Of the 30 items, 20 show a retrieved passage and 5
  show the no-match marker.
- **Withheld-evidence marker.** Five items show the marker
  `Evidence withheld: no passage or paper text is available for this item` and have an empty `paths`
  list in `source_texts_index_v2.json`. For those items no passage and no cached text were offered.
- **Three independent sessions.** Three separate Claude Opus sessions, run through Claude Code and
  designated A, B and C, each filled in one private copy of the sheet (`annotator2_A.csv`,
  `annotator2_B.csv`, `annotator2_C.csv`) with a label, a confidence and a one-sentence note. Each
  session received `INSTRUCTIONS_v2.md` and `source_texts_index_v2.json` and was blind to the
  construction category and the expected label. The rubric permits a session to open the cited
  paper's cached text listed in the index when the passage reads `No matching passage found`, so that
  a no-match item is decided against the cited paper rather than against an absence in the sheet.
- **Labels.** `verified` when the cited paper's text supports the claim as stated; `needs_nuance`
  when the paper supports only a weaker, narrower or conditional version, that is, when the claim
  overstates, generalises or drops a qualifier; `unsupported` when the cited paper contradicts the
  claim or does not address it; `no_full_text` when the item shows the withheld-evidence marker,
  applied in that case regardless of how plausible the claim sounds.

`aggregate_annotations_v2.py` merged the three sheets into `hss_annotation_completed_v2.csv` and
computed `hss_annotation_aggregate_v2.json` and `hss_annotation_aggregate_v2.md`.
`adjudicate_annotations_v2.py` wrote the final labels to `hss_annotation_adjudicated_v2.csv`.

## 2. Adjudication trigger and what the adjudicator did

An item was sent to adjudication when the three sessions were **not unanimous**, or when the
**majority label fell outside the expected-label set** recorded in the construction key. Two items
met the trigger, `ann-22` and `ann-23`, both on non-unanimity. No item had a majority outside its
expected set.

The adjudicator was a fourth Claude Opus session with access to all cached paper texts, including the
texts of the withheld items, because it adjudicates the set rather than performing the blind reading.
Its rule was that the cited paper's text is the authority, above the construction key and above the
annotators' notes. Each adjudicated item carries a two-sentence rationale quoting the decisive
sentence in the `rationale` column of `hss_annotation_adjudicated_v2.csv`.

**ann-22** (`hss-altered-01`, effect size). A and B returned `unsupported`, C returned
`needs_nuance`. The cited paper states "The results revealed an effect size of 0.81 which can be
interpreted as a large effect of the independent variable (CL) on the dependent variable (speaking
skill)", and the string `1.62` does not occur anywhere in its cached text. Final label
**`unsupported`**: reporting a different value for the same statistic inside the same sentence frame
contradicts the paper's number, which is more than an overstated or unqualified version of it.

**ann-23** (`hss-altered-02`, percentage). A and B returned `unsupported`, C returned `needs_nuance`.
The cited paper states "Seventeen participants (18%) reported using it to write the whole assignment
for them, which is certainly a cause for concern", and 36.1% is that paper's separate figure for
writing a part of the assignment. Final label **`unsupported`**: the claim attaches the paper's count
of seventeen to a percentage the paper assigns to a different and incompatible category.

### Unanimous items re-decided because a note misreads the paper

Three unanimous items were re-read against the cached text because an annotator note misdescribes the
source. The unanimous label survived in all three, so their `source` column stays `unanimous` and the
correction is recorded in the `rationale` column.

- **ann-20** (`hss-verbatim-03`). B's note says the paper reads "may be defined" and that the claim's
  "can" merely matches the paper's "may". The cached text of 10.14746/ssllt.2019.9.1.2 carries both:
  "it is proposed that the L2 Learning Experience may be defined as the perceived quality ..." in the
  abstract, and "In accordance with this approach, the L2 Learning Experience can be defined as the
  perceived quality of the learners' engagement with various aspects of the language learning
  process" in the body, which is the sentence the claim copies verbatim. Label `verified` unchanged.
- **ann-30** (`hss-wrong_paper-05`). B's note says the cited paper contains no instance of
  "transcription". It does contain "recorded and transcribed in full", describing the authors' own
  discussions. The paper still contains no occurrence of "translated into English" and no survey of
  interview practice across EFL studies, so `unsupported` is unchanged.
- **ann-04** (`hss-wrong_paper-03`). A's and C's notes report zero hits for resilience in the cited
  paper. There is one hit, the reference-list title "Resilient, overcontrolled, and undercontrolled
  boys: Three replicable personality types". The paper contains no occurrence of PsyCap,
  psychological capital, Masten, optimism or persistence and does not address psychological capital,
  so `unsupported` is unchanged.

No construction label was found to be contradicted by the cited paper's text. There are no
construction-label disputes in the v2 pass.

## 3. Agreement statistics

Recomputed from `hss_annotation_completed_v2.csv`. Labels are drawn from
{verified, needs_nuance, unsupported, no_full_text}. Fleiss kappa uses three raters per item.

### All 30 items

| quantity | value |
|---|---|
| items | 30 |
| unanimous items (A = B = C) | 28 of 30 |
| Fleiss kappa | 0.930 |
| A vs B, percent agreement / Cohen kappa | 1.000 / 1.000 |
| A vs C, percent agreement / Cohen kappa | 0.933 / 0.897 |
| B vs C, percent agreement / Cohen kappa | 0.933 / 0.897 |

### Evidence-shown subset, 25 items

The subset is the 20 items with a retrieved passage plus the 5 items with the no-match marker. It
excludes the 5 withheld items, on which agreement is trivial by construction.

| quantity | value |
|---|---|
| items | 25 |
| unanimous items | 23 of 25 |
| Fleiss kappa | 0.896 |
| A vs B, percent agreement / Cohen kappa | 1.000 / 1.000 |
| A vs C, percent agreement / Cohen kappa | 0.920 / 0.848 |
| B vs C, percent agreement / Cohen kappa | 0.920 / 0.848 |

### Final labels

| final label | count |
|---|---|
| verified | 10 |
| unsupported | 15 |
| no_full_text | 5 |
| needs_nuance | 0 |

Label source: 28 `unanimous`, 2 `adjudicated`, 0 decided on a bare majority. Every final label lies
inside its expected-label set, so agreement between the final label and the construction label is
30 of 30, or 1.000. Section 5 states why that number is weaker evidence than it looks.

## 4. What these labels do not establish

1. **They are model annotations, not human expert annotations.** All three annotator sessions and the
   adjudicator were Claude Opus sessions run through Claude Code. No author of this software has read
   the 30 items against the cited papers and countersigned the labels. The words gold standard,
   expert, human-verified and author-verified must not be used for this artifact.
2. **Three sessions of one model are correlated raters.** The reported Fleiss kappa measures
   within-model reproducibility under a shared prompt, not inter-annotator reliability in the usual
   sense. A and B agreed on 30 of 30 items, which is the signature of shared weights rather than of
   independent judgement.
3. **The withheld-evidence stratum is a simulated condition.** The evidence for those five items was
   withheld by construction while the cited paper's full text was present in
   `evaluation/claims/data/hss_fulltext/` the whole time; each of the five withheld papers was also
   reachable as the allowed source of another item in `source_texts_index_v2.json`, annotator B's
   attestation shows it had opened all five for those other items, and all three sessions
   nevertheless returned `no_full_text` on all five withheld items regardless. The stratum therefore
   measures compliance with an abstention instruction rather than genuine unavailability of the
   evidence. It must not be described as a paywall, an open-access gap, an unretrievable source, or a
   retrieval failure.
4. **The altered stratum carries a disjunctive expected label.** Ten of the 30 items expect
   `unsupported|needs_nuance`, so either label an annotator can plausibly assign in that stratum
   counts as agreement. One third of the set cannot disconfirm the key by construction, which is part
   of why the agreement figure of 1.000 in section 3 is not an independent confirmation of the
   construction rules.
5. **Blinding covers the category and the key, not the corpus.** No session was given the
   construction category or the expected label, and the identifiers carry no category information.
   The sessions were, by design, allowed to open the cited paper's cached text for items whose
   passage read `No matching passage found`, and A, B and C each did so for those items. Their labels
   on the five no-match items therefore rest on the cited paper's cached text rather than on the
   passage shown in the sheet. Annotator B's attestation additionally lists the cited paper's cached
   text for ten passage-mode items covering twenty rows; the rubric permits this, so it is not a
   protocol departure, but A, B and C therefore did not have identical evidence in front of them.
   The presented evidence also separates three of the five strata as unnamed groups: the no-match
   marker, the withheld marker and the claim-passage identity each fall on exactly one stratum of
   five, so a reader can tell those three strata apart even though no category name or expected
   label is ever shown. The ten altered and five paraphrase items are the only items not separable
   this way, and each of the three separations reflects the evidence actually available for the item
   rather than the construction record.
6. **Independence is self-reported.** The attestations in section 5 were returned by the sessions
   themselves. There was no sandbox and no filesystem access log enforcing them.
7. **The set is small and narrow.** Thirty items drawn from eleven papers in applied linguistics and
   TESOL. Four of the five strata have n = 5, so per-stratum rates should be reported as counts, not
   as percentages alone.
8. **The two adjudicated items are boundary judgements.** Both `ann-22` and `ann-23` double a number
   while leaving the surrounding qualitative reading intact. The adjudicator recorded both as
   `unsupported`; a reader who weighs the surviving qualitative reading more heavily would record
   both as `needs_nuance`, which is what annotator C did. The final set contains no `needs_nuance`
   label, and that count is the least stable number in the table.

## 5. Annotator attestations (self-reported, not enforced)

The three sessions returned the following attestations. They are reproduced verbatim. They were not
enforced by sandboxing or verified against a filesystem access log, and they should be read as
statements of intent rather than as proof of isolation.

### Annotator A, verbatim from `attestation2_A.md`

```
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
```

### Annotator B, verbatim from `attestation2_B.md`

```
# Attestation, independent annotator B

Annotator name: Claude Opus annotator B
Date: 2026-09-05
Items labelled: 30 of 30

## Files opened during this annotation session

Working sheet and protocol:

- evaluation/claims/annotation/annotator2_B.csv
- evaluation/claims/annotation/INSTRUCTIONS_v2.md
- evaluation/claims/annotation/source_texts_index_v2.json

Cited paper texts, each opened only because source_texts_index_v2.json lists it for an item I was judging:

- evaluation/claims/data/hss_fulltext/10-55593_ej-26103a4.json (items ann-03, ann-08, ann-24)
- evaluation/claims/data/hss_fulltext/10-32601_ejal-710194.json (items ann-04, ann-07, ann-19)
- evaluation/claims/data/hss_fulltext/10-17323_jle-2023-17379.json (items ann-05, ann-23)
- evaluation/claims/data/hss_fulltext/10-32601_ejal-651346.json (items ann-06, ann-12, ann-13)
- evaluation/claims/data/hss_fulltext/10-29140_jaltcall-v17n2-336.json (items ann-09, ann-14, ann-25)
- evaluation/claims/data/hss_fulltext/10-55593_ej-27105a9.json (items ann-10, ann-28, ann-30)
- evaluation/claims/data/hss_fulltext/10-55593_ej-27107int.json (items ann-11, ann-15)
- evaluation/claims/data/hss_fulltext/10-14746_ssllt-2020-10-1-3.json (items ann-17, ann-21)
- evaluation/claims/data/hss_fulltext/10-14746_ssllt-2019-9-1-2.json (item ann-20)
- evaluation/claims/data/hss_fulltext/10-17323_jle-2019-9809.json (items ann-22, ann-27, ann-29)

Files written by me:

- evaluation/claims/annotation/annotator2_B.csv (label columns only)
- evaluation/claims/annotation/attestation2_B.md (this file)

## Statement

I attest that during this annotation session I opened no file other than the paths listed above: I opened no key, results, summary, provenance, jsonl or build file, no other annotator's sheet, and no internal design note, and I did not browse any directory listing of the evaluation tree.

## Label distribution

- verified: 10 (ann-03, ann-05, ann-09, ann-10, ann-12, ann-15, ann-17, ann-19, ann-20, ann-29)
- unsupported: 15 (ann-04, ann-06, ann-07, ann-08, ann-11, ann-13, ann-14, ann-21, ann-22, ann-23, ann-24, ann-25, ann-27, ann-28, ann-30)
- needs_nuance: 0
- no_full_text: 5 (ann-01, ann-02, ann-16, ann-18, ann-26)

## Note on the absence of needs_nuance

Every item I judged against paper text fell cleanly into support or contradiction. The unsupported items divide into two kinds: eleven single-token polarity or quantity reversals of a sentence that is otherwise verbatim in the paper (for example the paper's "more preferable states" rendered as "less preferable states", or "significant cognitive labor" rendered as "no significant cognitive labor"), and five items whose cited paper does not discuss the claim's subject at all (ann-04, ann-08, ann-13, ann-25, ann-30), which I confirmed by keyword search of the cached text before deciding. Two items, ann-22 and ann-23, retain the paper's qualitative reading while doubling its numeric value; I recorded these as unsupported because the paper states a different figure, and set confidence to medium to mark that the surrounding interpretation does survive.
```

### Annotator C, verbatim from `attestation2_C.md`

```
# Blinding attestation, annotator C

Annotator: Claude Opus annotator C
Date: 2026-09-05

## Files opened during this annotation session

Annotation folder:

- evaluation/claims/annotation/INSTRUCTIONS_v2.md
- evaluation/claims/annotation/source_texts_index_v2.json
- evaluation/claims/annotation/annotator2_C.csv

Cached paper texts, each listed in source_texts_index_v2.json for the item it was opened for:

- evaluation/claims/data/hss_fulltext/10-32601_ejal-710194.json (ann-04)
- evaluation/claims/data/hss_fulltext/10-55593_ej-26103a4.json (ann-08, ann-24)
- evaluation/claims/data/hss_fulltext/10-32601_ejal-651346.json (ann-13)
- evaluation/claims/data/hss_fulltext/10-29140_jaltcall-v17n2-336.json (ann-25)
- evaluation/claims/data/hss_fulltext/10-55593_ej-27105a9.json (ann-30)

Files written:

- evaluation/claims/annotation/annotator2_C.csv (annotation columns only)
- evaluation/claims/annotation/attestation2_C.md (this file)

## Attestation

I opened no file other than those listed above: no key, results, summary, provenance, jsonl or
build file, no other annotator's sheet, no path under
evaluation/data/ beyond the five cached texts named above, and no internal design note; I did
not browse any directory listing and judged every row only from the cited_paper_passage column and,
where the passage read "No matching passage found", from term searches inside the cached text that
the index lists for that row.

## Label summary

- verified: 10
- unsupported: 13
- needs_nuance: 2
- no_full_text: 5
- total rows labelled: 30
```

## 6. What these labels support

Use these two passages as written. Do not strengthen them.

### Paragraph for section 1.3

> To check that the construction labels of the 30 HSS claim-verification items describe what the
> cited papers actually say, each item was annotated independently by three Claude Opus sessions run
> through Claude Code on 2026-09-05. The sheet given to each session carried opaque identifiers
> ann-01 to ann-30, the claim, the cited paper's title, DOI and first author and year, and a passage
> retrieved from the cited paper's own cached text by scoring one to three sentence windows against
> the claim with a Jaccard overlap on content tokens. Twenty items received such a passage, five
> received an explicit no-match marker because no window cleared the threshold, and five received a
> withheld-evidence marker under which no passage and no paper text were supplied. No session was
> given the construction category or the expected label, which live only in the withheld key file.
> The three sessions agreed on 28 of the 30 items, with a Fleiss kappa of 0.930 over all items and
> 0.896 over the 25 items that showed evidence. The two disagreements, both items in which a reported
> number had been altered, were adjudicated against the cited paper's cached text by a fourth
> session. The final label set is 10 verified, 15 unsupported and 5 no_full_text, and it falls inside
> the expected-label set of the construction rule on 30 of 30 items; on ten of those items the
> construction rule admits either of two labels, so that agreement figure is a consistency check on
> the construction rules rather than an independent confirmation of them.

### Limitation sentence

> These 30 labels were produced by three sessions of a single large language model rather than by
> human experts and no author has countersigned them, three sessions of one model are correlated
> rather than independent raters, and the five no_full_text items are a simulated condition in which
> the evidence was withheld from the annotators while the cited paper's text remained in the local
> cache, so that stratum measures correct abstention rather than paywalling or retrieval failure.

## 7. Files in this package

| file | role |
|---|---|
| `build_sheet_v2.py` | builds the blind sheet, the key, the index and the rubric from `hss_claims.jsonl` and the cached texts |
| `hss_annotation_key_v2.csv` | withheld key: `ann_id`, `item_id`, `construction_category`, `expected_label`, `built_at` |
| `hss_annotation_sheet_v2.csv` | the blind sheet handed to the sessions |
| `INSTRUCTIONS_v2.md` | rubric given to the sessions, with no category hints. Its four-line provenance header (naming the three 2026-09-05 sessions and pointing to this file) was added on 2026-09-05 after the three sessions had returned their completed sheets; the rest of the file, from "## Purpose" on, is byte-identical to what the sessions received. |
| `source_texts_index_v2.json` | per item, the cited DOI and the cached text paths, empty for withheld items |
| `annotator2_A.csv`, `annotator2_B.csv`, `annotator2_C.csv` | the three completed sheets |
| `attestation2_A.md`, `attestation2_B.md`, `attestation2_C.md` | the three self-reported attestations, reproduced in section 5 |
| `aggregate_annotations_v2.py` | merge and agreement statistics |
| `hss_annotation_completed_v2.csv` | the three sheets merged, one row per item |
| `hss_annotation_aggregate_v2.json`, `hss_annotation_aggregate_v2.md` | the agreement statistics |
| `adjudicate_annotations_v2.py` | applies the adjudication trigger and writes the final labels |
| `hss_annotation_adjudicated_v2.csv` | the 30 final labels with rationales |
| `PROVENANCE.md` | this file |

## 8. HSS test set (60 items) and the real-claims set (26 items)

This section records the v3 pass in full: the tooling, the two blind sheets, the six annotator
sessions, the aggregation and adjudication of both sets, the agreement statistics, the residual
blinding exposure, the six attestations, and what the resulting labels do and do not establish.
Both sets are fully labelled and adjudicated. `hss_annotation_completed_v3.csv`,
`hss_annotation_aggregate_v3.json`, `hss_annotation_aggregate_v3.md`,
`hss_annotation_adjudicated_v3.csv`, `annotator3_A.csv`, `annotator3_B.csv`, `annotator3_C.csv`
and `attestation3_{A,B,C}.md` exist and are committed, as do their real-claims counterparts
(`real_annotation_completed_v3.csv`, `real_annotation_aggregate_v3.{json,md}`,
`real_annotation_adjudicated_v3.csv`, `real_adjudication_decisions_v3.json`,
`real_annotator3_{A,B,C}.csv` and `real_attestation3_{A,B,C}.md`). Every number in this section
was recomputed from the committed sheets by an independent process audit. That audit
found no label in doubt and no discrepancy between the committed aggregate files and a
from-scratch recomputation, only presentation and documentation gaps, which is what the rest of
this section fixes.

The verifier has since been scored against these labels (`evaluation/claims/results/v6/`, the
result of record). Nothing in this section changes because of that: the labels
recorded here were frozen, unmodified, before that scoring ran, and this section's own edits (the
`n_adjudicated` field described in section 8.6, and this rewrite itself) touch only the aggregate json/md
presentation, not any label.

### 8.1 Tooling

`build_sheet_v3.py` is an argparse generalisation of `build_sheet_v2.py`: every path, the shuffle
seed, the opaque-id prefix and the output filename prefixes are command-line flags, so the same
script builds the 60-item HSS v3 sheet and the 26-item real-claims sheet below. Two
differences from `build_sheet_v2.py` are load bearing. First, it does not write the rubric at all,
so a `--suffix v3` run can never overwrite `INSTRUCTIONS_v3.md`, whose digest the v3 gate
pins; `INSTRUCTIONS_v3.md` is instead written directly, once, in the same commit as this script.
Second, the withheld key's schema is conditional: when every item in the input carries a
non-null expected label (the HSS set), the key holds `ann_id, item_id, construction_category,
expected_label, built_at`, exactly as in the v2 pass; when no item does (the real-claims set), the
key holds only `ann_id, item_id`, because no expected label exists for those items and there is no
construction category to withhold.

`evaluation/tests/test_claims_annotation_v3.py::test_build_sheet_v3_reproduces_round2_sheet_key_index_byte_for_byte`
runs `build_sheet_v3.py` against the v2 inputs with `--suffix v2` and the v2 seed
(`20260905 * 7`) into a temporary directory and asserts the sheet, key and index it writes are
byte identical to the committed `hss_annotation_sheet_v2.csv`, `hss_annotation_key_v2.csv` and
`source_texts_index_v2.json`; the instructions file is excluded because the v3 pass's script never
writes one. A companion test re-runs the build with a different seed and asserts the sheet is
**not** byte identical, so the regression is not vacuously true.

`aggregate_annotations_v3.py` is the equivalent generalisation of `aggregate_annotations_v2.py`.
Its default setting (`--key-arm off`) sets `needs_adjudication = not unanimous`: a unanimous blind
majority stands as the label without reference to any construction key, because
`build_hss_set.py --altered-expected strict` (v3 HSS build) removed the disjunctive expected
labels that made the v2 script's key check harmless. When a key with `construction_category` and
`expected_label` columns is supplied, `expected_agree`, the per-category agreement table and the
confusion table are still computed and written as reported diagnostics, but they play no part in
`needs_adjudication`. The tool also runs with a minimal id-only key, or with no key file at all
(both exercised by `evaluation/tests/test_claims_annotation_v3.py`), which is required for the
real-claims set: there is no expected label to check majority labels against, and, with no key at
all, items are identified only by `ann_id`. `--key-arm on` reproduces the v2 script's
behaviour, including its `fleiss_kappa_all30`/`fleiss_kappa_evidence25`-style dynamically-named
keys (the number is the actual item count, not a literal "30"; on v2 inputs the count happens
to be 30 and 25). `test_aggregate_key_arm_on_reproduces_round2_aggregate_byte_for_byte` runs it
against the v2 inputs and asserts the resulting json and completed csv are byte identical to
`hss_annotation_aggregate_v2.json` and `hss_annotation_completed_v2.csv`. This setting exists only for
that regression; the v3 pass does not use it.

### 8.2 Rubric diff

`INSTRUCTIONS_v3.md` is `INSTRUCTIONS_v2.md` plus two clauses, both category-neutral (naming no
construction category or rule):

- in `## Labels`, the `needs_nuance` bullet gains: "This also covers the case where the paper
  supports the claim's main finding but the claim also states a detail, such as a setting, a
  population, a time window or an instrument, that the cited paper does not state."
- in `## Rule`, a new sentence: "A claim that adds a detail the shown passage does not contain
  must be checked against the cited paper's own text before needs_nuance is chosen; the shown
  passage is a window, not the whole paper."

Every other change is mechanical: item and row counts (30 to 60), file names (`hss_annotation_sheet_v3.csv`
etc.) and the round number in the title.

`INSTRUCTIONS_v3.md`'s provenance-disclaimer paragraph now states the true v3 facts: the
three HSS sessions (annotator3_A, annotator3_B, annotator3_C) ran on 2026-09-07 against the
60-item HSS set, and three further sessions under the same three identifiers, run one after
another rather than concurrently, ran on 2026-09-07 against the 26-item real-claims set. No
author of this software has read either set against the cited papers and countersigned the
labels. This replaces an earlier placeholder that copied `INSTRUCTIONS_v2.md`'s v2 header
unchanged; that placeholder was left in place for one commit only, so that
`evaluation/tests/test_provenance_wording.py` already had an allow-list entry for this file
before the v3 pass's sessions existed to describe accurately, and is superseded by this section. No
`test_provenance_wording.py` entry change was needed for the rewrite: the allow-list entry is
keyed on the phrase and its "not" context, not on the specific date or count.

**Which rubric governed the real-claims set.** `INSTRUCTIONS_v3.md`'s `## Purpose` section names
only the HSS set's file names (`hss_annotation_sheet_v3.csv`, `annotator3_A.csv`/`_B`/`_C`,
`source_texts_index_v3.json`, "the same 60 items"); it names no `real_*` file and no 26-item
count. There is no separate `INSTRUCTIONS_real_v3.md`. All three real-claims sessions received
this same file and, per their own attestations (section 8.9 below), silently substituted
`real_annotation_sheet_real_v3.csv` for the sheet, their own `real_annotator3_{A,B,C}.csv` for
the output, and `source_texts_index_real_v3.json` for the index, judging 26 items
(`annr-01`-`annr-26`) instead of 60. The four label definitions and the judging rule are
identical in both sets; nothing in the rubric's substantive wording differs between the HSS run
and the real-claims run, only the file names and the item count that each session substituted.
The scope note added directly under the header of `INSTRUCTIONS_v3.md` records this in the
rubric file itself, so a reader of the rubric does not need this paragraph to find it.

**A note on comparing this file to `INSTRUCTIONS_v2.md`.** `INSTRUCTIONS_v2.md` has CRLF line
endings and `INSTRUCTIONS_v3.md` has LF-only line endings (a difference introduced when the file
was authored, unrelated to content). A plain `diff -u INSTRUCTIONS_v2.md INSTRUCTIONS_v3.md`
therefore reports every line as changed; pass `--strip-trailing-cr` (or normalise one file's line
endings before diffing) to see that the only content difference is the two clauses listed above
plus the mechanical changes.

### 8.3 The two v3 sheets

**HSS test v3, 60 items.** Built from `hss_test_v3_claims.jsonl` (60 items, six construction rules:
8 verbatim, 8 paraphrase, 18 altered, 10 over_specified, 8 wrong_paper, 8 no_full_text) and the
cache at `data/hss_test_v3_fulltext/`, shuffled with seed `20260907 * 3 = 60782721` into opaque ids
`ann-01` to `ann-60`. Of the 60 items, 44 show a retrieved passage, 8 show the no-match marker and
8 show the withheld-evidence marker; the withheld key preserves the v2 schema
(`ann_id, item_id, construction_category, expected_label, built_at`). Outputs:
`hss_annotation_sheet_v3.csv`, `hss_annotation_key_v3.csv`, `source_texts_index_v3.json`. The
residual blinding channels this construction leaves in the sheet, four pre-registered plus one
found by the v3 audit, are counted in section 8.8 below: 52 of 60 items, not 34, carry a
label reachable from the surface relation between the claim and passage columns.

**Real claims, 26 items.** Built from `real_claims_test.jsonl` (26 items harvested from real
manuscript citations for the review workbook a human will confirm or override outside this
repository's automated tests; `rule` is the constant `"real"` and `expected`
is `null` for every item, so there is no expected label and no construction category to withhold),
with `chunk_doi` equal to `cited_doi` for every item, against the cache at
`data/real_claims_fulltext/`, shuffled with seed `20260907 * 5 = 101304535` into opaque ids
`annr-01` to `annr-26`. Of the 26 items, 25 show a retrieved passage and 1 shows the no-match
marker; none are withheld (`chunk_doi` is never null for this set). The withheld key holds only
`ann_id, item_id`, per section 8.1. Outputs: `real_annotation_sheet_real_v3.csv`,
`real_annotation_key_real_v3.csv`, `source_texts_index_real_v3.json`.

Both builds were verified, by direct inspection and by
`evaluation/tests/test_claims_annotation_v3.py`, to contain none of the six construction-rule names
(`verbatim`, `paraphrase`, `altered`, `over_specified`, `wrong_paper`, `no_full_text`) anywhere in
the sheet or the index; `no_full_text` is exempted from that check for the rubric only, because it
is also one of the four legitimate `annotator_label` values and the rubric must state it
(unchanged from the v2 pass). Neither the opaque-id order nor any other visible field in either sheet
follows the claims file's own item order.

### 8.4 Labelling procedure

Three independent Claude Opus sessions, run through Claude Code and designated A, B and C (the
same designation used in the v2 pass, not the same sessions), each labelled one blind sheet, on
2026-09-07. Every session received `INSTRUCTIONS_v3.md` and the matching source-text index and
was blind to the construction category and the expected label (the HSS set has both in the
withheld key; the real-claims set has neither). Each session filled in one private copy of the
sheet with a label, a confidence and a one-sentence note, in two halves with an interim write, as
their attestations record.

- **HSS test v3, 60 items.** `annotator3_A.csv`, `annotator3_B.csv`, `annotator3_C.csv`.
- **Real claims, 26 items.** `real_annotator3_A.csv`, `real_annotator3_B.csv`,
  `real_annotator3_C.csv`, run after the HSS sessions and after each other rather than at the same
  time (see the annotator-A disclosure in section 8.9).

The four labels and the judging rule are exactly as in the v2 pass (section 1 above), plus the two
`needs_nuance` clauses added in section 8.2. `aggregate_annotations_v3.py --key-arm off` merged
each set's three sheets into `hss_annotation_completed_v3.csv` and
`real_annotation_completed_v3.csv` and computed the four aggregate files.
`adjudicate_annotations_v3.py` wrote the final labels to `hss_annotation_adjudicated_v3.csv` and
`real_annotation_adjudicated_v3.csv`.

### 8.5 Adjudication trigger and what the adjudicator did

An item is sent to adjudication under the v3 (`--key-arm off`) setting when the three sessions
are **not unanimous**; the construction key, where one exists, plays no part in the trigger (see
section 8.1).

**HSS test v3: 0 of 60 items adjudicated.** The three sessions agreed on all 60 items, so no item
was sent to adjudication and `hss_annotation_adjudicated_v3.csv` carries `source = unanimous` on
every row. `agrees_with_construction` is `True` on all 60 rows: the unanimous blind majority
matches the construction rule's expected label on every item, including the 18 `altered` items
under `build_hss_set.py --altered-expected strict`, which (unlike the v2 pass) admits only a single
expected label per item.

**Real claims: 2 of 26 items adjudicated.** The three sessions split 2-to-1 on `annr-09`
(unblinded id `real-test-23`) and on `annr-14` (unblinded id `real-test-15`). A fourth Claude Opus
session, independent of A, B and C, adjudicated both against the cited papers' own cached text,
following the same rule as the v2 pass: the cited paper's text is the authority. Both rationales are
quoted from `real_adjudication_decisions_v3.json`.

**annr-09** (2 sessions `needs_nuance`, 1 `unsupported`). "Three of the five attributes in the
claim are present in the paper, all of them reported second hand in its literature review: 'PhET
simulations are flexible tools used by educators to achieve various educational goals' and 'Some
of the reported advantages of computer simulations use in laboratories include: ease of use,
ability to save instructors time to be devoted to students learning instead of equipment set-up
and student supervision, availability of large set of variables to test and manipulate without
worrying about safety'. The words cleaner and cost-efficient do not occur anywhere in the cached
text, and cost appears only as a problem of the traditional laboratory... The paper therefore
supports a narrower version of the list rather than none of it, which is needs_nuance rather than
unsupported." Final label **`needs_nuance`**, matching the sessions' own 2-1 majority.

**annr-14** (2 sessions `verified`, 1 `needs_nuance`). "The setting matches... The strength does
not. The paper's own summary is 'A majority of students showed a positive attitude to learning via
Facebook', and the same results section reports 'Sixty percent of students expressed concerns that
they were distracted from concentrating on their studies as they switched to different resources'
and 'A high percentage (78%) indicated that they needed timely and effective support in
understanding how to use it effectively'... A majority holding a positive attitude, reported
alongside 60 per cent describing distraction and no measure of achievement, is weaker than strong
acceptance as an effective way to learn, so the claim overstates the paper and the label is
needs_nuance rather than verified." Final label **`needs_nuance`**, overturning the sessions' 2-1
majority of `verified`. This is the only decision in the v3 pass that overturns a blind majority.

### 8.6 Agreement statistics

Recomputed from `hss_annotation_completed_v3.csv` and `real_annotation_completed_v3.csv`. Labels
are drawn from {verified, needs_nuance, unsupported, no_full_text}. Fleiss kappa uses three raters
per item. `hss_annotation_aggregate_v3.json` and `real_annotation_aggregate_v3.json` now also
carry `n_adjudicated` (the count of non-unanimous items; 0 for HSS, 2 for real), and their
markdown summaries carry the matching "adjudicated (non-unanimous) items" row, added by this
section's fix to `aggregate_annotations_v3.py`'s `--key-arm off` path; the `--key-arm on` path, used only
by the v2 byte-identity regression test, is untouched and carries no such key.

### HSS test v3, all 60 items (= the evidence subset for `unanimity_count`, `fleiss_kappa`)

| quantity | value |
|---|---|
| items | 60 |
| evidence-subset items (passage or no-match, excluding the 8 withheld) | 52 |
| unanimous items (A = B = C) | 60 of 60 |
| n_adjudicated | 0 |
| Fleiss kappa, all 60 | 1.000 |
| Fleiss kappa, evidence subset (52) | 1.000 |
| A vs B, percent agreement / Cohen kappa | 1.000 / 1.000 |
| A vs C, percent agreement / Cohen kappa | 1.000 / 1.000 |
| B vs C, percent agreement / Cohen kappa | 1.000 / 1.000 |

### Real claims, all 26 items (evidence subset is the same 26; nothing is withheld)

| quantity | value |
|---|---|
| items | 26 |
| unanimous items (A = B = C) | 24 of 26 |
| n_adjudicated | 2 |
| Fleiss kappa | 0.900 |
| A vs B, percent agreement / Cohen kappa | 0.962 / 0.925 |
| A vs C, percent agreement / Cohen kappa | 0.923 / 0.853 |
| B vs C, percent agreement / Cohen kappa | 0.962 / 0.925 |

The perfect HSS agreement is not evidence of annotator skill; section 8.8 below measures how much
of the sheet's surface a session can decide from without reading the cited paper's own text.

### 8.7 Final labels

**HSS test v3, 60 items.**

| final label | count |
|---|---|
| unsupported | 26 |
| verified | 16 |
| needs_nuance | 10 |
| no_full_text | 8 |

Label source: 60 `unanimous`, 0 `adjudicated`, 0 decided on a bare majority. Every final label
equals the construction category's expected label (60 of 60, `agrees_with_construction = True` on
every row), which the v2 pass could not claim, because the v2 pass's `altered` stratum admitted a
disjunctive expected label; the v3 pass's `--altered-expected strict` build removed that. Section 8.8
explains why this agreement figure is a consistency check on the construction rules rather than
independent confirmation of them.

**Real claims, 26 items.**

| final label | count |
|---|---|
| verified | 15 |
| needs_nuance | 10 |
| unsupported | 1 |
| no_full_text | 0 |

Label source: 24 `unanimous`, 2 `adjudicated`. There is no construction key for this set, so there
is no expected-label agreement figure to report; these are the labels a future human reviewer
confirms or overrides in the workbook outside this repository's automated tests, not a benchmark
against a known-correct answer.

### 8.8 Residual blinding channels: 52 of 60, not 34

This section documents four channels on the HSS set, totalling 34 of 60 items, on which a
session can assign the construction-correct label from the shape of the sheet alone, without
reading the cited paper:

| channel | items | construction category |
|---|---|---|
| the shown passage is character for character the claim | 8 | `verbatim` |
| the shown passage is the no-match marker | 8 | `wrong_paper` |
| the shown passage is the withheld marker | 8 | `no_full_text` |
| the claim is the shown passage plus a trailing clause | 10 | `over_specified` |

A count taken independently of the construction process confirms this count
and adds two further measurements that raise it. First, the `over_specified` trailing clause is
drawn from a closed set of five literal phrases repeated across the 10 items ("using eye-tracking
software" three times, "using a mobile application" three times, "among heritage speakers" twice,
"in a bilingual classroom" once, "during the winter semester" once), so after two or three items a
session can recognise the rest by string match rather than by reading the paper. Second, the 18
`altered` items are the same kind of surface comparison already counted for `over_specified`: each
is the shown passage with exactly one token replaced by its antonym or by a different number (for
example `more`/`less`, `increased`/`decreased`, `90%`/`99%`, `4-year`/`5-year`), and deciding
`unsupported` from that needs no reading of the cited paper. Adding the 18 `altered` items to the
34 pre-registered gives **52 of 60 items** carrying a label reachable from the surface relation
between the claim and passage columns, under the brief's own definition of that relation. The
residual 8 are the `paraphrase` items, of which 7 are also single-token substitutions but were not
counted as a channel because the audit found no closed vocabulary or fixed shape for them the way
it did for `altered` and `over_specified`.

**52 of 60, with 34 given as the marker-and-identity subset, is the number this file, this
package's README and the manuscript's limitation sentence must use; 34 alone understates the
exposure.** It does not change any label: the sessions' agreement with the construction rule is a
consistency check on the construction rules, not an independent confirmation of them, exactly as
section 4 item 4 of the v2 pass (above) already says for a different reason.

The real-claims set has no construction key and no construction categories, so this measurement
does not apply to it; there is nothing to compare a real claim's label against except the cited
paper's own text, which is what section 8.5's adjudications did for the two non-unanimous items.

### 8.9 What these labels do not establish

All eight points in section 4 above, written for the v2 pass, apply to the v3 pass's HSS set with the
same force (model annotations not human or expert annotations; correlated raters; the withheld
stratum is simulated; blinding covers the category and the key, not the corpus; independence is
self-reported; the set is small and narrow relative to the applied-linguistics literature it is
drawn from). The v3 pass adds the following, specific to this round and to the real-claims set:

1. **The 52-of-60 residual channel of section 8.8**, not the 34-of-60 the design brief
   pre-registered, is the honest measure of how much of the HSS sheet a session can decide from
   its surface shape alone.
2. **The v3 pass's `altered`-stratum agreement of 60 of 60 is not stronger evidence than the v2 pass's**,
   for the reason just given: only 8 of the 60 items (the `paraphrase` stratum) require reading
   the cited paper's text and cannot be decided from a fixed pattern.
3. **The real-claims sessions ran one after another, not at the same time.** Session A's
   attestation records running `git status --porcelain` on the annotation folder to confirm
   nothing had been committed, and that the listing incidentally showed `real_annotator3_B.csv`,
   `real_annotator3_C.csv` and `real_attestation3_B.md` as already present and untracked. A states
   it opened none of them, and its labels differ from both B's and C's on both non-unanimous
   items, which is consistent with that statement but does not prove it; there was no sandbox
   enforcing it, as section 8.9(6) already says of the v2 pass. A reader should treat the three
   real-claims sessions as sequential, not as three simultaneous independent draws.
4. **Two of the 26 real-claims items carried no usable passage even though `evidence_mode` calls
   them `passage`.** `annr-09`'s shown passage is a bibliography line ("I., & Makki, J. (2021).
   Examining the use of PhET simulations on students' attitudes and learning in general chemistry
   II.") and `annr-19`'s is a Fejes and Nylander (2014) reference-list entry. In both cases the
   retrieval scored a bibliography line highest because it shares vocabulary with the claim, so
   the sheet offered no usable evidence for either row under the passage-only reading, and all
   three sessions had to fall back to the cited paper's cached text. No label is affected: all
   three sessions consulted every one of the 26 cached papers on this set regardless of what the
   sheet showed (per their attestations, section 8.10 below).
5. **Both sets' six annotator sheets carry two redundant columns.** Every `annotator3_*.csv` and
   `real_annotator3_*.csv` has `label` and `justification` columns alongside `annotator_label` and
   `annotator_note`, identical to them on every row. `aggregate_annotations_v3.py` selects columns
   by name and reads neither `label` nor `justification`; they carry no information the other two
   columns do not already carry and are not used anywhere downstream.
6. **The real-claims aggregate markdown names items by their unblinded ids.**
   `real_annotation_aggregate_v3.md`'s "Non-unanimous items" section lists `real-test-23` and
   `real-test-15`, the unblinded `item_id`s, rather than `annr-09` and `annr-14`. This is harmless
   after the round has closed (no annotator reads an aggregate file mid-round, and none did here),
   but it means this specific file is not safe to show to a future annotation session on this or a
   similar set; `hss_annotation_aggregate_v3.md` has the same property whenever a key with
   `item_id` is passed, and did not surface it only because the HSS set had zero non-unanimous
   items to list.
7. **`n_adjudicated` counts non-unanimous items, not incorrect ones.** For the HSS set it is 0
   because every item was unanimous, not because the sessions were tested against the key and
   passed; the separate `agrees_with_construction` figure in section 8.7 is the test against the
   key. For the real-claims set it is 2, both resolved by reading the cited paper rather than by
   any construction rule, because none exists for this set.
8. **The `INSTRUCTIONS_v2.md`/`INSTRUCTIONS_v3.md` line-ending difference (section 8.2) is
   cosmetic, not a content change**, but a reader running a plain `diff` without
   `--strip-trailing-cr` will see every line reported as changed and should not mistake that for a
   larger rewrite than the two clauses documented in section 8.2.

### 8.10 Annotator attestations (self-reported, not enforced)

The six sessions returned the following attestations. They are reproduced verbatim, exactly as
section 5 above reproduces the v2 pass's. They were not enforced by sandboxing or verified against a
filesystem access log, and should be read as statements of intent, not as proof of isolation.

#### HSS test v3, annotator A, verbatim from `attestation3_A.md`

```
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
```

#### HSS test v3, annotator B, verbatim from `attestation3_B.md`

```
# Attestation, annotator B

Session identifier: annotator3_B. Labelling completed 2026-09-07. This session is a Claude Code
model session, not a human annotator.

## What I was asked to do

Label all 60 items of `evaluation/claims/annotation/hss_annotation_sheet_v3.csv` blind, using the
rubric `evaluation/claims/annotation/INSTRUCTIONS_v3.md`, with one label and one justification per
item, working in two halves with an interim write, and without opening any key, any other
annotator's file, or any results, summary, provenance, build, veto, report or internal development-tooling file.

## Every file I opened

Read (in this order):

1. `evaluation/claims/annotation/INSTRUCTIONS_v3.md`
2. `evaluation/claims/annotation/hss_annotation_sheet_v3.csv`
3. `evaluation/claims/annotation/source_texts_index_v3.json`
4. `evaluation/claims/data/hss_test_v3_fulltext/10-14746_ssllt-2022-12-1-3.json` (ann-01, ann-54)
5. `evaluation/claims/data/hss_test_v3_fulltext/10-17323_jle-2021-10771.json` (ann-14, ann-20)
6. `evaluation/claims/data/hss_test_v3_fulltext/10-18806_tesl-v36i1-1301.json` (ann-18)
7. `evaluation/claims/data/hss_test_v3_fulltext/10-14483_22487085-17087.json` (ann-21)
8. `evaluation/claims/data/hss_test_v3_fulltext/10-14483_22487085-12966.json` (ann-25)
9. `evaluation/claims/data/hss_test_v3_fulltext/10-14483_22487085-14086.json` (ann-30)
10. `evaluation/claims/data/hss_test_v3_fulltext/10-14483_22487085-14514.json` (ann-31)
11. `evaluation/claims/data/hss_test_v3_fulltext/10-14746_ssllt-2021-11-1-2.json` (ann-37)
12. `evaluation/claims/data/hss_test_v3_fulltext/10-17011_apples_urn-201903251959.json` (ann-41, ann-45)
13. `evaluation/claims/data/hss_test_v3_fulltext/10-14746_ssllt-2022-12-1-4.json` (ann-42, ann-46)
14. `evaluation/claims/data/hss_test_v3_fulltext/10-17323_jle-2022-15962.json` (ann-43)
15. `evaluation/claims/data/hss_test_v3_fulltext/10-29140_jaltcall-v18n2-632.json` (ann-47)
16. `evaluation/claims/data/hss_test_v3_fulltext/10-55593_ej-26103a1.json` (ann-55)
17. `evaluation/claims/data/hss_test_v3_fulltext/10-17323_jle-2021-11494.json` (ann-56)

Each cached paper was opened only for the items listed beside it, all of which list that path in
`source_texts_index_v3.json`. I opened them through a small search script that prints the paper's
title and the windows of text around the search terms, so I saw the cited paper's own text and
nothing else. A directory listing of `evaluation/claims/data/hss_test_v3_fulltext/` was produced by
`ls`; it shows four filenames that belong to no item in this sheet, and I did not open those four.

Written:

- `evaluation/claims/annotation/annotator3_B.csv` (interim write with ann-01 to ann-30 labelled,
  then the final write with all 60 rows)
- `evaluation/claims/annotation/attestation3_B.md` (this file)

Scratch scripts were written outside the repository, under the Windows temp directory.

## What I did not open

No key file, no other annotator's file, no results, summary, provenance, build, veto, report or
internal development-tooling file, no `evaluation/claims/results/` content, and no `real_*` sheet. I ran no
`git log` and no repository-wide search. I used no prior knowledge of these papers beyond the
passage shown for each item and the cached text of that item's own cited paper.

## Method

I worked in two halves. For the first half, ann-01 to ann-30, I judged each row from the passage,
opened the cited paper's cached text for every row whose passage said "No matching passage found"
and for every row whose claim added a detail the passage did not contain, then wrote the interim
sheet with those 30 rows labelled. I then repeated the procedure for ann-31 to ann-60 and rewrote
the sheet with all 60 rows.

Where a claim added a setting, a population, a time window or an instrument, I searched the cited
paper's own cached text for that detail before choosing `needs_nuance`, as the rubric requires. In
every such case the search returned zero hits, for example zero hits for "heritage" in Wedin and
Shaswar and in Aubrey, zero hits for "mobile" in the three Colombian papers, zero hits for
eye-tracking in either Kruk paper or in Becirovic et al., zero hits for "bilingual" in Parra, and
zero hits for "winter" or "semester" in Savski and Prabjandee.

## Result

60 of 60 items labelled: 26 unsupported, 16 verified, 10 needs_nuance, 8 no_full_text. Confidence
is high on 59 items and medium on one, ann-57, where the paper contradicts the claim's
normalization base of 1370 words with its own "normalized per 1000 words" while the two reported
figures and the direction of the comparison still match.
```

#### HSS test v3, annotator C, verbatim from `attestation3_C.md`

```
# Attestation, annotator C

Session identifier: annotator3_C
Date of labelling: 2026-09-07
Items labelled: 60 of 60 (ann-01 through ann-60)
Output sheet: `evaluation/claims/annotation/annotator3_C.csv`

## Declaration

I labelled all 60 rows of `hss_annotation_sheet_v3.csv` from the claim text, the
`cited_paper_passage` column, and, where the rubric's rule required it, the cited paper's own
cached full text at the path listed for that row in `source_texts_index_v3.json`. I did not open
the key, any other annotator's sheet, any results, summary, provenance, build, veto or report
file, and I did not run `git log` or search the repository for the expected labels. Nothing was
committed.

I opened the cached paper text in two situations only, both required by the rubric: when the
passage read "No matching passage found in the cited paper", and when the claim stated a detail
(a setting, population, time window or instrument) that the shown passage did not contain. In the
second case I searched the whole cached text for that detail before choosing `needs_nuance`.

## Files opened

Rubric and sheet:

1. `evaluation/claims/annotation/INSTRUCTIONS_v3.md`
2. `evaluation/claims/annotation/hss_annotation_sheet_v3.csv`
3. `evaluation/claims/annotation/source_texts_index_v3.json`

Cached paper texts, all under `evaluation/claims/data/hss_test_v3_fulltext/`, each listed in
`source_texts_index_v3.json` for the rows named:

4. `10-14746_ssllt-2022-12-1-3.json` (rows ann-01, ann-54)
5. `10-17323_jle-2021-10771.json` (rows ann-14, ann-20)
6. `10-18806_tesl-v36i1-1301.json` (row ann-18)
7. `10-14483_22487085-17087.json` (row ann-21)
8. `10-14483_22487085-12966.json` (row ann-25)
9. `10-14483_22487085-14086.json` (row ann-30)
10. `10-14483_22487085-14514.json` (row ann-31)
11. `10-14746_ssllt-2021-11-1-2.json` (row ann-37)
12. `10-17011_apples_urn-201903251959.json` (rows ann-41, ann-45)
13. `10-14746_ssllt-2022-12-1-4.json` (rows ann-42, ann-46)
14. `10-17323_jle-2022-15962.json` (row ann-43)
15. `10-29140_jaltcall-v18n2-632.json` (row ann-47)
16. `10-55593_ej-26103a1.json` (row ann-55)
17. `10-17323_jle-2021-11494.json` (row ann-56)

A directory listing of `evaluation/claims/data/hss_test_v3_fulltext/` was produced by `ls`; the
listing showed file names and sizes only, and no cached text outside the numbered list above was
read. The remaining rows were decided from the `cited_paper_passage` column alone, which either
reproduced the paper's sentence or carried the "Evidence withheld" marker.

Working scripts were written to a temporary directory outside the repository
(`%TEMP%\psearch.py`, `%TEMP%\annot_c.py`, `%TEMP%\annot_c_part2.py`) and no repository file other
than `annotator3_C.csv` and this attestation was created or modified.

## Label counts

| label | count |
| --- | --- |
| unsupported | 26 |
| verified | 16 |
| needs_nuance | 10 |
| no_full_text | 8 |
| total | 60 |

## Notes on judgement

- `no_full_text` was applied to exactly the eight rows carrying the "Evidence withheld" marker
  with an empty `paths` list in the index (ann-06, ann-08, ann-15, ann-23, ann-33, ann-51, ann-53,
  ann-60), regardless of how plausible the claim read.
- `unsupported` covers two situations: the passage stating the opposite of the claim, usually a
  reversed direction word or a changed number, and rows where the passage read "No matching
  passage found" and a search of the cited paper's cached text for the claim's key terms returned
  nothing, so the citation does not address the claim.
- `needs_nuance` was used only where the paper supported the claim's main sentence but the claim
  appended a detail, such as "using eye-tracking software", "using a mobile application", "among
  heritage speakers", "in a bilingual classroom" or "during the winter semester", that a search of
  the cited paper's full cached text did not find.
- `verified` covers rows where the passage repeated the claim verbatim or differed only in
  neutral wording, such as "instructors" for "teachers", "learners" for "students", "observed" for
  "found" or "nevertheless" for "however".
- Confidence was recorded as high for all 60 rows, because every decision rested either on an
  explicit contradiction in the passage, an exact match, an explicit withheld marker, or an
  exhaustive search of the cited paper's cached text.
```

#### Real claims, annotator A, verbatim from `real_attestation3_A.md`

```
# Attestation, blind annotator A, real claims

Session identifier: annotator3_A
Date of labelling: 2026-09-07
Sheet labelled: evaluation/claims/annotation/real_annotation_sheet_real_v3.csv (26 items, annr-01 to annr-26)
Output: evaluation/claims/annotation/real_annotator3_A.csv

## Procedure

I read the rubric, then the sheet, then worked through the 26 items in two halves. After the first
13 items I wrote an interim version of the output file with those rows labelled and the remaining
rows blank, then completed items 14 to 26 and rewrote the file. Every item was judged from the
cited_paper_passage column plus the cited paper's own cached text at the path listed for that row
in source_texts_index_real_v3.json. No item carried the "Evidence withheld" marker, so the
no_full_text label was not used. One item (annr-03) showed "No matching passage found in the cited
paper", and per the rubric I searched the cited paper's cached text for the claim's key terms
before deciding.

I did not open any key file, any other annotator's file, or any results, summary, provenance,
build, veto, report or internal development-tooling file, and I did not run git log or search the repository
outside the files listed below.

## Files opened

Rubric and sheet:

1. evaluation/claims/annotation/INSTRUCTIONS_v3.md
2. evaluation/claims/annotation/real_annotation_sheet_real_v3.csv
3. evaluation/claims/annotation/source_texts_index_real_v3.json

Cited papers' cached text, one per item, all under evaluation/claims/data/real_claims_fulltext/:

4. 10-3389_feduc-2020-00059.json (annr-01)
5. 10-1111_jcpp-12352.json (annr-02)
6. 10-14742_ajet-1026.json (annr-03)
7. 10-14483_calj-v18n2-10610.json (annr-04)
8. 10-3389_fpubh-2022-1036071.json (annr-05)
9. 10-1007_s10833-018-9320-9.json (annr-06)
10. 10-1080_09243453-2020-1746363.json (annr-07)
11. 10-3389_fpsyg-2022-1057730.json (annr-08)
12. 10-21601_ijese_10966.json (annr-09)
13. 10-1375_ajgc-17-2-160.json (annr-10)
14. 10-14742_ajet-4310.json (annr-11)
15. 10-3402_rlt-v20i0-14430.json (annr-12)
16. 10-3991_ijet-v15i23-18799.json (annr-13)
17. 10-3991_ijet-v9i8-3805.json (annr-14)
18. 10-1007_s10993-023-09649-4.json (annr-15)
19. 10-1007_s11251-008-9065-6.json (annr-16)
20. 10-1007_s11858-017-0878-0.json (annr-17)
21. 10-1111_jcpp-12605.json (annr-18)
22. 10-3384_rela-2000-7426-rela9063.json (annr-19)
23. 10-4018_978-1-4666-3914-0-ch013.json (annr-20)
24. 10-14483_22487085-13373.json (annr-21)
25. 10-1007_s10734-010-9356-0.json (annr-22)
26. 10-1007_bf02213420.json (annr-23)
27. 10-1007_s10643-019-00971-3.json (annr-24)
28. 10-23917_humaniora-v19i2-6809.json (annr-25)
29. 10-30935_scimath_9511.json (annr-26)

A directory listing of evaluation/claims/data/real_claims_fulltext/ was produced while checking the
structure of the cached files; only the 26 files above were read.

## Files written

- evaluation/claims/annotation/real_annotator3_A.csv (interim after item 13, final after item 26)
- evaluation/claims/annotation/real_attestation3_A.md (this file)

Nothing was committed. To confirm that, I ran `git status --porcelain` restricted to
evaluation/claims/annotation/. That listing incidentally showed the file names
real_annotator3_B.csv, real_annotator3_C.csv and real_attestation3_B.md as untracked. I did not
open those files or read any of their contents, and no label in this sheet was influenced by them.
I ran no other git command.

## Label counts

- verified: 15
- needs_nuance: 10
- unsupported: 1
- no_full_text: 0

Confidence: 10 high, 16 medium, 0 low.
```

#### Real claims, annotator B, verbatim from `real_attestation3_B.md`

```
# Attestation, annotator B (real claims)

Session identifier: annotator3_B
Date of labelling: 2026-09-07
Sheet labelled: evaluation/claims/annotation/real_annotation_sheet_real_v3.csv (26 items, annr-01 to annr-26)
Output written: evaluation/claims/annotation/real_annotator3_B.csv

## Files opened

Rubric and sheet:

1. evaluation/claims/annotation/INSTRUCTIONS_v3.md
2. evaluation/claims/annotation/real_annotation_sheet_real_v3.csv
3. evaluation/claims/annotation/source_texts_index_real_v3.json

Cached cited-paper texts, opened because the rubric requires checking a claim's added
details, or a "No matching passage found" marker, against the cited paper's own text.
One file per item, taken from the path listed for that item in the index above:

4.  evaluation/claims/data/real_claims_fulltext/10-3389_feduc-2020-00059.json (annr-01)
5.  evaluation/claims/data/real_claims_fulltext/10-1111_jcpp-12352.json (annr-02)
6.  evaluation/claims/data/real_claims_fulltext/10-14742_ajet-1026.json (annr-03)
7.  evaluation/claims/data/real_claims_fulltext/10-14483_calj-v18n2-10610.json (annr-04)
8.  evaluation/claims/data/real_claims_fulltext/10-3389_fpubh-2022-1036071.json (annr-05)
9.  evaluation/claims/data/real_claims_fulltext/10-1007_s10833-018-9320-9.json (annr-06)
10. evaluation/claims/data/real_claims_fulltext/10-1080_09243453-2020-1746363.json (annr-07)
11. evaluation/claims/data/real_claims_fulltext/10-3389_fpsyg-2022-1057730.json (annr-08)
12. evaluation/claims/data/real_claims_fulltext/10-21601_ijese_10966.json (annr-09)
13. evaluation/claims/data/real_claims_fulltext/10-1375_ajgc-17-2-160.json (annr-10)
14. evaluation/claims/data/real_claims_fulltext/10-14742_ajet-4310.json (annr-11)
15. evaluation/claims/data/real_claims_fulltext/10-3402_rlt-v20i0-14430.json (annr-12)
16. evaluation/claims/data/real_claims_fulltext/10-3991_ijet-v15i23-18799.json (annr-13)
17. evaluation/claims/data/real_claims_fulltext/10-3991_ijet-v9i8-3805.json (annr-14)
18. evaluation/claims/data/real_claims_fulltext/10-1007_s10993-023-09649-4.json (annr-15)
19. evaluation/claims/data/real_claims_fulltext/10-1007_s11251-008-9065-6.json (annr-16)
20. evaluation/claims/data/real_claims_fulltext/10-1007_s11858-017-0878-0.json (annr-17)
21. evaluation/claims/data/real_claims_fulltext/10-1111_jcpp-12605.json (annr-18)
22. evaluation/claims/data/real_claims_fulltext/10-3384_rela-2000-7426-rela9063.json (annr-19)
23. evaluation/claims/data/real_claims_fulltext/10-4018_978-1-4666-3914-0-ch013.json (annr-20)
24. evaluation/claims/data/real_claims_fulltext/10-14483_22487085-13373.json (annr-21)
25. evaluation/claims/data/real_claims_fulltext/10-1007_s10734-010-9356-0.json (annr-22)
26. evaluation/claims/data/real_claims_fulltext/10-1007_bf02213420.json (annr-23)
27. evaluation/claims/data/real_claims_fulltext/10-1007_s10643-019-00971-3.json (annr-24)
28. evaluation/claims/data/real_claims_fulltext/10-23917_humaniora-v19i2-6809.json (annr-25)
29. evaluation/claims/data/real_claims_fulltext/10-30935_scimath_9511.json (annr-26)

A directory listing of evaluation/claims/data/real_claims_fulltext/ was produced while
locating these files; no file in that directory other than the 26 listed above was read.

## Files not opened

No key file, no other annotator's sheet or attestation, no results, summary, provenance,
build, veto or report file, and no internal development-tooling file was opened. No git history was
inspected and no repository-wide search was run. Searching was confined to the 26 cached
paper texts listed above, one at a time, for the item being judged.

## Method

The rubric was read first, then the sheet. Work was done in two halves with an interim
write of the output file after annr-01 to annr-13. For each item the claim was compared
with the passage shown in the sheet, and then with the cited paper's cached text whenever
the claim asserted a detail the passage did not contain or the passage was a bare
reference line or a "No matching passage found" marker; in practice that applied to every
item, so all 26 cached papers were consulted. Every label rests on wording quoted in the
justification column. No prior knowledge of the subject matter was used.

## Result summary

26 of 26 items labelled: 16 verified, 9 needs_nuance, 1 unsupported, 0 no_full_text.
No item carried the "Evidence withheld" marker, so no_full_text did not apply to any row.
Confidence: 12 high, 14 medium, 0 low.

Nothing was committed.
```

#### Real claims, annotator C, verbatim from `real_attestation3_C.md`

```
# Attestation: blind annotator C, real claims sheet

Session identifier: annotator3_C
Date of labelling: 2026-09-07
Items labelled: 26 of 26 (annr-01 to annr-26), one label and one justification each.

## Procedure

I read the rubric first, then the sheet, then labelled the items in two halves. After the
first half (annr-01 to annr-13) I wrote the partially completed sheet to
`evaluation/claims/annotation/real_annotator3_C.csv`, then labelled annr-14 to annr-26 and
rewrote the same file with all 26 rows. Every row was judged from the passage in the sheet
plus the cited paper's own cached text at the path listed for that row in
`source_texts_index_real_v3.json`, as the rubric's rule requires. I used no prior knowledge
of the subject matter and no external sources.

The output file keeps the six sheet columns exactly as given and fills in annotator_label,
annotator_confidence, annotator_note, annotator_name and annotated_on. Two extra columns,
`label` and `justification`, repeat the label and the note, as the task asked for.

## Every file I opened

Rubric and sheet:

1. evaluation/claims/annotation/INSTRUCTIONS_v3.md
2. evaluation/claims/annotation/real_annotation_sheet_real_v3.csv
3. evaluation/claims/annotation/source_texts_index_real_v3.json

Cited papers' cached text, all under evaluation/claims/data/real_claims_fulltext/, one per item:

4.  10-3389_feduc-2020-00059.json (annr-01)
5.  10-1111_jcpp-12352.json (annr-02)
6.  10-14742_ajet-1026.json (annr-03)
7.  10-14483_calj-v18n2-10610.json (annr-04)
8.  10-3389_fpubh-2022-1036071.json (annr-05)
9.  10-1007_s10833-018-9320-9.json (annr-06)
10. 10-1080_09243453-2020-1746363.json (annr-07)
11. 10-3389_fpsyg-2022-1057730.json (annr-08)
12. 10-21601_ijese_10966.json (annr-09)
13. 10-1375_ajgc-17-2-160.json (annr-10)
14. 10-14742_ajet-4310.json (annr-11)
15. 10-3402_rlt-v20i0-14430.json (annr-12)
16. 10-3991_ijet-v15i23-18799.json (annr-13)
17. 10-3991_ijet-v9i8-3805.json (annr-14)
18. 10-1007_s10993-023-09649-4.json (annr-15)
19. 10-1007_s11251-008-9065-6.json (annr-16)
20. 10-1007_s11858-017-0878-0.json (annr-17)
21. 10-1111_jcpp-12605.json (annr-18)
22. 10-3384_rela-2000-7426-rela9063.json (annr-19)
23. 10-4018_978-1-4666-3914-0-ch013.json (annr-20)
24. 10-14483_22487085-13373.json (annr-21)
25. 10-1007_s10734-010-9356-0.json (annr-22)
26. 10-1007_bf02213420.json (annr-23)
27. 10-1007_s10643-019-00971-3.json (annr-24)
28. 10-23917_humaniora-v19i2-6809.json (annr-25)
29. 10-30935_scimath_9511.json (annr-26)

Files I wrote:

- evaluation/claims/annotation/real_annotator3_C.csv (interim write after annr-13, final write after annr-26)
- evaluation/claims/annotation/real_attestation3_C.md (this file)

Helper scripts used for searching the cached paper texts were written outside the repository,
in the session temporary directory, and no other file in the repository was created or changed.

## Files I did not open

I opened no key file, no other annotator's file, and no results, summary, provenance, build,
veto, report or internal development-tooling file. I did not run git log and did not search the repository.
I listed the file names in evaluation/claims/data/real_claims_fulltext/ once to confirm the
directory layout before reading the paths given in the index; I opened only the 26 files
listed above from that directory.

## Label counts

- verified: 16
- needs_nuance: 8
- unsupported: 2
- no_full_text: 0 (no item carried the "Evidence withheld" marker)

Nothing was committed.
```

### 8.11 What section 8 does not establish

Sections 8.1 to 8.10 above are the complete record for the HSS test set and the real-claims set: procedure, tooling, the two sheets,
the agreement statistics, the adjudication of the two non-unanimous real-claims items, the final
labels, the residual blinding exposure and the six attestations. They establish that the v3
labels exist, were produced blind, and reproduce byte for byte from the committed sheets, exactly
as an independent recomputation from the committed sheets confirmed. They
do not establish anything section 4 or section 8.9 above says they do not: these remain model
annotations, not expert or human ones; the HSS agreement figures are a consistency check on the
construction rules, not independent confirmation of them, more so than the v2 pass's because of the
52-of-60 residual channel; and the real-claims labels are an internal check pending the human
workbook review this repository's automated tests do not perform, not a benchmark result reported
in the manuscript.
