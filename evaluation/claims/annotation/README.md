# HSS claim-set annotation package

This folder is an independent check on the by-construction labels of the HSS claim-verification
sets: each item was annotated blind, three times, and adjudicated on disagreement, so that a
reader has something other than the construction rule itself to compare the verifier against.
**These are model annotations, not human or expert
annotations**; see `PROVENANCE.md` for exactly what they do and do not establish.

## The 30-item HSS claim set

Built from scratch with opaque identifiers, no category-to-label mapping in the rubric, and
passages retrieved independently of the construction record. Three separate Claude Opus sessions
(A, B, C), run through Claude Code on 2026-09-05, each labelled all 30 items blind; a fourth
Claude Opus session adjudicated the two items on which the three sessions disagreed. Full
method, blinding checks and recomputed agreement statistics are in
[`PROVENANCE.md`](PROVENANCE.md); read it before citing any number from this folder.

| file | role |
|---|---|
| `build_sheet_v2.py` | builds the blind sheet, the key, the index and the rubric |
| `hss_annotation_key_v2.csv` | withheld construction key: `ann_id`, `item_id`, `construction_category`, `expected_label`, the labels the blind annotators did not see |
| `hss_annotation_sheet_v2.csv` | the blind sheet handed to the three sessions |
| `INSTRUCTIONS_v2.md` | rubric given to the sessions, with no category hints. Its four-line provenance header was added on 2026-09-05 after the sessions returned their sheets; the rest of the file is byte-identical to what the sessions received (see `PROVENANCE.md` section 7) |
| `source_texts_index_v2.json` | per item, the cited DOI and cached-text paths (empty for withheld items) |
| `annotator2_A.csv`, `annotator2_B.csv`, `annotator2_C.csv` | the three completed sheets |
| `attestation2_A.md`, `attestation2_B.md`, `attestation2_C.md` | the three self-reported blinding attestations |
| `aggregate_annotations_v2.py` | merges the three sheets and computes agreement statistics |
| `hss_annotation_completed_v2.csv` | the three sheets merged, one row per item |
| `hss_annotation_aggregate_v2.json`, `hss_annotation_aggregate_v2.md` | the agreement statistics |
| `adjudicate_annotations_v2.py` | applies the adjudication trigger and writes the final labels |
| `hss_annotation_adjudicated_v2.csv` | the 30 final labels (`item_id`, `final_label`, `expected_label`, ...) used for scoring |
| `PROVENANCE.md` | method, blinding checks, agreement statistics, limitations |

## The 60-item HSS test set and the 26-item real-claims set

Two sets, both labelled and adjudicated: the 60-item HSS test set (six construction rules: 8
verbatim, 8 paraphrase, 18 altered, 10 over_specified, 8 wrong_paper, 8 no_full_text) and the
26-item real-claims set (real citations, no construction key). Three separate Claude Opus
sessions (A, B, C), run through Claude Code on 2026-09-07, each labelled all 60 HSS items blind,
then three further sessions with the same identifiers, run one after another rather than at the
same time, each labelled all 26 real-claims items blind against the same rubric with the file
names substituted. The HSS sessions were unanimous on all 60 items, so no HSS item went to
adjudication. The real-claims sessions split on 2 of 26 items; a fourth Claude Opus session
adjudicated both against the cited papers' own cached text. Full method, the rubric diff, the
agreement statistics, the residual blinding-channel count, the six attestations and what the
labels do and do not establish are in [`PROVENANCE.md`](PROVENANCE.md) section 8.

| file | role |
|---|---|
| `build_sheet_v3.py` | argparse generalisation of `build_sheet_v2.py`; builds the sheet, key and index only, never the rubric |
| `INSTRUCTIONS_v3.md` | `INSTRUCTIONS_v2.md` plus two category-neutral clauses on `needs_nuance`; governs both sets (see `PROVENANCE.md` section 8.2) |
| `aggregate_annotations_v3.py` | argparse generalisation of `aggregate_annotations_v2.py`; the default `--key-arm off` setting drops the construction-key trigger, works with a minimal id-only key or with no key at all; that path also carries `n_adjudicated` |
| `hss_annotation_sheet_v3.csv`, `hss_annotation_key_v3.csv`, `source_texts_index_v3.json` | the 60-item HSS blind sheet, withheld construction key and index, ids `ann-01`-`ann-60`; the key holds the labels the blind annotators did not see |
| `real_annotation_sheet_real_v3.csv`, `real_annotation_key_real_v3.csv`, `source_texts_index_real_v3.json` | the 26-item real-claims blind sheet, key and index, ids `annr-01`-`annr-26`; the key holds only the id mapping because no expected label exists for this set |
| `annotator3_A.csv`, `annotator3_B.csv`, `annotator3_C.csv` | the three completed HSS sheets |
| `attestation3_A.md`, `attestation3_B.md`, `attestation3_C.md` | the three HSS blinding attestations |
| `real_annotator3_A.csv`, `real_annotator3_B.csv`, `real_annotator3_C.csv` | the three completed real-claims sheets |
| `real_attestation3_A.md`, `real_attestation3_B.md`, `real_attestation3_C.md` | the three real-claims blinding attestations |
| `hss_annotation_completed_v3.csv`, `hss_annotation_aggregate_v3.json`, `hss_annotation_aggregate_v3.md` | HSS merge and agreement statistics |
| `real_annotation_completed_v3.csv`, `real_annotation_aggregate_v3.json`, `real_annotation_aggregate_v3.md` | real-claims merge and agreement statistics |
| `adjudicate_annotations_v3.py` | applies the adjudication trigger and writes the final labels for either set |
| `hss_annotation_adjudicated_v3.csv` | the 60 final HSS labels used for scoring; `source` is `unanimous` on all 60 rows |
| `real_adjudication_decisions_v3.json` | the adjudicator's rationale for the 2 real-claims items sent to adjudication |
| `real_annotation_adjudicated_v3.csv` | the 26 final real-claims labels used for scoring; `source` is `adjudicated` on 2 rows and `unanimous` on 24 |

All six attestation files (`attestation3_A.md`, `attestation3_B.md`, `attestation3_C.md`,
`real_attestation3_A.md`, `real_attestation3_B.md`, `real_attestation3_C.md`) had their H1
heading rewritten to drop the pass number. Five of the six also had one sentence naming the
internal tooling directory reworded to say "internal development-tooling file" instead of
the directory's own name; `attestation3_C.md` had only the heading changed, since it names
no such directory. Every other byte of these six files is unchanged.

## Scoring the verifier against these labels

`claims/summarize.py --labels <csv>` reads a committed `hss_run<A|B>.jsonl` and scores its
`predicted_status` against a `final_label` csv, joined on `item_id`; `--real-labels <csv>` does
the same for `real_run<A|B>.jsonl` against the real-claims set. Neither calls the verifier, an
LLM or the network. From the repository root:

```
evaluation/.venv/Scripts/python evaluation/claims/summarize.py --results-dir evaluation/claims/results/v6 \
    --labels evaluation/claims/annotation/hss_annotation_adjudicated_v3.csv \
    --real-labels evaluation/claims/annotation/real_annotation_adjudicated_v3.csv
```

(POSIX: `evaluation/.venv/bin/python`.) This adds `hss_vs_annotation` and `real_vs_annotation`
blocks to `summary.json` (accuracy, per-label precision/recall/F1, a confusion table and
Cohen's kappa per run), without altering any other block; the committed `results/v6/summary.md`
keeps these blocks in `summary.json` for other consumers but no longer renders them as
markdown tables (see `results/v6/summary.md`'s own header note), so re-running this command
against `results/v6/` would reintroduce those tables in `summary.md`, which would then need
the same trim before being committed again.
