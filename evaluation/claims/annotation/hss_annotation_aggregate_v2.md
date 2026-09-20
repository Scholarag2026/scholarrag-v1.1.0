# HSS annotation aggregate (round 2)

Aggregation of the round 2 blind triple annotation (annotators A, B, C) for the 30-item HSS claim-verification set, produced by aggregate_annotations_v2.py.

## Summary

| metric | value |
|---|---|
| items | 30 |
| evidence-subset items (passage or no_match, excl. withheld) | 25 |
| unanimous items (A == B == C) | 28 |
| Fleiss' kappa (all 30 items) | 0.930 |
| Fleiss' kappa (evidence subset, 25 items) | 0.896 |
| majority vs expected agreement (overall) | 1.000 |

## Pairwise agreement, all items

| pair | percent agreement | Cohen's kappa |
|---|---|---|
| AB | 1.000 | 1.000 |
| AC | 0.933 | 0.897 |
| BC | 0.933 | 0.897 |

## Pairwise agreement, evidence subset

| pair | percent agreement | Cohen's kappa |
|---|---|---|
| AB | 1.000 | 1.000 |
| AC | 0.920 | 0.848 |
| BC | 0.920 | 0.848 |

## Majority vs expected agreement by construction category

| construction_category | n | agreement rate |
|---|---|---|
| altered | 10 | 1.000 |
| no_full_text | 5 | 1.000 |
| paraphrase | 5 | 1.000 |
| verbatim | 5 | 1.000 |
| wrong_paper | 5 | 1.000 |

## Confusion table: majority label vs expected_label

| majority \ expected | no_full_text | unsupported | unsupported\|needs_nuance | verified |
|---|---|---|---|---|
| no_full_text | 5 | 0 | 0 | 0 |
| unsupported | 0 | 5 | 10 | 0 |
| verified | 0 | 0 | 0 | 10 |

## Items needing adjudication (not unanimous OR majority outside expected set)

- hss-altered-01
- hss-altered-02
