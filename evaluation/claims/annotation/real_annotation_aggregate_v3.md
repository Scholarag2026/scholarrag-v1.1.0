# Real-claims annotation aggregate (v3, key-arm off)

Aggregation of the round-3 blind triple annotation (annotators A, B, C) for the 26-item set, produced by aggregate_annotations_v3.py.

## Summary

| metric | value |
|---|---|
| items | 26 |
| evidence-subset items (passage or no_match, excl. withheld) | 26 |
| unanimous items (A == B == C) | 24 |
| adjudicated (non-unanimous) items | 2 |
| Fleiss' kappa (all 26 items) | 0.900 |
| Fleiss' kappa (evidence subset, 26 items) | 0.900 |

## Pairwise agreement, all items

| pair | percent agreement | Cohen's kappa |
|---|---|---|
| AB | 0.962 | 0.925 |
| AC | 0.923 | 0.853 |
| BC | 0.962 | 0.925 |

## Pairwise agreement, evidence subset

| pair | percent agreement | Cohen's kappa |
|---|---|---|
| AB | 0.962 | 0.925 |
| AC | 0.923 | 0.853 |
| BC | 0.962 | 0.925 |

## Non-unanimous items (adjudication trigger)

- real-test-23
- real-test-15
