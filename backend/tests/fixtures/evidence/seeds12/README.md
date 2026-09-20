# Offline evidence-acceptance fixture: the twelve demo seed papers

What `backend/tests/test_evidence_fixture_targets.py` measures the evidence step against,
so the acceptance rules in `app.services.deep_analysis` can be checked on real extracted
text, real model output and real PDF-extraction artefacts without a network call.

One JSON file per paper, plus `manifest.json`. Each paper file holds:

- `paper_id`, `title`, `year`: the paper as stored in the demo project.
- `chunks`: `{"section", "text"}` entries, in the same order and with the same section
  labels as the paper's stored `fulltext_chunks`, reduced to excerpts (below).
- `proposed_items`: every evidence item the deep-analysis agent proposed for this paper,
  exactly as it came back from the model, including the items code rejects.
- `accepted_at_capture`, `rejected_at_capture`: what
  `app.services.deep_analysis.validate_evidence_items` decided for those items against
  the paper's **full** stored chunks on the day the fixture was captured.

## Where this came from

The twelve papers are the demo protocol's seed set (`demo/protocol.json`), as acquired by
the 2026-09-10 demo run in the `scholarrag-rev` stack: full text from Unpaywall's
open-access locations, extracted and chunked by `app.services.fulltext`, and read out of
that project's own `papers.metadata.fulltext_chunks`. The proposed items are one live
deep-analysis call per paper, run offline from the API by
`backend/scripts/evidence_diagnosis.py`, with the prompt whose version `manifest.json`
records.

## How the chunks were reduced

Storing twelve articles in full would put about 831,000 characters of published text in
the repository. Each chunk here keeps only the windows around the places one of this
paper's proposed quotes is found: roughly 400 characters either side, widened to whole
lines so the boundary checks read the same physical lines they read in the full chunk, and
the windows joined by the marker `[TEXT OMITTED FROM FIXTURE].` That comes to about
129,000 characters, non-contiguous, and it is derived test data rather than a copy of the
articles.

The reduction is checked, not assumed: `evidence_diagnosis.py build-fixture` refuses to
write anything unless `validate_evidence_items` returns the identical accepted rows,
quote for quote and chunk index for chunk index, on the reduced chunks and on the full
ones. A quote that is present nowhere in the paper contributes no window, so it is still
rejected here for the same reason it was rejected there.

## Re-capturing it

The fixture pins model output from one dated run, so it does not move when the prompt or
the model does. Re-capturing needs the demo stack up on that project's own volume and one
paid analysis call per paper; `manifest.json`'s `analysis_prompt_version` says which
prompt the stored items answered.
