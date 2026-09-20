# Changelog

All notable changes to ScholarRAG are documented in this file. The format follows
Keep a Changelog.

## [1.1.0] - 2026-09-19

This release answers the SoftwareX submission SOFTX-D-26-00953 reviews and hardens the
writing and retrieval pipeline built to answer them.

### Fixed

- The all-Docker path boots from a clean clone. The backend service runs
  `alembic upgrade head` before uvicorn and the frontend builds a dedicated `dev`
  stage, so the frontend no longer crash-loops on a missing `next` binary and
  registration works on first start.
- The test suite can no longer drop the application database. It reads
  `TEST_DATABASE_URL` only and refuses to run when that resolves to `DATABASE_URL`
  or names a database that does not end in `_test`. `INSTALL.md` documents the
  warning and the recovery recipe.
- A screening batch that fails is recorded as `unscreened` instead of being
  included by default.
- A search round in which every query fails ends with `stop_reason: retrieval_failed`
  and the provider's reason, instead of ending with an empty result that is
  indistinguishable from a search that found nothing. Contact addresses are
  redacted from stored error messages.
- Token usage is read correctly from both pydantic-ai 1.x and 2.x run results.

- `overall_status` for a paper record is derived from the record checks alone.
  Venue indexing, recency, and citation count became indicators and cannot turn a
  record green.
- A citation is now made only for a proposition the cited paper's own supplied
  material actually states; a statement the library does not support is written
  without a citation and flagged `[NEEDS CITATION]` instead of being attached to
  whichever paper's citation was due next.
- Finalizing and healing a section's citations is more consistent across a revision: a
  citation naming several papers in one mention heals as one link and keeps its
  verdict-to-link pairing across a sentence rewrite, repeated healing is idempotent, and
  matching a `[NEEDS CITATION]` marker or an uncited finding sentence no longer depends
  on incidental whitespace.
- A body sentence a citation-link call's own results leave unclassified as neither a
  citation nor an uncited entry is no longer invisible to both verification and
  finalize; it is retried once, together with any other such sentence from the same
  call, and one still unresolved after that retry is kept in the delivered text as an
  unclassified finding rather than silently dropped or left unchecked.
- An emptied section (finalize can legitimately remove every sentence) is treated as a
  real completion instead of being left stale in the editor; regenerating one section of
  a draft replaces only that section's own first heading, not every heading sharing its
  type; a paper counts as grounded only once its full-text analysis has actually been
  saved, so a failed analysis is counted as failed rather than silently as done; and both
  of a revision's citation-link calls keep their own provenance instead of the second
  overwriting the first.

### Added

- Optional venue filter. `wos_filter` accepts `auto`, `on`, and `off`; Smart Search
  runs without any Web of Science journal list; `on` with an empty list returns
  HTTP 409.
- Screening record export,
  `GET /api/v1/tasks/{task_id}/screening-record?format=json|csv`, with criteria,
  reconciling flow counts, per-record exclusion reasons, unscreened records, and
  retrieval failures.
- Deterministic routing between two screening model stages. Code demotes to needs review
  an exclusion carrying no verbatim anchor, a record with no abstract, a non-article type
  and a table-of-contents abstract, and queues a record whose criterion only full text can
  settle. Every provisional inclusion is then re-examined once per job, in reasoning mode,
  for population, outcome and study type, and is either confirmed or returned to needs
  review naming what is unestablished. The judge's own decision, tokens and latency are
  exported per record in the screening record, beside the first pass's own status,
  criterion and quotation.
- Claim verification results can now be exported as an auditable claim-verification
  record (JSON or CSV), listing every checked claim with its matched citation, source
  paper, full-text acquisition route, evidence quote/location, and the model's verdict,
  mirroring the existing screening-record export. New endpoint:
  `GET /api/v1/tasks/{task_id}/claim-record?format=json|csv` (owner-only; 404 unless
  the task is a completed `claim_verify` job); the claim-verification report UI gained
  a "Download claim record (CSV)" button next to its model/provenance line.
- Per-call provenance for every evaluated model call: configured model, model
  reported by the provider, provider response id, system fingerprint when
  returned, temperature, prompt version hash, token counts, and timestamp.
  Surfaced in the screening record, the claim-verification report, and the
  writing job result.
- Code-level citation audit of generated text against the project library, with
  counts of matched citations, unmatched citations, and `[NEEDS CITATION]` flags.
  The flag itself remains a prompt instruction; the counting is code.
- `full_text_coverage` on the claim-verification report.
- Reviewer-executable demonstration in `demo/`: a fixed protocol, twelve
  open-access seed DOIs, one command, expected outputs, exit codes, tolerances,
  and troubleshooting. Runs with the venue filter off, so no licensed data is
  needed.
- Evaluation harness in `evaluation/`: screening against three SYNERGY datasets
  with baselines and run-to-run agreement, claim verification against SciFact and
  a constructed humanities and social science set, with cost, latency, and prompt
  versions recorded.
- Expandable abstracts in the search results and the library.
- Data-processing notice on the registration page in both locales, linking to
  `PRIVACY.md`.
- Root documentation set: `README.md`, `INSTALL.md`, `PRIVACY.md`,
  `CHANGELOG.md`, `CITATION.cff`, `LICENSE.txt`, and a dependency table stating
  key, licence, cost, and fallback for every external service.
- `backend/requirements.lock.txt`, the full transitive dependency set frozen in a
  clean `python:3.12-slim` container.
- Smart Search stop rule: a run no longer ends for lack of new inclusions before a
  minimum number of rounds have completed (`SMART_SEARCH_MIN_ROUNDS`, default 3), and
  afterward only once a run of consecutive rounds have each added nothing
  (`SMART_SEARCH_DRY_ROUND_PATIENCE`, default 2); every other stop reason is unaffected.
  A per-round log of the queries issued, their result counts and their screening
  outcomes is recorded in the job's provenance and shown in the Smart Search UI, and a
  replay mode (`queries_override`, `publication_date_max` on the search request) lets a
  run be repeated against materially the same corpus.
- Evidence-grounded drafting: full-text analysis proposes an evidence table of
  verbatim, paper-specific quotes, and code, not the model, decides which are accepted;
  the writer is given that accepted evidence, not a summary of it, as its context for a
  section.
- A target-length option for AI Write (`target_words`): a body over a hard maximum 25%
  above the target is regenerated once. The write job now gates on verification before
  saving: every cited sentence is checked against the paper it cites, a section with an
  unverified claim is revised once with its already-verified sentences pinned unchanged,
  and the section is finalized before saving -- removing any sentence whose citation is
  not verified, any uncited "finding" sentence, and an uncited framing sentence left
  dangling by one of those removals, and stripping any leftover `[NEEDS CITATION]`
  marker. A section that finalize reduces to fewer than three cited sentences is
  written once more from its own original prompt, and the attempt that finalizes with
  more cited sentences is the one saved. The standalone verify-and-heal action now runs
  the same verify-revise-finalize step over a whole draft.
- Citation links are narrowed to the sentence's own proposition rather than the whole
  cited paper, and the exported claim record carries `claim_sentence`, the full sentence
  a narrowed claim was cut from.
- Full-text acquisition tries every Unpaywall open-access location for a paper, not
  only the first, retries a PDF download that comes back blocked or as the wrong content
  type, and reports `publisher_interstitial` when every attempt looked like a publisher
  block rather than folding that case into a bare `abstract_only`.
- The AI Write dialog asks for a target word count; the claim-verification report shows
  the final, all-verified view of a draft instead of a raw-pass-versus-healed comparison;
  and the draft editor refetches the section the write job already finalized and saved,
  rather than rebuilding and re-saving it client-side.

### Changed

- Screening and claim verification run at temperature 0.
- Screening returns one INCLUDE, EXCLUDE or NEEDS_REVIEW decision per paper, each with
  its criterion, a verbatim quotation from the record and a short stated reason.
- Claim verification decision rules. The verifier decomposes each drafted statement into
  atomic assertions and gives each one a verdict before deciding the claim's overall status;
  any one contradicted assertion makes the whole claim `unsupported` rather than
  `needs_nuance`, so a claim the source states the opposite of is no longer confused with one
  the source is merely silent about. Seven code-level guards (a no-full-text-with-chunks
  check, citation attribution, per-assertion status consistency, quote fidelity, scale
  fidelity and two numeric checks) run after the model call and can only make a status
  stricter, never weaker; a separate, older numeric check now runs as a report-only
  diagnostic and records a reason without ever changing a status. Each guard that fires
  appends a machine-readable reason, shown next to the model's own answer in both the
  claim-verification report and the exported
  claim record. Prompt versions are tracked by a SHA-256 hash of the prompt text, recorded
  with every call, so a report states exactly which decision rules produced it; consecutive
  prompt versions are evaluated and reported side by side in `evaluation/README.md`.
- Verification badges use record-level wording: `record verified`,
  `metadata mismatch`, `DOI not found`, `no DOI`. Indicators render as neutral
  chips.
- Dependencies are pinned to exact versions.
- The development compose stack no longer declares RabbitMQ or MinIO. Host
  ports are configurable through `POSTGRES_PORT`, `BACKEND_PORT`, and
  `FRONTEND_PORT`, and browser-facing origins through
  `PUBLIC_FRONTEND_ORIGIN`, `PUBLIC_BACKEND_ORIGIN`, and `PUBLIC_API_URL`.
- `.env.example` matches `backend/app/config.py` again. Dead variables were
  removed.
- Package version strings are aligned to `1.1.0` across `backend/app/main.py`,
  `backend/pyproject.toml`, and `frontend/package.json`. The FastAPI
  application title is now `ScholarRAG API`, matching the product name used
  everywhere else in the documentation.

### Removed

- The Clarivate Web of Science journal lists are no longer tracked in the
  repository and are not part of the release archive. Users supply them under
  their own licence.

### Known limitations

- Scholarly appropriateness of a citation is not assessed.
- External manuscripts cannot be imported and audited; verification applies to
  text in the project's own drafts.
- The citation graph is a visualization built from the project library after
  retrieval and screening; network structure does not influence retrieval,
  screening, or drafting.
- Background jobs run in the API process and uploads are on local disk, so a
  restart interrupts jobs and a container rebuild without a volume loses
  uploads.
- The provider does not offer dated model snapshots, so runs are reproducible in
  configuration but not pinned to an immutable model.
- Hosted-instance usage is not instrumented, and no usage figure is reported.
- A citation naming several papers in one rendered mention (e.g. `(Jones, 2019; Lee,
  2021)`) is kept as written, unsplit, when only some of its keys verify; the mention's
  own text is not cut into a verified part and a dropped part, on either the write job's
  own gate or the standalone verify-and-heal action.

Note on the requested comparison table against functionally similar open-source tools:
this is a manuscript-only point and is not addressed by a code or documentation
change in this release.

## [1.0.0] - 2026-07-29

Initial SoftwareX submission, tag `v1.0.0`.
