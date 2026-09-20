# ScholarRAG

ScholarRAG is an open-source retrieval-augmented research platform
that retrieves literature from OpenAlex, screens papers in two stages with an
exportable screening record, and acquires full text where it is available. A drafting
stage writes academic text section by section, with citations drawn from the
project's paper library. Generated text is checked at two levels: record-level checks
against Crossref, and claim-level checks of drafted statements against the cited full
text.

## What "verified" means here

ScholarRAG uses "verified" at three distinct levels. This document, the interface, and
job records all use the same words.

**Level 1: record verification.** A paper's bibliographic record is checked against
Crossref. The result is one of: `record verified`, `metadata mismatch`,
`DOI not found`, `no DOI`.

**Level 2: claim verification.** A statement drafted in ScholarRAG's own editor is
checked against the full text of the paper it cites. The check reaches one of four
decision outcomes: `verified`, `needs nuance`, `unsupported`, `no full text`. A fifth
value, `error`, records that the verification call itself failed, for example on a
provider quota error, and is not a decision about the claim.

**Level 3: scholarly appropriateness.** Whether a citation is the right one to make,
in the context of an argument, is not checked by the software. That judgement is left
to the author and to peer review.

Claim verification applies to text written in ScholarRAG's own drafting editor and
checked against the project's paper library. ScholarRAG does not import and audit an
external manuscript.

## Requirements at a glance

| Service | Purpose | Key or licence | Cost | Behaviour when absent |
|---|---|---|---|---|
| DeepSeek (`deepseek-flash`, the provider's current name for the deployment previously aliased `deepseek-chat`) | Query generation, relevance screening, claim verification, drafting, chat, deep paper analysis, and dataset coding and analysis | `DEEPSEEK_API_KEY`, account at platform.deepseek.com | Paid, metered per token. List price on 2026-09-16: input 0.15 / 0.30 USD per 1M tokens (off-peak / peak), output 0.60 / 1.20 USD per 1M tokens (`api-docs.deepseek.com/quick_start/pricing`). The evaluation cost figures in `evaluation/README.md` and `evaluation/common.py` were computed with the list price read on 2026-09-02 (input 0.22 / 0.44, output 0.66 / 1.32 USD per 1M tokens, cache miss), which was the provider's price at the time those runs were made | Mandatory. Drafting fails with an error. Screening calls fail and the affected papers are recorded as `unscreened`. Claim-verification calls fail and are recorded with status `error`. Retrieval, the library, the citation graph and record verification still work |
| OpenAlex | The only literature search source; also citation-graph relations and metadata | No key required. `OPENALEX_EMAIL` joins the polite pool; `OPENALEX_API_KEY` draws on an account budget | Free | Without a key the anonymous per-IP daily budget applies and can return HTTP 429. A search round where every query fails ends with `stop_reason: retrieval_failed` |
| Crossref | DOI existence and metadata match for record verification | No key. Contact address in the User-Agent | Free | The paper is reported without a verified record |
| Unpaywall | Open-access PDF discovery | No key. `UNPAYWALL_EMAIL` required by the API | Free | The lookup is skipped when the variable is empty; acquisition falls back to other sources |
| Web of Science Core Collection journal lists (SCIE, SSCI, AHCI, ESCI) | Optional Stage 1 venue filter and the indexing badges | Clarivate licence held by the user. Not distributed with ScholarRAG | Subscription held by the user's institution | The table stays empty, badges are inactive, and Smart Search runs on the whole OpenAlex result set |
| PostgreSQL 17 | All application data | Open source (PostgreSQL licence) | Free | Mandatory; the backend cannot start |
| Docker and Docker Compose >= 2.24 | The documented one-command path | Open source | Free | Optional; `INSTALL.md` documents a manual path |

The relevance screener's second pass, which re-checks a paper already marked INCLUDE, also
runs on `deepseek-flash` but with reasoning left on at effort high (rather than disabled, as
every other call above uses) and its answer schema sent in the prompt rather than as a
forced tool choice, since the served model rejects that combination while reasoning is on.
Every LLM call's own provenance record carries the model DeepSeek actually served
(`model_reported`) and its `system_fingerprint`, regardless of which configured name
requested it, so a served-model or deployment change on the provider's side is always
visible after the fact rather than assumed from the configured name alone.

## Quick start

```bash
cp .env.example .env
# edit .env and set DEEPSEEK_API_KEY
docker compose up -d
```

Open http://localhost:3000 for the app and http://localhost:8000/docs for the API
docs. Docker Compose must be >= 2.24, because both compose files use the
`env_file: {path, required}` mapping form. See `INSTALL.md` for the manual path,
port overrides, and the test suite.

## Configuration

`backend/app/config.py` is the complete list of settable variables: any field of the
`Settings` class there can be set as an environment variable of the same name.
`.env.example` carries the variables a deployment normally sets; copy it to `.env` and
fill in the values described there and in "Requirements at a glance" above. Every name
`.env.example` lists maps one to one to a field of `backend/app/config.py`, apart from
`POSTGRES_USER`, `POSTGRES_PASSWORD` and `POSTGRES_DB`, which Docker Compose reads
directly and which are not `Settings` fields. Two settings are easy to miss:

- **An OpenAlex account key is recommended for any real search, not only a heavy one.**
  Smart Search's stop rule (below) never ends a search before at least
  `SMART_SEARCH_MIN_ROUNDS` rounds (default 3) have run, so even a narrow topic makes
  several rounds of OpenAlex calls. `OPENALEX_API_KEY` (free at
  openalex.org/settings/api) draws on the account's own daily budget instead of the
  anonymous per-IP one, which a shared network can exhaust well before a search is done;
  without it, a search can end with `stop_reason: retrieval_failed`.
- **Writer-only overrides.** `WRITING_MODEL`, `WRITING_THINKING` and `WRITING_MAX_TOKENS`
  affect only the AI Write call (`call_deepseek` in `app/services/writing.py`); every
  other agent (query generation, screening, analysis, citation linking, claim
  verification) always uses `DEEPSEEK_MODEL`, unaffected by any of the three. Left unset,
  `WRITING_MODEL` also uses `DEEPSEEK_MODEL`, so the writer behaves like every other agent
  by default; set, it substitutes a different model for that one call. `WRITING_THINKING`
  is `"enabled"` or `"disabled"`; left unset, the request carries no `thinking` field at
  all and the provider's own default applies. `WRITING_MAX_TOKENS` caps that one call's
  `max_tokens` (default 4096).

## Smart Search

Smart Search runs a query-generation agent against OpenAlex over successive rounds,
screens every identified record in up to two stages (an optional Web of Science venue
filter, then LLM relevance screening against the stated inclusion and exclusion
criteria), and adds every included record to the project's paper library.

**Stop rule.** A round that adds no newly included paper does not end the search by
itself before `smart_search_min_rounds` rounds (default 3) have run. Once that many
rounds have happened, the search stops after `smart_search_dry_round_patience` (default
2) consecutive rounds have each added zero newly included papers, reported as
`stop_reason: "no_new_included"`. Every other stop reason (`time_limit`, `max_scanned`,
`cancelled`, `retrieval_failed`, `query_generation_failed`, `screening_failed`,
`no_new_papers`, `no_wos_papers`) can still end a search at any round and is unaffected
by either setting. The two thresholds are tuned with the environment variables
`SMART_SEARCH_MIN_ROUNDS` (default 3) and `SMART_SEARCH_DRY_ROUND_PATIENCE` (default 2),
set directly rather than through `.env.example`.

**Per-round log.** The job result's `provenance.rounds` carries one entry per round whose
queries actually ran: the queries themselves, how many results each returned and how
many were newly unique, how many records the round screened, how many it newly included,
how many it flagged needs-review, and whether the round replayed a fixed query list
instead of generating one. `provenance.settings` records the `min_rounds` and
`dry_round_patience` thresholds the run applied, next to the round log itself.

**Replay mode.** `POST /projects/{id}/smart-search` accepts two optional fields for
reproducing an earlier run: `queries_override`, one list of literal search queries per
round, used instead of the query-generation agent's own output for as many rounds as are
given (a round beyond the given lists falls back to normal generation); and
`publication_date_max`, a date forwarded as a retrieval cutoff to every OpenAlex search
call the job makes, so a later run can see materially the same corpus as of that date
rather than however the live index has grown since.

## Running without Web of Science

`wos_filter` is `auto` by default: Stage 1 applies only if a journal list has already
been imported. `wos_filter=on` requires an imported list and otherwise returns
HTTP 409. `wos_filter=off` never applies Stage 1. With no list imported, the WoS
badges are inactive, `stage1_applied` is false in the job result, and Smart Search
runs on the whole OpenAlex result set.

## Screening record

`GET /api/v1/tasks/{task_id}/screening-record?format=json|csv` exports the screening
decisions for a Smart Search job. The JSON response has the keys `criteria`, `flow`,
`provenance`, `retrieval_failures`, `records`. The CSV has the header
`outcome,stage,reason,title,doi,year,journal,journal_issn,openalex_id`. The counts in
`flow` can populate a PRISMA flow diagram; PRISMA itself is a reporting guideline, not
a screening method.

## Full-text acquisition

`POST /projects/{id}/acquire-full-texts` fetches open-access full text for every paper
in the library that does not already have it. Candidate PDF locations are tried in
order until one downloads and its content passes an identity check against the paper:
every Unpaywall open-access location, looked up only when the paper has a DOI (ordered
published version, then accepted version, then submitted version, then anything else),
then the paper's own stored full-text URL, then an open-access URL already present in
the paper's OpenAlex metadata, whether or not the paper has a DOI.

A response that looks like a publisher interstitial (an HTML page, or a body that does
not start with the PDF file signature) or an HTTP 403, 429 or server error is retried
twice, after 3 and 10 seconds, using a browser-like `User-Agent` on the two retries; an
HTTP 404 or 410 is treated as a genuinely missing document and is not retried. Every one
of these outcomes is recorded as `abstract_only`, with a `fulltext_reason` that narrows
why: a paper for which every attempted candidate looked like an interstitial is recorded
`abstract_only` with reason `publisher_interstitial`; a PDF that downloads but turns out
to describe a different paper is recorded `abstract_only` with reason `wrong_work`. A
paper still without full text once every candidate is exhausted, for any other reason,
is `abstract_only` with no more specific reason, and can be completed by pasting or
uploading the text through the UI.

## Drafting and verification

`POST /drafts/{id}/generate` writes one section at a time. For a section that draws on
the paper library, every selected paper with full text but no analysis yet is analyzed
first; the analysis agent proposes evidence items (verbatim quotes from that paper), and
code, not the model, decides which are accepted, so the writer is given accepted
evidence rather than a summary of it.

An optional `target_words` field asks for a body of about that many words (headings
excluded) plus a hard maximum 25% over it; a target of 400, for example, allows up to
500. A body still over the hard maximum after generation is regenerated once, naming the
previous word count so the model can see what to cut; this happens at most once, never in
an unbounded loop. What actually guarantees the saved body is at or under the hard
maximum is a further, unconditional step: right before saving, trailing content is
trimmed from the final text, after finalization and any revision, until the word count is
at or under the hard maximum, regardless of whether a regeneration happened.

A second, structured call over the generated text then builds a citation-link map: each
in-text citation is matched to the specific sentence that makes it and narrowed to that
sentence's own proposition, rather than to the whole paper it cites. Every body sentence
is expected to come back classified as either a citation link or an uncited entry
(framing or an uncited finding); a sentence the call's own results leave unclassified is
sent back once more, as its own small batch of gap sentences, to the same linker; a
sentence still unresolved after that one retry is kept in the delivered text as an
unclassified finding rather than being silently dropped or left unchecked. Every cited
sentence is checked with the claim-verification agent (frozen prompt and guards, see
"Claim-verification record" below) against the full text of the paper it cites. When a
guard finds a quoted span that is not verbatim in the source text and nothing else about
the claim is wrong, one bounded repair turn continues that same model conversation,
naming the exact non-verbatim segments, before the guards re-check the repaired answer;
every other guard outcome goes straight to the gate below without a repair turn. This is
the gate: when any claim's status is neither `verified` nor the uncheckable
`no_full_text`, the whole section is regenerated exactly once, with every sentence that
already verified pinned to survive the rewrite unchanged and the sentences that failed
named with the reason they failed; only a sentence whose text actually changed afterward
is sent back for a second verification call. This one revision happens at most once per
generation.

The section is then finalized before it is saved: a cited sentence whose status is still
not verified is removed, along with an uncited sentence that reads as a stated finding
rather than as scene-setting; a citation that could only be checked against
`no_full_text` is dropped, and its sentence with it if no verified citation remains on
that sentence; an uncited framing sentence left dangling by one of those removals (for
example, one announcing a count of points that no longer follow it, or opening with a
discourse connective whose antecedent was just removed) is also removed, so the
remaining framing stays coherent; and any leftover `[NEEDS CITATION]` marker is
stripped. The same step drops a paragraph that only echoes the section's own title or a
heading, and an uncited paragraph that reads as a heading rather than a sentence. A
marker-stripped sentence left as a fragment is dropped too, and a trailing clause that no
verified proposition covers is cut away with the rest of the sentence kept.
Only this finalized, fully-verified text is written to the draft, and the job
result records what finalize removed and why, alongside one claim report over the
section's own claims, which is all-verified by construction of the step just described.

One backstop follows finalize. It fires when a section that carried at least one
citation before finalize comes out of it with fewer than three distinct cited sentences.
That section is written once more, whole, from its own original prompt rather than from a
revised draft. The second attempt runs the same write, link and verification passes and
finalizes again. Whichever attempt survives finalize with more distinct cited sentences is
the one saved, and the first attempt wins a tie. The backstop runs at most once per
generation, independently of the gate's own revision above. It does not run when the
citation-link call itself failed, since that outcome is already recorded as an outage. The
job result names which attempt was saved.

The same claim-verification agent and finalize step power a standalone action,
`POST /projects/{id}/verify-and-heal`, that re-verifies and re-finalizes a whole draft on
demand, including text a user typed or pasted directly rather than generated.

## Claim-verification record

Claim verification results can be exported as an auditable claim-verification record
(JSON or CSV), listing every checked claim with its matched citation, source paper,
full-text acquisition route, evidence quote/location, and the model's verdict, mirroring
the screening-record export above. Endpoint:
`GET /api/v1/tasks/{task_id}/claim-record?format=json|csv` (owner-only; 404 unless the
task is a completed `claim_verify` job); the claim-verification report UI has a
"Download claim record (CSV)" button next to its model/provenance line. Each row also
carries `claim_sentence`, the full sentence the claim was cut from (equal to `claim`
itself when no citation-link proposition narrowed it to a shorter span), `model_status`
and `machine_reasons`, the reason codes a code-level guard appends whenever it makes the
model's own verdict stricter, alongside the SHA-256 prompt version that produced the run.

## Reproducibility

See `demo/README.md` for a reviewer-executable end-to-end run and
`evaluation/README.md` for the screening and claim-verification evaluations. What is
pinned: temperature 0 for the screener's own batch pass and for claim verification;
reasoning mode at effort high (not a temperature, which the provider does not apply while
reasoning) for the screener's second-pass judge; prompt version hashes; the model reported
by the provider and the system fingerprint recorded with every call; and an exact
dependency set (`backend/requirements.txt` pins plus `backend/requirements.lock.txt`).
What is not pinned: the provider decides what a model alias currently resolves to, and live
retrieval results drift over time.

**Running the demo.** With the stack up (`docker compose up -d`), `python
demo/run_demo.py` drives the whole workflow above through the public API and compares
the result with `demo/expected/`; see `demo/README.md` for prerequisites, expected
output, tolerances and troubleshooting.

**Running the tests.** `cd backend && python -m pytest tests -q` runs the backend suite
against a dedicated `TEST_DATABASE_URL` (see `INSTALL.md` "Running the test suite" for
the database setup and the guard that refuses to run against the application database).
`python -m pytest demo/tests -q` runs the demo script's own unit tests; `cd evaluation
&& python -m pytest tests -q` runs the evaluation harness's own unit tests. Neither of
those last two contacts a network or an LLM; see `evaluation/README.md` for the paid
screening and claim-verification runs themselves.

## Data notice

The four Web of Science Core Collection journal-list CSVs (SCIE, SSCI, AHCI, ESCI)
that the optional venue filter reads are licensed content of Clarivate. They are not
tracked in this repository and not in the release archive. A user with a Clarivate licence
places the four files, under their exact file names, in `backend/data/wos/` and
restarts the backend. The boot import is advisory-locked and import-if-empty, so it
is safe to run with multiple backend workers. The filter is a journal-level venue
filter matched on exact ISSN or eISSN: a record with no ISSN is never matched, and
books, chapters, and non-indexed venues are excluded by construction. Database
coverage is not a measure of research quality.

## Privacy

`PRIVACY.md` states what is sent to the LLM provider and to the scholarly data
services, what is stored and where, and the conditions of the authors' hosted
deployment. Read it before pointing ScholarRAG at confidential material.

## Architecture

| Layer | Technology |
|---|---|
| Frontend | Next.js 15 (App Router), React 19, TypeScript |
| Backend | FastAPI (Python >= 3.12), SQLAlchemy 2.0 (async), Alembic, Pydantic AI |
| Database | PostgreSQL 17 |
| LLM | DeepSeek (`deepseek-flash`) via an OpenAI-compatible API |
| Scholarly data | OpenAlex (retrieval), Crossref (record verification), Unpaywall (open-access location) |
| Jobs | In-process background tasks with polling; no message broker |

```
backend/          FastAPI application
  app/api/        REST routers
  app/agents/     Pydantic AI agents (search, screening, analysis, writing, claim verification, ...)
  app/services/   Service layer (retrieval, verification, writing pipeline, WoS import, ...)
  app/models/     SQLAlchemy models
  alembic/        Database migrations
  data/wos/       Web of Science Core Collection journal lists (see Data notice)
  tests/          Backend test suite
frontend/         Next.js application (src/, messages/ for i18n)
demo/             Reviewer-executable end-to-end demonstration
evaluation/       Screening and claim-verification evaluation harness
```

SoftwareX recommends a `repo/src` directory; this monorepo's equivalent source roots
are `backend/app/` and `frontend/src/`.

## Known limitations in this release

- Background jobs run inside the uvicorn process, so a restart interrupts them; they
  are marked interrupted at shutdown.
- Uploaded files live on the container filesystem, so a rebuild without a persistent
  volume loses them.
- Retrieval depends on the OpenAlex per-IP daily budget when no account key is
  configured.
- The DeepSeek model alias resolves to whatever the provider currently serves; the
  provider does not offer dated model snapshots.
- Scholarly appropriateness of a citation, level 3 above, is not assessed by the
  software.

## Version, licence, citation, support

Version 1.1.0. Source code is MIT licensed, see `LICENSE.txt`. See `CITATION.cff` to
cite this software and `CHANGELOG.md` for release notes. Questions:
yc47728@um.edu.mo (Yunjie Xu, corresponding author).
