# ScholarRAG reviewer-executable demonstration

This folder lets a reviewer reproduce the end-to-end example in the manuscript
(Section 3, Figures 3, 5 and 6) on their own machine with one command. It drives a
locally running ScholarRAG instance through the public HTTP API only; nothing in the
application code is bypassed.

Release: **ScholarRAG v1.1.0**, tag `v1.1.0`. No third-party archive copy is deposited;
on publication the journal copies the accepted version to its own GitHub archive.

## What the demo does

`run_demo.py` performs, in order:

| Step | API call(s) | Output |
|---|---|---|
| 1. Register a throw-away demo user (`scholarrag-demo-<random>@example.com`) | `POST /auth/register` (falls back to `POST /auth/login` on 409) | `credentials.json` |
| 2. Create a project | `POST /projects` | - |
| 3. Smart Search with the fixed research question and inclusion/exclusion criteria from `protocol.json`, venue filter **off** (`wos_filter: "off"`, so no Web of Science list is needed) | `POST /projects/{id}/smart-search`, polled via `GET /tasks/{id}` | - |
| 4. Export the screening record | `GET /tasks/{id}/screening-record?format=json` and `?format=csv` | `screening_record.json`, `screening_record.csv` |
| 5. Add the 12 fixed open-access seed papers from `seed_dois.json` | `POST /projects/{id}/papers` (once per DOI) | - |
| 6. Check that both protocol claims carry a citation the backend verifier can parse and that it keys to the resolved seed paper; **stop with exit code 1 otherwise** (before any LLM step) | - | - |
| 7. Acquire their full texts (server-side Unpaywall -> publisher PDF cascade); if any seed is still abstract-only afterwards, wait 60 s and request acquisition once more for whichever seeds are still abstract-only (see the note below the table) | `POST /projects/{id}/acquire-full-texts`, polled; repeated once if needed | - |
| 8. AI Write the literature-review section described in `protocol.json`. The write job itself now runs a gated loop -- compose, build the citation-link map, verify every claim the citation-link map itself resolved against full text (a citation the link-map call does not return is invisible to this gate; see the note below the table), and (only if at least one claim is not verified) revise the whole section once, keeping every already-verified sentence verbatim, then finalize: remove any sentence whose final status is still not verified, strip any surviving `[NEEDS CITATION]` marker, and save that finalized text to the draft itself, before this script ever reads it back | `POST /projects/{id}/drafts`, `POST /drafts/{id}/generate`, polled | `writing_result.json` (`content` is already the finalized section; `loop_stats`, `claim_report` and `metrics` describe the loop's own run) |
| 9. Append the two constructed claims from `protocol.json` (one expected `verified`, one expected `unsupported`) to the draft | `PUT /drafts/{id}` | `draft_content.json` |
| 10. Run the standalone verify-and-heal action on the now-appended draft and fetch the report. This re-verifies every claim in the saved content (the write job's own finalized section plus the two just-appended fixtures) and heals (removes) any sentence that is not verified, enforcing the same exit invariant a second time over the whole draft | `POST /projects/{id}/verify-and-heal`, polled; `GET /drafts/{id}/claim-verification` | `claim_report.json` |
| 11. Build the delivered-evidence record: one row per claim that survived into the healed draft (`claim_report.json` -> `final_report.verifications`), each row naming the source passage its own evidence quotes were located in | `GET /papers/{id}/fulltext-chunks/{index}` (once per distinct chunk a surviving claim cites) | `delivered_evidence.json` |
| 12. Print a summary table, compare with `expected/`, save the summary | - | `summary.json` |

**Retrying full-text acquisition (step 7).** A seed can come back `abstract_only` because
Unpaywall or the publisher host was transiently unreachable, not because no open-access copy
exists. `DemoRunner._acquire_full_texts` gives a seed still `abstract_only` after the first
call exactly one more chance: if `fulltext.abstract_only > 0`, it waits 60 s and calls
`POST /projects/{id}/acquire-full-texts` again with no `paper_ids` filter; the backend's own
idempotency check (a paper already marked `acquired` is skipped) means this second call only
ever re-attempts the seeds still abstract-only after the first, never the ones already
acquired. `summary.json` -> `fulltext` keeps both calls' own results, in order, under
`attempts`, and its top-level `acquired`/`abstract_only` report the state after the retry (the
same two keys, unchanged, when no retry was needed). `already_acquired` and
`reference_chunks_dropped` are not recombined after a retry and stay at the first attempt's
own values, so on a run where a retry actually fires those two keys describe the first attempt
only, not the whole run. This retry is exercised by
`demo/tests/test_run_demo.py` against `FakeServer`, not against a live host. **The promoted
baseline in `demo/expected/` did not need it**: all 12 seeds acquired full text on the first
attempt (`fulltext.retry_attempted: false`, and `summary.json` -> `fulltext` -> `attempts`
carries the one attempt made, not a second -- see "Approximate run time and cost" below).

After step 12, `run_demo.py` writes and verifies every section named in `demo/sections.json`
(`--sections <path>`, default `demo/sections.json` when that file exists; pass a different
path, or remove/rename the file, to skip -- `demo/sections.json` is a fixed list of `{title,
instructions, target_words}` objects, the same shape as `protocol.json`'s own `section`), each
through the
identical write-then-heal sequence steps 8 and 10 use on a fresh draft of the same project,
producing its own `writing_result_<n>.json`/`draft_content_<n>.json`/`claim_report_<n>.json`
(`<n>` = 2, 3, 4, ... -- the protocol section's own unindexed files, described in "Expected
outputs" below, are section 1) and its own entry in `summary.json` -> `sections`. Every
section's own `delivered_evidence.json` row is merged into the ONE file step 11 writes, and
`summary.json` -> `delivered_check` records the whole run's own violations.

`demo/sections.json` ships six extra sections, sized from a completed seven-section run
rather than extrapolated from the protocol section alone. On the current baseline, a full
seven-section run (protocol plus all six extra sections) produced 68 rows in `delivered_evidence.json`
(11, 10, 14, 12, 7, 5 and 9 rows per section, in the order written), against the human-review
workbook's Delivered sheet target of about 35 rows; the actual count varies baseline to
baseline, not only section to section, since it depends on model drift in how many
sentences a section's own citations survive verification. Check the next run's own
`delivered_evidence.json` (`{run_id, rows, ...}`, row count = `len(rows)`) against this
68-row measurement rather than treating either number as exact.

Right after every
section is saved (the protocol section and each extra one), `demo/check_delivered.py` -- a
self-contained, offline, no-network/no-database/no-LLM module -- checks ten of its own eleven
invariants over that section's own three files (no sentence the citation-link call left
unclassified or tagged an uncited "finding" survives; no `[NEEDS CITATION]` marker or
unresolved citation remains; no two consecutive heading nodes carrying the same text;
every final-report row is verified
and its `claim_text` occurs verbatim in the draft; no sentence truncated at an abbreviation;
every protocol fixture claim reached its expected status, checked on the protocol section
only; the deterministic coherence pass `app.services.fulltext` runs after finalize -- no body
paragraph other than the document's own opening one carries zero citations, no enumeration
opener has fewer sentences than it announces (unless the next sentence resolves the same count
as a colon-free list of its own), and no uncited sentence opens with a discourse
connective/ordinal or an unresolved deictic once an earlier enumeration shortfall fired in the
same paragraph; no heading is left with no body of its own, whether because it is a
section's own last node or because another heading immediately follows it; and no delivered
cited sentence asserts material outside its own verified claims and
citations -- a content-bearing residue that is not one of three closed discourse frames, or a
citation-link proposition with no final-report row behind it at all -- and no delivered
sentence carries a comparative meta-evaluation of the literature). Three further rules catch
three more shapes an empty, self-referential or heading-shaped section leaves behind: a
section whose own final report carries no verified row at all despite its own citation audit
finding a real citation, or whose own `loop_stats` records that the empty-section regeneration backstop
below also finalized to zero cited sentences on its second attempt (`empty_cited_section`);
a delivered body paragraph that echoes the section's own title or one of its headings
rather than stating a finding or framing sentence of its own (`title_echo_in_body`); a
delivered body paragraph that reads as a heading rather than a sentence at all -- no
sentence-final punctuation, at most ten words, no citation and no finite verb, whatever
its position (`heading_shaped_paragraph`); and a heading node anywhere but the section's
own first node, whether or not it has a body of its own -- a generated section carries
only its own title heading now, so every other heading the writer wrote is dropped whole
at finalize and should never survive (`non_title_heading_in_section`). Rule 5, every
`delivered_evidence.json` row's evidence quotes are verbatim in their own `source_passage`,
is a whole-run artefact and is checked once, after step 11, over the
merged file every section's own rows were folded into. Any violation is exit code 2; run it
by hand against any run directory with `python demo/check_delivered.py demo/output/<run>`.

Step 8's own gate is driven only by `claims_from_citation_links(citation_links)`
(`writing.py`), the citations the citation-link call itself resolved -- not by the broader,
four-step fallback chain `extract_claims_from_document` uses (see "Why step 6 matters"
below). A citation the link-map call does not return is invisible to that gate and to
finalize's own removal rule (rule 1 only strips an uncited sentence the linker itself
tagged "finding"), so such a sentence can reach the saved draft without the write job's own
loop ever having verified it, to be picked up only later, by step 10's broader extraction.
On the current baseline every one of the section's own citation occurrences resolved
through the link map (`citation_coverage.by_source.mapping` below), so this gap was not
exercised, but the loop's own gate does not itself guarantee that a future run's citation
will always be covered; see the citation-coverage paragraph below for the exact count.

As a backstop for whatever still slips past that gate, a section whose own citation link
map or citation audit found a real citation but that finalize reduces to fewer than three
distinct cited sentences is regenerated once, whole -- a fresh write call, then the same
citation-link and verification passes, finalized again -- and ships the second attempt's own
outcome either way, recording `loop_stats.empty_section_regenerated` and, when the second
attempt still finalizes to zero cited sentences, `loop_stats.empty_section_after_regeneration`
(that flag stays tied to the literal zero, not the three-sentence trigger, since the checker's
own `empty_cited_section` rule treats it as an unconditional violation and a one- or
two-sentence section is a thinness for a reader to judge, not that violation); there is never
a third attempt.

Every claim verification call, in step 8's own gated loop and in step 10 alike, first tries
to relocate a quote onto the source text before deciding whether the quote is verbatim.
Before the fixed guards check a quote for an exact match, two narrow, provably harmless
differences between the model's copy and the source -- an article substituted, inserted or
dropped ("a"/"an"/"the"), or one whole parenthetical citation dropped from the quote -- are
repointed at the source's own wording, so a claim is never marked less than fully verified
over a difference of that kind. If the quote is still not verbatim after relocation and the
claim would otherwise be marked `needs_nuance` for exactly that reason, the verification
model is asked, in the same conversation, to repair its own quote against the passage it was
already shown, naming every segment that did not match, and that repair turn's own answer is
taken as final, whatever it is. Every pass runs through the same relocation step and the same
guards; only the repair turn's own model call is conditional. Both relocation and the repair
turn run before the guards that decide the reported status, and both are recorded:
`claim_report.json` -> `provenance` carries the guard set's own digest, the relocation rule's
own version and the repair prompt's own version, and each verification carries
`quote_relocations` (empty when nothing needed relocating) and `passes` (one entry, or two
when the repair turn ran). The frozen `quote_not_verbatim` guard reads only the top-level
`evidence_quote`/`evidence_quotes`; it never reads an assertion's own `quote`/`quotes`. If a
repair turn's answer removes a paraphrase from the guarded top-level list without also
correcting the same paraphrase where it sits on an assertion, that paraphrase would otherwise
still reach the stored report and the claim-verification screen, verified and unflagged, on a
field the guard cannot see. One further write-time step closes that gap: after the guards have
already decided the status, any assertion `quote`/`quotes` entry that is not verbatim in the
paper's own chunks (the same check the guard itself uses) is dropped from what is stored and
returned, so a paraphrase can never survive on an assertion even when it is absent from the
guarded list. This runs after the status is decided and only removes text, so it never changes
a verdict.

When relocation and the repair turn still leave a claim's own quote not verbatim, the frozen
`quote_not_verbatim` guard demotes it from `verified` to `needs_nuance`, exactly as it is
meant to. This script's own fixture check (`is_guard_demotion` in `run_demo.py`) tells that
shape apart from a genuine wrong answer: a fixture expecting `verified` still counts as
passing (`matches_expected: true`) when the verification model's own `model_status` was
`verified` and the frozen guard alone -- no other reason alongside it -- is what changed the
reported status. When this happens the printed per-claim line reads `GUARD DEMOTION` instead
of `OK`/`MISMATCH`. That allowance covers the raw *status*, and `claim_problems` grants the
SAME allowance to the healed report: a fixture whose `expected_status` is `verified` and
whose raw `status` is a recorded guard demotion (the frozen guard alone, no other reason
alongside it, demoted a `verified` `model_status`) is accepted here too, even once the
standalone verify-and-heal action removes it from `final_report` the same way it would
remove any other non-`verified` sentence, matching the allowance the printed line above and
`check_delivered.check_fixture_claims`'s own rule 7 already grant. `claim_problems` only
reports a problem here when NEITHER a `verified`-fixture's `final_status` is `verified` NOR
any `verified`-fixture is itself a recorded guard demotion; a genuine loss of the `verified`
fixture in some other shape is still a real failure to demonstrate the claim this run exists
to demonstrate. On the current baseline neither fixture needs this allowance: `supported-1`
verifies (`model_status: verified`, `machine_reasons: []`, `guard_demotion: false`,
`matches_expected: true`, `pass_count: 2`) after one repair turn that returned the source's
own verbatim wording, with no quote relocation needed. `unsupported-1` was correctly caught
as designed (`model_status: unsupported`, machine reason `assertion_status_inconsistent`,
one pass, `final_status: "not_in_report"` once the standalone verify-and-heal action removes
it). See "Verdict distribution, before and after finalize" below for the full detail.

The repair turn is not just a backstop for other claims -- it fired on the `supported-1`
fixture itself this run, and six times in all across the whole baseline, every one resolving
cleanly. At the write stage, the "Direct versus indirect" extra section's own write job hit
the repair turn twice: once on the `Razmi and Ghane (2024)` row attributing the effect's
magnitude partly to the absence of revision opportunities in their design, and once on the
`Zhang and Hyland (2021)` row about an integrated feedback approach promoting behavioural,
affective and cognitive engagement; both first-pass quotes paraphrased the source rather than
quoting it verbatim (`machine_reasons: ["quote_not_verbatim"]`), and both repair turns
returned the source's own verbatim wording, closing `verified` with `machine_reasons: []`.
The standalone verify-and-heal action (step 10) then hit the repair turn four more times: on
the `supported-1` fixture itself (the protocol section's own heal-stage provenance reports 13
calls for 12 rows, the one surplus call being this repair), on the same two "Direct versus
indirect" rows re-verified as part of the whole draft (that section's own heal-stage
provenance reports 14 calls for 12 rows), and on the "Peer written corrective feedback"
extra section's own `assignment differences may have confounded this comparison` row (that
section's own heal-stage provenance reports 10 calls for 9 rows). Every one of the six
resolved to `verified` with `machine_reasons: []` and no guard demotion. No other
verification row, at either stage, on any of the seven sections, carried more than one
pass.

Because the write job's own gated loop already finalizes the section before step 9 even
runs, `writing_result.json` -> `claim_report` (the write job's own internal report, over
its own section's claims only, before the two fixtures exist) always shows
`unsupported_count: 0`, `nuance_count: 0` and `abstract_only_count: 0` by construction:
`finalize_generated_section` removes any sentence whose citation is not verified before the
draft is ever saved, and `surviving_verifications` (which this report is built from) can
only return rows finalize kept, so those three counts can never take another value. The
number worth quoting from that report is `verified_count` (a real measurement), together
with the finalize removal counts in `writing_result.json` -> `loop_stats`
(`sentences_removed_unverified`, `sentences_removed_uncited_finding`,
`sentences_removed_no_full_text`, `citations_dropped_no_full_text`,
`needs_citation_markers_removed`, `sentences_removed_dangling` -- the deterministic
coherence pass's own removal count, added on top of the three ordinary
removal rules; and `sentences_residue_excised` (a trailing
residue -- unverified material outside every kept proposition -- cut away while the
rest of the sentence survives), `sentences_removed_coverage_incomplete` (no safe cut
existed, so the whole sentence was removed for a residue or a comparative meta-
evaluation) and `frame_spans_accepted` (the number of residue spans the deterministic
frame classifier accepted as attribution or a short lead-in rather than flagging)) --
the number and kind of sentences the loop actually took out, which is where the
enforcement is visible.

`claim_report.json` (step 10's report) is a different, later measurement: it re-verifies
every claim in the draft as it stands after step 9, which is the write job's own
already-finalized section plus the two just-appended fixture claims, and its top-level
`verified_count`/`unsupported_count`/`nuance_count`/`abstract_only_count` are real counts
over that whole draft, not forced to any value. In practice the section's own claims
already survived the write job's finalize, so this top-level count is dominated by the two
fixtures: one designed to verify and one designed not to, so `unsupported_count: 1` (or
`nuance_count: 1`, depending which status the model actually returns for it) on a normal
run is expected, not a defect. `claim_report.json` -> `final_report` is the separate,
post-heal view: `final_report.verifications` and
`final_report.verified_count` describe the draft *after* healing removed whatever the
top-level counts found, so `final_report.verified_count` is the number to quote as "how
many claims survive in the draft now", and `finalize_stats` alongside it names what
healing removed.

**The two claims are appended by the script, not written by the model.** They are
inserted verbatim as the last two paragraphs of the draft, so that the claim-verification
step has one claim that the cited paper supports and one that it does not.
`protocol.json` documents how each claim was constructed (both cite Mao, Lee & Li, 2024,
*Language Teaching*, an open-access synthesis of written-corrective-feedback studies).

Why step 6 matters: it is a check on the two constructed protocol claims only, before
any paid LLM call runs. It applies the backend's author-year regex, `(Surname, YEAR)` /
`(Surname et al., YEAR)` with a single ASCII capitalised surname, matches it to a
library paper by `surname_year` derived from the first author's last-name token,
rebuilds that same key from the metadata resolved in step 5, and refuses to continue
when the two disagree, so a run can never silently verify nothing. That regex is not
what the pipeline uses to extract claims from the model's own generated text. Extraction
from a generated section tries, in order: (1) the writer's own sentence-to-reference map
first (a second DeepSeek call made over the finished section; see `writing_result.json` ->
`citation_links` below); (2) the author-year regexes -- both parenthetical and narrative,
including a multi-token surname such as "Bonilla Lopez et al. (2018)" -- for a sentence
with no surviving link; (3) an audit-key fallback, for a sentence still without one, keyed
from `citation_audit`'s own surname-and-year match; and (4) a numbered-citation resolver
last, when the draft carries a matching reference list. The regex (2) and the audit-key
fallback (3) share one source label, `author-year`, in `citation_coverage.by_source`; see
"Approximate run time and cost" below. In the promoted run, the audit's own span
recognition (`citation_audit.citation_spans`, also reported as `claim_report.json` ->
`citation_coverage.found`) recognises 11 citation occurrences across the whole draft: 9 are
in the generated section and the other 2 are the fixture paragraphs step 9 appends; all 11
are recorded under `by_source.mapping`, 0 under the author-year fallback and 0 under the
numbered resolver, because by the time step 10's coverage count runs, the standalone
verify-and-heal action's own repair step has already written a link back onto every
paragraph it resolved (see the `draft_content.json` row above on how a repaired link is
stored), so a fixture citation the writer's own link-map call never saw can still show up
under `mapping` once healing has run once over the draft. This is separate from the
`citation_audit` block reported in `writing_result.json`, which counts DISTINCT citations
recognised across the section as the loop's own verify-and-revise cycle produced it, before
finalize's own removals ran: 6 there on this run (0 unmatched, `needs_citation_flags: 2`),
over text finalize had not yet pruned. All six of those distinct citations (`Razmi, 2024`,
`López, 2018`, `Luo, 2025`, `Koltovskaia, 2022`, `Zhang, 2021` and `Mao, 2024`) survive into
the section that actually reached the saved draft; finalize's own removals dropped only
redundant mentions of an already-cited paper, not a citation outright. `citation_audit` and `citation_coverage.by_source.mapping` are still two different
counts in general and need not agree; on this run they diverge because one counts distinct
citations (6) and the other counts occurrences including the two appended fixtures (11), not
because either counted wrong. The loop's own finalize step removed four sentences from the
section on this run: two uncited "finding" sentences that carried no citation at all and two
cited sentences that still did not verify even after the section was revised once; nothing
was dropped as dangling
(`sentences_removed_uncited_finding: 2`, `sentences_removed_unverified: 2`,
`sentences_removed_dangling: 0`), and a fifth sentence was kept with a trailing, uncited
clause cut away (`sentences_residue_excised: 1`), leaving the section that reached the
saved draft (see "Verdict distribution, before and after finalize" below). A citation the
link map does not resolve is possible in principle -- the note under
"What the demo does" above explains what happens to it in that case -- it simply did not
occur on this run.

Step 5 resolves each seed DOI against the public OpenAlex API
(`https://api.openalex.org/works/...`) to obtain the author list, abstract and ISSN; the
seed file itself is the source of truth for DOI, title, year, journal and OpenAlex id.
No personal data is sent unless you opt in through `OPENALEX_EMAIL` (polite pool) or
`OPENALEX_API_KEY` in your environment, the same variables the backend uses. A 429 or
5xx from OpenAlex is retried with backoff; if it still fails, Crossref
(`https://api.crossref.org/works/<doi>`, no key) supplies the author list, year and
journal so that the citations can still be matched. The included papers from Smart
Search are *not* added to the library - only the 12 fixed seeds are, so that the
full-text and verification steps are the same for every run.

## Prerequisites

- Docker Desktop / Docker Engine with Compose v2.
- The `docker-compose.yml` of ScholarRAG **v1.1.0** (this tree). Its backend service
  starts with `alembic upgrade head && uvicorn ...`, so the database schema is created
  automatically on first start, and its `env_file: .env` is optional (`required: false`).
  Older compose files start `uvicorn` directly and require `.env` to exist; with them
  the demo fails at `POST /auth/register` with HTTP 500 (see Troubleshooting).
- Python 3.12 or newer with the `httpx` package (`pip install httpx`); no other
  third-party dependency. Where `python` is not on `PATH` (Debian/Ubuntu, macOS), use
  `python3` in the commands below.
- A DeepSeek API key: `DEEPSEEK_API_KEY` is the only mandatory setting in `.env`.
  Screening, writing and claim verification call DeepSeek with the paper
  titles/abstracts, the retrieved full-text passages and the draft text; nothing else
  leaves your machine. The Web of Science journal list is **not** required.
- A free OpenAlex API key, recommended the same way the DeepSeek key above is: sign up
  at `openalex.org/settings/api` and set `OPENALEX_API_KEY=<your key>` in `.env`. It
  raises the anonymous per-IP rate limit and the daily request budget; it never changes
  which papers Smart Search finds or how they are screened, so the demo runs the same
  with or without it, only more reliably under load.
- Optional: `UNPAYWALL_EMAIL=<your address>` and `OPENALEX_EMAIL=<your address>` in
  `.env`. Unpaywall asks for a contact address and the backend skips Unpaywall when the
  variable is empty; full-text acquisition still succeeds without it because the script
  stores each seed's verified `oa_pdf_url` as the paper's `full_text_url`, which is the
  second step of the backend's cascade and points at the same PDF.
- Outbound network access to `api.openalex.org`, `api.crossref.org`,
  `api.unpaywall.org`, `api.deepseek.com` and the publishers' PDF hosts.

All 12 seed PDFs were verified on 2026-09-02 with a client identical to the backend's
download path (`httpx`, 30 s timeout, redirects followed, default User-Agent, no extra
headers; HTTP 200, `content-type: application/pdf`, body starting with `%PDF`). See
`SOURCES.md` for the audit trail.

## Commands

From the repository root, on a fresh clone:

```bash
cp .env.example .env
# edit .env: set DEEPSEEK_API_KEY=<your key> (mandatory); optionally JWT_SECRET_KEY,
#            UNPAYWALL_EMAIL and OPENALEX_EMAIL
docker compose up -d            # first run builds the images (several minutes)
docker compose ps               # wait until backend and postgres report healthy/running
python demo/run_demo.py         # the whole workflow; prints progress and a summary table
                                # (python3 demo/run_demo.py where python is not on PATH)
```

`docker compose` reads `.env` through the backend's `env_file` entry, which is marked
optional: the stack starts without the file, but every LLM step then fails until
`DEEPSEEK_API_KEY` is set. On start the backend runs `alembic upgrade head` before
`uvicorn`, so no manual migration is needed. The backend listens on
`http://localhost:8000`, the web interface on `http://localhost:3000`.

Useful flags:

```
--api-url URL     API base (default http://localhost:8000/api/v1)
--protocol PATH   protocol file (default demo/protocol.json)
--seeds PATH      seed file (default demo/seed_dois.json)
--expected DIR    directory with expected/summary.json (default demo/expected)
--output DIR      where to write outputs (default demo/output/<YYYYMMDD-HHMMSS>)
--skip-search     skip Smart Search; run only the seed -> full text -> write -> verify path
--search-only     stop after exporting the screening record; write a summary with only
                  the screening block (exit codes unchanged; mutually exclusive with
                  --skip-search)
--replay-queries  read provenance.rounds[].queries[].query from --expected's
                  screening_record.json and send them as queries_override instead of
                  letting the query generator agent invent new ones (default off)
--timeout SEC     wait per background task, every task except Smart Search (default 2400
                  = 40 min)
--search-timeout SEC  wait for Smart Search specifically, which has its own larger
                  budget (default 4500 = 75 min)
--strict          exit code 3 when the comparison with demo/expected/ is outside tolerance
```

### Replaying the shipped search

`--replay-queries` makes a run ask Smart Search the same questions, round by round, that
the run in `--expected` (`demo/expected/` by default) already asked, instead of letting
the query generator agent write new ones. It reads
`provenance.rounds[].queries[].query` from that directory's `screening_record.json` and
sends the resulting list of lists as `queries_override` on the Smart Search request; the
backend then uses list `r` for round `r` and only falls back to generating queries for a
round past the end of the list. Combine it with `--search-only` to check just the search
step's own reproducibility without waiting for full-text acquisition, writing and
verification to run too. When `demo/expected/summary.json` is present, the comparison
table also reports `screening.replay_queries` as `identical` or lists the rounds that
differ.

**`python demo/run_demo.py --replay-queries --search-only` works against the
`demo/expected/` shipped in this tree.** `--replay-queries` needs `--expected` pointed at
a run whose `screening_record.json` carries `provenance.rounds` -- the per-round query
log -- and the baseline promoted here (2026-09-16) carries it, so the plain command above
is enough; against an older baseline with no `rounds` key at all it fails immediately with
`demo/expected/screening_record.json has no provenance.rounds to replay` (validated before
any HTTP call, so nothing is registered or created on the live stack when this happens).
Point `--expected` at a different run directory to replay that run's own queries instead:
`python demo/run_demo.py --replay-queries --search-only --expected demo/output/<a run's timestamp>`.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | every step completed and both protocol claims received their expected status |
| 1 | `DEMO FAILED: ...` - a step failed: unreachable API, HTTP 4xx/5xx, a background task that ended `failed`/`cancelled`, a task that exceeded `--timeout`, or a protocol claim whose citation the backend could not parse or match (step 6) |
| 2 | `DEMO CLAIM MISMATCH: ...` - the run completed but a protocol claim did not receive its expected status, or the verification report was empty; independent of `demo/expected/` |
| 3 | `DEMO DRIFT` - `--strict` was given and the comparison with `demo/expected/` is outside tolerance |

A transient failure while polling a task (connection reset, HTTP 5xx) is retried up to
five times at the poll interval before the run is abandoned, because the backend job
keeps running regardless.

After the run you can open `http://localhost:3000`, log in with the e-mail and password
in `output/<timestamp>/credentials.json`, and inspect the same project, screening record,
library badges and claim report in the web interface (this is how Figures 5 and 6 were
captured).

## Expected outputs

`demo/output/<timestamp>/`:

| File | Content |
|---|---|
| `screening_record.json` | `{criteria, flow, provenance, retrieval_failures, records}`; `flow` has `identified, duplicates_removed, stage1_screened, stage1_excluded, stage2_screened, stage2_excluded, needs_review, unscreened, included, rounds, stop_reason, needs_review_by_reason`; `needs_review_by_reason` breaks the `needs_review` count down by reason (e.g. `no_abstract`, `unanchored_exclude`, `undecidable`); `records` has one entry per de-duplicated record that reached screening (914 of the 2,947 `identified` records on the current baseline, after `duplicates_removed`) with `outcome` (`included`/`excluded`/`needs_review`/`unscreened`), `stage`, `status` (the screener's own protocol-eligibility decision, `INCLUDE`/`EXCLUDE`/`NEEDS_REVIEW`, empty on a record the LLM screener never reached), `criterion` and `quote` (the exclusion criterion id and verbatim quote behind an `EXCLUDE`, or any anchor a guard demoted on a `NEEDS_REVIEW`; empty otherwise), `needs_review_reason` (why a `needs_review` record needs a human look: a guard reason such as `unanchored_exclude` or `cut_abstract`, or the model's own `no_abstract`/`undecidable`; empty for every other outcome), `to_confirm` (comma-joined full-text inclusion criterion ids still to confirm, non-empty only on `included` records), `reason`, `abstract` (empty string when the paper has none) and the other paper fields; `retrieval_failures` lists `[query, error]` for every search query that raised, whether the cause is transient (a provider 5xx that clears on its own) or deterministic (a malformed query the provider rejects every time); it does not stop the run either way (empty on the current baseline -- see "Approximate run time and cost" below; the mechanism is real even though this baseline did not hit it). The no-abstract guard only demotes an `EXCLUDE`, so an abstract-less record the model itself decided to `INCLUDE` is not touched: in the baseline in `demo/expected/`, 127 of the 914 Stage-2-screened records carry no abstract -- 103 were routed to `needs_review` for that reason and 24 were nonetheless `excluded` under the `TOPIC` criterion, decided from the title alone (0 were `included` by the model itself). |
| `screening_record.csv` | one row per record; header `outcome,stage,status,criterion,quote,needs_review_reason,to_confirm,second_pass,second_pass_input_tokens,second_pass_output_tokens,second_pass_latency_s,reason,title,doi,year,journal,journal_issn,openalex_id,abstract` (19 columns; the four `second_pass*` fields, filled in only for the once-per-job second-pass judge's own candidates, sit between `to_confirm` and `reason`; the same fields as `screening_record.json`'s `records`, in the order `CSV_HEADER` fixes) |
| `writing_result.json` | `content` (the write job's own gated loop already verified, revised if needed, and finalized this text -- it is not the raw first-pass generation), `citation_audit` (`matched`, `unmatched`, `needs_citation_flags`, `total`), `citation_links` (the writer's own sentence-to-reference map, repaired by finalize: one entry per surviving sentence-level citation mention, with `paragraph_index`, `sentence`, `citation_text`, `keys` and `proposition` (the single fact-claim the sentence makes, used to check a revision's replacement sentence still asserts the same thing); `null` when the link-mapping call itself failed), `loop_stats` (`length_regenerated` -- the pre-revision body was over the hard maximum and was regenerated once; `loop_revised` -- the verification gate found a problem and the whole section was rewritten once, independent of `length_regenerated`; `first_pass_verified_rate`; `survival_rate`, defined as `final_words / target_words`, so a value over 1.0 means the finished section ran long, not that "more of it survived"; `sentences_removed_unverified`, `sentences_removed_uncited_finding`, `sentences_removed_no_full_text`, `citations_dropped_no_full_text`, `needs_citation_markers_removed`, `sentences_removed_dangling`, `sentences_removed_title_echo` -- a body paragraph that only restated the section's own title or one of its headings, dropped rather than delivered --, `sentences_removed_heading_shaped` -- an uncited paragraph that read as a heading rather than a sentence (no closing punctuation, at most ten words, no citation, no finite verb), dropped by the same independent guard the checker's own rule 16 also runs --, `link_map_failed`, `link_coverage_retry_failed` -- a coverage-gap retry call raised outright, as opposed to answering but never naming the gap sentence, which is instead removed as an ordinary uncited finding --, `links_unmatched_to_sentence` -- a citation link whose own sentence matched no fragment of the paragraph it was bucketed against, the shape the sentence-snap fix targets and this counts as a backstop --, `empty_section_regenerated` and `empty_section_after_regeneration` -- the empty-section regeneration backstop described above --), `claim_report` (`verified_count` is a real measurement; `unsupported_count`, `nuance_count` and `abstract_only_count` are always 0 by construction -- finalize already removed anything that would have made them non-zero, so quote `verified_count` and the `loop_stats` removal counts instead of these three), `metrics.analysis` (`papers`, `papers_attempted`, `papers_failed` -- a paper whose grounding call or whose analysis write failed is not counted in `papers` but is counted in `papers_attempted`, so the gap is visible instead of silently reducing the corpus a report is built on; `evidence_items`, `rejected_items`, `verbatim_rate`) and `provenance` (model configured/reported, temperature, prompt version, and `input_tokens`/`output_tokens` -- all four for the first generation call alone; `length` (`target_words`, `hard_maximum`, `words` -- the pre-revision, pre-finalize body length -- `final_words` -- the length of the write job's own finalized body, after any revision and the write job's own finalize removals, computed in `writing.py` before this file is ever saved; step 9 (protocol section only) then appends the two fixture paragraphs, and the later standalone verify-and-heal action's own `finalize_stats` (see `claim_report.json` below) can remove sentences from that appended text -- so the length of the draft actually saved (`summary.json` -> `writing.length.final_words`, a recount of that saved draft, see "Verdict distribution, before and after finalize" above) can come out either higher than this field, when the surviving fixture paragraph outweighs anything the heal removes, as on the protocol section of this baseline (341 saved against 296 recorded here), or lower, when the heal removes more than the fixture adds --, `regenerated`, `over_target`), present only when the request carried a `target_words`; `citation_link_call`, the most recent citation-link call's own record; `citation_link_calls`, the full list of every citation-link call made, in order (one call per map build -- one for the first-pass text, none of its own from an over-length regeneration since that runs before the map is built, one more each time the coverage-gap closer's own unclassified-sentence retry fires, and one more for a gate-triggered revision's own map -- up to four in all; the promoted baseline's own four are the first-pass map and its coverage-gap retry, then the revision's map and its own retry; this is the only place every call's own response id, fingerprint and token split survive); `analysis_calls`, one entry per full-text grounding call; `regeneration_calls`, one entry per empty-section regeneration write call (empty when the backstop never fired); and `total_calls`/`total_input_tokens`/`total_output_tokens`, the folded totals across every DeepSeek call the section made) |
| `draft_content.json` | the Tiptap document saved to the draft, as it stands after step 10's standalone verify-and-heal action has run (the file is written three times over the script's own run; this is the last write): the write job's own finalized paragraphs plus whichever of the two appended fixture claims the heal action kept. A fixture the heal action removes (its own sentence not verified) is therefore absent from this file by design, not a defect -- on the current baseline `unsupported-1` is absent (correctly caught, as designed) and `supported-1` survives (verified after one repair turn, no guard demotion; see "Verdict distribution, before and after finalize" below). Each generated paragraph node carries the same links as `writing_result.json` -> `citation_links` restricted to that paragraph, under a `citationLinks` attribute (`sentence`, `citation_text`, `keys`, `evidence_ids`, `proposition` -- the narrowed, verbatim span of the sentence that citation actually supports, so a grouped-citation sentence is narrowed to the one fact each citation supports rather than sent to the verifier whole once per citation), which is what `extract_claims_from_document` and the export/paper-chat citation renderer both read first. This is the only place the `proposition` field can be inspected in a saved artefact -- `demo/expected/` never ships `draft_content.json` by design (only `summary.json`, `screening_record.json`/`.csv`, `claim_report.json` and `delivered_evidence.json` are promoted there); read it from the promoted run's own directory, `demo/output/20260916-154140/` (kept on disk alongside `demo/expected/` so that `demo/tests` has a real draft to check against; see "Approximate run time and cost" below and the note on `demo/output/` below the "Expected outputs" table). A link the heal repairs for a regex- or numbered-resolved citation (one the writer's own link map did not return) carries only `sentence`, `keys` and `citation_text`: `_claims_for_paragraph_text` writes such a link as a bare single-key entry, with no `proposition` and no `evidence_ids`, because neither exists for a claim resolved outside the link map. On the current baseline the 10 `citationLinks` entries of the generated section went through the link map and carry `sentence`, `citation_text`, `keys`, `evidence_ids` and `proposition`, with no `paragraph_index` (the node's own position already is one). The surviving `supported-1` fixture paragraph's own link entry is a third shape neither of those: the standalone verify-and-heal action's own repair step wrote it back with the full field set (`sentence`, `citation_text`, `keys`, `evidence_ids`, `proposition`) PLUS `paragraph_index: 0` -- `{"keys": ["mao_2024"], "sentence": ..., "proposition": ..., "evidence_ids": [], "citation_text": "(Mao et al., 2024)", "paragraph_index": 0}`, at node 4 (the fifth and last of the draft's 5 nodes). Read this row, not the two-shape description above, as the shape a repaired fixture paragraph's own link carries. |
| `claim_report.json` | `verifications[]` (twenty-one fields per verification, including `claim_text`, `claim_sentence` (the full sentence the `claim_text` proposition was cut from), `status`, `model_status` (the model's own verdict before any guard demotes it), `machine_reasons` (the guard reasons that fired, whether or not they changed the status), `citation`, `paper_id`, `paper_doi`, `paper_title`, `acquisition_route`, `assertions`, `evidence_quote`, `evidence_quotes`, `evidence_location`, `explanation`, `unstated_details`, `suggested_revision`, `model_reported`, `diagnostics`, `quote_relocations` (empty when nothing needed relocating onto the source before guarding) and `passes` (one entry, or two when the repair turn ran)), the top-level counts (real, pre-heal measurements over the whole draft as it stood when step 10 ran -- not forced to any value; see "What the demo does" above for why the write job's own section rarely contributes to them), `full_text_coverage`, `citation_coverage`, the aggregated `provenance` (including `guard_digest`, `quote_relocation_version` and `verification_policy_version`), and three additive keys from the standalone verify-and-heal action: `healed` (`true`), `finalize_stats` (the same shape as `writing_result.json` -> `loop_stats`'s removal counts, for whatever this pass itself removed), and `final_report` (`verifications`, `verified_count`), the post-heal "every row verified" view of the draft as it stands after this step; `evidence_location` (`"chunk k (section)"`) is a best-effort exact-substring match of `evidence_quote` against the chunk text shown to the model and is `null` whenever the model paraphrased rather than quoted verbatim, including for some `verified` claims. The match is resolved through 30-character-or-longer segments of the quote (`_quote_segments`), not the quote as a whole, and returns the first chunk, in the order shown to the model, that contains a segment: on a paper whose chunks repeat wording (an abstract restating a results-section sentence, for instance), the label can therefore name a chunk that carries only part of the quote verbatim while a later chunk carries all of it -- read the label as locating a matching segment of the quote, not as a guarantee that the whole quote is verbatim in exactly that chunk. `citation_coverage` (`found`, `linked`, `sent_to_verifier`, `unresolved`, `unresolved_citations`, `by_source`) reports how many citation occurrences in the draft (`found`) were resolved to a citation key at all (`linked`) versus not (`unresolved`, listed verbatim in `unresolved_citations`); `sent_to_verifier` counts claims rather than citations, since a single occurrence naming several papers together produces one claim per key. `unresolved` covers two different cases: an occurrence the writer's map and the author-year regexes could not key at all (no claim is produced), and, separately, an occurrence that does resolve to a key naming a paper absent from the library (still sent to the verifier as a claim and reported `no_full_text`, never silently dropped). On the current baseline (see "Approximate run time and cost" below) neither case occurs: `unresolved` is 0 and `unresolved_citations` is empty |
| `delivered_evidence.json` | `{run_id, rows, unlocated_rows}` (no top-level `draft_id`/`section_title` -- a multi-section run's own rows come from more than one draft, so those two fields moved onto each row instead, `section_title` and `draft_id`); one row per claim in `claim_report.json` -> `final_report.verifications` of EVERY section written (the protocol section, then each of `demo/sections.json`'s own extra sections in order, `demo.run_demo.merge_delivered_evidence_records`), each with `section_title`, `draft_id`, `sentence`, `citation_text`, `paper_title`, `paper_doi`, `claim_text`, `status`, `evidence_quotes`, `sentence_in_draft` (whether the sentence occurs verbatim in that row's own section's `draft_content.json`/`draft_content_<n>.json`), `source_located`, `source_passage` (the chunk excerpt(s) each quote was found in, joined; `null` when nothing was located), `source_locator` (`chunk_index`, `section` -- the row's own primary chunk, the one `evidence_location` names), `excerpt_locations` (one `{chunk_index, section}` entry per excerpt actually joined into `source_passage`, in the same order: a quote the primary chunk does not contain is located by searching every other chunk of the paper, so a multi-quote row's passage can span more than one chunk, and `excerpt_locations` records exactly which one each excerpt came from rather than leaving every excerpt attributed to the row's single `source_locator`), `passage_located` (true only when every one of `evidence_quotes` was located and a passage is shown; also false, with `source_passage` falling back to the start of the chunk, when the row carries no evidence quote to locate at all), `unlocated_quotes`, `location_section_mismatch` and `fetch_error`. `unlocated_rows` counts rows whose `passage_located` is false, so a reader can see that count without scanning every row. A chunk's own `section` label (`source_locator`, `excerpt_locations`, and the parenthesised part of `evidence_location`) names the section a chunk STARTS IN, not what the chunk actually contains -- the demo's own chunking can put most of a paper's body text into a chunk labelled `abstract` simply because that chunk begins where the abstract does, so a row can read "chunk 2 (abstract)" even though the quoted evidence came from body text further down the same chunk. `location_section_mismatch` is therefore always `false`: it compares the label `evidence_location` itself named against the fetched chunk's own label under that same index, which is always the same value, so it is not a check that the fetched chunk's content matches its label. A quote genuinely verbatim only in the paper's own abstract, never in any numbered chunk, is still located: the read-only chunk endpoint returns the paper's `abstract` alongside every chunk, and the locator tries it last, reporting `{"chunk_index": null, "section": "abstract"}` in `excerpt_locations` rather than leaving the row `passage_located: false` for evidence the verifier had already accepted; none of the rows on the current baseline needed this fallback. `source_passage` is the source text exactly as extracted from the PDF, original line breaks included, so a quote may wrap across a line break inside it: on the current baseline (all seven sections, 68 rows) all 215 `evidence_quotes` match their own `source_passage` under the same letters-and-digits fold the backend guard uses, but a plain, exact substring search finds only 39 of the 215 directly, because the other 176 need the fold -- most commonly a line break inside the quoted span the quote itself does not carry, but also curly-quote, dash and other punctuation differences -- not because the passage is unfaithful. On the current baseline all 68 rows are `passage_located: true` with no `unlocated_quotes` and no `fetch_error` (`unlocated_rows: 0`) |
| `writing_result_<n>.json`, `draft_content_<n>.json`, `claim_report_<n>.json` | one triple per extra section named in `demo/sections.json` (`<n>` = 2, 3, 4, ... -- the protocol section's own unindexed files above are section 1), each the exact same shape as the unindexed file it is named after, produced by the identical write-then-heal sequence steps 8 and 10 run on the protocol section. `draft_content_<n>.json` is written twice -- once right after the write job, again after the section's own standalone verify-and-heal action -- and the file left on disk is always the second (healed) write, exactly like the protocol section's own `draft_content.json` |
| `summary.json` | everything the summary table shows: step timings, flow counts, models reported, temperatures, prompt versions, token counts (including `writing.total_input_tokens`/`writing.total_output_tokens`, the folded totals across every DeepSeek call the write job made), `writing.loop_stats`, `writing.metrics` (`analysis.papers`/`papers_attempted`/`papers_failed`/`evidence_items`/`rejected_items`/`verbatim_rate`), `writing.citation_link_call` and `writing.citation_link_calls`, `writing.length` (including `final_words`) and `writing.loop_stats.survival_rate` -- both of these two are recounted from the saved draft after step 10's heal (`run_demo.py::_verify_claims`/`_count_body_words`, "Verdict distribution, before and after finalize" above), so they can differ, in either direction, from `writing_result.json`'s own pre-heal `final_words`/`survival_rate`, per-claim status, citation key and quote, citation audit, `verification.citation_coverage` (the same object as `claim_report.json`) and `verification.claim_text_in_draft` (whether every `claim_text` in the report occurs verbatim in `draft_content.json`); `sections` (one entry per extra section, `{title, task_id, draft_id, verify_task_id, writing, verification, delivered_evidence}`; each entry's own `writing` and `verification` are small summaries (`writing`: `section_type`, `loop_stats`, `final_words`; `verification`: 6 keys), not the fuller top-level `writing`/`verification` objects the protocol section carries -- per-section token counts and provenance live only in the run directory's own `writing_result_<n>.json` and `claim_report_<n>.json` -- `null` on a run given no `demo/sections.json`); `delivered_check` (`{violations, ok}`: every violation `demo/check_delivered.py` found across every section written and the merged `delivered_evidence.json`, each `{rule, section, detail, sentence}`, and `ok` (`true` when there are none) -- folded into the exit code by `main()`) |
| `credentials.json` | the demo account (local only; delete it when done) |

By default every run's own output directory is git-ignored (`demo/output/.gitignore`
ignores every entry except itself), so an ordinary run never adds `credentials.json` or the
multi-megabyte outputs to a commit. A promoted baseline's run directory is the deliberate
exception: it is force-added (`git add -f`) alongside `demo/expected/` when a baseline is
promoted, specifically so `demo/tests/test_expected_baseline.py` has a real
`draft_content.json` to check the promoted `claim_report.json` against (`demo/expected/`
itself never ships `draft_content.json` or `writing_result.json`, see the table above). The
ignore rule is what keeps `credentials.json` and every un-promoted run directory out of a
commit; it does not apply to the handful of promoted run directories the test suite reads.

`demo/expected/` holds the same files from the authors' run (date, model and cost are
recorded in `expected/summary.json`), minus `credentials.json`, `draft_content.json`
and `writing_result.json` -- it ships exactly five files: `summary.json`,
`screening_record.json`, `screening_record.csv`, `claim_report.json` and
`delivered_evidence.json`.
`expected/summary.json` also has its own `comparison` key stripped, since that key only
ever holds a *run's* comparison against whatever `demo/expected/` happened to be at the
time it ran, not a comparison of the baseline against itself; `demo/tests/test_expected_
baseline.py` pins that the shipped `summary.json` carries no `comparison` key, so a future
manual promotion cannot ship it by mistake again. Because
live retrieval drifts, a fresh run's comparison against this baseline uses these
tolerances:

- screening flow counts: within +-30 % of the expected value **or** within +-2 records,
  whichever is larger (so an expected value of 0, e.g. `unscreened` or the `stage1_*`
  counts with the venue filter off, tolerates 0-2);
- `citation_coverage.found`, `citation_coverage.linked` and
  `citation_coverage.by_source.mapping`: the same +-30 % / +-2 band as the screening flow
  counts;
- `citation_coverage.unresolved`: one-sided -- a fresh run's value may never be **above**
  the baseline's; a fall, of any size, is never reported as drift;
- `writing.total_calls`: exact, against whatever `demo/expected/summary.json` itself
  recorded for that run -- not a fixed number. The gated write loop can make anywhere from
  two DeepSeek calls (one generation call, one citation-link call, if every claim verifies
  on the first pass and the body never exceeds the hard maximum) up to a first-pass
  generation, an over-length regeneration, a full-section revision, one citation-link call
  per map build (the first-pass map, the revision's own map, and one more each time the
  coverage-gap closer's own unclassified-sentence retry fires) and one or more
  verification-agent batches, each folded into this one total; see `writing_result.json` ->
  `provenance` in "Expected outputs" above for what each call type is called;
- `verification.claim_text_in_draft`: must be complete on every run -- `check_claim_text_
  in_draft` checks `final_report.verifications` only, so every surviving `claim_text` is
  expected to occur verbatim in `draft_content.json`, not a tolerance band;
- claim statuses (`verified` / `unsupported`): exact;
- model name reported by the provider: exact.

Mismatches are printed as `DRIFT` lines and do not change the exit code unless `--strict`
is given. The claim check (exit code 2) applies whether or not `demo/expected/` exists.

`--replay-queries` together with the protocol's `publication_date_max` cutoff narrows how
much a run can drift from `demo/expected/`, but it does not make two runs identical.
OpenAlex's index keeps growing after the cutoff date, since a record can still be added or
have its own publication date corrected after the fact even though it describes work
published on or before that date, so the exact same query can return a slightly different
set of works on two different days. The relevance screener is also not perfectly
deterministic run to run even at temperature 0, so a handful of records near a criterion's
boundary can be screened differently. The tolerances above still apply to a replayed run;
replaying narrows drift, it does not eliminate it.

What `--replay-queries` does and does not fix, in general, from the mechanism itself rather
than a specific pair of runs: it makes retrieval fully reproducible (the same query issued
in the same round returns the same records, so `identified`, `duplicates_removed` and the
set of records that reach Stage 2 screening are byte-identical between a run and its own
replay), but it does not make the screener's own verdicts on that identical input
reproducible, since the relevance screener is not perfectly deterministic run to run even at
`temperature=0`: a handful of records near a criterion's boundary can be screened
differently between a run and its replay, moving the `included` count by a few records
either way without the underlying retrieval having changed at all. This has been checked at
the record level, over the whole run: a `--replay-queries` run reproduced all seventeen
replayed rounds and all 118 queries of the run it replayed identically, in order, with
`returned` and `new_unique` equal on every single query, the seventeen per-round `screened`
counts identical, and cumulative `new_unique` over those seventeen rounds equal on both
sides (914), while the screener's own verdicts on that byte-identical input still moved by
one candidate over the whole run (169 against 170). One consequence follows directly:
because the screener's own drift can add or remove an inclusion in a round that would
otherwise have been dry, a replayed run's own dry-round-patience counter can behave
differently from the run it replays, and it can then outlive the replayed round list and
fall back to live query generation for as many further rounds as the stop condition needs;
on the current baseline that did not happen, since the one-candidate drift fell in a round
that was not decisive and the run stopped at round 17, exactly where its own replayed round
list ends, with no live query generation at all. The tolerances above still apply, and
replaying narrows drift, it does not eliminate it.

## Approximate run time and cost

The numbers below are read from the `demo/expected/` baseline as it currently stands, a
cold run made 2026-09-16 against a freshly created database volume (`docker compose down -v`
before boot, so nothing was reused from an earlier session), with all six extra sections of
`demo/sections.json` written and verified after the protocol section: it has `loop_stats`, a
`length` record and the exit invariant on all seven sections, no saved draft contains a
`[NEEDS CITATION ...]` marker in any form the widened pattern in `_NEEDS_CITATION_RE`
matches (the bare form, or one with a colon and an explanatory clause, an em dash, or any
other text before the closing bracket), and every section's own `writing_result*.json` ->
`claim_report` shows `unsupported_count: 0`/`nuance_count: 0`/`abstract_only_count: 0` by
construction (see
"What the demo does" above). They are **not final**: `demo/expected/` is replaced whenever
the demo is re-run against a changed screener, verifier or writing prompt, live retrieval
scale varies run to run, and this table is rewritten from the new baseline each time. Treat
it as "what the shipped baseline currently says", not as a permanent number. Full detail,
including the record-level replay comparison ("What `--replay-queries` does and does not
fix" above) and the verdict distribution before and after finalize ("Verdict distribution,
before and after finalize" below), is in this document.

| Quantity | Current `demo/expected/` baseline |
|---|---|
| Date | 2026-09-16 |
| Screener prompt | v3, `sha256:9bc741046cd6` (frozen) |
| Screener second-pass judge prompt | `sha256:7aa4b622c72e` (frozen; one call per batch of candidates, made once per job over every record the first pass marked `INCLUDE` -- see "Screening second-pass judge" below) |
| Claim-verification prompt | v3, `sha256:49fcfbfaf2f6` (frozen; assertion-level verdicts plus attribution/numeric/quote-fidelity guards, quote relocation and a repair turn ahead of the guards) |
| Writing prompt | protocol section (assembled) `sha256:f87fbf277c5c`; `prompt_template_version` `sha256:beb00d2d0e42`, the one part of the assembled prompt that stays the same across sections. Each of the six extra sections carries its own assembled digest (`sha256:c43fcef67920`, `sha256:c3c9ad6a4400`, `sha256:2ba9aca46738`, `sha256:de5cfe5a54e9`, `sha256:c66ea9019ac6`, `sha256:020227f65e8f`) because the assembled prompt includes that section's own instructions, which differ section to section; not compared run to run -- the full hash also includes the AI-merged journal/writing-rules text, which varies run to run -- author-year citations are fixed in drafts, the project's chosen style applies only at export |
| Citation-link prompt | `sha256:5a8092ebd77b` (the DeepSeek call that builds the sentence-to-reference map, made once per map build -- an over-length regeneration makes none of its own, since it runs before the map is built, the coverage-gap closer's own unclassified-sentence retry makes one more whenever it fires, and a gate-triggered revision makes one more for its own map; not compared run to run; the protocol section made four such calls this run, 23 across all seven sections, distributed 4/4/3/3/3/2/4) |
| Model reported by the provider | `deepseek-flash` (configured as `deepseek-chat`; the second-pass judge configures `deepseek-flash` directly). System fingerprint `aeb56401ca74e127821c4f9126dcb669` on every call that reports one |
| Wall-clock time, whole demo (7 sections) | 1,197.0 s (about 19.95 minutes), from `summary.json` -> `steps[].elapsed_seconds`, which sum to 1,196.9 s, 0.1 s under the recorded elapsed time (the twelve step timings are already rounded to one decimal): Smart Search 506.9 s, full-text acquisition 50.2 s (12 of the 12 seed PDFs downloaded fresh, see the acquisition note below), the protocol section's own AI Write step 211.1 s, its standalone verify-and-heal action 22.2 s, all six extra sections combined (their own write-then-heal sequence, one after another) 400.3 s, building the merged delivered-evidence record 0.4 s, and the offline delivered-text check 0.1 s; the remaining 5.7 s is register demo user (0.1 s), create project (0.0 s), add seed papers (5.5 s), check claim citations (0.0 s) and append claim fixtures (0.1 s). Cumulatively from `run_started_at` 19:42:08 UTC these give Smart Search 19:42:08-19:50:35, full-text acquisition 19:50:41-19:51:31, the protocol section's AI Write step 19:51:31-19:55:02, its verify-and-heal action 19:55:02-19:55:24, the six extra sections 19:55:24-20:02:04, the delivered-evidence step and the delivered-text check 20:02:04-20:02:05 (`run_finished_at`) |
| Smart Search: records identified / included | 2,947 / 108 (17 rounds, `stop_reason: no_new_included`, `min_rounds: 3`, `dry_round_patience: 2`) |
| Smart Search: needs review / excluded, of 914 screened | 180 (`no_abstract` 103, `not_established` 61, `undecidable` 15, `unanchored_exclude` 1) / 626 |
| Screening second-pass judge | 169 candidates (every first-pass `INCLUDE`), 34 calls, 61 demotions to `needs_review` (reason `not_established`), 0 `unavailable`; 169 - 61 - 0 = 108, the final `included` count |
| DeepSeek tokens (query generation + screening first pass + screening second-pass judge + writing across all 7 sections + claim verification across all 7 sections) | 0 + 488,999 + 0 + 3,020,950 + 1,297,327 = 4,807,276 in; 0 + 63,050 + 0 + 205,222 + 74,226 = 342,498 out; 5,149,774 total. The second-pass judge's own 34 calls report zero tokens both ways (`calls_missing_usage: 34`, see "Screening second-pass judge" below), so its own share of this total is a floor, not a true zero. `writing.total_input_tokens`/`writing.total_output_tokens` on each section's own `writing_result*.json` already fold in every DeepSeek call that section's write job made -- generation, the citation-link call(s), the verification-gate batch(es), and, for the protocol section only, the 12 full-text grounding calls (one per acquired seed paper: grounding is per-paper, not per-section, so every extra section reuses the protocol section's own grounding and makes none of its own, `metrics.analysis.papers_attempted: 0` on all six); `screening.provenance.query_generator` (0 calls -- all seventeen rounds were replayed from the baseline, so no query was generated live) is added separately since it is not part of `screening.provenance.screener` |
| API cost, off-peak rate, at the rates published on 2026-09-16 (this run's own UTC timestamps, 19:42-20:02 on a Wednesday, fall outside both of DeepSeek's peak windows) | about US$0.93: 4,807,276 input tokens at US$0.15/1M plus 342,498 output tokens at US$0.6/1M (`api-docs.deepseek.com/quick_start/pricing`); the run directory is named on the host's local (UTC-4) clock, so `20260916-154140` is not a UTC timestamp; at peak rates this usage would cost about US$1.85 |

The confirmed DeepSeek rate for the model reported on this baseline is **US$0.15 per 1M
input tokens and US$0.6 per 1M output tokens off-peak, US$0.30/US$1.20 at peak**
(01:00-04:00 and 06:00-10:00 UTC, Monday-Friday). Re-check
`api-docs.deepseek.com/quick_start/pricing` before quoting a cost from a future run, since
the page itself says prices may change.

Everything except DeepSeek usage is free (OpenAlex, Crossref, Unpaywall).

On this baseline, Smart Search replayed all seventeen rounds of the recorded queries
verbatim, identical in retrieval to the run they were replayed from, and no query was
generated live at any point; the run stopped exactly where its own replay list ends, with
rounds 16 and 17 both dry (`min_rounds: 3`, `dry_round_patience: 2`): 2,947 records
identified, 2,033 duplicates removed, 914 reaching Stage 2, of which 108 were included, 626
excluded and 180 flagged `needs_review` (103 for carrying no abstract, 61 `not_established`,
15 `undecidable`, 1 `unanchored_exclude`), stopping when two consecutive rounds produced no
newly included paper (`stop_reason: no_new_included`): the ordinary, successful stop
condition, not an error. None of the 118 queries this run issued (all 118 replayed) was
rejected by OpenAlex -- `screening_record.json` -> `retrieval_failures` is empty on this
baseline (the mechanism that catches a malformed query or a transient provider outage is
still real; it simply did not fire here). Live retrieval against the public OpenAlex index
lands at a different scale run to run, in either direction, so the flow counts above are
read from this run's own `screening_record.json` rather than treated as a fixed number a
reviewer should expect (see "Replaying the shipped search" above for how `--replay-queries`
narrows, but does not eliminate, that drift). The rest of the pipeline (the 12 fixed seed
papers, full-text acquisition, AI writing across all seven sections and claim verification)
completed normally on real DeepSeek calls, against the frozen screener prompt
(`sha256:9bc741046cd6`) and the frozen claim-verification prompt (`sha256:49fcfbfaf2f6`).

**Screening second-pass judge.** After the first-pass screener marks a record `INCLUDE`,
one further, once-per-job stage (`relevance_screener_second_pass_v2`) reviews every such
candidate together and can demote a candidate to `needs_review` (reason `not_established`)
when the population, outcome or study-type anchor it relied on does not hold up under a
second look; a candidate it does not demote stays `included`. On this baseline the judge
reviewed 169 candidates in 34 calls and demoted 61, leaving 108 `included` (169 - 61 = 108,
matching the flow's own `included` count exactly), with 0 reported `unavailable`. Each of
the 169 reviewed records carries its own decision in the screening-record CSV's four
`second_pass*` columns (`second_pass`, `second_pass_input_tokens`,
`second_pass_output_tokens`, `second_pass_latency_s`); the other 745 of the 914 screened
records carry these four columns blank, since the judge only ever reviews a first-pass
`INCLUDE`. The provider returned no usage for any of the judge's 34 calls (`RequestUsage()`
all-zero when the raw response's own `usage` field is `None`), so both the per-call
`second_pass_input_tokens`/`second_pass_output_tokens` columns and this stage's own share of
the token and cost totals above are a floor, not a true zero; per-call latencies still
recorded real wall-clock time for all 34 calls (1.333 s-124.941 s, summing to 1,036.1 s of
work overlapped under concurrency into a 194.14 s stage wall time), confirming the calls
were real even though their own token counts were not returned.

Claim verification consumes the sentence-to-reference map the writer returns alongside the
generated text, falling back to the author-year regexes and then to an audit-key fallback
for a sentence with no surviving link. `claim_report.json` -> `citation_coverage` (and the
identical object under `summary.json` -> `verification`) reports how that resolution went
on the protocol section of this baseline: 11 citation occurrences found across the whole
draft (`found`), of which all 11 resolved to a citation key (`linked` 11, `unresolved` 0)
and 12 reached the verifier as claims (`sent_to_verifier` 12, since the two appended
fixtures each become a claim of their own). `by_source` counts the 11 occurrences directly:
all 11 are recorded under `by_source.mapping` (0 under the numbered-citation resolver and 0
under the author-year fallback) -- see "Why step 6 matters" above for the mechanism by
which a fixture paragraph can land under `mapping` even though the writer's own link-map
call never saw it. `writing_result.json` -> `citation_audit` reports its own `total: 6`,
the count of DISTINCT citations recognised across the generated section alone, before the
two appended fixture paragraphs exist and before step 10's own repair step writes a link
back onto anything (0 unmatched, `needs_citation_flags: 2`). The gap between the 11
occurrences above and this 6 is occurrences counted against distinct citations, plus the two
fixture paragraphs' own citation; two of the section's own cited sentences were removed at
the write stage this run (`sentences_removed_unverified: 2`), and the appended
`unsupported-1` fixture was removed at the heal stage. `demo/tests/test_expected_baseline.py`
pins `unresolved == 0` and `found == linked` for the baseline currently shipped.

**Verdict distribution, before and after finalize, as they were on this run, not adjusted
for what the fixtures were designed to show.** The standalone verify-and-heal action (step
10) sent 12 claims to the verifier: the 10 cited claims of the write job's own
already-finalized section, plus the 2 appended fixtures. Before finalize (the report's
top-level counts): 11 `verified`, 0 `needs_nuance`, 1 `unsupported`, 0 `abstract_only`
(every paper any of these 12 claims cites has full text, `full_text_coverage: 1.0`), 0
`error`, `guarded_count: 0` (the frozen `quote_not_verbatim` guard changed no status this
run), `contradicted_count: 1`. Both fixtures resolve cleanly on this baseline, with no
guard intervention: `supported-1` verifies after one repair turn (`model_status: verified`,
`machine_reasons: []`, `guard_demotion: false`) -- pass 1 quoted a non-verbatim paraphrase
(`machine_reasons: ["quote_not_verbatim"]`) and pass 2 returned the source's own verbatim
wording, with no quote relocation needed. `unsupported-1` was correctly caught
(`model_status: unsupported`, machine reason
`assertion_status_inconsistent`, one pass, `contradicted_count: 1` counting exactly this
claim). After finalize (`finalize_stats`): 1 sentence removed, under
`sentences_removed_unverified` (0 under every other reason) -- `unsupported-1` only --
leaving `final_report` with exactly 11 `verified` rows, ten of them the section's own
real citations and one the surviving `supported-1` fixture; `unsupported-1` does not
survive into the delivered text on this baseline (`final_status: "not_in_report"`), exactly
as designed. Separately, inside the write job's own gated loop (`writing_result.json` ->
`loop_stats`, a pre-fixture measurement over the section alone, taken before step 9 appends
the two fixture paragraphs and before step 10's own heal can remove anything further):
`first_pass_verified_rate: 0.600` (six of the section's ten originally cited claims
verified on the loop's own first pass), `loop_revised: true` (the whole section was
rewritten once because some first-pass verifications did not verify), `length_regenerated:
false` (the revised body, at 474 words, exceeded the 400-word target, `over_target: true`,
but stayed under the 500-word hard maximum, which is why no regeneration fired),
`survival_rate: 0.740` (`final_words / target_words`, 296/400 -- under 1.0 because
finalize's own removals shortened the section, not because it ran long),
`sentences_removed_uncited_finding: 2`, `sentences_removed_unverified: 2` and
`sentences_removed_dangling: 0` (0 under `sentences_removed_no_full_text` and
`citations_dropped_no_full_text`) -- the loop's own finalize step, ahead of step 10, removed
four sentences in all: two uncited "finding" sentences that carried no citation at all and
two cited sentences that still did not verify after the revision; nothing was dropped as
dangling. A fifth sentence was kept with a trailing, uncited clause cut away
(`sentences_residue_excised: 1`), leaving the section that step 10 re-verifies.
`needs_citation_markers_removed: 0` on this run means no sentence reached the saved draft
with a stripped `[NEEDS CITATION]` marker still standing; `writing_result.json` ->
`citation_audit.needs_citation_flags` is 2 over the pre-finalize text -- two sentences
carried the marker before finalize and both were removed outright, counted under
`sentences_removed_uncited_finding` above, rather than kept with the marker stripped.
Markers were stripped and the sentence kept on four other sections this run instead (one
each on sections 2, 3 and 7 and five on section 5, numbering the protocol section 1),
eight in all across the run.

`summary.json` -> `writing.loop_stats.survival_rate` and `writing.length.final_words` record a
different, later pair on this same section: `0.8525` and 341 (341/400), not the `0.740` and 296
above. `run_demo.py`'s own `_verify_claims` overwrites both fields, after step 10's heal, with
a fresh count of the words in the draft actually saved to disk (`_count_body_words`) -- so they
cover the section as delivered, fixture paragraph included, rather than the write job's own
pre-fixture figure. On this section the saved draft (341 words) is longer than the pre-fixture
measurement (296 words) because the surviving `supported-1` fixture paragraph (45 words) is
appended after `writing_result.json` is written; on a different section the same recount can
instead come out lower than `writing_result.json`'s own figure, when step 10 removes a sentence
the write job's own finalize pass had kept, though no section of this baseline shows that
direction: the heal stage removed nothing further from any of the six extra sections, so each
one's `summary.json` word count equals its own `writing_result*.json` figure exactly. Read
`writing_result.json`'s `loop_stats`/`length` as the write job's own pre-fixture, pre-heal
measurement, and `summary.json`'s copy of the same two field names as the post-heal recount of
the section actually delivered; the two can differ in either direction and neither is wrong.

Full-text acquisition on this run acquired all 12 of the 12 seed PDFs fresh, against the
empty database volume this cold run started from, with no fallback to abstract-only:
`fulltext.acquired` 12, `fulltext.abstract_only` 0, `already_acquired` 0, so the one-time
acquisition retry described above ("Retrying full-text acquisition") was not needed. The 12
acquired papers came from all six publisher hosts named in `SOURCES.md` (`www.cambridge.org`
x4, `www.jowr.org` x3, `ueaeprints.uea.ac.uk` x2, `hal.science` x1, `lirias.kuleuven.be` x1,
`link.springer.com` x1), with 14 reference chunks dropped across them
(`reference_chunks_dropped`). The write job's own grounding step ran a fresh analysis call
for each of the 12 acquired papers: `writing_result.json` -> `metrics.analysis` reports
`papers: 12`, `papers_attempted: 12`, `papers_failed: 0`, `evidence_items: 120`,
`rejected_items: 9`, `verbatim_rate: 0.930` (`analysis_calls` has 12 entries in
`provenance`, one per acquired paper; `papers_used: 12`, the size of the project's whole
library).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `docker compose up` says `env file .env not found` | Only with an older `docker-compose.yml` (v1.1.0 marks the file optional). Create it: `cp .env.example .env` and set `DEEPSEEK_API_KEY`. |
| `POST /auth/register returned HTTP 500` and `docker compose logs backend` shows `relation "users" does not exist` | The database schema was not created: the backend was started without its `alembic upgrade head` step (older compose file or a custom command). Run `docker compose exec backend alembic upgrade head` and re-run the demo. |
| `python: command not found` | Use `python3 demo/run_demo.py`. |
| `could not reach http://localhost:8000/api/v1` | The stack is not up or still starting (the first start builds images): `docker compose ps`, `docker compose logs backend`. |
| `POST /auth/register returned HTTP 422` | Backend rejected the generated e-mail/password (should not happen); re-run. |
| Smart Search `HTTP 409: Smart search already in progress` | A previous run in the same project is still running; the demo always creates a fresh project, so this indicates a stuck job - restart the backend. |
| Smart Search completes with `stop_reason: retrieval_failed` and `identified = 0` | Every OpenAlex query failed; `screening_record.json` -> `retrieval_failures` and the task message carry the provider's reason. Seen by the authors on 2026-09-03: HTTP 429 `Insufficient budget ... Resets at midnight UTC` - OpenAlex's free daily budget for the calling IP was exhausted. Wait for the reset (00:00 UTC) or set `OPENALEX_API_KEY` (a funded OpenAlex account) in `.env`, restart the backend (`docker compose restart backend`) and re-run. |
| Smart Search stops with `stop_reason: time_limit` or `unscreened > 0` | OpenAlex rate limiting (HTTP 429) or DeepSeek quota; the record lists the affected papers as `unscreened` with the error. Wait and re-run. |
| `DEMO FAILED: Protocol claims cannot be verified: ... no authors/year resolved` | Neither OpenAlex nor Crossref answered for the claim's seed paper (network block or a long 429 quota block on OpenAlex). Set `OPENALEX_EMAIL`/`OPENALEX_API_KEY` or wait, then re-run. |
| `DEMO FAILED: Protocol claims cannot be verified: ... citation key ... differs` | Only after editing `protocol.json` or `seed_dois.json`: the cited surname/year must equal the last-name token of the seed paper's first author and its year. |
| `fulltext.acquired = 0`, `fulltext.already_acquired = 12` | The seed papers were already acquired by an earlier demo run in this database: full text is stored per paper, not per project, and the backend skips papers whose status is already `acquired`. Not an error; the verification step uses the stored text. On the current baseline (a cold run against a freshly reset volume) this did not occur: all 12 of the 12 PDFs were downloaded fresh, taking 50.2 s of the 1,197.0 s total. For a clean re-acquisition run `docker compose down -v` first. |
| `fulltext.abstract_only > 0` | Unpaywall or the publisher host refused or changed the PDF link since 2026-09-02; a claim on that paper would be reported as `no_full_text` instead of `verified`. Not seen on the current baseline (all 12 of 12 seed PDFs acquired fresh; see "Approximate run time and cost" above), but the mechanism is real: a transient host or Unpaywall miss on any seed's own host would raise this count without being a defect. Paste the PDF text through the UI (Library -> paper -> Paste full text) and re-run verification, or accept the deviation. |
| Exit code 2, `claim.*.status = not_in_report` | The verifier did not pick up the appended claim although its citation passed step 6: check that the seed paper is in the library with authors and that the draft was saved (`draft_content.json`). |
| Exit code 2, `claim.*.status = nuance` or the opposite of expected | Model drift: the verifier is deterministic in settings (temperature 0) but not across provider model versions; `summary.json` records the model reported. |
| Exit code 2, `DEMO CLAIM MISMATCH: no fixture expecting "verified" survived into the final report, and none is a recorded guard demotion` | A genuine loss of the `verified` fixture, not the accepted "printed line reads `GUARD DEMOTION` instead of `OK`" case described under "What the demo does" above (a fixture expecting `verified` coming back `needs_nuance` because the frozen `quote_not_verbatim` guard alone demoted it -- not a failure, and not what happened on the current baseline, where `supported-1` verifies directly with no guard demotion at all): check `claim_report.json`'s own raw verification for `supported-1` -- if its `status` disagrees with `model_status`/`machine_reasons` in some OTHER shape than `needs_nuance`/`verified`/`["quote_not_verbatim"]`, or if it never verified at all, this is model drift or a real regression, not the accepted paraphrase-drift case. |
| `DEMO FAILED: ... task ... failed: ...` | The backend task raised; the message is the backend error. `docker compose logs backend` shows the traceback. |
| `DEMO FAILED: GET /tasks/... returned HTTP 401: Invalid token` | The access token issued at registration lives for `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` (15 by default) and the write job's grounding/verify/revise passes can make a single run take longer than that. The script refreshes the token once, automatically, on any 401 and retries the failed call, so this should no longer stop a run; if it still does, the refresh call itself failed (`docker compose logs backend`, or the refresh token was somehow lost) -- restart the demo. |
| Timeouts | Increase `--timeout` (every task except Smart Search) or `--search-timeout` (Smart Search's own, larger budget); a timeout at the default means the backend itself is stuck or a very slow connection. |
| Empty database after `docker compose down -v` | Expected: the demo account and project live in the local Postgres volume. Just run the demo again. |

## Tests

The runner has an offline test suite (fake API via `httpx.MockTransport`; no network,
no database, no LLM), plus contract tests that apply the backend's citation regex to the
protocol claims and check the seed audit trail:

```bash
python -m pytest demo/tests -q
python -m ruff check demo --config evaluation/ruff.toml
```

Seed re-verification (network; needs `UNPAYWALL_EMAIL` in the environment or `.env`):

```bash
python demo/tools/verify_pdf.py demo/tools/candidates_filtered.json demo/tools/verified.json --browser-ua
python demo/tools/build_sources.py     # regenerates SOURCES.md from the audit files
```
