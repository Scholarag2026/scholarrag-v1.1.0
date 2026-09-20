# ScholarRAG - Privacy Notice

## Scope

ScholarRAG is self-hosted software. When you run your own instance, you are the
data controller for the data it stores. This notice also describes the authors'
hosted instance at `https://dr-app.zeabur.app`; see "The authors' hosted instance"
below for the conditions that apply there specifically.

## What is sent to the model provider

ScholarRAG sends text to DeepSeek (`https://api.deepseek.com/v1/chat/completions`,
configured model `deepseek-flash`, the provider's current name for the same deployment
previously aliased `deepseek-chat`).
Every feature under `backend/app/agents/` that uses the model sends some project
content; the table below states the features that send the most, including the two
that send content you upload yourself:

| Feature | What is sent |
|---|---|
| Relevance screening | The research topic, the inclusion and exclusion criteria, then per paper the title and the abstract truncated to 10,000 characters (`ABSTRACT_CHAR_LIMIT`) |
| Claim verification | The claim text and up to 40 chunks of up to 2000 characters each from the full text of the cited paper |
| Drafting | The section prompt, the selected papers' metadata, abstracts and stored analyses, and any journal guidelines you supply |
| Paper chat | The question, the paper's text chunks, and the stored chat history |
| Deep paper analysis | The paper's title, authors, year, and complete extracted full text, up to the 200 KB stored limit. When a paper has no extracted full text, ScholarRAG builds a fallback analysis from the abstract locally and makes no call to DeepSeek |
| Qualitative coding | The text segments from the dataset you upload (the columns marked as text), the project description used as the research question, and any existing codebook |
| Quantitative analysis | The dataset's column names, types, roles, and missing and unique counts, then the computed descriptive statistics (numeric, categorical, and normality summaries). For a categorical column this includes two fields taken from the data itself: the modal value (`mode`) and up to five category labels (`frequencies`). Whole rows are never sent |
| Query generation and expansion | The research question, the project description, the titles of papers already retrieved, and, when expanding a search, their abstracts truncated to 200 characters |

Other agents under `backend/app/agents/` (quality scoring, gap analysis, research
design, paper selection, metadata extraction, journal-guidelines lookup, text
refinement, compliance-rules merging, data-collection protocol generation,
field-foundations identification, scope refinement, writing-rules merging, and
query generation) send comparable project content, such as paper metadata,
abstracts, and stored analyses, to produce their outputs.

## DeepSeek's retention and training-use position

DeepSeek's privacy policy (`https://cdn.deepseek.com/policies/en-US/deepseek-privacy-policy.html`,
dated "Last Update: Feb 10, 2026" on the page itself, read 2026-09-14) states that it keeps
personal data, including the text submitted as model input, for as long as an account exists and
for as long as needed for legal or legitimate-business purposes, with no fixed deletion date given.
It also states that user input may be used to train and improve DeepSeek's models, and that account
holders may have the right, depending on where they live and applicable law, to opt out of that
training use, exercised by emailing DeepSeek at `privacy@deepseek.com`. ScholarRAG calls the
DeepSeek API under the operator's own `DEEPSEEK_API_KEY`, not a per-user DeepSeek account, so that
opt-out is available to the instance operator, not to individual ScholarRAG users. The same policy
states that it does not cover personal data collected by a downstream application built on
DeepSeek's open platform, naming the developer of that application as the controller for that data
instead; this section, and this document as a whole, is ScholarRAG's own disclosure for that
reason. The registration form (`frontend/src/app/(auth)/register/page.tsx`) shows a data-processing
notice linking to this document before the account is created; consent is not captured by a
separate step.

## What is sent to the scholarly services

ScholarRAG queries OpenAlex, Crossref, Unpaywall, and publisher hosts. What each
receives:

- **OpenAlex**: the search strings generated from your research question and
  project description, plus your configured contact address as a `mailto`
  parameter and in the User-Agent header when `OPENALEX_EMAIL` is set.
- **Crossref** and **Unpaywall**: the DOIs of the papers in your project, plus
  your configured contact address in the User-Agent header (Crossref) or as a
  `mailto` parameter (Unpaywall, when `UNPAYWALL_EMAIL` is set).
- **Publisher hosts**: the PDF URL, when ScholarRAG downloads a paper's open-access
  full text.

No uploaded files, draft text, or chat messages are sent to these services.

## What is stored

ScholarRAG stores application data in 18 PostgreSQL tables: `users`, `projects`,
`project_papers`, `papers`, `paper_analyses`, `drafts`, `draft_versions`,
`chat_messages`, `analysis_jobs`, `datasets`, `codebooks`, `coding_sessions`,
`citation_edges`, `teams`, `team_members`, `team_projects`, `wos_journals`,
`evidence` (one accepted, verbatim evidence quote from one paper's full text, used by claim
verification).
Passwords are stored as Argon2 hashes. Draft text, chat messages, and job results
(including screening records and provenance) are stored in clear text in
PostgreSQL. Uploaded files are stored on disk at `STORAGE_PATH`, which defaults to
`./data/uploads` relative to the backend's working directory.

## Shared paper records

`papers` rows are global to a ScholarRAG instance and are deduplicated by DOI.
Extracted full text is stored on the paper record. On a shared instance, full text
that one user acquires or pastes in for a given DOI is reused when another user adds
a paper with the same DOI. If you need to keep material confidential, run a
single-tenant deployment.

## Retention and deletion

Projects, drafts, papers, datasets, and chat history each have a self-service delete
route in the application. There is no self-service account deletion route; deleting
a user account is an administrator action. On the authors' hosted instance, account
deletion therefore requires an operator request. Data is kept until deleted; there
is no automatic expiry.

## The authors' hosted instance

`https://dr-app.zeabur.app` is a hosted deployment operated by the authors, not a
production service. There is no service-level agreement and no backup guarantee.
Data on this instance may be deleted without notice. Operators can read stored
content when maintaining the service. Do not upload confidential or personal data
to this instance. Self-hosting your own instance is the privacy-preserving option if
this matters to you.

## Model identity and provenance

Every evaluated LLM call is recorded with an `LLMCallProvenance` entry: the agent
that made the call, the configured model, the model the provider reported back, the
provider name, the provider's response id, the system fingerprint when the provider
returns one, the temperature used, a prompt version hash, input and output token
counts, and a timestamp. This record surfaces in the screening record export, the
claim-verification report, and the writing job result. Screening and claim
verification run at temperature 0; drafting runs at temperature 0.7.

## Turning features off

An empty `UNPAYWALL_EMAIL` disables the Unpaywall lookup. An empty `OPENALEX_EMAIL`
keeps your address out of OpenAlex requests. No LLM feature runs without
`DEEPSEEK_API_KEY` configured.
