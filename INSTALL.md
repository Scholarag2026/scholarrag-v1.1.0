# ScholarRAG - Installation

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.12 | backend |
| Node.js | >= 22.6 | frontend |
| Docker and Docker Compose | >= 2.24 | for the one-command path |
| PostgreSQL | 17 | only if not using Docker |
| DeepSeek API key | - | https://platform.deepseek.com |

OpenAlex, Crossref, and Unpaywall need no key, only a contact email address (see
`.env.example`).

## 1. Configure

```bash
cp .env.example .env
```

Four values matter most:

- `DEEPSEEK_API_KEY` (mandatory; nothing that calls the LLM works without it)
- `JWT_SECRET_KEY` (mandatory outside `APP_ENV=development|test|local`)
- `OPENALEX_EMAIL` (joins the OpenAlex polite pool)
- `UNPAYWALL_EMAIL` (required by the Unpaywall API)

The stack boots without a `.env` file, because `env_file` is `required: false` in
both compose files, but every step that calls the LLM then fails.

## 2. Run with Docker

Development stack:

```bash
docker compose up --build -d
```

The backend service runs `alembic upgrade head` before starting uvicorn, so there is
no manual migration step.

Production-style stack:

Before starting it, set `APP_ENV=production` in `.env` and replace the
placeholder secrets: `JWT_SECRET_KEY=change-me` and the default
`POSTGRES_PASSWORD=secret`. Generate a strong value with, for example:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

The backend refuses to start when `APP_ENV` is outside
`development|test|local` and `JWT_SECRET_KEY` is empty or still `change-me`.
Leaving `APP_ENV=development` in `.env`, which `docker-compose.prod.yml` does
not set or override, downgrades that refusal to a log warning, and every
token stays signed with the publicly documented placeholder key.

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

`NEXT_PUBLIC_API_URL` is a frontend build argument, not a runtime environment
variable, so it must be set before the frontend image is built, not after.

### Port collisions

If the default ports are already in use, override them on the command line or in
your shell, not in `.env`: `POSTGRES_PORT`, `BACKEND_PORT`, `FRONTEND_PORT`.
Browser-facing origins are separate variables,
`PUBLIC_FRONTEND_ORIGIN`, `PUBLIC_BACKEND_ORIGIN`, `PUBLIC_API_URL`, deliberately not
the `.env` names `FRONTEND_URL`/`BACKEND_URL`/`NEXT_PUBLIC_API_URL`: a `.env` file
that set those would pin the browser-facing origins to the default ports regardless
of which host ports the containers are actually bound to.

Worked example, running two stacks side by side under different ports and a project
name:

```bash
POSTGRES_PORT=55433 BACKEND_PORT=58000 FRONTEND_PORT=53000 docker compose -p scholarrag up -d
```

Measured on a Windows laptop: first build takes about 6 minutes. Neither compose file
defines a healthcheck for the backend or frontend service, only for postgres, so
`docker compose ps` shows the backend and frontend as running rather than healthy; confirm the
backend is actually up by requesting its `/health` endpoint, which starts responding about
7 minutes after the build starts.

## 3. Run without Docker

`backend/app/config.py` loads `.env` with `env_file=".env"`, and pydantic-settings
resolves a relative `env_file` against the process working directory, not against
the location of `backend/app/`. Copy `.env` into `backend/` before starting uvicorn
from inside that directory, otherwise every setting falls back to its code default:
`DEEPSEEK_API_KEY=""`, `OPENALEX_EMAIL=""`, `UNPAYWALL_EMAIL=""`, and
`JWT_SECRET_KEY="change-me"`. Every LLM feature then fails with no message that
points at the missing file. `STORAGE_PATH=./data/uploads` is also resolved against
the working directory, so keeping the copy under `backend/` also keeps uploads
under `backend/data/uploads` instead of `<repo>/data/uploads`.

Create the PostgreSQL role first, then create the database owned by it; the
defaults below match `.env.example`. PostgreSQL 15 and later grant the
`CREATE` privilege on schema `public` only to the database owner, so creating
the database before the role, or with a different owner, leaves Alembic
unable to create any table:

```bash
psql -c "CREATE ROLE deepresearch LOGIN PASSWORD 'secret'"
createdb -O deepresearch deepresearch    # or: psql -c "CREATE DATABASE deepresearch OWNER deepresearch"
```

Then:

```bash
cp .env backend/.env
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt        # or: pip install -r requirements.lock.txt
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

Alembic reads an exported `DATABASE_URL` if one is set in the shell; otherwise it
uses the default in `backend/alembic.ini`, which matches the values above.

```bash
cd frontend
npm ci
npm run dev
```

## 4. Optional: import the Web of Science journal lists

Place these four files, under these exact names, in `backend/data/wos/`:

- `Science Citation Index Expanded (SCIE).csv`
- `Social Sciences Citation Index (SSCI).csv`
- `Arts & Humanities Citation Index (AHCI).csv`
- `Emerging Sources Citation Index (ESCI).csv`

Restart the backend. Everything works without them; see README "Data notice".

## Running the test suite

The backend test suite creates and drops every application table on the database it
connects to. It reads that database from `TEST_DATABASE_URL` only. Before any test
runs, `backend/tests/conftest.py` refuses to start when `TEST_DATABASE_URL` resolves
to the same server, port, and database as `DATABASE_URL`, or when the database name
does not end in `_test`. Set `SCHOLARRAG_ALLOW_DESTRUCTIVE_TESTS=1` only if you
intend to lose that data.

The `deepresearch` role must already exist (see "Run without Docker" above);
the test database must be owned by it, for the same schema-ownership reason.

```bash
createdb -O deepresearch deepresearch_test    # or: psql -c "CREATE DATABASE deepresearch_test OWNER deepresearch"
cd backend
TEST_DATABASE_URL=postgresql+asyncpg://deepresearch:secret@localhost:5432/deepresearch_test \
DEEPSEEK_API_KEY=test-key-not-real \
python -m pytest tests -q
```

Recovery for a database wiped by the v1.0.0 test suite: v1.0.0 defaulted to
`DATABASE_URL` and dropped every table, but not the `alembic_version` bookkeeping
table, so Alembic then reports the schema as up to date while every application
table is gone. Restore it with:

```bash
cd backend
alembic stamp base
alembic upgrade head
```

Application rows are not recoverable this way; restore from a database backup if you
have one.

## Dependency versions

`backend/requirements.txt` pins all 28 direct dependencies to exact versions.
`backend/requirements.lock.txt` is the full transitive dependency freeze, generated
from a clean `python:3.12-slim` container on 2026-09-05:

```bash
docker run --rm -v "<repo>/backend:/w" python:3.12-slim \
  sh -c "pip install -q -r /w/requirements.txt && pip freeze"
```

## Troubleshooting

| Symptom | Cause and remedy |
|---|---|
| `RuntimeError: JWT_SECRET_KEY is unset ...` | `APP_ENV` is not `development`, `test`, or `local`, and `JWT_SECRET_KEY` is unset, `change-me`, or shorter than the 32-character production minimum (`MIN_PRODUCTION_JWT_SECRET_LENGTH` in `app/config.py`). Set a strong `JWT_SECRET_KEY` of at least 32 characters, or set `APP_ENV=development`. |
| OpenAlex requests return HTTP 429 | The anonymous per-IP daily budget is exhausted. Set `OPENALEX_EMAIL` to join the polite pool, and set `OPENALEX_API_KEY` if you have an OpenAlex account. |
| DeepSeek quota or key errors | Screening calls fail and the affected papers are recorded as `unscreened`; a drafting job fails outright. Check `DEEPSEEK_API_KEY` and your account balance. |
| `relation "users" does not exist` | The backend started against an older compose file that never ran migrations. Run `alembic upgrade head` in the backend container or venv. |
| Port already in use | Set `POSTGRES_PORT`, `BACKEND_PORT`, or `FRONTEND_PORT` (see "Port collisions" above). |
| Web of Science badges missing | No journal list is imported. See "Optional: import the Web of Science journal lists" above; ScholarRAG still runs without them. |
| Application database wiped | See "Running the test suite" above for the recovery recipe. |

See `demo/README.md` for the demo-specific troubleshooting table.
