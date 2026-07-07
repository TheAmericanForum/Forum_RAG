# Forum RAG — Agentic RAG over policy-discussion transcripts

Ingests facilitated community-conversation transcripts (verbose AssemblyAI JSON) from
Google Drive, indexes them in a vector store, and answers questions about the **proposals
and trade-offs** raised across the policy areas — with **citations to the exact phrase,
speaker, and timestamp**, each with a permanent, shareable link to the full passage.
Multi-tenant: one codebase serves several branded deployments (e.g. separate state
forums). Built to deploy on **Heroku** (nothing persists to local disk).

## How it works

- **Ingestion** (`ingest_data.py`, run by Heroku Scheduler): Drive → parse turns → chunk →
  classify policy area(s) → embed (OpenAI) → upsert to Qdrant. Incremental: a file is skipped
  when its md5 already matches what's stored in Qdrant. No local state.
- **Query** (`app.py` web service + `main.py` CLI): a retrieval planner plans and runs
  searches via a `search_transcripts` tool (multiple searches per question), then a
  second model writes a cited answer using Anthropic's native **Citations** over the
  retrieved passages. Retrieval and synthesis are split across two models (see table
  below) so the fast planning step doesn't wait on the more expensive answer model.
- **Citations**: every chunk carries a stable `citation_id` (a UUID, assigned once and
  carried forward across re-ingestion). Each footnote links to `/source/<citation_id>`,
  a permalink to the full passage — safe to share even after the transcript is
  re-ingested or re-chunked. `backfill_citation_ids.py` is a one-off migration for data
  ingested before citation_id existed; `report_classifications.py` audits how each
  transcript was classified.

| Component | Tech |
|---|---|
| Retrieval planner (plans/runs searches) | Claude Sonnet 4.6 |
| Answer synthesis (writes the cited answer) | Claude Opus 4.8 |
| Policy-area classifier | Claude Haiku 4.5 (structured output, multi-label) |
| Embeddings | OpenAI `text-embedding-3-large` (3072-dim) |
| Vector store | Qdrant (hosted; embedded mode for local dev only) |
| Web | FastAPI + minimal UI |

## Multi-tenant branding

One codebase, one Heroku app per tenant. `TENANT` (env var, defaults to `sc`) selects
`branding/<tenant>/`, which holds:

| File | Purpose |
|---|---|
| `brand.yaml` | App name, tagline, chat-avatar initials |
| `policy_areas.yaml` | This tenant's policy areas — overrides config.yaml's list when present |
| `theme.css` | Tenant color/style overrides, served at `/brand/theme.css` |
| `logo-mark.svg`, `favicon.svg` | Tenant logo/favicon, served at `/brand/...` |

Set `QDRANT_COLLECTION` per tenant too, so deployments don't share one Qdrant collection.

## Access control

Sign-in is Google SSO, gated by an email allowlist:

| Var | Purpose |
|---|---|
| `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET` | OAuth2 "Web application" credentials (Google Cloud Console), redirect URI `<APP_URL>/auth/callback` |
| `SESSION_SECRET_KEY` | Signs the login session cookie |
| `ALLOWED_EMAILS` | Comma-separated allowed emails |
| `ALLOWED_EMAILS_FILE_ID` | Optional: a Drive file ID for a shared, per-tenant allowlist (see `allowed_emails.txt`) — re-fetched at most every 5 minutes, no redeploy needed to update it |
| `CONTACT_EMAIL` | Optional "request access" mailto link shown to signed-in-but-unauthorized users |

## Configuration

Non-secret settings live in [config.yaml](config.yaml) — **fill in the 4 `policy_areas`**
(names + one-line descriptions) as a fallback for any tenant that ships no
`branding/<tenant>/policy_areas.yaml` of its own; until you do, chunks are labeled
`other` and area filtering is a no-op.

Secrets come from env vars (local `.env`, see [.env.example](.env.example)) / Heroku config vars:

| Var | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude (retrieval planner, synthesis, classifier) |
| `OPENAI_API_KEY` | embeddings |
| `QDRANT_URL`, `QDRANT_API_KEY` | hosted Qdrant (required in production) |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | service-account key JSON (full contents, one line) |
| `DRIVE_FOLDER_IDS` | comma-separated Drive folder IDs to ingest recursively |

> If `QDRANT_URL` is unset, an embedded on-disk Qdrant (`./.qdrant`) is used — **local dev only**;
> it does **not** persist on Heroku's ephemeral filesystem.

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in keys (QDRANT_URL optional for local embedded mode)

# Ingest the bundled sample (no Drive needed):
python ingest_data.py --source local:./data

# Ask via CLI:
python main.py --show-progress "What trade-offs were raised about <area>?"

# Or run the web UI:
      # open http://127.0.0.1:5050
```

`--source drive` ingests from Google Drive instead (needs `GOOGLE_SERVICE_ACCOUNT_JSON` +
`DRIVE_FOLDER_IDS`, and the folders shared with the service-account email).

## Deploy to Heroku

```bash
heroku create
# Provision hosted Qdrant (Qdrant Cloud free tier) and set its URL + key, plus the rest:
heroku config:set ANTHROPIC_API_KEY=... OPENAI_API_KEY=... \
  QDRANT_URL=... QDRANT_API_KEY=... \
  DRIVE_FOLDER_IDS=... GOOGLE_SERVICE_ACCOUNT_JSON="$(cat service_account.json)" \
  TENANT=sc GOOGLE_OAUTH_CLIENT_ID=... GOOGLE_OAUTH_CLIENT_SECRET=... \
  SESSION_SECRET_KEY=... ALLOWED_EMAILS=...
git push heroku main

# One-off backfill, then schedule ongoing ingestion:
heroku run python ingest_data.py
heroku addons:create scheduler:standard      # add job:  python ingest_data.py  (e.g. hourly)
```

- `Procfile` runs the web dyno (`uvicorn app:app`) and a `release:` step that ensures the Qdrant
  collection exists on deploy.
- The query stream emits an early progress event so the first byte arrives within Heroku's 30 s
  router window during retrieval.
- Repeat with a different `TENANT`, Heroku app, and Qdrant collection for each additional
  branded deployment.

## Layout

```
ingest_data.py   app.py   main.py                    # entry points
backfill_citation_ids.py   report_classifications.py  # one-off / audit scripts
config.yaml  Procfile  runtime.txt  requirements.txt
templates/index.html   static/app.js                  # web UI
branding/<tenant>/                                     # per-tenant brand + policy areas
forum_rag/    config, parse, chunk, embed, store, classify, drive, agent, auth
data/                                                  # sample transcript (gitignored)
```
