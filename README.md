# SC_RAG — Agentic RAG over policy-discussion transcripts

Ingests facilitated community-conversation transcripts (verbose AssemblyAI JSON) from
Google Drive, indexes them in a vector store, and answers questions about the **proposals
and trade-offs** raised across the policy areas — with **citations to the exact phrase,
speaker, and timestamp**. Built to deploy on **Heroku** (nothing persists to local disk).

## How it works

- **Ingestion** (`ingest_data.py`, run by Heroku Scheduler): Drive → parse turns → chunk →
  classify policy area(s) → embed (OpenAI) → upsert to Qdrant. Incremental: a file is skipped
  when its md5 already matches what's stored in Qdrant. No local state.
- **Query** (`app.py` web service + `main.py` CLI): Claude **Opus 4.8** plans retrieval via a
  `search_transcripts` tool (multiple searches per question), then writes a cited answer using
  Anthropic's native **Citations** over the retrieved passages.

| Component | Tech |
|---|---|
| LLM (agent + answer) | Claude Opus 4.8 |
| Policy-area classifier | Claude Haiku 4.5 (structured output, multi-label) |
| Embeddings | OpenAI `text-embedding-3-large` (3072-dim) |
| Vector store | Qdrant (hosted; embedded mode for local dev only) |
| Web | FastAPI + minimal UI |

## Configuration

Non-secret settings live in [config.yaml](config.yaml) — **fill in the 4 `policy_areas`**
(names + one-line descriptions); until you do, chunks are labeled `other` and area filtering
is a no-op.

Secrets come from env vars (local `.env`, see [.env.example](.env.example)) / Heroku config vars:

| Var | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude (agent + classifier) |
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
uvicorn app:app --reload --port 5050        # open http://127.0.0.1:5050
```

`--source drive` ingests from Google Drive instead (needs `GOOGLE_SERVICE_ACCOUNT_JSON` +
`DRIVE_FOLDER_IDS`, and the folders shared with the service-account email).

## Deploy to Heroku

```bash
heroku create
# Provision hosted Qdrant (Qdrant Cloud free tier) and set its URL + key, plus the rest:
heroku config:set ANTHROPIC_API_KEY=... OPENAI_API_KEY=... \
  QDRANT_URL=... QDRANT_API_KEY=... \
  DRIVE_FOLDER_IDS=... GOOGLE_SERVICE_ACCOUNT_JSON="$(cat service_account.json)"
git push heroku main

# One-off backfill, then schedule ongoing ingestion:
heroku run python ingest_data.py
heroku addons:create scheduler:standard      # add job:  python ingest_data.py  (e.g. hourly)
```

- `Procfile` runs the web dyno (`uvicorn app:app`) and a `release:` step that ensures the Qdrant
  collection exists on deploy.
- The query stream emits an early progress event so the first byte arrives within Heroku's 30 s
  router window during retrieval.

## Layout

```
ingest_data.py   app.py   main.py            # entry points
config.yaml  Procfile  runtime.txt  requirements.txt
templates/index.html   static/app.js         # web UI
sc_rag/  config, parse, chunk, embed, store, classify, drive, agent
data/                                          # sample transcript (gitignored)
```
