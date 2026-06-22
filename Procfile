web: uvicorn app:app --host 0.0.0.0 --port ${PORT}
release: python -c "from sc_rag.store import ensure_collection; ensure_collection()"
