web: uvicorn app:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips=*
release: python -c "from forum_rag.store import ensure_collection; ensure_collection()"
