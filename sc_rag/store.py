"""Qdrant vector store: collection management, incremental upsert, filtered search.

Hosted by default (QDRANT_URL). Embedded on-disk mode is used only for local dev
when QDRANT_URL is unset — note that does NOT persist on Heroku's ephemeral disk.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from .chunk import Chunk
from .config import get_settings
from .errors import ExternalServiceError

log = logging.getLogger(__name__)

_client: Optional[QdrantClient] = None

_PAYLOAD_INDEXES = [
    ("policy_areas", qm.PayloadSchemaType.KEYWORD),
    ("session", qm.PayloadSchemaType.KEYWORD),
    ("speakers", qm.PayloadSchemaType.KEYWORD),
    ("drive_file_id", qm.PayloadSchemaType.KEYWORD),
    ("transcript_id", qm.PayloadSchemaType.KEYWORD),
]


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        s = get_settings()
        try:
            if s.qdrant_url:
                _client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
            else:
                _client = QdrantClient(path=s.qdrant.local_path)
        except Exception as e:
            log.error("Failed to connect to Qdrant: %s", e)
            raise ExternalServiceError(f"Could not connect to Qdrant: {e}") from e
    return _client


def ensure_collection() -> None:
    s = get_settings()
    client = get_client()
    name = s.qdrant.collection
    try:
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(
                    size=s.qdrant.vector_size, distance=qm.Distance.COSINE
                ),
            )
            log.info("Created Qdrant collection %r", name)
    except Exception as e:
        log.error("Failed to ensure Qdrant collection %r exists: %s", name, e)
        raise ExternalServiceError(f"Could not create/verify Qdrant collection {name!r}: {e}") from e

    for field_name, schema in _PAYLOAD_INDEXES:
        try:
            client.create_payload_index(name, field_name=field_name, field_schema=schema)
        except Exception as e:
            log.debug("Payload index on %r not (re)created (likely already exists): %s", field_name, e)


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def stored_md5_for_file(drive_file_id: str) -> Optional[str]:
    """Return the source_md5 already indexed for this file, or None if not present."""
    s = get_settings()
    try:
        points, _ = get_client().scroll(
            collection_name=s.qdrant.collection,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(key="drive_file_id", match=qm.MatchValue(value=drive_file_id))]
            ),
            limit=1,
            with_payload=["source_md5"],
            with_vectors=False,
        )
    except Exception as e:
        log.error("Qdrant scroll failed for drive_file_id=%r: %s", drive_file_id, e)
        raise ExternalServiceError(f"Qdrant lookup failed for {drive_file_id!r}: {e}") from e
    if points:
        return (points[0].payload or {}).get("source_md5")
    return None


def delete_file(drive_file_id: str) -> None:
    s = get_settings()
    try:
        get_client().delete(
            collection_name=s.qdrant.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="drive_file_id", match=qm.MatchValue(value=drive_file_id))]
                )
            ),
        )
    except Exception as e:
        log.error("Qdrant delete failed for drive_file_id=%r: %s", drive_file_id, e)
        raise ExternalServiceError(f"Qdrant delete failed for {drive_file_id!r}: {e}") from e


def upsert_chunks(
    chunks: list[Chunk],
    vectors: list[list[float]],
    *,
    drive_file_id: str,
    source_md5: str,
    policy_areas_by_chunk: list[list[str]],
) -> None:
    s = get_settings()
    points = []
    for ch, vec, areas in zip(chunks, vectors, policy_areas_by_chunk):
        payload = {
            "chunk_id": ch.chunk_id,
            "transcript_id": ch.transcript_id,
            "drive_file_id": drive_file_id,
            "source_md5": source_md5,
            "text": ch.text,
            "speakers": ch.speakers,
            "start_ms": ch.start_ms,
            "end_ms": ch.end_ms,
            "turn_start": ch.turn_start,
            "turn_end": ch.turn_end,
            "review_flagged": ch.review_flagged,
            "session": ch.session,
            "table": ch.table,
            "date": ch.date,
            "source_file": ch.source_file,
            "policy_areas": areas,
        }
        points.append(qm.PointStruct(id=_point_id(ch.chunk_id), vector=vec, payload=payload))
    try:
        get_client().upsert(collection_name=s.qdrant.collection, points=points)
    except Exception as e:
        log.error("Qdrant upsert failed for drive_file_id=%r (%d points): %s", drive_file_id, len(points), e)
        raise ExternalServiceError(f"Qdrant upsert failed for {drive_file_id!r}: {e}") from e


def search(
    query_vector: list[float],
    *,
    top_k: int = 8,
    policy_area: Optional[str] = None,
    session: Optional[str] = None,
    speaker: Optional[str] = None,
) -> list[dict[str, Any]]:
    s = get_settings()
    must = []
    if policy_area:
        must.append(qm.FieldCondition(key="policy_areas", match=qm.MatchAny(any=[policy_area])))
    if session:
        must.append(qm.FieldCondition(key="session", match=qm.MatchValue(value=session)))
    if speaker:
        must.append(qm.FieldCondition(key="speakers", match=qm.MatchAny(any=[speaker])))
    flt = qm.Filter(must=must) if must else None

    try:
        res = get_client().query_points(
            collection_name=s.qdrant.collection,
            query=query_vector,
            limit=top_k,
            query_filter=flt,
            with_payload=True,
        )
    except Exception as e:
        log.error("Qdrant search failed: %s", e)
        raise ExternalServiceError(f"Qdrant search failed: {e}") from e
    return [{"score": p.score, **(p.payload or {})} for p in res.points]
