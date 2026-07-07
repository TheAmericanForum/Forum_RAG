"""Qdrant vector store: collection management, incremental upsert, filtered search.

Hosted by default (QDRANT_URL). Embedded on-disk mode is used only for local dev
when QDRANT_URL is unset — note that does NOT persist on Heroku's ephemeral disk.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Iterator, Optional

from qdrant_client import QdrantClient
from qdrant_client import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

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
    ("citation_id", qm.PayloadSchemaType.KEYWORD),
]


def get_client() -> QdrantClient:
    """Return the process-wide Qdrant client, creating it on first use."""
    global _client
    if _client is None:
        settings = get_settings()
        try:
            if settings.qdrant_url:
                _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
            else:
                _client = QdrantClient(path=settings.qdrant.local_path)
        except Exception as e:
            log.error("Failed to connect to Qdrant: %s", e)
            raise ExternalServiceError(f"Could not connect to Qdrant: {e}") from e
    return _client


def ensure_collection() -> None:
    """Create the configured Qdrant collection and its payload indexes if missing."""
    settings = get_settings()
    client = get_client()
    name = settings.qdrant.collection
    try:
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(
                    size=settings.qdrant.vector_size, distance=qm.Distance.COSINE
                ),
            )
            log.info("Created Qdrant collection %r", name)
    except Exception as e:
        log.error("Failed to ensure Qdrant collection %r exists: %s", name, e)
        raise ExternalServiceError(f"Could not create/verify Qdrant collection {name!r}: {e}") from e

    for field_name, schema in _PAYLOAD_INDEXES:
        try:
            client.create_payload_index(name, field_name=field_name, field_schema=schema)
        except UnexpectedResponse as e:
            # Qdrant returns 4xx with "already exists" in the message when the index
            # is already present — that specific case is expected/benign on repeat
            # deploys. Any other failure (bad connectivity, permissions, wrong
            # collection) should NOT be silently swallowed, since that would let
            # ingestion proceed against a broken collection.
            if e.status_code and e.status_code < 500 and b"already exists" in e.content.lower():
                log.debug("Payload index on %r already exists, skipping.", field_name)
            else:
                log.error("Failed to create payload index on %r: %s", field_name, e)
                raise ExternalServiceError(
                    f"Could not create payload index on {field_name!r}: {e}"
                ) from e


def _point_id(chunk_id: str) -> str:
    """Deterministic Qdrant point ID for a chunk_id (UUID5), so re-ingesting the same
    chunk overwrites the existing point instead of creating a duplicate."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def iter_all_points(
    scroll_filter: Optional[qm.Filter] = None, with_payload: Any = True
) -> Iterator[Any]:
    """Yield every point in the configured collection matching `scroll_filter`.

    Wraps Qdrant's scroll-with-pagination pattern (page through `next_page_offset`
    until it comes back None) in the same try/except -> ExternalServiceError used by
    every other function here, so callers don't need to talk to the raw client
    directly and lose that error handling.
    """
    settings = get_settings()
    offset = None
    try:
        while True:
            points, offset = get_client().scroll(
                collection_name=settings.qdrant.collection,
                scroll_filter=scroll_filter,
                offset=offset,
                limit=256,
                with_payload=with_payload,
                with_vectors=False,
            )
            yield from points
            if offset is None:
                break
    except Exception as e:
        log.error("Qdrant scroll failed: %s", e)
        raise ExternalServiceError(f"Qdrant scroll failed: {e}") from e


def stored_md5_for_file(drive_file_id: str) -> Optional[str]:
    """Return the source_md5 already indexed for this file, or None if not present."""
    settings = get_settings()
    try:
        points, _ = get_client().scroll(
            collection_name=settings.qdrant.collection,
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


def existing_citation_ids_for_file(drive_file_id: str) -> dict[str, str]:
    """Return {chunk_id: citation_id} for every already-indexed chunk of this file.

    Must be called before delete_file() so re-ingestion can carry forward each
    chunk's permanent citation_id instead of minting a new one.
    """
    scroll_filter = qm.Filter(
        must=[qm.FieldCondition(key="drive_file_id", match=qm.MatchValue(value=drive_file_id))]
    )
    out: dict[str, str] = {}
    for point in iter_all_points(scroll_filter, with_payload=["chunk_id", "citation_id"]):
        payload = point.payload or {}
        citation_id = payload.get("citation_id")
        chunk_id = payload.get("chunk_id")
        if citation_id and chunk_id:
            out[chunk_id] = citation_id
    return out


def get_by_citation_id(citation_id: str) -> Optional[dict[str, Any]]:
    settings = get_settings()
    try:
        points, _ = get_client().scroll(
            collection_name=settings.qdrant.collection,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(key="citation_id", match=qm.MatchValue(value=citation_id))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        log.error("Qdrant scroll failed for citation_id=%r: %s", citation_id, e)
        raise ExternalServiceError(f"Qdrant lookup failed for {citation_id!r}: {e}") from e
    return points[0].payload if points else None


def delete_file(drive_file_id: str) -> None:
    """Delete every point belonging to a Drive file (by drive_file_id)."""
    settings = get_settings()
    try:
        get_client().delete(
            collection_name=settings.qdrant.collection,
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
    existing_citation_ids: Optional[dict[str, str]] = None,
) -> None:
    """Upsert a file's chunks (with their embeddings) into the collection.

    Each chunk's Qdrant point ID is derived deterministically from its chunk_id
    (see _point_id), so re-ingesting the same file overwrites existing points
    instead of duplicating them.
    """
    settings = get_settings()
    existing_citation_ids = existing_citation_ids or {}
    points = []
    for chunk, vector, policy_areas in zip(chunks, vectors, policy_areas_by_chunk):
        # chunk_id is derived from turn boundaries (see chunk.py), so re-chunking a
        # file with different parameters changes chunk_id and this lookup misses —
        # the chunk is then treated as brand-new and gets a freshly minted
        # citation_id, which breaks any permalink issued against the old one.
        citation_id = existing_citation_ids.get(chunk.chunk_id) or str(uuid.uuid4())
        payload = {
            "chunk_id": chunk.chunk_id,
            "citation_id": citation_id,
            "transcript_id": chunk.transcript_id,
            "drive_file_id": drive_file_id,
            "source_md5": source_md5,
            "text": chunk.text,
            "speakers": chunk.speakers,
            "start_ms": chunk.start_ms,
            "end_ms": chunk.end_ms,
            "turn_start": chunk.turn_start,
            "turn_end": chunk.turn_end,
            "review_flagged": chunk.review_flagged,
            "session": chunk.session,
            "table": chunk.table,
            "date": chunk.date,
            "source_file": chunk.source_file,
            "policy_areas": policy_areas,
        }
        points.append(qm.PointStruct(id=_point_id(chunk.chunk_id), vector=vector, payload=payload))
    try:
        get_client().upsert(collection_name=settings.qdrant.collection, points=points)
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
    """Vector-search the collection, optionally AND-filtered by policy_area/session/speaker."""
    settings = get_settings()
    must = []
    if policy_area:
        must.append(qm.FieldCondition(key="policy_areas", match=qm.MatchAny(any=[policy_area])))
    if session:
        must.append(qm.FieldCondition(key="session", match=qm.MatchValue(value=session)))
    if speaker:
        must.append(qm.FieldCondition(key="speakers", match=qm.MatchAny(any=[speaker])))
    query_filter = qm.Filter(must=must) if must else None

    try:
        search_results = get_client().query_points(
            collection_name=settings.qdrant.collection,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
    except Exception as e:
        log.error("Qdrant search failed: %s", e)
        raise ExternalServiceError(f"Qdrant search failed: {e}") from e
    return [{"score": point.score, **(point.payload or {})} for point in search_results.points]
