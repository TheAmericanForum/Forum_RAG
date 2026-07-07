"""Group consecutive transcript turns into passage-sized, citeable chunks."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .parse import Transcript, Turn

log = logging.getLogger(__name__)

_enc = None
_enc_tried = False


def _count_tokens(text: str) -> int:
    """Token count via tiktoken when available; fall back to a rough estimate."""
    global _enc, _enc_tried
    if not _enc_tried:
        _enc_tried = True
        try:
            import tiktoken

            _enc = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            # tiktoken missing, or its BPE data failed to load/fetch (e.g. offline).
            # Chunk boundaries will be slightly less precise (char-count estimate
            # instead of exact token count) but ingestion should still proceed.
            log.warning("tiktoken unavailable, falling back to char-based token estimate: %s", e)
            _enc = None
    if _enc is None:
        return max(1, len(text) // 4)
    return len(_enc.encode(text))


@dataclass
class Chunk:
    # Stable derived ID, "{transcript_id}:{turn_start}-{turn_end}". Consumed downstream
    # by store.py's _point_id (deterministic Qdrant point ID, so re-ingesting the same
    # chunk overwrites rather than duplicates) and by the citation_id carry-forward
    # logic in store.py's upsert_chunks (re-chunking with different boundaries changes
    # this ID, which mints a fresh citation permalink rather than reusing the old one).
    chunk_id: str
    transcript_id: str
    text: str
    speakers: list[str]
    start_ms: int
    end_ms: int
    turn_start: int
    turn_end: int
    review_flagged: bool
    # transcript-level metadata carried onto every chunk
    session: str
    table: Optional[str]
    date: Optional[str]
    source_file: str


def _fmt_turn(turn: Turn) -> str:
    return f"{turn.speaker}: {turn.text}"


def chunk_transcript(
    transcript: Transcript,
    *,
    target_tokens: int = 350,
    overlap_turns: int = 1,
    max_turns_per_chunk: int = 40,
) -> list[Chunk]:
    """Group a transcript's turns into overlapping chunks for embedding/citation.

    Each chunk grows turn-by-turn until either stop condition is hit, whichever comes
    first: `max_turns_per_chunk` (hard cap) or `target_tokens` (soft budget, checked
    only after appending a turn — so a chunk can overshoot target_tokens by up to one
    turn's worth of tokens; that's intentional, not a bug, since turns aren't split).
    Consecutive chunks overlap by `overlap_turns` turns so a claim made right at a
    chunk boundary still has surrounding context in at least one chunk. The overlap
    step is guarded (`next_i if next_i > i else j`) to guarantee forward progress: if
    a single turn's group is smaller than `overlap_turns`, stepping back by
    `overlap_turns` could otherwise not advance past the start of the current chunk,
    looping forever.
    """
    turns = [turn for turn in transcript.turns if turn.text]
    chunks: list[Chunk] = []
    total_turns = len(turns)
    chunk_start_idx = 0

    while chunk_start_idx < total_turns:
        group: list[Turn] = []
        tokens = 0
        scan_idx = chunk_start_idx
        while scan_idx < total_turns and len(group) < max_turns_per_chunk:
            group.append(turns[scan_idx])
            tokens += _count_tokens(turns[scan_idx].text)
            scan_idx += 1
            if tokens >= target_tokens:
                break

        if not group:
            break

        text = "\n".join(_fmt_turn(turn) for turn in group)
        chunks.append(
            Chunk(
                chunk_id=f"{transcript.transcript_id}:{group[0].turn_index}-{group[-1].turn_index}",
                transcript_id=transcript.transcript_id,
                text=text,
                speakers=sorted({turn.speaker for turn in group}),
                start_ms=group[0].start_ms,
                end_ms=group[-1].end_ms,
                turn_start=group[0].turn_index,
                turn_end=group[-1].turn_index,
                review_flagged=any(turn.review_flag for turn in group),
                session=transcript.session,
                table=transcript.table,
                date=transcript.date,
                source_file=transcript.source_file,
            )
        )

        # Advance with overlap, guaranteeing forward progress (see docstring above).
        next_start_idx = scan_idx - overlap_turns
        chunk_start_idx = next_start_idx if next_start_idx > chunk_start_idx else scan_idx

    return chunks
