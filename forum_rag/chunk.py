"""Group consecutive transcript turns into passage-sized, citeable chunks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .parse import Transcript, Turn

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
        except Exception:
            _enc = None
    if _enc is None:
        return max(1, len(text) // 4)
    return len(_enc.encode(text))


@dataclass
class Chunk:
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


def _fmt_turn(t: Turn) -> str:
    return f"{t.speaker}: {t.text}"


def chunk_transcript(
    t: Transcript,
    *,
    target_tokens: int = 350,
    overlap_turns: int = 1,
    max_turns_per_chunk: int = 40,
) -> list[Chunk]:
    turns = [x for x in t.turns if x.text]
    chunks: list[Chunk] = []
    n = len(turns)
    i = 0

    while i < n:
        group: list[Turn] = []
        tokens = 0
        j = i
        while j < n and len(group) < max_turns_per_chunk:
            group.append(turns[j])
            tokens += _count_tokens(turns[j].text)
            j += 1
            if tokens >= target_tokens:
                break

        if not group:
            break

        text = "\n".join(_fmt_turn(x) for x in group)
        chunks.append(
            Chunk(
                chunk_id=f"{t.transcript_id}:{group[0].turn_index}-{group[-1].turn_index}",
                transcript_id=t.transcript_id,
                text=text,
                speakers=sorted({x.speaker for x in group}),
                start_ms=group[0].start_ms,
                end_ms=group[-1].end_ms,
                turn_start=group[0].turn_index,
                turn_end=group[-1].turn_index,
                review_flagged=any(x.review_flag for x in group),
                session=t.session,
                table=t.table,
                date=t.date,
                source_file=t.source_file,
            )
        )

        # advance with overlap, guaranteeing forward progress
        next_i = j - overlap_turns
        i = next_i if next_i > i else j

    return chunks
